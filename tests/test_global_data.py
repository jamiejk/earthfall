"""Non-US terrain and imagery: CRS choice, dispatch, and each source's URL / tiling logic, with HTTP mocked."""
import io
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoblock_core import global_dem, utm  # noqa: E402

rasterio = pytest.importorskip("rasterio")


class Resp:
    def __init__(self, content=b"", status=200, kind="application/json", payload=None, text=None):
        self.content, self.status_code, self.headers = content, status, {"content-type": kind}
        self._payload, self.text = payload, text if text is not None else ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class Client:
    """Answers GETs from a function of (url, params); records the URLs asked for."""

    def __init__(self, answer):
        self.answer, self.urls = answer, []

    def get(self, url, params=None):
        self.urls.append((url, params))
        return self.answer(url, params)

    def close(self):
        pass


# --- CRS ----------------------------------------------------------------------------------------------------

def test_us_zone_keeps_nad83_and_navd88():
    assert utm.projected_crs(11) == "EPSG:26911"
    assert utm.projected_crs("18") == "EPSG:26918"
    assert utm.vertical_crs(11) == "EPSG:5703"


def test_wgs84_codes_pass_through():
    assert utm.projected_crs(32632) == "EPSG:32632"
    assert utm.projected_crs(32758) == "EPSG:32758"
    assert utm.vertical_crs(32632, "EPSG:5728") == "EPSG:5728"
    assert utm.vertical_crs(32632) == "EPSG:3855"
    for bad in (0, 61, 26911, 32600, 32661, 32700):
        with pytest.raises(ValueError):
            utm.projected_crs(bad)


def test_wgs84_zone_choice():
    assert utm.wgs84_utm_code(45.98369, 7.7853) == 32632      # Gornergrat
    assert utm.wgs84_utm_code(-44.6716, 167.9254) == 32758    # Milford Sound, south
    assert utm.wgs84_utm_code(35.3606, 138.7274) == 32654     # Mount Fuji
    assert utm.wgs84_utm_code(60.39, 5.32) == 32632           # Bergen: the Norway exception
    assert utm.wgs84_utm_code(-33.9576, 18.4037) == 32734     # Table Mountain
    with pytest.raises(ValueError):
        utm.wgs84_utm_code(85, 0)


def test_prepare_terrain_accepts_both_zone_kinds(tmp_path):
    from rasterio.transform import from_origin
    from touchdown.prepare_terrain import prepare
    from pyproj import Transformer
    lat, lon = -44.6716, 167.9254
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32758", always_xy=True).transform(lon, lat)

    def dem(path, radius, step):
        n = int(2 * radius / step) + 1
        with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32", crs="EPSG:32758",
                           transform=from_origin(x - radius, y + radius, step, step)) as dst:
            dst.write(np.full((1, n, n), 12.5, np.float32))
        return path

    meta = prepare(dem(tmp_path / "n.tif", 460, 2), dem(tmp_path / "f.tif", 17200, 50), lat, lon, 32758, tmp_path / "dem")
    assert meta["utm_zone"] == 32758 and meta["landing"] == pytest.approx(12.5)
    with pytest.raises(ValueError, match="UTM zone must be 1..60"):
        prepare(tmp_path / "n.tif", tmp_path / "f.tif", lat, lon, 0, tmp_path / "dem2")


# --- dispatch -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("country,source", [("CH", "swissalti3d"), ("FR", "ign-lidarhd"), ("NO", "kartverket-dtm"),
                                            ("NZ", "linz-dem1m"), ("JP", "gsi-dem5"), ("ES", "es-mdt05"),
                                            ("AT", "tirol-dtm"), ("IT", "glo30"),
                                            ("GB", "glo30"), ("PE", "glo30")])
def test_plan_by_catalogue_country(country, source):
    p = global_dem.plan(46.0, 7.7, country)
    assert p.near == source and p.far == "glo30" and p.projected_crs == "EPSG:32632"


@pytest.mark.parametrize("lat,lon,country", [(45.98369, 7.7853, "CH"), (46.5475, 7.9853, "CH"), (45.8786, 6.8873, "FR"),
                                             (42.15, 9.10, "FR"), (58.9863, 6.1903, "NO"), (35.3606, 138.7274, "JP"),
                                             (-44.6716, 167.9254, "NZ"), (35.10, 129.07, ""), (51.5, -0.12, ""),
                                             (46.63686, 12.31058, ""), (37.7, -119.6, "")])
def test_custom_points_by_outline(lat, lon, country):
    assert global_dem.country_at(lat, lon) == country
    assert global_dem.plan(lat, lon).near == global_dem.COUNTRY_SOURCE.get(country, "glo30")


def test_every_non_us_catalogue_place_has_a_plan():
    from geoblock_core.places import landing, places
    for place in places():
        if place.get("country") == "US":
            continue
        lat, lon = landing(place)
        p = global_dem.plan(lat, lon, place["country"])
        assert p.near in global_dem.PROVIDERS and utm.is_wgs84_utm(p.zone), place["id"]


def test_terrain_route_sends_wgs84_zones_abroad_and_keeps_the_us_path(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    from geoblock_core import providers
    from touchdown import terrain_route
    calls = []
    monkeypatch.setattr(global_dem, "acquire", lambda *a: calls.append(("global", a)) or ("n", "f"))
    assert terrain_route.acquire(45.98, 7.78, "swissalti3d", 32632, tmp_path) == ("n", "f")
    assert calls[0][1][2:4] == ("swissalti3d", 32632)

    seen = []

    class Fake:
        def __init__(self, *a, **kw):
            self.args = a

        def acquire(self, spec, cache):
            seen.append((type(self).__name__, self.args, spec.projected_crs, spec.vertical_crs, spec.radius_m))
            return Path("x"), None

    monkeypatch.setattr(providers, "USGS1mDEMProvider", type("USGS1mDEMProvider", (Fake,), {}))
    monkeypatch.setattr(providers, "USGS3DEPProvider", type("USGS3DEPProvider", (Fake,), {}))
    terrain_route.acquire(37.7156, -119.6772, "CA_YosemiteNP_2019", 11, tmp_path)
    assert seen == [("USGS1mDEMProvider", ("CA_YosemiteNP_2019", 11), "EPSG:26911", "EPSG:5703", 400),
                    ("USGS3DEPProvider", (), "EPSG:26911", "EPSG:5703", 17000)]


def _tiny_dem(path: Path, crs="EPSG:32632", value=100.0):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32", crs=crs,
                       transform=from_origin(0, 4, 1, 1)) as dst:
        dst.write(np.full((1, 4, 4), value, np.float32))
    return path


def test_acquire_falls_back_to_glo30_and_says_so(monkeypatch, tmp_path, capsys):
    made = []

    class FakeGLO:
        def acquire(self, spec, cache):
            made.append(spec.radius_m)
            return _tiny_dem(tmp_path / f"glo_{len(made)}.tif"), None

    class Broken:
        def acquire(self, spec, cache):
            raise ConnectionError("server down")

    monkeypatch.setattr(global_dem, "GLO30Provider", FakeGLO)
    monkeypatch.setitem(global_dem.PROVIDERS, "swissalti3d", Broken())
    near, far = global_dem.acquire(45.98, 7.78, "swissalti3d", 32632, tmp_path)
    assert made == [400, 17000]
    with rasterio.open(near) as src:
        assert src.tags()["dem_fallback_1m_share"] == "0.000"
        assert "server down" in src.tags()["dem_fallback_reason"]
    assert "near DEM from GLO-30" in capsys.readouterr().out


# --- providers: URLs and tiling ------------------------------------------------------------------------------

def _stac_item(year, square, res_keys=("0.5", "2")):
    return {"id": f"swissalti3d_{year}_{square}",
            "assets": {f"swissalti3d_{year}_{square}_{r}_2056_5728.tif":
                       {"href": f"https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_{year}_{square}/"
                                f"swissalti3d_{year}_{square}_{r}_2056_5728.tif"} for r in res_keys}}


def test_swissalti3d_keeps_the_newest_tile_and_follows_paging():
    pages = {None: {"features": [_stac_item(2019, "2624-1092"), _stac_item(2021, "2624-1092")],
                    "links": [{"rel": "next", "href": "https://next.page"}]},
             "https://next.page": {"features": [_stac_item(2020, "2625-1092")], "links": []}}
    client = Client(lambda url, params: Resp(payload=pages[None if params else url]))
    urls = global_dem.SwissALTI3DProvider(0.5, client=client).asset_urls((7.78, 45.98, 7.79, 45.99))
    assert [u.rsplit("/", 1)[-1] for u in urls] == ["swissalti3d_2020_2625-1092_0.5_2056_5728.tif",
                                                    "swissalti3d_2021_2624-1092_0.5_2056_5728.tif"]
    assert client.urls[0][1]["bbox"] == "7.780000,45.980000,7.790000,45.990000"
    urls2 = global_dem.SwissALTI3DProvider(2.0, client=Client(lambda u, p: Resp(payload=pages[None if p else u])))
    assert all(u.endswith("_2_2056_5728.tif") for u in urls2.asset_urls((7.78, 45.98, 7.79, 45.99)))


def test_ign_and_kartverket_urls_and_chunks():
    ign = global_dem.IGNRGEAltiProvider()
    url = ign.url((1000000.0, 6500000.0, 1000900.0, 6500900.0), 900, 900)
    assert url.startswith("https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap")
    assert "LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES" in url and "CRS=EPSG:2154" in url
    assert "BBOX=1000000.00,6500000.00,1000900.00,6500900.00&WIDTH=900&HEIGHT=900" in url
    kv = global_dem.KartverketDTMProvider().url((-30000.0, 6580000.0, -29000.0, 6581000.0), 1000, 1000)
    assert "wcs.hoyde-dtm-nhm-25833" in kv and "COVERAGE=nhm_dtm_topo_25833" in kv and "CRS=EPSG:25833" in kv
    jobs = ign.chunks((0.4, 0.2, 4100.0, 1000.7))
    assert {(w, h) for _, w, h, _, _ in jobs} == {(2000, 1001), (100, 1001)} and len(jobs) == 3
    assert jobs[0][0] == (0.0, 0.0, 2000.0, 1001.0) and jobs[-1][4] == 4000


def _geotiff_bytes(arr, nodata=None):
    from rasterio.transform import from_origin
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
                      transform=from_origin(0, 10, 2, 2), nodata=nodata) as dst:
            dst.write(arr[None].astype("float32"))
        buf.write(mem.read())
    return buf.getvalue()


def test_grid_service_writes_the_requested_grid_in_the_national_crs(tmp_path):
    def answer(url, params):
        w = int(url.split("WIDTH=")[1].split("&")[0])
        h = int(url.split("HEIGHT=")[1].split("&")[0])
        return Resp(_geotiff_bytes(np.full((h, w), 1234.5)), kind="image/geotiff")

    spec = global_dem.spec_for(45.8786, 6.8873, 32632, 50)
    path, record = global_dem.IGNRGEAltiProvider(client=Client(answer)).acquire(spec, tmp_path)
    with rasterio.open(path) as src:
        assert src.crs.to_string() == "EPSG:2154" and src.res == (1.0, 1.0)
        assert float(src.read(1).mean()) == pytest.approx(1234.5)
        assert src.tags()["vertical_crs"] == "EPSG:5720"
    assert record.vertical_crs == "EPSG:5720" and record.nominal_resolution_m == 1.0


def test_grid_service_refuses_an_xml_error_and_a_hole(tmp_path):
    spec = global_dem.spec_for(58.9863, 6.1903, 32632, 50)
    xml = Client(lambda u, p: Resp(b"<ServiceException/>", kind="application/xml", text="<ServiceException/>"))
    with pytest.raises(ValueError, match="answered"):
        global_dem.KartverketDTMProvider(client=xml).acquire(spec, tmp_path)
    holes = Client(lambda u, p: Resp(_geotiff_bytes(np.full((int(u.split("HEIGHT=")[1].split("&")[0]),
                                                              int(u.split("WIDTH=")[1].split("&")[0])), -9999.0)),
                                     kind="image/tiff"))
    with pytest.raises(ValueError, match="covers only 0%"):
        global_dem.KartverketDTMProvider(client=holes).acquire(spec, tmp_path / "b")


def test_nz_sheet_and_tile_names():
    # verified against the nz-elevation / nz-imagery buckets, 18 Oct: CB08 spans E 1180000-1204000, N 5010000-5046000
    assert global_dem.nz_sheet(1197759, 5040989) == "CB08"
    assert global_dem.nz_sheet(1180000.5, 5045999.5) == "CB08"
    assert global_dem.nz_sheet(1179999.5, 5045999.5) == "CB07"
    assert global_dem.nz_sheet(1180000.5, 5046000.5) == "CA08"
    assert global_dem.nz_tile(1194400.5, 5045999.5, 5000) == "CB08_5000_0107"   # bucket item bounds 1194400.., ..5046000
    assert global_dem.nz_tile(1201400, 5045000, 5000) == "CB08_5000_0109"
    assert global_dem.nz_tile(1180000.5, 5045999.5, 10000) == "CB08_10000_0101"
    assert global_dem.nz_tile(1180000.5, 5045999.5, 500) == "CB08_500_001001"
    assert global_dem.nz_sheets((1179000, 5040000, 1181000, 5047000)) == ["CA07", "CA08", "CB07", "CB08"]
    urls = global_dem.LINZDEMProvider().urls((1195000, 5040000, 1196000, 5041000), "x/dem_1m/2193")
    assert urls == ["https://nz-elevation.s3.ap-southeast-2.amazonaws.com/x/dem_1m/2193/CB08.tiff"]


def _png(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), rgb).save(buf, "PNG")
    return buf.getvalue()


def test_gsi_decode():
    rgb = np.array([[[0, 0, 100], [0, 1, 0], [128, 0, 0], [255, 255, 255], [0x0F, 0x42, 0x40]]], np.uint8)
    h = global_dem.gsi_decode(rgb)
    assert h[0, 0] == pytest.approx(1.0) and h[0, 1] == pytest.approx(2.56)
    assert np.isnan(h[0, 2]) and h[0, 3] == pytest.approx(-0.01) and h[0, 4] == pytest.approx(10000.0)


def test_gsi_tiles_layers_and_fallback(tmp_path):
    p = global_dem.GSIDEMProvider()
    assert p.url("dem5a_png", 15, 29070, 12944) == "https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/15/29070/12944.png"
    assert p.tiles((138.72, 35.36, 138.73, 35.37), 15) == [(29010, 12938), (29011, 12938), (29010, 12939), (29011, 12939)]

    def answer(url, params):  # dem5a has nothing here, dem5b a flat 3776.00 m
        if "dem5a_png" in url:
            return Resp(status=404)
        return Resp(_png((0x05, 0xC3, 0x40)), kind="image/png")  # 377664 * 0.01 = 3776.64 m

    spec = global_dem.spec_for(35.3606, 138.7274, 32654, 60)
    path, record = global_dem.GSIDEMProvider(client=Client(answer)).acquire(spec, tmp_path)
    with rasterio.open(path) as src:
        assert src.crs.to_string() == "EPSG:32654" and src.res == (2.0, 2.0)
        assert float(np.median(src.read(1))) == pytest.approx(3776.64, abs=0.01)
        assert src.tags()["layers"].startswith("dem5b_png:")
    assert record.nominal_resolution_m == 5 and record.vertical_crs == "EPSG:6695"


# --- imagery: open Sentinel-2 fill and LINZ ------------------------------------------------------------------

def test_mgrs_tiles():
    from touchdown import aerial_open as ao
    assert ao.mgrs_tile(45.98369, 7.7853) == "32TMR"     # scenes listed in sentinel-cogs, 18 Oct
    assert ao.mgrs_tile(-44.6716, 167.9254) == "58GGR"   # idem
    assert ao.tile_prefix("32TMR", 2026, 7) == "sentinel-s2-l2a-cogs/32/T/MR/2026/7/"
    assert ao.tile_prefix("58GGR", 2026, 1) == "sentinel-s2-l2a-cogs/58/G/GR/2026/1/"
    assert len(ao.mgrs_tiles((7.3, 45.6, 8.3, 46.3))) >= 2
    assert ao.summer_months(46) == (7, 8, 6, 9) and ao.summer_months(-44)[0] == 1 and len(ao.summer_months(0)) == 12


def test_s2_scenes_from_the_bucket_listing():
    from touchdown import aerial_open as ao
    listing = ("<?xml version='1.0'?><ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>"
               "<CommonPrefixes><Prefix>sentinel-s2-l2a-cogs/32/T/MR/2026/7/S2A_32TMR_20260701_0_L2A/</Prefix>"
               "</CommonPrefixes><CommonPrefixes><Prefix>sentinel-s2-l2a-cogs/32/T/MR/2026/7/S2B_32TMR_20260704_0_L2A/"
               "</Prefix></CommonPrefixes></ListBucketResult>")
    cloud = {"S2A_32TMR_20260701_0_L2A": 3.0, "S2B_32TMR_20260704_0_L2A": 80.0}

    def answer(url, params):
        if params:
            assert params["prefix"].startswith("sentinel-s2-l2a-cogs/32/T/MR/")
            return Resp(text=listing if params["prefix"].endswith("/2026/7/") else listing.split("<Common")[0]
                        + "</ListBucketResult>", kind="application/xml")
        scene = url.rsplit("/", 1)[-1].removesuffix(".json")
        return Resp(payload={"id": scene, "properties": {"eo:cloud_cover": cloud[scene], "datetime": "2026-07-01T10:00:00Z",
                                                         "s2:nodata_pixel_percentage": 1.0},
                             "assets": {"visual": {"href": f"https://b/{scene}/TCI.tif"},
                                        "scl": {"href": f"https://b/{scene}/SCL.tif"}}})

    found = ao.scenes_from_bucket("32TMR", 46.0, today=date(2026, 10, 18), client=Client(answer))
    assert [s["id"] for s in found] == ["S2A_32TMR_20260701_0_L2A"]
    assert found[0]["visual"].endswith("/TCI.tif") and found[0]["scl"].endswith("/SCL.tif")


def test_s2_median_ignores_clouds():
    from touchdown import aerial_open as ao
    a, b, c = (np.full((3, 2, 2), v, np.uint8) for v in (10, 20, 250))
    ok = np.ones((2, 2), bool)
    cloudy = np.zeros((2, 2), bool)
    rgb, have = ao.s2_median([a, b, c], [ok, ok, cloudy])
    assert (rgb == 15).all() and have.all()
    rgb, have = ao.s2_median([a], [cloudy])
    assert not have.any()


def test_linz_collection_choice_and_tiles():
    from touchdown import aerial_open as ao
    catalog = (("southland/southland_2023_0.25m/rgb/2193/collection.json", ""),
               ("southland/southland_2023_0.1m/rgb/2193/collection.json", ""),
               ("southland/southland_2005-2011_0.75m/rgb/2193/collection.json", ""),
               ("southland/southland_2023_0.25m/rgbnir/2193/collection.json", ""),
               ("auckland/auckland_2024_0.06m/rgb/2193/collection.json", ""),
               ("new-zealand/new-zealand_2024-2025_10m/rgb/2193/collection.json", ""))
    bbox = (167.92, -44.676, 167.93, -44.668)
    assert ao.nz_candidates(bbox, catalog) == ["southland/southland_2023_0.1m/rgb/2193/collection.json",
                                               "southland/southland_2023_0.25m/rgb/2193/collection.json"]
    assert ao.slug_year("southland/southland_2005-2011_0.75m/rgb") == 2011
    tiles = ao.nz_tiles_over(bbox, 5000)
    assert tiles and all(t.startswith("CB08_5000_") for t in tiles)


def test_aerial_chain_uses_the_open_fill_and_eox_only_on_request(monkeypatch):
    pytest.importorskip("cv2")
    from touchdown import aerial
    monkeypatch.delenv("TOUCHDOWN_IMAGERY_FILL", raising=False)
    bbox = (-72.55, -13.17, -72.53, -13.16)  # Machu Picchu: no national source
    assert [s.id for s in aerial.chain(aerial.pick("PE"), bbox, 0.4)] == ["s2-l2a"]
    monkeypatch.setenv("TOUCHDOWN_IMAGERY_FILL", "s2cloudless")
    assert [s.id for s in aerial.chain(aerial.pick("PE"), bbox, 0.4)] == ["s2cloudless"]
    monkeypatch.setenv("TOUCHDOWN_IMAGERY_FILL", "nonsense")
    with pytest.raises(ValueError):
        aerial.fill()


def test_linz_only_for_fine_squares_and_a_failing_source_is_skipped(monkeypatch):
    pytest.importorskip("cv2")
    from touchdown import aerial
    monkeypatch.delenv("TOUCHDOWN_IMAGERY_FILL", raising=False)
    monkeypatch.delenv("LINZ_BASEMAPS_KEY", raising=False)
    bbox = (167.90, -44.69, 167.95, -44.65)
    assert [s.id for s in aerial.chain("linz-nz", bbox, 0.44)] == ["linz-nz", "s2-l2a"]
    assert [s.id for s in aerial.chain("linz-nz", bbox, 11.9)] == ["s2-l2a"]

    def boom(b, p):
        raise ConnectionError("down")

    def green(b, p):
        out = np.zeros((p, p, 4), np.uint8)
        out[..., 1], out[..., 3] = 200, 255
        return out

    sources = dict(aerial.SOURCES)
    sources["linz-nz"] = aerial.Source("linz-nz", "x", "NZ", ((166.0, -47.4, 178.7, -34.2),), fetch=boom)
    sources["s2-l2a"] = aerial.Source("s2-l2a", "y", "", ((-180.0, -85.0, 180.0, 85.0),), fetch=green)
    monkeypatch.setattr(aerial, "SOURCES", sources)
    out, cover = aerial.square_rgba("linz-nz", bbox, 16)
    assert cover == {"linz-nz": 0.0, "s2-l2a": 1.0} and (out[..., 1] == 200).all()


# --- the supported-places file ------------------------------------------------------------------------------

def test_place_data_support_json_is_current_and_tiered():
    pytest.importorskip("cv2")
    from touchdown import place_data_support as pds
    monkey = dict(__import__("os").environ)
    if monkey.get("TOUCHDOWN_IMAGERY_FILL"):
        pytest.skip("fill overridden in the environment")
    built = pds.build()
    assert json.loads(pds.OUT.read_text()) == json.loads(json.dumps(built, ensure_ascii=False))
    rows = {r["id"]: r for r in built["places"]}
    assert rows["gornergrat-matterhorn"]["tier"] == "fine"
    assert rows["milford-sound"]["imagery"]["source"] == "linz-nz"
    assert rows["tre-cime-di-lavaredo"]["tier"] == "coarse"
    assert rows["yosemite-glacier-point"]["terrain"]["projected_crs"].startswith("EPSG:269")
    assert pds.tier(5.0, 0.5) == "fine" and pds.tier(10.0, 0.6) == "coarse" and pds.tier(1.0, 10.0) == "coarse"


def test_manifest_gen_supports_fine_places_outside_the_us():
    from touchdown import manifest_gen
    ok, _ = manifest_gen.support(manifest_gen.find("gornergrat-matterhorn"))
    assert ok
    ok, why = manifest_gen.support(manifest_gen.find("ben-nevis"))
    assert not ok and "30 m" in why
    m = manifest_gen.generate("milford-sound", "subject.png", "a person in a red jacket")
    assert m["terrain"]["enabled"] and "project" not in m["terrain"] and "utm_zone" not in m["terrain"]
    assert m["terrain"]["heading"] >= 0  # catalogue view toward Mitre Peak
    m = manifest_gen.generate("mount-fuji", "subject.png", "a person")
    assert m["terrain"]["heading"] == -1.0 and m["terrain"]["embedding_rank"]
    assert manifest_gen.naip_words(manifest_gen.find("mount-fuji"), None) is None


def test_app_picker_matches_manifest_gen_outside_the_us():
    import importlib
    from touchdown import manifest_gen
    places = importlib.import_module("app.server.places")
    for pid in ("gornergrat-matterhorn", "aiguille-du-midi", "milford-sound", "ben-nevis", "machu-picchu"):
        place = manifest_gen.find(pid)
        assert places.support(place)[0] == manifest_gen.support(place)[0], pid


def test_view_search_needs_sky_in_the_final_frame():
    import numpy as np
    from touchdown import terrain_route
    flat_near, flat_far = np.zeros((401, 401)), np.zeros((681, 681))
    assert abs(terrain_route.horizon_deg(flat_near, flat_far, 0, 0, 90)) < 1
    # a slope rising 40 m within 100 m ahead (east) fills the frame: Delicate Arch's slickrock bowl
    xs = (np.arange(401) - 200) * 2.0
    wall = np.tile(np.clip(xs - 20, 0, None) * 0.4, (401, 1))
    assert terrain_route.horizon_deg(wall, flat_far, 0, 0, 90) > terrain_route.SKY_MAX_HORIZON_DEG
    assert terrain_route.horizon_deg(wall, flat_far, 0, 0, 270) < 1  # facing away: open sky


def test_horseshoe_bend_looks_at_the_loop():
    from geoblock_core.places import find_place, view_heading
    assert 270 < view_heading(find_place("horseshoe-bend")) < 295  # west-north-west from the overlook


def test_new_national_sources_18_oct():
    from touchdown import aerial
    # Tre Cime: South Tyrol's orthophoto first, the fill after it (the Veneto side of the square has no source)
    assert aerial.chain(aerial.pick("IT"), (12.30, 46.63, 12.32, 46.64), 0.4)[0].id == "bz-it"
    assert aerial.pick("HK") == "hk-imagery" and aerial.pick("AT") == "basemap-at" and aerial.pick("DE") == "dop20-by"
    url = aerial.tile_url(aerial.SOURCES["bz-it"], 1, 1, 0)
    assert "BBOX=0.000,0.000,20037508.343,20037508.343&" in url
    assert aerial.tile_url(aerial.SOURCES["basemap-at"], 3, 4, 5).endswith("/google3857/3/5/4.jpeg")
    tirol = global_dem.TirolDTMProvider()
    assert tirol.zone_provider(11.38).crs == "EPSG:31254" and tirol.zone_provider(12.69).crs == "EPSG:31255"
    assert "COVERAGEID=Gelaendemodell_50cm_M31&SUBSET=x(" in tirol.zone_provider(12.69).url((0, 0, 10, 10), 20, 20)
    assert global_dem.plan(41.593, 1.837).near == "es-mdt05" and global_dem.plan(47.07, 12.69).near == "tirol-dtm"


def test_coverage_bytes_takes_the_image_part_of_a_multipart_wcs_answer():
    from geoblock_core.global_dem import _coverage_bytes
    body = (b'--wcs\r\nContent-Type: text/xml\r\nContent-ID: GML-Part\r\n\r\n<gmlcov/>\r\n'
            b'--wcs\r\nContent-Type: image/tiff\r\nContent-Transfer-Encoding: binary\r\n\r\nII*\x00tiff\r\n--wcs--\r\n')
    kind = 'multipart/related; boundary="wcs";start="GML-Part";type="text/xml"'
    assert _coverage_bytes(kind, body).startswith(b"II*\x00")
    assert _coverage_bytes(kind, body.replace(b"image/tiff", b"text/plain")) is None
