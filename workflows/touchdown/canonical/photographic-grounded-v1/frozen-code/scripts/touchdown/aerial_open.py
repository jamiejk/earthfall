"""Openly licensed imagery read straight from cloud-optimised GeoTIFFs, for touchdown.aerial.

- ``s2_l2a_rgba``: Copernicus Sentinel-2 L2A true colour (10 m) from the ``sentinel-cogs`` AWS open-data bucket,
  a per-pixel median of a few low-cloud summer scenes with the scene classification's cloud, shadow and cirrus
  pixels dropped. Copernicus data are free for any use, commercial included (Copernicus data policy; credit
  "Contains modified Copernicus Sentinel data <year>"). This is the default fill under the national sources; the
  EOX cloudless mosaic (CC BY-NC-SA) is an explicit opt-in only.
- ``linz_rgba``: New Zealand aerial photos (5-50 cm, CC BY 4.0) from LINZ's open ``nz-imagery`` bucket. No key
  (unlike the LINZ Basemaps tile API).

Scenes are found through the Earth Search STAC API when it answers, else by listing the bucket itself (one
MGRS tile, month by month, reading each scene's STAC item for its cloud share). Both outputs follow the contract
of nadir_dive._naip: plate carree RGBA over (west, south, east, north), ``pixels`` rows, alpha 0 where nothing.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache

import numpy as np

USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"
S2_BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
S2_PREFIX = "sentinel-s2-l2a-cogs"
S2_STAC = "https://earth-search.aws.element84.com/v1/search"
S2_SCENES = 4               # scenes per MGRS tile in the median
S2_MAX_CLOUD = 20.0         # percent, per scene
S2_DROP_SCL = (0, 1, 3, 8, 9, 10)  # no data, saturated, cloud shadow, cloud medium/high, cirrus
NZ_BUCKET = "https://nz-imagery.s3.ap-southeast-2.amazonaws.com"
NZ_MAX_GSD = 0.5            # metres; the 10 m satellite collections are left to the Sentinel-2 fill
NZ_COLLECTIONS = 3          # collections tried per square, newest first


def _client():
    import httpx
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)


def _grid(bbox, pixels: int):
    from rasterio.transform import from_bounds
    west, south, east, north = bbox
    height, width = pixels, round(pixels * (east - west) / (north - south))
    return from_bounds(west, south, east, north, width, height), height, width


def _warp(uri: str, bbox, pixels: int, bands, resampling, metres_per_px: float, native_m: float):
    """``bands`` of a COG warped onto the bbox's plate carree grid, from the overview nearest the output scale."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    transform, height, width = _grid(bbox, pixels)
    level = int(math.floor(math.log2(max(metres_per_px / native_m, 1.0)))) - 1  # -1 = full resolution
    kw = {"overview_level": level} if level >= 0 else {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
        try:
            src = rasterio.open(uri, **kw)
        except rasterio.errors.RasterioIOError:
            src = rasterio.open(uri)  # fewer overviews than asked for
        with src, WarpedVRT(src, crs="EPSG:4326", transform=transform, width=width, height=height,
                            resampling=resampling) as vrt:
            data = vrt.read(bands)
            mask = vrt.dataset_mask()
    return data, mask


# --- Sentinel-2 L2A ------------------------------------------------------------------------------------------

BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"
ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
COL_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")


def mgrs_tile(lat: float, lon: float) -> str:
    """The Sentinel-2 (MGRS) tile id holding a point, e.g. 32TMR; plain 6-degree UTM zones, as the S2 grid uses."""
    from pyproj import Transformer
    zone = min(int((lon + 180) // 6) + 1, 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    easting, northing = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    band = BAND_LETTERS[min(int((lat + 80) // 8), len(BAND_LETTERS) - 1)]
    col = COL_SETS[(zone - 1) % 3][int(easting // 100_000) - 1]
    row = ROW_LETTERS[(int(northing // 100_000) + (5 if zone % 2 == 0 else 0)) % 20]
    return f"{zone}{band}{col}{row}"


def mgrs_tiles(bbox) -> list[str]:
    west, south, east, north = bbox
    n = max(2, math.ceil(max(east - west, north - south) / 0.25) + 1)  # a point every ~25 km
    out = []
    for lat in np.linspace(south, north, n):
        for lon in np.linspace(west, east, n):
            t = mgrs_tile(float(lat), float(lon))
            if t not in out:
                out.append(t)
    return out


def summer_months(lat: float) -> tuple[int, ...]:
    """Low-snow, long-day months: June-September north of the tropics, December-March south; any in the tropics."""
    if lat >= 23:
        return (7, 8, 6, 9)
    if lat <= -23:
        return (1, 2, 12, 3)
    return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


def tile_prefix(tile: str, year: int, month: int) -> str:
    m = re.fullmatch(r"(\d{1,2})([C-X])([A-Z]{2})", tile)
    if not m:
        raise ValueError(f"not an MGRS tile id: {tile}")
    return f"{S2_PREFIX}/{int(m.group(1))}/{m.group(2)}/{m.group(3)}/{year}/{month}/"


def list_scenes(client, tile: str, year: int, month: int) -> list[str]:
    """Scene folder names in the bucket for one tile and month (S3 ListObjectsV2, anonymous)."""
    r = client.get(f"{S2_BUCKET}/", params={"list-type": "2", "prefix": tile_prefix(tile, year, month),
                                           "delimiter": "/"})
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [p.text.rstrip("/").rsplit("/", 1)[-1] for p in ET.fromstring(r.text).findall(".//s3:CommonPrefixes/s3:Prefix", ns)]


def _scene_url(tile: str, year: int, month: int, scene: str, name: str) -> str:
    return f"{S2_BUCKET}/{tile_prefix(tile, year, month)}{scene}/{name}"


def scenes_from_bucket(tile: str, lat: float, today: date | None = None, client=None) -> list[dict]:
    """Up to S2_SCENES low-cloud scenes of ``tile`` from the last three years' summer months, newest years first."""
    today = today or date.today()
    own = client is None
    client = client or _client()
    picked: list[dict] = []
    try:
        for year in range(today.year - (0 if today.month > 9 else 1), today.year - 4, -1):
            for month in summer_months(lat):
                if date(year, month, 1) > today:
                    continue
                names = list_scenes(client, tile, year, month)

                def item(scene, year=year, month=month):
                    r = client.get(_scene_url(tile, year, month, scene, scene + ".json"))
                    return r.json() if r.status_code == 200 else None

                with ThreadPoolExecutor(6) as pool:
                    items = [i for i in pool.map(item, names) if i]
                for it in items:
                    p = it.get("properties", {})
                    if p.get("eo:cloud_cover", 100) <= S2_MAX_CLOUD and p.get("s2:nodata_pixel_percentage", 0) < 50:
                        picked.append({"id": it["id"], "cloud": p["eo:cloud_cover"],
                                       "date": p.get("datetime", "")[:10],
                                       "visual": it["assets"]["visual"]["href"], "scl": it["assets"]["scl"]["href"]})
            if len(picked) >= S2_SCENES:
                break
    finally:
        if own:
            client.close()
    return sorted(picked, key=lambda s: s["cloud"])[:S2_SCENES]


def scenes_from_stac(bbox, lat: float, today: date | None = None, client=None) -> list[dict]:
    """The same from Earth Search: one search per summer season, least cloud first."""
    today = today or date.today()
    own = client is None
    client = client or _client()
    out: list[dict] = []
    try:
        months = summer_months(lat)
        for year in range(today.year, today.year - 4, -1):
            if len(months) == 12:
                start, end = f"{year}-01-01", f"{year}-12-31"
            elif lat >= 0:
                start, end = f"{year}-06-01", f"{year}-09-30"
            else:
                start, end = f"{year - 1}-12-01", f"{year}-03-31"
            body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                    "limit": 50, "query": {"eo:cloud_cover": {"lt": S2_MAX_CLOUD}},
                    "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]}
            r = client.post(S2_STAC, json=body)
            r.raise_for_status()
            for it in r.json().get("features", []):
                p = it["properties"]
                if p.get("s2:nodata_pixel_percentage", 0) < 50:
                    out.append({"id": it["id"], "cloud": p["eo:cloud_cover"], "date": p.get("datetime", "")[:10],
                                "tile": f"{p.get('mgrs:utm_zone')}{p.get('mgrs:latitude_band')}{p.get('mgrs:grid_square')}",
                                "visual": it["assets"]["visual"]["href"], "scl": it["assets"]["scl"]["href"]})
            if len(out) >= 3 * S2_SCENES:
                break
    finally:
        if own:
            client.close()
    by_tile: dict[str, list[dict]] = {}
    for s in sorted(out, key=lambda s: s["cloud"]):
        by_tile.setdefault(s["tile"], []).append(s)
    return [s for scenes in by_tile.values() for s in scenes[:S2_SCENES]]


def s2_scenes(bbox, today: date | None = None) -> list[dict]:
    lat = (bbox[1] + bbox[3]) / 2
    try:
        found = scenes_from_stac(bbox, lat, today)
        if found:
            return found
    except Exception as exc:  # noqa: BLE001 - the API is optional: the bucket listing works without it
        print(f"  s2-l2a: Earth Search unavailable ({type(exc).__name__}); listing the bucket", flush=True)
    return [s for tile in mgrs_tiles(bbox) for s in scenes_from_bucket(tile, lat, today)]


def s2_median(stack: list[np.ndarray], valid: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over the scenes valid there; (RGB uint8, any-valid mask)."""
    data = np.stack(stack).astype(np.float32)          # (n, 3, h, w)
    ok = np.stack(valid)[:, None]                      # (n, 1, h, w)
    data[~np.broadcast_to(ok, data.shape)] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(data, axis=0)
    have = ok.any(axis=0)[0]
    return np.nan_to_num(med, nan=0).clip(0, 255).astype(np.uint8), have


def s2_l2a_rgba(bbox, pixels: int, scenes: list[dict] | None = None) -> np.ndarray:
    from rasterio.enums import Resampling
    scenes = s2_scenes(bbox) if scenes is None else scenes
    _, height, width = _grid(bbox, pixels)
    out = np.zeros((height, width, 4), np.uint8)
    if not scenes:
        return out
    mpp = (bbox[3] - bbox[1]) * 111_320 / pixels
    print(f"  s2-l2a: median of {len(scenes)} scenes ({', '.join(s['id'] for s in scenes)})", flush=True)

    def read(scene):
        rgb, mask = _warp("/vsicurl/" + scene["visual"], bbox, pixels, (1, 2, 3), Resampling.bilinear, mpp, 10.0)
        scl, _ = _warp("/vsicurl/" + scene["scl"], bbox, pixels, (1,), Resampling.nearest, mpp, 20.0)
        good = (mask > 0) & ~np.isin(scl[0], S2_DROP_SCL) & (rgb.max(axis=0) > 0)
        return rgb, good

    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(read, scenes))
    rgb, have = s2_median([p[0] for p in parts], [p[1] for p in parts])
    out[..., :3] = np.moveaxis(rgb, 0, -1)
    out[..., 3] = np.where(have, 255, 0)
    return out


# --- LINZ nz-imagery -----------------------------------------------------------------------------------------

# Rough region boxes (west, south, east, north) for picking which of the ~900 collections to open; generous.
NZ_REGIONS = {
    "northland": (172.4, -36.6, 174.8, -34.3), "auckland": (174.1, -37.4, 175.6, -35.8),
    "waikato": (174.4, -39.5, 176.7, -36.3), "bay-of-plenty": (175.7, -39.3, 178.7, -37.2),
    "gisborne": (177.0, -39.1, 178.7, -37.4), "hawkes-bay": (175.9, -40.6, 178.2, -38.4),
    "taranaki": (173.6, -40.0, 175.1, -38.6), "manawatu-whanganui": (174.7, -40.9, 176.7, -38.5),
    "wellington": (174.5, -41.7, 176.7, -40.5), "tasman": (171.9, -42.5, 173.3, -40.4),
    "nelson": (173.0, -41.5, 173.7, -40.9), "marlborough": (172.7, -42.7, 174.6, -40.6),
    "west-coast": (167.9, -44.6, 172.9, -40.6), "canterbury": (169.4, -45.0, 174.1, -41.8),
    "otago": (168.1, -46.8, 171.4, -43.9), "southland": (166.3, -47.4, 169.5, -44.4),
}


def _intersects(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


@lru_cache(maxsize=1)
def nz_catalog() -> tuple[tuple[str, str], ...]:
    with _client() as client:
        r = client.get(f"{NZ_BUCKET}/catalog.json")
        r.raise_for_status()
        return tuple((link["href"].removeprefix("./"), link.get("title", "")) for link in r.json()["links"]
                     if link.get("rel") == "child")


def slug_year(href: str) -> int:
    years = [int(y) for y in re.findall(r"(?<=[_-])((?:19|20)\d\d)(?=[-_])", href)]
    return max(years) if years else 0


def nz_candidates(bbox, catalog=None) -> list[str]:
    """RGB collections in regions touching ``bbox``, newest first (finer first within a year)."""
    catalog = nz_catalog() if catalog is None else catalog
    regions = {r for r, box in NZ_REGIONS.items() if _intersects(box, bbox)}
    out = []
    for href, _ in catalog:
        parts = href.split("/")
        if len(parts) < 5 or parts[0] not in regions or parts[2] != "rgb":
            continue
        gsd = re.search(r"_(\d+(?:\.\d+)?)m$", parts[1])
        if gsd and float(gsd.group(1)) <= NZ_MAX_GSD:
            out.append((slug_year(href), -float(gsd.group(1)), href))
    return [h for _, _, h in sorted(out, reverse=True)]


@lru_cache(maxsize=64)
def nz_collection(href: str) -> dict:
    with _client() as client:
        r = client.get(f"{NZ_BUCKET}/{href}")
        r.raise_for_status()
        d = r.json()
    items = tuple(link["href"].removeprefix("./").removesuffix(".json") for link in d["links"] if link["rel"] == "item")
    scale = int(items[0].split("_")[1]) if items else 0
    return {"bbox": tuple(d["extent"]["spatial"]["bbox"][0]), "gsd": float(d.get("gsd") or 0), "title": d.get("title", ""),
            "items": frozenset(items), "scale": scale, "root": f"{NZ_BUCKET}/{href.rsplit('/', 1)[0]}"}


def nz_tiles_over(bbox, scale: int) -> list[str]:
    """LINZ tile names at ``scale`` covering a WGS84 bbox (geoblock_core.global_dem.nz_tile)."""
    from pyproj import Transformer
    from geoblock_core.global_dem import NZ_SHEET_HEIGHT, NZ_SHEET_WIDTH, nz_tile
    to = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    xs, ys = to.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    per = 50_000 // scale
    tw, th = NZ_SHEET_WIDTH / per, NZ_SHEET_HEIGHT / per
    out = []
    for y in np.arange(max(ys), min(ys) - th, -th):
        for x in np.arange(min(xs), max(xs) + tw, tw):
            name = nz_tile(min(x, max(xs)), max(y, min(ys)), scale)
            if name not in out:
                out.append(name)
    return out


def linz_rgba(bbox, pixels: int) -> np.ndarray:
    from rasterio.enums import Resampling
    _, height, width = _grid(bbox, pixels)
    out = np.zeros((height, width, 4), np.uint8)
    mpp = (bbox[3] - bbox[1]) * 111_320 / pixels
    used = 0
    for href in nz_candidates(bbox):
        if used == NZ_COLLECTIONS or (out[..., 3] > 0).all():
            break
        c = nz_collection(href)
        if not _intersects(c["bbox"], bbox) or not c["scale"]:
            continue
        names = [n for n in nz_tiles_over(bbox, c["scale"]) if n in c["items"]]
        if not names:
            continue
        used += 1
        print(f"  linz-nz: {c['title']}, {len(names)} tiles", flush=True)
        for name in names:
            rgb, mask = _warp(f"/vsicurl/{c['root']}/{name}.tiff", bbox, pixels, (1, 2, 3), Resampling.bilinear,
                              mpp, c["gsd"] or 0.25)
            take = (mask > 0) & (out[..., 3] == 0)
            out[take, :3] = np.moveaxis(rgb, 0, -1)[take]
            out[take, 3] = 255
    return out

