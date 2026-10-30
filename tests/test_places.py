import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geoblock_core.landcover import describe, look_from_landcover, tile_name
from geoblock_core.lidar import area_km2, contains, survey_year, surveys_at
from geoblock_core.looks import look_text, render_to_real_prompt, scene_palette
from geoblock_core.places import (bearing_deg, catalog_option, catalog_report, find_place, place_for_pipeline,
                                  place_report, places, view_heading)
from geoblock_core.providers import CopernicusDEMProvider, USGS3DEPProvider
from geoblock_core.run import main
from geoblock_core.scene import build_landform_scene, terrain_family, wgs84_utm_crs
from geoblock_core.workflow import ShotRequest, blocking_graph

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_is_measured_and_unique():
    ids = [place["id"] for place in places()]
    assert len(ids) == len(set(ids)) >= 40
    for place in places():
        coverage = place.get("coverage", {})
        if place["id"].startswith("custom-") or not coverage.get("terrain"):
            continue  # a custom point registered for this process by another test, not a catalogue entry
        assert coverage["terrain"] in ("usgs3dep", "copernicus"), place["id"]
        if not place.get("measured", True):
            # Listed, but its terrain evidence has not been measured into the catalogue yet (Bryant Park was added
            # for the retreat experiment). It is not offered for a build until it is measured.
            continue
        assert coverage["relief_m"] > 0 and coverage["look"]["all"], place["id"]
        assert "hls" in coverage["imagery"]
        if coverage["terrain"] == "usgs3dep":
            assert "naip" in coverage["imagery"]


def test_catalog_heights_match_known_summits():
    # Published summit heights; 30 m and 10 m grids smooth sharp tops a little.
    for place_id, height in (("mount-whitney", 4421), ("half-dome-summit", 2693), ("mount-fuji", 3776),
                             ("kilimanjaro-uhuru", 5895), ("ben-nevis", 1345), ("table-mountain", 1067)):
        assert find_place(place_id)["coverage"]["elevation_m"] == pytest.approx(height, abs=40), place_id


def test_find_place_by_id_or_unique_words():
    assert find_place("gornergrat-matterhorn")["name"] == "Gornergrat"
    assert find_place("mather")["id"] == "grand-canyon-mather-point"
    with pytest.raises(KeyError, match="several"):
        find_place("yosemite")
    with pytest.raises(KeyError, match="No catalog place"):
        find_place("atlantis")


def test_view_heading_faces_the_named_view():
    assert view_heading(find_place("gornergrat")) == pytest.approx(265, abs=2)  # the Matterhorn is west
    assert view_heading(find_place("mather")) == -1
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0)
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90)


def test_pipeline_place_uses_the_moved_landing_and_names_the_view():
    place = find_place("table-mountain")
    chosen = place_for_pipeline(place)
    assert (chosen["latitude"], chosen["longitude"]) == (place["coverage"]["landing"]["latitude"],
                                                         place["coverage"]["landing"]["longitude"])
    zermatt = place_for_pipeline(find_place("gornergrat"))
    assert zermatt["look"]["far"].startswith("the Matterhorn stands out")
    assert zermatt["catalog_id"] == "gornergrat-matterhorn" and zermatt["heading_deg"] > 0
    assert place_for_pipeline(find_place("mather"))["palette"] == "grand-canyon"


def test_reports_list_sources_and_licences():
    report = place_report(find_place("mather"))
    assert "AZ_GrandCanyonNP_1_2019" in report and "Public domain" in report and "NAIP" in report
    assert "swissALTI3D" in place_report(find_place("gornergrat"))
    assert "Copernicus DEM licence" in place_report(find_place("gornergrat"))
    assert catalog_report("utah").startswith("6 of")


def test_terrain_family_and_utm_zone():
    assert terrain_family(36.06, -112.1) == "usgs3dep"
    assert terrain_family(45.98, 7.78) == "copernicus"
    assert terrain_family(20.7, -156.25) == "copernicus"  # Hawaii: 3DEP seamless is CONUS-only here
    assert wgs84_utm_crs(7.78, 45.98) == "EPSG:32632"
    assert wgs84_utm_crs(167.9, -44.67) == "EPSG:32758"


def test_copernicus_tile_names_use_south_west_corner():
    assert CopernicusDEMProvider.tile_names(7.5, 45.5, 8.2, 45.9) == [
        "Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "Copernicus_DSM_COG_10_N45_00_E008_00_DEM"]
    assert CopernicusDEMProvider.tile_names(167.8, -44.8, 167.9, -44.6) == ["Copernicus_DSM_COG_10_S45_00_E167_00_DEM"]
    assert tile_name(45.98, 7.78) == "ESA_WorldCover_10m_2021_v200_N45E006_Map"
    assert tile_name(-44.6, 167.9) == "ESA_WorldCover_10m_2021_v200_S45E165_Map"


LAT, LON = 45.9836, 7.7853


def copernicus_tiles(root: Path) -> Path:
    """One land tile with a ridge at the landmark; its eastern neighbour is missing (sea)."""
    name = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"
    step = 1 / 3600
    rows, cols = np.indices((3600, 3600))
    lon = 7 + (cols + 0.5) * step
    heights = 3000 - 2000 * np.abs(lon - LON)  # a north-south ridge through the landmark
    (root / name).mkdir(parents=True)
    with rasterio.open(root / name / f"{name}.tif", "w", driver="GTiff", width=3600, height=3600, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(7, 46, step, step)) as dst:
        dst.write(heights.astype("float32"), 1)
    (root / "tileList.txt").write_text(name + "\n")
    return root


def test_copernicus_scene_is_egm2008_on_wgs84_utm(tmp_path):
    root = copernicus_tiles(tmp_path / "cop")
    scene = build_landform_scene(name="Ridge", latitude=LAT, longitude=LON, radius_m=1000,
                                 output_root=tmp_path / "scenes", raw_cache=tmp_path / "raw",
                                 location_source="test", provider=CopernicusDEMProvider(str(root)))
    assert scene["projected_crs"] == "EPSG:32632" and scene["vertical_datum"] == "EGM2008"
    assert scene["terrain_family"] == "copernicus" and scene["source_type"] == "DSM"
    assert scene["target_height_m"] == pytest.approx(3000, abs=5)
    assert scene["height_alignment"] == "verified"


def test_copernicus_fills_missing_sea_tiles(tmp_path):
    root = copernicus_tiles(tmp_path / "cop")
    scene = build_landform_scene(name="Coast", latitude=LAT, longitude=7.998, radius_m=1000,
                                 output_root=tmp_path / "scenes", raw_cache=tmp_path / "raw",
                                 location_source="test", provider=CopernicusDEMProvider(str(root)))
    with rasterio.open(scene["terrain_raster"]) as dem:
        heights = dem.read(1, masked=True)
    assert heights.mask.sum() == 0 and float(heights.min()) == pytest.approx(0)


def test_3dep_regional_scene_fills_offshore_tiles(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_scene_camera import LAT as RIM_LAT, LON as RIM_LON, write_tile
    root = tmp_path / "tiles"
    write_tile(root, "n37w113", -112.1, RIM_LAT + 0.05)  # reaches just past -112
    # The scene reaches into n37w112 too, which has no tile: treated as sea, not an error.
    scene = build_landform_scene(name="Edge", latitude=RIM_LAT, longitude=-111.9995, radius_m=1500,
                                 output_root=tmp_path / "s", raw_cache=tmp_path / "r", location_source="test",
                                 provider=USGS3DEPProvider(str(root)))
    assert scene["elevation_range_m"][0] == pytest.approx(0)


def test_lidar_index_lookup():
    square = {"type": "MultiPolygon", "coordinates": [[[[-112.2, 36.0], [-112.0, 36.0], [-112.0, 36.2],
                                                        [-112.2, 36.2], [-112.2, 36.0]]]]}
    assert contains(square, -112.1, 36.1) and not contains(square, -111.9, 36.1)
    assert area_km2(square) == pytest.approx(0.2 * 111.32 * 0.809 * 0.2 * 110.57, rel=0.02)
    index = [{"geometry": square, "properties": {"name": "AZ_Test_2019", "count": 10_000_000_000,
                                                 "url": "https://example/ept.json"}},
             {"geometry": square, "properties": {"name": "AZ_Old_2011", "count": 1, "url": "x"}}]
    found = surveys_at(36.1, -112.1, Path("/nonexistent"), index=index)
    assert [survey["name"] for survey in found] == ["AZ_Test_2019", "AZ_Old_2011"]
    assert found[0]["density_pts_m2"] == pytest.approx(10e9 / (area_km2(square) * 1e6), rel=0.01)
    assert survey_year("USGS_LPC_MI_31Co_Chippewa_2017_LAS_2019") == 2017
    assert survey_year("CA_SierraNevada_11_B22") == 2022


def test_landcover_words():
    share = {"70": 0.55, "60": 0.3, "30": 0.1, "80": 0.05}
    assert describe(share) == ("mostly glacier ice and snowfields, with bare rock, scree and sand with sparse "
                               "plants and some grassland and meadow")
    look = look_from_landcover({"near": {10: 1.0}, "view": {80: 0.6, 10: 0.4}}, "sea cliff")
    assert "open water toward the horizon" in look["far"]
    assert look["all"].format(name="X").startswith("the sea cliff ground underfoot at X is mostly forest")


def test_prompts_use_measured_look_unless_a_palette_is_curated():
    look = {"all": "A {name}", "high": "H {name}", "low": "L {name}", "far": "F"}
    assert look_text("default", "Here", "low", look=look) == "L Here"
    assert "South Rim" in look_text("grand-canyon", "Here", "high", look=look)
    assert "L Peak" in render_to_real_prompt("Peak", "default", "", "Sunny.", "Earth zoom", look=look)
    assert scene_palette("default", {"palette": "grand-canyon"}) == "grand-canyon"
    assert scene_palette("default", {"palette": "default"}) == "default"


def test_catalog_place_drives_the_graph_and_cli(tmp_path, capsys):
    graph = blocking_graph(ShotRequest(name="Gornergrat", place_id="gornergrat-matterhorn"))
    assert graph["2"]["class_type"] == "GeoBlockPlaceFromCatalog"
    assert graph["2"]["inputs"]["place"] == catalog_option("gornergrat-matterhorn")
    path = tmp_path / "graph.json"
    assert main(["--place", "mather", "--preset", "Earth zoom", "--ltx", "--save-graph", str(path)]) == 0
    saved = json.loads(path.read_text())
    assert saved["2"]["inputs"]["place"].startswith("grand-canyon-mather-point")
    assert saved["5"]["inputs"]["palette"] == "grand-canyon"
    assert main(["--place", "gornergrat", "--save-graph", str(path)]) == 0
    assert json.loads(path.read_text())["4"]["inputs"]["heading_degrees"] == pytest.approx(265, abs=2)
    assert main(["--list-places", "norway"]) == 0
    assert "preikestolen" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--place", "atlantis", "--save-graph", str(path)])


def test_near_view_sees_the_drop_only_from_the_brink():
    from geoblock_core.sightline import near_view_deg
    # A flat top at 2000 m that ends 30 m east of the person in a 500 m cliff.
    step = 1 / 111_320 / 0.7  # about 1 m at this latitude
    lon0, lat0 = -119.0 - 100 * step, 45.6 + 100 * step
    cols = np.arange(400)
    lon = lon0 + (cols + 0.5) * step
    east_m = (lon - -119.0) * 111_320 * np.cos(np.radians(45.6))
    heights = np.repeat(np.where(east_m < 30, 2000.0, 1500.0)[None, :], 200, axis=0)
    transform = from_origin(lon0, lat0, step, 1 / 110_574)
    back = near_view_deg(heights, transform, (-119.0, 45.6), 90)
    brink = near_view_deg(heights, transform, (-119.0 + 28 / (111_320 * np.cos(np.radians(45.6))), 45.6), 90)
    assert -5 < back < 0 and brink < -25


def test_lidar_brink_is_used_only_with_the_landing_patch(tmp_path):
    glacier = find_place("yosemite-glacier-point")
    brink = glacier["coverage"]["lidar_landing"]
    chosen = place_for_pipeline(glacier)
    assert (chosen["latitude"], chosen["longitude"]) == (glacier["latitude"], glacier["longitude"])
    assert chosen["lidar_landing"]["latitude"] == brink["latitude"] and 5 <= brink["moved_m"] <= 40
    assert "CA_YosemiteNP_2019_D19" in place_report(glacier) and "LIDAR LANDING" in place_report(glacier)
    path = tmp_path / "graph.json"
    assert main(["--place", "glacier point", "--preset", "Earth zoom", "--landing-lidar", "auto",
                 "--save-graph", str(path)]) == 0
    assert json.loads(path.read_text())["3"]["inputs"]["landing_lidar"] == "auto"
    assert place_for_pipeline(find_place("mather"))["lidar_landing"] is None


def test_1m_gaps_fill_from_the_seamless_dem(tmp_path, monkeypatch):
    from geoblock_core.models import GeoSceneSpec
    from geoblock_core.providers import USGS1mDEMProvider
    # One 1 m tile whose eastern third is outside the project (nodata), on NAD83 / UTM 11N.
    root = tmp_path / "1m"
    project, x, y = "CA_Test_2020", 273_000.0, 4_179_000.0
    tif = root / project / "TIFF" / f"USGS_1M_11_x27y418_{project}.tif"
    tif.parent.mkdir(parents=True)
    data = np.full((10_000, 10_000), 2000.0, dtype="float32")
    data[:, 3_400:] = -999999.0
    with rasterio.open(tif, "w", driver="GTiff", width=10_000, height=10_000, count=1, dtype="float32",
                       crs="EPSG:26911", transform=from_origin(270_000, 4_180_000, 1, 1), nodata=-999999.0,
                       tiled=True, compress="deflate") as dst:
        dst.write(data, 1)
    monkeypatch.setattr(USGS1mDEMProvider, "seamless_fill",
                        staticmethod(lambda spec, cache, transform, shape, crs: np.full(shape, 1990.0, "float32")))
    spec = GeoSceneSpec(project_id="t", name="t", longitude=-119.57, latitude=37.73, projected_crs="EPSG:26911",
                        origin_easting_m=x, origin_northing_m=y, radius_m=400, location_source="test")
    raster, record = USGS1mDEMProvider(project, 11, root=str(root)).acquire(spec, tmp_path / "raw")
    with rasterio.open(raster) as dem:
        heights = dem.read(1)
    assert set(np.unique(heights)) == {1990.0, 2000.0} and "filled from the 1/3 arc-second DEM" in record.dataset
    far = GeoSceneSpec(**{**spec.model_dump(), "origin_easting_m": 273_500.0, "project_id": "f"})
    # The 1 m project does not reach this square (its eastern third is nodata and the window straddles the edge),
    # so the whole square falls back to the 1/3 arc-second DEM. That is deliberate (Battery Spencer, 18 Oct: most
    # places have no 1 m lidar), and the record has to say so: the ground is then 10 m, not 1 m.
    raster, record = USGS1mDEMProvider(project, 11, root=str(root)).acquire(far, tmp_path / "raw")
    with rasterio.open(raster) as dem:
        heights = dem.read(1)
    assert set(np.unique(heights)) == {1990.0}
    assert "1/3 arc-second DEM" in record.dataset and record.nominal_resolution_m == 10


def test_figure_moves_off_a_step_the_camera_cannot_see_over(monkeypatch):
    from geoblock_core import zoom
    # Looking north: the path (y < -1.5) is 2.5 m above a rock step where the landmark point sits.
    monkeypatch.setattr(zoom, "mesh_height", lambda scene, x, y: 2000.0 if y < -1.5 else 1997.5)
    x, y = zoom.standing_spot({}, np.array([0.0, 1.0]))
    assert y < -1.5 - 0.6 and math.hypot(x, y) <= 2.5  # back on the path, level underfoot, nearest such point
    monkeypatch.setattr(zoom, "mesh_height", lambda scene, x, y: 2000.0)
    assert zoom.standing_spot({}, np.array([0.0, 1.0])) == (0.0, 0.0)
