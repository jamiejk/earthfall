import json
from datetime import date

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from geoblock_core import imagery


def swath(width: int, height: int, offset: int) -> Image.Image:
    """Green land with a diagonal swath edge: black below the line x + y = offset."""
    pixels = np.full((height, width, 3), (60, 120, 40), dtype=np.uint8)
    rows, cols = np.indices((height, width))
    pixels[cols + rows * width / height < offset] = 0
    return Image.fromarray(pixels)


def test_hls_dates_are_stacked_until_the_swath_gap_is_filled(monkeypatch):
    calls = []

    def fake_wms(layer, bbox, width, height, when=None, client=None):
        calls.append((layer, when))
        if layer == imagery.BLUE_MARBLE:
            return Image.new("RGB", (width, height), (10, 20, 200))
        # Each date covers a different part; the most recent has the black diagonal gap on its lower left.
        offset = {0: width // 2, 5: -1}.get((date(2025, 9, 30) - when).days, width * 3)
        return swath(width, height, offset)

    monkeypatch.setattr(imagery, "wms_image", fake_wms)
    image, dates, usable = imagery.clearest_hls((0, 0, 1, 1), 128, 128, end=date(2025, 9, 30), days=20)
    pixels = np.asarray(image)
    assert usable == 1.0 and (pixels.mean(axis=2) > 20).all()  # no black left
    assert (pixels == (60, 120, 40)).all(axis=2).mean() > 0.99
    assert dates == [f"{imagery.HLS}:2025-09-25"]  # the full date adds the most; nothing else was needed
    assert imagery.BLUE_MARBLE not in [layer for layer, _ in calls]


def test_gaps_no_date_fills_come_from_blue_marble(monkeypatch):
    monkeypatch.setattr(imagery, "wms_image", lambda layer, bbox, width, height, when=None, client=None:
                        Image.new("RGB", (width, height), (10, 20, 200)) if layer == imagery.BLUE_MARBLE
                        else swath(width, height, width // 2))
    image, dates, usable = imagery.clearest_hls((0, 0, 1, 1), 128, 128, end=date(2025, 9, 30), days=10)
    pixels = np.asarray(image)
    assert 0.8 < usable < 0.95 and len(dates) == 1  # every date is the same swath: one is enough
    assert (pixels.mean(axis=2) > 20).all()  # the gap is Blue Marble, matched to the HLS, not black


def test_naip_patch_keeps_its_brightness_and_fades_into_the_surrounding_imagery(tmp_path, monkeypatch):
    raster = tmp_path / "landing.tif"
    with rasterio.open(raster, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32",
                       crs="EPSG:26911", transform=from_origin(1000, 2000, 80, 80)) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="float32"))
    # Surrounding HLS: 1600 m square, the landing's 800 m in the middle; brownish there, blue outside.
    around = np.zeros((64, 64, 3), dtype=np.uint8)
    around[...] = (0, 0, 255)
    around[16:48, 16:48] = (150, 120, 90)
    Image.fromarray(around).save(tmp_path / "around.png")
    reference = {"key": "r", "texture": str(tmp_path / "around.png"), "bounds_projected_m": [600, 800, 2200, 2400]}
    seen = {}

    def fake_naip(bounds, epsg, pixels, client=None):
        seen.update(bounds=bounds, epsg=epsg)
        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(30, 60, (pixels, pixels, 3), dtype=np.uint8))  # dark, unmatched

    monkeypatch.setattr(imagery, "naip_image", fake_naip)
    manifest = imagery.fetch_naip_imagery({"terrain_raster": str(raster)}, tmp_path / "out", pixels=64,
                                          reference=reference)
    assert seen == {"bounds": (1000.0, 1200.0, 1800.0, 2000.0), "epsg": 26911}
    texture = np.asarray(Image.open(manifest["texture"])).astype(float)
    assert np.allclose(texture[16:48, 16:48].reshape(-1, 3).mean(axis=0), 44.5, atol=2)  # NAIP's own values
    assert np.allclose(texture[0].mean(axis=0), (150, 120, 90), atol=6)  # the edge is the surrounding colour
    assert manifest["faded_into"] and "NAIP" in manifest["credit"]
    assert json.loads((tmp_path / "out" / "imagery.json").read_text())["key"] == manifest["key"]


def test_later_dates_are_colour_matched_and_feathered_across_the_swath_edge():
    height = width = 200
    rows, cols = np.indices((height, width))
    first_covers = cols + rows < 200
    stack = np.zeros((height, width, 3), dtype=np.uint8)
    stack[first_covers] = (60, 120, 40)  # green date
    later = np.full((height, width, 3), (120, 80, 50), dtype=np.uint8)  # browner, darker date of the same ground
    fill = ~first_covers
    pixels = imagery.blend_swath(stack, later, first_covers, fill, feather_px=8)
    assert np.allclose(pixels[fill].mean(axis=0), (60, 120, 40), atol=2)
    stack[fill] = pixels[fill]
    step = np.abs(stack[100, 101].astype(int) - stack[100, 99].astype(int)).max()  # across the old boundary
    assert step <= 3


def test_dates_are_picked_by_the_new_ground_they_add(monkeypatch):
    # Three dates: two cover the same left half (cleanest, 50% each), one covers the right 40%.
    def fake_wms(layer, bbox, width, height, when=None, client=None):
        if layer == imagery.BLUE_MARBLE:
            return Image.new("RGB", (width, height), (10, 20, 200))
        if layer != imagery.HLS:
            raise ValueError("not served")
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        day = when.day
        if day in (30, 25):
            pixels[:, : width // 2] = (60, 120, 40)
        elif day == 20:
            pixels[:, int(width * 0.6):] = (70, 110, 50)
        return Image.fromarray(pixels)

    monkeypatch.setattr(imagery, "wms_image", fake_wms)
    _, dates, usable = imagery.clearest_hls((0, 0, 1, 1), 128, 128, end=date(2025, 9, 30), days=15)
    assert dates == [f"{imagery.HLS}:2025-09-30", f"{imagery.HLS}:2025-09-20"]
    assert 0.85 < usable < 0.92


def test_ground_shares_split_rock_trees_grass_and_shadow():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:20] = (200, 198, 190)  # pale granite
    image[20:30] = (40, 60, 30)  # dark conifers
    image[30:35] = (110, 160, 70)  # meadow
    image[35:] = (20, 30, 45)  # blue cliff shadow: left out
    shares = imagery.ground_shares(image)
    assert shares == {"trees": 0.286, "grass": 0.143, "rock": 0.571, "scrub": 0.0, "earth": 0.0}
    assert imagery.describe_ground(shares) == ("mostly bare pale rock slabs and boulders, with dark conifers and "
                                               "shrubs and some green grass")


def test_measured_ground_underfoot_replaces_the_worldcover_words():
    from geoblock_core.landcover import with_ground_underfoot
    from geoblock_core.looks import render_to_real_prompt
    look = {"all": "the cliff-top overlook ground underfoot at {name} is mostly forest; across the view it is rock",
            "high": "the ground near the camera at {name} is mostly forest", "low": "below is forest",
            "far": "the land fades"}
    look = with_ground_underfoot(look, "bare pale rock slabs")
    assert look["all"] == "the ground underfoot at {name} is bare pale rock slabs; across the view it is rock"
    assert look["high"] == "the ground near the camera at {name} is bare pale rock slabs"
    prompt = render_to_real_prompt("Glacier Point", "default", "", "", "Earth zoom", look=look, weather="clear")
    assert "one person standing at the viewpoint on bare pale rock slabs" in prompt


def test_tree_proxy_words_use_the_measured_ground_between_trees():
    from geoblock_core.looks import canopy_text
    summary = {"cover": 0.3, "height_p25_m": 8, "height_p75_m": 20, "height_p95_m": 30, "tall_share": 0.0}
    assert "and bare pale rock slabs between them" in canopy_text(summary, "default", "bare pale rock slabs")
    assert "grass" in canopy_text(summary, "default")


def test_mapped_roads_are_paved_not_rock():
    image = np.full((20, 20, 3), 200, dtype=np.uint8)  # all pale and colourless: rock to the photo alone
    paved = np.zeros((20, 20), dtype=bool)
    paved[:, :12] = True
    shares = imagery.ground_shares(image, paved)
    assert shares["paved"] == 0.6 and shares["rock"] == 0.4
    assert imagery.describe_ground(shares).startswith("mostly paved paths and roads, with bare pale rock")


def test_tan_earth_and_grey_green_scrub_are_not_rock():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:10] = (115, 108, 97)  # tan dry earth: faintly tinted, not pale
    image[10:] = (82, 90, 86)  # grey-green coastal scrub
    shares = imagery.ground_shares(image)
    assert shares["rock"] == 0 and shares["earth"] == 0.5 and shares["scrub"] == 0.5
