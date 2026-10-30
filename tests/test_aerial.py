import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("cv2")
from touchdown import aerial  # noqa: E402

BBOX = (7.70, 45.95, 7.80, 46.02)  # west, south, east, north


def _lonlat(gx, gy, z):
    n = 256 * 2 ** z
    lon = gx / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * gy / n))))
    return lon, lat


def coded_tiles(missing=lambda lon: False):
    """Tiles whose red channel encodes longitude and green latitude across BBOX, so the resampling can be checked."""
    def get(url):
        z, x, y = map(int, url.split("/")[1:4])
        gx, gy = np.meshgrid(np.arange(256) + x * 256 + 0.5, np.arange(256) + y * 256 + 0.5)
        lon, lat = _lonlat(gx, gy, z)
        if missing(lon.mean()):
            return None
        w, s, e, n = BBOX
        tile = np.zeros((256, 256, 4), np.uint8)
        tile[..., 0] = np.clip((lon - w) / (e - w) * 255, 0, 255)
        tile[..., 1] = np.clip((lat - s) / (n - s) * 255, 0, 255)
        tile[..., 2] = 128
        tile[..., 3] = 255
        return tile
    return get


def test_tiles_land_where_they_belong():
    src = aerial.Source("t", "test", "XX", (BBOX,), "t/{z}/{x}/{y}", max_zoom=19)
    out = aerial.tiles_rgba(src, BBOX, 256, get_tile=coded_tiles())
    h, w = out.shape[:2]
    assert h == 256 and w == round(256 * 0.10 / 0.07)
    cols = (np.arange(w) + 0.5) / w * 255
    rows = 255 - (np.arange(h) + 0.5) / h * 255
    assert np.abs(out[h // 2, :, 0].astype(float) - cols)[2:-2].max() < 3
    assert np.abs(out[:, w // 2, 1].astype(float) - rows)[2:-2].max() < 3
    assert (out[..., 3] == 255).all()


def test_zoom_matches_the_square():
    assert aerial.zoom_for(0.15, 46.0, 20) == 20
    assert aerial.zoom_for(0.15, 46.0, 18) == 18
    z = aerial.zoom_for(36.0, 46.0, 19)
    px = aerial.EARTH_M * math.cos(math.radians(46.0)) / 256 / 2 ** z
    assert px <= 36.0 < 2 * px


def test_chain_fills_what_the_first_source_lacks(monkeypatch):
    mid = (BBOX[0] + BBOX[2]) / 2
    a = aerial.Source("a", "half", "XX", (BBOX,), "a/{z}/{x}/{y}")
    g = aerial.Source("g", "global", "", ((-180, -85, 180, 85),), "g/{z}/{x}/{y}")
    monkeypatch.setattr(aerial, "SOURCES", {"a": a, "g": g})
    monkeypatch.setattr(aerial, "FALLBACK", "g")
    half = coded_tiles(missing=lambda lon: lon > mid)
    full = coded_tiles()
    out, cover = aerial.square_rgba("a", BBOX, 128, get_tile=lambda url: (half if url[0] == "a" else full)(url))
    assert (out[..., 3] == 255).all()
    assert 0.3 < cover["a"] < 0.7 and cover["a"] + cover["g"] == pytest.approx(1, abs=0.01)


def test_pick_by_country_then_by_point():
    assert aerial.pick("CH") == "swissimage-ch"
    assert aerial.pick("US") == "naip"
    assert aerial.pick("", 36.74, -5.17) == "pnoa-es"
    assert aerial.pick("IT", 46.63, 12.31) == "bz-it"  # Tre Cime: South Tyrol orthophoto (18 Oct)
    assert aerial.pick("PE", -13.16, -72.54) == "s2-l2a"  # Machu Picchu: no national source; open Sentinel-2 fill
    assert aerial.pick("NZ") == "linz-nz"  # the open bucket, not the keyed Basemaps API


def test_flat_snow_is_kept_but_flat_black_or_white_is_not():
    import io

    from PIL import Image

    class R:
        def __init__(self, rgb):
            buf = io.BytesIO()
            Image.new("RGB", (256, 256), rgb).save(buf, "PNG")
            self.content, self.status_code, self.headers = buf.getvalue(), 200, {"content-type": "image/png"}

        def raise_for_status(self):
            pass

    class C:
        def __init__(self, rgb):
            self.rgb = rgb

        def get(self, url):
            return R(self.rgb)

    assert aerial._get_tile(C((252, 252, 253)), "u") is not None
    assert aerial._get_tile(C((0, 0, 0)), "u") is None
    assert aerial._get_tile(C((255, 255, 255)), "u") is None


def test_dropped_tiles_are_asked_for_again():
    src = aerial.Source("t", "test", "XX", (BBOX,), "t/{z}/{x}/{y}", max_zoom=19)
    good, seen = coded_tiles(), set()

    def flaky(url):  # every tile fails the first time it is asked for
        if url not in seen:
            seen.add(url)
            return None if url.endswith("0") else good(url)
        return good(url)
    out = aerial.tiles_rgba(src, BBOX, 256, get_tile=flaky)
    assert (out[..., 3] == 255).all()


def test_backdrop_neighbour_only_fills_coarse_squares():
    bbox = (6.80, 45.85, 6.95, 45.92)  # Aiguille du Midi: touches the Swiss box
    assert "swissimage-ch" not in [s.id for s in aerial.chain("ign-fr", bbox, 0.44)]
    assert "swissimage-ch" in [s.id for s in aerial.chain("ign-fr", bbox, 12.0)]
