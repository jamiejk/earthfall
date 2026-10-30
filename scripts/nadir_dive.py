"""Grounded nadir dive: a straight-down camera falling from deep space to a few hundred metres above a real place,
drawn from real public-domain imagery at every scale, with a procedural cloud deck to punch through.

Why: LTX will not fly between two pinned stills that are far apart; it dissolves (fakeout ladder, zoom stills).
A straight-down view of real imagery needs no model at all: it is exactly what a camera looking down would see, so
the dive has true motion at every frame and cannot cross-fade. Only the last part (tilting up from straight down
to eye level, onto the person) is left to LTX.

Imagery (fetched once, cached in the output folder): Blue Marble for the globe and a 1,000 km bridge square, HLS (30 m) for 200 and 60 km
squares, NAIP (0.6 m) for 24, 8 and 2.4 km squares. Each finer square is colour-matched to the one above at low
frequency and feathered at its edge; per frame every square is sampled trilinearly at the screen's footprint,
so no square ever pops in.

  uv run python scripts/nadir_dive.py --place san-francisco-twin-peaks --output-dir outputs/runs/fakeout/nadir
  uv run python scripts/nadir_dive.py --synthetic        # offline test, no downloads
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

R = 6_371_000.0
ATMOSPHERE = 80_000.0
HAZE = np.array([0.60, 0.74, 0.96], dtype=np.float32)
# (seconds, metres): log-altitude keyframes, monotone-interpolated. Slow from deep space, fast through orbit
# heights, through the cloud deck briskly (dada893 held pure white for 9 frames at 72), then easing toward the hand-off at 400 m.
PATH = ((0.0, 12_000_000), (1.4, 600_000), (2.55, 8_000), (2.8, 5_800), (3.9, 1_100), (4.6, 400))
# Imagery squares: (name, width km, pixels, source).
LEVELS = (("bm1000", 1000, 2048, "bluemarble"), ("hls200", 200, 4096, "hls"), ("hls60", 60, 2048, "hls"), ("naip24", 24, 4096, "naip"),
          ("naip8", 8, 4096, "naip"), ("naip2", 2.4, 4096, "naip"))


# --- imagery -----------------------------------------------------------------------------------------------

def _square(lat: float, lon: float, width_m: float) -> tuple[float, float, float, float]:
    half_lat = width_m / 2 / 111_320
    half_lon = width_m / 2 / (111_320 * math.cos(math.radians(lat)))
    return lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat


def _naip(bbox, pixels: int) -> np.ndarray:
    """NAIP RGBA in EPSG:4326 (plate carree, ``pixels`` rows), in tiles of <= 2048 px. Each request's size has
    the bbox's own aspect in degrees: the image service widens a bbox whose aspect differs from the size, which
    misregistered the squares against each other (bf78a66: doubled roads and reservoir at 800-1,500 m)."""
    import httpx

    from geoblock_core.imagery import NAIP_SERVICE, USER_AGENT
    west, south, east, north = bbox
    height = pixels
    width = round(pixels * (east - west) / (north - south))
    rows, cols = math.ceil(height / 2048), math.ceil(width / 2048)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=240) as client:
        for row in range(rows):
            for col in range(cols):
                y0, y1 = height * row // rows, height * (row + 1) // rows
                x0, x1 = width * col // cols, width * (col + 1) // cols
                tb = (west + (east - west) * x0 / width, north - (north - south) * y1 / height,
                      west + (east - west) * x1 / width, north - (north - south) * y0 / height)
                params = {"bbox": ",".join(f"{v:.8f}" for v in tb), "bboxSR": 4326, "imageSR": 4326,
                          "size": f"{x1 - x0},{y1 - y0}", "format": "png", "interpolation": "RSP_BilinearInterpolation",
                          "f": "image"}
                for attempt in range(5):
                    try:
                        response = client.get(NAIP_SERVICE, params=params)
                        response.raise_for_status()
                        if not response.headers.get("content-type", "").startswith("image/"):
                            raise ValueError(response.text[:200])
                        break
                    except (httpx.HTTPError, ValueError):
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                tile = Image.open(io.BytesIO(response.content)).convert("RGBA")
                if tile.size != (x1 - x0, y1 - y0):
                    raise ValueError(f"NAIP returned {tile.size} for a {(x1 - x0, y1 - y0)} request")
                out[y0:y1, x0:x1] = np.asarray(tile)
    rgb = out[..., :3].astype(np.int32)
    out[..., 3] = np.where((out[..., 3] > 0) & (rgb.max(axis=2) > 6), 255, 0)  # nodata is black or transparent
    return out


def _water_mask(lat: float, lon: float, width_m: float) -> Image.Image:
    """ESA WorldCover 10 m permanent water (class 80) over the square, white = water."""
    from geoblock_core.landcover import read_classes
    classes = read_classes(lat, lon, width_m / 2, pixels=min(4096, round(width_m / 10)))
    return Image.fromarray(np.where(classes == 80, 255, 0).astype(np.uint8), "L")


WATER = np.array([8, 22, 42], dtype=np.float32)  # deep blue; the haze lifts it from altitude like Blue Marble's sea


def _flat_water(rgba: np.ndarray, water_path: str) -> np.ndarray:
    """Paint water one deep blue, with a soft (~16 px) shoreline: the HLS sea had a straight swath edge and sun
    glint, NAIP's had mosaic seams, and the same water in two squares never matched."""
    h, w = rgba.shape[:2]
    water = Image.open(water_path).convert("L").resize((max(1, w // 16), max(1, h // 16)), Image.BOX)
    water = np.clip(np.asarray(water.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255, 0, 1)[..., None]
    out = rgba.copy()
    out[..., :3] = (rgba[..., :3] * (1 - water) + WATER * water).astype(np.uint8)
    out[..., 3] = np.maximum(rgba[..., 3], (water[..., 0] * 255).astype(np.uint8))  # NAIP nodata at sea is sea
    return out


def fetch_imagery(lat: float, lon: float, folder: Path) -> dict:
    """Blue Marble globe + the LEVELS squares around (lat, lon), cached as PNGs with a manifest."""
    from geoblock_core.imagery import BLUE_MARBLE, CREDIT, clearest_hls, wms_image
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "nadir_imagery.json"
    key = [round(lat, 5), round(lon, 5), [list(level) for level in LEVELS], "naip-aspect-2", "water-2"]
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("key") == key:
        return json.loads(manifest_path.read_text())
    manifest = {"key": key, "latitude": lat, "longitude": lon, "levels": [], "credit": CREDIT,
                "sources": {"globe": BLUE_MARBLE, "hls": "NASA GIBS HLS", "naip": "USDA NAIP (USGS image service)"}}
    globe = folder / "globe_bluemarble.png"
    if not globe.is_file():
        wms_image(BLUE_MARBLE, (-180, -90, 180, 90), 8192, 4096).save(globe)
    manifest["globe"] = str(globe)
    for name, width_km, pixels, source in LEVELS:
        path = folder / (f"{name}_v2.png" if source == "naip" else f"{name}.png")
        bbox = _square(lat, lon, width_km * 1000)
        if not path.is_file():
            print(f"Fetching {name} ({source}, {width_km} km, {pixels} px)", flush=True)
            if source == "bluemarble":  # a bridge: takes the HLS colour inside, fades widely into the globe
                wms_image(BLUE_MARBLE, bbox, round(pixels * (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])),
                          pixels).convert("RGBA").save(path)
            elif source == "hls":
                image, used, share = clearest_hls(bbox, pixels, pixels)
                image.convert("RGBA").save(path)
                print(f"  HLS dates {used}, {share:.0%} HLS (rest Blue Marble)")
            else:
                Image.fromarray(_naip(bbox, pixels), "RGBA").save(path)
        entry = {}
        if source in ("naip", "hls"):  # water in both has glint, haze and hard swath/mosaic seams
            water = folder / f"{name}_water.png"
            if not water.is_file():
                _water_mask(lat, lon, width_km * 1000).save(water)
            entry["water"] = str(water)
        manifest["levels"].append({"name": name, "path": str(path), "bbox": list(bbox), **entry,
                                   "feather": 0.35 if source == "bluemarble" else 0.2})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --- textures ----------------------------------------------------------------------------------------------

class Level:
    """One georeferenced square (or the whole globe) with a mip chain, colour-matched to the level above."""

    def __init__(self, rgba: np.ndarray, bbox: tuple[float, float, float, float], globe: bool = False):
        self.bbox, self.globe = bbox, globe
        self.opaque = bool(globe or rgba[..., 3].min() == 255)  # before feathering: no nodata anywhere
        _, south, _, north = bbox
        self.metres_per_px = (north - south) * 111_320 / rgba.shape[0]
        self.mips = [rgba]
        while min(self.mips[-1].shape[:2]) > 8:
            m = self.mips[-1].astype(np.float32)
            m = m[: m.shape[0] // 2 * 2, : m.shape[1] // 2 * 2]
            m = (m[0::2, 0::2] + m[1::2, 0::2] + m[0::2, 1::2] + m[1::2, 1::2]) / 4
            self.mips.append(m.astype(np.uint8))

    def uv(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        west, south, east, north = self.bbox
        return (lon - west) / (east - west), (north - lat) / (north - south)

    def sample(self, u: np.ndarray, v: np.ndarray, lod: np.ndarray) -> np.ndarray:
        """Trilinear RGBA in 0..1 at texture coords u, v (0..1) and mip level lod (float, per pixel)."""
        lod = np.clip(lod, 0, len(self.mips) - 1.001)
        out = np.zeros(u.shape + (4,), dtype=np.float32)
        base = np.floor(lod).astype(np.int32)
        frac = (lod - base)[..., None]
        for level in np.unique(base):
            sel = base == level
            a = self._bilinear(self.mips[level], u[sel], v[sel])
            b = self._bilinear(self.mips[level + 1], u[sel], v[sel])
            out[sel] = a * (1 - frac[sel]) + b * frac[sel]
        return out

    def _bilinear(self, image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x = u * w - 0.5
        y = v * h - 0.5
        if self.globe:
            x = np.mod(x, w)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        x1 = (x0 + 1) % w if self.globe else np.clip(x0 + 1, 0, w - 1)
        x0 = x0 % w if self.globe else np.clip(x0, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        y0 = np.clip(y0, 0, h - 1)
        top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
        bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
        return ((top * (1 - fy) + bottom * fy) / 255).astype(np.float32)


def _towards(image: np.ndarray, reference: np.ndarray, weight: np.ndarray, block: int) -> np.ndarray:
    """Shift ``image``'s low-frequency colour (blocks of ``block`` px, smoothly upsampled) toward ``reference``'s
    on the same grid, by ``weight`` (0..1 per pixel), keeping ``image``'s detail. Only blocks where both are
    valid move; nodata never pulls colour toward black."""
    h, w = image.shape[:2]
    gh, gw = max(1, h // block), max(1, w // block)
    both = ((image[..., 3] == 255) & (reference[..., 3] == 255)).astype(np.float32)

    def low(x: np.ndarray) -> np.ndarray:
        return np.dstack([np.asarray(Image.fromarray(x[..., c]).resize((gw, gh), Image.BOX)) for c in range(x.shape[2])])

    m = low(both[..., None])
    shift = (low(reference[..., :3].astype(np.float32) * both[..., None]) -
             low(image[..., :3].astype(np.float32) * both[..., None])) / np.maximum(m, 1e-3)
    shift *= np.clip(m / 0.5, 0, 1)  # blocks with little valid overlap barely move
    shift = np.dstack([np.asarray(Image.fromarray(shift[..., c]).resize((w, h), Image.BICUBIC)) for c in range(3)])
    out = image.copy()
    out[..., :3] = np.clip(image[..., :3] + shift * weight[..., None], 0, 255).astype(np.uint8)
    return out


def _feather(rgba: np.ndarray, edge: float = 0.08) -> np.ndarray:
    """Alpha fades to 0 over the outer ``edge`` of the square (smoothstep), multiplied into nodata alpha."""
    h, w = rgba.shape[:2]
    ys, xs = np.linspace(0, 1, h), np.linspace(0, 1, w)
    dist = np.minimum.outer(np.minimum(ys, 1 - ys), np.minimum(xs, 1 - xs))
    t = np.clip(dist / edge, 0, 1)
    out = rgba.copy()
    out[..., 3] = (rgba[..., 3] * (t * t * (3 - 2 * t))).astype(np.uint8)
    return out


def _overlap(coarse: np.ndarray, cbox, fine: np.ndarray, fbox) -> tuple[np.ndarray, np.ndarray]:
    """The part of ``coarse`` under ``fine``'s bbox, and ``fine`` resized to it (both RGBA uint8)."""
    h, w = coarse.shape[:2]
    west, south, east, north = cbox
    x0 = int((fbox[0] - west) / (east - west) * w)
    x1 = math.ceil((fbox[2] - west) / (east - west) * w)
    y0 = int((north - fbox[3]) / (north - south) * h)
    y1 = math.ceil((north - fbox[1]) / (north - south) * h)
    part = coarse[max(y0, 0):y1, max(x0, 0):x1]
    small = np.asarray(Image.fromarray(fine, "RGBA").resize((part.shape[1], part.shape[0]), Image.BOX))
    return part, small


def _gain(target: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain taking ``target``'s mean to ``reference``'s where both are valid. Multiplicative only: a
    mean-and-spread match added +50 to the HLS (its contrast is lower than NAIP's over the city), which lifted
    the whole 200 km square, ocean included, to a grey box."""
    ok = (target[..., 3] == 255) & (reference[..., 3] == 255)
    if ok.sum() < 100:
        return np.ones(3, np.float32), np.zeros(3, np.float32)
    t = target[..., :3][ok].astype(np.float32)
    r = reference[..., :3][ok].astype(np.float32)
    scale = float(np.clip(r.mean() / max(t.mean(), 1.0), 0.8, 1.6))  # brightness only: per-channel gains from
    return np.full(3, scale, np.float32), np.zeros(3, np.float32)  # grey-blue city NAIP turned HLS land purple


def _feather_alpha(h: int, w: int, edge: float = 0.08) -> np.ndarray:
    ys, xs = np.linspace(0, 1, h), np.linspace(0, 1, w)
    t = np.clip(np.minimum.outer(np.minimum(ys, 1 - ys), np.minimum(xs, 1 - xs)) / edge, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def load_levels(manifest: dict) -> list[Level]:
    """Globe first, then squares coarse to fine. Colour always comes from the finer imagery, finest first: each
    coarser square takes the mean and spread of the one inside it (the dark GIBS HLS takes NAIP's colour), then,
    under that square, its low-frequency colour, tapered with the square's feathered edge so neither image has a
    seam. (3efc291 matched the other way, finer to coarser, which carried Blue Marble's dark green all the way
    down to the NAIP: the near-black ground.)"""
    globe_rgb = np.asarray(Image.open(manifest["globe"]).convert("RGBA"))
    raw = [(np.asarray(Image.open(e["path"]).convert("RGBA")).copy(), tuple(e["bbox"])) for e in manifest["levels"]]
    raw = [(_flat_water(rgba, e["water"]) if e.get("water") else rgba, bbox)
           for (rgba, bbox), e in zip(raw, manifest["levels"], strict=True)]
    for k in range(len(raw) - 2, -1, -1):
        (coarse, cbox), (fine, fbox) = raw[k], raw[k + 1]
        part, small = _overlap(coarse, cbox, fine, fbox)
        scale, offset = _gain(part, small)
        if manifest["levels"][k].get("feather", 0.08) > 0.2:  # the Blue Marble bridge keeps the globe's colour
            scale = np.ones(3, np.float32)
        coarse[..., :3] = np.clip(coarse[..., :3] * scale + offset, 0, 255).astype(np.uint8)
        part, small = _overlap(coarse, cbox, fine, fbox)
        weight = _feather_alpha(*part.shape[:2], manifest["levels"][k + 1].get("feather", 0.08)) * (small[..., 3] == 255)
        part[...] = _towards(part, small, weight, max(4, part.shape[1] // 24))
        print(f"  {manifest['levels'][k]['name']}: colour x{np.round(scale, 2)} {np.round(offset)}, "
              f"mean {np.round(coarse[..., :3].reshape(-1, 3).mean(axis=0))}")
    levels = [Level(globe_rgb, (-180, -90, 180, 90), globe=True)]
    for (rgba, bbox), entry in zip(raw, manifest["levels"], strict=True):
        level = Level(_feather(rgba, entry.get("feather", 0.08)), bbox)
        level.opaque = bool(rgba[..., 3].min() == 255)
        levels.append(level)
    return levels


def _composite(levels: list[Level], lat: np.ndarray, lon: np.ndarray, _unused: np.ndarray, *,
               bbox_scale: float | np.ndarray) -> np.ndarray:
    """RGB 0..1 of the ground at lat/lon for a screen footprint of ``bbox_scale`` metres per pixel."""
    footprint = np.broadcast_to(np.asarray(bbox_scale, dtype=np.float32), lat.shape)
    rgb = np.zeros(lat.shape + (3,), dtype=np.float32)
    first = 0
    for k, level in enumerate(levels):  # start at the finest level whose solid core covers every point
        if level.opaque and not level.globe:
            u, v = level.uv(lat, lon)
            if u.min() > 0.08 and u.max() < 0.92 and v.min() > 0.08 and v.max() < 0.92:
                first = k
    for level in levels[first:]:
        u, v = level.uv(lat, lon)
        if level.globe:
            inside = np.ones(lat.shape, dtype=bool)
        else:
            inside = (u > 0) & (u < 1) & (v > 0) & (v < 1)
            if not inside.any():
                continue
        lod = np.log2(np.maximum(footprint[inside] / level.metres_per_px, 1.0))
        texel = level.sample(u[inside], v[inside], lod)
        a = texel[:, 3:4]
        rgb[inside] = rgb[inside] * (1 - a) + texel[:, :3] * a
    return rgb


# --- clouds ------------------------------------------------------------------------------------------------

def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy, seed), _hash(ix + 1, iy, seed)
    c, d = _hash(ix, iy + 1, seed), _hash(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def cloud_density(east: np.ndarray, north: np.ndarray, footprint: float, seed: int, coverage: float) -> np.ndarray:
    """Cumulus-like fbm in 0..1 on the tangent plane (metres), with octaves down to the screen footprint so the
    texture keeps getting finer as the camera closes in, and never aliases. Thicker over the target (within
    ~12 km), so the dive always passes through cloud."""
    if coverage <= 0:
        return np.zeros(east.shape, dtype=np.float32)
    r2 = east * east + north * north
    coverage = coverage * np.exp(-r2 / 400_000.0 ** 2) + 0.35 * np.exp(-r2 / 12_000.0 ** 2)  # a regional deck
    total = np.zeros(east.shape, dtype=np.float32)
    weight_sum = 0.0
    wavelength, amplitude = 40_000.0, 1.0
    while wavelength > 300.0:  # >= 2 px per octave; fade out the last one
        fade = float(np.clip((wavelength / max(footprint, 1e-3) - 2) / 2, 0, 1))
        if fade <= 0:
            break
        angle = 0.61 * seed  # each octave on its own rotated lattice: aligned octaves stack into "+" shaped clouds
        x = (east * math.cos(angle) - north * math.sin(angle)) / wavelength
        y = (east * math.sin(angle) + north * math.cos(angle)) / wavelength
        total += amplitude * fade * _value_noise(x, y, seed)
        weight_sum += amplitude * fade
        wavelength /= 2.03
        amplitude *= 0.55
        seed += 7
    n = total / max(weight_sum, 1e-6)
    return np.clip((n - (1 - coverage)) / 0.15, 0, 1)


# --- camera ------------------------------------------------------------------------------------------------

def _frame_basis(lat: float, lon: float, bearing: float):
    la, lo = math.radians(lat), math.radians(lon)
    up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.cross(up, east)
    b = math.radians(bearing)
    screen_up = north * math.cos(b) + east * math.sin(b)
    screen_right = np.cross(screen_up, up)
    return up, east, north, screen_right, screen_up


def altitude_curve(frames: int, path=PATH) -> np.ndarray:
    """Altitude (m) per frame: monotone cubic (Fritsch-Carlson) in log altitude through ``path`` keyframes."""
    t = np.array([p[0] for p in path])
    y = np.log(np.array([p[1] for p in path], dtype=np.float64))
    d = np.diff(y) / np.diff(t)
    m = np.concatenate([[d[0]], (d[:-1] + d[1:]) / 2, [d[-1]]])
    for k in range(len(d)):
        if d[k] == 0:
            m[k] = m[k + 1] = 0
        else:
            a, b = m[k] / d[k], m[k + 1] / d[k]
            s = a * a + b * b
            if s > 9:
                m[k], m[k + 1] = 3 * a / math.sqrt(s) * d[k], 3 * b / math.sqrt(s) * d[k]
    times = np.linspace(t[0], t[-1], frames)
    out = np.empty(frames)
    for i, x in enumerate(times):
        k = min(np.searchsorted(t, x, side="right") - 1, len(t) - 2)
        h = t[k + 1] - t[k]
        s = (x - t[k]) / h
        h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
        h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        out[i] = h00 * y[k] + h10 * h * m[k] + h01 * y[k + 1] + h11 * h * m[k + 1]
    return np.exp(out)


def render_frame(levels: list[Level], lat0: float, lon0: float, bearing: float, altitude: float,
                 size: tuple[int, int], fov_deg: float, deck: tuple[float, float], coverage: float,
                 seed: int) -> np.ndarray:
    width, height = size
    up, east, north, right, sup = _frame_basis(lat0, lon0, bearing)
    ground_r = R
    cam = up * (R + altitude)
    tan = math.tan(math.radians(fov_deg) / 2)
    xs = (np.arange(width) + 0.5) / width * 2 - 1
    ys = 1 - (np.arange(height) + 0.5) / height * 2
    gx, gy = np.meshgrid(xs * tan, ys * tan * height / width)
    dirs = -up[None, None] + gx[..., None] * right[None, None] + gy[..., None] * sup[None, None]
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs.reshape(-1, 3)
    pixel_angle = 2 * tan / width

    def hit(radius: float):
        b = dirs @ cam
        c = cam @ cam - radius * radius
        disc = b * b - c
        t = -b - np.sqrt(np.maximum(disc, 0))
        return t, disc >= 0

    t_ground, on_earth = hit(ground_r)
    point = cam + dirs * t_ground[:, None]
    lat = np.degrees(np.arcsin(np.clip(point[:, 2] / ground_r, -1, 1)))
    lon = np.degrees(np.arctan2(point[:, 1], point[:, 0]))
    cos_view = np.clip(-(dirs * point).sum(axis=1) / ground_r, 0.05, 1)
    footprint = t_ground * pixel_angle / cos_view
    rgb = np.zeros((dirs.shape[0], 3), dtype=np.float32)
    idx = np.nonzero(on_earth)[0]
    if idx.size:
        rgb[idx] = _composite(levels, lat[idx], lon[idx], None, bbox_scale=footprint[idx])
        rgb[idx] *= (0.75 + 0.25 * cos_view[idx])[:, None]

    # cloud deck: fixed sub-layers between deck[0] and deck[1]; each adds extinction for the part of it below
    # the camera, so the camera fades into and out of the fog continuously and the texture expands past it.
    low, high = deck
    layers = 6
    thickness = (high - low) / layers
    tangent = lambda p: (p @ east, p @ north)
    shadow = np.zeros(idx.size, dtype=np.float32)
    if idx.size:  # sun overhead: each sub-layer shades the ground point below it, whether above or below the camera
        ge, gn = tangent(point[idx])
        for k in range(layers):
            shadow += 0.25 * cloud_density(ge, gn, 600.0, seed + 101 * k, coverage)
    clouds = []
    for k in range(layers):
        bottom = low + k * thickness
        if bottom >= altitude:
            continue
        inside = min(1.0, (altitude - bottom) / thickness)  # share of this sub-layer below the camera
        mid = bottom + thickness * inside / 2
        t_layer, ok = hit(R + mid)
        p = cam + dirs * t_layer[:, None]
        e, n = tangent(p)
        local = float(np.median(t_layer[ok])) * pixel_angle if ok.any() else 1.0
        density = cloud_density(e, n, local, seed + 101 * k, coverage)
        cos_layer = np.clip(-(dirs * p).sum(axis=1) / np.linalg.norm(p, axis=1), 0.05, 1)
        alpha = 1 - np.exp(-2.2 * density * inside / cos_layer)
        alpha[~ok] = 0
        shade = 0.80 + 0.20 * (k + 1) / layers  # lighter toward the top of the deck
        clouds.append((alpha, shade))
    if idx.size:
        rgb[idx] *= (0.88 + 0.12 * np.exp(-shadow))[:, None]  # soft: the thick patch over the place darkened the landing ~25%
    for alpha, shade in clouds:  # nearest sub-layer composited last
        rgb = rgb * (1 - alpha[:, None]) + shade * alpha[:, None]

    # atmosphere: haze by path length through an 80 km shell, a glow at the limb, black space
    t_top, through = hit(R + ATMOSPHERE)
    inside_atm = altitude < ATMOSPHERE
    start = np.zeros_like(t_top) if inside_atm else np.maximum(t_top, 0)
    b = dirs @ cam
    far_side = -b + np.sqrt(np.maximum(b * b - (cam @ cam - (R + ATMOSPHERE) ** 2), 0))
    end = np.where(on_earth, t_ground, far_side)
    path = np.where(through | inside_atm, np.maximum(end - start, 0), 0)
    haze = 1 - np.exp(-path / 350_000)
    rgb = rgb * (1 - haze[:, None]) + HAZE * haze[:, None]
    rgb[~on_earth & ~(through | inside_atm)] = 0
    return (np.clip(rgb, 0, 1).reshape(height, width, 3) * 255).astype(np.uint8)


_JOB: tuple = ()


def _render_one(item: tuple[int, float, float]) -> tuple[int, float]:
    levels, lat, lon, size, fov_deg, deck, coverage, seed, tmp = _JOB
    i, h, bearing = item
    Image.fromarray(render_frame(levels, lat, lon, bearing, h, size, fov_deg, deck, coverage, seed)).save(
        f"{tmp}/{i:05d}.png")
    return i, h


def render_dive(levels: list[Level], lat: float, lon: float, bearing: float, out: Path, *, frames: int = 111,
                size: tuple[int, int] = (1920, 1088), fov_deg: float = 60.0, deck: tuple[float, float] = (6_100, 7_600),
                coverage: float = 0.35, seed: int = 20261009, workers: int | None = None,
                end_m: float | None = None, spin: float = 0.0) -> Path:
    """Write the dive as an H.264 MP4 at 24 fps and its last frame as <out>_last.png; returns the MP4 path.
    ``end_m`` replaces the last altitude (e.g. to land on another clip's first frame); ``spin`` degrees of turn
    are spread over the dive in step with the zoom, ending on ``bearing``."""
    import multiprocessing as mp
    import os
    path = PATH if end_m is None else (*PATH[:-1], (PATH[-1][0], end_m))
    heights = altitude_curve(frames, path)
    progress = np.log(heights[0] / heights) / np.log(heights[0] / heights[-1])
    bearings = bearing - spin * (1 - progress)
    global _JOB
    with tempfile.TemporaryDirectory() as tmp:
        _JOB = (levels, lat, lon, size, fov_deg, deck, coverage, seed, tmp)
        started = time.monotonic()
        workers = max(1, min(workers or min((os.cpu_count() or 1) - 2, 4), frames))
        with mp.get_context("fork").Pool(workers) as pool:  # forked workers share the imagery copy-on-write
            for done, _ in enumerate(pool.imap_unordered(_render_one, [(i, h, b) for i, (h, b) in enumerate(zip(heights, bearings, strict=True))])):
                if done % 10 == 0:
                    print(f"  {done}/{frames} frames ({time.monotonic() - started:.0f} s, {workers} workers)", flush=True)
        Image.open(f"{tmp}/{frames - 1:05d}.png").save(out.with_name(out.stem + "_last.png"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%05d.png", "-c:v",
                        "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def synthetic_manifest(lat: float, lon: float, folder: Path) -> dict:
    """Offline stand-in imagery: a coloured noise globe and squares of finer noise with roads (grid lines), so
    geometry, matching, feathering and the cloud deck can be checked without downloads."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    def noise(n: int, cells: int) -> np.ndarray:
        base = rng.random((cells, cells, 3)).astype(np.float32)
        return np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC))

    globe = np.asarray(Image.fromarray(noise(1024, 64)).resize((2048, 1024)))
    globe = (globe * np.array([0.3, 0.5, 0.7])).astype(np.uint8)
    Image.fromarray(globe).save(folder / "globe_bluemarble.png")
    manifest = {"globe": str(folder / "globe_bluemarble.png"), "levels": []}
    for name, width_km, _, _ in LEVELS:
        n = 1024
        img = noise(n, 128).astype(np.float32) * np.array([0.6, 0.6, 0.5])
        img[:: 64] = img[:, :: 64] = 230
        rgba = np.dstack([img.astype(np.uint8), np.full((n, n), 255, np.uint8)])
        Image.fromarray(rgba, "RGBA").save(folder / f"{name}.png")
        manifest["levels"].append({"name": name, "path": str(folder / f"{name}.png"),
                                   "bbox": list(_square(lat, lon, width_km * 1000))})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--place", default="san-francisco-twin-peaks")
    parser.add_argument("--frames", type=int, default=111)
    parser.add_argument("--size", default="1920x1088")
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--coverage", type=float, default=0.35, help="cloud cover of the regional deck, 0..1 (0 = none); always thick over the place")
    parser.add_argument("--workers", type=int, help="render processes (default: CPUs - 2, at most 4; each needs several GB, so never alongside an LTX job)")
    parser.add_argument("--synthetic", action="store_true", help="offline stand-in imagery")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/fakeout/nadir"))
    args = parser.parse_args()
    from geoblock_core.places import find_place, landing, view_heading
    place = find_place(args.place)
    lat, lon = landing(place)
    bearing = max(view_heading(place), 0.0)  # screen-up = the way the person looks, so LTX tilts up toward it
    folder = args.output_dir / ("synthetic" if args.synthetic else f"imagery_{args.place}")
    manifest = synthetic_manifest(lat, lon, folder) if args.synthetic else fetch_imagery(lat, lon, folder)
    levels = load_levels(manifest)
    width, height = (int(v) for v in args.size.split("x"))
    out = render_dive(levels, lat, lon, bearing, args.output_dir / f"nadir_dive_{args.place}.mp4",
                      frames=args.frames, size=(width, height), fov_deg=args.fov, coverage=args.coverage,
                      workers=args.workers)
    print(f"Dive: {out}\nLast frame: {out.with_name(out.stem + '_last.png')}\nBearing {bearing:.0f} deg")


if __name__ == "__main__":
    main()
