"""Calm the sea in the NAIP squares: sun glint and capture-date seams over water.

NAIP is flown tile by tile on different days, so open water can be a pale glint wash on one side of a dead straight
seam and dark sea on the other (Battery Spencer, transfer test 2: the strait off the Golden Gate Bridge). No inpaint
fixes that, because the pixels are valid.

Here the sea comes from Overture's base water polygons (ocean, bay, strait, ...). Within it, the broad brightness of
the water (a water-only blur, sigma 25 m) is pulled down to the place's typical sea colour wherever it is brighter
than that, in proportion to how much brighter, and the fine texture riding on it is kept, damped where it was glint.
Nothing is classified pixel by pixel, so there are no blocky patches, and a seam in the broad brightness disappears
with it. Water darker than the typical sea (shallows, swell shadows) is left alone. Roads and railways over the water
(bridges, causeways, piers), rasterised from Overture transportation segments with slack for relief displacement,
mark a corridor where only glint-bright pixels are pulled, so the deck stays as photographed.

Two halves, because the desktop runs the build and the ComfyUI jobs in different Pythons:
- ``masks()`` reads Overture (needs pyarrow; run through ``comfy_job sea_masks``) and writes a sea mask and a keep
  mask (structures) per square;
- ``apply()`` / ``calm()`` are numpy/OpenCV only and run in the build itself.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SEA_SUBTYPES = {"ocean"}
SEA_CLASSES = {"ocean", "sea", "bay", "strait", "sound", "lagoon", "fjord"}
DECK_M = {"motorway": 34.0, "trunk": 30.0, "primary": 24.0, "secondary": 18.0}  # kept width over water, by class
DECK_DEFAULT_M = 12.0
DECK_SLACK_M = 50.0         # added each side: a high deck leans away from its map line in the orthophoto (relief
                            # displacement: the Golden Gate deck sits ~40 m off its OSM line in NAIP)
DECK_LIFT = 55.0            # inside the corridor, pull pixels this much brighter (luma) than the sea, fully at
                            # +25 more: glint (~+90..105 at the Golden Gate), not the grey deck (~+40..55) or the red cables; bright
                            # blobs under CAR_M (cars, lane paint) are dropped from that so the deck keeps them
CAR_M = 6.0
FALLBACK = (54, 86, 100)    # a mid-blue sea for squares whose water is all glint
BROAD_M = 25.0              # the broad brightness: water-only blur this wide (sigma, metres); small enough that
                            # a capture-date seam stays sharp in it instead of bleeding into the dark side
FULL_LIFT = 30.0            # broad luma this far above the typical sea is pulled all the way down
SPECKLE_DARK = 0.5          # speckle (Liberty Island NAIP: black and saturated-blue blocks over the harbour): water
SPECKLE_BLUE = 12.0         # under this share of the square's median water luma, or blue this far above green
SPECKLE_STD = 8.0           # ... and only where the luma's local std (5 px) passes this: the defect is blocky; deep blue
                            # ocean and kelp are smooth (Battery Spencer 24300 m: 31% dark or blue, 0.3% with this)
SPECKLE_GATE = 0.03         # a square is repainted only when that much of its water is speckle (Liberty Island 2700 m:
                            # 10.3%, 8100 m: 8.3%; every Battery Spencer square 1.5% or less)
SPECKLE_CELL_M = 30.0       # speckle density blur (sigma, metres); the region is where density passes SPECKLE_DENSITY
SPECKLE_DENSITY = 0.05
VERSION = "3"               # mask format; calm() changes need no new masks               # masks on disk from an older version are rewritten


def _geoms(wkb: bytes) -> list[tuple[int, list[list[tuple[float, float]]]]]:
    """WKB -> [(kind, parts)]: kind 3 = polygon (parts = rings, the first the outline), 2 = line (one part)."""
    out: list[tuple[int, list[list[tuple[float, float]]]]] = []

    def points(order: str, off: int, dims: int) -> tuple[list[tuple[float, float]], int]:
        (m,) = struct.unpack_from(order + "I", wkb, off)
        pts = struct.unpack_from(order + "d" * (m * dims), wkb, off + 4)
        return list(zip(pts[0::dims], pts[1::dims])), off + 4 + 8 * m * dims

    def read(off: int) -> int:
        order = "<" if wkb[off] == 1 else ">"
        raw = struct.unpack_from(order + "I", wkb, off + 1)[0]
        dims = 3 if (raw & 0x80000000 or 1000 <= (raw & 0x0FFFFFFF) < 2000) else 2
        kind = (raw & 0x0FFFFFFF) % 1000
        off += 5
        if kind == 2:
            line, off = points(order, off, dims)
            out.append((2, [line]))
        elif kind == 3:
            (n,) = struct.unpack_from(order + "I", wkb, off)
            off += 4
            rings = []
            for _ in range(n):
                ring, off = points(order, off, dims)
                rings.append(ring)
            out.append((3, rings))
        elif kind in (5, 6, 7):
            (n,) = struct.unpack_from(order + "I", wkb, off)
            off += 4
            for _ in range(n):
                off = read(off)
        return off

    read(0)
    return out


def _read(theme: str, kind: str, box: tuple[float, float, float, float], cache_dir: Path, columns: list[str]):
    """Overture rows of one type whose bbox overlaps ``box`` (unlike buildings.read_overture, which keeps rows by
    bbox centre and so loses a sea polygon whose centre is far offshore)."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from geoblock_core import buildings as B
    west, south, east, north = box
    rows = []
    with B._client() as client:
        for key, groups in B.row_group_index(kind, cache_dir, theme=theme):
            hits = [g[0] for g in groups if g[1] <= east and g[3] >= west and g[2] <= north and g[4] >= south]
            if not hits:
                continue
            with B.RangeFile(f"{B.BUCKET}/{key}", client) as handle:
                table = pq.ParquetFile(handle).read_row_groups(hits, columns=columns + ["bbox"])
            for row in table.to_pylist():
                b = row.pop("bbox")
                if b["xmin"] <= east and b["xmax"] >= west and b["ymin"] <= north and b["ymax"] >= south:
                    rows.append(row)
    return rows


def masks(lat: float, lon: float, folder: str, levels: list[int], pixels: int, cache_dir: str) -> dict[str, list[str]]:
    """Per NAIP square: naip_<w>_sea.png (255 = sea) and naip_<w>_keep.png (255 = road/rail over it, as photographed).
    Returns width -> [sea path, keep path]."""
    import math

    import nadir_dive
    folder_p, cache = Path(folder), Path(cache_dir)
    squares = {w: nadir_dive._square(lat, lon, w) for w in levels}
    widest = squares[max(levels)]
    sea_rows = _read("base", "water", widest, cache, ["subtype", "class", "geometry"])
    seas = [g for r in sea_rows if r["subtype"] in SEA_SUBTYPES or r["class"] in SEA_CLASSES
            for g in _geoms(r["geometry"]) if g[0] == 3]
    # structures only matter where the dive gets close: the squares up to 8.1 km
    near = squares[max(w for w in levels if w <= 8100)] if any(w <= 8100 for w in levels) else widest
    seg_rows = _read("transportation", "segment", near, cache, ["subtype", "class", "geometry"])
    decks = [(DECK_M.get(r["class"] or "", DECK_DEFAULT_M), g[1][0]) for r in seg_rows if r["subtype"] in ("road", "rail")
             for g in _geoms(r["geometry"]) if g[0] == 2]
    out = {}
    for w, (west, south, east, north) in squares.items():
        sx, sy = pixels / (east - west), pixels / (north - south)

        def px(pts):
            return [((x - west) * sx, (north - y) * sy) for x, y in pts]

        sea = Image.new("L", (pixels, pixels), 0)
        pen = ImageDraw.Draw(sea)
        for _, rings in seas:
            for k, ring in enumerate(rings):  # outline filled, holes (islands) cut back out
                if len(ring) >= 3:
                    pen.polygon(px(ring), fill=255 if k == 0 else 0)
        keep = Image.new("L", (pixels, pixels), 0)
        pen = ImageDraw.Draw(keep)
        m_per_px = w / pixels
        for width_m, line in decks:
            if len(line) >= 2:
                pen.line(px(line), fill=255, width=max(1, int(math.ceil((width_m + 2 * DECK_SLACK_M) / m_per_px))))
        paths = [folder_p / f"naip_{w:05d}_sea.png", folder_p / f"naip_{w:05d}_keep.png"]
        for im, p in zip((sea, keep), paths):
            im.save(p, pnginfo=_info())
        out[str(w)] = [str(p) for p in paths]
        print(f"  sea {w} m: {np.asarray(sea).mean() / 255:.1%}, kept structures "
              f"{(np.asarray(keep) & np.asarray(sea)).astype(bool).mean():.2%}", flush=True)
    return out


def _info():
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("sea_version", VERSION)
    return info


def current(path: Path) -> bool:
    """A mask on disk written by this version of masks()."""
    try:
        return path.is_file() and Image.open(path).info.get("sea_version") == VERSION
    except OSError:
        return False


def _luma(a: np.ndarray) -> np.ndarray:
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _broad(a: np.ndarray, water: np.ndarray, sigma_px: float, s: int = 512) -> np.ndarray:
    """Water-only blur (normalised convolution) at ``s`` px, scaled back up: the water's colour, no land in it."""
    import cv2
    n = a.shape[0]
    s = min(s, n)
    f = s / n
    small = cv2.resize(a * water[..., None], (s, s), interpolation=cv2.INTER_AREA)
    wt = cv2.resize(water.astype(np.float32), (s, s), interpolation=cv2.INTER_AREA)
    sig = max(0.8, sigma_px * f)
    num = cv2.GaussianBlur(small, (0, 0), sig)
    den = cv2.GaussianBlur(wt, (0, 0), sig)[..., None]
    low = num / np.maximum(den, 1e-3)
    return cv2.resize(low, (n, n), interpolation=cv2.INTER_CUBIC)


def sea_colour(lows: dict[int, np.ndarray], seas: dict[int, np.ndarray]) -> np.ndarray:
    """The place's typical sea: the median broad colour of the darker third of the water, over every square together
    so all levels agree; FALLBACK when there is no clear water to measure."""
    picks = []
    for w, low in lows.items():
        water = seas[w]
        if water.mean() < 0.01:
            continue
        px = low[water][::7]
        lum = _luma(px)
        picks.append(px[lum <= np.percentile(lum, 35)])
    if not picks:
        return np.float32(FALLBACK)
    colour = np.median(np.concatenate(picks), axis=0)
    return colour if _luma(colour) < 140 else np.float32(FALLBACK)


def calm(arr: np.ndarray, water: np.ndarray, keep: np.ndarray, low: np.ndarray, colour: np.ndarray,
         metres_per_px: float) -> tuple[np.ndarray, float]:
    """Pull the water's broad colour down to ``colour`` where brighter (fully at FULL_LIFT luma above it), damp the
    fine texture there by up to 60%, leave ``keep`` (structures) as photographed; returns (image, mean pull)."""
    import cv2
    a = arr.astype(np.float32)
    # near: the water's colour at ~3 m. Pull only where both the broad and the near colour are brighter than the sea:
    # a boat or whitecap on dark sea (bright near, dark broad) and the dark side of a seam or a swell shadow inside
    # glint (dark near) are left alone, and the seam itself stays sharp
    near = _broad(a, water.astype(bool), 3.0 / max(metres_per_px, 0.05), a.shape[0]) if metres_per_px < 1.5 else a
    lift = np.minimum(_luma(low), _luma(near)) - _luma(colour)
    alpha = np.clip(lift / FULL_LIFT, 0, 1) * water
    feather = max(1.0, 3.0 / max(metres_per_px, 0.1))
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), feather) * water
    # the structure corridor (bridges): there the pull ignores the broad colour (the deck's shadow darkens it) and
    # goes by each pixel's own brightness, so glint beside the deck goes and the deck, its cars and the cables stay;
    # the corridor's edge is feathered so it never shows as a step against the open-sea pull
    bright = (_luma(a) - _luma(colour) - DECK_LIFT) / 25.0
    k = max(1, int(round(CAR_M / max(metres_per_px, 0.05))))
    if k > 1:  # open: bright blobs smaller than a car or lane line are not glint
        hard = cv2.morphologyEx((bright > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        bright = np.where(hard > 0, bright, 0)
    bright = cv2.GaussianBlur(np.clip(bright, 0, 1).astype(np.float32), (0, 0), 1.0) * water
    inside = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), max(2.0, 10.0 / max(metres_per_px, 0.05)))
    inside = np.maximum(inside, keep)
    alpha = (alpha * (1 - inside) + bright * inside)[..., None]
    detail = (a - near) * (1 - 0.6 * alpha)  # the fine texture only, so the glint's own brightness never returns
    out = a * (1 - alpha) + (colour + detail) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), float(alpha.mean())


def speckle(a: np.ndarray, water: np.ndarray, keep: np.ndarray, metres_per_px: float) -> tuple[np.ndarray, float]:
    """Blocky dark / saturated-blue speckle in the water (a NAIP tile defect, not glint: calm() cannot pull it).
    Returns (region, share): the speckled water outside ``keep`` as a bool mask, empty unless the square's share of
    speckle pixels reaches SPECKLE_GATE, so clean squares are left exactly as they were."""
    import cv2
    open_water = water & ~keep
    if open_water.mean() < 0.001:
        return np.zeros_like(water), 0.0
    lum = _luma(a)
    m = cv2.blur(lum, (5, 5))
    busy = np.sqrt(np.maximum(cv2.blur(lum * lum, (5, 5)) - m * m, 0)) > SPECKLE_STD
    odd = (lum < SPECKLE_DARK * np.median(lum[open_water])) | (a[..., 2] - a[..., 1] > SPECKLE_BLUE)
    bad = odd & busy & open_water
    share = float(bad.sum() / open_water.sum())
    if share < SPECKLE_GATE:
        return np.zeros_like(water), share
    density = cv2.GaussianBlur(bad.astype(np.float32), (0, 0), max(2.0, SPECKLE_CELL_M / metres_per_px))
    # inside the structure corridor only the speckle pixels themselves (a little grown) go: piers and seawalls are
    # grey or bright, never black or saturated blue, so they stay as photographed
    near = cv2.dilate((odd & water & keep).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return ((density > SPECKLE_DENSITY) & open_water) | (near & water & (density > SPECKLE_DENSITY)), share


def repaint(a: np.ndarray, region: np.ndarray, colour: np.ndarray, metres_per_px: float, amp: float,
            seed: int) -> np.ndarray:
    """``region`` replaced by ``colour`` with a faint wind-ripple texture (band-limited noise stretched across the
    wind, ``amp`` luma), its edge feathered over ~8 m so it meets the photographed water without a step."""
    import cv2
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n)).astype(np.float32)
    k = max(0.6, 1.2 / metres_per_px)  # ripples ~2-4 m across, longer along the wind
    tex = cv2.GaussianBlur(noise, (0, 0), sigmaX=3 * k, sigmaY=k)
    tex *= amp / max(float(tex.std()), 1e-6)
    alpha = cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(1.0, 8.0 / metres_per_px))
    alpha = (np.minimum(alpha * 2, 1) * region)[..., None]  # full inside, feathered only on the inner edge
    sea = colour[None, None, :] + tex[..., None]
    return np.clip(a * (1 - alpha) + sea * alpha, 0, 255)


def apply(paths: dict[int, Path], masks: dict[int, tuple[Path, Path]], pixels: int) -> tuple[dict[int, Path], dict]:
    """Calm every square; writes naip_<w>_calm.png next to it and returns the new paths + facts."""
    import cv2
    squares = {w: np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for w, p in paths.items()}
    seas = {w: np.asarray(Image.open(masks[w][0]).convert("L")) > 127 for w in paths}
    keeps = {w: (np.asarray(Image.open(masks[w][1]).convert("L")) > 127) & seas[w] for w in paths}
    if not any(s.mean() > 0.001 for s in seas.values()):
        return paths, {"sea": "none"}
    wet = [w for w in paths if seas[w].mean() > 0.001]
    # speckle first: it is left out of the sea colour (its black blocks dragged it dark at Liberty Island, so the
    # clean water and its ships were pulled down), then painted over with that colour before the calm
    specks = {w: speckle(squares[w], seas[w], keeps[w], w / pixels) for w in wet}
    clean = {w: seas[w] & ~specks[w][0] for w in wet}
    lows = {w: _broad(squares[w], clean[w], BROAD_M / (w / pixels)) for w in wet}
    colour = sea_colour(lows, clean)
    repainted = {}
    for w in wet:
        region = specks[w][0]
        if region.any():
            lum = _luma(squares[w])
            m = cv2.blur(lum, (9, 9))
            std = np.sqrt(np.maximum(cv2.blur(lum * lum, (9, 9)) - m * m, 0))
            ok = clean[w] & ~keeps[w]
            amp = float(np.clip(np.median(std[ok]) if ok.any() else 2.0, 1.0, 6.0))
            # painted in this square's median clean water, so it meets the water beside it (a capture seam's
            # clean side at Liberty Island) without a tone step; the calm then treats it like any other water
            fill = np.median(lows[w][ok], axis=0) if ok.sum() > 1000 else colour
            squares[w] = repaint(squares[w], region, fill, w / pixels, amp, seed=w)
            lows[w] = _broad(squares[w], seas[w], BROAD_M / (w / pixels))
            repainted[w] = round(float(region.sum() / seas[w].sum()), 4)
    out, pulls = dict(paths), {}
    for w, low in lows.items():
        calmed, pull = calm(squares[w], seas[w], keeps[w], low, colour, w / pixels)
        pulls[w] = round(pull, 4)
        if pull > 0.0005 or w in repainted:
            p = paths[w].with_name(paths[w].stem + "_calm.png")
            Image.fromarray(calmed).save(p)
            out[w] = p
    return out, {"sea_colour": [round(float(c)) for c in colour], "sea_pull": pulls,
                 "kept_structures": {w: round(float(k.mean()), 4) for w, k in keeps.items() if k.any()},
                 "speckle": {w: round(specks[w][1], 4) for w in wet}, "speckle_repainted": repainted}
