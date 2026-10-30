"""Measured tree positions for the blocking, so video models keep trees where they really are.

A 1 m canopy height model (CHM) is built for the area the camera sees up close, then each
tree top becomes one proxy in the Blender blocking. Sources, best first:

- USGS 3DEP lidar (``geoblock_core.lidar``), where a survey covers the place. Heights are
  measured per point against a ground grid from the survey's own ground class. Canyon
  surveys leave cliff rock unclassified alongside vegetation, so steep ground is masked out.
- Meta / WRI global canopy height, 1.2 m, CC BY 4.0, model-derived from 2009–2020 imagery.
  Used outside lidar coverage. It misses many small trees in sparse woodland.

At Mather Point the lidar crowns line up with the dark crowns in its intensity image, and
lidar canopy cover tracks Sentinel-2 NDVI better than Meta's (docs/tree-proxies.md).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = "2"  # 2: Meta canopy fills holes in the lidar survey
META_CHM = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/{quadkey}.tif"
META_ATTRIBUTION = ("Meta and World Resources Institute (2024), Global Canopy Height Maps, 1 m, CC BY 4.0 "
                    "(Tolan et al., Remote Sensing of Environment 300, 113888)")
MIN_TREE_M = 2.0
MAX_TREE_M = 45.0
MAX_SLOPE_DEG = 35.0   # steeper ground is cliff: lidar rock returns there look like canopy
VOID_BLOCK_M = 20.0     # a block this size with no lidar points at all is a hole in the survey
VEGETATION = (3, 4, 5)  # ASPRS low/medium/high vegetation, when a survey classifies it
UNCLASSIFIED = (1,)     # otherwise: vegetation, buildings and rock together


# ---------------------------------------------------------------- canopy height model

def _fill_gaps(grid: np.ndarray, max_passes: int = 64) -> np.ndarray:
    """Fill NaN cells from the mean of their filled neighbours, growing inward."""
    grid = grid.copy()
    for _ in range(max_passes):
        missing = np.isnan(grid)
        if not missing.any():
            break
        padded = np.pad(grid, 1, constant_values=np.nan)
        stack = np.stack([padded[1 + dy:padded.shape[0] - 1 + dy, 1 + dx:padded.shape[1] - 1 + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        counts = np.sum(~np.isnan(stack), axis=0)
        means = np.nansum(stack, axis=0) / np.maximum(counts, 1)
        grow = missing & (counts > 0)
        grid[grow] = means[grow]
    if np.isnan(grid).any():
        grid[np.isnan(grid)] = np.nanmean(grid) if np.isfinite(grid).any() else 0.0
    return grid


def slope_degrees(ground: np.ndarray, cell_m: float) -> np.ndarray:
    dy, dx = np.gradient(ground, cell_m)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def lidar_chm(points, bounds: tuple[float, float, float, float], cell_m: float = 1.0, *,
              voids: bool = False):
    """Canopy height and ground slope grids (row 0 = north edge) from classified lidar points.

    ``points`` is one dict of arrays or an iterable of them (``lidar.iter_points``). Everything is gridded
    as it streams (mean ground height, and the highest vegetation and unclassified return per cell), so
    memory follows the grid size, not the point count: dense city surveys run to hundreds of millions of
    points. With ``voids``, also returns a mask of cells in VOID_BLOCK_M blocks the survey has no points in
    at all (a missing source tile leaves a rectangle with no trees).
    """
    if isinstance(points, dict):
        points = [points]
    x0, y0, x1, y1 = bounds
    cols, rows = math.ceil((x1 - x0) / cell_m), math.ceil((y1 - y0) / cell_m)
    total, count = np.zeros(rows * cols), np.zeros(rows * cols)
    top = {"vegetation": np.full(rows * cols, -np.inf, dtype=np.float32),
           "unclassified": np.full(rows * cols, -np.inf, dtype=np.float32)}
    block = max(1, round(VOID_BLOCK_M / cell_m))
    seen = np.zeros((math.ceil(rows / block), math.ceil(cols / block)), dtype=bool)
    for part in points:
        col = ((part["x"] - x0) / cell_m).astype(np.int64)
        row = ((y1 - part["y"]) / cell_m).astype(np.int64)
        inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        col, row, z, cls = col[inside], row[inside], part["z"][inside], part["classification"][inside]
        seen[row // block, col // block] = True
        cell = row * cols + col
        ground = cls == 2
        total += np.bincount(cell[ground], weights=z[ground], minlength=rows * cols)
        count += np.bincount(cell[ground], minlength=rows * cols)
        for name, classes in (("vegetation", VEGETATION), ("unclassified", UNCLASSIFIED)):
            pick = np.isin(cls, classes)
            np.maximum.at(top[name], cell[pick], z[pick].astype(np.float32))
    if count.sum() < 100:
        raise ValueError("The lidar survey has too few ground points here to measure tree heights")
    dtm = _fill_gaps(np.where(count > 0, total / np.maximum(count, 1), np.nan).reshape(rows, cols))
    # Surveys that split vegetation (3-5) keep buildings and rock out of it; otherwise use class 1.
    highest = top["vegetation"] if np.isfinite(top["vegetation"]).any() else top["unclassified"]
    chm = np.clip(np.nan_to_num(highest.reshape(rows, cols) - dtm, neginf=0.0), 0, None)
    chm[chm > MAX_TREE_M] = 0  # towers, cables and cliff overhangs
    slope = slope_degrees(dtm, cell_m)
    chm[slope > MAX_SLOPE_DEG] = 0
    if voids:
        return chm, slope, ~np.repeat(np.repeat(seen, block, axis=0), block, axis=1)[:rows, :cols]
    return chm, slope


def _quadkey(longitude: float, latitude: float, zoom: int = 9) -> str:
    n = 2 ** zoom
    x = int((longitude + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2 * n)
    return "".join(str(((x >> bit) & 1) + 2 * ((y >> bit) & 1)) for bit in range(zoom - 1, -1, -1))


def meta_chm(bounds: tuple[float, float, float, float], crs: str, cell_m: float = 1.0) -> np.ndarray:
    """Meta / WRI canopy heights resampled onto the same grid as ``lidar_chm``."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds

    x0, y0, x1, y1 = bounds
    cols, rows = math.ceil((x1 - x0) / cell_m), math.ceil((y1 - y0) / cell_m)
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    chm = np.zeros((rows, cols), dtype=np.float32)
    keys = {_quadkey(lon, lat) for lon in (west, east) for lat in (south, north)}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for key in sorted(keys):
            part = np.zeros_like(chm)
            with rasterio.open("/vsicurl/" + META_CHM.format(quadkey=key)) as src:
                reproject(rasterio.band(src, 1), part, dst_transform=from_origin(x0, y1, cell_m, cell_m),
                          dst_crs=crs, resampling=Resampling.bilinear)
            chm = np.maximum(chm, part)
    return chm.astype(float)


# ---------------------------------------------------------------- tree tops

def _max_filter(grid: np.ndarray, radius: int) -> np.ndarray:
    out = grid.copy()
    padded = np.pad(grid, radius, constant_values=-np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out = np.maximum(out, padded[radius + dy:radius + dy + grid.shape[0],
                                             radius + dx:radius + dx + grid.shape[1]])
    return out


def find_trees(chm: np.ndarray, cell_m: float = 1.0, *, min_height: float = MIN_TREE_M) -> np.ndarray:
    """Tree tops as rows of (col, row, height, crown radius), in cells and metres, tallest first.

    A top is a local maximum of the lightly smoothed CHM; tops closer than a crown's
    likely radius to a taller one are merged into it. The crown radius is how far the
    canopy stays above half the tree's height, averaged over eight directions.
    """
    smooth = chm.copy()
    smooth[1:-1, 1:-1] = (4 * chm[1:-1, 1:-1] + chm[:-2, 1:-1] + chm[2:, 1:-1] + chm[1:-1, :-2] + chm[1:-1, 2:]) / 8
    peaks = (smooth >= _max_filter(smooth, max(1, round(1.5 / cell_m)))) & (chm >= min_height)
    rows, cols = np.nonzero(peaks)
    heights = chm[rows, cols]
    order = np.argsort(-heights, kind="stable")
    rows, cols, heights = rows[order], cols[order], heights[order]

    directions = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    max_steps = max(2, round(8 / cell_m))
    reach = np.zeros((len(rows), 8))
    alive = np.ones((len(rows), 8), dtype=bool)
    for step in range(1, max_steps + 1):
        c = np.rint(cols[:, None] + directions[None, :, 0] * step).astype(int)
        r = np.rint(rows[:, None] + directions[None, :, 1] * step).astype(int)
        inside = (c >= 0) & (c < chm.shape[1]) & (r >= 0) & (r < chm.shape[0])
        value = np.where(inside, chm[np.clip(r, 0, chm.shape[0] - 1), np.clip(c, 0, chm.shape[1] - 1)], 0)
        alive &= inside & (value >= 0.5 * heights[:, None]) & (value <= 1.05 * heights[:, None])
        reach += alive
    radius = np.clip((reach.mean(axis=1) + 0.5) * cell_m, 0.6, 0.3 * heights + 1.5)

    # Greedy merge, tallest first, on a coarse bucket grid.
    keep = np.zeros(len(rows), dtype=bool)
    bucket_m = 4.0
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(len(rows)):
        x, y = cols[i] * cell_m, rows[i] * cell_m
        bx, by = int(x // bucket_m), int(y // bucket_m)
        separation = max(1.5, 0.6 * radius[i])
        clash = False
        for key in ((bx + dx, by + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            for j in buckets.get(key, ()):
                if math.hypot(x - cols[j] * cell_m, y - rows[j] * cell_m) < max(separation, 0.6 * radius[j]):
                    clash = True
                    break
            if clash:
                break
        if not clash:
            keep[i] = True
            buckets.setdefault((bx, by), []).append(i)
    return np.column_stack((cols[keep], rows[keep], heights[keep], radius[keep]))


# ---------------------------------------------------------------- scene integration

def near_field_bounds(scene: dict, route: dict, margin_m: float = 1000, max_side_m: float = 3500,
                      low_m: float = 300) -> tuple[float, float, float, float]:
    """Projected box around the low part of the camera path (plus margin), inside the terrain square.

    Only frames within ``low_m`` of the landmark's height (or of the path's lowest point) count:
    from higher up a tree is a pixel or two wide, so far canopy is left to the terrain colours.
    The box is capped at ``max_side_m`` around the lowest camera position.
    """
    ox, oy = scene["origin_projected_m"]
    radius = float(scene["radius_m"])
    positions = np.asarray(route["positions"], dtype=float)
    looks = np.asarray(route.get("look_at", route["positions"]), dtype=float)[:, :2]
    if "above_ground_m" in route:  # the Earth zoom plans its clearance over the terrain
        altitude = np.asarray(route["above_ground_m"], dtype=float)
    else:
        target_z = route.get("target_local_m", [0, 0, scene.get("target_height_m", positions[:, 2].min())])[2]
        altitude = positions[:, 2] - target_z
    low = altitude <= max(low_m, altitude.min() + low_m)
    xy = positions[low, :2]
    points = np.vstack((xy, xy + np.clip(looks[low] - xy, -margin_m, margin_m)))
    lower, upper = points.min(axis=0) - margin_m, points.max(axis=0) + margin_m
    centre = positions[np.argmin(altitude), :2]
    for axis in (0, 1):
        if upper[axis] - lower[axis] > max_side_m:
            lower[axis] = min(max(lower[axis], centre[axis] - max_side_m / 2), upper[axis] - max_side_m)
            upper[axis] = lower[axis] + max_side_m
    lower, upper = np.maximum(lower, -radius), np.minimum(upper, radius)
    return (ox + lower[0], oy + lower[1], ox + upper[0], oy + upper[1])


def building_mask(manifest: dict, bounds: tuple[float, float, float, float], cell_m: float = 1.0,
                  margin_m: float = 1.0) -> np.ndarray:
    """Canopy-grid cells under a building footprint (``buildings.fetch_buildings``), grown by ``margin_m``.

    Surveys that don't split out vegetation leave roofs in the same unclassified class as trees, so in a city
    every roof would become a tree; footprints keep them out."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from .buildings import load_buildings
    x0, y0, x1, y1 = bounds
    cols, rows = math.ceil((x1 - x0) / cell_m), math.ceil((y1 - y0) / cell_m)
    ox, oy = manifest["origin_projected_m"]
    rings, _, _ = load_buildings(manifest)
    shapes = [({"type": "Polygon", "coordinates": [(ring + (ox, oy)).tolist()]}, 1) for ring in rings
              if len(ring) >= 3 and ring[:, 0].max() + ox >= x0 and ring[:, 0].min() + ox <= x1
              and ring[:, 1].max() + oy >= y0 and ring[:, 1].min() + oy <= y1]
    if not shapes:
        return np.zeros((rows, cols), dtype=bool)
    mask = rasterize(shapes, out_shape=(rows, cols), transform=from_origin(x0, y1, cell_m, cell_m),
                     all_touched=True, dtype="uint8").astype(bool)
    grow = round(margin_m / cell_m)
    return _max_filter(mask.astype(float), grow) > 0 if grow else mask


def fetch_trees(scene: dict, route: dict, folder: Path, *, cache_dir: Path, source: str = "auto",
                cell_m: float = 1.0, buildings: dict | None = None) -> dict:
    """Write ``trees.json`` (local east/north metres, heights and crown radii) and return its manifest.
    With a ``buildings`` manifest, no tree stands on a building footprint."""
    from . import lidar

    if source not in ("auto", "lidar", "meta"):
        raise ValueError(f"Unknown tree source {source!r}")
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    bounds = near_field_bounds(scene, route)
    key = hashlib.sha256(json.dumps([VERSION, scene["projected_crs"], [round(v, 1) for v in bounds], source,
                                     cell_m, *([buildings["key"]] if buildings else [])]).encode()).hexdigest()[:16]
    manifest_path = folder / f"trees-{key}.json"
    if manifest_path.is_file():
        return {**json.loads(manifest_path.read_text()), "path": str(manifest_path)}
    place = scene["place"]
    project = lidar.project_for(place["longitude"], place["latitude"], cache_dir) if source != "meta" else None
    if source == "lidar" and project is None:
        raise ValueError("No USGS 3DEP lidar survey covers this place; use the Meta canopy source")
    if project:
        counted = []

        def stream():
            for part in lidar.iter_points(project["ept"], bounds, scene["projected_crs"], cache_dir / "ept"):
                counted.append(len(part["x"]))
                yield part

        chm, slope, void = lidar_chm(stream(), bounds, cell_m, voids=True)
        used = {"source": "lidar", "dataset": project["name"], "attribution": lidar.ATTRIBUTION,
                "license": "Public domain (US Government)", "points": int(sum(counted))}
        if source == "auto" and void.mean() > 0.001:  # holes in the survey: fill them from the Meta map
            try:
                meta = meta_chm(bounds, scene["projected_crs"], cell_m)
            except Exception as error:  # the lidar alone is still usable
                print(f"Meta canopy fill skipped: {error}", flush=True)
            else:
                fill = void & (slope <= MAX_SLOPE_DEG)
                chm[fill] = meta[fill]
                used.update(dataset=f"{project['name']} + Meta/WRI Global Canopy Height v1 in survey gaps",
                            attribution=f"{lidar.ATTRIBUTION}; {META_ATTRIBUTION}",
                            license="Public domain (US Government); CC BY 4.0",
                            meta_fill_share=round(float(void.mean()), 4))
    else:
        chm = meta_chm(bounds, scene["projected_crs"], cell_m)
        used = {"source": "meta", "dataset": "Meta/WRI Global Canopy Height v1", "attribution": META_ATTRIBUTION,
                "license": "CC BY 4.0"}
    if buildings:
        roofs = building_mask(buildings, bounds, cell_m)
        used["building_cells_cleared"] = int((roofs & (chm >= MIN_TREE_M)).sum())
        chm[roofs] = 0
    tops = find_trees(chm, cell_m)
    ox, oy = scene["origin_projected_m"]
    x0, _, _, y1 = bounds
    east = x0 + (tops[:, 0] + 0.5) * cell_m - ox
    north = y1 - (tops[:, 1] + 0.5) * cell_m - oy
    trees = np.round(np.column_stack((east, north, tops[:, 2], tops[:, 3])), 2)
    manifest = {"key": key, "version": VERSION, **used, "bounds_projected_m": [round(v, 2) for v in bounds],
                "crs": scene["projected_crs"], "cell_m": cell_m, "count": int(len(trees)),
                "canopy_cover": round(float((chm >= MIN_TREE_M).mean()), 4),
                "columns": ["east_m", "north_m", "height_m", "crown_radius_m"], "trees": trees.tolist()}
    manifest_path.write_text(json.dumps(manifest) + "\n")
    return {**manifest, "path": str(manifest_path)}


AROUND_M = 250.0  # the ground a landing's keyframes show around the person


def canopy_summary(manifest: dict, around: tuple[float, float] | None = None, radius_m: float = AROUND_M) -> dict:
    """Cover and height spread of the measured trees, for prompts (``looks.canopy_text``).

    With ``around`` (local east, north), only the trees within ``radius_m`` of it count, and cover is their crown
    area over the circle: Twin Peaks' near field is 27% woodland, but its summit is open grass and road, and
    words for the whole area made the painter stand the person among tall pines.
    """
    trees = np.asarray(manifest["trees"], dtype=float).reshape(-1, 4)
    cover = float(manifest["canopy_cover"])
    if around is not None:
        trees = trees[np.hypot(trees[:, 0] - around[0], trees[:, 1] - around[1]) <= radius_m]
        cover = min(0.95, float((math.pi * trees[:, 3] ** 2).sum()) / (math.pi * radius_m ** 2))
    heights = trees[:, 2]
    count = len(heights)
    if not len(heights):
        heights = np.zeros(1)
    p25, p75, p95 = np.percentile(heights, [25, 75, 95])
    return {"cover": round(cover, 4), "count": int(count if around is not None else manifest["count"]),
            **({"around_m": radius_m} if around is not None else {}),
            "height_p25_m": round(float(p25), 1), "height_p75_m": round(float(p75), 1),
            "height_p95_m": round(float(p95), 1), "tall_share": round(float((heights >= CONIFER_FROM_M).mean()), 3),
            "source": manifest["source"]}


def tree_report(manifest: dict) -> str:
    heights = np.asarray(manifest["trees"], dtype=float).reshape(-1, 4)[:, 2]
    spread = f"median {np.median(heights):.1f} m, tallest {heights.max():.1f} m" if len(heights) else "none found"
    return (f"TREES: {manifest['count']} from {manifest['dataset']} ({spread}; canopy cover "
            f"{100 * manifest['canopy_cover']:.0f}%)\nTREE DATA: {manifest['attribution']} · {manifest['license']}")


def clear_of(standing, points, margin_m: float = 1.5) -> np.ndarray:
    """Drop trees the camera (or the stand-in) would be inside or brush through.

    ``standing`` rows are (east, north, ground z, height, crown radius); ``points`` rows are (x, y, z).
    A tree goes when any point is within its crown radius plus ``margin_m`` across and below its top.
    """
    standing = np.asarray(standing, dtype=float).reshape(-1, 5)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    keep = np.ones(len(standing), dtype=bool)
    for start in range(0, len(points), 64):
        chunk = points[start:start + 64]
        across = np.hypot(standing[:, None, 0] - chunk[None, :, 0], standing[:, None, 1] - chunk[None, :, 1])
        hit = (across < standing[:, None, 4] + margin_m) & (chunk[None, :, 2] < standing[:, None, 2] + standing[:, None, 3] + margin_m)
        keep &= ~hit.any(axis=1)
    return standing[keep]


FEATHER_M = 500.0  # width of the band in which the tree area thins out to nothing at its edge


def feather_edges(standing, band_m: float = FEATHER_M, seed: int = 0) -> np.ndarray:
    """Thin trees out toward the edge of the measured area so it doesn't show as a hard rectangle from the air.

    ``standing`` rows start with (east, north). A tree survives with a probability that rises smoothly from 0
    at the edge of the area the trees cover to 1 at ``band_m`` inside it (at most a fifth of the shorter side).
    """
    standing = np.asarray(standing, dtype=float)
    if len(standing) < 2:
        return standing
    lo, hi = standing[:, :2].min(axis=0), standing[:, :2].max(axis=0)
    band = min(band_m, 0.2 * float((hi - lo).min()))
    if band <= 0:
        return standing
    inside = np.minimum(standing[:, :2] - lo, hi - standing[:, :2]).min(axis=1)
    u = np.clip(inside / band, 0, 1)
    keep = np.random.default_rng(seed).random(len(standing)) < u * u * (3 - 2 * u)
    return standing[keep]


# ---------------------------------------------------------------- proxy geometry

# Lathe profiles as (height fraction, crown-radius fraction). Short trees are the rounded
# pinyon/juniper shape, crown down to the ground; tall ones are conifers on a visible trunk.
ROUND_CROWN = ((0.06, 0.55), (0.35, 1.0), (0.65, 0.85), (0.88, 0.45))
# Ponderosa-like: an open, round-topped crown high on the trunk. A clean cone read as a Christmas-tree
# plantation to the keyframe painter, which copied the shapes literally.
CONIFER_CROWN = ((0.3, 0.75), (0.55, 1.0), (0.78, 0.8), (0.93, 0.4))
JITTER = 0.25  # per-vertex crown radius variation, and lean of the upper crown, as fractions of the radius
CONIFER_FROM_M = 12.0
SIDES = 8


def _lathe(profile, sides: int = SIDES) -> tuple[np.ndarray, np.ndarray]:
    """Closed unit solid: bottom centre, one ring per profile row, apex. Vertices are (cos, sin, h, f)."""
    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    rows = [(0.0, 0.0, profile[0][0], 0.0)]
    rows += [(math.cos(a), math.sin(a), h, f) for h, f in profile for a in angles]
    rows.append((0.0, 0.0, 1.0, 0.0))
    tri = [[0, 1 + (i + 1) % sides, 1 + i] for i in range(sides)]
    for ring in range(len(profile) - 1):
        a, b = 1 + ring * sides, 1 + (ring + 1) * sides
        for i in range(sides):
            j = (i + 1) % sides
            tri += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    top, tip = 1 + (len(profile) - 1) * sides, len(rows) - 1
    tri += [[top + i, top + (i + 1) % sides, tip] for i in range(sides)]
    return np.array(rows), np.array(tri)


def proxy_mesh(trees, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One low-poly mesh for all trees: rows of (east, north, ground z, height, crown radius).

    Returns (vertices, triangles). Each tree gets a fixed random turn, lumpy crown edges and a slight lean,
    so no two proxies share a silhouette and none reads as a clean geometric shape.
    """
    trees = np.asarray(trees, dtype=float).reshape(-1, 5)
    rng = np.random.default_rng(seed)
    turns = rng.uniform(0, 2 * math.pi, len(trees))
    conifer = trees[:, 3] >= CONIFER_FROM_M
    verts, faces, count = [], [], 0
    for profile, chosen in ((ROUND_CROWN, ~conifer), (CONIFER_CROWN, conifer)):
        if not chosen.any():
            continue
        unit, tri = _lathe(profile)
        x, y, z, h, r = (trees[chosen, k][:, None] for k in range(5))
        turn = turns[chosen][:, None]
        cos, sin = np.cos(turn), np.sin(turn)
        ux, uy = unit[None, :, 0] * cos - unit[None, :, 1] * sin, unit[None, :, 0] * sin + unit[None, :, 1] * cos
        lumps = rng.uniform(1 - JITTER, 1 + JITTER, (len(x), len(unit)))
        lean = rng.uniform(-JITTER, JITTER, (len(x), 1, 2)) * r[:, :, None] * unit[None, :, 2, None] ** 2
        spread = r * unit[None, :, 3] * lumps
        block = np.stack((x + spread * ux + lean[..., 0], y + spread * uy + lean[..., 1], z + h * unit[None, :, 2]),
                         axis=-1)
        offsets = count + len(unit) * np.arange(len(block))[:, None, None]
        verts.append(block.reshape(-1, 3))
        faces.append((tri[None] + offsets).reshape(-1, 3))
        count += block.shape[0] * block.shape[1]
    if conifer.any():  # a closed four-sided trunk from just below the ground into the crown
        x, y, z, h, r = (trees[conifer, k][:, None] for k in range(5))
        w = np.maximum(0.15, 0.08 * r)
        sx, sy = np.array([-1, 1, 1, -1] * 2)[None], np.array([-1, -1, 1, 1] * 2)[None]
        sz = np.array([0] * 4 + [1] * 4)[None]
        block = np.stack((x + w * sx, y + w * sy, np.where(sz, z + h * CONIFER_CROWN[0][0], z - 0.3)), axis=-1)
        tri = np.array([[i, (i + 1) % 4, 4 + (i + 1) % 4] for i in range(4)]
                       + [[i, 4 + (i + 1) % 4, 4 + i] for i in range(4)] + [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]])
        verts.append(block.reshape(-1, 3))
        faces.append((tri[None] + count + 8 * np.arange(len(block))[:, None, None]).reshape(-1, 3))
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.vstack(verts), np.vstack(faces).astype(int)
