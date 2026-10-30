"""Measured building blocks for the blocking, so video models keep a city's buildings where they really are.

Footprints and heights come from Overture Maps (buildings theme), read straight from its public GeoParquet
release on S3: only the row groups whose bounding boxes overlap the scene are fetched. Where Overture splits
a building into parts (a tower on a podium), the parts are used instead, each with its own height and base.
Buildings with no measured height get one from their floor count, else a default low-rise height.

Each building becomes a flat-roofed prism in the Blender blocking, standing on the terrain mesh as rendered.
Heights are what Overture carries: mostly OpenStreetMap, which in many cities was imported from municipal
lidar surveys. Roof shapes are not modelled.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import urllib.parse

import numpy as np

VERSION = "1"
RELEASE = "2026-09-23.1"
BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
ATTRIBUTION = ("Overture Maps Foundation buildings, release {release}: © OpenStreetMap contributors (ODbL), "
               "Microsoft and Esri Community Maps contributors (CDLA Permissive 2.0), and others")
LICENSE = "ODbL 1.0 (OpenStreetMap-derived rows); CDLA Permissive 2.0 (other rows)"
HALF_SIZE_M = 7_000.0   # buildings within this many metres east/west and north/south of the landing
FLOOR_M = 3.2            # storey height when only a floor count is known
DEFAULT_HEIGHT_M = 7.0   # no height and no floor count: a two-storey building
MIN_AREA_M2 = 15.0       # sheds and kiosks are below a pixel from anywhere but the landing
SIMPLIFY_M = 0.5
SMALL_HOUSE_M2 = 150.0   # a footprint up to about one terraced or detached house


# ---------------------------------------------------------------- reading Overture

class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, in 1 MB blocks, for pyarrow's Parquet reader."""

    BLOCK = 1 << 20

    def __init__(self, url: str, client):
        self.url, self.client, self.position, self.blocks = url, client, 0, {}
        head = client.head(url)
        head.raise_for_status()
        self.size = int(head.headers["content-length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def _block(self, index: int) -> bytes:
        if index not in self.blocks:
            start = index * self.BLOCK
            end = min(self.size, start + self.BLOCK) - 1
            for attempt in range(4):  # under load S3 (or a proxy) sometimes answers 404 for a file that exists
                reply = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
                if reply.status_code != 404 and reply.status_code < 500 or attempt == 3:
                    break
                time.sleep(2 ** attempt)
            reply.raise_for_status()
            self.blocks[index] = reply.content
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        end = min(self.size, self.position + len(buffer))
        out = bytearray()
        while self.position + len(out) < end:
            at = self.position + len(out)
            block = self._block(at // self.BLOCK)
            out += block[at % self.BLOCK:at % self.BLOCK + end - at]
        buffer[:len(out)] = out
        self.position += len(out)
        return len(out)


def _client():
    import httpx
    return httpx.Client(timeout=120, follow_redirects=True)


def _keys(client, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}" + (
            f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        listing = client.get(url).text
        keys += re.findall(r"<Key>([^<]*\.parquet)</Key>", listing)
        more = re.search(r"<NextContinuationToken>([^<]*)<", listing)
        if not more:
            return keys
        token = more.group(1)


def _bbox_stats(row_group) -> tuple | None:
    found = {}
    for i in range(row_group.num_columns):
        column = row_group.column(i)
        if column.path_in_schema.startswith("bbox.") and column.statistics and column.statistics.has_min_max:
            found[column.path_in_schema] = (column.statistics.min, column.statistics.max)
    try:
        return (found["bbox.xmin"][0], found["bbox.ymin"][0], found["bbox.xmax"][1], found["bbox.ymax"][1])
    except KeyError:
        return None


def row_group_index(kind: str, cache_dir: Path, release: str = RELEASE, theme: str = "buildings") -> list:
    """[key, [[row group, west, south, east, north], ...]] for every file of one Overture type, cached."""
    import pyarrow.parquet as pq
    name = f"{kind}-row-groups.json" if theme == "buildings" else f"{theme}-{kind}-row-groups.json"
    path = cache_dir / "overture" / release / name
    if path.is_file():
        return json.loads(path.read_text())
    with _client() as client:
        keys = _keys(client, f"release/{release}/theme={theme}/type={kind}/")

        def footer(key: str) -> list:
            with RangeFile(f"{BUCKET}/{key}", client) as handle:
                meta = pq.ParquetFile(handle).metadata
                return [key, [[i, *box] for i in range(meta.num_row_groups)
                              if (box := _bbox_stats(meta.row_group(i)))]]

        with ThreadPoolExecutor(16) as pool:
            index = list(pool.map(footer, keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index) + "\n")
    return index


def read_overture(kind: str, box: tuple[float, float, float, float], cache_dir: Path, columns: list[str],
                  release: str = RELEASE, theme: str = "buildings") -> list[dict]:
    """Rows of one Overture type whose bounding-box centre is inside ``box`` (west, south, east, north)."""
    import pyarrow.parquet as pq
    west, south, east, north = box
    wanted = {}
    for key, groups in row_group_index(kind, cache_dir, release, theme):
        hits = [g[0] for g in groups if g[1] <= east and g[3] >= west and g[2] <= north and g[4] >= south]
        if hits:
            wanted[key] = hits
    with _client() as client:
        def read(item) -> list[dict]:
            key, groups = item
            with RangeFile(f"{BUCKET}/{key}", client) as handle:
                table = pq.ParquetFile(handle).read_row_groups(groups, columns=columns + ["bbox"])
            rows = []
            for row in table.to_pylist():
                b = row.pop("bbox")
                if west <= (b["xmin"] + b["xmax"]) / 2 <= east and south <= (b["ymin"] + b["ymax"]) / 2 <= north:
                    rows.append(row)
            return rows

        with ThreadPoolExecutor(8) as pool:
            return [row for rows in pool.map(read, wanted.items()) for row in rows]


# ---------------------------------------------------------------- footprints and heights

def building_height(height: float | None, floors: int | None) -> tuple[float, str]:
    """Height above the ground and where it came from: measured, from the floor count, or the default."""
    if height is not None and math.isfinite(height) and height > 0:
        return float(min(height, 650.0)), "height"
    if floors:
        return float(floors) * FLOOR_M, "floors"
    return DEFAULT_HEIGHT_M, "default"


def footprint_rings(geometries, transform) -> list[np.ndarray | None]:
    """Each footprint's outer ring in local metres, counter-clockwise, simplified, without the closing point.
    A multipolygon keeps its largest part. None for footprints too small to matter. ``transform`` maps an
    (N, 2) array of lon/lat to local metres."""
    import shapely
    geometries = np.asarray(geometries, dtype=object)
    multi = shapely.get_type_id(geometries) == 6
    for i in np.flatnonzero(multi):
        geometries[i] = max(geometries[i].geoms, key=lambda part: part.area)
    usable = shapely.get_type_id(geometries) == 3
    rings: list[np.ndarray | None] = [None] * len(geometries)
    if not usable.any():
        return rings
    exteriors = shapely.get_exterior_ring(geometries[usable])
    coords, owner = shapely.get_coordinates(exteriors, return_index=True)
    exteriors = shapely.set_coordinates(exteriors.copy(), transform(coords))
    polygons = shapely.simplify(shapely.polygons(exteriors), SIMPLIFY_M, preserve_topology=True)
    polygons = shapely.orient_polygons(polygons, exterior_cw=False)
    good = (shapely.get_type_id(polygons) == 3) & (shapely.area(polygons) >= MIN_AREA_M2)
    coords, owner = shapely.get_coordinates(shapely.get_exterior_ring(polygons), return_index=True)
    bounds = np.searchsorted(owner, np.arange(len(polygons) + 1))
    for slot, index in enumerate(np.flatnonzero(usable)):
        ring = coords[bounds[slot]:bounds[slot + 1] - 1]
        if good[slot] and len(ring) >= 3:
            rings[index] = ring
    return rings


def fetch_buildings(scene: dict, folder: Path, *, cache_dir: Path, half_size_m: float = HALF_SIZE_M,
                    release: str = RELEASE) -> dict:
    """Write ``buildings-<key>.npz`` (rings in local east/north metres, base and height above ground) and a
    manifest beside it; return the manifest with its ``path``."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    half = min(half_size_m, float(scene["radius_m"]))
    ox, oy = scene["origin_projected_m"]
    key = hashlib.sha256(json.dumps([VERSION, release, scene["projected_crs"], round(ox, 1), round(oy, 1),
                                     round(half, 1)]).encode()).hexdigest()[:16]
    manifest_path = folder / f"buildings-{key}.json"
    if manifest_path.is_file():
        return {**json.loads(manifest_path.read_text()), "path": str(manifest_path)}
    import shapely  # only when reading Overture: a ComfyUI that finds the prefetched blocks needs neither
    from pyproj import Transformer

    to_ll = Transformer.from_crs(scene["projected_crs"], "EPSG:4326", always_xy=True)
    corners = [to_ll.transform(ox + dx, oy + dy) for dx in (-half, half) for dy in (-half, half)]
    box = (min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners),
           max(c[1] for c in corners))
    columns = ["id", "geometry", "height", "num_floors", "min_height", "is_underground", "sources"]
    buildings = read_overture("building", box, cache_dir, columns + ["has_parts"], release)
    parts = read_overture("building_part", box, cache_dir, columns + ["building_id"], release)
    # A building with parts is drawn as its parts (a tower on a podium), if all of them were read.
    parted = {}
    for part in parts:
        parted.setdefault(part["building_id"], []).append(part)
    rows = []
    for building in buildings:
        if building.get("is_underground"):
            continue
        own = parted.get(building["id"]) if building.get("has_parts") else None
        rows += [p for p in own if not p.get("is_underground")] if own else [building]
    to_local = Transformer.from_crs("EPSG:4326", scene["projected_crs"], always_xy=True)

    def transform(coords: np.ndarray) -> np.ndarray:
        x, y = to_local.transform(coords[:, 0], coords[:, 1])
        return np.column_stack((np.asarray(x) - ox, np.asarray(y) - oy))

    rings = footprint_rings(shapely.from_wkb([r["geometry"] for r in rows]), transform)
    kept, sources = [], {"height": 0, "floors": 0, "default": 0}
    for row, ring in zip(rows, rings):
        if ring is None or np.abs(ring).max() > half:
            continue
        height, source = building_height(row.get("height"), row.get("num_floors"))
        base = float(row.get("min_height") or 0.0)
        if height <= base + 1.0:
            continue
        sources[source] += 1
        kept.append((ring, base, height, (row.get("sources") or [{}])[0].get("dataset") or ""))
    offsets = np.cumsum([0] + [len(ring) for ring, *_ in kept]).astype("int64")
    data_path = manifest_path.with_suffix(".npz")
    np.savez_compressed(data_path,
                        xy=np.concatenate([ring for ring, *_ in kept]).astype("float32") if kept else np.zeros((0, 2), "float32"),
                        offsets=offsets, base_m=np.array([k[1] for k in kept], "float32"),
                        height_m=np.array([k[2] for k in kept], "float32"))
    heights = np.array([k[2] for k in kept]) if kept else np.zeros(0)
    areas = footprint_areas([ring for ring, *_ in kept])
    datasets = {}
    for *_, dataset in kept:
        datasets[dataset] = datasets.get(dataset, 0) + 1
    manifest = {"key": key, "version": VERSION, "release": release, "crs": scene["projected_crs"],
                "origin_projected_m": [ox, oy], "half_size_m": half, "count": len(kept),
                "parts_used": sum(1 for r in rows if "building_id" in r), "height_source": sources,
                "datasets": dict(sorted(datasets.items(), key=lambda item: -item[1])),
                "height_p50_m": round(float(np.median(heights)), 1) if len(heights) else None,
                "height_p95_m": round(float(np.percentile(heights, 95)), 1) if len(heights) else None,
                "height_max_m": round(float(heights.max()), 1) if len(heights) else None,
                "towers_over_100m": int((heights > 100).sum()), **footprint_stats(areas), "data": str(data_path),
                "attribution": ATTRIBUTION.format(release=release), "license": LICENSE}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "path": str(manifest_path)}


def load_buildings(manifest: dict) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Rings, base heights and heights above the ground from a manifest's npz."""
    with np.load(manifest["data"]) as data:  # each data[...] decompresses the whole array: read once
        xy, offsets = data["xy"].astype("float64"), data["offsets"]
        base, height = data["base_m"].astype("float64"), data["height_m"].astype("float64")
    return np.split(xy, offsets[1:-1]), base, height


def building_report(manifest: dict) -> str:
    sources = manifest["height_source"]
    return (f"BUILDINGS: {manifest['count']:,} from Overture {manifest['release']} within "
            f"{manifest['half_size_m'] / 1000:g} km ({sources['height']:,} measured heights, {sources['floors']:,} "
            f"from floor counts, {sources['default']:,} at the {DEFAULT_HEIGHT_M:g} m default); median "
            f"{manifest['height_p50_m']} m, tallest {manifest['height_max_m']} m, {manifest['towers_over_100m']} over "
            f"100 m · {manifest['license']}")


# ---------------------------------------------------------------- geometry

def extrude(rings: list[np.ndarray], bottoms: np.ndarray, tops: np.ndarray) -> tuple:
    """Flat-roofed prisms: vertices (N, 3), loop vertex indices, and each face's loop start and size.

    Each ring (counter-clockwise, no closing point) gets one roof n-gon and one quad per wall, facing out.
    """
    sizes = np.array([len(ring) for ring in rings], dtype="int64")
    if not len(sizes):
        return np.zeros((0, 3)), np.zeros(0, "int64"), np.zeros(0, "int64"), np.zeros(0, "int64")
    xy = np.concatenate(rings)
    starts = np.concatenate(([0], np.cumsum(2 * sizes)[:-1]))  # each building: n bottom then n top vertices
    vertices = np.zeros((2 * len(xy), 3))
    index = np.repeat(starts, sizes) + (np.arange(len(xy)) - np.repeat(np.cumsum(sizes) - sizes, sizes))
    vertices[index, :2] = xy
    vertices[index, 2] = np.repeat(bottoms, sizes)
    top_index = index + np.repeat(sizes, sizes)
    vertices[top_index, :2] = xy
    vertices[top_index, 2] = np.repeat(tops, sizes)
    loops, totals = [], []
    for start, n in zip(starts, sizes):
        bottom = start + np.arange(n)
        top = bottom + n
        loops.append(top)  # roof, counter-clockwise seen from above
        totals.append(n)
        nxt = np.roll(np.arange(n), -1)
        walls = np.column_stack((bottom, bottom[nxt], top[nxt], top))  # outward for a counter-clockwise ring
        loops.append(walls.ravel())
        totals += [4] * n
    loops = np.concatenate(loops).astype("int64")
    totals = np.array(totals, dtype="int64")
    loop_starts = np.concatenate(([0], np.cumsum(totals)[:-1]))
    return vertices, loops, loop_starts, totals


def clear_of_path(rings: list[np.ndarray], tops: np.ndarray, points, margin_m: float = 25.0,
                  headroom_m: float = 15.0) -> np.ndarray:
    """Mask of buildings to keep: those that do not stand within ``margin_m`` of a camera point and reach to
    within ``headroom_m`` below it."""
    keep = np.ones(len(rings), dtype=bool)
    if not len(points) or not len(rings):
        return keep
    points = np.asarray(points, dtype=float)
    lows = np.array([ring.min(axis=0) for ring in rings]) - margin_m
    highs = np.array([ring.max(axis=0) for ring in rings]) + margin_m
    for x, y, z in points:
        near = (lows[:, 0] <= x) & (highs[:, 0] >= x) & (lows[:, 1] <= y) & (highs[:, 1] >= y)
        keep &= ~(near & (tops > z - headroom_m))
    return keep


def footprint_areas(rings: list[np.ndarray]) -> np.ndarray:
    """Shoelace areas in m² of counter-clockwise rings."""
    return np.array([0.5 * float(np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1)))
                     for r in rings])


def footprint_stats(areas: np.ndarray) -> dict:
    return {"footprint_p50_m2": round(float(np.median(areas))) if len(areas) else None,
            "small_share": round(float((areas < SMALL_HOUSE_M2).mean()), 3) if len(areas) else None}


def city_summary(manifest: dict) -> dict:
    stats = {key: manifest[key] for key in ("footprint_p50_m2", "small_share") if key in manifest}
    if not stats and manifest.get("data"):  # manifests written before footprint stats: measure the rings
        stats = footprint_stats(footprint_areas(load_buildings(manifest)[0]))
    return {"count": manifest["count"], "height_p50_m": manifest["height_p50_m"],
            "height_p95_m": manifest["height_p95_m"], "height_max_m": manifest["height_max_m"],
            "towers_over_100m": manifest["towers_over_100m"], **stats}


def city_text(summary: dict) -> str:
    """The measured buildings, in words, for prompts."""
    storeys = max(1, round(summary["height_p50_m"] / FLOOR_M))
    if summary.get("small_share", 0) >= 0.5:  # mostly house-sized footprints: rows of houses, not slabs
        words = (f"a dense city of real buildings, mostly narrow {storeys}–{storeys + 1} storey houses side by side "
                 "in long rows along a street grid, each house its own facade colour, with a few larger apartment "
                 "blocks")
    else:
        words = (f"a dense city of real buildings, mostly {storeys}–{storeys + 2} storey houses and apartment "
                 "blocks packed along a street grid")
    if summary.get("towers_over_100m", 0) >= 5:
        words += (f", rising to a cluster of downtown skyscrapers up to {summary['height_max_m']:.0f} m tall, glass "
                  "and stone towers")
    return (words + ". Each pale block in Image 1 is one real building: paint it as a finished building with "
            "windows, facades and rooftops at exactly that footprint and height, and do not add, move or remove "
            "buildings")
