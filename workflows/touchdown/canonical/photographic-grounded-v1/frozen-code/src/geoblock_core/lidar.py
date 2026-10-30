"""Which USGS 3DEP lidar surveys cover a place, from the public Entwine Point Tile (EPT) index.

The index (Hobu's ``resources.geojson``, about 9 MB, one MultiPolygon per survey) is read once and
cached. It answers "is there lidar here, from when, how dense" without touching any point data; the
points themselves stay in the public ``usgs-lidar-public`` bucket until a shot asks for them.

``iter_points`` / ``read_points`` then stream only the octree nodes inside a box, down to the depth that
gives roughly the requested point spacing (needs ``laspy[lazrs]``: ``uv sync --extra buildings``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import re
import time

import httpx
import numpy as np
from pyproj import CRS, Transformer

INDEX_URL = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
LICENSE = "Public domain (U.S. Geological Survey 3DEP)"
ATTRIBUTION = "U.S. Geological Survey, 3D Elevation Program lidar, via the usgs-lidar-public EPT mirror"


def load_index(cache: Path, url: str | None = None) -> list[dict]:
    url = url or os.environ.get("GEOBLOCK_EPT_INDEX") or INDEX_URL
    path = cache / "usgs_lidar" / "resources.geojson"
    if not path.is_file():
        if url.startswith(("http://", "https://")):
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                text = response.text
        else:
            text = Path(url).read_text()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return json.loads(path.read_text())["features"]


def _rings(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    return []


def _inside_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def contains(geometry: dict, longitude: float, latitude: float) -> bool:
    for polygon in _rings(geometry):
        if polygon and _inside_ring(longitude, latitude, polygon[0]) and not any(
                _inside_ring(longitude, latitude, hole) for hole in polygon[1:]):
            return True
    return False


def area_km2(geometry: dict) -> float:
    """Approximate polygon area on the sphere (shoelace in locally scaled degrees)."""
    total = 0.0
    for polygon in _rings(geometry):
        for index, ring in enumerate(polygon):
            if len(ring) < 3:
                continue
            lat0 = math.radians(sum(point[1] for point in ring) / len(ring))
            kx, ky = 111.32 * math.cos(lat0), 110.57
            area = 0.0
            for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
                area += (x0 * kx) * (y1 * ky) - (x1 * kx) * (y0 * ky)
            total += (-1 if index else 1) * abs(area) / 2
    return total


def survey_year(name: str) -> int | None:
    """Acquisition year from a survey name: the first four-digit year (a later one is usually the
    delivery, as in ``..._2017_LAS_2019``), else a two-digit work-unit suffix such as ``_B22``."""
    years = re.findall(r"(?<![0-9])(19[89][0-9]|20[0-4][0-9])(?![0-9])", name)
    if years:
        return int(years[0])
    suffix = re.search(r"_[A-Z]([0-4][0-9])$", name)
    return 2000 + int(suffix.group(1)) if suffix else None


def surveys_at(latitude: float, longitude: float, cache: Path, *, index: list[dict] | None = None) -> list[dict]:
    """3DEP lidar surveys whose footprint contains the point, newest first."""
    found = []
    for feature in index if index is not None else load_index(cache):
        geometry = feature.get("geometry") or {}
        if not contains(geometry, longitude, latitude):
            continue
        props = feature["properties"]
        area = area_km2(geometry)
        found.append({
            "name": props["name"],
            "year": survey_year(props["name"]),
            "points": int(props.get("count") or 0),
            "density_pts_m2": round(props["count"] / (area * 1e6), 1) if area > 0 and props.get("count") else None,
            "ept": props["url"],
            "license": LICENSE,
        })
    return sorted(found, key=lambda item: (item["year"] or 0, item["density_pts_m2"] or 0), reverse=True)


# USGS 3DEP 1 m DEMs: bare-earth rasters made from the lidar surveys, one folder per project in the public
# prd-tnm bucket, 10 km tiles named USGS_1M_<utm zone>_x<easting // 10 km>y<ceil(northing / 10 km)>_<project>.
DEM1M_ROOT = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects"
DEM1M_LIST = "https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/1m/Projects/&delimiter=/"
DEM1M_LICENSE = "Public domain (U.S. Geological Survey 3DEP)"

# Rough state boxes (west, south, east, north), only to pick which projects to probe; names start with the state.
STATES = {
    "AL": (-88.5, 30.1, -84.9, 35.0), "AZ": (-114.9, 31.3, -109.0, 37.0), "AR": (-94.7, 33.0, -89.6, 36.5),
    "CA": (-124.5, 32.5, -114.1, 42.0), "CO": (-109.1, 36.9, -102.0, 41.0), "CT": (-73.7, 40.9, -71.8, 42.1),
    "DE": (-75.8, 38.4, -75.0, 39.9), "FL": (-87.7, 24.4, -80.0, 31.0), "GA": (-85.6, 30.3, -80.8, 35.0),
    "ID": (-117.3, 41.9, -111.0, 49.0), "IL": (-91.6, 36.9, -87.0, 42.5), "IN": (-88.1, 37.7, -84.8, 41.8),
    "IA": (-96.7, 40.3, -90.1, 43.5), "KS": (-102.1, 36.9, -94.6, 40.0), "KY": (-89.6, 36.5, -81.9, 39.2),
    "LA": (-94.1, 28.9, -88.8, 33.1), "ME": (-71.1, 43.0, -66.9, 47.5), "MD": (-79.5, 37.9, -75.0, 39.8),
    "MA": (-73.5, 41.2, -69.9, 42.9), "MI": (-90.5, 41.7, -82.1, 48.3), "MN": (-97.3, 43.5, -89.5, 49.4),
    "MS": (-91.7, 30.1, -88.1, 35.0), "MO": (-95.8, 35.9, -89.1, 40.7), "MT": (-116.1, 44.3, -104.0, 49.0),
    "NE": (-104.1, 40.0, -95.3, 43.0), "NV": (-120.0, 35.0, -114.0, 42.0), "NH": (-72.6, 42.7, -70.6, 45.4),
    "NJ": (-75.6, 38.9, -73.9, 41.4), "NM": (-109.1, 31.3, -103.0, 37.0), "NY": (-79.8, 40.5, -71.8, 45.1),
    "NC": (-84.4, 33.8, -75.4, 36.6), "ND": (-104.1, 45.9, -96.5, 49.0), "OH": (-84.9, 38.4, -80.5, 42.0),
    "OK": (-103.0, 33.6, -94.4, 37.0), "OR": (-124.6, 41.9, -116.4, 46.3), "PA": (-80.6, 39.7, -74.7, 42.3),
    "RI": (-71.9, 41.1, -71.1, 42.1), "SC": (-83.4, 32.0, -78.5, 35.3), "SD": (-104.1, 42.4, -96.4, 46.0),
    "TN": (-90.4, 34.9, -81.6, 36.7), "TX": (-106.7, 25.8, -93.5, 36.5), "UT": (-114.1, 37.0, -109.0, 42.0),
    "VT": (-73.5, 42.7, -71.4, 45.1), "VA": (-83.7, 36.5, -75.2, 39.5), "WA": (-124.8, 45.5, -116.9, 49.0),
    "WV": (-82.7, 37.2, -77.7, 40.7), "WI": (-92.9, 42.4, -86.8, 47.1), "WY": (-111.1, 41.0, -104.0, 45.0),
}


def states_at(latitude: float, longitude: float, margin: float = 0.25) -> list[str]:
    return [code for code, (w, s, e, n) in STATES.items()
            if w - margin <= longitude <= e + margin and s - margin <= latitude <= n + margin]


def dem1m_projects(cache: Path) -> list[str]:
    """All 3DEP 1 m DEM project folders (one paginated bucket listing, cached)."""
    path = cache / "usgs_1m" / "projects.txt"
    if path.is_file():
        return path.read_text().split()
    names, token = [], None
    with httpx.Client(timeout=60) as client:
        while True:
            response = client.get(DEM1M_LIST, params={"continuation-token": token} if token else None)
            response.raise_for_status()
            names += re.findall(r"<Prefix>StagedProducts/Elevation/1m/Projects/([^<]+)/</Prefix>", response.text)
            match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", response.text)
            if not match:
                break
            token = match.group(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n")
    return names


def dem1m_tile(zone: int, easting: float, northing: float, project: str) -> str:
    name = f"USGS_1M_{zone}_x{int(easting // 10_000)}y{math.ceil(northing / 10_000)}_{project}.tif"
    return f"{DEM1M_ROOT}/{project}/TIFF/{name}"


def dem1m_projects_at(latitude: float, longitude: float, cache: Path, *, probe=None) -> list[dict]:
    """3DEP 1 m DEM projects with a tile under the point, newest first. Probes only projects named for the
    states around the point (plus the few not named by state), one HEAD request each."""
    from concurrent.futures import ThreadPoolExecutor
    from pyproj import Transformer
    states = states_at(latitude, longitude)
    projects = [name for name in dem1m_projects(cache)
                if name[:2] in states or not re.match(r"[A-Z]{2}_", name)]
    zone = int((longitude + 180) // 6) + 1
    zones = sorted({zone} | {z for z in (zone - 1, zone + 1)
                             if abs(longitude - (-180 + 6 * (z - 1) + 3)) < 3.6})  # near a zone edge
    candidates = []
    for z in zones:
        east, north = Transformer.from_crs("EPSG:4326", f"EPSG:{26900 + z}", always_xy=True).transform(longitude, latitude)
        candidates += [(project, z, dem1m_tile(z, east, north, project)) for project in projects]

    def exists(url: str) -> bool:
        if probe:
            return probe(url)
        try:
            return httpx.head(url, timeout=30).status_code == 200
        except httpx.HTTPError:
            return False

    with ThreadPoolExecutor(16) as pool:
        found = [(project, z, url) for (project, z, url), ok in zip(candidates, pool.map(exists, [c[2] for c in candidates])) if ok]
    result = [{"project": project, "zone": z, "year": survey_year(project), "tile": url, "resolution_m": 1,
               "license": DEM1M_LICENSE} for project, z, url in found]
    return sorted(result, key=lambda item: item["year"] or 0, reverse=True)


# ---------------------------------------------------------------- point data (EPT octree)

USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"


def _get(client: httpx.Client, url: str, cache: Path | None = None) -> bytes:
    if cache and cache.is_file():
        return cache.read_bytes()
    response = client.get(url)
    response.raise_for_status()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_suffix(cache.suffix + ".part")
        part.write_bytes(response.content)
        part.replace(cache)
    return response.content


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=180, follow_redirects=True)


def project_for(longitude: float, latitude: float, cache_dir: Path) -> dict | None:
    """The newest survey at the point as {name, ept (base URL), points, candidates}, or None."""
    surveys = surveys_at(latitude, longitude, cache_dir)
    if not surveys:
        return None
    best = surveys[0]
    return {"name": best["name"], "ept": best["ept"].rsplit("/", 1)[0] + "/", "points": best["points"],
            "candidates": [item["name"] for item in surveys]}


def _vertical_scale(srs: dict) -> float:
    """Metres per vertical unit; a few surveys deliver heights in US survey feet."""
    wkt = (srs.get("wkt") or "").lower()
    vertical = wkt[wkt.find("vert_cs"):] if "vert_cs" in wkt else wkt[wkt.find("vertcrs"):] if "vertcrs" in wkt else ""
    return 0.3048006096 if "foot" in vertical or "feet" in vertical else 1.0


def node_depth(bounds: list[float], span: int, spacing: float) -> int:
    """Shallowest octree depth whose voxel grid is at least as fine as ``spacing`` (EPT units)."""
    size = bounds[3] - bounds[0]
    return max(0, math.ceil(math.log2(size / span / spacing)))


def read_points(ept: str, bounds: tuple[float, float, float, float], crs: str, cache_dir: Path, *,
                spacing_m: float = 0.5) -> dict[str, np.ndarray]:
    """All points inside ``bounds`` at once; see ``iter_points``. Large areas need a lot of memory."""
    parts = list(iter_points(ept, bounds, crs, cache_dir, spacing_m=spacing_m))
    if not parts:
        raise ValueError(f"The lidar survey {ept} has no points in this area")
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


MAX_POINTS = 150_000_000


def iter_points(ept: str, bounds: tuple[float, float, float, float], crs: str, cache_dir: Path, *,
                spacing_m: float = 0.5, max_points: int = MAX_POINTS):
    """Points inside ``bounds`` (minx, miny, maxx, maxy in ``crs``), one octree node at a time.

    Yields dicts with x, y, z (``crs`` metres), classification, return_number, number_of_returns.
    """
    import laspy

    cache = cache_dir / ept.rstrip("/").rsplit("/", 1)[-1]
    with _client() as client:
        meta = json.loads(_get(client, ept + "ept.json", cache / "ept.json"))
        srs = meta["srs"]
        source = CRS.from_user_input(f"EPSG:{srs['horizontal']}" if srs.get("horizontal") else srs["wkt"])
        to_ept = Transformer.from_crs(crs, source, always_xy=True)
        xs, ys = to_ept.transform(*np.meshgrid(np.linspace(bounds[0], bounds[2], 9),
                                               np.linspace(bounds[1], bounds[3], 9)))
        query = (xs.min(), ys.min(), xs.max(), ys.max())
        # EPT units per metre on the ground (Web Mercator is stretched by 1/cos(latitude)).
        units_per_m = math.hypot(xs[0, -1] - xs[0, 0], ys[0, -1] - ys[0, 0]) / (bounds[2] - bounds[0])
        cube = meta["bounds"]
        max_depth = node_depth(cube, meta.get("span", 128), spacing_m * units_per_m)
        hierarchy = json.loads(_get(client, ept + "ept-hierarchy/0-0-0-0.json", cache / "ept-hierarchy/0-0-0-0.json"))

        def box(key: str):
            depth, *index = (int(part) for part in key.split("-"))
            scale = 2 ** depth
            low = [cube[a] + n * (cube[a + 3] - cube[a]) / scale for a, n in enumerate(index)]
            high = [cube[a] + (n + 1) * (cube[a + 3] - cube[a]) / scale for a, n in enumerate(index)]
            return low, high

        keys, todo = [], ["0-0-0-0"]
        while todo:
            key = todo.pop()
            low, high = box(key)
            if not (low[0] < query[2] and high[0] > query[0] and low[1] < query[3] and high[1] > query[1]):
                continue
            count = hierarchy.get(key)
            if count is None:
                continue
            if count < 0:  # the subtree's counts live in their own hierarchy file
                hierarchy.update(json.loads(_get(client, ept + f"ept-hierarchy/{key}.json",
                                                 cache / f"ept-hierarchy/{key}.json")))
                count = hierarchy[key]
            if count > 0:
                keys.append(key)
            depth, ix, iy, iz = (int(part) for part in key.split("-"))
            if depth < max_depth:
                todo += [f"{depth + 1}-{2 * ix + dx}-{2 * iy + dy}-{2 * iz + dz}"
                         for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]

        source_to_crs = Transformer.from_crs(source, crs, always_xy=True)
        vertical = _vertical_scale(srs)

        missing = []

        def load(key: str) -> dict[str, np.ndarray]:
            for attempt in range(3):
                try:
                    data = _get(client, ept + f"ept-data/{key}.laz", cache / f"ept-data/{key}.laz")
                    break
                except (httpx.HTTPStatusError, httpx.TransportError) as error:
                    # Under load S3 (or a proxy) sometimes answers 404 or drops the connection for a tile that
                    # exists, and a few surveys list tiles the bucket really doesn't have.
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if status is not None and status != 404 and status < 500:
                        raise
                    time.sleep(2 ** attempt)
            else:
                missing.append(key)
                empty = np.zeros(0)
                return {name: empty for name in ("x", "y", "z", "classification", "return_number",
                                                 "number_of_returns")}
            las = laspy.read(io.BytesIO(data))
            x, y = np.asarray(las.x), np.asarray(las.y)
            near = (x > query[0]) & (x < query[2]) & (y > query[1]) & (y < query[3])
            x, y = source_to_crs.transform(x[near], y[near])
            inside = (x > bounds[0]) & (x < bounds[2]) & (y > bounds[1]) & (y < bounds[3])
            return {"x": np.asarray(x)[inside], "y": np.asarray(y)[inside],
                    "z": np.asarray(las.z)[near][inside] * vertical,
                    "classification": np.asarray(las.classification)[near][inside],
                    "return_number": np.asarray(las.return_number)[near][inside],
                    "number_of_returns": np.asarray(las.number_of_returns)[near][inside]}

        # Dense city surveys (80+ points/m2) would mean a billion points for a few km: drop the finest levels
        # until the read fits the budget. The coarser levels still sample the whole area evenly.
        per_depth: dict[int, int] = {}
        for key in keys:
            per_depth[int(key.split("-")[0])] = per_depth.get(int(key.split("-")[0]), 0) + hierarchy[key]
        deepest = max(per_depth, default=0)
        while deepest > 0 and sum(per_depth.values()) > max_points:
            per_depth.pop(deepest)
            deepest -= 1
        if deepest < max(int(key.split("-")[0]) for key in keys or ["0"]):
            print(f"lidar: dense survey, reading down to octree level {deepest}", flush=True)
        keys = [key for key in keys if int(key.split("-")[0]) <= deepest]
        # A few nodes in flight at a time, so memory stays near one node per worker.
        total = sum(hierarchy[key] for key in keys)
        print(f"lidar: reading {len(keys)} tiles, {total / 1e6:.0f} M points", flush=True)
        step = max(16, len(keys) // 10)
        with ThreadPoolExecutor(8) as pool:
            for start in range(0, len(keys), 16):
                yield from pool.map(load, keys[start:start + 16])
                if (start + 16) // step > start // step:
                    print(f"lidar: {min(start + 16, len(keys))} of {len(keys)} tiles", flush=True)
        if missing:
            print(f"lidar: {len(missing)} of {len(keys)} tiles are listed but missing from the bucket", flush=True)
