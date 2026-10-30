"""Measure Woodside roof heights from USGS LPC, using OSM only for footprints.

EPT is an official public mirror of the USGS point cloud. The selected survey
classifies ground (2), but leaves roofs and vegetation together in class 1.
Consequently a footprint passes only when it has a dense, coherent elevated
surface; uncertain footprints are recorded as rejected, never given a guessed
height. Run with ``uv run --extra buildings python scripts/prepare_measured_buildings.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import laspy
import numpy as np
from pyproj import Transformer
import rasterio
from shapely import contains_xy
from shapely.geometry import Point, Polygon


ROOT = Path(__file__).resolve().parents[1]
EPT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_CaliforniaGaps_3_B23/"
LAT, LON = 37.4291056, -122.2580225


def fetch(url: str, path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read()
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(data)
    temp.replace(path)
    return data


def intersects(key: str, bounds: list[float], query: tuple[float, ...]) -> bool:
    depth, ix, iy, iz = (int(part) for part in key.split("-"))
    scale = 2 ** depth
    low = [bounds[j] + index * (bounds[j + 3] - bounds[j]) / scale
           for j, index in enumerate((ix, iy, iz))]
    high = [bounds[j] + (index + 1) * (bounds[j + 3] - bounds[j]) / scale
            for j, index in enumerate((ix, iy, iz))]
    return all(low[j] < query[j + 3] and high[j] > query[j] for j in range(3))


def ept_keys(cache: Path, query: tuple[float, ...], max_depth: int = 12) -> tuple[list[str], dict]:
    meta = json.loads(fetch(EPT + "ept.json", cache / "ept.json"))
    bounds = meta["bounds"]
    hierarchy = json.loads(fetch(EPT + "ept-hierarchy/0-0-0-0.json",
                                 cache / "ept-hierarchy/0-0-0-0.json"))
    todo = ["0-0-0-0"]
    keys = []
    while todo:
        key = todo.pop()
        if not intersects(key, bounds, query):
            continue
        count = hierarchy.get(key)
        if count is None:
            continue
        if count < 0:
            hierarchy.update(json.loads(fetch(EPT + f"ept-hierarchy/{key}.json",
                                               cache / f"ept-hierarchy/{key}.json")))
            count = hierarchy[key]
        if count > 0:
            keys.append(key)
        depth, ix, iy, iz = (int(part) for part in key.split("-"))
        if depth < max_depth:
            todo.extend(f"{depth+1}-{ix*2+dx}-{iy*2+dy}-{iz*2+dz}"
                        for dx in range(2) for dy in range(2) for dz in range(2))
    return keys, meta


def osm_footprints(path: Path, origin: tuple[float, float], radius: float) -> list[dict]:
    root = ET.parse(path).getroot()
    nodes = {n.attrib["id"]: (float(n.attrib["lon"]), float(n.attrib["lat"]))
             for n in root.findall("node")}
    project = Transformer.from_crs(4326, 26910, always_xy=True)
    out = []
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if "building" not in tags:
            continue
        coordinates = [project.transform(*nodes[ref.attrib["ref"]])
                       for ref in way.findall("nd") if ref.attrib["ref"] in nodes]
        if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
            continue
        poly = Polygon(coordinates)
        if not poly.is_valid or poly.area < 15 or poly.distance(Point(origin)) > radius:
            continue
        out.append({"id": way.attrib["id"], "polygon": poly, "tags": tags})
    return out


def measure(footprint: dict, x: np.ndarray, y: np.ndarray, z: np.ndarray,
            classification: np.ndarray, dem: rasterio.DatasetReader) -> dict:
    poly = footprint["polygon"]
    inner = poly.buffer(-1.0)
    result = {"osm_way_id": footprint["id"], "name": footprint["tags"].get("name"),
              "area_m2": round(poly.area, 1), "footprint_utm_m":
              [[round(a, 2), round(b, 2)] for a, b in poly.exterior.coords]}
    if inner.is_empty:
        result.update(status="rejected", reason="footprint too narrow for one-metre roof inset")
        return result
    minx, miny, maxx, maxy = inner.bounds
    candidate = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    indices = np.flatnonzero(candidate)
    if len(indices) < 25:
        result.update(status="rejected", reason="too few lidar returns in footprint")
        return result
    inside = contains_xy(inner, x[indices], y[indices])
    indices = indices[inside]
    ground = float(next(dem.sample([(poly.centroid.x, poly.centroid.y)]))[0])
    if not np.isfinite(ground) or ground == dem.nodata:
        result.update(status="rejected", reason="no valid ground elevation")
        return result
    # USGS class 1 includes roofs and canopy. A coherent roof should cover
    # several separated parts of the footprint at a similar elevation.
    roof = indices[(classification[indices] == 1) & (z[indices] > ground + 2.5)]
    if len(roof) < max(20, int(inner.area * 0.08)):
        result.update(status="rejected", reason="insufficient elevated returns")
        return result
    heights = z[roof] - ground
    p20, p50, p80 = np.percentile(heights, [20, 50, 80])
    if p50 < 3 or p50 > 35 or p80 - p20 > 4.5:
        result.update(status="rejected", reason="elevated surface inconsistent with a roof")
        return result
    # Require elevated points across a coarse 3x3 footprint grid, avoiding a
    # single tree crown whose points happen to fall inside an OSM polygon.
    occupied = {(min(2, int(3 * (x[i] - minx) / max(maxx-minx, .01))),
                 min(2, int(3 * (y[i] - miny) / max(maxy-miny, .01)))) for i in roof}
    if len(occupied) < 4:
        result.update(status="rejected", reason="elevated returns cover too little footprint")
        return result
    result.update(status="accepted", ground_z_m=round(ground, 2),
                  roof_z_m=round(ground + float(p50), 2), height_m=round(float(p50), 2),
                  roof_returns=len(roof), occupied_grid_cells=len(occupied),
                  height_p20_p80_m=[round(float(p20), 2), round(float(p80), 2)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-m", type=float, default=200)
    parser.add_argument("--osm", type=Path, default=ROOT / "data/woodside_library_osm_600m.xml")
    parser.add_argument("--dem", type=Path, default=ROOT / "outputs/woodside_library_3dep/terrain_metric.tif")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/woodside_library_3dep/measured_buildings.json")
    args = parser.parse_args()
    cache = ROOT / "data/woodside_lpc_ept"
    to_mercator = Transformer.from_crs(4326, 3857, always_xy=True)
    to_utm = Transformer.from_crs(3857, 26910, always_xy=True)
    origin = Transformer.from_crs(4326, 26910, always_xy=True).transform(LON, LAT)
    mx, my = to_mercator.transform(LON, LAT)
    # 3857 expands metre distances at this latitude; include a 35 m buffer.
    r = (args.radius_m + 35) / np.cos(np.radians(LAT))
    query = (mx-r, my-r, 0, mx+r, my+r, 300)
    keys, meta = ept_keys(cache, query)
    print(f"Fetching {len(keys)} EPT nodes", flush=True)
    def read_key(key: str):
        path = cache / f"ept-data/{key}.laz"
        fetch(EPT + f"ept-data/{key}.laz", path)
        cloud = laspy.read(path)
        x, y = to_utm.transform(np.asarray(cloud.x), np.asarray(cloud.y))
        keep = ((x > origin[0] - args.radius_m - 35) &
                (x < origin[0] + args.radius_m + 35) &
                (y > origin[1] - args.radius_m - 35) &
                (y < origin[1] + args.radius_m + 35))
        return (x[keep], y[keep], np.asarray(cloud.z)[keep],
                np.asarray(cloud.classification)[keep])
    with ThreadPoolExecutor(max_workers=8) as pool:
        chunks = list(pool.map(read_key, keys))
    x, y, z, classification = (np.concatenate([chunk[i] for chunk in chunks])
                                for i in range(4))
    print(f"Local points: {len(x)}; classes: {dict(Counter(classification.tolist()))}", flush=True)
    # Check vertical agreement with the paired 3DEP ground raster before
    # interpreting any roof return as a height above ground.
    with rasterio.open(args.dem) as dem:
        ground_idx = np.flatnonzero(classification == 2)
        if len(ground_idx) < 100:
            raise ValueError("LPC contains too few classified ground returns")
        sample_idx = ground_idx[::max(1, len(ground_idx)//1000)]
        sampled = np.array([v[0] for v in dem.sample(zip(x[sample_idx], y[sample_idx]))])
        offset = float(np.median(z[sample_idx] - sampled))
        if abs(offset) > 1.0:
            raise ValueError(f"LPC and DEM vertical reference disagree by {offset:.2f} m")
        buildings = [measure(fp, x, y, z, classification, dem)
                     for fp in osm_footprints(args.osm, origin, args.radius_m)]
    accepted = sum(b["status"] == "accepted" for b in buildings)
    target = next((b for b in buildings if b["osm_way_id"] == "382886306"), None)
    manifest = {
        "schema_version": "1.0", "location": "Woodside Library, 3140 Woodside Road",
        "origin_utm_m": origin, "radius_m": args.radius_m,
        "source_type": "LPC", "has_measured_structure_height": accepted > 0,
        "target_osm_way_id": "382886306", "target_has_measured_height":
        bool(target and target["status"] == "accepted"),
        "height_method": "class-1 elevated returns within OSM footprint, robust median above 3DEP ground; conservative surface-consistency filter",
        "height_limitations": "Class 1 mixes roofs and vegetation. Accepted forms are approximate block volumes, not surveyed roof shapes; reject list records uncertain footprints.",
        "lidar_source": EPT + "ept.json", "lidar_srs": meta["srs"]["horizontal"],
        "lidar_vertical_alignment": "checked against 3DEP NAVD88 ground raster",
        "ground_median_difference_m": round(offset, 3),
        "ground_source": str(args.dem), "footprint_source": "OpenStreetMap contributors (ODbL)",
        "footprint_file_sha256": hashlib.sha256(args.osm.read_bytes()).hexdigest(),
        "ept_nodes": len(keys), "local_point_count": len(x),
        "accepted_buildings": accepted, "rejected_buildings": len(buildings) - accepted,
        "retrieval_utc": datetime.now(timezone.utc).isoformat(), "buildings": buildings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Accepted {accepted}/{len(buildings)} building footprints; saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
