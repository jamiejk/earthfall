"""Measure every catalog place's coverage and write it back into src/geoblock_core/data/places.json.

For each landing point: which terrain family covers it and the ground height and relief there (a 1.5 km
window of that DEM), which USGS 3DEP lidar surveys contain it (EPT index), national lidar outside the US,
the imagery sources, and ESA WorldCover land cover near the person and across the view, turned into
prompt words. Only small windows are read; nothing is kept beyond the cache directory.

    uv run python scripts/build_place_catalog.py [--only ID ...] [--cache data/cache/raw]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pyproj import Transformer
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoblock_core.landcover import cached_summary, look_from_landcover, read_classes  # noqa: E402
from geoblock_core.lidar import dem1m_projects_at, load_index, surveys_at  # noqa: E402
from geoblock_core.models import GeoSceneSpec  # noqa: E402
from geoblock_core.places import CATALOG, NATIONAL_LIDAR, view_heading  # noqa: E402
from geoblock_core.providers import USGS1mDEMProvider  # noqa: E402
from geoblock_core.sightline import _sample, clearest_nearby, near_view_deg, steady_clearance_deg  # noqa: E402
from geoblock_core.scene import default_provider, nad83_utm_crs, terrain_family, wgs84_utm_crs  # noqa: E402

CONUS_OR_US = {"US"}


def national_lidar(place: dict) -> list[str] | None:
    country = place.get("country", "")
    if country == "GB":
        region = place.get("region", "")
        country = "GB-WLS" if region == "Wales" else "GB-SCT" if region == "Scotland" else "GB-ENG"
    entry = NATIONAL_LIDAR.get(country)
    return list(entry) if entry else None


def dem_window(place_id: str, lat: float, lon: float, cache: Path, family: str, window_m: float):
    crs = nad83_utm_crs(lon, lat) if family == "usgs3dep" else wgs84_utm_crs(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    spec = GeoSceneSpec(project_id=place_id, name=place_id, longitude=lon, latitude=lat, projected_crs=crs,
                        origin_easting_m=x, origin_northing_m=y, radius_m=window_m, location_source="catalog")
    return default_provider(family, 10).acquire(spec, cache)


# How far the named view's top must stand above the foreground at eye level. 2-3 degrees left only the
# Matterhorn's tip over Gornergrat's summit hump in the first blocking render.
WANTED_CLEARANCE_DEG = 4.0


def check_view(place: dict, coverage: dict, cache: Path, family: str) -> None:
    """Move the landing (up to 200 m) if nearer ground hides the named view from eye level."""
    target = place.get("view_toward")
    if not target:
        return
    lat, lon = (coverage["landing"]["latitude"], coverage["landing"]["longitude"]) if "landing" in coverage \
        else (place["latitude"], place["longitude"])
    kx = 111_320 * np.cos(np.radians(lat))
    reach = float(np.hypot((target["longitude"] - lon) * kx, (target["latitude"] - lat) * 110_574)) + 900
    raster, _ = dem_window(place["id"], lat, lon, cache, family, reach)
    with rasterio.open(raster) as dem:
        heights = dem.read(1, masked=True).astype("float64").filled(np.nan)
        transform = dem.transform
    eye, goal = (lon, lat), (target["longitude"], target["latitude"])
    structure = target.get("structure_m", 0.0)
    before = steady_clearance_deg(heights, transform, eye, goal, structure_m=structure)
    coverage["view_clearance_deg"] = before
    if before >= WANTED_CLEARANCE_DEG or not place.get("view_snap", True):
        return
    (new_lon, new_lat), after, moved = clearest_nearby(heights, transform, eye, goal, structure_m=structure,
                                                       wanted_deg=WANTED_CLEARANCE_DEG)
    if moved == 0 or (after < WANTED_CLEARANCE_DEG and after - before < 2.0):
        # Not worth leaving the curated spot for: say so and keep it.
        coverage["warnings"].append(f"{target['name']} clears the foreground by only {before:.1f}° at eye level")
        return
    curated = float(np.hypot((new_lon - place["longitude"]) * kx, (new_lat - place["latitude"]) * 110_574))
    coverage["landing"] = {"latitude": round(new_lat, 6), "longitude": round(new_lon, 6),
                           "moved_m": round(curated, 1),
                           "reason": f"nearer ground hid {target['name']} at eye level ({before:.1f}° clear); "
                                     f"moved {moved:.0f} m to where it clears by {after:.1f}°"}
    coverage["view_clearance_deg"] = after
    if after < WANTED_CLEARANCE_DEG:
        coverage["warnings"].append(f"{target['name']} clears the foreground by only {after:.1f}° at eye level")
    coverage.update(measure_terrain({**place, "latitude": new_lat, "longitude": new_lon, "snap_m": 0}, cache, family))


# An overlook's person should stand where the drop is in view. On the 1 m lidar, the first Glacier Point
# landing stood 25 m back from the brink and saw only 8.6 degrees down, so the valley floor was hidden.
BRINK_DEG = -20.0
WANTED_BRINK_DEG = -25.0
BRINK_REACH_M = 40


def check_brink(place: dict, coverage: dict, cache: Path) -> None:
    """US places: list the 3DEP 1 m lidar DEM projects at the landing, measure on the newest usable one how
    far below level the person can see toward the view, and, for an overlook whose edge hides the drop, find
    the brink (up to 40 m toward the view, never down a face). The brink is only used with the 1 m landing
    patch: the 10 m DEM rounds the edge off, so on the default mesh the person would stand on the slope."""
    lat, lon = (coverage["landing"]["latitude"], coverage["landing"]["longitude"]) if "landing" in coverage \
        else (place["latitude"], place["longitude"])
    projects = dem1m_projects_at(lat, lon, cache)
    coverage["dem1m"] = [{"project": p["project"], "zone": p["zone"], "year": p["year"]} for p in projects]
    crs = nad83_utm_crs(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    spec = GeoSceneSpec(project_id=place["id"], name=place["id"], longitude=lon, latitude=lat, projected_crs=crs,
                        origin_easting_m=x, origin_northing_m=y, radius_m=200, location_source="catalog")
    for project in projects:
        try:
            raster, _ = USGS1mDEMProvider(project["project"], project["zone"]).acquire(spec, cache)
            break
        except ValueError:
            continue
    else:
        return
    from rasterio.vrt import WarpedVRT
    with rasterio.open(raster) as src, WarpedVRT(src, crs="EPSG:4326") as dem:
        heights = dem.read(1, masked=True).astype("float64").filled(np.nan)
        transform = dem.transform
    heading = view_heading({**place, "coverage": coverage})
    if heading < 0:
        return  # the rim reveal picks its own heading at render time
    before = near_view_deg(heights, transform, (lon, lat), heading)
    coverage["near_view_deg"] = before
    if before <= BRINK_DEG or coverage["drop_below_m"] < 150:
        return
    kx, ky = 111_320 * np.cos(np.radians(lat)), 110_574.0
    ground = float(_sample(heights, transform, np.array([lon]), np.array([lat]))[0])
    for step in range(1, BRINK_REACH_M + 1):
        point = (lon + step * np.sin(np.radians(heading)) / kx, lat + step * np.cos(np.radians(heading)) / ky)
        if float(_sample(heights, transform, np.array([point[0]]), np.array([point[1]]))[0]) < ground - 3:
            break  # going down the face, not along the top
        after = near_view_deg(heights, transform, point, heading)
        if after <= WANTED_BRINK_DEG:
            coverage["lidar_landing"] = {
                "latitude": round(point[1], 6), "longitude": round(point[0], 6), "moved_m": step,
                "near_view_deg": after, "project": project["project"],
                "reason": f"on 1 m lidar the edge hides the drop ({abs(before):.0f}° down); the brink {step} m "
                          f"toward the view sees {abs(after):.0f}° down"}
            return
    coverage["warnings"].append(f"the person sees only {abs(before):.0f}° below level toward the view, and no "
                                f"brink within {BRINK_REACH_M} m; nearer ground hides the drop")


def measure_terrain(place: dict, cache: Path, family: str, window_m: float = 1500) -> dict:
    lat, lon = place["latitude"], place["longitude"]
    raster, record = dem_window(place["id"], lat, lon, cache, family, window_m)
    with rasterio.open(raster) as dem:
        heights = dem.read(1, masked=True).astype("float64").filled(np.nan)
        rows, cols = np.indices(heights.shape)
        xs, ys = rasterio.transform.xy(dem.transform, rows.ravel(), cols.ravel())
        to_dem = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
        px, py = to_dem.transform(lon, lat)
        here = float(next(dem.sample([(px, py)]))[0])
    xs, ys = np.asarray(xs).reshape(heights.shape), np.asarray(ys).reshape(heights.shape)
    kx = 111_320 * np.cos(np.radians(lat))
    result = {}
    snap = place.get("snap_m", 0)
    if snap:
        # Stand on top: a 30 m surface model smears a cliff edge, so a point on the brink can read as the face.
        distance = np.hypot((xs - px) * kx, (ys - py) * 111_320)
        zone = (distance <= snap) & np.isfinite(heights)
        top = heights[zone].max()
        candidates = zone & (heights >= top - 3)
        index = np.unravel_index(np.where(candidates, distance, np.inf).argmin(), heights.shape)
        to_wgs = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True)
        new_lon, new_lat = to_wgs.transform(xs[index], ys[index])
        result["landing"] = {"latitude": round(new_lat, 6), "longitude": round(new_lon, 6),
                             "moved_m": round(float(distance[index]), 1),
                             "reason": f"highest ground within {snap:g} m of the curated point"}
        px, py, here = xs[index], ys[index], float(heights[index])
    distance = np.hypot((xs - px) * kx, (ys - py) * 111_320)
    near = heights[(distance <= 1000) & np.isfinite(heights)]
    result.update({"elevation_m": round(here, 1), "relief_m": round(float(near.max() - near.min()), 1),
                   "drop_below_m": round(float(here - near.min()), 1),
                   "rise_above_m": round(float(near.max() - here), 1), "terrain_dataset": record.dataset})
    return result


CITY_MIN_BUILDINGS = 2000  # within the building area: a town or city, so the look and the blocking get buildings


def measure_city(place: dict, coverage: dict, cache: Path, family: str) -> None:
    """City places: count Overture buildings around the landing and how tall they are (the same area and data
    ``--buildings`` stands in the blocking), for the prompts' city words."""
    from geoblock_core.buildings import city_summary, fetch_buildings
    lat, lon = (coverage["landing"]["latitude"], coverage["landing"]["longitude"]) if "landing" in coverage \
        else (place["latitude"], place["longitude"])
    crs = nad83_utm_crs(lon, lat) if family == "usgs3dep" else wgs84_utm_crs(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    manifest = fetch_buildings({"projected_crs": crs, "origin_projected_m": [x, y],
                                "radius_m": 1000 * place.get("radius_km", 12)},
                               cache / "catalog-buildings" / place["id"], cache_dir=cache)
    if manifest["count"] >= CITY_MIN_BUILDINGS:
        coverage["city"] = {**city_summary(manifest), "measured_height_share": round(
            manifest["height_source"]["height"] / manifest["count"], 3), "release": manifest["release"],
            "source": "Overture Maps buildings", "license": manifest["license"]}


def measure(place: dict, cache: Path, ept_index: list[dict]) -> dict:
    lat, lon = place["latitude"], place["longitude"]
    family = terrain_family(lat, lon, 1000 * place.get("radius_km", 12))
    coverage: dict = {"terrain": family}
    coverage.update(measure_terrain(place, cache, family))
    coverage["warnings"] = []
    check_view(place, coverage, cache, family)
    if family == "usgs3dep":
        check_brink(place, coverage, cache)
    if "landing" in coverage:
        lat, lon = coverage["landing"]["latitude"], coverage["landing"]["longitude"]
    coverage["lidar"] = surveys_at(lat, lon, cache, index=ept_index) if place.get("country") in CONUS_OR_US else []
    if not coverage["lidar"]:
        lidar = national_lidar(place)
        if lidar:
            coverage["national_lidar"] = lidar
    coverage["imagery"] = ["hls"] + (["naip"] if family == "usgs3dep" else [])
    summary = cached_summary(lat, lon, cache)
    coverage["landcover"] = {key: summary[key] for key in ("near", "view", "near_m", "view_m", "source", "license")}
    coverage["look"] = look_from_landcover(summary, place.get("kind", ""))
    if "city" in place.get("kind", ""):
        measure_city(place, coverage, cache, family)
    centre = read_classes(lat, lon, 30, pixels=6)
    warnings = coverage["warnings"]
    if np.mean(centre == 80) > 0.5:
        warnings.append("the landing point reads as open water in WorldCover; move it onto land")
    if coverage["relief_m"] < 40:
        warnings.append(f"only {coverage['relief_m']:.0f} m of relief within 1 km; the blocking will look flat")
    coverage["warnings"] = warnings
    coverage["view_heading_deg"] = view_heading({**place, "coverage": coverage})
    return coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], help="Measure only this place id (repeatable)")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/raw")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text())
    index = load_index(args.cache)
    for place in catalog["places"]:
        if args.only and place["id"] not in args.only:
            continue
        try:
            place["coverage"] = measure(place, args.cache, index)
        except Exception as exc:  # report and keep going: one bad tile should not stop the catalog
            print(f"{place['id']}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        c = place["coverage"]
        print(f"{place['id']:<32} {c['terrain']:<10} ground {c['elevation_m']:7.0f} m  relief {c['relief_m']:5.0f} m  "
              f"drop {c['drop_below_m']:5.0f} m  lidar {len(c['lidar'])}/{len(c.get('dem1m', []))}  "
              f"view {c.get('view_clearance_deg', '-')}  down {c.get('near_view_deg', '-')}  "
              f"{('moved ' + str(c['landing']['moved_m']) + ' m') if 'landing' in c else ''}  "
              f"{('brink ' + str(c['lidar_landing']['moved_m']) + ' m') if 'lidar_landing' in c else ''}  "
              f"{'; '.join(c['warnings'])}", flush=True)
    args.catalog.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
