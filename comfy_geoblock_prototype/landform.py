"""Place → landform scene → camera preset → blocking video, for any landmark (3DEP in the US, Copernicus elsewhere).

Kept free of ComfyUI imports so the same steps run in tests and in the nodes.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.blender import render_route
from geoblock_core.camera import Terrain, plan_route, route_preview, route_report
from geoblock_core.place_photos import fetch_place_photos, photo_report
from geoblock_core.scene import build_landform_scene, scene_report


def data_dir() -> Path:
    return Path(os.environ.get("GEOBLOCK_DATA_DIR") or ROOT / "outputs").expanduser()


def place_from_coordinates(name: str, latitude: float, longitude: float,
                           radius_km: float) -> tuple[dict, str]:
    name = name.strip()
    if not name:
        raise ValueError("Name the landmark at these coordinates.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Latitude or longitude is out of range.")
    place = {"query": name, "display_name": name, "latitude": latitude, "longitude": longitude,
             "requested_radius_km": radius_km, "source": "user-supplied coordinates",
             "source_id": None, "feature_type": None, "extent": None}
    return place, (f"PLACE: {name}\nCOORDINATES: {latitude:.6f}, {longitude:.6f}\n"
                   f"REQUESTED RADIUS: {radius_km:g} km\nSOURCE: user-supplied coordinates")


def build_scene(place: dict, radius_km: float = 0, landing_lidar: str = "none") -> tuple[dict, str]:
    if place.get("fixture_only"):
        raise ValueError("Use Search/Choose Landmark or Place From Coordinates; fixture places "
                         "belong to the example Prepared Scene node.")
    radius_m = 1000 * (radius_km or place["requested_radius_km"])
    if not 0.5 <= radius_m / 1000 <= 30:
        raise ValueError("Landform scenes must be 0.5–30 km in radius.")
    use_lidar = landing_lidar.strip().lower() not in ("", "none", "off")
    brink = place.get("lidar_landing") if use_lidar else None
    curated = place
    if brink:  # an overlook's brink, found on the 1 m lidar; the 10 m DEM rounds it off
        place = {**place, "latitude": brink["latitude"], "longitude": brink["longitude"],
                 "heading_deg": brink["heading_deg"]}
    source = place["source"] + (f" {place['source_id']}" if place.get("source_id") else "")
    scene = build_landform_scene(
        name=place["display_name"], latitude=place["latitude"], longitude=place["longitude"],
        radius_m=radius_m, output_root=data_dir() / "scenes", raw_cache=ROOT / "data/cache/raw",
        location_source=source)
    scene.update(place_extras(place))
    report = scene_report(scene)
    if use_lidar:
        scene["landing_scene"], note = landing_patch(scene, landing_lidar.strip())
        if brink and scene["landing_scene"] is None:  # no patch after all: back from the rounded-off brink
            scene, report = build_scene({**curated, "lidar_landing": None}, radius_km, "none")
            return scene, report + "\n" + note
        report += "\n" + note + (f"\nLIDAR LANDING: {brink['reason']}" if brink else "")
    if scene.get("view_heading_deg", -1) >= 0:
        report += f"\nVIEW HEADING: {scene['view_heading_deg']:.0f}° (from the place catalog)"
    if scene.get("look"):
        report += f"\nLOOK: {scene['look']['all'].format(name=scene['name'])}"
    return scene, report


def landing_patch(scene: dict, choice: str) -> tuple[dict | None, str]:
    """A 1 m lidar terrain patch for the landing: ``auto`` picks the newest 3DEP 1 m project under the
    landmark, or name a project. US only; elsewhere, or with no project, the shot keeps the scene's terrain."""
    if scene.get("terrain_family") != "usgs3dep":
        return None, "LANDING LIDAR: none (3DEP 1 m covers the US only)"
    from geoblock_core.lidar import dem1m_projects_at
    from geoblock_core.scene import build_landing_scene
    place = scene["place"]
    projects = dem1m_projects_at(place["latitude"], place["longitude"], ROOT / "data/cache/raw")
    if choice.lower() != "auto":
        projects = [p for p in projects if p["project"] == choice]
    if not projects:
        return None, f"LANDING LIDAR: none ({'no 3DEP 1 m project under the landmark' if choice.lower() == 'auto' else choice + ' does not cover the landmark'})"
    errors = []
    for project in projects:  # newest first; fall back if one has a hole at the landmark
        try:
            landing = build_landing_scene(scene, project=project["project"], zone=project["zone"],
                                          output_root=data_dir() / "scenes", raw_cache=ROOT / "data/cache/raw")
        except ValueError as exc:
            errors.append(f"{project['project']}: {exc}")
            continue
        others = ", ".join(p["project"] for p in projects if p is not project)
        return landing, (f"LANDING LIDAR: {project['project']} 1 m bare earth, {2 * landing['radius_m']:g} m square "
                         f"at the landmark" + (f" (also available: {others})" if others else ""))
    return None, "LANDING LIDAR: none usable (" + "; ".join(errors) + ")"


def place_extras(place: dict) -> dict:
    """What a catalog place adds to its scene: the view heading, its palette, kind and measured look. A place
    from search or coordinates gets its look measured from ESA WorldCover (skipped if that is unreachable)."""
    extras = {"view_heading_deg": place.get("heading_deg", -1), "palette": place.get("palette", "default"),
              "kind": place.get("kind") or "", "catalog_id": place.get("catalog_id")}
    look = place.get("look")
    if not look:
        try:
            from geoblock_core.landcover import cached_summary, look_from_landcover
            look = look_from_landcover(cached_summary(place["latitude"], place["longitude"],
                                                      ROOT / "data/cache/raw"), extras["kind"])
        except Exception:  # landcover only sharpens prompts; never block a scene on it
            look = None
    extras["look"] = look
    return extras


def place_from_catalog(place_id: str, radius_km: float = 0) -> tuple[dict, str]:
    from geoblock_core.places import find_place, place_for_pipeline, place_report
    entry = find_place(place_id.split(" ", 1)[0])
    return place_for_pipeline(entry, radius_km or None), place_report(entry)


def plan_shot(scene: dict, preset: str, duration_s: float, heading_deg: float, scale_m: float,
              lens_mm: float, clearance_m: float, orbit_degrees: float,
              dive_tempo: str = "standard") -> tuple[dict, str, str]:
    terrain = Terrain.from_scene(scene)
    if heading_deg < 0 and scene.get("view_heading_deg", -1) >= 0:
        heading_deg = scene["view_heading_deg"]  # the catalog place's view, e.g. toward the Matterhorn
    route = plan_route(scene, preset, duration_s, heading_deg=heading_deg, scale_m=scale_m or None,
                       lens_mm=lens_mm, clearance_m=clearance_m, orbit_degrees=orbit_degrees,
                       terrain=terrain, dive_tempo=dive_tempo)
    preview = data_dir() / "routes" / f"{scene['project_id']}-{route['route_sha256'][:12]}.png"
    route_preview(route, terrain, preview)
    return route, str(preview), route_report(route)


def render_shot(scene: dict, route: dict, output_root: Path, engine: str, width: int, height: int,
                samples: int, depth: bool = False, palette: str = "default", weather: str = "clear", imagery: bool = False,
                trees: str = "off", buildings: bool = False, progress=None) -> tuple[dict, str]:
    from geoblock_core.looks import scene_palette
    palette = scene_palette(palette, scene)
    if route.get("kind") == "earth_zoom":
        from geoblock_core.zoom import render_earth_zoom
        manifest = render_earth_zoom(scene, route, output_root, raw_cache=ROOT / "data/cache/raw", width=width,
                                     height=height, samples=samples, palette=palette, weather=weather,
                                     imagery=True, trees=trees, buildings=buildings, progress=progress)
        return manifest, (f"EARTH ZOOM: {Path(manifest['video']).name} ({manifest['frames']} frames, "
                          f"{manifest['resolution'][0]}×{manifest['resolution'][1]})\nATTRIBUTION: "
                          f"{manifest['attribution']}")
    imagery_path = None
    if imagery:  # NASA GIBS HLS 30 m, warped onto the terrain grid and cached beside the scene
        from geoblock_core.imagery import fetch_terrain_imagery
        fetch_terrain_imagery(scene, Path(scene["terrain_dir"]) / "imagery")
        imagery_path = Path(scene["terrain_dir"]) / "imagery" / "imagery.json"
    tree_path, tree_line = None, ""
    if trees != "off":  # measured tree tops near the camera path: 3DEP lidar where surveyed, else Meta canopy
        from geoblock_core.trees import fetch_trees, tree_report
        tree_manifest = fetch_trees(scene, route, Path(scene["terrain_dir"]) / "trees",
                                    cache_dir=ROOT / "data/cache/raw", source=trees)
        tree_path, tree_line = Path(tree_manifest["path"]), "\n" + tree_report(tree_manifest)
    manifest = render_route(scene, route, output_root, engine=engine, width=width, height=height,
                            samples=samples, depth=depth, palette=palette, weather=weather, imagery=imagery_path,
                            trees=tree_path, progress=progress)
    report = (f"CONTROL VIDEO: {Path(manifest['video']).name} ({manifest['frames']} frames, "
              f"{manifest['resolution'][0]}×{manifest['resolution'][1]}, {manifest['engine']}, "
              f"Blender {manifest['blender_version']})\n"
              f"OPENING FRAME: {Path(manifest['first_frame']).name}\n"
              f"DEPTH VIDEO: {Path(manifest['depth_video']).name if manifest.get('depth_video') else 'not rendered'}\n"
              f"ATTRIBUTION: {manifest['attribution']}{tree_line}")
    return manifest, report


def place_photos(place: dict, count: int, radius_km: float, heading_deg: float,
                 exclude_page_ids: str) -> tuple[list[str], str]:
    exclude = tuple(int(value) for value in exclude_page_ids.replace(",", " ").split())
    folder = data_dir() / "place_photos" / f"{place['latitude']:.4f}_{place['longitude']:.4f}"
    manifest = fetch_place_photos(place["display_name"], place["latitude"], place["longitude"], count, folder,
                                  radius_km=radius_km, heading_deg=heading_deg, exclude=exclude)
    return [item["local_path"] for item in manifest["selected"]], photo_report(manifest)
