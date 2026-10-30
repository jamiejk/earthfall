"""Small GeoBlock preparation prototype with two explicitly prepared places."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_landform_heights, qualify_structure_heights

DEMO_SPEC = ROOT / "examples/worms_head_full.json"
DEMO_BASE = ROOT / "outputs/B_lidar_full_2m"
DEMO_PACK = ROOT / "data/reference_images/worms_head/h3_reference_pack.json"
DEMO_HEIGHT_EVIDENCE = DEMO_BASE / "landform_height_evidence.json"
DEMO_LAT = 51.5643
DEMO_LON = -4.3241
WOODSIDE_SPEC = ROOT / "examples/woodside_library.json"
WOODSIDE_BASE = ROOT / "outputs/woodside_library_3dep"
WOODSIDE_BUILDINGS = WOODSIDE_BASE / "measured_buildings.json"
WOODSIDE_CLOSE_REFERENCES = ROOT / "data/woodside_library_close_reference_manifest.json"
WOODSIDE_STYLE_DIR = ROOT / "assets/style_references/woodside_clearline"
WOODSIDE_STYLE_MANIFEST = WOODSIDE_STYLE_DIR / "manifest.json"
WOODSIDE_LAT = 37.4291056
WOODSIDE_LON = -122.2580225
ROUTES = {
    "Wide turn and return": DEMO_BASE / "flyover_reference_v2_h3.mp4",
    "Wide turn, slower return": DEMO_BASE / "flyover_reference_v3_h3.mp4",
}
WOODSIDE_CLOSE_ROUTE = "Close measured-building orbit"


def qualify_scene_height(scene: dict) -> str:
    """Every blocking scene must have measured height for its target feature."""
    if scene.get("fixture") == "worms_head":
        path = Path(scene.get("geometry_evidence_path", ""))
        if not path.is_file():
            raise ValueError("Worm's Head has no landmark-height evidence manifest.")
        spec = json.loads(DEMO_SPEC.read_text())
        target = (spec["origin_easting_m"], spec["origin_northing_m"])
        return qualify_landform_heights(json.loads(path.read_text()), Path(scene["terrain_path"]), target)
    if scene.get("fixture") == "woodside_library":
        path = Path(scene.get("geometry_evidence_path", ""))
        if not path.is_file():
            raise ValueError("Woodside has no measured building-height evidence manifest.")
        return qualify_structure_heights(json.loads(path.read_text()), "382886306", 200)
    raise ValueError("No qualified height model exists for this scene.")


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a, b = math.radians(lat1), math.radians(lat2)
    dlat, dlon = b - a, math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
    return 12742.0176 * math.asin(math.sqrt(h))


def resolve_place(query: str, mode: str, latitude: float, longitude: float, radius_km: float) -> tuple[dict, str]:
    query = query.strip()
    if not query:
        raise ValueError("Enter a place or landmark name.")
    if not 0.1 <= radius_km <= 20:
        raise ValueError("Coverage radius must be between 0.1 and 20 km.")
    if mode == "Exact coordinates":
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Latitude or longitude is out of range.")
        lat, lon = latitude, longitude
        source = "user-supplied coordinates"
        display_name = query
    elif mode == "Search place name":
        if "worm" in query.casefold() and "head" in query.casefold():
            lat, lon = DEMO_LAT, DEMO_LON
            source = "pinned Worm's Head landmark coordinate (reviewed demo)"
            display_name = "Worm's Head, Rhossili, Wales"
        elif "woodside" in query.casefold() and "library" in query.casefold():
            lat, lon = WOODSIDE_LAT, WOODSIDE_LON
            source = "pinned Woodside Library address point (OpenStreetMap node 368169993)"
            display_name = "Woodside Library, 3140 Woodside Road, Woodside, CA"
        else:
            raise ValueError(
                "Name lookup is pinned to Worm's Head and Woodside Library in this prototype. "
                "For another place, choose Exact coordinates; a configurable geocoder adapter is future work."
            )
    else:
        raise ValueError(f"Unknown location mode: {mode}")
    place = {
        "query": query, "display_name": display_name, "latitude": lat,
        "longitude": lon, "requested_radius_km": radius_km, "source": source,
        "fixture_only": True,
    }
    report = (
        f"PLACE: {display_name}\n"
        f"COORDINATES: {lat:.6f}, {lon:.6f}\n"
        f"REQUESTED RADIUS: {radius_km:g} km\nSOURCE: {source}\n"
        "Check the resolved location and full terrain extent before generation."
    )
    return place, report


def load_demo_terrain(place: dict) -> tuple[dict, str]:
    if not place.get("fixture_only"):
        raise ValueError(
            "This node loads prepared examples only. A user-selected landmark needs "
            "a validated terrain and geometry provider before scene building."
        )
    worm_distance = _distance_km(place["latitude"], place["longitude"], DEMO_LAT, DEMO_LON)
    wood_distance = _distance_km(place["latitude"], place["longitude"], WOODSIDE_LAT, WOODSIDE_LON)
    if worm_distance <= 1.0:
        fixture, base, spec_path, preview, radius = (
            "worms_head", DEMO_BASE, DEMO_SPEC, DEMO_BASE / "sea_preview_lookout.png", 2.5)
    elif wood_distance <= 0.03:
        fixture, base, spec_path, preview, radius = (
            "woodside_library", WOODSIDE_BASE, WOODSIDE_SPEC,
            WOODSIDE_BASE / "blocking_measured_storyboard/frame_0096.png", 1.2)
    else:
        raise ValueError(
            "This prototype has terrain only for Worm's Head and Woodside Library. "
            "The chosen place is outside the demo area; no unrelated terrain will be substituted."
        )
    if place["requested_radius_km"] > radius:
        raise ValueError(f"The prepared {fixture} terrain covers {radius:g} km radius or less.")
    spec = json.loads(spec_path.read_text())
    metadata = json.loads((base / "metadata.json").read_text())
    route_files = ROUTES.values() if fixture == "worms_head" else (
        base / "woodside_library_measured_blocking_flyover.mp4",)
    for path in (preview, base / "terrain.glb", *route_files):
        if not path.is_file():
            raise FileNotFoundError(f"Demo asset missing: {path}")
    scene = {
        "place": place,
        "project_id": spec["project_id"],
        "name": spec["name"],
        "actual_radius_km": spec["radius_m"] / 1000,
        "projected_crs": spec["projected_crs"],
        "vertical_crs": spec["vertical_crs"],
        "preview_path": str(preview),
        "terrain_path": str(base / "terrain.glb"),
        "height_alignment": metadata.get("height_alignment", "unknown"),
        "source_dataset": metadata["source"]["dataset"],
        "height_model_kind": "landform" if fixture == "worms_head" else "building",
        "geometry_evidence_path": str(DEMO_HEIGHT_EVIDENCE if fixture == "worms_head" else WOODSIDE_BUILDINGS),
        "fixture": fixture,
        "demo_only": True,
    }
    height_report = qualify_scene_height(scene)
    report = (
        f"TERRAIN: {scene['name']}\n"
        f"AREA: {2 * radius:g} × {2 * radius:g} km prepared square\n"
        f"SOURCE: {scene['source_dataset']}\n"
        f"HEIGHT ALIGNMENT: {scene['height_alignment']}\n"
        f"3D HEIGHT EVIDENCE: {height_report}\n" +
        ("Sea height is an illustrative visual setting, not a verified tide level."
         if fixture == "worms_head" else
         "Building blocks use LiDAR-derived heights and OSM footprints; uncertain buildings are omitted.")
    )
    return scene, report


def plan_shot(scene: dict, route: str, duration_seconds: float) -> tuple[dict, str]:
    if not scene.get("demo_only"):
        raise ValueError("No prepared terrain scene was selected.")
    qualify_scene_height(scene)
    if not 5 <= duration_seconds <= 15:
        raise ValueError("MiniMax H3 prototype duration must be 5–15 seconds.")
    # H3 output lengths follow 17*n+5 frames at 24 fps. Do not exceed 15 s.
    n = max(7, min(20, round((duration_seconds * 24 - 5) / 17)))
    frames = 17 * n + 5
    fixture = scene["fixture"]
    if route == "Auto route":
        route = "Wide turn, slower return" if fixture == "worms_head" else WOODSIDE_CLOSE_ROUTE
    if fixture == "worms_head" and route in ROUTES:
        source_video = ROUTES[route]
        start_view, target, path = ("Rhossili headland cliff-edge lookout", "Worm's Head and Outer Head",
                                    "fly out over the sea, pass Outer Head, turn broadly, return east")
    elif fixture == "woodside_library" and route == WOODSIDE_CLOSE_ROUTE:
        source_video = WOODSIDE_BASE / "woodside_library_measured_blocking_flyover.mp4"
        if not source_video.is_file():
            raise FileNotFoundError(f"Measured building control video missing: {source_video}")
        start_view, target, path = ("southwest of Woodside Library at building scale",
                                    "LiDAR-height Woodside Library building",
                                    "close oblique arc around measured building volumes")
    else:
        raise ValueError(f"Camera route {route!r} is not available for {fixture}.")
    shot = {
        "route": route,
        "source_video": str(source_video),
        "source_frames": 360 if fixture == "worms_head" else 192,
        "start_view": start_view,
        "target": target,
        "frame_count": frames,
        "fps": 24,
        "actual_duration_seconds": round(frames / 24, 3),
        "requested_duration_seconds": duration_seconds,
    }
    report = (
        f"SHOT: {route}\nSTART: {shot['start_view']}\n"
        f"PATH: {path}\n"
        f"TIMING: {frames} frames at 24 fps ({frames/24:.2f} s)\n"
        "The camera route is pinned for this prototype; editable waypoints are the next step."
    )
    return shot, report


def build_pack(scene: dict, shot: dict, style: str, surf: str, comfy_input_dir: Path) -> tuple[dict, str]:
    if not scene.get("demo_only"):
        raise ValueError("No prepared reference pack exists for this scene.")
    qualify_scene_height(scene)
    fixture = scene["fixture"]
    expected_source = (ROUTES.get(shot.get("route")) if fixture == "worms_head" else
                       WOODSIDE_BASE / "woodside_library_measured_blocking_flyover.mp4"
                       if shot.get("route") == WOODSIDE_CLOSE_ROUTE else None)
    if expected_source is None or Path(shot.get("source_video", "")) != expected_source:
        raise ValueError("Shot control is not a qualified 3D blocking reference for this scene.")
    if fixture == "worms_head":
        pack = json.loads(DEMO_PACK.read_text())
        chosen = ("commons_115697390.jpg", "commons_133768741.jpg")
        items = [next(item for item in pack["image_references"]
                      if Path(item["path"]).name == name) for name in chosen]
        attribution_path = str(DEMO_PACK.parent / "ATTRIBUTION.md")
    else:
        items = json.loads(WOODSIDE_CLOSE_REFERENCES.read_text())["images"]
        attribution_path = str(WOODSIDE_CLOSE_REFERENCES)
    if style == "Clear-line comic" and fixture != "woodside_library":
        raise ValueError("The clear-line style references are prepared for Woodside only.")
    photos = []
    for item in items:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Reference image checksum changed: {path}")
        photos.append(str(path))
    source = Path(shot["source_video"])
    frames = shot["frame_count"]
    key = hashlib.sha256(f"{source}:{source.stat().st_mtime_ns}:{frames}".encode()).hexdigest()[:12]
    stage_dir = comfy_input_dir / "geoblock_prototype"
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = stage_dir / f"{fixture}_{key}_{frames}f.mp4"
    if not destination.exists():
        temp = destination.with_name(destination.stem + ".part.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-vf", f"setpts=(PTS-STARTPTS)*{frames}/{shot['source_frames']},fps=24,scale=864:480:flags=lanczos",
                "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", str(temp),
            ], check=True, capture_output=True, text=True)
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
    style_text = ({
        "Photographic": "Photographic daylight coastal landscape; preserve the real landform silhouette.",
        "Clay study": "Simple matte clay maquette, untextured terrain, restrained colours; preserve the landform silhouette.",
    } if fixture == "worms_head" else {
        "Photographic": "Photographic daylight wooded California landscape; preserve the road and hill layout.",
        "Clay study": "Simple matte clay maquette with restrained colours; preserve the measured terrain shape.",
        "Clear-line comic": (
            "Original Tintin-era ligne claire / clear-line comic illustration: crisp even ink contours, flat "
            "daylight colours, legible tree canopies and simple roof volumes. No photographic "
            "texture, clay shading, mesh facets or visible 3D-model appearance."
        ),
    })[style]
    surf_text = {
        "Not applicable": "",
        "Breaking waves": "Distinct waves move toward shore, break against the rocks and leave receding white foam.",
        "Calm sea": "The sea is calm with gentle motion and minimal foam.",
    }[surf]
    if fixture == "worms_head" and not surf_text:
        raise ValueError("Choose a sea-motion setting for the Worm's Head shot.")
    worm_prompt = (
        "<Picture 1> is the opening cliff-edge view. <Picture 2> shows the connected Worm's Head "
        "promontory and tidal causeway. <Video 1> defines the continuous camera route and timing. "
        "Start at the Rhossili headland, fly west over the sea past Outer Head, make a broad smooth "
        "180-degree turn, then return toward the headland. Keep the causeway connected and the coast "
        "recognizable. One uninterrupted shot without cuts, jumps or morphing. "
        + style_text + " " + surf_text + " No people, no text."
    )
    close_prompt = (
        "<Picture 1> fixes the opening close oblique camera view of Woodside Library. "
        "<Picture 2> is the 220-metre address-centred NAIP crop for building placement, roads "
        "and trees, not an overhead camera angle. <Video 1> fixes the continuous close camera "
        "movement and relative building volumes measured from lidar. Preserve the scale, depth "
        "and parallax of these buildings throughout the shot. "
        + ("<Picture 3> and <Picture 4> are original clear-line style references for linework "
           "and colour. " if style == "Clear-line comic" else "") +
        "Draw finished roofs, facades, trees and streets in that same arrangement. "
        "One continuous shot with no wide panorama, overhead map view, cuts, morphing or text. "
        + style_text
    )
    prompt = worm_prompt if fixture == "worms_head" else close_prompt
    style_paths = load_style_refs(scene)[0] if style == "Clear-line comic" else []
    result = {
        "video_path": str(destination), "video_relative_input": str(destination.relative_to(comfy_input_dir)),
        "photo_paths": photos, "style_image_paths": style_paths,
        "prompt": prompt, "frame_count": frames,
        "attribution_path": attribution_path,
    }
    report = (
        f"CONTROL VIDEO: {destination.name} ({frames} frames)\n"
        f"IMAGES: two pinned {'Wikimedia Commons views' if fixture == 'worms_head' else 'opening guide frame and close NAIP crop'}, checksum-checked\n"
        f"STYLE: {style}" + (f"; SURF: {surf}\n" if fixture == "worms_head" else "\n") +
        (f"STYLE IMAGES: {len(style_paths)} original clear-line references\n" if style_paths else "") +
        f"ATTRIBUTION: {result['attribution_path']}\n"
        "Ready to wire into MiniMax H3 Reference to Video. No H3 inference was started."
    )
    return result, report


def load_style_refs(scene: dict) -> tuple[list[str], str]:
    if scene.get("fixture") != "woodside_library":
        raise ValueError("Clear-line style references are prepared for Woodside Library only.")
    manifest = json.loads(WOODSIDE_STYLE_MANIFEST.read_text())
    paths = []
    for item in manifest["images"]:
        path = WOODSIDE_STYLE_DIR / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Style reference checksum changed: {path}")
        paths.append(str(path))
    if len(paths) != 2:
        raise ValueError("Woodside clear-line pack must contain two style images.")
    return paths, "Two original clear-line style frames; visual style only, not geographic evidence."
