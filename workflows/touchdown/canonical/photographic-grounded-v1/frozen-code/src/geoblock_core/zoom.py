"""Earth Zoom: plan and render a descent from orbit, through the clouds, to the landmark.

Modelled on the commercial Earth Zoom effects: the curved horizon from low orbit, a dive through
cloud, a forward swoop down along the shot's heading, then eye level at the landmark for the second
half of the shot (where a subject can stand), looking out over the measured view.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .camera import FPS, PRESETS, h3_frame_count, plan_route
from .ride import ARRIVAL_EASE_FRAMES, RIDE_FLOOR_FRAMES, SMOOTH_TEMPOS, ride_floor
from .weather import WEATHER

PRESET = "Earth zoom"
DESCRIPTION = PRESETS[PRESET]
ORBIT_ALTITUDE_M = 420_000.0   # about the ISS: the curved horizon, not the whole globe
ORBIT_DISTANCE_M = 650_000.0   # start this far back along the heading, looking toward the landmark
REGIONAL_RADIUS_M = 100_000.0
CLOUD_BASE_M, CLOUD_TOP_M = 2_500.0, 8_500.0  # cloud deck, above the landmark
REGIONAL_SPACING_M = 150.0


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# The dive spends this many times longer inside the cloud deck than the plain descent curve would, and the part
# above the deck goes faster to make up for it. The plain curve crossed the deck in 8 frames of a 10 s shot (a
# third of a second), which read as a cloud patch and a smear; the Higgsfield whiteout lasts about a second. At
# 3x (22 frames, f24-46) Jamie saw a pause in the cloud instead of flying through it (r5): 1.8x is ~13 frames.
DECK_SLOWDOWN, ABOVE_DECK_PACE = 1.8, 0.75
# Dive tempos, (above-deck pace, deck slowdown). The descent is geometric, so below the deck the picture grows
# about 11% a frame (f56-100) whatever the ground flow cap; Twin Peaks r21 smeared the terraced rows there already
# in LTX's stage 1. "low-swoop" spends less of the dive above the deck (orbit to the deck in ~0.5 s) and gives the
# frames to the low swoop: ~9% a frame. Its deck takes ~20 frames, 12 of them solid white before the 8-frame
# thinning: with 1.3x (r23/r24) only 4 were solid, half an LTX latent, and LTX cut from orbit to clear ground
# whatever the white guide said. (r5's pause in a 22-frame deck came from a repainted cloud keyframe freezing the
# dive; the cloud keyframe now pins the blocking.)
DIVE_TEMPOS = {"standard": (ABOVE_DECK_PACE, DECK_SLOWDOWN), "low-swoop": (0.25, 2.2), "smooth-swoop": (0.25, 2.2)}
MAX_GROUND_FLOW = 0.45  # ground passing under the swooping camera, in camera heights per frame


def _ease_into(log_values: np.ndarray, end: int, frames: int, velocity_after: float) -> np.ndarray:
    """``log_values`` (per frame) with its last ``frames`` steps before ``end`` reshaped so the per-frame change
    goes smoothly from what it was to ``velocity_after``, the first step after ``end``; the values at both ends of
    the window stay put, so the shot still arrives at ``end``. A bump shaped correction (zero at both ends) puts
    back the change the braking took out."""
    out = np.array(log_values, dtype=float)
    start = max(1, end - frames)
    steps = np.diff(out)[start:end]
    if len(steps) < 2:
        return out
    ramp = _smooth((np.arange(len(steps)) + 0.5) / len(steps))
    blended = (1 - ramp) * steps + ramp * velocity_after
    bump = np.sin(np.pi * (np.arange(len(steps)) + 0.5) / len(steps)) ** 2
    blended += bump * (steps.sum() - blended.sum()) / bump.sum()
    out[start + 1:end + 1] = out[start] + np.cumsum(blended)
    return out


def _plain_descent(t, arrive_at: float = 0.45):
    s = np.clip(np.asarray(t, dtype=float) / arrive_at, 0.0, 1.0)
    return 0.6 * _smooth(s) + 0.4 * s


def _plain_height(t, arrive_at: float = 0.45, eye_m: float = 1.7):
    return np.exp(np.log(ORBIT_ALTITUDE_M) + (np.log(eye_m) - np.log(ORBIT_ALTITUDE_M))
                  * _plain_descent(t, arrive_at) ** 0.85)


def _dive_clock(t, arrive_at: float = 0.45, tempo: str = "standard") -> np.ndarray:
    """The plain descent's time for shot time ``t``: slowed inside the cloud deck, quicker above it, unchanged
    from ``arrive_at`` on. Pace changes are eased over a few hundred metres of height, so the camera speed
    never jumps."""
    plain = np.linspace(0.0, arrive_at, 4001)
    up = np.log10(_plain_height(plain, arrive_at))
    edge = 0.25  # decades of height to ease each pace change over
    inside = (_smooth((np.log10(CLOUD_TOP_M) - up) / edge + 0.5) * _smooth((up - np.log10(CLOUD_BASE_M)) / edge + 0.5))
    above = _smooth((up - np.log10(CLOUD_TOP_M)) / edge + 0.5)
    above_pace, deck_slowdown = DIVE_TEMPOS[tempo]
    pace = 1.0 + (deck_slowdown - 1.0) * inside + (above_pace - 1.0) * above  # shot time per plain time
    shot = np.concatenate(([0.0], np.cumsum(0.5 * (pace[1:] + pace[:-1]) * np.diff(plain))))
    shot *= arrive_at / shot[-1]
    t = np.asarray(t, dtype=float)
    return np.where(t < arrive_at, np.interp(t, shot, plain), t)


def _descent(t, arrive_at: float = 0.45, tempo: str = "standard"):
    """Progress of the dive from orbit to eye level, 0..1 (eased in, then steady, lingering in the cloud)."""
    return _plain_descent(_dive_clock(t, arrive_at, tempo), arrive_at)


def descent_height_m(t, arrive_at: float = 0.45, eye_m: float = 1.7, tempo: str = "standard"):
    """Camera height above the landmark (above the ground once low) over the shot, t in 0..1."""
    return np.exp(np.log(ORBIT_ALTITUDE_M) + (np.log(eye_m) - np.log(ORBIT_ALTITUDE_M)) * _descent(t, arrive_at, tempo) ** 0.85)


def cloud_entry_fraction(arrive_at: float = 0.45, tempo: str = "standard") -> float:
    """When in the shot the camera reaches the top of the cloud deck (the whiteout). The height curve depends
    only on the fraction of the shot, so this is the same for every place and duration."""
    t = np.linspace(0.0, arrive_at, 4001)
    return float(t[np.argmax(descent_height_m(t, arrive_at, tempo=tempo) <= CLOUD_TOP_M)])


def height_fraction(height_m: float, arrive_at: float = 0.45, tempo: str = "standard") -> float:
    """When in the shot the camera first comes down to ``height_m`` above the landmark."""
    t = np.linspace(0.0, arrive_at, 4001)
    return float(t[np.argmax(descent_height_m(t, arrive_at, tempo=tempo) <= height_m)])


def cloud_middle_fraction(arrive_at: float = 0.45, tempo: str = "standard") -> float:
    """When in the shot the camera is halfway through the cloud deck (in log height): the whiteout's keyframe."""
    return height_fraction(math.sqrt(CLOUD_BASE_M * CLOUD_TOP_M), arrive_at, tempo)


def plan_earth_zoom(scene: dict, duration_s: float, *, heading_deg: float | None = None,
                    lens_mm: float = 24, arrive_at: float = 0.45, eye_m: float = 1.7,
                    tempo: str = "standard") -> dict:
    """Per-frame positions, look targets and up vectors, in the local scene's metres.

    Shaped like the commercial Earth Zoom effects: low orbit looking toward the horizon, a dive into
    the cloud deck, a fast forward swoop that levels off low over the ground, then the rest of the
    shot at eye level on the landmark, pushing in slowly. ``arrive_at`` is the fraction of the shot
    spent getting down.
    """
    from .camera import Terrain
    rim = plan_route(scene, "Rim reveal", duration_s, heading_deg=heading_deg, lens_mm=lens_mm)
    heading = math.radians(rim["heading_deg"])
    forward = np.array([math.sin(heading), math.cos(heading), 0.0])
    terrain = Terrain.from_scene(scene)
    target_h = float(scene["target_height_m"])
    frames = h3_frame_count(duration_s)
    t = np.linspace(0.0, 1.0, frames)
    # Descent: distance behind the landmark and height above it fall geometrically. Height falls on a
    # faster curve than distance, so the camera gets low while still travelling: the swoop.
    s = _descent(t, arrive_at, tempo)
    hold = np.clip((t - arrive_at) / (1 - arrive_at), 0.0, 1.0)
    smooth_ride = tempo in SMOOTH_TEMPOS
    if smooth_ride:  # already moving when it arrives, and still moving at the end (r44's last frames were still)
        hold = 0.7 * hold + 0.3 * (1 - (1 - hold) ** 2)
    else:
        hold = _smooth(hold)
    hold_back = 2.0 + 10.0 * (1 - hold)  # eye-level push-in, 12 m -> 2 m
    distance = np.exp(np.log(ORBIT_DISTANCE_M) + (np.log(hold_back) - np.log(ORBIT_DISTANCE_M)) * s ** 0.9)
    height = descent_height_m(t, arrive_at, eye_m, tempo)
    arrive = int(np.ceil(arrive_at * (frames - 1) - 1e-9))  # the first frame at eye level
    if smooth_ride and arrive + 1 < frames:
        log_distance = np.log(distance)
        distance = np.exp(_ease_into(log_distance, arrive, ARRIVAL_EASE_FRAMES,
                                     log_distance[arrive + 1] - log_distance[arrive]))
        height = np.exp(_ease_into(np.log(height), arrive, ARRIVAL_EASE_FRAMES, 0.0))
    # Keep the ground from streaming past faster than MAX_GROUND_FLOW camera heights a frame: on the plain curve
    # the swoop kept about 0.7 all the way down (the camera falls as fast as it slows), which LTX smeared at
    # f60-100 in every place. A soft maximum with that floor keeps the swoop higher while it is fast, then lets
    # it settle to eye level as it slows.
    floor = np.abs(np.gradient(distance)) / MAX_GROUND_FLOW
    height = (height ** 4 + floor ** 4) ** 0.25
    subject = standing_spot(scene, forward[:2]) if scene.get("landing_scene") else (0.0, 0.0)
    xy = np.array(subject)[None, :] - distance[:, None] * forward[None, :2]
    ground = np.nan_to_num(terrain.sample(xy[:, 0], xy[:, 1]), nan=target_h)
    # Below the cloud deck the height is above the real ground; above it, above the landmark.
    landmark_ground = float(np.nan_to_num(terrain.sample(np.array([subject[0]]), np.array([subject[1]])),
                                          nan=target_h)[0])
    ride = np.maximum(ground, landmark_ground)  # see over rises to the view
    if smooth_ride:
        ride = np.array(ride_floor(ride.tolist(), RIDE_FLOOR_FRAMES))
    base = np.where(height < 3000, ride, target_h)
    positions = np.column_stack((xy, base + height))
    # Look at the landmark on the way down, then level out toward the view past it.
    level = _smooth((t - 0.8 * arrive_at) / (0.2 * arrive_at))[:, None]
    at_landmark = np.array([subject[0], subject[1], target_h])[None, :] - positions
    ahead = np.repeat((forward + np.array([0, 0, -0.05]))[None, :], frames, axis=0)
    directions = (1 - level) * at_landmark / np.linalg.norm(at_landmark, axis=1, keepdims=True) + level * ahead
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ups = np.repeat(np.array([[0.0, 0.0, 1.0]]), frames, axis=0)
    reach = np.maximum(np.linalg.norm(at_landmark, axis=1), 50.0)[:, None]
    look_at = positions + directions * reach
    clearance = positions[:, 2] - ground
    route = {
        "schema_version": "1.1", "preset": PRESET, "description": DESCRIPTION, "kind": "earth_zoom",
        "scene": scene["project_id"], "terrain_path": scene["terrain_path"], "fps": FPS, "frames": frames,
        "duration_s": round(frames / FPS, 3), "requested_duration_s": duration_s,
        "chain_segments": [[0, frames]], "heading_deg": rim["heading_deg"], "heading_source": rim["heading_source"],
        "lens_mm": lens_mm, "positions": positions.round(3).tolist(), "look_at": look_at.round(3).tolist(),
        "up": ups.round(6).tolist(), "altitude_m": positions[:, 2].round(1).tolist(),
        "above_ground_m": clearance.round(2).tolist(),
        "arrive_frame": int(arrive_at * (frames - 1)), "eye_height_m": eye_m,
        "min_clearance_m": round(float(clearance.min()), 2), "max_lift_m": 0.0,
        "scale_m": float(ORBIT_DISTANCE_M), "target_local_m": [subject[0], subject[1], target_h],
        "subject_local_m": [round(subject[0], 2), round(subject[1], 2)],
    }
    if tempo != "standard":  # standard routes keep their hash, so their renders are reused
        route["dive_tempo"] = tempo
    route["route_sha256"] = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()
    return route


def standing_spot(scene: dict, forward: np.ndarray, reach_m: float = 10.0) -> tuple[float, float]:
    """Where the figure stands on the 1 m landing patch: the landmark point, or the nearest point within
    ``reach_m`` that is level underfoot and that the eye-level camera, 2-12 m behind it, sees down to the
    chest. 1 m lidar resolves steps a 10 m DEM smooths away: Mather Point's catalog point is on a rock
    step 2-3 m below the path behind it, so the figure stood on a slope, hidden under the camera's view."""
    def level(x: float, y: float) -> float:
        heights = [mesh_height(scene, x + dx, y + dy) for dx, dy in ((0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6))]
        return max(heights) - min(heights)

    def seen(x: float, y: float) -> float:
        ground, worst = mesh_height(scene, x, y), math.inf
        for back in (2.0, 6.0, 12.0):
            cx, cy = x - forward[0] * back, y - forward[1] * back
            eye = max(mesh_height(scene, cx, cy), ground) + 1.7
            for t in np.linspace(0.05, 0.95, 19):
                line = eye + t * (ground + 1.2 - eye)
                worst = min(worst, line - mesh_height(scene, cx + t * (x - cx), cy + t * (y - cy)))
        return worst

    if level(0.0, 0.0) < 0.4 and seen(0.0, 0.0) > 0.3:
        return 0.0, 0.0
    for radius in np.arange(0.5, reach_m + 0.01, 0.5):
        for angle in np.linspace(0, 2 * math.pi, int(8 * radius) + 8, endpoint=False):
            x, y = float(radius * math.sin(angle)), float(radius * math.cos(angle))
            if level(x, y) < 0.4 and seen(x, y) > 0.3:
                return x, y
    return 0.0, 0.0


def cloud_deck_m(scene: dict) -> tuple[float, float]:
    """Cloud base to top for the dive: from about 2.5 km above the landmark, 6 km thick (about a second)."""
    base = float(scene["target_height_m"]) + CLOUD_BASE_M
    return base, base + CLOUD_TOP_M - CLOUD_BASE_M


ZOOM_SCRIPT = Path(__file__).with_name("blender_earth_zoom.py")


def regional_scene(scene: dict, output_root: Path, raw_cache: Path) -> dict:
    """The wide, coarse terrain the descent passes over between the globe and the local scene."""
    from .scene import build_landform_scene, default_provider
    place = scene["place"]
    # Same terrain family as the local scene, so both share one projected CRS and height datum.
    family = scene.get("terrain_family", "usgs3dep")
    return build_landform_scene(
        name=f"{scene['name']} (region)", latitude=place["latitude"], longitude=place["longitude"],
        radius_m=REGIONAL_RADIUS_M, output_root=output_root, raw_cache=raw_cache,
        location_source=place.get("source", "landmark"), raster_spacing_m=REGIONAL_SPACING_M,
        provider=default_provider(family, REGIONAL_SPACING_M))


def render_earth_zoom(scene: dict, route: dict, output_root: Path, *, raw_cache: Path, width: int = 1280,
                      height: int = 720, samples: int = 16, palette: str = "default", weather: str = "clear",
                      imagery: bool = True, trees: str = "off", buildings: bool = False,
                      progress=None) -> dict:
    """Build the regional terrain, globe textures and imagery, then render the zoom in Blender."""
    import subprocess
    from .blender import find_blender
    from .earth import fetch_globe
    import httpx
    from .imagery import fetch_naip_imagery, fetch_terrain_imagery
    region = regional_scene(scene, Path(scene["terrain_dir"]).parent, raw_cache)
    place = scene["place"]
    globe = fetch_globe(place["latitude"], place["longitude"], Path(region["terrain_dir"]) / "globe")
    zoom = {"route": route, "globe": globe, "regional_scene": region, "local_scene": scene,
            "cloud_deck_m": cloud_deck_m(scene)}
    landing = scene.get("landing_scene")
    if landing:
        zoom["landing_scene"] = landing
    if imagery:
        for key, terrain in (("regional_imagery", region), ("local_imagery", scene)):
            fetch_terrain_imagery(terrain, Path(terrain["terrain_dir"]) / "imagery")
            zoom[key] = str(Path(terrain["terrain_dir"]) / "imagery" / "imagery.json")
        if landing:  # NAIP aerial photos on the 1 m patch; 30 m HLS if NAIP is unreachable
            folder = Path(landing["terrain_dir"]) / "imagery"
            try:
                fetch_naip_imagery(landing, folder, reference=json.loads(Path(zoom["local_imagery"]).read_text()))
            except (httpx.HTTPError, ValueError) as exc:
                print(f"NAIP for the landing patch failed ({exc}); using HLS", flush=True)
                fetch_terrain_imagery(landing, folder, pixels=2048)
            zoom["landing_imagery"] = str(folder / "imagery.json")
    building_hash, building_manifest = None, None
    if buildings:  # Overture footprints and heights around the landing
        from .buildings import building_report, fetch_buildings
        building_manifest = fetch_buildings(scene, Path(scene["terrain_dir"]) / "buildings", cache_dir=raw_cache)
        print(building_report(building_manifest), flush=True)
        zoom["buildings"] = building_manifest["path"]
        zoom["buildings_attribution"] = building_manifest["attribution"]
        building_hash = building_manifest["key"]
    tree_hash = None
    if trees != "off":
        from .trees import fetch_trees
        tree_manifest = fetch_trees(scene, route, Path(scene["terrain_dir"]) / "trees", cache_dir=raw_cache,
                                    source=trees, buildings=building_manifest)
        zoom["trees"] = tree_manifest["path"]
        tree_hash = tree_manifest["key"]
    settings = {"route": route["route_sha256"], "width": width, "height": height, "samples": samples,
                "palette": palette, "weather": weather, "weather_values": json.loads(json.dumps(WEATHER[weather])), "imagery": imagery,
                "landing": landing["terrain_path"] if landing else None,
                **({"trees": tree_hash, "tree_code": hashlib.sha256(ZOOM_SCRIPT.with_name("trees.py").read_bytes()).hexdigest()}
                   if tree_hash else {}),
                **({"buildings": building_hash,
                    "building_code": hashlib.sha256(ZOOM_SCRIPT.with_name("buildings.py").read_bytes()).hexdigest()}
                   if building_hash else {}),
                "script": hashlib.sha256(ZOOM_SCRIPT.read_bytes()).hexdigest(),
                "base": hashlib.sha256(ZOOM_SCRIPT.with_name("blender_render.py").read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    output_dir = output_root / scene["project_id"] / f"earth-zoom-{key}"
    manifest_path = output_dir / "render.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("settings") == settings and Path(manifest.get("video") or "").is_file():
            return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom_path = output_dir / "zoom.json"
    zoom_path.write_text(json.dumps(zoom) + "\n")
    command = find_blender()
    command[command.index(str(ZOOM_SCRIPT.with_name("blender_render.py")))] = str(ZOOM_SCRIPT)
    command += ["--", "--zoom", str(zoom_path), "--output-dir", str(output_dir), "--width", str(width),
                "--height", str(height), "--samples", str(samples), "--palette", palette, "--weather", weather]
    log_lines: list[str] = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            log_lines.append(line)
            if line.startswith("GEOBLOCK_PROGRESS") and progress:
                done, total = line.split()[1].split("/")
                progress(int(done), int(total))
    (output_dir / "blender.log").write_text("".join(log_lines))
    if process.returncode != 0:
        raise RuntimeError(f"Earth zoom render failed (exit {process.returncode}); see {output_dir / 'blender.log'}\n"
                           + "".join(log_lines[-25:]))
    manifest = json.loads(manifest_path.read_text())
    manifest["settings"] = settings
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@functools.lru_cache(maxsize=2)
def _mesh_grid(path: str) -> tuple:
    """The terrain mesh's regular height grid (heights[row north, col east]), as exported by processing."""
    import trimesh
    vertices = trimesh.load(path, force="mesh").vertices  # glTF: east, up, -north
    east, up, north = vertices[:, 0], vertices[:, 1], -vertices[:, 2]
    xs, ys = np.unique(east.round(3)), np.unique(north.round(3))
    heights = np.full((len(ys), len(xs)), np.nan)
    heights[np.searchsorted(ys, north.round(3)), np.searchsorted(xs, east.round(3))] = up
    return xs, ys, heights


def mesh_height(scene: dict, x: float, y: float) -> float:
    """Height of the rendered terrain mesh at local (x, y), on the same triangles Blender ray-casts onto: the
    1 m landing patch where there is one, else the local terrain."""
    landing = scene.get("landing_scene")
    path = scene["terrain_path"]
    if landing and max(abs(x), abs(y)) < landing["radius_m"] - 2 * landing["mesh_spacing_m"]:
        path = landing["terrain_path"]
    xs, ys, heights = _mesh_grid(path)
    step = xs[1] - xs[0]
    i, j = int((x - xs[0]) // step), int((y - ys[0]) // step)
    tx, ty = (x - xs[i]) / step, (y - ys[j]) / step
    if tx + ty <= 1:  # faces (a, b, c) and (b, d, c) of each grid square
        return float(heights[j, i] + tx * (heights[j, i + 1] - heights[j, i]) + ty * (heights[j + 1, i] - heights[j, i]))
    return float(heights[j + 1, i + 1] + (1 - tx) * (heights[j + 1, i] - heights[j + 1, i + 1])
                 + (1 - ty) * (heights[j, i + 1] - heights[j + 1, i + 1]))


def _camera(scene: dict, route: dict, frame: int) -> tuple:
    """Blender's Earth zoom camera at ``frame``: position, forward, right and up unit vectors, and the focal
    length in image widths (36 mm sensor fitted to the width). Near the ground it keeps its height above the
    higher of the ground below it and the landmark's ground, as blender_earth_zoom places it."""
    p, look, up = (np.array(route[key][frame], dtype=float) for key in ("positions", "look_at", "up"))
    if route.get("dive_tempo") in SMOOTH_TEMPOS:
        shift = _ride_shifts(scene, route)[frame]
    else:
        shift = _ground_shift(scene, route, frame)
    p[2] += shift
    look[2] += shift
    forward = (look - p) / np.linalg.norm(look - p)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return p, forward, right, np.cross(right, forward), route["lens_mm"] / 36.0


def _ground_shift(scene: dict, route: dict, frame: int) -> float:
    """How far blender_earth_zoom lifts the camera at ``frame`` to keep its height over the mesh (0 above 200 m)."""
    height, (x, y, z) = route["above_ground_m"][frame], route["positions"][frame]
    if height >= 200:
        return 0.0
    sx, sy = route.get("subject_local_m", (0.0, 0.0))
    return max(mesh_height(scene, x, y), mesh_height(scene, sx, sy)) + height - z


def _ride_shifts(scene: dict, route: dict) -> list:
    """Every frame's lift on a smooth ride: the upper envelope blender_earth_zoom rides from the first low frame."""
    key = (id(scene), route.get("route_sha256"))
    if key not in _RIDE_CACHE:
        shifts = [_ground_shift(scene, route, frame) for frame in range(len(route["positions"]))]
        low = next((i for i, h in enumerate(route["above_ground_m"]) if h < 200), len(shifts))
        shifts[low:] = ride_floor(shifts[low:], RIDE_FLOOR_FRAMES)
        _RIDE_CACHE.clear()
        _RIDE_CACHE[key] = shifts
    return _RIDE_CACHE[key]


_RIDE_CACHE: dict = {}


def reproject_hold(scene: dict, route: dict, from_frame: int, to_frame: int, image: np.ndarray,
                   far_m: float = 4000.0, steps: int = 320, grid: int = 320, grain: float = 0.0,
                   fill: np.ndarray | None = None, near_m: float = 0.0) -> tuple:
    """``image`` (painted for ``from_frame``) as ``to_frame``'s camera sees it, moved through the scene's terrain:
    each pixel of the later view is traced to the ground (or to the sky and anything past ``far_m``, which only
    turns with the camera) and looked up where the earlier camera saw that point. The hold dollies in on a fixed
    lens, so the distant city stays about the same size while the near ground and the person grow; the old
    person-scaled crop (push_in_crop) zoomed the whole picture instead, blowing the skyline up ~6x by f240
    (second-opinion review, r18). Buildings are not in the heightfield: they move like the ground behind them,
    a small error over a dolly of a few metres. Returns the image, a mask of pixels the earlier view covers (the
    rest is filled from around it, or from ``fill``, for the painter to redo), how much each pixel was stretched and blurred
    (0..1), for a repaint of that ground, and where it may be pasted back over a repaint (kept_mask)."""
    from .camera import Terrain
    rows, cols = image.shape[:2]
    terrains = [Terrain.from_scene(scene)]
    if scene.get("landing_scene"):
        terrains.insert(0, Terrain.from_scene(scene["landing_scene"]))

    def ground(x, y):
        out = np.full(np.shape(x), np.nan)
        for terrain in terrains:  # the 1 m landing patch first, where it reaches
            out = np.where(np.isnan(out), terrain.sample(x, y), out)
        return out

    pa, fa, ra, ua, focal = _camera(scene, route, from_frame)
    pb, fb, rb, ub, _ = _camera(scene, route, to_frame)
    aspect = cols / rows
    # Traced on a grid of at most ``grid`` columns, then the lookup positions are interpolated to full size.
    grid_cols = min(cols, grid)
    grid_rows = max(1, round(grid_cols / aspect))
    u = (np.arange(grid_cols) + 0.5) / grid_cols - 0.5
    v = (np.arange(grid_rows) + 0.5) / grid_rows - 0.5
    distances = np.geomspace(0.3, far_m, steps)
    source = np.zeros((grid_rows, grid_cols, 2))
    seen = np.zeros((grid_rows, grid_cols), bool)
    reach = np.full((grid_rows, grid_cols), np.inf)  # how far the later camera's ray goes before the ground
    for r in range(grid_rows):
        rays = fb[None, :] + (u[:, None] / focal) * rb[None, :] - (v[r] / (focal * aspect)) * ub[None, :]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        points = pb[None, None, :] + rays[:, None, :] * distances[None, :, None]  # cols x steps x 3
        below = points[..., 2] <= ground(points[..., 0], points[..., 1])
        hit = below.any(axis=1)
        first = np.argmax(below, axis=1)
        target = np.where(hit[:, None], points[np.arange(grid_cols), first], rays * 1e9)  # far: at infinity
        reach[r] = np.where(hit, distances[first], np.inf)
        rel = target - pa[None, :]
        depth = rel @ fa
        sx = 0.5 + focal * (rel @ ra) / np.maximum(depth, 1e-6)
        sy = 0.5 - focal * aspect * (rel @ ua) / np.maximum(depth, 1e-6)
        source[r, :, 0], source[r, :, 1] = sx * cols - 0.5, sy * rows - 0.5
        seen[r] = (depth > 0) & (sx >= 0) & (sx <= 1) & (sy >= 0) & (sy <= 1)
    if (grid_rows, grid_cols) != (rows, cols):
        from PIL import Image
        source = np.dstack([np.asarray(Image.fromarray(source[..., k].astype(np.float32)).resize(
            (cols, rows), Image.BILINEAR)) for k in range(2)])
        seen = np.asarray(Image.fromarray(seen.astype(np.float32)).resize((cols, rows), Image.BILINEAR)) > 0.5
        reach = np.asarray(Image.fromarray(np.minimum(reach, 1e6).astype(np.float32)).resize(
            (cols, rows), Image.BILINEAR))
    x = np.clip(source[..., 0], 0, cols - 1)
    y = np.clip(source[..., 1], 0, rows - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, cols - 1), np.minimum(y0 + 1, rows - 1)
    tx, ty = (x - x0)[..., None], (y - y0)[..., None]
    out = (image[y0, x0] * (1 - tx) * (1 - ty) + image[y0, x1] * tx * (1 - ty)
           + image[y1, x0] * (1 - tx) * ty + image[y1, x1] * tx * ty)
    weight = stretch_weight(source)
    if near_m > 0:
        # Ground within near_m of the later camera goes to the painter too, however little it was magnified:
        # bushes and grass a few metres away stand up off the heightfield, so moved as ground they shear into
        # long streaks (Twin Peaks r46-r48 f168-192, the f112 plate's near sage bushes).
        weight = np.maximum(weight, np.clip((NEAR_FADE * near_m - reach) / ((NEAR_FADE - 1) * near_m), 0.0, 1.0))
    kept = kept_mask(seen, magnification(source)) * (1.0 - weight)
    covered = seen
    if fill is not None:  # the later camera's own blocking frame where the earlier plate doesn't reach
        out = _blend_unseen(out, seen, fill)
        covered = np.ones_like(seen)
    soft = _soften_stretched(_fill(out, ~covered), weight)
    return _add_grain(soft, weight, grain, seed=from_frame * 1000 + to_frame), seen, weight, kept


def unseen_mask(seen: np.ndarray) -> np.ndarray:
    """1 where ``seen`` is false, feathered outward over cols/60: where a fill went in for the painter to redo."""
    from .people import _box_blur
    cols = seen.shape[1]
    alpha = np.clip(_box_blur((~seen).astype(np.float64), max(1, round(cols / 60))), 0.0, 1.0)
    return np.maximum(alpha, (~seen).astype(np.float64))


def _blend_unseen(image: np.ndarray, seen: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """``fill`` (resized to ``image``) where ``seen`` is false, with a feathered edge. Filled from around it, the
    unseen 57% of Twin Peaks r42's f88 landing plate was flat tiles and klein lost the road there; the blocking
    frame has the road, the slope and the kerb where they are."""
    from PIL import Image
    from .people import _box_blur
    rows, cols = image.shape[:2]
    if fill.shape[:2] != (rows, cols):
        fill = np.dstack([np.asarray(Image.fromarray(fill[..., k].astype(np.float32)).resize(
            (cols, rows), Image.BILINEAR)) for k in range(3)])
    alpha = unseen_mask(seen)  # fully the blocking inside, feathered outward only
    return image * (1 - alpha[..., None]) + fill[..., :3] * alpha[..., None]


NEAR_FADE = 1.5  # reproject_hold near_m: the near ground's repaint weight fades out from near_m to 1.5x it
STRETCH_SOFTEN = (2.0, 4.0)  # reprojected ground stretched more than this (start, full) is blurred for the painter


KEEP_MAGNIFICATION = (1.15, 1.35)  # kept_mask: pasted back whole below the first, not at all past the second


def kept_mask(seen: np.ndarray, magnification: np.ndarray) -> np.ndarray:
    """0..1 per pixel: where a reprojected plate can stand in for a repaint, the view the earlier plate covers
    and magnifies little (KEEP_MAGNIFICATION), eased off over a feathered edge. Repainted whole at the hold plate's
    denoise, every hold keyframe drew its own midground (r44: a winding path and lawn appeared from f192 that
    f112-f176 lacked), which LTX showed as details popping in; pasted back from the earlier plate, it stays. Kept
    up to the 2x stretch blur, the near bushes (not in the heightfield) came back as long radial streaks (r46)."""
    from .people import _box_blur
    start, end = KEEP_MAGNIFICATION
    keep = seen.astype(np.float64) * np.clip((end - magnification) / (end - start), 0.0, 1.0)
    radius = max(1, round(seen.shape[1] / 120))
    return np.clip(np.minimum(keep, _box_blur(_box_blur(keep, radius), radius)), 0.0, 1.0)


def magnification(source: np.ndarray) -> np.ndarray:
    """How far the lookup ``source`` (source pixel per output pixel) magnifies the earlier plate, per pixel, in
    its most stretched direction: the inverse of the smallest singular value of its Jacobian."""
    dxdu, dxdv = np.gradient(source[..., 0], axis=1), np.gradient(source[..., 0], axis=0)
    dydu, dydv = np.gradient(source[..., 1], axis=1), np.gradient(source[..., 1], axis=0)
    a = dxdu ** 2 + dydu ** 2
    b = dxdu * dxdv + dydu * dydv
    c = dxdv ** 2 + dydv ** 2
    smallest = np.sqrt(np.maximum((a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b ** 2), 0.0))
    return 1.0 / np.maximum(smallest, 1e-6)


def stretch_weight(source: np.ndarray) -> np.ndarray:
    """0..1 per pixel: how far past STRETCH_SOFTEN the lookup ``source`` (source pixel per output pixel)
    magnifies the earlier plate, feathered. The stretch is the larger singular value of how far the lookup moves
    per output pixel, inverted."""
    from .people import _box_blur
    stretch = magnification(source)
    start, full = STRETCH_SOFTEN
    weight = np.clip((stretch - start) / (full - start), 0.0, 1.0)
    if not weight.any():
        return weight
    radius = max(1, round(source.shape[1] / 120))
    return _box_blur(_box_blur(weight, radius), radius)  # feathered edge


def _soften_stretched(image: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Blur where the reprojection magnified the earlier plate (``weight`` from stretch_weight): the near ground
    of a later hold frame comes from a few pixels of the earlier one, drawn out into radial streaks that klein
    kept as furrows and green stripes (r22k f176/f208). Blurred, it is only colour and layout (r23k: the streaks
    went, but klein left the blur as defocus, so the composite repaints that ground too)."""
    from .people import _box_blur
    if not weight.any():
        return image
    radius = max(1, round(image.shape[1] / 120))
    soft = image
    for _ in range(3):  # about a Gaussian over ~cols/40
        soft = _box_blur(soft, 2 * radius)
    return image * (1 - weight[..., None]) + soft * weight[..., None]


def _add_grain(image: np.ndarray, weight: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Fine noise of standard deviation ``amount`` where ``weight`` (stretch_weight) says the ground was blurred.
    klein repainting the hold plates kept the blur as defocus at any denoise (Twin Peaks r34: f208 the same at 0.6
    and 0.8); noise gives it texture to paint into grass and gravel. Mostly grey, a little colour, at one and two
    pixels. Seeded, so a run is reproducible."""
    from .people import _box_blur
    if amount <= 0 or not weight.any():
        return image
    rng = np.random.default_rng(seed)
    rows, cols = image.shape[:2]
    fine = rng.standard_normal((rows, cols, 1)) + 0.3 * rng.standard_normal((rows, cols, 3))
    coarse = _box_blur(rng.standard_normal((rows, cols)), 1)[..., None] * 2.0  # box blur of 3 px: std ~1/3
    noise = (fine + coarse) / np.sqrt(1.0 + 0.09 + 4.0 / 9.0)
    return np.clip(image + amount * noise * weight[..., None], 0.0, 1.0)


def stand_in_on_screen(scene: dict, route: dict, frame: int, aspect: float = 16 / 9,
                       height_m: float = 1.75) -> tuple[float, float, float]:
    """Where the Earth zoom's stand-in figure sits in ``frame``: the centre (x, y) and height of its feet-to-head
    line, as fractions of the image width, height and height. Mirrors blender_earth_zoom: near the ground the
    camera keeps its height above the higher of the ground below it and the landmark's ground, and the figure
    stands on the landmark (local origin). Blender's 36 mm sensor, fitted to the image width."""
    p, forward, right, true_up, focal = _camera(scene, route, frame)
    sx, sy = route.get("subject_local_m", (0.0, 0.0))
    landmark = mesh_height(scene, sx, sy)
    points = []
    for z in (landmark, landmark + height_m):
        v = np.array([sx, sy, z]) - p
        depth = max(float(v @ forward), 1e-3)
        points.append((0.5 + focal * float(v @ right) / depth, 0.5 - focal * aspect * float(v @ true_up) / depth))
    (x0, y0), (x1, y1) = points
    return (x0 + x1) / 2, (y0 + y1) / 2, abs(y0 - y1)


def push_in_crop(scene: dict, route: dict, from_frame: int, to_frame: int, aspect: float = 16 / 9) -> tuple:
    """The part of ``from_frame``'s picture that ``to_frame`` sees around the stand-in, as (x, y, width, height)
    fractions of the image: scaled by how much larger the figure gets and shifted so the figure lands where it
    stands in ``to_frame``. Can reach past the image edges."""
    xa, ya, ha = stand_in_on_screen(scene, route, from_frame, aspect)
    xb, yb, hb = stand_in_on_screen(scene, route, to_frame, aspect)
    scale = hb / ha
    return xa - xb / scale, ya - yb / scale, 1 / scale, 1 / scale


CHROMA_GREEN = 0.15  # a sheet border this much greener than its red and blue is a chroma-key wall


def _background(sheet: np.ndarray, margin: float = 0.06) -> np.ndarray:
    """Mask of a subject sheet's plain background: pixels about its border colour (as far off as the border's
    own shading, plus ``margin``) that connect to the border, so a grey jacket inside the figure is kept even
    when it is as grey as the wall behind it."""
    border = np.concatenate([sheet[:4].reshape(-1, 3), sheet[-4:].reshape(-1, 3),
                             sheet[:, :4].reshape(-1, 3), sheet[:, -4:].reshape(-1, 3)])
    colour = np.median(border, axis=0)
    tolerance = float(np.percentile(np.linalg.norm(border - colour, axis=1), 98)) + margin
    near = np.linalg.norm(sheet - colour, axis=2) < tolerance
    green = colour[1] - max(colour[0], colour[2])
    if green > CHROMA_GREEN:
        # A green wall keyed by how green each pixel is, not by distance to one colour: sunlit for the landing's
        # light, it had a dark green shadow of the person beside them and a paler floor, and by colour distance
        # those were kept as the figure (r19k). Green straps are far less green than the wall's darkest shadow.
        dominance = sheet[..., 1] - np.maximum(sheet[..., 0], sheet[..., 2])
        near = dominance > 0.5 * green
    filled = np.zeros_like(near)
    filled[0], filled[-1], filled[:, 0], filled[:, -1] = near[0], near[-1], near[:, 0], near[:, -1]
    while True:
        grown = filled.copy()
        grown[1:] |= filled[:-1]
        grown[:-1] |= filled[1:]
        grown[:, 1:] |= filled[:, :-1]
        grown[:, :-1] |= filled[:, 1:]
        grown &= near
        if (grown == filled).all():
            return filled
        filled = grown


def _central_figure(figure: np.ndarray) -> np.ndarray:
    """Only the part of a keyed sheet's figure mask joined to its centre column (the sheet asks for the person
    centred): a speck kept at the wall-floor seam stretched the crop to the sheet's edge (r19k)."""
    middle = figure.shape[1] // 2
    seed = np.zeros_like(figure)
    band = slice(max(0, middle - figure.shape[1] // 20), middle + figure.shape[1] // 20 + 1)
    seed[:, band] = figure[:, band]
    if not seed.any():
        return figure
    while True:
        grown = seed.copy()
        grown[1:] |= seed[:-1]
        grown[:-1] |= seed[1:]
        grown[:, 1:] |= seed[:, :-1]
        grown[:, :-1] |= seed[:, 1:]
        grown &= figure
        if (grown == seed).all():
            return seed
        seed = grown


def _fill(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """``hole`` filled smoothly from the pixels around it (push-pull: holes take the colour of an ever coarser
    average of their surroundings), so a painted-out area reads as soft background rather than streaks."""
    if not hole.any() or hole.all():
        return image
    weight = (~hole).astype(np.float64)
    levels = [(image * weight[..., None], weight)]
    while max(levels[-1][1].shape) > 1 and (levels[-1][1] == 0).any():
        colour, w = levels[-1]
        rows, cols = (w.shape[0] + 1) // 2 * 2, (w.shape[1] + 1) // 2 * 2
        colour = np.pad(colour, ((0, rows - w.shape[0]), (0, cols - w.shape[1]), (0, 0)))
        w = np.pad(w, ((0, rows - w.shape[0]), (0, cols - w.shape[1])))
        levels.append((colour.reshape(rows // 2, 2, cols // 2, 2, -1).sum(axis=(1, 3)),
                       w.reshape(rows // 2, 2, cols // 2, 2).sum(axis=(1, 3))))
    colour, w = levels[-1]
    filled = colour / np.maximum(w, 1e-9)[..., None]
    for colour, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)[:w.shape[0], :w.shape[1]]
        known = np.clip(w, 0, 1)[..., None]
        filled = np.where(known > 0, colour / np.maximum(w, 1e-9)[..., None], up)
    out = image.copy()
    out[hole] = filled[hole]
    return out


def erase_stand_in(frame: np.ndarray, centre_x: float, feet_y: float, height: float) -> np.ndarray:
    """The grey stand-in figure painted out of a blocking frame: its box (body and arm cylinders) is filled in
    smoothly from the scene around it, so no grey head or arms show around a pasted person."""
    rows, cols = frame.shape[:2]
    half = 0.22 * height * rows  # arms included
    x0, x1 = max(0, round(centre_x * cols - half)), min(cols, round(centre_x * cols + half))
    y0, y1 = max(0, round((feet_y - 1.12 * height) * rows)), min(rows, round(feet_y * rows) + 2)
    if x1 <= x0 or y1 <= y0:
        return frame
    hole = np.zeros(frame.shape[:2], bool)
    hole[y0:y1, x0:x1] = True
    return _fill(frame, hole)


def _despill(figure: np.ndarray, background: np.ndarray, colour: np.ndarray, band: int = 4,
             spill: float = 0.04) -> np.ndarray:
    """A chroma-key wall's colour taken back out of the figure's outline (``band`` pixels in from the wall),
    where the wall's light rims hair and sleeves, and any key colour more than ``spill`` above the other two
    channels elsewhere on the figure. Only for a strongly coloured wall."""
    key = int(np.argmax(colour))
    others = [c for c in range(3) if c != key]
    if colour[key] - colour[others].max() < 0.2:
        return figure
    edge = background.copy()
    for _ in range(band):
        grown = edge.copy()
        grown[1:] |= edge[:-1]
        grown[:-1] |= edge[1:]
        grown[:, 1:] |= edge[:, :-1]
        grown[:, :-1] |= edge[:, 1:]
        edge = grown
    edge &= ~background
    out = figure.copy()
    limit = out[..., others].max(axis=-1)
    # Along the edge the key colour is cut back fully; inside, only beyond a little headroom, so green clothes
    # stay greenish while the wall's spill on skin and pale fabric goes (r10k: green hands and sides at f176).
    out[..., key] = np.minimum(out[..., key], np.where(edge, limit, limit + spill))
    return out


def shadow_factor(clean: np.ndarray, shadowed: np.ndarray) -> np.ndarray:
    """How much the stand-in's shadow darkens each pixel: the clean plate rendered with the stand-in hidden from
    the camera but still casting its shadow, over the plate without it (1 where nothing changes). Multiplied onto
    a painted plate under the pasted person, it gives them a ground shadow with the blocking sun's direction
    and length (Twin Peaks r9k: pasted with none, the person floated)."""
    lum = lambda image: image @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731
    ratio = lum(shadowed) / np.maximum(lum(clean), 1e-3)
    return np.clip(np.where(ratio < 0.97, ratio, 1.0), 0.25, 1.0)


def shadow_near(shade: np.ndarray, centre_x: float, feet_y: float, height: float) -> np.ndarray:
    """``shade`` kept only within two person-heights of the feet: anything else the two plate passes differ by
    is not the person's shadow."""
    rows, cols = shade.shape
    yy, xx = np.ogrid[:rows, :cols]
    near = np.hypot(xx - centre_x * cols, (yy - feet_y * rows)) < 2.0 * height * rows
    return np.where(near, shade, 1.0)


def contact_shadow(frame: np.ndarray, centre_x: float, feet_y: float, height: float,
                   strength: float = 0.45) -> np.ndarray:
    """A soft dark ellipse on the ground under the feet, where the person meets it. The rendered shadow can fall
    behind them (sun behind the camera) and a pasted person with no contact floats (r10k)."""
    rows, cols = frame.shape[:2]
    cx, cy = centre_x * cols, feet_y * rows
    ax, ay = max(1.0, 0.16 * height * rows), max(1.0, 0.025 * height * rows)
    yy, xx = np.ogrid[:rows, :cols]
    d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    dark = strength * np.clip(1 - d, 0, 1) ** 1.5
    return frame * (1 - dark)[..., None]


def _match_light(patch: np.ndarray, alpha: np.ndarray, scene: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """The pasted person's colour cast and brightness moved ``amount`` of the way toward the ``scene`` pixels
    around them, so a studio-lit figure doesn't sit on a sunlit hillside in flat white light (r10k)."""
    inside = alpha > 0.5
    if len(scene) < 16 or inside.sum() < 16:
        return patch
    lum = np.array([0.2126, 0.7152, 0.0722])
    scene, person = scene.mean(axis=0), patch[inside].mean(axis=0)
    if float(scene @ lum) < 0.02:  # nothing to match (a black test frame)
        return patch
    cast = (scene / max(float(scene @ lum), 1e-3)) / np.maximum(person / max(float(person @ lum), 1e-3), 1e-3)
    bright = np.clip(float(scene @ lum) / max(float(person @ lum), 1e-3), 0.6, 1.4)
    gain = 1 + amount * (cast * bright - 1)
    return np.clip(patch * gain, 0, 1)


def place_subject(frame: np.ndarray, sheet: np.ndarray, centre_x: float, feet_y: float, height: float,
                  clean: np.ndarray | None = None, erase: bool = True, shade: np.ndarray | None = None,
                  with_mask: bool = False, ring: float = 0.12):
    """The person from a subject sheet (one figure on a plain background) pasted over a blocking frame, feet at
    (``centre_x``, ``feet_y``) and ``height`` tall, all as fractions of the frame. Float RGB arrays in 0-1.

    klein draws the hiker full-length and centred whatever size the grey stand-in is (Twin Peaks: twice its
    height at f112, smaller than it at f240), so keyframes disagree with the push-in and LTX swings between
    them. Given the person already at the stand-in's place and size, klein keeps them there. ``clean`` is the
    same frame rendered without the stand-in; without it the stand-in is painted out, unless ``erase`` is off
    (a frame with no stand-in in it). ``shade`` (the stand-in's shadow, see shadow_factor) darkens the ground
    under the paste. ``with_mask`` also returns the person's mask grown by ``ring`` of their height, for a blend
    pass that relights only them and the ground at their feet."""
    rows, cols = frame.shape[:2]
    if clean is not None and clean.shape == frame.shape:
        frame = clean
    elif erase:
        frame = erase_stand_in(frame, centre_x, feet_y, height)
    if shade is not None and shade.shape == frame.shape[:2] and (shade < 0.97).any():
        frame = frame * shade[..., None]
    else:  # no rendered shadow to give them: at least a soft one where the feet meet the ground
        frame = contact_shadow(frame, centre_x, feet_y, height)
    full = np.zeros((rows, cols))
    background = _background(sheet)
    border = np.concatenate([sheet[:4].reshape(-1, 3), sheet[-4:].reshape(-1, 3)])
    sheet = _despill(sheet, background, np.median(border, axis=0))
    matte = _central_figure(~background).astype(np.float64)
    ys, xs = np.nonzero(matte > 0.5)
    if not len(ys):
        return (frame, full) if with_mask else frame
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    figure, matte = sheet[top:bottom, left:right], matte[top:bottom, left:right]
    scale = height * rows / figure.shape[0]
    size = (max(1, round(figure.shape[1] * scale)), max(1, round(figure.shape[0] * scale)))
    from PIL import Image
    figure = np.asarray(Image.fromarray((figure * 255).astype(np.uint8)).resize(size, Image.LANCZOS)) / 255
    matte = np.asarray(Image.fromarray((matte * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) / 255
    x0, y0 = round(centre_x * cols - size[0] / 2), round(feet_y * rows - size[1])
    fx0, fy0 = max(0, -x0), max(0, -y0)  # parts of the figure outside the frame are cut off, like the stand-in's
    x1, y1 = min(cols, x0 + size[0]), min(rows, y0 + size[1])
    out = frame.copy()
    if x1 <= max(0, x0) or y1 <= max(0, y0):
        return (out, full) if with_mask else out
    alpha = matte[fy0:fy0 + y1 - max(0, y0), fx0:fx0 + x1 - max(0, x0), None]
    patch = figure[fy0:fy0 + y1 - max(0, y0), fx0:fx0 + x1 - max(0, x0)]
    region = out[max(0, y0):y1, max(0, x0):x1]
    pad = max(4, round(0.25 * size[1]))  # the scene around where they stand (no person in it yet)
    patch = _match_light(patch, alpha[..., 0],
                         out[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].reshape(-1, 3))
    out[max(0, y0):y1, max(0, x0):x1] = region * (1 - alpha) + patch * alpha
    if not with_mask:
        return out
    full[max(0, y0):y1, max(0, x0):x1] = alpha[..., 0] > 0.5
    grown = full > 0
    for _ in range(max(1, round(ring * height * rows))):
        step = grown.copy()
        step[1:] |= grown[:-1]
        step[:-1] |= grown[1:]
        step[:, 1:] |= grown[:, :-1]
        step[:, :-1] |= grown[:, 1:]
        grown = step
    return out, grown.astype(np.float64)


PUNCH_THROUGH_CUT = 2.5  # a frame change this many times the window's median counts as a cut


def punch_through(frames: np.ndarray, start: int, end: int, span: int = 6, bloom: float = 0.35) -> tuple:
    """The Earth zoom's break-out of the cloud deck as a fast bright dissolve. Holding the whiteout, LTX will not
    paint the part-cleared frames between cloud and ground and cuts from one to the other in a frame or two,
    wherever the guides are (Twin Peaks r21-r27, f36->37 on the standard route). The largest frame-to-frame
    change in ``start``..``end`` is taken as that cut, and the ``span`` frames around it are replaced by a
    cross-fade from the last frame before them to the first after, lifted toward white at its middle, as a camera
    bursting through cloud sees. A clip with no change above PUNCH_THROUGH_CUT times the window's median (LTX
    followed the blocking's haze, r28) is left alone. ``frames`` is (n, h, w, c) in 0..1; returns the new frames
    and the cut's index, or None."""
    start, end = max(1, start), min(len(frames) - 1, end)
    if end <= start:
        return frames, None
    sample = frames[max(0, start - 1):end + 1, ::8, ::8, :3].astype(np.float32)
    change = np.abs(np.diff(sample, axis=0)).mean(axis=(1, 2, 3))
    if change.max() < PUNCH_THROUGH_CUT * max(float(np.median(change)), 1e-6):
        return frames, None
    cut = start + int(np.argmax(change))  # frames[cut] differs most from frames[cut - 1]
    first = max(1, cut - span // 2)
    last = min(len(frames) - 2, first + span - 1)
    before, after = frames[first - 1].astype(np.float32), frames[last + 1].astype(np.float32)
    out = frames.copy()
    for i in range(first, last + 1):
        w = (i - first + 1) / (last - first + 2)
        mix = (1 - w) * before + w * after
        lift = bloom * math.sin(math.pi * w)
        mix[..., :3] = mix[..., :3] + (1 - mix[..., :3]) * lift
        out[i] = np.clip(mix, 0, 1).astype(frames.dtype)
    return out, cut
