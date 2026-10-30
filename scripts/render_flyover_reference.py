"""Animate a 15-second clay flyover of Worm's Head with a broad sea turn.

The route starts at the saved Rhossili lookout camera, passes west of the Outer
Head, turns smoothly around it, then returns toward the headland. Coordinates
are British National Grid metres; sea height remains an illustrative setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_landform_heights
BASE = ROOT / "outputs/B_lidar_full_2m"
FPS = 24
FRAMES = 360

# Seconds, easting, northing, height in the unverified LiDAR source frame.
ROUTE_WIDE = [
    (0.0, 240320, 187520, 72),
    (0.8, 240210, 187470, 79),
    (3.0, 239670, 187250, 101),
    (5.4, 238950, 187420, 115),
    (6.8, 238500, 187610, 122),
    (7.8, 238180, 187760, 128),
    (8.6, 237940, 187850, 133),
    (9.3, 237790, 187940, 137),
    (10.0, 237760, 188080, 139),
    (10.7, 237870, 188210, 137),
    (11.4, 238080, 188270, 132),
    (12.2, 238430, 188230, 125),
    (13.2, 239120, 187970, 112),
    (14.0, 239680, 187760, 104),
    (15.0, 240270, 187590, 95),
]
ROUTE_SLOW_RETURN = ROUTE_WIDE[:-3] + [
    (13.2, 238600, 188150, 120),
    (14.0, 238650, 188140, 118),
    (15.0, 238700, 188130, 116),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=BASE / "sea_scene_lookout.blend")
    parser.add_argument("--route", choices=("wide", "slow-return"), default="slow-return")
    parser.add_argument("--output", type=Path, default=BASE / "flyover_reference_v3.blend")
    parser.add_argument("--frames-dir", type=Path, default=BASE / "flyover_v3_frames")
    parser.add_argument("--storyboard", action="store_true", help="Render route checkpoints")
    parser.add_argument("--render", action="store_true", help="Render all 360 animation frames")
    cli = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(cli)


def route_position(seconds: float, route=ROUTE_WIDE) -> Vector:
    seconds = max(route[0][0], min(route[-1][0], seconds))
    index = min(next((i for i in range(len(route) - 1) if seconds <= route[i + 1][0]), len(route) - 2), len(route) - 2)
    t0, *p0 = route[index]
    t1, *p1 = route[index + 1]
    dt = t1 - t0
    u = (seconds - t0) / dt
    before = route[max(0, index - 1)]
    after = route[min(len(route) - 1, index + 2)]
    slope0 = (Vector(p1) - Vector(before[1:])) / (t1 - before[0])
    slope1 = (Vector(after[1:]) - Vector(p0)) / (after[0] - t0)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * Vector(p0) + h10 * dt * slope0 + h01 * Vector(p1) + h11 * dt * slope1


def smoothstep(start: float, end: float, value: float) -> float:
    x = max(0.0, min(1.0, (value - start) / (end - start)))
    return x * x * (3 - 2 * x)


def main() -> None:
    options = parse_args()
    evidence_path = options.scene.with_name("landform_height_evidence.json")
    terrain_path = options.scene.with_name("terrain.glb")
    if not evidence_path.is_file():
        raise ValueError("A measured landmark-height manifest is required for the flyover")
    qualify_landform_heights(json.loads(evidence_path.read_text()), terrain_path)
    route = ROUTE_SLOW_RETURN if options.route == "slow-return" else ROUTE_WIDE
    bpy.ops.wm.open_mainfile(filepath=str(options.scene.resolve()))
    scene = bpy.context.scene
    if Path(scene.get("source_terrain", "")).resolve() != terrain_path.resolve():
        raise ValueError("The Blender scene terrain does not match its measured-height evidence")
    camera = scene.camera
    start_rotation = camera.rotation_euler.to_quaternion()
    camera.rotation_mode = "QUATERNION"
    previous_rotation = None
    terrain = next(o for o in scene.objects if o.name.startswith("LiDAR terrain"))
    sea = next(o for o in scene.objects if o.name.startswith("SEA"))
    metadata_path = options.scene.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    origin_x = float(metadata["origin_easting_m"])
    origin_y = float(metadata["origin_northing_m"])

    # Workbench renders the clay study quickly and keeps the silhouette legible.
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.background_type = "WORLD"
    scene.world.color = (0.49, 0.59, 0.68)
    terrain.active_material.diffuse_color = (0.61, 0.61, 0.55, 1)
    sea.active_material.diffuse_color = (0.07, 0.25, 0.32, 1)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAMES
    camera.data.lens = 38
    camera.data.clip_end = 50000

    for frame in range(1, FRAMES + 1):
        seconds = (frame - 1) / FPS
        p = route_position(seconds, route)
        before = route_position(max(0, seconds - 0.35), route)
        after = route_position(min(15, seconds + 0.35), route)
        tangent = Vector((after.x - before.x, after.y - before.y))
        tangent.normalize()
        right = Vector((tangent.y, -tangent.x))
        heading = (tangent + 0.42 * right).normalized()
        direction = Vector((heading.x, heading.y, -0.065)).normalized()
        # Hold the promontory in frame through the wider turn instead of
        # looking out over empty sea while the flight path bends northward.
        tip_weight = smoothstep(5.0, 7.2, seconds) * (1 - smoothstep(11.3, 13.3, seconds))
        tip = Vector((238900 - p.x, 187750 - p.y, 45 - p.z)).normalized()
        direction = ((1 - tip_weight) * direction + tip_weight * tip).normalized()
        rotation = direction.to_track_quat("-Z", "Y")
        if seconds < 1.0:
            blend = 0.5 - 0.5 * math.cos(math.pi * seconds)
            rotation = start_rotation.slerp(rotation, blend)
        if previous_rotation is not None:
            difference = previous_rotation.rotation_difference(rotation).angle
            maximum_step = math.radians(6.0)
            if difference > maximum_step:
                rotation = previous_rotation.slerp(rotation, maximum_step / difference)
            else:
                rotation = previous_rotation.slerp(rotation, 0.45)
        camera.location = (p.x - origin_x, p.y - origin_y, p.z)
        camera.rotation_quaternion = rotation
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        previous_rotation = rotation.copy()

    scene["route_description"] = (
        "Rhossili lookout -> past Outer Head -> broad 180-degree sea arc -> "
        + ("slower partial Rhossili return" if options.route == "slow-return" else "Rhossili return")
    )
    scene["route_height_status"] = "Visual camera path; sea and terrain vertical datum not certified"
    scene["reference_target"] = "MiniMax H3 Ref2VA, maximum 15 second video reference"
    scene.frame_set(1)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(options.output.resolve()))

    if options.storyboard or options.render:
        options.frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        frames = (1, 72, 144, 168, 192, 216, 240, 264, 288, 312, 360) if options.storyboard else range(1, FRAMES + 1)
        for frame in frames:
            scene.frame_set(frame)
            scene.render.filepath = str((options.frames_dir / f"frame_{frame:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
            if frame == 1 or frame % 24 == 0 or frame == FRAMES:
                print(f"Rendered {frame}/{FRAMES}", flush=True)
    print(f"Saved {options.output}")


if __name__ == "__main__":
    main()
