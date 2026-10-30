"""Render a few Worm's Head flyover checkpoints with forge3d 1.40.1.

Run with ``uv run --no-sync --with forge3d==1.40.1 python
scripts/try_forge3d_worms_head.py``. This is a renderer comparison, not an H3
blocking control: the illustrative sea is made by filling DEM no-data cells.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

import forge3d as f3d


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/B_lidar_full_2m"
DEFAULT_FRAMES = (1, 144, 216, 288)


def blender_camera_poses(blend_path: Path, frames: tuple[int, ...]) -> list[dict]:
    script = """
import bpy, json
from mathutils import Vector
scene = bpy.context.scene
rows = []
for frame in FRAMES:
    scene.frame_set(frame)
    camera = scene.camera
    rows.append({
        "frame": frame,
        "eye": list(camera.matrix_world.translation),
        "direction": list(camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))),
    })
print("GEOBLOCK_CAMERA_JSON " + json.dumps(rows))
""".replace("FRAMES", repr(frames))
    result = subprocess.run(
        ["blender", "-b", str(blend_path), "--python-expr", script],
        text=True,
        capture_output=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("GEOBLOCK_CAMERA_JSON "):
            return json.loads(line.removeprefix("GEOBLOCK_CAMERA_JSON "))
    raise RuntimeError(f"Blender did not return camera poses: {result.stdout}\n{result.stderr}")


def forge_camera(pose: dict, target_distance_m: float = 1000.0) -> f3d.OrbitCamera:
    eye = np.asarray(pose["eye"], dtype=np.float64)
    direction = np.asarray(pose["direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    target = eye + target_distance_m * direction
    back = -direction
    return f3d.OrbitCamera(
        target=tuple(float(value) for value in target),
        distance=target_distance_m,
        # forge3d mesh:zup uses azimuth from +east and a polar angle from +up.
        azimuth_deg=math.degrees(math.atan2(back[1], back[0])),
        elevation_deg=math.degrees(math.acos(np.clip(back[2], -1.0, 1.0))),
        fov_deg=30.0,  # Blender 38 mm lens, 36 mm sensor, 16:9 vertical FOV.
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--grid-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--sea-fill-height", type=float, default=-5.0)
    parser.add_argument("--output-dir", type=Path, default=BASE / "forge3d_probe")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = tuple(args.frames)
    if any(frame < 1 or frame > 360 for frame in frames):
        parser.error("frames must be between 1 and 360")

    sys.path.insert(0, str(ROOT / "src"))
    from geoblock_core.height_policy import qualify_landform_heights

    evidence = json.loads((BASE / "landform_height_evidence.json").read_text())
    qualify_landform_heights(evidence, BASE / "terrain.glb")

    with rasterio.open(BASE / "terrain_metric.tif") as source:
        masked = source.read(
            1,
            out_shape=(args.grid_size, args.grid_size),
            resampling=Resampling.nearest,
            masked=True,
        )
        valid = ~np.ma.getmaskarray(masked)
        terrain = np.ascontiguousarray(masked.filled(args.sea_fill_height), dtype=np.float32)
        resolution = [source.width * source.res[0] / args.grid_size,
                      source.height * source.res[1] / args.grid_size]
    if valid.sum() == 0:
        raise ValueError("No measured land heights in the selected raster")
    if args.sea_fill_height >= float(masked.min()):
        raise ValueError("Sea fill must sit below the lowest measured land height")
    water_mask_path = args.output_dir / "water_mask.npy"
    np.save(water_mask_path, np.ascontiguousarray(~valid, dtype=np.float32))

    poses = blender_camera_poses(BASE / "flyover_reference_v3.blend", frames)
    results = []
    for pose in poses:
        frame = pose["frame"]
        output_path = args.output_dir / f"forge3d_frame_{frame:04d}.png"
        scene = f3d.MapScene(
            terrain=f3d.TerrainSource(
                data=terrain,
                crs="EPSG:27700",
                metadata={
                    "width": args.grid_size,
                    "height": args.grid_size,
                    "resolution": resolution,
                    "source_id": "Welsh-Government-LiDAR-DTM",
                },
                elevation_sampling_available=True,
            ),
            camera=forge_camera(pose),
            lighting=f3d.LightingPreset(
                name="daylight",
                sun_direction=(0.4, 0.7, 0.4),
                intensity=1.0,
                settings={
                    "cli_params": {"camera_mode": "mesh:zup"},
                    "hue_variation_strength": 0.0,
                    "sky": {"enabled": True, "model": "approximate", "aerial_perspective": False},
                    "water": {"enabled": True, "mask_path": str(water_mask_path), "level": 0.0},
                },
            ),
            output=f3d.OutputSpec(width=args.width, height=args.height, path=str(output_path)),
        )
        report = scene.validate()
        if report.render_blocked(scene.render_policy):
            raise RuntimeError(f"forge3d rejected frame {frame}: {report.diagnostics}")
        started = time.perf_counter()
        render = scene.render()
        elapsed = time.perf_counter() - started
        if render.status != "ok":
            raise RuntimeError(f"forge3d frame {frame} status: {render.status}")
        results.append({"frame": frame, "image": str(output_path), "render_s": round(elapsed, 3)})
        print(f"Rendered frame {frame}: {output_path} ({elapsed:.1f} s)", flush=True)

    manifest = {
        "forge3d_version": getattr(f3d, "__version__", "unknown"),
        "source_raster": str(BASE / "terrain_metric.tif"),
        "source_blender_camera": str(BASE / "flyover_reference_v3.blend"),
        "grid_size": args.grid_size,
        "valid_land_fraction": round(float(valid.mean()), 6),
        "sea_fill_height_m_in_unverified_lidar_frame": args.sea_fill_height,
        "status": "exploratory renderer comparison; water fill and vertical datum are not verified",
        "renders": results,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
