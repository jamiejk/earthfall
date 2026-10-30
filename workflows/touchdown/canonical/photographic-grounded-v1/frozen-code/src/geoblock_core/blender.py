"""Launch the headless Blender renderer for a planned camera route."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable

from .weather import DEFAULT_WEATHER, WEATHER

RENDER_SCRIPT = Path(__file__).with_name("blender_render.py")
ENGINES = ("eevee", "workbench", "cycles")
PALETTES = ("default", "grand-canyon")
TREE_SOURCES = ("off", "auto", "lidar", "meta")


def find_blender() -> list[str]:
    """Command prefix for Blender: GEOBLOCK_BLENDER, else ``blender`` on PATH.

    GEOBLOCK_BLENDER may name a Blender executable or a Python interpreter that
    has the ``bpy`` module installed (useful on servers without a Blender build).
    """
    configured = os.environ.get("GEOBLOCK_BLENDER", "").strip()
    executable = configured or shutil.which("blender")
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(
            "Blender not found. Install Blender 4.2+ and put it on PATH, or set GEOBLOCK_BLENDER "
            "to the Blender executable (or to a Python with the bpy module).")
    if Path(executable).name.startswith("python"):
        return [executable, str(RENDER_SCRIPT)]
    return [executable, "-b", "--factory-startup", "--python", str(RENDER_SCRIPT)]


def render_route(scene: dict, route: dict, output_root: Path, *, engine: str = "eevee",
                 width: int = 1280, height: int = 720, samples: int = 16, frame_step: int = 1,
                 depth: bool = False, palette: str = "default", weather: str = DEFAULT_WEATHER,
                 imagery: Path | None = None, trees: Path | None = None,
                 progress: Callable[[int, int], None] | None = None) -> dict:
    """Render a route to ``control.mp4``; reuse an identical earlier render."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown render engine {engine!r}")
    if palette not in PALETTES:
        raise ValueError(f"Unknown palette {palette!r}")
    if weather not in WEATHER:
        raise ValueError(f"Unknown weather {weather!r}")
    settings = {"route": route["route_sha256"], "engine": engine, "width": width, "height": height,
                "samples": samples, "frame_step": frame_step, "depth": depth, "palette": palette, "weather": weather,
                "imagery": hashlib.sha256(imagery.read_bytes()).hexdigest() if imagery else None,
                **({"trees": hashlib.sha256(trees.read_bytes()).hexdigest(),  # the proxy shapes live in trees.py
                    "tree_code": hashlib.sha256(RENDER_SCRIPT.with_name("trees.py").read_bytes()).hexdigest()}
                   if trees else {}),
                "script": hashlib.sha256(RENDER_SCRIPT.read_bytes()).hexdigest(),
                "weather_values": json.loads(json.dumps(WEATHER[weather]))}  # editing a preset re-renders
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    output_dir = output_root / scene["project_id"] / f"{route['preset'].lower().replace(' ', '-')}-{key}"
    manifest_path = output_dir / "render.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("settings") == settings and manifest.get("video") and Path(manifest["video"]).is_file():
            return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scene.json"
    route_path = output_dir / "route.json"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    route_path.write_text(json.dumps(route) + "\n")
    command = find_blender() + [
        "--", "--scene", str(scene_path), "--route", str(route_path), "--output-dir", str(output_dir),
        "--engine", engine, "--width", str(width), "--height", str(height),
        "--samples", str(samples), "--frame-step", str(frame_step), "--palette", palette, "--weather", weather]
    command += (["--depth"] if depth else []) + (["--imagery", str(imagery)] if imagery else [])
    command += ["--trees", str(trees)] if trees else []
    log_lines: list[str] = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            log_lines.append(line)
            if line.startswith("GEOBLOCK_PROGRESS") and progress:
                done, total = line.split()[1].split("/")
                progress(int(done), int(total))
    (output_dir / "blender.log").write_text("".join(log_lines))
    if process.returncode != 0:
        tail = "".join(log_lines[-25:])
        raise RuntimeError(f"Blender render failed (exit {process.returncode}); see {output_dir / 'blender.log'}\n{tail}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("video") or (depth and not manifest.get("depth_video")):
        raise RuntimeError("Blender rendered frames but ffmpeg was not found to encode control.mp4.")
    manifest["settings"] = settings
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
