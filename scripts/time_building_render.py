"""Time the Earth zoom blocking with and without parts of the building blocks, to find what makes EEVEE slow.

Re-renders a few frames of an existing Earth zoom (its zoom.json) once per variant and prints the wall time
of each: as is, blocks casting no shadows, n-gon roofs triangulated up front, both, and no buildings at all.
Each run includes scene setup (terrain, globe, ray casts), which is the same for all but the last.
``--variants clouds`` times the cloud deck instead: shadows on, dithered instead of blended, fewer sheets, none.

    uv run python scripts/time_building_render.py outputs/geoblock/<scene>/earth-zoom-<key>/zoom.json \\
        [--frames 120:123] [--width 1920 --height 1088] [--samples 16]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoblock_core.blender import find_blender  # noqa: E402

ZOOM_SCRIPT = ROOT / "src/geoblock_core/blender_earth_zoom.py"
VARIANTS = {
    "as is": {},
    "no block shadows": {"GEOBLOCK_BUILDING_SHADOWS": "0"},
    "triangulated roofs": {"GEOBLOCK_BUILDING_ROOFS": "tri"},
    "both": {"GEOBLOCK_BUILDING_SHADOWS": "0", "GEOBLOCK_BUILDING_ROOFS": "tri"},
    "no buildings": None,
}
# The cloud deck the camera dives through: 13 blended sheets 500 m apart, casting no shadows, by default.
CLOUD_VARIANTS = {
    "as is": {},
    "cloud shadows": {"GEOBLOCK_CLOUD_SHADOWS": "1"},
    "dithered clouds": {"GEOBLOCK_CLOUD_DITHER": "1"},
    "5 sheets": {"GEOBLOCK_CLOUD_SPACING": "1500"},
    "no cloud deck": {"GEOBLOCK_CLOUD_SPACING": "0"},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zoom", type=Path)
    parser.add_argument("--frames", default="120:123")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--engine", default="eevee")
    parser.add_argument("--variants", choices=("buildings", "clouds"), default="buildings",
                        help="which A/B set to time; for clouds use frames inside the deck, e.g. --frames 30:32")
    args = parser.parse_args(argv)
    zoom = json.loads(args.zoom.read_text())
    command = find_blender()
    command[command.index(str(ZOOM_SCRIPT.with_name("blender_render.py")))] = str(ZOOM_SCRIPT)
    results = []
    with tempfile.TemporaryDirectory() as temp:
        for name, env in (CLOUD_VARIANTS if args.variants == "clouds" else VARIANTS).items():
            out = Path(temp) / name.replace(" ", "-")
            out.mkdir()
            zoom_path = args.zoom
            if env is None:
                zoom_path = out / "zoom.json"
                zoom_path.write_text(json.dumps({k: v for k, v in zoom.items() if k != "buildings"}))
            started = time.time()
            done = subprocess.run(command + ["--", "--zoom", str(zoom_path), "--output-dir", str(out),
                                             "--width", str(args.width), "--height", str(args.height),
                                             "--samples", str(args.samples), "--frame-range", args.frames,
                                             "--engine", args.engine],
                                  env={**os.environ, **(env or {})}, capture_output=True, text=True)
            seconds = time.time() - started
            status = "ok" if done.returncode == 0 else f"failed: {(done.stdout + done.stderr)[-400:]}"
            results.append((name, seconds, status))
            print(f"{name:<20} {seconds:7.0f} s  {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
