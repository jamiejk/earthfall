"""Build a Touchdown clip for a place, and put it where the builds live.

    python scripts/build_place.py canyonlands-grand-view
    python scripts/build_place.py horseshoe-bend --ending close --name horseshoe-bend

Where things go, so there is one answer to "where is the latest test run":

    outputs/touchdown/<name>/          the build: every stage, the manifest, the logs, and the final clip
    outputs/touchdown/<name>/sound/<key>/touchdown.mp4    the full-resolution clip
    app/demo-clips/                   the 720p copies the app's Examples panel plays (tracked in git)

The build itself writes into the repository's ``outputs/`` (which git ignores), so a run is never lost to a
temporary directory, and the gallery is rebuilt from it with ``scripts/make_demo_gallery.py``.

It needs the ComfyUI the build uses to be reachable at ``--url`` (the Comfy API proxy, by default), and on a 24 GB
card it needs ``TOUCHDOWN_COMFY_SERVER_CMD`` so the build can restart ComfyUI between the stills and the video
stages: the FLUX 3 stills and the LTX 2.5 transformer cannot both be resident, and ComfyUI does not release the
stills' memory when it is asked to unload it. See ``app/README.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs" / "touchdown"

# The closer ending: the figure at six metres, from the knees up, looking out at the view, and the ground asked
# for explicitly (a figure that close is where the model invents a patch of coloured sand under them).
CLOSE_ENDING = re.compile(
    r"In the center, about (?:ten|six|four|three) metres from the camera[^.]*\."
    r"(?:[^.]*\.){0,3}")


def end_prompt(text: str) -> str:
    """Ask for the closer ending, whatever the template's wording was."""
    text = re.sub(
        r"In the center, about \w+ metres from the camera[^.]*\.[^.]*\.[^.]*\.",
        "In the center, about six metres from the camera on the ground, add {person}, seen from the knees up, "
        "their head about a quarter of the height of the frame, turned three-quarters away from the camera to look "
        "out across the view. They are not posing for the camera. The ground under and around them is the same as "
        "the rest of the foreground: no circle, no patch, no disc of red sand, no ring of darker sand under them.",
        text, count=1)
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("place", help="a catalogue place id, or 'custom' with --lat/--lon")
    p.add_argument("--name", help="the name of the run (the folder under outputs/touchdown)")
    p.add_argument("--person", default="a person in everyday outdoor clothes")
    p.add_argument("--subject", default=str(ROOT / "shots" / "subjects" / "test-subject-red-coat.jpg"))
    p.add_argument("--ending", choices=("close", "full"), default="close",
                   help="close: the figure at six metres, from the knees up (the default)")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--dry-run", action="store_true", help="write the manifest and stop")
    args = p.parse_args()

    name = args.name or args.place
    run = OUT / name
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / "manifest.toml"

    cmd = [sys.executable, "-m", "touchdown.manifest_gen", args.place,
           "--subject", args.subject, "--person", args.person, "--shot-name", name,
           "--cache", str(run / "cache"), "-o", str(manifest)]
    if args.place == "custom":
        cmd += ["--lat", str(args.lat), "--lon", str(args.lon)]
    env = {**os.environ, "PYTHONPATH": f"{SCRIPTS}{os.pathsep}{ROOT / 'src'}"}
    done = subprocess.run(cmd, cwd=SCRIPTS, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout + done.stderr, file=sys.stderr)
        return done.returncode

    text = manifest.read_text()
    if args.ending == "close":
        text = end_prompt(text)
    text = text.replace('flux3_url = "http://127.0.0.1:8193"',
                        'flux3_url = "http://127.0.0.1:8193"')
    text = re.sub(r"\[face\]\s*\n\s*enabled = true", "[face]\nenabled = false", text)
    text = text.replace('output = "outputs/touchdown"', f'output = "{run}"')
    manifest.write_text(text)
    print(f"manifest: {manifest}")
    if args.dry_run:
        return 0

    build = [sys.executable, "-m", "touchdown.build", str(manifest), "--descent", "45", "--url",
             os.environ.get("TOUCHDOWN_URL", "http://127.0.0.1:8193")]
    print(" ".join(build), flush=True)
    return subprocess.run(build, cwd=SCRIPTS, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
