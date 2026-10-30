"""Ending-only test: re-generate a build's near portrait from its own arrival still and seed.

Variant "current" keeps the live conditioning (a frame-0 LTXVAddGuide at strength 1.0 in
both sampling stages). Variant "native" swaps that guide for the conditioning in Comfy's
LTX 2.5 image-to-video template: LTXVPreprocess (compression 18) into
LTXVImgToVideoInplace at 0.7 for the first stage and 1.0 for the refine. Everything else
(model, sigmas, seed, size, frame count) is unchanged, so the conditioning is the only
difference between the two. Both stages are saved: *_stage1_* is the base resolution,
*_ltx25_* the refined clip.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROMPT = (
    "The camera settles into a steady three-quarter portrait with both knees visible. A gust bends the "
    "foreground grasses and flower stems sideways, then they spring partly back. The same breeze lifts loose "
    "hair and flutters the coat hem as the character gives a small wave. The village buildings retain their "
    "shape and position throughout."
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True, help="Studio job folder (outputs/touchdown/studio/studio-...)")
    p.add_argument("--variant", choices=["current", "native"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--prompt-file", type=Path, help="Read the prompt from this file instead")
    p.add_argument("--name", help="Output subfolder (default: the variant)")
    a = p.parse_args()
    if a.prompt_file:
        a.prompt = a.prompt_file.read_text().strip()
    import fakeout_earth_zoom as f
    from touchdown import studio_build as s
    from touchdown.portrait_flight import native
    s.REPO = Path(__file__).resolve().parents[2]
    cfg = json.loads((a.job / "prepared/scene-config.json").read_text())
    scene = a.job / "prepared/scene"
    before = json.loads((a.job / "prepared/motion/near/provenance.json").read_text())
    frames, width, height, seed = int(cfg["video"].get("near_frames", 49)), 1280, 736, cfg["seed"] + 31
    if (frames, width, height, seed) != (before["frames"], before["base_width"], before["base_height"], before["seed"]):
        raise ValueError("Settings differ from the build's own near portrait")
    paths = (a.job / "flight/pinned-model-paths.json").resolve()
    command = os.environ["TOUCHDOWN_COMFY_SERVER_CMD"]
    if "--extra-model-paths-config" not in shlex.split(command):
        os.environ["TOUCHDOWN_COMFY_SERVER_CMD"] = (command + " --deterministic --extra-model-paths-config "
                                                    + shlex.quote(str(paths)))
    folder = a.output / (a.name or a.variant)
    folder.mkdir(parents=True, exist_ok=True)
    with s.runtime("comfy", folder), f._client(s.VIDEO_URL) as client:
        graph = {}
        image = f._load(graph, client, "44", scene / "arrival.png")
        wanted = f._ltx(graph, SimpleNamespace(no_upscale=False), frames, [(0, image)], a.prompt, seed,
                        f"touchdown/ending_{a.variant}")
        graph["308"]["inputs"].update(width=width, height=height)
        if a.variant == "native":
            native(graph, image)
        (folder / "provenance.json").write_text(json.dumps(dict(
            job=str(a.job), variant=a.variant, frames=frames, width=width, height=height, seed=seed,
            prompt=a.prompt, original_near=json.loads((a.job / "prepared/motion/near/result.json").read_text())["video"],
        ), indent=2))
        saved = f._submit(client, graph, tuple(wanted), folder)
    (folder / "result.json").write_text(json.dumps(dict(refined=saved[f.LTX_VIDEO], base=saved[f.STAGE1_VIDEO]),
                                                    indent=2))
    print(json.dumps(saved), flush=True)


if __name__ == "__main__":
    main()
