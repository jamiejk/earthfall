"""Render a measured-terrain guide with LTX 2.5 Layout To Render.

    python scripts/touchdown/terrain_v2v.py --guide terrain/guide.mp4 \
      --first terrain/guide_frames/0001.png --last compatible_end.png \
      --output outputs/.../ltx --seed 7

The first image is an appearance reference and should be a photographic version
of the guide's starting view. For a clay guide, do not supply the raw clay image
or an unrelated eye-level endpoint as this reference.

The last still must be painted from the guide's own eye-level final view.
An unrelated attractive still is not a usable endpoint: in the experiment,
the guide dominated that pin and the person never appeared.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geoblock_core.workflow import LTX_VIDEO, ShotRequest, _add_ltx25
import fakeout_earth_zoom as fakeout


def render(guide: Path, first: Path, last: Path, output: Path, seed: int, frames: int,
           prompt: str, url: str, ltx_upscale: bool = False, mids: list[tuple[int, Path]] | None = None) -> Path:
    """LTX 2.5 Layout To Render over the measured guide.

    ``mids`` are extra pinned frames: ``(frame index, image)``. A photographic frame pinned in the middle of the
    swoop gives the model something to copy but the shaded guide, which is where the "3D terrain model" look and
    the invented mid-ground came from.
    """
    output.mkdir(parents=True, exist_ok=True)
    with fakeout._client(url) as client:
        graph = {
            "50": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(guide))}},
            "60": {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}},
            "40": {"class_type": "LoadImage", "inputs": {"image": client.assets.from_file(str(first))}},
            "41": {"class_type": "LTXVPreprocess", "inputs": {"image": ["40", 0], "img_compression": 18}},
            "42": {"class_type": "LoadImage", "inputs": {"image": client.assets.from_file(str(last))}},
            "43": {"class_type": "LTXVPreprocess", "inputs": {"image": ["42", 0], "img_compression": 18}},
        }
        mid_refs = []
        for k, (index, image) in enumerate(mids or []):
            load = f"m{k}"
            pre = f"m{k}p"
            graph[load] = {"class_type": "LoadImage", "inputs": {"image": client.assets.from_file(str(image))}}
            graph[pre] = {"class_type": "LTXVPreprocess", "inputs": {"image": [load, 0], "img_compression": 18}}
            mid_refs.append((index, [pre, 0]))
        request = ShotRequest(name="terrain-v2v", output_prefix="geoblock/touchdown_terrain_v2v",
                              seed=seed, ltx_upscale=ltx_upscale, pin_last_keyframe=True)
        _add_ltx25(graph, request, frames, ["41", 0], ["43", 0], mid_refs, prompt, layout=True)
        saved = fakeout._submit(client, graph, (LTX_VIDEO,), output)
    return Path(saved[LTX_VIDEO])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guide", type=Path, required=True)
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--last", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--url", default="http://127.0.0.1:8191")
    p.add_argument("--prompt", default=(
        "One continuous forward descent following the exact straight camera path of the guide video. "
        "The road, hills and distant skyline remain in their measured positions. The camera levels out "
        "toward a person standing at the landing and eases into an eye-level view. Natural daylight, "
        "realistic depth and parallax, photorealistic."))
    a = p.parse_args()
    print(render(a.guide, a.first, a.last, a.output, a.seed, a.frames, a.prompt, a.url))


if __name__ == "__main__":
    main()
