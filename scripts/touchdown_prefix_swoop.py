"""Video-prefix A/B for the swoop (Astra's review, 2026-10-10): does LTX 2.5 carry the computed dive's motion on
when the swoop is conditioned on the dive's last N frames instead of its last frame only?

Today the swoop is a first/last clip: frame 0 pinned to the dive's last frame, the last frame pinned to the end still.
LTX then starts almost still (flow ~2-4 against the dive's ~20) and _match_speed drops frames to hide the brake.
Here frame 0 is pinned to a short clip instead (LTXVAddGuide takes an 8n+1-frame image batch), so the model sees
the incoming velocity and pitch rate. Everything else (prompt, seed, length, end still) is the same, so the only
difference between the runs is the prefix length.

For each prefix length it writes the swoop, a joined preview (the dive minus its last N frames, then the swoop, whose
first N frames re-render that prefix) and prints optical flow across the join.

    uv run python scripts/touchdown_prefix_swoop.py --dive tiltdive_v3.mp4 --end-still ground_end.png \
        --prefix 1,17 --seed 42 --frames 121 --output-dir outputs/runs/touchdown/prefix-ab
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakeout_earth_zoom as f  # noqa: E402

# tswoop.py's prompt (swoop-fix B), with the person and place as arguments
PROMPT = ("One continuous shot with no cuts. The camera is already swooping forward and down over {approach}, about a "
          "hundred metres up. It keeps tilting up toward the horizon in one smooth, quick, steady move as it "
          "descends, levels out at eye level, and glides toward {person} standing {scene}, who stays still and looks "
          "at the camera. Steady camera, smooth continuous motion, real aerial photography, natural colour, sharp "
          "photographic detail, photorealistic.")
APPROACH = ("the grassy summit of Twin Peaks in San Francisco, looking forward and down at the winding two-lane road "
            "with a yellow centre line that crosses a rounded hilltop of dry grass and low green scrub")
SCENE = "by a large oak tree with the downtown skyline and the bay behind"


def _flow(path: str, lo: int, hi: int) -> list[float]:
    import cv2
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='between(n\\,{lo}\\,{hi})',"
                          "scale=480:272,format=gray", "-vsync", "0", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    g = np.frombuffer(raw, np.uint8).reshape(-1, 272, 480)
    return [round(float(np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0),
                                       axis=2).mean()), 1) for a, b in zip(g[:-1], g[1:])]


def _count(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return int(out.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dive", type=Path, required=True, help="the computed dive (its last frames become the prefix)")
    p.add_argument("--end-still", type=Path, required=True)
    p.add_argument("--prefix", default="1,17", help="comma-separated prefix lengths, each 8n+1 (1 = today's swoop)")
    p.add_argument("--frames", type=int, default=121)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--person", default=f.PERSON)
    p.add_argument("--scene", default=SCENE)
    p.add_argument("--approach", default=APPROACH)
    p.add_argument("--prompt", help="override the whole prompt")
    p.add_argument("--url", default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/runs/touchdown/prefix-ab"))
    a = p.parse_args()
    lengths = [int(x) for x in a.prefix.split(",")]
    if any((n - 1) % 8 for n in lengths) or a.frames % 8 != 1:
        raise SystemExit("prefix lengths and --frames must be 8n+1")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = a.prompt or PROMPT.format(person=a.person, scene=a.scene, approach=a.approach)
    total = _count(a.dive)
    for n in lengths:
        tag = f"p{n}_s{a.seed}"
        prefix = a.output_dir / f"prefix_{n}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.dive), "-vf",
                        f"select='gte(n\\,{total - n})',setpts=N/24/TB", "-vsync", "0", "-c:v", "libx264", "-crf", "12",
                        "-pix_fmt", "yuv420p", str(prefix)], check=True)
        args = SimpleNamespace(no_upscale=False, output_dir=a.output_dir / tag, url=a.url)
        started = time.monotonic()
        with f._client(a.url) as client:
            graph: dict = {}
            if n == 1:
                first = a.output_dir / "prefix_1.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(prefix), "-frames:v", "1", str(first)], check=True)
                start = f._load(graph, client, "44", first)
            else:
                graph["47"] = {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(prefix))}}
                graph["48"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["47", 0]}}
                start = ["48", 0]
            pins = [(0, start), (-1, f._load(graph, client, "45", a.end_still))]
            saved = f._submit(client, graph, tuple(f._ltx(graph, args, a.frames, pins, prompt, a.seed,
                                                          f"geoblock/touchdown_prefix_{tag}")), args.output_dir)
        swoop = saved[f.LTX_VIDEO]
        joined = str(a.output_dir / f"joined_{tag}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.dive), "-i", swoop, "-filter_complex",
                        f"[0:v]trim=end_frame={total - n},setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];"
                        "[a][b]concat=n=2:v=1[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "14", "-pix_fmt",
                        "yuv420p", joined], check=True)
        j = total - 1  # last frame of the dive's own motion in the joined clip (n=1) / last prefix frame (n>1)
        print(f"prefix {n}: {time.monotonic() - started:.0f} s, swoop {swoop}")
        try:
            import cv2  # noqa: F401  (optional: the project venv has no OpenCV unless installed with --extra touchdown)
        except ImportError:
            print("  (no OpenCV here: install with --extra touchdown to print flow across the join)")
            continue
        print(f"  flow, dive's last 8 frames:  {_flow(str(a.dive), total - 9, total - 1)}")
        print(f"  flow across the join (from joined f{j - 4}): {_flow(joined, j - 4, j + 30)}")


if __name__ == "__main__":
    main()
