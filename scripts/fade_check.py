"""Does a clip fly or dissolve? Per window of frames: mean optical flow (px/frame at 480 px wide), mean frame
difference, and the share of that difference the flow explains. A camera move has flow and a high "explained"; a
cross-fade changes pixels without flow (explained near 0). Whiteout frames have no texture, so they are skipped.

  uv run --with opencv-python-headless python scripts/fade_check.py clip.mp4 [--window 12]

Found on the high-merge chain (9 Oct): f0-119 flow ~0.1 px/frame, explained ~0 = orbit fading to white and the
city fading in, not a flown dive; r3-cloud's dive had 3-12 px/frame.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def measure(path: str) -> list[tuple[float, float, float, float]]:
    """(flow, diff, explained, texture) for each frame pair."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frames.append(cv2.cvtColor(cv2.resize(frame, (480, round(480 * h / w))), cv2.COLOR_BGR2GRAY).astype(np.float32))
    rows = []
    for a, b in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        gx, gy = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
        warped = cv2.remap(b, (gx + flow[..., 0]).astype(np.float32), (gy + flow[..., 1]).astype(np.float32),
                           cv2.INTER_LINEAR)
        diff = float(np.abs(b - a).mean())
        residual = float(np.abs(warped - a)[8:-8, 8:-8].mean())
        rows.append((float(np.linalg.norm(flow, axis=2).mean()), diff, 1 - residual / max(diff, 1e-6),
                     float(cv2.Laplacian(a, cv2.CV_32F).var())))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--window", type=int, default=12)
    args = parser.parse_args()
    rows = np.array(measure(args.clip))
    print(f"{args.clip}: {len(rows) + 1} frames\n frames     flow  diff  explained  texture  verdict")
    for start in range(0, len(rows), args.window):
        flow, diff, explained, texture = rows[start:start + args.window].mean(axis=0)
        verdict = ("whiteout" if texture < 100 else "DISSOLVE?" if diff > 1.5 and explained < 0.3
                   else "still" if diff <= 1.5 else "moves")
        print(f"{start:4d}-{start + args.window:<4d} {flow:6.2f} {diff:5.2f} {explained:9.2f} {texture:8.0f}  {verdict}")


if __name__ == "__main__":
    main()
