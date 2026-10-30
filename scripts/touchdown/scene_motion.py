"""Measure movement inside the scene of a portrait clip, with the camera move taken out.

Each pair of frames is aligned with a RANSAC homography fitted to features tracked across
the whole frame (the camera move and the dominant scene plane). Dense optical flow between
the aligned frames is then what is left: parallax at other depths plus anything that moves
by itself. Parallax in a small region is close to one shift for the whole region, so each
region reports its *internal* motion: the mean distance of its residual flow from the
region's own median flow. Wind in grass, ripples on water and fluttering cloth show up
there; a photograph pushed in by the camera stays near zero.

Regions are boxes given as fractions of the first frame and carried through the clip by
the accumulated homographies; a region is skipped once less than a third of it is in view.
Prints JSON: per region, the mean internal motion in pixels per frame at 640 px width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640


def frames(path: Path, width: int = WIDTH):
    capture = cv2.VideoCapture(str(path))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        yield cv2.cvtColor(cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2GRAY)


def homography(a, b):
    points = cv2.goodFeaturesToTrack(a, 1500, 0.005, 6)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(21, 21), maxLevel=4)
    good = status.ravel() == 1
    h, _ = cv2.findHomography(points[good], moved[good], cv2.RANSAC, 1.0)
    return h if h is not None else np.eye(3)


def measure(path: Path, regions: dict, start: int = 0, stop: int | None = None) -> dict:
    clip = list(frames(path))[start:stop]
    height, width = clip[0].shape
    corners = {name: np.float32([[x0 * width, y0 * height], [x1 * width, y0 * height], [x1 * width, y1 * height],
                                 [x0 * width, y1 * height]]).reshape(-1, 1, 2)
               for name, (x0, y0, x1, y1) in regions.items()}
    totals = {name: [] for name in regions}
    camera = []
    for a, b in zip(clip, clip[1:]):
        h = homography(a, b)
        aligned = cv2.warpPerspective(a, h, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.ones_like(a), h, (width, height), flags=cv2.INTER_NEAREST) > 0
        flow = cv2.calcOpticalFlowFarneback(aligned, b, None, 0.5, 4, 15, 3, 5, 1.2, 0)
        grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), -1).reshape(-1, 1, 2).astype(np.float32)
        camera.append(float(np.linalg.norm(cv2.perspectiveTransform(grid[::997], h) - grid[::997], axis=-1).mean()))
        for name in regions:
            corners[name] = cv2.perspectiveTransform(corners[name], h)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [corners[name].astype(np.int32)], 1)
            area = abs(cv2.contourArea(corners[name]))
            inside = (mask > 0) & valid
            if area == 0 or inside.sum() < area / 3 or inside.sum() < 200:
                totals[name].append(None)
                continue
            vectors = flow[inside]
            totals[name].append(float(np.linalg.norm(vectors - np.median(vectors, axis=0), axis=1).mean()))
    report = {name: dict(internal_motion=round(float(np.mean([v for v in values if v is not None])), 3)
                         if any(v is not None for v in values) else None,
                         frames_in_view=sum(v is not None for v in values),
                         per_frame=[None if v is None else round(v, 3) for v in values])
              for name, values in totals.items()}
    return dict(video=str(path), frames=len(clip), camera_motion=round(float(np.mean(camera)), 3), regions=report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--regions", type=json.loads, required=True,
                   help='JSON {"name": [x0, y0, x1, y1], ...} as fractions of the first frame')
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--per-frame", action="store_true")
    a = p.parse_args()
    for video in a.videos:
        result = measure(video, a.regions, a.start, a.stop)
        if not a.per_frame:
            for region in result["regions"].values():
                region.pop("per_frame")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
