"""Landmark surface stability: does the subject hold still between frames?

A photographic landmark is supposed to be masonry, rock or stone that the camera moves over. LTX
redraws a little of it every frame, which reads as bending, rocking or reshuffling. This module
measures that, in a form that means the same thing in every scene and at every resolution.

THE TRAP THIS MODULE EXISTS TO AVOID. The obvious metric -- the detail left after dense flow
removes the camera -- is proportional to how much detail the clip carries. Measured across the
Stonehenge takes it tracks detail energy at r = 0.999, so a sharper take always scores worse and
the 2x refine looked like it caused the warble when it had only made the stones visible. Two
independent corrections follow, and both are mandatory:

  * report ``redraw`` -- the residual divided by the clip's own detail energy. This is the only
    figure comparable between clips.
  * report ``control`` -- the same measurement run on a synthetic pan of one real frame from the
    clip, which cannot warp at all. It is the floor: interpolation and flow error. On Stonehenge
    that floor is 2.50 against a delivered 4.56, so over half of the raw number was never the
    landmark moving.

A scene passes on ``redraw``, not on the residual. Nothing here judges geography, camera motion,
identity or whether the shot is any good.

WHAT THIS CANNOT SEE. Dense flow is deliberately allowed to explain as much as it can, so the metric
catches detail that is REPLACED, not detail that moves. A stone that bends or rocks smoothly is
partly absorbed into the flow field and scores lower than the eye would; a stone whose surface is
repainted scores high. That asymmetry is the point -- repainting is what LTX does -- but it means a
low reading is evidence of stability, not proof of it. Look at the frames too.

usage:
  surface_stability.py CLIP [--window START:END] [--roi Y0,Y1,X0,X1] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SAMPLE_WIDTH = 960      # every clip is measured at one width, so detail means the same thing
SIGMA = 1.2             # high-pass radius, in sample pixels: surface relief, not silhouette
# Calibrated against a judgement, not a preference. With camera alignment working, the delivered
# Stonehenge take Jamie rejected for deforming stones measures 0.288 and every ordinary take sits
# between 0.255 and 0.318, while the one configuration that did hold the surface (pin 0.90) measures
# 0.228. A ceiling between them passes what he could accept and fails what he would not. Recalibrate
# for a new scene on its first accepted take: override with --max.
REDRAW_MAX = 0.25
# Pure flow sanity. Real footage on this pipeline measures a 0.47 floor; a synthetic noise field
# measured 1.07, so past this the number is about the content, not the landmark and cannot be read.
CONTROL_MAX = 0.75
REPEAT_FRACTION = 0.12  # a pair below this share of the median difference is a held frame


def highpass(gray: np.ndarray) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), SIGMA)


def roi_box(shape: tuple[int, int], roi: tuple[float, float, float, float] | None) -> tuple[int, int, int, int]:
    """Pixel bounds of the measured region: the whole frame unless a fraction is supplied."""
    h, w = shape
    if roi is None:
        return 0, h, 0, w
    y0, y1, x0, x1 = roi
    return int(h * y0), int(h * y1), int(w * x0), int(w * x1)


def _warp(prev: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Resample ``prev`` so it lands on ``cur``.

    SUBTRACT the flow. Farneback here reports the displacement that carries a pixel of ``prev`` to
    its place in ``cur``, so the sampling coordinate in ``prev`` is the current pixel MINUS that
    displacement. Adding it doubles the motion and, on real footage, leaves the "camera removed"
    residual almost exactly equal to the unaligned difference -- which is how a sign error hides:
    the numbers stay plausible, they just stop measuring what they claim. ``measure`` reports the
    unaligned difference and the gain alongside, so a dead flow step shows up as gain near zero.
    """
    h, w = prev.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev, gx - flow[..., 0], gy - flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def measure(pairs, roi=None) -> dict:
    """Measure a stream of consecutive grey frame pairs.

    ``pairs`` yields (previous, current). Both the residual and the detail energy are taken over
    the same region of the same frames, so ``redraw`` is a ratio of like for like.
    """
    residuals, unaligned, details = [], [], []
    box = None
    for prev, cur in pairs:
        if box is None:
            box = roi_box(cur.shape, roi)
        y0, y1, x0, x1 = box
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 4, 21, 3, 5, 1.2, 0)
        a, b = highpass(_warp(prev, flow))[y0:y1, x0:x1], highpass(cur)[y0:y1, x0:x1]
        residuals.append(float(np.abs(b - a).mean()))
        unaligned.append(float(np.abs(b - highpass(prev)[y0:y1, x0:x1]).mean()))
        details.append(float(b.std()))
    if not residuals:
        raise ValueError("no frame pairs to measure")
    res, raw, det = np.array(residuals), np.array(unaligned), float(np.mean(details))
    return {
        "pairs": len(res),
        "warble": float(res.mean()),
        "raw": float(raw.mean()),
        # the share of the frame-to-frame detail change that camera alignment explained. Near zero
        # means the flow step did nothing and the warble figure is just unaligned difference.
        "gain": float(1.0 - res.mean() / raw.mean()) if raw.mean() else float("nan"),
        "detail": det,
        "redraw": float(res.mean() / det) if det else float("nan"),
        "peak": float(res.max()),
        "spread": float(res.std() / res.mean()) if res.mean() else float("nan"),
        "per_pair": [round(float(v), 4) for v in res],
        "box": list(box),
    }


def rigid_control(frame: np.ndarray, dx: float = 4.0, dy: float = 1.5, pairs: int = 24, roi=None) -> dict:
    """The floor: one real frame panned by a known camera move, so it provably cannot warp.

    Run through the identical measurement, on the identical region, at the identical sample width.
    A camera that pans in whole pixels would leave a perfect measurement at zero, so whatever this
    returns is interpolation, sub-pixel resampling and flow error -- the part of ``warble`` that is
    never the landmark.
    """
    h, w = frame.shape[:2]
    shifted = []
    for i in range(pairs + 1):
        m = np.float32([[1, 0, -i * dx], [0, 1, -i * dy]])
        shifted.append(cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE))
    stats = measure(((shifted[i], shifted[i + 1]) for i in range(pairs)), roi=roi)
    stats["dx_per_frame"], stats["dy_per_frame"] = dx, dy
    return stats


def repeats(diffs: list[float], fraction: float = REPEAT_FRACTION) -> list[dict]:
    """Frame pairs that are far quieter than the clip's own median: held or duplicated frames."""
    if len(diffs) < 3:
        return []
    median = float(np.median(diffs))
    if median <= 0:
        return []
    return [{"pair": i + 1, "difference": round(float(d), 4), "share_of_median": round(float(d / median), 3)}
            for i, d in enumerate(diffs) if d < median * fraction]


def jump_cuts(diffs: list[float], ratio: float = 5.0, neighbour: float = 2.0) -> list[dict]:
    """Adjacent frame pairs that are a cut rather than fast motion.

    A generated flight can contain a hard cut: on the first Teotihuacan film the model jumped from a
    straight-on view of the pyramid to the figure on the platform at source frames 157->158, 8.3x the
    median difference, and nothing in the pipeline noticed. The discriminator is isolation -- a cut is
    one pair far above the median with ordinary pairs on both sides, while genuinely fast motion
    raises a run of neighbouring pairs together.
    """
    if len(diffs) < 3:
        return []
    median = float(np.median(diffs))
    if median <= 0:
        return []
    found = []
    for i, value in enumerate(diffs):
        if value < median * ratio:
            continue
        left = diffs[i - 1] if i else 0.0
        right = diffs[i + 1] if i + 1 < len(diffs) else 0.0
        if left < median * neighbour and right < median * neighbour:
            found.append({"pair": i + 1, "difference": round(float(value), 3),
                          "times_median": round(float(value / median), 2)})
    return found


def frame_differences(path, width: int = SAMPLE_WIDTH, window: tuple[float, float] | None = None) -> list[float]:
    diffs, prev = [], None
    for gray in _grey_frames(path, width, window):
        if prev is not None:
            diffs.append(float(np.abs(gray - prev).mean()))
        prev = gray
    return diffs


def _grey_frames(path, width: int = SAMPLE_WIDTH, window: tuple[float, float] | None = None):
    """Grey frames at one sample width, optionally only a window of the clip.

    Decodes here rather than through ``video.frames`` because that helper scales to a fixed size
    but cannot seek, and a window is what keeps two clips comparable: a raw generation job starts
    at delivery frame 72, so sampling "3.2-6.4 s of the file" on both samples different content and
    silently reverses the comparison.
    """
    height = _height_for(path, width)
    command = ["ffmpeg", "-v", "error"]
    if window:
        command += ["-ss", f"{window[0]:.3f}", "-t", f"{window[1] - window[0]:.3f}"]
    command += ["-i", str(path), "-vf", f"scale={width}:{height}:flags=area",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = width * height
    try:
        while chunk := proc.stdout.read(size):
            if len(chunk) < size:
                break
            yield np.frombuffer(chunk, np.uint8).reshape((height, width)).astype(np.float32)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()


def _height_for(path, width: int) -> int:
    """Decode at the clip's own aspect: the sample grid must not resample one clip differently."""
    probe = cv2.VideoCapture(str(path))
    try:
        h, w = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        probe.release()
    if not h or not w:
        raise ValueError(f"cannot read frame size from {path}")
    return max(2, int(round(width * h / w / 2)) * 2)


def assess(path, roi=None, window=None, control: tuple[float, float] = (4.0, 1.5),
           max_redraw: float = REDRAW_MAX) -> dict:
    """Full report for one clip: residual, redraw, its own floor, and the repeat scan."""
    grey = list(_grey_frames(path, SAMPLE_WIDTH, window))
    if len(grey) < 3:
        raise ValueError(f"{path} has too few frames in the window to measure")
    stats = measure(((grey[i], grey[i + 1]) for i in range(len(grey) - 1)), roi=roi)
    floor = rigid_control(grey[0], dx=control[0], dy=control[1], roi=roi)
    diffs = [float(np.abs(grey[i] - grey[i - 1]).mean()) for i in range(1, len(grey))]
    stats.update({
        "clip": str(path),
        "window": list(window) if window else None,
        "roi": list(roi) if roi else None,
        "control": {k: floor[k] for k in ("warble", "detail", "redraw", "peak", "gain")},
        "redraw_above_floor": round(stats["redraw"] - floor["redraw"], 4),
        "repeats": repeats(diffs),
        "cuts": jump_cuts(diffs),
    })
    stats["verdict"] = _verdict(stats, max_redraw)
    return stats


def _verdict(stats: dict, max_redraw: float = REDRAW_MAX) -> dict:
    reasons = []
    # A residual below the clip's own floor means less frame-to-frame change than an artificial pan
    # of the same frame produces -- haze, cloud, or a near-static section. That is not evidence of
    # stability, it is a window where the measurement has nothing to measure, so it cannot be a pass.
    if stats["redraw_above_floor"] < -0.05:
        reasons.append(f"residual sits {abs(stats['redraw_above_floor']):.2f} BELOW this clip's own "
                       f"measurement floor: no measurable surface change in this window")
        return {"status": "unverified", "reasons": reasons}
    if stats["control"]["redraw"] > CONTROL_MAX:
        reasons.append(f"control floor {stats['control']['redraw']:.2f} exceeds {CONTROL_MAX}: flow unreliable")
        return {"status": "unverified", "reasons": reasons}
    if stats["redraw"] > max_redraw:
        reasons.append(f"redraw {stats['redraw']:.2f} exceeds {max_redraw:.2f}")
    for cut in stats.get("cuts", []):
        reasons.append(f"hard cut at frame pair {cut['pair']} ({cut['times_median']}x the median difference)")
    if stats["repeats"]:
        reasons.append(f"{len(stats['repeats'])} held frame pair(s)")
    return {"status": "fail" if reasons else "pass", "reasons": reasons}


def _window(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    start, end = (float(v) for v in text.split(":"))
    if end <= start:
        raise ValueError("--window wants START:END with END after START")
    return start, end


def _roi(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    values = tuple(float(v) for v in text.split(","))
    if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError("--roi wants Y0,Y1,X0,X1 as fractions of the frame")
    return values


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip", type=Path)
    p.add_argument("--window", help="START:END in the clip's own seconds; default the whole clip")
    p.add_argument("--roi", help="Y0,Y1,X0,X1 fractions; default the whole frame")
    p.add_argument("--control", default="4.0,1.5", help="control pan in px per frame, default 4.0,1.5")
    p.add_argument("--max", type=float, default=REDRAW_MAX,
                   help=f"redraw ceiling for this scene, default {REDRAW_MAX}")
    p.add_argument("--json", type=Path, help="write the report here")
    a = p.parse_args(argv)
    dx, dy = (float(v) for v in a.control.split(","))
    report = assess(a.clip, roi=_roi(a.roi), window=_window(a.window), control=(dx, dy),
                    max_redraw=a.max)
    c = report["control"]
    print(f"{report['clip']}")
    print(f"  warble {report['warble']:.3f}  detail {report['detail']:.2f}  redraw {report['redraw']:.3f}  "
          f"peak {report['peak']:.3f}  spread {report['spread']:.2f}")
    print(f"  floor  {c['warble']:.3f}  detail {c['detail']:.2f}  redraw {c['redraw']:.3f}   "
          f"above floor {report['redraw_above_floor']:+.3f}")
    print(f"  camera alignment removed {report['gain'] * 100:.0f}% of the frame-to-frame detail change "
          f"(raw {report['raw']:.3f})")
    print(f"  pairs {report['pairs']}  repeats {len(report['repeats'])}  "
          f"verdict {report['verdict']['status'].upper()}")
    for reason in report["verdict"]["reasons"]:
        print(f"    - {reason}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1))
        print(f"  wrote {a.json}")
    return 0 if report["verdict"]["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
