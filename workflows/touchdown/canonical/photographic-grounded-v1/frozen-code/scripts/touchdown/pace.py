"""Deterministic late-dive acceleration without interpolation or an extra model render."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import video as V


def frame_indices(total: int, bounds: dict, *, peak: float, dive_start: float, swoop_end: float,
                  entry_rate: float = 1.0, entry_ease: int = 6) -> tuple[list[int], dict]:
    """Ease from 1x to ``peak`` at the dive/swoop join, then back to 1x in the early swoop. ``entry_rate`` > 1 also
    runs the start of the dive faster: the rate eases up to it over the last ``entry_ease`` cloud frames and climbs
    from it to ``peak`` across the dive (Monument Valley v8: ~2 s of pale flat map after the cloud)."""
    if peak < 1 or not 0 <= dive_start < 1 or not 0 < swoop_end <= 1:
        raise ValueError("pace needs peak >= 1 and fractions within their segments")
    dive = bounds["dive"]
    swoop = bounds["swoop"]
    a = dive[0] + round(dive_start * (dive[1] - dive[0]))
    b = swoop[0]
    c = b + round(swoop_end * (swoop[1] - b))
    if not 0 <= a < b < c < total:
        raise ValueError(f"invalid pace window {a}, {b}, {c} for {total} frames")

    def smooth(x: float) -> float:
        x = float(np.clip(x, 0, 1))
        return x * x * (3 - 2 * x)

    rates = np.ones(total - 1, dtype=np.float64)
    for i in range(a, c):
        rise = smooth((i - a) / (b - a)) if i <= b else 1 - smooth((i - b) / (c - b))
        rates[i] = 1 + (peak - 1) * rise
    if entry_rate > 1:
        d0 = dive[0]
        for i in range(max(d0 - entry_ease, 0), b):
            if i < d0:
                floor = 1 + (entry_rate - 1) * smooth((i - (d0 - entry_ease)) / entry_ease)
            else:
                floor = entry_rate + (peak - entry_rate) * smooth((i - d0) / (b - d0))
            rates[i] = max(rates[i], floor)
    playback_time = np.r_[0, np.cumsum(1 / rates)]
    selected = np.rint(np.interp(np.arange(0, playback_time[-1], 1),
                                  playback_time, np.arange(total))).astype(int).tolist()
    if selected[-1] != total - 1:
        selected.append(total - 1)
    selected = list(dict.fromkeys(selected))
    new_bounds = {name: [int(np.searchsorted(selected, lo, side="left")),
                         int(np.searchsorted(selected, hi, side="left"))]
                  for name, (lo, hi) in bounds.items()}
    return selected, {"segments": new_bounds, "window": [a, b, c], "peak": peak, "entry_rate": entry_rate,
                      "original_frames": total, "paced_frames": len(selected)}


def render(source: Path, bounds: dict, output: Path, segments_path: Path,
           *, peak: float, dive_start: float, swoop_end: float, entry_rate: float = 1.0) -> dict:
    total = V.count(source)
    selected, facts = frame_indices(total, bounds, peak=peak, dive_start=dive_start, swoop_end=swoop_end,
                                    entry_rate=entry_rate)
    keep = set(selected)
    with V.Writer(output) as writer:
        for i, image in enumerate(V.frames(source)):
            if i in keep:
                writer.write(image)
    if V.count(output) != len(selected):
        raise RuntimeError(f"pace wrote {V.count(output)} frames, expected {len(selected)}")
    segments_path.write_text(json.dumps(facts["segments"], indent=2) + "\n")
    return facts
