"""A flock of birds crossing close to the lens at the computed-to-generated hand-over (Jamie, 17 Oct: "even birds
flying across or other distractions might help"). The eye follows the birds while the scene behind them changes
from the measured guide to the model's approach.

Birds are drawn, not generated: gull-like silhouettes (two-segment wings flapping, a small body), dark against the
sky and terrain, each on its own straight path across the frame, growing as it nears the camera, with a little
motion blur. No model, no seed lottery; ``seed`` only places the flock.

    python -m touchdown.birds in.mp4 out.mp4 --at 120 --frames 30
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import video as V

SS = 2  # supersampling for clean edges


def _bird(draw: ImageDraw.ImageDraw, x: float, y: float, span: float, phase: float, heading: float,
          shade: int) -> None:
    """One bird at (x, y), wingspan ``span`` px, flap ``phase`` (radians), flying toward ``heading`` (radians)."""
    flap = math.sin(phase)  # +1 wings up, -1 wings down
    half = span / 2
    tilt = 0.25 * math.sin(heading)  # bank the wings slightly along the flight path

    def p(u: float, v: float) -> tuple[float, float]:  # bird space (u along the wings, v up) -> screen
        return (x + u * SS, y + (-v + u * tilt) * SS)

    lift = 0.45 * half * flap  # wing tips rise and fall
    elbow = 0.2 * half * flap
    chord = 0.14 * span
    for side in (-1, 1):
        wing = [p(0, chord * 0.4), p(side * half * 0.45, elbow + chord * 0.5), p(side * half, lift),
                p(side * half * 0.5, elbow - chord * 0.2), p(0, -chord * 0.3)]
        draw.polygon(wing, fill=(shade, shade, shade, 235))
    body = 0.11 * span
    draw.ellipse([x - body * 0.35 * SS, y - body * 0.2 * SS, x + body * 0.35 * SS, y + body * 0.2 * SS],
                 fill=(shade, shade, shade, 245))


def flock(count: int, seed: int, size: float) -> list[dict]:
    """Paths for ``count`` birds: start/end positions (fractions of the frame), size, flap rate and phase."""
    rng = np.random.default_rng(seed)
    side = 1 if rng.random() < 0.5 else -1  # the flock crosses left-to-right or right-to-left
    birds = []
    for k in range(count):
        near = rng.random() ** 1.5  # most birds farther away, a few big ones close to the lens
        y0 = 0.15 + 0.55 * rng.random()
        birds.append({"x0": 0.5 - side * (0.75 + 0.3 * rng.random()), "x1": 0.5 + side * (0.75 + 0.3 * rng.random()),
                      "y0": y0, "y1": y0 + (rng.random() - 0.65) * 0.35,
                      "span": size * (0.45 + 1.9 * near), "rate": 0.55 + 0.25 * rng.random(),
                      "phase": rng.random() * math.tau, "delay": rng.random() * 0.35,
                      "shade": int(18 + 34 * (1 - near))})
    return sorted(birds, key=lambda b: b["span"])  # far birds first, so near ones draw over them


def overlay(src: Path, out: Path, at: int, frames: int = 30, count: int = 9, seed: int = 0,
            size: float = 0.07) -> dict:
    """Copy ``src`` to ``out`` with the flock crossing frames [at - frames // 2, at + frames - frames // 2)."""
    first = at - frames // 2
    birds = flock(count, seed, size)
    drawn = 0
    with V.Writer(out) as w:
        for i, im in enumerate(V.frames(src)):
            j = i - first
            if 0 <= j < frames:
                h, wd = im.shape[:2]
                layer = Image.new("RGBA", (wd * SS, h * SS), (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)
                for b in birds:
                    u = (j / max(frames - 1, 1) - b["delay"]) / (1 - b["delay"])
                    if not 0 <= u <= 1:
                        continue
                    x = (b["x0"] + (b["x1"] - b["x0"]) * u) * wd * SS
                    y = (b["y0"] + (b["y1"] - b["y0"]) * u) * h * SS
                    grow = 1 + 0.35 * u  # nearing the camera
                    heading = math.atan2(b["y1"] - b["y0"], b["x1"] - b["x0"])
                    _bird(draw, x, y, b["span"] * wd * grow, b["phase"] + j * b["rate"] * math.tau / 4, heading,
                          b["shade"])
                    drawn += 1
                layer = layer.resize((wd, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8))
                base = Image.fromarray(im).convert("RGBA")
                im = np.asarray(Image.alpha_composite(base, layer).convert("RGB"))
            w.write(im)
    return {"first": first, "frames": frames, "count": count, "seed": seed, "size": size, "drawn": drawn}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--at", type=int, required=True, help="frame the flock is centred on (the hand-over)")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=float, default=0.07, help="typical wingspan as a share of the frame width")
    a = p.parse_args()
    print(overlay(a.src, a.out, a.at, a.frames, a.count, a.seed, a.size))


if __name__ == "__main__":
    main()
