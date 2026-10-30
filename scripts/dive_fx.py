"""Camera shake and speed lines for the computed dive (Jamie, 2026-10-10: "a little clinical right now").

Post-process on the dive's frame range of a finished clip (start = the cloud exit), CPU only (numpy + PIL).
Merged with the desktop's shake_streaks.py (refine/, 2026-10-10): its shake profile, its order (blur about the
true vanishing point first, then shake the whole picture) and its softer edge mask.

- Shake: two bands of smooth noise (slow drift ~1.5 Hz, fast buffeting ~9 Hz) on x, y and roll; a kick at the
  start of the range easing to 35% by mid-dive. The frame is scaled up just enough to hide the edges; that scale
  follows the envelope, not each frame's offset, so the zoom does not pump.
- Speed: strength follows the clip's own motion inside the range (smoothed). A radial zoom blur on the outer
  part of the frame (the centre, where the dive is headed, stays sharp), plus thin white streaks that spawn at
  the edge of the clear centre and race outwards for a few frames.
- Both taper to 0 at the ends of the range, so nothing pops at a join.

    python scripts/dive_fx.py in.mp4 out.mp4 --frames 33:104   # naip_middle_short_a7_wb_final: cloud exit f33, swoop f104
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _small(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((160, 90)), dtype=np.float32)


def envelope(files: list[Path], start: int, end: int, taper: int) -> np.ndarray:
    """Per-frame strength 0..1 over [start, end): motion (mean abs diff to the next frame) smoothed over 9 frames,
    over its 90th percentile, times a smoothstep taper at both ends. 0 outside the range."""
    small = [_small(f) for f in files[start:end + 1]]
    motion = np.array([np.abs(b - a).mean() for a, b in pairwise(small)], np.float32)
    motion = np.convolve(np.pad(motion, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    strength = np.clip(motion / max(float(np.percentile(motion, 90)), 1e-3), 0, 1)
    k = np.arange(len(strength), dtype=np.float32)
    edge = np.clip(np.minimum(k + 1, len(strength) - k) / max(taper, 1), 0, 1)
    strength *= edge * edge * (3 - 2 * edge)
    out = np.zeros(len(files), np.float32)
    out[start:start + len(strength)] = strength
    return out


def _noise(n: int, hz: float, rng: np.random.Generator, fps: float = 24.0) -> np.ndarray:
    """Smooth 1-D noise in -1..1: random keys every 1/hz s, cosine-interpolated."""
    step = fps / hz
    keys = rng.uniform(-1, 1, int(n / step) + 3)
    t = np.arange(n) / step
    i = t.astype(int)
    f = (1 - np.cos((t - i) * np.pi)) / 2
    return keys[i] * (1 - f) + keys[i + 1] * f


def shake(frame: Image.Image, dx: float, dy: float, roll_deg: float, scale: float) -> Image.Image:
    """Translate by (dx, dy) px, roll and scale about the centre (inverse affine, bicubic)."""
    w, h = frame.size
    a = np.deg2rad(roll_deg)
    c, s = np.cos(a) / scale, np.sin(a) / scale
    cx, cy = w / 2, h / 2
    # output (x, y) samples input at R^-1 (x - centre - d) + centre
    coeffs = (c, s, cx - c * (cx + dx) - s * (cy + dy), -s, c, cy + s * (cx + dx) - c * (cy + dy))
    return frame.transform(frame.size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)


def zoom_blur(frame: Image.Image, amount: float, passes: int = 5) -> Image.Image:
    """Radial blur away from the centre, blended in by a radial mask (0 inside 25% of the half-diagonal, rising as
    r**1.5 to 1 at the corners). Each pass averages the image with a copy scaled up by a halving step (amount, amount/2, ...), so the
    result is a smooth 2**passes-sample streak, not a few ghosted copies."""
    if amount < 1e-3:
        return frame
    w, h = frame.size
    acc = np.asarray(frame, np.float32)
    for j in range(passes):
        s = 1 + amount / 2 ** j
        cw, ch = w / s, h / s
        box = ((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2)
        img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
        acc = (acc + np.asarray(img.resize((w, h), Image.Resampling.BILINEAR, box=box), np.float32)) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    m = (np.clip((r - 0.25) / 0.75, 0, 1) ** 1.5)[..., None]  # the desktop's softer mask
    base = np.asarray(frame, np.float32)
    return Image.fromarray(np.clip(base * (1 - m) + acc * m, 0, 255).astype(np.uint8))


class Streaks:
    """Thin light streaks racing outwards from the edge of the clear centre; each lives a few frames."""

    def __init__(self, size: tuple[int, int], rng: np.random.Generator, rate: int = 14, width: int = 3,
                 alpha: float = 1.0):
        self.w, self.h = size
        self.rng, self.rate, self.width, self.gain = rng, rate, width, alpha
        self.live: list[list[float]] = []  # angle, radius (fraction of half-diagonal), length, life left, alpha

    def draw(self, frame: Image.Image, strength: float) -> Image.Image:
        rng = self.rng
        for _ in range(rng.poisson(self.rate * strength)):
            self.live.append([rng.uniform(0, 2 * np.pi), rng.uniform(0.35, 0.6), rng.uniform(0.08, 0.2),
                              int(rng.integers(3, 6)), rng.uniform(0.35, 0.7)])
        if not self.live:
            return frame
        half = np.hypot(self.w, self.h) / 2
        layer = Image.new("L", frame.size, 0)
        pen = ImageDraw.Draw(layer)
        for s in self.live:
            angle, r, length, _, alpha = s
            u, v = np.cos(angle), np.sin(angle)
            r0, r1 = r * half, (r + length * (0.5 + strength)) * half
            pen.line((self.w / 2 + u * r0, self.h / 2 + v * r0, self.w / 2 + u * r1, self.h / 2 + v * r1),
                     fill=int(255 * self.gain * alpha * max(strength, 0.3)), width=self.width)
            s[1] += 0.06 + 0.12 * strength
            s[3] -= 1
        self.live = [s for s in self.live if s[3] > 0 and s[1] < 1.1]
        mask = np.asarray(layer.filter(ImageFilter.GaussianBlur(1.2)), np.float32)[..., None] / 255
        base = np.asarray(frame, np.float32)
        return Image.fromarray(np.clip(base * (1 - mask) + 245 * mask, 0, 255).astype(np.uint8))


def apply(clip: str, out: str, start: int, end: int | None, *, shake_px: float = 12.0, roll_deg: float = 0.4,
          blur: float = 0.035, streaks: int = 14, taper: int = 12, seed: int = 0, streak_width: int = 3,
          streak_alpha: float = 1.0) -> str:
    """Shake + speed lines on frames [start, end) of ``clip``; other frames pass through. Returns ``out``."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", clip, f"{tmp}/%05d.png"], check=True)
        files = sorted(Path(tmp).glob("*.png"))
        end = len(files) if end is None else min(end, len(files))
        env = envelope(files, start, end - 1, taper)  # speed: blur and streaks
        n = len(files)
        # shake: a kick at the start of the range (the cloud exit), easing to 35% by mid-dive (the desktop's
        # shake_streaks.py profile), with the same end tapers as the speed envelope
        k = np.clip((np.arange(n, dtype=np.float32) - start) / max(end - start, 1), 0, 1)
        calm = np.clip((k - 0.25) / 0.3, 0, 1)
        jolt = np.zeros(n, np.float32)
        lo, hi = start, end - 1
        edge = np.clip(np.minimum(np.arange(n) - lo + 1, hi - np.arange(n) + 1) / max(taper / 2, 1), 0, 1)
        jolt[lo:hi + 1] = (edge * edge * (3 - 2 * edge) * (1 - 0.65 * calm * calm * (3 - 2 * calm)))[lo:hi + 1]
        drift = [_noise(n, 1.5, rng) for _ in range(3)]
        buffet = [_noise(n, 9.0, rng) for _ in range(3)]
        lines = Streaks(Image.open(files[0]).size, rng, streaks, streak_width, streak_alpha)
        w, h = Image.open(files[0]).size
        for i in range(start, end):
            e, j = float(env[i]), float(jolt[i])
            if e <= 0 and j <= 0 and not lines.live:
                continue
            frame = Image.open(files[i]).convert("RGB")
            # speed first, about the true vanishing point; the shake then moves the whole picture like a camera
            frame = zoom_blur(frame, blur * e)
            frame = lines.draw(frame, e if i < end - taper else 0.0)
            jx, jy, jr = (0.6 * d[i] + 0.4 * b[i] for d, b in zip(drift, buffet, strict=True))
            dx, dy, roll = jx * shake_px * j, jy * shake_px * j, jr * roll_deg * j
            # a scale that follows the envelope only (not each frame's offset, which would pump the zoom) and
            # covers the largest possible offset and roll at this strength
            cover = 1 + 2 * shake_px * j * (1 / w + 1 / h) + np.deg2rad(roll_deg * j) * (w / h + h / w) / 2
            frame = shake(frame, dx, dy, roll, cover)
            frame.save(files[i])
        print(f"dive_fx: frames {start}-{end - 1}, peak strength {env.max():.2f}")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", out], check=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip")
    p.add_argument("out")
    p.add_argument("--frames", default="0:", help="start:end (end exclusive, blank = to the end)")
    p.add_argument("--shake-px", type=float, default=12.0, help="peak shake at 1080p")
    p.add_argument("--roll", type=float, default=0.4, help="peak roll in degrees")
    p.add_argument("--blur", type=float, default=0.035, help="peak radial blur (scale spread at the corners)")
    p.add_argument("--streaks", type=int, default=14, help="new streaks per frame at peak (0 = none)")
    p.add_argument("--taper", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    s, _, e = a.frames.partition(":")
    apply(a.clip, a.out, int(s or 0), int(e) if e else None, shake_px=a.shake_px, roll_deg=a.roll, blur=a.blur,
          streaks=a.streaks, taper=a.taper, seed=a.seed)


if __name__ == "__main__":
    main()
