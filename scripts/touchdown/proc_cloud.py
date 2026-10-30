"""Procedural cloud plunge (Jamie, 17 Oct: try a procedural cloud instead of the LTX one). No model, so no seed lottery.

A horizontal deck of cumulus, ray-cast as stacked sheets of tileable fractal noise between a base and a top height.
Each sheet's threshold rises away from the deck's middle, so the billows come out as domes, lit on top and
blue-grey underneath, hazing out toward the horizon. The camera looks forward and down at ``pitch`` and dives along
its view line from above the deck into it, so the frame fills with cloud and ends opaque white, as the terrain
route's cloud veil expects. The first ``blend`` frames carry over from the opener's last kept frame under a radial
zoom blur.

    python -m touchdown.proc_cloud start.png out.mp4 --frames 32
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from . import video as V

SUN = np.array([250, 249, 246], np.float32)
SHADE = np.array([128, 141, 165], np.float32)
GROUND = np.array([96, 108, 126], np.float32)
SKY_LOW = np.array([186, 206, 230], np.float32)


def fbm(size: int, seed: int, octaves: int = 6, base: int = 4) -> np.ndarray:
    """Tileable fractal value noise, 0..1, (size, size); ``size`` must be divisible by base * 2**(octaves-1)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = base * 2 ** o
        grid = np.tile(rng.random((cells, cells), dtype=np.float32), (3, 3))  # tile so the cubic resize wraps
        layer = cv2.resize(grid, (3 * size, 3 * size), interpolation=cv2.INTER_CUBIC)[size:2 * size, size:2 * size]
        out += amp * layer
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / max(float(np.ptp(out)), 1e-6)


def _radial_blur(img: np.ndarray, amount: float, centre: tuple[float, float]) -> np.ndarray:
    if amount <= 0.002:
        return img
    h, w = img.shape[:2]
    acc = np.zeros_like(img, dtype=np.float32)
    n = 6
    for k in range(n):
        s = 1 + amount * k / (n - 1)
        m = np.float32([[s, 0, (1 - s) * centre[0] * w], [0, s, (1 - s) * centre[1] * h]])
        acc += cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def render(start: np.ndarray, out: Path, frames: int = 32, seed: int = 0, pitch_deg: float = 45.0, start_pitch_deg: float = 14.0,
           layers: int = 36, blend: int = 6, work: tuple[int, int] = (960, 544), size: int = 1536,
           tile_m: float = 40000.0) -> dict:
    """Write ``frames`` frames to ``out``; the last one is opaque white."""
    w, h = work
    tex = fbm(size, seed, base=12)  # one tile spans ``tile_m``, so the billows don't visibly repeat
    base_m, top_m = 2000.0, 3600.0
    mid, half = (base_m + top_m) / 2, (top_m - base_m) / 2
    heights = np.linspace(top_m, base_m, layers)
    tan = math.tan(math.radians(30))
    xs = (np.arange(w) + 0.5) / w * 2 - 1
    ys = 1 - (np.arange(h) + 0.5) / h * 2
    gx, gy = np.meshgrid(xs * tan, ys * tan * h / w)

    def camera(pitch_rad: float):
        """forward vector and per-pixel unit rays for a camera facing +y, pitched down"""
        fwd = np.array([0, math.cos(pitch_rad), -math.sin(pitch_rad)])
        right = np.array([1.0, 0, 0])
        up = np.cross(right, fwd)
        r = fwd[None, None] + gx[..., None] * right + gy[..., None] * up
        return fwd, r / np.linalg.norm(r, axis=-1, keepdims=True)

    sky_top = start[: max(start.shape[0] // 12, 1)].reshape(-1, 3).astype(np.float32).mean(0)  # the opener's sky
    start_f = cv2.resize(start, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    # dive along the view line from 12 km above the top into the deck's middle, accelerating
    z0, z1 = top_m + 12000.0, mid - 200.0
    pitch = math.radians(pitch_deg)
    pitch0 = math.radians(start_pitch_deg)  # the opener looks out at the horizon; tip down to ``pitch`` in the dive
    means = []
    with V.Writer(out) as writer:
        for i in range(frames):
            t = (i + 1) / frames
            s = t ** 2.0  # slow high up (matching the opener's scale), fast into the deck
            p_now = pitch0 + (pitch - pitch0) * V.smoothstep(min(t / 0.6, 1.0))
            fwd, rays = camera(p_now)
            dz = rays[..., 2]
            horizon = np.clip(1 - np.abs(dz) * 6, 0, 1)[..., None]
            back = np.where((dz > 0)[..., None], sky_top + (SKY_LOW - sky_top) * horizon,
                            GROUND + (SKY_LOW - GROUND) * horizon)
            cam = np.array([0.0, (z0 - z1) * s, z0 - (z0 - z1) * s])  # down and forward at 45 degrees
            colour = np.zeros((h, w, 3), np.float32)
            trans = np.ones((h, w), np.float32)
            tz = (heights[:, None, None] - cam[2]) / np.where(np.abs(dz) < 1e-4, 1e-4, dz)[None]
            order = np.argsort(np.where(tz > 0, tz, np.inf), axis=0)
            for rank in range(layers):
                k = order[rank]
                tk = np.take_along_axis(tz, k[None], 0)[0]
                valid = (tk > 0) & np.isfinite(tk)
                zk = heights[k]
                # each sheet drifts sideways with height, so the billows round over instead of extruding as columns
                drift = (zk - base_m) * 0.35
                px = cam[0] + rays[..., 0] * tk + drift
                py = cam[1] + rays[..., 1] * tk + 0.6 * drift
                u = ((px / tile_m * size) % size).astype(np.float32)
                v = ((py / tile_m * size) % size).astype(np.float32)
                n = cv2.remap(tex, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                rel = (zk - base_m) / (top_m - base_m)  # 0 at the base, 1 at the top
                thresh = 0.18 + 0.62 * rel ** 1.4  # domes: a solid base, narrowing toward the top
                a = np.clip((n - thresh) / 0.12, 0, 1) * 0.45
                # far sheets fade into haze; very near ones soften so they stream past the lens
                dist = np.where(valid, tk, 1e9)
                a *= np.exp(-dist / 60000.0) * np.clip(dist / 120.0, 0, 1)
                a *= valid
                lit = np.clip((zk - base_m) / (top_m - base_m), 0, 1)[..., None]
                lit = lit * 0.75 + 0.25 * np.clip((n - thresh)[..., None] * 4, 0, 1)
                col = SHADE + (SUN - SHADE) * np.clip(lit, 0, 1)
                haze = np.clip(dist / 80000.0, 0, 1)[..., None]
                col = col * (1 - haze) + SKY_LOW * haze
                colour += (trans * a)[..., None] * col
                trans *= 1 - a
            img = colour + trans[..., None] * back
            # inside the deck everything goes milky, then opaque white over the last frames
            inside = V.smoothstep(np.clip((top_m + 200 - cam[2]) / 1200.0, 0, 1))
            img = img * (1 - 0.5 * inside) + SUN * 0.5 * inside
            whiteout = V.smoothstep(np.clip((t - 0.78) / 0.22, 0, 1))
            img = img * (1 - whiteout) + SUN * whiteout
            if i < blend:  # carry over from the opener frame, under a zoom blur that hides the swap
                b = V.smoothstep((i + 1) / (blend + 1))
                img = start_f * (1 - b) + img * b
                img = _radial_blur(img, 0.06 * math.sin(math.pi * (i + 1) / (blend + 1)), (0.5, 0.45))
            frame = np.clip(cv2.resize(img, (V.W, V.H), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
            writer.write(frame)
            means.append(float(frame.mean()))
    return {"engine": "procedural", "frames": frames, "layers": layers, "seed": seed, "pitch_deg": pitch_deg,
            "first_mean": round(means[0], 1), "last_mean": round(means[-1], 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pitch", type=float, default=45.0)
    a = p.parse_args()
    from PIL import Image
    start = np.asarray(Image.open(a.start).convert("RGB").resize((V.W, V.H)))
    print(render(start, a.out, a.frames, a.seed, a.pitch))


if __name__ == "__main__":
    main()
