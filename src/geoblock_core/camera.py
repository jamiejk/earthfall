"""Preset camera routes computed from a landmark and its measured terrain.

Coordinates are local metres: x east and y north of the scene origin (the
landmark), z the absolute height in the scene's vertical CRS. This matches the
Blender frame of the glTF terrain, so routes can be keyed without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio

FPS = 24
H3_MAX_FRAMES = 17 * 20 + 5  # 345 frames, 14.4 s: the longest H3 length inside 15 s

PRESETS = {
    "Rim reveal": "Skim low toward the edge, then fly out over the drop as the view opens.",
    "Orbit": "Circle the landmark while keeping it centred.",
    "Flyover": "Pass straight over the landmark, looking ahead and down.",
    "Push-in": "Approach the landmark from out over the low ground.",
    "Pull-back rise": "Start close on the landmark and pull away while climbing.",
    "Earth zoom": "Start in low orbit above the curved horizon, dive through the clouds and swoop forward down "
                  "to eye level at the landmark, then push in slowly.",
}


@dataclass
class Terrain:
    """A metric heightfield with bilinear sampling in local coordinates."""

    heights: np.ndarray
    x0: float  # local x of the first column centre
    y0: float  # local y of the first row centre (north edge)
    spacing: float

    @classmethod
    def from_scene(cls, scene: dict) -> "Terrain":
        ox, oy = scene["origin_projected_m"]
        with rasterio.open(scene["terrain_raster"]) as dataset:
            heights = dataset.read(1, masked=True).astype("float32").filled(np.nan)
            spacing = float(dataset.res[0])
            x0 = dataset.bounds.left + spacing / 2 - ox
            y0 = dataset.bounds.top - spacing / 2 - oy
        return cls(heights, x0, y0, spacing)

    def sample(self, x, y) -> np.ndarray:
        col = (np.asarray(x, dtype="float64") - self.x0) / self.spacing
        row = (self.y0 - np.asarray(y, dtype="float64")) / self.spacing
        rows, cols = self.heights.shape
        inside = (col >= 0) & (row >= 0) & (col <= cols - 1) & (row <= rows - 1)
        col = np.clip(col, 0, cols - 1)
        row = np.clip(row, 0, rows - 1)
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        c1 = np.minimum(c0 + 1, cols - 1)
        r1 = np.minimum(r0 + 1, rows - 1)
        tx, ty = col - c0, row - r0
        h = self.heights
        out = (h[r0, c0] * (1 - tx) * (1 - ty) + h[r0, c1] * tx * (1 - ty) +
               h[r1, c0] * (1 - tx) * ty + h[r1, c1] * tx * ty)
        return np.where(inside, out, np.nan)

    @property
    def half_extent(self) -> float:
        return min(abs(self.x0), abs(self.y0)) + self.spacing / 2


def h3_frame_count(duration_s: float) -> int:
    """Nearest MiniMax H3 length (17n + 5 frames at 24 fps) to the request."""
    n = max(1, round((duration_s * FPS - 5) / 17))
    if duration_s <= 15:  # a single clip must stay within H3's 15 s reference limit
        n = min(n, (H3_MAX_FRAMES - 5) // 17)
    return 17 * n + 5


def chain_segments(frames: int, overlap: int = 22) -> list[tuple[int, int]]:
    """Split a long route into overlapping H3-length windows for clip chaining."""
    if frames <= H3_MAX_FRAMES:
        return [(0, frames)]
    segments, start = [], 0
    while True:
        end = min(start + H3_MAX_FRAMES, frames)
        segments.append((start, end))
        if end == frames:
            return segments
        start = end - overlap


def auto_heading(terrain: Terrain, target_h: float, distance: float) -> float:
    """Azimuth (degrees clockwise from north) toward the deepest drop from the landmark."""
    azimuths = np.arange(0, 360, 5, dtype="float64")
    radii = distance * np.array([0.25, 0.5, 0.75, 1.0])
    a = np.radians(azimuths)[:, None]
    heights = terrain.sample(radii * np.sin(a), radii * np.cos(a))
    drop = np.nanmean(target_h - heights, axis=1)
    drop = np.nan_to_num(drop, nan=-1e9)
    # Smooth over neighbouring directions so a single gully does not win.
    smooth = sum(np.roll(drop, k) for k in (-3, -2, -1, 0, 1, 2, 3)) / 7
    return float(azimuths[int(np.argmax(smooth))])


def _ease(s: np.ndarray) -> np.ndarray:
    return s * s * s * (s * (6 * s - 15) + 10)


def _route(preset: str, s: np.ndarray, target: np.ndarray, heading: float, scale: float,
           depth: float, orbit_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Unconstrained camera positions and look-at points for path parameter s in [0, 1]."""
    forward = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading)), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    s = s[:, None]
    lerp = lambda a, b, u: a + (b - a) * u
    if preset == "Rim reveal":
        start = target - 0.30 * scale * forward + up * 25
        end = target + 0.55 * scale * forward + up * (0.12 * scale)
        position = lerp(start, end, s)
        # Stay low over the approach, then climb once past the edge.
        position[:, 2] = lerp(start[2], end[2], s[:, 0] ** 1.6)
        near_look = target + 0.05 * scale * forward - up * 40
        far_look = target + 1.6 * scale * forward - up * (0.35 * depth)
        look = lerp(near_look, far_look, _ease(np.clip(s * 1.4 - 0.15, 0, 1)))
    elif preset == "Orbit":
        start_angle = math.radians(heading + 180 - orbit_degrees / 2)
        angle = start_angle + math.radians(orbit_degrees) * s[:, 0]
        radius = 0.45 * scale
        position = np.column_stack((target[0] + radius * np.sin(angle),
                                    target[1] + radius * np.cos(angle),
                                    np.full(len(angle), target[2] + 0.18 * scale)))
        look = np.repeat((target - up * 0.1 * depth)[None, :], len(angle), axis=0)
    elif preset == "Flyover":
        start = target - 0.7 * scale * forward + up * (0.10 * scale)
        end = target + 0.9 * scale * forward + up * (0.10 * scale)
        position = lerp(start, end, s)
        look = position + forward * scale * 0.6 - up * (0.10 * scale + 0.3 * depth)
    elif preset == "Push-in":
        start = target + 1.1 * scale * forward + up * (0.30 * scale)
        end = target + 0.22 * scale * forward + up * (0.06 * scale)
        position = lerp(start, end, s)
        look = np.repeat((target - up * 10)[None, :], len(s), axis=0)
    elif preset == "Pull-back rise":
        start = target + 0.15 * scale * forward + up * (0.05 * scale)
        end = target + 1.4 * scale * forward + up * (0.75 * scale)
        position = lerp(start, end, s ** 0.9)
        look = np.repeat((target - up * 10)[None, :], len(s), axis=0)
    else:
        raise ValueError(f"Unknown camera preset {preset!r}; choose one of {', '.join(PRESETS)}")
    return position, look


def _lift_for_clearance(position: np.ndarray, ground: np.ndarray, clearance: float) -> np.ndarray:
    """Smoothly raise the route wherever it passes too close to the terrain."""
    deficit = np.nan_to_num(clearance - (position[:, 2] - ground), nan=0.0).clip(min=0)
    if not deficit.any():
        return np.zeros(len(position))
    window = FPS  # one second either side
    padded = np.pad(deficit, window, mode="edge")
    envelope = np.array([padded[i:i + 2 * window + 1].max() for i in range(len(deficit))])
    kernel = np.hanning(2 * window + 1)
    kernel /= kernel.sum()
    lifted = np.convolve(np.pad(envelope, window, mode="edge"), kernel, mode="valid")
    return np.maximum(lifted, deficit)


def plan_route(scene: dict, preset: str, duration_s: float, *, heading_deg: float | None = None,
               scale_m: float | None = None, lens_mm: float = 24, clearance_m: float = 25,
               orbit_degrees: float = 120, ease: bool = True, terrain: Terrain | None = None,
               dive_tempo: str = "standard") -> dict:
    """Compute a per-frame camera route and check it against the measured terrain."""
    if preset == "Earth zoom":
        from .zoom import plan_earth_zoom
        return plan_earth_zoom(scene, duration_s, heading_deg=heading_deg, lens_mm=lens_mm, tempo=dive_tempo)
    if not 2 <= duration_s <= 60:
        raise ValueError("Route duration must be between 2 and 60 seconds.")
    terrain = terrain or Terrain.from_scene(scene)
    target_h = float(scene["target_height_m"])
    target = np.array([0.0, 0.0, target_h])
    scale = float(scale_m or 0.45 * terrain.half_extent)
    if heading_deg is None or heading_deg < 0:
        heading = auto_heading(terrain, target_h, scale)
        heading_source = "auto: toward the deepest measured drop"
    else:
        heading = float(heading_deg) % 360
        heading_source = "user"
    ring = terrain.sample(*(scale * np.array([[math.sin(math.radians(heading))],
                                              [math.cos(math.radians(heading))]])))
    depth = float(np.nan_to_num(target_h - ring[0], nan=0.0).clip(min=0))
    frames = h3_frame_count(duration_s)
    s = np.linspace(0.0, 1.0, frames)
    position, look = _route(preset, _ease(s) if ease else s, target, heading, scale, depth, orbit_degrees)
    ground = terrain.sample(position[:, 0], position[:, 1])
    if np.isnan(ground).any():
        first = int(np.argmax(np.isnan(ground)))
        raise ValueError(f"{preset} leaves the measured terrain at frame {first + 1}; "
                         "reduce the shot scale or build a larger scene.")
    lift = _lift_for_clearance(position, ground, clearance_m)
    position[:, 2] += lift
    look[:, 2] += lift * 0.5
    clearance = position[:, 2] - ground
    route = {
        "schema_version": "1.0",
        "preset": preset,
        "description": PRESETS[preset],
        "scene": scene["project_id"],
        "terrain_path": scene["terrain_path"],
        "fps": FPS,
        "frames": frames,
        "duration_s": round(frames / FPS, 3),
        "requested_duration_s": duration_s,
        "chain_segments": chain_segments(frames),
        "heading_deg": heading,
        "heading_source": heading_source,
        "scale_m": scale,
        "lens_mm": lens_mm,
        "target_local_m": target.tolist(),
        "min_clearance_m": round(float(clearance.min()), 2),
        "max_lift_m": round(float(lift.max()), 2),
        "positions": np.round(position, 3).tolist(),
        "look_at": np.round(look, 3).tolist(),
    }
    route["route_sha256"] = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()
    return route


def route_report(route: dict) -> str:
    chained = len(route["chain_segments"])
    return (
        f"PRESET: {route['preset']} — {route['description']}\n"
        f"HEADING: {route['heading_deg']:.0f}° ({route['heading_source']}); scale {route['scale_m']:.0f} m\n"
        f"TIMING: {route['frames']} frames at {route['fps']} fps ({route['duration_s']:.2f} s)"
        + (f"; {chained} chained H3 segments" if chained > 1 else "") + "\n"
        f"CLEARANCE: min {route['min_clearance_m']:.0f} m above measured ground"
        + (f"; raised up to {route['max_lift_m']:.0f} m to clear terrain" if route["max_lift_m"] > 0 else "")
    )


def route_preview(route: dict, terrain: Terrain, output: Path, size: int = 900) -> Path:
    """Top-down hillshade with the camera path, start, end and landmark marked."""
    h = terrain.heights
    filled = np.where(np.isfinite(h), h, np.nanmin(h))
    gy, gx = np.gradient(filled, terrain.spacing)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    sun_alt, sun_az = math.radians(40), math.radians(315)
    shade = (math.sin(sun_alt) * np.cos(slope) +
             math.cos(sun_alt) * np.sin(slope) * np.cos(sun_az - aspect)).clip(0, 1)
    lo, hi = np.nanpercentile(h, [2, 98])
    tint = np.nan_to_num((h - lo) / max(hi - lo, 1), nan=0).clip(0, 1)
    rgb = np.stack([110 + 120 * tint, 95 + 100 * tint, 80 + 70 * tint], axis=-1) * (0.35 + 0.65 * shade[..., None])
    rgb[~np.isfinite(h)] = (29, 45, 65)
    image = Image.fromarray(rgb.clip(0, 255).astype("uint8")).resize((size, size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    rows, cols = h.shape
    to_px = lambda x, y: ((x - terrain.x0) / terrain.spacing / (cols - 1) * (size - 1),
                          (terrain.y0 - y) / terrain.spacing / (rows - 1) * (size - 1))
    path = [to_px(x, y) for x, y, _ in route["positions"]]
    draw.line(path, fill=(255, 210, 40), width=4)
    for index in range(0, len(path), max(1, len(path) // 12)):
        px, py = path[index]
        lx, ly = to_px(*route["look_at"][index][:2])
        length = math.hypot(lx - px, ly - py) or 1
        draw.line((px, py, px + 30 * (lx - px) / length, py + 30 * (ly - py) / length), fill=(255, 255, 255), width=2)
    sx, sy = path[0]
    ex, ey = path[-1]
    tx, ty = to_px(0, 0)
    draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=(60, 200, 90))
    draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=(220, 60, 60))
    draw.rectangle((tx - 6, ty - 6, tx + 6, ty + 6), outline=(255, 255, 255), width=3)
    draw.text((12, 10), f"{route['preset']}  {route['duration_s']:.1f} s  heading {route['heading_deg']:.0f}°  "
              "green start, red end, square landmark", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
