"""Can a person standing at the landing actually see the named view, or does nearer ground hide it?

Works on a geographic (lon/lat) DEM array, as the providers return it. The Earth zoom ends at eye level
just behind the person, looking toward the view, so a crest 50 m in front can hide a mountain 10 km away
(the first Gornergrat blocking showed only the Matterhorn's tip over the summit hump).
"""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
REFRACTION = 0.13  # standard atmospheric refraction coefficient


def _metres_per_degree(latitude: float) -> tuple[float, float]:
    return 111_320 * math.cos(math.radians(latitude)), 110_574.0


def _sample(heights: np.ndarray, transform, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear heights at (lon, lat): the blocking mesh is interpolated too, so nearest-cell steps would
    invent ledges."""
    cols, rows = ~transform * (np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    cols, rows = cols - 0.5, rows - 0.5  # cell centres
    c0 = np.clip(np.floor(cols).astype(int), 0, heights.shape[1] - 2)
    r0 = np.clip(np.floor(rows).astype(int), 0, heights.shape[0] - 2)
    fc, fr = np.clip(cols - c0, 0, 1), np.clip(rows - r0, 0, 1)
    top = heights[r0, c0] * (1 - fc) + heights[r0, c0 + 1] * fc
    bottom = heights[r0 + 1, c0] * (1 - fc) + heights[r0 + 1, c0 + 1] * fc
    return top * (1 - fr) + bottom * fr


def clearance_deg(heights: np.ndarray, transform, person: tuple[float, float], target: tuple[float, float], *,
                  eye_height_m: float = 1.7, camera_back_m: tuple[float, ...] = (2.0, 12.0),
                  near_m: float = 20.0, target_radius_m: float = 400.0, step_m: float = 10.0,
                  structure_m: float = 0.0, foreground_share: float = 0.6) -> float:
    """How far the view target's top stands above the foreground, in degrees, as the Earth zoom's eye-level
    camera sees it: from ``camera_back_m`` behind the person (zoom.plan_earth_zoom holds 12 m back, pushing
    in to 2 m), at eye height above the higher of its own ground and the person's. The worst of those.

    ``person`` and ``target`` are (longitude, latitude). The target's top is the highest ground within
    ``target_radius_m`` of it, plus ``structure_m`` for a built top the terrain model does not hold (a
    bridge tower, a statue). The foreground runs from ``near_m`` past the person to ``foreground_share`` of
    the way to the target, so the target's own flanks are not counted. Positive: clear by that angle.
    """
    kx, ky = _metres_per_degree(person[1])
    dx, dy = (target[0] - person[0]) * kx, (target[1] - person[1]) * ky
    distance = math.hypot(dx, dy)
    if distance <= near_m + target_radius_m:
        return 90.0
    ux, uy = dx / distance, dy / distance
    person_ground = float(_sample(heights, transform, np.array([person[0]]), np.array([person[1]]))[0])
    # The target's summit: the highest point within target_radius_m of it.
    ring = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    rr, aa = np.meshgrid(np.arange(0, target_radius_m + 1, step_m * 2), ring)
    tx, ty = dx + (rr * np.sin(aa)).ravel(), dy + (rr * np.cos(aa)).ravel()
    tz = _sample(heights, transform, person[0] + tx / kx, person[1] + ty / ky) + structure_m
    worst = 90.0
    for back in camera_back_m:
        cx, cy = -ux * back, -uy * back
        camera_ground = float(_sample(heights, transform, np.array([person[0] + cx / kx]),
                                      np.array([person[1] + cy / ky]))[0])
        z_eye = max(camera_ground, person_ground) + eye_height_m

        def angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            d = np.hypot(x - cx, y - cy)
            drop = d * d * (1 - REFRACTION) / (2 * EARTH_RADIUS_M)
            return np.degrees(np.arctan2(z - drop - z_eye, d))

        along = np.arange(near_m, max(near_m + step_m, foreground_share * distance), step_m)
        fx, fy = ux * along, uy * along
        foreground = angle(fx, fy, _sample(heights, transform, person[0] + fx / kx, person[1] + fy / ky))
        top = float(np.nanmax(angle(tx, ty, tz)))
        worst = min(worst, top - float(np.nanmax(foreground)))
    return round(worst, 2)


def steady_clearance_deg(heights: np.ndarray, transform, person: tuple[float, float], target: tuple[float, float],
                         *, wobble_m: float = 15.0, **kwargs) -> float:
    """clearance_deg at the person and ``wobble_m`` either side in both directions, worst case. A 30 m
    surface model can put a building-sized hump right beside the point; a view that only clears on one
    exact cell is not a view the blocking will show."""
    kx, ky = _metres_per_degree(person[1])
    return min(clearance_deg(heights, transform, (person[0] + x / kx, person[1] + y / ky), target, **kwargs)
               for x, y in ((0, 0), (wobble_m, 0), (-wobble_m, 0), (0, wobble_m), (0, -wobble_m)))


def clearest_nearby(heights: np.ndarray, transform, eye: tuple[float, float], target: tuple[float, float], *,
                    radius_m: float = 200.0, step_m: float = 10.0, wanted_deg: float = 4.0,
                    structure_m: float = 0.0) -> tuple[tuple[float, float], float, float]:
    """The point within ``radius_m`` of ``eye`` nearest to it whose view of the target clears by
    ``wanted_deg``, else the one that clears most. Returns ((lon, lat), clearance_deg, moved_m)."""
    kx, ky = _metres_per_degree(eye[1])
    offsets = [(x, y) for x in np.arange(-radius_m, radius_m + 1, step_m)
               for y in np.arange(-radius_m, radius_m + 1, step_m) if math.hypot(x, y) <= radius_m]
    offsets.sort(key=lambda o: math.hypot(*o))
    best = (eye, steady_clearance_deg(heights, transform, eye, target, structure_m=structure_m), 0.0)
    if best[1] >= wanted_deg:
        return best
    for x, y in offsets[1:]:
        point = (eye[0] + x / kx, eye[1] + y / ky)
        value = steady_clearance_deg(heights, transform, point, target, structure_m=structure_m)
        if value >= wanted_deg:
            return point, value, round(math.hypot(x, y), 1)
        if value > best[1]:
            best = (point, value, round(math.hypot(x, y), 1))
    return best


def near_view_deg(heights: np.ndarray, transform, person: tuple[float, float], heading_deg: float, *,
                  eye_height_m: float = 1.7, near_m: float = 3.0, far_m: float = 300.0, step_m: float = 1.0) -> float:
    """How far below level the near ground lets a person at ``person`` see, looking along ``heading_deg``:
    the highest angle of the ground from ``near_m`` to ``far_m`` ahead, from eye height. On a brink it is
    steep (-30° or lower: the drop is in view); set back on a flat top it is a few degrees, because the edge
    hides everything below it. Use a 1 m terrain model: a 20-30 m one rounds the brink off."""
    kx, ky = _metres_per_degree(person[1])
    heading = math.radians(heading_deg)
    ux, uy = math.sin(heading), math.cos(heading)
    z_eye = float(_sample(heights, transform, np.array([person[0]]), np.array([person[1]]))[0]) + eye_height_m
    along = np.arange(near_m, far_m, step_m)
    ground = _sample(heights, transform, person[0] + ux * along / kx, person[1] + uy * along / ky)
    return round(float(np.nanmax(np.degrees(np.arctan2(ground - z_eye, along)))), 1)
