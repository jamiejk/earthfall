"""Deterministic inputs and landing choice for the measured-terrain route.

The landing scorer deliberately has a small search radius: the place catalogue
still decides *where* the shot is, while this step avoids putting the subject
on a car, a roof edge, or a steep patch immediately around that coordinate.
It is a rejection heuristic, not a semantic understanding of the scene.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer


def acquire(lat: float, lon: float, project: str, zone: int, cache: Path) -> tuple[Path, Path]:
    """Fetch the 1 m project and wider 1/3 arc-second USGS DEM into the normal raw cache.

    Outside the US ``zone`` is a WGS 84 / UTM EPSG code and ``project`` a geoblock_core.global_dem source id
    (swissalti3d, ign-rgealti, kartverket-dtm, linz-dem1m, gsi-dem5, glo30): the national near DEM and GLO-30 far."""
    from geoblock_core.models import GeoSceneSpec
    from geoblock_core.providers import USGS1mDEMProvider, USGS3DEPProvider
    from geoblock_core.utm import is_wgs84_utm, projected_crs, vertical_crs

    if is_wgs84_utm(zone):
        from geoblock_core import global_dem
        return global_dem.acquire(lat, lon, project, zone, cache)
    crs = projected_crs(zone)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)

    def spec(radius: float) -> GeoSceneSpec:
        return GeoSceneSpec(project_id="touchdown-terrain", name="Touchdown terrain",
                            longitude=lon, latitude=lat, projected_crs=crs,
                            vertical_crs=vertical_crs(zone), radius_m=radius,
                            origin_easting_m=x, origin_northing_m=y,
                            location_source="place catalogue")

    if project == "3dep13":  # no 3DEP 1 m project at the spot (Manhattan, 17 Oct): the ~10 m 1/3 arc-second DEM
        near, _ = USGS3DEPProvider(read_resolution_deg=0.00009).acquire(spec(400), cache)
    else:
        near, _ = USGS1mDEMProvider(project, zone).acquire(spec(400), cache)
    far, _ = USGS3DEPProvider(read_resolution_deg=0.0002).acquire(spec(17000), cache)
    return near, far


def _landing_candidates(terrain: Path, image: Path, radius_m: int,
                        *, varied_surface: bool = False, photo_op: bool = False,
                        natural: bool = False) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Return safe standing patches and the source arrays used to score them.

    The default caps want a neutral, grey-ish patch, which suits grass verges and plazas but rejects red desert
    sand and accepts car parks (Monument Valley, 17 Oct). ``natural`` drops the saturation cap and scores a
    neutral patch as worse, not better, so the person stands on coloured natural ground."""
    if not 1 <= radius_m <= 280:
        raise ValueError("landing search radius must be 1..280 m")
    height = np.load(terrain / "height.npy")
    source = cv2.imread(str(image))
    if source is None:
        raise ValueError(f"Could not read landing image: {image}")
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    if height.shape != (401, 401) or rgb.shape[:2] != (2048, 2048):
        raise ValueError("landing scorer expects 401x401 DEM and 2048x2048 NAIP 900 m image")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    candidates = []
    rejected = {"relief": 0, "brightness": 0, "saturation": 0, "texture": 0}
    for y in range(-radius_m, radius_m + 1, 2):
        for x in range(-radius_m, radius_m + 1, 2):
            if math.hypot(x, y) > radius_m:
                continue
            row, col = round(200 - y / 2), round(200 + x / 2)
            patch_h = height[row - 1:row + 2, col - 1:col + 2] if photo_op else height[row - 2:row + 3, col - 2:col + 3]
            px, py = round(1024 + x / 900 * 2048), round(1024 - y / 900 * 2048)
            sat = hsv[py - 7:py + 8, px - 7:px + 8, 1]
            lum = gray[py - 7:py + 8, px - 7:px + 8]
            relief = float(np.ptp(patch_h))
            brightness = float(lum.mean())
            saturation = float(sat.mean())
            texture = float(lum.std())
            saturation_cap, texture_cap = (75, 55) if varied_surface else (42, 35)
            brightness_low, brightness_high = (70, 215) if varied_surface else (80, 205)
            if natural:
                saturation_cap, texture_cap, brightness_low, brightness_high = 255, 55, 60, 230
            reason = ("relief" if relief > 1.2 else
                      "brightness" if not brightness_low <= brightness <= brightness_high else
                      "saturation" if saturation > saturation_cap else
                      "texture" if texture > texture_cap else None)
            if reason:
                rejected[reason] += 1
                continue
            colour = (0.03 * max(0.0, 60 - saturation) if natural  # neutral grey reads as pavement or roof
                      else (0.012 if varied_surface else 0.025) * saturation)
            score = (1.8 * relief + (0.016 if varied_surface or natural else 0.035) * texture
                     + colour
                     + 0.018 * abs(brightness - 165)
                     + (0.02 if varied_surface else 0.025) * math.hypot(x, y))
            candidates.append((score, x, y, relief, brightness, saturation, texture))
    if not candidates:
        raise ValueError("No plausible level landing patch near the catalogue coordinate "
                         f"(rejected within {radius_m} m: {rejected})")
    return sorted(candidates), height, gray


def choose_landing(terrain: Path, image: Path, *, radius_m: int = 150) -> dict:
    """Pick a nearby level, neutral, low-texture patch from the DEM and NAIP.

    The scorer uses only fixed measurements, so it cannot silently choose a
    different seed or ask a model to judge a stage. It checks the local standing
    patch; a drop outside that patch does not exclude an overlook. It refuses
    the place if no plausible patch exists; callers can then supply an
    explicit target or a different catalogue point.
    """
    candidates, _, _ = _landing_candidates(terrain, image, radius_m)
    score, x, y, relief, brightness, saturation, texture = min(candidates)
    return {"x": x, "y": y, "score": round(score, 3), "relief_m": round(relief, 2),
            "brightness": round(brightness, 1), "saturation": round(saturation, 1),
            "texture": round(texture, 1), "candidates": len(candidates)}


def large_pavement_mask(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Keep broad connected smooth surfaces at the fixed 900 m imagery scale."""
    lum = gray.astype(np.float32)
    mean = cv2.blur(lum, (25, 25))
    deviation = np.sqrt(np.maximum(0, cv2.blur(lum * lum, (25, 25)) - mean * mean))
    saturation = cv2.blur(hsv[:, :, 1].astype(np.float32), (25, 25))
    mask = ((saturation < 75) & (deviation < 14) & (mean > 80) & (mean < 185)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    return np.isin(labels, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 5000]).astype(np.uint8)


def pavement_corridor(hsv: np.ndarray, gray: np.ndarray, x: float, y: float, heading: float,
                      large_mask: np.ndarray | None = None) -> dict:
    """Check near-ground spans and an optional broad swath across the full 500 m route.

    This is a large-pavement heuristic, not a road segmentation model. Testing
    three lateral strips avoids treating a single pale rock as a parking lot.
    The caller can require a clean route for a natural-ground shot.
    """
    angle = math.radians(heading)
    dx, dy = math.sin(angle), math.cos(angle)
    rx, ry = math.cos(angle), -math.sin(angle)
    center, wide = [], []
    for distance in range(20, 221, 10):
        flags = []
        for lateral in (-12, 0, 12):
            sx, sy = x - dx * distance + rx * lateral, y - dy * distance + ry * lateral
            px, py = round(1024 + sx / 900 * 2048), round(1024 - sy / 900 * 2048)
            if not (12 <= px < 2036 and 12 <= py < 2036):
                flags.append(False)
                continue
            saturation = float(hsv[py - 12:py + 13, px - 12:px + 13, 1].mean())
            crop = gray[py - 12:py + 13, px - 12:px + 13]
            flags.append(saturation < 75 and float(crop.std()) < 14
                         and 80 < float(crop.mean()) < 185)
        center.append(flags[1])
        wide.append(sum(flags) >= 2)
    longest = max((len(run) for run in "".join("1" if flag else "0" for flag in center).split("0")), default=0)
    surface = {}
    clear = longest < 40 and sum(center) < 5 and sum(wide) < 5
    if large_mask is not None:
        rows, cols = np.mgrid[0:400, 0:120].astype(np.float32)
        distance = 10 + rows / 399 * 490
        lateral = (cols / 119 - 0.5) * 180
        sx, sy = x - dx * distance + rx * lateral, y - dy * distance + ry * lateral
        px, py = (1024 + sx / 900 * 2048).astype(np.float32), (1024 - sy / 900 * 2048).astype(np.float32)
        valid = (px >= 24) & (px < 2024) & (py >= 24) & (py < 2024)
        sampled = cv2.remap(large_mask, px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        fractions = [float(sampled[start:start + 100].mean()) for start in (0, 100, 200, 300)]
        coverage = float(valid.mean())
        surface = {"pavement_swath_fractions": [round(v, 3) for v in fractions],
                   "corridor_imagery_coverage": round(coverage, 3)}
        clear = clear and max(fractions) < 0.05 and coverage >= 0.85
    return {"pavement_center_samples": sum(center), "pavement_wide_samples": sum(wide),
            "pavement_longest_m": longest * 10,
            "pavement_clear": clear, **surface}


def _height_at(near: np.ndarray, far: np.ndarray):
    """Ground height at (x, y) metres from the origin: the 2 m near DEM within 396 m, the 50 m far DEM beyond."""
    def height(px: float, py: float) -> float:
        if abs(px) <= 396 and abs(py) <= 396:
            return float(near[round(200 - py / 2), round(200 + px / 2)])
        return float(far[int(np.clip(round(340 - py / 50), 0, 680)), int(np.clip(round(340 + px / 50), 0, 680))])
    return height


# the guide's final frame is 60 degrees wide (terrain_blender) and about 36 degrees tall, near level; ground that
# rises above this angle across most of the width leaves the frame without sky, and the end-still skyline gate
# then has nothing to check (Delicate Arch 18 Oct: the landing sat in the slickrock bowl, horizon 13-22 degrees up)
SKY_MAX_HORIZON_DEG = 12.0


def horizon_deg(near: np.ndarray, far: np.ndarray, x: float, y: float, heading: float) -> float:
    """Median elevation angle of the skyline across the guide's final frame (13 bearings over +-30 degrees), seen from
    the final camera: 10 m behind the landing on ``heading``, 2 m up. Terrain out to 3 km."""
    rad = math.radians(heading)
    cx, cy = x - math.sin(rad) * 10, y - math.cos(rad) * 10
    height = _height_at(near, far)
    eye = height(cx, cy) + 2.0
    steps = np.concatenate((np.arange(5.0, 400.0, 5.0), np.arange(400.0, 3001.0, 50.0)))
    angles = []
    for yaw in np.linspace(-30, 30, 13):
        t = math.radians(heading + yaw)
        dx, dy = math.sin(t), math.cos(t)
        angles.append(max(math.degrees(math.atan2(height(cx + dx * d, cy + dy * d) - eye, d)) for d in steps))
    return float(np.median(angles))


def view_clearance_deg(near: np.ndarray, far: np.ndarray, x: float, y: float, heading: float,
                       target: tuple[float, float, float]) -> float:
    """How far the top of the named view (``target`` = x, y metres from the origin, height of its top above the
    ground there) stands above the nearer ground, in degrees, seen from the guide's final camera: 10 m behind the
    landing on ``heading``, 2 m up (the catalogue's eye-level test, place_catalog WANTED_CLEARANCE_DEG). Battery Spencer
    18 Oct: the search moved the landing 24 m onto the battery, where a mound hid the bridge, and FLUX drew a view
    the guide never showed."""
    rad = math.radians(heading)
    cx, cy = x - math.sin(rad) * 10, y - math.cos(rad) * 10
    height = _height_at(near, far)
    eye = height(cx, cy) + 2.0
    tx, ty, top = target
    distance = math.hypot(tx - cx, ty - cy)
    goal = math.degrees(math.atan2(height(tx, ty) + top - eye, distance))
    blocking = -90.0
    for d in np.arange(5.0, max(distance - 50, 6.0), 5.0 if distance < 1000 else 10.0):
        f = d / distance
        blocking = max(blocking, math.degrees(math.atan2(height(cx + (tx - cx) * f, cy + (ty - cy) * f) - eye, d)))
    return goal - blocking


def search_views(terrain: Path, image: Path, *, heading: float | None = None,
                 radius_m: int = 150, limit: int = 8, varied_surface: bool = False,
                 avoid_pavement: bool = False, photo_views: list[dict] | None = None,
                 photo_anchor_radius_m: float = 10, natural: bool = False,
                 view_target: tuple[float, float, float] | None = None, min_clearance_deg: float = 4.0) -> list[dict]:
    """Search cheap landing/heading pairs before rendering a guide or model job.

    This is a conservative raster preflight, not a semantic road detector. It
    favours an open, level final approach and, for an automatic heading, a
    visible distant terrain drop ahead of the subject. The selected landing
    and camera heading are returned together so the camera route stays straight.
    """
    if heading is not None and not 0 <= heading < 360:
        raise ValueError("heading must be in [0, 360) degrees or automatic")
    if limit < 1:
        raise ValueError("view search limit must be positive")
    candidates, near, gray = _landing_candidates(terrain, image, radius_m,
                                                 varied_surface=varied_surface, photo_op=bool(photo_views),
                                                 natural=natural)
    if photo_views:
        if not 5 <= photo_anchor_radius_m <= 40:
            raise ValueError("Photo anchor neighbourhood must be 5..40 m")
        candidates = [candidate for candidate in candidates if min(
            math.hypot(candidate[1] - photo["x"], candidate[2] - photo["y"]) for photo in photo_views) <= photo_anchor_radius_m]
        if not candidates:
            raise ValueError(f"No level standing patch within {photo_anchor_radius_m:g} m of a photo camera anchor")
    aerial_hsv = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2HSV) if avoid_pavement else None
    broad_pavement = large_pavement_mask(aerial_hsv, gray) if avoid_pavement else None
    far = np.load(terrain / "height_far.npy")
    if far.shape != (681, 681):
        raise ValueError("view scorer expects a 681x681 far DEM")
    # Diversity prevents hundreds of adjacent pixels on one patch from
    # crowding out other viable viewpoints.
    landings = []
    for candidate in candidates:
        if all(math.hypot(candidate[1] - old[1], candidate[2] - old[2]) >= 18 for old in landings):
            landings.append(candidate)
        if len(landings) == 64:
            break
    headings = [heading] if heading is not None else list(range(0, 360, 15))
    scores = []
    for standing in landings:
        base, x, y, relief, brightness, saturation, texture = standing
        local_height = float(near[round(200 - y / 2), round(200 + x / 2)])
        for angle in headings:
            rad = math.radians(angle)
            dx, dy = math.sin(rad), math.cos(rad)
            # The camera flies in from behind the landing, ending 10 m short.
            # Check the visible final 180 m, where the 2D texture is exposed.
            samples = []
            for distance in range(10, 181, 10):
                sx, sy = x - dx * distance, y - dy * distance
                row, col = round(200 - sy / 2), round(200 + sx / 2)
                px, py = round(1024 + sx / 900 * 2048), round(1024 - sy / 900 * 2048)
                if not (2 <= row < 399 and 2 <= col < 399 and 5 <= px < 2043 and 5 <= py < 2043):
                    break
                patch = near[row - 2:row + 3, col - 2:col + 3]
                crop = gray[py - 5:py + 6, px - 5:px + 6]
                samples.append((float(np.ptp(patch)), float(crop.std()), float(crop.mean())))
            if len(samples) != 18:
                continue
            reliefs, textures, brightnesses = np.asarray(samples).T
            # Too much foreground relief makes the last metres look like a
            # cliff or a stretched 2D texture; abrupt image changes indicate
            # boundaries the generator may turn into a new surface.
            if np.max(reliefs) > 8:
                continue
            corridor = (0.18 * float(np.mean(reliefs)) + 0.025 * float(np.mean(textures))
                        + 0.008 * float(np.mean(np.abs(np.diff(brightnesses)))))
            drop = 0.0
            if heading is None:
                distant = []
                for distance in (1000, 3000):
                    sx, sy = x + dx * distance, y + dy * distance
                    row, col = round(340 - sy / 50), round(340 + sx / 50)
                    distant.append(float(far[row, col]))
                drop = max(0.0, local_height - min(distant))
            score = base + corridor - min(drop, 1000) * 0.002
            if photo_views:
                reference_cost = min(0.03 * math.hypot(x - photo["x"], y - photo["y"]) +
                    (abs((angle - photo["heading_deg"] + 180) % 360 - 180) / 90
                     if photo.get("heading_deg") is not None else 0) for photo in photo_views)
                score += reference_cost
            clearance = view_clearance_deg(near, far, x, y, angle, view_target) if view_target else None
            scores.append((score, standing, angle, corridor, drop, clearance))
    if not scores:
        raise ValueError("No level, image-supported straight approach near the catalogue coordinate")
    if view_target:
        # keep only views that see the named view over the nearer ground; if none does, the clearest ones
        clear = [row for row in scores if row[5] >= min_clearance_deg]
        scores = clear or sorted(scores, key=lambda row: -row[5])[:max(limit, 4)]
    chosen, lowest = [], 90.0
    for score, standing, angle, corridor, drop, clearance in sorted(scores, key=lambda row: (row[0], row[1][1], row[1][2], row[2])):
        _, x, y, relief, brightness, saturation, texture = standing
        horizon = horizon_deg(near, far, x, y, angle)
        lowest = min(lowest, horizon)
        if horizon > SKY_MAX_HORIZON_DEG:
            continue
        surface = pavement_corridor(aerial_hsv, gray, x, y, angle, broad_pavement) if aerial_hsv is not None else None
        if surface is not None and not surface["pavement_clear"]:
            continue
        if sum(math.hypot(x - old["x"], y - old["y"]) < 18 for old in chosen) >= 2:
            continue
        if any(math.hypot(x - old["x"], y - old["y"]) < 18
               and abs((angle - old["heading"] + 180) % 360 - 180) < 25 for old in chosen):
            continue
        chosen.append({"x": x, "y": y, "heading": float(angle), "score": round(score, 3),
                       "standing_score": round(standing[0], 3), "corridor_score": round(corridor, 3),
                       "distant_drop_m": round(drop), "relief_m": round(relief, 2),
                       "brightness": round(brightness, 1), "saturation": round(saturation, 1),
                       "texture": round(texture, 1), "landings_checked": len(landings),
                       "views_checked": len(scores), "horizon_deg": round(horizon, 1),
                       **(surface or {}),
                       **({"view_clearance_deg": round(clearance, 2)} if clearance is not None else {})})
        if photo_views:
            photo = min(photo_views, key=lambda p: math.hypot(x - p["x"], y - p["y"]))
            chosen[-1]["photo_anchor"] = {"page_id": photo["page_id"],
                "camera_distance_m": round(math.hypot(x - photo["x"], y - photo["y"]), 1),
                "reported_heading_deg": photo.get("heading_deg")}
        if len(chosen) == limit:
            break
    if not chosen:
        if lowest > SKY_MAX_HORIZON_DEG:
            raise ValueError(f"No landing near here keeps sky in the final view: the ground ahead rises above "
                             f"{SKY_MAX_HORIZON_DEG:g} degrees (lowest median skyline {lowest:.1f} degrees)")
        raise ValueError("No safe view has a clear natural-ground approach corridor")
    return chosen


def choose_view(terrain: Path, image: Path, *, heading: float | None = None,
                radius_m: int = 150, avoid_pavement: bool = False, natural: bool = False,
                view_target: tuple[float, float, float] | None = None) -> dict:
    """The best geometry-only landing/view pair (no embedding model required)."""
    return search_views(terrain, image, heading=heading, radius_m=radius_m, limit=1,
                        avoid_pavement=avoid_pavement, natural=natural, view_target=view_target)[0]


def terrain_meta(folder: Path) -> dict:
    return json.loads((folder / "terrain_guide.json").read_text())
