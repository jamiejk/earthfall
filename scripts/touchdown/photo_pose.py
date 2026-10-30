"""Where a real photograph was taken, in the Blender map's coordinates (18 Oct, [stills] end_from_photo).

Battery Spencer v11: with the Golden Gate's OSM towers in the map, the skyline and the water fixed the end camera's
direction but not where on the headland the photographer stood, and the map's near tower ended behind the figure
while the photo had it in the left third (it swapped through a dissolve). Here the pose is solved from known 3D
points instead:

- prior: the photo's EXIF GPS position and altitude, compass direction and 35 mm-equivalent focal length when present;
- orientation: pitch and yaw from the far skyline (end_register.fit_camera), the zoom held near the EXIF lens;
- position: the OSM structures that carry a colour tag (the bridge, #f04a00) projected through the camera as solid
  prisms and matched to the same colour in the photo (overlap of the two masks), searched around the prior, then
  pitch/yaw/zoom refined locally on the same score. That is a dense form of PnP on the mapped towers.

The camera follows terrain_blender.py exactly: Blender's 60 degree horizontal field (``zoom`` scales the lens), the
guide's last pose turned by ``yaw`` about the world vertical and ``pitch`` about the camera's own x axis, and moved
by lift (up), back (against the heading) and side (to the right of the heading) without re-aiming.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 544
FOCAL = (W / 2) / math.tan(math.radians(30))


def exif_prior(path: Path) -> dict | None:
    from PIL import ExifTags, Image
    with Image.open(path) as im:
        ex = im.getexif()
        size = im.size
    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in ex.get_ifd(0x8825).items()}
    camera = ex.get_ifd(0x8769)
    out = {}
    if "GPSLatitude" in gps and "GPSLongitude" in gps:
        def deg(v, ref):
            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            return -d if ref in ("S", "W") else d
        out["latitude"] = deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        out["longitude"] = deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        if gps.get("GPSAltitude") is not None:
            out["altitude_m"] = float(gps["GPSAltitude"]) * (-1 if gps.get("GPSAltitudeRef") in (b"\x01", 1) else 1)
        if gps.get("GPSImgDirection") is not None:
            out["direction_deg"] = float(gps["GPSImgDirection"])
            out["direction_ref"] = "magnetic" if gps.get("GPSImgDirectionRef") == "M" else "true"
    f35 = camera.get(41989)
    if f35:
        aspect = size[0] / size[1]
        width = 43.27 * aspect / math.hypot(aspect, 1)  # 35 mm-equivalent is quoted on the diagonal
        out["focal_35mm"] = float(f35)
        out["hfov_deg"] = math.degrees(2 * math.atan(width / 2 / float(f35)))
    return out or None


def base_rotation(position: np.ndarray, aim: np.ndarray) -> np.ndarray:
    """Columns = camera x (right), y (up), z (backward) in world coordinates, as Blender's to_track_quat('-Z', 'Y')."""
    f = aim - position
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.column_stack((r, u, -f))


def rotation(r0: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    a, b = math.radians(yaw), math.radians(pitch)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rz @ r0 @ rx


def project(points: np.ndarray, centre: np.ndarray, rot: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    cam = (points - centre) @ rot  # world -> camera (rot columns are the camera axes)
    depth = -cam[:, 2]
    f = FOCAL * zoom
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack((W / 2 + f * cam[:, 0] / depth, H / 2 - f * cam[:, 1] / depth))
    return uv, depth


def prisms(structures: dict, ground) -> list[tuple]:
    """(bottom ring xyz, top ring xyz, structure id, kind) for the colour-tagged structures (bridges and their parts)."""
    out = []
    for item in structures["structures"]:
        if not (item.get("bridge") or item["kind"] == "deck"):
            continue
        ring = np.array(item["ring"], float)
        g = min(ground(x, y) for x, y in ring) - 0.5
        bottom = np.column_stack((ring, np.full(len(ring), g + item["min"])))
        top = np.column_stack((ring, np.full(len(ring), g + item["max"])))
        out.append((bottom, top, item["id"], item["kind"]))
    return out


def mask(shapes: list[tuple], centre: np.ndarray, rot: np.ndarray, zoom: float) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for bottom, top, *_ in shapes:
        pts = np.vstack((bottom, top))
        uv, depth = project(pts, centre, rot, zoom)
        if (depth <= 1).any():
            continue
        n = len(bottom)
        polys = [uv[n:], uv[:n]] + [np.array([uv[i], uv[(i + 1) % n], uv[n + (i + 1) % n], uv[n + i]]) for i in range(n)]
        cv2.fillPoly(m, [np.round(p).astype(np.int32) for p in polys], 1)
    return m.astype(bool)


def colour_mask(image: np.ndarray, hues: list[int]) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(int)
    near = np.zeros(image.shape[:2], bool)
    for h in hues:
        near |= np.minimum(abs(hsv[..., 0] - h), 180 - abs(hsv[..., 0] - h)) <= 18
    # a tagged colour in a real photo: in shadow or against the light the Golden Gate's #f04a00 (hue 9) reads deep
    # red (hue 176-179) and dark (value 46-76), so the hue band and the floors are wide
    return near & (hsv[..., 1] > 90) & (hsv[..., 2] > 30)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def tower_groups(shapes: list[tuple]) -> list[dict]:
    """Tower parts 50 m apart or more are separate towers."""
    groups = []
    for shape in shapes:
        if shape[3] != "tower":
            continue
        c = shape[0][:, :2].mean(axis=0)
        for g in groups:
            if np.linalg.norm(g["xy"] - c) < 50:
                g["shapes"].append(shape)
                break
        else:
            groups.append({"xy": c, "shapes": [shape]})
    return groups


def column_towers(m: np.ndarray, min_rows: int, min_cols: int, gap: int = 40) -> list[dict]:
    """Towers seen as runs of columns covered over at least ``min_rows`` rows: centre x, width and top row per
    cluster (columns closer than ``gap`` px belong to one tower: its two legs)."""
    counts = m.sum(axis=0)
    cols = np.nonzero(counts >= min_rows)[0]
    if not len(cols):
        return []
    runs, start = [], cols[0]
    for a, b in zip(cols[:-1], cols[1:]):
        if b - a > 1:
            runs.append((start, a))
            start = b
    runs.append((start, cols[-1]))
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_cols]
    clusters = []
    for r in runs:
        if clusters and r[0] - clusters[-1][1] <= gap:
            clusters[-1] = (clusters[-1][0], r[1])
        else:
            clusters.append(r)
    out = []
    for a, b in clusters:
        rows = np.nonzero(m[:, a:b + 1].sum(axis=1) >= max(1, (b - a + 1) // 4))[0]
        out.append({"x": (a + b) / 2, "width": float(b - a + 1), "top": int(rows.min()) if len(rows) else None})
    return out


def photo_towers(target: np.ndarray, solid: np.ndarray) -> list[dict]:
    """The photo's towers, found without the projection: wide ones on the mask without thin runs (cables and
    hangers share the colour), narrow far ones as 3+ adjacent tall columns of the plain mask (hangers are 1 px)."""
    wide = column_towers(solid, int(0.30 * H), 6)
    narrow = [t for t in column_towers(target, int(0.08 * H), 3, gap=6)
              if t["width"] <= 20 and all(abs(t["x"] - w["x"]) > w["width"] / 2 + 15 for w in wide)]
    return [{**t, "kind": "wide"} for t in wide] + [{**t, "kind": "narrow"} for t in narrow]


def tower_errors(groups: list[dict], seen: list[dict], centre: np.ndarray, rot: np.ndarray, zoom: float) -> list[dict]:
    """Per projected tower, the photo tower nearest to it (``seen``, from photo_towers): centre-column and width
    differences in pixels. A projected tower with no photo tower within 120 px is reported unmatched."""
    out = []
    for g in groups:
        m = mask(g["shapes"], centre, rot, zoom)
        if m.sum() < 30:
            out.append({"at_m": g["xy"].round(1).tolist(), "error_px": None, "reason": "out of view"})
            continue
        ys, xs = np.nonzero(m)
        x, width = (xs.min() + xs.max()) / 2, float(xs.max() - xs.min() + 1)
        cands = [t for t in seen if abs(t["x"] - x) <= 120]
        row = {"at_m": g["xy"].round(1).tolist(), "projected_x": round(x, 1), "projected_width_px": width}
        if not cands:
            out.append({**row, "error_px": None, "reason": "no photo tower near it"})
            continue
        t = min(cands, key=lambda t: abs(t["x"] - x) + abs(t["width"] - width))
        dx = x - t["x"]
        dw = width - t["width"] if min(width, t["width"]) > 20 else 0.0  # a narrow far tower: its legs' core only
        out.append({**row, "photo_x": t["x"], "photo_width_px": t["width"], "dx_px": round(dx, 1),
                    "dwidth_px": round(dw, 1), "error_px": round(math.hypot(dx, dw), 1)})
    return out


def solve(photo: np.ndarray, structures: dict, ground, guide_pose: dict, heading: float, fit: dict,
          prior_xy: tuple[float, float] | None, landing_xy: tuple[float, float], radius_m: float = 300.0,
          span_m: float = 20.0) -> dict:
    """The camera position, height and yaw that put the colour-tagged OSM towers where the photo shows them (pitch
    and zoom from the skyline/EXIF fit). Returns the pose as terrain_blender end-fit offsets and the per-tower errors."""
    hues = sorted({int(cv2.cvtColor(np.uint8([[[c[2] * 255, c[1] * 255, c[0] * 255]]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
                   for c in (s["rgb"] for s in structures["structures"] if s.get("bridge") or s["kind"] == "deck")})
    target = colour_mask(photo, hues)
    solid = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 5), np.uint8)).astype(bool)
    shapes = prisms(structures, ground)
    groups = tower_groups(shapes)
    seen = photo_towers(target, solid)
    if not groups or not seen:
        return {"applied": False, "reason": "no colour-tagged OSM tower, or none of its colour in the photo"}
    p0 = np.array(guide_pose["camera_xyz_m"], float)
    aim = np.array(guide_pose["aim_xyz_m"], float)
    r0 = base_rotation(p0, aim)
    h = math.radians(heading)
    fwd, right = np.array([math.sin(h), math.cos(h), 0.0]), np.array([math.cos(h), -math.sin(h), 0.0])
    pitch, zoom = fit["pitch_deg"], fit["zoom"]
    sx, sy = prior_xy if prior_xy else (p0[0], p0[1])

    def cost(x, y, z, yaw):
        errs = tower_errors(groups, seen, np.array([x, y, z]), rotation(r0, yaw, pitch), zoom)
        got = [e["error_px"] for e in errs if e.get("error_px") is not None]
        if len(got) < min(len(groups), len(seen)):  # fewer towers matched than the photo shows: heavy
            return 1e3 + 100 * (min(len(groups), len(seen)) - len(got)) + sum(got)
        return math.sqrt(sum(v * v for v in got) / len(got))

    best = None
    for step, span, yaws in ((4.0, span_m, np.arange(-4, 4.01, 1.0)), (1.0, 4.0, np.arange(-1, 1.01, 0.25))):
        cx, cy = (best[1], best[2]) if best else (sx, sy)
        cyaw = best[4] if best else fit["yaw_deg"]
        cdz = best[3] - ground(best[1], best[2]) if best else None
        for dx in np.arange(-span, span + 1e-6, step):
            for dy in np.arange(-span, span + 1e-6, step):
                x, y = cx + dx, cy + dy
                if math.hypot(x - landing_xy[0], y - landing_xy[1]) > radius_m:
                    continue
                g = ground(x, y)
                for dz in ((1.6, 4.0, 8.0) if cdz is None else (max(1.6, cdz - 1), cdz, cdz + 1)):
                    for dyaw in yaws:
                        c = cost(x, y, g + dz, cyaw + dyaw)
                        if best is None or c < best[0]:
                            best = (c, x, y, g + dz, cyaw + dyaw)
    c, x, y, z, yaw = best
    centre = np.array([x, y, z])
    rot = rotation(r0, yaw, pitch)
    errors = tower_errors(groups, seen, centre, rot, zoom)
    offset = centre - p0
    return {"applied": True, "rms_tower_error_px": round(c, 1), "x": round(x, 1), "y": round(y, 1), "z": round(z, 2),
            "height_above_ground_m": round(z - ground(x, y), 1),
            "distance_from_prior_m": round(math.hypot(x - sx, y - sy), 1) if prior_xy else None,
            "distance_from_landing_m": round(math.hypot(x - landing_xy[0], y - landing_xy[1]), 1),
            "yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2), "zoom": round(zoom, 3),
            "lift_m": round(float(offset[2]), 2), "back_m": round(float(-offset[:2] @ fwd[:2]), 2),
            "side_m": round(float(offset[:2] @ right[:2]), 2), "towers": errors, "photo_towers": seen,
            "structure_overlap": round(iou(mask(shapes, centre, rot, zoom), target), 3)}


def figure_mask(end: np.ndarray, photo: np.ndarray) -> np.ndarray | None:
    """The figure FLUX added: where the end still differs from the photo it kept (Lab distance over 30), the largest
    connected region in the middle 40% of the columns. Both images BGR at the same size."""
    a = cv2.cvtColor(end, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = (np.linalg.norm(a - b, axis=2) > 30).astype(np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(diff)
    w = end.shape[1]
    keep = [i for i in range(1, n) if 0.3 * w <= cents[i][0] <= 0.7 * w]
    if not keep:
        return None
    i = max(keep, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return labels == i


def figure_ground(end: np.ndarray, photo: np.ndarray, centre: np.ndarray, rot: np.ndarray, zoom: float, ground) -> dict:
    """With the solved camera: the figure's feet (bottom of its box) cast onto the DEM, and the height that implies.
    Images at W x H."""
    m = figure_mask(end, photo)
    if m is None:
        return {"found": False}
    ys, xs = np.nonzero(m)
    x0, y0, x1 = int(xs.min()), int(ys.min()), int(xs.max())
    # the feet: the lowest row before the mask widens past the figure (FLUX often adds a strip of ground to stand
    # her on, and that strip differs from the photo too)
    rows = np.arange(m.shape[0])
    widths = np.array([np.ptp(np.nonzero(r)[0]) + 1 if r.any() else 0 for r in m])
    top_half = widths[y0:y0 + max(1, (ys.max() - y0) // 2)]
    shoulders = float(np.max(top_half)) if len(top_half) else float(x1 - x0)
    body = rows[(widths > 0) & (widths <= 1.25 * shoulders) & (rows >= y0)]
    y1 = int(body.max()) if len(body) else int(ys.max())
    feet_u = float(np.median(np.nonzero(m[y1])[0]))
    f = FOCAL * zoom

    def ray(u, v):
        d = rot @ np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])
        return d / np.linalg.norm(d)

    d = ray(feet_u, y1)
    hit = None
    for t in np.arange(1.0, 600.0, 0.25):
        p = centre + t * d
        if p[2] <= ground(p[0], p[1]):
            hit = p
            break
    if hit is None:
        return {"found": True, "box": [x0, y0, x1, y1], "ground_point": None, "reason": "feet ray misses the DEM"}
    gz = ground(hit[0], hit[1])
    h = ray(feet_u, y0)
    dist = math.hypot(hit[0] - centre[0], hit[1] - centre[1])
    head_z = centre[2] + dist * h[2] / math.hypot(h[0], h[1])
    implied = float(head_z - gz)
    # where a 1.7 m figure stands on the feet ray: the horizontal distance at which head and feet rays are 1.7 m apart
    slope = d[2] / math.hypot(d[0], d[1]) - h[2] / math.hypot(h[0], h[1])
    d17 = 1.7 / -slope if slope < 0 else None
    coat = end[y0 + (y1 - y0) // 5:y0 + (y1 - y0) // 2][m[y0 + (y1 - y0) // 5:y0 + (y1 - y0) // 2]]
    rgb = (coat[:, ::-1].astype(float).mean(axis=0) / 255).round(3).tolist() if len(coat) else None
    out = {"found": True, "box": [x0, y0, x1, y1], "feet_px": [round(feet_u, 1), y1],
           "dem_hit": [round(float(hit[0]), 2), round(float(hit[1]), 2)], "dem_hit_distance_m": round(dist, 2),
           "implied_height_m": round(implied, 2), "coat_rgb": rgb}
    if 1.45 <= implied <= 2.0:
        return {**out, "placement": "on the DEM", "ground_point": out["dem_hit"], "base_z": round(gz, 2),
                "height_m": 1.7, "scale": 1.0}
    if implied > 2.0 and d17:
        # the DEM falls away below her: she stands on ground FLUX added (Battery Spencer: a ledge in front of the cliff
        # edge, where the feet ray reaches the water 350 m out). Keep the figure as drawn, at the 1.7 m scale on its ray
        k = d17 / math.hypot(d[0], d[1])
        p = centre + k * d
        return {**out, "placement": "on ground the still added (1.7 m scale on the feet ray)",
                "ground_point": [round(float(p[0]), 2), round(float(p[1]), 2)], "base_z": round(float(p[2]), 2),
                "height_above_dem_m": round(float(p[2] - ground(p[0], p[1])), 2), "distance_m": round(d17, 2),
                "height_m": 1.7, "scale": 1.0}
    # the figure is too small for where its feet meet the DEM: scale it to 1.7 m there
    return {**out, "placement": "on the DEM, figure rescaled", "ground_point": out["dem_hit"], "base_z": round(gz, 2),
            "height_m": 1.7, "scale": round(1.7 / max(implied, 0.1), 3)}


def rescale_figure(end_path: Path, photo_path: Path, box: list, scale: float, out: Path) -> None:
    """Scale the added figure about its feet onto the untouched photo (the space it vacates is the photo itself)."""
    end = cv2.imread(str(end_path))
    photo = cv2.resize(cv2.imread(str(photo_path)), (end.shape[1], end.shape[0]), interpolation=cv2.INTER_AREA)
    m = figure_mask(end, photo).astype(np.uint8)
    k = end.shape[1] / W
    fx, fy = (box[0] + box[2]) / 2 * k, box[3] * k
    a = np.float32([[scale, 0, fx * (1 - scale)], [0, scale, fy * (1 - scale)]])
    size = (end.shape[1], end.shape[0])
    moved = cv2.warpAffine(end, a, size, flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(cv2.warpAffine(m * 255, a, size, flags=cv2.INTER_LINEAR), (0, 0), 1.5)[..., None] / 255.0
    cv2.imwrite(str(out), (photo * (1 - alpha) + moved * alpha).astype(np.uint8))
