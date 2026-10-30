"""Register a FLUX end still onto the terrain guide's last frame by its skyline (17 Oct).

FLUX 3 keeps the kind of distant terrain but not its scale and position: Grand Canyon v5 drew the canyon ~15% smaller,
Monument Valley v6 set the horizon at ~40% of the frame where the guide has it at 54%. One similarity transform
(uniform scale >= 1 and an x/y shift, no rotation) for the whole still, applied as zoom-in + crop, so nothing is
warped around the person and nothing is padded. It is fitted from the per-column sky/terrain boundary of both images
(sky mask against sky mask: the guide's smooth texture gives SIFT nothing to hold on to). A fit that needs scale < 1,
a crop outside the frame, or a crop that cuts the person's box is not applied; the end-still gate then decides as before.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H = 960, 544  # the guide frame and the gate's working size


def sky_mask(image: np.ndarray) -> np.ndarray:
    """Bright, blue or neutral pixels (sky and cloud), not yellow/red/green terrain; BGR in."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16)
    return (lab[..., 2] < 132) & (lab[..., 0] > 140)


def skyline(image: np.ndarray, ignore: tuple[int, int] | None = None) -> np.ndarray:
    """Per column, the first row where the sky ends (8 terrain rows in a row); NaN where the top is not sky or for
    columns in ``ignore`` (x0, x1: the person)."""
    sky = sky_mask(image)
    terrain = ~sky
    run = cv2.erode(terrain.astype(np.uint8), np.ones((8, 1), np.uint8), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    line = np.full(image.shape[1], np.nan)
    for x in range(image.shape[1]):
        if not sky[:6, x].all():
            continue
        hits = np.flatnonzero(run[:, x])
        if len(hits) and hits[0] < image.shape[0] - 8:
            line[x] = hits[0]
    if ignore is not None:
        line[max(ignore[0], 0):ignore[1]] = np.nan
    return line


def fit(guide_line: np.ndarray, end_line: np.ndarray, scales=np.arange(0.8, 1.3601, 0.01), step: int = 2,
        height: int = H, inside_only: bool = False) -> dict:
    """Crop window (x0, y0, scale) of the end image whose skyline best matches the guide's: guide pixel (u, v) is
    end pixel (x0 + u / s, y0 + v / s). y0 is the median offset over columns for each (s, x0); the score is the
    median absolute residual, plus a small preference for less zoom and a centred window (a flat rim or plain
    horizon fixes only the vertical offset, Grand Canyon / Monument Valley 17 Oct). ``inside_only``: only crops
    inside the frame (scale >= 1). Returns the best fit with its residual and the number of columns used."""
    n = len(guide_line)
    u = np.arange(n)
    good_u = ~np.isnan(guide_line)
    best = None
    for s in scales:
        span = n / s
        if inside_only and s < 1:
            continue
        xs = (np.arange(0.0, n - span + 1e-6, step) if inside_only else
              np.arange(min(0.0, n - span) - 0.25 * n, max(0.0, n - span) + 0.25 * n + 1, step))
        for x0 in xs:
            x = np.round(x0 + u / s).astype(int)
            inside = (x >= 0) & (x < n)
            e = np.full(n, np.nan)
            e[inside] = end_line[x[inside]]
            ok = good_u & ~np.isnan(e)
            if ok.sum() < 0.25 * n:
                continue
            offsets = e[ok] - guide_line[ok] / s
            y0 = float(np.median(offsets))
            if inside_only and not (-0.5 <= y0 <= height - height / s + 0.5):
                continue
            residual = float(np.median(np.abs(offsets - y0)))
            score = (residual + 40.0 * (ok.sum() < 0.4 * n) + 2.0 * abs(s - 1)
                     + 2.0 * abs(x0 - (n - span) / 2) / n)
            if best is None or score < best["score"]:
                best = {"scale": round(float(s), 3), "x0": float(x0), "y0": y0, "residual_px": round(residual, 2),
                        "columns": int(ok.sum()), "score": score}
    return best or {}


def person_box(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Standing figure's box (x0, y0, x1, y1) from the YuNet face: hair above, about nine heads down, three and a half wide."""
    from .face_refine import _detector, _face
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = _face(_detector((image.shape[1], image.shape[0])), rgb)
    if face is None:
        return None
    x, y, w, h = (float(v) for v in face[:4])
    cx = x + w / 2
    return (int(cx - 1.8 * h), int(y - 0.5 * h), int(cx + 1.8 * h), int(min(y + 9.2 * h, image.shape[0])))


def register(guide: Path, end: Path, out: Path, max_scale: float = 1.35, max_residual: float = 12.0) -> dict:
    """Write the registered still to ``out`` when the fit is usable; return what was fitted and why it was (not) used."""
    g = cv2.resize(cv2.imread(str(guide)), (W, H), interpolation=cv2.INTER_AREA)
    full = cv2.imread(str(end))
    e = cv2.resize(full, (W, H), interpolation=cv2.INTER_AREA)
    box = person_box(e)
    if box is None:
        return {"applied": False, "reason": "no person found to keep in the crop"}
    pad = (box[2] - box[0]) // 6
    gl, el = skyline(g), skyline(e, (box[0] - pad, box[2] + pad))
    free = fit(gl, el)  # what the skylines alone ask for, logged
    fitted = fit(gl, el, scales=np.arange(1.0, max_scale + 1e-4, 0.01), inside_only=True)
    if not fitted:
        return {"applied": False, "reason": "no zoom + crop inside the frame matches the skylines",
                "unconstrained": {k: v for k, v in free.items() if k != "score"}, "person_box": list(box)}
    fitted.pop("score")
    s, x0, y0 = fitted["scale"], fitted["x0"], fitted["y0"]
    report = {**fitted, "person_box": list(box), "unconstrained": {k: v for k, v in free.items() if k != "score"}}
    span_w, span_h = W / s, H / s
    if s < 1.0 - 1e-6:
        return {**report, "applied": False, "reason": "needs scale < 1 (the still is zoomed in past the guide)"}
    if s > max_scale + 1e-6:
        return {**report, "applied": False, "reason": f"needs scale > {max_scale}"}
    if fitted["residual_px"] > max_residual:
        return {**report, "applied": False, "reason": "skylines do not agree under one zoom + shift"}
    if x0 < -0.5 or y0 < -0.5 or x0 + span_w > W + 0.5 or y0 + span_h > H + 0.5:
        return {**report, "applied": False, "reason": "the crop would leave the frame"}
    if box[0] < x0 or box[1] < y0 or box[2] > x0 + span_w or box[3] > y0 + span_h:
        return {**report, "applied": False, "reason": "the crop would cut the person"}
    if s < 1.005 and abs(x0) < 1 and abs(y0) < 1:
        return {**report, "applied": False, "reason": "already aligned"}
    k = full.shape[1] / W  # back to the still's own resolution
    X0, Y0 = max(x0 * k, 0), max(y0 * k, 0)
    crop = full[int(round(Y0)):int(round(Y0 + span_h * k)), int(round(X0)):int(round(X0 + span_w * k))]
    cv2.imwrite(str(out), cv2.resize(crop, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_LANCZOS4))
    return {**report, "applied": True}


FOCAL = (W / 2) / np.tan(np.radians(30))  # the guide's 60-degree horizontal FOV, in pixels


def fit_camera(guide_line: np.ndarray, end_line: np.ndarray, pitch=(-8.0, 8.0), yaw=(-6.0, 6.0),
               zoom=(0.75, 1.39)) -> dict:
    """The end camera the still implies, the other way round from ``fit``: pitch up / yaw left (degrees) and zoom
    (lens ratio; 0.75-1.39 is the guide's 60-degree FOV +-25%) that carry the guide's skyline, taken as far away,
    onto the still's. Coarse grid, then a finer one around the best. Score: median absolute residual (px) plus a
    small preference for less change."""
    cx, cy = W / 2, H / 2
    u = np.flatnonzero(~np.isnan(guide_line))
    phi = np.arctan((u - cx) / FOCAL)
    theta = np.arctan((guide_line[u] - cy) / FOCAL)

    def score(p, y, z):
        f = FOCAL * z
        x = np.round(cx + f * np.tan(phi + np.radians(y))).astype(int)
        inside = (x >= 0) & (x < W)
        e = end_line[x[inside]]
        ok = ~np.isnan(e)
        if ok.sum() < 0.25 * W:
            return None
        r = np.abs(cy + f * np.tan(theta[inside][ok] + np.radians(p)) - e[ok])
        return float(np.median(r)), int(ok.sum())

    best = None
    for grid in ((2.0, 2.0, 0.04), (0.25, 0.25, 0.01)):
        dp, dy, dz = grid
        if best is None:
            ps, ys, zs = (np.arange(pitch[0], pitch[1] + 1e-6, dp), np.arange(yaw[0], yaw[1] + 1e-6, dy),
                          np.arange(zoom[0], zoom[1] + 1e-6, dz))
        else:
            ps = np.clip(np.arange(best["pitch_deg"] - 2, best["pitch_deg"] + 2 + 1e-6, dp), *pitch)
            ys = np.clip(np.arange(best["yaw_deg"] - 2, best["yaw_deg"] + 2 + 1e-6, dy), *yaw)
            zs = np.clip(np.arange(best["zoom"] - 0.04, best["zoom"] + 0.04 + 1e-6, dz), *zoom)
        for p in np.unique(ps):
            for y in np.unique(ys):
                for z in np.unique(zs):
                    got = score(p, y, z)
                    if got is None:
                        continue
                    s = got[0] + 0.1 * abs(p) + 0.1 * abs(y) + 4 * abs(z - 1)
                    if best is None or s < best["score"]:
                        best = {"pitch_deg": round(float(p), 2), "yaw_deg": round(float(y), 2),
                                "zoom": round(float(z), 3), "residual_px": round(got[0], 2), "columns": got[1],
                                "score": s}
    if best:
        best.pop("score")
    return best or {}
