"""Objective failure checks for a guided terrain approach.

These catch selected structural substitutions, a missing end pin, or a
dissolve/held segment. They do not certify subject identity or full 3D geometry.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import video as V


def assess_surface_layout(endpoint: Path, surface_mask: Path) -> dict:
    """Reject a disappeared mapped paved surface outside the inserted person.

    This checks neutral surface retention and nearby boundary edges. It cannot
    certify a railing, exact materials, or the full 3D shape.
    """
    image = cv2.imread(str(endpoint))
    mask = cv2.imread(str(surface_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError('Shared surface endpoint or mask is missing')
    image = cv2.resize(image, (960, 544))
    mask = (cv2.resize(mask, (960, 544), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    mask[:240] = 0
    mask[:, 400:560] = 0
    mask[:12] = mask[-12:] = 0
    mask[:, :12] = mask[:, -12:] = 0
    pixels = int(mask.sum())
    if pixels < 1000:
        return {'assessed': False, 'reason': 'Too little unoccluded mapped surface visible', 'warnings': []}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    neutral = float(((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 65))[mask.astype(bool)].mean())
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0
    boundary[:, 398:562] = False
    boundary[238:242] = False
    boundary[:14] = boundary[-14:] = False
    boundary[:, :14] = boundary[:, -14:] = False
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5,5), 0), 40, 90)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_support = float((distance[boundary] <= 10).mean()) if boundary.any() else None
    warnings = []
    if neutral < 0.55:
        warnings.append('mapped paved standing surface became different terrain')
    if edge_support is not None and edge_support < 0.5:
        warnings.append('mapped path boundaries disappeared or moved')
    return {'assessed': True, 'visible_pixels': pixels, 'neutral_fraction': round(neutral,3),
            'boundary_support': round(edge_support,3) if edge_support is not None else None,
            'warnings': warnings}


def ground_patch(still: Path, box: tuple | None, band_m: float = 1.6) -> dict:
    """The colour of the ground under the figure, against the ground beside it.

    The end still is painted from the outline map, which carries no ground texture, so the model invents the
    ground around the person. Monument Valley v14 showed the invention as a soft orange oval around the figure
    (the review of 20 Oct). This measures it: the patch directly under the figure against a reference band at the
    same height further out. It is a diagnostic, not a gate: real desert ground is saturated too.
    """
    image = cv2.imread(str(still))
    if image is None:
        raise ValueError("could not read the still for the ground check")
    image = cv2.resize(image, (960, 544), interpolation=cv2.INTER_AREA)
    if box is None:
        return {"assessed": False, "reason": "no person found in the still"}
    x0, y0, x1, y1 = box
    height = max(1, y1 - y0)
    feet = min(543, y1)
    row = feet - int(height * 0.08)  # just above the feet
    under = image[max(0, row - int(height * 0.25)):feet + 1,
                  max(0, x0 - int((x1 - x0) * 0.6)):min(959, x1 + int((x1 - x0) * 0.6))]
    left = image[max(0, row - int(height * 0.25)):feet + 1, 0:max(1, x0 - int((x1 - x0) * 0.6))]
    right = image[max(0, row - int(height * 0.25)):feet + 1, min(960, x1 + int((x1 - x0) * 0.6)):960]
    if under.size == 0 or left.size == 0 or right.size == 0:
        return {"assessed": False, "reason": "the figure is at the edge of the frame"}
    reference = np.concatenate([left.reshape(-1, 3), right.reshape(-1, 3)])

    def stats(patch: np.ndarray) -> tuple[float, float]:
        hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        return float(hsv[:, 1].mean()), float(hsv[:, 0].mean())

    sat_under, hue_under = stats(under)
    sat_ref, hue_ref = stats(reference)
    # hue is circular: 0 and 179 are neighbours
    hue_delta = abs(hue_under - hue_ref)
    hue_delta = min(hue_delta, 180.0 - hue_delta)
    saturated = sat_under - sat_ref > 30 and hue_delta > 10
    return {"assessed": True,
            "saturation_under_figure": round(sat_under, 1),
            "saturation_beside": round(sat_ref, 1),
            "hue_delta": round(hue_delta, 1),
            "patched_ground": bool(saturated),
            "note": "a strongly coloured patch of ground under the figure (the 'landing pad' artefact)" if saturated
                    else "the ground under the figure matches the ground beside it"}


def assess_endpoint_layout(guide: Path, endpoint: Path) -> dict:
    """Catch a high-confidence road/pavement-to-cliff endpoint substitution.

    This deliberately narrow gate compares the lower foreground outside the
    central person area. It does not certify topology or fine 3D geometry;
    those require semantic masks or registered landmarks.
    """
    reference = cv2.imread(str(guide))
    result = cv2.imread(str(endpoint))
    if reference is None or result is None:
        raise ValueError("Could not read guide or endpoint for layout check")
    reference = cv2.resize(reference, (960, 544))
    result = cv2.resize(result, (960, 544))

    def neutral_foreground(bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Only the foreground is inspected; the person and their shadow may
        # legitimately change the centre of an edited endpoint.
        lower = hsv[300:520]
        bands = np.concatenate((lower[:, :330], lower[:, 630:]), axis=1)
        return float(np.mean((bands[:, :, 1] < 35) & (bands[:, :, 2] > 90)))

    before = neutral_foreground(reference)
    after = neutral_foreground(result)
    warnings = []
    if before > 0.70 and after < 0.35:
        warnings.append("neutral paved foreground in the guide became coloured terrain in the endpoint")
    largest_changed = 0.0
    if before > 0.70:
        guide_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        end_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        neutral_guide = (guide_hsv[:, :, 1] < 35) & (guide_hsv[:, :, 2] > 90)
        neutral_end = (end_hsv[:, :, 1] < 35) & (end_hsv[:, :, 2] > 90)
        changed = (neutral_guide & ~neutral_end)[300:520].astype(np.uint8)
        changed[:, 450:510] = 0  # the inserted person can cover the measured ground
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        components, _, stats, _ = cv2.connectedComponentsWithStats(changed)
        if components > 1:
            largest_changed = float(stats[1:, cv2.CC_STAT_AREA].max() / changed.size)
        if largest_changed > 0.05:
            warnings.append("a large new foreground surface replaced measured pavement")
    # Geometrically matched ridge/landscape features provide a viewpoint check
    # even when FLUX changes colour and adds detail. The guide's foreground is
    # often too smooth for feature matching, so this never certifies the ground.
    feature_mask = np.zeros((544, 960), np.uint8)
    feature_mask[120:320] = 255
    feature_mask[120:320, 350:610] = 0
    sift = cv2.SIFT_create(nfeatures=2500)
    points_a, desc_a = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), feature_mask)
    points_b, desc_b = sift.detectAndCompute(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), feature_mask)
    aligned = 0
    median_shift = None
    if desc_a is not None and desc_b is not None and len(desc_a) >= 2 and len(desc_b) >= 2:
        pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
        matches = [first for first, second in pairs if first.distance < 0.75 * second.distance]
        if len(matches) >= 12:
            source = np.float32([points_a[m.queryIdx].pt for m in matches])
            target = np.float32([points_b[m.trainIdx].pt for m in matches])
            _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4)
            if inliers is not None:
                kept = inliers.ravel().astype(bool)
                aligned = int(kept.sum())
                if aligned >= 12:
                    median_shift = float(np.median(np.linalg.norm(source[kept] - target[kept], axis=1)))
                    if median_shift > 20:
                        warnings.append("distant landscape landmarks moved relative to the measured camera")
    # Too few matches is not a pass: when the guide's landscape band has plenty of features but the still matches
    # almost none of them, the distant landmarks were redrawn or lost (Grand Canyon v5's third seed passed this way,
    # 17 Oct). A featureless guide (haze, plain sky) still passes, since there is nothing to measure; a guide with
    # a few features is the case that cannot be verified (Battery Spencer 18 Oct: 2 features, a mound filled the
    # view, and an unrelated FLUX view passed and ghosted through the join); redrawing the still cannot fix it.
    guide_features = 0 if desc_a is None else len(desc_a)
    if 0 < guide_features < 60:
        warnings.append(f"end still unverifiable: map frame has {guide_features} landscape features")
    elif guide_features == 0:
        pass  # nothing in the map frame to check the still against
    elif median_shift is None:
        warnings.append(f"distant landscape landmarks moved or vanished: {aligned} aligned matches "
                        f"of {guide_features} guide features, too few to verify the camera")
    return {"guide_neutral_foreground": round(before, 3),
            "endpoint_neutral_foreground": round(after, 3),
            "largest_changed_foreground": round(largest_changed, 3),
            "landmark_inliers": aligned,
            "landmark_median_shift_px": round(median_shift, 2) if median_shift is not None else None,
            "landmark_guide_features": guide_features,
            "warnings": warnings}


def assess(clip: Path, endpoint: Path, window: int = 10, expected_frames: int = 81, tail_ok: bool = False) -> dict:
    """Windowed motion checks and end-pin convergence. With ``tail_ok`` (a free first/last approach), still windows
    at the very end are the camera having arrived early, not a dissolve: they are reported as ``settled_at`` (the
    first frame of the trailing hold) instead of warnings, so the caller can cut the dead hold."""
    import fade_check

    rows = np.asarray(fade_check.measure(str(clip)), dtype=np.float32)
    if expected_frames < 2 or len(rows) != expected_frames - 1:
        raise ValueError(f"guided approach must contain {expected_frames} frames, got {len(rows) + 1}")
    blocks = []
    for start in range(0, len(rows), window):
        flow, change, explained, texture = rows[start:start + window].mean(axis=0)
        blocks.append({"frames": [start, min(start + window, len(rows))],
                       "flow": round(float(flow), 2), "change": round(float(change), 2),
                       "explained": round(float(explained), 2), "texture": round(float(texture), 1)})
    last = cv2.resize(cv2.cvtColor(V.frame(clip, -1), cv2.COLOR_RGB2GRAY), (240, 136))
    pinned = cv2.cvtColor(np.asarray(Image.open(endpoint).convert("RGB").resize((240, 136))), cv2.COLOR_RGB2GRAY)
    correlation = float(np.corrcoef(last.ravel(), pinned.ravel())[0, 1])
    mae = float(np.abs(last.astype(np.float32) - pinned).mean())
    settled_at = None
    if tail_ok:
        tail = len(blocks)
        while tail > 1 and blocks[tail - 1]["flow"] < 0.5:
            tail -= 1
        if tail < len(blocks):
            settled_at = blocks[tail]["frames"][0]
            blocks_checked = blocks[:tail]
        else:
            blocks_checked = blocks
    else:
        blocks_checked = blocks
    warnings = [f"dissolve or still at frames {b['frames']}" for b in blocks_checked
                if b["texture"] > 100 and (b["explained"] < 0.3 or b["flow"] < 0.5)]
    if not np.isfinite(correlation) or correlation < 0.7 or mae > 35:
        warnings.append("LTX did not converge to the pinned end image")
    return {"windows": blocks, "end_correlation": round(correlation, 3),
            "end_mae": round(mae, 1), "settled_at": settled_at, "warnings": warnings}


def assess_surface_sequence(clip: Path, masks: Path) -> dict:
    """Diagnostic sample of shared surfaces; thresholds are not temporally calibrated."""
    from tempfile import TemporaryDirectory
    samples = []
    with TemporaryDirectory(prefix='touchdown-surface-') as directory:
        for index in (0, 16, 32, 48, 64, 80):
            mask = masks / f'{index + 1:04d}.png'
            if not mask.is_file():
                raise ValueError(f'Missing shared surface mask: {mask}')
            image = Path(directory) / f'{index}.png'
            Image.fromarray(V.frame(clip, index)).save(image)
            samples.append({'frame': index, **assess_surface_layout(image, mask)})
    return {'mode': 'diagnostic; not a calibrated temporal gate', 'samples': samples,
            'flagged_frames': [s['frame'] for s in samples if s['warnings']]}


def assess_photo_geometry(guide: Path, still: Path, max_residual: float = 12.0, min_water_iou: float = 0.5,
                          min_columns: float = 0.4) -> dict:
    """The end-still landmark test for [stills] end_from_photo (18 Oct). SIFT finds nothing in common between the
    smooth DEM + aerial render and a real photograph even where they visibly line up (Battery Spencer v10: 0 of 115
    at a 3 px skyline fit), so a photo end still is checked geometrically against the fitted guide instead: the
    sky/terrain boundary per column (the person's columns ignored) must agree within ``max_residual`` px over at
    least ``min_columns`` of the frame, and where either image shows water below the skyline (over 5%), the water
    layouts must overlap by ``min_water_iou``. Returns the numbers and the warnings, in the same form as
    assess_endpoint_layout."""
    from .end_register import H, W, person_box, skyline
    g = cv2.resize(cv2.imread(str(guide)), (W, H), interpolation=cv2.INTER_AREA)
    e = cv2.resize(cv2.imread(str(still)), (W, H), interpolation=cv2.INTER_AREA)
    box = person_box(e)
    pad = (box[2] - box[0]) // 6 if box else 0
    gl, el = skyline(g), skyline(e, (box[0] - pad, box[2] + pad) if box else None)
    ok = np.isfinite(gl) & np.isfinite(el)
    residual = float(np.median(np.abs(gl[ok] - el[ok]))) if ok.any() else None

    def water(im, line):
        f = im.astype(int)
        rows = np.arange(H)[:, None]
        below = rows > np.where(np.isfinite(line), line, H)[None, :] + 4
        mask = (f[..., 0] > f[..., 2] + 12) & below
        if box:
            mask[:, max(box[0] - pad, 0):box[2] + pad] = False
        return mask

    gw, ew = water(g, gl), water(e, el)
    has_water = max(gw.mean(), ew.mean()) > 0.05
    iou = float((gw & ew).sum() / max((gw | ew).sum(), 1)) if has_water else None
    warnings = []
    if ok.mean() < min_columns:
        warnings.append(f"distant landscape landmarks unverifiable: skyline found in only {ok.mean():.0%} of columns")
    elif residual > max_residual:
        warnings.append(f"distant landscape landmarks moved: skyline differs by {residual:.1f} px")
    if iou is not None and iou < min_water_iou:
        warnings.append(f"distant landscape landmarks moved: water below the skyline overlaps {iou:.2f}")
    return {"check": "photo geometry", "skyline_residual_px": round(residual, 2) if residual is not None else None,
            "skyline_columns": round(float(ok.mean()), 3), "water_iou": round(iou, 3) if iou is not None else None,
            "guide_water": round(float(gw.mean()), 3), "still_water": round(float(ew.mean()), 3), "warnings": warnings}
