"""Track one edited face plate onto the approach, then release it to LTX's living end.

The detector finds the figure automatically; no face position or per-frame paint is supplied by a person.
Only the interior of the face is blended. Its hair, body and the scene remain the video frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import video as V

DETECTOR = Path(__file__).resolve().parents[2] / "assets/models/face_detection_yunet_2023mar.onnx"


def _detector(size: tuple[int, int]):
    if not DETECTOR.is_file():
        raise FileNotFoundError(f"YuNet face detector is missing: {DETECTOR}")
    return cv2.FaceDetectorYN.create(str(DETECTOR), "", size, score_threshold=0.5)


def _face(detector, image: np.ndarray):
    _, hits = detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if hits is None:
        return None
    width = image.shape[1]
    candidates = [f for f in hits if abs(f[0] + f[2] / 2 - width / 2) < width * 0.2 and f[2] >= 15]
    return max(candidates, key=lambda f: float(f[-1])) if candidates else None


def extract_reference(video: Path, frame_index: int, output: Path) -> dict:
    """Crop around the automatically detected late-approach face for FLUX editing."""
    image = V.frame(video, frame_index)
    return extract_plate(image, output, frame=frame_index)


def extract_plate(image: np.ndarray, output: Path, frame: int | None = None) -> dict:
    """Crop a square plate around the largest centred face in an image.

    Used for the generated frame and for the subject's own photograph: the photograph is sharp, while a face in a
    generated 1920-wide frame is about 50 px across, so pasting that back can only soften it.
    """
    face = _face(_detector((image.shape[1], image.shape[0])), image)
    if face is None:
        return {"detected": False, "frame": frame}
    side = round(max(face[2], face[3]) * 4.7)
    side = min(side, image.shape[1], image.shape[0])
    x = int(np.clip(round(face[0] + face[2] / 2 - side / 2), 0, image.shape[1] - side))
    y = int(np.clip(round(face[1] + face[3] / 2 - side / 2), 0, image.shape[0] - side))
    Image.fromarray(image[y:y + side, x:x + side]).save(output)
    return {"detected": True, "frame": frame, "box": [round(float(v), 1) for v in face[:4]],
            "crop": [x, y, side]}


def extract_photo_plate(photo: Path, output: Path) -> dict:
    """The subject's own face, as a plate to warp onto the approach.

    The app's photo is a real portrait: its face is sharp and it is the person the clip is about. A face cropped out
    of a generated frame is tiny and soft, so pasting it back cannot add the detail the shot is missing.
    """
    image = np.asarray(Image.open(photo).convert("RGB"))
    found = extract_plate(image, output)
    found["source"] = str(photo)
    return found


def _smooth(points: np.ndarray) -> np.ndarray:
    """Five-tap temporal smoothing removes subpixel detector jitter before warping the face."""
    padded = np.pad(points, ((2, 2), (0, 0), (0, 0)), mode="edge")
    return sum(weight * padded[i:i + len(points)] for i, weight in enumerate((1, 4, 6, 4, 1))) / 16


def render(source: Path, plate_path: Path, output: Path, end_start: int, *, lead: int = 14,
           ramp: int = 12, hold_after: int = 8, fade: int = 16, strength: float = 0.65) -> dict:
    """Blend a tracked facial interior only where the approach becomes close enough to need it.

    A face pasted from another picture carries that picture's light, and the seam across the cheeks is what makes
    it look stitched on. So the plate is matched to the frame it goes into (its mean and spread in Lab) before the
    warp, and the mask is a soft ellipse well inside the face.
    """
    plate = np.asarray(Image.open(plate_path).convert("RGB"))
    ref = _face(_detector((plate.shape[1], plate.shape[0])), plate)
    if ref is None:
        raise ValueError("FLUX face plate contains no detectable centered face")
    ref_points = ref[4:14].reshape(5, 2)
    mask = np.zeros(plate.shape[:2], np.float32)
    centre = (round(ref[0] + ref[2] / 2), round(ref[1] + ref[3] / 2))
    cv2.ellipse(mask, centre, (round(ref[2] * 0.45), round(ref[3] * 0.46)), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 18)

    first, last = end_start - lead, end_start + hold_after + fade
    detector = _detector((V.W, V.H))
    found: dict[int, np.ndarray] = {}
    boxes: dict[int, np.ndarray] = {}
    for i, image in enumerate(V.frames(source)):
        if i < first:
            continue
        if i > last:
            break
        face = _face(detector, image)
        if face is not None:
            found[i] = face[4:14].reshape(5, 2)
            boxes[i] = np.array([face[0] + face[2] / 2, face[1] + face[3] / 2, face[2]], np.float32)
    # A detection that breaks from its neighbours' track is a false face in the scenery, not her: Grand Canyon v7
    # f162-163 found a 570 px "face" in the canyon (hers was 36 px) and blended a giant ghost face into the sky
    dropped = []
    ids = sorted(found)
    for k, i in enumerate(ids):
        near = [boxes[j] for j in ids[max(0, k - 3):k + 4] if j != i]
        if len(near) < 2:
            continue
        cx, cy, w = np.median(np.stack(near), axis=0)
        x, y, wi = boxes[i]
        if not 0.6 * w <= wi <= 1.6 * w or np.hypot(x - cx, y - cy) > 1.5 * w + 8:
            dropped.append(i)
    for i in dropped:
        found.pop(i)
    if not found:
        raise ValueError("no face detected in the approach")
    # the patch starts where her face is first actually found (with a long ``lead`` that can be well before the
    # end, while she is still small: MV v11's face was a smear from 4 s) and is feathered in over ``ramp`` frames;
    # nothing is blended before the first detection (no extrapolated position: the Grand Canyon ghost face)
    first_seen = min(found)
    if len(found) < (last - first_seen + 1) * 0.6:
        raise ValueError(f"face detected in only {len(found)} of {last - first_seen + 1} approach frames")
    ids = np.array(sorted(found))
    points = np.stack([found[i] for i in ids])
    dense = np.empty((last - first + 1, 5, 2), np.float32)
    for landmark in range(5):
        for axis in range(2):
            dense[:, landmark, axis] = np.interp(np.arange(first, last + 1), ids,
                                                 points[:, landmark, axis])
    dense = _smooth(dense)

    with V.Writer(output) as writer:
        for i, image in enumerate(V.frames(source)):
            if first <= i <= last:
                matrix, _ = cv2.estimateAffinePartial2D(ref_points, dense[i - first], method=cv2.LMEDS)
                if matrix is None:
                    raise ValueError(f"could not align edited face at frame {i}")
                patch = cv2.warpAffine(plate, matrix, (V.W, V.H), flags=cv2.INTER_AREA)
                # the plate carries its own light: match it to the frame before it goes in, or the face reads as
                # a patch rather than part of the picture
                patch = V.match_lab(patch, image)
                alpha = cv2.warpAffine(mask, matrix, (V.W, V.H), flags=cv2.INTER_LINEAR)
                gain = strength * V.smoothstep((i - first_seen) / max(1, ramp)) if i >= first_seen else 0.0
                gain *= 1 - V.smoothstep((i - end_start - hold_after) / max(1, fade))
                alpha = (alpha * gain)[..., None]
                image = np.clip(image * (1 - alpha) + patch * alpha, 0, 255).astype(np.uint8)
            writer.write(image)
    return {"detected_frames": len(found), "window": [first, last], "first_seen": int(first_seen), "strength": strength,
            "dropped_frames": dropped}
