"""Pace a photographic descent and retain the native moving portrait as whole frames."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from touchdown.authored_scene import digest


class JoinMismatch(ValueError):
    pass


def match_metrics(a, b):
    """Measure the whole-frame seam. No image is warped, pasted or sharpened."""
    size = (640, 368)
    a, b = cv2.resize(a, size), cv2.resize(b, size)
    sift = cv2.SIFT_create(nfeatures=1600)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return dict(acceptable=False, reason="Insufficient photographic detail for a seam comparison")
    matches = [m[0] for m in cv2.BFMatcher().knnMatch(da, db, k=2)
               if len(m) == 2 and m[0].distance < .7 * m[1].distance]
    if len(matches) < 30:
        return dict(acceptable=False, reason="Too few independent scene features agree", matches=len(matches))
    left = np.float32([ka[m.queryIdx].pt for m in matches])
    right = np.float32([kb[m.trainIdx].pt for m in matches])
    h, mask = cv2.findHomography(left, right, cv2.RANSAC, 2.5)
    if h is None:
        return dict(acceptable=False, reason="Scene registration could not be measured")
    points = np.float32([[[0, 0], [640, 0], [0, 368], [640, 368], [320, 184]]])
    movement = np.linalg.norm(cv2.perspectiveTransform(points, h) - points, axis=2)
    pixel_change = float(np.abs(a.astype(float)-b).mean())
    inliers = int(mask.sum())
    displacement = float(movement.max())
    # Conservative operational limits, recorded for human inspection. They
    # aren't a calibrated perceptual continuity certificate.
    acceptable = inliers >= 30 and displacement <= 8 and pixel_change <= 16
    return dict(acceptable=acceptable, inliers=inliers, matches=len(matches),
                max_displacement_at_640px=round(displacement, 3), mean_pixel_change=round(pixel_change, 3))


def video_info(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS))


def frame_at(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    if not ok:
        raise ValueError(f"Could not decode photographic frame {frame}")
    return image


def hermite_indices(source_end, target_end, start_speed, end_speed):
    if source_end < target_end or target_end < 1:
        raise ValueError("Pacing must shorten, without duplicating frames")
    result = []
    for i in range(target_end+1):
        t = i/target_end
        position = ((-2*t**3+3*t*t)*source_end + (t**3-2*t*t+t)*target_end*start_speed
                    + (t**3-t*t)*target_end*end_speed)
        result.append(round(position))
    if result[0] != 0 or result[-1] != source_end or not all(a < b for a, b in zip(result, result[1:])):
        raise ValueError("Pacing would repeat or reverse frames")
    return result


def compose(output: Path, near_folder: Path, high_frame: int, target_seconds=None,
            descent_seconds: float = 3.0):
    """Reach the wide aerial early, descend, then use the real near video.

    The orbital section and the ground approach used to be fixed at 2.2 s and 1.5 s on screen, which
    capped every delivery at about 6.7 s once the portrait was added: a ten-second target was
    unreachable by construction, not by model behaviour. Given ``target_seconds`` the approach keeps
    its own frames up to ``descent_seconds`` and the orbital section takes whatever is left, so the
    film lands on the requested length. Without it the old fixed timings apply.

    Both inputs are photographic generations bound to the same arrival. The
    join is selected from their corresponding prefix frames; a mismatch holds
    the take rather than hiding it with a dissolve or face replacement.
    """
    source = output / "flight/earth_zoom.mp4"
    flight_record = json.loads((output / "flight/result.json").read_text())
    near_record = json.loads((near_folder / "result.json").read_text())
    native = Path(near_record["video"])
    if digest(source) != flight_record["descent_sha256"] or digest(native) != near_record["sha256"]:
        raise ValueError("A photographic source changed before delivery")
    for record in (flight_record, near_record):
        if digest(Path(record["workflow"])) != record["workflow_sha256"]:
            raise ValueError("A photographic source workflow changed")
    for part in flight_record.get("source_segments", []):
        if digest(Path(part["video"])) != part["sha256"] or digest(Path(part["workflow"])) != part["workflow_sha256"]:
            raise ValueError("A photographic flight segment changed")
    provenance = json.loads((output / "flight/provenance.json").read_text())["parent"]
    if provenance["near_video_sha256"] != near_record["sha256"]:
        raise ValueError("The native portrait is not the flight's verified parent")
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(native))
    try:
        total, width, height, fps = video_info(a)
        near_count, near_width, near_height, near_fps = video_info(b)
        if fps != 24 or near_fps != 24 or abs(width/height-near_width/near_height) > .002:
            raise ValueError("Native portrait and flight must share 24 fps and aspect ratio")
        flight_size = (width, height)
        width, height = near_width, near_height
        expected_start = total-near_count
        if not 0 < high_frame < expected_start:
            raise ValueError("The high aerial must precede the moving portrait")
        candidates = []
        for n in range(min(9, near_count-1)):
            metrics = match_metrics(frame_at(a, expected_start+n), frame_at(b, n))
            candidates.append(dict(native_frame=n, flight_frame=expected_start+n, **metrics))
        viable = [c for c in candidates if c["acceptable"]]
        if not viable:
            raise JoinMismatch("The native portrait does not join the flight closely enough; no dissolve was added")
        # Prefer an already moving near frame when its agreement is as good.
        selected = min(viable, key=lambda c: c["mean_pixel_change"] + c["max_displacement_at_640px"]*.5)
        join = selected["flight_frame"]
        lower_target = min(join-high_frame, max(1, round(descent_seconds*fps)))
        if target_seconds:
            want = round(sum(target_seconds) / 2 * fps)
            head = want - near_count - lower_target
            upper_target = min(high_frame, max(round(1.0*fps), head))
        else:
            upper_target = min(high_frame, max(1, round(2.2*fps)))
            lower_target = min(join-high_frame, max(1, round(1.5*fps)))
        upper = hermite_indices(high_frame, upper_target, high_frame/upper_target, (join-high_frame)/lower_target)
        lower = hermite_indices(join-high_frame, lower_target, (join-high_frame)/lower_target, 1.0)
        mapping = upper[:-1] + [high_frame+i for i in lower[:-1]]
        mapping += list(range(join, total))
        if mapping[0] != 0 or mapping[-1] != total-1 or not all(x < y for x, y in zip(mapping, mapping[1:])):
            raise ValueError("Delivery would repeat, reverse or omit a sequence endpoint")
        target = output / "earth_zoom.mp4"
        encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], stdin=subprocess.PIPE)
        a.set(cv2.CAP_PROP_POS_FRAMES, 0); b.set(cv2.CAP_PROP_POS_FRAMES, selected["native_frame"])
        index = -1
        try:
            for source_frame in mapping:
                if source_frame < join:
                    while index < source_frame:
                        ok, image = a.read(); index += 1
                        if not ok:
                            raise ValueError("Short flight source")
                else:
                    ok, image = b.read()
                    if not ok:
                        raise ValueError("Short native portrait source")
                if image.shape[1::-1] != (width, height):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError("Photographic delivery encoding failed")
        record = dict(source=str(source), source_sha256=digest(source), output_sha256=digest(target),
                      native_portrait=str(native), native_portrait_sha256=digest(native),
                      source_high_frame=high_frame, target_high_frame=upper_target,
                      native_start_frame=upper_target+lower_target, native_first_frame=selected["native_frame"],
                      native_frames=near_count-selected["native_frame"], join=selected, join_candidates=candidates,
                      flight_size=list(flight_size), delivered_size=[width, height],
                      flight_resize="whole-frame Lanczos" if flight_size != (width, height) else "none",
                      output_frames=len(mapping), fps=24, source_frame_mapping=mapping,
                      interpolation=False, crossfade=False, face_composite=False, final_portrait_speed=1.0)
        (output / "pacing.json").write_text(json.dumps(record, indent=2))
        return record
    finally:
        a.release(); b.release()
