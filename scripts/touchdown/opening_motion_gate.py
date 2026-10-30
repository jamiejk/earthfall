"""Reject a confidently stationary surface before cloud obscuration.

This is a negative check, not proof of physical camera travel. A changed image,
feature expansion or insufficient matches cannot by themselves earn a pass.
"""
from pathlib import Path
import json


def compare(first, later):
    import cv2
    import numpy as np
    sift = cv2.SIFT_create(nfeatures=3000)
    k0, d0 = sift.detectAndCompute(first, None)
    k1, d1 = sift.detectAndCompute(later, None)
    if d0 is None or d1 is None or len(d1) < 2:
        return dict(stationary=False, usable=False, reason='Insufficient features')
    pairs = cv2.BFMatcher().knnMatch(d0, d1, k=2)
    good = [a for pair in pairs if len(pair) == 2
            for a, b in [pair] if a.distance < .7*b.distance
            and k0[a.queryIdx].pt[1] > first.shape[0]*90/368]
    if len(good) < 20:
        return dict(stationary=False, usable=False, matches=len(good))
    src = np.float32([k0[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    # A fixed RNG makes the diagnostic reproducible across repeated reviews.
    cv2.setRNGSeed(0)
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None:
        return dict(stationary=False, usable=False, matches=len(good))
    keep = mask.ravel() > 0
    inliers = int(keep.sum())
    scale = float(np.hypot(*matrix[:, 0]))
    movement = float(np.median(np.linalg.norm(dst[keep]-src[keep], axis=1)))
    # Require broad spatial support so one small stationary patch cannot decide.
    points = src[keep]
    span = np.ptp(points, axis=0) / [first.shape[1], first.shape[0]]
    supported = (inliers >= 20 and inliers/len(good) >= .65
                 and span[0] >= .35 and span[1] >= .25)
    return dict(stationary=bool(supported and abs(scale-1) < .01 and movement < 1),
                usable=bool(supported), matches=len(good), inliers=inliers,
                scale=scale, median_displacement_pixels=movement,
                support_span=span.tolist())


def check(video, folder):
    import cv2
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    frames = {}
    try:
        for index in (0, 4, 8, 12):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Cannot decode opening motion evidence')
            frame = cv2.resize(frame, (640, 368))
            cv2.imwrite(str(folder/f'frame-{index:04}.png'), frame)
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()
    observations = {str(i): compare(frames[0], frames[i]) for i in (4, 8, 12)}
    stationary = observations['4']['stationary'] and observations['8']['stationary']
    reason = ('Matched surface features remain stationary through the first eight frames; '
              'cloud obscuration is not evidence of camera approach.' if stationary else
              'No confident stationary opening detected. This does not establish camera '
              'travel, parallax or freedom from fades; chronological review is still required.')
    result = dict(version='opening-static-surface-v1', status='fail' if stationary else 'unclear',
                  reason=reason, observations=observations, review_width=640,
                  auto_pass=False)
    record = folder/'response.json'
    record.write_text(json.dumps(result, indent=2)+'\n')
    return result, record
