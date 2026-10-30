"""Chronological comparison and unchanged-tail audit for an opening-only repair."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def frames(path):
    cap = cv2.VideoCapture(str(path))
    result = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        result.append(cv2.resize(image, (640, 368)))
    cap.release()
    return result


def tile(image, label):
    image = cv2.resize(image, (384, 221))
    header = np.zeros((30, 384, 3), dtype=np.uint8)
    cv2.putText(header, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def expansion(seq, n):
    detector = cv2.SIFT_create(nfeatures=2000)
    a, b = [cv2.cvtColor(seq[i], cv2.COLOR_BGR2GRAY) for i in (0, n)]
    mask = np.zeros_like(a)
    mask[100:] = 255  # Ground region; exclude the sky and horizon from source matches.
    ka, da = detector.detectAndCompute(a, mask)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None or len(db) < 2:
        return dict(status='unclear')
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in pairs if x.distance < .7*y.distance]
    if len(good) < 10:
        return dict(status='unclear', matches=len(good))
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or int(inliers.sum()) < 10:
        return dict(status='unclear', matches=len(good))
    return dict(status='tracked', elapsed_seconds=n/24, inliers=int(inliers.sum()),
        apparent_surface_scale=float(np.hypot(matrix[0, 0], matrix[1, 0])),
        note='Two-dimensional feature expansion; does not by itself prove physical parallax.')


def main(root):
    plan = json.loads((root / 'experiment.json').read_text())
    original = frames(plan['source'])
    changed = frames(root / 'opening-rush-preview.mp4')
    assert len(original) == len(changed) == 196
    errors = [float(np.abs(a.astype(float)-b).mean()) for a, b in zip(original[40:], changed[40:])]
    assert max(errors) < 2, 'Later footage differs by more than expected re-encoding noise'
    board = [np.hstack([tile(seq[n], f'{name} {n/24:.3f}s') for n in (0, 4, 8, 12, 16)])
             for name, seq in [('Original', original), ('Rush', changed)]]
    cv2.imwrite(str(root / 'opening-comparison.jpg'), np.vstack(board))
    indexes = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39, 40)
    tiles = [tile(changed[n], f'frame {n} / {n/24:.3f}s') for n in indexes]
    cv2.imwrite(str(root / 'opening-chronology.jpg'), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, 12, 4)]))
    graph = json.loads(Path(json.loads((root / 'generation-result.json').read_text())['workflow']).read_text())
    baseline_result = Path(plan['source']).parent / 'flight/parent/result.json'
    base_graph = json.loads(Path(json.loads(baseline_result.read_text())['workflow']).read_text())
    kinds = {'UNETLoader', 'CLIPLoader', 'VAELoader'}
    def models(g):
        return {k: n['inputs'] for k, n in g.items() if n['class_type'] in kinds}
    assert models(graph) == models(base_graph)
    classes = [n['class_type'] for n in graph.values()]
    assert 'TouchdownPromptRelay' in classes
    assert not {'LTXICLoRALoaderModelOnly', 'LTXAddVideoICLoRAGuide'} & set(classes)
    audit = dict(retained_tail_frames=156, frame_numbers_and_timing_unchanged=True,
        tail_mean_absolute_pixel_error=float(np.mean(errors)), tail_max_frame_mean_error=max(errors),
        interpretation='Tail matches source at identical frame numbers within H.264 re-encoding differences',
        models_match_source=True, prompt_relay=True, layout_guidance=False,
        early_surface_expansion={name: [expansion(seq, n) for n in (4, 8, 12)]
                                 for name, seq in [('original', original), ('candidate', changed)]},
        semantic_camera_motion='Requires visual review; no automated pass inferred from cloud motion')
    (root / 'review-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
