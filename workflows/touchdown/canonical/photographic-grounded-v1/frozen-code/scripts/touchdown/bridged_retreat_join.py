"""Replace a map/landing cut with an LTX bridge and a close-up-origin retreat.

Only stream existing generated frames; no image zoom, dissolve or blur ramp.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import video as V


def timeline(base, prefix_frames, bridge_frames, approach_frames, end_frames=48):
    if not base['dive'][0] < prefix_frames <= base['dive'][1]:
        raise ValueError('Prefix must end within the computed dive')
    if min(bridge_frames, approach_frames) < 2 or end_frames < 1:
        raise ValueError('Generated clips and living-end span must be nonempty')
    total = prefix_frames+bridge_frames-1+approach_frames-1
    segments = {key: [a, min(b, prefix_frames)] for key, (a, b) in base.items()
                if key in ('opener', 'cloud', 'dive') and a < prefix_frames}
    segments['swoop'] = [prefix_frames, total]
    segments['end'] = [max(prefix_frames+bridge_frames-1, total-end_frames), total]
    return segments, total


def kept_indices(frames, step):
    if frames < 2 or step < 1:
        raise ValueError('Bridge frame count and step must be positive')
    indices = list(range(0, frames, step))
    if indices[-1] != frames-1:
        indices.append(frames-1)
    return indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--approach', type=Path, required=True)
    p.add_argument('--prefix-frames', type=int, required=True)
    p.add_argument('--bridge-step', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    base = json.loads((a.base/'segments.json').read_text())
    bridge_indices = kept_indices(V.count(a.bridge), a.bridge_step)
    segments, total = timeline(base, a.prefix_frames, len(bridge_indices), V.count(a.approach))
    (a.output/'segments.json').write_text(json.dumps(segments, indent=2))
    silent = a.output/'silent.mp4'
    with V.Writer(silent) as writer:
        for i, frame in enumerate(V.frames(a.base/'touchdown.mp4')):
            if i >= a.prefix_frames: break
            writer.write(frame)
        for clip in (a.bridge, a.approach):
            keep = set(bridge_indices) if clip == a.bridge else None
            for i, frame in enumerate(V.frames(clip)):
                if i and (keep is None or i in keep):
                    writer.write(frame)  # shared endpoint already appears in previous clip
    if V.count(silent) != total:
        raise ValueError('Assembled frame count differs from timeline')
    subprocess.run([sys.executable, '-m', 'touchdown.wind_bed', '--video', str(silent),
        '--segments', str(a.output/'segments.json'), '--out', str(a.output/'touchdown.mp4'),
        '--seed', '7'], check=True)
    frames = V.small(silent)
    boundaries = [a.prefix_frames, a.prefix_frames+len(bridge_indices)-1]
    (a.output/'join-audit.json').write_text(json.dumps({'frames': total, 'seconds': total/V.FPS,
        'joins': [{'frame': i, 'mae_240px': float(np.abs(frames[i]-frames[i-1]).mean())}
                  for i in boundaries], 'sources': {key: str(getattr(a, key))
                  for key in ('base', 'bridge', 'approach')}, 'prefix_frames': a.prefix_frames,
        'digital_zoom': False, 'crossfade': False, 'blur_join': False,
        'bridge_indices': bridge_indices, 'bridge_step': a.bridge_step,
        'limitations': ['Small endpoint differences do not certify the generated bridge interior',
                       'Generated city geometry and identity need visual assessment']}, indent=2))
    print(a.output/'touchdown.mp4', flush=True)


if __name__ == '__main__':
    main()
