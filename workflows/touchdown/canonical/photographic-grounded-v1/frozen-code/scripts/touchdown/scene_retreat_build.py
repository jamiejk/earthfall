"""Experimental full shot: photo scene, reversed retreat, computed dive, close-up.

One fixed seed per generated component; cache jobs, never rank generated takes.
The map/retreat blur join conceals detail, not a solved geographical alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import dive, video as V
from .build import Build
from .portrait_scene import extract_person


def crop_portrait(image: Image.Image, box: tuple[int, int, int, int]):
    """Crop chest/shoulders and nearby scene from an upright full-body box."""
    x0, y0, x1, y1 = box
    height = y1-y0
    ch = math.ceil(height*.48/9)*9
    cw = ch*16//9
    if ch < 36 or cw > image.width or ch > image.height:
        raise ValueError('Person framing cannot support a close-up crop')
    left = int(np.clip(round((x0+x1)/2-cw/2), 0, image.width-cw))
    top = int(np.clip(round(y0-height*.08), 0, image.height-ch))
    rect = (left, top, left+cw, top+ch)
    return image.crop(rect), rect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--comfy-python', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8191')
    a = p.parse_args()
    b = Build(a.manifest, None, a.url, False, a.comfy_python, unattended=True)
    shot = b.m['shot']
    endpoint = b.run('stills', (), b.stills)

    def job(stage, name, **kw):
        params = {'url': a.url, **kw}
        files = []
        for key in ('reference', 'identity'):
            if params.get(key): files.append(params[key])
        files.extend(str(pin[1]) for pin in params.get('pins', []))
        fingerprint = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}
        helper_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            'comfy_job': Path(__file__).with_name('comfy_job.py'),
            'fakeout_earth_zoom': Path(__file__).parent.parent/'fakeout_earth_zoom.py'}.items()}
        key = hashlib.sha256(json.dumps([name, params, fingerprint, helper_hashes],
            sort_keys=True).encode()).hexdigest()[:12]
        folder = b.root/stage/key
        done = folder/'result.json'
        if done.is_file():
            result = Path(json.loads(done.read_text())['path'])
            if result.is_file():
                print(f'[{stage}] cached {result}', flush=True)
                return result
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'parameters.json').write_text(json.dumps(params, indent=2))
        start = time.monotonic()
        result = Path(b.comfy(name, folder=str(folder), **kw))
        done.write_text(json.dumps({'path': str(result)}, indent=2))
        b.log(stage=stage, seed=kw.get('seed'), seconds=round(time.monotonic()-start, 1),
              candidate_count=1, path=str(result))
        return result

    close = b.root/'closeup'
    close.mkdir(parents=True, exist_ok=True)
    image = Image.open(endpoint).convert('RGB')
    _, box = extract_person(np.asarray(image))
    cropped, rect = crop_portrait(image, box)
    crop = close/'crop.png'
    cropped.save(crop)
    (close/'crop.json').write_text(json.dumps({'source': str(endpoint), 'person_box': box,
        'crop_box': rect, 'limitations': ['Mustard-jacket-specific extraction; not general identity detection']}, indent=2))
    upscaled = job('closeup-upscale', 'closeup_upscale', reference=str(crop))
    refined = job('closeup-refine', 'closeup_refine', reference=str(upscaled),
                  identity=shot['subject'], seed=b.m['closeup']['seed'])
    retreat = job('retreat', 'ltx', frames=b.m['retreat']['frames'], pins=[[0, str(endpoint)]],
                  prompt=b.m['retreat']['prompt'], seed=b.m['retreat']['seed'],
                  prefix=f"geoblock/touchdown_{b.name}_retreat")
    approach = b.root/'approach.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(retreat),
                    '-vf', 'reverse', '-an', '-c:v', 'libx264', '-threads', '2',
                    '-crf', '16', str(approach)], check=True)
    push = job('push-in', 'ltx', frames=b.m['closeup']['frames'],
               pins=[[0, str(endpoint)], [b.m['closeup']['frames']-1, str(refined)]],
               prompt=b.m['closeup']['prompt'], seed=b.m['closeup']['video_seed'],
               prefix=f"geoblock/touchdown_{b.name}_closeup")
    b.run('opener', ('stills',), b.opener)
    paths = dive.fetch(shot['latitude'], shot['longitude'], b.root/'imagery')

    def computed(folder):
        cfg = b.m['dive']
        facts = dive.render(paths, folder/'dive.mp4',
            **{k: cfg[k] for k in ('from_m', 'to_m', 'nadir_frames', 'pitch_frames', 'pitch_deg', 'clear', 'aim')},
            spin=shot['heading'])
        return folder/'dive.mp4', facts

    b.run('dive', ('stills',), computed,
          extra={'route': 'photo-scene-retreat', 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    b.run('cloud', ('opener', 'dive'), b.cloud)
    keep = int((b.out['cloud'].parent/'keep.txt').read_text())
    small = V.small(approach)
    motion = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    # Discard only leading near-static frames; no candidate selection or visual judge.
    skip = next((i for i, d in enumerate(motion[:48]) if d > 1.5), 0)
    join_cfg = b.m.get('join', {})
    span = int(join_cfg.get('blur_frames', 12))
    sigma = float(join_cfg.get('blur_sigma_px', 36))
    if span < 1 or sigma < 0: raise ValueError('Invalid blur join configuration')
    counts = {'opener': keep, 'cloud': V.count(b.out['cloud']), 'dive': V.count(b.out['dive']),
              'swoop': V.count(approach)-skip, 'end': V.count(push)-1}
    segments, cursor = {}, 0
    for name, count in counts.items():
        segments[name] = [cursor, cursor+count]
        cursor += count
    (b.root/'segments.json').write_text(json.dumps(segments, indent=2))
    cloud_exit = V.frame(b.out['cloud'], -1)
    silent = b.root/'silent.mp4'
    with V.Writer(silent) as writer:
        for i, im in enumerate(V.frames(b.out['opener'])):
            if i >= keep: break
            writer.write(im)
        writer.extend(V.frames(b.out['cloud']))
        for i, im in enumerate(V.frames(b.out['dive'])):
            if i < 12:
                weight = float(V.smoothstep(i/11))
                # Opaque cloud thins over the map; no image-image crossfade at landing.
                im = np.clip(cloud_exit*(1-weight)+im*weight, 0, 255).astype(np.uint8)
            distance = counts['dive']-1-i
            if distance < span and sigma:
                im = cv2.GaussianBlur(im, (0, 0), sigma*float(V.smoothstep(1-distance/span)))
            writer.write(im)
        for i, im in enumerate(V.frames(approach)):
            if i < skip: continue
            distance = i-skip
            if distance < span and sigma:
                im = cv2.GaussianBlur(im, (0, 0), sigma*float(V.smoothstep(1-distance/span)))
            writer.write(im)
        for i, im in enumerate(V.frames(push)):
            if i: writer.write(im)
    subprocess.run([sys.executable, '-m', 'touchdown.wind_bed', '--video', str(silent),
        '--segments', str(b.root/'segments.json'), '--out', str(b.root/'touchdown.mp4'),
        '--seed', str(b.m['sound']['wind_seed'])], check=True)
    result = V.small(silent)
    boundaries = {name: {'frame': bounds[0], 'mae_240px': float(np.abs(result[bounds[0]]-result[bounds[0]-1]).mean())}
                  for name, bounds in segments.items() if bounds[0]}
    (b.root/'audit.json').write_text(json.dumps({'frames': len(result), 'fps': V.FPS,
        'seconds': len(result)/V.FPS, 'boundaries': boundaries, 'approach_prefix_skipped': skip,
        'blur_sigma_px': sigma, 'blur_frames': span, 'one_seed_per_component': True,
        'limitations': ['Map and generated aerial scene are not geometrically calibrated',
                       'Blur is concealment, not proof of continuity',
                       'Identity and rigid buildings are not certified by pixel metrics']}, indent=2))
    (b.root/'report.md').write_text(
        f'# {b.name}: experimental full shot\n\n'
        f'{len(result)} frames, {len(result)/V.FPS:.2f} seconds, {V.FPS} fps.\n\n'
        'One fixed seed per generated component; no generated-take selection. '
        'The map/retreat blur join is not geometrically calibrated and remains experimental. '
        'See audit.json, segments.json and build_log.jsonl for boundaries and timings.\n\n'
        'Video: touchdown.mp4\n')
    print(b.root/'touchdown.mp4', flush=True)


if __name__ == '__main__':
    main()
