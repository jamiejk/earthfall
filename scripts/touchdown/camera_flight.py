"""Reproducible single-pass Earth-camera experiment for any saved photographic scene.

prepare -> render -> review -> assemble. No stage silently changes the seed,
reference image, model files or recipe when it resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import cv2

from touchdown import flight_regression as regression, split_flight
from touchdown.authored_scene import digest
from touchdown.flight_sequence import review
from touchdown.photo_delivery import JoinMismatch, match_metrics, video_info
from touchdown.photo_gate import require_arrival_gate
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.recipe_snapshot import capture as capture_recipe

PRESET = Path(__file__).with_name('presets') / 'earth-camera-v1.json'
read, write = regression.read, regression.write


def graph_snapshot(graph):
    # Match geoblock_core.run._submit's graph persistence: SDK Asset objects
    # remain live for submission and get string representations in the audit.
    return json.loads(json.dumps(graph, default=str))


def fingerprint(path):
    """Stream large weights; do not load a 14 GB file into RAM to hash it."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    stat = path.stat()
    return dict(path=str(path.resolve()), sha256=h.hexdigest(), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def flight_prompt(cfg, preset):
    w = split_flight.words(cfg)
    ft = preset['review']['regional_altitude_ft']
    metres = preset['review']['regional_altitude_m']
    if 'prompt_template' in preset:
        return preset['prompt_template'].format(**w, altitude_ft=f'{ft:,}', altitude_m=f'{metres:,}')
    return (
        f"One continuous live-action Earth zoom out camera flight from this exact integrated travel photograph at {w['place']}. "
        "Begin with a gentle half-second physical retreat, then accelerate BACKWARD AND UPWARD along one heading. "
        f"Preserve the person and the same connected {w['support']}. Preserve the landmark and its surroundings as rigid "
        "objects. The person shrinks with increasing camera distance. Rise out over the wider countryside and shoreline "
        f"of {w['region']}, with natural {w['palette']} colours, {w['light']}. "
        f"By the FOURTH SECOND reach {ft:,} FEET, about {metres:,} METRES above the ground. "
        "For the next second keep climbing while clearly showing a vast broad regional view covering tens of kilometres. "
        "Individual people are invisible and the original landmark is microscopic. Show connected mountain ranges, "
        "water basins, valleys and settlement patterns appropriate to the photographed region, not enlarged buildings. "
        "There are NO foreground clouds obscuring this broad regional interval. "
        "Only AFTER establishing this altitude physically enter the high cloud deck. Spend the NEXT TWO SECONDS "
        "travelling through deep, opaque, billowing cloud: near cloud edges sweep past the lens with strong parallax, "
        "cloud depth surrounds the camera and briefly fills the whole frame. The land never fades into cloud by opacity. "
        "Break out above cloud tops and continue climbing rapidly until the broad curved planetary limb and black space "
        "are visible for the LAST TWO SECONDS. Clouds shrink into weather systems over continental terrain. "
        "One scene at every distance, fixed focal length, physical perspective and parallax throughout. "
        "No separate views dissolving into each other, no fade, no wipe, no cut, no optical crop zoom, no static-photo animation."
    )


def prepare(source, root, model_roots, preset_path=PRESET):
    regression.check_root(source, root)
    if (root / 'acceptance.json').exists():
        raise ValueError('Approved runs are frozen; choose a new root')
    cfg, preset = read(source / 'scene-config.json'), read(preset_path)
    near = split_flight._checked(read(source / 'motion/near/result.json'))
    verify_lineage(cfg, source / 'scene', asset_plan(cfg, source / 'scene'))
    if read(source / 'motion/near/provenance.json')['parent']['arrival_sha256'] != digest(source / 'scene/arrival.png'):
        raise ValueError('Portrait belongs to another arrival')
    sources = {str(p): digest(p) for p in regression.source_files(source, cfg, near)}
    code = [Path(__file__), Path(review.__code__.co_filename), Path(split_flight.__file__).with_name('photo_delivery.py'),
            Path(split_flight.__file__).parents[1] / 'fakeout_earth_zoom.py', Path(capture_recipe.__code__.co_filename)]
    identity = dict(source=str(source), source_hashes=sources, preset=preset, preset_sha256=digest(preset_path),
                    code_hashes={str(p):digest(p) for p in code}, seed=cfg['seed'] + preset['seed_offset'])
    path = root / 'experiment.json'
    if path.exists():
        existing = read(path)
        if existing['identity'] != identity:
            raise ValueError('Inputs, code or preset changed; choose a fresh root')
        for record in existing['model_fingerprints'].values():
            stat = Path(record['path']).stat()
            if stat.st_size != record['size'] or stat.st_mtime_ns != record['mtime_ns']:
                raise ValueError('Model weights changed since preparation')
        capture_recipe(root)
        return existing
    preparation = root / 'preparation.json'
    if root.exists() and any(root.iterdir()):
        if not preparation.exists() or read(preparation)['identity'] != identity:
            raise ValueError('New experiment root must be empty or contain this matching preparation')
    root.mkdir(parents=True, exist_ok=True)
    write(preparation, dict(identity=identity))
    require_arrival_gate(cfg, source / 'scene', root)
    cache_path = root / 'model-fingerprints.json'
    models = read(cache_path) if cache_path.exists() else {}
    for role, spec in preset['models'].items():
        matches = [r / spec['kind'] / spec['name'] for r in model_roots if (r / spec['kind'] / spec['name']).is_file()]
        if not matches:
            raise ValueError(f"Installed model not found for {role}: {spec['name']}; supply --model-root")
        stat = matches[0].stat()
        previous = models.get(role)
        if previous and (previous['path'],previous['size'],previous['mtime_ns']) == (str(matches[0].resolve()),stat.st_size,stat.st_mtime_ns):
            continue
        print(f'Fingerprinting {role}: {spec["name"]}', flush=True)
        models[role] = fingerprint(matches[0])
        write(cache_path, models)
    manifest = dict(identity=identity, config=cfg, native_portrait=near, model_fingerprints=models,
                    prompt=flight_prompt(cfg, preset), input_boundary='Arrival image only; no orbital or high-aerial still pins',
                    order=list(preset['review']['required_order']), exact_gpu_bitwise_reproduction='not guaranteed across hardware/kernel versions')
    write(path, manifest)
    capture_recipe(root)
    write(root / 'status.json', dict(stage='prepared', status='ready'))
    return manifest


def encode(frames, target, size, fps):
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-threads', '4', '-crf', '17', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait():
        raise RuntimeError('Video encoding failed')


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                return frames
            frames.append(image)
    finally:
        cap.release()


def render(root, manifest):
    from touchdown import studio_build as studio
    import fakeout_earth_zoom as f
    preset, cfg = manifest['identity']['preset'], manifest['config']
    target = root / 'flight-forward.mp4'
    if (root / 'result.json').exists():
        record = split_flight._checked(read(root / 'result.json'))
    else:
        command = os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
        if preset['deterministic'] and '--deterministic' not in command.split():
            command += ' --deterministic'
        paths = dict(is_default=True)
        for role, spec in preset['models'].items():
            directory = str(Path(manifest['model_fingerprints'][role]['path']).parent)
            if directory not in paths.get(spec['kind'], '').splitlines():
                paths[spec['kind']] = '\n'.join(filter(None, (paths.get(spec['kind']), directory)))
        model_config = root / 'pinned-model-paths.json'
        write(model_config, dict(pinned_camera=paths))
        command += ' --extra-model-paths-config ' + shlex.quote(str(model_config))
        os.environ['TOUCHDOWN_COMFY_SERVER_CMD'] = command
        args = SimpleNamespace(**{'wan_' + k: v['name'] for k, v in preset['models'].items()},
                               wan_lora_strength=preset['lora_strength'])
        write(root / 'status.json', dict(stage='render', status='running'))
        start = time.monotonic()
        with studio.runtime('comfy', root):
            import urllib.request
            with urllib.request.urlopen(studio.COMFY_URL.rstrip('/') + '/system_stats', timeout=10) as response:
                write(root / 'runtime-system-stats.json', json.load(response))
            with f._client(studio.VIDEO_URL) as client:
                graph = {}
                image = f._load(graph, client, '44', Path(manifest['identity']['source']) / 'scene/arrival.png')
                node = f._wan(graph, image, preset['frames'], manifest['prompt'], manifest['identity']['seed'],
                              'touchdown/earth_camera', args)
                graph['506']['inputs']['device'] = 'cpu'
                graph['510']['inputs'].update(width=preset['width'], height=preset['height'])
                graph['514']['inputs']['fps'] = preset['source_fps']
                graph['509']['inputs']['text'] += (', crossfade, dissolve, fade in, fade out, double exposure, '
                    'transparent scenery, superimposed horizons, iris transition, slideshow, Ken Burns, '
                    'static landscape with opacity changes, low cloud exit, enlarged monument in space, '
                    'grassy rooftop, raised isolated platform, miniature landscape, decorative cloud streaks')
                write(root / 'submitted-graph.json', graph_snapshot(graph))
                saved = f._submit(client, graph, (node,), root / 'generation')
        video = Path(saved[node]); workflow = video.parent / 'graph.json'
        record = dict(video=str(video), sha256=digest(video), workflow=str(workflow),
                      workflow_sha256=digest(workflow), elapsed_seconds=round(time.monotonic()-start, 1))
        write(root / 'result.json', record)
    frames = read_frames(record['video'])
    if len(frames) != preset['frames']:
        raise ValueError('Unexpected generated frame count')
    encode(reversed(frames), target, (preset['width'], preset['height']), preset['delivery_fps'])
    write(root / 'flight-preview.json', dict(video=str(target), sha256=digest(target),
          source=record, source_frame_mapping=list(range(len(frames)-1, -1, -1)),
          fps=preset['delivery_fps'], source_fps=preset['source_fps'], speed=preset['delivery_fps']/preset['source_fps'],
          assembly_crossfade=False, scene_transition_quality='unreviewed'))
    return target


def load_prepared(source, root):
    """Review/assembly can evolve without regenerating an immutable model output."""
    regression.check_root(source, root)
    if (root/'acceptance.json').exists():
        raise ValueError('Approved runs are frozen; choose a fresh review output')
    manifest = read(root/'experiment.json')
    if manifest['identity']['source'] != str(source):
        raise ValueError('Prepared run belongs to a different source')
    regression.verify_snapshot(manifest['identity']['source_hashes'])
    capture_recipe(root)
    return manifest


def assemble(root, manifest, flight_review, policy=None):
    if flight_review['status'] != 'pass':
        raise ValueError('Full ordered flight review did not pass; raw flight remains available')
    source, preset = Path(manifest['identity']['source']), manifest['identity']['preset']
    if (flight_review['identity']['video_sha256'] != digest(root/'flight-forward.mp4')
            or flight_review['identity']['policy'] != (policy or preset['review'])):
        raise ValueError('Flight review is stale for these pixels or requirements')
    flight = split_flight._checked(read(root / 'result.json'))
    if read(root/'flight-preview.json')['source'] != flight:
        raise ValueError('Reviewed preview belongs to a different generation')
    near = split_flight._checked(manifest['native_portrait'])
    generated = read_frames(flight['video'])
    native = cv2.VideoCapture(near['video'])
    try:
        _, nw, nh, fps = video_info(native)
        if fps != preset['delivery_fps']:
            raise ValueError('Native portrait fps differs from delivery')
        w, h = preset['width'], preset['height']
        fitted_h = round((w / (nw / nh)) / 2) * 2
        if fitted_h > h or h - fitted_h > h * .01:
            raise ValueError('Aspect correction would require substantial crop')
        top = (h-fitted_h)//2
        def fit(frame):
            return cv2.resize(frame[top:top+fitted_h], (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        ok, first = native.read()
        if not ok:
            raise ValueError('Empty native portrait')
        seam = match_metrics(fit(generated[0]), first)
        write(root / 'portrait-seam.json', seam)
        if not seam['acceptable']:
            raise JoinMismatch('Camera flight does not match portrait; no dissolve added')
        ratios = split_flight.face_ratios(near['video'])
        stop = split_flight.FACE_STOP.get(manifest['config']['video'].get('framing'))
        end = split_flight.ending_frame(ratios, stop, 0)
        # A stop at frame six can truncate the requested action; the source's
        # existing measured end is reused when it was reviewed for this framing.
        source_pacing = source / 'motion/pacing.json'
        if source_pacing.exists():
            end = read(source_pacing).get('ending', {}).get('native_last_frame', end)
        native.set(cv2.CAP_PROP_POS_FRAMES, 0)
        def frames():
            for frame in reversed(generated[1:]):
                yield fit(frame)
            for i in range(end+1):
                ok, frame = native.read()
                if not ok:
                    raise ValueError('Short native portrait')
                yield frame
        target = root / 'earth_zoom.mp4'
        encode(frames(), target, (nw, nh), fps)
        write(root / 'delivery.json', dict(video=str(target), sha256=digest(target),
              source=flight, native_portrait=near, flight_review=str(root/'review/report.json'), seam=seam,
              flight_frames=len(generated)-1, native_frames=end+1, native_start_frame=len(generated)-1,
              output_frames=len(generated)+end, fps=fps, source_crop=[0,top,w,top+fitted_h],
              assembly_code_sha256=digest(Path(__file__)), review_policy=flight_review['identity']['policy'],
              assembly_crossfade=False, optical_flow_interpolation=False, face_composite=False,
              publication='review candidate; human playback review required'))
    finally:
        native.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'render', 'review', 'assemble', 'run'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, action='append', default=[])
    parser.add_argument('--preset', type=Path, default=PRESET)
    parser.add_argument('--review-preset', type=Path)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    stage = args.stage
    manifest = (load_prepared(source, root) if stage in ('review','assemble')
                else prepare(source, root, args.model_root, args.preset))
    policy = (read(args.review_preset)['review'] if args.review_preset
              else manifest['identity']['preset']['review'])
    report = None
    try:
        if stage in ('render', 'run'):
            render(root, manifest)
        if stage in ('review', 'run'):
            report = review(root/'flight-forward.mp4', root/'review', policy)
            write(root/'status.json', dict(stage='review', status='review_candidate' if report['status']=='pass' else 'held',
                                         failures=report['verdict']['failures']))
        if stage in ('assemble', 'run'):
            report = read(root/'review/report.json')
            if report['status']=='pass':
                assemble(root, manifest, report, policy)
                write(root/'status.json', dict(stage='assembled', status='review_candidate',
                                             publication='Human playback review required'))
            elif stage=='assemble':
                raise ValueError('Sequence review failed; refusing assembly')
        regression.verify_snapshot(manifest['identity']['source_hashes'])
        return 2 if report is not None and report['status'] != 'pass' else 0
    except (Exception, SystemExit) as exc:
        write(root/'status.json', dict(stage=stage,status='held',error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
