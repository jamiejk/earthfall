"""Isolated photographic LTX opening repair; retain source frame 40 onward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'outputs/touchdown/studio/studio-eilean-donan-photographic-transfer-v1-20261027'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root):
    import cv2
    if root.exists():
        raise ValueError('A fresh experiment root is required')
    root.mkdir(parents=True)
    shutil.copytree(BASE / 'frozen-code', root / 'frozen-code')
    shutil.copy2(BASE / 'pinned-model-paths.json', root / 'pinned-model-paths.json')
    source = BASE / 'run/motion/earth_zoom.mp4'
    cap = cv2.VideoCapture(str(source))
    info = [int(cap.get(7)), int(cap.get(3)), int(cap.get(4)), cap.get(5)]
    ok, first = cap.read()
    cap.release()
    assert ok and info == [196, 2560, 1472, 24.0]
    cv2.imwrite(str(root / 'orbit-start.png'), cv2.resize(first, (1280, 736)))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vf',
        'trim=start_frame=40:end_frame=49,reverse,scale=1280:736,setpts=PTS-STARTPTS',
        '-an', '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p',
        str(root / 'cloud-retreat-prefix.mp4')], check=True)
    global_text = (
        'One uninterrupted live-action photographic camera flight, filmed with a fixed focal length '
        'rectilinear lens. The CAMERA rapidly translates BACKWARD AND UPWARD away from Earth along '
        'one fixed heading. This is a powerful sustained physical retreat, with strong perspective '
        'and parallax. Nearby cloud structures recede much faster than distant terrain. '
        'Stable natural daylight and exposure. No hovering, sideways orbit, lens zoom, cuts, '
        'dissolves, fades, flat cloud overlay, morphing or CGI.'
    )
    beats = [
        dict(name='bound_cloud_motion', start_frame=0, end_frame=9, prompt=(
            'Continue the supplied moving cloud passage in its exact direction, retreating BACKWARD '
            'AND UPWARD through substantial three-dimensional clouds. Opaque cloud surrounds the camera. '
            'The camera keeps travelling fast, not watching clouds drift past a stationary viewpoint.')),
        dict(name='rapid_cloud_exit', start_frame=9, end_frame=25, prompt=(
            'The camera rockets BACKWARD AND UPWARD out through the upper cloud surface. Nearby '
            'billowing cloud banks rapidly recede into depth, revealing the connected planetary '
            'surface beneath. Terrain and cloud formations contract toward the same vanishing point '
            'as the camera gains distance. Maintain one heading and strong outward camera translation.')),
        dict(name='rapid_planet_retreat', start_frame=25, end_frame=49, prompt=(
            'Keep rocketing BACKWARD AND UPWARD away from the planet, never slowing. The same terrain '
            'and weather systems visibly SHRINK and more surrounding surface enters from every edge '
            'because the camera is rapidly gaining altitude. Initially the planetary surface fills '
            'the entire image with NO visible horizon or space. Only during the FINAL twelve frames '
            'does the curved Earth horizon enter from ABOVE the top edge, descending slightly into '
            'view and revealing a narrow blue limb and strip of black space. Reach the supplied '
            'close-Earth endpoint while still moving rapidly. Never reveal a small isolated globe.')),
    ]
    timing = dict(version='photographic-opening-rush-v1', fps=24, source_frames=49,
        global_prompt=global_text, generation_beats=beats,
        single_prompt=global_text + ' ' + ' '.join(b['prompt'] for b in beats),
        epsilon=.1, temporal_stride=8)
    write(root / 'timing.json', timing)
    files = [p for p in (root / 'frozen-code').rglob('*') if p.is_file()
             and p.name != 'extra-paths.json' and '__pycache__' not in p.parts]
    files += [source, root / 'orbit-start.png', root / 'cloud-retreat-prefix.mp4',
              root / 'pinned-model-paths.json', root / 'timing.json', Path(__file__).resolve()]
    write(root / 'experiment.json', dict(source=str(source), source_info=info,
        parent_seed=2026997968, generated_frames=49, generated_size=[1280, 736],
        generated_delivery_frames=list(range(48, 8, -1)), retained_source_frames=list(range(40, 196)),
        source_prefix_frames=list(range(48, 39, -1)),
        intended_change='Physical rapid push toward Earth after reversal; horizon exits above by about 0.5s',
        scope='Only delivery frames 0..39 replaced; frame 40 onward and total timing retained',
        inference='Same frozen photographic LTX 2.5 distilled graph builder and Prompt Relay; no Layout LoRA',
        crossfade=False, interpolation=False, digital_zoom=False, app_promoted=False,
        hashes={str(p.resolve()): sha(p) for p in files}))
    write(root / 'status.json', dict(status='prepared'))


def verify(root):
    for path, expected in read(root / 'experiment.json')['hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Changed input: {path}')


def worker(root):
    # Import only after selecting a fresh copy of the archived photographic code.
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key, value = line.split('=', 1)
        if key.startswith('TOUCHDOWN_'):
            os.environ[key] = json.loads(value)
    code = root / 'frozen-code'
    sys.path[:0] = [str(code / 'scripts'), str(code / 'src')]
    os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:2])
    from touchdown import split_flight as flight, studio_build as studio
    import fakeout_earth_zoom as f
    assert Path(flight.__file__).is_relative_to(code)
    studio.REPO = REPO
    verify(root)
    command = os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--deterministic' not in shlex.split(command):
        command += ' --deterministic'
    command += ' --extra-model-paths-config ' + shlex.quote(str(root / 'pinned-model-paths.json'))
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD'] = command
    flight.enable_prompt_relay()
    write(root / 'runtime-launch.json', dict(command=shlex.split(os.environ['TOUCHDOWN_COMFY_SERVER_CMD']),
        frozen_code=str(code), deterministic=True,
        relay_paths_sha256=sha(code / 'outputs/touchdown/relay-runtime/extra-paths.json')))
    folder = root / 'generation'
    folder.mkdir()
    write(root / 'status.json', dict(status='generating', started_at=time.time()))
    started = time.monotonic()
    try:
        with studio.runtime('comfy', root):
            with f._client(studio.VIDEO_URL) as client:
                graph = {'50': {'class_type': 'LoadVideo', 'inputs': {
                    'file': client.assets.from_file(str(root / 'cloud-retreat-prefix.mp4'))}},
                    '51': {'class_type': 'GetVideoComponents', 'inputs': {'video': ['50', 0]}}}
                orbit = f._load(graph, client, '46', root / 'orbit-start.png')
                saved, _ = flight._generate(f, client, graph, 49,
                    [(0, ['51', 0]), (-1, orbit)], {}, read(root / 'timing.json'),
                    flight.guide_times(0, 9) + [48], 2026997968,
                    'touchdown/opening_rush_v1', folder, dict(width=1280, height=736))
                record = flight._record(saved, f)
                write(root / 'generation-result.json', record)
        verify(root)
        write(root / 'status.json', dict(status='generated_awaiting_review', elapsed_seconds=round(time.monotonic()-started, 1)))
    except BaseException as exc:
        write(root / 'status.json', dict(status='held', error=f'{type(exc).__name__}: {exc}'))
        raise


def assemble(root):
    verify(root)
    manifest = read(root / 'experiment.json')
    result = read(root / 'generation-result.json')
    assert sha(Path(result['video'])) == result['sha256']
    # Reverse generated 48..9, then use original 40..195. No blend or retiming.
    dest = root / 'opening-rush-preview.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', result['video'], '-i', manifest['source'],
        '-filter_complex', '[0:v]trim=start_frame=9:end_frame=49,reverse,scale=2560:1472,setpts=PTS-STARTPTS[a];'
        '[1:v]trim=start_frame=40,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]',
        '-map', '[v]', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '16',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)], check=True)
    import cv2
    cap = cv2.VideoCapture(str(dest))
    assert (int(cap.get(7)), cap.get(5)) == (196, 24)
    cap.release()
    write(root / 'assembly.json', dict(video=str(dest), sha256=sha(dest), frames=196, fps=24,
        replacement_frames=40, retained_source_frames=156, retained_start_seconds=40/24,
        duration_seconds=196/24, crossfade=False, retiming=False,
        note='Later source frames retained at identical times; ordinary H.264 re-encoding changes pixels slightly.'))
    print(dest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'assemble'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.root.resolve())
