"""Still-only geographic reference experiment: overhead, native Fill, spherical projection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'outputs/touchdown/studio/studio-eilean-donan-photographic-transfer-v1-20261027'
SOURCE = BASE / 'run/motion/earth_zoom.mp4'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key, value = line.split('=', 1)
        if key.startswith('TOUCHDOWN_'):
            os.environ[key] = json.loads(value)


def prepare(root):
    import cv2
    if root.exists():
        raise ValueError('Fresh experiment directory required')
    root.mkdir(parents=True)
    cap = cv2.VideoCapture(str(SOURCE))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 64)
    ok, image = cap.read()
    cap.release()
    assert ok
    cv2.imwrite(str(root / 'cloud-exit-frame-064.png'), image)
    files = [SOURCE, root / 'cloud-exit-frame-064.png', Path(__file__).resolve(),
        REPO / 'scripts/touchdown/fill_outpaint.py', REPO / 'scripts/touchdown/flux3.py',
        REPO / 'scripts/touchdown/authored_scene.py', REPO / 'workflows/touchdown/flux-fill-outpaint.api.json']
    write(root / 'experiment.json', dict(source=str(SOURCE), frame=64, seconds=64/24,
        scope='Still references only; no video submissions, source replacement or app promotion',
        stages=['Approximate overhead from cloud-exit photograph', 'Two native Fill expansions, each 2x in each dimension',
                'Deterministic spherical texture projection', 'Optional photographic FLUX finishing after guide review'],
        limitations=['Generated overhead and expanded surroundings are approximate, not measured geography',
                    'Spherical curvature supplies a cinematic scale guide, not validated metric altitude',
                    'Outpaint raw centre may drift; no pasted-back centre is accepted as proof'],
        hashes={str(p.resolve()): sha(p) for p in files}, app_promoted=False))
    write(root / 'status.json', dict(status='prepared', video_submissions=0))


def verify(root):
    for path, expected in read(root / 'experiment.json')['hashes'].items():
        assert sha(Path(path)) == expected, path


def still(root, name):
    environment()
    verify(root)
    from touchdown import studio_build as studio
    from touchdown.authored_scene import generate_asset
    cfg = read(REPO / 'outputs/touchdown/studio/studio-eilean-donan-e1-20261026/scene-config.json')
    if name == 'overhead':
        refs = [str(root / 'cloud-exit-frame-064.png')]
        prompt = (
            'Create one natural, realistic vertical overhead aerial photograph of the SAME Scottish '
            'coastal landscape shown in image 1. Change only the viewing direction to look straight '
            'DOWN, with no sky or horizon. Preserve the distinctive island and surrounding water, '
            'irregular opposing shorelines, mountain-valley directions and their left-to-right order. '
            'Keep the island around the centre of the image. Expand the view only enough to reveal '
            'the land surrounding this same connected sea loch. Real muted green-brown Highland '
            'mountains, dark blue water, natural sunlight and subtle haze matching the reference. '
            'The tiny castle is part of the island, never enlarged into a giant landmark. '
            'No people, no map labels, no borders, no clouds hiding the coastline, no collage, '
            'no rectangular inset, no stylized map or rendered terrain. One continuous photograph.'
        )
        seed = 2026997937
    else:
        refs = [str(root / 'globe-guide.png'), str(root / 'cloud-exit-frame-064.png')]
        prompt = (
            'Finish image 1 as a believable natural photograph from close low Earth orbit. '
            'Image 1 fixes the curved Earth horizon, camera heading and the broad land/water '
            'pattern projected onto the planetary surface: preserve those positions and shapes. '
            'Image 2 supplies the destination landscape, natural green-brown mountains, dark blue '
            'sea-loch water and lighting only. Its island and castle are microscopic at orbital '
            'scale and must not be enlarged. Keep Earth filling the frame, a thin curved blue '
            'atmospheric rim near the top and a narrow strip of black space. Add photographic '
            'atmospheric depth and broad small-scale weather systems, with gaps revealing the '
            'same land and water beneath. Remove texture stretching and artificial rendering '
            'artifacts. No new dominant coast, giant island, close castle, pasted local scene, '
            'separate whole globe, text, illustration or CGI. One full-frame photograph.'
        )
        seed = 2026997938
    spec = dict(name=name, references=refs, prompt=prompt, seed=seed, models=cfg['models'],
                width=1920, height=1088)
    write(root / f'{name}-spec.json', spec)
    command = os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD'] = command + ' --deterministic --extra-model-paths-config ' + shlex.quote(str(BASE / 'pinned-model-paths.json'))
    write(root / 'status.json', dict(status='generating_'+name, video_submissions=0))
    with studio.runtime('comfy', root):
        generate_asset(studio.COMFY_URL, spec, root)
    verify(root)
    write(root / 'status.json', dict(status=name+'_awaiting_visual_review', video_submissions=0))


def expand(root, index):
    environment()
    verify(root)
    from touchdown import fill_outpaint as fill
    source = root / ('overhead.png' if index == 1 else f'expand-{index-1}/raw-outpaint.png')
    folder = root / f'expand-{index}'
    graph, photo = fill.prepare(source, folder)
    # Same native Fill conditioning, padding geometry and sampler as coherent raw trial.
    graph['23']['inputs']['text'] = (
        'Continue this vertical overhead aerial photograph naturally beyond all four edges, '
        'with the same straight-down camera direction and continuous landscape. Extend the '
        'existing connected Scottish sea lochs, irregular coastlines, dark blue open water, '
        'branching inlets, green-brown Highland mountains and valleys. Preserve the existing '
        'central water connections and shorelines; surrounding terrain is a wider geographic '
        'continuation. No new giant island or castle, no person, no buildings enlarged, '
        'no sky, no horizon, no clouds, no picture frame or inset, no labels or map graphics. '
        'Natural photographic detail and consistent muted sunlight and colour.'
    )
    fill.validate(graph)
    write(folder / 'graph.api.json', graph)
    # UI prompt reflects the API prompt for this saved variant.
    ui = read(folder / 'workflow.json')
    node = next(n for n in ui['nodes'] if n['id'] == 23)
    node['widgets_values'][0] = graph['23']['inputs']['text']
    write(folder / 'workflow.json', ui)
    write(root / 'status.json', dict(status='outpainting_'+str(index), video_submissions=0))
    fill.run(folder, graph, photo)
    verify(root)
    write(root / 'status.json', dict(status='outpaint_'+str(index)+'_awaiting_visual_review', video_submissions=0))


def project(root):
    import cv2
    import numpy as np
    verify(root)
    texture_path = root / 'expand-2/raw-outpaint.png'
    texture = cv2.imread(str(texture_path))
    assert texture is not None
    w, h = 1920, 1088
    # Ray/sphere intersections. Unit Earth, camera height 0.08 radii. Heading +Y.
    altitude, pitch, horizontal_fov = .08, np.deg2rad(45), np.deg2rad(80)
    origin = np.array([0., 0., 1+altitude])
    forward = np.array([0., np.cos(pitch), -np.sin(pitch)])
    up = np.array([0., np.sin(pitch), np.cos(pitch)])
    xx, yy = np.meshgrid((np.arange(w)+.5-w/2)/(w/2), -(np.arange(h)+.5-h/2)/(w/2))
    ray = forward + np.tan(horizontal_fov/2)*(xx[..., None]*np.array([1., 0., 0.])+yy[..., None]*up)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    b = ray @ origin
    discriminant = b*b-(origin@origin-1)
    valid = (discriminant >= 0) & (b < 0)
    t = -b-np.sqrt(np.maximum(0, discriminant))
    point = origin+ray*t[..., None]
    longitude = np.arctan2(point[..., 0], point[..., 2])
    latitude = np.arcsin(np.clip(point[..., 1], -1, 1))
    center_t = (1+altitude)*np.sin(pitch)-np.sqrt(((1+altitude)*np.sin(pitch))**2-((1+altitude)**2-1))
    center_latitude = np.arcsin(center_t*np.cos(pitch))
    # A synthetic broad-region texture, explicitly without geographic/metric registration.
    lon_span, lat_span = 1.6, .9
    u = .5+longitude/lon_span
    v = .5-(latitude-center_latitude)/lat_span
    coverage = valid & (u>=0) & (u<=1) & (v>=0) & (v<=1)
    if int(coverage.sum()) != int(valid.sum()):
        raise ValueError('Projected Earth extends beyond supplied texture; refusing edge smearing')
    mapped = cv2.remap(texture, (u*(texture.shape[1]-1)).astype(np.float32),
        (v*(texture.shape[0]-1)).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Broad atmospheric haze, no generated clouds to conceal the reference geometry.
    normal = point/np.linalg.norm(point, axis=-1, keepdims=True)
    view_cosine = np.clip(np.sum(normal*(-ray), axis=-1), 0, 1)
    haze = np.clip((1-view_cosine)**3*.65, 0, .65)[..., None]
    image = mapped.astype(float)*(1-haze)+np.array([205., 157., 105.])*haze
    image[~valid] = [8, 5, 3]
    cv2.imwrite(str(root / 'globe-guide.png'), image.clip(0, 255).astype(np.uint8))
    # Track the nested source footprint rather than asserting literal pixel preservation.
    source_box = (u>=.375) & (u<=.625) & (v>=.375) & (v<=.625) & valid
    outline = (source_box.astype(np.uint8)*255)
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image.clip(0, 255).astype(np.uint8)
    cv2.drawContours(annotated, contours, -1, (0, 200, 255), 2)
    cv2.imwrite(str(root / 'globe-source-footprint.png'), annotated)
    write(root / 'projection.json', dict(texture=str(texture_path), texture_sha256=sha(texture_path),
        guide_sha256=sha(root/'globe-guide.png'), radius=1, height_in_radii=altitude,
        pitch_degrees=45, horizontal_fov_degrees=80, longitude_span_radians=lon_span,
        latitude_span_radians=lat_span, center_latitude_radians=float(center_latitude),
        original_overhead_nominal_texture_rectangle=[.375, .375, .625, .625],
        destination_projected_to=[w/2, h/2], metric_registration=False,
        note='Approximate synthetic geography; source footprint is ancestry, not pixel-exact overlap'))
    write(root / 'status.json', dict(status='projected_guide_awaiting_review', video_submissions=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'overhead', 'expand-1', 'expand-2', 'project', 'orbit'])
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.stage == 'prepare':
        prepare(root)
    elif args.stage.startswith('expand-'):
        expand(root, int(args.stage[-1]))
    elif args.stage == 'project':
        project(root)
    else:
        still(root, args.stage)
