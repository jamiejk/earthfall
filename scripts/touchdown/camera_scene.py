"""Versioned camera-first Chillon pilot, separate from the photographic app.

The segment occupies original delivery frames 96..192, at 24 fps. Its geometry
and camera are authored from the retained photographs; surveyed accuracy is not
claimed. No orbital/cloud wrapper or final native-portrait join is produced.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from touchdown.authored_scene import digest, generate_asset
from touchdown.flight_contract import BOUNDS, FPS
from touchdown.flight_regression import read, write, check_root, verify_snapshot
from touchdown.recipe_snapshot import capture

START, END = 96, 192
FRAMES = END-START+1
VERSION = 'camera-scene-pilot-v2'


from touchdown.shoreline_camera import shore, ground_height, camera_pose, project


def prepare(source,root):
    check_root(source,root)
    cfg=read(source/'scene-config.json')
    files=[source/'scene-config.json',source/'setup.json',source/'scene/arrival.png',Path(cfg['subject'])]
    files += [Path(r['local_path']) for r in cfg['references']]
    hashes={str(p.resolve()):digest(p) for p in files}
    from touchdown.flight_runtime import recipe_hashes
    code=[Path(__file__),Path(__file__).with_name('camera_scene_blender.py')]
    identity=dict(version=VERSION,source=str(source),source_hashes=hashes,
                  code_hashes=recipe_hashes(),seed=cfg['seed']+2048,
                  product_clock=dict(fps=FPS,bounds=list(BOUNDS),regional_altitude_m=9144),
                  segment=dict(start_frame=START,end_frame=END,frames=FRAMES),
                  architecture='Persistent Blender geometry -> separately versioned LTX Layout To Render pilot',
                  blender_version=subprocess.check_output(['blender','--version'],text=True).splitlines()[:3])
    path=root/'experiment.json'
    if path.exists():
        if read(path)['identity']!=identity:raise ValueError('Pilot inputs/code changed; use a fresh root')
        verify_snapshot(hashes);capture(root);return cfg
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh pilot root required')
    write(path,dict(identity=identity));capture(root)
    (root/'assets').mkdir()
    for name in ['arrival','subject']:
        p=source/'scene/arrival.png' if name=='arrival' else Path(cfg['subject'])
        shutil.copy2(p,root/'assets'/f'{name}.png')
    for i,r in enumerate(cfg['references']):shutil.copy2(r['local_path'],root/'assets'/f'location-{i}.jpg')
    write(root/'attribution.json',cfg['references'])
    poses=[camera_pose(d) for d in range(START,END+1)]
    write(root/'render.json',dict(width=1280,height=736,fps=FPS,frames=FRAMES,start_delivery_frame=START,
        poses=poses,seed=20261026,renderer_sha256=digest(code[1]),
        scope='Camera/appearance diagnostic only; no orbit, cloud or final native-portrait join',
        phases=[dict(name='regional',start=96,end=108),dict(name='approach',start=108,end=156),dict(name='settle',start=156,end=193)],
        witness_points=dict(foreground_stone=[-2.5,-12,ground_height(-2.5,-12)+.8],
                            castle_keep=[-42,185,28.],person=[0,0,ground_height(0,0)+1.])))
    return cfg


def render(root,preview=False):
    import fcntl
    args=['blender','-b','-t','8','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('camera_scene_blender.py')),'--',str(root)]
    if preview:args+=['--preview']
    lock_path=Path(__file__).resolve().parents[2]/'outputs/.touchdown-photo-gpu.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(args,check=True)
    if not preview:
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','24','-i',str(root/'frames/%04d.png'),
            '-frames:v',str(FRAMES),'-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'guide.mp4')],check=True)


def paint(root,cfg):
    from touchdown import studio_build as studio
    require_raw_review(root)
    for name,index in [('first',0),('last',FRAMES-1)]:
        guide=root/'frames'/f'{index:04d}.png'
        prompt=(
            'EDIT IMAGE 1 ONLY into ONE real live-action photographic view: same camera pose, '
            'lens, shoreline, path, landmark footprint, roof and tower silhouettes, trees and ground. '
            'Replace synthetic material appearance with natural photographic detail and atmospheric depth. '
            'Bright natural daylight, one exposure, sharp natural textures, no miniature or CGI look. '
            'No new towers or platforms, no layout changes, no collage, no inset, no text. ')
        refs=[str(guide)]
        if name=='first':
            prompt+=('The camera is 30,000 feet above ground. One broad lake occupies the LEFT of the picture; '
                     'connected regional green terrain and distant tiny settlement patterns occupy the RIGHT. '
                     'The shoreline in image 1 remains in exactly the same position. Show many kilometres of land. '
                     'Any local structures are microscopic. No person, no readable buildings, no castle close-up, no sky or horizon.')
        else:
            refs.append(str(root/'assets/subject.png'))
            prompt+=('Image 2 supplies ONLY the person identity: same face, brown hair and beige trench coat. '
                     'Keep the pose and scale from image 1. The person stands on connected ground-level grass '
                     'beside the shore path, arms open, knees-up portrait. Preserve the small distant castle on '
                     'the LEFT: irregular pale masonry, slate pitched roofs, conical turrets and square keep. '
                     'The castle stays far behind the person. No roof or isolated pedestal beneath them.')
        spec=dict(name=f'photo-{name}',references=refs,
                  width=1280,height=736,models=cfg['models'],seed=cfg['seed']+2049+index,prompt=prompt)
        with studio.runtime('comfy',root):generate_asset(studio.COMFY_URL,spec,root)


def require_raw_review(root):
    review=read(root/'raw-guide-review.json')
    if review['status']!='pass' or review['guide_sha256']!=digest(root/'guide.mp4') or review['geometry_sha256']!=digest(root/'geometry-audit.json'):
        raise ValueError('Raw camera guide must pass review before photographic inference')
    return review


def review_photos(root,visual_note):
    require_raw_review(root)
    if not visual_note:raise ValueError('Record the actual photographic-anchor layout findings')
    write(root/'photo-anchor-review.json',dict(status='pass',reviewer='Agent sampled-image inspection',
        visual_note=visual_note,guide_sha256=digest(root/'guide.mp4'),
        first_sha256=digest(root/'photo-first.png'),last_sha256=digest(root/'photo-last.png'),
        scope='Anchor suitability for this bounded treatment; not video acceptance'))


def require_photo_review(root):
    r=read(root/'photo-anchor-review.json')
    if (r['status']!='pass' or r['guide_sha256']!=digest(root/'guide.mp4')
        or r['first_sha256']!=digest(root/'photo-first.png') or r['last_sha256']!=digest(root/'photo-last.png')):
        raise ValueError('Photographic anchors changed or failed their layout review')


def review_raw(root,visual_note):
    import cv2
    if not visual_note:raise ValueError('Record sampled-frame visual findings before accepting the raw guide')
    audit=read(root/'geometry-audit.json');render=read(root/'render.json')
    if render['poses']!=[camera_pose(d) for d in range(START,END+1)]:raise ValueError('Camera poses changed')
    cap=cv2.VideoCapture(str(root/'guide.mp4'))
    n,fps=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),cap.get(cv2.CAP_PROP_FPS);cap.release()
    if (n,fps)!=(FRAMES,24):raise ValueError('Raw guide changed the segment clock')
    if audit['image_planes'] or audit['view_crossfades']:raise ValueError('Guide contains image replacement')
    witness=[r['points']['foreground_stone'] for r in audit['witness_projection']]
    visible=[p for p in witness if p[2]>0 and 0<p[0]<1 and 0<p[1]<1]
    if len(visible)<4 or witness[-1][2]>0:raise ValueError('Foreground witness does not pass the camera')
    result=dict(status='pass',guide_sha256=digest(root/'guide.mp4'),geometry_sha256=digest(root/'geometry-audit.json'),
        frames=n,fps=fps,delivery_mapping=list(range(START,END+1)),visible_witness_frames=len(visible),
        visual_note=visual_note,reviewer='Agent sampled-frame inspection plus camera/geometry audit',
        scope='Motion prototype only; photographic finish and identity not certified')
    write(root/'raw-guide-review.json',result)
    return result


def setup_runtime(root):
    from touchdown.flight_runtime import capture_runtime
    from touchdown.phase_recovery import stream_fingerprint
    import shlex
    repo=Path(__file__).resolve().parents[2]
    prior=read(repo/'outputs/touchdown/studio/studio-chillon-phase-recovery-v1b-20261026/experiment.json')
    models=prior['model_fingerprints']
    for m in models.values():
        stat=Path(m['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('Previously fingerprinted weights changed')
    lora=Path('/run/media/jamie/Data/Image Creation/Comfy/models/loras/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors')
    cache=root/'model-fingerprints.json'
    if cache.exists():
        recorded=read(cache);m=recorded['layout_lora'];stat=lora.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('Layout LoRA changed')
    else:
        models['layout_lora']=stream_fingerprint(lora)
        write(cache,models)
    paths={**prior['model_paths'],'loras':str(lora.resolve().parent)}
    config=root/'pinned-model-paths.json';write(config,dict(pilot=dict(is_default=True,**paths)))
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--deterministic' not in command.split():command+=' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --extra-model-paths-config '+shlex.quote(str(config))
    capture_runtime(root)


def treat(root,cfg):
    require_raw_review(root)
    require_photo_review(root)
    from geoblock_core.workflow import ShotRequest,LTX_VIDEO,_add_ltx25
    import fakeout_earth_zoom as f
    from touchdown import studio_build as studio
    output=root/'ltx';output.mkdir(exist_ok=True)
    prompt=(
        'One uninterrupted live-action fixed 35mm camera descent following the supplied geometric camera guide. '
        'The first half second remains a broad clear regional view 30,000 feet above the single large lake, '
        'then descend along that same shoreline for two seconds, passing foreground rocks and trees, '
        'and spend the final one and a half seconds slowing to the same person on connected grass. '
        'Preserve the exact guide movement, landmark geometry, paths, shoreline, occlusion and timing. '
        'Chillon Castle has pale irregular stone, pitched slate roofs, conical turrets and a square keep. '
        'Bright natural daylight over a turquoise lake, green Swiss shore and atmospheric mountain distances. '
        'Real photographic materials, natural irregular detail, no miniature or CGI appearance. '
        'The same woman with brown hair and a beige trench coat opens both arms during the finish. '
        'Her shoes remain grounded until the camera crops them. No invented pedestal, roof or separate platform. '
        'Physical perspective and depth throughout; no cuts, dissolves, wipes, overlays, scenery swaps or pauses.')
    with studio.runtime('comfy',root),f._client(studio.VIDEO_URL) as client:
        graph={
            '50':dict(class_type='LoadVideo',inputs=dict(file=client.assets.from_file(str(root/'guide.mp4')))),
            '60':dict(class_type='GetVideoComponents',inputs=dict(video=['50',0])),
            '40':dict(class_type='LoadImage',inputs=dict(image=client.assets.from_file(str(root/'photo-first.png')))),
            '41':dict(class_type='LTXVPreprocess',inputs=dict(image=['40',0],img_compression=18)),
            '42':dict(class_type='LoadImage',inputs=dict(image=client.assets.from_file(str(root/'photo-last.png')))),
            '43':dict(class_type='LTXVPreprocess',inputs=dict(image=['42',0],img_compression=18))}
        request=ShotRequest(name=VERSION,output_prefix='touchdown/chillon_camera_scene',seed=cfg['seed']+2048,
                           ltx_upscale=False,pin_last_keyframe=True)
        _add_ltx25(graph,request,FRAMES,['41',0],['43',0],[],prompt,layout=True)
        graph['308']['inputs'].update(width=1280,height=736)
        if graph['308']['inputs']['length']!=FRAMES or graph['307']['inputs']['frame_rate']!=24:raise ValueError('Treatment clock changed')
        write(root/'treatment-spec.json',dict(version=VERSION,seed=request.seed,frames=FRAMES,fps=24,
            delivery_mapping=list(range(START,END+1)),layout_strength=1.,lora_strength=1.,prompt=prompt,
            guide_sha256=digest(root/'guide.mp4'),first_sha256=digest(root/'photo-first.png'),last_sha256=digest(root/'photo-last.png'),
            scope='Separate architecture experiment; app Prompt Relay contract is not overwritten'))
        saved=f._submit(client,graph,(LTX_VIDEO,),output)
    target=Path(saved[LTX_VIDEO])
    write(root/'treatment-result.json',dict(video=str(target),sha256=digest(target)))
    compare(root,target)
    return target


def compare(root,treated):
    import cv2
    import numpy as np
    caps=[cv2.VideoCapture(str(p)) for p in [root/'guide.mp4',treated]]
    if any((int(c.get(cv2.CAP_PROP_FRAME_COUNT)),c.get(cv2.CAP_PROP_FPS))!=(FRAMES,24) for c in caps):
        raise ValueError('Guide/treatment clocks differ; no retiming permitted')
    dest=root/'comparison.mp4'
    enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x410',
        '-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    try:
        for i in range(FRAMES):
            image=np.zeros((410,1280,3),np.uint8)
            for j,c in enumerate(caps):
                ok,frame=c.read()
                if not ok:raise ValueError('Short comparison decode')
                image[42:,j*640:(j+1)*640]=cv2.resize(frame,(640,368),interpolation=cv2.INTER_AREA)
                cv2.putText(image,['Persistent 3D camera guide','LTX photographic treatment'][j],(j*640+10,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            d=START+i;phase='regional' if d<108 else 'approach' if d<156 else 'settle'
            cv2.putText(image,f'Delivery {d/24:.3f}s | {phase} | camera pilot, not a complete flight',(10,36),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1,cv2.LINE_AA)
            enc.stdin.write(image.tobytes())
    finally:
        enc.stdin.close()
        for c in caps:c.release()
    if enc.wait():raise RuntimeError('Comparison encoding failed')
    write(root/'comparison.json',dict(video=str(dest),sha256=digest(dest),frames=FRAMES,fps=24,
        delivery_mapping=list(range(START,END+1)),source_speed=1,scope='Camera segment diagnostic only'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','preview','render','review','paint','photo-review','treat'],required=True)
    p.add_argument('--visual-note',default='')
    a=p.parse_args()
    env=Path('/tmp/touchdown-generation-runtime.env')
    if env.exists():
        for line in env.read_text().splitlines():
            key,value=line.split('=',1)
            if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    os.environ['PYTHONPATH']=os.pathsep.join(str(Path(__file__).resolve().parents[2]/q) for q in ['scripts','src'])
    cfg=prepare(a.source.resolve(),a.root.resolve())
    if a.stage in ['preview','render']:render(a.root.resolve(),preview=a.stage=='preview')
    elif a.stage=='review':review_raw(a.root.resolve(),a.visual_note)
    elif a.stage=='photo-review':review_photos(a.root.resolve(),a.visual_note)
    elif a.stage=='paint':
        setup_runtime(a.root.resolve());paint(a.root.resolve(),cfg)
    elif a.stage=='treat':
        setup_runtime(a.root.resolve());treat(a.root.resolve(),cfg)
    verify_snapshot(read(a.root.resolve()/'experiment.json')['identity']['source_hashes'])


if __name__=='__main__':raise SystemExit(main())
