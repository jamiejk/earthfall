"""Recorded new-scene stages for the experimental photographic/grounded-orbit chain.

The photographic flight is generated separately with photographic_transfer.py.
This runner takes that fresh flight and a reviewed per-scene layout, never a
hard-coded previous destination. Main-module imports stay stdlib-only so the
opening worker can select the archived inference package before importing it.
"""
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
import urllib.parse
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[2]
FILL_CHECKS = REPO/'outputs/touchdown/studio/studio-eilean-donan-globe-reference-v1-20261027'
TIMING = REPO/'outputs/touchdown/studio/studio-eilean-donan-grounded-opening-soft-v2-20261027/timing.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    path = Path('/tmp/touchdown-generation-runtime.env')
    if not path.is_file():  # an app worker already passes its environment
        return
    for line in path.read_text().splitlines():
        key, value = line.split('=', 1)
        if key.startswith('TOUCHDOWN_'):
            os.environ[key] = json.loads(value)


def case(root):
    selected = read(root/'selection.json')['selected']
    cfg = read(Path(selected['source'])/'scene-config.json')
    return selected, cfg


def verify(root):
    for path, expected in read(root/'terrain/experiment.json')['hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Changed frozen input: '+path)


def terrain_prepare(root):
    import cv2
    import numpy as np
    selected, cfg = case(root)
    layout = read(root/'scene-layout.json')
    folder = root/'terrain'
    if folder.exists():
        raise ValueError('Fresh terrain directory required')
    folder.mkdir()
    source = root/'flight/run/motion/earth_zoom.mp4'
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, layout['cloud_exit_frame'])
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Cannot extract reviewed cloud-exit frame')
    cv2.imwrite(str(folder/'cloud-exit.png'), image)
    w, h = 1920, 1088
    colours = layout['colours']
    def bgr(colour):
        return tuple(int(colour[i:i+2], 16) for i in (5, 3, 1))
    image = np.full((h,w,3), bgr(colours['background']), np.uint8)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="{colours["background"]}"/>']
    for feature in layout['features']:
        points = [(round(x*w), round(y*h)) for x,y in feature['polygon']]
        colour = colours[feature['material']]
        cv2.fillPoly(image, [np.array(points,np.int32)], bgr(colour))
        svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{colour}"/>')
    svg.append('</svg>')
    (folder/'regional-layout.svg').write_text('\n'.join(svg)+'\n')
    cv2.imwrite(str(folder/'regional-layout.png'),image)
    files = [root/'selection.json', root/'scene-layout.json', source, folder/'cloud-exit.png',
             folder/'regional-layout.png', folder/'regional-layout.svg', Path(__file__).resolve(),
             REPO/'scripts/touchdown/fill_outpaint.py', REPO/'scripts/touchdown/flux3.py',
             REPO/'scripts/touchdown/authored_scene.py', REPO/'workflows/touchdown/flux-fill-outpaint.api.json']
    write(folder/'experiment.json',dict(place=selected['place'],seed=cfg['seed'],target_uv=layout['target_uv'],
        expansions=3,expansion_per_axis=2,total_expansion_per_axis=8,
        source_photo_frame=layout['cloud_exit_frame'],layout_authored=True,measured_geography=False,
        hashes={str(p.resolve()):sha(p) for p in files},app_promoted=False))
    write(root/'status.json',dict(stage='terrain',status='prepared',app_promoted=False))


def still(root, name):
    environment(); verify(root)
    from touchdown import studio_build as studio
    from touchdown.authored_scene import generate_asset
    selected, cfg = case(root)
    layout = read(root/'scene-layout.json')
    folder = root/'terrain'
    region = cfg['words']['region']
    if name == 'texture':
        refs = [folder/'regional-layout.png',folder/'cloud-exit.png']
        prompt = (
            'Turn the flat plan in image 1 into a realistic vertical NADIR satellite orthophotograph. '
            'The camera points STRAIGHT DOWN at the surface beneath it. Preserve the exact shapes '
            'and positions of land, sea and other broad geographic features in image 1. '
            +layout['layout_description']+' '
            f'Image 2 supplies ONLY the natural photographic colours, terrain character and daylight of {region}. '
            'Do not copy its camera angle or enlarge the local landmark. Replace the flat colours '
            'with fine overhead photographic ground and water textures. All terrain is seen from '
            'directly above at uniform map scale right to every edge. Local buildings and people '
            'are too small to resolve. No sky, horizon, foreground mountain faces, vanishing point, '
            'text, outlines, map symbols, borders or picture inset. One continuous photographic surface.')
        seed = cfg['seed']+1
    else:
        refs = [folder/'globe-guide.png']
        prompt = (
            'Finish image 1 as a believable natural close-Earth orbital photograph. Preserve the '
            'curved horizon position, camera direction and positions and relative sizes of visible '
            'land and water regions projected onto Earth. The central destination remains microscopic; '
            'no individual landmark, castle, person or building is visible. Earth fills the frame '
            'with a thin blue atmospheric rim near the top and a narrow strip of black space. '
            'Add fine photographic surface detail, atmospheric scattering and restrained white weather '
            'systems with broad gaps revealing the SAME geographic shapes. '
            f'Natural muted colours and terrain appropriate to {region}. '
            'No close landscape pasted onto Earth, isolated whole globe, illustration, CGI, panels '
            'or text. Retain guide geometry while improving photographic appearance.')
        seed = cfg['seed']+2
    spec = dict(name=name,references=list(map(str,refs)),prompt=prompt,seed=seed,
                models=cfg['models'],width=1920,height=1088)
    write(folder/f'{name}-spec.json',spec)
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(root/'flight/pinned-model-paths.json'))
    write(root/'status.json',dict(stage=name,status='generating',app_promoted=False))
    with studio.runtime('comfy',folder):
        generate_asset(studio.COMFY_URL,spec,folder)
    verify(root)
    write(root/'status.json',dict(stage=name,status='awaiting_review',app_promoted=False))


def model_checks(folder):
    if not (FILL_CHECKS/'fill-model-stat-before-verification-completes.json').is_file():
        return fresh_model_checks(folder)
    stats = read(FILL_CHECKS/'fill-model-stat-before-verification-completes.json')
    checked = read(FILL_CHECKS/'expand-1/model-fingerprints.json')
    for record in checked:
        path=Path(record['path']).resolve();old=stats[str(path)];s=path.stat()
        if record['sha256']!=old['expected_sha256'] or (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)!=(
            old['size'],old['mtime_ns'],old['ctime_ns'],old['inode'],old['device']):
            raise ValueError('Previously byte-verified Fill model changed: '+str(path))
    write(folder/'model-checks.json',dict(policy='Reuse completed full-byte checks only with identical size/mtime/ctime/inode/device',
        previous_checks=str(FILL_CHECKS/'expand-1/model-fingerprints.json'),
        prior_record_sha256=sha(FILL_CHECKS/'expand-1/model-fingerprints.json'),models=checked,fresh_byte_hashes=False))


def fresh_model_checks(folder):
    """Full-byte check of the Fill models against workflows/touchdown/flux-fill-models.json, cached per file stat."""
    expected = read(REPO/'workflows/touchdown/flux-fill-models.json')
    cache_path = REPO/'models/FLUX-Fill/.verified.json'
    cache = read(cache_path) if cache_path.is_file() else {}
    checked = []
    for record in expected:
        path = (REPO/record['path']).resolve()
        if not path.is_file():
            raise ValueError(f"Missing Fill model {record['filename']}: put it at {path}")
        s = path.stat(); key = [s.st_size, s.st_mtime_ns, s.st_ino]
        if s.st_size != record['size']:
            raise ValueError('Fill model has the wrong size: '+str(path))
        if cache.get(str(path), {}).get('stat') != key:
            cache[str(path)] = dict(stat=key, sha256=sha(path))
        if cache[str(path)]['sha256'] != record['sha256']:
            raise ValueError('Fill model differs from the recorded download: '+str(path))
        checked.append({**record, 'path': str(path)})
    write(cache_path, cache)
    write(folder/'model-checks.json', dict(policy='Full-byte SHA-256 against flux-fill-models.json, reused while size/mtime/inode are unchanged',
        models=checked, fresh_byte_hashes=True))


def expand(root):
    environment();verify(root)
    from touchdown import fill_outpaint as fill,studio_build as studio,flux3
    from touchdown.flight_runtime import capture_runtime
    selected,cfg=case(root);folder=root/'terrain';model_checks(folder)
    layout=read(root/'scene-layout.json');target=layout['target_uv']
    paths=folder/'fill-paths.json'
    write(paths,dict(scene_fill=dict(is_default=True,**{role:str((REPO/'models/FLUX-Fill'/role).resolve())
        for role in ('diffusion_models','text_encoders','vae')})))
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(paths))
    capture_runtime(folder)
    (folder/'decisions').mkdir(exist_ok=True)
    with studio.runtime('comfy',folder):
        for index in range(1,4):
            verify(root)
            source=folder/('texture.png' if index==1 else f'expand-{index-1}/raw-outpaint.png')
            part=folder/f'expand-{index}'
            graph,photo=fill.prepare(source,part)
            graph['23']['inputs']['text']=(
                'Extend this vertical NADIR satellite orthophotograph beyond every edge. Keep the '
                'same completely straight-down view, uniform ground scale, illumination and texture. '
                f'Continue the existing connected geographic surface of {cfg["words"]["region"]}. '
                +layout['expansion_description']+' '
                'Preserve existing land/water boundaries through the central source area. New '
                'surroundings form a much wider region. Fine small-scale overhead topographic texture. '
                'No sky, horizon, oblique side faces, close foreground, giant landmark, repeated '
                'castle, labels, frame or inset photograph.')
            fill.validate(graph)
            ui=read(part/'workflow.json')
            next(n for n in ui['nodes'] if n['id']==23)['widgets_values'][0]=graph['23']['inputs']['text']
            write(part/'workflow.json',ui)
            graph['17']['inputs']['image']=flux3._upload(studio.COMFY_URL,photo)
            write(part/'submitted-graph.json',graph)
            response=flux3._call(studio.COMFY_URL,'/prompt',dict(prompt=graph,client_id=uuid.uuid4().hex,
                extra_data={'extra_pnginfo':{'workflow':ui}}))
            write(part/'submission.json',response);pid=response['prompt_id'];started=time.monotonic()
            write(root/'status.json',dict(stage='expand',index=index,status='generating',prompt_id=pid))
            print('START expansion',index,pid,flush=True)
            while True:
                history=flux3._call(studio.COMFY_URL,f'/history/{pid}').get(pid,{})
                if history.get('status',{}).get('status_str')=='error':
                    write(part/'history.json',history);raise RuntimeError('Native Fill failed')
                if history.get('status',{}).get('completed'):break
                if time.monotonic()-started>1800:raise TimeoutError('Native Fill timed out')
                time.sleep(2)
            write(part/'history.json',history)
            for node,name in [('9','raw-outpaint'),('47','padding-mask'),('48','padded-input')]:
                item=history['outputs'][node]['images'][0]
                with urllib.request.urlopen(studio.COMFY_URL+'/view?'+urllib.parse.urlencode(item),timeout=60) as r:
                    (part/f'{name}.png').write_bytes(r.read())
            write(part/'geometry.json',dict(expansions=index,original_target_uv=target,
                nominal_target_uv=[.5+(v-.5)/(2**index) for v in target],
                original_source_width_fraction=1/(2**index),raw_sha256=sha(part/'raw-outpaint.png'),
                source_sha256=sha(source),exact_pixel_preservation=False,elapsed_seconds=round(time.monotonic()-started,1)))
            write(root/'status.json',dict(stage='expand',index=index,status='awaiting_review'))
            print('REVIEW expansion',index,flush=True)
            decision=folder/f'decisions/expand-{index}.json';deadline=time.monotonic()+900
            while not decision.exists():
                if time.monotonic()>deadline:raise TimeoutError('Review checkpoint expired')
                time.sleep(1)
            if read(decision).get('continue') is not True:
                write(root/'status.json',dict(stage='expand',index=index,status='held'));return
    verify(root)
    write(root/'status.json',dict(stage='expand',status='complete'))


def project(root):
    import cv2
    import numpy as np
    verify(root);folder=root/'terrain';layout=read(root/'scene-layout.json')
    target=np.array([.5+(v-.5)/8 for v in layout['target_uv']])
    texture=folder/'expand-3/raw-outpaint.png';image=cv2.imread(str(texture));assert image is not None
    w,h=1920,1088;origin=np.array([0.,0.,1.08]);pitch=np.deg2rad(45)
    forward=np.array([0.,np.cos(pitch),-np.sin(pitch)]);up=np.array([0.,np.sin(pitch),np.cos(pitch)])
    x,y=np.meshgrid((np.arange(w)+.5-w/2)/(w/2),-(np.arange(h)+.5-h/2)/(w/2))
    ray=forward+np.tan(np.deg2rad(40))*(x[...,None]*np.array([1.,0.,0.])+y[...,None]*up)
    ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
    b=ray@origin;disc=b*b-(origin@origin-1);valid=(disc>=0)&(b<0)
    point=origin+ray*(-b-np.sqrt(np.maximum(0,disc)))[...,None]
    lon=np.arctan2(point[...,0],point[...,2]);lat=np.arcsin(np.clip(point[...,1],-1,1))
    t=1.08*np.sin(pitch)-np.sqrt((1.08*np.sin(pitch))**2-(1.08**2-1));center_lat=np.arcsin(t*np.cos(pitch))
    u=target[0]+.48*np.tanh(lon/.075);v=target[1]-.48*np.tanh((lat-center_lat)/.075)
    assert np.all(((u>=0)&(u<=1)&(v>=0)&(v<=1))[valid])
    mapped=cv2.remap(image,(u*(w-1)).astype(np.float32),(v*(h-1)).astype(np.float32),cv2.INTER_LINEAR)
    normal=point/np.linalg.norm(point,axis=-1,keepdims=True);cosine=np.clip(np.sum(normal*(-ray),axis=-1),0,1)
    haze=((1-cosine)**3*.65)[...,None]
    output=(mapped*(1-haze)+np.array([205.,157.,105.])*haze).clip(0,255).astype(np.uint8)
    output[~valid]=[8,5,3];cv2.imwrite(str(folder/'globe-guide.png'),output)
    annotated=output.copy();cv2.drawMarker(annotated,(w//2,h//2),(0,220,255),cv2.MARKER_CROSS,24,2)
    cv2.imwrite(str(folder/'globe-target-diagnostic.png'),annotated)
    write(folder/'projection.json',dict(texture=str(texture),texture_sha256=sha(texture),
        original_target_uv=layout['target_uv'],expanded_target_uv=target.tolist(),nominal_target_screen_xy=[w/2,h/2],
        height_in_radii=.08,pitch_degrees=45,horizontal_fov_degrees=80,
        mapping='u=target_x+0.48*tanh(longitude/0.075); v=target_y-0.48*tanh((latitude-centre_latitude)/0.075)',
        uniform_angular_scale=False,metric_geography=False,target_is_nominal_ancestry=True,
        guide_sha256=sha(folder/'globe-guide.png')))
    write(root/'status.json',dict(stage='project',status='awaiting_review'))


def opening_prepare(root):
    import cv2
    folder=root/'opening'
    if folder.exists():raise ValueError('Fresh opening root required')
    folder.mkdir();selected,cfg=case(root)
    source=root/'flight/run/motion/earth_zoom.mp4';cap=cv2.VideoCapture(str(source))
    info=[int(cap.get(7)),int(cap.get(3)),int(cap.get(4)),cap.get(5)];cap.release()
    assert info[0]>49 and info[3]==24
    shutil.copytree(root/'flight/frozen-code',folder/'frozen-code')
    shutil.copy2(root/'flight/pinned-model-paths.json',folder/'pinned-model-paths.json')
    shutil.copy2(TIMING,folder/'timing.json')
    orbit=root/'terrain/orbit.png';shutil.copy2(orbit,folder/'grounded-orbit-full.png')
    image=cv2.imread(str(orbit));h,w=image.shape[:2];cw=round(h*1280/736);left=(w-cw)//2
    image=cv2.resize(image[:,left:left+cw],(1280,736),interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(folder/'orbit-start.png'),image)
    zw,zh=round(1280/1.8),round(736/1.8);x,y=(1280-zw)//2,(736-zh)//2
    cv2.imwrite(str(folder/'closer-earth.png'),cv2.resize(image[y:y+zh,x:x+zw],(1280,736),interpolation=cv2.INTER_LANCZOS4))
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vf',
        'trim=start_frame=40:end_frame=49,reverse,scale=1280:736,setpts=PTS-STARTPTS',
        '-an','-c:v','libx264','-crf','12','-pix_fmt','yuv420p',str(folder/'cloud-retreat-prefix.mp4')],check=True)
    files=[p for p in (folder/'frozen-code').rglob('*') if p.is_file()
           and '__pycache__' not in p.parts and p.name!='extra-paths.json']
    files += [source,orbit,folder/'orbit-start.png',folder/'closer-earth.png',folder/'grounded-orbit-full.png',
              folder/'cloud-retreat-prefix.mp4',folder/'pinned-model-paths.json',folder/'timing.json',Path(__file__).resolve()]
    write(folder/'experiment.json',dict(place=selected['place'],source=str(source),source_info=info,
        parent_seed=cfg['seed']+32,generated_frames=49,generated_size=[1280,736],
        source_prefix_frames=list(range(48,39,-1)),generated_delivery_frames=list(range(48,8,-1)),
        retained_source_frames=list(range(40,info[0])),strengths={'40':.5},closer_scale=1.8,
        closer_crop=[x,y,zw,zh],orbital_aspect_crop=[left,0,left+cw,h],crossfade=False,retiming=False,
        hashes={str(p.resolve()):sha(p) for p in files},app_promoted=False))
    write(root/'status.json',dict(stage='opening',status='prepared'))


def opening_verify(folder):
    for path,expected in read(folder/'experiment.json')['hashes'].items():
        if sha(Path(path))!=expected:raise ValueError('Changed opening input: '+path)


def opening_worker(root):
    environment();folder=root/'opening';opening_verify(folder);plan=read(folder/'experiment.json')
    code=folder/'frozen-code';sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    os.environ['PYTHONPATH']=os.pathsep.join(sys.path[:2])
    from touchdown import split_flight as flight,studio_build as studio
    import fakeout_earth_zoom as f
    assert Path(flight.__file__).is_relative_to(code)
    studio.REPO=REPO
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--deterministic' not in shlex.split(command):command+=' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --extra-model-paths-config '+shlex.quote(str(folder/'pinned-model-paths.json'))
    flight.enable_prompt_relay();part=folder/'generation';part.mkdir()
    write(root/'status.json',dict(stage='opening',status='generating'))
    with studio.runtime('comfy',folder):
        with f._client(studio.VIDEO_URL) as client:
            graph={'50':{'class_type':'LoadVideo','inputs':{'file':client.assets.from_file(str(folder/'cloud-retreat-prefix.mp4'))}},
                   '51':{'class_type':'GetVideoComponents','inputs':{'video':['50',0]}}}
            closer=f._load(graph,client,'47',folder/'closer-earth.png');orbit=f._load(graph,client,'46',folder/'orbit-start.png')
            saved,_=flight._generate(f,client,graph,49,[(0,['51',0]),(40,closer),(-1,orbit)],{40:.5},
                read(folder/'timing.json'),[0,4.5,40,48],plan['parent_seed'],'touchdown/grounded_scene_opening',part,dict(width=1280,height=736))
            write(folder/'generation-result.json',flight._record(saved,f))
    opening_verify(folder);write(root/'status.json',dict(stage='opening',status='awaiting_review'))


def assemble(root):
    folder=root/'opening';opening_verify(folder);plan=read(folder/'experiment.json');result=read(folder/'generation-result.json')
    assert sha(Path(result['video']))==result['sha256']
    count,w,h,fps=plan['source_info'];dest=root/'complete-flight.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',result['video'],'-i',plan['source'],'-filter_complex',
        f'[0:v]trim=start_frame=9:end_frame=49,reverse,scale={w}:{h},setpts=PTS-STARTPTS[a];'
        '[1:v]trim=start_frame=40,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]',
        '-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(dest)],check=True)
    import cv2
    cap=cv2.VideoCapture(str(dest));assert (int(cap.get(7)),cap.get(5))==(count,24);cap.release()
    write(root/'assembly.json',dict(video=str(dest),sha256=sha(dest),frames=count,fps=24,
        replacement_frames=40,retained_source_frames=count-40,duration_seconds=count/24,
        crossfade=False,retiming=False,export='H.264 CRF16; later source frames re-encoded at identical frame positions'))
    write(root/'status.json',dict(stage='assembled',status='awaiting_review',app_promoted=False));print(dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['terrain-prepare','texture','expand','project','orbit','opening-prepare','opening-worker','assemble'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.stage in ('texture','orbit'):still(root,a.stage)
    else:globals()[a.stage.replace('-','_')](root)
