"""Flat regional layout -> reviewed native Fill expansions -> tracked globe -> FLUX still."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import time
import urllib.parse
import urllib.request
import uuid

from touchdown.orbital_reference import BASE, REPO, environment, read, sha, write

OLD = REPO/'outputs/touchdown/studio/studio-eilean-donan-globe-reference-v1-20261027'
W, H = 1920, 1088
TARGET = (.52, .58)
EXPANSIONS = 3


def prepare(root):
    import cv2
    import numpy as np
    if root.exists():
        raise ValueError('Use a fresh experiment root')
    root.mkdir(parents=True)
    shutil.copy2(OLD/'cloud-exit-frame-064.png', root/'cloud-exit.png')
    # Interpretive plan, not satellite registration. No oblique image is wrapped as a map.
    polygons = [
        [(0.14,0),(1,0),(1,1),(.78,1),(.73,.88),(.76,.76),(.86,.65),(.93,.48),(.91,.37),(.78,.29),(.57,.28),(.36,.22),(.23,.14)],
        [(0,.88),(.12,.85),(.22,.88),(.34,.84),(.42,.77),(.48,.75),(.56,.79),(.64,.88),(.68,1),(0,1)],
        [(.503,.568),(.514,.559),(.531,.561),(.542,.573),(.535,.585),(.517,.588),(.505,.581)],
        [(.532,.600),(.540,.597),(.547,.602),(.542,.617),(.531,.625),(.524,.622)],
    ]
    water, land = '#23495c', '#68754b'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="{water}"/>']
    image = np.full((H,W,3),(92,73,35),np.uint8)
    for poly in polygons:
        points = [(round(x*W),round(y*H)) for x,y in poly]
        svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{land}"/>')
        cv2.fillPoly(image,[np.array(points,np.int32)],(75,117,104))
    svg.append('</svg>')
    (root/'regional-layout.svg').write_text('\n'.join(svg)+'\n')
    cv2.imwrite(str(root/'regional-layout.png'),image)
    write(root/'layout.json',dict(polygons=polygons,target_uv=TARGET,heading='Image top points away from the arrival shore',
        provenance='Authored approximate plan inferred from cloud-exit island, upper/right shore and later near shore',
        measured_geography=False,water_colour=water,land_colour=land))
    files = [root/'cloud-exit.png',root/'regional-layout.svg',root/'regional-layout.png',root/'layout.json',
             Path(__file__).resolve(),REPO/'scripts/touchdown/fill_outpaint.py',REPO/'scripts/touchdown/flux3.py',
             REPO/'scripts/touchdown/authored_scene.py',REPO/'workflows/touchdown/flux-fill-outpaint.api.json']
    write(root/'experiment.json',dict(source=str(BASE/'run/motion/earth_zoom.mp4'),
        source_sha256=sha(BASE/'run/motion/earth_zoom.mp4'),expansions=EXPANSIONS,expansion_per_axis=2,
        total_expansion_per_axis=2**EXPANSIONS,target_uv=TARGET,
        source_photo_conditioning='Texture stage uses layout as image 1; cloud-exit photo as appearance-only image 2',
        scope='Still-only experimental authored continuity; no measured map, no video generation or promotion',
        hashes={str(p.resolve()):sha(p) for p in files},app_promoted=False))
    write(root/'status.json',dict(status='prepared',video_jobs=0))


def verify(root):
    plan=read(root/'experiment.json')
    for name,expected in plan['hashes'].items():
        if sha(Path(name))!=expected:
            raise ValueError('Changed experiment input: '+name)
    if sha(Path(plan['source']))!=plan['source_sha256']:
        raise ValueError('Original film changed')


def still(root,name):
    environment();verify(root)
    from touchdown import studio_build as studio
    from touchdown.authored_scene import generate_asset
    cfg=read(REPO/'outputs/touchdown/studio/studio-eilean-donan-e1-20261026/scene-config.json')
    if name=='texture':
        refs=[root/'regional-layout.png',root/'cloud-exit.png']
        prompt=(
            'Turn the flat plan in image 1 into a realistic vertical NADIR satellite orthophotograph. '
            'The camera points STRAIGHT DOWN at the surface beneath it. Preserve the exact blue-water '
            'and green-land shapes and their positions in image 1, including the small central island '
            'and nearby narrow rocky spit. Image 2 supplies ONLY natural Scottish Highland colours, '
            'mountain vegetation, dark blue sea-loch water and daylight. Do not copy its camera angle '
            'or enlarge its castle. Replace flat green with fine overhead vegetation and mountain '
            'ridge textures; replace blue with natural water texture. All land and water are seen '
            'from directly above at uniform map scale, right to every edge. Local buildings and '
            'people are too small to resolve. No sky, horizon, side faces of hills, foreground, '
            'atmospheric vanishing point, text, outlines, map symbols, borders or picture inset. '
            'One uninterrupted photographic geographic surface, not an oblique landscape photograph.')
        seed=2026997937
    else:
        refs=[root/'globe-guide.png']
        prompt=(
            'Finish image 1 as a believable natural close-Earth orbital photograph. Preserve the '
            'supplied curved horizon position, camera direction, and the positions and relative sizes '
            'of visible land and water regions projected onto Earth. The tiny destination in the '
            'centre remains microscopic; no castle, people or individual buildings are visible. '
            'Earth fills the frame, with a thin blue atmospheric rim near the top and a narrow strip '
            'of black space above it. Add fine photographic surface detail, atmospheric scattering '
            'and restrained white weather systems with broad gaps revealing the same geographic '
            'shapes. Natural muted blue water and green-brown Scottish regional terrain. No giant '
            'local island, close landscape pasted onto Earth, isolated whole globe, illustration, '
            'CGI, panels or text. Retain the guide geometry while improving photographic appearance.')
        seed=2026997938
    spec=dict(name=name,references=list(map(str,refs)),prompt=prompt,seed=seed,models=cfg['models'],width=W,height=H)
    write(root/f'{name}-spec.json',spec)
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(BASE/'pinned-model-paths.json'))
    write(root/'status.json',dict(status='generating_'+name,video_jobs=0))
    with studio.runtime('comfy',root):
        generate_asset(studio.COMFY_URL,spec,root)
    verify(root)
    write(root/'status.json',dict(status=name+'_awaiting_review',video_jobs=0))


def model_checks(root):
    stats=read(OLD/'fill-model-stat-before-verification-completes.json')
    checked=read(OLD/'expand-1/model-fingerprints.json')
    for record in checked:
        path=Path(record['path']).resolve();old=stats[str(path)];s=path.stat()
        if record['sha256']!=old['expected_sha256'] or (
            s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)!=(
            old['size'],old['mtime_ns'],old['ctime_ns'],old['inode'],old['device']):
            raise ValueError('Previously byte-verified Fill model changed: '+str(path))
    write(root/'fill-model-checks.json',dict(policy='Reuse previous full-byte hashes with matching size/mtime/ctime/inode/device',
        previous_checks=str(OLD/'expand-1/model-fingerprints.json'),previous_checks_sha256=sha(OLD/'expand-1/model-fingerprints.json'),
        stat_record_sha256=sha(OLD/'fill-model-stat-before-verification-completes.json'),models=checked,fresh_byte_hashes=False))


def expansions(root):
    environment();verify(root);model_checks(root)
    from touchdown import studio_build as studio,fill_outpaint as fill,flux3
    from touchdown.flight_runtime import capture_runtime
    paths=root/'fill-paths.json'
    write(paths,dict(flat_region_fill=dict(is_default=True,**{role:str((REPO/'models/FLUX-Fill'/role).resolve())
        for role in ('diffusion_models','text_encoders','vae')})))
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(paths))
    capture_runtime(root)
    (root/'decisions').mkdir(exist_ok=True)
    with studio.runtime('comfy',root):
        for index in range(1,EXPANSIONS+1):
            verify(root)
            source=root/('texture.png' if index==1 else f'expand-{index-1}/raw-outpaint.png')
            folder=root/f'expand-{index}'
            graph,photo=fill.prepare(source,folder)
            graph['23']['inputs']['text']=(
                'Extend this vertical NADIR satellite orthophotograph beyond every edge. Keep the '
                'same completely straight-down view, uniform ground scale, illumination and photographic '
                'texture. Continue the connected irregular sea lochs and dark blue coastal water, '
                'branching inlets, green-brown mountain ridges and drainage in the Scottish Highlands. '
                'Preserve the existing land/water boundaries through the central source area. '
                'New surroundings form a much wider region. Fine small-scale topographic texture '
                'seen from directly above. No sky or horizon anywhere, no oblique mountain side faces, '
                'no close foreground, giant island, castle, labels, frame or inset photograph.')
            fill.validate(graph)
            ui=read(folder/'workflow.json')
            next(n for n in ui['nodes'] if n['id']==23)['widgets_values'][0]=graph['23']['inputs']['text']
            write(folder/'workflow.json',ui)
            graph['17']['inputs']['image']=flux3._upload(studio.COMFY_URL,photo)
            write(folder/'submitted-graph.json',graph)
            response=flux3._call(studio.COMFY_URL,'/prompt',dict(prompt=graph,client_id=uuid.uuid4().hex,
                extra_data={'extra_pnginfo':{'workflow':ui}}))
            write(folder/'submission.json',response)
            started=time.monotonic();pid=response['prompt_id']
            write(root/'status.json',dict(status='expanding',index=index,prompt_id=pid,video_jobs=0))
            print('START expansion',index,pid,flush=True)
            while True:
                history=flux3._call(studio.COMFY_URL,f'/history/{pid}').get(pid,{})
                if history.get('status',{}).get('status_str')=='error':
                    write(folder/'history.json',history);raise RuntimeError('Native Fill failed')
                if history.get('status',{}).get('completed'):break
                if time.monotonic()-started>1800:raise TimeoutError('Native Fill timed out')
                time.sleep(2)
            write(folder/'history.json',history)
            for node,name in [('9','raw-outpaint'),('47','padding-mask'),('48','padded-input')]:
                item=history['outputs'][node]['images'][0]
                with urllib.request.urlopen(studio.COMFY_URL+'/view?'+urllib.parse.urlencode(item),timeout=60) as r:
                    (folder/f'{name}.png').write_bytes(r.read())
            target=[.5+(v-.5)/(2**index) for v in TARGET]
            write(folder/'geometry.json',dict(expansions=index,original_target_uv=TARGET,
                nominal_target_uv=target,original_source_width_fraction=1/(2**index),
                raw_sha256=sha(folder/'raw-outpaint.png'),source_sha256=sha(source),
                exact_pixel_preservation=False,elapsed_seconds=round(time.monotonic()-started,1)))
            write(root/'status.json',dict(status='expansion_awaiting_review',index=index,video_jobs=0))
            print('REVIEW expansion',index,flush=True)
            decision=root/f'decisions/expand-{index}.json';deadline=time.monotonic()+900
            while not decision.exists():
                if time.monotonic()>deadline:raise TimeoutError('Review checkpoint expired; saved result retained')
                time.sleep(1)
            if read(decision).get('continue') is not True:
                write(root/'status.json',dict(status='held_expansion_review',index=index,video_jobs=0));return
    verify(root)
    write(root/'status.json',dict(status='expanded_awaiting_projection',video_jobs=0))


def project(root):
    import cv2
    import numpy as np
    verify(root)
    texture=root/f'expand-{EXPANSIONS}/raw-outpaint.png'
    image=cv2.imread(str(texture));assert image is not None
    target=np.array([.5+(v-.5)/(2**EXPANSIONS) for v in TARGET])
    origin=np.array([0.,0.,1.08]);pitch=np.deg2rad(45)
    forward=np.array([0.,np.cos(pitch),-np.sin(pitch)]);up=np.array([0.,np.sin(pitch),np.cos(pitch)])
    x,y=np.meshgrid((np.arange(W)+.5-W/2)/(W/2),-(np.arange(H)+.5-H/2)/(W/2))
    ray=forward+np.tan(np.deg2rad(40))*(x[...,None]*np.array([1.,0.,0.])+y[...,None]*up)
    ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
    b=ray@origin;disc=b*b-(origin@origin-1);valid=(disc>=0)&(b<0)
    point=origin+ray*(-b-np.sqrt(np.maximum(0,disc)))[...,None]
    lon=np.arctan2(point[...,0],point[...,2]);lat=np.arcsin(np.clip(point[...,1],-1,1))
    t=1.08*np.sin(pitch)-np.sqrt((1.08*np.sin(pitch))**2-(1.08**2-1));center_lat=np.arcsin(t*np.cos(pitch))
    u=target[0]+lon/1.6;v=target[1]-(lat-center_lat)/.9
    assert np.all(((u>=0)&(u<=1)&(v>=0)&(v<=1))[valid]),'Texture coverage failed'
    mapped=cv2.remap(image,(u*(W-1)).astype(np.float32),(v*(H-1)).astype(np.float32),cv2.INTER_LINEAR)
    normal=point/np.linalg.norm(point,axis=-1,keepdims=True)
    cosine=np.clip(np.sum(normal*(-ray),axis=-1),0,1);haze=((1-cosine)**3*.65)[...,None]
    output=(mapped*(1-haze)+np.array([205.,157.,105.])*haze).clip(0,255).astype(np.uint8)
    output[~valid]=[8,5,3];cv2.imwrite(str(root/'globe-guide.png'),output)
    annotated=output.copy();cv2.drawMarker(annotated,(W//2,H//2),(0,220,255),cv2.MARKER_CROSS,24,2)
    cv2.imwrite(str(root/'globe-target-diagnostic.png'),annotated)
    write(root/'projection.json',dict(texture=str(texture),texture_sha256=sha(texture),
        original_target_uv=TARGET,expanded_target_uv=target.tolist(),nominal_target_screen_xy=[W/2,H/2],
        height_in_radii=.08,pitch_degrees=45,horizontal_fov_degrees=80,longitude_span_radians=1.6,
        latitude_span_radians=.9,original_source_width_fraction=1/8,metric_geography=False,
        target_is_nominal_ancestry=True,guide_sha256=sha(root/'globe-guide.png')))
    write(root/'status.json',dict(status='projected_awaiting_review',video_jobs=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','texture','expand','project','orbit'])
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.stage in ('texture','orbit'):still(root,a.stage)
    elif a.stage=='expand':expansions(root)
    else:globals()[a.stage](root)
