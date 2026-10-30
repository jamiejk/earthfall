"""Fast fixed-lens Earth-to-person motion revision, without image crossfades."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from touchdown.authored_film import encode
from touchdown.authored_scene import digest
from touchdown.earth_fall_clouds import CloudField, Earth, opening_pose, rays

OPENING_FRAMES=48
DESCENT_FRAMES=120
FPS=24


def descent_pose(t):
    if not 0<=t<=1:raise ValueError('progress must be in [0,1]')
    distance=2.8+597.2*(1-t)**4
    return dict(position=[0.,-distance,distance-1.28],target=[0.,0.,1.16],focal_length_mm=35.)


def prepare(source,out,width,height,preview,globe,connected=False):
    out.mkdir(parents=True,exist_ok=True)
    for name in ['master.png','ground.png','person.png','person_rgba.png','orbit.png','under_cloud.png',
                 'route_photo.png','route_wide_photo.png','route_photo.json','route_wide_photo.json','ATTRIBUTION.md']:
        shutil.copy2(source/name,out/name)
    previous=json.loads((source/'render.json').read_text())
    renderer='connected_canyon_blender.py' if connected else 'authored_spur_blender.py'
    cfg=dict(width=width,height=height,frames=DESCENT_FRAMES,fps=FPS,preview=preview,matte=previous['matte'],
             poses=[descent_pose(i/(DESCENT_FRAMES-1)) for i in range(DESCENT_FRAMES)],
             projection_poses=previous.get('projection_poses',previous['poses']),regional_ground=True,
             assets={name:digest(out/f'{name}.png') for name in ['master','ground','person','route_photo','route_wide_photo','under_cloud']},
             scene='connected_canyon' if connected else 'earth_fall_fast_45_degree',cloud_motion=connected,
             renderer_sha256=digest(Path(__file__).with_name(renderer)))
    if connected:
        from PIL import Image
        terrain=Path('outputs/touchdown/canyonlands-grand-view/terrain/b46baae38211/dem').resolve()
        cfg['terrain_cache']=str(terrain)
        cfg['terrain_hashes']={name:digest(terrain/name) for name in ['height_detail.npy','height_far.npy']}
        master=Image.open(out/'master.png');mw,mh=master.size
        master.crop((round(mw*.18),round(mh*.30),round(mw*.50),round(mh*.43))).save(out/'cliff_texture.png')
        master.crop((0,0,mw,round(mh*.275))).save(out/'sky_texture.png')
    (out/'render.json').write_text(json.dumps(cfg,indent=2))
    (out/'flight.json').write_text(json.dumps(dict(opening_frames=OPENING_FRAMES,descent_frames=DESCENT_FRAMES,
        fps=FPS,fixed_lens_mm=35,opening_poses=[opening_pose(i/(OPENING_FRAMES-1)) for i in range(OPENING_FRAMES)],
        cloud_exit_altitude_m=400,globe=str(globe.resolve()),
        inputs={name:digest(globe/name) for name in ['globe_ground.jpg','cap_ground.jpg','globe.json']},
        cloud_code_sha256=digest(Path(__file__).with_name('earth_fall_clouds.py')),
        scope='Spatial camera travel; hard splice while inside opaque cloud. No crop zoom, timed fade or optical zoom.'),indent=2))
    with (out/'ATTRIBUTION.md').open('a') as f:
        f.write('\nOrbital ground: cached NASA GIBS Blue Marble global and regional textures, public domain; see source globe.json.\n')


def render_opening(out,globe,preview=False):
    import numpy as np
    from PIL import Image
    cfg=json.loads((out/'render.json').read_text());w,h=cfg['width'],cfg['height']
    # Ray casting at 960px keeps the frame review fast; geometry and camera are
    # unchanged at output resolution. Cloud scattering is supersampled by layers.
    work_w=min(w,960);work_h=round(work_w*h/w)
    earth=Earth(globe);clouds=CloudField()
    directory=out/'opening';directory.mkdir(exist_ok=True)
    indices=[0,8,16,24,32,40,47] if preview else range(OPENING_FRAMES)
    records=[]
    for i in indices:
        pose=opening_pose(i/(OPENING_FRAMES-1));direction=rays(work_w,work_h,pose['pitch_deg'])
        background=earth.render(pose['position'],direction)
        colour,trans=clouds.transport(pose['position'],direction,1200,3800,spherical=True,layers=40,
                                     time_s=i/FPS if cfg.get('cloud_motion') else None)
        im=Image.fromarray(np.clip(colour+background*trans[...,None],0,255).astype('uint8'))
        im.resize((w,h),Image.Resampling.BICUBIC).save(directory/f'{i:04d}.png')
        records.append(dict(frame=i,minimum_transmission=float(trans.min()),maximum_transmission=float(trans.max()),
                            position=pose['position']))
        print(f'opening {i+1}/{OPENING_FRAMES}',flush=True)
    (out/'opening_transport.json').write_text(json.dumps(records,indent=2))


def compose(out,approach_video=None):
    import numpy as np
    import cv2
    from PIL import Image,ImageOps
    cfg=json.loads((out/'render.json').read_text());w,h=cfg['width'],cfg['height']
    work_w=min(w,960);work_h=round(work_w*h/w);clouds=CloudField()
    suffix='_living' if approach_video else ''
    directory=out/f'film{suffix}_frames';directory.mkdir(exist_ok=True)
    approach=[]
    if approach_video:
        cap=cv2.VideoCapture(str(approach_video))
        while True:
            ok,frame=cap.read()
            if not ok:break
            approach.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        if len(approach)<DESCENT_FRAMES:raise ValueError('The treated approach must contain at least 120 frames')
    for i in range(OPENING_FRAMES):
        # Preserve the rendered frame exactly. No incoming blend or zoom.
        shutil.copy2(out/'opening'/f'{i:04d}.png',directory/f'{i:04d}.png')
    records=[]
    incoming_cloud=np.asarray(Image.open(out/'opening'/f'{OPENING_FRAMES-1:04d}.png'),np.float32).mean((0,1))
    lighting_gain=None
    for i,pose in enumerate(cfg['poses']):
        if approach:
            image=np.asarray(ImageOps.fit(Image.fromarray(approach[i]),(w,h),Image.Resampling.LANCZOS),np.float32)
        else:
            image=np.asarray(Image.open(out/'frames'/f'{i:04d}.png').convert('RGB'),np.float32)
        x,y,z=pose['position'];target=pose['target']
        pitch=math.degrees(math.atan2(z-target[2],target[1]-y))
        direction=rays(work_w,work_h,pitch)
        colour,trans=clouds.transport(pose['position'],direction,250,900,layers=48,tile_m=5000.,ragged_base=400,
                                     time_s=(OPENING_FRAMES+i)/FPS if cfg.get('cloud_motion') else None)
        if lighting_gain is None:
            lighting_gain=incoming_cloud/np.maximum(colour.mean((0,1)),1)
        # One constant lighting calibration across the whole local cloud field.
        # This matches illumination at the hidden splice; it is not a dissolve.
        colour*=lighting_gain
        colour=cv2.resize(colour,(w,h),interpolation=cv2.INTER_CUBIC)
        trans=cv2.resize(trans,(w,h),interpolation=cv2.INTER_LINEAR)
        result=np.clip(colour+image*trans[...,None],0,255).astype('uint8')
        Image.fromarray(result).save(directory/f'{OPENING_FRAMES+i:04d}.png')
        records.append(dict(frame=OPENING_FRAMES+i,camera_altitude=z,mean_transmission=float(trans.mean()),
                            minimum_transmission=float(trans.min()),maximum_transmission=float(trans.max())))
    (out/f'exit_transport{suffix}.json').write_text(json.dumps(records,indent=2))
    flight=json.loads((out/'flight.json').read_text())
    flight['exit_cloud_code_sha256']=digest(Path(__file__).with_name('earth_fall_clouds.py'))
    flight['cloud_exit']='Spatially varying billow bottoms around 400m, rather than a flat layer or timed dissolve'
    flight['local_cloud_lighting_gain']=lighting_gain.tolist()
    if approach_video:flight['approach_video']=dict(path=str(approach_video),sha256=digest(approach_video),source_frames=len(approach),used_frames=DESCENT_FRAMES)
    (out/f'flight{suffix}.json').write_text(json.dumps(flight,indent=2))
    encode(directory/'%04d.png',OPENING_FRAMES+DESCENT_FRAMES,out/f'earth_fall{suffix}.mp4')
    print(out/f'earth_fall{suffix}.mp4',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/touchdown/authored-spur-v2'))
    p.add_argument('--output',type=Path,default=Path('outputs/touchdown/earth-fall-v3'))
    p.add_argument('--globe',type=Path,default=Path('outputs/scenes/mather-point-grand-canyon-region-36p0617--112p1078-r100200/globe'))
    p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080)
    p.add_argument('--preview',action='store_true');p.add_argument('--compose-only',action='store_true')
    p.add_argument('--connected',action='store_true',help='Connected terrain and evolving clouds')
    p.add_argument('--approach-video',type=Path,help='With --compose-only, assemble a treated approach separately')
    a=p.parse_args()
    if min(a.width,a.height)<16 or a.width%2 or a.height%2:p.error('Use positive even dimensions')
    out=a.output.resolve()
    if a.compose_only:compose(out,a.approach_video);return
    prepare(a.source.resolve(),out,a.width,a.height,a.preview,a.globe,a.connected)
    script=Path(__file__).with_name('connected_canyon_blender.py' if a.connected else 'authored_spur_blender.py')
    subprocess.run(['blender','-b','-t','8','--factory-startup','--python',str(script),'--',str(out)],check=True)
    render_opening(out,a.globe,preview=a.preview)
    if not a.preview:
        encode(out/'frames'/'%04d.png',DESCENT_FRAMES,out/'approach.mp4')
        compose(out)

if __name__=='__main__':main()
