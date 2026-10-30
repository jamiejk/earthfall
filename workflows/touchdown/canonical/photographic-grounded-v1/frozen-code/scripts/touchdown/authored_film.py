"""Render the authored sandstone-spur approach and a ten-second editing proof.

The opening uses still plates and an authored cloud transition. The approach uses
persistent Blender geometry. This experiment does not produce an AI video pass.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.authored_scene import digest, person_matte


def centre(y):
    return 5.5 * math.sin(y * .038) * min(1., abs(y)/14.)


def surface(x, y):
    """Continuous ridge surface, with a flat contact patch at the destination."""
    away = min(1., (x*x+y*y)/5.)
    lateral = x-centre(y)
    return away * (max(0., -y)*.055 + .10*math.sin(y*.43) +
                   .055*math.sin(x*2.1+y*.4) + .025*math.sin(x*5-y*2) + .018*lateral*lateral)


def spur_pose(t):
    if not 0 <= t <= 1:
        raise ValueError('Camera progress must be in [0, 1]')
    r=1-t
    distance=2.8+102.2*r**2.4
    y=-distance
    x=centre(y)+3.4*r**3
    z=1.52+31*r**2.6
    target_y=min(0.,y+max(7.,distance*.76))
    target_x=centre(target_y)
    return dict(position=[x,y,z],target=[target_x,target_y,surface(target_x,target_y)+1.16],
                focal_length_mm=35+3*t)


def cloud_mask(width,height,progress):
    """Deterministic expanding cloud billows; no random flicker between frames."""
    import numpy as np
    from PIL import Image
    import cv2
    # Cached stationary density field, sampled along an expanding flight path.
    if not hasattr(cloud_mask,'field'):
        rng=np.random.default_rng(20261022)
        field=np.zeros((768,1024),np.float32)
        for size,weight in [(8,.45),(20,.28),(48,.16),(120,.08),(256,.03)]:
            layer=Image.fromarray((rng.random((size,size))*255).astype('uint8')).resize((1024,768),Image.Resampling.BICUBIC)
            field+=np.asarray(layer,dtype=np.float32)/255*weight
        cloud_mask.field=field
    yy,xx=np.mgrid[:height,:width].astype(np.float32)
    scale=1+progress*3
    u=(xx/width-.5)/scale*850+512+progress*50
    v=(yy/height-.5)/scale*650+384-progress*90
    noise=cv2.remap(cloud_mask.field,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    # Opaque centre interval conceals the change in scale; soft fronts move apart.
    envelope=max(0.,min(1.,(progress-.04)/.24,(1-progress)/.30))
    density=np.clip((noise-.29)*3.3,0,1)
    opacity=np.clip(envelope*1.75*density,0,1)
    occlusion=max(0.,min(1.,(progress-.26)/.08,(.82-progress)/.07))
    opacity=np.maximum(opacity,occlusion)
    colour=np.stack([noise*24+186,noise*24+194,noise*24+204],axis=-1)
    return colour,opacity


def compose(folder, approach_video=None):
    import numpy as np
    from PIL import Image,ImageOps
    cfg=json.loads((folder/'render.json').read_text())
    w,h=cfg['width'],cfg['height'];fps=cfg['fps'];n=cfg['frames']
    # Fixed total of ten seconds. The persistent set occupies the last 7.5s.
    lead=round(2.5*fps); total=lead+n
    suffix='_ltx' if approach_video else ''
    film=folder/f'film{suffix}_frames';film.mkdir(exist_ok=True)
    orbit_path=folder/('opening_orbit.png' if (folder/'opening_orbit.png').exists() else 'orbit.png')
    orbit=Image.open(orbit_path).convert('RGB')
    aerial=Image.open(folder/'under_cloud.png').convert('RGB')
    def zoom(im,scale,anchor):
        iw,ih=im.size; cw=iw/scale;ch=cw*h/w
        if ch>ih:ch=ih/scale;cw=ch*w/h
        x=max(0,min(iw-cw,iw*anchor[0]-cw/2));y=max(0,min(ih-ch,ih*anchor[1]-ch/2))
        return im.transform((w,h),Image.Transform.EXTENT,(x,y,x+cw,y+ch),Image.Resampling.BICUBIC)
    # Reuse the project's ray-cast cloud deck rather than a flat noise wipe.
    from touchdown.proc_cloud import render as render_cloud
    import cv2
    approach=[]
    if approach_video:
        cap=cv2.VideoCapture(str(approach_video))
        while True:
            ok,frame=cap.read()
            if not ok:break
            approach.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        if len(approach)<2:raise ValueError('Approach video is empty or unreadable')
    cloud_path=folder/'cloud_deck.mp4'
    cloud_key=digest(orbit_path)
    cloud_record=folder/'cloud_deck.json'
    if not cloud_path.exists() or not cloud_record.exists() or json.loads(cloud_record.read_text()).get('source')!=cloud_key:
        start=np.asarray(zoom(orbit,1+1.3*(.7/1.65)**2,(.5,.64)))
        render_cloud(start,cloud_path,frames=48,seed=17,blend=8,work=(960,544),size=768)
        cloud_record.write_text(json.dumps(dict(source=cloud_key,frames=48,seed=17)))
    cap=cv2.VideoCapture(str(cloud_path));clouds=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        clouds.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((w,h),Image.Resampling.BICUBIC))
    cap.release()
    if len(clouds)!=48:raise ValueError('Incomplete cloud deck')
    for i in range(total):
        t=i/fps
        if i>=lead:
            if approach:
                index=round((i-lead)*(len(approach)-1)/(n-1))
                im=Image.fromarray(approach[index]).resize((w,h),Image.Resampling.LANCZOS)
            else:
                im=Image.open(folder/'frames'/f'{i-lead:04d}.png').convert('RGB')
        elif t<1.65:
            im=zoom(orbit,1+1.3*(t/1.65)**2,(.5,.64))
        else:
            im=zoom(aerial,1+1.3*((t-1.65)/.85),(.52,.57))
        if .7<=t<2.7:
            im=clouds[min(47,round((t-.7)*fps))]
        elif 2.7<=t<3.35:
            # The fully opaque tail is the splice. Exit into the same set while
            # soft vapour clears; all subsequent frames remain in that set.
            colour,alpha=cloud_mask(w,h,.70+.30*(t-2.7)/.65)
            colour=np.asarray(clouds[-1],dtype=np.float32)
            frame=np.asarray(im,dtype=np.float32)
            im=Image.fromarray(np.clip(frame*(1-alpha[:,:,None])+colour*alpha[:,:,None],0,255).astype('uint8'))
        im.save(film/f'{i:04d}.jpg',quality=95)
    encode(film/'%04d.jpg',total,folder/f'earth_zoom{suffix}.mp4',fps)
    (folder/f'edit{suffix}.json').write_text(json.dumps(dict(duration=total/fps,fps=fps,opening_frames=lead,
        approach_frames=n,cloud_interval=[.7,3.35],
        limitation='Opening uses an animated orbital still and a ray-cast procedural cloud deck. '+
                    ('Approach is an LTX treatment and may alter the guide scene.' if approach_video else 'Approach is the deterministic set render.'),
        approach_video=dict(path=str(approach_video),sha256=digest(approach_video),source_frames=len(approach)) if approach_video else None,
        sources={'orbit':digest(orbit_path),'under_cloud':digest(folder/'under_cloud.png')}),indent=2))


def encode(pattern,frames,path,fps=24):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),
                    '-i',str(pattern),'-frames:v',str(frames),'-c:v','libx264','-crf','16',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)


def paint_route(folder, manifest, url, source, anchor=48):
    from PIL import Image,ImageOps
    from touchdown.authored_scene import read_scene, generate_asset
    cfg=read_scene(manifest)
    name='route_photo' if anchor==48 else 'route_wide_photo'
    layout=folder/f'{name}_layout.png'
    ImageOps.grayscale(Image.open(folder/'preview'/f'{anchor:04d}.png')).convert('RGB').save(layout)
    spec=dict(name=name,width=2688,height=1536,seed=2026102205,
        models=cfg['models'],context_parent=str(folder/'master.png'),
        references=[str(layout),str(source/'master_look_reference.png')]+
                   [r['local_path'] for r in cfg['references']],
        prompt=('Photorealistic National Geographic landscape photograph in Canyonlands, Utah. '
        'Image 1 is a crude geometry guide: preserve its camera viewpoint, winding central sandstone '
        'spur, broken dropoffs, and the positions of the large rock shoulders along the route. '
        'Replace all smooth CGI cylinders and stripes with naturally fractured, eroded sandstone '
        'buttresses, irregular cap rock, narrow ledges, deep cracks, talus and sparse desert scrub. '
        'Image 2 supplies the desired cream/copper stone palette and cool storm light. '
        'Images 3, 4 and 5 are real photographs of the location: their natural rock detail, irregular '
        'weathering and atmospheric depth should dominate the appearance. The central spur is a '
        'continuous walkable pale sandstone route curving toward the distant overlook. '
        'Keep this route clearly visible. Warm sun from the left, soft cool shadows. '
        'One seamless convincing photograph. Entirely uninhabited, no people anywhere, no red objects, '
        'no text or panels, no straight image borders, no manmade structures.'))
    generate_asset(url,spec,folder)


def video_pass(folder,url):
    from touchdown.terrain_v2v import render
    out=folder/'ltx'
    prompt=('One continuous cinematic drone descent along a winding sandstone spur in Canyonlands, Utah. '
        'Follow the guide camera, passing the foreground rock shoulders on either side, descending and '
        'slowing into an eye-level portrait of the same young woman in a red raincoat over a grey hoodie. '
        'The rock route, cliff edges and background remain stable throughout the move. '
        'Real naturally weathered sandstone, fine cracks, irregular erosion, sparse scrub, '
        'warm sunlight from the left, cool storm sky and soft atmospheric depth. '
        'Photographic travel cinematography. Ground the woman naturally in the scene, preserve her face, '
        'hair and clothing, very subtle natural movement, both feet planted. '
        'No scenery swaps, no extra people, no artificial cylinders, no repeating patterns.')
    path=render(folder/'approach.mp4',folder/'frames/0000.png',folder/'frames/0176.png',out,
                seed=2026102207,frames=177,prompt=prompt,url=url,
                mids=[(96,folder/'frames/0096.png'),(144,folder/'frames/0144.png')])
    (out/'source.json').write_text(json.dumps(dict(prompt=prompt,seed=2026102207,frames=177,
        guide_sha256=digest(folder/'approach.mp4'),video=str(path)),indent=2))
    print(path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/touchdown/authored-canyonlands'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080)
    p.add_argument('--preview',action='store_true');p.add_argument('--compose-only',action='store_true')
    p.add_argument('--approach-video',type=Path,help='Optional treated approach for a separate earth_zoom_ltx.mp4 edit')
    p.add_argument('--paint-route',action='store_true')
    p.add_argument('--video-pass',action='store_true')
    p.add_argument('--anchor',type=int,choices=[0,48],default=48)
    p.add_argument('--manifest',type=Path,default=Path('shots/authored/canyonlands.toml'))
    p.add_argument('--url',help='Plain ComfyUI for painting (default 8192); SDK proxy for video (default 8193)')
    a=p.parse_args()
    if min(a.width,a.height)<16 or a.width%2 or a.height%2:p.error('Use positive even output dimensions')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.video_pass:
        video_pass(out,a.url or 'http://127.0.0.1:8193');return
    if a.paint_route:
        paint_route(out,a.manifest,a.url or 'http://127.0.0.1:8192',a.source.resolve(),a.anchor);return
    if a.compose_only:
        compose(out,a.approach_video);return
    source=a.source.resolve()
    for name in ['master.png','ground.png','person.png','orbit.png','under_cloud.png','ATTRIBUTION.md']:
        shutil.copy2(source/name,out/name)
    config=tomllib.loads(a.manifest.read_text())
    if config.get('orbit_layout'):
        opening=(a.manifest.resolve().parent/config['orbit_layout']).resolve()
        shutil.copy2(opening,out/'opening_orbit.png')
        (out/'opening_orbit.json').write_text(json.dumps(dict(source=str(opening),sha256=digest(opening),
            reason='Reviewed regional orbital scale and cloud distribution'),indent=2))
    for name in ['route_photo','route_wide_photo']:
        record=out/f'{name}.json'
        if record.exists() and json.loads(record.read_text()).get('context_parent_sha256')!=digest(out/'master.png'):
            raise ValueError(f'{name} uses a different master; regenerate the route photograph')
    matte=person_matte(out/'person.png',out/'person_rgba.png')
    n=180
    cfg=dict(width=a.width,height=a.height,frames=n,fps=24,preview=a.preview,matte=matte,
             poses=[spur_pose(i/(n-1)) for i in range(n)],
             assets={name:digest(out/f'{name}.png') for name in ['master','ground','person','route_photo','route_wide_photo']
                     if (out/f'{name}.png').exists()},
             scene='sandstone_spur_v2',projection='Per-pixel camera projection with spatial coverage masks',
             renderer_sha256=digest(Path(__file__).with_name('authored_spur_blender.py')))
    (out/'render.json').write_text(json.dumps(cfg,indent=2))
    script=Path(__file__).with_name('authored_spur_blender.py')
    subprocess.run(['blender','-b','-t','8','--factory-startup','--python',str(script),'--',str(out)],check=True)
    if not a.preview:
        encode(out/'frames'/'%04d.png',n,out/'approach.mp4')
        compose(out)
        print(out/'earth_zoom.mp4',flush=True)

if __name__=='__main__':main()
