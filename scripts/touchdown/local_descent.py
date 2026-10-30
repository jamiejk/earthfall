"""Frozen 49-frame local descent diagnostic; never assembles/promotes a film."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess

import numpy as np
from touchdown.authored_scene import digest
from touchdown.flight_regression import read, write, verify_snapshot

ROOT = Path('outputs/touchdown/studio/studio-chillon-local-descent-v1-20261027').resolve()
PHOTO = Path('outputs/touchdown/studio/studio-chillon-neutral-ladder-v1-20261027/view-3.png').resolve()
SOURCE = Path('outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026').resolve()
SEED = 1399629552
PROMPT = ('A continuous live-action aerial camera descent over a blue-green lake shore. '
          'The camera travels forward and downward toward the SAME stationary adult woman, '
          'keeping her near the lower middle throughout, with natural ground parallax. '
          'She stands with both feet planted on grass, arms relaxed by her sides, wearing '
          'a beige knee-length trench coat and black shoes, brown hair, natural adult anatomy. '
          'Water stays left, grass in the middle, a narrow gravel path stays right. '
          'The camera looks downward throughout. Natural autumn daylight, photographic grass, '
          'gravel, fabric and rippling water. One uninterrupted moving shot.')


def project(position, target, points):
    position, target, points = [np.asarray(p,dtype=float) for p in [position,target,points]]
    f = target-position; f /= np.linalg.norm(f)
    r = np.cross(f,[0,0,1]); r /= np.linalg.norm(r)
    up = np.cross(r,f)
    d = points-position
    depth = d@f
    return np.stack([.5+(d@r)/depth*35/36, .5-(d@up)/depth*35/36*960/544],axis=-1)


def prepare(root):
    from scipy.optimize import least_squares
    if any(root.iterdir()):
        raise ValueError('Fresh output root required')
    desired = np.array([[.547,.542],[.547,.687]])
    def residual(v):
        h,tx,tz=v
        return (project([0,-.9*h,h],[tx,0,tz],[[0,0,1.78],[0,0,0]])-desired).ravel()
    fit = least_squares(residual,[14,-.8,3.],bounds=([3,-5,-3],[40,5,8]))
    h,tx,tz = fit.x
    initial_position = [0,-.9*h,h]
    initial_target = [tx,0,tz]
    # Layout calibration is approximate: shore/path are parallel ribbons and
    # only the subject's head/feet are numerically registered to the photo.
    def ground_x(screen_x):
        f = least_squares(lambda x:project(initial_position,initial_target,[[x[0],0,0]])[0,:1]-[screen_x],[0.])
        return float(f.x[0])
    poses=[]
    target_screen = desired.mean(axis=0)
    for i in range(49):
        u=i/48
        p=(1-u)*np.array(initial_position)+u*np.array([0,-5.5,3.])
        a=least_squares(lambda x:project(p,[x[0],0,x[1]],[[0,0,.89]])[0]-target_screen,[tx,tz])
        poses.append(dict(frame=i,position=p.tolist(),target=[float(a.x[0]),0,float(a.x[1])]))
    write(root/'camera-plan.json',dict(poses=poses,shore_x=ground_x(.265),path_x=ground_x(.665),
        fit_head_foot=project(initial_position,initial_target,[[0,0,1.78],[0,0,0]]).tolist(),
        desired_head_foot=desired.tolist(),fit_error_pixels=(residual(fit.x).reshape(2,2)*[960,544]).tolist(),
        calibration='Approximate manual subject/shore/path registration; no surveyed/depth reconstruction',
        lens_mm=35,frames=49,fps=24,temporal_resampling=False))
    shutil.copy2(PHOTO,root/'appearance.png')
    sources=[Path(__file__).resolve(),Path(__file__).with_name('local_descent_blender.py').resolve(),
             Path('src/geoblock_core/workflow.py').resolve(),PHOTO]
    write(root/'experiment.json',dict(seed=SEED,prompt=PROMPT,frames=49,fps=24,
        appearance='One elevated neutral photograph as untimed IC appearance reference in both stages',
        motion='New continuous, subject-tracked grey local camera guide in both stages',
        image_endpoint_pins=False,production_clock=dict(bounds=[0,29,58,108,156,192],fps=24,unchanged=True),
        source_hashes={str(p):digest(p) for p in sources},
        scope='Standalone diagnostic, no product timeline assignment; not a controlled single-variable comparison',
        changes=['bounded local geometry registered approximately to photo 3','no timed image endpoints',
                 'no castle/trees in guide or prompt','Layout adapter restored versus photo-only tests'],
        acceptance=['continuous camera descent, no cut/dissolve','subject remains planted and tracked',
                    'photographic adult appearance','near/far ground has relative perspective motion']))
    write(root/'status.json',dict(status='prepared',app_promoted=False))


def graph(guide,photo):
    from geoblock_core.workflow import ShotRequest,_add_ltx25
    g={'50':dict(class_type='LoadVideo',inputs=dict(file=guide)),
       '60':dict(class_type='GetVideoComponents',inputs=dict(video=['50',0])),
       '40':dict(class_type='LoadImage',inputs=dict(image=photo)),
       '41':dict(class_type='LTXVPreprocess',inputs=dict(image=['40',0],img_compression=18))}
    request=ShotRequest(name='local-descent',seed=SEED,output_prefix='touchdown/local_descent_v1',
                       ltx_upscale=True,save_stage1=True,pin_last_keyframe=False)
    _add_ltx25(g,request,49,['41',0],['41',0],[],PROMPT,layout=True)
    g['360']['inputs']['noise_seed']=SEED
    g['306']['inputs']['text']=''
    validate_graph(g)
    return g


def validate_graph(g):
    assert g['308']['inputs']==dict(width=960,height=544,length=49,batch_size=1)
    assert g['301']['inputs']['lora_name']=='ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors'
    assert g['301']['inputs']['strength_model']==1
    assert not any(n['class_type']=='LTXVAddGuide' for n in g.values())
    for motion,appearance in [('310','311'),('340','341')]:
        for node,image,index in [(motion,['60',0],0),(appearance,['41',0],-1)]:
            assert g[node]['class_type']=='LTXAddVideoICLoRAGuide'
            v=g[node]['inputs']
            assert (v['image'],v['frame_idx'],v['strength'])==(image,index,1)
    for noise,guider,sampler,sigma,steps in [('390','391','392','393',8),('360','361','362','363',3)]:
        assert g[noise]['inputs']['noise_seed']==SEED
        assert g[guider]['inputs']['cfg']==1
        assert g[sampler]['inputs']['sampler_name']=='euler_ancestral'
        assert len(g[sigma]['inputs']['sigmas'].split(','))==steps+1
    assert g['307']['inputs']['frame_rate']==g['371']['inputs']['fps']==24
    assert g['331']['inputs']['samples']==['395',2]


def board(root,video,name):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    cap=cv2.VideoCapture(str(video))
    sheet=Image.new('RGB',(1440,888),(25,25,25))
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/google-noto/NotoSans-Regular.ttf',16)
    folder=root/(name+'-samples');folder.mkdir(exist_ok=True)
    for n,i in enumerate(range(0,49,6)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read()
        if not ok:raise ValueError('Missing video frame')
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.save(folder/f'{i:04d}.png')
        x=n%3*480;y=n//3*296
        sheet.paste(im.resize((480,272)),(x,y+24));draw.text((x+8,y+2),f'{name}: frame {i}',font=font,fill='white')
    cap.release();sheet.save(root/(name+'-sheet.jpg'),quality=94)


def run(root):
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    from touchdown import studio_build as studio
    from touchdown.layout_recipe_trial import setup_runtime
    from touchdown.nested_local import video_meta
    import fakeout_earth_zoom as f
    from geoblock_core.workflow import LTX_VIDEO,STAGE1_VIDEO
    if (root/'submitted-graph.json').exists():raise ValueError('Attempt frozen; no automatic retry')
    verify_snapshot(read(root/'experiment.json')['source_hashes'])
    verify_snapshot(read(root/'guide-review.json')['hashes'])
    if read(root/'guide-review.json')['status']!='pass':raise ValueError('Guide rejected')
    runtime=root/'runtime';runtime.mkdir(exist_ok=False)
    setup_runtime(SOURCE,runtime)
    custom=Path('outputs/review/neutral-ladder-20261027/runtime_nodes').resolve()
    paths=read(runtime/'pinned-model-paths.json')
    paths['mask_attention_compatibility']={'custom_nodes':str(custom)}
    write(runtime/'pinned-model-paths.json',paths)
    shutil.copy2(custom/'TouchdownMaskAttention/__init__.py',runtime/'mask-attention-shim.py')
    with studio.runtime('comfy',runtime),f._client(studio.VIDEO_URL) as client:
        a=client.assets.from_file(str(root/'guide.mp4'));b=client.assets.from_file(str(root/'appearance.png'))
        g=graph(a,b)
        write(root/'submitted-graph.json',json.loads(json.dumps(g,default=str)))
        write(root/'status.json',dict(status='running',app_promoted=False))
        saved=f._submit(client,g,(LTX_VIDEO,STAGE1_VIDEO),root/'ltx')
        for name,node in [('result',LTX_VIDEO),('stage1-result',STAGE1_VIDEO)]:
            p=Path(saved[node]);write(root/(name+'.json'),dict(video=str(p),sha256=digest(p),**video_meta(p)))
            board(root,p,name)
    shutil.copy2(read(root/'result.json')['video'],root/'result.mp4')
    write(root/'status.json',dict(status='held_awaiting_visual_review',app_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render','preview','run']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare(root)
    elif a.stage in ['render','preview']:
        args=['blender','-b','-t','8','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('local_descent_blender.py')),'--',str(root)]
        if a.stage=='preview':args.append('--preview')
        subprocess.run(args,check=True)
        if a.stage=='render':
            subprocess.run(['ffmpeg','-v','error','-y','-framerate','24','-i',str(root/'frames/%04d.png'),'-frames:v','49','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'guide.mp4')],check=True)
            board(root,root/'guide.mp4','guide')
    else:run(root)
