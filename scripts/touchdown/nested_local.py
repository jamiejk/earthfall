"""Views 1–4 of the user's backwards-built photographic expansion experiment.

No viewport, video guide or Layout-To-Render adapter. The 49-frame pair clips
are unit experiments at 24 fps, not a re-timed replacement for the 8 s product.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from touchdown.authored_scene import digest, generate_asset
from touchdown.flight_contract import BOUNDS, FPS
from touchdown.flight_regression import read, write, verify_snapshot, check_root
from touchdown.recipe_snapshot import capture
from touchdown.layout_recipe_trial import setup_runtime

VERSION='nested-local-v1'
SIZE=(1920,1088)
PAIR_FRAMES=49
PAIRS=((2,1),(3,2),(4,3))


def expansion_geometry(size, factor=2):
    if factor!=2 or size[0]%4 or size[1]%4:
        raise ValueError('This controlled experiment uses an exact 2x expansion and divisible-by-four working size')
    w,h=size
    return dict(factor=2,working_canvas=[w,h],working_centre=[w//4,h//4,w//2,h//2],
                full_canvas=[w*2,h*2],full_centre=[w//2,h//2,w,h],
                parent_to_working=dict(scale=.5,offset=[.25,.25]))


def outpaint_canvas(original):
    """RGBA canvas: alpha 1 protects the original's reduced centre; 0 paints."""
    g=expansion_geometry(original.size)
    w,h=g['working_canvas'];x,y,cw,ch=g['working_centre']
    rgb=np.asarray(original.convert('RGB'))
    rng=np.random.default_rng(20261027)
    bg=np.clip(np.median(rgb.reshape(-1,3),axis=0)+rng.normal(0,3,(h,w,3)),0,255).astype('uint8')
    canvas=Image.fromarray(np.dstack([bg,np.zeros((h,w),dtype='uint8')]),'RGBA')
    inner=original.convert('RGBA').resize((cw,ch),Image.Resampling.LANCZOS)
    canvas.paste(inner,(x,y))
    return canvas


def restore_native_centre(original, generated):
    """Preserve the *entire* native-resolution source, not just a feathered core.

    Surroundings are generated at working size and enlarged. Do not describe
    those borders as native-resolution 4K generation.
    """
    g=expansion_geometry(original.size)
    full=generated.convert('RGB').resize(tuple(g['full_canvas']),Image.Resampling.LANCZOS)
    x,y,w,h=g['full_centre'];full.paste(original.convert('RGB'),(x,y))
    if not np.array_equal(np.asarray(full.crop((x,y,x+w,y+h))),np.asarray(original.convert('RGB'))):
        raise ValueError('Native source pixels were not protected')
    return full


def image_plan(root,seed,models):
    common=('One natural photographic scene in soft overcast autumn daylight. The SAME woman in a beige '
            'trench coat stands still on the SAME connected grass beside the gravel path, arms open; '
            'the SAME Chillon Castle is behind her on the LEFT across the water, and irregular mature '
            'deciduous trees line the RIGHT. Preserve this approach heading and landmark arrangement. '
            'No duplicated people or castles, no new buildings, no pedestal, no inset or collage, no text. ')
    return {
        '2':dict(name='view-2-generated',references=[],inpaint=str(root/'outpaint-canvas.png'),
            width=SIZE[0],height=SIZE[1],models=models,seed=seed+301,
            prompt=common+'Outpaint the masked surroundings of the existing central photograph into one seamless '
            'wider ground-level environmental view. Continue the water and distant landscape on the left, '
            'the grass and gravel in the foreground, and natural trees at right. Complete only the lower legs '
            'and shoes that continue the existing person; do not add another person. The existing centre is '
            'already the photograph, not a framed picture to depict. Keep its light, horizon and perspective. '
            'The lens sees more of the same scene in every direction; do not climb or change the viewpoint.'),
        '3':dict(name='view-3',references=[str(root/'view-2.png')],width=SIZE[0],height=SIZE[1],models=models,seed=seed+302,
            prompt=common+'CHANGE THE CAMERA VIEWPOINT: move backward along the approach path and rise to about '
            '12 metres above ground, roughly 25 metres from the woman, looking down at about 25 degrees. '
            'Show the meadow/path in front of her and the tops of her shoulders and nearby trees. She becomes '
            'a small full-body adult near the lower middle, standing at the SAME place and facing the arriving '
            'camera. The castle remains behind-left, water left, path/trees right. Preserve the same roof groups, '
            'shoreline and season. This is a new elevated photograph, not an outpaint, crop or miniature version '
            'of the original eye-level photograph. Do not retain the original eye-level viewpoint.'),
        '4':dict(name='view-4',references=[str(root/'view-3.png'),str(root/'view-2.png')],width=SIZE[0],height=SIZE[1],models=models,seed=seed+303,
            prompt=common+'CHANGE THE CAMERA VIEWPOINT AGAIN: a modest oblique local aerial, about 40 metres '
            'above ground and 70 metres back along the SAME path/heading, looking downward about 35 degrees. '
            'Image 1 supplies the preceding elevated view; image 2 supplies nearby landmark/subject appearance '
            'only. Show a broader stretch of the SAME lakeshore and meadow. The woman is a tiny full-body '
            'figure near the lower middle of the shore path, not a large foreground portrait. The castle and '
            'its actual roofs stay behind-left across the water; grass/path/trees stay right. No new island, '
            'town, palace, coastline reversal, horizon rotation or duplicated landmark. Real aerial photography.')}


def prepare(source,root,proposal):
    from touchdown.flight_runtime import recipe_hashes
    check_root(source,root)
    previous=read(source/'experiment.json')['identity']
    scene=Path(previous['scene_source']);cfg=read(scene/'scene-config.json')
    endpoint=source/'photo-096.png'
    with Image.open(endpoint) as im:
        if im.size!=SIZE:raise ValueError('Native final reference size changed')
    files=[endpoint,source/'photo-096.json',source/'experiment.json',source/'model-fingerprints.json',
           source/'pinned-model-paths.json',scene/'scene-config.json',proposal]
    files += [Path(r['local_path']) for r in cfg['references']]
    seed=previous['seed']
    plan=image_plan(root,seed,cfg['models'])
    identity=dict(version=VERSION,source=str(source),proposal=str(proposal),source_hashes={str(p):digest(p) for p in files},
        code_hashes=recipe_hashes(),seed=seed,working_size=list(SIZE),expansion=expansion_geometry(SIZE),
        image_plan=plan,product_clock=dict(fps=FPS,bounds=list(BOUNDS),unchanged=True),
        pairs=[dict(first=a,last=b,frames=PAIR_FRAMES,fps=24,seed=seed,candidate=1) for a,b in PAIRS],
        architecture='Photographic expansion/viewpoint chain -> adjacent plain LTX 2.5 image-to-video pairs; no viewport/IC-LoRA',
        scope='Standalone local diagnostic clips; NOT a complete film or a replacement delivery timeline',
        candidate_policy='One seed per initial image/pair; changed recipes and failed candidates retained. No automatic rerolls.')
    path=root/'experiment.json'
    if path.exists():
        if read(path)['identity']!=identity:raise ValueError('Changed experiment; use a fresh root')
        verify_snapshot(identity['source_hashes']);capture(root)
        if digest(root/'view-1.png')!=digest(endpoint):raise ValueError('Native endpoint changed')
        return identity
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh root required')
    write(path,dict(identity=identity));capture(root)
    shutil.copy2(endpoint,root/'view-1.png');shutil.copy2(proposal,root/'proposal.txt')
    write(root/'attribution.json',cfg['references'])
    with Image.open(endpoint) as im:outpaint_canvas(im).save(root/'outpaint-canvas.png')
    write(root/'image-plan.json',plan)
    write(root/'status.json',dict(status='prepared',stage='outpaint_mask_preflight',full_flight=False,app_promoted=False))
    return identity


def verify_canvas(root):
    with Image.open(root/'outpaint-canvas.png') as im:rgba=np.asarray(im)
    x,y,w,h=expansion_geometry(SIZE)['working_centre']
    expected=np.zeros((SIZE[1],SIZE[0]),dtype='uint8');expected[y:y+h,x:x+w]=255
    if not np.array_equal(rgba[:,:,3],expected):raise ValueError('Protected-centre alpha mask changed')
    graph=__import__('touchdown.flux3',fromlist=['graph']).graph('mask probe',1,[],'probe',*SIZE,inpaint='canvas.png')
    if graph['17']['class_type']!='SetLatentNoiseMask' or graph['17']['inputs']['mask']!=['m0',1] or graph['14']['inputs']['latent_image']!=['17',0]:
        raise ValueError('FLUX graph bypasses the canvas mask')
    return dict(canvas_sha256=digest(root/'outpaint-canvas.png'),protected_pixels=int((expected==255).sum()),
                editable_pixels=int((expected==0).sum()),mask='LoadImage 1-alpha -> SetLatentNoiseMask -> KSampler')


def require_view(root,key):
    if key==1:return
    review=read(root/f'view-{key}-review.json')
    if review['status']!='pass' or review['sha256']!=digest(root/f'view-{key}.png'):
        raise ValueError('Previous view changed or failed inspection')


def make_view(source,root,identity,key):
    from touchdown import studio_build as studio
    require_view(root,key-1)
    if (root/f'view-{key}.png').exists():raise ValueError('Existing image attempt is frozen; do not overwrite')
    if key==2:verify_canvas(root)
    setup_runtime(source,root)
    with studio.runtime('comfy',root):
        if key==2:
            from touchdown import flux3
            info=flux3._call(studio.COMFY_URL,'/object_info')
            needed=['LoadImage','VAEEncode','SetLatentNoiseMask','KSampler']
            if any(n not in info for n in needed):raise ValueError('Installed ComfyUI lacks the masked outpaint path')
            write(root/'mask-preflight.json',dict(status='pass',**verify_canvas(root),available_nodes=needed))
        generated=generate_asset(studio.COMFY_URL,identity['image_plan'][str(key)],root)
    if key==2:
        with Image.open(root/'view-1.png') as original,Image.open(generated) as painted:
            full=restore_native_centre(original,painted);full.save(root/'view-2-full.png')
            full.resize(SIZE,Image.Resampling.LANCZOS).save(root/'view-2.png')
            x,y,w,h=expansion_geometry(SIZE)['working_centre']
            target=np.asarray(original.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)).astype('int16')
            raw=np.asarray(painted.convert('RGB'))[y:y+h,x:x+w].astype('int16')
            write(root/'expansion-audit.json',dict(status='pass',geometry=expansion_geometry(SIZE),
                native_centre_exact=True,source_sha256=digest(root/'view-1.png'),full_sha256=digest(root/'view-2-full.png'),
                working_sha256=digest(root/'view-2.png'),raw_vae_centre_mean_abs_error=float(np.abs(raw-target).mean()),
                border_resolution='Generated at 1920x1088 working canvas then enlarged; native centre restored exactly',
                relation='Literal nested overlap, not a calibrated dolly or recovered 3D geometry'))


def review_view(root,key,note):
    if not note:raise ValueError('Inspect actual scene continuity and perspective before passing a view')
    require_view(root,key-1)
    if key==2:
        a=read(root/'expansion-audit.json')
        if not a['native_centre_exact']:raise ValueError('Original centre not preserved')
    write(root/f'view-{key}-review.json',dict(status='pass',sha256=digest(root/f'view-{key}.png'),
        note=note,reviewer='Agent image inspection',scope='Photographic scene/perspective candidate; not video acceptance'))


def pair_prompt(first,last):
    move={2:'The camera advances gently at eye level from the wider environmental view to the closer knees-up portrait.',
          3:'The camera descends smoothly from the modest elevated view and moves forward along the path to the wide ground-level view.',
          4:'The camera descends and advances from the local oblique aerial to the lower elevated approach, keeping the same heading.'}[first]
    return (f'One uninterrupted live-action shot. {move} '
        'The same woman in a beige trench coat remains planted on the same grass with her arms open; '
        'the camera travels, she does not walk. The same castle stays behind-left, lake left and path/trees right. '
        'Foreground grass and branches pass faster than distant castle masonry, with continuous perspective and depth. '
        'Preserve her face, proportions, clothing, the roof groups, shoreline and soft overcast autumn lighting. '
        'No cuts, dissolves, opacity changes, picture-in-picture, new landscape, body morphing or stationary photo hold.')


def pair_graph(first,last,seed,a,b):
    from geoblock_core.workflow import _add_ltx25,ShotRequest
    graph={'40':dict(class_type='LoadImage',inputs=dict(image=a)),
           '41':dict(class_type='LTXVPreprocess',inputs=dict(image=['40',0],img_compression=18)),
           '42':dict(class_type='LoadImage',inputs=dict(image=b)),
           '43':dict(class_type='LTXVPreprocess',inputs=dict(image=['42',0],img_compression=18))}
    request=ShotRequest(name=VERSION,seed=seed,output_prefix=f'touchdown/nested_local_{first}_to_{last}',
        pin_last_keyframe=True,ltx_upscale=True,save_stage1=True)
    _add_ltx25(graph,request,PAIR_FRAMES,None,['43',0],[(0,['41',0])],pair_prompt(first,last),layout=False)
    graph['360']['inputs']['noise_seed']=seed;graph['306']['inputs']['text']=''
    validate_pair_graph(graph,seed)
    return graph


def validate_pair_graph(graph,seed):
    forbidden={'LoadVideo','GetVideoComponents','LTXICLoRALoaderModelOnly','LTXAddVideoICLoRAGuide','TouchdownPromptRelay'}
    if any(v['class_type'] in forbidden for v in graph.values()):raise ValueError('Standalone test must not use viewport/video/layout/phase conditioning')
    if graph['308']['inputs']!=dict(width=960,height=544,length=49,batch_size=1) or graph['307']['inputs']['frame_rate']!=24:
        raise ValueError('Diagnostic pair clock/size changed')
    for base in [310,340]:
        for k,(index,image) in enumerate([(0,['41',0]),(-1,['43',0])]):
            v=graph[str(base+2+k)]['inputs']
            if (v['frame_idx'],v['image'],v['strength'])!=(index,image,1):raise ValueError('Adjacent first/last references are not connected on both stages')
    for base,steps in [(390,8),(360,3)]:
        if graph[str(base)]['inputs']['noise_seed']!=seed or graph[str(base+1)]['inputs']['model']!=['300',0] or graph[str(base+1)]['inputs']['cfg']!=1:
            raise ValueError('Plain LTX distilled sampling changed')
        if len(graph[str(base+3)]['inputs']['sigmas'].split(','))-1!=steps:raise ValueError('Two-stage step schedule changed')
    if graph['370']['inputs']['samples']!=['365',2]:raise ValueError('Stage 2 bypassed')


def video_meta(path):
    import cv2
    c=cv2.VideoCapture(str(path));d=dict(frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),fps=c.get(cv2.CAP_PROP_FPS),
        width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)));c.release()
    if (d['frames'],d['fps'])!=(PAIR_FRAMES,24):raise ValueError('Pair video clock changed')
    return d


def render_pairs(source,root,identity):
    import fakeout_earth_zoom as f
    from touchdown import studio_build as studio
    from geoblock_core.workflow import LTX_VIDEO,STAGE1_VIDEO
    for k in [2,3,4]:require_view(root,k)
    if (root/'pairs-submitted.json').exists():raise ValueError('Pair attempt already started; preserve the attempt, do not resubmit')
    setup_runtime(source,root)
    write(root/'pairs-submitted.json',dict(pairs=identity['pairs'],input_hashes={str(root/f'view-{k}.png'):digest(root/f'view-{k}.png') for k in range(1,5)}))
    with studio.runtime('comfy',root),f._client(studio.VIDEO_URL) as client:
        assets={k:client.assets.from_file(str(root/f'view-{k}.png')) for k in range(1,5)}
        for first,last in PAIRS:
            folder=root/f'pair-{first}-{last}';folder.mkdir()
            graph=pair_graph(first,last,identity['seed'],assets[first],assets[last])
            write(folder/'submitted-graph.json',json.loads(json.dumps(graph,default=str)))
            write(folder/'attempt.json',dict(candidate=1,seed=identity['seed'],first=first,last=last,
                frames=49,fps=24,prompt=pair_prompt(first,last),scope='Diagnostic pair, no delivery-time assignment',
                first_sha256=digest(root/f'view-{first}.png'),last_sha256=digest(root/f'view-{last}.png')))
            saved=f._submit(client,graph,(LTX_VIDEO,STAGE1_VIDEO),folder/'ltx')
            for name,node in [('result',LTX_VIDEO),('stage1-result',STAGE1_VIDEO)]:
                path=Path(saved[node]);meta=video_meta(path)
                write(folder/(name+'.json'),dict(video=str(path),sha256=digest(path),**meta))
    write(root/'status.json',dict(status='held',stage='pair_motion_review',pairs=3,video_candidates_per_pair=1,
        full_flight=False,app_promoted=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','outpaint','elevated','local','review','pairs'],required=True)
    p.add_argument('--key',type=int,choices=[2,3,4]);p.add_argument('--note',default='');a=p.parse_args()
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    os.environ['PYTHONPATH']=os.pathsep.join(str(Path(__file__).resolve().parents[2]/q) for q in ['scripts','src'])
    source,root,proposal=a.source.resolve(),a.root.resolve(),a.proposal.resolve();identity=prepare(source,root,proposal)
    if a.stage=='outpaint':make_view(source,root,identity,2)
    elif a.stage=='elevated':make_view(source,root,identity,3)
    elif a.stage=='local':make_view(source,root,identity,4)
    elif a.stage=='review':
        if a.key is None:raise ValueError('Review requires a keyframe number')
        review_view(root,a.key,a.note)
    elif a.stage=='pairs':render_pairs(source,root,identity)
    verify_snapshot(identity['source_hashes'])


if __name__=='__main__':main()
