"""Neutral-pose photographic ladder and gated plain-LTX unit transitions.

Each 49-frame clip is a standalone diagnostic, never an edited replacement for
the fixed eight-second product clock. No viewport or Layout-To-Render input.
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
from touchdown.flight_regression import read, write, verify_snapshot
from touchdown.layout_recipe_trial import setup_runtime
from touchdown.recipe_snapshot import capture

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026'
VERSION='neutral-photographic-ladder-v1'
SEED=1399629552
ROLES={1:'neutral close',2:'wide ground',3:'elevated approach',4:'local oblique aerial',
       5:'regional 30,000 ft',6:'cloud-covered high aerial',7:'curved Earth horizon'}


def plan(root, cfg):
    common=('One continuous real photographic scene in soft overcast autumn daylight. '
        'The same Chillon Castle has irregular grey stone walls and pitched roofs, behind-left across the '
        'water from the destination. Lake on the LEFT, connected mainland grass, gravel path and natural '
        'mature trees on the RIGHT. Keep the same approach heading and shoreline, realistic atmospheric '
        'depth and adult proportions. No duplicated landmarks, no collage, inset photograph, pedestal, '
        'toy geometry, labels or text. ')
    person=('Exactly one adult woman with the same brown hair, face and beige knee-length trench coat. '
        'She is planted on the grass beside the path, facing the arriving camera. NEUTRAL POSE: '
        'both upper arms hang vertically beside her torso, elbows relaxed, hands down by her thighs, '
        'shoulders relaxed, calm neutral expression, both feet planted. ')
    defs={
        1:([SOURCE/'photo-096.png'], common+person+
            'Edit the supplied photograph: change ONLY the woman’s pose to the neutral standing pose. '
            'Preserve the camera, knees-up crop, subject position/scale, castle, tree trunks, shoreline '
            'and overcast light. Her hands hang beside the coat, not outward; no greeting gesture.'),
        3:([root/'view-2.png'],common+person+
            'Generate a NEW HIGHER CAMERA VIEW, roughly 12 metres above the ground and 25 metres back '
            'along the same approach, pitched downward about 25 degrees. We see the tops of shoulders '
            'and ground behind her feet. The full-body woman is substantially smaller than in the '
            'reference, around one fifth of image height, near lower centre-right. Show more grass '
            'and the path in the foreground. The castle remains behind-left across the water. '
            'Do not retain the reference’s eye-level portrait camera or just shrink its photograph.'),
        4:([root/'view-3.png',root/'view-2.png'],common+
            'A NEW low-altitude oblique aerial photograph from about 60 metres high and 120 metres '
            'back, pitched down about 40 degrees along the SAME heading. Image 1 is the preceding '
            'approach; image 2 supplies the destination appearance only. The woman, still neutral '
            'and planted, is at most 2 percent of image height and is not a foreground portrait. '
            'See treetops, castle roof planes, a broader stretch of the same lakeshore and park. '
            'Lake left, shore/path right. No enlarged figure, island substitution or mirrored geography.'),
        5:([root/'view-4.png',SOURCE/'photo-000.png'],
            'One realistic REGIONAL aerial photograph from 30,000 feet / 9,144 metres, covering tens '
            'of kilometres of the same lake region in soft overcast autumn daylight. Image 1 anchors '
            'the destination and approach direction; image 2 supplies only the very broad camera '
            'scale, not extra islands or buildings. Lake Geneva water fills the LEFT, continuous '
            'shoreline and foothill terrain occupy the RIGHT, lake stretching into distant haze. '
            'Settlements are tiny patterns, not readable buildings. The woman, individual trees, '
            'gravel path and castle architectural details are UNRESOLVABLE. Same destination on '
            'the right-hand shore near the lower-middle. A new regional viewpoint, not a scaled '
            'local park photo, no giant castle, no inset, no text.'),
        6:([root/'view-5.png'],
            'Create one higher oblique aerial photograph above the SAME lake region in image 1. '
            'Preserve the broad shoreline direction, lake LEFT and land RIGHT, same destination '
            'along the right shore. Add a high cloud deck between the camera and terrain: large '
            'three-dimensional white/grey cloud masses at several distances, with breaks revealing '
            'the same broad land/water pattern. Soft overcast daylight, realistic atmospheric haze. '
            'The clouds occupy most of the picture, with visible near cloud edges for a forward '
            'flight into them. No figure, no readable castle or trees, no tiny model park, no text.'),
        7:([root/'view-6.png'],
            'One realistic very high orbital oblique photograph looking towards the SAME cloud-covered '
            'European region. Change camera scale radically upward from image 1: a close curved '
            'EARTH HORIZON near the upper quarter, thin blue atmospheric rim and narrow black space '
            'above, broad clouds and land below. Planet fills the frame; never show an isolated '
            'whole globe. Carry the broad cloud arrangement and daylight direction forward, aim '
            'towards the same destination. Geographic detail becomes very small; the lake is only '
            'a subtle possible blue mark, never an enormous lake taking up half the planet. '
            'No foreground trees, people, castle, miniature landscape, inset or text.')}
    return {str(k):dict(name=f'view-{k}',role=ROLES[k],references=[str(p) for p in refs],
        width=1920,height=1088,models=cfg['models'],seed=SEED+600+k,prompt=prompt)
        for k,(refs,prompt) in defs.items()}


def prepare(root):
    from touchdown.flight_runtime import recipe_hashes
    cfg=read(Path(read(SOURCE/'experiment.json')['identity']['scene_source'])/'scene-config.json')
    identity=dict(version=VERSION,source=str(SOURCE),code_hashes=recipe_hashes(),seed=SEED,
        source_hashes={str(SOURCE/n):digest(SOURCE/n) for n in ['photo-096.png','photo-000.png',
            'experiment.json','model-fingerprints.json','pinned-model-paths.json']},
        plan=plan(root,cfg),roles=ROLES,product_clock=dict(fps=FPS,bounds=list(BOUNDS),unchanged=True),
        still_policy='Neutral subject, raw native Fill expansion; no pixel-exact overlap claim',
        video_policy='49-frame/24fps adjacent diagnostics, 8+3-step LTX; not an 8-second product',
        pose_policy='All still references neutral; close LTX action test leaves final pose unpinned',
        candidate_policy='One initial sample per declared stage, no automatic rerolls; failures held')
    path=root/'experiment.json'
    if path.exists():
        if read(path)['identity']!=json.loads(json.dumps(identity)):
            raise ValueError('Recipe changed; use a new root')
        verify_snapshot(identity['source_hashes']);capture(root)
    else:
        if root.exists() and any(root.iterdir()):raise ValueError('Fresh root required')
        write(path,dict(identity=identity));capture(root)
        write(root/'status.json',dict(status='prepared',video_submissions=0,app_promoted=False))
    return identity


def require(root,key):
    r=read(root/f'view-{key}-review.json')
    if r['status']!='pass' or r['sha256']!=digest(root/f'view-{key}.png'):
        raise ValueError('Preceding frame failed or changed')


def photo(root,identity,key):
    if key>1:require(root,key-1)
    if (root/f'view-{key}.png').exists():raise ValueError('Existing candidate cannot be overwritten')
    folder=root/f'photo-stage-{key}';folder.mkdir()
    if key==2:
        # Fresh child process avoids inheriting another stage's model-path override.
        subprocess.run([str(REPO/'.venv/bin/python'),'-m','touchdown.fill_outpaint',
            '--source',str(root/'view-1.png'),'--root',str(folder/'fill'), '--run'],check=True)
        shutil.copy2(folder/'fill/raw-outpaint.png',root/'view-2.png')
        write(root/'view-2.json',dict(engine='native FLUX.1 Fill, official-example recipe',
            input_sha256=digest(root/'view-1.png'),output_sha256=digest(root/'view-2.png'),
            original_pixels_exact=False,raw_composite_restoration=False,
            working_source_size=[960,544],working_rectangle=[480,272,960,544],
            scope='Coherent raw outpaint; exact source-pixel preservation is not claimed'))
        return
    from touchdown import studio_build as studio
    setup_runtime(SOURCE,folder)
    with studio.runtime('comfy',folder):
        generated=generate_asset(studio.COMFY_URL,identity['plan'][str(key)],folder)
    shutil.copy2(generated,root/f'view-{key}.png')


def review(root,key,note,fail):
    if not note:raise ValueError('Actual image inspection note required')
    write(root/f'view-{key}-review.json',dict(status='fail' if fail else 'pass',
        sha256=digest(root/f'view-{key}.png'),note=note,reviewer='assistant_visual_inspection',
        scope='Still continuity and neutral pose, not video acceptance'))


def pair_prompt(first,last):
    moves={2:'Advance gently from the wide full-body view to the closer portrait.',
        3:'Descend and advance from the elevated view to the wide ground-level view.',
        4:'Descend from the local aerial to the low elevated approach along the shore.',
        5:'Fly downward from the vast regional view to the local oblique aerial.',
        6:'Fly physically through the near clouds and emerge into the regional 30,000-foot view.',
        7:'Fly forward and down from the close Earth horizon toward the three-dimensional cloud deck.'}
    action=('The planted woman starts with both arms relaxed at her sides. As the camera settles '
        'she smoothly raises both arms outward into a welcoming open pose, moving naturally at '
        'shoulders and elbows. She does not walk. The final pose is free to change.' if first==2 else
        'Where visible the same woman remains planted with relaxed arms down and natural adult proportions.')
    return ('One uninterrupted live-action camera flight. '+moves[first]+' '+action+
        ' Lake stays left, destination shore and path stay right, castle behind-left when resolved. '
        'Natural continuous perspective change and parallax: nearby grass, branches or cloud edges '
        'pass faster than distant landmarks. Maintain coherent daylight, shore shape and architecture. '
        'No cuts, crossfades, opacity replacement, flat photo zoom, miniature scenery or body morphing.')


def pair_graph(first,last,a,b):
    from geoblock_core.workflow import ShotRequest,_add_ltx25
    graph={'40':dict(class_type='LoadImage',inputs=dict(image=a)),
        '41':dict(class_type='LTXVPreprocess',inputs=dict(image=['40',0],img_compression=18)),
        '42':dict(class_type='LoadImage',inputs=dict(image=b)),
        '43':dict(class_type='LTXVPreprocess',inputs=dict(image=['42',0],img_compression=18))}
    action=first==2
    request=ShotRequest(name=VERSION,seed=SEED,output_prefix=f'touchdown/neutral_{first}_{last}',
        pin_last_keyframe=not action,ltx_upscale=True,save_stage1=True)
    mids=[(0,['41',0])]+([(24,['43',0])] if action else [])
    _add_ltx25(graph,request,49,None,['43',0],mids,pair_prompt(first,last),layout=False)
    graph['360']['inputs']['noise_seed']=SEED;graph['306']['inputs']['text']=''
    if action:
        for node in ['313','343']:graph[node]['inputs']['strength']=.5
    validate_pair(graph,first)
    return graph


def validate_pair(graph,first):
    forbidden={'LoadVideo','GetVideoComponents','LTXICLoRALoaderModelOnly',
               'LTXAddVideoICLoRAGuide','TouchdownPromptRelay'}
    if any(n['class_type'] in forbidden for n in graph.values()):raise ValueError('Unexpected viewport/relay conditioning in unit test')
    if graph['308']['inputs']!=dict(width=960,height=544,length=49,batch_size=1):raise ValueError('Changed diagnostic size/clock')
    if graph['307']['inputs']['frame_rate']!=24:raise ValueError('Changed fps')
    expected=[(0,['41',0],1),(24,['43',0],.5)] if first==2 else [(0,['41',0],1),(-1,['43',0],1)]
    for base in [312,342]:
        for k,(frame,image,strength) in enumerate(expected):
            p=graph[str(base+k)]['inputs']
            if (p['frame_idx'],p['image'],p['strength'])!=(frame,image,strength):raise ValueError('Reference/pose timing changed')
    if first==2 and any(n['inputs']['frame_idx']==-1 for n in graph.values() if n['class_type']=='LTXVAddGuide'):
        raise ValueError('Neutral final pose must not be pinned in the action test')
    for base,steps in [(390,8),(360,3)]:
        if graph[str(base)]['inputs']['noise_seed']!=SEED or graph[str(base+1)]['inputs']['cfg']!=1:
            raise ValueError('Frozen seed/CFG changed')
        if graph[str(base+2)]['inputs']['sampler_name']!='euler_ancestral' or len(graph[str(base+3)]['inputs']['sigmas'].split(','))-1!=steps:
            raise ValueError('Two-stage recipe changed')


def pair(root,first):
    from touchdown import studio_build as studio
    from touchdown.nested_local import video_meta
    import fakeout_earth_zoom as f
    from geoblock_core.workflow import LTX_VIDEO,STAGE1_VIDEO
    last=first-1
    require(root,first);require(root,last)
    folder=root/f'pair-{first}-{last}';folder.mkdir()
    setup_runtime(SOURCE,folder)
    write(folder/'attempt.json',dict(seed=SEED,frames=49,fps=24,prompt=pair_prompt(first,last),
        inputs={str(k):digest(root/f'view-{k}.png') for k in [first,last]},
        final_pose_unpinned=first==2,scope='Standalone unit clip, not product delivery timing'))
    with studio.runtime('comfy',folder),f._client(studio.VIDEO_URL) as client:
        a,b=[client.assets.from_file(str(root/f'view-{k}.png')) for k in [first,last]]
        graph=pair_graph(first,last,a,b)
        write(folder/'submitted-graph.json',json.loads(json.dumps(graph,default=str)))
        saved=f._submit(client,graph,(LTX_VIDEO,STAGE1_VIDEO),folder/'ltx')
        for name,node in [('result',LTX_VIDEO),('stage1-result',STAGE1_VIDEO)]:
            path=Path(saved[node]);write(folder/(name+'.json'),dict(video=str(path),sha256=digest(path),**video_meta(path)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','photo','review','pair'],required=True)
    p.add_argument('--key',type=int,choices=range(1,8));p.add_argument('--note',default='');p.add_argument('--fail',action='store_true')
    a=p.parse_args()
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    root=a.root.resolve();identity=prepare(root)
    if a.stage=='photo':photo(root,identity,a.key)
    elif a.stage=='review':review(root,a.key,a.note,a.fail)
    elif a.stage=='pair':pair(root,a.key)


if __name__=='__main__':main()
