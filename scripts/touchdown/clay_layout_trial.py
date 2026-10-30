"""Neutral-clay + five photographic stage keyframes on the frozen camera clock.

Separate experiment; changes the guide's contents and appearance anchors together.
The corrected LTX two-stage recipe, seeds, strengths, camera poses and clock stay fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from touchdown.authored_scene import digest, generate_asset
from touchdown.clay_layout import START, END, FRAMES, FPS, KEYFRAMES, camera_pose, HEIGHT, HEAD_HEIGHT
from touchdown.flight_contract import BOUNDS
from touchdown.flight_regression import read, write, verify_snapshot, check_root
from touchdown.recipe_snapshot import capture
from touchdown.layout_recipe_trial import SIZE, PROMPT, metadata, setup_runtime

VERSION='minimal-clay-photographic-stages-v1'


def photograph_plan(cfg, root):
    common=(
        'Create ONE convincing real photograph, not an illustration or a textured render. '
        'Image 1 is only an approximate camera-and-occupancy sketch: use its viewpoint, horizon or downward '
        'pitch, shoreline and path position, and the subject location and scale. Its simple shapes are NOT '
        'the desired architecture, anatomy, materials or vegetation. Replace them with real-world detail. '
        'Soft natural daylight, a blue-green Swiss lake, natural meadow grass and gravel, atmospheric depth. '
        'Real irregular mature trees and weathered masonry, no clipped spherical topiary, no toy castle, '
        'no smooth plastic surfaces, no tilt-shift or miniature aesthetic, no text, collage or inset. ')
    specs=[]
    for i, role in KEYFRAMES:
        refs=[root/'frames'/f'{i:04d}.png']
        if i==0:
            # Retain the existing broad regional reference rather than accepting
            # a freshly painted close aerial as a 30,000-foot view.
            prompt='Reuse retained photographic regional starting view; whole-frame spatial resize only.'
        elif i==12:
            refs.append(root/'photo-000.png')
            prompt=('Create one real very high-altitude aerial photograph. Image 2 is the correct REGIONAL '
                'camera scale: retain that vast extent, tiny unresolved settlements and distant haze. '
                'Image 1 supplies only the broad shoreline and updated camera orientation. The aircraft is '
                '30,000 feet above ground. The frame covers tens of kilometres, not a local park. The lake '
                'fills the left, regional land the right. Individual trees, shoreline paths, people and '
                'castle details must NOT be readable. Natural satellite-like landscape detail, no new islands, '
                'no sky/horizon, no eye-level view, no model or miniature appearance.')
        elif i==36:
            prompt=('One real low-altitude aerial PHOTOGRAPH, looking steeply downward from approximately '
                '120 metres, not an oblique view toward the horizon. Image 1 is a neutral camera/occupancy sketch '
                'only, not the desired grey materials. The lake occupies the LEFT HALF and natural Swiss '
                'shoreline meadow the RIGHT HALF. Follow the single shoreline and narrow gravel route in '
                'the sketch. Fine irregular natural grass, wild plants, small scattered shrubs and a few '
                'mature irregular trees; photographic ground textures and natural soft daylight, blue-green '
                'lake water. The bottom half shows close shore terrain, not kilometres of regional fields. '
                'No sky/horizon, no islands, no town close-up, no toy objects, no repeated tree spheres, '
                'no new buildings or large castle. One tiny adult in beige at the sketch person position '
                'only if resolvable; never a large foreground portrait. No illustration or render appearance.')
        else:
            refs.append(root/'assets/subject.png')
            if i==60:
                refs.append(root/'assets/location-1.jpg')
                prompt=(common+'Image 2 supplies ONLY the adult woman identity, brown hair and beige trench coat. '
                    'Image 3 supplies ONLY real Chillon Castle architectural detail and natural landscape appearance, '
                    'not its close-up viewpoint. The large plain mass at the LEFT BACKGROUND of image 1 is only '
                    'castle occupancy: replace it with a real distant Chillon Castle, irregular historic stone '
                    'facades, varied pitched roofs, a square keep and turrets. Keep its broad background position '
                    'and scale, not the cuboid silhouette. One full-sized adult woman stands on connected grass '
                    'beside the path at the exact position/scale of the mannequin: head to shoes visible, '
                    'normal adult height and anatomy, about 7.5 head lengths tall, long natural legs, '
                    'a knee-length trench coat and dark trousers, both shoes planted on the grass. '
                    'Her arms are low, beginning to open, matching the sketch pose. Realistic portrait photograph '
                    'from about eight metres away, not a short or child-proportioned figure. '
                    'No tree spheres or rows of identical trees, no other people, no pedestal beneath her.')
            else:
                refs.append(root/'assets/location-1.jpg')
                prompt=(common+'Image 2 supplies ONLY the same adult woman identity. Image 3 supplies the SAME '
                    'real Chillon Castle architecture, NOT its close-up camera position. '
                    'Image 1 sets the new close camera position and crop: a normally proportioned adult woman '
                    'in a beige knee-length trench coat, BOTH ARMS OPEN ALMOST HORIZONTALLY at shoulder height, '
                    'hands extended far to both sides exactly as the mannequin pose, knees-up portrait '
                    'with the lower legs cropped by the bottom edge. Keep the photographed castle from image 3 '
                    'distant at the LEFT BACKGROUND, with the same irregular roofs and mature vegetation. '
                    'Natural human head/body ratio and slender adult limbs, not an oversized head or squat torso. '
                    'One woman on connected grass beside the same gravel shore path; no extra people or platforms.')
        spec=dict(name=f'photo-{i:03d}',role=role,local_frame=i,delivery_frame=START+i,
            references=[str(p) for p in refs],width=SIZE[0],height=SIZE[1],models=cfg['models'],
            seed=cfg['seed']+2049+i,prompt=prompt)
        if i==0:spec['reuse_path']=cfg['pilot_regional_reference']
        specs.append(spec)
    return specs


def prepare(source,root):
    from touchdown.flight_runtime import recipe_hashes
    check_root(source,root)
    previous=read(source/'experiment.json')['identity']
    camera_source=Path(previous['source'])
    scene_source=Path(read(camera_source/'experiment.json')['identity']['source'])
    cfg=read(scene_source/'scene-config.json')
    cfg['pilot_regional_reference']=str(camera_source/'photo-first.png')
    files=[source/n for n in ['experiment.json','submitted-graph.json','treatment-result.json','model-fingerprints.json','pinned-model-paths.json']]
    files += [Path(read(source/'treatment-result.json')['video']),camera_source/'render.json',camera_source/'photo-first.png',scene_source/'scene-config.json',Path(cfg['subject'])]
    files += [Path(r['local_path']) for r in cfg['references']]
    previous_poses=read(camera_source/'render.json')['poses']
    poses=[camera_pose(d) for d in range(START,END+1)]
    if poses!=previous_poses:raise ValueError('Original camera path changed')
    identity=dict(version=VERSION,source=str(source),scene_source=str(scene_source),
        source_hashes={str(p.resolve()):digest(p) for p in files},code_hashes=recipe_hashes(),
        seed=previous['seed'],fps=FPS,frames=FRAMES,delivery_mapping=list(range(START,END+1)),
        full_product_clock=dict(fps=FPS,bounds=list(BOUNDS)),keyframes=[dict(local_frame=i,stage=n) for i,n in KEYFRAMES],
        guide=dict(width=960,height=544,colour='one neutral grey material',textures=False,trees=False,
            anatomy=dict(height_m=HEIGHT,head_height_m=HEAD_HEIGHT,head_units=HEIGHT/HEAD_HEIGHT)),
        recipe='Corrected two-stage LTX 2.5 Layout-To-Render; 8+3 steps; full LoRA/guide strengths; same seed each stage',
        changed=['Minimal untextured grey scene, no tree/prop/architectural detail','Normally proportioned adult mannequin',
                 'Five stage-specific photographic keyframes (regional start retained), all applied in both stages'],
        retained=['Camera position/aim/lens per frame','97 frames at 24 fps, original 4–8 second mapping',
                  'Model files, strengths, seed, two-stage sigma schedules, output resolution'],
        scope='Camera segment only; no orbital/cloud wrapper or full-flight acceptance')
    manifest=root/'experiment.json'
    if manifest.exists():
        if read(manifest)['identity']!=identity:raise ValueError('Frozen recipe changed; use a new root')
        verify_snapshot(identity['source_hashes']);capture(root)
        verify_snapshot(read(root/'prepared-inputs.json')['hashes'])
        if read(root/'photograph-plan.json')['specs']!=photograph_plan(cfg,root):
            raise ValueError('Frozen stage-photograph plan changed')
        return cfg,identity
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh experiment root required')
    write(manifest,dict(identity=identity));capture(root)
    assets=root/'assets';assets.mkdir()
    shutil.copy2(cfg['subject'],assets/'subject.png')
    for i,ref in enumerate(cfg['references']):shutil.copy2(ref['local_path'],assets/f'location-{i}.jpg')
    write(root/'attribution.json',cfg['references'])
    write(root/'render.json',dict(width=960,height=544,frames=FRAMES,fps=FPS,poses=poses,seed=20261026,
        start_delivery_frame=START,phases=[dict(name='regional',start=96,end=108),dict(name='approach',start=108,end=156),dict(name='settle',start=156,end=193)]))
    write(root/'photograph-plan.json',dict(specs=photograph_plan(cfg,root)))
    paths=[root/n for n in ['render.json','photograph-plan.json','assets/subject.png','assets/location-0.jpg','assets/location-1.jpg']]
    write(root/'prepared-inputs.json',dict(hashes={str(p):digest(p) for p in paths}))
    return cfg,identity


def render(root,preview=False):
    import fcntl
    lockpath=Path(__file__).resolve().parents[2]/'outputs/.touchdown-photo-gpu.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        args=['blender','-b','-t','8','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('clay_layout_blender.py')),'--',str(root)]
        if preview:args.append('--preview')
        subprocess.run(args,check=True)
    if not preview:
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','24','-i',str(root/'frames/%04d.png'),
            '-frames:v',str(FRAMES),'-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'guide.mp4')],check=True)


def guide_review(root,note):
    import cv2
    import numpy as np
    audit=read(root/'geometry-audit.json')
    if not note:raise ValueError('Guide requires actual sampled-image inspection')
    if audit['camera_poses']!=[camera_pose(d) for d in range(START,END+1)]:raise ValueError('Camera changed')
    if audit['tree_meshes'] or audit['texture_nodes'] or audit['image_planes'] or audit['view_crossfades']:
        raise ValueError('Motion guide retains appearance/detail or view replacements')
    if audit['material_count']!=1 or any(abs(c[0]-c[1])+abs(c[1]-c[2])>1e-6 for c in audit['material_base_colours']):
        raise ValueError('Guide must use neutral clay')
    if audit['head_units']<7.3 or audit['human_height_m']!=HEIGHT:raise ValueError('Adult proportions changed')
    meta=metadata(root/'guide.mp4');cap=cv2.VideoCapture(str(root/'guide.mp4'));error=0
    for i in range(FRAMES):
        ok,bgr=cap.read()
        if not ok:raise ValueError('Short guide')
        error=max(error,int(np.abs(bgr[:,:,0].astype('int16')-bgr[:,:,2].astype('int16')).max()))
    cap.release()
    if error>2:raise ValueError('Rendered guide has chromatic colour')
    review=dict(status='pass',guide_sha256=digest(root/'guide.mp4'),geometry_sha256=digest(root/'geometry-audit.json'),
        frames=FRAMES,fps=FPS,max_pixel_chroma_error=error,visual_note=note,scope='Minimal motion guide only')
    review['stage_layout_hashes']={str(root/'frames'/f'{i:04d}.png'):digest(root/'frames'/f'{i:04d}.png') for i,_ in KEYFRAMES}
    write(root/'raw-guide-review.json',review)


def require_guide(root):
    r=read(root/'raw-guide-review.json')
    if r['status']!='pass' or r['guide_sha256']!=digest(root/'guide.mp4') or r['geometry_sha256']!=digest(root/'geometry-audit.json'):
        raise ValueError('Guide changed or failed inspection')
    verify_snapshot(r.get('stage_layout_hashes',{}))


def paint(source,root):
    from touchdown import studio_build as studio
    require_guide(root);setup_runtime(source,root)
    with studio.runtime('comfy',root):
        for spec in read(root/'photograph-plan.json')['specs']:
            if spec.get('reuse_path'):
                from PIL import Image
                path=root/(spec['name']+'.png');source_image=Path(spec['reuse_path'])
                with Image.open(source_image) as im:im.convert('RGB').resize(SIZE,Image.Resampling.LANCZOS).save(path)
                write(root/(spec['name']+'.json'),dict(spec=spec,source_sha256=digest(source_image),output_sha256=digest(path),operation='Retained regional appearance, whole-frame resize'))
            else:generate_asset(studio.COMFY_URL,spec,root)


def photo_review(root,note):
    from PIL import Image
    require_guide(root)
    if not note:raise ValueError('All five stage images require inspection for photography, scale and anatomy')
    hashes={}
    for i,_ in KEYFRAMES:
        path=root/f'photo-{i:03d}.png'
        with Image.open(path) as im:
            if im.size!=SIZE:raise ValueError('Photograph size changed')
        hashes[str(path)]=digest(path)
    write(root/'photo-anchor-review.json',dict(status='pass',guide_sha256=digest(root/'guide.mp4'),hashes=hashes,
        plan_sha256=digest(root/'photograph-plan.json'),visual_note=note,
        scope='Stage-keyframe appearance/layout review; not video acceptance'))


def require_photos(root):
    require_guide(root);r=read(root/'photo-anchor-review.json')
    if r['status']!='pass' or r['guide_sha256']!=digest(root/'guide.mp4') or r['plan_sha256']!=digest(root/'photograph-plan.json'):
        raise ValueError('Stage photographs rejected or their frozen plan changed')
    expected={str(root/f'photo-{i:03d}.png') for i,_ in KEYFRAMES}
    if set(r['hashes'])!=expected:raise ValueError('Missing stage photograph')
    verify_snapshot(r['hashes'])


def build_graph(seed,assets):
    from geoblock_core.workflow import ShotRequest,_add_ltx25
    graph={'50':dict(class_type='LoadVideo',inputs=dict(file=assets['guide'])),
           '60':dict(class_type='GetVideoComponents',inputs=dict(video=['50',0]))}
    for i,_ in KEYFRAMES:
        node=f'p{i:03d}'
        graph[node]=dict(class_type='LoadImage',inputs=dict(image=assets[i]))
        graph[node+'c']=dict(class_type='LTXVPreprocess',inputs=dict(image=[node,0],img_compression=18))
    mids=[(i,[f'p{i:03d}c',0]) for i,_ in KEYFRAMES if i!=96]
    request=ShotRequest(name=VERSION,output_prefix='touchdown/chillon_minimal_clay',seed=seed,
        ltx_upscale=True,save_stage1=True,pin_last_keyframe=True)
    _add_ltx25(graph,request,FRAMES,['p000c',0],['p096c',0],mids,PROMPT,layout=True)
    graph['360']['inputs']['noise_seed']=seed;graph['306']['inputs']['text']=''
    validate_graph(graph,seed)
    return graph


def validate_graph(graph,seed):
    def inp(n):return graph[n]['inputs']
    if inp('308')!=dict(width=960,height=544,length=97,batch_size=1) or inp('307')['frame_rate']!=24:
        raise ValueError('Original clock or two-stage dimensions changed')
    if inp('301')['strength_model']!=1 or inp('301')['lora_name']!='ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors':
        raise ValueError('Layout LoRA/strength changed')
    for layout,look,base in [('310','311',312),('340','341',342)]:
        if graph[layout]['class_type']!='LTXAddVideoICLoRAGuide' or (inp(layout)['image'],inp(layout)['frame_idx'],inp(layout)['strength'])!=(['60',0],0,1):
            raise ValueError('Both sampling stages require the whole motion guide')
        if (inp(look)['image'],inp(look)['frame_idx'],inp(look)['strength'])!=(['p000c',0],-1,1):
            raise ValueError('First-frame appearance reference lost')
        for k,(index,_) in enumerate(KEYFRAMES):
            node=str(base+k);v=inp(node);idx=-1 if index==96 else index
            if graph[node]['class_type']!='LTXVAddGuide' or (v['image'],v['frame_idx'],v['strength'])!=([f'p{index:03d}c',0],idx,1):
                raise ValueError('Every stage photograph must be pinned at its original time in both stages')
    if graph['331']['class_type']!='LTXVLatentUpsampler' or inp('370')['samples']!=['365',2]:
        raise ValueError('Output must come from stage 2')
    for base,steps in [(390,8),(360,3)]:
        if inp(str(base))['noise_seed']!=seed or inp(str(base+1))['cfg']!=1 or inp(str(base+1))['model']!=['301',0]:
            raise ValueError('Frozen distilled sampling changed')
        if inp(str(base+2))['sampler_name']!='euler_ancestral' or len(inp(str(base+3))['sigmas'].split(','))-1!=steps:
            raise ValueError('Published 8+3 sampling recipe changed')


def treat(source,root,identity):
    import fakeout_earth_zoom as f
    from touchdown import studio_build as studio
    from geoblock_core.workflow import LTX_VIDEO,STAGE1_VIDEO
    require_photos(root)
    if (root/'treatment-result.json').exists() or (root/'submitted-graph.json').exists():
        raise ValueError('An existing attempt is frozen; do not submit twice')
    setup_runtime(source,root)
    with studio.runtime('comfy',root),f._client(studio.VIDEO_URL) as client:
        assets={'guide':client.assets.from_file(str(root/'guide.mp4'))}
        assets.update({i:client.assets.from_file(str(root/f'photo-{i:03d}.png')) for i,_ in KEYFRAMES})
        graph=build_graph(identity['seed'],assets)
        write(root/'submitted-graph.json',json.loads(json.dumps(graph,default=str)))
        write(root/'status.json',dict(status='running',stage='two_stage_generation',app_promoted=False))
        saved=f._submit(client,graph,(LTX_VIDEO,STAGE1_VIDEO),root/'ltx')
    for name,node,size in [('treatment',LTX_VIDEO,SIZE),('stage1',STAGE1_VIDEO,(960,544))]:
        path=Path(saved[node]);meta=metadata(path)
        if (meta['width'],meta['height'])!=size:raise ValueError('Unexpected rendered stage size')
        write(root/f'{name}-result.json',dict(video=str(path),sha256=digest(path),**meta))
    write(root/'status.json',dict(status='held',stage='awaiting_visual_review',ltx_submissions=1,app_promoted=False))
    compare(source,root)


def compare(source,root):
    import cv2
    import numpy as np
    videos=[('Minimal neutral clay',root/'guide.mp4'),('Previous coloured guide result',Path(read(source/'treatment-result.json')['video'])),
            ('Clay + photographic stage keyframes',Path(read(root/'treatment-result.json')['video']))]
    for _,p in videos:metadata(p)
    caps=[cv2.VideoCapture(str(p)) for _,p in videos];dest=root/'comparison.mp4'
    enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x410','-r','24','-i','pipe:0','-an',
        '-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    try:
        for i in range(FRAMES):
            board=np.zeros((410,1920,3),np.uint8)
            for j,(cap,(name,_)) in enumerate(zip(caps,videos)):
                ok,frame=cap.read()
                if not ok:raise ValueError('Comparison decode ended early')
                board[42:,j*640:(j+1)*640]=cv2.resize(frame,(640,368),interpolation=cv2.INTER_AREA)
                cv2.putText(board,name,(j*640+8,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            d=START+i;phase='regional' if d<108 else 'approach' if d<156 else 'settle'
            cv2.putText(board,f'Delivery {d/24:.3f}s | {phase} | original 24 fps | camera segment only',(8,36),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
            enc.stdin.write(board.tobytes())
    finally:
        enc.stdin.close()
        for cap in caps:cap.release()
    if enc.wait():raise RuntimeError('Comparison encoding failed')
    write(root/'comparison.json',dict(video=str(dest),sha256=digest(dest),**metadata(dest),delivery_mapping=list(range(START,END+1)),source_speed=1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','preview','render','review-guide','paint','review-photos','treat','compare'],required=True)
    p.add_argument('--note',default='');a=p.parse_args()
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    os.environ['PYTHONPATH']=os.pathsep.join(str(Path(__file__).resolve().parents[2]/q) for q in ['scripts','src'])
    source,root=a.source.resolve(),a.root.resolve();cfg,identity=prepare(source,root)
    if a.stage in ['preview','render']:render(root,a.stage=='preview')
    elif a.stage=='review-guide':guide_review(root,a.note)
    elif a.stage=='paint':paint(source,root)
    elif a.stage=='review-photos':photo_review(root,a.note)
    elif a.stage=='treat':treat(source,root,identity)
    elif a.stage=='compare':compare(source,root)
    verify_snapshot(identity['source_hashes'])


if __name__=='__main__':main()
