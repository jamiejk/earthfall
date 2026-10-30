"""Frozen published-recipe correction of the camera-scene v2 pilot.

Keeps its 97 delivery frames (96..192), camera, weights and appearance anchors.
Only the inference recipe changes: 960x544 -> 1920x1088, both guide sets,
8+3 distilled steps, a short appearance prompt and the same seed in both stages.
This is a recipe-bundle comparison, not an isolated test of upscale alone.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from touchdown.authored_scene import digest
from touchdown.flight_regression import check_root, read, write, verify_snapshot
from touchdown.camera_scene import START, END, FRAMES, require_raw_review, require_photo_review
from touchdown.recipe_snapshot import capture

VERSION = 'layout-recipe-trial-v1'
SIZE = (1920, 1088)
PROMPT = ('A photoreal aerial view of a broad blue lake and its green Swiss shoreline in natural daylight, '
          'descending toward a woman with brown hair and a beige trench coat opening her arms beside a '
          'gravel path, with a pale stone castle, slate roofs and trees behind her. '
          'Natural irregular foliage, weathered masonry, grass and fabric, realistic atmospheric depth '
          'and consistent daylight throughout the continuous camera move.')


def build_graph(seed, guide, first, last):
    from geoblock_core.workflow import ShotRequest, _add_ltx25
    graph = {
        '50': dict(class_type='LoadVideo', inputs=dict(file=guide)),
        '60': dict(class_type='GetVideoComponents', inputs=dict(video=['50', 0])),
        '40': dict(class_type='LoadImage', inputs=dict(image=first)),
        '41': dict(class_type='LTXVPreprocess', inputs=dict(image=['40', 0], img_compression=18)),
        '42': dict(class_type='LoadImage', inputs=dict(image=last)),
        '43': dict(class_type='LTXVPreprocess', inputs=dict(image=['42', 0], img_compression=18)),
    }
    request = ShotRequest(name=VERSION, seed=seed, output_prefix='touchdown/chillon_published_recipe',
                          ltx_upscale=True, save_stage1=True, pin_last_keyframe=True)
    _add_ltx25(graph, request, FRAMES, ['41', 0], ['43', 0], [], PROMPT, layout=True)
    # The published graph uses the same supplied seed for each sampler.
    graph['360']['inputs']['noise_seed'] = seed
    graph['306']['inputs']['text'] = ''
    validate_graph(graph, seed)
    return graph


def validate_graph(graph, seed):
    """Fail before submission if a helper change silently drops the recipe/clock."""
    def inputs(node): return graph[node]['inputs']
    if inputs('308') != dict(width=960, height=544, length=FRAMES, batch_size=1):
        raise ValueError('First-stage dimensions or original frame count changed')
    if inputs('307')['frame_rate'] != 24 or any(inputs(n)['fps'] != 24 for n in ['371', '376']):
        raise ValueError('Original 24 fps clock changed')
    if inputs('301')['lora_name'] != 'ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors' or inputs('301')['strength_model'] != 1:
        raise ValueError('Layout LoRA or full strength changed')
    if graph['331']['class_type'] != 'LTXVLatentUpsampler' or inputs('331')['samples'] != ['395', 2]:
        raise ValueError('Second stage must upscale the cropped first-stage latent')
    for node, frame, image in [('310', 0, ['60', 0]), ('311', -1, ['41', 0]),
                               ('340', 0, ['60', 0]), ('341', -1, ['41', 0])]:
        v = inputs(node)
        if graph[node]['class_type'] != 'LTXAddVideoICLoRAGuide' or (v['frame_idx'], v['image'], v['strength']) != (frame, image, 1):
            raise ValueError('Both stages require full layout and first-frame appearance conditioning')
    if inputs('340')['positive'] != ['307', 0] or inputs('340')['latent'] != ['331', 0]:
        raise ValueError('Second-stage references must be re-encoded onto the upscaled latent')
    for node in ['312', '342']:
        v = inputs(node)
        if graph[node]['class_type'] != 'LTXVAddGuide' or (v['image'], v['frame_idx'], v['strength']) != (['43', 0], -1, 1):
            raise ValueError('The identical final appearance anchor is required in both stages')
    for base, steps in [('390', 8), ('360', 3)]:
        b = int(base)
        if inputs(base)['noise_seed'] != seed or inputs(str(b+1))['cfg'] != 1 or inputs(str(b+1))['model'] != ['301', 0]:
            raise ValueError('Both stages require the frozen seed, CFG 1 and Layout LoRA')
        if inputs(str(b+2))['sampler_name'] != 'euler_ancestral':
            raise ValueError('Published distilled sampler changed')
        if len(inputs(str(b+3))['sigmas'].split(','))-1 != steps:
            raise ValueError('Published 8+3 step schedule changed')
    if inputs('370')['samples'] != ['365', 2]:
        raise ValueError('Output must decode the cropped second-stage result')


def metadata(video):
    import cv2
    c = cv2.VideoCapture(str(video))
    record = dict(frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)), fps=c.get(cv2.CAP_PROP_FPS),
                  width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    c.release()
    if (record['frames'], record['fps']) != (FRAMES, 24):
        raise ValueError(f'Changed clock: {video}: {record}')
    return record


def prepare(source, root, reference):
    from PIL import Image
    from touchdown.flight_runtime import recipe_hashes
    check_root(source, root)
    require_raw_review(source); require_photo_review(source)
    prior = read(source/'experiment.json')['identity']
    verify_snapshot(prior['source_hashes'])
    files = [source/n for n in ['experiment.json', 'guide.mp4', 'geometry-audit.json', 'render.json',
             'raw-guide-review.json', 'photo-anchor-review.json', 'photo-first.png', 'photo-last.png',
             'treatment-spec.json', 'treatment-result.json', 'model-fingerprints.json', 'pinned-model-paths.json']]
    files += [Path(read(source/'treatment-result.json')['video'])]
    files += [reference/n for n in ['official-workflow.json', 'official-ic-lora.py']]
    files += [Path('.firecrawl/ltx25-layout-to-render-20261026.md').resolve()]
    identity = dict(version=VERSION, source=str(source), source_hashes={str(p):digest(p) for p in files},
        code_hashes=recipe_hashes(), seed=read(source/'treatment-spec.json')['seed'],
        frames=FRAMES, fps=24, delivery_mapping=list(range(START, END+1)), output_size=list(SIZE),
        prompt=PROMPT, strengths=dict(lora=1, layout=1),
        changes=['two-stage 960x544 -> 1920x1088', 'short appearance prompt',
                 'same seed in both samplers, as in the published graph', 'empty negative prompt'],
        retained=['97-frame original clock', 'all original camera poses', 'same first and last keyframes',
                  'same quantized base and encoder, VAE and Layout LoRA'],
        spatial_conversion='Lanczos whole-frame resize, no crop/padding; aspect changes by 1.47%; no temporal conversion',
        limitations=['int8 convrot base/encoder remain, rather than published BF16 builds',
                     'final appearance reference already contains simplified architecture and rounded trees',
                     'recipe bundle comparison; cannot attribute effect to one setting alone'])
    path=root/'experiment.json'
    if path.exists():
        if read(path)['identity'] != identity: raise ValueError('Recipe/input changed; use a new run root')
        verify_snapshot(identity['source_hashes']); capture(root)
        verify_snapshot(read(root/'prepared-inputs.json')['hashes'])
        return identity
    if root.exists() and any(root.iterdir()): raise ValueError('Fresh experiment root required')
    write(path, dict(identity=identity)); capture(root)
    refs=root/'published-reference'; refs.mkdir()
    for name in ['official-workflow.json', 'official-ic-lora.py']: shutil.copy2(reference/name, refs/name)
    shutil.copy2('.firecrawl/ltx25-layout-to-render-20261026.md', refs/'model-card.md')
    for name in ['first', 'last']:
        with Image.open(source/f'photo-{name}.png') as im:
            im.convert('RGB').resize(SIZE, Image.Resampling.LANCZOS).save(root/f'photo-{name}.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source/'guide.mp4'), '-vf',
                    f'scale={SIZE[0]}:{SIZE[1]}:flags=lanczos,setsar=1', '-an', '-fps_mode', 'passthrough',
                    '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(root/'guide.mp4')], check=True)
    for name in ['geometry-audit.json', 'render.json']: shutil.copy2(source/name, root/name)
    guide_meta=metadata(root/'guide.mp4')
    write(root/'prepared-inputs.json', dict(hashes={str(root/n):digest(root/n) for n in ['guide.mp4','photo-first.png','photo-last.png',
          'geometry-audit.json','render.json']}, guide=guide_meta, temporal_mapping=list(range(START,END+1))))
    write(root/'preflight.json', dict(status='pass', frames=FRAMES, fps=24,
        stages=[dict(width=960,height=544,steps=8),dict(width=1920,height=1088,steps=3)],
        guide_in_both_stages=True, appearance_in_both_stages=True, camera_poses_unchanged=True,
        source_appearance_review=read(source/'photo-anchor-review.json'),
        appearance_limit='The existing endpoint retains visibly rendered scenery; no claim it is a fully photographic anchor.'))
    return identity


def setup_runtime(source, root):
    from touchdown.flight_runtime import capture_runtime
    from touchdown.phase_recovery import stream_fingerprint
    models=read(source/'model-fingerprints.json')
    for record in models.values():
        s=Path(record['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(record['size'],record['mtime_ns']): raise ValueError('Pinned model changed')
    upscaler=Path('/run/media/jamie/Image Creation/Live-Models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors')
    models['spatial_upscaler']=stream_fingerprint(upscaler)
    write(root/'model-fingerprints.json', models)
    paths=read(source/'pinned-model-paths.json')
    paths['pilot']['latent_upscale_models']=str(upscaler.parent)
    config=root/'pinned-model-paths.json'; write(config, paths)
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--extra-model-paths-config' in command: raise ValueError('Uncontrolled model path override')
    if '--deterministic' not in command.split(): command+=' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --extra-model-paths-config '+shlex.quote(str(config))
    capture_runtime(root)
    cwd=Path(os.environ['TOUCHDOWN_COMFY_SERVER_CWD'])
    sources=set((cwd/'custom_nodes').rglob('*.py'))
    write(root/'custom-node-fingerprints.json',dict(files={str(p):digest(p) for p in sorted(sources)}))


def treat(source, root, identity):
    import fakeout_earth_zoom as f
    from touchdown import studio_build as studio
    from geoblock_core.workflow import LTX_VIDEO, STAGE1_VIDEO
    if (root/'treatment-result.json').exists(): raise ValueError('Completed treatment is frozen; no automatic reroll')
    setup_runtime(source, root)
    with studio.runtime('comfy',root), f._client(studio.VIDEO_URL) as client:
        assets={n:client.assets.from_file(str(root/n)) for n in ['guide.mp4','photo-first.png','photo-last.png']}
        graph=build_graph(identity['seed'],assets['guide.mp4'],assets['photo-first.png'],assets['photo-last.png'])
        # Keep the exact graph even if inference fails before the SDK saves it.
        # SDK file assets are typed references; retain them like _submit does.
        write(root/'submitted-graph.json',json.loads(json.dumps(graph,default=str)))
        write(root/'status.json',dict(status='running',stage='two_stage_generation',publication='held'))
        saved=f._submit(client,graph,(LTX_VIDEO,STAGE1_VIDEO),root/'ltx')
    videos={}
    for name,node,size in [('stage1',STAGE1_VIDEO,(960,544)),('treatment',LTX_VIDEO,SIZE)]:
        p=Path(saved[node]); meta=metadata(p)
        if (meta['width'],meta['height'])!=size: raise ValueError('Rendered stage size differs from recipe')
        videos[name]=dict(video=str(p),sha256=digest(p),**meta)
    write(root/'treatment-result.json', videos['treatment'])
    write(root/'stage1-result.json', videos['stage1'])
    write(root/'status.json',dict(status='held',stage='awaiting_visual_review',videos=videos,app_promoted=False))
    comparison(source, root)


def comparison(source, root):
    import cv2
    import numpy as np
    records=[('Persistent camera guide',root/'guide.mp4'),
             ('Previous: single stage',Path(read(source/'treatment-result.json')['video'])),
             ('Published recipe: two stages',Path(read(root/'treatment-result.json')['video']))]
    for _,p in records: metadata(p)
    caps=[cv2.VideoCapture(str(p)) for _,p in records]
    dest=root/'comparison.mp4'
    enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x410','-r','24','-i','pipe:0',
                          '-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    try:
        for i in range(FRAMES):
            board=np.zeros((410,1920,3),np.uint8)
            for j,(cap,(name,_)) in enumerate(zip(caps,records)):
                ok,frame=cap.read()
                if not ok: raise ValueError('Comparison source ended early')
                board[42:,j*640:(j+1)*640]=cv2.resize(frame,(640,368),interpolation=cv2.INTER_AREA)
                cv2.putText(board,name,(j*640+8,18),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
            d=START+i;phase='regional' if d<108 else 'approach' if d<156 else 'settle'
            cv2.putText(board,f'Delivery {d/24:.3f}s | {phase} | original 24 fps | camera segment only',
                        (8,36),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
            enc.stdin.write(board.tobytes())
    finally:
        enc.stdin.close()
        for c in caps:c.release()
    if enc.wait():raise RuntimeError('Comparison encoding failed')
    write(root/'comparison.json',dict(video=str(dest),sha256=digest(dest),**metadata(dest),
        delivery_mapping=list(range(START,END+1)),source_speed=1,
        panels=[dict(title=n,video=str(p),sha256=digest(p)) for n,p in records]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','treat','compare'],required=True)
    a=p.parse_args()
    env=Path('/tmp/touchdown-generation-runtime.env')
    for line in env.read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    os.environ['PYTHONPATH']=os.pathsep.join(str(Path(__file__).resolve().parents[2]/q) for q in ['scripts','src'])
    source,root,reference=a.source.resolve(),a.root.resolve(),a.reference.resolve()
    identity=prepare(source,root,reference)
    if a.stage=='treat':treat(source,root,identity)
    elif a.stage=='compare':comparison(source,root)
    verify_snapshot(identity['source_hashes'])


if __name__=='__main__':main()
