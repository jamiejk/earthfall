"""Bounded photo-anchor experiment; preserve the existing orbital opening."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np

from touchdown.authored_scene import digest, generate_asset
from touchdown.photo_delivery import frame_at, match_metrics, video_info
from touchdown.photo_gate import require_arrival_gate
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.relay_experiment import REPO, RUNTIME, write
from touchdown.relay_timing import schedule, wire_relay

PARENT=REPO/'outputs/touchdown/studio/stonehenge-ltx-relay-handles48-20261023/relay'
SOURCE=REPO/'outputs/touchdown/studio/studio-9kaxjysulpa1'
START=72
BASE_LENGTH=121          # the eight-second generation clock, kept for beat proportions
LENGTH=BASE_LENGTH       # generation frames for this delivery
ANCHOR_OUTPUT=120
DELIVERY=192             # delivery frames; 192 = 8 s, 240 = 10 s


def approach_schedule(cfg,frames=None):
    """Beat schedule for the generated approach. A longer delivery keeps the same beat
    proportions as the eight-second clip and rescales them onto the new clock, so the
    content progression is preserved rather than being cut short or padded."""
    frames=LENGTH if frames is None else frames
    original=schedule(cfg)
    timing=copy.deepcopy(original)
    timing.update(source_frames=frames,delivery_frames=frames-1,native_portrait_frames=25,
                  parent_delivery_start_frame=START,opening_handle_frames=0)
    timing['global_prompt'] += (' The ancient standing stones are solid immovable masonry. Every upright and lintel '
        'retains its shape, orientation and relationship to its neighbours. Only the camera and the person move. '
        'No bending, stretching, rocking, reshuffling or growing stones.')
    beats=[dict(b) for b in original['generation_beats'] if b['start_frame']<BASE_LENGTH]
    beats[-1]['end_frame']=min(beats[-1]['end_frame'],BASE_LENGTH)
    delta=frames-BASE_LENGTH
    if delta:
        # Extra delivery time belongs to the settle, which is the beat nearest the portrait:
        # that is where the figure is invented. Every other beat keeps its original delivery
        # time, so the approach and the high aerial are unchanged.
        beats[0]['end_frame']+=delta
        for beat in beats[1:]:
            beat['start_frame']+=delta;beat['end_frame']+=delta
    timing['generation_beats']=beats
    timing['single_prompt']=timing['global_prompt']+' '+' '.join(b['prompt'] for b in timing['generation_beats'])
    timing['forward_beats']=[dict(b,start_frame=max(START,b['start_frame'])-START,
        end_frame=b['end_frame']-START,start_seconds=(max(START,b['start_frame'])-START)/24,
        end_seconds=(b['end_frame']-START)/24) for b in original['forward_beats'] if b['end_frame']>START]
    return timing


def output_to_source(frame):
    if not START<=frame<DELIVERY:raise ValueError('Approach frame outside the retained delivery')
    return DELIVERY-frame


def checked_record(record):
    for path,key in [('video','sha256'),('workflow','workflow_sha256')]:
        if digest(Path(record[path]))!=record[key]:raise ValueError('Changed source '+path)


def inputs(root):
    cfg=json.loads((SOURCE/'scene-config.json').read_text())
    parent=json.loads((PARENT/'delivery.json').read_text());checked_record(parent['flight']);checked_record(parent['native_portrait'])
    if digest(Path(parent['video']))!=parent['sha256']:raise ValueError('Changed parent film')
    verify_lineage(cfg,SOURCE/'scene',asset_plan(cfg,SOURCE/'scene'))
    require_arrival_gate(cfg,SOURCE/'scene',root)
    if digest(SOURCE/'scene/arrival.png')!=parent['arrival_sha256']:raise ValueError('Changed arrival')
    return cfg,parent


def anchor(root):
    from touchdown.studio_build import runtime,COMFY_URL
    cfg,parent=inputs(root)
    cap=cv2.VideoCapture(parent['video'])
    try:
        for name,index in [('far',START),('composition',ANCHOR_OUTPUT)]:
            cv2.imwrite(str(root/(name+'.png')),frame_at(cap,index))
    finally:cap.release()
    prompt=('Edit image 1 into ONE single full-frame live-action aerial photograph. KEEP its exact camera '
        'position, perspective, heading, composition, stone-circle footprint, monument scale, paths, grass '
        'and tiny beige-coated person. Repair the stones into physically solid ancient weathered masonry '
        'with sharp photographic surface detail and coherent lintels supported by paired uprights. '
        'Use image 2 ONLY as a visual reference for the same landmark’s stone identities, materials and '
        'natural light. Image 2 must NOT appear as another picture or panel. Output exactly one camera view '
        'filling the entire image: no montage, grid, multiple views, split-screen or borders. Natural soft '
        'daylight and real grass, no illustration, miniature model, CGI or text.')
    spec=dict(name='approach-anchor',references=[str(root/'composition.png'),str(SOURCE/'scene/arrival.png')],
        prompt=prompt,models=cfg['models'],seed=cfg['seed']+502,width=2560,height=1472)
    with runtime('comfy',root):generate_asset(COMFY_URL,spec,root)
    write(root/'anchor-binding.json',dict(delivery_frame=ANCHOR_OUTPUT,delivery_seconds=ANCHOR_OUTPUT/24,
        generation_frame=output_to_source(ANCHOR_OUTPUT),clip_delivery_start=START,
        guide_type='LTXVAddGuide single image',strength=1,reference=str(root/'approach-anchor.png'),
        sha256=digest(root/'approach-anchor.png'),geometry_guides=False))
    print('ANCHOR READY',root/'approach-anchor.png',flush=True)


def assemble(root,folder,parent,portrait=24,grade=False):
    """Deliver the film. ``portrait`` is how many frames of the real moving portrait replace the
    generated tail; 24 is the one-second ending, larger values bring the photographic figure in
    earlier at the cost of more of the generated approach being dropped.

    ``grade`` matches the generated approach's mean level to the preserved opening at the 3 s join
    and to the real portrait at the end, ramping linearly between them. A two-stage refine
    re-renders the clip at a different exposure, which otherwise lands as a visible darkening at
    the cut; the correction is a per-channel gain with no colour rotation."""
    if not 1<=portrait<=72:raise ValueError('Portrait window must be between 1 and 72 frames')
    record=json.loads((folder/'result.json').read_text());checked_record(record)
    captures=[cv2.VideoCapture(record['video']),cv2.VideoCapture(parent['video']),cv2.VideoCapture(parent['native_portrait']['video'])]
    a,b,n=captures
    try:
        if video_info(a)[0:1]!=(LENGTH,) or video_info(a)[3]!=24:raise ValueError('Wrong approach clock')
        if video_info(b)!=(192,2560,1472,24):raise ValueError('Wrong parent clock')
        if video_info(n)[0]<portrait:raise ValueError('Native portrait is shorter than the requested window')
        near_seam=match_metrics(frame_at(a,portrait),frame_at(n,0))
        far_seam=match_metrics(frame_at(a,DELIVERY-START),frame_at(b,72))
        native=near_seam['acceptable']
        gains=None
        if grade:
            # Per-frame gain toward a target curve that starts at the preserved opening's level and
            # ends at the real portrait's, then box-smoothed so legitimate lighting changes inside
            # the shot survive while the refine's exposure drift is removed. A single head gain
            # cannot hold a drift that keeps moving after the seam.
            sources=list(range(DELIVERY-START,portrait,-1))
            head=frame_at(b,START-1).reshape(-1,3).mean(0).astype('float64')
            tail=frame_at(n,0).reshape(-1,3).mean(0).astype('float64')
            raw=[]
            for k,src in enumerate(sources):
                t=k/max(1,len(sources)-1)
                desired=head*(1.-t)+tail*t
                actual=frame_at(a,src).reshape(-1,3).mean(0).astype('float64')
                raw.append(np.clip(desired/np.maximum(actual,1.),0.7,1.4))
            raw=np.array(raw)
            window=25
            kernel=np.ones(window)/window
            norm=np.convolve(np.ones(len(raw)),kernel,mode='same')
            gains=np.stack([np.convolve(raw[:,c],kernel,mode='same')/norm for c in range(3)],axis=1)
            summary=dict(head=[round(float(v),4) for v in raw[0]],
                         tail=[round(float(v),4) for v in raw[-1]],
                         frames=len(raw),smooth_window=window,
                         min=[round(float(v),4) for v in gains.min(0)],
                         max=[round(float(v),4) for v in gains.max(0)])
        target=folder/'earth_zoom.mp4'
        enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','2560x1472',
            '-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        try:
            span=max(1,DELIVERY-portrait-START)
            for output in range(DELIVERY):
                source=DELIVERY-output
                image=(frame_at(b,output) if output<START
                    else frame_at(n,portrait-1-source) if native and source<portrait
                    else frame_at(a,source))
                if grade and START<=output<DELIVERY-portrait:
                    factor=gains[output-START]
                    image=np.clip(image.astype('float32')*np.array(factor,dtype='float32'),0,255).astype('uint8')
                if image.shape[1::-1]!=(2560,1472):image=cv2.resize(image,(2560,1472),interpolation=cv2.INTER_LANCZOS4)
                enc.stdin.write(image.tobytes())
        finally:enc.stdin.close();code=enc.wait()
        if code:raise RuntimeError('Export failed')
        c=cv2.VideoCapture(str(target));info=video_info(c);c.release()
        if info!=(DELIVERY,2560,1472,24):raise ValueError('Wrong delivery clock')
        delivery=dict(video=str(target),sha256=digest(target),flight=record,native_portrait=parent['native_portrait'],
            arrival_sha256=parent['arrival_sha256'],seam=near_seam,far_seam=far_seam,native_portrait_used=native,
            native_start_frame=(DELIVERY-portrait) if native else None,native_first_frame=0,
            native_frames=portrait if native else 0,
            portrait_window_frames=portrait,
            colour_match=summary if grade else None,
            portrait_prefix_frames=25,preserved_opening=dict(video=parent['video'],sha256=parent['sha256'],frames=list(range(START))),
            flight_source_mapping=list(range(DELIVERY-START,0,-1)),output_frames=DELIVERY,delivered_fps=24,
            duration=DELIVERY/24,delivered_size=[2560,1472],
            source_speed=1.,native_speed=1.,interpolation=False,crossfade=False,face_composite=False,three_dimensional_guides=False)
        write(folder/'delivery.json',delivery)
        return delivery
    finally:
        for c in captures:c.release()


def anchor_guides(graph,frame,strength):
    """Set the blend strength of every guide pinned at ``frame`` (stage 1 and stage 2 each get one)."""
    nodes=[node for node,body in graph.items()
        if body.get('class_type')=='LTXVAddGuide' and body['inputs'].get('frame_idx')==frame]
    if not nodes:raise ValueError('No anchor guide at frame %d'%frame)
    for node in nodes:graph[node]['inputs']['strength']=strength
    return sorted(nodes,key=int)


def flight(root,variants,anchor_strength=1.0,anchor_frame=None,extras=(),portrait=24,upscale=False,base=(1280,736),grade=False):
    from touchdown.studio_build import runtime,VIDEO_URL
    import fakeout_earth_zoom as f
    # A root has to be seeded with an already-reviewed anchor; a mistyped --root otherwise
    # fails much later with a bare FileNotFoundError that says nothing about the mistake.
    missing=[name for name in ('approach-anchor.png','anchor-visual-review.json','anchor-binding.json','far.png')
        if not (root/name).exists()]
    if missing:raise SystemExit(f'{root} is not a prepared anchor root: missing {", ".join(missing)}. '
        'Copy the anchor artifacts from a reviewed root first, or run the anchor stage.')
    cfg,parent=inputs(root)
    review=json.loads((root/'anchor-visual-review.json').read_text())
    if review['status']!='pass' or review['sha256']!=digest(root/'approach-anchor.png'):
        raise ValueError('Reference must pass visual review before animation')
    timing=approach_schedule(cfg,LENGTH);write(root/'timing.json',timing)
    node=RUNTIME/'custom_nodes/TouchdownPromptRelay';upstream=json.loads((node/'upstream.json').read_text())
    for name,meta in upstream['files'].items():
        if digest(node/name)!=meta['sha256']:raise ValueError('Changed pinned attention source')
    shutil.copyfile(REPO/'scripts/touchdown/relay_node.py',node/'__init__.py')
    config=RUNTIME/'extra-paths.json';write(config,dict(touchdown_relay=dict(custom_nodes=str(node.parent))))
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --extra-model-paths-config '+shlex.quote(str(config))
    prefix=root/'near-retreat-prefix.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',parent['native_portrait']['video'],'-vf',
        'trim=end_frame=25,reverse,setpts=PTS-STARTPTS','-an','-c:v','libx264','-crf','12',str(prefix)],check=True)
    provenance=dict(source=str(SOURCE),parent_delivery=parent,anchor=json.loads((root/'anchor-binding.json').read_text()),
        models=f.LTX25_MODELS,seed=cfg['seed']+32,frames=LENGTH,fps=24,preserved_opening_frames=START,
        upstream=upstream,geometry_guides=False,comparison=('Same shorter approach and rigidity text; photo guide '
            f'strength {anchor_strength} at generation frame {anchor_frame if anchor_frame is not None else output_to_source(ANCHOR_OUTPUT)}.'))
    write(root/'provenance.json',provenance)
    snapshot=root/'operator-scripts';snapshot.mkdir(exist_ok=True)
    for name in ('anchored_approach.py','relay_timing.py','relay_node.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    with runtime('comfy',root):
        for variant in variants:
            folder=root/variant;folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():assemble(root,folder,parent);continue
            started=time.monotonic();print('START',variant,flush=True)
            with f._client(VIDEO_URL) as client:
                graph={'50':dict(class_type='LoadVideo',inputs=dict(file=client.assets.from_file(str(prefix)))),
                    '51':dict(class_type='GetVideoComponents',inputs=dict(video=['50',0]))}
                pins=[(0,['51',0])];guide_frames=[0,4.5,12.5,20.5]
                anchor_source=output_to_source(ANCHOR_OUTPUT) if anchor_frame is None else anchor_frame
                entries=[]
                if variant=='relay':
                    entries.append((anchor_source,f._load(graph,client,'45',root/'approach-anchor.png')))
                for k,(frame,path,strength) in enumerate(extras):
                    entries.append((frame,f._load(graph,client,str(47+k),Path(path))))
                for frame,image in sorted(entries,key=lambda item:item[0]):
                    if frame in (0,LENGTH-1):raise ValueError('An extra anchor cannot reuse a reserved pin frame')
                    pins.append((frame,image));guide_frames.append(frame)
                pins.append((LENGTH-1,f._load(graph,client,'46',root/'far.png')));guide_frames.append(LENGTH-1)
                wanted=f._ltx(graph,SimpleNamespace(no_upscale=not upscale),LENGTH,pins,timing['single_prompt'],cfg['seed']+32,
                    'touchdown/stonehenge_short_approach_'+variant)
                nodes=anchor_guides(graph,anchor_source,anchor_strength) if variant=='relay' else []
                for frame,path,strength in extras:nodes+=anchor_guides(graph,frame,strength)
                graph['308']['inputs'].update(width=base[0],height=base[1])
                wire_relay(graph,timing,folder/'attention-audit.json',guide_frames,enabled=True)
                saved=f._submit(client,graph,tuple(wanted),folder)
            video=Path(saved[f.LTX_VIDEO]);workflow=video.parent/'graph.json'
            write(folder/'result.json',dict(video=str(video),sha256=digest(video),workflow=str(workflow),workflow_sha256=digest(workflow),
                variant=variant,photo_anchor=variant=='relay',anchor_strength=anchor_strength if variant=='relay' else None,
                anchor_generation_frame=anchor_source if variant=='relay' else None,
                anchor_delivery_frame=(192-anchor_source) if variant=='relay' else None,
                anchor_guide_nodes=nodes,portrait_window_frames=portrait,upscale=upscale,
                extra_anchors=[dict(generation_frame=f,delivery_frame=192-f,path=str(p),strength=s) for f,p,s in extras],
                elapsed_seconds=round(time.monotonic()-started,1)))
            delivery=assemble(root,folder,parent,portrait,grade)
            print('COMPLETE',variant,'native',delivery['native_portrait_used'],'far seam',delivery['far_seam'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['anchor','flight']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=['baseline','relay'],default=['baseline','relay'])
    p.add_argument('--anchor-strength',type=float,default=1.0,help='LTXVAddGuide blend strength for the photo anchor')
    p.add_argument('--anchor-frame',type=int,default=None,help='generation frame for the anchor; default is the 5-second anchor')
    p.add_argument('--extra-anchor',action='append',default=[],metavar='FRAME:PATH:STRENGTH',
        help='one more pin, repeatable; FRAME is a generation frame (delivery frame = DELIVERY-FRAME)')
    p.add_argument('--portrait-frames',type=int,default=24,
        help='how many frames of the real moving portrait end the film (default 24 = one second)')
    p.add_argument('--delivery-seconds',type=float,default=8.0,
        help='total delivery length; the retained opening is always the first 3 s')
    p.add_argument('--upscale',action='store_true',
        help='use the model 2x latent spatial upscaler and refine stage (stage 2) instead of a display resize')
    p.add_argument('--base-resolution',default='1280x736',
        help='generation size WxH; stage 2 doubles it when --upscale is set')
    p.add_argument('--grade-match',action='store_true',
        help='ramp a per-channel gain across the generated approach so both joins match their neighbour')
    a=p.parse_args()
    global DELIVERY,LENGTH
    DELIVERY=int(round(24*a.delivery_seconds))
    LENGTH=DELIVERY-START+1
    if LENGTH%8!=1:raise SystemExit(f'a {a.delivery_seconds}s delivery needs an 8n+1 generation clock, got {LENGTH}')
    if not 1<=a.portrait_frames<LENGTH:raise SystemExit('portrait window must be smaller than the generated clock')
    extras=[]
    for spec in a.extra_anchor:
        try:frame,path,strength=spec.rsplit(':',2)
        except ValueError:raise SystemExit('--extra-anchor must be FRAME:PATH:STRENGTH, got '+spec)
        extras.append((int(frame),path,float(strength)))
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1);os.environ[key]=json.loads(value)
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.stage=='anchor':anchor(root)
    else:
        try:base=tuple(int(v) for v in a.base_resolution.lower().split("x"))
        except ValueError:raise SystemExit('--base-resolution must be WxH, got '+a.base_resolution)
        if len(base)!=2 or base[0]%32 or base[1]%32:
            raise SystemExit('--base-resolution must be two numbers divisible by 32')
        flight(root,a.variants,a.anchor_strength,a.anchor_frame,extras,a.portrait_frames,a.upscale,base,a.grade_match)


if __name__=='__main__':main()
