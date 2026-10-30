"""Recover the specified LTX/Prompt Relay flight without changing its delivery clock.

All scene-dependent text comes from the saved source. Original inputs are immutable;
only the 30,000-ft regional reference is regenerated in the new experiment root.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time

from touchdown import flight_regression as regression, split_flight as flight, flight_contract
from touchdown.authored_scene import generate_asset, digest
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.recipe_snapshot import capture
from touchdown.flight_sequence import review
from touchdown.phase_review import review_opening, review_final

read,write=regression.read,regression.write
PRESET=Path(__file__).with_name('presets')/'split-phase-v1.json'


def configuration(source_cfg):
    from touchdown.flight_recipe import configuration as shared
    return shared(regression.configuration(source_cfg))

def stream_fingerprint(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):sha.update(block)
    stat=path.stat()
    return dict(path=str(path.resolve()),sha256=sha.hexdigest(),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def prepare(source,root,model_roots,model_cache=None):
    regression.check_root(source,root)
    if (root/'acceptance.json').exists():raise ValueError('Approved runs are frozen')
    original=read(source/'scene-config.json');cfg=configuration(original)
    near=flight._checked(read(source/'motion/near/result.json'))
    verify_lineage(original,source/'scene',asset_plan(original,source/'scene'))
    hashes={str(p):digest(p) for p in regression.source_files(source,original,near)}
    from geoblock_core import workflow
    from touchdown.flight_runtime import recipe_hashes
    identity=dict(source=str(source),source_hashes=hashes,config=cfg,contract_sha256=digest(PRESET),
                  code_hashes=recipe_hashes())
    path=root/'experiment.json'
    if path.exists():
        prior=read(path)
        if prior['identity']!=identity:raise ValueError('Recovery inputs/code changed; choose a fresh root')
        for record in prior['model_fingerprints'].values():
            stat=Path(record['path']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(record['size'],record['mtime_ns']):
                raise ValueError('Pinned model weights changed')
        regression.verify_snapshot(hashes);capture(root)
        return prior
    preparation=root/'preparation.json'
    if root.exists() and any(root.iterdir()) and (not preparation.exists() or read(preparation)!=identity):
        raise ValueError('Fresh recovery root required')
    write(preparation,identity)
    weights={f'ltx_{k}':v for k,v in workflow.LTX25_MODELS.items() if k in ['transformer','text_encoder','video_vae']}
    weights.update({f'still_{k}':v for k,v in cfg['models'].items()})
    models=(read(root/'model-fingerprints.json') if (root/'model-fingerprints.json').exists()
            else read(model_cache) if model_cache else {})
    paths={}
    for role,name in weights.items():
        matches=[(kind,p) for r in model_roots for kind in ['diffusion_models','text_encoders','vae'] if (p:=r/kind/name).is_file()]
        if not matches:raise ValueError(f'Missing installed model: {name}')
        kind,p=matches[0];stat=p.stat();previous=models.get(role)
        if not previous or (previous['path'],previous['size'],previous['mtime_ns'])!=(str(p.resolve()),stat.st_size,stat.st_mtime_ns):
            print(f'Fingerprinting {role}: {name}',flush=True)
            models[role]=stream_fingerprint(p);write(root/'model-fingerprints.json',models)
        paths.setdefault(kind,set()).add(str(p.resolve().parent))
    manifest=dict(identity=identity,native_portrait=near,model_fingerprints=models,
                  model_paths={kind:'\n'.join(sorted(dirs)) for kind,dirs in paths.items()},
                  seed=cfg['seed'],parent_seed=cfg['seed']+32,descent_seed=cfg['seed']+33,
                  engine='Existing split_flight.parent/descent with TouchdownPromptRelay; no Wan',
                  bitwise_gpu_reproduction='not guaranteed across hardware/kernel versions')
    write(path,manifest);capture(root)
    scene=root/'scene';scene.mkdir(exist_ok=True)
    for name in ['arrival','orbit']:
        for suffix in ['.png','.json','.workflow.json']:
            target=scene/(name+suffix)
            if name=='arrival' and suffix=='.png':
                target.symlink_to((source/'scene'/target.name).resolve())
            else:shutil.copy2(source/'scene'/target.name,target)
    write(root/'scene-config.json',cfg)
    o=flight.options(cfg)
    timings={stage:builder(cfg,o) for stage,builder in [('parent',flight.parent_timing),('descent',flight.descent_timing)]}
    for stage,timing in timings.items():flight_contract.validate_timing(timing,o,stage)
    write(root/'phase-preflight.json',dict(status='pass',contract=cfg['video']['phase_contract'],timings=timings,
        scene_input_boundary='Existing photographic roots, arrival, orbit, native portrait; fresh 30,000-ft regional reference'))
    write(root/'status.json',dict(stage='prepared',status='ready',publication='held for phase and motion review'))
    return manifest


def run(source,root,model_roots,prepare_only=False,model_cache=None):
    m=prepare(source,root,model_roots,model_cache)
    if prepare_only:return 0
    cfg=m['identity']['config'];scene=root/'scene';output=root/'motion';near=source/'motion/near'
    repo=Path(__file__).resolve().parents[2]
    os.environ['PYTHONPATH']=os.pathsep.join(str(p) for p in [repo/'scripts',repo/'src'])
    from touchdown import studio_build as studio
    from touchdown.studio_brief import approach_view_review
    config=root/'pinned-model-paths.json'
    write(config,dict(recovery=dict(is_default=True,**m['model_paths'])))
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--deterministic' not in command.split():command+=' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --extra-model-paths-config '+shlex.quote(str(config))
    stage='relay';started=time.monotonic()
    try:
        write(root/'relay-upstream.json',flight.enable_prompt_relay())
        from touchdown.flight_runtime import capture_runtime
        capture_runtime(root)
        for stage in ['regional_reference','regional_review','parent','parent_review','anchor','descent','review']:
            regression.verify_snapshot(m['identity']['source_hashes'])
            write(root/'status.json',dict(stage=stage,status='running',publication='held for phase and motion review'))
            print('START '+stage,flush=True)
            if stage=='regional_reference':
                spec=next(s for s in asset_plan(cfg,scene) if s['name']=='high')
                with studio.runtime('comfy',root):generate_asset(studio.COMFY_URL,spec,scene)
                verify_lineage(cfg,scene,asset_plan(cfg,scene))
            elif stage=='regional_review':
                from touchdown.flight_sequence import observed_chat,FrameObservation,classify
                with studio.runtime('vision',root) as metadata:
                    observation=observed_chat(studio.VISION_URL,[(scene/'high.png','REGIONAL REFERENCE')],
                        'Describe only the actual image. Is this many kilometres of connected regional terrain/water '
                        'with microscopic landmarks and no readable facades or people? A close castle with distant '
                        'hills is detailed_ground. Cloud-obscured or ambiguous texture is unclear. Independently '
                        'report visible scale, dominant nearby objects, clouds and planetary limb. Do not infer altitude.',
                        FrameObservation,root/'gates/regional.json',metadata)
                write(root/'regional-reference-review.json',dict(observation=observation,sha256=digest(scene/'high.png')))
                if classify(observation)!='high_regional':raise studio.Held('Regional reference fails broad, clear regional-scale review')
            elif stage=='parent':
                if not (output/'flight/anchor-binding.json').exists():
                    with studio.runtime('comfy',root):flight.parent(cfg,scene,output,studio.VIDEO_URL,studio.COMFY_URL,near)
            elif stage=='parent_review':
                result=review_opening(root)
                if result['status']!='pass':raise studio.Held('; '.join(result['failures']))
            elif stage=='anchor':
                anchor=output/'flight/anchor.png';path=output/'flight/anchor-review.json'
                if path.exists():answer=read(path)
                else:
                    with studio.runtime('vision',root):answer=approach_view_review(studio.VISION_URL,anchor,root/'gates/anchor.json')
                    write(path,dict(answer,sha256=digest(anchor)))
                if answer['status']!='pass' or answer.get('sha256',digest(anchor))!=digest(anchor):raise studio.Held('Anchor failed its review')
            elif stage=='descent':
                if not (output/'pacing.json').exists():
                    with studio.runtime('comfy',root):flight.descent(cfg,scene,output,studio.VIDEO_URL,near)
                pacing=read(output/'pacing.json')
                mapping=[(s,int(i)) for s,i in (r.split(':') for r in pacing['source_frame_mapping'])]
                flight_contract.validate_mapping(mapping,cfg['video']['split']['prefix_frames'])
                if pacing['dropped_hold_frames'] or pacing['fps']!=24:raise ValueError('Delivery was retimed')
            else:
                result=review_final(root,cfg)
                from touchdown.flight_report import export
                export(output/'earth_zoom.mp4',root/'board',root/'gates/flight-sequence/report.json')
                write(root/'status.json',dict(stage='review',status='review_candidate' if result['status']=='pass' else 'held',
                    failures=result['verdict']['failures'],elapsed_seconds=round(time.monotonic()-started,1),publication='Human playback review required'))
                return 0 if result['status']=='pass' else 2
            print('COMPLETE '+stage,flush=True)
    except Exception as exc:
        write(root/'status.json',dict(stage=stage,status='held',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=round(time.monotonic()-started,1)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-root',type=Path,action='append',required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--model-cache',type=Path,help='Reuse streamed hashes only for matching resolved path, size and modification time')
    args=p.parse_args()
    return run(args.source.resolve(),args.root.resolve(),args.model_root,args.prepare_only,args.model_cache)


if __name__=='__main__':raise SystemExit(main())
