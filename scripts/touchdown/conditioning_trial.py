"""Matched parent-only LTX conditioning trial, preserving all scene pixels and timing.

No seed rerolls, regional-image regeneration, descent, or delivery publication.
Run --stage probe first: real encoders/connectors, zero diffusion steps.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

from touchdown import split_flight as flight, flight_contract as contract
from touchdown.flight_regression import read, write, verify_snapshot, check_root
from touchdown.authored_scene import digest
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.flight_runtime import recipe_hashes, capture_runtime
from touchdown.recipe_snapshot import capture

REPO = Path(__file__).resolve().parents[2]
VARIANTS = contract.CONTEXT_STRATEGIES


def prepare(source, root, strategy):
    check_root(source, root)
    prior = read(source/'experiment.json')
    verify_snapshot(prior['identity']['source_hashes'])
    cfg = contract.configure(prior['identity']['config'], strategy)
    # Only implementation metadata changes; no new scene/reference prompts.
    for field in ['prompts', 'words', 'seed', 'subject', 'references']:
        if cfg.get(field) != prior['identity']['config'].get(field):
            raise ValueError('Matched scene inputs changed')
    near = Path(prior['identity']['source'])/'motion/near'
    flight._checked(read(near/'result.json'))
    files = [source/'experiment.json', near/'result.json', near/'provenance.json']
    files += [source/'scene'/(name+suffix) for name in ['arrival','orbit','high']
              for suffix in ['.png','.json','.workflow.json']]
    hashes = dict(prior['identity']['source_hashes'])
    hashes.update({str(p.resolve()):digest(p) for p in files})
    manifest = dict(identity=dict(source=str(source), source_hashes=hashes, config=cfg,
                    code_hashes=recipe_hashes()), near=str(near), strategy=strategy,
                    model_fingerprints=prior['model_fingerprints'], model_paths=prior['model_paths'],
                    seed=cfg['seed'], parent_seed=cfg['seed']+32,
                    scope='Matched parent and encoder probe; not a completed delivery')
    for record in manifest['model_fingerprints'].values():
        stat=Path(record['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(record['size'],record['mtime_ns']):
            raise ValueError('Previously fingerprinted model changed')
    target = root/'experiment.json'
    if target.exists():
        if read(target) != manifest:
            raise ValueError('Experiment identity changed; use a fresh root')
        verify_snapshot(hashes)
        capture(root)
        return manifest
    if root.exists() and any(root.iterdir()):
        raise ValueError('Fresh variant root required')
    write(target,manifest)
    capture(root)
    scene=root/'scene';scene.mkdir()
    for name in ['arrival','orbit','high']:
        for suffix in ['.png','.json','.workflow.json']:
            p=source/'scene'/(name+suffix)
            if suffix=='.png':(scene/p.name).symlink_to(p.resolve())
            else:shutil.copy2(p,scene/p.name)
    verify_lineage(cfg,scene,asset_plan(cfg,scene))
    write(root/'scene-config.json',cfg)
    timing=flight.parent_timing(cfg,flight.options(cfg))
    contract.validate_timing(timing,flight.options(cfg),'parent')
    write(root/'timing.json',timing)
    return manifest


def configure_runtime(root, manifest):
    os.environ['PYTHONPATH']=os.pathsep.join(str(REPO/p) for p in ['scripts','src'])
    path=root/'pinned-model-paths.json'
    write(path,dict(recovery=dict(is_default=True,**manifest['model_paths'])))
    import shlex
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--deterministic' not in command.split():command+=' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --extra-model-paths-config '+shlex.quote(str(path))
    write(root/'relay-upstream.json',flight.enable_prompt_relay())
    capture_runtime(root)


def preview(root):
    from touchdown.phase_review import parent_preview
    return parent_preview(root)

def run_variant(source, root, strategy, stage):
    import fakeout_earth_zoom as f
    from touchdown import studio_build as studio
    from touchdown.phase_review import review_opening
    manifest=prepare(source,root,strategy)
    configure_runtime(root,manifest)
    cfg=manifest['identity']['config']
    verify_snapshot(manifest['identity']['source_hashes'])
    started=time.monotonic()
    write(root/'status.json',dict(stage=stage,status='running',publication='diagnostic only'))
    try:
        if stage=='probe':
            folder=root/'probe';folder.mkdir(exist_ok=True)
            if not (folder/'attention-audit.json').exists():
                original=read(source/'motion/flight/parent/contract-graph.json')
                graph={k:copy.deepcopy(original[k]) for k in ['300','304','308']}
                inputs=copy.deepcopy(original['380']['inputs'])
                inputs.update(timing_json=json.dumps(read(root/'timing.json')),audit_path=str(folder/'attention-audit.json'))
                graph['900']=dict(class_type='TouchdownRelayProbe',inputs=inputs)
                with studio.runtime('comfy',root), f._client(studio.VIDEO_URL) as client:
                    f._submit(client,graph,(),folder)
            audit=read(folder/'attention-audit.json')
            if audit.get('probe',{}).get('status')!='pass':raise ValueError('Encoder probe incomplete')
        elif stage=='render':
            audit=read(root/'probe/attention-audit.json')
            if audit.get('probe',{}).get('status')!='pass':raise ValueError('Run the encoder probe first')
            output=root/'motion'
            if not (output/'flight/parent/result.json').exists():
                with studio.runtime('comfy',root):
                    flight.parent(cfg,root/'scene',output,studio.VIDEO_URL,studio.COMFY_URL,Path(manifest['near']))
            preview(root)
            result=review_opening(root,cfg)
            write(root/'status.json',dict(stage=stage,status='review_candidate' if result['status']=='pass' else 'held',
                failures=result['failures'],elapsed_seconds=round(time.monotonic()-started,1),publication='Parent diagnostic only'))
            verify_snapshot(manifest['identity']['source_hashes'])
            return result
        verify_snapshot(manifest['identity']['source_hashes'])
        write(root/'status.json',dict(stage=stage,status='pass',elapsed_seconds=round(time.monotonic()-started,1)))
        return read(root/'status.json')
    except BaseException as exc:
        write(root/'status.json',dict(stage=stage,status='held',error=f'{type(exc).__name__}: {exc}'))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['probe','render'],required=True)
    p.add_argument('--variant',choices=['all',*VARIANTS],default='all')
    args=p.parse_args()
    # Load the established local runtime settings without printing credentials.
    runtime=Path('/tmp/touchdown-generation-runtime.env')
    if runtime.exists():
        for line in runtime.read_text().splitlines():
            key,value=line.split('=',1)
            if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    base=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    variants=VARIANTS if args.variant=='all' else [args.variant]
    for strategy in variants:
        os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=base
        print(f'START {args.stage} {strategy}',flush=True)
        run_variant(args.source.resolve(),args.root.resolve()/strategy,strategy,args.stage)
        print(f'COMPLETE {args.stage} {strategy}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
