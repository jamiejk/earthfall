"""One seeded photographic control against a saved split-flight experiment.

Reuses the source scene, native portrait and seed. All new files go to a separate
root; source paths and hashes remain explicit. Run with PYTHONPATH=scripts:src.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from touchdown import split_flight as flight
from touchdown.authored_scene import digest
from touchdown.photo_gate import require_arrival_gate
from touchdown.photo_only import asset_plan, verify_lineage


CONTROL = dict(parent_layout=False, high_motion_frames=0, seam_carry_frames=0,
               seam_bridge_frames=0, high_strength=.5, anchor_delivery_frame=120,
               anchor_strength=.5)
PRESET_PATH = Path(__file__).with_name('presets') / 'photographic-flight-v1.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_root(source, root):
    source, root = source.resolve(), root.resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError('Source and experiment root must be separate, non-nested directories')


def configuration(cfg):
    cfg = json.loads(json.dumps(cfg))
    # The approved settings are a versioned snapshot, independent of live defaults.
    preset = read(PRESET_PATH)
    missing = set(flight.DEFAULTS) - preset['split'].keys()
    if missing:
        raise ValueError(f'Frozen preset lacks new split settings: {sorted(missing)}')
    cfg.setdefault('video', {})['split'] = dict(preset['split'])
    if os.environ.get('TOUCHDOWN_SPLIT'):
        raise ValueError('Unset TOUCHDOWN_SPLIT: this control has explicit saved settings')
    flight.options(cfg)
    return cfg


def source_files(source, cfg, near):
    files = [source / 'scene-config.json', source / 'motion/near/result.json',
             source / 'motion/near/provenance.json', Path(near['video']), Path(near['workflow']),
             Path(cfg['subject']), Path(cfg['gate']['arrival_report'])]
    files += [Path(r['local_path']) for r in cfg['references']]
    if cfg.get('regional_context'):
        files.append(Path(cfg['regional_context']))
    for spec in asset_plan(cfg, source / 'scene'):
        files += [source / 'scene' / (spec['name'] + suffix)
                  for suffix in ('.png', '.json', '.workflow.json')]
    baseline = source / 'motion/earth_zoom.mp4'
    if baseline.is_file():
        files += [baseline, source / 'motion/pacing.json', source / 'motion/flight/provenance.json']
    return sorted(set(p.resolve() for p in files))


def verify_snapshot(snapshot):
    for path, expected in snapshot.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Reused source changed: {path}')


def prepare(source, root):
    check_root(source, root)
    original = read(source / 'scene-config.json')
    cfg = configuration(original)
    near = flight._checked(read(source / 'motion/near/result.json'))
    ancestry = verify_lineage(original, source / 'scene', asset_plan(original, source / 'scene'))
    if len(ancestry['generated']) != len(asset_plan(original, source / 'scene')):
        raise ValueError('The saved photographic scene is incomplete')
    if read(source / 'motion/near/provenance.json')['parent']['arrival_sha256'] != digest(source / 'scene/arrival.png'):
        raise ValueError('The native portrait belongs to a different arrival')
    snapshot = {str(p): digest(p) for p in source_files(source, original, near)}
    manifest_path = root / 'experiment.json'
    if manifest_path.exists():
        existing = read(manifest_path)
        if existing['source'] != str(source) or existing['source_hashes'] != snapshot or existing['config'] != cfg:
            raise ValueError('Existing experiment differs; choose a fresh root')
        verify_snapshot(existing['source_hashes'])
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError('Experiment root must be empty or contain this experiment manifest')
    root.mkdir(parents=True, exist_ok=True)
    require_arrival_gate(cfg, source / 'scene', root / 'motion')
    manifest = dict(source=str(source), root=str(root), config=cfg, source_hashes=snapshot,
                    preset=dict(path=str(PRESET_PATH.resolve()), sha256=digest(PRESET_PATH), name=read(PRESET_PATH)['name']),
                    seed=cfg['seed'], parent_seed=cfg['seed'] + int(cfg['video'].get('flight_seed_offset', 32)),
                    descent_seed=cfg['seed'] + int(cfg['video'].get('flight_seed_offset', 32)) + 1,
                    native_portrait=near, baseline_options=read(source / 'motion/flight/provenance.json')['options'],
                    control_options=flight.options(cfg), comparison='Earlier photographic treatment; same source assets and seed',
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    code_hashes={str(p): digest(p) for p in [Path(__file__), Path(flight.__file__)]})
    write(manifest_path, manifest)
    write(root / 'control-config.json', cfg)
    write(root / 'status.json', dict(stage='prepared', status='ready', publication='held for motion review'))
    return manifest


def summary(source):
    pacing = read(source / 'motion/pacing.json')
    zoom = {k: v for k, v in pacing.get('zoom', {}).items() if k != 'log_scale'}
    return dict(video=str(source / 'motion/earth_zoom.mp4'), sha256=pacing['output_sha256'],
                duration_seconds=pacing['output_frames'] / pacing['fps'],
                native_start_seconds=pacing['native_start_frame'] / pacing['fps'],
                native_frames=pacing['native_frames'], dropped_hold_frames=len(pacing['dropped_hold_frames']),
                generated_cuts=pacing['generated_cuts'], motion=pacing['motion'], zoom=zoom,
                assembly_crossfade=pacing['crossfade'], generated_transition_review='pending visual review')


def run(source, root):
    manifest = prepare(source, root)
    cfg, scene, output, near = manifest['config'], source / 'scene', root / 'motion', source / 'motion/near'
    # ComfyUI starts in its own directory; relative PYTHONPATH entries break the
    # pinned relay node's import of touchdown.relay_timing there.
    repo = Path(__file__).resolve().parents[2]
    os.environ['PYTHONPATH'] = os.pathsep.join(str(p) for p in (repo / 'scripts', repo / 'src'))
    attempts_path = root / 'run-attempts.json'
    attempts = read(attempts_path) if attempts_path.exists() else []
    attempts.append(dict(started_at=time.time(), code_hashes={str(p): digest(p)
                    for p in (Path(__file__), Path(flight.__file__))}))
    write(attempts_path, attempts)
    # Import after runtime configuration has been supplied by the caller.
    from touchdown import studio_build as studio
    from touchdown.studio_brief import approach_view_review
    stage = 'runtime'
    started = time.monotonic()
    try:
        relay = flight.enable_prompt_relay()
        write(root / 'relay-upstream.json', relay)
        for stage in ('parent', 'anchor', 'descent'):
            verify_snapshot(manifest['source_hashes'])
            write(root / 'status.json', dict(stage=stage, status='running', publication='held for motion review'))
            print(f'START {stage}', flush=True)
            if stage == 'parent':
                binding_path = output / 'flight/anchor-binding.json'
                if not binding_path.exists():
                    with studio.runtime('comfy', root):
                        flight.parent(cfg, scene, output, studio.VIDEO_URL, studio.COMFY_URL, near)
                binding = read(binding_path)
                flight._checked(binding['parent'])
                for name in ('far', 'composition', 'anchor'):
                    if digest(output / f'flight/{name}.png') != binding[f'{name}_sha256']:
                        raise ValueError(f'Changed experiment {name}')
            elif stage == 'anchor':
                anchor = output / 'flight/anchor.png'
                review_path = output / 'flight/anchor-review.json'
                review = read(review_path) if review_path.exists() else None
                if review is None:
                    with studio.runtime('vision', root):
                        review = approach_view_review(studio.VISION_URL, anchor, root / 'gates/scales/anchor.json')
                    write(review_path, dict(review, sha256=digest(anchor)))
                if review.get('status') != 'pass' or review.get('sha256', digest(anchor)) != digest(anchor):
                    raise studio.Held('Control anchor did not pass its aerial review')
            elif not (output / 'pacing.json').exists():
                with studio.runtime('comfy', root):
                    flight.descent(cfg, scene, output, studio.VIDEO_URL, near)
            print(f'COMPLETE {stage}', flush=True)
        verify_snapshot(manifest['source_hashes'])
        comparison = dict(baseline=summary(source), control=summary(root),
                          source_unchanged=True, motion_quality_verdict='pending visual review',
                          elapsed_seconds=round(time.monotonic() - started, 1))
        write(root / 'comparison.json', comparison)
        write(root / 'status.json', dict(stage='comparison', status='rendered', publication='held for motion review'))
        print(json.dumps(comparison, indent=2), flush=True)
    except (Exception, SystemExit) as exc:
        write(root / 'status.json', dict(stage=stage, status='held', error=f'{type(exc).__name__}: {exc}',
                                        elapsed_seconds=round(time.monotonic() - started, 1),
                                        publication='held; completed stages retained'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    if args.prepare_only:
        manifest = prepare(source, root)
        print(json.dumps({k: manifest[k] for k in ('source', 'root', 'seed', 'parent_seed', 'descent_seed', 'control_options')}, indent=2))
    else:
        run(source, root)


if __name__ == '__main__':
    main()
