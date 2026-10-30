"""Isolated replay of the archived accepted photographic pipeline, not live defaults."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys

REPO=Path(__file__).resolve().parents[2]
ARCHIVE=REPO/'outputs/touchdown/baselines/pena-photographic-v1-20261026'
SOURCE=REPO/'outputs/touchdown/studio/studio-chambord-b12-20261025'
ROOT=REPO/'outputs/touchdown/studio/studio-chambord-photographic-recipe-v1-20261027'


def environment():
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)


def prepare(root):
    from touchdown.flight_regression import read,write
    from touchdown.authored_scene import digest
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh root required')
    root.mkdir(parents=True)
    code=root/'frozen-code';shutil.copytree(ARCHIVE/'code',code)
    # Runtime launcher is infrastructure, not part of the archived inference recipe.
    shutil.copy2(REPO/'scripts/start_touchdown_vision_checker.sh',code/'scripts/start_touchdown_vision_checker.sh')
    relay=code/'outputs/touchdown/relay-runtime/custom_nodes/TouchdownPromptRelay';relay.mkdir(parents=True)
    original=REPO/'outputs/touchdown/relay-runtime/custom_nodes/TouchdownPromptRelay'
    for name in ['patches.py','prompt_relay.py','upstream.json']:shutil.copy2(original/name,relay/name)
    saved=read(SOURCE/'motion/flight/provenance.json')['options']
    preset=read(code/'scripts/touchdown/presets/photographic-flight-v1.json')
    common={k:saved[k] for k in ['parent_handles','opening_frames','anchor_delivery_frame','anchor_strength','high_delivery_frame','high_strength','descent_frames','prefix_frames','width','height']}
    assert all(preset['split'][k]==v for k,v in common.items())
    graph_records=[]
    for name,base in [('chambord',SOURCE),('pena',REPO/'outputs/touchdown/studio/studio-pena-control-20261026')]:
        for stage in ['parent','descent']:
            record=read(base/f'motion/flight/{stage}/result.json');p=Path(record['workflow']);g=read(p)
            graph_records.append(dict(case=name,stage=stage,path=str(p),sha256=digest(p),
                models={k:n['inputs'] for k,n in g.items() if n.get('class_type') in ['UNETLoader','CLIPLoader','VAELoader']},
                relay_nodes=[k for k,n in g.items() if n.get('class_type')=='TouchdownPromptRelay'],
                layout_nodes=[k for k,n in g.items() if n.get('class_type') in ['LTXICLoRALoaderModelOnly','LTXAddVideoICLoRAGuide']]))
    assert all(g['relay_nodes'] and not g['layout_nodes'] for g in graph_records)
    files=list(p for p in code.rglob('*') if p.is_file())+[Path(__file__).resolve()]
    write(root/'recipe.json',dict(name='chambord-pena-photographic-recipe-v1',source=str(SOURCE),
        code_origin=str(ARCHIVE/'code'),common_settings=common,full_split_settings=preset['split'],historical_graphs=graph_records,
        reused=['arrival/high/orbit photographs','native moving portrait','subject/place references','scene seed'],
        fresh=['parent LTX generation','FLUX approach-anchor repair','descent LTX generation','historical assembly'],
        scope='Flight-stage reproduction; not fresh scene/portrait generation or arbitrary-location reliability',
        timing='Historical assembly and hold trimming retained explicitly; not strict 8s/30000ft compliance',
        inputs_and_code={str(p):digest(p) for p in files},promotion=False))
    write(root/'status.json',dict(status='prepared',app_promoted=False))
    print('Prepared frozen code and shared historical recipe',flush=True)


def worker(root):
    environment()
    code=root/'frozen-code'
    sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    os.environ['PYTHONPATH']=os.pathsep.join([str(code/'scripts'),str(code/'src')])
    # Import exclusively from the archived code after putting it first.
    from touchdown import flight_regression,studio_build,relay_experiment
    assert Path(flight_regression.__file__).is_relative_to(code)
    assert Path(studio_build.__file__).is_relative_to(code)
    # Share the live workspace GPU lock and unchanged local vision launcher.
    studio_build.REPO=REPO
    # Include pinned weight paths and deterministic execution; the inference code stays archived.
    config=root/'pinned-model-paths.json'
    original=REPO/'outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026/pinned-model-paths.json'
    shutil.copy2(original,config)
    import shlex
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--extra-model-paths-config' in command:raise ValueError('Unexpected runtime model override')
    if '--deterministic' not in command.split():command+=' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --extra-model-paths-config '+shlex.quote(str(config))
    from touchdown.authored_scene import digest
    from touchdown.flight_regression import read,write,verify_snapshot
    verify_snapshot(read(root/'recipe.json')['inputs_and_code'])
    write(root/'execution.json',dict(command=shlex.split(os.environ['TOUCHDOWN_COMFY_SERVER_CMD']),
        cwd=os.environ['TOUCHDOWN_COMFY_SERVER_CWD'],code=str(code),
        orchestration_only_change='studio_build.REPO uses live workspace for shared GPU lock and vision launcher',
        deterministic=True,recipe_sha256=digest(root/'recipe.json')))
    flight_regression.run(SOURCE,root/'run')
    verify_snapshot(read(root/'recipe.json')['inputs_and_code'])
    write(root/'status.json',dict(status='rendered_awaiting_visual_review',app_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();(prepare if a.stage=='prepare' else worker)(a.root.resolve())
