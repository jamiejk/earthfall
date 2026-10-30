"""Isolated execution of canonical stages; frozen inference imports stay isolated."""
import argparse,json,os,shutil,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
CANON=REPO/'workflows/touchdown/canonical/photographic-grounded-v1'
def read(p):return json.loads(p.read_text())
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2)+'\n')

def prepare(root):
    from touchdown import studio_build as s
    from touchdown.canonical_scene import scene_config
    s.REPO=REPO
    import shlex
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--extra-model-paths-config' in shlex.split(command):raise ValueError('Unexpected preparation model override')
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --deterministic --extra-model-paths-config '+shlex.quote(str(root/'flight/pinned-model-paths.json'))
    setup=read(root/'setup.json');base=root/'prepared';folder=root/'inputs';scene=base/'scene';scene.mkdir(parents=True)
    output=base/'motion'
    for item in [setup['subject'],*setup['references']]:
        if Path(item['name']).name!=item['name'] or s.digest(folder/item['name'])!=item['sha256']:raise ValueError('Input changed')
    with s.runtime('vision',base):
        brief=s.grounding(setup,folder,s.VISION_URL,base/'grounding.json')
        try:facts=s.region_facts(setup,s.VISION_URL,base/'region-facts.json')
        except OSError:facts=None
        if facts:brief.update(regional_relief=facts['relief'],local_relief=facts['local_relief'],nearest_settlement=facts['nearest_settlement'],coastal=facts.get('sea_fraction',0)>=.03,local_coastal=facts.get('local_sea_fraction',0)>=.03,local_features=facts.get('local_features'),local_cover=facts.get('local_cover'))
    if len(brief['selected_references'])<2:raise s.Held('Fewer than two useful location references.')
    cfg=scene_config(setup,brief,folder,base,2081561474)
    cfg['video']['split']=read(CANON/'flight-preset.json')['split']
    if cfg['video'].get('phase_contract') or '30,000' in cfg['prompts']['high']:raise ValueError('Strict phase regression is forbidden')
    write(base/'scene-config.json',cfg)
    write(base/'attribution.json',cfg['references'])
    for attempt in range(2):
        cfg['seed']=2081561474+attempt*100
        with s.runtime('comfy',base):s.generate_asset(s.COMFY_URL,s.asset_plan(cfg,scene)[0],scene)
        with s.runtime('vision',base) as metadata:
            gate=base/f'gates/arrival-{attempt}'
            evidence=s.vision.prepare(scene/'arrival.png',[0.0],gate/'evidence',subject=folder/setup['subject']['name'])
            report=s.vision.check(evidence,cfg['gate']['description'],cfg['gate']['checks'],s.MODEL,s.VISION_URL,gate/'report.json',metadata=metadata,allowed_supports=cfg['gate']['allowed_supports'])
        if report['status']=='pass' and not report.get('error'):
            cfg['gate']['arrival_report']=str(gate/'report.json');break
    else:raise s.Held('Arrival did not pass the reference and foot-contact gate.')
    write(base/'scene-config.json',cfg)
    with s.runtime('comfy',base):
        plan=s.asset_plan(cfg,scene)
        for i,spec in enumerate(plan[1:],1):
            s.verify_lineage(cfg,scene,plan[:i]);s.generate_asset(s.COMFY_URL,spec,scene)
        s.verify_lineage(cfg,scene,plan)
        s.portrait_flight.run(cfg,scene,output,'near',s.VIDEO_URL)
    write(base/'preparation.json',dict(recipe='photographic-grounded-v1',seed=cfg['seed'],arrival_gate=cfg['gate']['arrival_report'],input_hashes={p['name']:p['sha256'] for p in [setup['subject'],*setup['references']]},source_scope='Fresh upstream photographs and native portrait; canonical control reused these inputs'))


def protect_person(f):
    """Keep the frozen hold drop off the frames that show the person moving (the file itself is left as frozen).

    The descent's last frames before the cut to the native portrait were guided by native[0..n-1] played in step, so
    she is already moving there; the frozen collapse_holds drops slow-camera frames right up to the native portrait
    and played that motion at 2-7x (Longleat descent steps 1,1,2,3,7,4,3,2,2,2,2), then normal speed after the cut.
    Holds are now dropped only before those guided frames; the step size still comes from the same median."""
    ending_frame,guided=f.ending_frame,{}
    def capture(ratios,stop,first,*a,**k):
        guided['n']=first;return ending_frame(ratios,stop,first,*a,**k)
    def collapse(diffs,end,floor=f.HOLD_FLOOR,max_drop=f.HOLD_MAX_DROP):
        total=len(diffs)+1;part=diffs[:max(0,end-1)]
        if len(part)<f.FPS:return list(range(total))
        import numpy as np;step=floor*float(np.median(part))
        stop=end-guided.get('n',0)
        keep,moved,dropped=[0],0.,0
        for i in range(1,total):
            moved+=diffs[i-1]
            if i>=stop-1 or moved>=step or dropped>=max_drop:keep.append(i);moved,dropped=0.,0
            else:dropped+=1
        return keep
    f.ending_frame,f.collapse_holds=capture,collapse


def flight(root):
    code=root/'flight/frozen-code';sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    os.environ['PYTHONPATH']=os.pathsep.join(sys.path[:2])
    from touchdown import split_flight as f,studio_build as s
    protect_person(f)
    from touchdown.studio_brief import approach_view_review
    from touchdown.photo_only import asset_plan,verify_lineage
    assert Path(f.__file__).is_relative_to(code)
    s.REPO=REPO
    source=root/'prepared';cfg=read(source/'scene-config.json');scene=source/'scene';near=source/'motion/near'
    cfg['video']['split']=read(CANON/'flight-preset.json')['split']
    expected={**f.DEFAULTS,**cfg['video']['split']}
    expected.setdefault('end_face_ratio',f.FACE_STOP.get(cfg['video'].get('framing')))
    if f.options(cfg)!=expected:raise ValueError('Frozen flight options differ')
    import shlex
    command=os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--extra-model-paths-config' in shlex.split(command):raise ValueError('Unexpected model override')
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']=command+' --deterministic --extra-model-paths-config '+shlex.quote(str(root/'flight/pinned-model-paths.json'))
    verify_lineage(cfg,scene,asset_plan(cfg,scene));f._checked(read(near/'result.json'))
    output=root/'flight/run/motion';output.mkdir(parents=True)
    files=[p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    frozen={str(p):s.digest(p) for p in files}
    write(root/'flight/run/experiment.json',dict(config=cfg,source_hashes=frozen,source=str(source),inference_code=str(code),preset_sha256=s.digest(CANON/'flight-preset.json')))
    f.enable_prompt_relay()
    with s.runtime('comfy',root/'flight'):f.parent(cfg,scene,output,s.VIDEO_URL,s.COMFY_URL,near)
    with s.runtime('vision',root/'flight'):
        review=approach_view_review(s.VISION_URL,output/'flight/anchor.png',root/'flight/gates/anchor.json')
    write(output/'flight/anchor-review.json',dict(review,sha256=s.digest(output/'flight/anchor.png')))
    if review['status']!='pass':raise s.Held('The repaired aerial anchor did not pass Qwen review.')
    with s.runtime('comfy',root/'flight'):f.descent(cfg,scene,output,s.VIDEO_URL,near)
    for path,expected in frozen.items():
        if s.digest(Path(path))!=expected:raise ValueError('Frozen source changed: '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.stage=='prepare':prepare(root)
    elif a.stage=='flight':flight(root)
    elif a.stage=='layout-normalize':
        from touchdown.canonical_texture import normalise
        normalise(root)
    elif a.stage=='texture':
        from touchdown.canonical_texture import generate
        generate(root)
    elif a.stage.startswith('generic-opening-'):
        # Loaded by path: the worker imports the job's frozen touchdown package, which must not be shadowed.
        import importlib.util
        spec=importlib.util.spec_from_file_location('generic_opening',REPO/'scripts/touchdown/generic_opening.py')
        g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
        getattr(g,a.stage.removeprefix('generic-opening-'))(root)
    elif a.stage.startswith('expand-'):
        from touchdown.canonical_fill import expand_one
        expand_one(root,int(a.stage[-1]))
    else:
        # Imported only in a fresh subprocess, before any inference package is loaded.
        import importlib.util
        spec=importlib.util.spec_from_file_location('recorded_grounded_stage',REPO/'scripts/touchdown/scene_grounded_workflow.py')
        g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
        g.TIMING=CANON/'opening-timing.json'
        if a.stage in ('texture','orbit'):g.still(root,a.stage)
        else:getattr(g,a.stage.replace('-','_'))(root)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        if type(exc).__name__=='Held':
            write(Path(sys.argv[sys.argv.index('--root')+1])/'stage-held.json',{'reason':str(exc)})
            raise SystemExit(3)
        raise
