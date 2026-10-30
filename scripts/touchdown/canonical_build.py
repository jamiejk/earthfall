"""App integration of the canonical photographic recipes, with recorded local vision checkpoints.

The accepted inference helpers remain unchanged. This driver prepares fresh inputs, runs the frozen flight and the
opening, and records bound vision decisions. Create runs photographic-generic-orbit-v1: a cached generic orbit and
cumulus dive joined inside the flight's own cloud (generic_opening.py). TOUCHDOWN_CANONICAL_RECIPE=
photographic-grounded-v1 selects the earlier destination-specific opening (layout, texture, expand, project, orbit).
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
CANON=REPO/'workflows/touchdown/canonical/photographic-grounded-v1'
# Frozen inference code and Prompt Relay node, vendored from the accepted run; initialise() checks them against recipe.json.
ARCHIVE=CANON/'frozen-code'
RELAY=CANON/'relay-runtime/TouchdownPromptRelay'
LEGACY_MODEL_PATHS=REPO/'outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026/pinned-model-paths.json'
GROUNDED,GENERIC='photographic-grounded-v1','photographic-generic-orbit-v1'
RECIPES={GROUNDED:('references','flight','layout','texture','expand','project','orbit','opening','review'),
         GENERIC:('references','flight','opening','review')}
RECIPE=os.environ.get('TOUCHDOWN_CANONICAL_RECIPE',GENERIC)

def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def model_paths():
    """ComfyUI extra model paths for every canonical ComfyUI start, or None to use ComfyUI's own model folders.

    TOUCHDOWN_MODEL_PATHS names an extra_model_paths file; without it the development machine's recorded pin is used if present.
    """
    configured=os.environ.get('TOUCHDOWN_MODEL_PATHS')
    if configured:
        if not Path(configured).is_file():raise ValueError('TOUCHDOWN_MODEL_PATHS is not a file: '+configured)
        return Path(configured)
    return LEGACY_MODEL_PATHS if LEGACY_MODEL_PATHS.is_file() else None


def check(control=False):
    from touchdown.studio_build import preflight
    from touchdown.scene_grounded_workflow import FILL_CHECKS,model_checks
    preflight()
    for path in [ARCHIVE/'scripts/touchdown/split_flight.py',CANON/'flight-preset.json',CANON/'opening-timing.json',REPO/'assets/models/face_detection_yunet_2023mar.onnx',*(RELAY/n for n in ('patches.py','prompt_relay.py','upstream.json'))]:
        if not path.is_file():raise ValueError('Missing canonical dependency: '+str(path))
    if RECIPE==GENERIC:
        for name in ('generic-orbit.png','cumulus.png','limb-fit.json'):
            if not (REPO/'assets/touchdown/opening'/name).is_file():raise ValueError('Missing cached opening still: '+name)
    model_paths()
    if control:
        # The accepted run's media lives only on the development machine; --check-control compares it too.
        for name,expected in read(CANON/'recipe.json')['files'].items():
            if sha(REPO/name)!=expected:raise ValueError('Changed canonical evidence: '+name)
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        Build(Path(temporary)).initialise()  # the frozen code a job would get, checked against the accepted control
        model_checks(Path(temporary))
    if read(ARCHIVE/'scripts/touchdown/presets/photographic-flight-v1.json')['split']!=read(CANON/'flight-preset.json')['split']:raise ValueError('Archived flight preset differs')
    print('Canonical grounded workflow preflight passed; automatic Qwen checkpoints required.',flush=True)


class Held(Exception):pass

class Build:
    def __init__(self,root,recipe=None):
        self.recipe=recipe or RECIPE
        if self.recipe not in RECIPES:raise ValueError('Unknown canonical recipe: '+self.recipe)
        self.root=root;self.states={s:'pending' for s in RECIPES[self.recipe]};self.reviews={}
    def progress(self,stage,message):
        for key in self.states:
            if self.states[key]=='running' and key!=stage:self.states[key]='done'
        self.states[stage]='running';write(self.root/'progress.json',dict(stages=self.states,message=message))
    def checkpoint(self,name,title,description,paths,*,reject_raises=True):
        assets=[]
        for path in paths:
            path=path.resolve()
            if not path.is_relative_to(self.root):raise ValueError('Review evidence outside job')
            # Names are API-safe and each checkpoint has its own folder.
            assets.append(dict(name=path.name.replace('_','-'),path=str(path.relative_to(self.root)),sha256=sha(path)))
        request=dict(id=name,title=title,description=description,assets=assets)
        write(self.root/'checkpoint.json',request)
        from touchdown.canonical_review import review
        try:reviewed=review(self.root,name,assets,description)
        except (ValueError,TimeoutError) as exc:
            raise Held(f'{title}: automatic evidence was incomplete or invalid: {exc}') from exc
        decision=dict(continue_=reviewed.pop('continue_'),**reviewed)
        decision['continue']=decision.pop('continue_')
        decision['reviewer']='qwen'
        # Bind the decision to exactly the public evidence uploaded by the worker.
        frozen={k:request[k] for k in ('id','assets','title','description')}
        frozen['assets']=[{k:v for k,v in a.items() if k!='path'} for a in assets]
        decision['evidence_sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
        write(self.root/'automatic-reviews'/name/'decision.json',decision)
        deadline=time.monotonic()+120
        decision_file=self.root/'decisions'/f'{name}.json'
        while not decision_file.exists():
            if time.monotonic()>deadline:raise Held(f'{title}: could not save the automatic review to the app.')
            time.sleep(.5)
        saved=read(decision_file)
        if saved.get('evidence_sha256')!=decision['evidence_sha256']:raise ValueError('Saved review does not match checkpoint')
        if any(sha(self.root/a['path'])!=a['sha256'] for a in assets):raise ValueError('Review evidence changed')
        self.reviews[name]={k:v for k,v in saved.items() if k!='review_report'}
        if saved.get('continue') is not True and reject_raises:raise Held(saved.get('reason','Review held'))
        return saved
    def stage(self,name):
        command=[sys.executable,str(REPO/'scripts/touchdown/canonical_stage.py'),name,'--root',str(self.root)]
        env={**os.environ,'PYTHONPATH':os.pathsep.join([str(REPO/'scripts'),str(REPO/'src')])}
        process=subprocess.run(command,env=env,cwd=REPO)
        if process.returncode==3:
            held=self.root/'stage-held.json'
            raise Held(read(held)['reason'] if held.exists() else f'{name} needs review')
        process.check_returncode()
    def initialise(self):
        root=self.root
        if (root/'flight').exists():raise ValueError('Use a new job for a new canonical run')
        code=root/'flight/frozen-code';shutil.copytree(ARCHIVE,code)
        shutil.copy2(REPO/'scripts/start_touchdown_vision_checker.sh',code/'scripts/start_touchdown_vision_checker.sh')
        relay=code/'outputs/touchdown/relay-runtime/custom_nodes/TouchdownPromptRelay';relay.mkdir(parents=True,exist_ok=True)
        for name in ('patches.py','prompt_relay.py','upstream.json'):shutil.copy2(RELAY/name,relay/name)
        detector=code/'assets/models/face_detection_yunet_2023mar.onnx';detector.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/'assets/models'/detector.name,detector)
        pinned=model_paths()
        if pinned:shutil.copy2(pinned,root/'flight/pinned-model-paths.json')
        else:write(root/'flight/pinned-model-paths.json',{})  # an empty extra-paths file: ComfyUI's own model folders
        control=read(CANON/'recipe.json')
        prefix=control['accepted_run']+'/flight/frozen-code/'
        for name,expected in control['files'].items():
            if name.startswith(prefix):
                cloned=code/name[len(prefix):]
                if not cloned.is_file() or sha(cloned)!=expected:
                    raise ValueError('Cloned inference differs from accepted control: '+name)
        write(root/'integration.json',dict(recipe=self.recipe,control_manifest_sha256=sha(CANON/'recipe.json'),driver_sha256=sha(Path(__file__)),fresh_upstream=True,layout_author="qwen",reviewer="qwen",review_required=True,frozen_code={str(p.relative_to(root)):sha(p) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts}))
    def run(self):
        root=self.root;self.initialise()
        self.progress('references','Preparing the location photos, the landing scene and your moving portrait')
        self.stage('prepare')
        cfg=read(root/'prepared/scene-config.json')
        write(root/'selection.json',dict(selected=dict(source=str(root/'prepared'),place=cfg['place']),selection='This saved app setup; no landmark substitution'))
        self.progress('flight','Creating the Earth flight')
        self.stage('flight')
        flight=root/'flight/run/motion/earth_zoom.mp4'
        self.checkpoint('flight','Review the fresh flight','Check the cloud passage, readable landmark approach and connected landing before building its Earth view.',[flight])
        if self.recipe==GENERIC:return self.generic_opening()
        self.progress('layout','Sketching the regional layout from the flight')
        decision=self.checkpoint('layout','Build the regional layout','Select a clear frame below the clouds and derive the broad land/water plan. Locate the destination and describe how its surroundings should continue. This is approximate authored geography.',[flight])
        layout=decision['layout']
        import cv2
        cap=cv2.VideoCapture(str(flight));count=int(cap.get(7));cap.release()
        if layout['cloud_exit_frame']>=count:raise Held('The selected cloud-exit frame is outside this flight')
        write(root/'scene-layout.json',layout);self.stage('layout-normalize');self.stage('terrain-prepare')
        self.progress('texture','Creating the overhead texture')
        self.stage('texture')
        self.checkpoint('texture','Review the overhead texture','Check straight-down geometry, land/water continuity and natural texture; reject oblique scenery or a giant landmark.',[root/'terrain/texture.png',root/'terrain/regional-layout.png',root/'terrain/cloud-exit.png'])
        self.progress('expand','Widening the overhead view in three passes')
        for index in range(1,4):
            name=f'expand-{index}'
            self.stage(name)
            self.checkpoint(name,f'Check outpaint {index} of 3','Check continuous land/water shapes, destination ancestry, invented features and text. Pass only if usable for the next expansion.',[root/f'terrain/{name}/raw-outpaint.png'])
        self.progress('project','Wrapping the texture onto the curved Earth')
        self.stage('project')
        self.checkpoint('project','Review the curved Earth guide','Check horizon, texture coverage and the marked destination ancestry. This is an artistic projection.',[root/'terrain/globe-guide.png',root/'terrain/globe-target-diagnostic.png'])
        self.progress('orbit','Finishing the view from orbit')
        self.stage('orbit')
        self.checkpoint('orbit','Review the orbital photograph','Check coastline, horizon and relative land/water sizes; reject visible text, CGI, giant landmarks or geography hidden by clouds.',[root/'terrain/orbit.png'])
        self.progress('opening','Generating the opening from orbit and joining it inside the cloud')
        self.stage('opening-prepare');self.stage('opening-worker');self.stage('assemble')
        final=root/'complete-flight.mp4'
        self.progress('review','Checking the finished film')
        self.checkpoint('final','Review the complete Earthfall','Check the advancing viewpoint, horizon leaving the frame, cloud join at frames 39–40, geography, landmark approach and the person. Approve only if the full film is usable.',[final])
        return self.finish()
    def generic_opening(self):
        root=self.root
        self.progress('opening','Approaching Earth from orbit and diving into the clouds')
        self.stage('generic-opening-prepare');self.stage('generic-opening-worker');self.stage('generic-opening-assemble')
        final=root/'complete-flight.mp4'
        self.progress('review','Checking the finished film')
        self.checkpoint('final','Review the complete Earthfall','The opening is a generic Earth from orbit, not the destination: check that the camera keeps moving in, dives into cloud and joins the flight inside the cloud at frames 39–40, then the landmark approach and the person. Approve only if the full film is usable.',[final])
        return self.finish()
    def finish(self):
        root=self.root;final=root/'complete-flight.mp4'
        assembly=read(root/'assembly.json')
        if sha(final)!=assembly['sha256']:raise ValueError('Delivered film changed after assembly')
        self.states={s:'done' for s in self.states}
        write(root/'progress.json',dict(stages=self.states,message='Canonical flight accepted'))
        return dict(status='done',checks_passed=True,recipe=self.recipe,video=final.name,video_sha256=sha(final),message='Reviewed canonical flight ready.',assembly=assembly,reviews=self.reviews,integration=read(root/'integration.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--check-control',action='store_true',help='also hash the accepted Bamburgh run (development machine only)');p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.check or a.check_control:check(a.check_control);return
    if not a.input or not a.output:p.error('--input and --output required')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.input.resolve()!=root/'setup.json':shutil.copy2(a.input,root/'setup.json')
    build=Build(root)
    try:outcome=build.run()
    except Held as exc:outcome=dict(status='held',recipe=build.recipe,checks_passed=False,error=str(exc),message=str(exc))
    except Exception as exc:outcome=dict(status='failed',recipe=build.recipe,checks_passed=False,error=f'{type(exc).__name__}: {exc}',message='Canonical generation stopped. Completed stages are preserved.')
    outcome.setdefault('reviews',build.reviews)
    final=root/'complete-flight.mp4'
    if outcome['status']=='held' and final.is_file():
        outcome.update(video=final.name,video_sha256=sha(final))
    write(root/'outcome.json',outcome)
    if outcome['status']!='done':raise SystemExit(3 if outcome['status']=='held' else 1)
if __name__=='__main__':main()
