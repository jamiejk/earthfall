"""The Create contract is the canonical chain, with automatic evidence-bound gates."""
import hashlib,json,sys,threading
from pathlib import Path
from contextlib import contextmanager
import pytest
from test_studio import env,save
from worker.client import ServerClient
from touchdown.canonical_build import Build,Held,GENERIC,GROUNDED
from server.checkpoints import IDS,validate_layout


def approve_checkpoints(worker,job_id,root,video=b'synthetic'):
    root.mkdir(parents=True,exist_ok=True);(root/'fixture.mp4').write_bytes(video)
    for name in IDS:
        payload=dict(id=name,title='Synthetic '+name,description='Explicit synthetic transport fixture',assets=[dict(name='fixture.mp4',path='fixture.mp4',sha256=hashlib.sha256(video).hexdigest())])
        public=worker.checkpoint(job_id,payload,root)
        report=root/f'{name}-report.json';report.write_text(json.dumps({'answer':{'status':'pass'}}))
        data=dict(continue_=True,reason='Explicit synthetic pass',record=report.name,record_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_sha256=public['evidence_sha256'])
        data['continue']=data.pop('continue_')
        if name=='layout':data['layout']=dict(cloud_exit_frame=64,target_uv=[.5,.5],colours={'background':'#697b50'},features=[],layout_description='Synthetic connected mainland',expansion_description='Continue synthetic connected mainland')
        worker.checkpoint_review(job_id,name,data,root)


def queued(env):
    client,settings,app=env;record=save(client,references=2).json()  # a fresh clone has no installed example references
    worker=ServerClient('http://testserver','worker',http=client)
    worker.heartbeat(canonical=True)
    result=client.post(f"/api/studio/setups/{record['id']}/render").json()
    return client,settings,app,worker,result


def test_create_isolated_from_earlier_worker_and_requires_all_gates(env):
    client,settings,app,worker,job=queued(env)
    assert job['workflow']=='photographic-grounded-v1'
    assert worker.claim(photographic=True) is None
    claimed=worker.claim(canonical=True);assert claimed['setup']['id']==job['setup_id']
    worker.http.post(f"/api/worker/jobs/{job['id']}/result",files={'file':('fixture.mp4',b'synthetic','video/mp4')})
    worker.review(job['id'],dict(recipe=job['workflow'],status='done',checks_passed=True,video_sha256=hashlib.sha256(b'synthetic').hexdigest()))
    worker.finish(job['id'],'done')
    assert app.state.jobs.get(job['id'])['status']=='held'


def test_checkpoint_order_immutability_auth_and_rejected_vision(env,tmp_path):
    client,settings,app,worker,job=queued(env);worker.claim(canonical=True)
    root=tmp_path/'evidence';root.mkdir();(root/'frame.jpg').write_bytes(b'synthetic image')
    payload=dict(id='layout',title='Plan',description='Test layout ordering',assets=[dict(name='frame.jpg',path='frame.jpg',sha256=hashlib.sha256(b'synthetic image').hexdigest())])
    with pytest.raises(Exception):worker.checkpoint(job['id'],payload,root)
    payload['id']='flight';public=worker.checkpoint(job['id'],payload,root)
    asset=f"/api/jobs/{job['id']}/checkpoints/flight/assets/frame.jpg"
    assert client.get(asset).content==b'synthetic image'
    assert worker.http.post(f"/api/worker/jobs/{job['id']}/checkpoints/flight/assets/frame.jpg",files={'file':('frame.jpg',b'changed')}).status_code==409
    report=root/'review.json';report.write_text(json.dumps({'answer':{'status':'unclear'}}))
    body=dict(continue_=True,reason='Should never be silently approved',record=report.name,record_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_sha256=public['evidence_sha256'])
    body['continue']=body.pop('continue_')
    with pytest.raises(Exception):worker.checkpoint_review(job['id'],'flight',body,root)
    body['continue']=False
    worker.checkpoint_review(job['id'],'flight',body,root)
    state=client.get(f"/api/jobs/{job['id']}/checkpoint").json()
    assert state['decision']['reviewer']=='qwen' and state['decision']['continue'] is False
    assert not state['can_decide']


def test_generic_orbit_job_needs_only_the_flight_and_final_checks(env,tmp_path):
    client,settings,app,worker,job=queued(env);worker.claim(canonical=True)
    root=tmp_path/'evidence';root.mkdir();video=b'synthetic generic film';(root/'film.mp4').write_bytes(video)
    def publish(name):
        return worker.checkpoint(job['id'],dict(id=name,title=name,description='Generic-orbit ordering',assets=[dict(name='film.mp4',path='film.mp4',sha256=hashlib.sha256(video).hexdigest())]),root)
    def approve(name,public):
        report=root/f'{name}-report.json';report.write_text(json.dumps({'answer':{'status':'pass'}}))
        body=dict(continue_=True,reason='Explicit synthetic pass',record=report.name,record_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_sha256=public['evidence_sha256'])
        body['continue']=body.pop('continue_');worker.checkpoint_review(job['id'],name,body,root)
    with pytest.raises(Exception):publish('final')            # the flight check comes first
    approve('flight',publish('flight'))
    approve('final',publish('final'))                          # no layout, texture, outpaint, projection or orbit check
    worker.http.post(f"/api/worker/jobs/{job['id']}/result",files={'file':('film.mp4',video,'video/mp4')})
    worker.review(job['id'],dict(recipe='photographic-generic-orbit-v1',status='done',checks_passed=True,video_sha256=hashlib.sha256(video).hexdigest()))
    worker.finish(job['id'],'done')
    assert app.state.jobs.get(job['id'])['status']=='done'


def test_a_grounded_job_cannot_skip_its_checks_by_claiming_the_generic_recipe(env,tmp_path):
    client,settings,app,worker,job=queued(env);worker.claim(canonical=True)
    root=tmp_path/'evidence';root.mkdir();video=b'synthetic film';(root/'film.mp4').write_bytes(video)
    def publish(name):
        return worker.checkpoint(job['id'],dict(id=name,title=name,description='Grounded ordering',assets=[dict(name='film.mp4',path='film.mp4',sha256=hashlib.sha256(video).hexdigest())]),root)
    report=root/'report.json';report.write_text(json.dumps({'answer':{'status':'pass'}}))
    public=publish('flight')
    body=dict(continue_=True,reason='Explicit synthetic pass',record=report.name,record_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_sha256=public['evidence_sha256'])
    body['continue']=body.pop('continue_');worker.checkpoint_review(job['id'],'flight',body,root)
    publish('layout')                                          # a grounded-only check, published but not approved
    with pytest.raises(Exception):publish('final')


def test_layout_schema_rejects_bad_coordinates_and_restored_strict_constraints():
    layout=dict(cloud_exit_frame=64,target_uv=[.5,.5],colours={'background':'#697b50'},features=[],layout_description='Connected regional land',expansion_description='Continue the connected countryside')
    assert validate_layout(layout)['measured_geography'] is False
    with pytest.raises(ValueError):validate_layout({**layout,'target_uv':[float('nan'),.5]})
    with pytest.raises(ValueError):validate_layout({**layout,'cloud_exit_frame':-1})
    from touchdown.canonical_scene import scene_config
    brief=dict(selected_references=[0,1],setting='Landmark on a broad lawn',surrounding_landscape='Connected countryside',regional_view='England',appearance='Person in red coat',support='grass',support_description='connected grassy lawn',light='Natural daylight')
    setup=dict(destination={'title':'Landmark'},subject={'name':'subject.jpg'},references=[{'name':'a.jpg'},{'name':'b.jpg'}],ending='wave',framing='full')
    cfg=scene_config(setup,brief,Path('/tmp'),Path('/tmp'),2081561474)
    assert 'phase_contract' not in cfg['video'] and '30,000' not in cfg['prompts']['high']
    assert 'TWO KILOMETRES' in cfg['prompts']['high']


def test_driver_orders_every_canonical_stage_and_stops_on_unclear(tmp_path,monkeypatch):
    import cv2,numpy as np
    root=tmp_path;build=Build(root,GROUNDED);calls=[]
    def initialise():
        (root/'prepared').mkdir();(root/'flight/run/motion').mkdir(parents=True)
        (root/'prepared/scene-config.json').write_text(json.dumps({'place':'Fixture'}))
        (root/'integration.json').write_text('{}')
        video=cv2.VideoWriter(str(root/'flight/run/motion/earth_zoom.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(64,64))
        for _ in range(195):video.write(np.zeros((64,64,3),dtype=np.uint8))
        video.release()
    monkeypatch.setattr(build,'initialise',initialise)
    monkeypatch.setattr(build,'stage',lambda name:calls.append(name))
    def checkpoint(name,*args,**kwargs):
        calls.append('check:'+name)
        if name=='layout':return {'layout':{'cloud_exit_frame':64}}
        if name=='texture':raise Held('unclear geometry')
        return {'continue':True}
    monkeypatch.setattr(build,'checkpoint',checkpoint)
    with pytest.raises(Held,match='unclear geometry'):build.run()
    assert calls==['prepare','flight','check:flight','check:layout','layout-normalize','terrain-prepare','texture','check:texture']


def test_complete_driver_keeps_all_nine_checks_and_original_assembly_clock(tmp_path,monkeypatch):
    import cv2,numpy as np,shutil
    from touchdown.canonical_build import sha
    root=tmp_path;build=Build(root,GROUNDED);calls=[]
    def initialise():
        (root/'prepared').mkdir();(root/'flight/run/motion').mkdir(parents=True)
        (root/'prepared/scene-config.json').write_text(json.dumps({'place':'Fixture'}));(root/'integration.json').write_text('{}')
        path=root/'flight/run/motion/earth_zoom.mp4'
        video=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24,(64,64))
        for _ in range(219):video.write(np.zeros((64,64,3),dtype=np.uint8))
        video.release()
    def stage(name):
        calls.append(name)
        if name=='assemble':
            shutil.copy2(root/'flight/run/motion/earth_zoom.mp4',root/'complete-flight.mp4')
            (root/'assembly.json').write_text(json.dumps({'frames':219,'fps':24,'replacement_frames':40,'sha256':sha(root/'complete-flight.mp4')}))
    def checkpoint(name,*args,**kwargs):
        calls.append('check:'+name);build.reviews[name]={'continue':True}
        return {'continue':True,'layout':{'cloud_exit_frame':64}}
    monkeypatch.setattr(build,'initialise',initialise);monkeypatch.setattr(build,'stage',stage);monkeypatch.setattr(build,'checkpoint',checkpoint)
    outcome=build.run()
    assert outcome['status']=='done' and outcome['recipe']=='photographic-grounded-v1'
    assert tuple(build.reviews)==IDS and outcome['assembly']['frames']==219
    assert calls==['prepare','flight','check:flight','check:layout','layout-normalize','terrain-prepare','texture','check:texture','expand-1','check:expand-1','expand-2','check:expand-2','expand-3','check:expand-3','project','check:project','orbit','check:orbit','opening-prepare','opening-worker','assemble','check:final']


def test_create_defaults_to_the_generic_orbit_recipe_with_four_stages(tmp_path,monkeypatch):
    import shutil
    from touchdown.canonical_build import sha
    root=tmp_path;build=Build(root);calls=[]
    assert build.recipe==GENERIC and tuple(build.states)==('references','flight','opening','review')
    def initialise():
        (root/'prepared').mkdir();(root/'flight/run/motion').mkdir(parents=True)
        (root/'prepared/scene-config.json').write_text(json.dumps({'place':'Fixture'}));(root/'integration.json').write_text('{}')
        (root/'flight/run/motion/earth_zoom.mp4').write_bytes(b'synthetic flight')
    def stage(name):
        calls.append(name)
        if name=='generic-opening-assemble':
            shutil.copy2(root/'flight/run/motion/earth_zoom.mp4',root/'complete-flight.mp4')
            (root/'assembly.json').write_text(json.dumps({'frames':214,'fps':24,'replacement_frames':40,'join_source_frame':32,'sha256':sha(root/'complete-flight.mp4')}))
    def checkpoint(name,*args,**kwargs):
        calls.append('check:'+name);build.reviews[name]={'continue':True}
        return {'continue':True}
    monkeypatch.setattr(build,'initialise',initialise);monkeypatch.setattr(build,'stage',stage);monkeypatch.setattr(build,'checkpoint',checkpoint)
    outcome=build.run()
    assert outcome['status']=='done' and outcome['recipe']==GENERIC and tuple(build.reviews)==('flight','final')
    assert calls==['prepare','flight','check:flight','generic-opening-prepare','generic-opening-worker','generic-opening-assemble','check:final']
    assert json.loads((root/'progress.json').read_text())['stages']=={s:'done' for s in ('references','flight','opening','review')}


def test_fill_graph_prompt_settings_match_canonical_expand():
    # The lifecycle change must not introduce a different Fill recipe.
    import ast
    from touchdown import canonical_fill,scene_grounded_workflow
    original=ast.parse(Path(scene_grounded_workflow.__file__).read_text())
    integrated=ast.parse(Path(canonical_fill.__file__).read_text())
    def assignments(tree):
        return [ast.dump(node,include_attributes=False) for node in ast.walk(tree)
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Subscript) and ast.unparse(t).startswith("graph[") for t in node.targets)]
    assert assignments(original)==assignments(integrated)


def test_real_archive_import_uses_canonical_preset_and_only_archived_defaults(tmp_path):
    import os,subprocess
    from touchdown.canonical_build import ARCHIVE
    if not ARCHIVE.is_dir():pytest.skip('Local canonical archive is not installed')
    build=Build(tmp_path);build.initialise();code=tmp_path/'flight/frozen-code'
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(code/'scripts'),str(code/'src')])}
    probe='''from pathlib import Path
import json,sys
from touchdown import split_flight as f
assert Path(f.__file__).is_relative_to(Path(sys.argv[1]))
preset=json.loads(Path('workflows/touchdown/canonical/photographic-grounded-v1/flight-preset.json').read_text())['split']
for framing in ('full','three-quarter','medium','close'):
 expected={**f.DEFAULTS,**preset,'end_face_ratio':f.FACE_STOP[framing]}
 assert f.options({'video':{'split':preset,'framing':framing}})==expected
assert preset['width']==1280 and preset['height']==736
assert preset['parent_layout'] is False
assert preset['high_motion_frames']==0
'''
    subprocess.run([sys.executable,'-c',probe,str(code)],env=env,cwd=Path(__file__).resolve().parents[1],check=True)


@pytest.mark.parametrize('index',[1,2,3])
def test_real_fill_stage_executes_preparation_submission_and_download(tmp_path,monkeypatch,index):
    """Exercise the actual wrapper with the real Fill graph and a fake Comfy transport."""
    import io
    from PIL import Image
    from touchdown import canonical_fill as stage,studio_build,flux3,flight_runtime
    terrain=tmp_path/'terrain';terrain.mkdir()
    source=terrain/'texture.png' if index==1 else terrain/f'expand-{index-1}/raw-outpaint.png'
    source.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(1920,1088),'tan').save(source)
    (tmp_path/'scene-layout.json').write_text(json.dumps({'target_uv':[.5,.4],'expansion_description':'Continue broad dry desert terrain.'}))
    monkeypatch.setenv('TOUCHDOWN_COMFY_SERVER_CMD','python main.py')
    monkeypatch.setattr(stage,'environment',lambda:None)
    monkeypatch.setattr(stage,'verify',lambda root:None)
    monkeypatch.setattr(stage,'case',lambda root:({}, {'words':{'region':'Synthetic desert'}}))
    monkeypatch.setattr(stage,'model_checks',lambda root:None)
    monkeypatch.setattr(flight_runtime,'capture_runtime',lambda root:None)
    @contextmanager
    def runtime(kind,folder):
        assert kind=='comfy'
        yield
    monkeypatch.setattr(studio_build,'runtime',runtime)
    monkeypatch.setattr(flux3,'_upload',lambda url,path:'synthetic-upload.png')
    submissions=[]
    def call(url,endpoint,payload=None):
        if endpoint=='/prompt':
            submissions.append(payload['prompt'])
            return {'prompt_id':'synthetic-job'}
        assert endpoint=='/history/synthetic-job'
        return {'synthetic-job':{'status':{'completed':True},'outputs':{key:{'images':[{'filename':key+'.png','type':'output','subfolder':''}]} for key in ('9','47','48')}}}
    monkeypatch.setattr(flux3,'_call',call)
    image=io.BytesIO();Image.new('RGB',(16,16),'tan').save(image,'PNG')
    monkeypatch.setattr(stage.urllib.request,'urlopen',lambda url,timeout:io.BytesIO(image.getvalue()))
    stage.expand_one(tmp_path,index)
    assert len(submissions)==1
    assert submissions[0]['17']['inputs']['image']=='synthetic-upload.png'
    part=terrain/f'expand-{index}'
    assert all((part/(name+'.png')).is_file() for name in ('raw-outpaint','padding-mask','padded-input'))
    geometry=json.loads((part/'geometry.json').read_text())
    assert geometry['expansions']==index
    assert geometry['source_sha256']==stage.sha(source)
    assert json.loads((tmp_path/'status.json').read_text())['status']=='complete'
