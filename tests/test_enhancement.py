"""Optional refinement must preserve originals, clock and queue isolation."""
import hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from server.main import create_app
from server.config import Settings
from worker.client import ServerClient
from worker.runner import WorkerConfig,run_job
from touchdown.enhance import chunks,graph

@pytest.fixture
def env(tmp_path):
    settings=Settings(data_dir=tmp_path/'data',worker_token='token',gallery_seed=[tmp_path/'none'])
    app=create_app(settings); client=TestClient(app)
    worker=ServerClient('http://testserver','token',http=TestClient(app))
    jobs=app.state.jobs
    parent=jobs.create('bamburgh','','',{'title':'Bamburgh'},workflow='photographic-v1')
    folder=settings.jobs_dir/parent['id'];folder.mkdir(parents=True)
    original=folder/'touchdown.mp4';original.write_bytes(b'original film')
    jobs.update(parent['id'],status='done',result=original.name)
    return client,worker,jobs,parent['id'],original,settings


def test_offline_capability_dedup_and_original_binding(env):
    client,worker,jobs,parent,original,_=env
    assert client.post(f'/api/jobs/{parent}/enhance').status_code==503
    worker.heartbeat(enhancement=True)
    first=client.post(f'/api/jobs/{parent}/enhance').json()
    assert first['parent_id']==parent
    assert client.post(f'/api/jobs/{parent}/enhance').json()['id']==first['id']
    assert worker.claim() is None
    job=worker.claim(enhancement=True)
    assert job['workflow']=='refine-details-v1'
    assert client.get(f"/api/worker/jobs/{job['id']}/source-video").status_code==401
    assert worker.http.get(f"/api/worker/jobs/{job['id']}/source-video").content==original.read_bytes()
    original.write_bytes(b'changed original')
    assert worker.http.get(f"/api/worker/jobs/{job['id']}/source-video").status_code==409


def test_worker_end_to_end_and_clock_hold(env,tmp_path):
    client,worker,jobs,parent,original,settings=env
    worker.heartbeat(enhancement=True)
    child=client.post(f'/api/jobs/{parent}/enhance').json()
    fake=tmp_path/'fake.py'
    fake.write_text('''import pathlib,sys,json,hashlib
source,root=map(pathlib.Path,sys.argv[1:])
video=root/'enhanced.mp4';video.write_bytes(b'enhanced film')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'outcome.json').write_text(json.dumps(dict(video=video.name,video_sha256=sha(video),source_sha256=sha(source),source_frames=195,output_frames=195,source_fps='24/1',output_fps='24/1',clock_preserved=True)))
''')
    cfg=WorkerConfig(repo=Path(__file__).resolve().parents[1],python=sys.executable,enhancement=True,output=tmp_path/'output',tick_s=.01,enhance_build_cmd='{python} '+str(fake)+' {input} {output}')
    job=worker.claim(enhancement=True)
    assert run_job(job,worker,cfg)=='done'
    assert jobs.get(child['id'])['status']=='done'
    assert original.read_bytes()==b'original film' and jobs.get(parent)['status']=='done'
    assert client.get(f"/api/jobs/{child['id']}").json()['video']
    assert worker.http.post(f"/api/worker/jobs/{parent}/result",files={'file':('bad.mp4',b'bad','video/mp4')}).status_code==409
    report=settings.jobs_dir/child['id']/'review.json'
    data=json.loads(report.read_text());data['output_frames']=194;report.write_text(json.dumps(data))
    worker.finish(child['id'],'done')
    assert jobs.get(child['id'])['status']=='held'
    assert original.read_bytes()==b'original film'


def test_cancelled_enhancement_never_replaces_original(env):
    client,worker,jobs,parent,original,_=env
    worker.heartbeat(enhancement=True)
    child=client.post(f'/api/jobs/{parent}/enhance').json()
    worker.claim(enhancement=True)
    client.post(f"/api/jobs/{child['id']}/cancel")
    assert worker.http.post(f"/api/worker/jobs/{child['id']}/result",files={'file':('x.mp4',b'x','video/mp4')}).status_code==409
    worker.finish(child['id'],'done')
    assert jobs.get(child['id'])['status']=='cancelled'
    assert original.read_bytes()==b'original film'
    assert jobs.get(parent)['status']=='done'


def test_bounded_recipe_and_original_frame_mapping():
    assert chunks(195)==[(0,49,49),(48,97,49),(96,145,49),(144,195,57)]
    for count in (1,48,49,50,57,58,96,195,400):
        plan=chunks(count)
        assert sum(last-first-int(i>0) for i,(first,last,_) in enumerate(plan))==count
        assert all(9<=length<=57 and length%8==1 for _,_,length in plan)
    g=graph('input.mp4',57,3840,2208,30,'test')
    assert g['456']['inputs']['tile_frames']==g['459']['inputs']['tile_frames']==57
    assert g['456']['inputs']['use_tiled_encode'] is True
    assert g['446']['inputs']['frame_rate']==g['480']['inputs']['fps']==30
    assert g['441']['inputs']['lora_name']=='ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors'
    assert g['459']['inputs']['tile_width']==1024
    assert not {'400','449','450'} & g.keys()
