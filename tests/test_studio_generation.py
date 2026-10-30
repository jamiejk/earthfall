"""Photographic submission and worker transport, with explicitly synthetic builds."""
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from test_studio import env, save  # shared studio fixture; all external discovery is mocked
from worker.client import ServerClient
from worker.runner import WorkerConfig, run_job
from touchdown.studio_brief import POSES, FRAMINGS, scene_config, adjudicate_crop


def online(client):
    return client.post("/api/worker/heartbeat?canonical=true", headers={"Authorization": "Bearer worker"})


def enqueue(client):
    record = save(client).json()
    online(client)
    response = client.post(f"/api/studio/setups/{record['id']}/render")
    assert response.status_code == 200, response.text
    return record, response.json()


def worker_for(client):
    return ServerClient("http://testserver", "worker", http=client)


def test_submission_is_idempotent_and_old_workers_cannot_claim_photo_jobs(env):
    client, settings, app = env
    record = save(client).json()
    route = f"/api/studio/setups/{record['id']}/render"
    assert client.post(route).status_code == 503
    assert online(client).status_code == 200
    assert client.get("/api/studio/availability").json()["render_available"]
    first = client.post(route).json()
    assert client.post(route).json()["id"] == first["id"]
    assert first["workflow"] == "photographic-grounded-v1" and first["setup_id"] == record["id"]
    worker = worker_for(client)
    assert worker.claim() is None
    assert worker.claim(photographic=True) is None
    job = worker.claim(canonical=True)
    assert job["id"] == first["id"] and job["setup"]["ending"] == record["ending"]
    assert len(job["setup"]["references"]) >= 2 and job["manifest"] is None
    assert worker.claim(canonical=True) is None
    ref = job["setup"]["references"][0]
    response = client.get(f"/api/worker/jobs/{job['id']}/inputs/{ref['name']}")
    assert hashlib.sha256(response.content).hexdigest() == ref["sha256"]
    assert client.get(f"/api/worker/jobs/{job['id']}/inputs/setup.json").status_code == 404
    app.state.canonical_worker_seen = time.time()-60
    assert not client.get("/api/studio/availability").json()["render_available"]


@pytest.fixture
def fake_builder(tmp_path):
    path = tmp_path / "fake_photographic.py"
    path.write_text('''import argparse,json,os,time,hashlib
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
r=Path(a.output);(r/'previews').mkdir(exist_ok=True)
Image.new('RGB',(80,60),'green').save(r/'previews/arrival.jpg')
(r/'progress.json').write_text(json.dumps({'stages':{'arrival':'done','flight':'running'},'message':'Synthetic test build'}))
time.sleep(float(os.environ.get('FAKE_PHOTO_DELAY','0')))
status=os.environ.get('FAKE_PHOTO_STATUS','done')
(r/'test.mp4').write_bytes(b'explicitly synthetic video transport fixture')
for name in ('flight','layout','texture','expand-1','expand-2','expand-3','project','orbit','final'):
 request=dict(id=name,title='Synthetic '+name,description='Explicit synthetic transport fixture',assets=[dict(name='test.mp4',path='test.mp4',sha256=hashlib.sha256((r/'test.mp4').read_bytes()).hexdigest())])
 tmp=r/'checkpoint.tmp';tmp.write_text(json.dumps(request));tmp.replace(r/'checkpoint.json')
 folder=r/'automatic-reviews'/name;folder.mkdir(parents=True)
 report=folder/'report.json';report.write_text(json.dumps({'answer':{'status':'pass'}}))
 public={**request,'assets':[{k:v for k,v in a.items() if k!='path'} for a in request['assets']]}
 decision={'continue':True,'reason':'Explicitly synthetic passed fixture','reviewer':'qwen','record':str(report.relative_to(r)),'record_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'evidence_sha256':hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()}
 if name=='layout':decision['layout']=dict(cloud_exit_frame=64,target_uv=[.5,.5],colours={'background':'#697b50'},features=[],layout_description='Synthetic connected land region',expansion_description='Continue the synthetic connected land region')
 tmp=folder/'decision.tmp';tmp.write_text(json.dumps(decision));tmp.replace(folder/'decision.json')
 deadline=time.monotonic()+10
 while not (r/'decisions'/f'{name}.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('Synthetic checkpoint transport timed out')
  time.sleep(.01)
(r/'outcome.json').write_text(json.dumps({'recipe':'photographic-grounded-v1','status':status,'checks_passed' :status=='done','video':'test.mp4','video_sha256':hashlib.sha256((r/'test.mp4').read_bytes()).hexdigest(),'message':'Synthetic test complete'}))
''')
    return path


@pytest.mark.parametrize("status", ["done", "held"])
def test_photo_worker_saves_result_and_review_and_reopens_the_job(env, fake_builder, monkeypatch, status):
    client, settings, _ = env
    record, public = enqueue(client)
    worker = worker_for(client)
    job = worker.claim(canonical=True)
    monkeypatch.setenv("FAKE_PHOTO_STATUS", status)
    cfg = WorkerConfig(repo=settings.repo, python=sys.executable, output=settings.data_dir / "worker-output",
                       canonical=True, canonical_build_cmd=f"{{python}} {fake_builder} --input {{input}} --output {{output}}", tick_s=.01)
    assert run_job(job, worker, cfg) == status
    state = client.get(f"/api/jobs/{job['id']}").json()
    assert state["status"] == status
    assert state["video"] if status == "done" else state["review_video"]
    assert client.get(state["review_url"]).json()["checks_passed"] == (status == "done")
    assert client.get(state["video"] or state["review_video"]).status_code == 200
    assert client.get(f"/api/studio/setups/{record['id']}").json()["latest_job"]["id"] == job["id"]
    if status == "done":
        assert client.post(f"/api/studio/setups/{record['id']}/render").json()["id"] == job["id"]
    else:
        assert client.post(f"/api/studio/setups/{record['id']}/render").json()["id"] != job["id"]


def test_cancellation_stops_the_photo_subprocess(env, fake_builder, monkeypatch):
    client, settings, _ = env
    _, public = enqueue(client)
    worker = worker_for(client)
    job = worker.claim(canonical=True)
    monkeypatch.setenv("FAKE_PHOTO_DELAY", "30")
    cfg = WorkerConfig(repo=settings.repo, python=sys.executable, output=settings.data_dir / "worker-output",
                       canonical=True, canonical_build_cmd=f"{{python}} {fake_builder} --input {{input}} --output {{output}}", tick_s=.02)
    result = []
    thread = threading.Thread(target=lambda: result.append(run_job(job, worker, cfg)))
    thread.start()
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["stages"].get("flight") == "running":
            break
        time.sleep(.02)
    client.post(f"/api/jobs/{job['id']}/cancel")
    thread.join(timeout=5)
    assert not thread.is_alive() and result == ["cancelled"]
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"


@pytest.mark.parametrize("review,cancel,expected", [(None, False, "held"), ("wrong-hash", False, "held"),
                                                   ("valid", False, "done"), ("valid", True, "cancelled")])
def test_result_requires_a_matching_delivered_review_and_honours_late_cancel(env, review, cancel, expected):
    client, settings, _ = env
    _, public = enqueue(client)
    worker = worker_for(client)
    job = worker.claim(canonical=True)
    route = f"/api/worker/jobs/{job['id']}"
    client.post(route+"/result", files={"file": ("test.mp4", b"synthetic", "video/mp4")})
    if review:
        from test_canonical_integration import approve_checkpoints
        approve_checkpoints(worker,job['id'],settings.data_dir/'fixture-reviews',b'synthetic')
        worker.review(job["id"], {"recipe":"photographic-grounded-v1", "status": "done", "checks_passed": True,
                                 "video_sha256": hashlib.sha256(b"synthetic").hexdigest() if review == "valid" else "0"*64})
    if cancel:
        client.post(f"/api/jobs/{job['id']}/cancel")
    client.post(route+"/finish", json={"status": "done"})
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == expected


def test_missing_or_changed_reference_is_not_queued(env):
    client, settings, _ = env
    record = save(client).json()
    online(client)
    path = settings.data_dir / "setups" / record["id"] / record["default_references"][0]["name"]
    path.write_bytes(b"changed")
    assert client.post(f"/api/studio/setups/{record['id']}/render").status_code == 409


@pytest.mark.parametrize("ending,framing", [("relaxed", "medium"), ("model", "close"), ("look-back", "three-quarter"),
                                            ("wave", "full"), ("arms-open", "three-quarter"), ("twirl", "full")])
def test_scene_prompts_use_saved_controls_and_keep_a_full_body_ground_anchor(tmp_path, ending, framing):
    brief = dict(selected_references=[0, 1], setting="Recognisable monument on a broad lawn", surrounding_landscape="Low trees and open countryside",
                 regional_view="Southern England", appearance="Person with short curls in a red jacket", support="grass",
                 support_description="short connected grass", light="Soft natural daylight")
    setup = dict(destination={"title": "Stonehenge"}, subject={"name": "subject.jpg"}, references=[{"name": "location-0.jpg"}, {"name": "location-1.jpg"}],
                 ending=ending, framing=framing)
    cfg = scene_config(setup, brief, tmp_path, tmp_path, 42)
    assert cfg['video']['phase_contract']['bounds']==[0,29,58,108,156,192]
    assert cfg['video']['phase_contract']['retiming'] is False
    assert cfg['video']['split']['high_motion_frames']==0
    assert cfg['video']['split']['seam_carry_frames']==0
    assert '30,000 FEET' in cfg['prompts']['high']
    assert 'mountain ranges' not in cfg['prompts']['high']  # generic wording must respect flat regional patterns
    assert "BOTH ACTUAL SHOES" in cfg["prompts"]["arrival"]
    assert POSES[ending] in cfg["prompts"]["near"] and FRAMINGS[framing] in cfg["prompts"]["near"]
    assert cfg["gate"]["allowed_supports"] == ["grass", "ground_paving", "natural_rock"]
    assert cfg["video"]["frames"] % 8 == cfg["video"]["near_frames"] % 8 == 1
    assert cfg["video"]["frames"] - cfg["video"]["near_frames"] == 168
    assert "Stonehenge" not in cfg["prompts"]["high"]
    assert "Stonehenge" not in cfg["prompts"]["orbit"]  # custom names without country/region cannot become planetary objects
    assert "Low trees" not in cfg["prompts"]["high"]  # close camera description cannot leak into the regional stage


@pytest.mark.parametrize("region,head,expected", [("chest", True, "fail"), ("knees", True, "pass"),
                                               ("unclear", True, "unclear"), ("knees", False, "fail")])
def test_request_primed_crop_approval_cannot_override_independent_observation(region, head, expected):
    raw = {"framing": {"status": "pass", "evidence": "Chest up and cropped at the knees"},
           "ending": {"status": "pass", "evidence": "Both arms open"}}
    result = adjudicate_crop(raw, {"head_complete": head, "lowest_visible_region": region, "evidence": "Independent visible body region"}, "three-quarter")
    assert result["framing"]["status"] == expected
    assert raw["framing"]["status"] == "pass"  # original model answer remains inspectable


def test_late_reveal_is_found_before_the_three_second_portrait_prefix(tmp_path, monkeypatch):
    from touchdown import studio_build
    class Capture:
        def get(self, key):
            # The fixture models a 24 fps clip; returning one number for every property made the
            # frame-rate read as 241 once the sampler started deriving its step from fps.
            return 24 if key == studio_build.cv2.CAP_PROP_FPS else 241
        def release(self):
            pass
    monkeypatch.setattr(studio_build.cv2, "VideoCapture", lambda *args: Capture())
    def evidence(source, times, output):
        # The window must still reach the late reveal and must never sample inside the portrait. The
        # exact grid is not the contract: a 241-frame clip asks for more half-second samples than the
        # checker accepts, so the spacing widens to stay within the limit.
        assert len(times) <= 12
        assert times[-1] >= 6.5
        assert max(times) < (241 - 73) / 24
        return dict(frames=[dict(frame_id=str(t), source_frame=round(t*24), image=str(tmp_path/'frame.jpg')) for t in times])
    def observe(url, images, *args):
        clear = float(images[0][1]) >= 6
        return dict(stage="aerial" if clear else "cloud", terrain_clear=clear, evidence="Visible scene")
    monkeypatch.setattr(studio_build.vision, "prepare", evidence)
    monkeypatch.setattr(studio_build, "chat", observe)
    found = studio_build.reveal_frame(tmp_path/'video.mp4', tmp_path, 73)
    assert 144 <= found < 168      # a late reveal, before the portrait prefix begins
