"""The Touchdown web app (app/): server API, queue, worker and job manifests, end to end with a fake build.

    PYTHONPATH=scripts:.:app python3 -m pytest -q tests/test_app.py
"""
import ast
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("tomli_w")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from server.config import Settings  # noqa: E402
from server.main import create_app  # noqa: E402
from worker import runner  # noqa: E402
from worker.client import ServerClient, loop  # noqa: E402
from worker.describe import describe_person  # noqa: E402

TOKEN = "test-worker-token"
FAKE = "{python} {app}/worker/fake_build.py {manifest}"


def jpeg(exif_gps: bool = False, size=(320, 480)) -> bytes:
    im = Image.new("RGB", size, (180, 40, 40))
    buf = io.BytesIO()
    if exif_gps:
        exif = Image.Exif()
        exif[0x010F] = "PhoneMaker"          # Make
        exif[0x8825] = {2: (37.0, 1.0, 0.0)}  # GPSInfo: latitude
        im.save(buf, "JPEG", exif=exif.tobytes())
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_STAGE_SECONDS", "0.05")
    monkeypatch.delenv("FAKE_FAIL_AT", raising=False)
    settings = Settings(data_dir=tmp_path / "data", repo=ROOT, worker_token=TOKEN, gallery_seed=[tmp_path / "none"])
    app = create_app(settings)
    client = TestClient(app)
    cfg = runner.WorkerConfig(repo=ROOT, python=sys.executable, build_cmd=FAKE, output=tmp_path / "outputs",
                              share_terrain=False, tick_s=0.05)
    worker = ServerClient("http://testserver", TOKEN, http=TestClient(app))
    return client, worker, cfg, settings


def submit(client, preset="monument-valley", wearing="", photo=None):
    return client.post("/api/jobs", data={"preset": preset, "wearing": wearing},
                       files={"photo": ("me.jpg", photo or jpeg(), "image/jpeg")})


def test_stages_match_the_build():
    tree = ast.parse((ROOT / "scripts/touchdown/build.py").read_text())
    stages = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "STAGES")
    assert tuple(stages) == runner.STAGES


def test_describe_person():
    assert describe_person(Path("x.jpg"), "", "the default") == "the default"
    assert describe_person(Path("x.jpg"), "a blue parka and jeans", "d") == "a person wearing a blue parka and jeans"
    assert describe_person(Path("x.jpg"), "blue parka {x}", "d") == "a person wearing blue parka x"
    assert describe_person(Path("x.jpg"), "wearing a hat", "d") == "a person wearing a hat"


def test_config_lists_three_presets_and_an_experimental_custom_location(app_env):
    client, *_ = app_env
    cfg = client.get("/api/config").json()
    assert [p["id"] for p in cfg["presets"]] == ["monument-valley", "grand-canyon-mather", "yosemite-tunnel-view"]
    assert all(p["default_person"] and "manifest" not in p for p in cfg["presets"])
    assert cfg["custom_location"]["enabled"] is True and cfg["custom_location"]["experimental"] is True
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_job_manifest_copies_the_preset_with_this_job(tmp_path):
    preset = ROOT / "shots/flux3/monument-valley.toml"
    photo = tmp_path / "subject.jpg"
    photo.write_bytes(jpeg())
    m = runner.write_job_manifest(preset, tmp_path / "job.toml", photo, "a person in a hat", "app-mv-abc", tmp_path)
    back = tomllib.loads((tmp_path / "job.toml").read_text())
    original = tomllib.loads(preset.read_text())
    assert back == m
    assert back["shot"]["subject"] == str(photo.resolve())
    assert back["shot"]["person"] == "a person in a hat"
    assert back["shot"]["name"] == "app-mv-abc"
    assert back["output"] == str(tmp_path)
    for table in ("terrain", "stills", "opener", "cloud", "swoop", "end", "face", "sound"):
        assert back[table] == original[table], table
    assert back["shot"]["scene"] == original["shot"]["scene"]
    # the job runs the stages its manifest enables (the face transplant is off in the presets)
    assert runner.enabled_stages(back) == runner.enabled_stages(original)


def test_upload_is_checked_and_exif_stripped(app_env):
    client, _, _, settings = app_env
    assert submit(client, photo=b"not an image").status_code == 400
    assert submit(client, preset="battery-spencer").status_code == 400
    r = submit(client, photo=jpeg(exif_gps=True))
    assert r.status_code == 200, r.text
    saved = next(settings.uploads.iterdir())
    with Image.open(saved) as im:
        assert not im.getexif()
        assert "exif" not in im.info
    settings.max_upload_mb = 0.001
    assert submit(client).status_code == 413


def test_end_to_end_with_a_fake_build(app_env):
    client, worker, cfg, settings = app_env
    job = submit(client, wearing="a green anorak").json()
    assert job["status"] == "queued" and job["position"] == 1
    loop(worker, cfg, once=True, log=lambda *_: None)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done", done
    preset = tomllib.loads((ROOT / "shots/flux3/monument-valley.toml").read_text())
    assert set(done["stages"].values()) == {"done"}
    assert list(done["stages"]) == runner.enabled_stages(preset)
    assert sorted(p["name"] for p in done["previews"]) == ["cloud.jpg", "end.jpg", "orbit.jpg"]
    assert client.get(done["previews"][0]["url"]).status_code == 200
    video = client.get(done["video"] + "?download=1")
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert "attachment" in video.headers["content-disposition"]
    manifest = cfg.output / f"app-monument-valley-{job['id'].lower()}" / "job.toml"
    assert tomllib.loads(manifest.read_text())["shot"]["person"] == "a person wearing a green anorak"
    events = client.get(f"/api/jobs/{job['id']}/events")
    assert '"status": "done"' in events.text


def test_one_job_at_a_time_and_queue_positions(app_env):
    client, worker, *_ = app_env
    a, b = submit(client).json(), submit(client).json()
    assert client.get(f"/api/jobs/{b['id']}").json()["position"] == 2
    first = worker.claim()
    assert first["id"] == a["id"] and first["manifest"] == "shots/flux3/monument-valley.toml"
    assert worker.claim() is None  # one runs at a time
    assert client.get(f"/api/jobs/{b['id']}").json()["position"] == 2


def test_queue_claims_remain_visible_during_concurrent_polling(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from server.db import Jobs

    jobs = Jobs(tmp_path / "queue.sqlite3")
    ids = [jobs.create("example", "", "subject.jpg")["id"] for _ in range(40)]
    start = threading.Barrier(5)

    def worker():
        start.wait()
        for expected in ids:
            job = jobs.claim()
            assert job and job["id"] == expected and job["status"] == "running"
            jobs.update(job["id"], stages={"stills": "running"})
            jobs.finish(job["id"], "done")

    def browser(offset):
        start.wait()
        for i in range(200):
            job = jobs.get(ids[(i + offset) % len(ids)])
            assert job and isinstance(job["stages"], dict) and isinstance(job["place"], dict)
            assert jobs.active_count() >= 0
            assert jobs.position(job) >= 0

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(worker), *(pool.submit(browser, i) for i in range(4))]
        for future in futures:
            future.result(timeout=15)
    assert all(jobs.get(job_id)["status"] == "done" for job_id in ids)


def test_cancel_a_queued_job(app_env):
    client, worker, *_ = app_env
    job = submit(client).json()
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert worker.claim() is None


def test_cancel_a_running_job_kills_only_its_build(app_env, monkeypatch):
    client, worker, cfg, _ = app_env
    monkeypatch.setenv("FAKE_STAGE_SECONDS", "15")
    job = submit(client).json()
    t = threading.Thread(target=loop, args=(worker, cfg), kwargs={"once": True, "log": lambda *_: None})
    t.start()
    # The job has to be claimed and its build started before a cancel means anything; a loaded machine (a GPU
    # build running next to the tests) can take a while to get there, so wait for the job's own status.
    deadline = time.time() + 60
    while time.time() < deadline:
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] == "running" and any(v == "running" for v in state["stages"].values()):
            break
        time.sleep(0.1)
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelling"
    t.join(timeout=60)
    assert not t.is_alive()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"


def test_a_failed_build_reports_its_tail(app_env, monkeypatch):
    client, worker, cfg, _ = app_env
    monkeypatch.setenv("FAKE_FAIL_AT", "opener")
    job = submit(client).json()
    loop(worker, cfg, once=True, log=lambda *_: None)
    failed = client.get(f"/api/jobs/{job['id']}").json()
    assert failed["status"] == "failed"
    assert "fake failure in opener" in failed["error"] and "exited with code 1" in failed["error"]
    assert failed["video"] is None


def test_worker_api_needs_the_token(app_env):
    client, *_ = app_env
    assert client.post("/api/worker/claim").status_code == 401
    assert client.post("/api/worker/claim", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_passcode(tmp_path):
    settings = Settings(data_dir=tmp_path, repo=ROOT, passcode="open sesame", gallery_seed=[])
    client = TestClient(create_app(settings))
    assert client.get("/api/config").json()["passcode_required"] is True
    assert client.get("/api/gallery").status_code == 401
    assert client.post("/api/login", json={"passcode": "wrong"}).status_code == 401
    assert client.post("/api/login", json={"passcode": "open sesame"}).status_code == 200
    assert client.get("/api/gallery").status_code == 200


def test_gallery_lists_seeded_videos(tmp_path):
    seed = tmp_path / "review" / "v14"
    seed.mkdir(parents=True)
    (seed / "touchdown.mp4").write_bytes(b"mp4")
    client = TestClient(create_app(Settings(data_dir=tmp_path / "d", repo=ROOT, gallery_seed=[seed.parent])))
    items = client.get("/api/gallery").json()["items"]
    assert [i["file"] for i in items] == ["v14-touchdown.mp4"]
    assert client.get(items[0]["url"]).content == b"mp4"
    assert client.get("/api/gallery/..%2Fsecret.mp4").status_code == 404


def test_terrain_is_shared_only_between_identical_terrain_inputs(tmp_path):
    preset = ROOT / "shots/flux3/monument-valley.toml"
    out = tmp_path / "outputs"
    root = out / "app-monument-valley-job1"
    root.mkdir(parents=True)
    job_manifest = root / "job.toml"
    job_manifest.write_text("x")
    donor = out / "flux3-monument-valley" / "terrain" / "donorkey"
    donor.mkdir(parents=True)
    (donor / "guide.mp4").write_bytes(b"guide")
    (donor / "done.json").write_text(json.dumps({"output": "guide.mp4"}))
    cfg = runner.WorkerConfig(repo=ROOT, output=out)

    def keys(neutral_donor):
        return lambda *_: {str(job_manifest): {"key": "jobkey", "neutral": "N"},
                           str(preset): {"key": "donorkey", "neutral": neutral_donor}}

    note = runner.share_terrain(cfg, preset, job_manifest, root, out, keys_fn=keys("other"))
    assert "no finished terrain" in note and not (root / "terrain").exists()
    note = runner.share_terrain(cfg, preset, job_manifest, root, out, keys_fn=keys("N"))
    assert note.startswith("terrain: reused")
    assert (root / "terrain" / "jobkey" / "guide.mp4").read_bytes() == b"guide"
    assert runner.stage_states(root, ["terrain", "stills"], {"terrain"}) == {"terrain": "cached", "stills": "pending"}


def test_build_command_template():
    cfg = runner.WorkerConfig(repo=ROOT, python="py", build_cmd=runner.DEFAULT_CMD, comfy_url="http://h:1", descent="45")
    assert runner.build_command(cfg, Path("/m/job.toml")) == ["py", "-m", "touchdown.build", "/m/job.toml",
                                                               "--descent", "45", "--url", "http://h:1"]


def test_real_terrain_keys_differ_by_person_but_neutral_keys_match(tmp_path):
    """With the real touchdown.build: a job's terrain key differs from the preset's (the [shot] table is hashed),
    while the neutral key, with subject/person/name blanked, is the same, so the preset's terrain can be reused."""
    pytest.importorskip("numpy")
    preset = ROOT / "shots/flux3/monument-valley.toml"
    photo = tmp_path / "subject.jpg"
    photo.write_bytes(jpeg())
    out = tmp_path / "outputs"
    job = tmp_path / "job.toml"
    runner.write_job_manifest(preset, job, photo, "a person wearing a hat", "app-monument-valley-x", out)
    other = ROOT / "shots/flux3/yosemite-tunnel-view.toml"
    cfg = runner.WorkerConfig(repo=ROOT, python=sys.executable)
    try:
        keys = runner.terrain_keys(cfg, [job, preset, other], out)
    except RuntimeError as e:
        pytest.skip(f"touchdown.build not importable here: {e}")
    assert keys[str(job)]["key"] != keys[str(preset)]["key"]
    assert keys[str(job)]["neutral"] == keys[str(preset)]["neutral"]
    assert keys[str(other)]["neutral"] != keys[str(preset)]["neutral"]


def test_progress_lines_stay_out_of_the_error_tail() -> None:
    from worker.runner import is_progress

    for bar in ("99%", "100%", " 45%|####      | 9/20 [00:03<00:04,  2.51it/s]", "12/20 [00:05<00:03, 1.2s/it]"):
        assert is_progress(bar), bar
    for real in ("ValueError: end still failed the gate", "stills seed 3: camera fit {}", "Built /x/touchdown.mp4", ""):
        assert not is_progress(real), real


# --- the place picker: catalogue places, generated manifests, custom US points ---------------------------------

def gen_env(app_env):
    """app_env with the worker generating manifests (no NAIP network call)."""
    client, worker, cfg, settings = app_env
    cfg.naip_words = False
    return client, worker, cfg, settings


def test_config_lists_catalogue_places_with_support_and_tested_flags(app_env):
    client, *_ = app_env
    places = {p["id"]: p for p in client.get("/api/config").json()["places"]}
    supported = [p for p in places.values() if p["supported"]]
    assert len([p for p in supported if p["country"] == "US"]) == 20  # 24 minus the two parked bridges, Delicate Arch and Bryant Park (not measured)
    assert not [p for p in supported if p["country"] != "US"]  # abroad parked (18 Oct)
    assert places["milford-sound"]["reason"].startswith("Coming soon")
    assert places["ben-nevis"]["supported"] is False and "30 m terrain" in places["ben-nevis"]["reason"]
    assert places["zion-angels-landing"]["supported"] is False
    assert places["grand-canyon-mather-point"]["preset"] == "grand-canyon-mather"
    assert {p["id"] for p in supported if p["tested"]} == {"monument-valley", "grand-canyon-mather-point",
                                                           "yosemite-tunnel-view"}
    assert places["horseshoe-bend"]["thumb"] == "/static/img/kind-canyon.svg"
    assert client.get(places["horseshoe-bend"]["thumb"]).status_code == 200


def test_every_catalogue_place_is_still_offered(app_env):
    """Nothing is dropped from the picker: every catalogue place is listed, with the reason it cannot be built."""
    from server import places
    client, *_ = app_env
    rows = client.get("/api/config").json()["places"]
    catalogue = {p["id"] for p in places.load_catalogue(ROOT)}
    assert {r["id"] for r in rows} == catalogue
    for row in rows:
        assert row["group"] in ("tested", "experimental", "soon")
        if row["supported"]:
            assert row["group"] == ("tested" if row["tested"] else "experimental")
            assert row["reason"] == ""
        else:
            assert row["group"] == "soon" and row["reason"]


def test_the_picker_groups_the_tested_places_first(app_env):
    """The list comes back tested first, the flagship first of all, then the experimental ones."""
    client, *_ = app_env
    rows = client.get("/api/config").json()["places"]
    groups = [r["group"] for r in rows]
    assert groups == sorted(groups, key=lambda g: ("tested", "experimental", "soon").index(g))
    tested = [r for r in rows if r["group"] == "tested"]
    assert [r["id"] for r in tested] == ["monument-valley", "grand-canyon-mather-point", "yosemite-tunnel-view"]
    assert tested[0]["flagship"] is True
    assert all(r["flagship"] is False for r in rows[3:])
    experimental = [r for r in rows if r["group"] == "experimental"]
    assert len(experimental) == 17  # the 20 buildable places minus the 3 with a real build
    assert "horseshoe-bend" in {r["id"] for r in experimental}
    assert all(r["supported"] and not r["tested"] for r in experimental)


def test_the_counts_come_from_the_data(app_env):
    """The numbers the picker shows are the ones the catalogue holds: 3 tested, 17 experimental, 20 buildable."""
    client, *_ = app_env
    rows = client.get("/api/config").json()["places"]
    tested = [r for r in rows if r["group"] == "tested"]
    experimental = [r for r in rows if r["group"] == "experimental"]
    buildable = tested + experimental
    assert (len(tested), len(experimental), len(buildable)) == (3, 17, 20)
    assert all(r["country"] == "US" for r in buildable)
    assert len([r for r in rows if r["group"] == "soon"]) == len(rows) - 20


# Photographic studio behavior is covered in test_studio.py and live Chromium checks.
# Legacy catalogue and job API checks below remain for existing terrain jobs.


def test_mark_tested_adds_a_place(app_env):
    from server import places
    client, _, _, settings = app_env
    places.mark_tested(settings, "horseshoe-bend", "real build 19 Oct")
    rows = {p["id"]: p for p in client.get("/api/config").json()["places"]}
    assert rows["horseshoe-bend"]["tested"] is True


def test_a_catalogue_place_with_a_preset_uses_the_hand_written_manifest(app_env):
    client, worker, *_ = app_env
    job = submit(client, preset="grand-canyon-mather-point").json()
    assert job["preset"] == "grand-canyon-mather" and job["title"] == "Grand Canyon, Mather Point"
    claimed = worker.claim()
    assert claimed["manifest"] == "shots/flux3/grand-canyon-mather.toml" and "place" not in claimed


def test_unsupported_places_are_refused(app_env):
    client, *_ = app_env
    for preset in ("ben-nevis", "zion-angels-landing", "no-such-place", "../shots/x"):
        assert submit(client, preset=preset).status_code == 400, preset


def test_end_to_end_with_a_generated_catalogue_manifest(app_env):
    client, worker, cfg, _ = gen_env(app_env)
    job = submit(client, preset="horseshoe-bend", wearing="a yellow jacket").json()
    assert job["title"] == "Horseshoe Bend"
    loop(worker, cfg, once=True, log=lambda *_: None)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done", done
    generated = cfg.output / "app-places" / "horseshoe-bend.toml"
    m = tomllib.loads((cfg.output / f"app-horseshoe-bend-{job['id'].lower()}" / "job.toml").read_text())
    assert generated.is_file()
    assert m["shot"]["place"] == "horseshoe-bend" and m["shot"]["person"] == "a person wearing a yellow jacket"
    assert m["terrain"]["project"] == "AZ_NorthEast_D23" and 280 < m["terrain"]["heading"] < 283
    assert "Horseshoe Bend" in m["stills"]["end_prompt"]


def test_custom_point_check(app_env, monkeypatch):
    from server import places
    client, *_ = app_env
    places._checked.clear()
    monkeypatch.setattr(places, "find_projects", lambda repo, lat, lon, cache: [{"project": "AZ_NorthEast_D23",
                                                                                "zone": 12, "year": 2023}])
    ok = client.get("/api/custom/check", params={"lat": 36.8619, "lon": -111.3743}).json()
    assert ok["ok"] and ok["project"] == "AZ_NorthEast_D23" and ok["state"] == "AZ"
    assert client.get("/api/custom/check", params={"lat": 21.3, "lon": -157.8}).json()["ok"] is False  # Hawaii
    assert client.get("/api/custom/check", params={"lat": 48.85, "lon": 2.35}).json()["ok"] is False   # Paris
    monkeypatch.setattr(places, "find_projects", lambda *a: [])
    none = client.get("/api/custom/check", params={"lat": 41.5, "lon": -100.0}).json()
    assert none["ok"] is False and "No USGS 1 m lidar" in none["message"]
    assert client.get("/api/custom/check", params={"lat": 99, "lon": 0}).status_code == 400


def test_end_to_end_with_a_custom_point(app_env, monkeypatch):
    from server import places
    client, worker, cfg, _ = gen_env(app_env)
    places._checked.clear()
    monkeypatch.setattr(places, "find_projects", lambda *a: [{"project": "AZ_NorthEast_D23", "zone": 12, "year": 2023}])
    r = client.post("/api/jobs", data={"preset": "custom", "latitude": "36.8619", "longitude": "-111.3743",
                                       "name": "Page {rim} \"overlook\""},
                    files={"photo": ("me.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["title"] == "Page rim overlook"
    loop(worker, cfg, once=True, log=lambda *_: None)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done", done
    m = tomllib.loads((cfg.output / f"app-custom-{job['id'].lower()}" / "job.toml").read_text())
    assert m["place"]["latitude"] == 36.8619 and m["place"]["name"] == "Page rim overlook"
    assert m["shot"]["place"] == m["place"]["id"] and m["terrain"]["project"] == "AZ_NorthEast_D23"
    monkeypatch.setattr(places, "find_projects", lambda *a: [])
    places._checked.clear()
    bad = client.post("/api/jobs", data={"preset": "custom", "latitude": "41.5", "longitude": "-100.0"},
                      files={"photo": ("me.jpg", jpeg(), "image/jpeg")})
    assert bad.status_code == 400


def test_geocode_goes_through_the_server_and_is_cleaned(app_env, monkeypatch):
    from server import places
    client, *_ = app_env
    calls = []

    def fake(path, params):
        calls.append((path, params))
        if path == "search":
            return [{"name": "Horseshoe {Bend}", "display_name": "Horseshoe Bend, Coconino County, Arizona",
                     "lat": "36.8791", "lon": "-111.5104", "address": {"state": "Arizona"}}]
        return {"name": "", "address": {"county": "Coconino County", "state": "Arizona"}}

    monkeypatch.setattr(places, "_nominatim", fake)
    rows = client.get("/api/geocode", params={"q": "horseshoe bend"}).json()["results"]
    assert rows == [{"name": "Horseshoe Bend", "label": "Horseshoe Bend, Coconino County, Arizona",
                     "latitude": 36.8791, "longitude": -111.5104, "state": "Arizona"}]
    assert calls[0][1]["countrycodes"] == "us"
    assert client.get("/api/geocode", params={"q": "ab"}).json()["results"] == []
    assert client.get("/api/geocode/reverse", params={"lat": 36.8, "lon": -111.5}).json()["name"] == "Coconino County"


def test_old_queue_database_gets_the_place_column(tmp_path):
    import sqlite3
    from server.db import Jobs
    path = tmp_path / "q.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL, preset TEXT "
                 "NOT NULL, person TEXT NOT NULL, photo TEXT NOT NULL, stages TEXT NOT NULL DEFAULT '{}', previews TEXT "
                 "NOT NULL DEFAULT '[]', message TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '', result TEXT "
                 "NOT NULL DEFAULT '', started REAL, finished REAL, heartbeat REAL, cancel_requested INTEGER NOT NULL "
                 "DEFAULT 0)")
    conn.execute("INSERT INTO jobs (id, created, status, preset, person, photo) VALUES ('a', 1, 'done', 'x', '', 'p')")
    conn.commit()
    conn.close()
    jobs = Jobs(path)
    assert jobs.get("a")["place"] == {}
    assert jobs.create("horseshoe-bend", "", "p.jpg", {"title": "Horseshoe Bend"})["place"]["title"] == "Horseshoe Bend"


def test_parked_places_are_catalogue_ids():
    import importlib
    places = importlib.import_module("app.server.places")
    ids = {p["id"] for p in places.load_catalogue(places.APP_DIR.parent)}
    assert set(places.PARKED) <= ids


def test_quality_choice_reaches_the_worker_and_the_manifest(app_env, tmp_path):
    client, worker, *_ = app_env
    assert client.post("/api/jobs", data={"preset": "monument-valley", "quality": "8k"},
                       files={"photo": ("me.jpg", jpeg(), "image/jpeg")}).status_code == 400
    job = client.post("/api/jobs", data={"preset": "monument-valley", "quality": "draft"},
                      files={"photo": ("me.jpg", jpeg(), "image/jpeg")}).json()
    assert job["quality"] == "draft"
    assert worker.claim()["quality"] == "draft"
    preset = ROOT / "shots/flux3/monument-valley.toml"
    photo = tmp_path / "subject.jpg"
    photo.write_bytes(jpeg())
    m = runner.write_job_manifest(preset, tmp_path / "job.toml", photo, "p", "n", tmp_path, "draft")
    assert m["video"]["quality"] == "draft"
    assert runner.write_job_manifest(preset, tmp_path / "j2.toml", photo, "p", "n", tmp_path)["video"]["quality"] == "full"


def test_a_failed_build_is_explained_in_a_sentence():
    """A judge should not have to read a CUDA traceback."""
    from worker.runner import explain
    assert "ran out of memory" in explain("torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory")
    assert "comfy-aimdo" in explain("RuntimeError: Fault failed: 2")
    assert "could not reach ComfyUI" in explain("Could not reach ComfyUI to submit the job.")
    assert "cv2" in explain("ModuleNotFoundError: No module named 'cv2'")
    assert "no level ground" in explain("No plausible level landing patch near the catalogue coordinate")
    assert explain("something else entirely").startswith("The build failed")
