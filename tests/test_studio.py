"""The photographic studio must save real choices without submitting terrain jobs."""
import io
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from server import places, studio, location_photos
from server.config import Settings
from server.main import create_app


def photo():
    buf = io.BytesIO()
    image = Image.new("RGB", (640, 480), "#b4c099")
    exif = Image.Exif()
    exif[0x010F] = "Private camera metadata"
    image.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    def no_photos(*args, **kwargs):
        raise ValueError("No test photos")
    monkeypatch.setattr(location_photos, "fetch_place_photos", no_photos)
    # Chichen Itza's curated Commons photos live under outputs/ on the development machine only. A fresh clone gets
    # stand-ins for the test and has them removed afterwards, so the default place still has its two photos.
    landmark = next(p for p in json.loads((ROOT / "app/server/studio_landmarks.json").read_text()) if p["id"] == "chichen-itza")
    stand_ins = [ROOT / r["path"] for r in landmark["references"] if not (ROOT / r["path"]).is_file()]
    for path in stand_ins:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(photo())
    settings = Settings(data_dir=tmp_path / "data", repo=ROOT, worker_token="worker", gallery_seed=[tmp_path / "none"])
    app = create_app(settings)
    try:
        with TestClient(app) as client:
            yield client, settings, app
    finally:
        for path in stand_ins:
            path.unlink(missing_ok=True)


def save(client, *, destination=None, data=None, subject=True, references=0):
    fields = {"destination_json": json.dumps(destination or {"id": "chichen-itza"}),
              "ending": "relaxed", "framing": "medium", **(data or {})}
    files = []
    if subject:
        files.append(("subject", ("untrusted../name.jpg", photo(), "image/jpeg")))
    files += [("references", (f"place-{i}.jpg", photo(), "image/jpeg")) for i in range(references)]
    return client.post("/api/studio/setups", data=fields, files=files)


def test_curated_config_uses_photos_and_never_claims_rendering_is_ready(env):
    client, _, _ = env
    config = client.get("/api/studio/config").json()
    assert config["render_available"] is False
    assert [p["id"] for p in config["landmarks"]] == ["chichen-itza", "eiffel-tower", "canyonlands"]
    for place in config["landmarks"]:
        assert place["thumb"].endswith(".jpg") and place["credit"]["license_url"]
        assert "example" not in place and "references" not in place
        assert client.get(place["thumb"]).status_code == 200


def test_destination_menu_includes_all_seven_modern_wonders_and_city_views(env):
    client, _, _ = env
    rows = client.get("/api/studio/config").json()["destinations"]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    wonders = {row["id"] for row in rows if row["group"] == "Seven wonders · modern"}
    assert wonders == {"chichen-itza", "great-wall", "petra", "colosseum", "taj-mahal", "machu-picchu", "christ-the-redeemer"}
    cities = {row["id"] for row in rows if row["group"] == "City views"}
    assert {"new-york-skyline", "london-westminster", "tokyo-shibuya"} <= cities


@pytest.mark.parametrize("place_id,title,country", [
    ("taj-mahal", "Taj Mahal", "India"),
    ("mount-fuji-kawaguchi", "Mount Fuji", "Japan"),
    ("tokyo-shibuya", "Tokyo · Shibuya", "Japan"),
])
def test_new_menu_destinations_are_canonical_and_reopen_without_fake_references(env, place_id, title, country):
    client, _, app = env
    response = save(client, destination={"id": place_id, "title": "Spoofed place", "latitude": 0, "longitude": 0})
    assert response.status_code == 200, response.text
    record = response.json()
    assert (record["destination"]["id"], record["destination"]["title"], record["destination"]["country"]) == (place_id, title, country)
    assert record["destination"]["latitude"] != 0
    assert record["preparation"] == "needs_location_photos"
    assert record["default_references"] == []
    assert client.get(f"/api/studio/setups/{record['id']}").json() == record
    assert app.state.jobs.active_count() == 0


def test_save_preserves_choices_strips_metadata_and_never_queues_a_build(env):
    client, settings, app = env
    response = save(client, data={"ending": "model", "framing": "full", "wearing": "  a blue jacket  "}, references=1)
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["workflow"] == "photographic-grounded-v1"
    assert (record["ending"], record["framing"], record["wearing"]) == ("model", "full", "a blue jacket")
    assert record["destination"]["country"] == "Mexico"
    assert record["preparation"] == "needs_reference_review"
    assert record["render_available"] is False
    assert client.get(f"/api/studio/setups/{record['id']}").json() == record
    result = client.get(record["subject"]["url"])
    assert result.headers["cache-control"] == "private, no-store"
    with Image.open(io.BytesIO(result.content)) as image:
        assert not image.getexif()
    assert "untrusted" not in json.dumps(record)
    assert app.state.jobs.active_count() == 0
    assert client.post("/api/worker/claim", headers={"Authorization": "Bearer worker"}).status_code == 204
    assert client.post(f"/api/studio/setups/{record['id']}/render").status_code == 503
    assert app.state.jobs.active_count() == 0
    assert len(list((settings.data_dir / "setups").glob("*/setup.json"))) == 1


def test_custom_place_needs_no_lidar_and_can_resume_with_uploaded_references(env, monkeypatch):
    client, _, _ = env
    monkeypatch.setattr(places, "check_point", lambda *_: pytest.fail("The photographic path must not check terrain coverage"))
    place = {"title": "A garden in Kyoto", "latitude": 35.01, "longitude": 135.76}
    first = save(client, destination=place, references=1).json()
    assert first["preparation"] == "needs_location_photos"
    reference = first["references"][0]
    resumed = save(client, destination=place, subject=False, references=1,
                   data={"base": first["id"], "retained": json.dumps([reference["name"]]), "ending": "look-back", "framing": "close"})
    assert resumed.status_code == 200, resumed.text
    second = resumed.json()
    assert second["id"] != first["id"]
    assert second["preparation"] == "needs_reference_review"
    assert len(second["references"]) == 2 and second["ending"] == "look-back" and second["framing"] == "close"
    assert client.get(second["subject"]["url"]).content == client.get(first["subject"]["url"]).content


def test_cannot_retain_location_references_after_changing_destination(env):
    client, _, _ = env
    first = save(client, references=1).json()
    response = save(client, destination={"title": "A different garden"}, subject=False,
                    data={"base": first["id"], "retained": json.dumps([first["references"][0]["name"]])})
    assert response.status_code == 422


@pytest.mark.parametrize("fields", [
    {"ending": "twirl", "framing": "close"}, {"ending": "arms-open", "framing": "medium"},
    {"ending": "invented"}, {"framing": "invented"}, {"retained": "{}"},
    {"retained": '["../../outside.jpg"]'}, {"destination_json": "[]"},
    {"destination_json": '{"title":"Moon","latitude":"NaN","longitude":0}'},
])
def test_invalid_or_incompatible_setups_are_rejected_without_saving(env, fields):
    client, settings, _ = env
    assert save(client, data=fields).status_code == 422
    assert not list((settings.data_dir / "setups").glob("*/setup.json"))


def test_failed_upload_cleans_partial_files(env):
    client, settings, _ = env
    response = client.post("/api/studio/setups", data={"destination_json": '{"id":"eiffel-tower"}'},
                           files=[("subject", ("portrait.jpg", photo(), "image/jpeg")),
                                  ("references", ("fake.jpg", b"not an image", "image/jpeg"))])
    assert response.status_code == 422
    assert not list((settings.data_dir / "setups").iterdir())


@pytest.mark.parametrize("fmt,extension,mime", [("HEIF", "heic", "image/heic"), ("HEIF", "heif", "application/octet-stream"), ("WEBP", "webp", "image/webp")])
def test_new_formats_work_for_portraits_references_and_ephemeral_previews(env, fmt, extension, mime):
    client, settings, _ = env
    image = Image.new("RGB", (120, 80), "red")
    exif = Image.Exif(); exif[274] = 6; exif[271] = "Private camera"
    buf = io.BytesIO(); image.save(buf, fmt, exif=exif.tobytes())
    data = buf.getvalue()
    preview = client.post("/api/studio/images/normalize", files={"file": (f"photo.{extension}", data, mime)})
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/jpeg"
    assert preview.headers["cache-control"] == "private, no-store"
    with Image.open(io.BytesIO(preview.content)) as clean:
        assert clean.size == (80, 120) and not clean.getexif()
    assert not list(settings.data_dir.glob(".preview-*"))
    response = client.post("/api/studio/setups", data={"destination_json": '{"id":"chichen-itza"}'},
                           files=[("subject", (f"photo.{extension}", data, mime)),
                                  ("references", (f"place.{extension}", data, mime))])
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["subject"]["name"].endswith(".jpg")
    assert record["references"][0]["name"].endswith(".jpg")
    for item in [record["subject"], *record["references"]]:
        with Image.open(io.BytesIO(client.get(item["url"]).content)) as clean:
            assert clean.size == (80, 120) and not clean.getexif()


def test_preview_rejects_invalid_and_oversized_uploads_without_leaving_files(env):
    client, settings, _ = env
    assert client.post("/api/studio/images/normalize", files={"file": ("wrong.heic", b"invalid", "image/heic")}).status_code == 422
    settings.max_upload_mb = 0.001
    assert client.post("/api/studio/images/normalize", files={"file": ("photo.webp", photo(), "image/webp")}).status_code == 413
    assert not list(settings.data_dir.glob(".preview-*"))


def test_reference_limit_and_upload_size_are_enforced(env):
    client, settings, _ = env
    assert save(client, references=5).status_code == 422
    settings.max_upload_mb = 0.001
    assert save(client).status_code == 413


def test_private_setup_routes_and_images_require_passcode(tmp_path):
    settings = Settings(data_dir=tmp_path, repo=ROOT, passcode="private", gallery_seed=[tmp_path / "none"])
    with TestClient(create_app(settings)) as owner:
        owner.post("/api/login", json={"passcode": "private"})
        record = save(owner).json()
        with TestClient(create_app(settings)) as visitor:
            for path in ["/api/studio/config", f"/api/studio/setups/{record['id']}", record["subject"]["url"]]:
                assert visitor.get(path).status_code == 401
            assert save(visitor).status_code == 401
            assert visitor.post("/api/studio/location-photos", json={"destination": {"id": "taj-mahal"}}).status_code == 401
            assert visitor.post("/api/studio/images/normalize", files={"file": ("p.jpg", photo(), "image/jpeg")}).status_code == 401
        assert owner.get(f"/api/studio/setups/{record['id']}/images/reference-3.jpg").status_code == 404
        assert owner.get("/api/studio/setups/not-a-setup").status_code == 404


def test_global_search_is_explicit_and_has_no_us_filter(env, monkeypatch):
    client, _, _ = env
    calls = []
    def search(path, params):
        calls.append(params)
        return [{"name": "Kyoto", "display_name": "Kyoto, Japan", "lat": "35", "lon": "135", "address": {}}]
    monkeypatch.setattr(places, "_nominatim", search)
    assert client.get("/api/studio/search?q=ab").json() == {"results": []}
    assert not calls
    response = client.get("/api/studio/search?q=Kyoto")
    assert response.json()["results"][0]["name"] == "Kyoto"
    assert "countrycodes" not in calls[0]
    monkeypatch.setattr(places, "_nominatim", lambda *_: (_ for _ in ()).throw(OSError()))
    response = client.get("/api/studio/search?q=Kyoto")
    assert response.status_code == 502 and "temporarily unavailable" in response.json()["detail"]


def test_missing_examples_are_not_advertised_and_example_allowlist_supports_ranges(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", repo=tmp_path / "repo", gallery_seed=[tmp_path / "none"])
    with TestClient(create_app(settings)) as client:
        assert all(p["example_url"] is None for p in client.get("/api/studio/config").json()["landmarks"])
        assert client.get("/api/studio/examples/chichen-itza").status_code == 404
        source = settings.repo / studio.LANDMARKS[0]["example"]
        source.parent.mkdir(parents=True)
        source.write_bytes(b"1234567890")
        response = client.get("/api/studio/examples/chichen-itza", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206 and response.content == b"3456"
        assert client.get("/api/studio/examples/not-allowed").status_code == 404


@pytest.fixture
def commons(monkeypatch):
    calls = []

    def fetch(name, lat, lon, count, folder, **kwargs):
        calls.append((name, lat, lon, kwargs))
        folder.mkdir(parents=True)
        selected = []
        for i, color in enumerate(["red", "green", "blue"]):
            path = folder / f"commons_{i}.jpg"
            Image.new("RGB", (960, 640), color).save(path)
            selected.append({"page_id": i, "local_path": str(path), "title": f"View {i}",
                             "artist": f"Photographer {i}", "license": "CC BY-SA 4.0",
                             "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                             "commons_page": f"https://commons.wikimedia.org/wiki/File:View_{i}.jpg"})
        return {"selected": selected}

    monkeypatch.setattr(location_photos, "fetch_place_photos", fetch)
    return calls


def test_discovery_caches_real_images_and_preserves_credits(env, commons):
    client, settings, _ = env
    request = {"destination": {"id": "taj-mahal", "latitude": 0, "longitude": 0}}
    batch = client.post("/api/studio/location-photos", json=request).json()
    assert batch["status"] == "found" and len(batch["photos"]) == 3
    assert commons[0][:3] == ("Taj Mahal", 27.1751, 78.0421)
    assert commons[0][3]["purpose"] == "photo_op"
    assert client.post("/api/studio/location-photos", json=request).json() == batch
    assert len(commons) == 1
    for p in batch["photos"]:
        result = client.get(p["image_url"])
        assert result.status_code == 200
        assert hashlib.sha256(result.content).hexdigest() == p["sha256"]
        assert p["artist"] and p["license_url"] and p["url"].startswith("https://commons.wikimedia.org/")
        assert "local_path" not in p
    assert not list((settings.data_dir / "location-photos").glob("*/raw"))
    assert client.get(f"/api/studio/location-photos/{batch['id']}/references.json").status_code == 404


def test_saving_without_uploads_discovers_and_snapshots_references(env, commons):
    client, settings, _ = env
    first = save(client, destination={"id": "taj-mahal"}).json()
    assert len(first["default_references"]) == 3 and first["references"] == []
    assert first["preparation"] == "needs_reference_review"
    original = client.get(first["default_references"][0]["image_url"]).content
    shutil.rmtree(settings.data_dir / "location-photos")
    assert client.get(first["default_references"][0]["image_url"]).content == original
    second = save(client, destination={"id": "taj-mahal"}, subject=False, data={
        "base": first["id"], "reference_set": first["reference_set"],
        "automatic": json.dumps([p["sha256"] for p in first["default_references"][:2]])}).json()
    assert len(commons) == 1  # revisions reopen without rediscovering their evidence
    assert len(second["default_references"]) == 2
    assert client.get(second["default_references"][0]["image_url"]).content == original
    assert client.get(f"/api/studio/setups/{second['id']}").json() == second


def test_removed_automatic_photo_can_be_added_back_and_destination_is_bound(env, commons):
    client, _, _ = env
    batch = client.post("/api/studio/location-photos", json={"destination": {"id": "taj-mahal"}}).json()
    fields = {"reference_set": batch["id"], "automatic": json.dumps([batch["photos"][0]["sha256"]])}
    first = save(client, destination={"id": "taj-mahal"}, data=fields).json()
    assert len(first["default_references"]) == 1
    assert first["preparation"] == "needs_location_photos"
    second = save(client, destination={"id": "taj-mahal"}, subject=False, data={
        **fields, "base": first["id"], "automatic": json.dumps([p["sha256"] for p in batch["photos"]])})
    assert second.status_code == 200 and len(second.json()["default_references"]) == 3
    assert save(client, destination={"id": "petra"}, data=fields).status_code == 422
    assert save(client, destination={"id": "taj-mahal"}, data={**fields, "automatic": '["invented-hash"]'}).status_code == 422


@pytest.mark.parametrize("exception,status", [(ValueError("none"), "empty"), (OSError("offline"), "unavailable"),
                                              (RuntimeError("API busy"), "unavailable")])
def test_empty_or_failed_search_is_cached_retryable_and_uploads_remain_available(env, monkeypatch, exception, status):
    client, _, _ = env
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        raise exception
    monkeypatch.setattr(location_photos, "fetch_place_photos", fetch)
    request = {"destination": {"id": "petra"}}
    batch = client.post("/api/studio/location-photos", json=request).json()
    assert batch["status"] == status and batch["photos"] == [] and "own" in batch["message"]
    client.post("/api/studio/location-photos", json=request)
    assert len(calls) == 1
    client.post("/api/studio/location-photos", json={**request, "retry": True})
    assert len(calls) == 2
    saved = save(client, destination={"id": "petra"}, references=2)
    assert saved.status_code == 200 and saved.json()["preparation"] == "needs_reference_review"


def test_custom_name_needs_a_unique_result_before_automatic_references(env, commons, monkeypatch):
    client, _, _ = env
    rows = [{"latitude": 35.01, "longitude": 135.76}]
    monkeypatch.setattr(places, "geocode", lambda q, **kw: rows)
    first = client.post("/api/studio/location-photos", json={"destination": {"title": "Kyoto garden"}}).json()
    assert first["status"] == "found" and commons[0][1:3] == (35.01, 135.76)
    rows.append({"latitude": 0, "longitude": 0})
    second = client.post("/api/studio/location-photos", json={"destination": {"title": "Another garden"}}).json()
    assert second["status"] == "empty" and len(commons) == 1
