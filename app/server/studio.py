"""Photographic studio: curated examples and private, resumable creative setups.

Saving a setup never enqueues a build. Create explicitly queues the grounded
canonical workflow for a capable worker; uploading a reference does not establish
that it passed a visual check.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import places
from .images import BadImage, sanitize
from .location_photos import ReferenceLibrary, destination_key

LANDMARKS = json.loads(Path(__file__).with_name("studio_landmarks.json").read_text())
DESTINATIONS = json.loads(Path(__file__).with_name("studio_destinations.json").read_text())
ENDINGS = [
    {"id": "relaxed", "label": "Relaxed", "description": "A natural stance and a small smile."},
    {"id": "model", "label": "Model pose", "description": "Hand on hip, a small turn toward the camera."},
    {"id": "look-back", "label": "Look back", "description": "Look at the landmark, then turn toward the camera."},
    {"id": "wave", "label": "Wave", "description": "A small greeting as the camera arrives.", "trial": True},
    {"id": "arms-open", "label": "Arms open", "description": "Open your arms into the view.", "trial": True},
    {"id": "twirl", "label": "Twirl", "description": "One turn, finishing toward the camera.", "trial": True},
]
FRAMINGS = [
    {"id": "full", "label": "Full body", "description": "Head to feet"},
    {"id": "three-quarter", "label": "Three-quarter", "description": "Knees up"},
    {"id": "medium", "label": "Waist-up", "description": "Medium shot"},
    {"id": "close", "label": "Close portrait", "description": "Head and shoulders"},
]
SETUP_RE = re.compile(r"^[a-f0-9]{32}$")
ASSET_RE = re.compile(r"^(subject|reference-[0-3]|location-[0-3])\.(jpg|png)$")


class ReferenceRequest(BaseModel):
    destination: dict
    retry: bool = False


def load_record(settings, setup_id: str) -> dict:
    if not SETUP_RE.fullmatch(setup_id):
        raise HTTPException(404, "No saved setup with that link.")
    try:
        return json.loads((settings.data_dir / "setups" / setup_id / "setup.json").read_text())
    except FileNotFoundError:
        raise HTTPException(404, "No saved setup with that link.") from None


def worker_setup(settings, record: dict) -> dict:
    folder = settings.data_dir / "setups" / record["id"]
    def photo(name, metadata=None):
        if not ASSET_RE.fullmatch(name) or not (folder / name).is_file():
            raise HTTPException(409, "A saved photo is unavailable. Save this setup again before creating a film.")
        sha = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if metadata and metadata.get("sha256") and metadata["sha256"] != sha:
            raise HTTPException(409, "A saved reference changed. Save this setup again.")
        return {**(metadata or {}), "name": name, "sha256": sha}
    references = [photo(p["name"], p) for p in record.get("default_references", []) if p.get("name")]
    references += [photo(n, {"source": "upload", "title": "User-supplied location photo"}) for n in record["references"]]
    return {k: record[k] for k in ("id", "destination", "ending", "framing", "wearing")} | {
        "subject": photo(record["subject"]), "references": references}


def destination(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Choose a place first.")
    known = next((p for p in DESTINATIONS if p["id"] == value.get("id")), None)
    if known:
        return {k: known[k] for k in ("id", "title", "region", "country", "latitude", "longitude")}
    title = places.clean_name(value.get("title", ""))
    if len(title) < 3:
        raise ValueError("Give your place a name of at least three characters.")
    result = {"id": "custom", "title": title, "region": places.clean_name(value.get("region", "")),
              "country": places.clean_name(value.get("country", ""))}
    lat, lon = value.get("latitude"), value.get("longitude")
    if lat is not None or lon is not None:
        lat, lon = float(lat), float(lon)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("That place has invalid coordinates.")
        result.update(latitude=lat, longitude=lon)
    return result


def install(app, settings, user, public_job) -> None:
    router = APIRouter(prefix="/api/studio", dependencies=[Depends(user)])
    store = settings.data_dir / "setups"
    library = ReferenceLibrary(settings, LANDMARKS)
    jobs = app.state.jobs

    def available():
        return bool(settings.worker_token) and time.time() - app.state.canonical_worker_seen < 45

    async def read_upload(file: UploadFile) -> bytes:
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"Each photo must be under {settings.max_upload_mb:g} MB.")
        return data

    def clean_upload(data: bytes, stem: Path) -> Path:
        try:
            return sanitize(data, stem)
        except BadImage as e:
            raise HTTPException(422, str(e)) from e

    @router.post("/images/normalize")
    async def normalize_image(file: UploadFile = File(...)):
        # Browser HEIF support varies. Return a clean preview without retaining
        # either the original upload or a temporary image on the server.
        data = await read_upload(file)
        with tempfile.TemporaryDirectory(prefix=".preview-", dir=settings.data_dir) as temp:
            path = clean_upload(data, Path(temp) / "preview")
            return Response(path.read_bytes(), media_type="image/png" if path.suffix == ".png" else "image/jpeg")

    def load(setup_id: str) -> dict:
        return load_record(settings, setup_id)

    def public(record: dict) -> dict:
        out = {k: v for k, v in record.items() if k not in ("subject", "references")}
        prefix = f"/api/studio/setups/{record['id']}/images/"
        out["subject"] = {"name": record["subject"], "url": prefix + record["subject"]}
        out["references"] = [{"name": n, "url": prefix + n} for n in record["references"]]
        out["default_references"] = [{**p, "image_url": prefix + p["name"]} if p.get("name") else p
                                     for p in record.get("default_references", [])]
        out["render_available"] = available()
        latest = jobs.latest_setup(record["id"])
        out["latest_job"] = public_job(latest) if latest else None
        return out

    @router.post("/location-photos")
    def location_photos(request: ReferenceRequest):
        try:
            dest = destination(request.destination)
        except (ValueError, TypeError, OverflowError):
            raise HTTPException(422, "Choose a valid place first.") from None
        return library.public(library.discover(dest, request.retry))

    @router.get("/location-photos/{key}/{name}")
    def location_photo(key: str, name: str):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise HTTPException(404, "No such location photo.")
        try:
            record = library.load(key)
        except FileNotFoundError:
            raise HTTPException(404, "No such location photo.") from None
        if name not in {p["name"] for p in record["photos"]}:
            raise HTTPException(404, "No such location photo.")
        return FileResponse(library.root / key / name)

    @router.get("/config")
    def config():
        rows = []
        for item in LANDMARKS:
            row = {k: v for k, v in item.items() if k not in ("example", "references")}
            row["example_url"] = (f"/api/studio/examples/{item['id']}"
                                  if (settings.repo / item["example"]).is_file() else None)
            row["reference_count"] = sum((settings.repo / r["path"]).is_file() for r in item["references"])
            rows.append(row)
        return {"landmarks": rows, "destinations": DESTINATIONS, "endings": ENDINGS, "framings": FRAMINGS,
                "render_available": available(), "max_references": 4, "max_upload_mb": settings.max_upload_mb}

    @router.get("/availability")
    def availability():
        return {"render_available": available()}

    @router.get("/examples/{landmark_id}")
    def example(landmark_id: str):
        item = next((p for p in LANDMARKS if p["id"] == landmark_id), None)
        path = settings.repo / item["example"] if item else None
        if path is None or not path.is_file():
            raise HTTPException(404, "This example isn't installed on this server.")
        return FileResponse(path, media_type="video/mp4")

    @router.get("/search")
    def search(q: str = ""):
        try:
            return {"results": places.geocode(q, country=None)}
        except Exception as e:
            raise HTTPException(502, "Place search is temporarily unavailable. Try again, or use the place name "
                                "and add your own location photos.") from e

    @router.get("/setups/{setup_id}")
    def get_setup(setup_id: str):
        return public(load(setup_id))

    @router.get("/setups/{setup_id}/images/{name}")
    def image(setup_id: str, name: str):
        record = load(setup_id)
        allowed = [record["subject"], *record["references"],
                   *[p["name"] for p in record.get("default_references", []) if p.get("name")]]
        if not ASSET_RE.fullmatch(name) or name not in allowed:
            raise HTTPException(404, "No such photo.")
        return FileResponse(store / setup_id / name, headers={"Cache-Control": "private, no-store"})

    @router.post("/setups")
    async def save_setup(destination_json: str = Form(...), ending: str = Form("relaxed"),
                         framing: str = Form("medium"), wearing: str = Form(""),
                         base: str = Form(""), retained: str = Form("[]"),
                         reference_set: str = Form(""), automatic: str = Form("null"),
                         subject: UploadFile | None = File(None),
                         references: list[UploadFile] = File(default=[])):
        try:
            dest = destination(json.loads(destination_json))
            keep = json.loads(retained)
            auto_keep = json.loads(automatic)
            if auto_keep is not None and (not isinstance(auto_keep, list) or
                    any(not isinstance(n, str) for n in auto_keep) or len(set(auto_keep)) != len(auto_keep)):
                raise ValueError("Invalid automatic photo selection.")
            if not isinstance(keep, list) or any(not isinstance(n, str) for n in keep) or len(set(keep)) != len(keep):
                raise ValueError("The saved photo selection is invalid.")
        except (ValueError, TypeError, OverflowError):
            raise HTTPException(422, "Choose a valid place and photo selection.") from None
        if ending not in {e["id"] for e in ENDINGS} or framing not in {f["id"] for f in FRAMINGS}:
            raise HTTPException(422, "Choose one of the available endings and framings.")
        if ending in ("twirl", "arms-open") and framing not in ("full", "three-quarter"):
            raise HTTPException(422, "Choose full body or three-quarter so there is room for this movement.")
        previous = load(base) if base else None
        if keep and (not previous or any(n not in previous["references"] for n in keep)):
            raise HTTPException(422, "Those saved location photos aren't part of this setup.")
        if previous and keep and previous["destination"] != dest:
            raise HTTPException(422, "Choose location photos for the new place.")
        if len(keep) + len(references) > 4:
            raise HTTPException(422, "Add up to four location photos.")
        if not subject and not previous:
            raise HTTPException(422, "Add the person's photo first.")
        store.mkdir(parents=True, exist_ok=True)
        setup_id = secrets.token_hex(16)
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=store) as temp:
            folder = Path(temp)

            async def upload(file: UploadFile, stem: str) -> str:
                return clean_upload(await read_upload(file), folder / stem).name

            def copy(name: str, stem: str) -> str:
                target = stem + Path(name).suffix
                shutil.copyfile(store / base / name, folder / target)
                return target

            photo = await upload(subject, "subject") if subject else copy(previous["subject"], "subject")
            refs = [copy(n, f"reference-{i}") for i, n in enumerate(keep)]
            for file in references:
                refs.append(await upload(file, f"reference-{len(refs)}"))
            key = destination_key(dest)
            if reference_set and reference_set != key:
                raise HTTPException(422, "Choose location photos for the new place.")
            # Revisions retain their own immutable copies even if the discovery cache is gone.
            previous_hashes = {p["sha256"] for p in previous.get("default_references", []) if p.get("sha256")} if previous else set()
            if (previous and previous.get("reference_set") == key and reference_set == key
                    and (auto_keep is None or set(auto_keep) <= previous_hashes)):
                batch = {"photos": previous["default_references"],
                         "status": previous.get("reference_search", "found")}
                source_folder = store / base
            else:
                batch = await run_in_threadpool(library.discover, dest)
                source_folder = library.root / key
            defaults = batch["photos"]
            hashes = {p["sha256"] for p in defaults}
            if auto_keep is not None and any(n not in hashes for n in auto_keep):
                raise HTTPException(422, "Those location photos aren't part of this place's selection.")
            default_refs = []
            for item in defaults:
                if auto_keep is not None and item["sha256"] not in auto_keep:
                    continue
                shutil.copyfile(source_folder / item["name"], folder / item["name"])
                default_refs.append(item)
            count = len(refs) + len(default_refs)
            record = {"id": setup_id, "created": time.time(), "workflow": "photographic-grounded-v1",
                      "destination": dest, "ending": ending, "framing": framing,
                      "wearing": " ".join(wearing.split())[:300], "subject": photo, "references": refs,
                      "default_references": default_refs,
                      "reference_set": key, "reference_search": batch["status"],
                      "preparation": "needs_location_photos" if count < 2 else "needs_reference_review",
                      "render_available": False}
            (folder / "setup.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            folder.rename(store / setup_id)
        return public(record)

    @router.post("/setups/{setup_id}/render")
    def render(setup_id: str, retry: bool = False):
        record = load(setup_id)
        if not available():
            raise HTTPException(503, "The generation worker is offline. Your setup is saved; try again when the worker is running.")
        inputs = worker_setup(settings, record)
        if len(inputs["references"]) < 2:
            raise HTTPException(422, "Add at least two location photos, then save this setup before creating a film.")
        jobs.sweep(settings.worker_timeout_s)
        with jobs.lock:
            latest = jobs.latest_setup(setup_id)
            if latest and (latest["status"] in ("queued", "running") or (latest["status"] == "done" and not retry)):
                return public_job(latest)
            if jobs.active_count() >= settings.max_queue:
                raise HTTPException(429, "The queue is full. Your setup is saved; please try again later.")
            job = jobs.create(record["destination"]["id"], record["wearing"], record["subject"],
                              {**record["destination"], "title": record["destination"]["title"]},
                              workflow="photographic-grounded-v1", setup_id=setup_id)
        return public_job(job)

    app.include_router(router)
