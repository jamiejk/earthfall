"""Touchdown web app: the front end, the JSON API, the job queue and the files.

User API (passcode-gated when TOUCHDOWN_PASSCODE is set):
    GET  /api/config                      presets, catalogue places, whether a passcode is needed
    POST /api/login                       {"passcode": "..."} -> cookie
    GET  /api/custom/check?lat=&lon=      can a custom US point be built (1 m lidar)?
    GET  /api/geocode?q=, /api/geocode/reverse?lat=&lon=   place names (OpenStreetMap Nominatim, via the server)
    POST /api/jobs                        multipart: preset (preset id, catalogue id or "custom"; custom also takes
                                          latitude, longitude, name), wearing, photo -> job
    GET  /api/jobs/{id}                   job + queue position
    GET  /api/jobs/{id}/events            server-sent events: the job, whenever it changes
    POST /api/jobs/{id}/cancel
    GET  /api/jobs/{id}/video             the finished mp4 (?download=1 for an attachment)
    GET  /api/jobs/{id}/previews/{name}   stage preview images
    GET  /api/gallery, /api/gallery/{file}

Worker API (Authorization: Bearer $TOUCHDOWN_WORKER_TOKEN); the worker pulls, so it needs no inbound port:
    POST /api/worker/claim                -> 200 job, or 204 when there is nothing to do
    GET  /api/worker/jobs/{id}/photo
    POST /api/worker/jobs/{id}/progress   {"stages": {...}, "message": "..."} -> {"cancel": bool}
    POST /api/worker/jobs/{id}/preview    multipart: name, file
    POST /api/worker/jobs/{id}/result     multipart: file (the mp4)
    POST /api/worker/jobs/{id}/finish     {"status": "done"|"failed"|"cancelled", "error": "..."}
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import shutil
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import places, presets, studio, enhancement, checkpoints
from .config import Settings
from .db import Jobs
from .images import BadImage, sanitize

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(mp4|jpg|png|webm)$")
MAX_WEARING = 300
COOKIE = "td_pass"


def public_job(jobs: Jobs, job: dict) -> dict:
    out = {k: job[k] for k in ("id", "status", "preset", "stages", "message", "error", "created", "started",
                               "finished", "cancel_requested", "quality", "workflow", "setup_id")}
    preset = presets.by_id(job["preset"])
    out["title"] = preset["title"] if preset else job["place"].get("title") or job["preset"]
    out["position"] = jobs.position(job)
    out["previews"] = [{"name": n, "url": f"/api/jobs/{job['id']}/previews/{n}"} for n in job["previews"]]
    out["video"] = f"/api/jobs/{job['id']}/video" if job["status"] == "done" and job["result"] else None
    out["review_video"] = f"/api/jobs/{job['id']}/video" if job["status"] == "held" and job["result"] else None
    out["setup_url"] = f"/#setup={job['setup_id']}" if job["setup_id"] else None
    out["review_url"] = f"/api/jobs/{job['id']}/review" if job.get("workflow") in ("photographic-v1", checkpoints.WORKFLOW) else None
    out["parent_id"] = job.get("parent_id") or None
    return out


def seed_gallery(settings: Settings) -> None:
    """Copy the example clips into the gallery folder, with their titles and posters.

    A clip's title comes from its sidecar ``<name>.json`` and its poster from ``<name>.jpg``; without them the
    gallery shows file names, which is not what a judge should read.
    """
    settings.gallery_dir.mkdir(parents=True, exist_ok=True)
    for folder in settings.gallery_seed:
        if not folder.is_dir():
            continue
        for mp4 in sorted(folder.rglob("*.mp4")):
            target = settings.gallery_dir / re.sub(r"[^A-Za-z0-9._-]", "-", f"{mp4.parent.name}-{mp4.name}")
            if not target.exists() and FILE_RE.match(target.name):
                shutil.copy(mp4, target)
                for suffix in (".json", ".jpg"):  # the title and the poster, if the clip has them
                    sidecar = mp4.with_suffix(suffix)
                    if sidecar.is_file():
                        shutil.copy(sidecar, target.with_suffix(suffix))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    for d in (settings.uploads, settings.jobs_dir, settings.gallery_dir):
        d.mkdir(parents=True, exist_ok=True)
    seed_gallery(settings)
    jobs = Jobs(settings.db_path)
    app = FastAPI(title="Touchdown", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.photo_worker_seen = 0.0
    app.state.canonical_worker_seen = 0.0

    def pass_cookie() -> str:
        return hmac.new(settings.cookie_secret.encode(), settings.passcode.encode(), hashlib.sha256).hexdigest()

    def user(request: Request) -> None:
        if not settings.passcode:
            return
        given = request.cookies.get(COOKIE, "")
        header = request.headers.get("x-touchdown-passcode", "")
        if hmac.compare_digest(given, pass_cookie()) or (header and hmac.compare_digest(header, settings.passcode)):
            return
        raise HTTPException(401, "passcode required")

    def worker(request: Request) -> None:
        token = settings.worker_token
        auth = request.headers.get("authorization", "")
        if not token:
            raise HTTPException(503, "no worker token is configured on the server")
        if not hmac.compare_digest(auth, f"Bearer {token}"):
            raise HTTPException(401, "bad worker token")

    def job_or_404(job_id: str) -> dict:
        jobs.sweep(settings.worker_timeout_s)
        job = jobs.get(job_id) if NAME_RE.match(job_id.lower()) else None
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if request.url.path.startswith(("/api/studio/", "/api/jobs/", "/api/worker/")):
            response.headers.setdefault("Cache-Control", "private, no-store")
        return response

    # --- user API ------------------------------------------------------------------------------------------

    @app.get("/api/config")
    def config():
        return {"passcode_required": bool(settings.passcode), "presets": presets.public(settings.repo),
                "places": places.public(settings), "default_person": presets.FALLBACK_PERSON,
                "custom_location": presets.CUSTOM_LOCATION, "max_upload_mb": settings.max_upload_mb,
                # True when the worker is running the stand-in build instead of the real pipeline, so the page can
                # say so instead of showing a test pattern as a finished Touchdown.
                "mock": "fake_build" in os.environ.get("TOUCHDOWN_BUILD_CMD", "")}

    def point(lat: float, lon: float) -> tuple[float, float]:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(400, "bad position")
        return round(lat, 5), round(lon, 5)

    @app.get("/api/custom/check", dependencies=[Depends(user)])
    def custom_check(lat: float, lon: float):
        return places.check_point(settings, *point(lat, lon))

    @app.get("/api/geocode", dependencies=[Depends(user)])
    def geocode(q: str = ""):
        try:
            return {"results": places.geocode(q)}
        except Exception as e:
            raise HTTPException(502, f"place search is unavailable ({type(e).__name__})") from e

    @app.get("/api/geocode/reverse", dependencies=[Depends(user)])
    def geocode_reverse(lat: float, lon: float):
        try:
            return places.reverse(*point(lat, lon))
        except Exception:
            return {"name": "", "state": ""}  # the name is a nicety: "Custom place" will do

    @app.post("/api/login")
    async def login(request: Request):
        body = await request.json()
        if not settings.passcode or not hmac.compare_digest(str(body.get("passcode", "")), settings.passcode):
            raise HTTPException(401, "wrong passcode")
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, pass_cookie(), httponly=True, samesite="lax", max_age=30 * 86400,
                            secure=request.url.scheme == "https" or
                            request.headers.get("x-forwarded-proto") == "https")
        return response

    @app.post("/api/jobs", dependencies=[Depends(user)])
    async def create_job(preset: str = Form(...), wearing: str = Form(""), photo: UploadFile = File(...),
                         latitude: float | None = Form(None), longitude: float | None = Form(None),
                         name: str = Form(""), quality: str = Form("full")):
        if quality not in ("full", "draft"):
            raise HTTPException(400, "quality must be full or draft")
        place: dict = {}
        if preset == "custom":
            if latitude is None or longitude is None:
                raise HTTPException(400, "a custom place needs a position")
            lat, lon = point(latitude, longitude)
            check = await asyncio.to_thread(places.check_point, settings, lat, lon)
            if not check.get("ok"):
                raise HTTPException(400, check.get("message", "that point can't be built"))
            place = {"title": places.clean_name(name) or "Custom place", "custom": True, "latitude": lat,
                     "longitude": lon, "project": check["project"], "zone": check["zone"],
                     "region": check.get("state", "")}
        elif presets.by_id(preset) is None:
            entry = places.by_id(settings, preset) if NAME_RE.match(preset) else None
            if entry is None or not places.support(entry)[0]:
                raise HTTPException(400, "unknown place")
            hand = next((p for p in presets.PRESETS if p["place"] == preset), None)
            if hand:  # a catalogue place with a hand-written manifest uses it
                preset = hand["id"]
            else:
                place = {"title": entry["name"], "catalogue": entry["id"]}
        jobs.sweep(settings.worker_timeout_s)
        if jobs.active_count() >= settings.max_queue:
            raise HTTPException(429, "the queue is full; please try again later")
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = await photo.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"photos must be under {settings.max_upload_mb:g} MB")
        stem = settings.uploads / f"{time.time_ns():x}"
        try:
            path = sanitize(data, stem)
        except BadImage as e:
            raise HTTPException(400, str(e)) from e
        wearing = " ".join(wearing.split())[:MAX_WEARING]
        job = jobs.create(preset, wearing, path.name, place, quality)
        return public_job(jobs, job)

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(user)])
    def get_job(job_id: str):
        return public_job(jobs, job_or_404(job_id))

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(user)])
    async def job_events(job_id: str, request: Request):
        job_or_404(job_id)

        async def stream():
            last = None
            idle = 0.0
            while not await request.is_disconnected():
                job = job_or_404(job_id)
                body = json.dumps(public_job(jobs, job))
                if body != last:
                    last, idle = body, 0.0
                    yield f"data: {body}\n\n"
                    if job["status"] in ("done", "failed", "cancelled", "held"):
                        return
                elif idle > 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"  # Cloudflare drops idle connections after ~100 s
                await asyncio.sleep(1.0)
                idle += 1.0

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(user)])
    def cancel_job(job_id: str):
        job_or_404(job_id)
        return {"status": jobs.cancel(job_id)}

    @app.get("/api/jobs/{job_id}/video", dependencies=[Depends(user)])
    def video(job_id: str, download: int = 0):
        job = job_or_404(job_id)
        path = settings.jobs_dir / job["id"] / (job["result"] or "-")
        if job["status"] not in ("done", "held") or not path.is_file():
            raise HTTPException(404, "no video yet")
        return FileResponse(path, media_type="video/mp4",
                            filename=f"touchdown-{job['preset']}-{job['id']}.mp4" if download else None)

    @app.get("/api/jobs/{job_id}/previews/{name}", dependencies=[Depends(user)])
    def preview(job_id: str, name: str):
        job = job_or_404(job_id)
        if name not in job["previews"]:
            raise HTTPException(404, "no such preview")
        return FileResponse(settings.jobs_dir / job["id"] / "previews" / name)

    @app.get("/api/gallery", dependencies=[Depends(user)])
    def gallery():
        items = []
        for mp4 in sorted(settings.gallery_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta_path = mp4.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            poster = mp4.with_suffix(".jpg")
            items.append({"file": mp4.name, "url": f"/api/gallery/{mp4.name}",
                          "title": meta.get("title") or mp4.stem.replace("-", " "),
                          "poster": f"/api/gallery/{poster.name}" if poster.is_file() else None})
        return {"items": items}

    @app.get("/api/gallery/{name}", dependencies=[Depends(user)])
    def gallery_file(name: str):
        path = settings.gallery_dir / name
        if not FILE_RE.match(name) or not path.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(path)

    # --- worker API ----------------------------------------------------------------------------------------

    @app.post("/api/worker/heartbeat", dependencies=[Depends(worker)])
    def worker_heartbeat(photographic: bool = False, enhancement: bool = False, canonical: bool = False):
        if canonical:
            app.state.canonical_worker_seen = time.time()
        if photographic:
            app.state.photo_worker_seen = time.time()
        if enhancement:
            app.state.enhance_worker_seen = time.time()
        return {"ok": True}

    @app.post("/api/worker/claim", dependencies=[Depends(worker)])
    def claim(photographic: bool = False, enhancement: bool = False, canonical: bool = False):
        jobs.sweep(settings.worker_timeout_s)
        job = jobs.claim(photographic=photographic, enhancement=enhancement, canonical=canonical)
        if job is None:
            return Response(status_code=204)
        preset = presets.by_id(job["preset"])
        out = {"id": job["id"], "preset": job["preset"], "manifest": preset["manifest"] if preset else None,
               "wearing": job["person"], "photo_name": job["photo"], "quality": job.get("quality") or "full",
               "workflow": job["workflow"], "setup_id": job["setup_id"]}
        if job["workflow"] in ("photographic-v1", checkpoints.WORKFLOW):
            record = studio.load_record(settings, job["setup_id"])
            out["setup"] = studio.worker_setup(settings, record)
            out["manifest"] = None
        if job["workflow"] == "refine-details-v1":
            out.update(source_sha256=job["source_sha256"], parent_id=job["parent_id"], manifest=None)
        if not preset:  # the worker generates the manifest (touchdown.manifest_gen)
            out["place"] = job["place"]
        return out

    @app.get("/api/worker/jobs/{job_id}/photo", dependencies=[Depends(worker)])
    def worker_photo(job_id: str):
        job = job_or_404(job_id)
        if job["workflow"] in ("photographic-v1", checkpoints.WORKFLOW):
            record = studio.load_record(settings, job["setup_id"])
            return FileResponse(settings.data_dir / "setups" / record["id"] / record["subject"])
        return FileResponse(settings.uploads / job["photo"])

    @app.get("/api/worker/jobs/{job_id}/inputs/{name}", dependencies=[Depends(worker)])
    def worker_input(job_id: str, name: str):
        job = job_or_404(job_id)
        record = studio.load_record(settings, job["setup_id"])
        allowed = [record["subject"], *record["references"],
                   *[p["name"] for p in record.get("default_references", []) if p.get("name")]]
        if not studio.ASSET_RE.fullmatch(name) or name not in allowed:
            raise HTTPException(404, "No such input photo.")
        return FileResponse(settings.data_dir / "setups" / record["id"] / name)

    @app.post("/api/worker/jobs/{job_id}/review", dependencies=[Depends(worker)])
    async def worker_review(job_id: str, request: Request):
        job = job_or_404(job_id)
        body = await request.json()
        if job["workflow"] not in ("photographic-v1", "refine-details-v1", checkpoints.WORKFLOW) or len(json.dumps(body)) > 200_000:
            raise HTTPException(400, "Invalid review.")
        folder = settings.jobs_dir / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "review.json").write_text(json.dumps(body, indent=2))
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/review", dependencies=[Depends(user)])
    def get_review(job_id: str):
        job = job_or_404(job_id)
        path = settings.jobs_dir / job["id"] / "review.json"
        if not path.is_file():
            raise HTTPException(404, "No review yet.")
        return FileResponse(path, media_type="application/json")

    @app.post("/api/worker/jobs/{job_id}/progress", dependencies=[Depends(worker)])
    async def worker_progress(job_id: str, request: Request):
        job = job_or_404(job_id)
        body = await request.json()
        fields = {"heartbeat": time.time()}
        if isinstance(body.get("stages"), dict):
            fields["stages"] = {str(k): str(v) for k, v in body["stages"].items()}
        if isinstance(body.get("message"), str):
            fields["message"] = body["message"][:300]
        if job["status"] == "running":
            jobs.update(job_id, **fields)
        return {"cancel": job["cancel_requested"] or job["status"] != "running"}

    @app.post("/api/worker/jobs/{job_id}/preview", dependencies=[Depends(worker)])
    async def worker_preview(job_id: str, name: str = Form(...), file: UploadFile = File(...)):
        job = job_or_404(job_id)
        stem = Path(name).stem
        if not NAME_RE.match(stem):
            raise HTTPException(400, "bad preview name")
        data = await file.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "preview too large")
        folder = settings.jobs_dir / job["id"] / "previews"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            out = sanitize(data, folder / stem)
        except BadImage as e:
            raise HTTPException(400, str(e)) from e
        for other in folder.glob(stem + ".*"):
            if other != out:
                other.unlink()
        jobs.add_preview(job_id, out.name)
        return {"ok": True, "name": out.name}

    @app.post("/api/worker/jobs/{job_id}/result", dependencies=[Depends(worker)])
    async def worker_result(job_id: str, file: UploadFile = File(...)):
        job = job_or_404(job_id)
        if job["status"] != "running" or job["cancel_requested"]:
            raise HTTPException(409, "This job is no longer accepting a result.")
        folder = settings.jobs_dir / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "touchdown.mp4.part"
        with tmp.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        tmp.replace(folder / "touchdown.mp4")
        jobs.update(job_id, result="touchdown.mp4")
        return {"ok": True}

    @app.post("/api/worker/jobs/{job_id}/finish", dependencies=[Depends(worker)])
    async def worker_finish(job_id: str, request: Request):
        job = job_or_404(job_id)
        body = await request.json()
        status = body.get("status")
        if status not in ("done", "failed", "cancelled", "held"):
            raise HTTPException(400, "bad status")
        if job["cancel_requested"]:
            status = "cancelled"
        if status == "done" and not job["result"]:
            status, body["error"] = "failed", "the worker finished without uploading a video"
        if status == "done" and job["workflow"] in ("photographic-v1", checkpoints.WORKFLOW):
            folder = settings.jobs_dir / job["id"]
            try:
                review = json.loads((folder / "review.json").read_text())
                valid = (review["status"] == "done" and review["checks_passed"] is True
                         and review["video_sha256"] == hashlib.sha256((folder / job["result"]).read_bytes()).hexdigest())
            except (OSError, ValueError, KeyError):
                valid = False
            if not valid:
                status, body["error"] = "held", "The delivered film needs reference/continuity review."
        if status == "done" and job["workflow"] == checkpoints.WORKFLOW:
            try:
                base = settings.jobs_dir / job_id / "checkpoints"
                review = json.loads((settings.jobs_dir / job_id / "review.json").read_text())
                if review.get("recipe") not in checkpoints.RECIPES:
                    raise ValueError("Wrong canonical recipe")
                for name in checkpoints.RECIPES[review["recipe"]]:
                    request = checkpoints.read(base / name / "request.json")
                    decision = checkpoints.read(base / name / "decision.json")
                    if decision.get("continue") is not True or decision.get("evidence_sha256") != request["evidence_sha256"]:
                        raise ValueError("Missing canonical review")
                    for asset in request['assets']:
                        if checkpoints.sha(base / name / asset['name']) != asset['sha256']:
                            raise ValueError("Changed canonical review evidence")
                final = checkpoints.read(base / "final" / "request.json")
                if not any(a["sha256"] == review["video_sha256"] for a in final["assets"]):
                    raise ValueError("Final review does not bind delivered film")
            except (OSError, ValueError, KeyError):
                status, body["error"] = "held", "Canonical review checkpoints are missing or do not match the delivered film."
        if status == "done" and job["workflow"] == "refine-details-v1":
            error = enhancement.completion_error(settings, jobs, job)
            if error:
                status, body["error"] = "held", error
        jobs.finish(job_id, status, error=str(body.get("error", "")), message=str(body.get("message", ""))[:300])
        return {"ok": True}

    # --- front end -----------------------------------------------------------------------------------------

    @app.get("/")
    def index():
        return FileResponse(settings.web_dir / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/workflow")
    @app.get("/workflow.html")
    def workflow():
        return RedirectResponse("/static/workflow.html")

    checkpoints.install(app, settings, jobs, user, worker, job_or_404)
    enhancement.install(app, settings, jobs, user, worker, job_or_404, public_job)
    studio.install(app, settings, user, lambda job: public_job(jobs, job))
    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")
    return app
