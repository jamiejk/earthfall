"""The worker loop: pull a job from the server over HTTP(S), run it, repeat. Needs no inbound port.

    TOUCHDOWN_SERVER=https://touchdown.example.com TOUCHDOWN_WORKER_TOKEN=... python -m worker
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import httpx

from .runner import WorkerConfig, run_job


class ServerClient:
    """The Reporter for runner.run_job, talking to the server's /api/worker endpoints."""

    def __init__(self, base_url: str, token: str, timeout: float = 60.0, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.http.headers["Authorization"] = f"Bearer {token}"

    def _retry(self, fn, tries: int = 5):
        for attempt in range(tries):
            try:
                r = fn()
                r.raise_for_status()
                return r
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise
                if attempt == tries - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def claim(self, photographic: bool = False, enhancement: bool = False, canonical: bool = False) -> dict | None:
        r = self.http.post("/api/worker/claim", params={"photographic": str(photographic).lower(), "enhancement": str(enhancement).lower(), "canonical": str(canonical).lower()})
        r.raise_for_status()
        return None if r.status_code == 204 else r.json()

    def input(self, job_id: str, name: str, dest: Path) -> None:
        r = self._retry(lambda: self.http.get(f"/api/worker/jobs/{job_id}/inputs/{name}"))
        dest.write_bytes(r.content)

    def review(self, job_id: str, body: dict) -> None:
        self._retry(lambda: self.http.post(f"/api/worker/jobs/{job_id}/review", json=body))

    def heartbeat(self, photographic: bool = False, enhancement: bool = False, canonical: bool = False) -> None:
        r = self.http.post("/api/worker/heartbeat", params={"photographic": str(photographic).lower(), "enhancement": str(enhancement).lower(), "canonical": str(canonical).lower()})
        r.raise_for_status()

    def checkpoint(self, job_id: str, data: dict, root: Path) -> dict:
        for asset in data['assets']:
            path = (root / asset['path']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Review asset is outside the job')
            def send():
                with path.open('rb') as stream:
                    return self.http.post(f"/api/worker/jobs/{job_id}/checkpoints/{data['id']}/assets/{asset['name']}", files={'file': (asset['name'], stream)}, timeout=600)
            self._retry(send)
        body = {**data, 'assets': [{k:v for k,v in a.items() if k != 'path'} for a in data['assets']]}
        return self._retry(lambda: self.http.post(f'/api/worker/jobs/{job_id}/checkpoint', json=body)).json()

    def checkpoint_review(self, job_id: str, checkpoint_id: str, data: dict, root: Path):
        path=(root/data['record']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Review report outside job')
        body={**data,'review_report':path.read_text()}
        return self._retry(lambda: self.http.post(f'/api/worker/jobs/{job_id}/checkpoints/{checkpoint_id}/decision',json=body)).json()

    def checkpoint_decision(self, job_id: str, checkpoint_id: str):
        return self._retry(lambda: self.http.get(f'/api/worker/jobs/{job_id}/checkpoints/{checkpoint_id}/decision')).json()

    def source_video(self, job_id: str, dest: Path) -> None:
        def download():
            with self.http.stream("GET", f"/api/worker/jobs/{job_id}/source-video", timeout=600) as response:
                response.raise_for_status()
                with dest.open("wb") as stream:
                    for chunk in response.iter_bytes(1 << 20):
                        stream.write(chunk)
            return response
        self._retry(download)

    def photo(self, job_id: str, dest: Path) -> None:
        r = self._retry(lambda: self.http.get(f"/api/worker/jobs/{job_id}/photo"))
        dest.write_bytes(r.content)

    def progress(self, job_id: str, stages: dict, message: str) -> bool:
        r = self.http.post(f"/api/worker/jobs/{job_id}/progress", json={"stages": stages, "message": message})
        r.raise_for_status()
        return bool(r.json().get("cancel"))

    def preview(self, job_id: str, name: str, path: Path) -> None:
        def send():
            with path.open("rb") as fh:
                return self.http.post(f"/api/worker/jobs/{job_id}/preview", data={"name": name},
                                      files={"file": (path.name, fh)})
        self._retry(send, tries=3)

    def result(self, job_id: str, path: Path) -> None:
        def send():
            with path.open("rb") as fh:
                return self.http.post(f"/api/worker/jobs/{job_id}/result", files={"file": ("touchdown.mp4", fh,
                                                                                            "video/mp4")},
                                      timeout=600)
        self._retry(send)

    def finish(self, job_id: str, status: str, error: str = "", message: str = "") -> None:
        self._retry(lambda: self.http.post(f"/api/worker/jobs/{job_id}/finish",
                                           json={"status": status, "error": error, "message": message}))


def loop(client: ServerClient, cfg: WorkerConfig | None = None, stop: threading.Event | None = None,
         poll_s: float = 5.0, once: bool = False, log=print) -> None:
    """Claim and run jobs until ``stop`` is set (or after one job with ``once``)."""
    cfg = cfg or WorkerConfig()
    stop = stop or threading.Event()
    announcer_stop = threading.Event()
    # The heartbeat remains active during long GPU stages, independently of claims.
    if cfg.photographic or cfg.enhancement or cfg.canonical:
        def heartbeat_loop():
            while not stop.is_set() and not announcer_stop.is_set():
                try:
                    client.heartbeat(photographic=cfg.photographic, enhancement=cfg.enhancement, canonical=cfg.canonical)
                except httpx.HTTPError:
                    pass
                announcer_stop.wait(15)
        threading.Thread(target=heartbeat_loop, daemon=True).start()
    warned = False
    while not stop.is_set():
        try:
            job = client.claim(photographic=cfg.photographic, enhancement=cfg.enhancement, canonical=cfg.canonical)
            warned = False
        except httpx.HTTPError as e:
            if not warned:
                log(f"worker: server not reachable yet ({e}); retrying")
                warned = True
            job = None
        if job is None:
            if once:
                announcer_stop.set()
                return
            stop.wait(poll_s)
            continue
        log(f"worker: job {job['id']} ({job['preset']})")
        try:
            status = run_job(job, client, cfg, log=log)
        except Exception as e:  # report and keep serving the queue
            status = "failed"
            try:
                client.finish(job["id"], "failed", error=f"worker error: {e!r}")
            except httpx.HTTPError:
                pass
        log(f"worker: job {job['id']} {status}")
        if once:
            announcer_stop.set()
            return
    announcer_stop.set()


def main() -> None:
    p = argparse.ArgumentParser(description="Touchdown GPU worker: pulls jobs from a Touchdown server")
    p.add_argument("--server", default=os.environ.get("TOUCHDOWN_SERVER", "http://127.0.0.1:8000"))
    p.add_argument("--token", default=os.environ.get("TOUCHDOWN_WORKER_TOKEN", ""))
    p.add_argument("--poll", type=float, default=float(os.environ.get("TOUCHDOWN_POLL", "5")))
    p.add_argument("--once", action="store_true", help="run at most one job, then exit")
    a = p.parse_args()
    if not a.token:
        raise SystemExit("set TOUCHDOWN_WORKER_TOKEN (the same value the server has)")
    cfg = WorkerConfig()
    if cfg.photographic:
        import subprocess
        subprocess.run([cfg.python, "-m", "touchdown.studio_build", "--check"], cwd=cfg.cwd(), env=cfg.env(), check=True)
    if cfg.canonical:
        import subprocess
        subprocess.run([cfg.python, "-m", "touchdown.canonical_build", "--check"], cwd=cfg.cwd(), env=cfg.env(), check=True)
    if cfg.enhancement:
        import subprocess
        subprocess.run([cfg.python, "-m", "touchdown.enhance", "--check"], cwd=cfg.cwd(), env=cfg.env(), check=True)
    print(f"worker: serving {a.server}; builds run in {cfg.cwd()} with: {cfg.canonical_build_cmd if cfg.canonical else cfg.photo_build_cmd if cfg.photographic else cfg.build_cmd}")
    loop(ServerClient(a.server, a.token), cfg, poll_s=a.poll, once=a.once)


if __name__ == "__main__":
    main()
