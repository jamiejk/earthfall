"""touchdown-app: run Touchdown.

    touchdown-app                  local mode (the default): web app + a worker in the same process, one machine
    touchdown-app --server-only    hosted mode: web app and queue only; a worker elsewhere pulls jobs
    touchdown-app publish <job>    copy a finished job's video into the example gallery
    touchdown-app mark-tested <place id> [--note "..."]   list a catalogue place as tested (it had a real build)
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import threading
import webbrowser


def serve(a) -> None:
    import uvicorn

    from .config import Settings
    from .main import create_app

    settings = Settings()
    install_examples(settings)
    if not a.server_only and not settings.worker_token:
        settings.worker_token = secrets.token_urlsafe(24)  # the in-process worker's token; never leaves the machine
    app = create_app(settings)
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    stop = threading.Event()
    if not a.server_only:
        from worker.client import ServerClient, loop
        from worker.runner import WorkerConfig

        cfg = WorkerConfig()
        if cfg.canonical:
            import subprocess
            try:
                subprocess.run([cfg.python, "-m", "touchdown.canonical_build", "--check"], cwd=cfg.cwd(), env=cfg.env(), check=True)
            except subprocess.CalledProcessError:
                raise SystemExit("Preflight failed: see the message above. Nothing was started.") from None
        if cfg.enhancement:
            import subprocess
            subprocess.run([cfg.python, "-m", "touchdown.enhance", "--check"], cwd=cfg.cwd(), env=cfg.env(), check=True)
        print(f"Touchdown (local): builds run in {cfg.cwd()} with: {cfg.build_cmd}")
        threading.Thread(target=loop, args=(ServerClient(url, settings.worker_token), cfg, stop),
                         kwargs={"poll_s": 2.0}, daemon=True, name="touchdown-worker").start()
        if not a.no_browser:
            threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"Touchdown: {url}  (data in {settings.data_dir})")
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning", proxy_headers=True,
                    forwarded_allow_ips=os.environ.get("TOUCHDOWN_FORWARDED_IPS", "127.0.0.1"))
    finally:
        stop.set()


def install_examples(settings) -> None:
    """Copy the example films in app/examples into the gallery and refresh their posters. Skipped, with a note, when
    the films or OpenCV are missing, so the app still starts."""
    import importlib.util

    script = settings.repo / "scripts" / "install_earthfall_examples.py"
    if not script.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("install_earthfall_examples", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install(settings.gallery_dir)
    except Exception as exc:  # a missing film or cv2: the page says the examples are not installed
        print(f"Example films not installed: {type(exc).__name__}: {exc}")


def publish(a) -> None:
    from .config import Settings
    from .db import Jobs

    settings = Settings()
    job = Jobs(settings.db_path).get(a.job)
    if not job or job["status"] != "done":
        raise SystemExit("no finished job with that id")
    settings.gallery_dir.mkdir(parents=True, exist_ok=True)
    name = f"{job['preset']}-{job['id'].lower()}"
    shutil.copy(settings.jobs_dir / job["id"] / job["result"], settings.gallery_dir / f"{name}.mp4")
    end = settings.jobs_dir / job["id"] / "previews" / "end.jpg"
    if end.is_file():
        shutil.copy(end, settings.gallery_dir / f"{name}.jpg")
    (settings.gallery_dir / f"{name}.json").write_text(json.dumps({"title": a.title or job["preset"]}))
    print(f"published {name}.mp4")


def mark_tested(a) -> None:
    from . import places
    from .config import Settings

    settings = Settings()
    if places.by_id(settings, a.place) is None:
        raise SystemExit(f"{a.place} is not a catalogue place id")
    print(f"marked {a.place} tested in {places.mark_tested(settings, a.place, a.note or '')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    pub = sub.add_parser("publish", help="copy a finished job's video into the gallery")
    pub.add_argument("job")
    pub.add_argument("--title")
    mark = sub.add_parser("mark-tested", help="list a catalogue place as tested after a real build")
    mark.add_argument("place")
    mark.add_argument("--note")
    p.add_argument("--local", action="store_true", help="web app + in-process worker (the default)")
    p.add_argument("--server-only", action="store_true", help="no worker here; a remote worker pulls jobs")
    p.add_argument("--host", default=os.environ.get("TOUCHDOWN_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TOUCHDOWN_PORT", "8000")))
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = p.parse_args()
    if a.cmd == "publish":
        publish(a)
    elif a.cmd == "mark-tested":
        mark_tested(a)
    else:
        if a.local and a.server_only:
            raise SystemExit("choose --local or --server-only")
        serve(a)


if __name__ == "__main__":
    main()
