"""Transport and supervise a photographic build; keep GPU work out of the web server."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from .runner import kill_build, output_dir

STAGES = ("references", "arrival", "arrival_check", "portrait", "flight", "pacing", "review")


def run_photo_job(job, rep, cfg, log=print):
    job_id = job["id"]
    root = output_dir(cfg, {}) / f"studio-{job_id.lower()}"
    root.mkdir(parents=True, exist_ok=True)
    stages = {s: "pending" for s in STAGES}
    proc = None
    try:
        setup = job["setup"]
        inputs = root / "inputs"
        inputs.mkdir(exist_ok=True)
        for item in [setup["subject"], *setup["references"]]:
            name = item["name"]
            if not re.fullmatch(r"(?:subject|reference-[0-3]|location-[0-3])\.(?:jpg|png)", name):
                raise ValueError("Invalid input filename")
            path = inputs / name
            rep.input(job_id, name, path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("A saved input changed during download")
            if rep.progress(job_id, stages, "Loading saved photographs"):
                rep.finish(job_id, "cancelled", message="Cancelled")
                return "cancelled"
        source = root / "setup.json"
        source.write_text(json.dumps(setup, indent=2))
        words = {"python": cfg.python, "input": str(source), "output": str(root), "repo": str(cfg.repo)}
        canonical = job.get('workflow') == 'photographic-grounded-v1'
        if canonical and not cfg.canonical:
            raise ValueError('This worker does not support the canonical recipe')
        command = [p.format(**words) for p in shlex.split(cfg.canonical_build_cmd if canonical else cfg.photo_build_cmd)]
        sent = {}
        published = set()
        with (root / "worker.log").open("a") as log_file:
            proc = subprocess.Popen(command, cwd=cfg.cwd(), env=cfg.env(), stdout=log_file, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while True:
                state = {}
                try:
                    state = json.loads((root / "progress.json").read_text())
                except (FileNotFoundError, ValueError):
                    pass
                stages = state.get("stages", stages)
                try:
                    stop = rep.progress(job_id, stages, state.get("message", "Preparing photographic generation"))
                except Exception as exc:
                    log(f"[{job_id}] progress unavailable: {type(exc).__name__}")
                    stop = False
                if stop:
                    kill_build(proc)
                    rep.finish(job_id, "cancelled", message="Cancelled")
                    return "cancelled"
                if canonical and (root / 'checkpoint.json').is_file():
                    checkpoint = json.loads((root / 'checkpoint.json').read_text())
                    checkpoint_id = checkpoint['id']
                    if checkpoint_id not in published:
                        rep.checkpoint(job_id, checkpoint, root)
                        published.add(checkpoint_id)
                    decision_file = root / 'decisions' / (checkpoint_id + '.json')
                    if not decision_file.exists():
                        automatic = root / 'automatic-reviews' / checkpoint_id / 'decision.json'
                        if automatic.exists():
                            decision = rep.checkpoint_review(job_id, checkpoint_id, json.loads(automatic.read_text()), root)
                        else:
                            decision = rep.checkpoint_decision(job_id, checkpoint_id)
                        if decision:
                            decision_file.parent.mkdir(exist_ok=True)
                            temporary = decision_file.with_suffix('.tmp')
                            temporary.write_text(json.dumps(decision)); temporary.replace(decision_file)
                for name in ("arrival", "orbit", "approach", "portrait"):
                    path = root / "previews" / f"{name}.jpg"
                    sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                    if sha and sent.get(name) != sha:
                        try:
                            rep.preview(job_id, name, path)
                            sent[name] = sha
                        except Exception as exc:
                            log(f"[{job_id}] preview unavailable: {type(exc).__name__}")
                if proc.poll() is not None:
                    break
                time.sleep(cfg.tick_s)
        outcome_path = root / "outcome.json"
        if not outcome_path.is_file():
            tail = (root / "worker.log").read_text()[-6000:]
            raise RuntimeError(f"Photographic build exited {proc.returncode}.\n{tail}")
        outcome = json.loads(outcome_path.read_text())
        status = outcome["status"]
        if status not in ("done", "held", "failed") or (status == "done" and proc.returncode != 0):
            raise ValueError("Invalid build completion")
        if outcome.get("video"):
            video = (root / outcome["video"]).resolve()
            if not video.is_relative_to(root.resolve()) or not video.is_file():
                raise ValueError("Invalid result path")
            if hashlib.sha256(video.read_bytes()).hexdigest() != outcome["video_sha256"]:
                raise ValueError("The delivered video changed after review")
            rep.result(job_id, video)
        rep.review(job_id, outcome)
        rep.finish(job_id, status, error=outcome.get("error", ""), message=outcome.get("message", status))
        return status
    except Exception as exc:
        rep.finish(job_id, "failed", error=f"Could not complete photographic generation: {exc}")
        return "failed"
    finally:
        if proc is not None and proc.poll() is None:
            kill_build(proc)
