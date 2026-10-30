"""Run one Touchdown job on the GPU machine: write its manifest, run the build, report progress, upload results.

The build itself is unchanged: ``python -m touchdown.build <manifest> --descent 45 --url <ComfyUI>`` (configurable
with TOUCHDOWN_BUILD_CMD). Progress is read from the output folders the build already writes: a stage's folder
``<root>/<stage>/<key>/`` appears when the stage starts and gets ``done.json`` when it finishes.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import tomli_w

from .describe import describe_person

APP_DIR = Path(__file__).resolve().parents[1]
# Must match STAGES in scripts/touchdown/build.py (tests/test_app.py checks); copied so the worker does not need the
# pipeline's imports (numpy, OpenCV) in its own Python.
STAGES = ("terrain", "stills", "opener", "dive", "cloud", "swoop", "body", "end", "face", "sound")
PREVIEWS = ("orbit", "cloud", "end")  # stills written by the stills stage, shown while the rest renders
DEFAULT_CMD = "{python} -m touchdown.build {manifest} --descent {descent} --url {url}"
FALLBACK_PERSON = "a person in everyday outdoor clothes"  # generated places have no test subject's description
# Used when the user leaves "what they're wearing" empty. The photo is the source of truth: the preset manifests
# describe the test subject ("a young woman ... long red raincoat"), and the end still is told to keep "their
# exact face, hair, body and clothing", so the test subject's description would contradict the photo.
PHOTO_PERSON = "the person in the reference photo"


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


def _repo_default() -> Path:
    if os.environ.get("TOUCHDOWN_REPO"):
        return Path(os.environ["TOUCHDOWN_REPO"]).resolve()
    return APP_DIR.parent if (APP_DIR.parent / "scripts" / "touchdown").is_dir() else Path.cwd().resolve()


class heartbeat:
    """Keep the server's view of a job alive while a long step runs.

    The server sweeps a job whose heartbeat is older than TOUCHDOWN_WORKER_TIMEOUT and answers the next progress
    post with "cancel", so a step that runs for minutes without posting (manifest generation, the terrain keys)
    used to kill a perfectly good build. This posts a progress message every ``every`` seconds instead.
    """

    def __init__(self, rep: "Reporter", job_id: str, message: str, every: float = 30.0):
        self.rep, self.job_id, self.message, self.every = rep, job_id, message, every
        self.cancelled = False
        self._stop = threading.Event()

    def __enter__(self):
        def beat():
            while not self._stop.wait(self.every):
                try:
                    if self.rep.progress(self.job_id, {}, self.message):
                        self.cancelled = True
                        return
                except Exception:  # a heartbeat is best effort
                    pass
        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


@dataclass
class WorkerConfig:
    repo: Path = field(default_factory=lambda: _repo_default())
    python: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_PYTHON", sys.executable))
    build_cmd: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_BUILD_CMD", DEFAULT_CMD))
    build_cwd: Path | None = field(default_factory=lambda: (Path(os.environ["TOUCHDOWN_BUILD_CWD"])
                                                            if os.environ.get("TOUCHDOWN_BUILD_CWD") else None))
    comfy_url: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_COMFY_URL", "http://127.0.0.1:8193"))
    # The stills stage talks to ComfyUI over its own address, which the manifests hard-code (8193 through the
    # proxy, 8192 for the plain server). A judge on stock ComfyUI (8188) can set these instead of editing TOML.
    flux3_url: str | None = field(default_factory=lambda: os.environ.get("TOUCHDOWN_FLUX3_URL") or None)
    flux3_free_url: str | None = field(default_factory=lambda: os.environ.get("TOUCHDOWN_FLUX3_FREE_URL") or None)
    # "klein" draws the stills with FLUX.2 [klein] 9B, which is open weights and already installed, instead of
    # FLUX 3 Image Klein, whose package is gated on Hugging Face. The stills are 1344x768 rather than 2K.
    stills: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_STILLS", "flux3").strip().lower())
    descent: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_DESCENT", "45"))
    output: Path | None = field(default_factory=lambda: (Path(os.environ["TOUCHDOWN_OUTPUT"])
                                                         if os.environ.get("TOUCHDOWN_OUTPUT") else None))
    share_terrain: bool = field(default_factory=lambda: _env_bool("TOUCHDOWN_SHARE_TERRAIN", True))
    tick_s: float = field(default_factory=lambda: float(os.environ.get("TOUCHDOWN_TICK", "2")))
    # generated places: measure the ground underfoot on NAIP for the place words (network; cached by point)
    naip_words: bool = field(default_factory=lambda: _env_bool("TOUCHDOWN_NAIP_WORDS", True))
    photographic: bool = field(default_factory=lambda: _env_bool("TOUCHDOWN_PHOTOGRAPHIC", False))
    photo_build_cmd: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_PHOTO_BUILD_CMD",
        "{python} -m touchdown.studio_build --input {input} --output {output}"))
    canonical: bool = field(default_factory=lambda: _env_bool("TOUCHDOWN_CANONICAL", False))
    canonical_build_cmd: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_CANONICAL_BUILD_CMD",
        "{python} -m touchdown.canonical_build --input {input} --output {output}"))
    enhancement: bool = field(default_factory=lambda: _env_bool("TOUCHDOWN_ENHANCEMENT", False))
    enhance_build_cmd: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_ENHANCE_BUILD_CMD",
        "{python} -m touchdown.enhance --input {input} --output {output}"))

    def cwd(self) -> Path:
        return (self.build_cwd or self.repo / "scripts").resolve()

    def env(self) -> dict:
        paths = [str(self.repo / "scripts"), str(self.repo)]
        if os.environ.get("PYTHONPATH"):
            paths.append(os.environ["PYTHONPATH"])
        return {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "PYTHONUNBUFFERED": "1"}


class Reporter(Protocol):
    def source_video(self, job_id: str, dest: Path) -> None: ...
    def review(self, job_id: str, body: dict) -> None: ...
    def photo(self, job_id: str, dest: Path) -> None: ...
    def progress(self, job_id: str, stages: dict, message: str) -> bool: ...  # True: stop the build
    def preview(self, job_id: str, name: str, path: Path) -> None: ...
    def result(self, job_id: str, path: Path) -> None: ...
    def finish(self, job_id: str, status: str, error: str = "", message: str = "") -> None: ...


# --- the job manifest ----------------------------------------------------------------------------------------

def job_name(job: dict) -> str:
    """Per-job ``[shot] name``: the job's own folder, so nobody's stills or video land in another job's folder."""
    return f"app-{job['preset']}-{job['id'].lower()}"


def output_dir(cfg: WorkerConfig, m: dict) -> Path:
    out = cfg.output or Path(m.get("output", "outputs/touchdown"))
    return (out if out.is_absolute() else cfg.cwd() / out).resolve()


def enabled_stages(m: dict) -> list[str]:
    """The stages this manifest runs, in order (as Build.build skips them)."""
    return [s for s in STAGES
            if not (s == "terrain" and not m.get("terrain", {}).get("enabled", False))
            and not (s == "face" and not m.get("face", {}).get("enabled", False))
            and not (s == "sound" and not m.get("sound", {}).get("enabled", True))]


def write_job_manifest(preset_path: Path, dest: Path, photo: Path, person: str, name: str, out_dir: Path,
                       quality: str = "full", flux3_url: str | None = None,
                       flux3_free_url: str | None = None, stills: str = "flux3") -> dict:
    """Copy the preset manifest to ``dest`` with this job's subject, person, name and output. The build resolves
    some paths against the manifest's folder; those are made absolute here because ``dest`` lives elsewhere."""
    m = tomllib.loads(preset_path.read_text())
    base = preset_path.parent
    shot = m["shot"]
    for k in ("end_still", "end_reference"):
        if shot.get(k):
            shot[k] = str((base / shot[k]).resolve())
    for k in ("near_source", "far_source"):
        if m.get("terrain", {}).get(k):
            m["terrain"][k] = str((base / m["terrain"][k]).resolve())
    for k in ("commons_refs", "mid_refs"):
        if m.get("stills", {}).get(k):
            m["stills"][k] = [str((base / p).resolve()) for p in m["stills"][k]]
    wf = m.get("sound", {}).get("workflow")
    if wf and not Path(wf).expanduser().is_absolute() and (base / wf).is_file():
        m["sound"]["workflow"] = str((base / wf).resolve())
    if flux3_url:
        m.setdefault("stills", {})["flux3_url"] = flux3_url
    if flux3_free_url:
        m.setdefault("stills", {})["flux3_free_url"] = flux3_free_url
    if stills == "klein":
        # Draw the stills with FLUX.2 [klein] instead of FLUX 3 Image Klein (a gated package). The outline
        # conditioning stays: it is what holds the skyline to a few pixels.
        m.setdefault("stills", {})["flux3"] = []
        m["stills"]["end_conditioning"] = m["stills"].get("end_conditioning", "structure")
    shot["subject"] = str(photo.resolve())
    shot["person"] = person
    shot["name"] = name
    m["output"] = str(out_dir)
    m.setdefault("video", {})["quality"] = quality  # full: LTX 2x pass to 1920x1088; draft: 960x544, faster
    dest.write_text(f"# Touchdown app job, from {preset_path}\n" + tomli_w.dumps(m))
    return m


# --- generated manifests (catalogue places without a hand-written manifest, and custom US points) -------------

GENERATED = "app-places"  # <output>/app-places/<preset>.toml: the generated manifest a job copies


def generate_manifest(cfg: WorkerConfig, job: dict, out_dir: Path, photo: Path, person: str) -> Path:
    """Write the generated manifest for ``job`` with touchdown.manifest_gen, run in the pipeline's Python (as the
    terrain keys are), and return its path. Nothing from the request reaches the command line except the
    catalogue id (checked against the catalogue by the server) and numbers and a cleaned name for a custom point."""
    place = job.get("place") or {}
    dest = out_dir / GENERATED / f"{job['preset']}.toml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cfg.python, "-m", "touchdown.manifest_gen", "--subject", str(photo), "--person", person,
           "--cache", str(out_dir), "-o", str(dest)]
    if place.get("custom"):
        cmd += ["--lat", repr(float(place["latitude"])), "--lon", repr(float(place["longitude"])),
                "--project", re.sub(r"[^A-Za-z0-9_]", "", str(place["project"])), "--zone", str(int(place["zone"])),
                "--name", describe_clean(place.get("title", "")), "--region", describe_clean(place.get("region", ""))]
    else:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", job["preset"]):
            raise ValueError(f"bad place id: {job['preset']}")
        cmd.insert(3, job["preset"])
    if cfg.naip_words:
        cmd.append("--naip")
    done = subprocess.run(cmd, cwd=cfg.cwd(), env=cfg.env(), capture_output=True, text=True, timeout=900)
    if done.returncode != 0 or not dest.is_file():
        raise RuntimeError("manifest generation failed: " + (done.stderr.strip()[-500:] or done.stdout[-500:]))
    return dest


def describe_clean(text: str) -> str:
    return re.sub(r"[{}\[\]<>\"\\\x00-\x1f]", "", " ".join(str(text or "").split()))[:80]


# --- sharing the terrain stage ---------------------------------------------------------------------------------

def terrain_keys(cfg: WorkerConfig, manifests: list[Path], out_dir: Path) -> dict:
    """Ask touchdown.build itself (in the pipeline's Python) for each manifest's terrain cache key."""
    cmd = [cfg.python, str(APP_DIR / "worker" / "terrain_key.py"), "--output", str(out_dir),
           "--descent", cfg.descent, *map(str, manifests)]
    done = subprocess.run(cmd, cwd=cfg.cwd(), env=cfg.env(), capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip()[-500:] or "terrain key helper failed")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


_PROGRESS = re.compile(r"^\s*(\d{1,3}(\.\d+)?%|\d+/\d+ \[|\S*\s*\d{1,3}%\|)|(it/s|s/it)\]?\s*$")


def is_progress(line: str) -> bool:
    """A progress-bar line (tqdm, ComfyUI sampling): '99%', ' 45%|####  | 9/20 [00:03<00:04, 2.5it/s]'."""
    return bool(line.strip()) and bool(_PROGRESS.search(line))


def share_terrain(cfg: WorkerConfig, preset_path: Path, job_manifest: Path, root: Path, out_dir: Path,
                  keys_fn: Callable[[WorkerConfig, list[Path], Path], dict] = terrain_keys) -> str:
    """Seed this job's terrain stage from a finished one for the same place.

    Every stage's cache key hashes the whole ``[shot]`` table, so a new person (photo path, description, name)
    changes the terrain key too, although the terrain stage never reads those fields: it uses the place, the
    scene words and the [terrain] table, which a job copies unchanged from the preset. So a terrain folder built
    for the preset (or an earlier job of the same preset) with the same code is the right result for this job.
    It is copied into the folder under this job's key; the build then reports it as cached. Everything that
    depends on the person (stills onwards) still re-runs. Returns a note for the log.
    """
    donors = [preset_path] + sorted((p for p in out_dir.glob(f"app-{preset_path.stem}-*/job.toml")
                                     if p.parent != root), key=lambda p: p.stat().st_mtime, reverse=True)[:6]
    keys = keys_fn(cfg, [job_manifest, *donors], out_dir)
    mine = keys.get(str(job_manifest))
    if not mine:
        return "terrain: no key for this job; it will build its own"
    target = root / "terrain" / mine["key"]
    if (target / "done.json").is_file():
        return "terrain: already present"
    for donor in donors:
        theirs = keys.get(str(donor))
        if not theirs or theirs["neutral"] != mine["neutral"]:  # different place, terrain settings or code
            continue
        name = tomllib.loads(donor.read_text())["shot"]["name"]
        folder = out_dir / name / "terrain" / theirs["key"]
        done = folder / "done.json"
        try:
            ok = done.is_file() and (folder / json.loads(done.read_text())["output"]).is_file()
        except (OSError, ValueError, KeyError):
            ok = False
        if ok:
            tmp = target.with_name(target.name + ".copying")
            shutil.rmtree(tmp, ignore_errors=True)
            # hard links, not copies: a 452 MB terrain folder took ~3 min to copy on the desktop's drive (18 Oct).
            # Later stages only read the terrain files or add new ones beside them, so sharing inodes is safe;
            # a different filesystem falls back to a real copy
            shutil.copytree(folder, tmp, symlinks=True, copy_function=_link_or_copy)
            tmp.rename(target)
            return f"terrain: reused {folder}"
    return "terrain: no finished terrain for this place yet; this job builds it"


# --- watching the build ----------------------------------------------------------------------------------------

def stage_states(root: Path, stages: list[str], cached: set[str]) -> dict:
    states = {}
    for stage in stages:
        subs = [p for p in (root / stage).glob("*") if p.is_dir() and not p.name.endswith(".copying")] \
            if (root / stage).is_dir() else []
        if any((p / "done.json").is_file() for p in subs):
            states[stage] = "cached" if stage in cached else "done"
        elif subs:
            states[stage] = "running"
        else:
            states[stage] = "pending"
    return states


def stills_folder(root: Path) -> Path | None:
    done = sorted((root / "stills").glob("*/done.json"), key=lambda p: p.stat().st_mtime)
    return done[-1].parent if done else None


def small_preview(src: Path, dest: Path) -> Path:
    try:
        from PIL import Image
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((960, 960))
            im.save(dest.with_suffix(".jpg"), "JPEG", quality=85)
            return dest.with_suffix(".jpg")
    except Exception:  # Pillow missing or an odd file: send the original
        return src


def final_video(root: Path, built: str | None) -> Path | None:
    if built and Path(built).is_file():
        return Path(built)
    for stage in ("sound", "face", "end"):
        found = sorted((root / stage).glob("*/touchdown.mp4"), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    return None


def kill_build(proc: subprocess.Popen) -> None:
    """Stop the build this worker started, by its PID. On POSIX it was started in its own session, so its process
    group is exactly the build and the children it spawned (Blender, ffmpeg, the ComfyUI client); nothing else is
    touched and nothing is matched by name."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=15)


def build_command(cfg: WorkerConfig, manifest: Path) -> list[str]:
    words = {"python": cfg.python, "manifest": str(manifest), "descent": cfg.descent, "url": cfg.comfy_url,
             "app": str(APP_DIR), "repo": str(cfg.repo)}
    return [part.format(**words) for part in shlex.split(cfg.build_cmd, posix=os.name != "nt")]


def explain(tail: str, code: int | None = None) -> str:
    """One sentence for what went wrong, above the raw log.

    A judge should not have to read a CUDA traceback. The failures worth naming are the ones with an action:
    the card was too full, the models were missing, the server was not there.
    """
    low = tail.lower()
    if "out of memory" in low or "allocation on device" in low or "exceed allowed memory" in low:
        return ("The graphics card ran out of memory. The LTX 2.5 transformer needs about 21 GB of the 24 GB card, "
                "so nothing else can be using it. Close other GPU work and try again.")
    if "fault failed" in low:
        return ("ComfyUI's memory manager failed while paging the model in. This is the comfy-aimdo version: "
                "ComfyUI v0.41.3 wants comfy-aimdo 0.5.6 and comfy-kitchen 0.2.37.")
    if "no connection" in low or "reach comfyui" in low or "connection refused" in low:
        return ("The build could not reach ComfyUI. Check that the server is running and that TOUCHDOWN_COMFY_URL "
                "(and TOUCHDOWN_FLUX3_URL for the stills) point at it.")
    if "no module named" in low:
        missing = tail.split("No module named")[-1].split("'")[1] if "No module named" in tail else ""
        return f"The build's Python is missing a package{f' ({missing})' if missing else ''}. See the install steps."
    if "no plausible level landing" in low or "no level" in low:
        return "This place has no level ground to stand on near the landing point, so the build stopped."
    if "stopped for picks" in low:
        return "The build stopped for a manual pick: the same stage produced several candidates and one has to be chosen."
    return (f"The build failed (exited with code {code})." if code is not None
            else "The build failed (see the log below).")


def run_job(job: dict, rep: Reporter, cfg: WorkerConfig | None = None, log=print) -> str:
    """Run one claimed job to the end. Returns the final status (done, failed or cancelled)."""
    cfg = cfg or WorkerConfig()
    if job.get("workflow") == "refine-details-v1":
        from .enhancement import run_enhancement_job
        return run_enhancement_job(job, rep, cfg, log)
    if job.get("workflow") in ("photographic-v1", "photographic-grounded-v1"):
        from .photographic import run_photo_job
        return run_photo_job(job, rep, cfg, log)
    job_id = job["id"]
    try:
        if job.get("manifest"):  # a hand-written preset manifest
            preset_path = (cfg.repo / job["manifest"]).resolve()
            if not preset_path.is_relative_to((cfg.repo / "shots").resolve()) or not preset_path.is_file():
                raise ValueError(f"preset manifest not found: {job['manifest']}")
            preset = tomllib.loads(preset_path.read_text())
            out_dir = output_dir(cfg, preset)
        else:  # generated below, once the photo is here
            preset_path = None
            out_dir = output_dir(cfg, {})
        name = job_name(job)
        root = out_dir / name
        root.mkdir(parents=True, exist_ok=True)
        photo = root / ("subject" + (Path(job.get("photo_name", "x.jpg")).suffix.lower() or ".jpg"))
        rep.photo(job_id, photo)
        if preset_path is None:
            person = describe_person(photo, job.get("wearing", ""), PHOTO_PERSON)
            with heartbeat(rep, job_id, "writing this place's manifest") as beat:
                preset_path = generate_manifest(cfg, job, out_dir, photo, person)
            log(f"[{job_id}] generated {preset_path}")
        else:
            person = describe_person(photo, job.get("wearing", ""), PHOTO_PERSON)
        manifest = root / "job.toml"
        m = write_job_manifest(preset_path, manifest, photo, person, name, out_dir, job.get("quality") or "full",
                               flux3_url=cfg.flux3_url, flux3_free_url=cfg.flux3_free_url, stills=cfg.stills)
        stages = enabled_stages(m)
    except Exception as e:
        rep.finish(job_id, "failed", error=f"could not prepare the job: {e}")
        return "failed"

    cached: set[str] = set()
    if cfg.share_terrain and "terrain" in stages:
        rep.progress(job_id, stage_states(root, stages, cached), "looking for this place's terrain")
        try:
            with heartbeat(rep, job_id, "looking for this place's terrain"):
                note = share_terrain(cfg, preset_path, manifest, root, out_dir)
        except Exception as e:  # sharing is an optimisation: the build makes its own terrain without it
            note = f"terrain: not shared ({e})"
        log(f"[{job_id}] {note}")
        if note.startswith("terrain: reused"):
            cached.add("terrain")

    cmd = build_command(cfg, manifest)
    log(f"[{job_id}] running {shlex.join(cmd)} in {cfg.cwd()}")
    tail: deque[str] = deque(maxlen=80)
    built: list[str] = []
    logfile = (root / "worker_build.log").open("a", encoding="utf-8")
    popen = dict(cwd=cfg.cwd(), env=cfg.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                 bufsize=1, encoding="utf-8", errors="replace")
    if os.name == "nt":
        popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **popen)
    except OSError as e:
        logfile.close()
        rep.finish(job_id, "failed", error=f"could not start the build: {e}")
        return "failed"

    def read():
        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()
            line = line.rstrip("\n").split("\r")[-1]
            if not is_progress(line):  # keep the real error in view, not tqdm bars (app first run, 18 Oct)
                tail.append(line)
            if line.startswith("Built "):
                built.append(line[len("Built "):].strip())
            for stage in stages:
                if line.startswith(f"[{stage}] cached"):
                    cached.add(stage)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    sent: set[str] = set()
    cancelled = False
    try:
        while True:
            running = proc.poll() is None
            states = stage_states(root, stages, cached)
            folder = stills_folder(root)
            if folder:
                for still in PREVIEWS:
                    src = folder / f"{still}.png"
                    if still not in sent and src.is_file():
                        try:
                            rep.preview(job_id, still, small_preview(src, root / f"preview_{still}"))
                            sent.add(still)
                        except Exception as e:
                            log(f"[{job_id}] preview {still} not sent: {e}")
            now = next((s for s, v in states.items() if v == "running"), None)
            idle = "starting" if set(states.values()) <= {"pending"} else "working"
            message = f"rendering: {now}" if now else (idle if running else "uploading")
            try:
                stop = rep.progress(job_id, states, message)
            except Exception as e:  # the server being briefly unreachable must not kill a 20-minute build
                log(f"[{job_id}] progress not sent: {e}")
                stop = False
            if stop and running:
                log(f"[{job_id}] cancel requested: stopping build PID {proc.pid}")
                kill_build(proc)
                cancelled = True
            if not running or cancelled:
                break
            time.sleep(cfg.tick_s)
    finally:
        if proc.poll() is None:
            kill_build(proc)
        reader.join(timeout=10)
        logfile.close()

    if cancelled:
        rep.finish(job_id, "cancelled", message="cancelled")
        return "cancelled"
    error_tail = "\n".join(tail)
    if proc.returncode != 0:
        rep.finish(job_id, "failed", error=f"{explain(error_tail, proc.returncode)}\n\n{error_tail}")
        return "failed"
    video = final_video(root, built[-1] if built else None)
    if video is None:
        reason = "the build stopped for a manual pick" if "Stopped for picks:" in error_tail else \
            "the build finished without a touchdown.mp4"
        rep.finish(job_id, "failed", error=f"{reason}\n{error_tail}")
        return "failed"
    rep.progress(job_id, stage_states(root, stages, cached), "uploading the video")
    rep.result(job_id, video)
    rep.finish(job_id, "done", message="done")
    return "done"
