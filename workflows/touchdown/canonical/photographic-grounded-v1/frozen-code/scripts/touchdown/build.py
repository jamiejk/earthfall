"""Touchdown: one reproducible build of the Earth-zoom cut from a shot manifest.

    uv run python -m touchdown.build shots/twin-peaks.toml            (run from scripts/, or with PYTHONPATH=scripts)
    uv run python -m touchdown.build shots/twin-peaks.toml --until swoop
    uv run python -m touchdown.build shots/twin-peaks.toml --dry-run  (print the plan and which stages are cached)
    uv run python -m touchdown.build shots/yosemite-one-pass.toml --unattended  (one seed per generated stage)

Needs OpenCV (``--extra touchdown``) and, for the FLUX/LTX jobs, comfy_sdk (``--extra comfy``). With no single
Python that has both, run the build in one that has OpenCV and pass ``--comfy-python`` a Python with the comfy extra:
    PYTHONPATH=scripts:src <ComfyUI venv>/bin/python -m touchdown.build shots/twin-peaks.toml \
        --url http://127.0.0.1:8191 --comfy-python <project .venv>/bin/python

The flat cut has eight required stages and an optional face stage; the measured-terrain route adds an initial terrain stage. Each one's output is cached under ``<output>/<shot>/<stage>/<key>/``, where the key hashes
the stage's settings, the keys of the stages it uses and the stage's own code. Changing anything upstream therefore
re-runs everything downstream of it and nothing else. In particular, a changed opener seed regenerates the cloud
candidate from the new opener; a longer cloud passage re-conforms the effects and the sound windows (placed by
segment, not by frame number).

    terrain  optional: USGS relief and NAIP texture, Blender dive/guide with one shared camera path
    stills   FLUX klein: terrain-aligned end still, orbit still, whiteout still, optional separate opener end pin
    opener   LTX 2.5 first/last: orbit -> whiteout -> opener end pin; only its first ``keep`` frames are used
    cloud    LTX 2.5 image-to-video from the opener's next frame: the plunge into cloud, retimed to keep its speed,
             cut at its first fully white frame
    dive     no model: a dive over real NAIP aerial photos that ends pitching forward (touchdown/dive.py)
    swoop    LTX 2.5 first/last: the dive's last frame -> the end still
    body     joins: colour and brightness ramps, speed match, flow retime, seam softening, shake and speed lines
    end      LTX 2.5 living end; terrain route pins the swoop's last nine frames and holds the final scene
    face     optional final-shot face patch, tracked onto the blurry approach and faded into the living end
    sound    procedural wind, or LTX 2.3 Foley V2A windows (scripts/foley_v2a.py)

A generated stage lists candidate ``seeds``. Without a ``pick`` the build renders every candidate, writes a contact
strip for each and stops so a person (or, later, a calibrated ranking) can choose; the pick is then recorded in the
manifest and in the build log as a manual intervention. Every stage's wall time is logged, cached or not, so a
complete build reports the true time from new inputs to a finished cut, candidates included.
With ``--unattended``, each generated stage must list exactly one seed and no pick; the build runs through without
candidate selection and writes an objective postflight audit.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from . import dive as divemod
from . import video as V

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: fakeout_earth_zoom, dive_fx, foley_v2a
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # geoblock_core
STAGES = ("terrain", "stills", "opener", "dive", "cloud", "swoop", "body", "end", "face", "sound")


class NeedsPick(Exception):
    pass


def guide_height(frame: int) -> float:
    """Height above the landing of measured guide frame ``frame`` (1-based), as terrain_blender.py flies it."""
    return 501 * math.exp(math.log(2 / 501) * (frame - 1) / 80)


def handover_frame(height_m: float) -> int:
    """The guide frame (1-based) nearest to ``height_m`` above the landing."""
    if not 2 < height_m <= 501:
        raise ValueError("[swoop] handover_m must be between 2 and 501 m")
    return int(round(1 + 80 * math.log(501 / height_m) / math.log(501 / 2)))


class LandmarkGate(Exception):
    """The end still moved the distant landmarks; the stills stage retries with another seed."""


class Build:
    def __init__(self, manifest: Path, output: Path | None, url: str | None, dry_run: bool,
                 comfy_python: str | None = None, descent: str | None = None, unattended: bool = False):
        self.manifest_path = manifest
        self.m = tomllib.loads(manifest.read_text())
        self.terrain_mode = self.m.get("terrain", {}).get("enabled", False)
        if isinstance(self.m.get("place"), dict):  # a custom point outside the catalogue (app custom locations)
            from geoblock_core.places import register_place
            register_place(self.m["place"])
        shot = self.m["shot"]
        # descent: how the camera comes down after the cloud. "45" leaves the cloud looking forward and dives at
        # ~45 degrees (Jamie's pick, 10 Oct); "90" tips straight down inside the cloud and dives looking down.
        # It is taken out of the shot block (which every stage's key includes) and resolved into the cloud and dive
        # blocks, so switching it re-renders only the cloud and what follows, never the stills or opener.
        self.descent = str(descent or shot.pop("descent", "45"))
        shot.pop("descent", None)
        _resolve_descent(self.m, self.descent)
        if self.terrain_mode:
            if self.descent != "45":
                raise ValueError("measured-terrain route currently supports only descent = '45'")
            if shot.get("end_still") or shot.get("end_reference"):
                raise ValueError("measured-terrain route builds its end still from the guide's final camera view")
        for k in ("end_still", "end_reference", "subject"):  # image paths are relative to the manifest
            if shot.get(k):
                shot[k] = str((manifest.parent / shot[k]).resolve())
        for k in ("near_source", "far_source"):
            if self.terrain_mode and self.m["terrain"].get(k):
                self.m["terrain"][k] = str((manifest.parent / self.m["terrain"][k]).resolve())
        for k in ("commons_refs", "mid_refs"):  # Commons photo lists for the photo-grounded stills
            if self.m.get("stills", {}).get(k):
                self.m["stills"][k] = [str((manifest.parent / p).resolve()) for p in self.m["stills"][k]]
        from .auto_switch import flux3_defaults
        self.defaults_filled = flux3_defaults(self.m)  # lock_pin "48" + keep_after_pin 8 unless the manifest sets them
        self.name = shot["name"]
        self.root = (output or Path(self.m.get("output", "outputs/touchdown"))) / self.name
        self.url = url
        # The plain ComfyUI behind the build's address, where the model unload is asked for (the API proxy does
        # not serve /free). TOUCHDOWN_COMFY_FREE_URL if the build's address is a proxy.
        self.free_url = os.environ.get("TOUCHDOWN_COMFY_FREE_URL") or None
        self.comfy_python = comfy_python or os.environ.get("TOUCHDOWN_COMFY_PYTHON")
        if self.comfy_python and os.sep in self.comfy_python:
            # Keep a venv's ``bin/python`` symlink intact: resolving it loses the venv's site-packages.
            self.comfy_python = str(Path(self.comfy_python).expanduser().absolute())
        self.dry_run = dry_run
        self.unattended = unattended
        if unattended:
            for stage in ("opener", "cloud", "swoop", "end"):
                cfg = self.m[stage]
                if "pick" in cfg or len(cfg.get("seeds", [])) != 1:
                    raise ValueError(f"--unattended requires exactly one seed and no pick in [{stage}]")
            if "start" in self.m["end"]:
                raise ValueError("--unattended requires an automatic [end] start (start_frac or settle)")
            if self.m["sound"].get("engine", "procedural") != "procedural":
                raise ValueError("--unattended currently requires procedural sound")
            if not self.m["sound"].get("enabled", True):
                raise ValueError("--unattended requires [sound] enabled = true")
        self.keys: dict[str, str] = {}
        self.asm: dict[str, str] = {}  # per stage: its pick + its dependencies' asm (see run)
        self.out: dict[str, Path] = {}
        self.stills_url = os.environ.get("TOUCHDOWN_STILLS_SERVER_URL") or self.m["stills"].get("server_url")
        self.stills_free_url = os.environ.get("TOUCHDOWN_STILLS_FREE_URL") or self.m["stills"].get("server_free_url")
        self.stills_server: subprocess.Popen | None = None
        self.server: subprocess.Popen | None = None  # the ComfyUI this build started, if it owns one
        self.log_path = self.root / "build_log.jsonl"
        self.words = {"ambient": "", **{k: v for k, v in shot.items() if isinstance(v, str)}}  # prompt {fields}
        self.started = time.monotonic()
        self.archive_manifest(manifest)

    def archive_manifest(self, manifest: Path) -> None:
        """Keep a copy of the manifest as this build resolved it (paths made absolute, defaults filled in).

        The manifest on disk is the input; this is what the build actually ran, which the report and the cache key
        alone do not show.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "manifest.json").write_text(json.dumps(
                {"source": str(manifest), "resolved": self.m, "models": self.model_identity(), "comfy_url": self.url},
                indent=2, default=str) + "\n")
        except OSError as exc:
            print(f"could not archive the resolved manifest: {exc}", flush=True)

    # --- caching ---------------------------------------------------------------------------------------------

    def model_identity(self) -> dict:
        """The model files this build will use.

        Part of the cache key: swapping a checkpoint on the ComfyUI server changes the output, and without this
        the cache would still report a hit.
        """
        ids: dict = {}
        try:
            from geoblock_core.workflow import LTX25_MODELS
            ids["ltx25"] = dict(LTX25_MODELS)
        except Exception:  # the pipeline's imports are optional for some stages
            pass
        try:
            import fakeout_earth_zoom as f
            ids["flux3"] = dict(f.FLUX3_MODELS)
            ids["klein"] = {"unet": f.LTX_MODELS["flux_unet"], "clip": f.LTX_MODELS["flux_clip"],
                            "vae": f.LTX_MODELS["flux_vae"]}
        except Exception:
            pass
        return ids

    def key(self, stage: str, deps: tuple[str, ...], extra: dict | None = None) -> str:
        # candidates are cached per seed inside the stage folder, so the seed list and the pick are not part of
        # the key: picking, or adding a seed, re-renders nothing that already exists
        cfg = {k: v for k, v in self.m.get(stage, {}).items() if k not in ("pick", "seeds", "picked_by")}
        payload = {"stage": stage, "cfg": cfg, "shot": self.m["shot"],
                   "deps": [(self.keys[d], self.asm[d]) for d in deps],
                   "models": self.model_identity(),
                   "code": inspect.getsource(getattr(Build, stage)) + (inspect.getsource(Build.horizon_hint) + inspect.getsource(Build.stills_once) + inspect.getsource(Build.fit_camera_to_end) + inspect.getsource(Build.blender_end_fit) + inspect.getsource(Build.photo_end_stills) + inspect.getsource(Build.auto_photo_end) if stage == "stills" else "") + (inspect.getsource(Build.commons_stills) if stage == "stills" else ""), "extra": extra or {}}
        helpers = {"stills": ("comfy_job.py", "../fakeout_earth_zoom.py", "terrain_quality.py", "end_register.py", "face_refine.py", "terrain_blender.py", "photo_op.py", "flux3.py", "photo_pose.py", "auto_switch.py", "osm_structures.py"),
                   "terrain": ("naip_match.py", "osm_structures.py", "terrain_route.py", "prepare_terrain.py", "terrain_blender.py", "view_rank.py", "photo_op.py", "platform_layout.py", "../../src/geoblock_core/place_photos.py", "../../src/geoblock_core/providers.py", "aerial.py", "../../src/geoblock_core/global_dem.py", "../../src/geoblock_core/utm.py"),
                   "opener": ("comfy_job.py", "../fakeout_earth_zoom.py"),
                   "cloud": ("comfy_job.py", "../fakeout_earth_zoom.py", "video.py", "proc_cloud.py"),
                   "dive": ("video.py", "sea.py", "aerial.py"),
                   "swoop": ("comfy_job.py", "../fakeout_earth_zoom.py", "terrain_v2v.py", "terrain_quality.py", "video.py"),
                   "body": ("video.py", "../fakeout_earth_zoom.py", "../dive_fx.py", "birds.py"),
                   "end": ("comfy_job.py", "../fakeout_earth_zoom.py", "video.py"),
                   "face": ("face_refine.py", "comfy_job.py", "../fakeout_earth_zoom.py", "video.py"),
                   "sound": ("wind_bed.py", "pace.py", "video.py")}
        payload["helpers"] = {p: hashlib.sha1((Path(__file__).parent / p).read_bytes()).hexdigest()
                              for p in helpers[stage]}
        if stage in ("opener", "cloud", "swoop", "end") and not self.full_res():
            payload["quality"] = "draft"  # only when set, so full-quality caches keep their keys
        payload["files"] = {k: hashlib.sha1(Path(self.m["shot"][k]).read_bytes()).hexdigest()
                            for k in ("end_still", "end_reference", "subject") if self.m["shot"].get(k)}
        if stage == 'stills' and self.m['stills'].get('photo_ranking'):
            p = Path(self.m['stills']['photo_ranking'])
            payload['files']['photo_ranking'] = hashlib.sha1(p.read_bytes()).hexdigest()
            photo = Path(json.loads(p.read_text())['ranked'][0]['local_path'])
            payload['files']['scene_photo'] = hashlib.sha1(photo.read_bytes()).hexdigest()
        if stage == "terrain":
            payload["module"] = inspect.getsource(Build.terrain)  # e.g. where the NAIP squares are cached
            if self.m['terrain'].get('platform_layout'):
                p = self.root.parent / 'platform_maps' / (self.m['shot']['place'] + '.json')
                if p.is_file():
                    payload['files']['platform_map'] = hashlib.sha1(p.read_bytes()).hexdigest()
            if self.m["terrain"].get("photo_op"):
                p = (Path(self.m["terrain"]["reference_manifest"]) if self.m["terrain"].get("reference_manifest")
                     else self.root.parent / "reference_photos" / self.m["shot"]["place"] / "manifest.json")
                if p.is_file():
                    payload["files"]["reference_manifest"] = hashlib.sha1(p.read_bytes()).hexdigest()
            for k in ("near_source", "far_source"):
                if self.m["terrain"].get(k):
                    p = Path(self.m["terrain"][k])
                    payload["files"][k] = (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
        if stage == "face":
            from .face_refine import DETECTOR
            payload["files"]["face_detector"] = hashlib.sha1(DETECTOR.read_bytes()).hexdigest()
        if stage == "dive" and not self.terrain_mode:
            payload["module"] = inspect.getsource(divemod) + (Path(__file__).parent / "sea.py").read_text()
        if stage == "body" and self.terrain_mode:
            payload["module"] = inspect.getsource(Build.terrain_body)
        if stage == "swoop" and self.terrain_mode:
            payload["module"] = inspect.getsource(Build.terrain_first_last)
        if stage == "sound":  # the sound thread tunes the wind in wind_bed.py: a change there re-runs the sound
            payload["module"] = (Path(__file__).parent / "wind_bed.py").read_text()
        k = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.keys[stage] = k
        return k

    def log(self, **entry) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def run(self, stage: str, deps: tuple[str, ...], fn, extra: dict | None = None) -> Path:
        """Run (or reuse) one stage; ``fn(folder)`` writes the stage's files and returns its main output."""
        folder = self.root / stage / self.key(stage, deps, extra)
        done = folder / "done.json"
        # the folder key leaves picks out, so candidates stay put when a pick changes; ``asm`` (this stage's pick and
        # its dependencies' asm) catches that instead: a changed pick re-assembles this stage and every one after
        # it from the candidates already rendered. A done.json from before asm existed is adopted as it is.
        pick = self.m.get(stage, {}).get("pick")
        if self.unattended and stage in ("opener", "cloud", "swoop", "end"):
            pick = self.m[stage]["seeds"][0]
        asm = hashlib.sha1(json.dumps([pick, [self.asm.get(d) for d in deps]]).encode()).hexdigest()[:10]
        self.asm[stage] = asm
        if done.is_file():
            record = json.loads(done.read_text())
            if record.setdefault("asm", asm) != asm:
                print(f"[{stage}] a pick changed: re-assembling from cached candidates", flush=True)
                done.unlink()
            else:
                done.write_text(json.dumps(record, indent=1))
        if done.is_file():
            out = folder / json.loads(done.read_text())["output"]
            if out.is_file():
                self.out[stage] = out
                print(f"[{stage}] cached {folder}", flush=True)
                return out
            print(f"[{stage}] cached output is missing: rebuilding {out}", flush=True)
            done.unlink()
        if self.dry_run:
            print(f"[{stage}] would run -> {folder}", flush=True)
            self.out[stage] = folder / "(not built)"
            return self.out[stage]
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        out, facts = fn(folder)
        seconds = round(time.monotonic() - t0, 1)
        done.write_text(json.dumps({"output": str(out.relative_to(folder)), "seconds": seconds,
                                    "facts": facts, "asm": asm}, indent=1))
        self.log(stage=stage, key=self.keys[stage], seconds=seconds, facts=facts)
        self.out[stage] = out
        return out

    def candidates(self, stage: str, folder: Path, render, choose=None) -> tuple[Path, dict]:
        """Render each candidate seed (cached per seed) and return the picked one, or stop for a pick.

        ``choose(made)`` picks the seed when the stage has its own objective checks: the first candidate that
        passes is taken, so a seed that dissolves or holds still is not a failed build, it is the next seed. It
        returns the seed, or None when no candidate passes.
        """
        cfg = self.m[stage]
        pick = cfg["seeds"][0] if self.unattended else cfg.get("pick")
        seeds = [pick] if pick is not None else list(cfg["seeds"])
        if pick is None and len(seeds) == 1:  # one seed per stage (Jamie, 11 Oct): it works in one run or not at all
            pick = seeds[0]
            cfg.setdefault("picked_by", "single")
        made = {}
        for seed in seeds:
            sub = folder / f"s{seed}"
            out = sub / "output.mp4"
            if not out.is_file():
                t0 = time.monotonic()
                sub.mkdir(parents=True, exist_ok=True)
                shutil.copy(render(seed, sub), out)
                self.log(stage=stage, candidate=seed, seconds=round(time.monotonic() - t0, 1))
            made[seed] = out
        if pick is None and choose is not None:
            pick = choose(made)
            if pick is None:
                raise ValueError(f"{stage}: no candidate passed the stage's own checks ({sorted(made)})")
            cfg.setdefault("picked_by", "checks")
        if pick is None:
            for seed, out in made.items():
                strip(out, out.with_name("strip.jpg"))
            self.log(stage=stage, needs_pick=sorted(made), strips=[str(p.with_name("strip.jpg")) for p in made.values()])
            raise NeedsPick(f"{stage}: pick one of {sorted(made)} (strips in {folder}), set [{stage}] pick = <seed>")
        picked_by = "precommitted" if self.unattended else cfg.get("picked_by", "human")
        self.log(stage=stage, picked=pick, picked_by=picked_by, intervention=picked_by == "human")
        return made[pick], {"seed": pick, "picked_by": picked_by}

    # --- ComfyUI ---------------------------------------------------------------------------------------------

    def comfy(self, job: str, **kw):
        """Run a ComfyUI job (touchdown.comfy_job) here, or in ``--comfy-python`` when the build's own Python
        lacks comfy_sdk (on the desktop: the build runs in the ComfyUI venv for OpenCV, the jobs in the project venv)."""
        kw.setdefault("url", self.url)
        # Unload whatever the last job left behind: the LTX 2.5 transformer is 21.5 GB and the FLUX 3 one 5.8 GB,
        # and on a 24 GB card the next stage's model does not fit while the last one is still resident.
        self.free_comfy(self.free_url or self.url)
        self.free_comfy(kw["url"])
        self.ensure_server()
        if not self.comfy_python:
            from touchdown import comfy_job
            return comfy_job.JOBS[job](**kw)
        here = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(here), str(here.parent / "src")])}
        done = subprocess.run([self.comfy_python, "-m", "touchdown.comfy_job", job, json.dumps(kw)], env=env,
                              stdout=subprocess.PIPE, text=True, check=True)
        print(done.stdout, end="", flush=True)
        return json.loads(done.stdout.strip().splitlines()[-1])["result"]

    @staticmethod
    def free_comfy(url: str) -> None:
        """Ask a ComfyUI server to unload its models before the next job.

        The LTX 2.5 transformer is 21.5 GB and the FLUX 3 transformer 5.8 GB; on a 24 GB card the next stage's
        model does not fit while the last one is still resident. The unload is best effort: a server that does not
        answer (or a proxy that does not serve /free) must not stop the build.
        """
        import urllib.request
        try:
            req = urllib.request.Request(url.rstrip("/") + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass

    def full_res(self) -> bool:
        """[video] quality: "full" (default; LTX's 2x latent upscale to ~1920x1088) or "draft" (960x544, no 2x pass:
        faster, softer; the joins still write 1920x1088)."""
        quality = str(self.m.get("video", {}).get("quality", "full"))
        if quality not in ("full", "draft"):
            raise ValueError(f'[video] quality must be "full" or "draft", not {quality!r}')
        return quality == "full"

    def ltx(self, folder: Path, frames: int, pins: list[tuple[int, Path]], prompt: str, seed: int, tag: str) -> Path:
        return Path(self.comfy("ltx", folder=str(folder), frames=frames, pins=[[i, str(p)] for i, p in pins],
                               prompt=prompt, seed=seed, prefix=f"geoblock/touchdown_{self.name}_{tag}_s{seed}",
                               ltx_upscale=self.full_res()))

    # --- stages ----------------------------------------------------------------------------------------------

    def terrain(self, folder: Path):
        """Fetch measured relief, choose an endpoint, and render a shared dive/guide camera path."""
        from geoblock_core.places import find_place, landing, view_heading
        from geoblock_core.utm import projected_crs
        from . import terrain_route
        from .prepare_terrain import prepare

        cfg = self.m["terrain"]
        place = find_place(self.m["shot"]["place"])
        lat, lon = landing(place)
        if (cfg.get("latitude") is None) != (cfg.get("longitude") is None):
            raise ValueError("[terrain] latitude and longitude must be supplied together")
        if cfg.get("latitude") is not None:  # land away from the catalogue point, e.g. inside the 1 m DEM project
            lat, lon = float(cfg["latitude"]), float(cfg["longitude"])
        projects = place.get("coverage", {}).get("dem1m", [])
        project = cfg.get("project") or (projects[0]["project"] if projects else None)
        zone = cfg.get("utm_zone") or (projects[0]["zone"] if projects else None)
        if not projects and not cfg.get("project") and place.get("country", "US") != "US":
            # outside the US: the national fine DEM (or GLO-30) and WGS 84 / UTM (geoblock_core.global_dem)
            from geoblock_core import global_dem
            plan = global_dem.plan(lat, lon, place.get("country"))
            project, zone = plan.near, cfg.get("utm_zone") or plan.zone
        if not project or not zone:
            raise ValueError("terrain route needs a catalogued 1 m DEM or [terrain] project and utm_zone")
        cache, imagery_source = self.terrain_imagery(place, lat, lon)
        paths = divemod.fetch(lat, lon, cache, divemod.FAR_LEVELS, source=imagery_source)
        blender_imagery = cache
        if cfg.get("naip_match", True):  # colour-match the nested squares (hard line in the high dive, 17 Oct)
            from .naip_match import match
            blender_imagery = match(cache)
        near = Path(cfg["near_source"]) if cfg.get("near_source") else None
        far = Path(cfg["far_source"]) if cfg.get("far_source") else None
        if near is None or far is None:
            fetched_near, fetched_far = terrain_route.acquire(lat, lon, project, int(zone),
                                                                self.root.parent / "dem_cache")
            near = near or fetched_near
            far = far or fetched_far
        dem = folder / "dem"
        prepare(near, far, lat, lon, int(zone), dem)
        import rasterio
        with rasterio.open(near) as src:  # providers.USGS1mDEMProvider: under 80% 1 m coverage -> 1/3" everywhere
            share = src.tags().get("dem_fallback_1m_share")
        dem_note = f"terrain: 10 m DEM fallback (1 m covers {100 * float(share):.0f}%)" if share is not None else None
        if dem_note:
            print(dem_note, flush=True)
        platform_args = []
        if cfg.get('platform_layout', False):
            from .platform_layout import fetch, project_layout
            source = fetch(lat, lon, self.root.parent / 'platform_maps' / (self.m['shot']['place'] + '.json'))
            layout = project_layout(source, json.loads((dem / 'metadata.json').read_text()), int(zone))
            layout_path = folder / 'platform_layout.json'
            layout_path.write_text(json.dumps(layout, indent=2) + '\n')
            platform_args = ['--platform-layout', str(layout_path)]
            if cfg.get('platform_fit', False):
                platform_args.append('--fit-platform')
        if cfg.get("buildings", False):  # Overture blocks in the measured views (cities: Jamie 17 Oct)
            origin = json.loads((dem / "metadata.json").read_text())
            manifest = self.comfy("buildings", x=origin["x"], y=origin["y"], zone=int(zone),
                                  radius_m=float(cfg.get("buildings_radius_m", 1500)),
                                  folder=str(self.root.parent / "buildings"), cache_dir=str(self.root.parent / "overture"))
            platform_args = [*platform_args, "--buildings", str(manifest)]
        if cfg.get("osm_structures", False):  # OSM bridges/towers/buildings with mapped heights (Battery Spencer v11)
            from .osm_structures import prepare as osm_prepare
            structures = folder / "osm_structures.json"
            osm_facts = osm_prepare(lat, lon, int(zone), dem, blender_imagery, structures,
                                    radius_m=float(cfg.get("osm_radius_m", 2500)))
            print(f"terrain: OSM structures {json.dumps(osm_facts)}", flush=True)
            platform_args = [*platform_args, "--structures", str(structures)]
        if dem_note and cfg.get("fallback_min_clearance_m", 9.0):
            # on the 10 m DEM the smoothed ground at the swoop's end reads as a blurred sandy rise (Battery Spencer
            # v12 f128-136): keep the camera this high until the last 12 guide frames
            platform_args = [*platform_args, "--min-clearance", str(float(cfg.get("fallback_min_clearance_m", 9.0)))]
        if cfg.get("glide") is not None:  # approach steepness; "auto" clears the Overture buildings (cities)
            platform_args = [*platform_args, "--glide", str(cfg["glide"])]
        target = (cfg.get("target_x"), cfg.get("target_y"))
        requested_heading = cfg.get("heading", view_heading(place))
        # -1 is the catalogue's "no known view direction" sentinel, never a
        # compass bearing. Resolve it with the same preflight as the landing.
        heading = None if requested_heading is None or float(requested_heading) == -1 else float(requested_heading)
        if heading is not None and not 0 <= heading < 360:
            raise ValueError("[terrain] heading must be 0..359.999 degrees or -1 for automatic")
        selection = None
        photo_views = None
        if cfg.get("photo_op", False):
            if not cfg.get("embedding_rank", False):
                raise ValueError("[terrain] photo_op requires embedding_rank")
            from geoblock_core.place_photos import fetch_place_photos
            references = self.root.parent / "reference_photos" / self.m["shot"]["place"]
            manifest = Path(cfg["reference_manifest"]) if cfg.get("reference_manifest") else references / "manifest.json"
            if not manifest.is_file():
                fetch_place_photos(place["name"], lat, lon, 12, references,
                                   radius_km=float(cfg.get("photo_radius_km", 0.6)), purpose="photo_op")
            if json.loads(manifest.read_text()).get("purpose") != "photo_op":
                raise ValueError("Reference manifest must come from photo_op discovery, not landscape-only filtering")
            env = dict(os.environ)
            if env.get("TOUCHDOWN_EMBED_DEPS"):
                env["PYTHONPATH"] = os.pathsep.join([env["TOUCHDOWN_EMBED_DEPS"], env.get("PYTHONPATH", "")])
            env["HF_HUB_OFFLINE"] = "1"
            ranking_path = folder / "photo_op_ranking.json"
            photo_query = cfg.get("photo_query") or (
                "An eye-level photograph showing a pedestrian photo spot, its standing surface and " + self.m["shot"]["scene"])
            subprocess.run([env.get("TOUCHDOWN_EMBED_PYTHON", sys.executable), "-m", "touchdown.photo_op",
                            "--manifest", str(manifest), "--query", photo_query,
                            "--output", str(ranking_path)], check=True, env=env)
            from pyproj import Transformer
            origin = json.loads((dem / "metadata.json").read_text())
            project_camera = Transformer.from_crs("EPSG:4326", projected_crs(zone), always_xy=True)
            photo_views = []
            for photo in json.loads(ranking_path.read_text())["anchors"]:
                east, north = project_camera.transform(photo["longitude"], photo["latitude"])
                photo_views.append({**photo, "x": east - origin["x"], "y": north - origin["y"]})
            (folder / "photo_anchors.json").write_text(json.dumps(photo_views, indent=2) + "\n")
        view_target = None
        if place.get("view_toward") and heading is not None and cfg.get("view_clearance", True):
            from pyproj import Transformer
            origin = json.loads((dem / "metadata.json").read_text())
            east, north = Transformer.from_crs("EPSG:4326", projected_crs(zone), always_xy=True).transform(
                place["view_toward"]["longitude"], place["view_toward"]["latitude"])
            view_target = (east - origin["x"], north - origin["y"], float(place["view_toward"].get("structure_m", 0.0)))
        if target == (None, None):
            from .auto_switch import search_relaxing_pavement
            if cfg.get("embedding_rank", False):
                # avoid_pavement is dropped (and logged) when no view has a clear natural-ground corridor: a headland
                # whose approach crosses a road or concrete (Battery Spencer) has none
                views, relaxed = search_relaxing_pavement(
                    lambda avoid_pavement: terrain_route.search_views(
                        dem, paths[900], heading=heading, radius_m=int(cfg.get("search_radius_m", 150)),
                        limit=int(cfg.get("view_previews", 8)), varied_surface=True, avoid_pavement=avoid_pavement,
                        photo_views=photo_views, photo_anchor_radius_m=float(cfg.get("photo_anchor_radius_m", 10)),
                        natural=cfg.get("surface") == "natural", view_target=view_target), cfg)
                if relaxed:
                    print("terrain: no clear natural-ground corridor; avoid_pavement relaxed", flush=True)
                plan = folder / "view_plan.json"
                plan.write_text(json.dumps(views, indent=2) + "\n")
                scene = folder / "scene"
                subprocess.run([cfg.get("blender", "blender"), "-b", "-t", "4", "-noaudio", "--python",
                                str(Path(__file__).with_name("terrain_blender.py")), "--", "--resolution", self.guide_resolution(), "--terrain", str(dem),
                                "--imagery", str(blender_imagery), "--output", str(scene), "--heading", "0",
                                "--preview-plan", str(plan), *platform_args], check=True)
                query = cfg.get("view_query") or (self.m["shot"]["scene"] + ". " +
                                                  self.m["shot"].get("swoop_toward", ""))
                route_query = cfg.get("route_query") or (
                    "Four views of a continuous straight camera descent across natural terrain toward " + query)
                ranking_path = folder / "view_ranking.json"
                env = dict(os.environ)
                if env.get("TOUCHDOWN_EMBED_DEPS"):
                    env["PYTHONPATH"] = os.pathsep.join([env["TOUCHDOWN_EMBED_DEPS"],
                                                          env.get("PYTHONPATH", "")])
                env["HF_HUB_OFFLINE"] = "1"
                subprocess.run([env.get("TOUCHDOWN_EMBED_PYTHON", sys.executable), "-m", "touchdown.view_rank",
                                "--plan", str(plan), "--previews", str(scene / "previews" / "final"),
                                "--contacts", str(scene / "previews" / "contacts"),
                                "--query", query, "--route-query", route_query,
                                "--output", str(ranking_path),
                                *(["--avoid-pavement"] if cfg.get("avoid_pavement", False) and not relaxed else [])],
                               check=True, env=env)
                ranking = json.loads(ranking_path.read_text())
                selection = ranking["selected"]
                selection["embedding_similarity"] = ranking["ranked"][0]["similarity"]
                selection["avoid_pavement_relaxed"] = relaxed
            else:
                selection, relaxed = search_relaxing_pavement(
                    lambda avoid_pavement: terrain_route.choose_view(
                        dem, paths[900], heading=heading, radius_m=int(cfg.get("search_radius_m", 150)),
                        avoid_pavement=avoid_pavement, natural=cfg.get("surface") == "natural",
                        view_target=view_target), cfg)
                selection = {**selection, "avoid_pavement_relaxed": relaxed}
            target = (selection["x"], selection["y"])
            heading = selection["heading"]
        elif None in target:
            raise ValueError("terrain target_x and target_y must be supplied together")
        elif cfg.get("embedding_rank", False):
            raise ValueError("[terrain] embedding_rank requires automatic target_x/target_y")
        elif heading is None:
            raise ValueError("[terrain] explicit target_x/target_y also needs an explicit heading")
        scene = folder / "scene"
        scene.mkdir(parents=True, exist_ok=True)
        clearance_args = [x for i, x in enumerate(platform_args) if x == "--min-clearance"
                          or (i and platform_args[i - 1] == "--min-clearance")]
        (scene / "blender_extra_args.json").write_text(json.dumps(clearance_args) + "\n")  # repeated by end-fit renders
        subprocess.run([cfg.get("blender", "blender"), "-b", "-t", "4", "-noaudio", "--python",
                        str(Path(__file__).with_name("terrain_blender.py")), "--", "--resolution", self.guide_resolution(), "--terrain", str(dem),
                        "--imagery", str(blender_imagery), "--output", str(scene), "--heading", str(heading),
                        "--target-x", str(target[0]), "--target-y", str(target[1]), *platform_args], check=True)
        meta = terrain_route.terrain_meta(scene)
        if meta["boundary_mae"] > 0.001:
            raise ValueError("measured dive/guide camera boundary is discontinuous")
        from pyproj import Transformer
        origin = json.loads((dem / "metadata.json").read_text())
        to_wgs84 = Transformer.from_crs(projected_crs(zone), "EPSG:4326", always_xy=True)
        target_lon, target_lat = to_wgs84.transform(origin["x"] + target[0], origin["y"] + target[1])
        final_camera = meta["guide_poses"][-1]["camera_xyz_m"]
        camera_lon, camera_lat = to_wgs84.transform(origin["x"] + final_camera[0],
                                                    origin["y"] + final_camera[1])
        meta.update({"projected_crs": projected_crs(zone),
                     "origin_easting_m": origin["x"], "origin_northing_m": origin["y"],
                     "target_wgs84": {"latitude": target_lat, "longitude": target_lon},
                     "final_camera_wgs84": {"latitude": camera_lat, "longitude": camera_lon}})
        (scene / "terrain_guide.json").write_text(json.dumps(meta, indent=2) + "\n")
        # Keep the per-frame poses in terrain_guide.json, not the compact build log.
        facts = {k: v for k, v in meta.items() if k != "guide_poses"}
        return scene / "guide.mp4", {**facts, "project": project, "utm_zone": zone,
                                      **({"dem_fallback": dem_note} if dem_note else {}),
                                      **({"imagery": imagery_source} if imagery_source != "naip" else {}),
                                      "heading_source": ("embedding_rank" if cfg.get("embedding_rank", False)
                                                         else "view_search" if requested_heading == -1 else "specified"),
                                      "selection": selection or "manifest"}

    def horizon_hint(self, reference) -> str:
        """A sentence giving the measured horizon's height in the frame (17 Oct: FLUX 3 kept Monument Valley's butte
        shapes but set the horizon ~70 px higher than the guide, landmark shift 90-98 px, twice). '' if unknown."""
        if not (self.terrain_mode and reference and self.m["stills"].get("horizon_hint", True)):
            return ""
        import cv2
        img = cv2.imread(str(reference))
        if img is None:
            return ""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sky = (img[..., 0].astype(int) > img[..., 2].astype(int) + 25) & (hsv[..., 2] > 150)
        rows = [int(np.argmin(sky[:, x])) for x in range(sky.shape[1]) if sky[0, x]]
        if len(rows) < sky.shape[1] // 2:
            return ""
        line = float(np.percentile(rows, 75)) / img.shape[0]  # the plain horizon, below most peaks
        return (f" The far horizon is exactly {round(100 * line)}% of the way down the frame, as in image 1; keep the "
                f"distant landforms at the same size and positions as in image 1, do not raise or lower the horizon.")

    def stills(self, folder: Path):
        """Draw the stills.

        The stills are the only stage that needs the FLUX 3 weights, and they are the only ones that leave a
        server holding them: ComfyUI's paging does not give the memory back when it is asked to unload them, and
        the LTX 2.5 transformer (21.5 GB) then cannot be faulted in on a 24 GB card. So the server is restarted
        after the stills, and the video stages get a clean one.
        """
        process = self.start_stills_server()
        try:
            self.warn_about_the_card()
            return self._stills(folder)
        finally:
            self.stop_stills_server(process)
            if self.server is not None:  # a server this build owns: give the card back before the video stages
                self.stop_server()

    def _stills(self, folder: Path):
        cfg, shot = self.m["stills"], self.m["shot"]
        seed = cfg["seed"]
        # orbit_region: what the orbit still shows. A mountain region painted from orbit came out as close snowy
        # peaks, and the opener and cloud carried them on as rocky "clouds" (Yosemite v1); an orbit-scale phrase
        # with a cloud deck below keeps the orbit far up and gives the plunge real cloud to enter
        words = {**self.words, "region": cfg.get("orbit_region") or self.words["region"]}
        if self.terrain_mode and cfg.get("end_from_photo"):
            return self.photo_end_stills(folder, words)
        if self.terrain_mode and cfg.get("commons_refs"):
            return self.commons_stills(folder, words)
        reference = (str(self.out["terrain"].parent / "guide_frames" / "0081.png")
                     if self.terrain_mode else shot.get("end_reference"))
        prompt = cfg.get("end_prompt")
        scene_reference = None
        if self.terrain_mode and cfg.get('photo_reference', False):
            ranking_path = (Path(cfg['photo_ranking']) if cfg.get('photo_ranking') else
                            self.out['terrain'].parent.parent / 'photo_op_ranking.json')
            ranked = json.loads(ranking_path.read_text())['ranked']
            scene_reference = ranked[0]['local_path']
            image_index = 3 if shot.get('subject') else 2
            prompt = (prompt or 'Preserve image 1 camera view and geometry. Add {person} standing {scene}.')
            prompt += (f' Image {image_index} documents the actual pedestrian setting and material treatment. '
                       'Keep the standing surface and path edges from image 1 in exactly the same positions. '
                       'Use the photo for materials; retain image 1 viewpoint and path layout.')
        if self.terrain_mode and not prompt:
            prompt = ("Make a real eye-level photograph from the exact same camera viewpoint as image 1. "
                      "Keep the skyline, road orientation and horizon where image 1 has them. "
                      "Add {person} standing {scene}, centered, full length, about ten metres from the camera. "
                      "Natural daylight, photographic detail, no text.")
        if self.terrain_mode and shot.get("subject") and prompt:
            prompt += " Use image 2 for the person's exact face, hair, body and clothing."
        # [stills] flux3 = ["end", "opener_end", "orbit"]: those stills from FLUX 3 Image Klein at 2K, on the
        # ComfyUI v0.41.3+ server at flux3_url (its proxy; the desktop's 8190 install is older), which is told to
        # unload afterwards (flux3_free_url = the server itself). FLUX.2 klein kept the flat whiteout better (17 Oct).
        # layout_retries: a FLUX 3 end still that moved the distant landmarks (Grand Canyon v5: canyon ~15% smaller,
        # 133 inliers, median shift 35 px) is redrawn with the next seed, up to this many times; the first passing
        # set is kept, else the build stops as before
        retries = int(cfg.get("layout_retries", 2)) if self.terrain_mode else 0
        for attempt in range(retries + 1):
            seed = cfg["seed"] + attempt
            try:
                return self.stills_once(folder, cfg, shot, seed, words, reference, prompt, scene_reference,
                                        locals().get("ranked"))
            except LandmarkGate as e:
                print(f"stills seed {seed}: {e}", flush=True)
                if attempt == retries:
                    switched = self.auto_photo_end(folder, words, failures=retries + 1)
                    if switched is not None:
                        return switched
                    raise ValueError("endpoint changed measured foreground: " + str(e)) from None

    def auto_photo_end(self, folder: Path, words: dict, failures: int):
        """After the FLUX end still failed the landmark gate on every seed: build it from a real photo instead
        (photo_end_stills) when the manifest leaves end_from_photo unset, a geotagged Commons photo of the place
        exists and OSM maps its structures with heights (auto_switch.use_photo_end). The OSM structures are
        prepared for the photo route's re-render if the terrain did not have them. Returns its result, or None."""
        from geoblock_core.place_photos import fetch_place_photos
        from geoblock_core.places import find_place, landing
        from . import terrain_route
        from .auto_switch import use_photo_end
        from .osm_structures import prepare as osm_prepare
        cfg, shot, tcfg = self.m["stills"], self.m["shot"], self.m["terrain"]
        if "end_from_photo" in cfg or not cfg.get("auto_photo_end", True):
            return None
        place = find_place(shot["place"])
        lat, lon = landing(place)
        terrain_dir = self.out["terrain"].parent.parent
        meta = terrain_route.terrain_meta(self.out["terrain"].parent)
        refs = self.root.parent / "reference_photos" / shot["place"]
        try:
            if not (refs / "manifest.json").is_file():
                fetch_place_photos(place["name"], lat, lon, 12, refs, radius_km=float(cfg.get("photo_radius_km", 0.6)),
                                   heading_deg=float(meta["heading"]), purpose="photo_op")
            photos = json.loads((refs / "manifest.json").read_text())["selected"]
        except (ValueError, OSError) as exc:
            print(f"auto photo end: no photos ({exc})", flush=True)
            return None
        structures_file = terrain_dir / "osm_structures.json"
        if not structures_file.is_file():
            imagery = self.root.parent / "imagery" / shot["place"] / self.imagery_subfolder(place, lat, lon)
            imagery = imagery / "matched" if tcfg.get("naip_match", True) else imagery
            osm_prepare(lat, lon, int(tcfg.get("utm_zone") or json.loads((terrain_dir / "dem" / "metadata.json").read_text())["utm_zone"]),
                        terrain_dir / "dem", imagery, structures_file, radius_m=float(tcfg.get("osm_radius_m", 2500)))
        switch, reason = use_photo_end(cfg, failures, photos, json.loads(structures_file.read_text()))
        print(f"auto photo end: {'yes' if switch else 'no'} ({reason})", flush=True)
        self.log(stage="stills", auto_photo_end=switch, reason=reason)
        if not switch:
            return None
        return self.photo_end_stills(folder, words)

    def imagery_subfolder(self, place: dict, lat: float, lon: float) -> str:
        """"" for NAIP (its old cache folder), else the touchdown.aerial source id ([terrain] imagery, default by the
        place's country), as the dive stage files them."""
        from touchdown import aerial
        source = self.m.get("terrain", {}).get("imagery", "auto")
        if source == "auto":
            source = aerial.pick(place.get("country", ""), lat, lon)
        return "" if source == "naip" else source

    def terrain_imagery(self, place: dict, lat: float, lon: float) -> tuple[Path, str]:
        """The aerial squares' folder for the terrain drape and its touchdown.aerial source id."""
        cache = self.root.parent / "imagery" / self.m["shot"]["place"]
        if self.m["terrain"].get("latitude") is not None:
            # the squares are centred on the landing and their file names carry no position: a moved landing
            # needs its own folder (run 2, 17 Oct, textured the Utah DEM with the Arizona car park)
            cache = cache.with_name(f"{cache.name}_{lat:.5f}_{lon:.5f}")
        sub = self.imagery_subfolder(place, lat, lon)
        return (cache / sub if sub else cache), (sub or "naip")

    def guide_resolution(self) -> str:
        """[terrain] guide_resolution (default 1920x1088; was 960x544, which left the swoop's land soft, MV v11)."""
        return str(self.m.get("terrain", {}).get("guide_resolution", "1920x1088"))

    def blender_end_fit(self, terrain: Path, scene: Path, meta: dict, render: Path, fit_json: Path | None,
                        person: list | None = None) -> None:
        """Re-render the guide's last frames in Blender with the end camera in ``fit_json`` (terrain_blender --end-fit)."""
        tcfg = self.m["terrain"]
        from geoblock_core.places import find_place, landing
        place = find_place(self.m["shot"]["place"])
        lat, lon = ((float(tcfg["latitude"]), float(tcfg["longitude"])) if tcfg.get("latitude") is not None
                    else landing(place))
        imagery, _ = self.terrain_imagery(place, lat, lon)
        if tcfg.get("naip_match", True):
            imagery = imagery / "matched"
        extra = []
        if (terrain / "platform_layout.json").is_file():
            extra = ["--platform-layout", str(terrain / "platform_layout.json")]
            if tcfg.get("platform_fit", False):
                extra.append("--fit-platform")
        if (scene / "glide.json").is_file():
            extra += ["--glide", str(json.loads((scene / "glide.json").read_text())["glide"])]
        if (terrain / "osm_structures.json").is_file():
            extra += ["--structures", str(terrain / "osm_structures.json")]
        if (scene / "blender_extra_args.json").is_file():
            extra += json.loads((scene / "blender_extra_args.json").read_text())
        extra += list(person or [])
        subprocess.run([tcfg.get("blender", "blender"), "-b", "-t", "4", "-noaudio", "--python",
                        str(Path(__file__).with_name("terrain_blender.py")), "--", "--resolution", self.guide_resolution(), "--terrain", str(terrain / "dem"),
                        "--imagery", str(imagery), "--output", str(render), "--heading", str(meta["heading"]),
                        "--target-x", str(meta["target_x"]), "--target-y", str(meta["target_y"]), *extra,
                        *(["--end-fit", str(fit_json)] if fit_json else [])], check=True, stdout=subprocess.DEVNULL)

    def fit_camera_to_end(self, folder: Path, guide: Path) -> dict:
        """Move the camera to the still when the still cannot be moved to the camera (17 Oct): fit the end camera's
        pitch, yaw and zoom from the skylines (end_register.fit_camera), re-render the guide's last ``ease_frames``
        frames in Blender with it, eased in, and write the whole fitted guide to ``<stills>/guide_fit/``; the swoop
        then follows that guide. Returns the fit, the Blender time and the fitted last frame (``guide``), or why not."""
        import time
        import cv2
        from . import terrain_route
        from .end_register import W, H, fit_camera, person_box, skyline
        cfg, tcfg = self.m["stills"], self.m["terrain"]
        if tcfg.get("buildings", False):
            return {"applied": False, "reason": "the Overture buildings are not re-rendered for a fitted end camera"}
        g = cv2.resize(cv2.imread(str(guide)), (W, H), interpolation=cv2.INTER_AREA)
        e = cv2.resize(cv2.imread(str(folder / "end.png")), (W, H), interpolation=cv2.INTER_AREA)
        box = person_box(e)
        if box is None:
            return {"applied": False, "reason": "no person found to mask from the still's skyline"}
        pad = (box[2] - box[0]) // 6
        limit = float(cfg.get("fit_camera_max_residual", 12))
        gl, el = skyline(g), skyline(e, (box[0] - pad, box[2] + pad))
        fit = fit_camera(gl, el)
        if not fit:
            return {"applied": False, "reason": "too few skyline columns"}
        lifts = None
        if fit["residual_px"] > limit and cfg.get("fit_camera_height", True):
            # fit_camera_height: the still's horizon is far above the map's because FLUX drew a higher viewpoint
            # (Battery Spencer v9, horizon ~20% vs ~52%): pitch down to -20 deg, and pick the end camera's lift (up
            # and back along the heading, 0-40 m) whose rendered last frame matches the still best
            wide = fit_camera(gl, el, pitch=(-20.0, 8.0))
            if wide and wide["residual_px"] <= limit:
                fit, lifts = wide, []
        if fit["residual_px"] > limit:
            return {**fit, "applied": False, "reason": "skylines do not agree under one pitch, yaw and zoom"}
        ease = int(cfg.get("fit_camera_ease", 24))
        scene = self.out["terrain"].parent
        terrain = scene.parent
        meta = terrain_route.terrain_meta(scene)
        out = folder / "guide_fit"
        render = out / "render"
        render.mkdir(parents=True, exist_ok=True)
        if lifts is not None:
            best = None
            for lift in (0.0, 10.0, 20.0, 30.0, 40.0):
                trial = out / f"lift_{int(lift)}"
                trial.mkdir(exist_ok=True)
                (trial / "end_fit.json").write_text(json.dumps({**fit, "lift_m": lift, "back_m": lift,
                                                                "ease_frames": 1}) + "\n")
                self.blender_end_fit(terrain, scene, meta, trial / "render", trial / "end_fit.json")
                got = self.end_layout(trial / "render" / "guide_frames" / "0081.png", folder / "end.png")
                row = {"lift_m": lift, "inliers": got["landmark_inliers"], "shift_px": got["landmark_median_shift_px"],
                       "passes": not any("landmark" in w or "unverifiable" in w for w in got["warnings"])}
                skyline_px = (got.get("photo_geometry") or {}).get("skyline_residual_px")
                if "photo_geometry" in got:
                    row["skyline_px"] = skyline_px
                lifts.append(row)
                key = ((row["passes"], -(skyline_px if skyline_px is not None else 999)) if "photo_geometry" in got
                       else (row["passes"], row["inliers"], -(row["shift_px"] or 999)))
                if best is None or key > best[0]:
                    best = (key, lift)
            fit = {**fit, "lift_m": best[1], "back_m": best[1], "lift_trials": lifts}
        (out / "end_fit.json").write_text(json.dumps({**fit, "ease_frames": ease}, indent=2) + "\n")
        start = time.time()
        self.blender_end_fit(terrain, scene, meta, render, out / "end_fit.json")
        seconds = round(time.time() - start, 1)
        frames = out / "guide_frames"
        frames.mkdir(exist_ok=True)
        for f in range(1, 82):
            source = render / "guide_frames" / f"{f:04d}.png" if f > 81 - ease else scene / "guide_frames" / f"{f:04d}.png"
            shutil.copy(source, frames / f"{f:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames / "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "guide.mp4")], check=True)
        return {**fit, "applied": True, "ease_frames": ease, "blender_seconds": seconds, "guide": frames / "0081.png"}

    def realism_score(self, folder: Path, still: Path, guide: Path | None = None) -> dict:
        """How photographic the still's background reads (EmbeddingGemma, photograph vs low-poly render).

        The 20 Oct review calls the "3D terrain model" look the main weakness. This is the only objective measure of
        it in the project: the margins were -0.03 on Monument Valley's faceted v12 still and +0.08 on the
        photographic v14. It needs the embedding environment, so it is a diagnostic when one is configured.
        """
        import subprocess
        out = folder / "realism.json"
        env = dict(os.environ)
        deps = env.get("TOUCHDOWN_EMBED_DEPS")
        here = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (deps, str(here), env.get("PYTHONPATH", "")) if p])
        env["HF_HUB_OFFLINE"] = "1"
        images = [str(still)] + ([str(guide)] if guide and guide.is_file() else [])
        try:
            subprocess.run([env.get("TOUCHDOWN_EMBED_PYTHON", sys.executable), "-m", "touchdown.realism",
                            "--images", *images, "--output", str(out)], env=env, check=True,
                           capture_output=True, timeout=600)
            scored = json.loads(out.read_text())
            for row in scored:
                row.pop("image", None)
            return {"assessed": True, "margins": scored}
        except Exception as exc:
            return {"assessed": False, "reason": f"{type(exc).__name__}: {exc}"}

    def stills_once(self, folder, cfg, shot, seed, words, reference, prompt, scene_reference, ranked):
        flux3 = list(cfg.get("flux3", []))
        # where the stills are drawn, and where that server is asked to unload afterwards (the unload goes to the
        # plain ComfyUI: the proxy does not serve /free)
        route = self.stills_route()
        # structure conditioning (FLUX 3 end stills by default, 18 Oct): FLUX gets the guide's outline map, not the
        # shaded render, so there are no low-poly facets or clay shading to copy (Monument Valley v12's buttes)
        # "both" adds the render as well (image 1), with the outline map as image 3: the outline alone lets the model
        # invent its own landforms, and the still then shows a landscape that is not the one the camera flew over
        # (the review of 21 Oct: the final view sat inside the dive's view rather than being it).
        conditioning = self.end_conditioning()
        structure = self.terrain_mode and reference and conditioning in ("structure", "both")
        both = self.terrain_mode and reference and conditioning == "both"
        model_reference = self.structure_map(Path(reference), folder / "end_structure.png") if structure else reference
        if both:
            # image 1 stays the layout (the prompt says "the horizon and the sky where image 1 has them"), and the
            # measured render goes in as image 3, for the land. The other order made the model take the render's
            # framing as well as its texture, and the still was no longer the view the camera flew into.
            scene_reference = str(reference)
            role = ("Image 1 is a structure map of the measured scene from the exact camera position: light grey is "
                    "sky, dark grey is land and buildings, black lines are their edges. Image 3 is the same scene "
                    "rendered with the real aerial photographs draped on the elevation model: keep its land, its "
                    "rock, its roads and its trees in the places they are in, and do not move the ridgelines.")
        else:
            role = self.STILL_ROLE["structure"] if structure else self.STILL_ROLE["render"]
        if prompt and structure:
            prompt = role + " " + prompt
        opener_end = cfg.get("opener_end_prompt", "").format(**self.words)
        out = self.comfy("stills", folder=str(folder), seed=seed, name=self.name, words=words, flux3=flux3, **route,
                         end_still=shot.get("end_still") and str(shot["end_still"]),
                         end_reference=model_reference and str(model_reference),
                         subject=shot.get("subject") and str(shot["subject"]),
                         scene_reference=scene_reference,
                         end_prompt=prompt and (prompt.format(**self.words) + self.horizon_hint(reference) + self.PHOTOREAL),
                         opener_end_prompt=(opener_end + self.PHOTOREAL) if opener_end else None,
                         cloud_prompt=cfg.get("cloud_prompt"))
        for name, path in out.items():
            shutil.copy(path, folder / f"{name}.png")
        facts = {"seed": seed, "end_given": bool(shot.get("end_still"))}
        if self.terrain_mode:
            guide = self.out["terrain"].parent / "guide_frames" / "0081.png"
            layout = self.end_layout(guide, folder / "end.png")
            unverifiable = [w for w in layout["warnings"] if w.startswith("end still unverifiable")]
            if unverifiable and cfg.get("layout_gate", "stop") != "warn":
                (folder / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
                raise ValueError(unverifiable[0])  # not a LandmarkGate: no seed retries, no camera fit
            # register_end: FLUX 3 draws the distant terrain at its own scale and height (Grand Canyon v5/v6,
            # Monument Valley v6); one zoom-in + crop of the whole still fitted from the skylines brings it onto the
            # guide when that is possible without padding or cutting the person (end_register.py). The gate judges
            # the registered still; the FLUX still is kept as end_flux.png
            # only a still that fails the landmark test is registered (Monument Valley v7 seed 2 passed as drawn,
            # 50 inliers 3.95 px; a 1% crop gave 41 inliers 5.16 px)
            if cfg.get("register_end", True) and any("landmarks moved" in w for w in layout["warnings"]):
                from .end_register import register
                registration = register(guide, folder / "end.png", folder / "end_registered.png")
                pre = {k: layout[k] for k in ("landmark_inliers", "landmark_median_shift_px", "warnings")}
                if registration.get("applied"):
                    shutil.move(folder / "end.png", folder / "end_flux.png")
                    shutil.move(folder / "end_registered.png", folder / "end.png")
                    layout = self.end_layout(guide, folder / "end.png")
                layout = {**layout, "registration": registration, "pre_registration": pre}
                print(f"stills seed {seed}: registration {json.dumps(registration)}; pre {pre['landmark_inliers']} "
                      f"inliers {pre['landmark_median_shift_px']} px, post {layout['landmark_inliers']} inliers "
                      f"{layout['landmark_median_shift_px']} px", flush=True)
            (folder / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
            # a medium close-up covers much of the measured foreground with the person, which this gate reads as an
            # invented foreground: ``layout_gate = "warn"`` records the warnings instead of stopping the build
            # "landmarks": stop only when the distant landmarks moved; a real wall, railing or kerb that the scene
            # describes but the measured pavement smears (Yosemite Tunnel View's granite wall, 17 Oct: landmark shift
            # 1 px, foreground flagged) is kept
            gate = cfg.get("layout_gate", "stop")
            stopping = [w for w in layout["warnings"] if gate == "stop" or (gate == "landmarks" and self.redrawable(w))]
            shutil.rmtree(folder / "guide_fit", ignore_errors=True)  # a previous seed's fitted guide
            if (stopping and all("landmarks moved" in w for w in stopping) and cfg.get("fit_camera_to_end", True)
                    and not (layout.get("registration") or {}).get("applied")):
                fitted = self.fit_camera_to_end(folder, guide)
                post = self.end_layout(fitted["guide"], folder / "end.png") if fitted.get("guide") else None
                layout = {**layout, "camera_fit": {**{k: v for k, v in fitted.items() if k != "guide"},
                                                   **({"post": {k: post[k] for k in ("landmark_inliers",
                                                       "landmark_median_shift_px", "warnings")}} if post else {})}}
                print(f"stills seed {seed}: camera fit {json.dumps(layout['camera_fit'])}", flush=True)
                if post is not None:
                    still = [w for w in post["warnings"] if gate == "stop" or (gate == "landmarks" and self.redrawable(w))]
                    if not still:
                        layout = {**layout, **{k: post[k] for k in post if k != "warnings"}, "warnings": post["warnings"]}
                        stopping = []
                    else:
                        shutil.rmtree(folder / "guide_fit", ignore_errors=True)
                (folder / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
            if stopping:
                if all(self.redrawable(w) for w in stopping):
                    raise LandmarkGate(f"{'; '.join(stopping)} ({layout.get('landmark_inliers')} inliers, "
                                       f"{layout.get('landmark_median_shift_px')} px, skyline "
                                       f"{(layout.get('photo_geometry') or {}).get('skyline_residual_px')} px)")
                raise ValueError("endpoint changed measured foreground: " + "; ".join(stopping))
            facts["layout"] = layout
            # The end still is painted from the outline map, so the ground under the figure is the model's. This
            # measures whether it invented a patch of coloured ground (the "landing pad" artefact, MV v14).
            try:
                import cv2
                from .end_register import person_box
                from .terrain_quality import ground_patch
                still_image = cv2.resize(cv2.imread(str(folder / "end.png")), (960, 544), interpolation=cv2.INTER_AREA)
                facts["ground_patch"] = ground_patch(folder / "end.png", person_box(still_image))
                print(f"ground under the figure: {json.dumps(facts['ground_patch'])}", flush=True)
            except Exception as exc:
                facts["ground_patch"] = {"assessed": False, "reason": f"{type(exc).__name__}: {exc}"}
            if os.environ.get("TOUCHDOWN_EMBED_PYTHON"):
                guide_frame = self.out["terrain"].parent / "guide_frames" / "0081.png"
                facts["realism"] = self.realism_score(folder, folder / "end.png", guide_frame)
                print(f"realism: {json.dumps(facts['realism'])}", flush=True)
            surface_mask = self.out['terrain'].parent / 'guide_frames_surface_masks' / '0081.png'
            if surface_mask.is_file():
                from .terrain_quality import assess_surface_layout
                surface = assess_surface_layout(folder / 'end.png', surface_mask)
                (folder / 'surface_layout.json').write_text(json.dumps(surface, indent=2) + '\n')
                facts['shared_surface'] = surface
                if surface['warnings']:
                    raise ValueError('endpoint changed shared platform layout: ' + '; '.join(surface['warnings']))
            if scene_reference and ranked:
                facts['scene_reference'] = {**{k: ranked[0].get(k) for k in ('commons_page', 'license', 'license_url', 'artist', 'title')}, 'path': scene_reference, 'scope': 'materials; pose alignment not established'}
            elif scene_reference:
                facts['scene_reference'] = {'path': scene_reference, 'scope': 'the measured render, as the layout reference'}
        return folder / "end.png", facts

    # Appended in code to every FLUX end and opener-end prompt, whatever the manifest says (Jamie 18 Oct: "how can we
    # set immutable characteristics of the output"). Positive wording: FLUX 3 Klein distilled runs at cfg 1, no negatives
    PHOTOREAL = (" The result is a real, unretouched camera photograph: natural rock, soil, sand and plants with fine "
                 "weathered detail and irregular natural edges, real sunlight and soft atmospheric haze on distant "
                 "landforms, like a professional landscape photo.")

    def end_conditioning(self) -> str:
        """[stills] end_conditioning for the terrain end still: "structure" (outline map; default when FLUX 3 draws
        the end) or "render" (the shaded Blender frame). The stills adherence test (17 Oct, ab65edc) found FLUX 3 +
        structure map held the skyline to 3 px with a real-photo look; the render route copies the model's facets."""
        cfg = self.m["stills"]
        return cfg.get("end_conditioning", "structure" if "end" in cfg.get("flux3", []) else "render")

    @staticmethod
    def redrawable(warning: str) -> bool:
        """Warnings that send the end still to the next seed (layout_retries) rather than stopping the build."""
        return "landmarks moved" in warning or "structure map" in warning

    def end_layout(self, guide: Path, still: Path) -> dict:
        """The end-still gate. With structure conditioning the SIFT landmark test is replaced by the skyline one
        (assess_photo_geometry): a still that no longer copies the render's texture shares few SIFT features with it
        even when the buttes line up (Battery Spencer v10: 0 of 115 at a 3 px skyline fit). A still that pasted the
        grey structure map in is redrawn. The foreground checks are as usual."""
        from .terrain_quality import assess_endpoint_layout, assess_photo_geometry
        layout = assess_endpoint_layout(guide, still)
        if self.end_conditioning() != "structure":
            return layout
        geometry = assess_photo_geometry(guide, still, max_residual=float(self.m["stills"].get("skyline_max_px", 12)))
        diagram = self.looks_like_diagram(still)
        warnings = [w for w in layout["warnings"] if "landmark" not in w and not w.startswith("end still unverifiable")]
        warnings += geometry["warnings"]
        if diagram > 0.15:
            warnings.append(f"end still pasted the structure map in ({diagram:.0%} flat grey)")
        sift = {k: layout[k] for k in ("landmark_inliers", "landmark_median_shift_px", "landmark_guide_features")
                if k in layout}
        return {**layout, "warnings": warnings, "sift_not_used": sift, "photo_geometry": geometry,
                "diagram_share": round(diagram, 3)}

    # What the photo-grounded stills must keep, by scene type (scene_type.py). The user picks the spot, so nothing
    # that is really there may be painted over: buildings, car parks and roads stay, made real (Jamie 17 Oct).
    KEEP = {"urban": "every building with its height, roofline and facade position, every street, kerb, pavement, "
                     "plaza, lawn, tree line and paved area",
            "landmark": "the landmark itself with its exact shape, size and position, and every building, path, "
                        "plaza, wall, paved area and landform around it",
            "nature": "every landform, ridge, rock, slope, path, road, building, car park and paved area"}
    STILL_ROLE = {"render": "Image 1 is a crude 3D render of the measured terrain and buildings from the exact camera "
                            "position (aerial photographs draped on an elevation model, buildings as plain grey blocks).",
                  "structure": "Image 1 is a structure map of the measured scene from the exact camera position: light "
                               "grey is sky, dark grey is land and buildings, black lines are their edges.",
                  "both": "Image 1 is a crude 3D render of the measured terrain and buildings from the exact camera "
                          "position (aerial photographs draped on an elevation model, so paved areas, roads and roofs "
                          "show their real colours); image 2 is its structure map (light grey sky, dark grey land, "
                          "black edges) for the exact outlines."}
    COMMONS_END = ("Make one real eye-level photograph of {person} standing {scene}. {role} Keep the camera direction, "
                   "the horizon and the distant landforms exactly, and keep {keep} exactly where image 1 has them; do "
                   "not invent, remove or move any structure. Image 1 only lacks realism: give everything real "
                   "materials, textures, colour and light, using the other images (photographs or ground swatches of "
                   "this place) for appearance only, never for their viewpoints or compositions. Put the person in the centre about ten metres "
                   "from the camera, full length, still and facing the camera, both feet planted with a natural "
                   "contact shadow. Exactly one person. Natural daylight, photorealistic, sharp detail, no text.")
    COMMONS_MID = ("Make one real aerial photograph taken from a drone about {height} metres up, looking forward and "
                   "down. {role} Keep its camera direction and the exact layout of everything below it: {keep}, all in "
                   "the same positions and sizes as in image 1; do not invent, remove or move any structure, do not "
                   "add landforms that image 1 does not show, and do not reproduce its flat smeared texture or grey "
                   "blocks. Give everything real "
                   "materials, colour, shadows and light, using the other images (photographs or ground swatches of this place) "
                   "for appearance only, never for their viewpoints or compositions. No people. Natural daylight, photorealistic, "
                   "no text.")

    AMBIENT = {"urban": "people walk past along the paths and pavements behind her, pigeons hop and flutter, tree "
                        "leaves sway, flags and awnings stir, and clouds move slowly above the buildings",
               "landmark": "visitors drift past in the distance, flags ripple, trees and plants sway in the breeze, "
                           "and clouds move slowly behind the landmark",
               "nature": "the wind streams across the ground, plants and scrub shiver and sway, fine dust or sand "
                         "blows past, and cloud shadows slide slowly across the land behind her"}

    def scene_ambient(self) -> str:
        """The living end's background motion for this scene type (scene_type.json beside the stills)."""
        st = self.out.get("stills")
        kind = "nature"
        if st and (st.parent / "scene_type.json").is_file():
            kind = json.loads((st.parent / "scene_type.json").read_text()).get("type", kind)
        return self.AMBIENT.get(kind, self.AMBIENT["nature"])

    def scene_type(self, folder: Path) -> dict:
        """landmark / urban / nature for this request, with EmbeddingGemma 2 (scene_type.py), cached in ``folder``."""
        out = folder / "scene_type.json"
        if out.is_file():
            return json.loads(out.read_text())
        cfg, shot = self.m["stills"], self.m["shot"]
        if cfg.get("scene_type"):  # a manifest override skips the model
            out.write_text(json.dumps({"type": cfg["scene_type"], "source": "manifest"}) + "\n")
            return json.loads(out.read_text())
        from geoblock_core.places import find_place
        place = find_place(shot["place"])
        text = f"{place.get('name', shot['place'])}, {place.get('region', '')}, {place.get('kind', '')}: {shot.get('scene', '')}"
        env = dict(os.environ)
        deps = env.get("TOUCHDOWN_EMBED_DEPS")
        here = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (deps, str(here), env.get("PYTHONPATH", "")) if p])
        env["HF_HUB_OFFLINE"] = "1"
        subprocess.run([env.get("TOUCHDOWN_EMBED_PYTHON", sys.executable), "-m", "touchdown.scene_type",
                        "--text", text, "--output", str(out)], env=env, check=True)
        return json.loads(out.read_text())

    @staticmethod
    def structure_map(guide: Path, out: Path) -> Path:
        """Sky light grey, land dark grey, edges black: the measured scene's layout without its smeared texture."""
        import cv2
        out.parent.mkdir(parents=True, exist_ok=True)  # cv2.imwrite does not create the folder, and fails quietly
        img = cv2.imread(str(guide))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sky = (img[..., 0].astype(int) > img[..., 2].astype(int) + 25) & (hsv[..., 2] > 150)
        m = np.full(img.shape, 70, np.uint8)
        m[sky] = 235
        m[cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 30, 90) > 0] = 0
        cv2.imwrite(str(out), m)
        return out

    @staticmethod
    def looks_like_diagram(still: Path) -> float:
        """Share of pixels at the structure map's flat dark grey (its land): a still that pasted the diagram in scores
        high. The map's light grey (its sky, 235) is not counted: an overcast sky is the same neutral near-white, and
        counting it rejected every photographic Monument Valley still with a cloudy sky as "pasted the structure map"
        (app first run and v14, 18 Oct: 16-36% "flat grey", all of it sky, 0.0% of the dark grey)."""
        import cv2
        img = cv2.imread(str(still)).astype(int)
        flat = (np.abs(img[..., 0] - img[..., 1]) < 4) & (np.abs(img[..., 1] - img[..., 2]) < 4)
        grey = flat & (np.abs(img[..., 0] - 70) < 6)
        return float(grey.mean())

    def prepared_refs(self, folder: Path, refs: list, tag: str) -> tuple[list[Path], list[dict]]:
        """Commons photos as references. ``ref_mode`` "full" passes them whole; "swatch" passes only a crop of their
        lower ground (materials and plants, no landforms or composition to copy); "auto" (default) keeps a photo
        whole only when its Commons coordinates are within ``ref_full_km`` (0.3) of the landing, because a model
        handed a photo of somewhere else drifts toward that photo's view (Monument Valley 8/9, 17 Oct)."""
        import cv2
        cfg = self.m["stills"]
        mode = cfg.get("ref_mode", "auto")
        limit = float(cfg.get("ref_full_km", 0.3))
        out, audit = [], []
        for r in map(Path, refs):
            distance = None
            manifest = r.parent / "manifest.json"
            if manifest.is_file():
                page = r.stem.split("_")[-1]
                for item in json.loads(manifest.read_text()).get("candidates", []):
                    if str(item.get("page_id")) == page:
                        distance = item.get("distance_km")
            whole = mode == "full" or (mode == "auto" and distance is not None and distance <= limit)
            if whole:
                out.append(r)
            else:
                img = cv2.imread(str(r))
                h, w = img.shape[:2]
                swatch = img[int(h * 0.62):int(h * 0.97), int(w * 0.05):int(w * 0.95)]
                path = folder / f"{tag}_swatch_{r.stem}.png"
                cv2.imwrite(str(path), swatch)
                out.append(path)
            audit.append({"photo": str(r), "distance_km": distance, "used": "whole" if whole else "ground swatch"})
        return out, audit

    def photo_end_stills(self, folder: Path, words: dict):
        """End still from a real photograph of a famous viewpoint (Jamie 18 Oct, "use real photos"). At Battery
        Spencer FLUX 3 drew the postcard view of the bridge on every seed, whatever the map frame showed, and no crop
        or camera turn reconciled them. Here the openly licensed Commons photos of the place are fetched and ranked
        (photo_op), the best one whose far skyline the map can be fitted to is cropped to the frame, FLUX 3 only adds
        the figure to it, and the map's end camera is fitted to the photo: pitch, yaw and zoom from the skylines
        (wide bounds: the photo is ground truth), the lift (up and back along the heading, 0-40 m) by which rendered
        last frame's skyline agrees best, then the last ``fit_camera_ease`` guide frames are re-rendered eased in
        (``<stills>/guide_fit/``, which the swoop follows). The end-still gate is unchanged: the still against the
        fitted guide, no seed retries."""
        import cv2
        from geoblock_core.place_photos import fetch_place_photos
        from geoblock_core.places import find_place, landing
        from . import flux3, terrain_route
        from .end_register import W, H, fit_camera, person_box, skyline
        from .terrain_quality import assess_endpoint_layout
        cfg, shot = self.m["stills"], self.m["shot"]
        seed = cfg["seed"]
        scene = self.out["terrain"].parent
        terrain_dir = scene.parent  # stays the terrain folder when ``scene`` becomes the photo camera line's render
        meta = terrain_route.terrain_meta(scene)
        guide = scene / "guide_frames" / "0081.png"
        place = find_place(shot["place"])
        lat, lon = landing(place)
        refs = self.root.parent / "reference_photos" / shot["place"]
        if not (refs / "manifest.json").is_file():
            fetch_place_photos(place["name"], lat, lon, 12, refs, radius_km=float(cfg.get("photo_radius_km", 0.6)),
                               heading_deg=float(meta["heading"]), purpose="photo_op")
        env = dict(os.environ)
        if env.get("TOUCHDOWN_EMBED_DEPS"):
            env["PYTHONPATH"] = os.pathsep.join([env["TOUCHDOWN_EMBED_DEPS"], env.get("PYTHONPATH", "")])
        env["HF_HUB_OFFLINE"] = "1"
        query = cfg.get("photo_query") or f"A photograph taken {shot['scene']}, the wide view in daylight."
        subprocess.run([env.get("TOUCHDOWN_EMBED_PYTHON", sys.executable), "-m", "touchdown.photo_op",
                        "--manifest", str(refs / "manifest.json"), "--query", query,
                        "--output", str(folder / "photo_ranking.json"), "--no-anchors-ok"], check=True, env=env)
        ranked = json.loads((folder / "photo_ranking.json").read_text())["ranked"]

        def framed(path: str):
            im = cv2.imread(path)
            h, w = im.shape[:2]
            aspect = flux3.W / flux3.H
            if w / h > aspect:
                nw = int(h * aspect)
                return im[:, (w - nw) // 2:(w - nw) // 2 + nw]
            nh = int(w / aspect)
            return im[(h - nh) // 2:(h - nh) // 2 + nh]

        g = cv2.resize(cv2.imread(str(guide)), (W, H), interpolation=cv2.INTER_AREA)
        gl = skyline(g)
        bounds = {"pitch": (-25.0, 10.0), "yaw": (-30.0, 30.0), "zoom": (0.5, 2.0)}
        limit = float(cfg.get("fit_camera_max_residual", 12))
        trials, chosen = [], None
        for photo in ranked:
            small = cv2.resize(framed(photo["local_path"]), (W, H), interpolation=cv2.INTER_AREA)
            line = skyline(small)
            fit = fit_camera(gl, line, **bounds) if np.isfinite(line).sum() >= 0.4 * W else {}
            trials.append({"page_id": photo["page_id"], "title": photo["title"], "similarity": photo["photo_op_similarity"],
                           "skyline_columns": int(np.isfinite(line).sum()), "fit": fit})
            if fit and fit["residual_px"] <= limit:
                chosen = photo
                break
        (folder / "photo_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
        if chosen is None:
            raise ValueError("no Commons photo of the place has a far skyline the map frame can be fitted to")
        cv2.imwrite(str(folder / "photo.png"), framed(chosen["local_path"]))
        refs_in = [folder / "photo.png"] + ([Path(shot["subject"])] if shot.get("subject") else [])
        prompt = cfg.get("photo_end_prompt") or (
            "Keep image 1 exactly as it is: the same real photograph, the same view, framing, bridge, water, hills, "
            "city, sky and light; do not move, redraw or restyle anything in it. Only add {person}, standing still "
            "facing the camera in the centre of the frame, full length at realistic human scale about ten metres from "
            "the camera, on natural ground at the near edge of the view (extend the near ground at the bottom of the "
            "frame just enough for them to stand on). Photorealistic, no text.").format(**self.words)
        if shot.get("subject"):
            prompt += " Use image 2 for the person's exact face, hair, body and clothing."
        # flux3.py speaks the plain ComfyUI API: the server itself (flux3_free_url), not the comfy-api-proxy (8193)
        flux3.still(cfg.get("flux3_free_url", "http://127.0.0.1:8192"), refs_in, prompt, seed,
                    f"geoblock/touchdown_{self.name}_end_photo_f3", folder / "end.png")
        # the map's end camera, fitted to the photo
        e = cv2.resize(cv2.imread(str(folder / "end.png")), (W, H), interpolation=cv2.INTER_AREA)
        box = person_box(e)
        el = skyline(e, (box[0] - (box[2] - box[0]) // 6, box[2] + (box[2] - box[0]) // 6) if box else None)
        # EXIF prior (the photo's GPS position and altitude, compass direction and lens): the lens fixes the zoom,
        # which the skyline alone traded against the camera height (v11: zoom 0.62 at 50 m up; EXIF: 30 mm equiv.,
        # zoom ~1.0, taken at the cliff edge)
        from . import photo_pose
        prior = photo_pose.exif_prior(Path(chosen["local_path"])) or {}
        if prior.get("hfov_deg"):
            zoom0 = math.tan(math.radians(30)) / math.tan(math.radians(prior["hfov_deg"] / 2))
            bounds = {**bounds, "zoom": (zoom0 * 0.95, zoom0 * 1.05)}
        fit = fit_camera(gl, el, **bounds)
        out = folder / "guide_fit"
        out.mkdir(parents=True, exist_ok=True)
        structures_file = terrain_dir / "osm_structures.json"
        if structures_file.is_file():
            # PnP on the OSM towers (photo_pose.solve), anchored on the EXIF GPS position: the near tower's parallax
            # places the camera, which the skyline and the water cannot (v11: the tower swapped through a dissolve)
            from pyproj import Transformer
            dmeta = json.loads((terrain_dir / "dem" / "metadata.json").read_text())
            datum = float(dmeta["landing"])
            near_h = np.load(terrain_dir / "dem" / "height.npy") - datum
            far_h = np.load(terrain_dir / "dem" / "height_far.npy") - datum

            def ground(x, y):
                if abs(x) <= 396 and abs(y) <= 396:
                    return float(near_h[round(200 - y / 2), round(200 + x / 2)])
                return float(far_h[int(np.clip(round(340 - y / 50), 0, 680)), int(np.clip(round(340 + x / 50), 0, 680))])

            prior_xy = None
            if prior.get("latitude") is not None:
                from geoblock_core.utm import projected_crs
                east, north = Transformer.from_crs("EPSG:4326", projected_crs(self.m['terrain'].get('utm_zone') or dmeta['utm_zone']),
                                                   always_xy=True).transform(prior["longitude"], prior["latitude"])
                prior_xy = (east - dmeta["x"], north - dmeta["y"])
            landing_xy = (float(meta["target_x"]), float(meta["target_y"]))
            pose = photo_pose.solve(cv2.resize(cv2.imread(str(folder / "photo.png")), (W, H), interpolation=cv2.INTER_AREA),
                                    json.loads(structures_file.read_text()), ground, meta["guide_poses"][-1],
                                    float(meta["heading"]), fit, prior_xy, landing_xy)
            pose["exif_prior"] = {**prior, "local_xy_m": [round(v, 1) for v in prior_xy] if prior_xy else None}
            (folder / "photo_pose.json").write_text(json.dumps(pose, indent=2, default=float) + "\n")
            print(f"photo pose: {json.dumps({k: v for k, v in pose.items() if k != 'photo_towers'}, default=float)}", flush=True)
            # sanity bounds: on or above the DEM, within 300 m of the landing, every matched tower under 10 px
            bad = []
            if not pose.get("applied"):
                bad.append(pose.get("reason", "not solved"))
            else:
                if pose["height_above_ground_m"] < 0:
                    bad.append("camera below the DEM surface")
                if pose["distance_from_landing_m"] > 300:
                    bad.append(f"camera {pose['distance_from_landing_m']} m from the landing")
                errs = [t["error_px"] for t in pose["towers"] if t.get("error_px") is not None]
                if not errs:
                    bad.append("no OSM tower matched in the photo")
                elif max(errs) >= 10:
                    bad.append(f"tower reprojection error {max(errs)} px")
            if bad:
                raise ValueError("photo end camera fit failed: " + "; ".join(bad))
            ease = int(cfg.get("fit_camera_ease", 24))
            fit = {**fit, **{k: pose[k] for k in ("yaw_deg", "pitch_deg", "zoom", "lift_m", "back_m", "side_m")},
                   "solved_by": "OSM tower PnP", "tower_errors_px": errs}
            # the figure: its feet cast onto the DEM with the solved camera; a 1.7 m proxy stands there in every guide
            # frame, so the swoop grows her with the geometry (v12: 16 px until f69, then 28x in 11 frames)
            gp = meta["guide_poses"][-1]
            rot = photo_pose.rotation(photo_pose.base_rotation(np.array(gp["camera_xyz_m"]), np.array(gp["aim_xyz_m"])),
                                      pose["yaw_deg"], pose["pitch_deg"])
            centre = np.array([pose["x"], pose["y"], pose["z"]])
            small = lambda p_: cv2.resize(cv2.imread(str(p_)), (W, H), interpolation=cv2.INTER_AREA)
            figure = photo_pose.figure_ground(small(folder / "end.png"), small(folder / "photo.png"), centre, rot,
                                              pose["zoom"], ground)
            if figure.get("found") and figure.get("scale", 1.0) and abs(figure["scale"] - 1) > 0.15:
                shutil.copy(folder / "end.png", folder / "end_unscaled.png")
                photo_pose.rescale_figure(folder / "end_unscaled.png", folder / "photo.png", figure["box"],
                                          figure["scale"], folder / "end.png")
                figure["rescaled_end_still"] = True
            (folder / "figure.json").write_text(json.dumps(figure, indent=2, default=float) + "\n")
            print(f"figure: {json.dumps(figure, default=float)}", flush=True)
            if not (figure.get("found") and figure.get("ground_point")):
                raise ValueError("photo end figure could not be placed: " + json.dumps(figure, default=float))
            gx, gy = figure["ground_point"]
            # "--flag=value": a negative coordinate as a separate argument reads as a flag to argparse
            person = [f"--person-at={gx},{gy}", "--person-height=1.7", f"--person-z={figure['base_z']}",
                      f"--aim-z={figure['base_z'] + 1.1}"]  # aim at her body (v14 first try: at the ground below her ledge)
            if figure.get("coat_rgb"):
                person += ["--person-rgb=" + ",".join(str(c) for c in figure["coat_rgb"])]
            fit = {**fit, "render_from": 1, "figure": figure}
            # fly the descent straight down the photo's camera line (v13: bending the old approach into the photo's
            # view in its last frames, 16.5 deg and ~30 m sideways, warped the whole scene f138-144): the dive and the
            # guide are re-rendered with the target at the figure's ground point and the photo camera's heading; the
            # small rest (forward to the photo camera, its height, the last degrees) eases in over the last frames
            forward = -rot[:, 2]
            heading_new = math.degrees(math.atan2(forward[0], forward[1])) % 360
            line = folder / "guide_fit" / "scene"
            line.mkdir(parents=True, exist_ok=True)
            shutil.copy(scene / "blender_extra_args.json", line / "blender_extra_args.json") if (scene / "blender_extra_args.json").is_file() else None
            # the base path's final camera (10 m short of the target, on the heading) is the photo camera itself, and
            # it aims along the photo's own elevation, so almost nothing is left to ease in (v14 tries: aiming at the
            # figure and moving the last 6.7 m forward without re-aiming pushed her out of the bottom of the frame)
            hn = math.radians(heading_new)
            tx, ty = float(centre[0] + 10 * math.sin(hn)), float(centre[1] + 10 * math.cos(hn))
            aim_z = float(centre[2] + 10 * forward[2] / math.hypot(forward[0], forward[1]))
            person = [a_ for a_ in person if not a_.startswith("--aim-z=")] + [f"--aim-z={aim_z:.3f}"]
            relined = {"heading": round(heading_new, 2), "target_x": round(tx, 2), "target_y": round(ty, 2)}
            self.blender_end_fit(terrain_dir, scene, relined, line, None, person=person)
            meta = terrain_route.terrain_meta(line)
            if meta["boundary_mae"] > 0.001:
                raise ValueError("photo camera line: dive/guide boundary is discontinuous")
            g0 = meta["guide_poses"][-1]
            p0n, r0n = np.array(g0["camera_xyz_m"]), photo_pose.base_rotation(np.array(g0["camera_xyz_m"]), np.array(g0["aim_xyz_m"]))
            f0 = -r0n[:, 2]
            ang = lambda v: math.atan2(v[1], v[0])
            elev = lambda v: math.atan2(v[2], math.hypot(v[0], v[1]))
            yaw_r = (math.degrees(ang(forward) - ang(f0)) + 180) % 360 - 180
            pitch_r = math.degrees(elev(forward) - elev(f0))
            hn = math.radians(heading_new)
            dvec = centre - p0n
            rest = {"yaw_deg": round(yaw_r, 3), "pitch_deg": round(pitch_r, 3), "zoom": pose["zoom"],
                    "lift_m": round(float(dvec[2]), 3), "back_m": round(float(-(dvec[0] * math.sin(hn) + dvec[1] * math.cos(hn))), 3),
                    "side_m": round(float(dvec[0] * math.cos(hn) - dvec[1] * math.sin(hn)), 3)}
            ease = int(cfg.get("fit_camera_ease", 40))
            x57 = max(0.0, (57 - (81 - ease)) / ease)
            late = 1 - x57 * x57 * (3 - 2 * x57)  # share of the eased rest still to come over the last 24 frames
            check = {"yaw_change_last24_deg": round(abs(yaw_r) * late, 2), "lateral_last24_m": round(abs(rest["side_m"]) * late, 2)}
            print(f"photo camera line: heading {heading_new:.1f}, target {tx:.1f},{ty:.1f} (figure {gx:.1f},{gy:.1f}); rest {json.dumps(rest)}; {json.dumps(check)}", flush=True)
            if check["yaw_change_last24_deg"] >= 3 or check["lateral_last24_m"] >= 5:
                raise ValueError(f"photo camera line still swings late: {json.dumps(check)}")
            scene = line
            fit = {**fit, **rest, "render_from": 1, "camera_line": {**relined, **check}}
        # (no OSM structures) the lift: the skyline barely tells lifts apart (Battery Spencer: 7, 7, 9 px at 0/20/40 m) but the ground
        # does: from the plateau a sandy mound fills the lower frame where the photo, taken at the edge, shows the
        # strait. Choose by the overlap of water below the skyline (blue over red), the smallest lift within 0.02 of
        # the best (0.08 at 0 m, 0.60 at 20 m, 0.63 at 30 m, flat above)
        def water_below(im):
            f = im.astype(int)
            line = skyline(im)
            rows = np.arange(im.shape[0])[:, None]
            return (f[..., 0] > f[..., 2] + 12) & (rows > np.where(np.isfinite(line), line, im.shape[0])[None, :] + 4)

        photo_small = cv2.resize(cv2.imread(str(folder / "photo.png")), (W, H), interpolation=cv2.INTER_AREA)
        photo_water = water_below(photo_small)
        # OSM structures with a colour tag (the Golden Gate's #f04a00) are a second, near-field cue: their pixels in
        # the render and in the photo should coincide, which places the camera sideways (the near tower's parallax:
        # the skyline and the water cannot see it)
        hues = []
        structures_file = terrain_dir / "osm_structures.json"
        if structures_file.is_file():
            for item in json.loads(structures_file.read_text())["structures"]:
                if item.get("bridge") or item["kind"] == "deck":
                    r_, g_, b_ = item["rgb"]
                    hsv = cv2.cvtColor(np.uint8([[[b_ * 255, g_ * 255, r_ * 255]]]), cv2.COLOR_BGR2HSV)[0, 0]
                    if hsv[1] > 110:
                        hues.append(int(hsv[0]))
        hues = sorted(set(hues))

        def coloured(im):
            if not hues:
                return None
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(int)
            near_hue = np.zeros(im.shape[:2], bool)
            for h_ in hues:
                near_hue |= np.minimum(abs(hsv[..., 0] - h_), 180 - abs(hsv[..., 0] - h_)) <= 8
            return near_hue & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)

        photo_colour = coloured(photo_small)

        def iou(a_, b_):
            return float((a_ & b_).sum() / max((a_ | b_).sum(), 1))

        def trial_pose(lift, side):
            trial = out / f"pose_l{int(lift)}_s{int(side)}"
            trial.mkdir(parents=True, exist_ok=True)
            (trial / "end_fit.json").write_text(json.dumps({**fit, "lift_m": lift, "back_m": lift, "side_m": side,
                                                            "ease_frames": 1}) + "\n")
            self.blender_end_fit(terrain_dir, scene, meta, trial / "render", trial / "end_fit.json")
            rendered = cv2.resize(cv2.imread(str(trial / "render" / "guide_frames" / "0081.png")), (W, H))
            rl = skyline(rendered)
            ok = np.isfinite(rl) & np.isfinite(el)
            row = {"lift_m": lift, "side_m": side,
                   "skyline_residual_px": round(float(np.median(np.abs(rl[ok] - el[ok]))), 2) if ok.sum() > 0.25 * W else None,
                   "water_iou": round(iou(water_below(rendered), photo_water), 3)}
            if photo_colour is not None:
                row["structure_iou"] = round(iou(coloured(rendered), photo_colour), 3)
            row["score"] = row["water_iou"] + row.get("structure_iou", 0.0)
            return row

        lifts = [] if structures_file.is_file() else [trial_pose(lift, 0.0) for lift in (0.0, 10.0, 20.0, 30.0, 40.0, 60.0)]
        if lifts and photo_colour is not None and photo_colour.mean() > 0.01:
            lift0 = max(lifts, key=lambda r: r["score"])["lift_m"]
            for side in (-100.0, -50.0, 50.0, 100.0):
                lifts.append(trial_pose(lift0, side))
            side0 = max(lifts, key=lambda r: r["score"])["side_m"]
            for side in (side0 - 25.0, side0 + 25.0):
                for lift in (max(lift0 - 10.0, 0.0), lift0, lift0 + 10.0):
                    if not any(r["side_m"] == side and r["lift_m"] == lift for r in lifts):
                        lifts.append(trial_pose(lift, side))
        if lifts:
            top = max(r["score"] for r in lifts)
            best = min((r for r in lifts if r["score"] >= top - 0.02), key=lambda r: (r["lift_m"], abs(r["side_m"])))
            ease = int(cfg.get("fit_camera_ease", 24))
            fit = {**fit, "lift_m": best["lift_m"], "back_m": best["lift_m"], "side_m": best["side_m"], "pose_trials": lifts}
        (out / "end_fit.json").write_text(json.dumps({**fit, "ease_frames": ease}, indent=2) + "\n")
        self.blender_end_fit(self.out["terrain"].parent.parent, scene, meta, out / "render", out / "end_fit.json",
                             person=locals().get("person"))
        frames = out / "guide_frames"
        frames.mkdir(exist_ok=True)
        for f in range(1, 82):
            rendered = out / "render" / "guide_frames" / f"{f:04d}.png"
            shutil.copy(rendered if rendered.is_file() else scene / "guide_frames" / f"{f:04d}.png", frames / f"{f:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames / "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "guide.mp4")], check=True)
        from .terrain_quality import assess_photo_geometry
        layout = assess_endpoint_layout(frames / "0081.png", folder / "end.png")
        # photo mode: the SIFT landmark test is replaced by the geometric one (skyline + water layout; calibrated 18 Oct:
        # rejects all 35 known-bad stills, skyline 19-161 px / water overlap <= 0.15, passes the known-good ones);
        # the rest of the gate is as usual
        geometry = assess_photo_geometry(frames / "0081.png", folder / "end.png")
        photo_facts = {k: chosen.get(k) for k in ("page_id", "title", "commons_page", "license", "license_url",
                                                   "artist", "distance_km", "photo_op_similarity")}
        sift = {k: layout[k] for k in ("landmark_inliers", "landmark_median_shift_px", "landmark_guide_features")
                if k in layout}
        layout = {**layout, "warnings": [w for w in layout["warnings"]
                                         if "landmark" not in w and not w.startswith("end still unverifiable")]
                  + geometry["warnings"], "sift_not_used": sift, "photo_geometry": geometry,
                  "photo": photo_facts, "camera_fit": fit}
        (folder / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
        print(f"end from photo: {json.dumps(photo_facts)}; camera fit {json.dumps(fit)}; gate "
              f"{json.dumps(geometry)} (SIFT, not used: {json.dumps(sift)})", flush=True)
        gate = cfg.get("layout_gate", "stop")
        stopping = [w for w in layout["warnings"] if gate == "stop" or (gate == "landmarks" and "landmark" in w)]
        if stopping and gate != "warn":
            raise ValueError("end still from photo failed the gate: " + "; ".join(stopping))
        rest = self.comfy("stills", folder=str(folder), seed=seed, name=self.name, words=words,
                          flux3=list(cfg.get("flux3", [])),
                          **self.stills_route(),
                          end_still=str(folder / "end.png"), end_reference=None, subject=shot.get("subject"),
                          scene_reference=None, end_prompt=None,
                          opener_end_prompt=cfg.get("opener_end_prompt", "").format(**self.words) or None)
        for name, path in rest.items():
            if name != "end":
                shutil.copy(path, folder / f"{name}.png")
        return folder / "end.png", {"seed": seed, "end_from_photo": photo_facts, "camera_fit": fit, "layout": layout}

    def commons_stills(self, folder: Path, words: dict):
        """Photo-grounded stills (Jamie 17 Oct). The end still and an optional aerial middle still keep the measured
        scene's layout (image 1: the guide render, or its structure map) including every building, road and paved
        area, and take only their appearance from Commons photographs of the place (``commons_refs``,
        ``mid_refs``). ``stills_model`` = "flux3" renders them with FLUX 3 Image Klein on ``flux3_url`` (the
        side-by-side ComfyUI v0.41 install); "klein2" uses FLUX.2 klein through the usual proxy. The scene type
        (landmark / urban / nature, from EmbeddingGemma) picks what the prompt tells the model to keep. Each still
        is checked against its guide frame: the landmark test stops the build when it can measure a shift, and a
        still that pasted the structure diagram in stops it too. The orbit, cloud and opener stills are as usual."""
        from .terrain_quality import assess_endpoint_layout
        cfg, shot = self.m["stills"], self.m["shot"]
        seed = cfg["seed"]
        frames = self.out["terrain"].parent / "guide_frames"
        kind = self.scene_type(folder)
        keep = self.KEEP.get(kind["type"], self.KEEP["nature"])
        model = cfg.get("stills_model", "flux3")

        def render(name: str, refs: list, prompt: str, s: int) -> Path:
            out = folder / f"{name}.png"
            if model == "flux3":
                from . import flux3
                # the FLUX 3 module talks to the plain ComfyUI API, not the SDK proxy
                url = self.stills_free_url or cfg.get("flux3_free_url") or cfg.get("flux3_url", "http://127.0.0.1:8192")
                return flux3.still(url, refs, prompt, s, f"geoblock/touchdown_{self.name}_{name}_f3", out)
            shutil.copy(self.comfy("klein_refs", folder=str(folder), refs=[str(r) for r in refs], prompt=prompt,
                                   seed=s, prefix=f"geoblock/touchdown_{self.name}_{name}_k2"), out)
            return out

        def image1(name: str, guide: Path, conditioning: str) -> list[Path]:
            if conditioning == "structure":
                return [self.structure_map(guide, folder / f"{name}_structure.png")]
            if conditioning == "both":
                return [guide, self.structure_map(guide, folder / f"{name}_structure.png")]
            return [guide]

        def gate(name: str, guide: Path, still: Path) -> dict:
            layout = assess_endpoint_layout(guide, still)
            layout["diagram_share"] = round(self.looks_like_diagram(still), 3)
            (folder / f"{name}_layout.json").write_text(json.dumps(layout, indent=2) + "\n")
            if layout["diagram_share"] > 0.15:
                raise ValueError(f"{name} still pasted the structure map in ({layout['diagram_share']:.0%} flat grey)")
            if any("landmarks moved" in w for w in layout["warnings"]) and cfg.get("layout_gate", "landmarks") != "warn":
                raise ValueError(f"{name} still moved the measured landmarks: " + "; ".join(layout["warnings"]))
            return layout

        cond = cfg.get("end_conditioning", "structure" if model == "flux3" else "render")
        end_photos, end_audit = self.prepared_refs(folder, cfg["commons_refs"], "end")
        refs = [*image1("end", frames / "0081.png", cond), *end_photos]
        if shot.get("subject"):
            refs.append(Path(shot["subject"]))
        prompt = (cfg.get("end_prompt") or self.COMMONS_END).format(role=self.STILL_ROLE[cond], keep=keep, **self.words)
        if shot.get("subject"):
            prompt += f" Image {len(refs)} shows the person: keep their exact face, hair, body and clothing."
        render("end", refs, prompt, seed)
        facts = {"seed": seed, "end_given": False, "scene_type": kind, "stills_model": model, "end_conditioning": cond,
                 "commons_refs": end_audit, "end_layout": gate("end", frames / "0081.png", folder / "end.png")}
        if cfg.get("mid_height_m"):
            k = handover_frame(float(cfg["mid_height_m"]))
            mcond = cfg.get("mid_conditioning", "render")
            mid_photos, mid_audit = self.prepared_refs(folder, cfg.get("mid_refs", cfg["commons_refs"]), "mid")
            mid_refs = [*image1("mid", frames / f"{k:04d}.png", mcond), *mid_photos]
            mid_prompt = (cfg.get("mid_prompt") or self.COMMONS_MID).format(
                height=round(guide_height(k)), role=self.STILL_ROLE[mcond], keep=keep, **self.words)
            render("mid", mid_refs, mid_prompt, seed + 7)
            facts.update(mid_guide_frame=k, mid_height_m=round(guide_height(k), 1), mid_conditioning=mcond,
                         mid_refs=mid_audit,
                         mid_layout=gate("mid", frames / f"{k:04d}.png", folder / "mid.png"))
        out = self.comfy("stills", folder=str(folder), seed=seed, name=self.name, words=words,
                         end_still=str(folder / "end.png"),
                         opener_end_prompt=cfg.get("opener_end_prompt", "").format(**self.words) or None)
        for name, path in out.items():
            if name != "end":
                shutil.copy(path, folder / f"{name}.png")
        return folder / "end.png", facts

    def opener(self, folder: Path):
        import fakeout_earth_zoom as f
        cfg, st = self.m["opener"], self.out["stills"].parent
        frames = cfg["frames"]
        # cloud_at = 0 drops the cloud pin: pinned 32 frames after a FLUX 3 orbit still that looks nothing like it,
        # the cloud still made LTX dissolve globe -> cloud inside the kept frames instead of flying (Jamie, 18 Oct)
        pins = [(0, st / "orbit.png")]
        if cfg.get("cloud_at", 0.15):
            pins.append((round(cfg["cloud_at"] * (frames - 1) / 8) * 8, st / "cloud.png"))
        pins.append((-1, st / ("opener_end.png" if (st / "opener_end.png").is_file() else "end.png")))
        prompt = cfg.get("prompt") or f.ZOOM_IN.format(**self.words)
        clip, facts = self.candidates("opener", folder, lambda s, d: self.ltx(d, frames, pins, prompt, s, "opener"))
        out = folder / "opener.mp4"
        shutil.copy(clip, out)
        return out, facts

    def cloud(self, folder: Path):
        cfg = self.m["cloud"]
        keep = self.m["opener"]["keep"]
        if cfg.get("start_at") == "auto":  # cut the opener before it fogs over, so the cloud is entered only once
            keep = _before_fog(self.out["opener"], keep, cfg.get("fog_contrast", 0.45))
        (folder / "keep.txt").write_text(str(keep))
        start = V.save_png(V.frame(self.out["opener"], keep), folder / "start.png")
        if cfg.get("engine", "ltx") == "procedural":  # Jamie 17 Oct: a model-free cloud deck, so no seed lottery
            from .proc_cloud import render as proc_cloud
            out = folder / "cloud.mp4"
            facts = proc_cloud(V.frame(self.out["opener"], keep - 1), out, frames=cfg.get("procedural_frames", 32),
                               seed=cfg.get("procedural_seed", 0), pitch_deg=cfg.get("pitch_deg", 45.0),
                               start_pitch_deg=cfg.get("start_pitch_deg", 14.0))
            return out, {**facts, "opener_keep": keep, "kept": V.count(out)}
        prompt = cfg["prompt"]
        # grounding (Jamie 17 Oct): ``prefix`` pins the opener's last n frames as a clip, so LTX carries the
        # opener's motion and scale into the cloud; ``end_pin`` pins the whiteout still at the end
        pins = [(0, start)]
        n = cfg.get("prefix", 1)
        if n > 1:
            if (n - 1) % 8 or n > keep + 1:
                raise ValueError("[cloud] prefix must be 8n+1 and fit inside the opener's kept frames")
            start = folder / f"prefix_{n}.mp4"
            with V.Writer(start) as w:
                for i, im in enumerate(V.frames(self.out["opener"])):
                    if keep - n < i <= keep:
                        w.write(im)
            pins = [(0, start)]
        if cfg.get("end_pin"):
            pins.append((-1, self.out["stills"].parent / "cloud.png"))
        clip, facts = self.candidates("cloud", folder, lambda s, d: self.ltx(d, cfg["frames"], pins, prompt, s, "cloud"))
        # retime: integer steps so the plunge keeps the opener's pace (LTX starts almost still), <= 2 once it whitens
        sm = V.small(clip, (480, 272))
        tiny = sm[:, ::4, ::4]
        # Opaque cloud can carry a warm low-frequency gradient (Grand Canyon: std 16), while still hiding
        # all scene detail. Require both low overall contrast and almost no dark pixels.
        white = [i for i in range(len(sm))
                 if tiny[i].mean() > cfg.get("white_mean", 215)
                 and tiny[i].std() < cfg.get("white_std", 25)
                 and (tiny[i] < 160).mean() < cfg.get("white_dark_share", 0.02)]
        if self.unattended and not white:
            raise ValueError("cloud candidate did not reach a full whiteout; one-pass build stopped")
        end = white[0] if white else len(sm) - 1
        v = V.flow(sm, 0, end, win=21)
        bright = tiny.mean(axis=(1, 2))
        low = int(np.argmin(bright[:end]))
        thr = bright[:end].min() + 0.08 * (bright[end] - bright[:end].min())
        rise = next((i for i in range(low + 1, end) if bright[i] > thr), end)
        # ramp: the step limit eases up from 1 over the first kept frames, so the plunge does not leap at the
        # join (Yosemite: a full cap-5 step straight after the opener read as a one-frame cloud-bank pop);
        # max_kept: raise the speed until the passage fits, to trim the cloud section
        q0 = n - 1  # with a prefix, the clip's first n frames re-render the opener's last n: start after them

        def retime(speed):
            idx, q = [q0], q0
            while True:
                cap = cfg["cap"] if q < rise else 2
                cap = min(cap, 1 + len(idx) - 1) if len(idx) - 1 < cfg.get("ramp", 0) else cap
                step = int(np.clip(round(speed / max(float(v[min(q, end - 1)]), 0.3)), 1, cap))
                if q + step > end:
                    if idx[-1] != end:
                        idx.append(end)  # the terrain handoff needs the measured opaque frame
                    return idx
                q += step
                idx.append(q)
        speed = cfg["speed"]
        idx = retime(speed)
        while cfg.get("max_kept") and len(idx) - 1 > cfg["max_kept"] and speed < 100:
            speed *= 1.1
            idx = retime(speed)
        # colour (build_cloud.py): the white exit is cooled toward the dive's first frame (a/b only, over the
        # last 16 kept frames; a full white balance turned the interior orange), then the whole passage is pulled
        # to the opener's last kept frame at the join (Lab mean, measured on the clip's frame 0, over 12 frames)
        joint = V.frame(self.out["opener"], keep - 1)
        dive0 = V.frame(self.out["dive"], 0)
        frame0 = V.frame(clip, q0)
        ab = V.lab_mean(dive0)[1:] - V.lab_mean(V.frame(clip, idx[-1]))[1:]
        off = V.lab_mean(joint) - V.lab_mean(frame0)
        pos = {i: k for k, i in enumerate(idx)}
        # exit_fade: the last kept frames fade into the opaque exit frame. The LTX cloud can jump to opaque in one
        # frame (Monument Valley 17 Oct: a dark band of sky vanished at the exit, change 37); flying into white
        # cloud should whiten progressively. Default on for the terrain route, whose hand-off needs that frame.
        fade = int(cfg.get("exit_fade", 8 if self.terrain_mode else 0))  # 4 left one 37-change step (17 Oct)
        last = len(idx) - 1
        white = V.lab(V.frame(clip, idx[-1]))
        white[..., 1:] += ab
        out = folder / "cloud.mp4"
        with V.Writer(out) as w:
            for i, im in enumerate(V.frames(clip)):
                k = pos.get(i)
                if not k:  # not kept, or frame 0 (it re-renders the opener's frame ``keep``; the join drops it)
                    continue
                x = V.lab(im)
                wab = V.smoothstep((k - (len(idx) - 16)) / 15)
                if wab > 0:
                    x[..., 1:] += wab * ab
                if k < 12:
                    x += (1 - V.smoothstep(k / 12)) * off
                if fade and last - fade < k < last:
                    t = V.smoothstep((k - (last - fade)) / fade)
                    x = x * (1 - t) + white * t
                w.write(V.rgb(x))
        return out, {**facts, "white_at": end, "kept": len(idx) - 1, "steps": np.diff(idx).tolist(), "opener_keep": keep,
                     "speed": round(speed, 2)}

    def dive(self, folder: Path):
        if self.terrain_mode:
            source = self.out["terrain"].parent / "dive.mp4"
            relined = self.out["stills"].parent / "guide_fit" / "scene" / "dive.mp4"
            if relined.is_file():  # the photo end (set, or switched on automatically) re-aimed the descent
                source = relined
            out = folder / "dive.mp4"
            shutil.copy(source, out)
            V.save_png(V.frame(out, -1), folder / "dive_last.png")
            return out, {"source": str(source), "frames": V.count(out)}
        from geoblock_core.places import find_place, landing, view_heading
        cfg = self.m["dive"]
        place = find_place(self.m["shot"]["place"])
        lat, lon = landing(place)
        spin = cfg.get("spin")
        if spin is None:
            spin = max(view_heading(place), 0.0)
        from touchdown import aerial
        source = cfg.get("imagery", "auto")  # a touchdown.aerial source id; auto = by the place's country
        if source == "auto":
            source = aerial.pick(place.get("country", ""), lat, lon)
        cache = self.root.parent / "imagery" / self.m["shot"]["place"]
        if source != "naip":  # NAIP keeps its old cache folder
            cache = cache / source
        out = folder / "dive.mp4"
        paths = divemod.fetch(lat, lon, cache, divemod.FAR_LEVELS if cfg.get("mode") == "oblique" else divemod.LEVELS,
                              source=source)
        sea = {}
        if cfg.get("sea", True):  # repaint sun glint and capture-date seams on open water (touchdown.sea)
            from touchdown import sea as seamod
            masks = {w: (cache / f"naip_{w:05d}_sea.png", cache / f"naip_{w:05d}_keep.png") for w in paths}
            if not all(seamod.current(m) for pair in masks.values() for m in pair):
                self.comfy("sea_masks", lat=lat, lon=lon, folder=str(cache), levels=list(paths), pixels=divemod.PIXELS,
                           cache_dir=str(self.root.parent / "imagery"))
            paths, sea = seamod.apply(paths, masks, divemod.PIXELS)
        if cfg.get("mode") == "oblique":
            facts = divemod.render_oblique(paths, out, from_height_m=cfg["from_height_m"], to_height_m=cfg["to_height_m"],
                                           frames=cfg["frames"], start_pitch_deg=cfg["start_pitch_deg"],
                                           pitch_deg=cfg["pitch_deg"], turn_frames=cfg["turn_frames"], heading=spin,
                                           clear=cfg["clear"], aim=cfg.get("aim", 15.0))
            return out, {**facts, **sea, "spin": spin, "lat": lat, "lon": lon, "imagery": source}
        facts = divemod.render(paths, out, from_m=cfg["from_m"], to_m=cfg["to_m"], nadir_frames=cfg["nadir_frames"],
                               pitch_frames=cfg["pitch_frames"], pitch_deg=cfg["pitch_deg"], spin=spin,
                               clear=cfg["clear"], aim=cfg.get("aim", 15.0))
        return out, {**facts, **sea, "spin": spin, "lat": lat, "lon": lon, "imagery": source}

    def swoop(self, folder: Path):
        cfg = self.m["swoop"]
        if self.terrain_mode and cfg.get("mode", "guide") == "first_last":
            return self.terrain_first_last(folder)
        if self.terrain_mode and cfg.get("mode", "guide") == "keyframes":
            return self.terrain_keyframes(folder)
        if self.terrain_mode:
            if cfg.get("frames", 81) != 81:
                raise ValueError("measured-terrain guide and LTX approach must both have 81 frames")
            scene = self.out["terrain"].parent
            end = self.out["stills"].parent / "end.png"
            fitted = self.out["stills"].parent / "guide_fit" / "guide.mp4"  # the camera moved to the end still
            guide_video = fitted if fitted.is_file() else scene / "guide.mp4"
            first_frame = self.out["stills"].parent / "guide_fit" / "guide_frames" / "0001.png"
            first_frame = first_frame if first_frame.is_file() else scene / "guide_frames" / "0001.png"
            prompt = cfg["prompt"].format(**self.words)
            mids = self.swoop_mid_pin(folder, cfg, scene)
            from .terrain_quality import assess

            def passing(made: dict) -> int | None:
                """The first candidate that passes the guided-approach checks: a seed that dissolves or holds
                still is not a failed build, it is the next seed (Horseshoe Bend, 21 Oct)."""
                for seed, out in made.items():
                    if not assess(out, end)["warnings"]:
                        return seed
                return None

            clip, facts = self.candidates("swoop", folder, lambda s, d: Path(self.comfy(
                "terrain_v2v", guide=str(guide_video), first=str(first_frame),
                ltx_upscale=bool(cfg.get("ltx_upscale", self.full_res())),  # MV v11: at 960x544 face and land were soft
                last=str(end), output=str(d), seed=s, frames=81, prompt=prompt, url=self.url,
                mids=mids)), choose=passing)
            quality = assess(clip, end)
            if (self.out["stills"].parent / "figure.json").is_file():
                # photo end: the swoop's last frame must show the figure where the end still has it (10% size, 15 px)
                import cv2
                from .end_register import W as RW, H as RH, person_box
                last = cv2.resize(cv2.cvtColor(V.frame(clip, V.count(clip) - 1), cv2.COLOR_RGB2BGR), (RW, RH))
                still = cv2.resize(cv2.imread(str(end)), (RW, RH))
                a_, b_ = person_box(last), person_box(still)
                fig = {"swoop_last_box": a_ and list(map(int, a_)), "end_box": b_ and list(map(int, b_))}
                if a_ and b_:
                    fig["size_ratio"] = round((a_[3] - a_[1]) / max(b_[3] - b_[1], 1), 3)
                    fig["centre_px"] = round(math.hypot((a_[0] + a_[2] - b_[0] - b_[2]) / 2, (a_[1] + a_[3] - b_[1] - b_[3]) / 2), 1)
                quality["figure"] = fig
                if not (a_ and b_) or abs(fig["size_ratio"] - 1) > 0.10 or fig["centre_px"] > 15:
                    quality["warnings"].append(f"figure in the swoop's last frame does not match the end still: {json.dumps(fig)}")
            (folder / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
            if quality["warnings"]:
                raise ValueError("guided approach failed objective checks: " + "; ".join(quality["warnings"]))
            masks = scene / 'guide_frames_surface_masks'
            if masks.is_dir():
                from .terrain_quality import assess_surface_sequence
                temporal = assess_surface_sequence(clip, masks)
                (folder / 'surface_sequence.json').write_text(json.dumps(temporal, indent=2) + '\n')
                facts['surface_sequence_flagged_frames'] = temporal['flagged_frames']
            out = folder / "swoop.mp4"
            shutil.copy(clip, out)
            return out, {**facts, "guide_boundary_mae": json.loads((scene / "terrain_guide.json").read_text())["boundary_mae"],
                         "end_correlation": quality["end_correlation"], "end_mae": quality["end_mae"]}
        first = self.out["dive"].with_name(self.out["dive"].stem + "_last.png")
        end = self.out["stills"].parent / "end.png"
        prompt = cfg["prompt"].format(**self.words)
        n = cfg.get("prefix", 1)  # >1: pin frame 0 to the dive's last n frames, so LTX sees the incoming speed (p9 A/B)
        if n > 1:
            if (n - 1) % 8:
                raise SystemExit("swoop.prefix must be 8n+1")
            first = folder / f"prefix_{n}.mp4"
            with V.Writer(first) as w:
                w.extend(list(V.frames(self.out["dive"]))[-n:])
        clip, facts = self.candidates("swoop", folder, lambda s, d: self.ltx(d, cfg["frames"], [(0, first), (-1, end)], prompt, s, "swoop"))
        out = folder / "swoop.mp4"
        shutil.copy(clip, out)
        return out, facts

    def terrain_first_last(self, folder: Path):
        """Hand over to a free LTX first/last clip at ``handover_m`` instead of following the measured guide to the
        ground (Jamie 17 Oct, option 1). The measured guide plays model-free down to the hand-over height; LTX gets
        its last ``prefix`` frames as a pinned clip and flies on to the end still, which FLUX painted from the
        guide's own final view, so both ends agree on the distant landmark. Below the hand-over the near ground is
        the model's, not the aerial DEM's."""
        cfg = self.m["swoop"]
        scene = self.out["terrain"].parent
        k = handover_frame(cfg.get("handover_m", 175.0))
        n = cfg.get("prefix", 9)
        if (n - 1) % 8 or n < 9 or k < n + 1:
            raise ValueError("[swoop] prefix must be 8n+1 (>= 9) and fit above the hand-over")
        frames = cfg.get("frames", 73)
        if (frames - 1) % 8:
            raise ValueError("[swoop] frames must be 8n+1")
        prefix = folder / f"prefix_{n}.mp4"
        from PIL import Image
        with V.Writer(prefix) as w:
            for f in range(k - n + 1, k + 1):
                w.write(np.asarray(Image.open(scene / "guide_frames" / f"{f:04d}.png").convert("RGB").resize((V.W, V.H))))
        end = self.out["stills"].parent / "end.png"
        prompt = cfg["prompt"].format(**self.words)
        clip, facts = self.candidates("swoop", folder, lambda s, d: self.ltx(d, frames, [(0, prefix), (-1, end)], prompt, s, "swoop"))
        from .terrain_quality import assess
        quality = assess(clip, end, expected_frames=frames, tail_ok=True)
        (folder / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
        if quality["warnings"]:
            raise ValueError("first/last approach failed objective checks: " + "; ".join(quality["warnings"]))
        out = folder / "swoop.mp4"
        kept = frames
        if quality["settled_at"] is not None:
            # LTX arrived early and held the end still (run 2, 17 Oct: still from f44 of 73). Cut the dead hold,
            # keeping nine settled frames for the living end's video prefix.
            kept = min(frames, quality["settled_at"] + 9)
            with V.Writer(out) as w:
                for i, im in enumerate(V.frames(clip)):
                    if i >= kept:
                        break
                    w.write(im)
        else:
            shutil.copy(clip, out)
        (folder / "handover.json").write_text(json.dumps({"guide_frame": k, "prefix": n,
                                                          "height_m": round(guide_height(k), 1)}))
        return out, {**facts, "mode": "first_last", "kept_frames": kept, "handover_frame": k, "handover_height_m": round(guide_height(k), 1),
                     "end_correlation": quality["end_correlation"], "end_mae": quality["end_mae"]}

    def terrain_keyframes(self, folder: Path):
        """Hand over at the top of the measured descent and fly two shorter LTX hops (Jamie 17 Oct): the dive's
        last ``prefix`` frames -> the photo-grounded aerial middle still (``[stills] mid_height_m``) -> the
        photo-grounded end still. The measured guide is not played at all; below the hand-over every frame is the
        model's, steered by stills taken from real photographs. Each hop must pass the dissolve/hold check."""
        cfg = self.m["swoop"]
        n = cfg.get("prefix", 9)
        fa, fb = cfg.get("frames_a", 97), cfg.get("frames_b", 97)
        if (n - 1) % 8 or n < 9 or (fa - 1) % 8 or (fb - 1) % 8:
            raise ValueError("[swoop] prefix, frames_a and frames_b must be 8n+1 (prefix >= 9)")
        stills = self.out["stills"].parent
        mid, end = stills / "mid.png", stills / "end.png"
        if not mid.is_file():
            raise ValueError("[swoop] mode = 'keyframes' needs [stills] commons_refs and mid_height_m")
        seed = (cfg.get("seeds") or [cfg.get("pick")])[0]
        from .terrain_quality import assess

        # mid_at: pin the middle keyframe at this frame of hop A instead of its last frame, let LTX fly on past it,
        # and cut hop A there. LTX parks on a last-frame pin (Monument Valley 17 Oct: the last 16 frames identical,
        # the hold check fired at 97 and at 57 frames); through an interior pin the camera keeps moving, so hop B
        # picks up from frames still in flight.
        mid_at = cfg.get("mid_at")
        if mid_at is not None and (mid_at % 8 or not n <= mid_at < fa - 1):
            raise ValueError("[swoop] mid_at must be a multiple of 8, at least prefix and before hop A's last frame")

        def hop(tag: str, source: list, frames: int, key: Path, prompt: str, at: int = -1) -> Path:
            sub = folder / tag / f"s{seed}"
            out = sub / "output.mp4"
            if not out.is_file():
                sub.mkdir(parents=True, exist_ok=True)
                prefix = sub / f"prefix_{n}.mp4"
                with V.Writer(prefix) as w:
                    w.extend(source[-n:])
                t0 = time.monotonic()
                shutil.copy(self.ltx(sub, frames, [(0, prefix), (at, key)], prompt.format(**self.words), seed,
                                     f"swoop_{tag}"), out)
                self.log(stage="swoop", hop=tag, candidate=seed, seconds=round(time.monotonic() - t0, 1))
            used, count = out, frames
            if at != -1:
                used, count = sub / "cut.mp4", at + 1
                with V.Writer(used) as w:
                    w.extend(list(V.frames(out))[:count])
            quality = assess(used, key, expected_frames=count)
            (folder / f"quality_{tag}.json").write_text(json.dumps(quality, indent=2) + "\n")
            if quality["warnings"]:
                raise ValueError(f"keyframe hop {tag} failed objective checks: " + "; ".join(quality["warnings"]))
            return used

        a = hop("a", list(V.frames(self.out["dive"])), fa, mid, cfg["prompt_a"], mid_at if mid_at is not None else -1)
        b = hop("b", list(V.frames(a)), fb, end, cfg["prompt_b"])
        out = folder / "swoop.mp4"
        with V.Writer(out) as w:
            w.extend(V.frames(a))
            w.extend(list(V.frames(b))[n:])  # b opens on a's last n frames
        (folder / "handover.json").write_text(json.dumps({"mode": "keyframes", "prefix": n, "guide_frame": None}))
        return out, {"seed": seed, "picked_by": "fixed", "mode": "keyframes", "prefix": n, "frames_a": fa,
                     "frames_b": fb, "hop_a": str(a), "hop_b": str(b)}

    def body(self, folder: Path):
        if self.terrain_mode:
            return self.terrain_body(folder)
        import fakeout_earth_zoom as f
        from dive_fx import apply as dive_fx
        cfg = self.m["body"]
        keep = int((self.out["cloud"].parent / "keep.txt").read_text())  # the opener frames the cloud stage kept
        dive_clip, swoop = self.out["dive"], self.out["swoop"]
        dive_last = dive_clip.with_name(dive_clip.stem + "_last.png")
        end_still = self.out["stills"].parent / "end.png"
        # 1. lead = opener[:keep] + cloud + dive, the dive's first 10 frames pulled to the cloud's white (Lab mean)
        lead = folder / "lead.mp4"
        top_n = keep + V.count(self.out["cloud"])
        white = V.frame(self.out["cloud"], -1)
        with V.Writer(lead) as w:
            for i, im in enumerate(V.frames(self.out["opener"])):
                if i >= keep:
                    break
                w.write(im)
            w.extend(V.frames(self.out["cloud"]))
            off = None
            n = self.m["swoop"].get("prefix", 1)  # with a prefix the swoop's first n frames re-render the dive's last n
            stop = V.count(dive_clip) - (n if n > 1 else 0)
            for i, im in enumerate(V.frames(dive_clip)):
                if i >= stop:
                    break
                if i < 10:
                    off = V.lab_mean(white) - V.lab_mean(im) if off is None else off
                    im = V.shift_lab(im, (1 - V.smoothstep(i / 10)) * off)
                w.write(im)
        # 2. the swoop (build2.py, as used for the current cut): white balance toward the end still; skip LTX's
        # near-still first frames (start at the first frame that differs from the dive's last frame by 0.8x the
        # dive's exit flow); hold one integer step S so its speed matches the dive; only ever step down (never
        # back up) once S x native flow falls below ``low``. No blending, so nothing ghosts.
        local = folder / "swoop.mp4"  # the helper writes next to its input: keep the swoop stage's cache clean
        shutil.copy(swoop, local)
        graded = Path(f._grade_ramp(str(local), dive_last, end_still))
        lead_s = V.small(lead, (480, 272))
        j = len(lead_s)
        exit_flow = float(np.nanmean(V.flow(lead_s, j - 4, j - 1, win=21)))
        sw_s = V.small(graded, (480, 272))
        v = np.nan_to_num(V.flow(sw_s, win=21), nan=0.0)
        if n > 1:  # the prefix re-render carries the dive's speed: play it 1:1 from frame 0, retime after it
            native = float(np.median(v[n:n + 35]))
            k0, idx = 0, list(range(n))
        else:
            native = float(np.median(v[5:40]))
            k0 = 1
            while k0 < 30 and V.flow(np.stack([lead_s[-1], sw_s[k0]]), win=21)[0] < 0.8 * exit_flow:
                k0 += 1
            idx = [k0]
        step = first_step = int(np.clip(round(exit_flow / max(native, 0.5)), 1, cfg.get("max_step", 3)))
        while True:
            q = idx[-1]
            if step > 1 and step * v[min(q, len(v) - 1)] < cfg.get("low", 7.0):
                step -= 1
            if q + step >= len(sw_s):
                break
            idx.append(q + step)
        # 3. seam: the first ``seam_span`` swoop frames blurred (sigma 2 -> 0) and pulled to the dive's last frame's
        # Lab mean and spread, so the soft photo hands over to sharp LTX over half a second, not in one frame
        import cv2
        span = cfg["seam_span"]
        ref = V.lab(V.frame(lead, -1)).reshape(-1, 3)
        mr, sr = ref.mean(0), ref.std(0)
        pos = {i: k for k, i in enumerate(idx)}
        retimed = folder / "joined.mp4"
        with V.Writer(retimed) as w:
            w.extend(V.frames(lead))
            for i, im in enumerate(V.frames(graded)):
                k = pos.get(i)
                if k is None:
                    continue
                if k < span:
                    t = (k + 1) / (span + 1)
                    wgt = 1 - V.smoothstep(t)
                    x = V.lab(cv2.GaussianBlur(im, (0, 0), 2.0 * wgt) if wgt > 0.05 else im)
                    flat = x.reshape(-1, 3)
                    m, sd = flat.mean(0), flat.std(0)
                    im = V.rgb((x - m) / np.maximum(sd, 1e-3) * (sd + wgt * (sr - sd)) + m + wgt * (mr - m))
                w.write(im)
        # 5. shake, radial blur and speed lines over the dive (the cloud exit to the join)
        out = folder / "body.mp4"
        fx = cfg.get("fx", {})
        dive_fx(str(retimed), str(out), top_n, j, shake_px=fx.get("shake_px", 12.0), roll_deg=fx.get("roll", 0.4),
                blur=fx.get("blur", 0.035), streaks=fx.get("streaks", 14), taper=fx.get("taper", 12), seed=fx.get("seed", 0),
                streak_width=fx.get("streak_width", 3), streak_alpha=fx.get("streak_alpha", 1.0))
        bounds = {"opener": [0, keep], "cloud": [keep, top_n], "dive": [top_n, j], "swoop": [j, V.count(out)]}
        (folder / "segments.json").write_text(json.dumps(bounds))
        return out, {"segments": bounds, "dive_exit_flow": round(exit_flow, 1), "swoop_native_flow": round(native, 1),
                     "step": first_step,
                     "swoop_start": k0, "swoop_kept": len(idx), "prefix": n,
                     "step_downs": [idx[i] for i in range(1, len(idx) - 1) if idx[i + 1] - idx[i] < idx[i] - idx[i - 1]]}

    def terrain_body(self, folder: Path):
        """Assemble the measured route without changing its guided camera timing."""
        from dive_fx import apply as dive_fx
        keep = int((self.out["cloud"].parent / "keep.txt").read_text())
        # skip: the opener eases out of its pinned orbit still, so its first frames barely move (every FLUX 3 cut,
        # 17 Oct: change ~1 for 0.4-0.5 s); start the cut that many frames in
        skip = min(int(self.m["opener"].get("skip", 0)), max(keep - 4, 0))
        top_n = keep - skip + V.count(self.out["cloud"])
        keyframes = self.m["swoop"].get("mode", "guide") == "keyframes"
        first_last = self.m["swoop"].get("mode", "guide") == "first_last" or keyframes
        computed = [im for im in V.frames(self.out["dive"])]
        if keyframes:  # the swoop opens on the dive's last ``prefix`` frames; the guide is not played
            computed = computed[:-self.m["swoop"].get("prefix", 9)]
        elif first_last:  # the measured guide plays model-free down to the hand-over, before the LTX prefix
            from PIL import Image
            ho = json.loads((self.out["swoop"].parent / "handover.json").read_text())
            scene = self.out["terrain"].parent
            computed += [np.asarray(Image.open(scene / "guide_frames" / f"{f:04d}.png").convert("RGB").resize((V.W, V.H)))
                         for f in range(2, ho["guide_frame"] - ho["prefix"] + 1)]
        dive_n = len(computed)
        swoop_n = V.count(self.out["swoop"])
        if not first_last and (dive_n != 64 or swoop_n != 81):
            raise ValueError(f"terrain route expected 64+81 frames, got {dive_n}+{swoop_n}")
        white = V.frame(self.out["cloud"], -1)
        dive0 = computed[0]
        dive_last = computed[-1]
        swoop0 = V.frame(self.out["swoop"], 0)
        import cv2
        seam_pair = np.stack([cv2.cvtColor(cv2.resize(im, (480, 272)), cv2.COLOR_RGB2GRAY)
                              for im in (dive_last, swoop0)])
        seam_flow = float(V.flow(seam_pair, win=21)[0])
        # the guide pins the swoop's first view to the dive's last view (first/last: the prefix follows on directly)
        duplicate_boundary = seam_flow < 1.0 and not first_last
        cloud_off = V.lab_mean(white) - V.lab_mean(dive0)
        handoff_off = V.lab_mean(dive_last) - V.lab_mean(swoop0)
        raw = folder / "raw.mp4"
        with V.Writer(raw) as w:
            for i, im in enumerate(V.frames(self.out["opener"])):
                if i >= keep:
                    break
                if i < skip:
                    continue
                w.write(im)
            w.extend(V.frames(self.out["cloud"]))
            # the dive is revealed through the cloud's own white, so the hand-over is a fade rather than a cut.
            # The reveal is slow enough to hide the render's own haze edge (which is a hard line where the hazy
            # upper frame meets the ground), and long enough that the aerial does not arrive in one frame.
            reveal = int(self.m["body"].get("dive_reveal", 10))
            for i, im in enumerate(computed):
                if i < reveal:
                    im = V.shift_lab(im, (1 - V.smoothstep(i / reveal)) * cloud_off)
                    # The cloud ends fully opaque. Reveal the measured landscape through that same cloud
                    # texture instead of cutting from white to a detailed aerial in one frame.
                    veil = 0.9 * (1 - V.smoothstep(i / reveal))  # 0.9: dive frame 0 already shows the ground,
                    # instead of repeating the cloud's opaque last frame (a held frame in both 17 Oct cuts)
                    im = np.clip(im.astype(np.float32) * (1 - veil)
                                 + white.astype(np.float32) * veil, 0, 255).astype(np.uint8)
                w.write(im)
            for i, im in enumerate(V.frames(self.out["swoop"])):
                if i == 0 and duplicate_boundary:
                    continue  # avoid a one-frame stop before the LTX camera starts moving
                if i < 8:
                    im = V.shift_lab(im, (1 - V.smoothstep(i / 8)) * handoff_off)
                w.write(im)
        out = folder / "body.mp4"
        fx = self.m["body"].get("fx", {})
        # ``extend``: carry the shake, blur and speed lines this many frames past the hand-over, so the join is
        # hidden in motion as in Higgsfield's swoop (Jamie 17 Oct, option 2); they taper out over ``taper`` frames
        birds = self.m["body"].get("birds", {})
        if birds.get("enabled"):
            from .birds import overlay
            flocked = folder / "birds.mp4"
            bird_facts = overlay(raw, flocked, at=top_n + dive_n + int(birds.get("offset", 0)),
                                 frames=int(birds.get("frames", 30)), count=int(birds.get("count", 9)),
                                 seed=int(birds.get("seed", 0)), size=float(birds.get("size", 0.07)))
            raw = flocked
        dive_fx(str(raw), str(out), top_n, top_n + dive_n + int(fx.get("extend", 0)),
                shake_px=fx.get("shake_px", 12.0), roll_deg=fx.get("roll", 0.4),
                blur=fx.get("blur", 0.035), streaks=fx.get("streaks", 12),
                taper=fx.get("taper", 12), seed=fx.get("seed", 0),
                streak_width=fx.get("streak_width", 2), streak_alpha=fx.get("streak_alpha", 0.8))
        bounds = {"opener": [0, keep - skip], "cloud": [keep - skip, top_n],
                  "dive": [top_n, top_n + dive_n],
                  "swoop": [top_n + dive_n, top_n + dive_n + swoop_n - int(duplicate_boundary)]}
        (folder / "segments.json").write_text(json.dumps(bounds))
        return out, {"segments": bounds, "first_last": first_last, "fx_extend": int(fx.get("extend", 0)),
                     "birds": bird_facts if birds.get("enabled") else None,
                     "handoff_lab_offset": np.round(handoff_off, 2).tolist(),
                     "handoff_flow": round(seam_flow, 2), "dropped_duplicate_boundary": duplicate_boundary}

    def end_still_full(self) -> np.ndarray:
        """The end still at its full size, if FLUX 3 drew one.

        The video stages work at 960x544, so the copy of the end still they use has been scaled down twice. The
        full-size copy is what the last second of the shot is held on: the video model's face is a blur by then,
        the photograph's is not.
        """
        import cv2
        folder = self.out["stills"].parent
        full = folder / "end_full.png"
        path = full if full.is_file() else folder / "end.png"
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("the end still is missing")
        if full.is_file():
            # the still may have been registered onto the guide (a zoom and a shift): apply the same crop here,
            # so the held frame matches what the video was pinned to
            layout = {}
            if (folder / "layout.json").is_file():
                try:
                    layout = json.loads((folder / "layout.json").read_text())
                except ValueError:
                    layout = {}
            reg = (layout.get("registration") or {})
            if reg.get("applied"):
                k = image.shape[1] / 960.0
                span_w, span_h = 960.0 / reg["scale"] * k, 544.0 / reg["scale"] * k
                x0, y0 = max(reg["x0"] * k, 0), max(reg["y0"] * k, 0)
                crop = image[int(round(y0)):int(round(y0 + span_h)), int(round(x0)):int(round(x0 + span_w))]
                if crop.size:
                    image = crop
        return V.fit_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def fit_to_frame(still: np.ndarray, frame: np.ndarray) -> tuple[float, float, float]:
        """The zoom and offset that put the still's person where the frame's person is.

        The video model's version of the scene is framed a little differently from the photograph the still was
        drawn from, so cutting straight from one to the other jumps. Fitting the still to the person in the last
        video frame leaves the cut as a change of sharpness and nothing else. (z, tx, ty) for a warpAffine.
        """
        from .end_register import person_box
        try:
            a, b = person_box(still), person_box(frame)
        except Exception:
            a = b = None
        if not a or not b:
            return 1.0, 0.0, 0.0
        h = (b[3] - b[1]) / max(1, a[3] - a[1])
        w = (b[2] - b[0]) / max(1, a[2] - a[0])
        z = (h + w) / 2
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return z, bcx - z * acx, bcy - z * acy

    def stills_route(self) -> dict:
        """Where the stills are drawn, and where that server is asked to unload afterwards.

        With a stills server of its own (started by the build, or one that is already running), the FLUX 3 weights
        live on that server and the video server keeps the card to itself. Without one, the stills are drawn on the
        build's own server.
        """
        cfg = self.m["stills"]
        if self.stills_url:
            return {"url": self.stills_url, "free_url": self.stills_free_url or self.stills_url}
        if cfg.get("flux3"):
            return {"url": cfg["flux3_url"], "free_url": cfg.get("flux3_free_url")}
        return {"free_url": self.free_url or self.url}

    # --- the ComfyUI the build drives -----------------------------------------------------

    def server_command(self) -> str | None:
        """The command that starts the ComfyUI this build uses, if the build is to own it.

        ComfyUI's VRAM paging holds the weights it has loaded, and asking it to unload them does not give the
        memory back: the stills' FLUX 3 pages stay resident and the LTX 2.5 transformer cannot be faulted in on a
        24 GB card. A server the build owns can be restarted, which does give the memory back. So the rule is: the
        stills get a clean server, and the video stages get a fresh one.
        """
        return (self.m.get("video", {}).get("server_cmd")
                or os.environ.get("TOUCHDOWN_COMFY_SERVER_CMD") or None)

    def warn_about_the_card(self) -> None:
        """Say so when the build cannot give the card back between the stills and the video stages."""
        if self.server_command():
            return
        if not self.m["stills"].get("flux3"):
            return  # FLUX.2 klein is small enough to share the card with the video stages
        print("WARNING: the stills are drawn with FLUX 3 (about 10 GB with its text encoder) and the video stages "
              "need the LTX 2.5 transformer (21.5 GB). On a 24 GB card the video stages can fail with a VRAM "
              "error, because ComfyUI does not release the stills' memory when it is asked to unload it.\n"
              "         Set TOUCHDOWN_COMFY_SERVER_CMD (and TOUCHDOWN_COMFY_SERVER_CWD, TOUCHDOWN_COMFY_SERVER_URL) "
              "so the build can restart ComfyUI between the stills and the video stages.", flush=True)

    def ensure_server(self) -> None:
        """Make sure the ComfyUI this build uses is running, and that it is a clean one.

        If a server is already at that address it is stopped first: it would still be holding the last build's
        models (about 19 GB of staged pages that ComfyUI does not release when it is asked to unload them), and
        the next stage would not fit.
        """
        cmd = self.server_command()
        if not cmd:
            return
        if self.server is not None and self.server.poll() is None:
            return
        url = (self.m.get("video", {}).get("server_url")
               or os.environ.get("TOUCHDOWN_COMFY_SERVER_URL") or "").rstrip("/")
        if url and self.server_up(url):
            print(f"a ComfyUI is already running at {url}: stopping it, so the build starts from a clean card", flush=True)
            self.stop_listener(self.server_port(url))
            for _ in range(30):
                if not self.server_up(url):
                    break
                time.sleep(1)
        cwd = self.m.get("video", {}).get("server_cwd") or os.environ.get("TOUCHDOWN_COMFY_SERVER_CWD")
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(f"starting ComfyUI: {cmd}", flush=True)
        self.server = subprocess.Popen(shlex.split(cmd), cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
        if not url:
            return
        for _ in range(150):  # a cold ComfyUI takes a while to load its custom nodes
            if self.server_up(url):
                print(f"ComfyUI up at {url}", flush=True)
                return
            if self.server.poll() is not None:
                raise ValueError(f"ComfyUI exited: {cmd}")
            time.sleep(2)
        self.stop_server()
        raise ValueError(f"ComfyUI did not come up at {url}: {cmd}")

    @staticmethod
    def server_port(url: str) -> int | None:
        """The port in a ComfyUI address, or None."""
        try:
            return int(url.rstrip("/").rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def stop_listener(port: int | None) -> None:
        """Stop whatever is listening on a port. The build owns that address."""
        if not port:
            return
        import signal
        try:
            out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True).stdout
        except OSError:
            return
        for line in out.splitlines():
            if f":{port} " not in line or "pid=" not in line:
                continue
            try:
                pid = int(line.split("pid=")[1].split(",")[0])
                os.kill(pid, signal.SIGTERM)
                print(f"stopped the process on port {port} (pid {pid})", flush=True)
            except (ValueError, ProcessLookupError, PermissionError):
                continue

    def stop_server(self) -> None:
        """Stop the ComfyUI this build started, so its memory goes back to the card."""
        if self.server is None:
            return
        process, self.server = self.server, None
        try:
            os.killpg(os.getpgid(process.pid), 15)
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=60)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), 9)
            except Exception:
                pass
        print("ComfyUI stopped; its memory is back on the card", flush=True)

    def start_stills_server(self) -> subprocess.Popen | None:
        """Start a ComfyUI for the stills, if the manifest asks for one.

        A 24 GB card cannot hold the FLUX 3 stills (about 10 GB with the text encoder) and the LTX 2.5
        transformer (21.5 GB) at the same time, and ComfyUI's paging does not release the stills' pages when it is
        asked to unload them. So the stills get their own server, and the build stops it when they are done: the
        video stages then have the card to themselves.

        The command comes from [stills] server_cmd (or TOUCHDOWN_STILLS_SERVER_CMD); the address from
        server_url. Returns the process, so the caller can stop it.
        """
        cfg = self.m["stills"]
        cmd = cfg.get("server_cmd") or os.environ.get("TOUCHDOWN_STILLS_SERVER_CMD")
        if not cmd:
            return None
        cwd = cfg.get("server_cwd") or os.environ.get("TOUCHDOWN_STILLS_SERVER_CWD")
        url = cfg.get("server_url") or os.environ.get("TOUCHDOWN_STILLS_SERVER_URL")
        if not url:
            raise ValueError("[stills] server_cmd needs server_url (the address the stills are drawn on)")
        if self.server_up(url):
            print(f"stills server already up at {url}: it keeps its models, so the video stages may not fit", flush=True)
            return None
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(shlex.split(cmd), cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(150):  # a cold ComfyUI takes a while to load its custom nodes
            if self.server_up(url):
                print(f"stills server up at {url}", flush=True)
                return process
            if process.poll() is not None:
                raise ValueError(f"the stills server exited: {cmd}")
            time.sleep(2)
        process.terminate()
        raise ValueError(f"the stills server did not come up at {url}: {cmd}")

    @staticmethod
    def server_up(url: str) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=5).read()
            return True
        except Exception:
            return False

    @staticmethod
    def stop_stills_server(process: subprocess.Popen | None) -> None:
        """Stop the stills server, so its models give their memory back to the card."""
        if process is None:
            return
        try:
            os.killpg(os.getpgid(process.pid), 15)
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), 9)
            except Exception:
                pass
        print("stills server stopped; its memory is back on the card", flush=True)

    def end(self, folder: Path):
        cfg = self.m["end"]
        body = self.out["body"]
        bounds = json.loads((body.parent / "segments.json").read_text())
        j, n = bounds["swoop"]
        if self.terrain_mode and cfg.get("prefix_frames"):
            prefix_n = int(cfg["prefix_frames"])
            if prefix_n < 9 or prefix_n % 8 != 1 or prefix_n > n - j:
                raise ValueError("[end] prefix_frames must be 8n+1 and fit within the swoop")
            k = n - prefix_n
            prefix = folder / "prefix.mp4"
            with V.Writer(prefix) as w:
                for i, im in enumerate(V.frames(body)):
                    if k <= i < n:
                        w.write(im)
            prompt = cfg["lock_prompt"].format(scene_ambient=self.scene_ambient(), **self.words)
            pin = self.out["stills"].parent / "end.png"
            # lock_pin: where the end still is pinned. "last" (the original) makes LTX converge on it and freeze
            # (17 Oct: change 0-1 over the last 3 s of both cuts); the 9-frame prefix already ends on that still,
            # so "none" lets the scene move after the hand-over, and a number pins it that early instead.
            where = cfg.get("lock_pin", "last")
            pins = [(0, prefix)] + ([] if where == "none" else [(-1 if where == "last" else int(where), pin)])
            clip, facts = self.candidates("end", folder,
                lambda s, d_: self.ltx(d_, cfg["frames"], pins, prompt, s, "end_locked"))
            out = folder / "final.mp4"
            hold = int(cfg.get("still_hold", 0))  # frames of the end still held at the very end
            with V.Writer(out) as w:
                for i, im in enumerate(V.frames(body)):
                    if i >= k:
                        break
                    w.write(im)
                # keep_after_pin: end the clip this many frames after a numeric pin (Grand Canyon v7: the face
                # matched the still at the pin, f48, then drifted, looking down and away by the last frame; Jamie
                # wants a faster sequence anyway)
                keep = cfg.get("keep_after_pin")
                last = int(where) + int(keep) if keep is not None and where not in ("none", "last") else None
                if hold and where not in ("none", "last"):
                    # the hold takes over at the pin: the clip is pinned to the still there, so the two agree and
                    # the dissolve is invisible. The frames after the pin are the model drifting away from the still.
                    last = int(where)
                previous = None
                last_written = 0
                for i, im in enumerate(V.frames(clip)):
                    if last is not None and i > last:
                        break
                    im = V.push_in(im, i, cfg["frames"], prefix_n,
                                   float(cfg.get("zoom", 1)),
                                   float(cfg.get("zoom_focus_x", 0.5)),
                                   float(cfg.get("zoom_focus_y", 0.46)))
                    previous = im
                    last_written = i
                    w.write(im)
                if hold and previous is not None:
                    # Hold on the end still: it is the only copy of the face that is sharp. The video model works
                    # at 960x544 and its output is upscaled, so by the last second the person is a blur; the
                    # photograph FLUX 3 drew is not. The cut is made at the pin, where the clip and the still
                    # already agree, and the push-in continues. There is no dissolve: crossfading a video frame
                    # into a photograph of the same scene ghosts two faces over each other.
                    still = self.end_still_full()
                    zoom = float(cfg.get("zoom", 1))
                    grow = float(cfg.get("still_hold_zoom", 0.06))
                    fade = min(int(cfg.get("still_hold_fade", 0)), hold)
                    z_fit, tx_fit, ty_fit = self.fit_to_frame(still, previous)
                    # where the push-in would have ended, so the hold carries on moving
                    z_end, cx_end, cy_end = V.push_in_zoom(1, 1, 0, zoom * (1 + grow),
                                                           float(cfg.get("zoom_focus_x", 0.5)),
                                                           float(cfg.get("zoom_focus_y", 0.46)))
                    tx_end, ty_end = V.W / 2 - z_end * cx_end, V.H / 2 - z_end * cy_end
                    import cv2
                    for i in range(hold):
                        t = i / max(1, hold - 1)
                        z = z_fit + (z_end - z_fit) * t
                        tx = tx_fit + (tx_end - tx_fit) * t
                        ty = ty_fit + (ty_end - ty_fit) * t
                        matrix = np.float32([[z, 0, tx], [0, z, ty]])
                        im = cv2.warpAffine(still, matrix, (V.W, V.H), flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE)
                        if i < fade:
                            im = V.blend(previous, im, V.smoothstep((i + 1) / (fade + 1)))
                        w.write(im)
            bounds["swoop"][1] = k
            bounds["end"] = [k, V.count(out)]
            (folder / "segments.json").write_text(json.dumps(bounds))
            return out, {**facts, "settle_frame": k, "segments": bounds,
                         "seconds_total": round(bounds["end"][1] / V.FPS, 2)}
        if self.terrain_mode:
            k = n - 1  # keep the full measured approach, then animate its final view
        elif cfg.get("start") is not None:  # a person picked the frame (``start`` frames after the swoop's first)
            k = j + int(cfg["start"])
            self.log(stage="end", manual="living-end start frame", start=cfg["start"], intervention=True)
        elif cfg.get("start_frac") is not None:  # a share of the kept swoop: v5 started at 64 of 77 (0.83), the
            k = j + round(float(cfg["start_frac"]) * (n - j))  # oblique swoop settled at ~120 of 156 (0.77)
        else:  # where the swoop settles: the first of 3 steps below ``settle`` in the swoop's second half
            d = V.diff(V.small(body))
            k = next((i for i in range(j + (n - j) // 2, n - 3) if (d[i:i + 3] < cfg["settle"]).all()), n - 1)
        start = V.save_png(V.frame(body, k), folder / "start.png")
        prompt = cfg["prompt"].format(**self.words)
        clip, facts = self.candidates("end", folder, lambda s, d_: self.ltx(d_, cfg["frames"], [(0, start)], prompt, s, "end"))
        out = folder / "final.mp4"
        prev = V.frame(body, k - 1)
        off = None
        span = cfg.get("ramp", 24)
        with V.Writer(out) as w:
            for i, im in enumerate(V.frames(body)):
                if i >= k:
                    break
                w.write(im)
            for i, im in enumerate(V.frames(clip)):
                if i < span:
                    off = V.lab_mean(prev) - V.lab_mean(im) if off is None else off
                    im = V.shift_lab(im, (1 - V.smoothstep(i / span)) * off)
                w.write(im)
        bounds["swoop"][1] = k
        bounds["end"] = [k, V.count(out)]
        (folder / "segments.json").write_text(json.dumps(bounds))
        return out, {**facts, "settle_frame": k, "segments": bounds, "seconds_total": round(bounds["end"][1] / V.FPS, 2)}

    MID_STILL_PROMPT = ("A real aerial photograph from the same camera position and direction as image 1: the same "
                        "ground, the same ridgelines, the same horizon and the same sky, but photographed rather "
                        "than drawn. Real rock, sand and scrub with natural texture and sunlight, the landforms in "
                        "exactly the places image 1 shows them. No text.")

    def swoop_mid_pin(self, folder: Path, cfg: dict, scene: Path) -> list:
        """Photographic frames pinned along the swoop.

        The swoop is a video-to-video pass over the *shaded* Blender guide, and the model copies that shading:
        the low-poly look, and the mid-ground landforms it invents. Worse, it does not follow the guide's pace: it
        reaches the landing early, so the character is standing there by the middle of the clip, on ground that is
        not the ground the camera just flew over. A photographic frame pinned at a point on the measured path
        forces the clip through that point, so the descent keeps its shape and its place.

        ``mid_pin_frame`` is one frame index or a list of them (0-based, into the 81-frame guide).
        """
        wanted = cfg.get("mid_pin_frame")
        if wanted is None:
            return []
        indices = [int(wanted)] if isinstance(wanted, (int, str)) and str(wanted).isdigit() else [int(i) for i in wanted]
        scfg = self.m["stills"]
        route = self.stills_route() if scfg.get("flux3") else {}
        out = folder / "mid"
        out.mkdir(parents=True, exist_ok=True)
        pins = []
        for index in indices:
            frames = scene / "guide_frames" / f"{int(index) + 1:04d}.png"
            if not frames.is_file():
                print(f"mid pin: no guide frame {frames}", flush=True)
                continue
            still = out / f"mid_{int(index):04d}.png"
            if not still.is_file():
                structure = self.structure_map(frames, out / f"mid_{int(index):04d}_structure.png")
                prompt = cfg.get("mid_prompt", self.MID_STILL_PROMPT)
                still = Path(self.comfy("mid_still", folder=str(out), seed=int(cfg.get("mid_seed", 0)) + index,
                                        name=self.name, prompt=prompt, reference=str(structure),
                                        flux3=bool(scfg.get("flux3")), **route))
            pins.append((int(index), still))
            print(f"mid pin: frame {index} pinned from {still}", flush=True)
        return pins

    def face(self, folder: Path):
        """Sharpen the approaching figure's face without fixing a hand-placed mask or editing every frame."""
        from . import face_refine

        cfg = self.m["face"]
        source = self.out["end"]
        bounds = json.loads((source.parent / "segments.json").read_text())
        boundary = bounds["end"][0]
        ref = folder / "reference.png"
        mode = cfg.get("source", "final_frame")
        if mode not in ("final_frame", "flux", "photo"):
            raise ValueError(f"[face] source must be final_frame, flux or photo, got {mode!r}")
        if mode == "photo":
            # The subject's own photograph: sharp, and the person the clip is about. A face cropped out of a
            # generated frame is about 50 px across, so pasting it back can only soften what is there.
            subject = self.m["shot"].get("subject")
            if not subject:
                raise ValueError("[face] source = 'photo' needs a subject photograph")
            extraction = face_refine.extract_photo_plate(Path(subject), ref)
            reference_index = None
        else:
            reference_index = (V.count(source) - 1 - int(cfg.get("reference_from_last", 20))
                               if mode == "final_frame" else boundary + int(cfg.get("reference_offset", 8)))
            extraction = face_refine.extract_reference(source, reference_index, ref)
        out = folder / "final.mp4"
        if not extraction["detected"]:
            shutil.copy(source, out)
            facts = {"skipped": "no centred face detected in the reference", "reference": extraction}
        else:
            plate = (ref if mode in ("final_frame", "photo") else
                     Path(self.comfy("face_plate", folder=str(folder / "plate_job"), reference=str(ref),
                                     seed=int(cfg["seed"]), prompt=cfg["prompt"].format(**self.words))))
            shutil.copy(plate, folder / "plate.png")
            try:
                facts = face_refine.render(source, folder / "plate.png", out, boundary,
                                           lead=int(cfg.get("lead", 14)), ramp=int(cfg.get("ramp", 12)),
                                           hold_after=int(cfg.get("hold_after", 8)), fade=int(cfg.get("fade", 16)),
                                           strength=float(cfg.get("strength", 0.65)))
            except ValueError as exc:  # a failed face track must not make an otherwise valid shot unusable
                shutil.copy(source, out)
                facts = {"skipped": str(exc)}
            facts["reference"] = extraction
            facts["source"] = mode
        (folder / "segments.json").write_text(json.dumps(bounds))
        return out, facts

    def sound(self, folder: Path):
        cfg = self.m["sound"]
        final = self.out["face"] if self.m.get("face", {}).get("enabled", False) else self.out["end"]
        bounds = json.loads((final.parent / "segments.json").read_text())
        total = bounds["end"][1]
        if cfg.get("engine", "procedural") == "procedural":
            # touchdown/wind_bed.py (owned by the sound thread): shaped-noise wind from the segments, no model, so
            # nothing can echo; its look (levels, turbulence) is tuned in that script, not here
            source, segments = final, final.parent / "segments.json"
            pacing = {}
            if cfg.get("pace"):
                from . import pace
                source, segments = folder / "paced.mp4", folder / "segments.json"
                pc = cfg["pace"]
                pacing = pace.render(final, bounds, source, segments,
                                     peak=pc["peak"], dive_start=pc["dive_start"], swoop_end=pc["swoop_end"],
                                     entry_rate=float(pc.get("entry_rate", 1.0)))
            out = folder / "touchdown.mp4"
            cmd = [sys.executable, str(Path(__file__).resolve().parent / "wind_bed.py"), "--video", str(source),
                   "--segments", str(segments), "--out", str(out),
                   "--seed", str(cfg.get("wind_seed", 7))]
            if cfg.get("wind_fade"):
                cmd += ["--fade", cfg["wind_fade"]]
            subprocess.run(cmd, check=True)
            return out, {"engine": "procedural", "wind_seed": cfg.get("wind_seed", 7), "pace": pacing}

        def at(anchor: str) -> int:
            """'swoop', 'end-36', 'swoop+6', 'total' -> frame number (segment starts; 'total' = last frame + 1)."""
            for sign in ("+", "-"):
                if sign in anchor:
                    name, num = anchor.split(sign)
                    return at(name) + (int(num) if sign == "+" else -int(num))
            return total if anchor == "total" else bounds[anchor][0]

        shots = []
        for win in cfg["windows"]:
            a, b = max(0, at(win["from"])), min(total, at(win["to"]))
            n = b - a if win.get("loop") else (b - a - 1) // 8 * 8 + 1  # rendered windows need 8n+1 frames
            shots.append({**{k: v for k, v in win.items() if k not in ("from", "to")}, "start": a, "frames": n})
        (folder / "shots.json").write_text(json.dumps(shots, indent=1))
        f0, f1 = at(cfg["fade"][0]), at(cfg["fade"][1])
        seed = cfg["pick"] if cfg.get("pick") is not None else cfg["seeds"][0]
        cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "foley_v2a.py"), "--video", str(final),
               "--shots", str(folder / "shots.json"), "--seeds", str(seed), "--out-dir", str(folder),
               "--fade-out", f"{f0}:{f1}:{cfg['fade'][2]}", "--server", cfg.get("server", "http://127.0.0.1:8190")]
        if cfg.get("workflow"):  # the official workflow (gated HF repo), not in this repo: a path, or relative to the manifest
            wf = Path(cfg["workflow"]).expanduser()
            if not wf.is_absolute() and (self.manifest_path.parent / wf).is_file():
                wf = self.manifest_path.parent / wf
            cmd += ["--workflow", str(wf)]
        for key in ("checkpoint", "text_encoder"):  # model files as named on this ComfyUI
            if cfg.get(key):
                cmd += ["--" + key.replace("_", "-"), cfg[key]]
        if cfg.get("lora"):
            cmd += ["--lora", cfg["lora"]]
        if cfg.get("decomb", True):
            cmd.append("--decomb")
        subprocess.run(cmd, check=True)
        tag = ("_dc" if cfg.get("decomb", True) else "") + "_fade"
        made = folder / f"{final.stem}_foley_s{seed}{tag}.mp4"
        if not made.is_file():
            raise RuntimeError(f"foley_v2a did not write {made}")
        out = folder / "touchdown.mp4"
        shutil.copy(made, out)
        return out, {"seed": seed, "windows": [(s["name"], s["start"], s["frames"]) for s in shots], "fade": [f0, f1]}

    # --- the whole build ---------------------------------------------------------------------------------------

    def build(self, until: str | None, only: list[str] | None = None) -> Path | None:
        deps = {"terrain": (), "stills": ("terrain",) if self.terrain_mode else (),
                "opener": ("stills",), "cloud": ("opener", "dive"),
                "dive": ("terrain", "stills") if self.terrain_mode else (), "swoop": ("dive", "stills"),
                "body": ("opener", "cloud", "dive", "swoop", "stills"), "end": ("body",),
                "face": ("end",),
                "sound": ("face",) if self.m.get("face", {}).get("enabled", False) else ("end",)}
        last = None
        waiting: dict[str, str] = {}  # stage -> why; stages that don't depend on one still run, so picks batch up
        try:
            for stage in STAGES:
                if stage == "terrain" and not self.terrain_mode:
                    continue
                if stage == "face" and not self.m.get("face", {}).get("enabled", False):
                    continue
                if stage == "sound" and not self.m.get("sound", {}).get("enabled", True):
                    continue
                if only and stage not in only:
                    continue
                if only and any(d not in only for d in deps[stage]):
                    raise SystemExit(f"--only {stage} also needs its inputs: {', '.join(deps[stage])}")
                if any(d in waiting for d in deps[stage]):
                    waiting[stage] = "waits on " + ", ".join(d for d in deps[stage] if d in waiting)
                else:
                    try:
                        last = self.run(stage, deps[stage], lambda folder, s=stage: getattr(self, s)(folder))
                    except NeedsPick as e:
                        waiting[stage] = str(e)
                if stage == until:
                    break
        finally:
            self.report()
            self.stop_server()  # a ComfyUI this build started does not outlive it
            self.stop_stills_server(self.stills_server)
        if waiting:
            print("\nStopped for picks:")
            for stage, why in waiting.items():
                print(f"  {stage}: {why}")
            return None
        return last

    def report(self) -> None:
        if not self.log_path.is_file():
            return
        rows = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        stage_s = sum(r.get("seconds", 0) for r in rows if "facts" in r)
        cand_s = sum(r.get("seconds", 0) for r in rows if "candidate" in r)
        picks = [r for r in rows if r.get("intervention")]
        lines = [f"# Touchdown build: {self.name}", "", f"Manifest: `{self.manifest_path}`", "",
                 f"- Logged stage time (all runs, cached reuse not counted): {stage_s / 60:.1f} min",
                 f"- of which candidate generations: {cand_s / 60:.1f} min",
                 "- Manual interventions: {} ({})".format(len(picks), ", ".join("{}={}".format(r["stage"], r.get("picked", r.get("manual"))) for r in picks)), "",
                 "| stage | key | seconds | facts |", "|---|---|---|---|"]
        for r in rows:
            if "facts" in r:
                lines.append(f"| {r['stage']} | {r['key']} | {r['seconds']} | {json.dumps(r['facts'])[:200]} |")
        (self.root / "report.md").write_text("\n".join(lines) + "\n")
        self.point_at_the_latest_clip()

    def point_at_the_latest_clip(self) -> None:
        """Copy the finished clip to ``<run>/latest.mp4``, so there is one obvious file per build."""
        sound = self.out.get("sound")
        if not sound or not Path(sound).is_file():
            return
        try:
            shutil.copy(sound, self.root / "latest.mp4")
        except OSError as exc:
            print(f"could not copy the clip to latest.mp4: {exc}", flush=True)


DESCENTS = {"45": "oblique", "90": "nadir"}


def _resolve_descent(m: dict, descent: str) -> None:
    """Fold the chosen descent into [cloud] and [dive]: cloud ``prompt_45`` / ``prompt_90`` -> ``prompt``, and the
    [dive.oblique] / [dive.nadir] sub-table -> the dive block (mode set to match). Manifests without the
    sub-tables (one fixed descent) pass through unchanged."""
    if descent not in DESCENTS:
        raise SystemExit(f"descent must be one of {sorted(DESCENTS)}, not {descent!r}")
    cloud = m.get("cloud", {})
    chosen = cloud.get(f"prompt_{descent}")
    for k in [k for k in cloud if k.startswith("prompt_")]:
        cloud.pop(k)
    if chosen:
        cloud["prompt"] = chosen
    dive = m.get("dive", {})
    subs = {k: dive.pop(k) for k in list(dive) if k in DESCENTS.values() and isinstance(dive[k], dict)}
    if subs:
        mode = DESCENTS[descent]
        dive.update(subs.get(mode, {}))
        dive["mode"] = mode


def _before_fog(opener: Path, keep: int, contrast: float = 0.45) -> int:
    """The last opener frame to keep before it fogs over: two frames before contrast falls under ``contrast`` of
    the opening frames' (the cloud stage starts from the frame after it), never later than ``keep``. Higher cuts
    earlier (t1 still showed haze at 0.45)."""
    sm = V.small(opener)
    std = sm.reshape(len(sm), -1).std(axis=1)
    ref = float(np.median(std[:6]))
    fog = next((i for i in range(6, len(std)) if std[i] < contrast * ref), len(std))
    return int(max(6, min(keep, fog - 2)))


def strip(clip: Path, out: Path, n: int = 8) -> None:
    total = V.count(clip)
    picks = "+".join(f"eq(n\\,{round(i * (total - 1) / (n - 1))})" for i in range(n))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", f"select='{picks}',scale=480:-1,tile={n // 2}x2",
                    "-frames:v", "1", "-vsync", "0", str(out)], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--until", choices=STAGES, help="stop after this stage")
    p.add_argument("--only", choices=STAGES, action="append",
                   help="run just these stages (with their inputs); e.g. --only dive, which needs no GPU")
    p.add_argument("--output", type=Path, help="default: the manifest's output, else outputs/touchdown")
    p.add_argument("--url", help="ComfyUI server (default: COMFY_BASE_URL)")
    p.add_argument("--comfy-python", help="run the ComfyUI jobs with this Python (one that has comfy_sdk); "
                                          "default: TOUCHDOWN_COMFY_PYTHON, else this one")
    p.add_argument("--descent", choices=sorted(DESCENTS), help="45: dive at ~45 degrees after the cloud (default); "
                   "90: tip straight down inside the cloud. Overrides [shot] descent")
    p.add_argument("--dry-run", action="store_true", help="show which stages are cached and which would run")
    p.add_argument("--unattended", action="store_true", help="run exactly one precommitted seed per generated stage, without picks")
    a = p.parse_args()
    b = Build(a.manifest, a.output, a.url, a.dry_run, a.comfy_python, a.descent, a.unattended)
    out = b.build(a.until, a.only)
    if out:
        if a.dry_run:
            print("\nDry run complete; no media generated")
            return
        if a.unattended and not a.until and not a.only and b.m.get("sound", {}).get("enabled", True):
            from .audit import inspect as audit_output
            end = b.out["end"]
            seed = b.m["end"]["seeds"][0]
            paced_segments = out.parent / "segments.json"
            result = audit_output(out, end.parent / f"s{seed}" / "output.mp4",
                                  paced_segments if paced_segments.exists() else end.parent / "segments.json")
            if b.terrain_mode:
                from .terrain_quality import assess
                approach = assess(b.out["swoop"], b.out["stills"].parent / "end.png",
                                  expected_frames=V.count(b.out["swoop"]),
                                  tail_ok=b.m["swoop"].get("mode") == "first_last")
                result["terrain_approach"] = approach
                result["objective_warnings"].extend(approach["warnings"])
            audit_path = b.root / "audit.json"
            audit_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Audit {audit_path}: {len(result['objective_warnings'])} objective warnings")
            if b.terrain_mode and approach["warnings"]:
                raise ValueError("guided terrain approach failed objective checks: " + "; ".join(approach["warnings"]))
        print(f"\nBuilt {out}\nReport {b.root / 'report.md'}")


if __name__ == "__main__":
    main()
