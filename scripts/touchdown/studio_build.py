"""Run one saved photographic studio setup, with isolated local GPU stages."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import time
from typing import Literal
import urllib.request

import cv2
from PIL import Image
from pydantic import BaseModel

from touchdown import portrait_flight, split_flight, vision_check as vision, photo_delivery
from touchdown.region_facts import region_facts
from touchdown.authored_scene import digest, generate_asset
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.studio_brief import (MODEL, MODEL_DIGEST, EndingReview, FRAMINGS, POSES, chat,
                                    grounding, scene_config, independent_delivery_review)
from touchdown.studio_brief import approach_view_review, ending_action_review, high_view_review, orbital_view_review
from touchdown.surface_stability import assess, frame_differences, jump_cuts

STAGES = ("references", "arrival", "arrival_check", "portrait", "flight", "pacing", "review")
COMFY_URL = os.environ.get("TOUCHDOWN_FLUX3_FREE_URL", "http://127.0.0.1:8192")
VIDEO_URL = os.environ.get("TOUCHDOWN_COMFY_URL", "http://127.0.0.1:8193")
VISION_URL = os.environ.get("TOUCHDOWN_VISION_URL", "http://127.0.0.1:11439")
REPO = Path(__file__).resolve().parents[2]


class Held(Exception):
    pass


def surface_report(root: Path) -> dict | None:
    """Measure landmark surface stability on the delivered clip, next to the outcome.

    Advisory on purpose: the redraw ceiling is calibrated on one scene, so a new landmark records
    its number and its verdict without holding the film. The last 3.2 seconds before the portrait
    begins are the approach where the landmark is the subject; earlier than that it is too small to
    measure or too far away to matter.
    """
    candidate = root / "motion/earth_zoom.mp4"
    if not candidate.is_file():
        return None
    info = cv2.VideoCapture(str(candidate))
    frames = int(info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = info.get(cv2.CAP_PROP_FPS) or 24
    info.release()
    pacing = json.loads((root / "motion/pacing.json").read_text()) if (root / "motion/pacing.json").is_file() else {}
    near_start = pacing.get("native_start_frame", frames - 96) / fps
    # The landmark is the subject only once the cloud has cleared. On a ten-second film the last
    # 3.2 s before the portrait is the approach; on a six-second film the same rule reaches back
    # into the cloud deck, where the whole frame changes and nothing can be measured. The cloud
    # exit frame is recorded by pacing, so never start before it.
    start = max(0.0, near_start - 3.2)
    if pacing.get("target_high_frame") is not None:
        start = max(start, pacing["target_high_frame"] / fps)
    window = (start, near_start)
    if window[1] - window[0] < 1.0:
        return None
    report = assess(candidate, window=window)
    write_json(root / "surface-stability.json", report)
    print(f"[surface] {window[0]:.2f}-{window[1]:.2f}s  redraw {report['redraw']:.3f}  "
          f"above floor {report['redraw_above_floor']:+.3f}  {report['verdict']['status'].upper()}",
          flush=True)
    return {k: report[k] for k in ("warble", "detail", "redraw", "control", "redraw_above_floor",
                                   "gain", "pairs", "repeats", "verdict", "window")}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def reachable(url, endpoint):
    try:
        with urllib.request.urlopen(url.rstrip("/") + endpoint, timeout=2):
            return True
    except Exception:
        return False


@contextmanager
def runtime(kind, root):
    # Also serialize manual review commands with queued GPU stages.
    import fcntl
    lock_path = REPO / "outputs/.touchdown-photo-gpu.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with _runtime(kind, root) as metadata:
            yield metadata


@contextmanager
def _runtime(kind, root):
    """Own ComfyUI's process. Never stop another process merely for using a port.

    Children share this build's process group, allowing the worker to cancel all
    of its GPU work together. Each stage exits before another model takes over.
    """
    vision_stage = kind == "vision"
    url, endpoint = (VISION_URL, "/api/version") if vision_stage else (COMFY_URL, "/system_stats")
    process = None
    if not reachable(url, endpoint):
        if vision_stage:
            command, cwd = ["bash", str(REPO / "scripts/start_touchdown_vision_checker.sh")], REPO
        else:
            command = shlex.split(os.environ["TOUCHDOWN_COMFY_SERVER_CMD"])
            cwd = os.environ["TOUCHDOWN_COMFY_SERVER_CWD"]
        with (root / f"{kind}-runtime.log").open("a") as log:
            process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    elif not vision_stage:
        raise RuntimeError("The generation ComfyUI port is already in use. The worker left that session untouched.")
    try:
        deadline = time.monotonic() + 180
        while not reachable(url, endpoint):
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"The {kind} runtime exited. See {kind}-runtime.log")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"The {kind} runtime did not start")
            time.sleep(1)
        if vision_stage:
            metadata = vision.model_record(VISION_URL, MODEL)
            if metadata["digest"] != MODEL_DIGEST:
                raise ValueError("The installed Qwen model differs from the pinned evaluated 4B build")
            # Cold file-backed tensors can load slowly after the large video
            # model exits. Warm explicitly before the bounded evidence calls.
            vision.api(VISION_URL, "/api/generate", dict(model=MODEL, stream=False, keep_alive="5m"), timeout=600)
            yield metadata
        else:
            yield None
    finally:
        if vision_stage:
            try:
                vision.api(VISION_URL, "/api/generate", dict(model=MODEL, keep_alive=0))
            except Exception:
                if process is None:
                    raise RuntimeError("Could not release the visual-check model before video generation")
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


class Reveal(BaseModel):
    stage: Literal["space", "cloud", "aerial", "approach", "portrait", "unclear"]
    terrain_clear: bool
    evidence: str


def reveal_times(count: int, near_frames: int, fps: float = 24.0, limit: int = 12) -> list[float]:
    """When to sample the cloud-to-ground reveal, in clip seconds.

    Sample every half second through the window that ends half a second before the native portrait;
    a late reveal on a short portrait job would otherwise be missed. The window can ask for more
    samples than the vision checker accepts -- 241 frames with a 73-frame portrait asks for 13 of a
    limit of 12 -- so when it does, the count is capped and the spacing widened to still cover the
    same window. Failing the whole build over the sample count is not a useful outcome.
    """
    end = max(1, count - near_frames - 1)
    slots = end // 12 + 1
    step = 12.0
    if slots > limit:
        slots, step = limit, end / limit
    return [round(i * step / fps, 3) for i in range(1, slots + 1)]


def reveal_frame_index(answers: list[dict]) -> int | None:
    """Where the cloud-to-ground reveal is established, as an index into ``answers``.

    The ground must be held rather than glimpsed, so one clear frame is not enough -- but demanding
    two CONSECUTIVE clear readings makes the whole build hostage to a single misread by the 4B
    checker, and on the first Teotihuacan take that is exactly what happened: frames at 5.8 s, 6.4 s
    and 7.0 s all show detailed ground, and the middle one was read as hazy, so a film with a clean
    reveal was held. Two qualifying frames in any run of three keeps the "held" requirement and
    survives one bad reading.
    """
    clear = [bool(a.get("terrain_clear")) and a.get("stage") in ("aerial", "approach", "portrait")
             for a in answers]
    for i in range(len(clear) - 2):
        if sum(clear[i:i + 3]) >= 2:
            return i + clear[i:i + 3].index(True)
    if len(clear) == 2 and all(clear):        # a two-sample window is still a pair
        return 0
    return None


def reveal_frame(source: Path, root: Path, near_frames: int = 49):
    cap = cv2.VideoCapture(str(source))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    times = reveal_times(count, near_frames, fps)
    evidence = vision.prepare(source, times, root / "pacing-evidence")
    answers = []
    for frame in evidence["frames"]:
        answer = chat(VISION_URL, [(Path(frame["image"]), frame["frame_id"])],
            "Classify this single film frame. Is at least half the view clearly visible detailed terrestrial ground "
            "UNDER the cloud layer, rather than a planet seen from space or a view ABOVE clouds? A blue curved "
            "Earth limb or continental-scale view is space. White cloud filling most of the lens is cloud. "
            "Aerial is resolved landscape or city beneath the clouds, approach is nearer ground, portrait is a "
            "prominent person. Set terrain_clear only for clearly visible ground under the cloud layer. "
            "Do not infer through white cloud or space haze. Describe the visible evidence.", Reveal,
            root / "pacing-observations" / f"{frame['frame_id']}.json")
        answers.append(answer)
        index = reveal_frame_index(answers)
        if index is not None:
            return evidence["frames"][index]["source_frame"]
    raise Held("The cloud-to-ground reveal was inconclusive. This take needs a pacing review.")


def sample_preview(source, at, target):
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read a preview frame")
    target.parent.mkdir(exist_ok=True, parents=True)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((960, 960))
    image.save(target, quality=88)


def review_summary(report):
    return {k: report[k] for k in ("status", "error", "checker_version", "model", "description", "answer") if k in report}


def reuse_arrival(cfg, source: Path, scene: Path, expected_hashes: dict):
    """Reuse an exact, previously reviewed photograph, then run a fresh arrival gate."""
    previous = json.loads((source / "outcome.json").read_text())
    old = json.loads((source / "scene-config.json").read_text())
    if previous.get("input_hashes") != expected_hashes or old["models"] != cfg["models"]:
        raise ValueError("The reusable arrival belongs to different inputs or models")
    if not previous.get("arrival_review", {}).get("status") == "pass":
        raise ValueError("The reusable arrival did not pass its original gate")
    if any(digest(Path(p)) not in set(expected_hashes.values()) for p in
           [old["subject"], *[r["local_path"] for r in old["references"]]]):
        raise ValueError("A reusable arrival reference no longer matches the saved inputs")
    cfg.update(subject=old["subject"], references=old["references"], seed=old["seed"])
    cfg["prompts"]["arrival"] = old["prompts"]["arrival"]
    for name in ("arrival.png", "arrival.json", "arrival.workflow.json"):
        shutil.copyfile(source / "scene" / name, scene / name)
    verify_lineage(cfg, scene, asset_plan(cfg, scene)[:1])
    return old["seed"]


def review_delivery(cfg, root, setup, folder, output, near_folder, outcome, progress):
    split = cfg["video"].get("flight_mode") == "split"
    candidate = output / ("earth_zoom.mp4" if split else "flight/earth_zoom.mp4")
    if cfg['video'].get('phase_contract'):
        from touchdown.phase_review import review_final
        result = review_final(root, cfg)
        outcome['flight_sequence_review'] = result
        if result['status'] != 'pass':
            raise Held('; '.join(result['verdict']['failures']))
    sample_preview(candidate, 0, root / "previews/orbit.jpg")
    with runtime("vision", root) as metadata:
        progress("pacing", "Checking the flight timing and ground reveal")
        if split:
            # The split flight is delivered at its own clock when it is assembled; nothing to retime.
            pacing = json.loads((output / "pacing.json").read_text())
        elif cfg["video"].get("preserve_native_portrait"):
            try:
                flight_record = json.loads((output / "flight/result.json").read_text())
                high_frame = flight_record.get("source_high_frame")
                if high_frame is None:
                    high_frame = cfg["video"]["frames"]-1-cfg["video"]["high_frame"] if cfg["video"].get("pin_high", True) else reveal_frame(candidate, root, cfg["video"]["near_frames"])
                pacing = photo_delivery.compose(output, near_folder, high_frame,
                                                target_seconds=cfg["video"].get("target_seconds"))
            except photo_delivery.JoinMismatch as exc:
                raise Held(str(exc)) from exc
        else:
            reveal = reveal_frame(candidate, root, cfg["video"]["near_frames"])
            portrait_flight.pace(output, reveal, 2.6)
            pacing = json.loads((output / "pacing.json").read_text())
        candidate = output / "earth_zoom.mp4"
        info = cv2.VideoCapture(str(candidate))
        frames = int(info.get(cv2.CAP_PROP_FRAME_COUNT)); fps = info.get(cv2.CAP_PROP_FPS)
        width, height = int(info.get(cv2.CAP_PROP_FRAME_WIDTH)), int(info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.release()
        if frames < 73 or fps != 24 or width < 1900 or height < 1000:
            raise ValueError("Unexpected delivered video dimensions or timing")
        duration = frames / fps
        near_start = pacing.get("native_start_frame", frames-cfg["video"]["near_frames"]) / fps
        # The generated flight is a 10-second budget, but the delivered length falls out of how much
        # of it the model spends near the ground: a take that reaches the cloud early delivers a
        # six-second film with a one-second approach. That is a defect, not a short film, so the
        # target is enforced here rather than assumed from the generation parameters.
        low, high = cfg["video"].get("target_seconds", (8.0, 10.0))
        outcome.update(duration_seconds=round(duration, 3), duration_target=[low, high])
        # Jamie (25 Oct): within half a second of the target either way is fine.
        if not low - .5 <= duration <= high + .5:
            raise Held(f"The film came out {duration:.2f}s, outside the {low:.0f}-{high:.0f}s target. "
                       f"The flight spent too much of its length above the ground, leaving only "
                       f"{near_start:.2f}s of approach before the portrait.")
        if cfg["video"].get("scale_anchors"):
            at = pacing["target_high_frame"] / fps
            high_reviews = []
            for i, time_seconds in enumerate([max(0, at-.25), at, at+.25]):
                path = root / f"gates/high-exit/frame-{i}.jpg"
                sample_preview(candidate, time_seconds, path)
                high_reviews.append(high_view_review(VISION_URL, path, path.with_suffix(".json")))
            outcome["cloud_exit_reviews"] = high_reviews
            if high_reviews[1]["status"] != "pass" or any(v["viewpoint"] in ("low_aerial", "ground_level") for v in high_reviews[:2]):
                raise Held("The delivered cloud exit did not establish a broad high aerial before the final descent.")
        times = {round(max(0, near_start - 1), 2), round(max(0, near_start - .5), 2),
                 round(near_start, 2), round(near_start + .5, 2), round(near_start + 1, 2), round(duration - .125, 2)}
        # A portrait that stops early would put the +0.5/+1 s samples past the end of the film.
        times = {t for t in times if t <= duration - .125}
        feet_until = split_flight.feet_visible_until(pacing) if split else None
        if feet_until is not None and setup["framing"] != "full":
            # The ground-contact checks need the shoes; once the push crops them the ending review takes over.
            times = {t for t in times if t < feet_until} or {round(max(0, near_start - .5), 2)}
        times = sorted(times)
        progress("review", "Checking the delivered approach and requested ending")
        checks = ["foot_contact", "support_context", "surface_continuity", "appearance", "scene_match"]
        evidence = vision.prepare(candidate, times, root / "gates/final/evidence", subject=folder / setup["subject"]["name"])
        report = vision.check(evidence, cfg["gate"]["description"], checks, MODEL, VISION_URL,
                              root / "gates/final/report.json", metadata=metadata, allowed_supports=cfg["gate"]["allowed_supports"])
        outcome["final_review"] = review_summary(report)
        ending_times = [round(near_start + (duration-near_start-.125)*i/4, 2) for i in range(5)]
        ending = vision.prepare(candidate, ending_times, root / "gates/ending/evidence")
        controls = chat(VISION_URL, [(Path(f["image"]), f["frame_id"]) for f in ending["frames"]],
            f"These are chronological frames from the ending. Requested action: {POSES[setup['ending']]} "
            f"Requested FINAL frame crop: {FRAMINGS[setup['framing']]}. Judge framing only from the last frame. "
            "Judge the action across the sequence. For arms-open both arms must be visibly open; a single small "
            "arm lift is insufficient. For a twirl require evidence of a complete turn. A relaxed stance does "
            "not require a measurable blink. Say unclear if evidence is insufficient. Do not certify facial sharpness.",
            EndingReview, root / "gates/ending/report.json")
        outcome["raw_ending_review"] = controls
        controls = {**controls, "ending": ending_action_review(VISION_URL, [(Path(f["image"]), f["frame_id"]) for f in ending["frames"]],
                                                                setup["ending"], root / "gates/ending/action.json")}
        controls, crop, orbit = independent_delivery_review(VISION_URL, Path(ending["frames"][-1]["image"]),
            root / "previews/orbit.jpg", controls, setup["framing"], root / "gates/ending")
        measured = split_flight.measured_crop(pacing, setup["framing"]) if split else None
        if measured:
            # The 4B model misreads crops (it called a knees-up frame "ends at feet"); face height decides.
            outcome["vision_framing_advisory"] = controls.get("framing")
            controls = {**controls, "framing": measured}
        outcome.update(ending_review=controls, final_crop_observation=crop, orbital_scale_review=orbit)
        outcome.update(duration=duration, width=width, height=height,
                       pacing=json.loads((output / "pacing.json").read_text()))
        sample_preview(candidate, max(0, near_start-.5), root / "previews/approach.jpg")
        sample_preview(candidate, duration-.125, root / "previews/portrait.jpg")
        if report["status"] != "pass" or report.get("error") or orbit["status"] != "pass" or any(v["status"] != "pass" for v in controls.values()):
            raise Held("This take needs review: one or more approach, ground-contact, framing or ending checks were fail or unclear.")
    outcome.update(status="done", checks_passed=True, message="Film ready. Please review facial sharpness and motion before sharing.")
    return candidate


def portrait_flight_takes(cfg, scene, output, outcome, progress):
    """Legacy single-generation flight (flight_mode unset), kept for existing scenes."""
    # A single generation can contain a hard cut: the first Teotihuacan flight jumped from a
    # straight-on view of the pyramid to the figure on the platform at source frame 157->158.
    # Pacing cannot repair that, and shipping it is worse than paying for another take, so the
    # flight is re-rolled once before the build gives up.
    for attempt in range(2):
        portrait_flight.run(cfg, scene, output, "flight", VIDEO_URL)
        cuts = jump_cuts(frame_differences(output / "flight/earth_zoom.mp4"))
        if not cuts:
            break
        outcome.setdefault("flight_cuts", []).append({"attempt": attempt + 1, "cuts": cuts})
        if attempt == 1:
            raise Held(f"The flight contained a hard cut on both attempts "
                       f"({len(cuts)} pair(s) far above the median difference). Pacing cannot "
                       f"repair a cut; a different seed or a different scene treatment is needed.")
        # Offset only the flight seed: cfg["seed"] pins the arrival that the lineage check
        # verifies, and moving it makes the whole run look stale.
        cfg["video"]["flight_seed_offset"] = int(cfg["video"].get("flight_seed_offset", 32)) + 500
        progress("flight", "The flight contained a hard cut; generating it again")


def build(setup_path: Path, root: Path, arrival_source: Path | None = None):
    root.mkdir(parents=True, exist_ok=True)
    setup = json.loads(setup_path.read_text())
    folder = root / "inputs"
    scene = root / "scene"
    scene.mkdir(exist_ok=True)
    output = root / "motion"
    states = {s: "pending" for s in STAGES}
    outcome = dict(status="failed", setup_id=setup["id"], checks_passed=False,
                   facial_clarity="Human review required; the small visual model does not certify sharpness.")
    candidate = None

    def progress(stage, message):
        for s, value in states.items():
            if value == "running":
                states[s] = "done"
        states[stage] = "running"
        write_json(root / "progress.json", dict(stages=states, message=message))
        print(f"[{stage}] {message}", flush=True)

    try:
        for p in [setup["subject"], *setup["references"]]:
            if Path(p["name"]).name != p["name"] or digest(folder / p["name"]) != p["sha256"]:
                raise ValueError("A saved input changed before generation")
        progress("references", "Reading the location photographs and choosing connected ground")
        with runtime("vision", root):
            brief = grounding(setup, folder, VISION_URL, root / "grounding.json")
            # Relief and settlement come from the coordinates: the photos show a courtyard, not the region.
            try:
                facts = region_facts(setup, VISION_URL, root / "region-facts.json")
            except OSError as exc:
                facts = None
                outcome["region_facts_error"] = str(exc)
            if facts:
                brief.update(regional_relief=facts["relief"], local_relief=facts["local_relief"],
                             nearest_settlement=facts["nearest_settlement"],
                             coastal=facts.get("sea_fraction", 0) >= .03,  # Edinburgh's Firth is 7% of its tile
                             local_coastal=facts.get("local_sea_fraction", 0) >= .03)
        if len(brief["selected_references"]) < 2:
            raise Held("Fewer than two location photos were useful for this setting. Add clearer references and save a revised setup.")
        seed = secrets.randbelow(2**31)
        outcome.update(seed=seed, brief=brief, input_hashes={p["name"]: p["sha256"] for p in [setup["subject"], *setup["references"]]})
        cfg = scene_config(setup, brief, folder, root, seed)
        if arrival_source:
            seed = reuse_arrival(cfg, arrival_source, scene, outcome["input_hashes"])
            outcome.update(seed=seed, reused_arrival=str(arrival_source), reused_arrival_sha256=digest(scene / "arrival.png"))
        outcome["selected_reference_hashes"] = [p["sha256"] for p in cfg["references"]]
        outcome["references"] = [{k: p[k] for k in ("source", "title", "artist", "license", "license_url", "url", "sha256", "source_sha256") if k in p}
                                  for p in cfg["references"]]
        write_json(root / "scene-config.json", cfg)
        # Retain source credits beside the render, including which references were actually used.
        write_json(root / "attribution.json", cfg["references"])
        for attempt in range(2):
            cfg["seed"] = seed + attempt * 100
            progress("arrival", f"Creating the photographed landing scene (take {attempt + 1}/2)")
            if attempt or not arrival_source:
                with runtime("comfy", root):
                    generate_asset(COMFY_URL, asset_plan(cfg, scene)[0], scene)
            target = root / "previews/arrival.jpg"
            target.parent.mkdir(exist_ok=True)
            with Image.open(scene / "arrival.png") as image:
                image.thumbnail((960, 960)); image.convert("RGB").save(target, quality=88)
            progress("arrival_check", "Checking visible shoes, connected ground, appearance and the setting")
            gate_folder = root / f"gates/arrival-{attempt}"
            with runtime("vision", root) as metadata:
                evidence = vision.prepare(scene / "arrival.png", [0.0], gate_folder / "evidence", subject=folder / setup["subject"]["name"])
                report = vision.check(evidence, cfg["gate"]["description"], cfg["gate"]["checks"], MODEL, VISION_URL,
                                      gate_folder / "report.json", metadata=metadata, allowed_supports=cfg["gate"]["allowed_supports"])
            outcome["arrival_review"] = review_summary(report)
            if report["status"] == "pass" and not report.get("error"):
                cfg["gate"]["arrival_report"] = str(gate_folder / "report.json")
                outcome["seed"] = cfg["seed"]
                break
        else:
            raise Held("The landing scene did not establish all required reference and ground-contact checks after two takes.")
        write_json(root / "scene-config.json", cfg)
        if cfg["video"].get("scale_anchors"):
            progress("arrival", "Creating broad high-aerial and orbital photographs from the location context")
            plan = asset_plan(cfg, scene)
            with runtime("comfy", root):
                for i, spec in enumerate(plan[1:], start=1):
                    verify_lineage(cfg, scene, plan[:i])
                    generate_asset(COMFY_URL, spec, scene)
            progress("arrival_check", "Checking the high view and orbital scale before flight generation")
            with runtime("vision", root):
                scale_checks = dict(high=high_view_review(VISION_URL, scene / "high.png", root / "gates/scales/high.json"),
                                    orbit=orbital_view_review(VISION_URL, scene / "orbit.png", root / "gates/scales/orbit.json"))
            outcome["scale_reference_reviews"] = scale_checks
            if any(v["status"] != "pass" for v in scale_checks.values()):
                raise Held("A high-aerial or orbital photograph did not establish the required scale. Video generation is held.")
        verify_lineage(cfg, scene, asset_plan(cfg, scene))
        split = cfg["video"].get("flight_mode") == "split"
        if split:
            outcome["prompt_relay"] = split_flight.enable_prompt_relay()
            from touchdown.flight_runtime import capture_runtime
            capture_runtime(root)
        with runtime("comfy", root):
            progress("portrait", "Creating the requested moving portrait and final framing")
            portrait_flight.run(cfg, scene, output, "near", VIDEO_URL)
            progress("flight", "Flying out through the location and clouds to orbit")
            if split:
                split_flight.parent(cfg, scene, output, VIDEO_URL, COMFY_URL, output / "near")
            else:
                portrait_flight_takes(cfg, scene, output, outcome, progress)
        if split:
            from touchdown.phase_review import review_opening
            result = review_opening(root, cfg)
            outcome['parent_phase_review'] = result
            if result['status'] != 'pass':
                from touchdown.phase_review import parent_preview
                candidate = parent_preview(root)
                outcome['video_kind'] = 'parent_diagnostic'
                raise Held('; '.join(result['failures']))
            # The descent's anchor is repaired from the parent flight; check it before animating it.
            anchor = output / "flight/anchor.png"
            with runtime("vision", root):
                review = approach_view_review(VISION_URL, anchor, root / "gates/scales/anchor.json")
            outcome["anchor_review"] = review
            write_json(output / "flight/anchor-review.json", dict(review, sha256=digest(anchor)))
            if review["status"] != "pass":
                raise Held("The descent's aerial anchor was not a clear aerial view without a prominent person.")
            with runtime("comfy", root):
                progress("flight", "Descending from the high aerial to the portrait")
                # One seed. A cut is reported and holds the take; it is not re-rolled.
                pacing = split_flight.descent(cfg, scene, output, VIDEO_URL, output / "near")
            outcome["flight_motion"] = pacing["motion"]
            if pacing["generated_cuts"]:
                raise Held(f"The flight contained {len(pacing['generated_cuts'])} hard cut(s); see pacing.json.")
        candidate = review_delivery(cfg, root, setup, folder, output, output / "near", outcome, progress)
        states["review"] = "done"
    except Held as exc:
        for stage in states:
            if states[stage] == "running":
                states[stage] = "held"
        outcome.update(status="held", error=str(exc), message=str(exc))
    except (Exception, SystemExit) as exc:
        for stage in states:
            if states[stage] == "running":
                states[stage] = "failed"
        outcome.update(status="failed", error=f"{type(exc).__name__}: {exc}", message="Generation stopped. Your setup and completed stages are preserved.")
        raise
    finally:
        delivered = output / "earth_zoom.mp4"
        if delivered.is_file():
            candidate = delivered
        elif (output / "flight/earth_zoom.mp4").is_file():
            candidate = output / "flight/earth_zoom.mp4"
        if candidate is not None and candidate.is_file():
            outcome.update(video=str(candidate.relative_to(root)), video_sha256=digest(candidate))
        write_json(root / "outcome.json", outcome)
        write_json(root / "progress.json", dict(stages=states, message=outcome.get("message", "Stopped")))
    return outcome


def preflight():
    import comfy_sdk  # noqa: F401
    if not shutil.which("ffmpeg") or not shutil.which("ollama"):
        raise ValueError("Install ffmpeg and the local Ollama runtime first")
    cmd = shlex.split(os.environ.get("TOUCHDOWN_COMFY_SERVER_CMD", ""))
    if not cmd or not Path(cmd[0]).is_file() or not Path(os.environ.get("TOUCHDOWN_COMFY_SERVER_CWD", "")).is_dir():
        raise ValueError("Configure TOUCHDOWN_COMFY_SERVER_CMD and TOUCHDOWN_COMFY_SERVER_CWD for the photographic worker")
    model_root = Path(os.environ.get("TOUCHDOWN_VISION_MODELS", str(REPO / "models/vision-checker/ollama")))
    manifest = model_root / "manifests/registry.ollama.ai/library/qwen3.5/4b"
    if not manifest.is_file() or digest(manifest) != MODEL_DIGEST:
        raise ValueError("The evaluated local Qwen3.5 4B model must already be installed")
    print("Photographic worker preflight passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-arrival", type=Path, help="Operator trial: reuse an exact checked arrival with identical saved input hashes")
    args = parser.parse_args()
    if args.check:
        preflight()
        return
    if not args.input or not args.output:
        parser.error("Supply --input and --output")
    result = build(args.input.resolve(), args.output.resolve(), args.reuse_arrival.resolve() if args.reuse_arrival else None)
    result["surface_stability"] = surface_report(args.output.resolve())
    if result["status"] != "done":
        raise SystemExit(3 if result["status"] == "held" else 1)


if __name__ == "__main__":
    main()
