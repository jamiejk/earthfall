"""Objective postflight checks for an unattended Touchdown render.

These checks catch broken media and abrupt frame changes. They do not certify scene
geography, person identity, road geometry, or whether a camera move looks natural.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from . import video as V


def living_face_turn(video: Path, first: int, last: int) -> dict:
    """Flag a sustained profile turn after a front-facing living-end handoff.

    YuNet's nose offset from the midpoint of the eyes is scale independent. A
    missing face also counts as a profile sample, but only after the first few
    sampled frames establish that a frontal face was visible to begin with.
    """
    from .face_refine import DETECTOR, _detector, _face

    if not DETECTOR.is_file():
        return {"assessed": False, "reason": "YuNet model unavailable", "warnings": []}
    detector = _detector((V.W, V.H))
    samples = []
    for index, frame in enumerate(V.frames(video)):
        if index >= last:
            break
        if index < first or (index - first) % 4:
            continue
        face = _face(detector, frame)
        offset = None
        if face is not None:
            points = face[4:14].reshape(5, 2)
            eye_span = abs(float(points[1, 0] - points[0, 0]))
            if eye_span >= 3:
                offset = abs(float(points[2, 0] - (points[0, 0] + points[1, 0]) / 2)) / eye_span
        samples.append({"frame": index, "nose_offset": round(offset, 2) if offset is not None else None})
    if len(samples) < 8:
        return {"assessed": False, "reason": "living end too short", "samples": samples, "warnings": []}
    frontal_start = sum(item["nose_offset"] is not None and item["nose_offset"] < 0.2
                        for item in samples[:5]) >= 4
    profile = [item["nose_offset"] is None or item["nose_offset"] > 0.5 for item in samples[5:]]
    sustained = any(all(profile[i:i + 3]) for i in range(len(profile) - 2))
    warning = "living-end frontal face becomes profile or undetectable" if frontal_start and sustained else None
    return {"assessed": frontal_start, "samples": samples, "warnings": [warning] if warning else []}


def inspect(video: Path, end_candidate: Path, segments: Path) -> dict:
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate",
         "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout)
    streams = probe["streams"]
    picture = next((s for s in streams if s["codec_type"] == "video"), None)
    if not picture or (picture["width"], picture["height"], picture["r_frame_rate"]) != (V.W, V.H, f"{V.FPS}/1"):
        raise ValueError(f"unexpected output video format: {picture}")
    if not any(s["codec_type"] == "audio" for s in streams):
        raise ValueError("final video has no audio stream")
    bounds = json.loads(segments.read_text())
    images = V.small(video)
    if len(images) != bounds["end"][1]:
        raise ValueError(f"final video has {len(images)} frames, segments expect {bounds['end'][1]}")
    changes = V.diff(images)
    joins = {}
    for name in ("cloud", "dive", "swoop", "end"):
        at = bounds[name][0]
        neighborhood = np.concatenate([changes[max(0, at - 9):max(0, at - 2)],
                                       changes[min(len(changes), at + 1):min(len(changes), at + 8)]])
        baseline = float(np.median(neighborhood)) if len(neighborhood) else 0.0
        jump = float(changes[at - 1])
        # a held frame at the join reads as a stop just as a spike reads as a cut (17 Oct: the cloud -> dive
        # hand-off repeated the opaque frame in both cuts and the audit called it clean)
        around = changes[max(0, at - 3):min(len(changes), at + 3)]
        held = bool(len(around) and float(around.min()) < 0.8 and baseline > 4.0)
        joins[name] = {"frame": at, "change": round(jump, 2), "nearby_median": round(baseline, 2),
                       "spike": bool(jump > max(30.0, 3.5 * baseline)), "held_frame": held}
    end_changes = V.diff(V.small(end_candidate))
    middle = end_changes[4:-4]
    end_baseline = float(np.median(middle)) if len(middle) else 0.0
    cuts = [int(i + 4) for i, value in enumerate(middle) if value > max(22.0, 4 * end_baseline)]
    face_turn = living_face_turn(video, bounds["end"][0], bounds["end"][1])
    living = changes[bounds["end"][0]:bounds["end"][1] - 1]
    living_motion = float(np.median(living)) if len(living) else 0.0
    return {"video": str(video), "frames": len(images), "seconds": round(len(images) / V.FPS, 2),
            "format": f"{V.W}x{V.H}@{V.FPS}", "audio": True, "joins": joins,
            "end_candidate_cut_frames": cuts, "living_face_turn": face_turn,
            "living_end_motion": round(living_motion, 2),
            "not_assessed": ["geographic continuity", "object and person identity", "natural camera motion",
                             "perceived audio quality"],
            "objective_warnings": [*(f"{name} join spike" for name, x in joins.items() if x["spike"]),
                                   *(f"{name} join held frame" for name, x in joins.items() if x["held_frame"]),
                                   *(["living end nearly static"] if living_motion < 1.0 else []),
                                   *(f"end candidate abrupt change at frame {i}" for i in cuts),
                                   *face_turn["warnings"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shot_root", type=Path)
    args = parser.parse_args()
    root = args.shot_root
    records = {}
    for stage in ("end", "sound"):
        found = list((root / stage).glob("*/done.json"))
        if not found:
            raise FileNotFoundError(f"no completed {stage} stage under {root}")
        done = max(found, key=lambda p: p.stat().st_mtime)
        records[stage] = (done.parent, json.loads(done.read_text()))
    end_dir, end_record = records["end"]
    sound_dir, sound_record = records["sound"]
    report = inspect(sound_dir / sound_record["output"],
                     end_dir / f"s{end_record['facts']['seed']}" / "output.mp4",
                     sound_dir / "segments.json" if (sound_dir / "segments.json").exists()
                     else end_dir / "segments.json")
    dest = root / "audit.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
