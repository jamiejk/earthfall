"""Calibrate the motion checks on clips whose verdict we already know (Astra's review: fade_check flagged parts of the
accepted cut as DISSOLVE?, so it cannot rank seeds until it is tested on known flights and known dissolves).

Each labelled range is a judgement made frame by frame by a person (high-merge and the ladders are Jamie's and our
confirmed dissolves; r3-cloud and the current cut are confirmed flights). For every 12-frame window the script
records fade_check's features (flow, difference, share of the difference the flow explains, texture) plus the
similarity-fit camera motion, then reports how fade_check's current rule scores against the labels and which single
threshold on each feature separates the two classes best.

    uv run --extra touchdown python scripts/touchdown/calibrate_checks.py --root /mnt/project-files --out calib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fade_check import measure

# (clip under --root, [(first frame, last frame, label)]); label: fly | dissolve | white (skipped)
LABELS = [
    ("fakeout/high-merge/high_merge_joined.mp4", [(0, 119, "dissolve"), (120, 280, "fly"), (288, 318, "dissolve")]),
    ("fakeout/r4-chain/fakeout_chain_s20261009_joined.mp4", [(0, 240, "dissolve")]),
    ("fakeout/ladder/fakeout_ladder_s20261009_joined.mp4", [(0, 328, "dissolve")]),
    ("fakeout/r3-cloud/fakeout_ltx_first-last_s20261009_ltx25_00004_.mp4", [(0, 28, "fly"), (29, 44, "white"), (45, 240, "fly")]),
    ("fakeout/nadir/dada893-swoops/refine/swoop-fix/cloud/a7wb_cloud2_s4_long.mp4",
     [(0, 44, "fly"), (45, 58, "white"), (59, 279, "fly")]),
]
WINDOW = 12


def windows(rows: np.ndarray, ranges) -> list[dict]:
    out = []
    for start in range(0, len(rows) - WINDOW + 1, WINDOW):
        end = start + WINDOW
        label = next((lab for a, b, lab in ranges if a <= start and end - 1 <= b), None)
        if label in (None, "white"):
            continue  # straddles a label boundary, or a whiteout
        flow, diff, explained, texture = rows[start:end].mean(axis=0)
        out.append({"start": start, "label": label, "flow": float(flow), "diff": float(diff),
                    "explained": float(explained), "texture": float(texture)})
    return out


def rule(w: dict) -> str:
    """fade_check's verdict today."""
    if w["texture"] < 100:
        return "whiteout"
    if w["diff"] > 1.5 and w["explained"] < 0.3:
        return "dissolve"
    return "still" if w["diff"] <= 1.5 else "fly"


def best_threshold(ws: list[dict], feature: str) -> tuple[float, float, str]:
    """The threshold and direction on one feature with the best balanced accuracy for fly vs dissolve."""
    vals = sorted({w[feature] for w in ws})
    best = (0.0, 0.0, ">")
    for t in vals:
        for direction in (">", "<"):
            fly_ok = np.mean([(w[feature] > t) == (direction == ">") for w in ws if w["label"] == "fly"])
            dis_ok = np.mean([(w[feature] > t) != (direction == ">") for w in ws if w["label"] == "dissolve"])
            score = (fly_ok + dis_ok) / 2
            if score > best[1]:
                best = (float(t), float(score), direction)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/mnt/project-files"))
    p.add_argument("--out", type=Path, default=Path("calibration.json"))
    a = p.parse_args()
    ws = []
    for clip, ranges in LABELS:
        rows = np.array(measure(str(a.root / clip)))
        for w in windows(rows, ranges):
            w["clip"] = clip
            ws.append(w)
        print(f"measured {clip}", flush=True)
    confusion: dict[str, dict[str, int]] = {}
    for w in ws:
        confusion.setdefault(w["label"], {}).setdefault(rule(w), 0)
        confusion[w["label"]][rule(w)] += 1
    thresholds = {f: best_threshold(ws, f) for f in ("flow", "diff", "explained")}
    a.out.write_text(json.dumps({"windows": ws, "fade_check_rule": confusion, "thresholds": thresholds}, indent=1))
    print("fade_check's rule against the labels (label -> verdict counts):")
    for label, counts in confusion.items():
        print(f"  {label}: {counts}")
    print("best single-feature thresholds (threshold, balanced accuracy, fly is above/below):")
    for f, (t, s, d) in thresholds.items():
        print(f"  {f}: {t:.3f}  {s:.2f}  fly {d}")
    wrong = [w for w in ws if (w["label"] == "fly") != (rule(w) in ("fly", "still"))]
    print(f"windows fade_check gets wrong: {len(wrong)} of {len(ws)}")
    for w in wrong:
        print(f"  {w['clip'].split('/')[-1]} f{w['start']}-{w['start'] + WINDOW - 1}: label {w['label']}, "
              f"verdict {rule(w)} (flow {w['flow']:.2f}, diff {w['diff']:.2f}, explained {w['explained']:.2f})")


if __name__ == "__main__":
    main()
