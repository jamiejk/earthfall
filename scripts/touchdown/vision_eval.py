"""Compare local visual checkers against a small, explicitly labelled development set."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.vision_check import api, check, model_record, prepare


def score(rows: list[dict]) -> dict:
    counts = dict(labelled_checks=0, exact_matches=0, known_failures=0, detected_failures=0,
                  false_approvals=0, unresolved_failures=0, known_passes=0, accepted_passes=0,
                  false_rejections=0, unresolved_passes=0, expected_unclear=0,
                  correct_abstentions=0, unsupported_approvals=0, unsupported_rejections=0, errors=0)
    mismatches = []
    for row in rows:
        if row.get("error"):
            counts["errors"] += 1
        for name, expected in row["expected"].items():
            actual = row["actual"].get(name, "error" if row.get("error") else "unclear")
            counts["labelled_checks"] += 1
            counts["exact_matches"] += actual == expected
            if expected == "fail":
                counts["known_failures"] += 1
                counts["detected_failures"] += actual == "fail"
                counts["false_approvals"] += actual == "pass"
                counts["unresolved_failures"] += actual not in {"pass", "fail"}
            elif expected == "pass":
                counts["known_passes"] += 1
                counts["accepted_passes"] += actual == "pass"
                counts["false_rejections"] += actual == "fail"
                counts["unresolved_passes"] += actual not in {"pass", "fail"}
            else:
                counts["expected_unclear"] += 1
                counts["correct_abstentions"] += actual == "unclear"
                counts["unsupported_approvals"] += actual == "pass"
                counts["unsupported_rejections"] += actual == "fail"
            if actual != expected:
                mismatches.append(dict(case=row["case"], check=name, expected=expected, actual=actual))
    return dict(counts=counts, mismatches=mismatches)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["qwen3.5:2b", "qwen3.5:4b"])
    p.add_argument("--only", nargs="+")
    p.add_argument("--url", default="http://127.0.0.1:11439")
    a = p.parse_args()
    manifest = json.loads(a.cases.read_text())
    root = (a.cases.resolve().parent / manifest["root"]).resolve()
    selected = [c for c in manifest["cases"] if not a.only or c["id"] in a.only]
    if not selected or (a.only and set(a.only)-{c["id"] for c in selected}):
        p.error("Unknown or empty case selection")
    if len({c["id"] for c in selected}) != len(selected):
        p.error("Duplicate case IDs")
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "labels.json").write_text(json.dumps(manifest, indent=2))
    evidence = {}
    for case in selected:
        if not re.fullmatch(r"[a-z0-9-]+", case["id"]):
            p.error("Case IDs must be lowercase slugs")
        evidence[case["id"]] = prepare(root/case["source"], case["times"], output/"evidence"/case["id"],
                                        case.get("support_box"), root/case["subject"] if case.get("subject") else None)
    comparison = dict(diagnostic_only=True, note=manifest["limitations"], models={})
    for model in a.models:
        meta = model_record(a.url, model)
        rows = []
        try:
            for case in selected:
                path = output/re.sub(r"[^a-zA-Z0-9.-]", "-", model)/f"{case['id']}.json"
                report = check(evidence[case["id"]], case["description"], list(case["expected"]),
                               model, a.url, path, metadata=meta, allowed_supports=case.get("allowed_supports"))
                actual = {k: v["status"] for k, v in report.get("answer", {}).get("checks", {}).items()}
                raw_actual = {k: v["status"] for k, v in report.get("raw_answer", {}).get("checks", {}).items()}
                row = dict(case=case["id"], group=case["group"], expected=case["expected"], actual=actual,
                           raw_actual=raw_actual, adjustments=report.get("adjustments", []),
                           error=report.get("error"), wall_seconds=report["wall_seconds"], report=str(path))
                rows.append(row)
                print(model, case["id"], json.dumps(actual), report.get("error") or "", flush=True)
                raw_rows = [{**r, "actual": r["raw_actual"]} for r in rows]
                comparison["models"][model] = dict(metadata=meta, rows=rows, raw_model=score(raw_rows), **score(rows))
                (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        finally:
            api(a.url, "/api/generate", dict(model=model, keep_alive=0))
    print(json.dumps({m: r["counts"] for m, r in comparison["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
