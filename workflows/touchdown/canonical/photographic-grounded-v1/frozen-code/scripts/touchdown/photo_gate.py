"""Bind a photographic arrival's video eligibility to its local visual report."""
from __future__ import annotations

import json
from pathlib import Path

from touchdown.authored_scene import digest


def require_arrival_gate(cfg: dict, scene: Path, output: Path) -> dict | None:
    policy = cfg.get("gate")
    if not policy:
        return None
    path = Path(policy["arrival_report"])
    report = json.loads(path.read_text())
    if report.get("status") != "pass" or report.get("error"):
        raise ValueError("Arrival gate is not pass; video generation is held")
    if report.get("checker_version") != policy["checker_version"]:
        raise ValueError("Arrival gate used a different checker version")
    if not policy["checks"]:
        raise ValueError("Arrival gate has no required checks")
    model = report.get("model", {})
    if model.get("name") != policy["model"] or model.get("digest") != policy["model_digest"]:
        raise ValueError("Arrival gate used a different model or model build")
    if report.get("description") != policy["description"]:
        raise ValueError("Arrival gate used a different scene brief")
    if set(report.get("allowed_supports") or []) != set(policy["allowed_supports"]):
        raise ValueError("Arrival gate used different allowed support types")
    source = scene / "arrival.png"
    evidence = report.get("evidence", {})
    if (Path(evidence.get("source", "")).resolve() != source.resolve()
            or evidence.get("source_sha256") != digest(source)):
        raise ValueError("Arrival changed after its gate check")
    inputs = evidence.get("inputs", [])
    if not inputs or any(digest(Path(r["path"])) != r["sha256"] for r in inputs):
        raise ValueError("Arrival gate evidence changed or is missing")
    findings = report.get("answer", {}).get("checks", {})
    if any(findings.get(k, {}).get("status") != "pass" for k in policy["checks"]):
        raise ValueError("A required arrival check is missing or not pass")
    decision = dict(status="pass", report=str(path.resolve()), report_sha256=digest(path),
                    source_sha256=digest(source), model=model,
                    required_checks=policy["checks"], allowed_supports=policy["allowed_supports"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrival-gate-verification.json").write_text(json.dumps(decision, indent=2))
    return decision
