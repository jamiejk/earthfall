import json

import pytest

from touchdown.authored_scene import digest
from touchdown.photo_gate import require_arrival_gate


def setup_gate(tmp_path):
    scene=tmp_path/"scene";scene.mkdir()
    arrival=scene/"arrival.png";arrival.write_bytes(b"arrival pixels")
    image=scene/"sample.jpg";image.write_bytes(b"inspected pixels")
    path=scene/"report.json"
    policy=dict(arrival_report=str(path),model="qwen3.5:4b",model_digest="pinned-build",
                checker_version="visual-check-4",description="A woman on ordinary grass.",
                checks=["foot_contact","support_context","appearance","scene_match"],allowed_supports=["grass"])
    report=dict(status="pass",checker_version=policy["checker_version"],
                model=dict(name=policy["model"],digest=policy["model_digest"]),description=policy["description"],
                allowed_supports=policy["allowed_supports"],
                evidence=dict(source=str(arrival),source_sha256=digest(arrival),
                              inputs=[dict(path=str(image),sha256=digest(image))]),
                answer=dict(checks={k:dict(status="pass") for k in policy["checks"]}))
    path.write_text(json.dumps(report))
    return dict(gate=policy),scene,path,report


def test_pass_is_bound_to_the_current_image_and_report(tmp_path):
    cfg,scene,path,report=setup_gate(tmp_path)
    result=require_arrival_gate(cfg,scene,tmp_path/"video")
    assert result["status"]=="pass" and result["report_sha256"]==digest(path)
    assert (tmp_path/"video/arrival-gate-verification.json").is_file()


@pytest.mark.parametrize("status",["fail","unclear"])
def test_nonpassing_gate_prevents_video_eligibility(tmp_path,status):
    cfg,scene,path,report=setup_gate(tmp_path)
    report["status"]=status;path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="held"):
        require_arrival_gate(cfg,scene,tmp_path/"video")
    assert not (tmp_path/"video").exists()


def test_changing_arrival_invalidates_an_old_pass(tmp_path):
    cfg,scene,path,report=setup_gate(tmp_path)
    (scene/"arrival.png").write_bytes(b"new image")
    with pytest.raises(ValueError,match="changed"):
        require_arrival_gate(cfg,scene,tmp_path/"video")


@pytest.mark.parametrize("change",["model","checker","brief","support","missing_check","unclear_check","sample"])
def test_a_pass_cannot_be_reused_with_different_evidence_or_policy(tmp_path,change):
    cfg,scene,path,report=setup_gate(tmp_path)
    if change=="model":report["model"]["digest"]="other-build"
    elif change=="checker":report["checker_version"]="older-checker"
    elif change=="brief":report["description"]="A person on a roof."
    elif change=="support":report["allowed_supports"]=["building_roof"]
    elif change=="missing_check":report["answer"]["checks"].pop("foot_contact")
    elif change=="unclear_check":report["answer"]["checks"]["foot_contact"]["status"]="unclear"
    else:(scene/"sample.jpg").write_bytes(b"changed sample")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        require_arrival_gate(cfg,scene,tmp_path/"video")
