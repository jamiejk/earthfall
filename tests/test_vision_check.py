"""Protect the observational checker against unsupported automatic approvals."""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from touchdown import vision_check as v
from touchdown.vision_eval import score


def evidence(tmp_path, frames=1):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (320, 180), "grey").save(source)
    result = v.prepare(source, [0.0], tmp_path/"evidence")
    if frames == 2:
        result["frames"].append({**result["frames"][0], "frame_id": "F02", "source_frame": 1})
    return result


def response(ids, checks, *, feet="not_visible", framing="waist_up", support="ground_paving", person="clear"):
    return dict(observations=[dict(frame_id=i, person_visibility=person, feet_visibility=feet,
                body_framing=framing, support_type=support, evidence="Visible image evidence.") for i in ids],
                checks={k: dict(status=s, frame_ids=ids, reason="Model's visual judgment.") for k,s in checks.items()})


@pytest.mark.parametrize("url", ["https://127.0.0.1:11439", "http://example.com", "http://localhost.evil.test",
                                 "http://user@localhost:11439", "http://localhost/path"])
def test_remote_or_ambiguous_servers_are_rejected_before_network(url):
    with pytest.raises(ValueError, match="loopback"):
        v.api(url, "/api/chat", {})


@pytest.mark.parametrize("raw_status", ["pass", "fail"])
def test_hidden_feet_do_not_establish_contact_or_failure(tmp_path, raw_status):
    e = evidence(tmp_path)
    result, notes = v.validated_answer(response(["F01"], {"foot_contact": raw_status}), e, ["foot_contact"])
    assert result["checks"]["foot_contact"]["status"] == "unclear"
    assert notes


def test_visible_feet_in_waist_up_observation_cannot_pass(tmp_path):
    e = evidence(tmp_path)
    result,_ = v.validated_answer(response(["F01"], {"foot_contact":"pass"}, feet="visible"), e, ["foot_contact"])
    assert result["checks"]["foot_contact"]["status"] == "unclear"


def test_subject_reference_cannot_supply_feet_missing_from_the_scene(tmp_path):
    e = evidence(tmp_path)
    raw = response(["F01"], {"foot_contact":"pass"}, feet="visible", framing="full_body")
    probes = {"F01": {"observation": {"main_person_present":True, "feet_visible":False,
                                     "support_type":"ground_paving", "evidence":"Feet are cropped out."}}}
    observed = v.independent_answer(raw, probes)
    checked,_ = v.validated_answer(observed,e,["foot_contact"])
    assert checked["checks"]["foot_contact"]["status"] == "unclear"
    assert raw["observations"][0]["feet_visibility"] == "visible"


def test_observed_roof_conflicts_with_ground_request_but_accepts_deliberate_roof(tmp_path):
    e = evidence(tmp_path)
    raw = response(["F01"], {"support_context":"pass"}, support="building_roof")
    ground,_ = v.validated_answer(raw, e, ["support_context"], ["ground_paving"])
    roof,_ = v.validated_answer(raw, e, ["support_context"], ["building_roof"])
    assert ground["checks"]["support_context"]["status"] == "fail"
    assert roof["checks"]["support_context"]["status"] == "pass"
    assert raw["checks"]["support_context"]["status"] == "pass"  # preserve the raw model answer


def test_no_person_is_insufficient_support_evidence(tmp_path):
    e = evidence(tmp_path)
    result,_ = v.validated_answer(response(["F01"], {"support_context":"fail"}, person="absent"),
                                 e, ["support_context"], ["ground_paving"])
    assert result["checks"]["support_context"]["status"] == "unclear"


@pytest.mark.parametrize("raw_status", ["pass", "fail"])
def test_single_frame_cannot_establish_temporal_continuity(tmp_path, raw_status):
    e = evidence(tmp_path)
    result,_ = v.validated_answer(response(["F01"], {"surface_continuity":raw_status}), e, ["surface_continuity"])
    assert result["checks"]["surface_continuity"]["status"] == "unclear"


def test_invented_evidence_ids_cannot_support_a_finding(tmp_path):
    e = evidence(tmp_path)
    raw = response(["F01"], {"foot_contact":"pass"})
    raw["checks"]["foot_contact"]["frame_ids"] = ["F99"]
    with pytest.raises(ValueError, match="invented"):
        v.validated_answer(raw, e, ["foot_contact"])


def test_duplicate_observations_are_invalid(tmp_path):
    e = evidence(tmp_path, 2)
    raw = response(["F01","F01"], {"surface_continuity":"pass"})
    with pytest.raises(ValueError, match="repeated"):
        v.validated_answer(raw, e, ["surface_continuity"])


def test_truncated_model_response_is_saved_as_unclear_not_success(tmp_path, monkeypatch):
    e = evidence(tmp_path)
    monkeypatch.setattr(v, "api", lambda *a,**kw: {"done":True,"done_reason":"length", "message":{"content":"{"}})
    out = tmp_path/"report.json"
    r = v.check(e,"Natural setting",["foot_contact"],"local:small","http://localhost",out,metadata={"digest":"a"})
    assert r["status"] == "unclear" and r["error"]
    assert json.loads(out.read_text())["raw_response"]["done_reason"] == "length"


def test_changed_evidence_is_rejected_before_inference(tmp_path, monkeypatch):
    e = evidence(tmp_path)
    Path(e["inputs"][0]["path"]).write_bytes(b"changed")
    monkeypatch.setattr(v,"api",lambda *a,**kw: pytest.fail("Should not run inference"))
    with pytest.raises(ValueError,match="pixels changed"):
        v.check(e,"Natural setting",["foot_contact"],"local:small","http://localhost",tmp_path/"r.json",metadata={"digest":"a"})


def test_invalid_crop_or_still_times_are_rejected(tmp_path):
    e = evidence(tmp_path);source=Path(e["source"])
    with pytest.raises(ValueError,match="crop"):
        v.prepare(source,[0.0],tmp_path/"bad",[.8,.2,.1,.9])
    with pytest.raises(ValueError,match="still"):
        v.prepare(source,[0.0,1.0],tmp_path/"bad")


def test_false_approvals_and_insufficient_evidence_are_scored_separately():
    result=score([dict(case="bad",expected={"support_context":"fail"},actual={"support_context":"pass"}),
                  dict(case="hidden",expected={"foot_contact":"unclear"},actual={"foot_contact":"pass"}),
                  dict(case="error",error="timeout",expected={"foot_contact":"pass"},actual={})])
    assert result["counts"]["false_approvals"] == 1
    assert result["counts"]["unsupported_approvals"] == 1
    assert result["counts"]["errors"] == 1
    assert result["counts"]["unresolved_passes"] == 1


@pytest.mark.parametrize("status,code",[("pass",0),("fail",1),("unclear",3)])
def test_cli_status_and_model_release(tmp_path,monkeypatch,status,code):
    monkeypatch.setattr(sys,"argv",["vision_check","--source",str(tmp_path/"input.jpg"),
        "--times","0","--description","Test","--checks","foot_contact","--output",str(tmp_path/"out")])
    monkeypatch.setattr(v,"model_record",lambda *a: {"name":"local:small","digest":"a"})
    monkeypatch.setattr(v,"prepare",lambda *a: {})
    monkeypatch.setattr(v,"check",lambda *a,**kw: {"status":status,"wall_seconds":0})
    calls=[]
    monkeypatch.setattr(v,"api",lambda *a,**kw: calls.append(a))
    with pytest.raises(SystemExit) as exc:
        v.main()
    assert exc.value.code==code
    assert calls[-1][1]=="/api/generate" and calls[-1][2]["keep_alive"]==0


def test_operational_error_is_not_counted_as_a_correct_abstention():
    result=score([dict(case="timeout",expected={"foot_contact":"unclear"},actual={},error="timeout")])
    assert result["counts"]["correct_abstentions"]==0
    assert result["counts"]["exact_matches"]==0


def test_cli_input_failure_is_distinct_from_a_visual_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,"argv",["vision_check","--source",str(tmp_path/"missing.jpg"),
        "--times","0","--description","Test","--checks","foot_contact","--output",str(tmp_path/"out")])
    monkeypatch.setattr(v,"model_record",lambda *a: {"name":"local:small","digest":"a"})
    def broken(*args):
        raise FileNotFoundError("Missing source")
    monkeypatch.setattr(v,"prepare",broken)
    monkeypatch.setattr(v,"api",lambda *a,**kw: {})
    with pytest.raises(SystemExit) as exc:
        v.main()
    assert exc.value.code==2
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert report["status"]=="unclear" and "FileNotFoundError" in report["error"]


def test_unvalidated_clarity_approval_is_withheld(tmp_path):
    e=evidence(tmp_path)
    raw=response(["F01"],{"figure_clarity":"pass"})
    result,notes=v.adjudicate(raw,e,["figure_clarity"],None,{})
    assert result["checks"]["figure_clarity"]["status"]=="unclear"
    assert raw["checks"]["figure_clarity"]["status"]=="pass"
    assert "figure_clarity" not in v.DEFAULT_CHECKS
    assert notes


def test_elevated_to_ground_switch_is_a_continuity_failure(tmp_path):
    e=evidence(tmp_path,2)
    raw=response(["F01","F02"],{"surface_continuity":"pass"})
    probes={i:{"observation":dict(main_person_present=True,feet_visible=True,support_type=s,evidence="Surface visible.")}
            for i,s in [("F01","building_roof"),("F02","ground_paving")]}
    result,_=v.adjudicate(raw,e,["surface_continuity"],None,probes)
    assert result["checks"]["surface_continuity"]["status"]=="fail"


def test_failed_frame_probe_prevents_geometry_approval(tmp_path):
    e=evidence(tmp_path)
    raw=response(["F01"],{"foot_contact":"pass"},feet="visible",framing="full_body")
    result,_=v.adjudicate(raw,e,["foot_contact"],None,{"F01":{"error":"timeout"}})
    assert result["checks"]["foot_contact"]["status"]=="unclear"
