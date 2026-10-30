"""Timing contracts and guide-safe temporal attention; no model downloads/GPU."""
import pytest
from touchdown.relay_timing import schedule, query_positions


def test_reverse_schedule_covers_the_exact_source_and_preserves_the_agreed_delivery_beats():
    s = schedule(dict(place="Stonehenge", prompts={}))
    assert s["delivery_frames"] == 192 and s["source_frames"] == 193
    assert [(b["start_frame"], b["end_frame"]) for b in s["forward_beats"]] == [(0,29),(29,58),(58,108),(108,156),(156,192)]
    assert [b["name"] for b in s["generation_beats"]] == ["settle","approach","high_aerial","clouds","orbit"]
    assert s["generation_beats"][0]["start_frame"] == 0
    assert s["generation_beats"][-1]["end_frame"] == 193
    assert "Stonehenge" not in s["global_prompt"]
    assert "two kilometres" in s["generation_beats"][1]["prompt"]
    for beat in s["forward_beats"]:
        for output_frame in range(beat["start_frame"], beat["end_frame"]):
            source_frame = 192-output_frame
            actual = next(b for b in s["generation_beats"] if b["start_frame"] <= source_frame < b["end_frame"])
            assert actual["name"] == beat["name"]


def test_appending_references_never_rescales_the_real_video_timeline():
    video = query_positions(25, 4, 100, [])
    guided = query_positions(25, 4, 144, video[:10]+[192])
    assert guided[:25] == video
    assert guided[25:] == video[:10]+[192]


def test_control_and_relay_use_identical_text_and_backend_with_only_weighting_switched():
    import copy
    from touchdown.relay_timing import wire_relay
    original={"305":dict(class_type="CLIPTextEncode",inputs=dict(text="old")),
              "307":dict(class_type="LTXVConditioning",inputs=dict(positive=["305",0])),
              "393":dict(class_type="CFGGuider",inputs=dict(model=["300",0]))}
    timing=schedule(dict(place="Stonehenge",prompts={}))
    control=wire_relay(copy.deepcopy(original),timing,"audit.json",[192],enabled=False)
    relay=wire_relay(copy.deepcopy(original),timing,"audit.json",[192],enabled=True)
    assert not control["380"]["inputs"]["enabled"]
    control["380"]["inputs"]["enabled"]=True
    assert control==relay


@pytest.mark.parametrize("queries", [103, 108, 140])
def test_unaccounted_tokens_hold_the_take_instead_of_silently_shifting_prompts(queries):
    with pytest.raises(ValueError):
        query_positions(25, 4, queries, [192])


def test_temporal_attention_assigns_orbit_and_portrait_to_their_times_even_with_guides():
    torch = pytest.importorskip("torch")
    from touchdown.relay_timing import temporal_mask
    s = schedule(dict(place="Stonehenge", prompts={}))
    positions = query_positions(25, 1, 27, [0,192])
    mask = temporal_mask(positions, 1, [(i+1,i+2) for i in range(5)], s["generation_beats"], 6,
                         .1, torch.float32, "cpu")
    assert torch.all(mask[:,0] == 0)  # shared photographic appearance
    assert mask[0,1] == 0 and mask[0,5] < -100  # retreat begins on portrait
    assert mask[24,5] == 0 and mask[24,1] < -100  # retreat ends in orbit
    assert torch.equal(mask[0], mask[25])
    assert mask[26,5] == 0


def test_attention_adapter_preserves_head_layout_and_honours_the_temporal_mask():
    torch=pytest.importorskip("torch")
    from touchdown.relay_timing import direct_attention
    torch.manual_seed(42)
    q=torch.randn(1,3,8);k=torch.randn(1,5,8);v=torch.randn(1,5,8)
    mask=torch.zeros(3,5);mask[0,2:]=-1000
    actual=direct_attention(None,q,k,v,2,mask=mask)
    expected=[]
    for h in range(2):
        a=q[:,:,h*4:(h+1)*4];b=k[:,:,h*4:(h+1)*4];c=v[:,:,h*4:(h+1)*4]
        expected.append(torch.softmax(a@b.transpose(1,2)/2+mask,dim=-1)@c)
    assert torch.allclose(actual,torch.cat(expected,dim=-1),atol=1e-6)


@pytest.mark.parametrize("prefix,handles", [(25,0),(73,0),(25,48)])
def test_export_is_exactly_eight_seconds_and_uses_unretimed_native_portrait(tmp_path, monkeypatch, prefix, handles):
    import json
    import subprocess
    import cv2
    import numpy as np
    from touchdown import relay_experiment as experiment
    from touchdown.authored_scene import digest
    def video(path, count, colour):
        enc=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24",
            "-s","96x64","-r","24","-i","pipe:0","-c:v","libx264","-crf","12",str(path)],stdin=subprocess.PIPE)
        for i in range(count):
            frame=np.full((64,96,3),colour,dtype=np.uint8)
            frame[:, :32] = i
            enc.stdin.write(frame.tobytes())
        enc.stdin.close();assert enc.wait()==0
        workflow=path.with_suffix(".json");workflow.write_text("{}")
        return dict(video=str(path),sha256=digest(path),workflow=str(workflow),workflow_sha256=digest(workflow))
    flight=video(tmp_path/"source.mp4",193+handles,(30,40,50))
    near=video(tmp_path/"native.mp4",73,(190,160,130))
    (tmp_path/"result.json").write_text(json.dumps(flight))
    monkeypatch.setattr(experiment,"match_metrics",lambda *args: dict(acceptable=True))
    result=experiment.assemble(tmp_path,near,"arrival-hash",prefix,handles)
    assert result["output_frames"]==192 and result["duration"]==8
    assert result["source_speed"]==result["native_speed"]==1
    assert result["discarded_orbital_source_frames"]==list(range(193,193+handles))
    assert result["flight_source_mapping"]==list(range(192,0,-1))
    start=193-prefix
    assert result["native_start_frame"]==start and result["native_frames"]==prefix-1
    cap=cv2.VideoCapture(result["video"])
    for index,expected in [(0,192),(start,0),(191,prefix-2)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();assert ok
        assert abs(float(im[20:40,8:24].mean())-expected)<4
        if index>=start:
            assert im[20:40,50:70,0].mean()>180  # native pixels, not the repainted flight
    cap.release()


def test_orbital_handles_extend_only_the_discarded_end_of_generation():
    cfg=dict(place="Stonehenge",prompts={})
    a,b=schedule(cfg),schedule(cfg,48)
    assert b["source_frames"]==241 and b["delivery_frames"]==192
    assert b["forward_beats"]==a["forward_beats"]
    assert b["generation_beats"][:-1]==a["generation_beats"][:-1]
    assert b["generation_beats"][-1]["start_frame"]==164
    assert b["generation_beats"][-1]["end_frame"]==241
    for frame in range(1,193):
        assert next(x["name"] for x in a["generation_beats"] if x["start_frame"]<=frame<x["end_frame"]) == next(x["name"] for x in b["generation_beats"] if x["start_frame"]<=frame<x["end_frame"])
