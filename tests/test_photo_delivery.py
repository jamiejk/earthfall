"""Photographic scale ancestry and whole-frame native-video delivery."""
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from touchdown.authored_scene import digest
from touchdown.photo_delivery import JoinMismatch, compose, hermite_indices, match_metrics
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown import studio_brief


def test_scale_views_do_not_carry_portrait_or_local_monument_pixels_into_orbit(tmp_path):
    cfg = dict(subject="identity.jpg", references=[dict(local_path="place-a.jpg"), dict(local_path="place-b.jpg")],
               models={}, seed=1, video=dict(scale_anchors=True),
               prompts=dict(arrival="person at landmark", high="regional aerial", orbit="planetary photograph"))
    plan = {p["name"]: p for p in asset_plan(cfg, tmp_path)}
    assert plan["arrival"]["references"] == ["place-a.jpg", "place-b.jpg", "identity.jpg"]
    assert plan["high"]["references"] == []  # regional description instead of close-up camera geometry
    assert plan["orbit"]["references"] == []
    cfg["video"] = {}
    assert [p["name"] for p in asset_plan(cfg, tmp_path)] == ["arrival"]  # accepted old scenes remain single-photo


def test_fast_descent_retiming_keeps_endpoints_and_never_blends_or_repeats():
    upper = hermite_indices(119, 53, 119/53, 49/36)
    lower = hermite_indices(49, 36, 49/36, 1.)
    mapping = upper[:-1] + [119+i for i in lower[:-1]] + list(range(168, 241))
    assert mapping[53] == 119 and mapping[89] == 168
    assert mapping[0] == 0 and mapping[-1] == 240
    assert all(a < b for a, b in zip(mapping, mapping[1:]))
    assert mapping[89:] == list(range(168, 241))  # native portrait moves at its own speed


def test_changed_photo_derived_context_invalidates_the_regional_image(tmp_path):
    subject = tmp_path / "subject.jpg"; subject.write_bytes(b"portrait")
    context = tmp_path / "grounding.json"; context.write_text('{"regional_kind":"farmland"}')
    cfg = dict(subject=str(subject), references=[], regional_context=str(context), models={}, seed=1,
               video=dict(scale_anchors=True), prompts=dict(arrival="portrait", high="fields", orbit="earth"))
    spec = asset_plan(cfg, tmp_path)[1]
    image = tmp_path / "high.png"; image.write_bytes(b"regional photo")
    (tmp_path / "high.json").write_text(json.dumps(dict(spec=spec, output_sha256=digest(image),
          references=[], context_parent_sha256=digest(context))))
    assert verify_lineage(cfg, tmp_path, [spec])["generated"]["high"]["context_parent_sha256"] == digest(context)
    context.write_text('{"regional_kind":"urban"}')
    with pytest.raises(ValueError, match="Changed photographic context"):
        verify_lineage(cfg, tmp_path, [spec])


@pytest.mark.parametrize("extent,expected", [("over_5_percent", "unclear"), ("unclear", "unclear"),
                                            ("1_to_5_percent", "pass"), ("none_resolved", "pass")])
def test_a_high_aerial_label_cannot_override_a_large_foreground_monument(tmp_path, monkeypatch, extent, expected):
    observed = dict(viewpoint="high_aerial", broad_region_visible=True, prominent_person_visible=False,
                    landmark_dominates=False, largest_structure_width=extent,
                    sky_or_horizon_visible=True, evidence="Observed landscape extent")
    monkeypatch.setattr(studio_brief, "chat", lambda *args: observed)
    assert studio_brief.high_view_review("unused", tmp_path / "image.jpg", tmp_path / "report.json")["status"] == expected


def texture(seed=1):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (192, 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_seam_comparison_rejects_an_unrelated_scene_or_camera_jump():
    image = texture()
    assert match_metrics(image, image)["acceptable"]
    assert not match_metrics(image, texture(2))["acceptable"]
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 15], [0, 1, 0]]), (320, 192))
    assert not match_metrics(image, shifted)["acceptable"]


def encode(path, frames):
    size = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", size,
                                "-r", "24", "-i", "pipe:0", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0


@pytest.fixture
def sources(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    flight = tmp_path / "flight"; near = tmp_path / "near"
    flight.mkdir(); near.mkdir()
    image = texture()
    native = [np.roll(image, i//4, axis=1) for i in range(17)]
    repainted = [image]*61 + [np.full_like(image, (180, 20, 120))]*16
    encode(flight / "earth_zoom.mp4", repainted)
    encode(near / "near.mp4", native)
    for folder, video in [(flight, flight / "earth_zoom.mp4"), (near, near / "near.mp4")]:
        graph = folder / "graph.json"; graph.write_text('{"1":{"class_type":"LoadImage"}}')
        record = dict(video=str(video), sha256=digest(video), workflow=str(graph), workflow_sha256=digest(graph))
        if folder == flight:
            record.update(descent=str(video), descent_sha256=digest(video))
        (folder / "result.json").write_text(json.dumps(record))
    (flight / "provenance.json").write_text(json.dumps(dict(parent=dict(near_video_sha256=digest(near / "near.mp4")))))
    return tmp_path, near


def test_delivered_ending_uses_native_frames_instead_of_the_repainted_figure(sources):
    output, near = sources
    record = compose(output, near, 30)
    assert not record["crossfade"] and not record["interpolation"] and not record["face_composite"]
    assert record["native_frames"] == 17 and record["native_start_frame"] == 60
    assert record["output_frames"] == 77
    cap = cv2.VideoCapture(str(output / "earth_zoom.mp4")); cap.set(1, 76); ok, delivered = cap.read(); cap.release(); assert ok
    cap = cv2.VideoCapture(str(near / "near.mp4")); cap.set(1, 16); ok, native = cap.read(); cap.release(); assert ok
    assert float(np.abs(delivered.astype(float)-native).mean()) < 12
    assert float(np.abs(delivered.astype(float)-np.array((180, 20, 120))).mean()) > 35


def test_a_target_length_controls_the_delivered_duration(tmp_path):
    """A ten-second target was unreachable while the head and descent were fixed at 2.2 s and 1.5 s."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    flight = tmp_path / "flight"; near = tmp_path / "near"
    flight.mkdir(); near.mkdir()
    image = texture()
    # A ten-second flight against a three-second portrait, as the studio pipeline generates: the
    # flight's tail IS the portrait, so the join is exact and only the pacing is under test.
    portrait = [np.roll(image, i, axis=1) for i in range(73)]
    encode(near / "near.mp4", portrait)
    encode(flight / "earth_zoom.mp4", [np.roll(image, 100 + i//4, axis=1) for i in range(241 - 73)] + portrait)
    for folder, video in [(flight, flight / "earth_zoom.mp4"), (near, near / "near.mp4")]:
        graph = folder / "graph.json"; graph.write_text('{"1":{"class_type":"LoadImage"}}')
        record = dict(video=str(video), sha256=digest(video), workflow=str(graph), workflow_sha256=digest(graph))
        if folder == flight:
            record.update(descent=str(video), descent_sha256=digest(video))
        (folder / "result.json").write_text(json.dumps(record))
    (flight / "provenance.json").write_text(json.dumps(dict(parent=dict(near_video_sha256=digest(near / "near.mp4")))))

    fixed = compose(tmp_path, near, 125)
    assert fixed["output_frames"] < 8*24          # the old fixed timings cannot reach the target

    target = compose(tmp_path, near, 125, target_seconds=(8.0, 10.0))
    assert 8*24 <= target["output_frames"] <= 10*24, target["output_frames"]
    assert target["native_frames"] == 73


def test_whole_flight_can_resize_to_the_native_portrait_without_resizing_the_portrait(sources):
    output, near = sources
    cap = cv2.VideoCapture(str(near / "near.mp4")); frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 384), interpolation=cv2.INTER_LANCZOS4))
    cap.release(); encode(near / "near.mp4", frames)
    record = json.loads((near / "result.json").read_text()); record["sha256"] = digest(near / "near.mp4")
    (near / "result.json").write_text(json.dumps(record))
    (output / "flight/provenance.json").write_text(json.dumps(dict(parent=dict(near_video_sha256=record["sha256"]))))
    delivery = compose(output, near, 30)
    assert delivery["flight_size"] == [320, 192] and delivery["delivered_size"] == [640, 384]
    assert delivery["flight_resize"] == "whole-frame Lanczos" and not delivery["face_composite"]
    cap = cv2.VideoCapture(str(output / "earth_zoom.mp4"));cap.set(1, 76);ok, delivered=cap.read();cap.release();assert ok
    assert delivered.shape == frames[-1].shape
    assert float(np.abs(delivered.astype(float)-frames[-1]).mean()) < 12


@pytest.mark.parametrize("tamper", ["video", "workflow", "parent"])
def test_changed_video_workflow_or_wrong_native_parent_cannot_be_spliced(sources, tamper):
    output, near = sources
    if tamper == "video":
        (near / "near.mp4").write_bytes(b"changed")
    elif tamper == "workflow":
        (near / "graph.json").write_text("{}")
    else:
        (output / "flight/provenance.json").write_text('{"parent":{"near_video_sha256":"wrong"}}')
    with pytest.raises(ValueError):
        compose(output, near, 30)
    assert not (output / "earth_zoom.mp4").exists()


def test_a_failed_seam_is_not_hidden_with_a_dissolve(sources):
    output, near = sources
    encode(near / "near.mp4", [texture(2)]*17)
    record = json.loads((near / "result.json").read_text()); record["sha256"] = digest(near / "near.mp4")
    (near / "result.json").write_text(json.dumps(record))
    (output / "flight/provenance.json").write_text(json.dumps(dict(parent=dict(near_video_sha256=record["sha256"]))))
    with pytest.raises(JoinMismatch):
        compose(output, near, 30)
    assert not (output / "earth_zoom.mp4").exists()
