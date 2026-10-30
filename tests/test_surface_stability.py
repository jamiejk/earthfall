"""The surface-stability gate has to mean the same thing on a different scene and a different clip.

Each test pins one property the module exists for:

  * a rigid subject must measure at the floor, not above it;
  * a sharper clip of the same motion must not score worse than a softer one;
  * a subject whose surface is repainted must be caught;
  * a held frame must be reported.

The fixtures pan by whole pixels. A half-pixel pan resamples every frame and an at-Nyquist synthetic
texture loses most of its detail to it, which pushes the floor to 1.0 where real footage measures
0.47 -- a fixture artefact that would hide every effect being tested. Real clips are measured with
the sub-pixel default, where that loss is part of the floor the control reports.
"""
import cv2
import numpy as np

from touchdown.surface_stability import (assess, jump_cuts, measure, repeats, rigid_control, roi_box)

FPS = 24
DX, DY = 4.0, 2.0


def texture(seed=42, shape=(368, 640)):
    """Photo-like: structure at a range of scales, not a noise field."""
    rng = np.random.default_rng(seed)
    h, w = shape
    n = rng.normal(size=(h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1e-6
    img = np.real(np.fft.ifft2(n / f ** 1.2))
    img = (img - img.min()) / (img.max() - img.min()) * 190 + 30
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def panned(frame, n, dx=DX, dy=DY):
    """The same picture, moved by a known camera pan. Nothing in it can warp."""
    h, w = frame.shape[:2]
    return [grey(cv2.warpAffine(frame, np.float32([[1, 0, -i * dx], [0, 1, -i * dy]]),
                                (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
            for i in range(n)]


def repainted(frame, n, dx=DX, dy=DY, seed=5):
    """The camera pan, carrying the picture, with the surface detail drawn fresh every frame.

    This is what LTX does to masonry: position and silhouette hold, the surface is re-invented.
    Smooth deformation would be the wrong fixture -- dense flow can explain any smooth motion, so
    a bending stone scores low by construction.
    """
    h, w = frame.shape[:2]
    carried_source = cv2.GaussianBlur(frame, (0, 0), 2.0)
    amplitude = float(np.std(grey(frame) - cv2.GaussianBlur(grey(frame), (0, 0), 2.0)))
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = np.float32([[1, 0, -i * dx], [0, 1, -i * dy]])
        carried = grey(cv2.warpAffine(carried_source, m, (w, h), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE))
        noise = cv2.GaussianBlur(rng.normal(0, amplitude, (h, w)).astype(np.float32), (0, 0), 0.8)
        out.append(np.clip(carried + noise, 0, 255))
    return out


def stats_of(frames, **kw):
    return measure(((frames[i], frames[i + 1]) for i in range(len(frames) - 1)), **kw)


def write_clip(path, frames):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def test_a_rigid_subject_measures_at_its_own_floor():
    frames = panned(texture(), 25)
    stats = stats_of(frames)
    floor = rigid_control(frames[0], dx=DX, dy=DY, pairs=24)
    assert stats["redraw"] < 0.25
    assert abs(stats["redraw"] - floor["redraw"]) < 0.05


def test_camera_alignment_actually_removes_the_camera_motion():
    """The sign-error guard. Warping the previous frame the wrong way leaves as much difference as
    no warp at all, and every other assertion here would still look plausible."""
    frames = panned(texture(), 25)
    stats = stats_of(frames)
    assert stats["gain"] > 0.9
    assert stats["warble"] < stats["raw"] * 0.2


def test_a_repainted_surface_measures_far_above_the_floor():
    frame = texture()
    rigid = stats_of(panned(frame, 25))
    repaint = stats_of(repainted(frame, 25))
    assert repaint["redraw"] > rigid["redraw"] + 0.30


def test_sharper_clip_of_the_same_motion_raises_warble_but_not_redraw():
    """The bug that made the refine look guilty: a crisper picture scores worse on absolute warble."""
    frame = texture()
    soft = panned(frame, 25)
    sharp = panned(cv2.addWeighted(frame, 1.7, cv2.GaussianBlur(frame, (0, 0), 2.0), -0.7, 0), 25)
    a, b = stats_of(soft), stats_of(sharp)
    assert b["detail"] > a["detail"] * 1.15
    assert b["warble"] > a["warble"] * 1.15
    assert abs(b["redraw"] - a["redraw"]) < 0.06


def test_a_held_frame_is_reported_as_a_repeat():
    frames = panned(texture(), 25)
    frames.insert(12, frames[11].copy())
    diffs = [float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames))]
    found = repeats(diffs)
    assert [r["pair"] for r in found] == [12]
    assert found[0]["share_of_median"] == 0.0


def test_an_isolated_jump_is_a_cut():
    diffs = [5.0] * 10 + [60.0] + [5.0] * 10
    found = jump_cuts(diffs)
    assert [c["pair"] for c in found] == [11]
    assert found[0]["times_median"] == 12.0


def test_a_run_of_fast_frames_is_not_a_cut():
    """Genuinely fast camera motion raises neighbouring pairs together and must not be reported."""
    diffs = [5.0] * 10 + [40.0, 45.0, 50.0, 44.0] + [5.0] * 10
    assert jump_cuts(diffs) == []


def test_the_real_cut_in_the_teotihuacan_flight_is_detected():
    """Source frames 157->158 measured 8.3x the median with ordinary neighbours."""
    diffs = [6.0] * 157 + [54.4] + [6.0] * 80
    found = jump_cuts(diffs)
    assert [c["pair"] for c in found] == [158]


def test_a_residual_below_the_floor_cannot_pass():
    """Haze, cloud and static sections measure below their own floor and must not be certified."""
    from touchdown.surface_stability import _verdict
    stats = dict(redraw=0.09, redraw_above_floor=-0.14, gain=0.23, repeats=[], cuts=[],
                 control={"redraw": 0.235})
    verdict = _verdict(stats)
    assert verdict["status"] == "unverified"
    assert "below" in verdict["reasons"][0].lower()


def test_roi_defaults_to_the_whole_frame_and_accepts_fractions():
    assert roi_box((100, 200), None) == (0, 100, 0, 200)
    assert roi_box((100, 200), (0.25, 0.5, 0.1, 0.9)) == (25, 50, 20, 180)


def test_assess_passes_a_rigid_clip_and_reports_its_floor(tmp_path):
    clip = write_clip(tmp_path / "rigid.mp4", panned(texture(), 30))
    report = assess(clip, control=(DX, DY))
    assert report["verdict"]["status"] == "pass", report["verdict"]
    assert report["control"]["redraw"] >= 0          # the floor is always measured, never assumed
    assert report["redraw_above_floor"] == round(report["redraw"] - report["control"]["redraw"], 4)


def test_assess_fails_a_repainted_clip_and_says_why(tmp_path):
    clip = write_clip(tmp_path / "repaint.mp4", repainted(texture(), 30))
    report = assess(clip, control=(DX, DY))
    assert report["verdict"]["status"] == "fail"
    assert any("redraw" in reason for reason in report["verdict"]["reasons"])


def test_assess_window_samples_only_the_requested_seconds(tmp_path):
    clip = write_clip(tmp_path / "rigid.mp4", panned(texture(), 48))
    whole = assess(clip, control=(DX, DY))
    windowed = assess(clip, window=(0.5, 1.0), control=(DX, DY))
    assert windowed["pairs"] < whole["pairs"]
    assert 9 <= windowed["pairs"] <= 14, windowed["pairs"]
    assert windowed["window"] == [0.5, 1.0]
