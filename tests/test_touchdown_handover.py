import numpy as np
import pytest

from touchdown.birds import flock
from touchdown.build import guide_height, handover_frame
from touchdown.proc_cloud import fbm


def test_handover_frame_inverts_the_guide_height_curve():
    assert handover_frame(501) == 1
    assert guide_height(1) == pytest.approx(501)
    assert guide_height(81) == pytest.approx(2)
    for h in (300, 175, 80, 30):
        assert abs(guide_height(handover_frame(h)) - h) / h < 0.05


def test_handover_height_is_bounded():
    with pytest.raises(ValueError):
        handover_frame(600)
    with pytest.raises(ValueError):
        handover_frame(1)


def test_cloud_noise_tiles_without_a_seam():
    n = fbm(384, seed=3, base=12)
    assert 0 <= n.min() and n.max() <= 1
    # the wrap-around step is no bigger than an ordinary neighbouring step
    inner = np.abs(np.diff(n, axis=1)).mean()
    edge = np.abs(n[:, 0] - n[:, -1]).mean()
    assert edge < 3 * inner


def test_flock_is_deterministic_and_crosses_the_frame():
    a, b = flock(9, seed=4, size=0.07), flock(9, seed=4, size=0.07)
    assert a == b
    for bird in a:
        assert (bird["x0"] < 0) != (bird["x1"] < 0) or (bird["x0"] > 1) != (bird["x1"] > 1)


def test_first_last_tail_hold_is_cut_not_failed(tmp_path, monkeypatch):
    import fade_check
    from PIL import Image
    from touchdown import terrain_quality

    moving, still = [8.0, 10.0, 0.6, 300.0], [0.3, 1.5, 0.4, 400.0]
    monkeypatch.setattr(fade_check, "measure", lambda _: [moving] * 44 + [still] * 28)
    frame = np.full((544, 960, 3), 120, np.uint8)
    monkeypatch.setattr(terrain_quality.V, "frame", lambda clip, i: frame)
    end = tmp_path / "end.png"
    Image.fromarray(frame).save(end)
    strict = terrain_quality.assess(tmp_path / "clip.mp4", end, expected_frames=73)
    assert any("dissolve or still" in w for w in strict["warnings"])
    loose = terrain_quality.assess(tmp_path / "clip.mp4", end, expected_frames=73, tail_ok=True)
    assert loose["settled_at"] == 50 and not any("dissolve" in w for w in loose["warnings"])


def test_natural_surface_accepts_red_sand_that_the_neutral_caps_reject(tmp_path):
    import cv2
    from touchdown.terrain_route import _landing_candidates

    np.save(tmp_path / "height.npy", np.zeros((401, 401), np.float32))
    sand = np.zeros((2048, 2048, 3), np.uint8) + np.uint8([90, 125, 195])  # BGR red desert sand
    cv2.imwrite(str(tmp_path / "naip.png"), sand)
    with pytest.raises(ValueError, match="saturation"):
        _landing_candidates(tmp_path, tmp_path / "naip.png", 20)
    assert _landing_candidates(tmp_path, tmp_path / "naip.png", 20, natural=True)[0]
