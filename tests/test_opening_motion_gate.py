"""A stationary feature field must fail; motion alone must never auto-pass."""
import cv2
import numpy as np
from touchdown.opening_motion_gate import compare, check


def surface():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (368, 640), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_stationary_surface_with_changing_cloud_region():
    first = surface()
    later = first.copy()
    later[:100] = 255
    assert compare(first, later)['stationary']


def test_expansion_is_not_stationary():
    first = surface()
    matrix = cv2.getRotationMatrix2D((320, 184), 0, 1.3)
    later = cv2.warpAffine(first, matrix, (640, 368))
    assert not compare(first, later)['stationary']


def test_no_features_is_unclear():
    blank = np.zeros((368, 640), np.uint8)
    result = compare(blank, blank)
    assert not result['usable'] and not result['stationary']


def test_negative_check_never_passes(tmp_path):
    video = tmp_path/'static.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 24, (640, 368))
    image = cv2.cvtColor(surface(), cv2.COLOR_GRAY2BGR)
    for _ in range(13):
        writer.write(image)
    writer.release()
    result, record = check(video, tmp_path/'review')
    assert result['status'] == 'fail'
    assert result['auto_pass'] is False
    assert record.is_file()


def test_final_review_holds_static_film_before_vision_can_pass(tmp_path, monkeypatch):
    from touchdown import canonical_review, studio_build
    from touchdown.canonical_build import sha
    video = tmp_path/'static.avi'
    # Final review receives an MP4 in production; the decoder uses the file's
    # contents, so keep this test independent of optional H.264 encoders.
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 24, (640, 368))
    image = cv2.cvtColor(surface(), cv2.COLOR_GRAY2BGR)
    for _ in range(13):
        writer.write(image)
    writer.release()

    def unexpected_vision(*args, **kwargs):
        raise AssertionError('A stationary opening must hold before Qwen review')

    monkeypatch.setattr(studio_build, 'runtime', unexpected_vision)
    result = canonical_review.review(tmp_path, 'final', [{'path':video.name}], 'Forward flight')
    assert result['continue_'] is False
    assert result['record_sha256'] == sha(tmp_path/result['record'])
