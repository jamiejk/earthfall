"""The generic-orbit opening's automatic join, zoom measurement and proxy checks, on small synthetic frames."""
import cv2
import numpy as np
import pytest

from touchdown import generic_opening as g


def sky_over_cloud(sky_rows, size=(184, 320)):
    """A frame of bright grey cloud with a band of blue sky across its top."""
    frame = np.full((*size, 3), 235, np.uint8)
    frame[:sky_rows] = (200, 140, 60)                    # BGR blue sky
    return frame


def flight(sky_by_frame):
    return [sky_over_cloud(rows) for rows in sky_by_frame]


def test_join_is_the_densest_white_frame_of_the_cloud_passage():
    frames = flight([60] * 20 + [10] * 5 + [0] * 165)     # sky shrinks, then full cloud from frame 25
    rng = np.random.default_rng(2)
    for k in range(25, 60):                               # thin, streaky cloud except a smooth bright bank at 29
        if k != 29:
            frames[k] = (frames[k].astype(np.int16) - rng.integers(0, 40, frames[k].shape)).clip(0, 255).astype(np.uint8)
    join, stats = g.detect_join(frames)
    assert join == 29 and stats[join]['texture'] < stats[25]['texture'] and stats[24]['sky'] > 0


def test_join_needs_the_next_eight_frames_to_stay_cloud():
    frames = flight([60] * 20 + [0] * 4 + [60] + [0] * 165)   # a sky frame inside any prefix starting before 25
    assert g.detect_join(frames)[0] >= 25


def test_join_keeps_the_film_near_eight_and_a_half_seconds():
    frames = flight([60] * 20 + [0] * 170)
    join, _ = g.detect_join(frames)
    seconds = (g.GENERATED - g.PREFIX + len(frames) - join) / 24
    assert g.FILM_SECONDS[0] <= seconds <= g.FILM_SECONDS[1]


def test_a_short_flight_still_joins_inside_its_cloud():
    """A 174-frame flight (Colosseum) cannot reach 8.1 s with any cloud join; it keeps its own length, not a hold."""
    frames = flight([60] * 30 + [0] * 20 + [60] * 124)
    join, _ = g.detect_join(frames)
    assert 30 <= join <= 41


def test_no_full_cloud_passage_holds_the_job():
    with pytest.raises(g.Held, match='full-frame cloud'):
        g.detect_join(flight([60] * 190))


def test_zoom_rate_measures_a_synthetic_zoom():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((736, 1280)) * 255).astype(np.uint8), (0, 0), 6)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    frames = []
    for k in range(6):
        m = cv2.getRotationMatrix2D((640, 368), 0, 1.1 ** k)
        frames.append(cv2.cvtColor(cv2.warpAffine(base, m, (1280, 736), borderMode=cv2.BORDER_REFLECT), cv2.COLOR_GRAY2BGR))
    rate, scales = g.zoom_rate(frames, 0)
    assert rate == pytest.approx(np.log(1.1), abs=.01) and len(scales) >= 3


def test_proxy_checks_flag_flat_white_frames_and_jumps():
    rng = np.random.default_rng(1)
    textured = [(rng.random((736, 1280, 3)) * 60 + 100).astype(np.uint8) for _ in range(48)]
    white = np.full((736, 1280, 3), 250, np.uint8)
    c = g.checks(textured[:10] + [white] + textured[10:])     # generation order; the white frame is play 38 of 0..48
    assert c['flat_white_play_frames'] == [38] and c['max_diff_last16'] > c['diff_gate']
    steady = [np.full((736, 1280, 3), 120 + k % 3, np.uint8) for k in range(49)]
    c = g.checks(steady)
    assert not c['flat_white_play_frames'] and c['max_diff_last16'] < c['diff_gate'] == g.MAX_DIFF


def test_diff_gate_follows_a_fast_flight():
    """A flight whose own pinned frames step by ~12 allows the join to step as much (x1.25), not a fixed 10.5."""
    frames = [np.full((736, 1280, 3), 60 + 12 * (k % 2), np.uint8) for k in range(g.PREFIX)]
    frames += [np.full((736, 1280, 3), 60, np.uint8)] * (g.GENERATED - g.PREFIX)
    c = g.checks(frames)
    assert c['diff_gate'] == pytest.approx(1.25 * 12, abs=.1)


def test_the_orbit_stays_visible_through_play_24():
    """The real proxy at play 0, 12 and 24 still shows ground or limb; the cloud closes only after the orbit."""
    orbit, still = cv2.imread(str(g.ASSETS / 'generic-orbit.png')), cv2.imread(str(g.ASSETS / 'cumulus.png'))
    pinned = [np.full((g.HEIGHT, g.WIDTH, 3), 225, np.uint8)] * g.PREFIX
    images, record = g.proxy_frames(orbit, still, pinned, g.read(g.ASSETS / 'limb-fit.json'), .1, plays={0, 12, 24})
    play = g.GENERATED - g.PREFIX
    ground = [g.ground_visible(images[g.GENERATED - 1 - p]) for p in (0, 12, 24)]
    assert min(ground) > g.GROUND_MIN and ground[0] > ground[2]
    assert record['deck_scale_per_play'][:g.ORBIT] == [round(float(np.exp(g.ORBIT_RATE)), 4)] * g.ORBIT and play == 40


def test_schedule_holds_the_orbit_then_reaches_the_flight_rate_smoothly():
    log_distance, rate = g.schedule(.105)
    assert len(log_distance) == g.GENERATED - g.PREFIX + 1
    assert np.allclose(rate[:g.ORBIT], g.ORBIT_RATE) and np.allclose(rate[g.DIVE:], .105)
    assert np.all(np.diff(rate) >= 0)                     # never slows down


def test_cached_stills_and_limb_fit_ship_with_the_repo():
    for name in ('generic-orbit.png', 'cumulus.png', 'generic-orbit.json', 'cumulus.json', 'limb-fit.json'):
        assert (g.ASSETS / name).is_file()
    assert cv2.imread(str(g.ASSETS / 'generic-orbit.png')).shape == (g.HEIGHT, g.WIDTH, 3)
    fit = g.read(g.ASSETS / 'limb-fit.json')
    assert 0 < fit['altitude_radii'] < .3 and fit['limb_rms_px'] < 10


def test_camera_mode_starts_near_three_percent_accelerates_and_meets_the_flight_rate():
    log_distance, rate = g.camera_schedule(.063)
    assert rate[0] == pytest.approx(g.CAMERA_START, abs=1e-3) and rate[-1] == pytest.approx(.063, abs=2e-3)
    assert np.all(np.diff(rate[:g.CAMERA_PEAK_BY]) >= 0) and rate.max() > 3 * g.CAMERA_START
    assert np.exp(log_distance[-1]) == pytest.approx(g.CAMERA_TOTAL, rel=.01)


def test_ending_hold_continues_the_native_portrait_after_the_delivered_end(tmp_path):
    """The frames after native_last_frame, up to ENDING_HOLD, resized to the film; none when the clip is used up."""
    video = tmp_path / 'native.mp4'
    g.encode([np.full((64, 96, 3), 4 * k, np.uint8) for k in range(40)], video, crf=0, size=(96, 64))
    pacing = dict(native_portrait=str(video), native_first_frame=4,
                  ending=dict(native_last_frame=10, native_available=40), portrait_sharpness_ramp=dict(sigma=0., frames=12))
    indices, images = g.ending_hold(pacing, (48, 32), hold=8)
    assert indices == list(range(11, 19)) and images[0].shape == (32, 48, 3)
    assert abs(images[0].mean() - 44) <= 5 and abs(images[-1].mean() - 72) <= 5   # yuv420p rounding
    assert len(g.ending_hold(pacing, (48, 32), hold=100)[0]) == 29          # capped by the clip
    assert g.ending_hold(dict(pacing, ending=dict(native_last_frame=39, native_available=40)), (48, 32)) == ([], [])


def test_the_ending_hold_counts_towards_the_preferred_film_length():
    frames = flight([60] * 20 + [0] * 170)
    join, _ = g.detect_join(frames, 24)
    seconds = (g.GENERATED - g.PREFIX + len(frames) + 24 - join) / 24
    assert g.FILM_SECONDS[0] <= seconds <= g.FILM_SECONDS[1]


def test_the_seam_cuts_to_the_flight_frame_most_like_the_openings_last_frame():
    """The opening's frame 9 resembles flight frame 29 more than the planned join 36; cut there, keeping frame 9."""
    flight = [np.full((184, 320, 3), 150 + 2 * k, np.uint8) for k in range(50)]   # all cloud-grey, brightening
    generated = [np.full((184, 320, 3), v, np.uint8) for v in [0] * 9 + [208, 230, 240]]
    g_frame, j, diff = g.best_seam(generated, flight, 9, 36)
    assert (g_frame, j) == (9, 29) and diff < 1


def test_the_slow_run_in_to_the_portrait_drops_alternate_still_frames(tmp_path):
    """Steady motion, then a near-stop in the last 20 frames before the portrait: every other slow frame goes."""
    values, v = [], 0.
    for k in range(120):
        values.append(v); v += 4. if k < 100 else .5
    video = tmp_path / 'flight.mp4'
    g.encode([np.full((184, 320, 3), int(x) % 256, np.uint8) for x in values], video, crf=0, size=(320, 184))
    drops = g.run_in_drops(video, 0, 120)
    assert drops and min(drops) >= 100 and all(b - a >= 2 for a, b in zip(drops, drops[1:]))
    assert g.run_in_drops(video, 0, 40) == []                      # too short to judge


def test_camera_fog_eases_the_frames_next_to_the_prefix_into_the_flights_own_cloud():
    fit = dict(altitude_radii=.05, pitch_deg=-20., focal_px=1000.)
    orbit = np.full((g.HEIGHT, g.WIDTH, 3), 60, np.uint8)
    pinned = [np.full((g.HEIGHT, g.WIDTH, 3), 200, np.uint8)] * g.PREFIX
    fog = [np.full((g.HEIGHT, g.WIDTH, 3), 210, np.uint8)] * 4
    plain, _ = g.camera_frames(orbit, pinned, fit, .1)
    fogged, record = g.camera_frames(orbit, pinned, fit, .1, fog)
    assert record['fog_frames'] == 4 and abs(float(fogged[g.PREFIX].mean()) - 210) < 1
    assert float(fogged[g.PREFIX + 2].mean()) > float(plain[g.PREFIX + 2].mean())
    assert all((a == b).all() for a, b in zip(plain[g.PREFIX + 4:], fogged[g.PREFIX + 4:]))


def test_run_in_stops_before_the_portrait_guided_frames():
    mapping = [f'parent:{i}' for i in range(5)] + [f'descent:{g}' for g in range(60, 20, -1)] + [f'native:{k}' for k in range(4, 30)]
    assert g.guided_start(dict(source_frame_mapping=mapping)) == 5 + (60 - 24)
    assert g.guided_start(dict(source_frame_mapping=['parent:0', 'native:0'])) == 1


def test_frozen_hold_drop_keeps_every_frame_that_shows_the_person():
    import importlib.util, types
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('canonical_stage', Path(g.__file__).with_name('canonical_stage.py'))
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
    f = types.SimpleNamespace(HOLD_FLOOR=.35, HOLD_MAX_DROP=12, FPS=24, ending_frame=lambda r, s, first: len(r) - 1)
    stage.protect_person(f)
    f.ending_frame([0] * 40, .15, 10)                     # the frozen assemble passes the portrait seam n = 10
    diffs = [1.] * 60 + [.05] * 40                         # the camera slows to a crawl over the last 40 frames
    keep = f.collapse_holds(diffs, 90)                     # native portrait from output frame 90
    assert keep[-21:] == list(range(80, 101))              # guided 80..89 and the portrait all kept
    assert len(keep) < 101                                 # the slow descent before them is still thinned
