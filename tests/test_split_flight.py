"""Split photographic flight: parent and descent clocks, guide times and whole-frame delivery."""
import json
import shutil

import cv2
import numpy as np
import pytest

from test_photo_delivery import encode
from touchdown.authored_scene import digest
from touchdown.photo_delivery import JoinMismatch
from touchdown import split_flight as sf
from touchdown.relay_timing import query_positions

CFG = dict(place="Teotihuacan", seed=7, video=dict(flight_mode="split", near_frames=73),
           words=dict(region="central Mexico", pattern="a broad arid valley", palette="ochre, grey-green"))


def beat_at(timing, frame):
    return next(b["name"] for b in timing["generation_beats"] if b["start_frame"] <= frame < b["end_frame"])


def test_parent_clock_matches_the_delivery_beats_and_names_the_place_only_near_the_ground():
    o = sf.options(CFG)
    t = sf.parent_timing(CFG, o)
    assert t["source_frames"] == 241 == sf.parent_frames(o)
    assert [b["name"] for b in t["generation_beats"]] == ["settle", "approach", "high_aerial", "clouds", "orbit"]
    for name, lo, hi in zip(("orbit", "clouds", "high_aerial", "approach", "settle"), sf.BOUNDS, sf.BOUNDS[1:]):
        for d in range(lo, hi):
            assert beat_at(t, 192 - d) == name
    places = [b["name"] for b in t["generation_beats"] if "Teotihuacan" in b["prompt"]]
    assert places == ["approach"] and "Teotihuacan" not in t["global_prompt"]
    assert t["single_prompt"] == t["global_prompt"] + " " + " ".join(b["prompt"] for b in t["generation_beats"])


def test_descent_clock_covers_the_parent_after_the_kept_opening():
    o = sf.options(CFG)
    t = sf.descent_timing(CFG, o)
    assert o["high_join"] == "parent" and t["source_frames"] == 121
    assert [(b["name"], b["start_frame"], b["end_frame"]) for b in t["generation_beats"]] == [
        ("settle", 0, 37), ("approach", 37, 85), ("high_aerial", 85, 121)]
    anchor = 192 - o["anchor_delivery_frame"]
    assert anchor == 56 and beat_at(t, anchor) == "approach"
    assert "Finish exactly in the supplied high aerial photograph" in t["single_prompt"]
    # The high still is pinned in the parent, just below the cloud and inside the kept opening.
    assert sf.BOUNDS[2] < o["high_delivery_frame"] < o["opening_frames"]


CLOUD = dict(high_join="cloud", opening_frames=40, descent_frames=153, high_strength=.8)


def test_the_cloud_join_variant_starts_the_descent_in_the_cloud():
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = CLOUD
    o = sf.options(cfg)
    t = sf.descent_timing(cfg, o)
    assert t["source_frames"] == 153
    assert [(b["name"], b["start_frame"], b["end_frame"]) for b in t["generation_beats"]] == [
        ("settle", 0, 37), ("approach", 37, 85), ("high_aerial", 85, 135), ("clouds", 135, 153)]
    high = 192 - o["high_delivery_frame"]
    assert high == 128 and high % 8 == 0 and beat_at(t, high) == "high_aerial"
    assert beat_at(t, 153 - sf.SEAM_FRAMES) == "clouds"
    assert "Pass exactly through the supplied high aerial" in t["single_prompt"] and "cloud tops in this part" in t["single_prompt"]


def test_guide_planes_sit_at_their_own_times():
    assert sf.guide_times(0, 25) == [0, 4.5, 12.5, 20.5]
    assert len(sf.guide_times(0, 73)) == 10
    guides = sf.guide_times(0, 25) + [72, 120]
    latents = 120 // 8 + 1
    assert query_positions(latents, 2, 2 * (latents + len(guides)), guides)[latents:] == guides
    assert sf.guide_times(113, 8) == [116.5]
    with pytest.raises(ValueError):
        sf.guide_times(112, 9)


@pytest.mark.parametrize("change", [dict(anchor_strength=1.0), dict(high_strength=1.0), dict(high_delivery_frame=40), dict(descent_frames=113), dict(parent_handles=4),
                                    dict(opening_frames=40, descent_frames=153), dict(high_delivery_frame=110),
                                    dict(CLOUD, opening_frames=72, descent_frames=121), dict(CLOUD, high_delivery_frame=110),
                                    dict(high_join="sideways")])
def test_invalid_split_settings_are_refused(change):
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = change
    with pytest.raises(ValueError):
        sf.options(cfg)


def test_the_portrait_prefix_fits_inside_the_settle_beat():
    o = sf.options(CFG)
    settle = sf.parent_timing(CFG, o)["generation_beats"][0]
    assert settle["name"] == "settle" and o["prefix_frames"] < settle["end_frame"]


def world(width):
    rng = np.random.default_rng(3)
    return cv2.GaussianBlur(rng.integers(0, 256, (184, width, 3), dtype=np.uint8), (3, 3), 0)


def segments(tmp_path, shift=0):
    """One camera pan cut into the three sources exactly as the pins bind them (delivery frame = position)."""
    pan = world(2 * 260 + 400)
    frame = lambda position: pan[:, 2 * position:2 * position + 320].copy()
    sources = dict(parent=[frame(192 - g) for g in range(193)],
                   descent=[frame(192 - g + shift) for g in range(121)],
                   native=[frame(168 + j + shift) for j in range(17)])
    records = {}
    for name, frames in sources.items():
        video, graph = tmp_path / f"{name}.mp4", tmp_path / f"{name}.json"
        encode(video, frames); graph.write_text("{}")
        records[name] = dict(video=str(video), sha256=digest(video), workflow=str(graph), workflow_sha256=digest(graph))
    return records


@pytest.fixture
def ffmpeg():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")


def options():
    return sf.options(dict(video=dict(split=dict(parent_handles=0))))


def test_delivery_keeps_the_opening_then_the_descent_then_the_native_portrait(tmp_path, ffmpeg):
    o = options(); r = segments(tmp_path)
    record = sf.assemble(tmp_path, r["native"], r["parent"], r["descent"], o)
    assert record["output_frames"] == 72 + (120 - (24 - record["native_first_frame"])) + (17 - record["native_first_frame"])
    assert record["high_seam"]["acceptable"] and record["portrait_seam"]["acceptable"]
    assert record["source_frame_mapping"][:2] == ["parent:192", "parent:191"]
    assert record["source_frame_mapping"][71:73] == ["parent:121", "descent:120"]
    assert record["target_high_frame"] == sf.BOUNDS[2] + (72 - sf.BOUNDS[2]) // 2
    assert record["source_frame_mapping"][-1] == "native:16"
    assert record["native_start_frame"] == record["output_frames"] - record["native_frames"] and not record["crossfade"]
    assert record["generated_cuts"] == [] and not record["motion"]["gate"]
    cap = cv2.VideoCapture(str(tmp_path / "earth_zoom.mp4"))
    frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(image.astype(float))
    cap.release()
    assert len(frames) == record["output_frames"]
    # A continuous pan: no repeated (frozen) frame anywhere, including at both joins.
    assert min(np.abs(a - b).mean() for a, b in zip(frames, frames[1:])) > 1


def test_a_descent_that_misses_the_opening_frame_holds_instead_of_dissolving(tmp_path, ffmpeg):
    o = options(); r = segments(tmp_path, shift=60)
    with pytest.raises(JoinMismatch, match="kept opening"):
        sf.assemble(tmp_path, r["native"], r["parent"], r["descent"], o)
    assert not (tmp_path / "pacing.json").exists()


def test_a_changed_segment_is_refused(tmp_path, ffmpeg):
    o = options(); r = segments(tmp_path)
    (tmp_path / "descent.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="segment changed"):
        sf.assemble(tmp_path, r["native"], r["parent"], r["descent"], o)


def test_the_slowest_half_second_is_reported_not_gated():
    report = sf.motion_report([4.0] * 60 + [0.2] * 12 + [4.0] * 40, 100)
    assert report["slowest_window_start_frame"] == 60 and report["slowest_window_ratio"] < .1
    assert report["gate"] is False


def test_new_studio_scenes_use_the_split_flight_with_their_own_regional_words(tmp_path):
    from touchdown.studio_brief import scene_config
    from touchdown.photo_only import asset_plan
    brief = dict(selected_references=[0, 1], setting="A stepped pyramid above a long avenue", surrounding_landscape="Dry scrub",
                 regional_view="Central Mexico", appearance="Person in a green coat", support="stone",
                 support_description="worn stone paving", light="Hard afternoon sun", regional_kind="desert")
    setup = dict(destination={"title": "Teotihuacan", "country": "Mexico"}, subject={"name": "subject.jpg"},
                 references=[{"name": "a.jpg"}, {"name": "b.jpg"}], ending="arms-open", framing="three-quarter")
    cfg = scene_config(setup, brief, tmp_path, tmp_path, 42)
    assert cfg["video"]["flight_mode"] == "split"
    assert [p["name"] for p in asset_plan(cfg, tmp_path)] == ["arrival", "high", "orbit"]
    o = sf.options(cfg)
    t = sf.parent_timing(cfg, o)
    assert "arid" in t["single_prompt"] and "Hard afternoon sun" in t["global_prompt"]
    assert "worn stone paving" in sf.descent_timing(cfg, o)["single_prompt"]
    assert "Teotihuacan" in sf.anchor_prompt(cfg)


def test_the_setting_text_cannot_place_the_person_on_a_different_surface(tmp_path):
    from touchdown.studio_brief import scene_config
    brief = dict(selected_references=[0, 1], setting="Ancient terraces above a river gorge. The person stands on a grassy patch "
                 "adjacent to ancient stone walls.", surrounding_landscape="Steep green mountains", regional_view="Andes",
                 appearance="Person in a green coat", support="ground_paving", support_description="weathered grey stone steps",
                 light="Soft cloudy light", regional_kind="mountains")
    setup = dict(destination={"title": "Machu Picchu", "country": "Peru"}, subject={"name": "subject.jpg"},
                 references=[{"name": "a.jpg"}, {"name": "b.jpg"}], ending="arms-open", framing="three-quarter")
    cfg = scene_config(setup, brief, tmp_path, tmp_path, 42)
    assert "grassy patch" not in cfg["prompts"]["arrival"] and "Ancient terraces" in cfg["prompts"]["arrival"]
    assert "weathered grey stone steps" in cfg["prompts"]["arrival"]


def test_the_portrait_ends_at_the_first_frame_that_reaches_the_requested_crop():
    ratios = [.09] * 30 + [.12, .14, .151, .17, .2, .3]
    assert sf.ending_frame(ratios, .15, 4) == 32
    assert sf.ending_frame([None] * 40, .15, 4) == 39
    assert sf.ending_frame(ratios, None, 4) == len(ratios) - 1
    assert sf.ending_frame([.2] * 40, .15, 4) == 4 + sf.MIN_NATIVE_FRAMES  # never shorter than the minimum push
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["framing"] = "three-quarter"
    assert sf.options(cfg)["end_face_ratio"] == sf.FACE_STOP["three-quarter"] and sf.options(CFG)["end_face_ratio"] is None


def test_hagia_sophia_ratios_stop_knees_up_and_the_crop_is_measured_not_guessed():
    # b3's native portrait as measured on the desktop: seam at native 24, knees-up at native 31 (8.29 s).
    ratios = [.06] * 24 + [.09, .1, .11, .125, .13, .14, .148, .155, .172, .2] + [.3] * 39
    end = sf.ending_frame(ratios, sf.FACE_STOP["three-quarter"], 24)
    assert end == 31
    pacing = dict(native_start_frame=192, native_first_frame=24,
                  ending=dict(stop_face_ratio=.15, native_last_frame=end, face_ratios=ratios))
    assert sf.measured_crop(pacing, "three-quarter")["status"] == "pass"
    assert sf.measured_crop(dict(pacing, ending=dict(pacing["ending"], native_last_frame=48)), "three-quarter")["status"] == "fail"
    assert sf.feet_visible_until(pacing) == (192 + 26 - 24) / 24
    assert sf.measured_crop(dict(pacing, ending=dict(pacing["ending"], stop_face_ratio=None)), "three-quarter") is None


def test_a_hang_in_the_flight_is_shortened_by_dropping_frames_and_the_portrait_is_kept_whole():
    diffs = [4.0] * 100 + [.1] * 20 + [4.0] * 50 + [.2] * 30
    keep = sf.collapse_holds(diffs, 170)
    dropped = sorted(set(range(201)) - set(keep))
    assert dropped and all(100 < i < 121 for i in dropped)
    assert keep[-31:] == list(range(170, 201))
    assert max(b - a for a, b in zip(keep, keep[1:])) <= sf.HOLD_MAX_DROP + 1
    assert sf.collapse_holds([4.0] * 200, 170) == list(range(201))


def test_the_orbit_shows_the_real_relief_and_a_city_only_as_sprawl(tmp_path):
    from touchdown.studio_brief import scene_config
    brief = dict(selected_references=[0, 1], setting="A red sandstone tomb facade cut into a cliff", surrounding_landscape="Rugged sandstone",
                 regional_view="Southern Jordan", appearance="Person in a green coat", support="sand",
                 support_description="pale packed sand", light="Hard afternoon sun", regional_kind="desert",
                 regional_relief="plateau_and_canyons", nearest_settlement="town")
    setup = dict(destination={"title": "Petra", "country": "Jordan"}, subject={"name": "subject.jpg"},
                 references=[{"name": "a.jpg"}, {"name": "b.jpg"}], ending="arms-open", framing="three-quarter")
    orbit = scene_config(setup, brief, tmp_path, tmp_path, 42)["prompts"]["orbit"]
    assert "escarpments" in orbit and "small town" in orbit and "urban blocks" not in orbit
    city = scene_config(setup, dict(brief, nearest_settlement="large_city"), tmp_path, tmp_path, 42)["prompts"]["orbit"]
    assert "urban sprawl" in city and "no street or building" in city


def test_relief_comes_from_the_elevation_of_the_region():
    from touchdown.region_facts import classify_relief
    rng = np.random.default_rng(1)
    assert classify_relief(rng.uniform(5, 150, (256, 256)))["relief"] == "plain"
    sierra = rng.uniform(300, 1500, (256, 256)); sierra[:8] = 3000
    assert classify_relief(sierra)["relief"] == "high_mountains"
    assert classify_relief(rng.uniform(800, 1500, (256, 256)))["relief"] == "plateau_and_canyons"
    bay = rng.uniform(5, 150, (256, 256)); bay[:80] = -20
    assert classify_relief(bay)["relief"] == "coastal_lowland"


def test_the_high_still_takes_its_relief_from_the_slopes_around_the_place():
    from touchdown.region_facts import classify_local, settlement_kind
    y, x = np.mgrid[0:256, 0:256].astype(float)
    # A high plain (1000 m, gentle undulation) beside a range at the tile's edge, like Segovia under the Guadarrama.
    meseta = 1000 + 20 * np.sin(x / 9) + 20 * np.cos(y / 11); meseta[:, 230:] = 2400
    assert classify_local(meseta, 40.95, -4.13)["local_relief"] == "plain"
    ranges = 1500 + 900 * np.sin(x / 2.5) * np.cos(y / 2.5)
    assert classify_local(ranges, 47.5, 10.7)["local_relief"] == "high_mountains"
    rolling = 150 + 120 * np.sin(x / 2) * np.cos(y / 3)
    assert classify_local(rolling, 43.2, 2.36)["local_relief"] == "hills"
    # Granada: the tile's corner is Mediterranean, the 25 km around the Alhambra is dry land.
    inland = rolling.copy(); inland[200:, 200:] = -20
    assert classify_local(inland, 37.18, -3.59)["local_sea_fraction"] == 0 and (inland <= 0).mean() > .03
    assert settlement_kind(dict(population=52000, distance_km=1)) == "town"
    assert settlement_kind(dict(population=230000, distance_km=0)) == "large_city"
    assert settlement_kind(dict(population=230000, distance_km=12)) == "none"



def test_the_sharp_portrait_is_eased_in_from_the_softer_descent():
    sharp = world(640)
    soft = cv2.GaussianBlur(sharp, (0, 0), 1.5)
    sigma = sf.sharpness_match(soft, sharp)
    assert 1.0 <= sigma <= 2.0
    assert sf.sharpness_match(sharp, sharp) == 0.0


def test_the_cloud_join_starts_where_the_parent_cloud_is_most_featureless():
    textured, white = world(320), np.full((184, 320, 3), 235, np.uint8)
    # Bright cloud edges and terrain at delivery 40-47 (Versailles e1); whiteout from 48 on.
    opening, scores = sf.cloud_opening(lambda d: white if d >= 48 else textured)
    assert opening == 48 and scores[48] < scores[40]
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = CLOUD
    o = sf.with_opening(sf.options(cfg), opening)
    assert o["descent_frames"] == 145 and (o["descent_frames"] - sf.SEAM_FRAMES) % 8 == 1
    t = sf.descent_timing(cfg, o)
    assert beat_at(t, 145 - sf.SEAM_FRAMES) == "clouds" and beat_at(t, 192 - o["high_delivery_frame"]) == "high_aerial"
    with pytest.raises(ValueError):
        sf.with_opening(o, 32)


def test_the_ending_action_is_judged_on_its_own_question(monkeypatch, tmp_path):
    from touchdown import studio_brief as sb
    assert set(sb.ACTION_QUESTIONS) == set(sb.POSES)
    asked = []
    def chat(url, frames, prompt, schema, output):
        asked.append(prompt)
        return dict(answer="yes", evidence="Both arms are held out to the sides.")
    monkeypatch.setattr(sb, "chat", chat)
    review = sb.ending_action_review("u", [], "arms-open", tmp_path / "a.json")
    assert review["status"] == "pass" and "BOTH" in asked[0] and "Ignore cropping" in asked[0]



def test_the_high_view_is_a_moving_guide_ending_on_the_pinned_frame(tmp_path):
    o = sf.options(CFG)
    start = sf.high_motion_start(o)
    assert o["high_motion_frames"] == 24 and start == 105 and start % 8 == 1
    assert start + o["high_motion_frames"] - 1 == 192 - o["high_delivery_frame"]
    assert sf.guide_times(start, 24) == [108.5, 116.5, 124.5]
    still = tmp_path / "high.png"
    y, x = np.mgrid[0:768, 0:1344]
    smooth = np.dstack([x * 255 // 1343, y * 255 // 767, (x + y) * 255 // 2110]).astype(np.uint8)
    cv2.imwrite(str(still), smooth)
    frames = sf.high_motion(still, tmp_path / "high-motion.mp4", 24, 1.06, 1280, 736)
    assert len(frames) == 24 and frames[0].shape == (736, 1280, 3)
    full = cv2.resize(cv2.imread(str(still)), (1280, 736), interpolation=cv2.INTER_AREA)
    # The last frame (the pinned high frame) is the whole still; the first is a 6 % tighter crop of its centre.
    assert np.abs(frames[-1].astype(int) - full.astype(int)).mean() < 3
    assert np.abs(frames[0].astype(int) - full.astype(int)).mean() > np.abs(frames[-1].astype(int) - full.astype(int)).mean()
    cap = cv2.VideoCapture(str(tmp_path / "high-motion.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    cap.release()


@pytest.mark.parametrize("change", [dict(high_motion_frames=20), dict(high_motion_frames=24, high_delivery_frame=60),
                                    dict(high_motion_frames=56), dict(high_zoom=1.0)])
def test_the_moving_high_guide_keeps_to_the_grid_and_the_high_aerial_beat(change):
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = change
    with pytest.raises(ValueError):
        sf.options(cfg)


def test_the_zoom_profile_reads_the_scale_change_per_frame(tmp_path, ffmpeg):
    # A steady zoom in on a textured field, then a hold for the last second.
    rng = np.random.default_rng(5)
    field = cv2.GaussianBlur(rng.integers(0, 256, (1200, 1200, 3), dtype=np.uint8), (5, 5), 0)
    frames = []
    for i in range(72):
        k = min(i, 47)
        size = int(1000 / 1.01 ** k)
        x0 = (1200 - size) // 2
        frames.append(cv2.resize(field[x0:x0 + size, x0:x0 + size], (320, 320), interpolation=cv2.INTER_AREA))
    encode(tmp_path / "zoom.mp4", frames)
    profile = sf.zoom_profile(tmp_path / "zoom.mp4", 72)
    tracked = [v for v in profile["log_scale"][:47] if v is not None]
    assert len(tracked) > 40 and abs(np.median(tracked) - np.log(1.01)) < .004
    assert profile["flat_from_frame"] is not None and 44 <= profile["flat_from_frame"] <= 52
    assert "stops" in profile["summary"]


def test_the_layout_proxy_pulls_back_at_a_constant_rate_and_swaps_inside_the_cloud(tmp_path, ffmpeg):
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = dict(parent_layout=True, width=320, height=184)
    o = sf.options(cfg)
    y, x = np.mgrid[0:768, 0:1344]
    high = np.dstack([x * 255 // 1343, y * 255 // 767, np.full_like(x, 60)]).astype(np.uint8)
    orbit = np.full((768, 1344, 3), (200, 80, 20), np.uint8)
    prefix = [np.full((184, 320, 3), 30 + i, np.uint8) for i in range(25)]
    frames = sf.layout_proxy(prefix, high, orbit, tmp_path / "proxy.mp4", o)
    assert len(frames) == sf.parent_frames(o) == 241
    assert np.array_equal(frames[0], prefix[0]) and np.array_equal(frames[24], prefix[24])
    high_g = 192 - o["high_delivery_frame"]
    assert np.abs(frames[high_g].astype(int) - sf._window(high, 1.0, 320, 184).astype(int)).mean() < 2
    # The nested portrait keeps shrinking, and the camera never stands still before the cloud.
    def portrait_width(frame):
        return int((np.abs(frame[92].astype(int) - 54).sum(axis=1) < 30).sum())
    widths = [portrait_width(frames[g]) for g in (30, 45, 60)]
    assert widths[0] > widths[1] > widths[2] > 0
    moves = [np.abs(frames[g + 1].astype(int) - frames[g].astype(int)).mean() for g in range(25, 136)]
    assert min(moves) > .05
    # Opaque cloud (no ground) between the high still and the orbit; the orbit fills the end.
    cloud = frames[192 - 40]
    assert cloud.mean() > 200 and cloud.std() < 20
    assert np.abs(frames[-1].astype(int) - np.array((200, 80, 20))).mean() < 3


def test_the_layout_proxy_needs_the_parent_join():
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = dict(CLOUD, parent_layout=True)
    with pytest.raises(ValueError):
        sf.options(cfg)


def test_the_seam_carry_continues_the_parent_zoom_at_its_own_speed(tmp_path, ffmpeg):
    rng = np.random.default_rng(9)
    field = cv2.GaussianBlur(rng.integers(0, 256, (900, 1500, 3), dtype=np.uint8), (5, 5), 0)
    # Parent generation order: retreating, so each frame widens by 1 % (zooming in when played forwards).
    seam = [sf._window(field, 0.6 * 1.01 ** i, 320, 184) for i in range(sf.SEAM_FRAMES)]
    encode(tmp_path / "seam.mp4", seam)
    frames, speed = sf.seam_carry(tmp_path / "seam.mp4", tmp_path / "carry.mp4", 16, 320, 184)
    assert len(frames) == 16 and abs(speed - np.log(1.01)) < .003
    # The last carry frame sits one step inside the seam's lowest frame; the first, sixteen steps inside.
    measured = [v for v in sf._log_scales(tmp_path / "carry.mp4", 16) if v is not None]
    assert len(measured) >= 12 and abs(np.median(measured) + speed) < .004
    o = sf.options(CFG)
    start = o["descent_frames"] - sf.SEAM_FRAMES - o["seam_carry_frames"]
    assert start == 89 and sf.guide_times(start, 24) == [92.5, 100.5, 108.5]
    # Accelerating: the carry eases from the parent's speed into the swoop.
    frames, _ = sf.seam_carry(tmp_path / "seam.mp4", tmp_path / "fast.mp4", 24, 320, 184, accel=4.0)
    fast = sf._log_scales(tmp_path / "fast.mp4", 24)
    early, late = [-v for v in fast[-6:] if v is not None], [-v for v in fast[:6] if v is not None]
    assert np.median(late) > 2.5 * np.median(early)


def test_the_bridge_grows_the_anchor_until_it_fills_the_anchor_frame(tmp_path, ffmpeg):
    rng = np.random.default_rng(4)
    field = cv2.GaussianBlur(rng.integers(0, 256, (900, 1500, 3), dtype=np.uint8), (5, 5), 0)
    seam = [sf._window(field, 0.6 * 1.01 ** i, 320, 184) for i in range(sf.SEAM_FRAMES)]
    encode(tmp_path / "seam.mp4", seam)
    anchor = tmp_path / "anchor.png"
    cv2.imwrite(str(anchor), np.full((736, 1280, 3), (40, 160, 90), np.uint8))
    frames, _ = sf.seam_carry(tmp_path / "seam.mp4", tmp_path / "carry.mp4", 24, 320, 184, accel=4.0, anchor=anchor, bridge=32)
    assert len(frames) == 56
    green = lambda f: float((np.abs(f.astype(int) - (40, 160, 90)).sum(axis=2) < 20).mean())
    # Retreat order: frame 0 sits next to the anchor frame and is the anchor; the last frames (next to the seam) have none.
    assert green(frames[0]) > .99 and green(frames[-1]) < .01
    shares = [green(frames[j]) for j in (0, 10, 20, 30)]
    assert shares == sorted(shares, reverse=True)
    o = sf.options(CFG)
    start = o["descent_frames"] - sf.SEAM_FRAMES - o["seam_carry_frames"] - o["seam_bridge_frames"]
    assert start == 57 == 192 - o["anchor_delivery_frame"] + 1 and start % 8 == 1
    frame, count = sf.carry_guide(o)
    assert (frame, count) == (57, 56) and len(sf.guide_times(frame, count)) == 7


def test_the_carry_is_the_parents_own_flight_down_to_the_anchor_by_default():
    o = sf.options(CFG)
    assert o["seam_carry_source"] == "parent"
    frame, count = sf.carry_guide(o)
    # Parent generation frames 57..112 sit right below the anchor (56) and right above the seam (113..120).
    assert frame == 192 - o["anchor_delivery_frame"] + 1 and frame + count == o["descent_frames"] - sf.SEAM_FRAMES
    with pytest.raises(ValueError):
        sf.options({**CFG, "video": {**CFG.get("video", {}), "split": {"seam_carry_source": "inset"}}})


def test_the_proxy_matches_the_high_guide_and_swells_into_the_swoop_below_it():
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = dict(parent_layout=True)
    o = sf.options(cfg)
    w = sf.proxy_window(o, 25)
    top, high_g = sf.high_motion_start(o), 192 - o["high_delivery_frame"]
    assert abs(w[25] - np.log(o["layout_nest"])) < .1 and abs(w[high_g]) < 1e-9
    assert abs(w[top] + np.log(o["high_zoom"])) < 1e-9
    rates = np.diff([w[g] for g in range(25, 164)])
    assert rates.min() > 0
    guide = rates[top - 25:high_g - 25]
    assert np.allclose(guide, guide[0]) and abs(rates[top - 26] - guide[0]) < .002
    assert rates.max() > 5 * guide[0]


def _nested_rungs(count=4, width=320, height=184, scale=16):
    """Rungs cut from one big texture: rung j covers 2 ** (j - top) of the top rung, as a perfect ladder would."""
    rng = np.random.default_rng(3)
    # Detail at every scale (an 8 px grain on the top rung would vanish once magnified, and vice versa).
    big = sum(cv2.resize(rng.random((height * f // 64, width * f // 64, 3)).astype(np.float32),
                         (width * scale, height * scale), interpolation=cv2.INTER_CUBIC) for f in (2, 8, 32)) / 3
    big = (big.clip(0, 1) * 255).astype(np.uint8)
    rungs = []
    for j in range(count):
        cover = 2.0 ** (j - (count - 1))
        rungs.append(sf._window(big, cover, width, height))
    return big, rungs


def test_the_ladder_proxy_zooms_through_the_rungs_without_an_edge():
    big, rungs = _nested_rungs()
    frames = [sf.ladder_frame(rungs, -k * .05, 320, 184).astype(int) for k in range(0, 60)]
    # Each frame matches the same window cut straight from the big texture: no inset, seam or jump at rung changes.
    for k in (0, 7, 14, 20, 21, 27, 41, 55):
        truth = sf._window(big, float(np.exp(-k * .05)), 320, 184).astype(int)
        assert np.abs(frames[k] - truth).mean() < 6
    steps = [np.abs(frames[k + 1] - frames[k]).mean() for k in range(59)]
    assert max(steps) < 3 * np.median(steps)


def test_outpaint_canvas_and_nest_back_keep_the_rung_below_at_the_centre():
    _, rungs = _nested_rungs(count=2)
    canvas = sf.outpaint_canvas(rungs[0])
    h, w = canvas.shape[:2]
    half = cv2.resize(rungs[0], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    assert canvas.shape[2] == 4
    assert np.array_equal(canvas[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2, :3], half)
    # FLUX repaints (alpha 0) the border and a band just inside the half, and keeps its core (alpha 255).
    assert canvas[:h // 4, :, 3].max() == 0 and canvas[h // 4 + 2, w // 2, 3] == 0 and canvas[h // 2, w // 2, 3] == 255
    painted = np.full_like(rungs[0], 128)
    nested = sf.nest_back(painted, rungs[0])
    core = nested[h // 4 + 20:h // 4 + h // 2 - 20, w // 4 + 30:w // 4 + w // 2 - 30].astype(int)
    assert np.abs(core - half[20:-20, 30:-30].astype(int)).mean() < 1
    assert np.abs(nested[:10].astype(int) - 128).mean() < 1


def test_with_a_ladder_the_swoop_ends_on_the_lowest_rung_at_the_anchor_frame():
    cfg = json.loads(json.dumps(CFG)); cfg["video"]["split"] = dict(parent_layout=True)
    o = sf.options(cfg)
    anchor = 192 - o["anchor_delivery_frame"]
    low = -o["layout_rungs"] * np.log(2)
    w = sf.proxy_window(o, 25, bottom=anchor, low=low)
    assert abs(w[anchor] - low) < 1e-9 and abs(w[192 - o["high_delivery_frame"]]) < 1e-9
    rates = np.diff([w[g] for g in range(25, 164)])
    assert rates.min() > 0 and abs(rates[0] - o["ladder_drift"]) < 1e-9


def test_flux_inpaint_graph_starts_from_the_canvas_and_repaints_only_its_mask():
    from touchdown import flux3
    g = flux3.graph("p", 1, ["rung0.png"], "x", 2560, 1472, inpaint="canvas.png")
    assert g["13"]["class_type"] == "VAEEncode" and g["13"]["inputs"]["pixels"] == ["m0", 0]
    assert g["17"]["class_type"] == "SetLatentNoiseMask" and g["17"]["inputs"]["mask"] == ["m0", 1]
    assert g["14"]["inputs"]["latent_image"] == ["17", 0]
    assert flux3.graph("p", 1, [], "x")["13"]["class_type"] == "EmptyFlux2LatentImage"


def test_ground_view_at_the_nadir_of_a_rung_height_is_that_rung():
    big, rungs = _nested_rungs(count=3)
    for i in range(3):
        view = sf.ground_view(rungs, sf.RUNG0_HEIGHT * 2 ** i, 90, 0, 320, 184).astype(int)
        # Haze aside (it grows with height), the nadir view from rung i's height is rung i.
        assert np.corrcoef(view.ravel(), rungs[i].astype(int).ravel())[0, 1] > .9


def test_the_crane_rises_from_eye_level_to_the_rung0_nadir_and_shows_sky_at_the_start():
    h0, p0, b0 = sf.crane(0, 31)
    h1, p1, b1 = sf.crane(31, 31)
    assert (round(h0, 2), p0, b0) == (1.7, 6, 15) and (round(h1, 2), p1, b1) == (sf.RUNG0_HEIGHT, 90, 0)
    _, rungs = _nested_rungs(count=3)
    low = sf.ground_view(rungs, h0, p0, b0, 320, 184)
    assert np.abs(low[:20].astype(np.float32) - sf.SKY).mean() < 10  # a horizon: sky at the top
    assert np.abs(low[-20:].astype(np.float32) - sf.SKY).mean() > 20  # ground at the bottom
