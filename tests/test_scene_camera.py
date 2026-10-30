import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geoblock_core.camera import (H3_MAX_FRAMES, PRESETS, Terrain, chain_segments,
                                  h3_frame_count, plan_route)
from geoblock_core.providers import USGS3DEPProvider
from geoblock_core.scene import build_landform_scene, plan_spacing, qualify_scene

LAT, LON = 36.0617, -112.1078
STEP = 1 / 10800  # 1/3 arc-second


def write_tile(root, name, west, north, size=1200):
    """A rim at the landmark latitude with an 800 m drop to the north."""
    rows, cols = np.indices((size, size))
    lat = north - (rows + 0.5) * STEP
    heights = 2000 - 800 / (1 + np.exp(-(lat - (LAT + 0.004)) / 0.002))
    directory = root / name
    directory.mkdir(parents=True)
    with rasterio.open(directory / f"USGS_13_{name}.tif", "w", driver="GTiff", width=size, height=size,
                       count=1, dtype="float32", crs="EPSG:4269", nodata=-999999,
                       transform=from_origin(west, north, STEP, STEP)) as dst:
        dst.write(heights.astype("float32"), 1)


@pytest.fixture
def tiles(tmp_path):
    root = tmp_path / "tiles"
    # A small stand-in for tile n37w113, positioned around the test landmark.
    write_tile(root, "n37w113", LON - 0.05, LAT + 0.05)
    return root


@pytest.fixture
def scene(tmp_path, tiles):
    return build_landform_scene(name="Synthetic rim", latitude=LAT, longitude=LON, radius_m=2000,
                                output_root=tmp_path / "scenes", raw_cache=tmp_path / "raw",
                                location_source="test", provider=USGS3DEPProvider(str(tiles)))


def test_tile_names_follow_north_west_corner_convention():
    assert USGS3DEPProvider.tile_names(-112.2, 35.9, -112.0, 36.1) == ["n36w113", "n37w113"]
    assert USGS3DEPProvider.tile_names(-113.1, 36.2, -112.9, 36.4) == ["n37w114", "n37w113"]


def test_provider_refuses_outside_conus(tmp_path, tiles):
    with pytest.raises(ValueError, match="conterminous"):
        build_landform_scene(name="Hawaii", latitude=19.8, longitude=-155.5, radius_m=500,
                             output_root=tmp_path / "s", raw_cache=tmp_path / "r",
                             location_source="test", provider=USGS3DEPProvider(str(tiles)))


def test_plan_spacing_keeps_mesh_bounded():
    assert plan_spacing(4000) == (4000.0, 10.0)
    assert plan_spacing(12000) == (12000.0, 20.0)
    radius, spacing = plan_spacing(12345)
    assert spacing == 30 and radius % spacing == 0 and radius >= 12345


def test_scene_is_verified_navd88_and_reused(scene, tmp_path, tiles):
    assert scene["height_alignment"] == "verified"
    assert scene["projected_crs"] == "EPSG:26912"
    expected = 2000 - 800 / (1 + np.exp(0.004 / 0.002))
    assert scene["target_height_m"] == pytest.approx(expected, abs=3)
    assert "local relief" in qualify_scene(scene)
    again = build_landform_scene(name="Synthetic rim", latitude=LAT, longitude=LON, radius_m=2000,
                                 output_root=tmp_path / "scenes", raw_cache=tmp_path / "raw",
                                 location_source="test", provider=USGS3DEPProvider(str(tiles)))
    assert again == json.loads((tmp_path / "scenes" / scene["terrain_dir"].split("/")[-1] / "scene.json").read_text())


def test_h3_frame_rules():
    assert h3_frame_count(10) == 243
    assert (h3_frame_count(14.9) - 5) % 17 == 0 and h3_frame_count(14.9) <= H3_MAX_FRAMES
    assert chain_segments(243) == [(0, 243)]
    segments = chain_segments(H3_MAX_FRAMES * 2)
    assert segments[0] == (0, H3_MAX_FRAMES) and segments[-1][1] == H3_MAX_FRAMES * 2
    assert all(end - start <= H3_MAX_FRAMES for start, end in segments)


@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "Earth zoom"])
def test_presets_clear_terrain_and_face_the_drop(scene, preset):
    terrain = Terrain.from_scene(scene)
    route = plan_route(scene, preset, 6, terrain=terrain, clearance_m=25)
    assert route["frames"] == len(route["positions"]) == len(route["look_at"]) == h3_frame_count(6)
    assert route["min_clearance_m"] >= 25 - 1e-6
    assert route["heading_deg"] in (355.0, 0.0, 5.0)  # the synthetic drop is due north
    positions = np.array(route["positions"])
    assert np.all(np.isfinite(positions))
    assert np.all(np.linalg.norm(np.diff(positions, axis=0), axis=1) < 200)  # no jumps between frames


def test_earth_zoom_dives_from_orbit_and_ends_at_eye_level(scene):
    route = plan_route(scene, "Earth zoom", 6)
    positions = np.array(route["positions"])
    assert route["kind"] == "earth_zoom" and route["frames"] == len(route["up"]) == h3_frame_count(6)
    assert positions[0, 2] > 300_000 and np.all(np.diff(np.linalg.norm(positions[:, :2], axis=1)) <= 1e-6)
    arrive = route["arrive_frame"]
    assert abs(positions[-1, 2] - positions[arrive, 2]) < 5.0  # level for the hold, following the ground
    assert np.linalg.norm(positions[-1, :2]) < 10  # a few metres from the landmark
    first = np.array(route["look_at"][0]) - positions[0]
    assert first[2] < 0 and np.dot(first[:2], -positions[0, :2]) > 0  # looking down toward the landmark
    # The dive spends about half a second in the cloud deck (the plain curve crossed it in a third of a second;
    # 3x lingered long enough to read as a pause), and the swoop keeps the ground passing under it below half a
    # camera height per frame.
    from geoblock_core.zoom import CLOUD_BASE_M, CLOUD_TOP_M, MAX_GROUND_FLOW
    above = positions[:, 2] - scene["target_height_m"]
    inside = ((above > CLOUD_BASE_M) & (above < CLOUD_TOP_M)).sum()  # a short whiteout: ~0.5 s, not a pause
    assert 0.05 * route["frames"] <= inside <= 0.085 * route["frames"]
    clearance = np.array(route["above_ground_m"])[:arrive]
    speed = np.abs(np.diff(np.linalg.norm(positions[:arrive + 1, :2], axis=1)))
    low = clearance < 500
    assert np.all(speed[low] / clearance[low] < MAX_GROUND_FLOW + 0.1)


def test_low_swoop_tempo_slows_the_swoop_and_keeps_a_solid_whiteout(scene):
    from geoblock_core.zoom import CLOUD_BASE_M, CLOUD_TOP_M
    standard, low = plan_route(scene, "Earth zoom", 10), plan_route(scene, "Earth zoom", 10, dive_tempo="low-swoop")
    assert "dive_tempo" not in standard and low["dive_tempo"] == "low-swoop"
    assert low["arrive_frame"] == standard["arrive_frame"]

    def growth(route):  # picture growth a frame over the low swoop, as the camera's log height change
        above = np.array(route["positions"])[:, 2] - scene["target_height_m"]
        return np.abs(np.diff(np.log(np.maximum(above, 1.0))))[56:100].mean()

    assert growth(low) < 0.9 * growth(standard)
    above = np.array(low["positions"])[:, 2] - scene["target_height_m"]
    inside = ((above > CLOUD_BASE_M) & (above < CLOUD_TOP_M)).sum()
    # at least 12 solid white frames (an LTX latent and a half) before the 8-frame thinning, and under a second
    assert 20 <= inside <= 24


def test_route_that_leaves_terrain_is_rejected(scene):
    with pytest.raises(ValueError, match="leaves the measured terrain"):
        plan_route(scene, "Pull-back rise", 6, scale_m=5000)


def test_user_heading_overrides_auto(scene):
    route = plan_route(scene, "Flyover", 6, heading_deg=90)
    assert route["heading_deg"] == 90 and route["heading_source"] == "user"


def test_place_subject_pastes_the_figure_at_the_stand_ins_size_and_feet():
    from geoblock_core.zoom import place_subject
    sheet = np.full((100, 60, 3), 0.5)
    sheet[10:90, 25:35] = (0.9, 0.1, 0.1)  # an 80 px tall red figure on plain grey
    frame = np.zeros((200, 300, 3))
    out = place_subject(frame, sheet, centre_x=0.6, feet_y=0.9, height=0.4)
    red = np.nonzero(out[..., 0] > 0.5)
    assert abs(red[0].max() + 1 - 180) <= 1 and abs(red[0].max() - red[0].min() + 1 - 80) <= 2
    assert abs(red[1].mean() - 180) <= 2
    cut = place_subject(frame, sheet, centre_x=0.5, feet_y=1.4, height=1.0)  # cut off at the bottom, like a close-up
    assert (cut[..., 0] > 0.5).any() and cut.shape == frame.shape


def test_place_subject_keys_out_a_shaded_wall_and_erases_the_stand_in():
    from geoblock_core.zoom import place_subject
    sheet = np.tile(np.linspace(0.75, 0.6, 60)[None, :, None], (100, 1, 3))  # a wall shaded left to right
    sheet[10:90, 25:35] = (0.9, 0.1, 0.1)
    sheet[40:60, 28:32] = 0.68  # a grey patch inside the figure, as grey as the wall
    frame = np.zeros((200, 300, 3))
    frame[100:180, 170:190] = 0.5  # the grey stand-in, 80 px tall, feet at row 180
    out = place_subject(frame, sheet, centre_x=0.6, feet_y=0.9, height=0.4)
    assert (np.abs(out[..., 0] - 0.5) < 0.02).sum() < 40  # stand-in gone
    kept = (np.abs(out[..., 1] - out[..., 0]) < 0.05) & (out[..., 1] > 0.6)  # grey, not red, not black
    assert kept.sum() >= 50  # the grey patch inside the figure is kept
    assert (out[100:180, 140:165] < 0.05).all()  # no wall box pasted beside the figure


def test_erasing_the_stand_in_leaves_no_streaks():
    from geoblock_core.zoom import erase_stand_in
    frame = np.zeros((200, 300, 3))
    frame[:, :, 1] = np.linspace(0, 1, 200)[:, None]  # a vertical gradient behind
    frame[::7, :, 0] = 1.0  # thin bright rows at the box's edges: a row-wise fill drags them across as streaks
    frame[100:180, 170:190] = 0.5
    out = erase_stand_in(frame, centre_x=0.6, feet_y=0.9, height=0.4)
    box = out[95:181, 165:195]
    assert (np.abs(box - 0.5).max(axis=2) < 0.02).sum() < 10  # stand-in gone
    assert box[..., 0].max() < 0.6  # no bright row carried through the box


def test_a_green_wall_is_keyed_out_and_its_rim_light_removed():
    from geoblock_core.zoom import place_subject
    sheet = np.zeros((100, 60, 3))
    sheet[...] = (0.1, 0.8, 0.15)  # chroma-key green
    sheet[10:90, 25:35] = (0.8, 0.8, 0.8)  # a light grey jacket
    sheet[10:90, 25] = (0.5, 0.8, 0.5)  # green light on its edge
    frame = np.zeros((200, 300, 3))
    out = place_subject(frame, sheet, centre_x=0.6, feet_y=0.9, height=0.4)
    person = out[100:180, 165:195]
    assert (person.min(axis=2) > 0.4).sum() > 400  # the grey jacket is kept, not keyed out with the wall
    assert (person[..., 1] - person[..., [0, 2]].max(axis=2) < 0.1).all()  # no green on it
    assert out[100:180, 120:160].max() < 0.05  # no wall beside it


def test_place_subject_uses_a_clean_plate_when_given():
    from geoblock_core.zoom import place_subject
    sheet = np.full((100, 60, 3), (0.1, 0.8, 0.15))
    sheet[10:90, 25:35] = (0.9, 0.1, 0.1)
    frame = np.zeros((200, 300, 3))
    frame[100:180, 170:190] = 0.5
    clean = np.full((200, 300, 3), 0.25)
    out = place_subject(frame, sheet, centre_x=0.6, feet_y=0.9, height=0.4, clean=clean)
    assert (np.abs(out - 0.5) < 0.02).all(axis=2).sum() == 0 and np.allclose(out[:, :100], 0.25)


def test_the_stand_ins_shadow_darkens_the_plate_and_the_mask_rings_the_person():
    from geoblock_core.zoom import place_subject, shadow_factor
    clean = np.full((200, 300, 3), 0.6)
    shadowed = clean.copy()
    shadowed[175:185, 185:230] = 0.3  # the stand-in's shadow, off to the right of its feet
    shade = shadow_factor(clean, shadowed)
    assert shade[180, 200] == 0.5 and shade[50, 50] == 1.0
    sheet = np.full((100, 60, 3), (0.1, 0.8, 0.15))
    sheet[10:90, 25:35] = (0.9, 0.1, 0.1)
    out, mask = place_subject(clean, sheet, centre_x=0.6, feet_y=0.9, height=0.4, erase=False, shade=shade,
                              with_mask=True)
    assert abs(out[180, 220, 0] - 0.3) < 0.01  # shadow on the ground beside the person
    assert mask[150, 180] == 1 and mask[150, 193] == 1 and mask[150, 230] == 0  # person plus a ~10 px ring


def test_the_pasted_person_takes_on_the_scenes_light():
    from geoblock_core.zoom import place_subject
    sheet = np.full((100, 60, 3), (0.1, 0.8, 0.15))
    sheet[10:90, 25:35] = 0.8  # a white-lit grey figure
    frame = np.zeros((200, 300, 3))
    frame[...] = (0.6, 0.45, 0.25)  # warm, dimmer evening ground
    out = place_subject(frame, sheet, centre_x=0.6, feet_y=0.9, height=0.4, erase=False)
    person = out[110:160, 178:182].reshape(-1, 3).mean(axis=0)
    assert person[0] > person[2] + 0.1 and person.mean() < 0.75  # warmer and dimmer than the studio


def test_chroma_key_drops_the_persons_wall_shadow_and_floor():
    from geoblock_core.zoom import place_subject
    sheet = np.zeros((120, 80, 3))
    sheet[:] = [0.52, 0.83, 0.41]  # sunlit green wall
    sheet[100:] = [0.57, 0.87, 0.45]  # paler green floor
    sheet[30:95, 5:30] = [0.01, 0.28, 0.0]  # the person's shadow on the wall (r19k)
    sheet[20:110, 34:46] = [0.8, 0.8, 0.78]  # the person, centred
    sheet[50:60, 36:44] = [0.29, 0.42, 0.24]  # an olive strap
    sheet[101, 1] = [0.5, 0.5, 0.5]  # a speck at the wall-floor seam
    frame = np.zeros((200, 200, 3))
    out, mask = place_subject(frame, sheet, 0.5, 0.9, 0.45, erase=False, with_mask=True, ring=0.0)
    ys, xs = np.nonzero(mask)
    # the person alone: 90 px tall scaled to 90 px, 12 px wide to 12 px, the strap kept inside them
    assert abs((ys.max() - ys.min() + 1) - 90) <= 2 and abs((xs.max() - xs.min() + 1) - 12) <= 2


def test_hold_reprojection_dollies_rather_than_zooms(scene):
    from geoblock_core.zoom import reproject_hold, stand_in_on_screen
    route = plan_route(scene, "Earth zoom", 6)
    arrive, last = route["arrive_frame"], route["frames"] - 1
    rows, cols = 54, 96
    ys, xs = np.mgrid[0:rows, 0:cols]
    image = np.dstack([(xs + 0.5) / cols, (ys + 0.5) / rows, np.zeros((rows, cols))])  # each pixel says where it is
    same, seen, _, _ = reproject_hold(scene, route, arrive, arrive, image)
    assert seen.mean() > 0.95 and np.abs(same - image)[seen].max() < 0.02
    later, seen, _, kept = reproject_hold(scene, route, arrive, last, image)
    # a ground point beyond the person lands where the earlier camera saw it
    from geoblock_core.zoom import _camera, mesh_height
    def screen(frame, point):
        p, forward, right, up, focal = _camera(scene, route, frame)
        rel = point - p
        depth = rel @ forward
        return 0.5 + focal * (rel @ right) / depth, 0.5 - focal * (cols / rows) * (rel @ up) / depth
    p, forward, _, _, _ = _camera(scene, route, last)
    sx, sy = route.get("subject_local_m", (0.0, 0.0))
    spot = np.array([sx, sy]) + 40 * forward[:2] / np.linalg.norm(forward[:2])
    point = np.array([*spot, mesh_height(scene, *spot)])
    (ua, va), (ub, vb) = screen(arrive, point), screen(last, point)
    assert 0 < ub < 1 and 0 < vb < 1
    got = later[int(vb * rows), int(ub * cols)]
    assert abs(got[0] - ua) < 0.03 and abs(got[1] - va) < 0.03
    xa, ya, ha = stand_in_on_screen(scene, route, arrive)
    xb, yb, hb = stand_in_on_screen(scene, route, last)
    # the sky only turns with the camera: a top-row span keeps about its width, unlike the person (hb / ha)
    top = later[0, :, 0]
    assert hb / ha > 1.5 and (top[-1] - top[0]) > 0.6


def test_reprojection_blurs_ground_it_stretched():
    from geoblock_core.zoom import _soften_stretched, stretch_weight
    rows, cols = 120, 240
    rng = np.random.default_rng(0)
    image = rng.random((rows, cols, 3))
    ys, xs = np.mgrid[0:rows, 0:cols].astype(float)
    source = np.dstack([xs, ys])
    source[60:, :, 1] = 60 + (ys[60:] - 60) * 0.2  # bottom half drawn out 5x downward: streaks
    weight = stretch_weight(source)
    assert weight[:40].max() == 0 and weight[90:].min() > 0.9
    out = _soften_stretched(image, weight)
    assert np.allclose(out[:40], image[:40])  # untouched where the plate kept its scale
    assert out[90:].std() < 0.3 * image[90:].std()  # only colour left where it was stretched


def test_grain_gives_the_blurred_ground_texture_only():
    from geoblock_core.zoom import _add_grain
    image = np.full((80, 160, 3), 0.5)
    weight = np.zeros((80, 160))
    weight[40:] = 1.0
    out = _add_grain(image, weight, 0.08, seed=3)
    assert np.array_equal(out[:40], image[:40])
    assert 0.05 < out[50:].std() < 0.11 and abs(out[50:].mean() - 0.5) < 0.01
    assert np.array_equal(out, _add_grain(image, weight, 0.08, seed=3))  # reproducible
    assert np.array_equal(_add_grain(image, weight, 0.0), image)


def test_punch_through_dissolves_the_cloud_cut():
    from geoblock_core.zoom import punch_through
    frames = np.ones((40, 16, 16, 3), np.float32)  # white cloud...
    frames[20:] = 0.2  # ...cut to clear ground at f20
    out, cut = punch_through(frames, 10, 30, span=6)
    assert cut == 20
    assert np.allclose(out[:17], frames[:17]) and np.allclose(out[23:], frames[23:])
    means = out[16:24].mean(axis=(1, 2, 3))
    assert np.all(np.diff(means[2:]) <= 1e-6)  # no cut left: it falls off over the span
    assert np.abs(np.diff(means)).max() < 0.35

    smooth = np.linspace(1, 0.2, 40)[:, None, None, None] * np.ones((40, 16, 16, 3), np.float32)
    left, cut = punch_through(smooth, 10, 30)  # a steady clearing, no cut: left alone
    assert cut is None and np.array_equal(left, smooth)


def test_unseen_area_takes_the_blocking_frame():
    from geoblock_core.zoom import _blend_unseen
    image = np.zeros((40, 80, 3))
    seen = np.ones((40, 80), bool)
    seen[20:] = False
    fill = np.ones((20, 40, 3))  # half size: resized to the image
    out = _blend_unseen(image, seen, fill)
    assert np.allclose(out[20:], 1.0) and np.allclose(out[:10], 0.0)
    assert 0.0 < out[19, 40, 0] < 1.0  # feathered into the seen side


def test_ride_floor_is_a_smooth_envelope():
    from geoblock_core.ride import ride_floor
    bumps = [0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    ride = ride_floor(bumps, 4)
    assert all(r >= b - 1e-9 for r, b in zip(ride, bumps))
    steps = np.diff(ride)
    assert np.abs(np.diff(steps)).max() < np.abs(np.diff(np.diff(bumps))).max() / 2
    assert ride_floor(bumps, 0) == bumps


def test_smooth_swoop_brakes_into_a_moving_push_in(scene):
    low = plan_route(scene, "Earth zoom", 10, dive_tempo="low-swoop")
    smooth = plan_route(scene, "Earth zoom", 10, dive_tempo="smooth-swoop")
    assert smooth["dive_tempo"] == "smooth-swoop" and smooth["arrive_frame"] == low["arrive_frame"]

    def speeds(route):  # along the ground, metres a frame
        xy = np.array(route["positions"])[:, :2]
        return np.linalg.norm(np.diff(xy, axis=0), axis=1)

    arrive = low["arrive_frame"]
    window = slice(arrive - 6, arrive + 6)
    # low-swoop stops dead at the arrival; smooth-swoop's speed changes by under a third a frame around it
    assert (speeds(low)[window][1:] / speeds(low)[window][:-1]).min() < 0.1
    ratio = speeds(smooth)[window][1:] / speeds(smooth)[window][:-1]
    assert ratio.min() > 0.7 and ratio.max() < 1.3
    assert speeds(smooth)[-1] > 0.3 * speeds(smooth)[arrive + 2]  # still pushing in at the end
    assert abs(np.linalg.norm(np.array(smooth["positions"])[-1, :2] - smooth["subject_local_m"]) - 2.0) < 0.05
    assert min(smooth["above_ground_m"]) > 1.0


def test_kept_mask_keeps_only_the_seen_little_magnified_view():
    from geoblock_core.zoom import kept_mask
    seen = np.ones((60, 120), bool)
    seen[:, :20] = False
    mag = np.ones((60, 120))
    mag[40:] = 1.6  # the near ground, stretched past the keep
    kept = kept_mask(seen, mag)
    assert kept[:, :20].max() == 0 and kept[40:].max() == 0
    assert kept[10:30, 40:100].min() > 0.99
    assert 0 < kept[38, 60] < 1  # feathered toward the stretched ground


def test_hold_reprojection_hands_the_near_ground_to_the_painter(scene):
    from geoblock_core.zoom import reproject_hold
    route = plan_route(scene, "Earth zoom", 6)
    arrive, last = route["arrive_frame"], route["frames"] - 1
    image = np.full((54, 96, 3), 0.5)
    _, _, plain, kept_plain = reproject_hold(scene, route, arrive, (arrive + last) // 2, image)
    _, _, near, kept_near = reproject_hold(scene, route, arrive, (arrive + last) // 2, image, near_m=8.0)
    assert np.all(near >= plain - 1e-9)
    assert near[-1].max() > 0.8 and near[:10].max() < 0.05  # the ground at the feet, not the sky
    assert kept_near.sum() <= kept_plain.sum() and np.all(kept_near <= 1 - near + 1e-9)
