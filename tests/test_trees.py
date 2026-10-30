import math

import numpy as np
import pytest

from geoblock_core.lidar import node_depth
from geoblock_core.trees import (MAX_SLOPE_DEG, _quadkey, find_trees, lidar_chm, near_field_bounds,
                                 proxy_mesh)


def crown(grid, col, row, height, radius):
    rows, cols = np.indices(grid.shape)
    distance = np.hypot(cols - col, rows - row)
    dome = height * np.sqrt(np.clip(1 - (distance / radius) ** 2, 0, 1))
    np.maximum(grid, np.where(distance <= radius, np.maximum(dome, 0.55 * height), 0), out=grid)


def test_find_trees_recovers_separate_crowns_heights_and_radii():
    chm = np.zeros((80, 80))
    crown(chm, 20, 20, 6.0, 3.0)
    crown(chm, 50, 25, 9.0, 4.0)
    crown(chm, 30, 60, 4.0, 2.0)
    chm[70, 70] = 1.5  # shrub below the 2 m tree threshold
    tops = find_trees(chm)
    assert len(tops) == 3
    by_height = tops[np.argsort(-tops[:, 2])]
    assert np.allclose(by_height[:, :2], [[50, 25], [20, 20], [30, 60]], atol=1)
    assert np.allclose(by_height[:, 2], [9, 6, 4], atol=0.01)
    assert np.allclose(by_height[:, 3], [4, 3, 2], atol=1.0)


def test_find_trees_merges_a_lumpy_crown_into_one_tree():
    chm = np.zeros((40, 40))
    crown(chm, 20, 20, 8.0, 4.0)
    chm[19, 21] = 7.9  # a second, slightly lower bump inside the same crown
    chm[21, 18] = 7.95
    assert len(find_trees(chm)) == 1


def test_lidar_chm_measures_above_sloping_ground_and_masks_cliffs():
    rng = np.random.default_rng(1)
    # 60 x 60 m: a gentle 5 % slope, then a cliff band on the east side.
    gx, gy = rng.uniform(0, 60, 20000), rng.uniform(0, 60, 20000)
    gz = 100 + 0.05 * gy + np.where(gx > 45, (gx - 45) * 3.0, 0)
    tree_x, tree_y = rng.normal(15, 1.0, 400), rng.normal(30, 1.0, 400)
    tree_z = 100 + 0.05 * tree_y + rng.uniform(4.0, 7.0, 400)
    rock_x, rock_y = rng.uniform(46, 60, 2000), rng.uniform(0, 60, 2000)
    rock_z = 100 + 0.05 * rock_y + (rock_x - 45) * 3.0 + 6  # unclassified rock overhang on the cliff
    points = {"x": np.concatenate((gx, tree_x, rock_x)), "y": np.concatenate((gy, tree_y, rock_y)),
              "z": np.concatenate((gz, tree_z, rock_z)),
              "classification": np.concatenate((np.full(20000, 2), np.ones(2400)))}
    chm, slope = lidar_chm(points, (0, 0, 60, 60))
    assert chm.shape == (60, 60)
    assert 6.0 < chm[30 - 1:30 + 2, 14:17].max() <= 7.2  # row 0 is the north edge: y=30 is row 30
    assert slope[:, 50:].min() > MAX_SLOPE_DEG and chm[:, 50:].max() == 0
    assert chm[:, :8].max() == 0  # bare ground away from the tree


def test_lidar_chm_prefers_vegetation_classes_when_present():
    x = np.repeat(np.arange(0.5, 20, 1.0), 20)
    y = np.tile(np.arange(0.5, 20, 1.0), 20)
    ground = {"x": x, "y": y, "z": np.zeros_like(x), "classification": np.full(len(x), 2)}
    extra = {"x": np.array([5.5, 15.5]), "y": np.array([5.5, 15.5]), "z": np.array([6.0, 9.0]),
             "classification": np.array([5, 6])}  # a tree and a building
    points = {key: np.concatenate((ground[key], extra[key])) for key in ground}
    chm, _ = lidar_chm(points, (0, 0, 20, 20))
    assert chm[14, 5] == pytest.approx(6.0) and chm[4, 15] == 0


def test_near_field_bounds_wraps_the_low_path_inside_the_terrain():
    scene = {"origin_projected_m": [1000.0, 2000.0], "radius_m": 1500}
    route = {"positions": [[0, -800, 30], [0, -50, 30]], "look_at": [[0, 400, 0], [0, 400, 0]],
             "target_local_m": [0, 0, 0]}
    x0, y0, x1, y1 = near_field_bounds(scene, route, margin_m=500)
    assert (x0, x1) == (500, 1500)
    assert y0 == 2000 - 1300 and y1 == 2000 + 400 + 500  # the look direction reaches 400 m past the end


def test_near_field_bounds_follows_the_low_frames_of_a_rim_reveal():
    # Skims the plateau at 25 m, then climbs out over the canyon to 650 m.
    scene = {"origin_projected_m": [0.0, 0.0], "radius_m": 12000}
    route = {"positions": [[0, -1600, 25], [0, -1100, 40], [0, 600, 220], [0, 2400, 530], [0, 3000, 650]],
             "look_at": [[0, 300, -40], [0, 2000, -100], [0, 4700, -200], [0, 7000, -300], [0, 8600, -330]],
             "target_local_m": [0, 0, 0]}
    x0, y0, x1, y1 = near_field_bounds(scene, route, margin_m=1000, max_side_m=3500)
    assert y0 <= -2600 and y1 - y0 <= 3500  # the plateau the camera skims is inside
    assert y1 < 2400                         # the high canyon frames are not


def test_near_field_bounds_trims_long_paths_around_the_lowest_point():
    scene = {"origin_projected_m": [0.0, 0.0], "radius_m": 20000}
    route = {"positions": [[0, -9000, 100], [0, 0, 30]], "look_at": [[0, 0, 0], [0, 500, 0]],
             "target_local_m": [0, 0, 0]}
    x0, y0, x1, y1 = near_field_bounds(scene, route, margin_m=1000, max_side_m=3000)
    assert y1 - y0 == 3000 and y0 <= 0 <= y1


def test_proxy_mesh_is_closed_and_stands_on_the_ground():
    vertices, faces = proxy_mesh([(0, 0, 50, 6, 3), (10, 0, 52, 20, 4)])
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts) == {2}  # every edge shared by two triangles: watertight crowns and trunk
    assert vertices[:, 2].max() == pytest.approx(72)
    near_first = np.hypot(vertices[:, 0], vertices[:, 1]) < 5
    assert vertices[near_first, 2].min() == pytest.approx(50 + 0.06 * 6)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    centres = vertices[faces].mean(axis=1)
    owner = np.where(centres[:, 0] < 5, 0, 10)
    outward = (centres[:, 0] - owner) * normals[:, 0] + centres[:, 1] * normals[:, 1]
    assert (outward >= -1e-6).mean() > 0.95


def test_quadkey_and_ept_depth():
    assert _quadkey(-112.1077, 36.0617) == "023102000"  # the Meta canopy tile checked at Mather Point
    assert node_depth([0, 0, 0, 128_000, 128_000, 128_000], 128, 1.0) == 10
    assert node_depth([0, 0, 0, 100, 100, 100], 128, 1.0) == 0


def test_blocking_graph_passes_the_tree_source_only_when_on():
    from geoblock_core.workflow import ShotRequest, blocking_graph
    plain = blocking_graph(ShotRequest(name="x", latitude=36.06, longitude=-112.1))
    assert "trees" not in plain["5"]["inputs"]  # older ComfyUI node sets keep working
    graph = blocking_graph(ShotRequest(name="x", latitude=36.06, longitude=-112.1, trees="auto"))
    assert graph["5"]["inputs"]["trees"] == "auto"


def test_clear_of_drops_trees_on_the_camera_path_and_around_the_person():
    from geoblock_core.trees import clear_of
    standing = [(0, 0, 100, 6, 2), (10, 0, 100, 6, 2), (30, 0, 100, 6, 2), (60, 0, 100, 6, 2)]
    camera = [(10, 1, 103), (30, 0, 140), (60, 30, 101.7)]  # inside a crown; high above one; beside one
    person = [(0, 2.5, 102)]
    kept = clear_of(standing, camera + person)
    assert kept[:, 0].tolist() == [30, 60]


def test_canopy_words_describe_the_measured_trees_and_reach_both_prompts():
    from geoblock_core.looks import canopy_text, keyframe_prompt, render_to_real_prompt
    from geoblock_core.trees import canopy_summary
    trees = [[0, 0, h, 2] for h in [3, 4, 5, 5, 6, 6, 7, 8, 9, 22]]
    summary = canopy_summary({"trees": trees, "count": 10, "canopy_cover": 0.36, "source": "lidar"})
    words = canopy_text(summary, "grand-canyon")
    assert "36%" in words and "pinyon pines and junipers" in words and "ponderosa pines" in words
    situation = {"over": "top", "above_ground_m": 25}
    key = keyframe_prompt("Mather Point", "grand-canyon", "Clear sky.", situation, canopy=words)
    assert words in key and "pale limestone ground between them" not in key.split(words)[0]
    stock = keyframe_prompt("Mather Point", "grand-canyon", "Clear sky.", situation)
    assert "36%" not in stock
    ltx = render_to_real_prompt("Mather Point", "grand-canyon", "Skim low.", "Clear sky.", canopy=words)
    assert "36%" in ltx and "keep their places and sizes" in ltx and "Image 1" not in ltx


def test_keyframe_nodes_get_the_canopy_words_only_when_measured():
    from geoblock_core.workflow import ShotRequest, ltx_graph
    request = ShotRequest(name="x", latitude=36.06, longitude=-112.1, keyframe_model="flux-klein", ltx_version="2.5")
    plain = ltx_graph(request)
    assert not any("canopy" in node["inputs"] for node in plain.values()
                   if node["class_type"] == "GeoBlockKeyframePrompt")
    request.canopy = "the flat ground is woodland"
    graph = ltx_graph(request)
    prompts = [node for node in graph.values() if node["class_type"] == "GeoBlockKeyframePrompt"]
    assert prompts and all(node["inputs"]["canopy"] == request.canopy for node in prompts)


def test_feather_edges_thins_the_border_and_keeps_the_middle():
    from geoblock_core.trees import feather_edges
    rng = np.random.default_rng(3)
    trees = np.column_stack((rng.uniform(0, 3000, 40000), rng.uniform(0, 3000, 40000),
                             np.full(40000, 8.0), np.full(40000, 2.0)))
    kept = feather_edges(trees, band_m=500)
    inside = np.minimum(kept[:, :2], 3000 - kept[:, :2]).min(axis=1)
    middle = (inside > 520).sum() / (np.minimum(trees[:, :2], 3000 - trees[:, :2]).min(axis=1) > 520).sum()
    assert middle == 1.0
    middle_density = (inside > 520).sum() / (3000 - 2 * 520) ** 2
    edge_density = (inside < 50).sum() / (3000 ** 2 - 2900 ** 2)
    assert edge_density < 0.05 * middle_density
    assert np.array_equal(kept, feather_edges(trees, band_m=500))  # same trees every render


def test_lidar_chm_reports_holes_in_the_survey():
    x = np.repeat(np.arange(0.5, 100, 1.0), 100)
    y = np.tile(np.arange(0.5, 100, 1.0), 100)
    hole = (x > 40) & (x < 80) & (y > 20) & (y < 60)  # a missing tile
    points = {"x": x[~hole], "y": y[~hole], "z": np.zeros((~hole).sum()), "classification": np.full((~hole).sum(), 2)}
    _, _, void = lidar_chm(points, (0, 0, 100, 100), voids=True)
    assert void[50, 60] and not void[50, 20] and not void[90, 60]  # row 50 is y = 49.5
    assert 0.08 < void.mean() < 0.17


def test_building_mask_keeps_roofs_out_of_the_canopy(tmp_path):
    from geoblock_core.trees import building_mask
    # One 10 m square building at local (20..30, 20..30), scene origin at projected (1000, 2000).
    ring = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], "float32")
    np.savez_compressed(tmp_path / "b.npz", xy=ring, offsets=np.array([0, 4]), base_m=np.zeros(1, "float32"),
                        height_m=np.full(1, 8.0, "float32"))
    manifest = {"origin_projected_m": [1000.0, 2000.0], "data": str(tmp_path / "b.npz")}
    mask = building_mask(manifest, (1000.0, 2000.0, 1050.0, 2050.0), 1.0, margin_m=1.0)
    assert mask.shape == (50, 50)
    # Rows run north to south: local north 25 is row 50 - 25 = 25.
    assert mask[25, 25] and mask[50 - 21, 20] and not mask[25, 40] and not mask[5, 5]
    assert 100 <= mask.sum() <= 12 * 12 + 8


def test_lidar_chm_streams_dense_surveys_without_keeping_the_points():
    import tracemalloc

    def parts(n_parts=40, size=100_000, seed=3):
        rng = np.random.default_rng(seed)
        for _ in range(n_parts):  # 4 M points over 100 x 100 m, about a city survey's density
            x, y = rng.uniform(0, 100, size), rng.uniform(0, 100, size)
            cls = rng.choice([2, 1, 5], size, p=[0.6, 0.2, 0.2])
            z = np.where(cls == 2, 0.0, np.where(cls == 5, 8.0, 20.0)) + rng.uniform(0, 0.5, size)
            yield {"x": x, "y": y, "z": z, "classification": cls}

    tracemalloc.start()
    chm, _ = lidar_chm(parts(), (0, 0, 100, 100))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 30e6  # keeping every candidate point needed about 120 MB here
    assert 7.5 < np.median(chm) < 9.0  # vegetation tops, not the 20 m class-1 "roofs"
    whole = {key: np.concatenate([part[key] for part in parts(4, 25_000)]) for key in ("x", "y", "z", "classification")}
    np.testing.assert_allclose(lidar_chm(whole, (0, 0, 100, 100))[0], lidar_chm(parts(4, 25_000), (0, 0, 100, 100))[0])


def test_canopy_words_around_an_open_viewpoint_say_it_is_open():
    from geoblock_core.looks import canopy_text
    from geoblock_core.trees import canopy_summary
    # A wooded area with a bare summit at the origin (Twin Peaks): 27% cover overall, 2 small trees near the top.
    rng = np.random.default_rng(0)
    far = np.column_stack((rng.uniform(400, 1500, 3000), rng.uniform(-1500, 1500, 3000),
                           rng.uniform(5, 40, 3000), np.full(3000, 3.0)))
    near = np.array([[60, 40, 4, 2], [-90, 120, 5, 2]])
    manifest = {"trees": np.vstack((far, near)).tolist(), "count": 3002, "canopy_cover": 0.27, "source": "lidar"}
    assert "woodland" in canopy_text(canopy_summary(manifest), "default")
    summary = canopy_summary(manifest, around=(0, 0))
    assert summary["count"] == 2 and summary["cover"] < 0.01
    words = canopy_text(summary, "default", "green grass")
    assert "open ground" in words and "only 2 scattered trees" in words and "conifers" not in words


def test_sparse_trees_around_a_viewpoint_get_no_species_or_heights():
    from geoblock_core.looks import canopy_text
    summary = {"cover": 0.11, "count": 40, "around_m": 250, "height_p25_m": 3, "height_p75_m": 9,
               "height_p95_m": 17, "tall_share": 0.2, "source": "lidar"}
    words = canopy_text(summary, "default", "green grass")
    assert "just a few small trees and bushes" in words and words.count("Image 1") == 2
    assert "conifers" not in words and "17 m" not in words
