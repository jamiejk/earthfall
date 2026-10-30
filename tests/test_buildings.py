import numpy as np
import pytest

shapely = pytest.importorskip("shapely")

from geoblock_core.buildings import (DEFAULT_HEIGHT_M, FLOOR_M, building_height, city_text, clear_of_path,
                                     extrude, footprint_rings)
from geoblock_core.looks import without_image_cues


def square(x, y, size):
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=float)


def test_building_height_prefers_measured_then_floors_then_default():
    assert building_height(42.0, 3) == (42.0, "height")
    assert building_height(None, 3) == (3 * FLOOR_M, "floors")
    assert building_height(float("nan"), None) == (DEFAULT_HEIGHT_M, "default")


def test_extrude_makes_closed_outward_prisms():
    rings = [square(0, 0, 10), square(20, 0, 5)[:3]]  # a box and a triangle
    vertices, loops, starts, totals = extrude(rings, np.array([0.0, 1.0]), np.array([30.0, 4.0]))
    assert len(vertices) == 2 * (4 + 3)
    assert list(totals) == [4, 4, 4, 4, 4, 3, 4, 4, 4]  # roof then walls, per building
    assert starts[-1] + totals[-1] == len(loops)
    for start, total in zip(starts, totals):
        face = vertices[loops[start:start + total]]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        centre = face.mean(axis=0)
        owner = rings[0] if centre[0] < 15 else rings[1]
        if total > 4 or np.allclose(face[:, 2], face[0, 2]):  # roof: faces up
            assert normal[2] > 0
        else:  # wall: faces away from the footprint's middle
            assert np.dot(normal[:2], centre[:2] - owner.mean(axis=0)) > 0


def test_footprint_rings_projects_orients_and_drops_tiny():
    big = shapely.Polygon([(0, 0), (0, 20), (20, 20), (20, 0)])  # clockwise
    tiny = shapely.Polygon([(50, 50), (52, 50), (52, 52), (50, 52)])
    multi = shapely.MultiPolygon([shapely.Polygon(square(100, 100, 30)), shapely.Polygon(square(200, 200, 5))])
    rings = footprint_rings([big, tiny, multi, None], lambda coords: coords * 1.0 + 1.0)
    assert rings[1] is None and rings[3] is None
    assert len(rings[0]) == 4 and shapely.Polygon(rings[0]).exterior.is_ccw
    assert rings[0].min() == 1.0  # the transform was applied
    assert rings[2].max() == pytest.approx(131.0)  # the multipolygon kept its larger part


def test_clear_of_path_drops_only_blocks_reaching_the_camera():
    rings = [square(0, 0, 10), square(500, 0, 10), square(0, 100, 10)]
    tops = np.array([80.0, 300.0, 20.0])
    keep = clear_of_path(rings, tops, [(5, 5, 90), (5, 105, 90)])
    assert list(keep) == [False, True, True]


def test_city_words_and_video_prompt_cue():
    text = city_text({"count": 9000, "height_p50_m": 7.0, "height_max_m": 326.0, "towers_over_100m": 50})
    assert "326 m" in text and "Each pale block in Image 1" in text
    video = without_image_cues(f"the land; {text}; the skyline beyond")
    assert "Image 1" not in video and video.endswith("; the skyline beyond") and "326 m" in video


def test_city_words_say_rows_of_houses_when_most_footprints_are_house_sized():
    from geoblock_core.buildings import footprint_areas, footprint_stats
    rings = [square(0, 0, 10)] * 7 + [square(0, 0, 80)] * 3  # 10 x 10 m houses and 80 x 80 m blocks
    stats = footprint_stats(footprint_areas(rings))
    assert stats == {"footprint_p50_m2": 100, "small_share": 0.7}
    summary = {"count": 9000, "height_p50_m": 7.0, "height_max_m": 30.0, "towers_over_100m": 0}
    assert "long rows" in city_text({**summary, **stats}) and "long rows" not in city_text(summary)
