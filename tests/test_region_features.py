"""OpenStreetMap land cover and water become one plain sentence for the aerial prompts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from touchdown.region_facts import describe_features

LAT, LON = 54.1205, -0.9068


def square(dlat, dlon, size, tags):
    points = [dict(lat=LAT + dlat + a * size, lon=LON + dlon + b * size) for a, b in ((0, 0), (0, 1), (1, 1), (1, 0), (0, 0))]
    return dict(tags=tags, geometry=points)


def test_lake_cover_and_direction():
    facts = describe_features([square(.005, -.005, .004, {"natural": "water", "name": "Great Lake"}),
                               square(-.02, -.02, .03, {"landuse": "farmland"})], LAT, LON)
    assert facts["local_cover"] == ["farmland"]
    assert facts["local_lakes"][0]["direction"] == "north" and 5 < facts["local_lakes"][0]["area_ha"] < 50
    assert "a lake of about" in facts["local_features"]


def test_no_water_is_said_plainly():
    assert "without lakes or rivers" in describe_features([], LAT, LON)["local_features"]


def test_multipolygon_lake_is_stitched():
    a = [dict(lat=LAT + .002, lon=LON), dict(lat=LAT + .002, lon=LON + .006), dict(lat=LAT + .006, lon=LON + .006)]
    b = [dict(lat=LAT + .006, lon=LON), dict(lat=LAT + .006, lon=LON + .006)][::-1] + [dict(lat=LAT + .002, lon=LON)]
    relation = dict(tags={"natural": "water", "name": "Great Lake"}, members=[
        dict(role="outer", geometry=a), dict(role="outer", geometry=b)])
    lake = describe_features([relation], LAT, LON)["local_lakes"][0]
    assert lake["name"] == "Great Lake" and lake["area_ha"] > 10
