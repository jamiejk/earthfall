"""Landmark choice must preserve identity and never claim geometry coverage."""

import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


path = Path(__file__).resolve().parents[1] / "comfy_geoblock_prototype/landmark_search.py"
spec = importlib.util.spec_from_file_location("geoblock_landmark_search", path)
search = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = search
spec.loader.exec_module(search)


class Response(io.BytesIO):
    pass


def test_user_search_can_choose_second_ambiguous_result_and_cache(monkeypatch, tmp_path):
    payload = {"features": [
        {"properties": {"name": "Example Bridge", "city": "Harbour", "country": "Exampleland",
                        "osm_key": "highway", "osm_value": "motorway", "osm_type": "W", "osm_id": 11},
         "geometry": {"type": "Point", "coordinates": [10, 20]}},
        {"properties": {"name": "Example Bridge", "city": "Harbour", "country": "Exampleland",
                        "osm_key": "bridge", "osm_value": "yes", "osm_type": "W", "osm_id": 12},
         "geometry": {"type": "Point", "coordinates": [10.1, 20.1]}},
        {"properties": {"name": "Example Bridge", "osm_type": "W", "osm_id": 11},
         "geometry": {"type": "Point", "coordinates": [10, 20]}},
    ]}
    calls = []

    def fake_open(request, timeout):
        calls.append(request.full_url)
        assert request.get_header("User-agent").startswith("GeoBlock/")
        return Response(json.dumps(payload).encode())

    monkeypatch.setattr(search, "urlopen", fake_open)
    results, report = search.search_landmarks("Example Bridge, Harbour", cache_dir=tmp_path)
    assert len(results["candidates"]) == 2
    assert "1. Example Bridge" in report and "2. Example Bridge" in report
    assert len(calls) == 1

    monkeypatch.setattr(search, "urlopen", lambda *args, **kwargs: pytest.fail("cache missed"))
    cached, cached_report = search.search_landmarks("Example Bridge, Harbour", cache_dir=tmp_path)
    assert cached == results and "(cached)" in cached_report

    place, selection = search.choose_landmark(results, 2, 1.5)
    assert (place["longitude"], place["latitude"]) == (10.1, 20.1)
    assert place["source_id"] == "W12"
    assert "not checked" in selection


def test_invalid_choice_or_endpoint_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="result number"):
        search.choose_landmark({"candidates": [{}]}, 2, 1)
    monkeypatch.setenv("GEOBLOCK_PHOTON_URL", "http://example.com/api")
    with pytest.raises(ValueError, match="HTTPS"):
        search.search_landmarks("Some landmark", cache_dir=tmp_path)


def test_user_landmark_cannot_reuse_a_nearby_example_scene():
    core_path = Path(__file__).resolve().parents[1] / "comfy_geoblock_prototype/core.py"
    core_spec = importlib.util.spec_from_file_location("geoblock_fixture_core", core_path)
    core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core)
    selected = {"latitude": core.DEMO_LAT, "longitude": core.DEMO_LON,
                "requested_radius_km": 1.0, "source": "Photon/OpenStreetMap"}
    with pytest.raises(ValueError, match="prepared examples only"):
        core.load_demo_terrain(selected)
