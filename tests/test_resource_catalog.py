"""Discovery should expose geometry-capable leads without claiming shot readiness."""

import importlib.util
from pathlib import Path
import sys


path = Path(__file__).resolve().parents[1] / "comfy_geoblock_prototype/resource_catalog.py"
spec = importlib.util.spec_from_file_location("geoblock_resource_catalog", path)
catalog = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = catalog
spec.loader.exec_module(catalog)


def test_alphabetical_categories_and_registry_warning():
    matches, report = catalog.discover_resources()
    assert [item.name for item in matches] == sorted((item.name for item in matches), key=str.casefold)
    assert "Open 3D City Model Catalog [Catalogs]" in report
    assert "Discovery only — no shot approval" in report


def test_place_search_only_returns_relevant_geometry_source():
    matches, report = catalog.discover_resources("All", "Wales landform")
    assert [item.name for item in matches] == ["Welsh Government LiDAR DTM"]
    assert "Measured bare-earth relief" in report
    matches, report = catalog.discover_resources("3D city models", "Woodside")
    assert matches == []
    assert "no flat substitute" in report


def test_city_dataset_candidates_and_restricted_mesh_exclusion():
    matches, report = catalog.discover_resources("3D city models", "Hamburg")
    assert [item.name for item in matches] == ["Hamburg LoD3.0-HH"]
    assert "target tile unchecked" in report
    names = [item.name for item in catalog.discover_resources("3D city models")[0]]
    assert "Helsinki 3D Mesh" in names
    assert not any("Berlin" in name for name in names)
