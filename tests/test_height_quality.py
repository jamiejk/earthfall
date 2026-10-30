"""Every blocking shot needs measured target height, not only close shots."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


module_path = Path(__file__).resolve().parents[1] / "comfy_geoblock_prototype/core.py"
spec = importlib.util.spec_from_file_location("geoblock_comfy_core", module_path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def valid_manifest():
    return {
        "source_type": "LPC",
        "has_measured_structure_height": True,
        "ground_median_difference_m": 0.04,
        "buildings": [
            {"osm_way_id": str(i), "status": "accepted" if i < 8 else "rejected",
             "roof_returns": 100, "height_m": 5.2}
            for i in range(10)
        ],
    }


def test_close_shot_accepts_complete_lidar_height_evidence():
    assert "80% accepted" in core.qualify_structure_heights(valid_manifest(), "0")
    with pytest.raises(ValueError, match="smaller than the blocking camera area"):
        core.qualify_structure_heights(valid_manifest() | {"radius_m": 200}, "0", 700)


def test_surface_model_with_measured_roof_pixels_can_qualify():
    manifest = valid_manifest() | {"source_type": "DSM"}
    for building in manifest["buildings"]:
        building.pop("roof_returns")
        building["surface_samples"] = 12
    assert "80% accepted" in core.qualify_structure_heights(manifest, "0")


@pytest.mark.parametrize("change,message", [
    ({"source_type": "DTM"}, "bare-earth DEM"),
    ({"has_measured_structure_height": False}, "no measured building"),
    ({"ground_median_difference_m": 1.5}, "datums are not aligned"),
])
def test_close_shot_rejects_ineligible_source(change, message):
    manifest = valid_manifest() | change
    with pytest.raises(ValueError, match=message):
        core.qualify_structure_heights(manifest, "0")


def test_close_shot_rejects_missing_target_and_low_coverage():
    with pytest.raises(ValueError, match="target landmark"):
        core.qualify_structure_heights(valid_manifest(), "missing")
    manifest = valid_manifest()
    for building in manifest["buildings"][5:]:
        building["status"] = "rejected"
    with pytest.raises(ValueError, match="at least 70%"):
        core.qualify_structure_heights(manifest, "0")


def test_landform_requires_measured_relief_and_matching_mesh(tmp_path):
    mesh = tmp_path / "terrain.glb"
    mesh.write_bytes(b"measured landform mesh")
    evidence = {
        "feature_kind": "landform", "source_type": "DTM", "target_height_m": 5.8,
        "valid_fraction": 0.4, "relief_p95_minus_p5_m": 36.7,
        "terrain_mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    }
    assert "36.7 m local relief" in core.qualify_landform_heights(evidence, mesh)
    with pytest.raises(ValueError, match="different target"):
        core.qualify_landform_heights(evidence | {"target_projected_m": [10, 20]}, mesh, (12, 20))
    with pytest.raises(ValueError, match="insufficient measured 3D relief"):
        core.qualify_landform_heights(evidence | {"relief_p95_minus_p5_m": 0}, mesh)
    mesh.write_bytes(b"flat replacement")
    with pytest.raises(ValueError, match="does not match"):
        core.qualify_landform_heights(evidence, mesh)


def test_reference_pack_requires_height_evidence(tmp_path):
    from geoblock_core.height_policy import qualify_pack_geometry

    with pytest.raises(ValueError, match="no measured 3D height evidence"):
        qualify_pack_geometry({"motion_reference": "flat.mp4"})
    mesh = tmp_path / "terrain.glb"
    mesh.write_bytes(b"measured landform mesh")
    evidence_path = tmp_path / "height.json"
    evidence_path.write_text(json.dumps({
        "feature_kind": "landform", "source_type": "DTM", "target_height_m": 5.8,
        "valid_fraction": 0.4, "relief_p95_minus_p5_m": 36.7,
        "terrain_mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "target_projected_m": [10, 20],
    }))
    pack = {"geometry_evidence": {"kind": "landform", "evidence_path": str(evidence_path),
                                  "terrain_path": str(mesh), "target_projected_m": [10, 20]}}
    assert "36.7 m local relief" in qualify_pack_geometry(pack)


def test_woodside_auto_route_requires_measured_blocks_and_rejects_flat_orbit(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "WOODSIDE_BASE", tmp_path)
    (tmp_path / "woodside_library_measured_blocking_flyover.mp4").write_bytes(b"control")
    manifest = valid_manifest() | {"radius_m": 200}
    manifest["buildings"][0]["osm_way_id"] = "382886306"
    path = tmp_path / "measured_buildings.json"
    path.write_text(json.dumps(manifest))
    scene = {"demo_only": True, "fixture": "woodside_library",
             "geometry_evidence_path": str(path), "terrain_path": str(tmp_path / "terrain.glb")}
    shot, _ = core.plan_shot(scene, "Auto route", 5.2)
    assert shot["route"] == "Close measured-building orbit"
    with pytest.raises(ValueError, match="not available"):
        core.plan_shot(scene, "Address orbit", 5.2)
    with pytest.raises(ValueError, match="not a qualified 3D blocking reference"):
        core.build_pack(scene, shot | {"source_video": str(tmp_path / "flat.mp4")},
                        "Clear-line comic", "Not applicable", tmp_path)
