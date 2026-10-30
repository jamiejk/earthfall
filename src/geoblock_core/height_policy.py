"""Eligibility checks for measured geometry used in every 3D blocking shot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualify_structure_heights(manifest: dict, target_way_id: str,
                              minimum_radius_m: float = 0) -> str:
    """Accept an urban block only with measured heights for the target and area."""
    if manifest.get("source_type") not in {"LPC", "DSM"}:
        raise ValueError("Building shots require a point cloud or surface model with structure heights; a bare-earth DEM or aerial image is insufficient.")
    if not manifest.get("has_measured_structure_height"):
        raise ValueError("Height source has no measured building elevations in this area.")
    offset = manifest.get("ground_median_difference_m")
    if not isinstance(offset, (int, float)) or not math.isfinite(offset) or abs(offset) > 1:
        raise ValueError("Building and ground height datums are not aligned.")
    if manifest.get("radius_m", 0) < minimum_radius_m:
        raise ValueError("Measured building coverage is smaller than the blocking camera area.")
    buildings = manifest.get("buildings", [])
    target = next((b for b in buildings if b.get("osm_way_id") == target_way_id), None)
    target_samples = (target.get("roof_returns", target.get("surface_samples", 0))
                      if target else 0)
    if (not target or target.get("status") != "accepted" or
            target_samples < (20 if manifest["source_type"] == "LPC" else 9) or
            not isinstance(target.get("height_m"), (int, float)) or target["height_m"] < 2):
        raise ValueError("The target landmark has no reliable measured roof height.")
    coverage = sum(b.get("status") == "accepted" for b in buildings) / max(1, len(buildings))
    if coverage < 0.7:
        raise ValueError(f"Only {coverage:.0%} of nearby buildings have usable heights; blocking shots need at least 70%.")
    return (f"{len(buildings)} footprints within {manifest.get('radius_m', '?')} m; "
            f"{coverage:.0%} accepted; target height {target['height_m']:.2f} m")


def measure_landform_evidence(terrain_dir: Path, target_m: tuple[float, float],
                              landmark: str, radius_m: float = 150) -> dict:
    """Measure relief around a landmark in a compiled terrain and record it."""
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    terrain = json.loads((terrain_dir / "metadata.json").read_text())
    if terrain["source"]["source_type"] not in {"DTM", "DSM", "LPC"}:
        raise ValueError("Landmark source is not a measured elevation dataset")
    x, y = target_m
    raster = terrain_dir / "terrain_metric.tif"
    with rasterio.open(raster) as dataset:
        center = next(dataset.sample([(x, y)], masked=True))[0]
        if np.ma.is_masked(center) or not np.isfinite(center):
            raise ValueError("No measured height at the landmark")
        window = from_bounds(x - radius_m, y - radius_m, x + radius_m, y + radius_m,
                             dataset.transform).round_offsets().round_lengths()
        neighborhood = dataset.read(1, window=window, masked=True)
    valid = neighborhood.compressed()
    coverage = len(valid) / neighborhood.size
    relief = float(np.percentile(valid, 95) - np.percentile(valid, 5)) if len(valid) else 0.0
    if coverage < 0.1 or relief < 3:
        raise ValueError("Landmark has insufficient measured 3D relief")
    record = {
        "schema_version": "1.0", "feature_kind": "landform",
        "landmark": landmark, "target_projected_m": [x, y],
        "target_height_m": float(center), "neighborhood_radius_m": radius_m,
        "valid_fraction": round(coverage, 4), "relief_p95_minus_p5_m": round(relief, 3),
        "source_type": terrain["source"]["source_type"],
        "source_dataset": terrain["source"]["dataset"],
        "height_alignment": terrain["height_alignment"],
        "terrain_mesh_sha256": _sha256(terrain_dir / "terrain.glb"),
        "terrain_raster_sha256": _sha256(raster),
    }
    if terrain["height_alignment"] != "verified":
        record["height_limitation"] = ("Relative landform relief is measured; "
                                       "the source vertical datum remains unverified.")
    (terrain_dir / "landform_height_evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def qualify_landform_heights(evidence: dict, terrain_path: Path,
                             expected_target_m: tuple[float, float] | None = None) -> str:
    """A landform DTM qualifies only with measured relief at the landmark."""
    if evidence.get("feature_kind") != "landform" or evidence.get("source_type") not in {"DTM", "DSM", "LPC"}:
        raise ValueError("Landform blocking requires measured terrain or point-cloud height evidence.")
    height = evidence.get("target_height_m")
    if not isinstance(height, (int, float)) or not math.isfinite(height):
        raise ValueError("The target landmark has no measured elevation.")
    if evidence.get("valid_fraction", 0) < 0.1 or evidence.get("relief_p95_minus_p5_m", 0) < 3:
        raise ValueError("The target landmark has insufficient measured 3D relief.")
    if expected_target_m is not None:
        measured_target = evidence.get("target_projected_m", [])
        if (len(measured_target) != 2 or
                math.hypot(measured_target[0] - expected_target_m[0],
                           measured_target[1] - expected_target_m[1]) > 1):
            raise ValueError("Landmark-height evidence is for a different target.")
    if _sha256(terrain_path) != evidence.get("terrain_mesh_sha256"):
        raise ValueError("The landform height evidence does not match the terrain mesh.")
    return (f"measured {evidence['source_type']} at landmark; "
            f"{evidence['relief_p95_minus_p5_m']:.1f} m local relief")


def qualify_pack_geometry(pack: dict) -> str:
    """Reject H3 reference packs without a qualified measured 3D source."""
    geometry = pack.get("geometry_evidence", {})
    evidence_path = Path(geometry.get("evidence_path", ""))
    if not evidence_path.is_file():
        raise ValueError("Reference pack has no measured 3D height evidence.")
    evidence = json.loads(evidence_path.read_text())
    if geometry.get("kind") == "landform":
        target = geometry.get("target_projected_m")
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError("Reference pack has no landmark target coordinates.")
        return qualify_landform_heights(evidence, Path(geometry["terrain_path"]), tuple(target))
    if geometry.get("kind") == "building":
        return qualify_structure_heights(evidence, str(geometry["target_osm_way_id"]),
                                         float(geometry["minimum_radius_m"]))
    raise ValueError("Reference pack has no qualified measured 3D geometry kind.")
