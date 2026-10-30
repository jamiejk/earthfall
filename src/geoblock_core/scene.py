"""Build a qualified landform scene for any user-chosen place."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from pyproj import Transformer

from .height_policy import measure_landform_evidence, qualify_landform_heights
from .models import GeoSceneSpec
from .processing import compile_terrain
from .providers import CopernicusDEMProvider, USGS3DEPProvider

SCENE_VERSION = "1"


def nad83_utm_crs(longitude: float, latitude: float) -> str:
    """NAD83 / UTM zone (EPSG:26901–26923) for northern-hemisphere North America."""
    zone = int((longitude + 180) // 6) + 1
    if latitude < 0 or not 1 <= zone <= 23:
        raise ValueError("NAD83 / UTM is defined here for North America only")
    return f"EPSG:{26900 + zone}"


def wgs84_utm_crs(longitude: float, latitude: float) -> str:
    """WGS 84 / UTM zone (EPSG:326xx north, 327xx south) for anywhere outside the polar caps."""
    if not -80 <= latitude <= 84:
        raise ValueError("UTM does not cover the polar regions")
    zone = min(int((longitude + 180) // 6) + 1, 60)
    return f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"


def in_conus(latitude: float, longitude: float, margin_deg: float = 0.0) -> bool:
    w, s, e, n = USGS3DEPProvider.conus
    return w + margin_deg <= longitude <= e - margin_deg and s + margin_deg <= latitude <= n - margin_deg


# How each terrain family is referenced: the projected CRS, the height datum and the variant name.
TERRAIN_FAMILIES = {
    "usgs3dep": {"vertical_crs": "EPSG:5703", "datum": "NAVD88", "variant": "USGS_3DEP_13"},
    "copernicus": {"vertical_crs": "EPSG:3855", "datum": "EGM2008", "variant": "COPERNICUS_GLO30"},
}


def terrain_family(latitude: float, longitude: float, radius_m: float = 0.0) -> str:
    """USGS 3DEP (10 m, bare earth) inside the conterminous US, Copernicus GLO-30 (30 m) elsewhere."""
    margin = radius_m / 111_000 + 0.01
    return "usgs3dep" if in_conus(latitude, longitude, margin) else "copernicus"


def default_provider(family: str, raster_spacing_m: float):
    if family == "usgs3dep":
        return USGS3DEPProvider(read_resolution_deg=raster_spacing_m / 111_000 / 2 if raster_spacing_m > 10 else None)
    return CopernicusDEMProvider(read_resolution_deg=raster_spacing_m / 111_000 / 2 if raster_spacing_m > 30 else None)


def plan_spacing(radius_m: float, raster_spacing_m: float = 10,
                 max_mesh_cells: int = 1200) -> tuple[float, float]:
    """Round the radius up to the mesh grid and keep the mesh near max_mesh_cells²."""
    if radius_m <= 0:
        raise ValueError("radius must be positive")
    factor = max(1, math.ceil(2 * radius_m / raster_spacing_m / max_mesh_cells))
    mesh_spacing = raster_spacing_m * factor
    radius = math.ceil(radius_m / mesh_spacing) * mesh_spacing
    return float(radius), float(mesh_spacing)


def slug(name: str, latitude: float, longitude: float) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:40] or "place"
    return f"{text}-{latitude:.4f}-{longitude:.4f}".replace(".", "p")


def build_landform_scene(*, name: str, latitude: float, longitude: float, radius_m: float,
                         output_root: Path, raw_cache: Path, location_source: str,
                         provider=None, evidence_radius_m: float = 300, raster_spacing_m: float = 10) -> dict:
    """Compile terrain around a landmark and qualify its measured relief.

    The landmark coordinate is the scene origin, so every camera preset can aim
    at local (0, 0, target height). Reuses a previous build when its inputs and
    files are unchanged. Terrain is USGS 3DEP inside the conterminous US and
    Copernicus GLO-30 elsewhere, unless a provider is given.
    """
    if provider is None:
        family = terrain_family(latitude, longitude, radius_m)
        provider = default_provider(family, raster_spacing_m)
    else:
        family = "copernicus" if isinstance(provider, CopernicusDEMProvider) else "usgs3dep"
    reference = TERRAIN_FAMILIES[family]
    projected = nad83_utm_crs(longitude, latitude) if family == "usgs3dep" else wgs84_utm_crs(longitude, latitude)
    radius, mesh_spacing = plan_spacing(radius_m, raster_spacing_m)
    project_id = slug(name, latitude, longitude)
    terrain_dir = output_root.resolve() / f"{project_id}-r{int(radius)}"
    scene_path = terrain_dir / "scene.json"
    request = {"version": SCENE_VERSION, "latitude": latitude, "longitude": longitude,
               "radius_m": radius, "mesh_spacing_m": mesh_spacing, "provider": type(provider).__name__,
               **({"raster_spacing_m": raster_spacing_m} if raster_spacing_m != 10 else {})}
    if scene_path.is_file():
        scene = json.loads(scene_path.read_text())
        if scene.get("request") == request:
            try:
                qualify_scene(scene)
                return scene
            except (ValueError, FileNotFoundError):
                pass
    x, y = Transformer.from_crs("EPSG:4326", projected, always_xy=True).transform(longitude, latitude)
    spec = GeoSceneSpec(project_id=project_id, name=name, longitude=longitude, latitude=latitude,
                        projected_crs=projected, vertical_crs=reference["vertical_crs"], radius_m=radius,
                        origin_easting_m=x, origin_northing_m=y, location_source=location_source,
                        origin_source="chosen landmark coordinate")
    terrain_dir.mkdir(parents=True, exist_ok=True)
    (terrain_dir / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n")
    asset = compile_terrain(spec, provider, terrain_dir, raw_cache, reference["variant"], raster_spacing_m,
                            mesh_spacing)
    evidence = measure_landform_evidence(terrain_dir, (x, y), name, evidence_radius_m)
    scene = {
        "request": request,
        "project_id": project_id,
        "name": name,
        "place": {"latitude": latitude, "longitude": longitude, "source": location_source},
        "terrain_dir": str(terrain_dir),
        "terrain_path": str(terrain_dir / "terrain.glb"),
        "terrain_raster": str(terrain_dir / "terrain_metric.tif"),
        "preview_path": str(terrain_dir / "preview_topdown.png"),
        "geometry_evidence_path": str(terrain_dir / "landform_height_evidence.json"),
        "projected_crs": projected,
        "vertical_crs": spec.vertical_crs,
        "vertical_datum": reference["datum"],
        "terrain_family": family,
        "source_type": asset.source.source_type,
        "origin_projected_m": [x, y],
        "radius_m": radius,
        "mesh_spacing_m": mesh_spacing,
        "target_height_m": evidence["target_height_m"],
        "elevation_range_m": [asset.elevation_min_m, asset.elevation_max_m],
        "source_dataset": asset.source.dataset,
        "attribution": asset.source.attribution,
        "height_alignment": asset.height_alignment,
        "height_model_kind": "landform",
    }
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    qualify_scene(scene)
    return scene


def qualify_scene(scene: dict) -> str:
    """Re-check a built scene's measured relief against its current mesh."""
    if scene.get("height_model_kind") != "landform":
        raise ValueError("Only landform scenes are supported by the generic builder.")
    evidence = json.loads(Path(scene["geometry_evidence_path"]).read_text())
    return qualify_landform_heights(evidence, Path(scene["terrain_path"]),
                                    tuple(scene["origin_projected_m"]))


def scene_report(scene: dict) -> str:
    lo, hi = scene["elevation_range_m"]
    return (
        f"SCENE: {scene['name']}\n"
        f"AREA: {2 * scene['radius_m'] / 1000:g} × {2 * scene['radius_m'] / 1000:g} km, "
        f"{scene['projected_crs']}, mesh {scene['mesh_spacing_m']:g} m\n"
        f"SOURCE: {scene['source_dataset']}\n"
        f"HEIGHTS: {scene.get('vertical_datum', 'NAVD88')} {lo:.0f}–{hi:.0f} m ({scene['height_alignment']}); "
        f"landmark {scene['target_height_m']:.0f} m\n"
        f"3D HEIGHT EVIDENCE: {qualify_scene(scene)}\n"
        f"ATTRIBUTION: {scene['attribution']}"
    )


LANDING_RADIUS_M = 400.0


def build_landing_scene(scene: dict, *, project: str, zone: int, output_root: Path, raw_cache: Path,
                        radius_m: float = LANDING_RADIUS_M) -> dict:
    """A 1 m lidar terrain patch around the landmark, nested inside the local scene for eye-level shots.

    Same origin and projected CRS as ``scene``; the Earth zoom cuts the local mesh out under it.
    """
    from .providers import USGS1mDEMProvider
    place = scene["place"]
    landing = build_landform_scene(
        name=f"{scene['name']} (landing)", latitude=place["latitude"], longitude=place["longitude"],
        radius_m=radius_m, output_root=output_root, raw_cache=raw_cache,
        location_source=place.get("source", "landmark"), raster_spacing_m=1,
        provider=USGS1mDEMProvider(project, zone))
    if landing["projected_crs"] != scene["projected_crs"]:
        raise ValueError("landing patch and local scene disagree on the projected CRS")
    return {key: landing[key] for key in ("project_id", "name", "terrain_dir", "terrain_path", "terrain_raster",
                                          "origin_projected_m", "radius_m", "mesh_spacing_m", "target_height_m",
                                          "elevation_range_m", "source_dataset", "attribution", "place",
                                          "projected_crs")}
