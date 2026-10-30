from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

from .models import GeoSceneSpec


@lru_cache(maxsize=16)
def horizontal_transformers(projected_crs: str):
    crs = CRS.from_user_input(projected_crs)
    if not crs.is_projected or len(crs.axis_info) < 2:
        raise ValueError("projected_crs must be a two-axis metric projected CRS")
    if any(axis.unit_name.lower() not in {"metre", "meter"} for axis in crs.axis_info[:2]):
        raise ValueError("projected_crs axes must be metres")
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True, allow_ballpark=False)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True, allow_ballpark=False)
    return forward, inverse


def validate_scene_coordinates(spec: GeoSceneSpec) -> float:
    forward, inverse = horizontal_transformers(spec.projected_crs)
    e, n = forward.transform(spec.longitude, spec.latitude)
    if not np.all(np.isfinite([e,n,spec.origin_easting_m,spec.origin_northing_m])):
        raise ValueError("scene coordinates are not finite")
    lon, lat = inverse.transform(e, n)
    roundtrip_m = float(np.hypot((lon-spec.longitude)*111320*np.cos(np.radians(lat)),
                                  (lat-spec.latitude)*111320))
    if roundtrip_m > 0.001:
        raise ValueError("horizontal coordinate round trip exceeds 1 mm")
    return roundtrip_m


def geographic_envelope(spec: GeoSceneSpec, margin_m: float = 120) -> tuple[float, float, float, float]:
    """Transform a densely sampled perimeter, retaining a fetch margin."""
    _, inverse = horizontal_transformers(spec.projected_crs)
    x0,y0,x1,y1=spec.bounds
    xs=np.linspace(x0-margin_m,x1+margin_m,17)
    ys=np.linspace(y0-margin_m,y1+margin_m,17)
    px=np.concatenate((xs,xs,np.full_like(ys,x0-margin_m),np.full_like(ys,x1+margin_m)))
    py=np.concatenate((np.full_like(xs,y0-margin_m),np.full_like(xs,y1+margin_m),ys,ys))
    lon,lat=inverse.transform(px,py)
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("area cannot be projected to geographic coordinates")
    if np.ptp(lon)>180 or min(lat)<-85 or max(lat)>85:
        raise ValueError("polar or antimeridian area is unsupported in this version")
    return tuple(float(v) for v in (min(lon),min(lat),max(lon),max(lat)))


def vertical_transformer(source_horizontal: str, source_vertical: str,
                         target_horizontal: str, target_vertical: str):
    """Select a grid-backed operation; never accept a ballpark height shift."""
    src = CRS.from_user_input(f"{source_horizontal}+{source_vertical}")
    dst = CRS.from_user_input(f"{target_horizontal}+{target_vertical}")
    group = TransformerGroup(src, dst, always_xy=True, allow_ballpark=False)
    if not group.best_available or not group.transformers:
        missing = sorted({grid.short_name for op in group.unavailable_operations
                          for grid in op.grids if not grid.available})
        raise ValueError("required vertical transformation grids are unavailable: " + ", ".join(missing))
    transformer = group.transformers[0]
    if transformer.accuracy < 0:
        raise ValueError("selected vertical operation has unknown accuracy")
    return transformer
