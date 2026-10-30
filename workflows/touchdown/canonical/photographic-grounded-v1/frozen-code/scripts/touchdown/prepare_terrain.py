"""Resample a local 1 m landing DEM and wide 3DEP DEM for terrain_blender.py.

The raster files can be fetched with the existing USGS1mDEMProvider and
USGS3DEPProvider. This step streams them from disk and produces small centred
arrays; it does not need a GPU.

    python scripts/touchdown/prepare_terrain.py --near project_1m.tif \
      --far seamless_13arcsec.tif --lat 37.7156 --lon -119.6772 \
      --utm-zone 11 --output outputs/touchdown/.../dem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


def prepare(near: Path, far: Path, lat: float, lon: float, zone: int, output: Path) -> dict:
    from geoblock_core.utm import projected_crs
    crs = projected_crs(zone)  # 1..60: NAD83 UTM as the 3DEP products in CONUS; 326xx/327xx: WGS 84 UTM abroad
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)

    def sample(source: Path, radius: int, step: int) -> np.ndarray:
        n = 2 * radius // step + 1
        out = np.full((n, n), np.nan, np.float32)
        with rasterio.open(source) as src:
            reproject(rasterio.band(src, 1), out,
                      dst_transform=from_origin(x - radius, y + radius, step, step),
                      dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        if not np.isfinite(out).all():
            raise ValueError(f"DEM has nodata inside the required {radius} m radius: {source}")
        return out

    landing = sample(near, 400, 2)
    valley = sample(far, 17000, 50)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "height.npy", landing)
    np.save(output / "height_detail.npy", sample(near, 400, 1))
    np.save(output / "height_far.npy", valley)
    meta = {"step": 2, "n": 401, "landing": float(landing[200, 200]), "x": x, "y": y,
            "lat": lat, "lon": lon, "utm_zone": zone,
            "near_source": str(near), "far_source": str(far),
            "render_step_m": 1, "search_step_m": 2}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--near", type=Path, required=True)
    p.add_argument("--far", type=Path, required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--utm-zone", type=int, required=True, help="1..60 (NAD83, US) or a WGS 84 UTM EPSG code")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.near, a.far, a.lat, a.lon, a.utm_zone, a.output), indent=2))


if __name__ == "__main__":
    main()
