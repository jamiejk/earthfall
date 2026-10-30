"""Select and crop a USGS 3DEP 1 m tile for an address-centred GeoBlock scene.

The script queries the official TNM Access product API, compares valid coverage
in the requested square, and reads only the matching GeoTIFF window. It does
not download whole 10 km tiles. Run with ``uv run python`` from this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import transform_bounds

from geoblock_core.models import GeoSceneSpec


API = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "Digital Elevation Model (DEM) 1 meter"


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--address", required=True)
    p.add_argument("--latitude", required=True, type=float)
    p.add_argument("--longitude", required=True, type=float)
    p.add_argument("--radius-m", type=float, default=1200)
    p.add_argument("--project-id", required=True)
    p.add_argument("--projected-crs", default="EPSG:26910")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--raster", required=True, type=Path)
    p.add_argument("--location-source", required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    a = arguments()
    east, north = Transformer.from_crs("EPSG:4326", a.projected_crs, always_xy=True).transform(
        a.longitude, a.latitude
    )
    spec = GeoSceneSpec(
        project_id=a.project_id,
        name=a.name,
        longitude=a.longitude,
        latitude=a.latitude,
        projected_crs=a.projected_crs,
        vertical_crs="EPSG:5703",  # NAVD88 height in metres
        radius_m=a.radius_m,
        origin_easting_m=east,
        origin_northing_m=north,
        location_source=a.location_source,
        origin_source=f"address point: {a.address}",
    )
    query_bounds = transform_bounds(a.projected_crs, "EPSG:4326", *spec.bounds, densify_pts=21)
    params = {"datasets": DATASET, "bbox": ",".join(map(str, query_bounds)), "max": 100}
    query_url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(query_url, headers={"User-Agent": "GeoBlock prototype/0.1"})
    with urllib.request.urlopen(request, timeout=40) as response:
        products = json.load(response)["items"]
    if not products:
        raise RuntimeError("TNM Access returned no 1 m 3DEP products for this area")

    best = None
    audit = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for product in products:
            url = product.get("downloadURL")
            if not url or not url.lower().endswith(".tif"):
                continue
            try:
                with rasterio.open(url) as src:
                    bounds = (spec.bounds if src.crs.to_string() == a.projected_crs else
                              transform_bounds(a.projected_crs, src.crs, *spec.bounds, densify_pts=21))
                    requested_window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                    window = requested_window.intersection(Window(0, 0, src.width, src.height))
                    pixels = src.read(1, window=window, masked=True)
                    valid = np.isfinite(pixels.filled(np.nan)) & ~np.ma.getmaskarray(pixels)
                    footprint_fraction = (window.width * window.height /
                                          (requested_window.width * requested_window.height))
                    fraction = float(valid.mean() * footprint_fraction)
                    entry = {"title": product["title"], "source_id": product.get("sourceId"),
                             "publication_date": product.get("publicationDate"),
                             "download_url": url, "valid_fraction_in_window": fraction,
                             "crs": src.crs.to_string(), "native_spacing_m": list(src.res)}
                    audit.append(entry)
                    if fraction > 0 and (best is None or
                            (fraction, entry["publication_date"] or "") > best[0]):
                        best = ((fraction, entry["publication_date"] or ""), product,
                                pixels.filled(src.nodata).astype("float32"),
                                window_transform(window, src.transform), src.crs.to_string(), src.nodata)
            except (OSError, ValueError) as exc:
                audit.append({"title": product.get("title"), "error": str(exc)})
    if best is None:
        raise RuntimeError("No queried 3DEP 1 m tile has valid elevation in this square")
    _, product, pixels, affine, crs, nodata = best
    if "NAVD88" not in (product.get("body") or product.get("moreInfo") or ""):
        raise RuntimeError("The selected USGS product did not document a NAVD88 height reference")

    a.spec.parent.mkdir(parents=True, exist_ok=True)
    a.raster.parent.mkdir(parents=True, exist_ok=True)
    a.spec.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with rasterio.open(a.raster, "w", driver="GTiff", height=pixels.shape[0], width=pixels.shape[1],
                       count=1, dtype="float32", crs=crs, transform=affine, nodata=nodata,
                       compress="deflate", predictor=3, tiled=True) as dst:
        dst.write(pixels, 1)
        dst.update_tags(source_product=product["title"], source_url=product["downloadURL"],
                        vertical_crs="EPSG:5703", vertical_datum="NAVD88")
    manifest = {
        "address": a.address,
        "address_point": {"latitude": a.latitude, "longitude": a.longitude,
                          "source": a.location_source},
        "query_url": query_url,
        "selected_product": {"title": product["title"], "source_id": product.get("sourceId"),
                             "publication_date": product.get("publicationDate"),
                             "download_url": product["downloadURL"],
                             "metadata_url": product.get("metaUrl"),
                             "vertical_reference": "NAVD88, metres; USGS product metadata"},
        "candidate_audit": audit,
        "crop": {"path": str(a.raster), "sha256": sha256(a.raster),
                 "width": pixels.shape[1], "height": pixels.shape[0],
                 "crs": crs, "nodata": nodata},
        "licence": "USGS public domain",
        "attribution": "USGS 3D Elevation Program, The National Map",
    }
    a.raster.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"spec": str(a.spec), "raster": str(a.raster),
                      "selected": product["title"], "candidates": audit}, indent=2))


if __name__ == "__main__":
    main()
