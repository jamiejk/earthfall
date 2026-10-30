"""Fetch two georeferenced NAIP aerial crops and record their source image."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

from PIL import Image
from pyproj import Transformer


SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"


def get(url, params):
    request = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": "GeoBlock prototype/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read(), response.headers.get("Content-Type", "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--name", default="woodside_library")
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    x, y = Transformer.from_crs("EPSG:4326", spec["projected_crs"], always_xy=True).transform(
        spec["longitude"], spec["latitude"])
    ident_params = {
        "f": "json", "geometry": json.dumps({"x": x, "y": y,
                                               "spatialReference": {"wkid": int(spec["projected_crs"].split(":")[1])}}),
        "geometryType": "esriGeometryPoint", "returnCatalogItems": "true",
        "returnGeometry": "false",
    }
    body, _ = get(SERVICE + "/identify", ident_params)
    found = json.loads(body)
    source = next((feature["attributes"] for feature in found.get("catalogItems", {}).get("features", [])
                   if feature["attributes"].get("Category") == 1), None)
    if source is None:
        raise RuntimeError("NAIP ImageServer has no primary image at the address point")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    image_payloads = {}
    for width_m in (660, 2400):
        half = width_m / 2
        params = {
            "f": "image", "bbox": f"{x-half},{y-half},{x+half},{y+half}",
            "bboxSR": spec["projected_crs"].split(":")[1],
            "imageSR": spec["projected_crs"].split(":")[1],
            "size": "1024,1024", "format": "png", "pixelType": "U8",
        }
        payload, content_type = get(SERVICE + "/exportImage", params)
        if not content_type.startswith("image/"):
            raise RuntimeError(f"NAIP export returned {content_type}: {payload[:200]!r}")
        with Image.open(io.BytesIO(payload)) as image:
            if image.size != (1024, 1024):
                raise RuntimeError(f"Unexpected NAIP image size {image.size}")
        path = a.output_dir / f"{a.name}_naip_{width_m}m.png"
        path.write_bytes(payload)
        image_payloads[width_m] = payload
        entries.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                        "width_m": width_m, "bbox_projected_m": [x-half, y-half, x+half, y+half],
                        "export_url": SERVICE + "/exportImage?" + urllib.parse.urlencode(params)})
    # Both crops share the same projected centre. Put native-detail centre
    # pixels into an upsampled wide drape without inventing any new detail.
    with Image.open(io.BytesIO(image_payloads[2400])) as wide, \
            Image.open(io.BytesIO(image_payloads[660])) as close:
        drape = wide.convert("RGB").resize((4096, 4096), Image.Resampling.BICUBIC)
        centre_size = round(4096 * 660 / 2400)
        centre = close.convert("RGB").resize((centre_size, centre_size), Image.Resampling.BICUBIC)
        start = (4096 - centre_size) // 2
        drape.paste(centre, (start, start))
        drape_path = a.output_dir / f"{a.name}_naip_drape_4096.png"
        drape.save(drape_path, optimize=True)
    manifest = {
        "source_service": SERVICE,
        "licence": "USDA NAIP public domain; USGS National Map distribution",
        "address_point": {"latitude": spec["latitude"], "longitude": spec["longitude"]},
        "identified_image": {key: source.get(key) for key in
                             ("OBJECTID", "Name", "Year", "acquisition_date", "agency", "vendor",
                              "resolution_value", "resolution_units", "download_url")},
        "images": entries,
        "drape": {"path": str(drape_path), "sha256": hashlib.sha256(drape_path.read_bytes()).hexdigest(),
                  "size_pixels": 4096,
                  "method": "wide 2.4 km crop upsampled; central 660 m crop placed by shared projected centre"},
    }
    out = a.output_dir / f"{a.name}_naip_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out), "identified_image": manifest["identified_image"],
                      "images": entries}, indent=2))


if __name__ == "__main__":
    main()
