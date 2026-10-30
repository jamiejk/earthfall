"""Globe and cloud textures for the Earth Zoom, from NASA GIBS (public domain).

- Ground: Blue Marble Next Generation (cloud-free monthly composite).
- Clouds: one day of VIIRS true colour. Cloud alpha is where that day is brighter and greyer
  than the cloud-free ground, so the clouds from orbit are real clouds on a real date.
- A high-resolution cap of both around the landmark covers the lower part of the descent.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .imagery import BLUE_MARBLE, CREDIT, USER_AGENT, WMS, wms_image

VIIRS = "VIIRS_SNPP_CorrectedReflectance_TrueColor"


def cloud_alpha(day: Image.Image, ground: Image.Image) -> Image.Image:
    """Greyscale alpha: 1 where ``day`` shows cloud over the cloud-free ``ground``."""
    d = np.asarray(day, dtype=np.float32) / 255
    g = np.asarray(ground.resize(day.size), dtype=np.float32) / 255
    brighter = d.mean(axis=2) - g.mean(axis=2)
    grey = 1 - np.clip((d.max(axis=2) - d.min(axis=2)) / 0.25, 0, 1)
    alpha = np.clip((brighter - 0.12) / 0.4, 0, 1) ** 1.5 * grey  # thin haze stays clear; clouds keep soft edges
    alpha[d.mean(axis=2) < 0.02] = 0  # swath gaps and polar night
    return Image.fromarray((alpha * 255).astype(np.uint8), "L")


def fetch_globe(latitude: float, longitude: float, folder: Path, *, day: date | None = None,
                globe_px: int = 8192, cap_deg: float = 6.0, cap_px: int = 4096) -> dict:
    """Write globe_ground.jpg, globe_clouds.png, cap_ground.jpg, cap_clouds.png and globe.json."""
    import httpx
    folder = folder.resolve()  # Blender runs from another directory
    folder.mkdir(parents=True, exist_ok=True)
    day = day or date(date.today().year - 1, 8, 16)
    manifest_path = folder / "globe.json"
    key = [latitude, longitude, day.isoformat(), globe_px, cap_deg, cap_px, "alpha-2"]
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("key") == key:
        return json.loads(manifest_path.read_text())
    half = cap_deg / 2
    cap = (longitude - half, max(-90.0, latitude - half), longitude + half, min(90.0, latitude + half))
    cap_h = max(64, round(cap_px * (cap[3] - cap[1]) / (cap[2] - cap[0])))
    def cached(name, layer, bbox, width, height, when):
        path = folder / name
        if path.is_file():
            return Image.open(path).convert("RGB")
        image = wms_image(layer, bbox, width, height, when, client)
        image.save(path, quality=92)
        return image

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=300) as client:
        ground = cached("globe_ground.jpg", BLUE_MARBLE, (-180, -90, 180, 90), globe_px, globe_px // 2, None)
        viirs = cached(f"globe_viirs_{day}.jpg", VIIRS, (-180, -90, 180, 90), globe_px, globe_px // 2, day)
        cap_ground = cached("cap_ground.jpg", BLUE_MARBLE, cap, cap_px, cap_h, None)
        cap_viirs = cached(f"cap_viirs_{day}.jpg", VIIRS, cap, cap_px, cap_h, day)
    cloud_alpha(viirs, ground).save(folder / "globe_clouds.png")
    cloud_alpha(cap_viirs, cap_ground).save(folder / "cap_clouds.png")
    manifest = {"key": key, "latitude": latitude, "longitude": longitude, "cloud_date": day.isoformat(),
                "cap_bounds_deg": list(cap), "ground_layer": BLUE_MARBLE, "cloud_layer": VIIRS,
                "files": {name: str(folder / name) for name in
                          ("globe_ground.jpg", "globe_clouds.png", "cap_ground.jpg", "cap_clouds.png")},
                "source": WMS, "license": "Public domain (NASA)", "credit": CREDIT}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
