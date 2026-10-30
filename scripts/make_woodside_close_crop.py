"""Derive a close address-centred NAIP reference from the checked 660 m crop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/woodside_library_naip_manifest.json"
DEST = ROOT / "data/woodside_library_naip_close_220m.png"
CLOSE_MANIFEST = ROOT / "data/woodside_library_close_reference_manifest.json"


def main() -> None:
    source_record = json.loads(MANIFEST.read_text())["images"][0]
    source = ROOT / source_record["path"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != source_record["sha256"]:
        raise ValueError("NAIP source crop checksum changed")
    with Image.open(source) as image:
        width, height = image.size
        crop_w = round(width * 220 / source_record["width_m"])
        crop_h = round(height * 220 / source_record["width_m"])
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        crop = image.crop((left, top, left + crop_w, top + crop_h))
        crop.resize((1024, 1024), Image.Resampling.LANCZOS).save(DEST)
    opening = ROOT / "outputs/woodside_library_3dep/blocking_measured_storyboard/frame_0001.png"
    if not opening.is_file():
        raise FileNotFoundError("Render the measured-building storyboard before preparing close references")
    east, north = 565645.8847276662, 4142734.5250981823
    record = {
        "source": str(source), "source_sha256": digest,
        "source_attribution": "USDA NAIP 18 May 2022, distributed by USGS; public domain",
        "crop_bounds_utm10_m": [east-110, north-110, east+110, north+110],
        "images": [
            {"path": str(opening), "sha256": hashlib.sha256(opening.read_bytes()).hexdigest(),
             "role": "Blender opening composition; not geographic imagery"},
            {"path": str(DEST), "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
             "role": "NAIP place reference, cropped to 220 m around address"},
        ],
    }
    CLOSE_MANIFEST.write_text(json.dumps(record, indent=2) + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
