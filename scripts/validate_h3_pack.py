"""Validate local inputs for a MiniMax H3 Ref2VA run."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_pack_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", nargs="?", type=Path,
                        default=Path("data/reference_images/worms_head/h3_reference_pack.json"))
    args = parser.parse_args()
    pack = json.loads(args.pack.read_text())
    print(f"3D height evidence: {qualify_pack_geometry(pack)}")
    images = pack["image_references"]
    assert 1 <= len(images) <= 9, "H3 Ref2VA supports at most nine images"
    assert len({item["sha256"] for item in images}) == len(images), "Duplicate image"
    for item in images:
        path = Path(item["path"])
        assert path.exists(), f"Missing image: {path}"
        assert path.stat().st_size <= 30 * 1024 * 1024, f"Image exceeds 30 MiB: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"Hash mismatch: {path}"
        with Image.open(path) as image:
            image.verify()
    video = Path(pack["motion_reference"])
    assert video.exists(), f"Missing video: {video}"
    assert hashlib.sha256(video.read_bytes()).hexdigest() == pack.get("motion_reference_sha256"), "Motion reference differs from the qualified 3D blocking video"
    assert video.stat().st_size <= 50 * 1024 * 1024, "Video exceeds 50 MiB"
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)], check=True, capture_output=True, text=True
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    assert 2 <= duration <= 15.05, f"Reference video duration out of range: {duration}"
    assert len(images) + 1 <= 12, "H3 Ref2VA supports at most 12 total files"
    print(f"Valid H3 pack: {len(images)} images, one {duration:.2f}s video")


if __name__ == "__main__":
    main()
