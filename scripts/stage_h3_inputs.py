"""Copy the validated GeoBlock H3 references into a ComfyUI input folder."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_pack_geometry


def copy_if_changed(source: Path, destination: Path, expected_sha256: str | None = None) -> None:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError(f"Source changed since pack creation: {source}")
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == source_hash:
        return
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path,
                        default=Path("data/reference_images/worms_head/h3_reference_pack.json"))
    parser.add_argument("--comfy-root", type=Path,
                        default=Path("/run/media/jamie/Data/Image Creation/Comfy"))
    args = parser.parse_args()
    pack = json.loads(args.pack.read_text())
    print(f"3D height evidence: {qualify_pack_geometry(pack)}")
    video = Path(pack["motion_reference"])
    if hashlib.sha256(video.read_bytes()).hexdigest() != pack.get("motion_reference_sha256"):
        raise ValueError("Motion reference differs from the qualified 3D blocking video.")
    target = args.comfy_root / "input" / "geoblock_worms_head"
    target.mkdir(parents=True, exist_ok=True)
    staged = set()
    for item in pack["image_references"]:
        source = Path(item["path"])
        staged.add(source.name)
        copy_if_changed(source, target / source.name, item["sha256"])
    staged.add(video.name)
    copy_if_changed(video, target / video.name)
    for old in target.iterdir():
        if old.is_file() and old.name.startswith("commons_") and old.name.endswith(".jpg") and old.name not in staged:
            old.unlink()
    print(f"Staged {len(pack['image_references'])} photos and one video in {target}")


if __name__ == "__main__":
    main()
