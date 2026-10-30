"""Record measured 3D relief at a natural landmark before blocking a shot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import measure_landform_evidence

DEFAULT_BASE = ROOT / "outputs/B_lidar_full_2m"
DEFAULT_SPEC = ROOT / "examples/worms_head_full.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--target-easting", type=float)
    parser.add_argument("--target-northing", type=float)
    args = parser.parse_args()
    scene = json.loads(args.spec.read_text())
    x = args.target_easting if args.target_easting is not None else scene["origin_easting_m"]
    y = args.target_northing if args.target_northing is not None else scene["origin_northing_m"]
    record = measure_landform_evidence(args.terrain_dir, (x, y), "Worm's Head")
    print(f"{args.terrain_dir / 'landform_height_evidence.json'}: "
          f"{record['relief_p95_minus_p5_m']:.1f} m local relief, {record['valid_fraction']:.0%} valid cells")


if __name__ == "__main__":
    main()
