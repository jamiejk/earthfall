"""Simplify an aerial drape into a repeatable flat-colour Blender guide texture."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--colors", type=int, default=32)
    a = p.parse_args()
    if a.size < 512 or not 8 <= a.colors <= 64:
        p.error("size must be >= 512 and colors must be 8–64")
    with Image.open(a.input) as source:
        image = source.convert("RGB").resize((a.size, a.size), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter(5))
    image = image.quantize(colors=a.colors, method=Image.Quantize.FASTOCTREE).convert("RGB")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
