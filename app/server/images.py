"""Decode uploaded photos and emit oriented JPEG/PNG pixels without metadata."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener(thumbnails=False)

MAX_SIDE = 2048          # plenty for the subject reference; keeps uploads and the worker's copy small
MAX_PIXELS = 40_000_000  # refuse decompression bombs before decoding
# MPO is a JPEG with extra frames (some Nikon/Fuji cameras); it decodes to its first image.
SUPPORTED_FORMATS = ("JPEG", "MPO", "PNG", "WEBP", "HEIF")
FORMAT_NAMES = "JPEG, PNG, WebP or HEIC/HEIF"


class BadImage(ValueError):
    pass


def sanitize(data: bytes, out_stem: Path) -> Path:
    """Write a clean copy of ``data`` to ``out_stem`` + .jpg/.png and return its path. No EXIF, GPS or other
    metadata survives: the pixels are decoded and encoded again."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            if fmt not in SUPPORTED_FORMATS:
                raise BadImage(f"please upload a {FORMAT_NAMES} photo")
            if probe.width * probe.height > MAX_PIXELS:
                raise BadImage("that photo is too large (too many pixels)")
        with Image.open(io.BytesIO(data)) as im:
            # HEIF opens its primary image; animated WebP imports its first frame.
            im = ImageOps.exif_transpose(im)
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                out = out_stem.with_suffix(".png")
                clean = Image.new("RGBA", im.size)
                clean.paste(im)
                clean.save(out, "PNG", optimize=True)
                return out
            out = out_stem.with_suffix(".jpg")
            clean = im.convert("RGB")
            clean.info.clear()
            clean.save(out, "JPEG", quality=92, exif=b"")
            return out
    except BadImage:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError, ValueError) as e:
        raise BadImage(f"that file is not a readable {FORMAT_NAMES} photo") from e
