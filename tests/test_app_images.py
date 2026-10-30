"""Actual HEIF/WebP bytes, orientation and sanitized output for the upload boundary."""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("pillow_heif")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from server import images


def encoded_photo(fmt, mode="RGB", orientation=6):
    image = Image.new(mode, (120, 80), (220, 30, 20, 90) if mode == "RGBA" else (220, 30, 20))
    exif = Image.Exif()
    exif[274] = orientation
    exif[271] = "Private camera"
    buf = io.BytesIO()
    image.save(buf, fmt, exif=exif.tobytes(), quality=95)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "HEIF"])
def test_formats_are_oriented_once_and_saved_without_metadata(tmp_path, fmt):
    output = images.sanitize(encoded_photo(fmt), tmp_path / "photo")
    assert output.suffix in (".jpg", ".png")
    with Image.open(output) as image:
        assert image.size == (80, 120)
        assert image.format in ("JPEG", "PNG")
        assert not image.getexif()
        assert not {"exif", "xmp", "icc_profile"} & image.info.keys()


@pytest.mark.parametrize("fmt", ["WEBP", "HEIF"])
def test_transparent_photos_remain_transparent_clean_pngs(tmp_path, fmt):
    output = images.sanitize(encoded_photo(fmt, mode="RGBA", orientation=1), tmp_path / "photo")
    assert output.suffix == ".png"
    with Image.open(output) as image:
        assert image.mode == "RGBA" and image.size == (120, 80)
        assert 80 <= image.getpixel((60, 40))[3] <= 100
        assert not image.getexif()


def test_animated_webp_imports_one_still_frame(tmp_path):
    first = Image.new("RGB", (120, 80), "red")
    second = Image.new("RGB", (120, 80), "blue")
    buf = io.BytesIO()
    first.save(buf, "WEBP", save_all=True, append_images=[second], duration=100, loop=0, lossless=True)
    output = images.sanitize(buf.getvalue(), tmp_path / "photo")
    with Image.open(output) as image:
        assert getattr(image, "n_frames", 1) == 1
        r, g, b = image.convert("RGB").getpixel((60, 40))
        assert r > 240 and g < 10 and b < 10


@pytest.mark.parametrize("fmt", ["HEIF", "WEBP"])
def test_pixel_limit_applies_to_new_formats_before_resize(tmp_path, monkeypatch, fmt):
    monkeypatch.setattr(images, "MAX_PIXELS", 100)
    with pytest.raises(images.BadImage, match="too many pixels"):
        images.sanitize(encoded_photo(fmt), tmp_path / "photo")
    assert not list(tmp_path.iterdir())


def test_new_format_resizes_to_existing_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_SIDE", 60)
    output = images.sanitize(encoded_photo("HEIF"), tmp_path / "photo")
    with Image.open(output) as image:
        assert image.size == (40, 60)


@pytest.mark.parametrize("data", [b"not an image", b"RIFF\x00\x00\x00\x00WEBP", b"\x00\x00\x00\x18ftypheic"])
def test_invalid_new_format_data_has_a_useful_error(tmp_path, data):
    with pytest.raises(images.BadImage, match="readable"):
        images.sanitize(data, tmp_path / "photo")


def test_unsupported_image_content_is_rejected(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, "GIF")
    with pytest.raises(images.BadImage, match="JPEG, PNG, WebP or HEIC/HEIF"):
        images.sanitize(buf.getvalue(), tmp_path / "photo")


def test_multi_picture_jpeg_keeps_its_first_image(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (220, 30, 20)).save(buf, "MPO", save_all=True,
                                                    append_images=[Image.new("RGB", (64, 48), (20, 30, 220))])
    out = images.sanitize(buf.getvalue(), tmp_path / "mpo")
    with Image.open(out) as clean:
        assert clean.format == "JPEG" and clean.getpixel((5, 5))[0] > 150
