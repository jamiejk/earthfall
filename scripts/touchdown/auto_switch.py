"""Per-place switches decided by the build instead of set by hand in each manifest (18 Oct).

Three switches were set place by place while the FLUX 3 runs were iterated:

- ``[end] lock_pin = "48"`` and ``keep_after_pin = 8``: unpinned, the living end tilted up to mostly sky (Grand
  Canyon v3, Monument Valley v7) and the face drifted after the pin (Grand Canyon v7). Now the defaults for every
  manifest that renders its stills with FLUX 3; a manifest that sets them keeps its own values.
- ``[terrain] avoid_pavement``: on, the landing search may find no view with a clear natural-ground corridor at a
  headland whose approach crosses a road or concrete (Battery Spencer); the search is then repeated without it and
  the relaxation logged.
- ``[stills] end_from_photo`` (+ ``[terrain] osm_structures``): when FLUX 3 draws the famous view instead of the
  measured one (Battery Spencer: every seed failed the end-still gate), the end still is built from a real photo, if
  one is geotagged and OSM maps the place's structures with heights.
"""
from __future__ import annotations

from pathlib import Path

NO_CORRIDOR = "No safe view has a clear natural-ground approach corridor"


def flux3_defaults(manifest: dict) -> dict:
    """``manifest`` with the FLUX 3 end defaults filled in where the manifest does not set them. Returns what was
    filled in (empty when nothing was)."""
    if not manifest.get("stills", {}).get("flux3"):
        return {}
    end = manifest.setdefault("end", {})
    filled = {}
    for key, value in (("lock_pin", "48"), ("keep_after_pin", 8)):
        if key not in end:
            end[key] = value
            filled[key] = value
    return filled


def search_relaxing_pavement(search, cfg: dict, **kwargs):
    """Run a landing/view search (``search(avoid_pavement=..., **kwargs)``) with the manifest's ``avoid_pavement``;
    if that finds no clear natural-ground corridor and ``auto_relax_pavement`` (default on) allows it, repeat once
    without the pavement test. Returns (result, relaxed: bool)."""
    avoid = bool(cfg.get("avoid_pavement", False))
    try:
        return search(avoid_pavement=avoid, **kwargs), False
    except ValueError as exc:
        if not (avoid and NO_CORRIDOR in str(exc) and cfg.get("auto_relax_pavement", True)):
            raise
    return search(avoid_pavement=False, **kwargs), True


def geotagged(photo: dict) -> bool:
    """A Commons photo whose camera position is known: an explicit camera-location template, or EXIF GPS in the
    downloaded file."""
    if photo.get("camera_location"):
        return True
    path = photo.get("local_path")
    if not path or not Path(path).is_file():
        return False
    from PIL import Image
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
    except OSError:
        return False
    return 2 in gps and 4 in gps  # GPSLatitude, GPSLongitude


def osm_heights(structures: dict | None, minimum: int = 1, tall_m: float = 20.0) -> bool:
    """Whether OSM maps structures with heights near the landing (osm_structures.prepare output): at least
    ``minimum`` structures taller than ``tall_m``."""
    if not structures:
        return False
    return sum(1 for s in structures.get("structures", []) if s.get("max", 0) > tall_m) >= minimum


def use_photo_end(cfg: dict, gate_failures: int, photos: list[dict], structures: dict | None) -> tuple[bool, str]:
    """Switch the end still to a real photo? Only when the manifest does not decide it (``end_from_photo`` unset),
    ``auto_photo_end`` is on (default), the FLUX end still failed the gate at least ``auto_photo_after`` (2) times,
    a geotagged photo of the place exists and OSM maps its structures with heights, including a bridge tower (the
    photo camera solve matches towers). Returns (switch, reason)."""
    if "end_from_photo" in cfg:
        return bool(cfg["end_from_photo"]), "set in the manifest"
    if not cfg.get("auto_photo_end", True):
        return False, "auto_photo_end is off"
    need = int(cfg.get("auto_photo_after", 2))
    if gate_failures < need:
        return False, f"the FLUX end still failed the gate {gate_failures} time(s), under {need}"
    if not any(geotagged(p) for p in photos):
        return False, "no geotagged Commons photo of the place"
    if not osm_heights(structures):
        return False, "OSM maps no structure with a height here"
    if not any(s.get("bridge") and s.get("kind") == "tower" for s in structures.get("structures", [])):
        # photo_pose.solve places the photo camera on bridge towers; without one it can only fail later
        # (Bixby Creek Bridge 18 Oct: an arch bridge, no towers)
        return False, "OSM maps no bridge tower for the photo camera solve"
    return True, f"the FLUX end still failed the gate {gate_failures} times; a geotagged photo and OSM heights exist"
