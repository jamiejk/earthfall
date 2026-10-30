"""What the ground at a place is covered with, from ESA WorldCover 10 m (CC BY 4.0), as prompt words.

Render-to-real models repaint the blocking with what the prompt says the place is made of. A curated
description only exists for a few places (looks.LOOKS); this one is measured for anywhere on Earth, so a
new landing spot gets words that match its real forest, rock, snow, water or town.

Only the window around the place is read from the public COG tiles (3 x 3 degrees, named by their
south-west corner), at a reduced overview for wide areas.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
LICENSE = "CC BY 4.0"
ATTRIBUTION = "ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium"

# WorldCover class code -> words for the real surface (no colours: weather and light come from elsewhere).
CLASSES = {
    10: "forest",
    20: "shrubland and scrub",
    30: "grassland and meadow",
    40: "farmland",
    50: "buildings, roads and town",
    60: "bare rock, scree and sand with sparse plants",
    70: "glacier ice and snowfields",
    80: "open water",
    90: "wetland and marsh",
    95: "mangrove",
    100: "moss and lichen tundra",
}


def tile_name(latitude: float, longitude: float) -> str:
    south, west = 3 * math.floor(latitude / 3), 3 * math.floor(longitude / 3)
    return (f"ESA_WorldCover_10m_2021_v200_{'N' if south >= 0 else 'S'}{abs(south):02d}"
            f"{'E' if west >= 0 else 'W'}{abs(west):03d}_Map")


def tile_uri(name: str, root: str | None = None) -> str:
    root = (root or os.environ.get("GEOBLOCK_WORLDCOVER_ROOT") or ROOT).rstrip("/")
    path = f"{root}/{name}.tif"
    return "/vsicurl/" + path if path.startswith(("http://", "https://")) else path


def read_classes(latitude: float, longitude: float, radius_m: float, *, pixels: int = 400,
                 root: str | None = None) -> np.ndarray:
    """WorldCover classes in a square of ``radius_m`` around the place, about ``pixels`` across."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * max(math.cos(math.radians(latitude)), 0.05))
    west, south, east, north = longitude - dlon, latitude - dlat, longitude + dlon, latitude + dlat
    names = sorted({tile_name(lat, lon) for lat in (south, north) for lon in (west, east)})
    resolution = max(2 * dlat / pixels, 1 / 12000)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        sources = []
        try:
            for name in names:
                try:
                    sources.append(rasterio.open(tile_uri(name, root)))
                except rasterio.errors.RasterioIOError:
                    continue  # open ocean has no tile
            if not sources:
                return np.full((pixels, pixels), 80, dtype=np.uint8)
            data, _ = merge(sources, bounds=(west, south, east, north), res=(resolution, resolution), nodata=0,
                            resampling=Resampling.mode)
        finally:
            for source in sources:
                source.close()
    data = data[0]
    data[data == 0] = 80  # outside every tile: sea
    return data


def fractions(classes: np.ndarray, mask: np.ndarray | None = None) -> dict[int, float]:
    values = classes[mask] if mask is not None else classes.ravel()
    codes, counts = np.unique(values, return_counts=True)
    total = counts.sum()
    return {int(code): round(float(count / total), 3) for code, count in zip(codes, counts) if code in CLASSES}


def describe(share: dict[int, float], *, limit: int = 3, floor: float = 0.06) -> str:
    """'mostly forest, with grassland and meadow and some bare rock' from class shares."""
    share = {int(code): value for code, value in share.items()}  # JSON keys arrive as strings
    ranked = [(code, value) for code, value in sorted(share.items(), key=lambda item: -item[1]) if value >= floor]
    if not ranked:
        ranked = sorted(share.items(), key=lambda item: -item[1])[:1]
    ranked = ranked[:limit]
    words = [CLASSES[code] for code, _ in ranked]
    lead = "mostly " if ranked[0][1] >= 0.5 else ""
    if len(words) == 1:
        return lead + words[0]
    if len(words) == 2:
        return f"{lead}{words[0]}, with {words[1]}"
    return f"{lead}{words[0]}, with {words[1]} and some {words[2]}"


def landcover_summary(latitude: float, longitude: float, *, near_m: float = 600, view_m: float = 8000,
                      root: str | None = None) -> dict:
    """Class shares close to the place (where a person stands) and across the wider view."""
    classes = read_classes(latitude, longitude, view_m, root=root)
    size = classes.shape[0]
    rows, cols = np.indices(classes.shape)
    centre = (size - 1) / 2
    distance = np.hypot(rows - centre, cols - centre) * (2 * view_m / size)
    near = fractions(classes, distance <= near_m)
    view = fractions(classes, distance > near_m)
    return {"near": near, "view": view, "near_m": near_m, "view_m": view_m,
            "source": "ESA WorldCover 10 m v200 (2021)", "license": LICENSE, "attribution": ATTRIBUTION}


def look_from_landcover(summary: dict, kind: str = "") -> dict:
    """A looks.LOOKS-style entry (all / high / low / far) from a landcover_summary.

    ``kind`` is the place's landform (summit, canyon rim, sea cliff, ...), if known.
    """
    near, view = describe(summary["near"]), describe(summary["view"])
    landform = f" {kind}" if kind else ""
    water = {int(code): value for code, value in summary["view"].items()}.get(80, 0)
    far = ("the land runs out to open water toward the horizon" if water >= 0.25
           else "the distant land fades naturally toward the horizon")
    return {
        "all": f"the{landform} ground underfoot at {{name}} is {near}; across the view the land is {view}",
        "high": f"the ground near the camera at {{name}} is {near}",
        "low": f"the land below and around {{name}} is {view}",
        "far": far,
    }


def cached_summary(latitude: float, longitude: float, cache: Path, **kwargs) -> dict:
    """landcover_summary, stored under ``cache`` so a scene rebuild does not read the tiles again."""
    key = f"{latitude:.5f}_{longitude:.5f}_{kwargs.get('near_m', 600)}_{kwargs.get('view_m', 8000)}"
    path = cache / "worldcover" / f"{key}.json"
    if path.is_file():
        return json.loads(path.read_text())
    summary = landcover_summary(latitude, longitude, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def with_ground_underfoot(look: dict, words: str, between_trees: str = "") -> dict:
    """``look`` with the ground near the person replaced by ``words`` measured on aerial photos
    (imagery.naip_ground). WorldCover's 600 m 'near' share is too coarse underfoot: at Glacier Point it
    says forest and grassland, and klein painted the granite overlook as a meadow."""
    look = dict(look)
    rest = look.get("all", "").split("; ", 1)
    look["all"] = f"the ground underfoot at {{name}} is {words}" + (f"; {rest[1]}" if len(rest) == 2 else "")
    look["high"] = f"the ground near the camera at {{name}} is {words}"
    look["underfoot"] = words
    if between_trees:  # for the tree proxies' prompt words (looks.canopy_text)
        look["between_trees"] = between_trees
    return look
