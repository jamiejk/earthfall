"""Regional relief and settlement for the orbit view, taken from the coordinates rather than the photographs.

The 4B vision model reads relief from the reference photos, which show a courtyard or a facade: it called
the Alhambra (Sierra Nevada, Granada) a plain with no settlement. Relief comes from open elevation tiles
(AWS Terrain Tiles, Terrarium encoding, no key); the settlement question is asked of the local model as
text only, with the place and its coordinates.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Literal
import urllib.request

import numpy as np
from PIL import Image
from pydantic import BaseModel

TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 8  # one 256 px tile is about 156 km across at the equator


def _position(lat: float, lon: float, z: int = ZOOM) -> tuple[float, float]:
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def tile(lat: float, lon: float, z: int = ZOOM) -> tuple[int, int]:
    x, y = _position(lat, lon, z)
    return int(x), int(y)


def pixel(lat: float, lon: float, z: int = ZOOM) -> tuple[int, int]:
    """The place's column and row inside its own 256 px tile."""
    x, y = _position(lat, lon, z)
    return int(x % 1 * 256), int(y % 1 * 256)


LOCAL_RADIUS = 24  # pixels at zoom 8: about 12 km either side, the several kilometres the high still shows


def slopes(heights: np.ndarray, lat: float, z: int = ZOOM) -> np.ndarray:
    metres = 156543.03 / 2 ** z * math.cos(math.radians(lat))
    gy, gx = np.gradient(heights, metres)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def classify_local(heights: np.ndarray, lat: float, lon: float) -> dict:
    """Relief of the ~25 km around the place, from slope rather than height.

    The 156 km tile decides the orbit, but it misreads the high still: Segovia's tile holds the Sierra de
    Guadarrama, so the high still drew eroded badlands above a gentle plain and LTX cut between it and the
    arrival (25 Oct). Measured local slope p50/p90 (degrees): Machu Picchu 23/33, Neuschwanstein 9/26, Grand
    Canyon 7/28, Petra 4/13, Alhambra 4/13 (median 807 m), Segovia 2/5, Carcassonne 2/5, Chambord 0.6/1.3.
    The sea is measured here too: Granada's tile reaches the Mediterranean 50 km away (12% sea), and the
    high still drew an estuary city on the coast (Alhambra a1, 25 Oct).
    """
    cx, cy = pixel(lat, lon)
    r = LOCAL_RADIUS
    window = (slice(max(0, cy - r), cy + r), slice(max(0, cx - r), cx + r))
    land = heights[window] > 0
    sea = round(1 - float(land.mean()), 3)
    if land.mean() < .05:
        return dict(local_relief="coastal_lowland", local_sea_fraction=sea)
    slope = slopes(heights, lat)[window][land]
    p50, p90 = (float(v) for v in np.percentile(slope, (50, 90)))
    median = float(np.median(heights[window][land]))
    if p90 >= 20 and p50 >= 8:
        relief = "high_mountains"
    elif median >= 900 and p90 >= 10:
        relief = "plateau_and_canyons"
    elif p90 >= 6:  # Segovia 5.4 and Avila 5.1 are rolling high plains; "hills" drew badlands there
        relief = "hills"
    elif sea >= .1:
        relief = "coastal_lowland"
    else:
        relief = "plain"
    return dict(local_relief=relief, local_sea_fraction=sea, local_slope_p50=round(p50, 1), local_slope_p90=round(p90, 1),
                local_median=round(median))


def elevations(lat: float, lon: float, z: int = ZOOM) -> np.ndarray:
    x, y = tile(lat, lon, z)
    with urllib.request.urlopen(TILES.format(z=z, x=x, y=y), timeout=30) as response:
        rgb = np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"), dtype=float)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def classify_relief(heights: np.ndarray) -> dict:
    sea = float((heights <= 0).mean())
    land = heights[heights > 0]
    if land.size < heights.size * .05:
        return dict(relief="coastal_lowland", sea_fraction=round(sea, 3))
    low, median, high, peak = (float(v) for v in np.percentile(land, (5, 50, 95, 99)))
    span = high - low
    # A range that is a small part of the tile still dominates the view (Sierra Nevada above Granada).
    if peak >= 2500 and peak - low >= 1500:
        relief = "high_mountains"
    elif median >= 700 and span >= 400:
        relief = "plateau_and_canyons"
    elif span >= 400:
        relief = "hills"
    elif sea >= .1:
        relief = "coastal_lowland"
    else:
        relief = "plain"
    return dict(relief=relief, sea_fraction=round(sea, 3), p5=round(low), median=round(median), p95=round(high), p99=round(peak))


class Settlement(BaseModel):
    town_name: str
    population: int
    distance_km: float
    evidence: str


def settlement_kind(found: dict) -> str:
    """The model calls Segovia (about 50,000 people) a large city, so it gives a population and code decides."""
    if found["distance_km"] > 5 or found["population"] < 2000:
        return "none"
    return "large_city" if found["population"] >= 150_000 else "town"


def settlement(url, place: str, region: str, country: str, lat: float, lon: float, output: Path) -> dict:
    from touchdown.studio_brief import chat
    found = chat(url, [], f"{place}, {region}, {country}, at latitude {lat:.4f}, longitude {lon:.4f}. From what you know "
                 "of this place: name the nearest town or city, its approximate population (the town itself, not its "
                 "province or region) and its distance in kilometres from the landmark (0 if the landmark is inside it).",
                 Settlement, output)
    return {**found, "nearest_settlement": settlement_kind(found)}


def region_facts(setup, url, output: Path) -> dict | None:
    """Relief and settlement for the destination, or None when the setup has no coordinates."""
    d = setup["destination"]
    if d.get("latitude") is None or d.get("longitude") is None:
        return None
    heights = elevations(d["latitude"], d["longitude"])
    facts = dict(source=TILES, zoom=ZOOM, **classify_relief(heights))
    facts.update(classify_local(heights, d["latitude"], d["longitude"]))
    found = settlement(url, d["title"], d.get("region", ""), d.get("country", ""), d["latitude"], d["longitude"],
                       output.with_name("settlement.json"))
    facts.update(nearest_settlement=found["nearest_settlement"], settlement_evidence=found["evidence"],
                 nearest_town=found["town_name"], town_population=found["population"], town_distance_km=found["distance_km"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(facts, indent=2))
    return facts
