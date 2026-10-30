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
import urllib.parse
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


OVERPASS_SERVERS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass-api.de/api/interpreter")
FEATURE_RADIUS = 3000  # metres: what the low aerial and the descent show
COVER = {  # OpenStreetMap tag value -> words for the prompt
    "forest": "woodland", "wood": "woodland", "farmland": "farmland", "meadow": "meadows", "grass": "grassland",
    "heath": "heath", "scrub": "scrub", "residential": "village houses", "park": "landscaped parkland",
    "garden": "formal gardens", "golf_course": "a golf course", "vineyard": "vineyards", "orchard": "orchards",
    "bare_rock": "bare rock", "beach": "beach", "sand": "sand", "industrial": "industrial buildings",
    "commercial": "town buildings", "retail": "town buildings",
}
DIRECTIONS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")


def _stitch(ways: list[list[dict]]) -> list[dict]:
    """Join a multipolygon's outer ways end to end (Castle Howard's Great Lake is one); returns the longest ring."""
    key = lambda p: (round(p["lat"], 7), round(p["lon"], 7))
    ways, rings = [list(w) for w in ways], []
    while ways:
        ring = ways.pop(0)
        while key(ring[0]) != key(ring[-1]):
            for i, w in enumerate(ways):
                if key(w[0]) == key(ring[-1]):
                    ring += w[1:]
                elif key(w[-1]) == key(ring[-1]):
                    ring += w[-2::-1]
                else:
                    continue
                ways.pop(i)
                break
            else:
                break
        rings.append(ring)
    return max(rings, key=len) if rings else []


def _ring_metres(points: list[dict], lat0: float, lon0: float) -> np.ndarray:
    k = 111_320.0
    return np.array([((p["lon"] - lon0) * k * math.cos(math.radians(lat0)), (p["lat"] - lat0) * k) for p in points])


def _area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def describe_features(elements: list[dict], lat: float, lon: float, radius: float = FEATURE_RADIUS) -> dict:
    """Plain words for the real land cover and water around the place, from OpenStreetMap elements with
    geometry. Areas are measured inside the circle only roughly (by the share of each outline's points that
    fall inside it), which is enough to rank them."""
    cover, lakes, rivers = {}, [], set()
    for e in elements:
        tags = e.get("tags", {})
        geometry = e.get("geometry") or _stitch([m.get("geometry", []) for m in e.get("members", [])
                                                 if m.get("role") == "outer" and m.get("geometry")])
        if len(geometry) < 2:
            continue
        xy = _ring_metres(geometry, lat, lon)
        if tags.get("waterway") in ("river", "canal"):
            if (np.hypot(*xy.T) <= radius).any():
                rivers.add(tags.get("name") or "")
            continue
        if len(geometry) < 4:
            continue
        inside = float((np.hypot(*xy.T) <= radius).mean())
        area = _area(xy) * inside
        if area <= 0:
            continue
        if tags.get("amenity") == "fountain" or tags.get("man_made") or tags.get("water") in (
                "basin", "fountain", "reflecting_pool", "wastewater", "moat"):
            continue  # palace fountains and basins are not lakes (Blenheim)
        if tags.get("natural") == "water" or tags.get("water"):
            centre = xy.mean(axis=0)
            if tags.get("water") in ("river", "canal", "stream"):
                rivers.add(tags.get("name") or "")
                continue
            lakes.append(dict(area_ha=round(area / 1e4, 1), name=tags.get("name"),
                              distance_km=round(float(np.hypot(*centre)) / 1000, 1),
                              direction=DIRECTIONS[round(math.degrees(math.atan2(centre[0], centre[1])) % 360 / 45) % 8]))
            continue
        word = next((COVER[tags[k]] for k in ("landuse", "natural", "leisure") if tags.get(k) in COVER), None)
        if word:
            cover[word] = cover.get(word, 0.0) + area
    circle = math.pi * radius ** 2
    ranked = [w for w, a in sorted(cover.items(), key=lambda x: -x[1]) if a >= circle * .03][:4]
    lakes = sorted((l for l in lakes if l["area_ha"] >= 1), key=lambda l: -l["area_ha"])[:2]
    parts = []
    if ranked:
        parts.append("The real surroundings within three kilometres are mostly " + ", ".join(ranked[:-1])
                     + (" and " if len(ranked) > 1 else "") + ranked[-1] + ".")
    for l in lakes:
        size = "a pond" if l["area_ha"] < 2 else "a lake"
        parts.append(f"There is {size} of about {l['area_ha']:g} hectares {l['distance_km']:g} km to the {l['direction']}.")
    names = sorted(n for n in rivers if n)
    if names:
        parts.append(f"{' and '.join(names[:2])} {'flow' if len(names[:2]) > 1 else 'flows'} nearby.")
    elif rivers:
        parts.append("A small river flows nearby.")
    if not lakes and not rivers:
        parts.append("Dry land continues on every side, without lakes or rivers.")
    return dict(local_features=" ".join(parts), local_cover=ranked, local_lakes=lakes, local_rivers=sorted(n for n in rivers if n),
                feature_radius_m=radius, feature_source="OpenStreetMap via Overpass")


CACHE = Path(__file__).resolve().parents[2] / "outputs/touchdown/osm-cache"


def local_features(lat: float, lon: float, radius: float = FEATURE_RADIUS) -> dict:
    """Cached per coordinates: the public Overpass servers took 1-2.5 minutes and often failed (29 Oct)."""
    cached = CACHE / f"{lat:.4f}_{lon:.4f}_{int(radius)}.json"
    if cached.is_file():
        return {**describe_features(json.loads(cached.read_text()), lat, lon, radius), "feature_server": "cache"}
    elements, server = _overpass(lat, lon, radius)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(elements))
    return {**describe_features(elements, lat, lon, radius), "feature_server": server}


def _overpass(lat: float, lon: float, radius: float):
    query = (f"[out:json][timeout:40];("
             f'way["natural"~"^(water|wood|heath|scrub|bare_rock|beach|sand)$"](around:{radius},{lat},{lon});'
             f'relation["natural"="water"](around:{radius},{lat},{lon});'
             f'way["waterway"~"^(river|canal)$"](around:{radius},{lat},{lon});'
             f'way["landuse"~"^(forest|farmland|meadow|grass|residential|vineyard|orchard|industrial|commercial|retail)$"](around:{radius},{lat},{lon});'
             f'relation["landuse"~"^(forest|farmland)$"](around:{radius},{lat},{lon});'
             f'way["leisure"~"^(park|garden|golf_course)$"](around:{radius},{lat},{lon});'
             ");out geom;")  # "out tags geom" drops relation members, and every big lake is a relation
    error = None
    for server in OVERPASS_SERVERS:  # the public servers are often busy (Blenheim: 504, 29 Oct)
        request = urllib.request.Request(server, data=("data=" + urllib.parse.quote(query)).encode(),
                                         headers={"User-Agent": "Earthfall/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read())["elements"], server
        except (OSError, ValueError, KeyError) as exc:
            error = exc
    raise error


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
    # Without inland water and land cover LTX invented a sea-sized lake below Castle Howard (29 Oct).
    try:
        facts.update(local_features(d["latitude"], d["longitude"]))
    except (OSError, ValueError, KeyError) as exc:
        facts.update(local_features=None, local_features_error=str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(facts, indent=2))
    return facts
