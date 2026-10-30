"""The fly-down place catalog: landing spots with their measured terrain, lidar and imagery coverage.

``data/places.json`` lists each place's curated landing point (where the person stands) and, optionally,
what the camera looks at past them. Everything under ``coverage`` is measured from the open datasets by
``scripts/build_place_catalog.py``, so listing and choosing places downloads nothing: terrain, imagery and
lidar are fetched only for the place a shot is built for, and only its window.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import re

CATALOG = Path(__file__).with_name("data") / "places.json"

# Wired terrain families (scene.terrain_family picks one by location).
TERRAIN_SOURCES = {
    "usgs3dep": {"dataset": "USGS 3DEP 1/3 arc-second DEM", "resolution_m": 10, "model": "bare earth (DTM)",
                 "vertical": "NAVD88", "license": "Public domain (U.S. Geological Survey)",
                 "url": "https://www.usgs.gov/3d-elevation-program"},
    "copernicus": {"dataset": "Copernicus DEM GLO-30", "resolution_m": 30,
                   "model": "surface (DSM: canopy and buildings included)", "vertical": "EGM2008",
                   "license": "Copernicus DEM licence: free for any use, including commercial, with attribution",
                   "url": "https://registry.opendata.aws/copernicus-dem/"},
}
IMAGERY_SOURCES = {
    "hls": {"dataset": "NASA HLS (Harmonized Landsat Sentinel-2) via GIBS", "resolution_m": 30,
            "coverage": "worldwide", "license": "Public domain (NASA)", "status": "draped with --imagery"},
    "naip": {"dataset": "USDA NAIP aerial photography", "resolution_m": 0.6, "coverage": "conterminous US",
             "license": "Public domain (USDA)", "status": "available, not draped yet"},
}
# National open lidar outside the US: listed so a user can see what exists; not wired into scenes yet.
NATIONAL_LIDAR = {
    "GB-ENG": ("Environment Agency National LIDAR Programme, 1 m", "Open Government Licence v3",
               "https://environment.data.gov.uk/survey"),
    "GB-WLS": ("Welsh Government LiDAR 2020-23, 1 m", "Open Government Licence v3",
               "https://datamap.gov.wales/maps/lidar-data-download/"),
    "GB-SCT": ("Scottish Public Sector LiDAR (partial coverage)", "Open Government Licence v3",
               "https://remotesensingdata.gov.scot/"),
    "CH": ("swisstopo swissALTI3D, 0.5 m", "Open (swisstopo OGD, free use with attribution)",
           "https://www.swisstopo.admin.ch/en/height-model-swissalti3d"),
    "FR": ("IGN LiDAR HD", "Etalab Open Licence 2.0", "https://geoservices.ign.fr/lidarhd"),
    "NO": ("Kartverket høydedata, 1 m", "CC BY 4.0", "https://hoydedata.no/"),
    "IE": ("Geological Survey Ireland lidar (partial coverage)", "CC BY 4.0",
           "https://www.gsi.ie/en-ie/data-and-maps/Pages/default.aspx"),
    "NZ": ("LINZ national 1 m DEM (lidar), public AWS bucket nz-elevation", "CC BY 4.0",
           "https://registry.opendata.aws/nz-elevation/"),
    "AU": ("Geoscience Australia ELVIS lidar (partial coverage)", "CC BY 4.0", "https://elevation.fsdf.org.au/"),
    "JP": ("GSI 5 m DEM (partly lidar)", "GSI terms (attribution)", "https://fgd.gsi.go.jp/download/"),
}


COUNTRY_NAMES = {"US": "United States USA America", "CH": "Switzerland", "FR": "France", "IT": "Italy",
                 "NO": "Norway", "GB": "United Kingdom UK Britain", "IE": "Ireland", "GR": "Greece",
                 "ZA": "South Africa", "TZ": "Tanzania", "NP": "Nepal", "JP": "Japan", "HK": "Hong Kong China",
                 "CN": "China", "AU": "Australia", "NZ": "New Zealand", "BR": "Brazil", "PE": "Peru"}


@lru_cache(maxsize=1)
def load_catalog(path: str | None = None) -> dict:
    return json.loads(Path(path or CATALOG).read_text())


def places() -> list[dict]:
    return load_catalog()["places"]


def register_place(place: dict) -> dict:
    """Add a place that is not in the catalogue (a custom point, e.g. from the Touchdown app) for this process,
    so ``find_place`` returns it by id. It needs at least id, name, latitude and longitude; an entry with the
    same id is replaced. Nothing is written to the catalogue file."""
    missing = [key for key in ("id", "name", "latitude", "longitude") if place.get(key) in (None, "")]
    if missing:
        raise ValueError("a registered place needs " + ", ".join(missing))
    entry = {"country": "", "region": "", "kind": "", "coverage": {}, **place}
    rows = load_catalog()["places"]
    rows[:] = [row for row in rows if row["id"] != entry["id"]] + [entry]
    return entry


def place_ids() -> list[str]:
    return [place["id"] for place in places()]


def catalog_option(place_id: str) -> str:
    """The GeoBlock Place From Catalog dropdown entry for a place id."""
    place = find_place(place_id)
    return f"{place['id']} · {place['name']}"


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def find_place(key: str) -> dict:
    """A catalog place by id, or by a unique match on its name, region or country."""
    key = key.strip()
    for place in places():
        if place["id"] == key:
            return place
    found = [place for place in places() if matches(place, key)]
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(f"No catalog place matches {key!r}; see geoblock-run --list-places")
    raise KeyError(f"{key!r} matches several places: " + ", ".join(place["id"] for place in found))


def bearing_deg(latitude: float, longitude: float, to_latitude: float, to_longitude: float) -> float:
    """Initial great-circle bearing, degrees clockwise from north."""
    p1, p2 = math.radians(latitude), math.radians(to_latitude)
    dl = math.radians(to_longitude - longitude)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def landing(place: dict) -> tuple[float, float]:
    """Where the person stands: the curated point, or where the catalog build moved it onto the top."""
    moved = place.get("coverage", {}).get("landing")
    if moved:
        return moved["latitude"], moved["longitude"]
    return place["latitude"], place["longitude"]


def view_heading(place: dict) -> float:
    """Camera heading at the landing: toward ``view_toward`` when set, else -1 (toward the measured drop)."""
    target = place.get("view_toward")
    if not target:
        return -1.0
    latitude, longitude = landing(place)
    return round(bearing_deg(latitude, longitude, target["latitude"], target["longitude"]), 1)


def matches(place: dict, text: str) -> bool:
    """Every word of ``text`` appears in the place's id, name, region, country (code or name) or kind."""
    words = _fold(text).split()
    fields = [str(place.get(key, "")) for key in ("id", "name", "region", "country", "kind")]
    haystack = _fold(" ".join(fields + [COUNTRY_NAMES.get(place.get("country", ""), "")]))
    return all(word in haystack for word in words)


def catalog_report(filter_text: str = "") -> str:
    """One line per place: id, name, terrain, lidar and the measured relief."""
    rows = [place for place in places() if not filter_text or matches(place, filter_text)]
    lines = [f"{len(rows)} of {len(places())} catalog places" + (f" matching {filter_text!r}" if filter_text else "")]
    for place in rows:
        coverage = place.get("coverage", {})
        terrain = TERRAIN_SOURCES.get(coverage.get("terrain"), {})
        lidar = coverage.get("lidar") or []
        lidar_text = (f"lidar {lidar[0]['name']} ({lidar[0].get('year') or '?'}, "
                      f"{lidar[0].get('density_pts_m2') or '?'} pts/m²)" if lidar
                      else coverage.get("national_lidar", ["no open lidar listed"])[0])
        relief = coverage.get("relief_m")
        lines.append(f"  {place['id']:<32} {place['name']} [{place.get('country', '')}] · "
                     f"{terrain.get('dataset', 'terrain unmeasured')} {terrain.get('resolution_m', '?')} m · "
                     f"{lidar_text}" + (f" · relief {relief:.0f} m" if relief is not None else ""))
    return "\n".join(lines)


def place_report(place: dict) -> str:
    coverage = place.get("coverage", {})
    terrain = TERRAIN_SOURCES.get(coverage.get("terrain"), {})
    lines = [f"PLACE: {place['name']} ({place['id']})",
             f"WHERE: {place.get('region', '')}, {place.get('country', '')} · {place.get('kind', '')}",
             "LANDING POINT: {:.5f}, {:.5f}".format(*landing(place))
             + (f" · ground {coverage['elevation_m']:.0f} m" if coverage.get("elevation_m") is not None else "")
             + (f" (moved {coverage['landing']['moved_m']:.0f} m: {coverage['landing']['reason']})"
                if coverage.get("landing") else "")]
    heading = view_heading(place)
    if heading >= 0:
        lines.append(f"VIEW: toward {place['view_toward']['name']}, heading {heading:.0f}°")
    else:
        lines.append("VIEW: toward the deepest measured drop")
    if terrain:
        lines.append(f"TERRAIN: {terrain['dataset']}, {terrain['resolution_m']} m, {terrain['model']}, "
                     f"{terrain['vertical']} · {terrain['license']}")
    for survey in coverage.get("lidar") or []:
        lines.append(f"LIDAR: {survey['name']} ({survey.get('year') or '?'}, {survey.get('density_pts_m2') or '?'} "
                     f"pts/m², {survey['points']:,} points) · {survey['license']}")
    if coverage.get("dem1m"):
        lines.append("LIDAR 1 m DEM (landing patch): " + ", ".join(p["project"] for p in coverage["dem1m"])
                     + " · Public domain (U.S. Geological Survey)")
    if coverage.get("near_view_deg") is not None:
        lines.append(f"DOWN VIEW: {abs(coverage['near_view_deg']):.0f}° below level toward the view, on 1 m lidar")
    if coverage.get("lidar_landing"):
        brink = coverage["lidar_landing"]
        lines.append(f"LIDAR LANDING: {brink['latitude']:.5f}, {brink['longitude']:.5f} ({brink['reason']}); "
                     "used with --landing-lidar")
    if coverage.get("national_lidar"):
        name, licence, url = coverage["national_lidar"]
        lines.append(f"NATIONAL LIDAR (not wired yet): {name} · {licence} · {url}")
    for key in coverage.get("imagery", []):
        source = IMAGERY_SOURCES[key]
        lines.append(f"IMAGERY: {source['dataset']}, {source['resolution_m']} m · {source['license']} · {source['status']}")
    if coverage.get("landcover"):
        from .landcover import describe
        lines.append(f"GROUND: {describe(coverage['landcover']['near'])}; VIEW: {describe(coverage['landcover']['view'])}")
    for warning in coverage.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


def place_for_pipeline(place: dict, radius_km: float | None = None) -> dict:
    """The GEOBLOCK_PLACE dict the scene builder takes, carrying the catalog's view and look."""
    coverage = place.get("coverage", {})
    latitude, longitude = landing(place)
    look = dict(coverage["look"]) if coverage.get("look") else None
    if look and place.get("view_toward"):
        look["far"] = f"{place['view_toward']['name']} stands out in the view beyond; " + look["far"]
    if look and coverage.get("city"):  # measured Overture buildings: name the city, not only "town" landcover
        from .buildings import city_text
        city = city_text(coverage["city"])
        look["low"] = f"{look['low']}; {city}"
        look["all"] = f"{look['all']}; {city}"
    return {"query": place["name"], "display_name": place["name"], "latitude": latitude,
            "longitude": longitude, "requested_radius_km": radius_km or place.get("radius_km", 12),
            "source": "GeoBlock place catalog", "source_id": place["id"], "feature_type": place.get("kind"),
            "extent": None, "catalog_id": place["id"], "heading_deg": view_heading(place),
            "palette": place.get("palette", "default"), "kind": place.get("kind", ""),
            "look": look, "lidar_landing": lidar_landing(place)}


def lidar_landing(place: dict) -> dict | None:
    """Where the person stands when the shot uses the 1 m lidar landing patch, if that differs from the
    landing: an overlook's brink, which the 10 m DEM rounds off. Carries the heading from there."""
    brink = place.get("coverage", {}).get("lidar_landing")
    if not brink:
        return None
    target = place.get("view_toward")
    heading = (round(bearing_deg(brink["latitude"], brink["longitude"], target["latitude"], target["longitude"]), 1)
               if target else view_heading(place))
    return {"latitude": brink["latitude"], "longitude": brink["longitude"], "heading_deg": heading,
            "reason": brink["reason"]}
