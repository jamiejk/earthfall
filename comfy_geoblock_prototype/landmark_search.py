"""User-triggered landmark search, independent of the prepared demo scenes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "https://photon.komoot.io/api"
DEFAULT_CACHE = ROOT / "data/landmark_search_cache"
USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"


def _endpoint() -> str:
    endpoint = os.environ.get("GEOBLOCK_PHOTON_URL", DEFAULT_ENDPOINT).rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path != "/api":
        raise ValueError("GEOBLOCK_PHOTON_URL must be a Photon /api URL")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for a remote Photon endpoint")
    return endpoint


def _candidate(feature: dict) -> dict | None:
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") != "Point" or len(coordinates) < 2:
        return None
    lon, lat = float(coordinates[0]), float(coordinates[1])
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    name = props.get("name")
    if not name:
        return None
    location_parts = []
    for part in (name, props.get("city"), props.get("state"), props.get("country")):
        if part and part not in location_parts:
            location_parts.append(part)
    osm_type, osm_id = props.get("osm_type"), props.get("osm_id")
    return {
        "name": name,
        "display_name": ", ".join(location_parts),
        "latitude": lat,
        "longitude": lon,
        "osm_id": f"{osm_type}{osm_id}" if osm_type and osm_id else None,
        "feature_type": "/".join(filter(None, (props.get("osm_key"), props.get("osm_value")))),
        "extent": props.get("extent"),
    }


def search_landmarks(query: str, *, cache_dir: Path = DEFAULT_CACHE) -> tuple[dict, str]:
    """Search Photon only when a user queues this node; cache repeated queries."""
    query = " ".join(query.split())
    if not 3 <= len(query) <= 200:
        raise ValueError("Enter a landmark name with at least three characters")
    endpoint = _endpoint()
    key = hashlib.sha256(f"{endpoint}\n{query.casefold()}".encode()).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        cache_status = "cached"
    else:
        url = endpoint + "?" + urlencode({"q": query, "limit": 5, "lang": "en"})
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ValueError("Landmark search is unavailable; retry or configure GEOBLOCK_PHOTON_URL") from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False))
        temp.replace(cache_path)
        cache_status = "fetched"
    candidates = []
    seen = set()
    for feature in payload.get("features", []):
        candidate = _candidate(feature)
        if candidate is None:
            continue
        identity = candidate["osm_id"] or (candidate["name"], candidate["latitude"], candidate["longitude"])
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("No landmark matches. Add a town or region to the search")
    result = {"query": query, "provider": "Photon/OpenStreetMap", "candidates": candidates}
    lines = [f"LANDMARK SEARCH: {query} ({cache_status})", "Choose a result number in the next node:"]
    for number, item in enumerate(candidates, 1):
        lines.append(f"{number}. {item['display_name']} [{item['feature_type'] or 'place'}] "
                     f"{item['latitude']:.6f}, {item['longitude']:.6f}"
                     + (f" · OSM {item['osm_id']}" if item["osm_id"] else ""))
    lines.append("Place data © OpenStreetMap contributors (ODbL). Search does not verify 3D coverage.")
    return result, "\n".join(lines)


def choose_landmark(results: dict, result_number: int, radius_km: float) -> tuple[dict, str]:
    if not 0.1 <= radius_km <= 20:
        raise ValueError("Coverage radius must be between 0.1 and 20 km")
    candidates = results.get("candidates", [])
    if not 1 <= result_number <= len(candidates):
        raise ValueError(f"Choose a result number from 1 to {len(candidates)}")
    selected = candidates[result_number - 1]
    place = {
        "query": results["query"],
        "display_name": selected["display_name"],
        "latitude": selected["latitude"],
        "longitude": selected["longitude"],
        "requested_radius_km": radius_km,
        "source": results["provider"],
        "source_id": selected["osm_id"],
        "feature_type": selected["feature_type"],
        "extent": selected["extent"],
    }
    report = (f"SELECTED: {place['display_name']}\n"
              f"COORDINATES: {place['latitude']:.6f}, {place['longitude']:.6f}\n"
              f"OSM FEATURE: {place['source_id'] or 'unavailable'} · {place['feature_type']}\n"
              f"REQUESTED AREA: {radius_km:g} km radius\n"
              "3D STATUS: not checked. The chosen landmark needs actual terrain/shape coverage before a shot.")
    return place, report
