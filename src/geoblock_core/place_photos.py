"""Find and download openly licensed photos of a place for H3's reference slots.

Wikimedia Commons is searched around the coordinates (geosearch) and by name.
Only JPEGs whose Commons metadata names CC0, CC BY, CC BY-SA or public domain
are eligible. Candidates are ranked by distance, name match, Commons quality
marks and shape, and at most one photo per author is kept so a burst of near
identical shots cannot fill every slot. Commons coordinates can describe the
subject rather than the camera, so the selection is a lead for a human to
review, and every pick keeps its page, author, licence and SHA-256.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlparse

import httpx
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"
# Commons began serving imageinfo thumbnails from thumb.wikimedia.org on 26 Oct 2026 (originals stay on upload).
UPLOAD_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}
LICENCES = re.compile(r"^(CC (?:BY|BY-SA)(?: 2\.0| 2\.5| 3\.0| 4\.0)?|CC0(?: 1\.0)?|Public domain)$", re.I)
QUALITY_CATEGORIES = ("Category:Quality images", "Category:Featured pictures on Wikimedia Commons",
                      "Category:Valued images sorted by promotion date")
# Titles that are about something other than the landscape.
OFF_SUBJECT = re.compile(
    r"\b(map|chart|diagram|sign|signs|notice|poster|plaque|logo|seal|menu|hotel|restaurant|lodge|"
    r"interior|museum|exhibit|archaeology|ranger|program|ceremony|dance|festival|visitors?|tourists?|"
    r"people|crowd|portrait|selfie|wedding|car|bus|shuttle|parking|construction|camp|campground|campsite|tents?|"
    r"bird|raven|corvus|condor|elk|deer|squirrel|lizard|snake|flower|plant)\b", re.I)
# Unusual conditions: fine as photos, poor as the look of an ordinary daylight shot.
UNUSUAL = re.compile(r"\b(fog|snow|storm|inversion|night|stars?|milky way|lightning|fire|smoke|eclipse)\b", re.I)
# Views from down in a valley or on a trail: real, but wrong for an aerial or rim-level camera.
BELOW_RIM = re.compile(r"\b(river|rapids?|riverside|creek|trail|trailhead|hike|hikers?|hiking|mules?|"
                       r"bridge|ranch|bottom|inner gorge|plateau point)\b", re.I)
WORD = re.compile(r"[a-z0-9]+")
COMPASS = {name: index * 22.5 for index, name in enumerate(
    ("north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast", "southeast",
     "south-southeast", "south", "south-southwest", "southwest", "west-southwest", "west",
     "west-northwest", "northwest", "north-northwest"))}
VIEW_DIRECTION = re.compile(r"\b(?:view|looking|panorama)\s+(" + "|".join(sorted(COMPASS, key=len, reverse=True)) + r")\b", re.I)
STOP = {"the", "of", "and", "at", "on", "in", "a", "point", "national", "park"}
PHOTO_OP_EXCLUDE = re.compile(r"\b(map|diagram|logo|seal|poster|interior|museum|exhibit|menu|selfie)\b", re.I)


def camera_location(wikitext: str) -> dict | None:
    """Read an explicit Commons camera template; never promote object coordinates."""
    match = re.search(r"\{\{\s*(?:location(?: dec)?|camera location)\s*\|([^{}]+)\}\}", wikitext, re.I)
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split("|")]
    try:
        if len(parts) >= 8 and parts[3].upper() in ("N", "S") and parts[7].upper() in ("E", "W"):
            lat = sum(float(parts[i]) / scale for i, scale in zip(range(3), (1, 60, 3600)))
            lon = sum(float(parts[i]) / scale for i, scale in zip(range(4, 7), (1, 60, 3600)))
            lat *= -1 if parts[3].upper() == "S" else 1
            lon *= -1 if parts[7].upper() == "W" else 1
        else:
            lat, lon = float(parts[0]), float(parts[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
    except (ValueError, IndexError):
        return None
    direction = re.search(r"heading\s*[:=]\s*([0-9.]+|[NSEW]{1,3})(?=[_|\s}]|$)", match.group(1), re.I)
    heading = None
    if direction:
        value = direction.group(1).upper()
        abbreviations = dict(zip(("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"), range(16)))
        heading = abbreviations[value] * 22.5 if value in abbreviations else float(value) % 360
    return {"latitude": lat, "longitude": lon, "heading_deg": heading,
            "source": "explicit Commons camera-location template", "accuracy_m": None}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain(value: str | None) -> str:
    parser = _Text()
    parser.feed(value or "")
    return html.unescape(" ".join("".join(parser.parts).split()))


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def _name_tokens(name: str) -> set[str]:
    return {word for word in WORD.findall(name.lower()) if word not in STOP and len(word) > 2}


def score_candidate(page: dict, name: str, latitude: float, longitude: float,
                    radius_km: float, heading_deg: float = -1, *, purpose: str = "landscape") -> dict | None:
    """A ranked candidate from one Commons API page, or None if it is ineligible."""
    info = (page.get("imageinfo") or [{}])[0]
    width, height = info.get("width", 0), info.get("height", 0)
    if info.get("mime") != "image/jpeg" or min(width, height) < 900:
        return None
    ext = info.get("extmetadata", {})
    licence = _plain(ext.get("LicenseShortName", {}).get("value"))
    if not LICENCES.fullmatch(licence):
        return None
    image_url = info.get("thumburl") or info.get("url")
    if not image_url or urlparse(image_url).hostname not in UPLOAD_HOSTS:
        return None
    title = page["title"].removeprefix("File:")
    if (PHOTO_OP_EXCLUDE if purpose == "photo_op" else OFF_SUBJECT).search(title):
        return None
    aspect = width / height
    if not 1.2 <= aspect <= 2.4:  # portrait frames and wide panoramas squash badly into a 16:9 slot
        return None
    revision = (page.get("revisions") or [{}])[0]
    camera = camera_location(revision.get("slots", {}).get("main", {}).get("content", ""))
    coord = ({"lat": camera["latitude"], "lon": camera["longitude"]} if camera
             else (page.get("coordinates") or [{}])[0])
    dist = None
    if "lat" in coord and "lon" in coord:
        dist = round(distance_km(latitude, longitude, coord["lat"], coord["lon"]), 3)
        if dist > radius_km:
            return None
    words = set(WORD.findall(title.lower()))
    tokens = _name_tokens(name)
    score = 3.0 * (len(tokens & words) / len(tokens) if tokens else 0)
    score += 2.0 * (1 - dist / radius_km) if dist is not None else 0
    categories = {category["title"] for category in page.get("categories", [])}
    score += 2.0 if categories & set(QUALITY_CATEGORIES) else 0
    score -= 2.0 if UNUSUAL.search(title) else 0
    score -= 1.5 if purpose != "photo_op" and BELOW_RIM.search(title) else 0
    score += 0.5 if re.search(r"\b(view|views|vista|overlook|rim|panorama)\b", title, re.I) else 0
    score += min(width * height / 12_000_000, 1.0)
    direction = VIEW_DIRECTION.search(title)
    if camera and direction and camera.get("heading_deg") is not None:
        difference = abs((camera["heading_deg"] - COMPASS[direction.group(1).lower()] + 180) % 360 - 180)
        if difference > 60:
            camera = {**camera, "reported_heading_deg": camera["heading_deg"],
                      "heading_deg": None, "heading_conflict": "camera metadata conflicts with title direction"}
    if direction and heading_deg >= 0:  # a titled view direction close to the shot's heading
        off = abs((COMPASS[direction.group(1).lower()] - heading_deg + 180) % 360 - 180)
        score += 1.5 if off <= 30 else 0.5 if off <= 60 else 0
    return {
        "page_id": page["pageid"], "title": title, "commons_page": info.get("descriptionurl"),
        "image_url": image_url, "license": licence, "license_url": _plain(ext.get("LicenseUrl", {}).get("value")),
        "artist": _plain(ext.get("Artist", {}).get("value")) or "author not given",
        "width": width, "height": height, "distance_km": dist,
        "quality_mark": bool(categories & set(QUALITY_CATEGORIES)), "score": round(score, 3),
        "camera_location": camera,
        "coordinate_role": "camera" if camera else "unverified page coordinate",
        "page_coordinate": (page.get("coordinates") or [None])[0],
        "description": _plain(ext.get("ImageDescription", {}).get("value")),
        "capture_time": _plain(ext.get("DateTimeOriginal", {}).get("value")) or None,
    }


def _query(client: httpx.Client, params: dict) -> list[dict]:
    base = {"action": "query", "format": "json", "formatversion": 2,
            "prop": "imageinfo|coordinates|categories|revisions", "iiprop": "url|size|extmetadata|mime",
            "rvprop": "content", "rvslots": "main",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|ImageDescription|DateTimeOriginal", "iiurlwidth": 2048,
            "clcategories": "|".join(QUALITY_CATEGORIES), "cllimit": "max", "colimit": "max"}
    for attempt in range(4):  # Commons search sheds load with "too busy"; back off and retry
        response = client.get(API, params={**base, **params})
        response.raise_for_status()
        result = response.json()
        if result.get("error", {}).get("code") != "cirrussearch-too-busy-error" or attempt == 3:
            break
        time.sleep(5 * 2 ** attempt)
    if "error" in result:
        raise RuntimeError(f"Commons API error: {result['error']}")
    return result.get("query", {}).get("pages", [])


def find_place_photos(name: str, latitude: float, longitude: float, *, radius_km: float = 2.0,
                      heading_deg: float = -1, client: httpx.Client | None = None,
                      purpose: str = "landscape") -> list[dict]:
    """Eligible Commons photos near the place, best first; ``heading_deg`` favours views that way."""
    if not 0.1 <= radius_km <= 10:
        raise ValueError("Commons geosearch radius must be 0.1–10 km.")
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)
    try:
        pages = _query(client, {"generator": "geosearch", "ggscoord": f"{latitude}|{longitude}",
                                "ggsradius": int(radius_km * 1000), "ggsnamespace": 6, "ggslimit": 100})
        if _name_tokens(name):
            pages += _query(client, {"generator": "search", "gsrsearch": name.replace(",", " "),
                                     "gsrnamespace": 6, "gsrlimit": 50})
    finally:
        if own:
            client.close()
    found: dict[int, dict] = {}
    for page in pages:
        item = score_candidate(page, name, latitude, longitude, radius_km, heading_deg, purpose=purpose)
        if item:
            found[item["page_id"]] = item
    return sorted(found.values(), key=lambda item: (-item["score"], item["page_id"]))


def pick_diverse(candidates: list[dict], count: int) -> list[dict]:
    """The best ``count`` candidates, at most one per author."""
    picked, authors = [], set()
    for item in candidates:
        # Commons edits often repeat one photographer's image with an editor
        # appended to the credit. Those are not independent scene references.
        author = re.split(r";|\bedited by\b", item["artist"].lower(), maxsplit=1)[0]
        author = re.sub(r"\s*\(talk\)", "", author).strip()
        if author in authors:
            continue
        authors.add(author)
        picked.append(item)
        if len(picked) == count:
            break
    return picked


def _download(client: httpx.Client, item: dict, folder: Path, max_bytes: int) -> dict:
    path = folder / f"commons_{item['page_id']}.jpg"
    if not path.exists():
        temp = path.with_suffix(".part")
        try:
            with client.stream("GET", item["image_url"]) as response:
                response.raise_for_status()
                if urlparse(str(response.url)).hostname not in UPLOAD_HOSTS:
                    raise ValueError(f"unexpected download host for {item['title']}")
                size = 0
                with temp.open("wb") as file:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"{item['title']} is larger than {max_bytes // 2**20} MB")
                        file.write(chunk)
            with Image.open(temp) as image:
                image.verify()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return {**item, "local_path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fetch_place_photos(name: str, latitude: float, longitude: float, count: int, folder: Path, *,
                       radius_km: float = 2.0, heading_deg: float = -1, exclude: tuple[int, ...] = (),
                       max_mb: int = 25, purpose: str = "landscape") -> dict:
    """Download up to ``count`` photos into ``folder`` and write manifest.json and ATTRIBUTION.md there.

    ``exclude`` lists Commons page IDs a reviewer rejected; the next best photos take their place.
    """
    if not 1 <= count <= (16 if purpose == "photo_op" else 8):
        raise ValueError("Ask for 1–8 landscape photos or 1–16 photo-op references.")
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        candidates = find_place_photos(name, latitude, longitude, radius_km=radius_km,
                                       heading_deg=heading_deg, client=client, purpose=purpose)
        selected = []
        for item in pick_diverse([c for c in candidates if c["page_id"] not in exclude], len(candidates)):
            try:
                selected.append(_download(client, item, folder, max_mb * 2**20))
            except (httpx.HTTPError, OSError, ValueError) as exc:
                print(f"Skipping {item['title']}: {exc}")
            if len(selected) == count:
                break
    if not selected:
        raise ValueError(f"No openly licensed Commons photos found within {radius_km:g} km of {name}.")
    manifest = {"place": name, "latitude": latitude, "longitude": longitude, "radius_km": radius_km,
                "source": "Wikimedia Commons", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": purpose,
                "note": "Commons coordinates may describe the subject, not the camera. Review the picks.",
                "excluded_page_ids": list(exclude),
                "selected": selected, "candidates": candidates}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (folder / "ATTRIBUTION.md").write_text(attribution(selected))
    return manifest


def photo_report(manifest: dict) -> str:
    lines = [f"PLACE PHOTOS: {len(manifest['selected'])} from Wikimedia Commons "
             f"({len(manifest['candidates'])} eligible within {manifest['radius_km']:g} km)"]
    lines += [f"{index}. page {item['page_id']}: {item['title']} by {item['artist']}, {item['license']}"
              for index, item in enumerate(manifest["selected"], start=1)]
    lines.append("Review them: exclude a page ID that shows people, signs or the wrong place.")
    return "\n".join(lines)


def attribution(selected: list[dict]) -> str:
    lines = ["# Place photo attribution", "", "Photos from Wikimedia Commons. See each page for full licence terms.", ""]
    lines += [f"- {item['title']} by {item['artist']}, {item['license']}: {item['commons_page']}"
              for item in selected]
    return "\n".join(lines) + "\n"


def route_waypoint(scene: dict, route: dict, frame: int) -> dict:
    """Latitude, longitude and compass heading of the route camera at ``frame`` (0-based)."""
    from pyproj import Transformer
    frame = max(0, min(frame, len(route["positions"]) - 1))
    (x, y, z), (tx, ty, _) = route["positions"][frame], route["look_at"][frame]
    east, north = scene["origin_projected_m"]
    to_wgs84 = Transformer.from_crs(scene["projected_crs"], "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(east + x, north + y)
    return {"frame": frame, "latitude": round(lat, 6), "longitude": round(lon, 6), "altitude_m": round(z, 1),
            "heading_deg": round(math.degrees(math.atan2(tx - x, ty - y)) % 360, 1)}


def fetch_waypoint_photos(scene: dict, route: dict, frame: int, count: int, folder: Path, *,
                          radius_km: float = 3.0, exclude: tuple[int, ...] = ()) -> dict:
    """Commons photos taken near where the camera is at ``frame``, favouring views along its heading."""
    point = route_waypoint(scene, route, frame)
    # Geosearch only: a name search would pull in the landmark's famous shots wherever they were taken.
    manifest = fetch_place_photos("", point["latitude"], point["longitude"], count, folder,
                                  radius_km=radius_km, heading_deg=point["heading_deg"], exclude=exclude)
    manifest["place"], manifest["waypoint"] = scene["name"], point
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
