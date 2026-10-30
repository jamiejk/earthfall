"""Discover and download licensed, location-relevant Wikimedia Commons photos.

The discovery manifest records every candidate so a human can audit relevance.
Coordinates on Commons may denote the subject rather than the camera position.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import httpx
from PIL import Image

UPLOAD_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}  # Commons thumbnails moved to thumb.* on 26 Oct 2026

API = "https://commons.wikimedia.org/w/api.php"
AGENT = "GeoBlockReferenceMiner/0.1 (Wikimedia Commons imagery research; geoblock@example.org)"
ALLOWED_LICENSES = re.compile(r"^(CC (?:BY|BY-SA)(?: 2\.0| 2\.5| 3\.0| 4\.0)?|CC0(?: 1\.0)?|Public domain)$", re.I)
BAD_TITLE = re.compile(r"\b(hotel|notice|sign|chart|map|poster|restaurant|menu|seal|fisherman|anchor|car park)\b", re.I)


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain(value: str | None) -> str:
    parser = _Text()
    parser.feed(value or "")
    return html.unescape(" ".join("".join(parser.parts).split()))


def distance_km(a: float, b: float, c: float, d: float) -> float:
    ar, cr = math.radians(a), math.radians(c)
    dr = math.radians(c - a)
    dl = math.radians(d - b)
    h = math.sin(dr / 2) ** 2 + math.cos(ar) * math.cos(cr) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def request_pages(client: httpx.Client, params: dict, max_pages: int = 8) -> list[dict]:
    base = {"action": "query", "format": "json", "formatversion": 2,
            "prop": "imageinfo|coordinates", "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": 2048, **params}
    found: list[dict] = []
    continuation: dict = {}
    for _ in range(max_pages):
        response = client.get(API, params={**base, **continuation})
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        found.extend(result.get("query", {}).get("pages", []))
        continuation = result.get("continue", {})
        if not continuation:
            break
    return found


def meta(value: dict, key: str) -> str:
    return plain(value.get(key, {}).get("value"))


def candidate(page: dict, latitude: float, longitude: float) -> dict | None:
    info = (page.get("imageinfo") or [{}])[0]
    if info.get("mime") != "image/jpeg" or min(info.get("width", 0), info.get("height", 0)) < 900:
        return None
    ext = info.get("extmetadata", {})
    license_name = meta(ext, "LicenseShortName")
    if not ALLOWED_LICENSES.fullmatch(license_name):
        return None
    title = page["title"].removeprefix("File:")
    if BAD_TITLE.search(title):
        return None
    image_url = info.get("thumburl") or info.get("url")
    if not image_url or urlparse(image_url).hostname not in UPLOAD_HOSTS:
        return None
    coord = (page.get("coordinates") or [{}])[0]
    dist = None
    if "lat" in coord and "lon" in coord:
        dist = round(distance_km(latitude, longitude, coord["lat"], coord["lon"]), 3)
    lower = title.lower().replace("’", "'")
    score = 0
    if "worm" in lower and "head" in lower:
        score += 5
    if "rhossili" in lower or "rhosili" in lower:
        score += 2
    if "causeway" in lower or "crossing" in lower:
        score += 2
    if "outer head" in lower or "inner head" in lower:
        score += 1
    if "panorama" in lower or "aerial" in lower:
        score += 1
    if dist is not None:
        score += max(0, 3 - dist)
        if dist > 12:
            score -= 10
    score += min(info["width"] * info["height"] / 8_000_000, 1)
    return {
        "page_id": page["pageid"], "title": title,
        "commons_page": info.get("descriptionurl"), "image_url": image_url,
        "original_url": info.get("url"),
        "license": license_name, "license_url": meta(ext, "LicenseUrl"),
        "artist": meta(ext, "Artist"), "credit": meta(ext, "Credit"),
        "description": meta(ext, "ImageDescription"),
        "width": info["width"], "height": info["height"],
        "coordinates": {"latitude": coord.get("lat"), "longitude": coord.get("lon")} if dist is not None else None,
        "distance_from_query_km": dist, "score": round(score, 3),
    }


def download(client: httpx.Client, item: dict, out: Path, max_bytes: int) -> dict:
    path = out / f"commons_{item['page_id']}.jpg"
    if not path.exists():
        with client.stream("GET", item["image_url"], follow_redirects=True) as response:
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";")[0] not in {"image/jpeg", "image/jpg"}:
                raise ValueError(f"Unexpected content type for {item['title']}")
            if urlparse(str(response.url)).hostname not in UPLOAD_HOSTS:
                raise ValueError(f"Unexpected download host for {item['title']}")
            digest = hashlib.sha256()
            size = 0
            temp = path.with_suffix(".part")
            try:
                with temp.open("wb") as file:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"Image exceeds limit: {item['title']}")
                        digest.update(chunk)
                        file.write(chunk)
                with Image.open(temp) as image:
                    image.verify()
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    data = path.read_bytes()
    with Image.open(path) as image:
        item["downloaded_dimensions"] = list(image.size)
    item["local_path"] = str(path.resolve())
    item["sha256"] = hashlib.sha256(data).hexdigest()
    item["bytes"] = len(data)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/worms_head_imagery.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reference_images/worms_head"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-mb", type=int, default=25)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    lat, lon = config["latitude"], config["longitude"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gathered: dict[int, dict] = {}
    with httpx.Client(headers={"User-Agent": AGENT}, timeout=60, follow_redirects=True) as client:
        queries = []
        for cat in config["categories"]:
            queries.append({"generator": "categorymembers", "gcmtitle": cat,
                            "gcmtype": "file", "gcmlimit": 50})
        for term in config["search_terms"]:
            queries.append({"generator": "search", "gsrsearch": term,
                            "gsrnamespace": 6, "gsrlimit": 50})
        queries.append({"generator": "geosearch", "ggscoord": f"{lat}|{lon}",
                        "ggsradius": 10000, "ggsnamespace": 6, "ggslimit": 50})
        for query in queries:
            for page in request_pages(client, query, max_pages=8):
                item = candidate(page, lat, lon)
                if item:
                    gathered[item["page_id"]] = item
        ranked = sorted(gathered.values(), key=lambda x: (-x["score"], x["page_id"]))
        pinned = config.get("selected_page_ids", [])
        selected = []
        for page_id in pinned:
            if page_id not in gathered:
                raise ValueError(f"Pinned Commons page {page_id} missing or fails licence/quality filters")
            selected.append(gathered[page_id])
        selected.extend(item for item in ranked if item["page_id"] not in pinned)
        selected = selected[:args.limit]
        for item in selected:
            item["role"] = config.get("roles", {}).get(str(item["page_id"]), "Place reference")
        if not args.discover_only:
            downloaded = []
            hashes = set()
            for item in selected:
                try:
                    item = download(client, item, args.output_dir, args.max_mb * 1024 * 1024)
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    print(f"Skipping {item['title']}: {exc}")
                    continue
                if item["sha256"] in hashes:
                    continue
                hashes.add(item["sha256"])
                downloaded.append(item)
            selected = downloaded
    manifest = {"location": config["name"], "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "query": {"latitude": lat, "longitude": lon},
                "source": "Wikimedia Commons MediaWiki API", "candidates": ranked,
                "selected": selected,
                "note": "Commons coordinates may describe the photo subject, not camera position. Review selected images visually."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    attribution = ["# Photo attribution", "", "Review each Commons page for the complete licence terms.", ""]
    for item in selected:
        attribution.append(f"- {item['title']} — {item['artist'] or 'author not provided'}; "
                           f"{item['license']} ({item['license_url'] or 'see Commons page'}); "
                           f"{item['commons_page']}")
    (args.output_dir / "ATTRIBUTION.md").write_text("\n".join(attribution) + "\n")
    if not args.discover_only:
        motion = Path(config["motion_reference"]).resolve()
        if not motion.exists():
            raise FileNotFoundError(f"Motion reference missing: {motion}")
        pack = {
            "mode": "MiniMax H3 Ref2VA",
            "location": config["name"],
            "motion_reference": str(motion),
            "image_references": [
                {"path": item["local_path"], "role": item["role"],
                 "commons_page": item["commons_page"], "sha256": item["sha256"]}
                for item in selected
            ],
            "prompt": (
                "A simple, restrained clay-model aerial flyover of the real Worm's Head "
                "promontory in Rhossili, Wales. Begin at the edge of the Rhossili headland "
                "looking west across the sea. Follow the motion reference: fly out over the "
                "water alongside the intertidal rocky causeway, pass Inner Head and Outer "
                "Head, turn smoothly through 180 degrees beyond Outer Head, then fly back "
                "east toward the Rhossili headland. Preserve the geography in the overhead "
                "and reverse-view photographs: Worm's Head joins the mainland via a tidal "
                "causeway at low tide; do not depict it as an isolated island. Use the photos "
                "for landform and sea appearance; keep the clay-model styling and camera "
                "path of the motion video. Continuous shot, calm daylight, no people, no text."
            ),
            "suggested_output": {"duration_seconds": 15, "fps": 24,
                                  "width": 1344, "height": 768},
        }
        (args.output_dir / "h3_reference_pack.json").write_text(
            json.dumps(pack, indent=2, ensure_ascii=False) + "\n"
        )
    print(f"{len(ranked)} eligible candidates; {len(selected)} selected. {args.output_dir / 'manifest.json'}")
    for item in selected:
        print(f"{item['page_id']}: {item['title']}")


if __name__ == "__main__":
    main()
