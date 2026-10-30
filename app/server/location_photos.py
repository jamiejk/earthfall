"""Cached Commons discovery. Downloaded references are leads, never gate approvals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time

import httpx
from geoblock_core.place_photos import fetch_place_photos

from . import places
from .images import BadImage, sanitize

# Bound discovery across requests, including two tabs requesting the same place.
_discovery_lock = threading.Lock()


def destination_key(destination: dict) -> str:
    return hashlib.sha256(json.dumps(destination, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ReferenceLibrary:
    def __init__(self, settings, landmarks):
        self.root = settings.data_dir / "location-photos"
        self.repo = settings.repo
        self.landmarks = landmarks

    def load(self, key: str) -> dict:
        return json.loads((self.root / key / "references.json").read_text())

    def discover(self, destination: dict, retry: bool = False) -> dict:
        key = destination_key(destination)
        self.root.mkdir(parents=True, exist_ok=True)
        with _discovery_lock:
            try:
                cached = self.load(key)
                intact = cached.get("version") == 1 and all((self.root / key / p["name"]).is_file() for p in cached["photos"])
                if intact and (cached["photos"] or (not retry and time.time() - cached["created"] < 600)):
                    return cached
            except (FileNotFoundError, ValueError, KeyError):
                pass
            with tempfile.TemporaryDirectory(prefix=".discover-", dir=self.root) as temp:
                folder = Path(temp)
                photos = []
                status = "unavailable"
                message = "Location photo search is unavailable just now. Try again, or add your own photos."
                try:
                    photos = self._retrieve(destination, folder)
                    status = "found" if len(photos) >= 2 else "insufficient"
                    message = ("Photos found on Wikimedia Commons. Check that they show the setting you want."
                               if len(photos) >= 2 else "We couldn't find enough useful location photos. Add your own to guide the setting.")
                except ValueError:
                    status = "empty"
                    message = "We couldn't find suitable location photos. Try a more specific place, or add your own."
                except (httpx.HTTPError, OSError, RuntimeError):
                    pass
                record = {"id": key, "version": 1, "destination": destination, "photos": photos,
                          "status": status, "message": message, "created": time.time()}
                # Only clean images and provenance enter the cache; discard raw downloads.
                for child in folder.iterdir():
                    if child.name not in {p["name"] for p in photos}:
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                (folder / "references.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                target = self.root / key
                if target.exists():
                    shutil.rmtree(target)
                folder.rename(target)
                return record

    def _retrieve(self, destination: dict, folder: Path) -> list[dict]:
        known = next((p for p in self.landmarks if p["id"] == destination["id"]), None)
        seeds = [r for r in known["references"] if (self.repo / r["path"]).is_file()] if known else []
        if len(seeds) >= 2:
            selected = [{**r, "local_path": str(self.repo / r["path"])} for r in seeds]
        else:
            latitude, longitude = destination.get("latitude"), destination.get("longitude")
            if latitude is None:
                matches = places.geocode(destination["title"], country=None)
                # Don't silently choose one of several different places.
                if len(matches) != 1:
                    raise ValueError("Choose a specific search result first.")
                latitude, longitude = matches[0]["latitude"], matches[0]["longitude"]
            manifest = fetch_place_photos(destination["title"], latitude, longitude, 3, folder / "raw",
                                          radius_km=2, purpose="photo_op")
            selected = manifest["selected"]
        photos = []
        for item in selected[:4]:
            original = Path(item["local_path"]).read_bytes()
            # One unreadable Commons file must not discard the good ones (Chatsworth, 29 Oct).
            try:
                path = sanitize(original, folder / f"location-{len(photos)}")
            except BadImage:
                continue
            photos.append({"name": path.name, "source": "Wikimedia Commons", "page_id": item["page_id"],
                           "title": item["title"], "artist": item["artist"], "license": item["license"],
                           "license_url": item["license_url"], "url": item.get("commons_page") or item["url"],
                           "source_sha256": hashlib.sha256(original).hexdigest(),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return photos

    @staticmethod
    def public(record: dict) -> dict:
        return {**record, "photos": [{**p, "image_url": f"/api/studio/location-photos/{record['id']}/{p['name']}"}
                                    for p in record["photos"]]}
