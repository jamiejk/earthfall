"""The three places with a hand-written, tested FLUX 3 manifest. Every other place in the picker comes from the
place catalogue (places.py) and gets a generated manifest on the worker (touchdown.manifest_gen).

The server only needs display data and the default person text; the worker reads the full manifest from its own
checkout of the repository (``manifest`` is relative to the repo root on both machines).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PRESETS: list[dict] = [
    {
        "id": "monument-valley",
        "place": "monument-valley",  # catalogue id
        "title": "Monument Valley",
        "region": "Utah / Arizona",
        "blurb": "Open red sand and slickrock, with the sandstone buttes rising behind.",
        "latitude": 37.012,
        "longitude": -110.075,
        "manifest": "shots/flux3/monument-valley.toml",
    },
    {
        "id": "grand-canyon-mather",
        "place": "grand-canyon-mather-point",
        "title": "Grand Canyon, Mather Point",
        "region": "Arizona",
        "blurb": "Level ground on the South Rim, layered canyon cliffs beyond.",
        "latitude": 36.0617,
        "longitude": -112.1078,
        "manifest": "shots/flux3/grand-canyon-mather.toml",
    },
    {
        "id": "yosemite-tunnel-view",
        "place": "yosemite-tunnel-view",
        "title": "Yosemite, Tunnel View",
        "region": "California",
        "blurb": "The paved overlook, with El Capitan, Bridalveil Fall and Half Dome across the valley.",
        "latitude": 37.7156,
        "longitude": -119.6772,
        "manifest": "shots/flux3/yosemite-tunnel-view.toml",
    },
]

# Any point in the lower 48 with USGS 1 m lidar: the worker generates its manifest (touchdown.manifest_gen) and the
# build searches the landing and the view itself. Nothing has been built this way yet, hence "experimental".
CUSTOM_LOCATION = {"enabled": True, "experimental": True,
                   "note": "Experimental: any spot in the lower 48 with 1 m lidar terrain. The place words and the "
                           "landing are worked out automatically and have not been tried in a real build yet."}

FALLBACK_PERSON = "a person in everyday outdoor clothes"
# The default when the user leaves "what they're wearing" empty. The preset manifests describe the test subject
# ("a young woman ... wearing a long red raincoat"), but the app always has the user's own photo, and the end
# still is told to keep "their exact face, hair, body and clothing". Describing the test subject there gives the
# model two contradictory instructions and a judge in a blue shirt can end up in a red raincoat.
PHOTO_PERSON = "the person in the reference photo"


def by_id(preset_id: str) -> dict | None:
    return next((p for p in PRESETS if p["id"] == preset_id), None)


def default_person(preset: dict, repo: Path) -> str:
    """The text used when the user leaves "what they're wearing" empty.

    The photo is the source of truth, so this is photo-neutral; the preset manifest's own description of the test
    subject is deliberately not used here (see PHOTO_PERSON).
    """
    return PHOTO_PERSON


def public(repo: Path) -> list[dict]:
    return [{k: v for k, v in p.items() if k not in ("manifest", "place")} | {"default_person": default_person(p, repo),
                                                                "thumb": f"/static/img/{p['id']}.svg"}
            for p in PRESETS]
