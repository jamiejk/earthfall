"""The place picker: catalogue places, which ones the pipeline can build, which ones have had a real build, and
the checks for a custom US point (1 m lidar coverage, and place-name search through OpenStreetMap Nominatim).

Catalogue places come from ``src/geoblock_core/data/places.json``. A place can be built when it is in the US and
the catalogue lists a USGS 3DEP 1 m lidar DEM and NAIP imagery for it, or outside the US when
``place_data_support.json`` rates its national terrain and aerial photos "fine" (the same test as
``touchdown.manifest_gen.support``); the worker then generates its manifest. The three places with a hand-written
manifest (presets.PRESETS) keep using it.

"Tested" means the place has had at least one real build. The committed list is ``app/tested_places.json``; the
desktop adds to it with ``touchdown-app mark-tested <place id>``, which writes ``<data>/tested_places.json``.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import httpx

from . import presets

APP_DIR = Path(__file__).resolve().parents[1]
TESTED_FILE = "tested_places.json"
LOWER_48 = (-125.0, 24.4, -66.9, 49.5)  # west, south, east, north (NAIP covers the conterminous US)
USER_AGENT = "Touchdown/0.1 (local app; https://github.com/jamiejk/geoblock-scenecompiler)"
NOMINATIM = os.environ.get("TOUCHDOWN_GEOCODER_URL", "https://nominatim.openstreetmap.org").rstrip("/")


def catalogue_path(repo: Path) -> Path:
    for candidate in (repo / "src" / "geoblock_core" / "data" / "places.json",
                      Path(__file__).resolve().parent / "places.json"):
        if candidate.is_file():
            return candidate
    return repo / "src" / "geoblock_core" / "data" / "places.json"


def load_catalogue(repo: Path) -> list[dict]:
    try:
        return json.loads(catalogue_path(repo).read_text())["places"]
    except (OSError, ValueError, KeyError):
        return []


SUPPORT_FILE = APP_DIR.parent / "src" / "geoblock_core" / "data" / "place_data_support.json"


def fine_outside_us() -> set[str]:
    """Non-US place ids with the "fine" data tier (national DEM of 5 m or finer, sub-metre aerial photos)."""
    try:
        rows = json.loads(SUPPORT_FILE.read_text())["places"]
    except (OSError, ValueError, KeyError):
        return set()
    return {r["id"] for r in rows if r.get("country") != "US" and r.get("tier") == "fine"}


def support(place: dict) -> tuple[bool, str]:
    """Mirror of touchdown.manifest_gen.support (the server does not import the pipeline)."""
    coverage = place.get("coverage", {})
    if not place.get("measured", True):
        return False, "Not measured yet: its terrain evidence is not in the catalogue."
    if place.get("country") != "US":
        if place.get("id") in fine_outside_us():
            return True, ""
        return False, "Coming soon: only 30 m terrain and 10 m satellite imagery here."
    if "naip" not in coverage.get("imagery", []):
        return False, "No NAIP aerial imagery here (lower 48 states only)."
    if not coverage.get("dem1m"):
        return False, "No 1 m lidar terrain listed for this spot."
    return True, ""


def art(place: dict) -> str:
    """Which generic illustration a place gets (web/img/kind-<art>.svg)."""
    kind = (place.get("kind") or "").lower()
    for words, name in ((("city", "park", "street"), "city"), (("sea", "headland", "coast", "shore", "fjord"), "coast"),
                        (("canyon", "badlands"), "canyon"), (("desert", "mesa", "slickrock", "monolith"), "desert"),
                        (("volcano", "caldera", "crater"), "volcano"),
                        (("summit", "peak", "ridge", "pass", "alpine", "mountain", "meadow"), "mountain"),
                        (("valley", "cliff", "overlook"), "valley")):
        if any(w in kind for w in words):
            return name
    return "mountain"


def tested(settings) -> dict[str, str]:
    """place id -> note, from the committed list and the desktop's own additions."""
    out: dict[str, str] = {}
    for path in (APP_DIR / TESTED_FILE, settings.data_dir / TESTED_FILE):
        try:
            out.update(json.loads(path.read_text()).get("tested", {}))
        except (OSError, ValueError, AttributeError):
            continue
    return out


def mark_tested(settings, place_id: str, note: str = "") -> Path:
    path = settings.data_dir / TESTED_FILE
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {"tested": {}}
    data.setdefault("tested", {})[place_id] = note or time.strftime("real build %Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


# places whose builds are known to fail; shown disabled with the reason
PARKED = {"golden-gate-battery-spencer": "Parked: the bridge warps in the final swing (18 Oct builds v8-v14)",
          "big-sur-bixby-bridge": "Parked: the bridge and sea layout missed in every end still (18 Oct)",
          "arches-delicate-arch": "Parked: the automatic viewpoint can't see the arch from the slickrock bowl (18 Oct, 2 tries)",
          # abroad (Jamie 18 Oct: park them): the data works, but the automatic viewpoint picks lake or mudflat
          # foregrounds and FLUX redraws another view (Matterhorn, Milford Sound abroad-1 builds)
          **{pid: "Coming soon: the automatic viewpoint isn't reliable here yet" for pid in (
              "gornergrat-matterhorn", "jungfraujoch-sphinx", "aiguille-du-midi", "mount-fuji", "milford-sound")}}


# The place the picker opens on: the flagship, the one with a hand-written manifest and a real build behind it.
FLAGSHIP = "monument-valley"

# The groups the picker shows, in order. A tested place has had a real Touchdown build; an experimental one has
# not, and a build for it may stop part-way or come out wrong; a "soon" one cannot be built at all yet.
GROUP_TESTED = "tested"
GROUP_EXPERIMENTAL = "experimental"
GROUP_SOON = "soon"


def public(settings) -> list[dict]:
    """Every catalogue place for the picker, with the group it belongs to.

    The groups are the honest ones: a "tested" place has had a real Touchdown build, an "experimental" one has
    not (a build for it may stop part-way, or come out wrong), and a "soon" one cannot be built at all yet.
    The list comes back in group order, with the flagship first inside the tested group, so the picker opens on
    the place with the most real builds behind it.
    """
    by_place = {p["place"]: p for p in presets.PRESETS}
    done = tested(settings)
    rows = []
    for place in load_catalogue(settings.repo):
        ok, why = support(place)
        if place["id"] in PARKED:
            ok, why = False, PARKED[place["id"]]
        preset = by_place.get(place["id"])
        is_tested = place["id"] in done or (preset is not None and preset["id"] in done)
        rows.append({"id": place["id"], "title": place["name"], "region": place.get("region", ""),
                     "country": place.get("country", ""), "kind": place.get("kind", ""),
                     "latitude": place["latitude"], "longitude": place["longitude"],
                     "supported": ok, "reason": why, "preset": preset["id"] if preset else None,
                     "tested": is_tested, "flagship": place["id"] == FLAGSHIP,
                     "group": GROUP_SOON if not ok else (GROUP_TESTED if is_tested else GROUP_EXPERIMENTAL),
                     "view": (place.get("view_toward") or {}).get("name", ""),
                     "thumb": f"/static/img/{preset['id']}.svg" if preset else f"/static/img/kind-{art(place)}.svg"})
    rank = {GROUP_TESTED: 0, GROUP_EXPERIMENTAL: 1, GROUP_SOON: 2}
    return sorted(rows, key=lambda r: (rank[r["group"]], not r["flagship"], r["country"] != "US", r["title"].lower()))


def by_id(settings, place_id: str) -> dict | None:
    return next((p for p in load_catalogue(settings.repo) if p["id"] == place_id), None)


# --- custom US points ----------------------------------------------------------------------------------------

def _core(repo: Path):
    """geoblock_core.lidar, from the installed package or this checkout's src/."""
    try:
        from geoblock_core import lidar
    except ImportError:
        sys.path.insert(0, str(repo / "src"))
        from geoblock_core import lidar
    return lidar


def find_projects(repo: Path, latitude: float, longitude: float, cache: Path) -> list[dict]:
    """USGS 3DEP 1 m DEM projects with a tile under the point, newest first: the catalogue builder's own check
    (geoblock_core.lidar.dem1m_projects_at, one HEAD request per candidate project). Tests replace this."""
    return _core(repo).dem1m_projects_at(latitude, longitude, cache)


def states_at(repo: Path, latitude: float, longitude: float) -> list[str]:
    return _core(repo).states_at(latitude, longitude, margin=0.0)


_checked: dict[tuple, dict] = {}
_check_lock = threading.Lock()


def check_point(settings, latitude: float, longitude: float) -> dict:
    """Is a custom point buildable? In the lower 48 (NAIP), with a 3DEP 1 m lidar project at the point. The
    10 m 1/3 arc-second fallback (terrain_route's "3dep13") exists in the pipeline but is not offered here."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return {"ok": False, "message": "That is not a valid position."}
    key = (round(latitude, 4), round(longitude, 4))
    with _check_lock:
        if key in _checked:
            return _checked[key]
    w, s, e, n = LOWER_48
    states = states_at(settings.repo, latitude, longitude) if (s <= latitude <= n and w <= longitude <= e) else []
    if not states:
        result = {"ok": False, "message": "Outside the lower 48 US states. The terrain and aerial photos the "
                                          "pipeline uses (USGS 3DEP lidar, NAIP) only cover those."}
    else:
        try:
            projects = find_projects(settings.repo, latitude, longitude, settings.data_dir / "coverage_cache")
        except Exception as e:  # network trouble: say so, don't cache it
            return {"ok": False, "message": f"Could not check the lidar coverage just now ({type(e).__name__})."}
        if projects:
            best = projects[0]
            result = {"ok": True, "project": best["project"], "zone": int(best["zone"]), "year": best.get("year"),
                      "state": states[0], "message": f"1 m lidar terrain: {best['project']}."}
        else:
            result = {"ok": False, "state": states[0],
                      "message": "No USGS 1 m lidar terrain here. The pipeline could fall back to the coarse 10 m "
                                 "elevation model, but that isn't offered in the app."}
    with _check_lock:
        _checked[key] = result
    return result


def clean_name(text: str) -> str:
    """A place name that goes into prompts: no braces, brackets, quotes or control characters."""
    text = re.sub(r"[{}\[\]<>\"\\\x00-\x1f]", "", str(text or ""))
    return " ".join(text.split())[:80].strip(" ,.;")


# --- place-name search (OpenStreetMap Nominatim, through the server) ----------------------------------------------
# Nominatim's usage policy: an identifying User-Agent, at most one request per second, cache results, and no
# search-as-you-type. The browser cannot set a User-Agent, so the server asks on its behalf, serialised to one
# request a second, cached, and only when the user presses Search or clicks the map.

_geo_lock = threading.Lock()
_geo_last = [0.0]
_geo_cache: dict[str, object] = {}


def _nominatim(path: str, params: dict) -> object:
    key = path + json.dumps(params, sort_keys=True)
    if key in _geo_cache:
        return _geo_cache[key]
    with _geo_lock:
        wait = 1.0 - (time.monotonic() - _geo_last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = httpx.get(f"{NOMINATIM}/{path}", params={**params, "format": "jsonv2"}, timeout=15,
                          headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
        finally:
            _geo_last[0] = time.monotonic()
    r.raise_for_status()
    data = r.json()
    if len(_geo_cache) > 500:
        _geo_cache.clear()
    _geo_cache[key] = data
    return data


def geocode(query: str, *, country: str | None = "us") -> list[dict]:
    query = " ".join(query.split())[:120]
    if len(query) < 3:
        return []
    params = {"q": query, "limit": 6, "addressdetails": 1}
    if country:
        params["countrycodes"] = country
    rows = _nominatim("search", params)
    return [{"name": clean_name(r.get("name") or r.get("display_name", "").split(",")[0]),
             "label": clean_name(r.get("display_name", ""))[:80], "latitude": float(r["lat"]),
             "longitude": float(r["lon"]), "state": clean_name((r.get("address") or {}).get("state", ""))}
            for r in rows if r.get("lat") and r.get("lon")]


def reverse(latitude: float, longitude: float) -> dict:
    r = _nominatim("reverse", {"lat": f"{latitude:.5f}", "lon": f"{longitude:.5f}", "zoom": 14, "addressdetails": 1})
    if not isinstance(r, dict) or r.get("error"):
        return {"name": "", "state": ""}
    address = r.get("address") or {}
    name = r.get("name") or next((address.get(k) for k in ("peak", "natural", "tourism", "park", "village", "town",
                                                            "city", "hamlet", "county") if address.get(k)), "")
    return {"name": clean_name(name), "state": clean_name(address.get("state", ""))}
