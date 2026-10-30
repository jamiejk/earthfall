"""FLUX 3 Touchdown manifests generated from the place catalogue, or from a custom US point (Jamie 18 Oct: "a series
of global landmarks as a dropdown, and then letting people generate custom locations as well").

The three working manifests in ``shots/flux3/`` were written by hand: every place word in them (the scene, the
approach, the swoop's view, the end-still prompt, the orbit region, ...) and the terrain table were set place by
place. This module derives them instead, from what the catalogue already measured for each place:

- the curated name, region, kind and ``view_toward`` (what the camera looks at past the person);
- the ESA WorldCover shares near the landing and across the view (``coverage.landcover``);
- the 1 m 3DEP lidar project (``coverage.dem1m``) and the catalogue view heading (-1 = unknown);
- optionally, the ground underfoot measured on NAIP (``geoblock_core.imagery.naip_ground``), which knows paving
  and pale rock that 10 m WorldCover cannot see. Pass ``ground=`` (or ``--naip`` on the command line).

The words come from fixed templates per terrain family (desert, canyon, mountain, valley, coast, volcano, forest,
hills, city) and scene type (landmark, urban, nature). No model is called.

Everything that is not per place (the opener, cloud, dive, body, end, face and sound settings and their prompts) is
taken from ``shots/flux3/monument-valley.toml``, the newest tuned FLUX 3 manifest, so a generated manifest follows
the hand-tuned defaults as they change. The place-specific keys in it are rebuilt, not copied.

Terrain settings follow what ``touchdown.build`` already decides by itself:
- ``project`` / ``utm_zone``: the catalogue's newest 1 m project (the build's own default when they are unset);
- the landing: the catalogue landing point; the build's search moves it at most ``search_radius_m``;
- ``heading``: the catalogue view heading toward ``view_toward``, or -1 (no known view), which makes the build
  search landing/heading pairs (terrain_route.search_views) and rank them with the existing embedding ranker,
  as the Grand Canyon manifest does;
- ``[end] lock_pin`` / ``keep_after_pin`` are left to auto_switch.flux3_defaults ("48", 8), and the end still to
  the outline-map (structure) conditioning default.

    python -m touchdown.manifest_gen monument-valley --subject ../shots/subjects/me.jpg --person "a person in a hat"
    python -m touchdown.manifest_gen --lat 36.06 --lon -112.11 --name "Somewhere" --project AZ_X --zone 12 ...
    python -m touchdown.manifest_gen --list          (the catalogue places it can build, and why the rest can't)
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))  # geoblock_core, as touchdown.build does

BASE_MANIFEST = REPO / "shots" / "flux3" / "monument-valley.toml"

# WorldCover class -> a few words for the ground (lower case; no colours: light and weather come from elsewhere)
COVER_WORDS = {10: "trees", 20: "scrub", 30: "grass", 40: "fields", 50: "roads and buildings",
               60: "sand and bare rock", 70: "snow", 80: "water", 90: "marsh", 95: "mangroves", 100: "tundra"}
# imagery.naip_ground classes -> short words, and the ground a person stands on
NAIP_WORDS = {"rock": "bare rock", "trees": "trees and shrubs", "grass": "grass", "earth": "bare earth and gravel",
              "scrub": "low scrub", "paved": "paved paths and roads"}
NAIP_STANDING = {"rock": "bare rock slabs", "trees": "a level open patch among the trees", "grass": "open grass",
                 "earth": "open bare earth and gravel", "scrub": "open ground among low scrub",
                 "paved": "open paved ground"}

# One entry per terrain family. backdrop: what fills the view past the person when the catalogue names nothing;
# beside: the same, said after a named view ("Half Dome and the forested valley").
FAMILIES = {
    "desert": {"noun": "desert", "backdrop": "the tall sandstone buttes and mesas", "beside": "the desert buttes",
               "standing": "open sand and slickrock", "features": "desert floor, buttes and distant skyline",
               "materials": "Draw every butte, mesa and slope as real weathered sandstone: sheer cliff faces with "
                            "horizontal rock layers, rounded eroded edges and fans of fallen rock at their bases.",
               "ambient": "fine sand streams and swirls across the rock in the wind, the sparse dry scrub shivers, "
                          "and high white clouds drift slowly over {toward}"},
    "canyon": {"noun": "canyon country", "backdrop": "the layered canyon cliffs", "beside": "the canyon cliffs",
               "standing": "a level open patch of natural ground", "features": "natural ground, canyon cliffs and rim",
               "materials": "Draw the canyon walls as real layered rock with ledges, talus slopes and scattered "
                            "trees.",
               "ambient": "the scrub and trees sway in the wind and thin clouds cast moving shadows across the "
                          "canyon walls"},
    "badlands": {"noun": "badlands", "backdrop": "the bare eroded badlands ridges", "beside": "the badlands",
                 "standing": "open bare ground", "features": "bare eroded ridges, rills and gullies",
                 "materials": "Draw the badlands as real bare, soft mudstone hills in pale yellow, tan and grey, "
                              "cut by dense rills and gullies, with no trees or plants on them.",
                 "ambient": "fine dust drifts across the bare ridges in the wind and cloud shadows slide slowly over "
                            "{toward}"},
    "mountain": {"noun": "mountains", "backdrop": "the surrounding peaks and ridges", "beside": "the surrounding peaks",
                 "standing": "open rocky ground", "features": "rocky ground, ridges and distant peaks",
                 "materials": "Draw the mountains as real rock with natural ridges, gullies and scree, and "
                              "vegetation on the lower slopes.",
                 "ambient": "the wind streams across the rocky ground, small plants shiver, and clouds drift slowly "
                            "past {toward}"},
    "valley": {"noun": "mountains and valleys", "backdrop": "the valley and its cliffs", "beside": "the valley",
               "standing": "a level open patch of ground", "features": "ground, forest, valley walls and cliffs",
               "materials": "Draw the cliffs and slopes as real rock with cracks and streaks, with forest on the "
                            "valley floor.",
               "ambient": "the trees sway, mist drifts in the valley, and clouds move slowly over {toward}"},
    "coast": {"noun": "coast", "backdrop": "the sea and the coastline", "beside": "the water",
              "standing": "open grassy ground", "features": "headland, shoreline and water",
              "materials": "Draw the coastline as real cliffs and shore, with natural waves on the water.",
              "ambient": "the grass and scrub ripple in the wind, small whitecaps move on the water and seabirds "
                         "glide past"},
    "volcano": {"noun": "volcanic highlands", "backdrop": "the volcano's slopes", "beside": "the volcanic slopes",
                "standing": "open volcanic rock and cinder", "features": "volcanic ground, slopes and ridges",
                "materials": "Draw the volcanic slopes as real rock, cinder and old lava with natural ridges and "
                             "gullies.",
                "ambient": "the wind streams across the cinder and rock, sparse plants shiver, and clouds drift "
                           "slowly past {toward}"},
    "forest": {"noun": "forested hills", "backdrop": "the forested hills", "beside": "the forested hills",
               "standing": "a level open patch of ground", "features": "ground, forest and distant hills",
               "materials": "Draw the hills as real land covered in trees, with natural ridges and clearings.",
               "ambient": "the trees sway in the wind and cloud shadows slide slowly across the hills"},
    "hills": {"noun": "hills", "backdrop": "the rolling hills", "beside": "the hills",
              "standing": "open grassy ground", "features": "ground, hills and distant skyline",
              "materials": "Draw the hills as real land with grass, scrub, trees and natural slopes.",
              "ambient": "the grass and scrub ripple in the wind and cloud shadows slide slowly across the hills"},
    "city": {"noun": "city", "backdrop": "the city's buildings and skyline", "beside": "the city around it",
             "standing": "open paved ground", "features": "streets, buildings and skyline",
             "materials": "Draw every building as a real building with windows, facades and rooftops, and every "
                          "street with real paving.",
             "ambient": "trees sway, flags and awnings stir, people pass in the distance, and clouds move slowly "
                        "above the buildings"},
}
ORBIT = "{region}, seen from about 400 kilometres up, small, flat and hazy at this scale"
LOWER_48 = (-125.0, 24.4, -66.9, 49.5)  # west, south, east, north: NAIP covers the conterminous US


def _clean(text: str, limit: int = 120) -> str:
    """Text that goes into prompts: no braces (the build calls str.format on them), quotes or control characters."""
    return re.sub(r"\s+", " ", re.sub(r"[{}\[\]<>\"\\\x00-\x1f]", "", str(text or ""))).strip(" ,.;")[:limit]


def _shares(table: dict | None) -> dict[int, float]:
    return {int(k): float(v) for k, v in (table or {}).items()}


def _top_words(shares: dict, words: dict, floor: float = 0.1, limit: int = 2) -> str:
    known = {k: v for k, v in shares.items() if k in words}
    ranked = [k for k, v in sorted(known.items(), key=lambda kv: -kv[1]) if v >= floor][:limit]
    if not ranked and known:
        ranked = [max(known, key=known.get)]
    return " and ".join(words[k] for k in ranked)


# --- what the catalogue says about a place --------------------------------------------------------------------

def find(place_id: str) -> dict:
    from geoblock_core.places import find_place
    return find_place(place_id)


SUPPORT_FILE = Path(__file__).resolve().parents[2] / "src" / "geoblock_core" / "data" / "place_data_support.json"


def data_tier(place_id: str) -> dict:
    """The place's entry in place_data_support.json (touchdown.place_data_support): its terrain and imagery sources
    and tier ("fine" = sub-metre imagery and a DEM of 5 m or finer at the landing). {} when not listed."""
    try:
        rows = json.loads(SUPPORT_FILE.read_text())["places"]
    except (OSError, ValueError, KeyError):
        return {}
    return next((r for r in rows if r.get("id") == place_id), {})


def support(place: dict) -> tuple[bool, str]:
    """Can the pipeline build this place unattended? Returns (ok, reason when not).
    US: a catalogued 1 m lidar DEM project and NAIP imagery (the dive squares and the landing search read NAIP).
    Elsewhere: the "fine" tier of place_data_support.json, a national DEM of 5 m or finer and national sub-metre
    aerial photos (geoblock_core.global_dem, touchdown.aerial); "coarse" places (GLO-30 and 10 m Sentinel-2) look
    soft close to the ground and stay off."""
    coverage = place.get("coverage", {})
    if place.get("country") != "US":
        if data_tier(place.get("id", "")).get("tier") == "fine":
            return True, ""
        return False, "coming soon: only 30 m terrain and 10 m satellite imagery here"
    if "naip" not in coverage.get("imagery", []):
        return False, "no NAIP aerial imagery here (NAIP covers the lower 48 states only)"
    if not coverage.get("dem1m"):
        return False, "no USGS 3DEP 1 m lidar DEM listed for this spot"
    return True, ""


def family(place: dict, ground: dict | None = None) -> str:
    """The terrain family the templates use: from the curated kind first, then the measured land cover."""
    kind = (place.get("kind") or "").lower()
    near = _shares(place.get("coverage", {}).get("landcover", {}).get("near"))
    view = _shares(place.get("coverage", {}).get("landcover", {}).get("view"))
    has = lambda *words: any(w in kind for w in words)  # noqa: E731
    if has("city", "street", "square", "plaza") or near.get(50, 0) >= 0.4:
        return "city"
    if has("sea", "headland", "coast", "shore", "beach", "fjord") or view.get(80, 0) >= 0.3:
        return "coast"
    if has("badlands"):  # soft bare mudstone, not layered canyon walls with trees (Zabriskie Point 18 Oct)
        return "badlands"
    if has("canyon", "gorge"):
        return "canyon"
    if has("desert", "mesa", "butte", "slickrock", "monolith", "arch"):
        return "desert"
    if has("volcano", "caldera", "crater"):
        return "volcano"
    if has("summit", "peak", "ridge", "pass", "alpine", "mountain", "meadow"):
        return "mountain"
    if has("valley", "cliff"):
        return "valley"
    land = {**near, **{k: v for k, v in view.items() if k not in near}} or near
    if ground and ground.get("shares"):
        shares = ground["shares"]
        if shares.get("paved", 0) >= 0.4:
            return "city"
        if shares.get("rock", 0) + shares.get("earth", 0) >= 0.6:
            return "desert"
    if not land:
        return "hills"
    top = max(land, key=land.get)
    return {10: "forest", 20: "hills", 30: "hills", 40: "hills", 50: "city", 60: "desert", 70: "mountain",
            80: "coast", 100: "mountain"}.get(top, "hills")


def scene_type(place: dict, fam: str) -> str:
    """landmark / urban / nature, as touchdown.scene_type classifies it, decided here from the catalogue so the
    words and the living end's motion agree (a city park is urban; a named view with a structure height, such
    as a bridge or a skyline, is a landmark; the rest is nature)."""
    kind = (place.get("kind") or "").lower()
    if any(w in kind for w in ("city park", "street", "square", "plaza")):
        return "urban"
    if float((place.get("view_toward") or {}).get("structure_m", 0) or 0) > 0:
        return "landmark"
    return "urban" if fam == "city" else "nature"


def paved(place: dict, fam: str, ground: dict | None) -> bool:
    """Does the person stand on paving? Then the landing search must not insist on natural ground."""
    if ground and ground.get("shares"):
        return ground["shares"].get("paved", 0) >= 0.3
    return fam == "city"


def name_ref(place: dict) -> str:
    """'the Monument Valley overlook', 'Mather Point, Grand Canyon': with an article when the name ends in a
    common noun."""
    name = _clean(place.get("name") or "this place")
    head = name.split(",")[0].strip()
    last = head.split()[-1] if head.split() else ""
    return ("the " + name) if last[:1].islower() and not name.lower().startswith("the ") else name


def region_text(place: dict, fam: str) -> str:
    region = _clean(place.get("region", ""))
    region = re.sub(r"\s*/\s*", " and ", region)
    noun = FAMILIES[fam]["noun"]
    return f"the {noun} of {region}" if region else f"the {noun} around {name_ref(place)}"


# --- the manifest -----------------------------------------------------------------------------------------------

def words(place: dict, ground: dict | None = None) -> dict:
    """The per-place words of a manifest, from the templates."""
    fam = family(place, ground)
    f = FAMILIES[fam]
    coverage = place.get("coverage", {})
    near = _shares(coverage.get("landcover", {}).get("near"))
    view_name = _clean((place.get("view_toward") or {}).get("name", ""))
    here = name_ref(place)
    is_paved = paved(place, fam, ground)
    if ground and ground.get("shares"):
        top = max(ground["shares"], key=ground["shares"].get)
        standing = NAIP_STANDING.get(top, f["standing"])
        if is_paved:
            standing = "open paved ground"
        underfoot = _clean(ground.get("words", ""), 160) or f["standing"]
        across = _top_words(ground["shares"], NAIP_WORDS) or "the open ground"
    else:
        standing = "open paved ground" if is_paved else f["standing"]
        from geoblock_core.landcover import describe
        underfoot = describe(near) if near else f["standing"]
        across = _top_words(near, COVER_WORDS) or "the open ground"
    backdrop = f"{view_name} and {f['beside']}" if view_name else f["backdrop"]
    toward = view_name or f["backdrop"]
    region = region_text(place, fam)
    distant = (f"{view_name} and the other distant landforms" if view_name else
               "the buildings and distant landforms" if fam == "city" else "the distant landforms")
    scene = f"on {standing} at {here}, with {backdrop} beyond"
    return {
        "family": fam, "scene_type": scene_type(place, fam), "paved": is_paved, "here": here,
        "standing": standing, "underfoot": underfoot, "backdrop": backdrop, "toward": toward,
        "scene": scene,
        "region": region,
        "approach": f"the {f['noun']} around {here}, looking forward and down across {across} toward {toward}",
        "swoop_view": f"the {f['noun']} at {here}, about a hundred and seventy metres up, looking forward and down "
                      f"across {across}",
        "swoop_toward": f"{toward} ahead",
        "ambient": f["ambient"].format(toward=toward),
        "orbit_region": ORBIT.format(region=region),
        "end_prompt": (f"Make a real eye-level photograph from exactly the same camera viewpoint and compass direction "
                       f"at {here}. Keep the outlines and positions of {distant}, the horizon and the sky where "
                       f"image 1 has them; do not rotate or move them. {f['materials']} In the center, about ten "
                       f"metres from the camera on {standing}, add {{person}}, standing still facing the camera, "
                       f"shown full length. The near ground is {underfoot}. Natural daylight and realistic human "
                       f"scale. No text."),
        "opener_end_prompt": (f"A real photograph taken at eye level about fifteen metres away: {{person}} stands "
                              f"{{scene}}, seen full length, small in the middle of the frame, facing the camera, on "
                              f"{standing} that runs straight from the bottom of the frame to their feet. The wide "
                              f"view of {backdrop} fills most of the frame. Natural daylight, a few soft clouds, 35mm "
                              f"lens, sharp, real photo, no text."),
        "swoop_prompt": (f"One continuous forward descent following the exact straight camera path of the guide "
                         f"video. The {f['features']} remain in their measured positions. The camera levels out "
                         f"toward {{person}} standing {{scene}} and eases into an eye-level view. Natural daylight, "
                         f"realistic depth and parallax, photorealistic."),
        "view_query": f"A level open patch of {standing.removeprefix('a level open patch of ').removeprefix('open ')} at {here}, with "
                      f"{backdrop} beyond.",
        "route_query": f"Four consecutive views of a straight forward camera descent over {across} toward an open "
                       f"viewpoint with {backdrop} beyond.",
        "water": _shares(coverage.get("landcover", {}).get("view")).get(80, 0) >= 0.05 or fam == "coast",
    }


def terrain_table(place: dict, w: dict) -> dict:
    """[terrain]: the lidar project (US) and the automatic landing/view search the build already runs."""
    from geoblock_core.places import view_heading
    projects = place.get("coverage", {}).get("dem1m") or []
    if place.get("country", "US") != "US":  # the build picks the national DEM and WGS 84 / UTM zone (global_dem.plan)
        table = {"enabled": True}
    elif not projects:
        raise ValueError(f"{place.get('id')}: no 1 m lidar DEM project; the terrain route needs one")
    else:
        table = {"enabled": True, "project": projects[0]["project"], "utm_zone": int(projects[0]["zone"])}
    heading = view_heading(place)
    if heading < 0:  # no known view direction: search landing/heading pairs on the DEM and NAIP, rank them
        table.update({"heading": -1.0, "embedding_rank": True, "view_previews": 8, "search_radius_m": 260})
        if not w["paved"]:  # what the ranker compares the previews with (the build has a default from the scene)
            table.update({"view_query": w["view_query"], "route_query": w["route_query"]})
    else:  # the catalogue's bearing toward view_toward; the landing is searched nearby, the view kept clear
        table.update({"heading": float(heading), "search_radius_m": 150})
    if not w["paved"]:
        table.update({"avoid_pavement": True, "surface": "natural"})
    return table


def base_manifest(path: Path = BASE_MANIFEST) -> dict:
    return tomllib.loads(path.read_text())


def place_from_point(latitude: float, longitude: float, *, name: str = "", project: str, zone: int,
                     region: str = "", landcover: dict | None = None, year: int | None = None) -> dict:
    """A catalogue-shaped entry for a custom US point (not in the catalogue). ``project``/``zone`` come from the
    1 m DEM coverage check (geoblock_core.lidar.dem1m_projects_at)."""
    latitude, longitude = float(latitude), float(longitude)
    if not (LOWER_48[1] <= latitude <= LOWER_48[3] and LOWER_48[0] <= longitude <= LOWER_48[2]):
        raise ValueError("custom places must be in the conterminous US (NAIP and 3DEP 1 m lidar)")
    point_id = (f"custom-{abs(latitude):.5f}{'n' if latitude >= 0 else 's'}-"
                f"{abs(longitude):.5f}{'e' if longitude >= 0 else 'w'}")
    dem = {"project": _clean(project, 80), "zone": int(zone), **({"year": int(year)} if year else {})}
    coverage = {"terrain": "usgs3dep", "imagery": ["naip"], "dem1m": [dem]}
    if landcover:
        coverage["landcover"] = landcover
    return {"id": point_id, "name": _clean(name, 80) or "Custom place", "region": _clean(region, 60),
            "country": "US", "kind": "", "latitude": latitude, "longitude": longitude, "coverage": coverage}


def generate(place: str | dict | tuple, subject: str, person: str, *, name: str | None = None,
             ground: dict | None = None, base: dict | None = None) -> dict:
    """A FLUX 3 Touchdown manifest for ``place``: a catalogue id, a catalogue-shaped dict (``place_from_point``),
    or ``(name, latitude, longitude, project, zone)``. ``subject`` is the person's photo (a path), ``person`` the
    words for them. ``ground``: an imagery.naip_ground result for the landing, if measured."""
    if isinstance(place, str):
        place = find(place)
    elif isinstance(place, tuple):
        label, latitude, longitude, project, zone = place
        place = place_from_point(latitude, longitude, name=label, project=project, zone=zone)
    custom = place["id"].startswith("custom-")
    if not custom:
        ok, why = support(place)
        if not ok:
            raise ValueError(f"{place['id']}: {why}")
    w = words(place, ground)
    m = copy.deepcopy(base or base_manifest())
    m["output"] = m.get("output", "outputs/touchdown")
    m["shot"] = {"name": name or f"auto-{place['id']}", "place": place["id"], "subject": str(subject),
                 "person": _clean(person, 300) or "a person in everyday outdoor clothes",
                 "scene": w["scene"], "region": w["region"], "approach": w["approach"],
                 "swoop_view": w["swoop_view"], "swoop_toward": w["swoop_toward"], "ambient": w["ambient"],
                 "descent": "45"}
    m["terrain"] = terrain_table(place, w)
    stills = {k: m["stills"][k] for k in ("flux3", "flux3_url", "flux3_free_url", "seed") if k in m["stills"]}
    stills.update({"orbit_region": w["orbit_region"], "end_prompt": w["end_prompt"],
                   "opener_end_prompt": w["opener_end_prompt"], "scene_type": w["scene_type"]})
    # a real wall or kerb the measured pavement smears is not a reason to stop (Yosemite); in badlands the pale grey
    # mudstone reads as "neutral pavement" and FLUX's pale rock as its replacement (Zabriskie, 3 tries 18 Oct); the
    # skyline and structure checks still stop the build
    if w["paved"] or w["family"] == "badlands":
        stills["layout_gate"] = "landmarks"
    m["stills"] = stills
    m["swoop"] = {**m["swoop"], "prompt": w["swoop_prompt"]}
    m["dive"] = {**m["dive"], "sea": bool(w["water"])}
    for key in ("lock_pin", "keep_after_pin"):  # auto_switch.flux3_defaults fills "48" and 8
        m["end"].pop(key, None)
    m["generated"] = {"by": "touchdown.manifest_gen", "family": w["family"], "base": BASE_MANIFEST.name,
                      "naip_ground": bool(ground)}
    if custom:
        m["place"] = place  # touchdown.build registers it, so find_place knows the id
    return m


def naip_words(place: dict, cache: Path) -> dict | None:
    """imagery.naip_ground at the landing (network; cached by point), or None when it can't be measured."""
    if place.get("country", "US") != "US":  # NAIP is US only: the catalogue's WorldCover words
        return None
    try:
        from geoblock_core.imagery import naip_ground
        from geoblock_core.places import landing
        latitude, longitude = landing(place)
        return naip_ground(latitude, longitude, cache)
    except Exception as error:  # no network, no NAIP here, a missing optional dependency: use the catalogue words
        print(f"manifest_gen: NAIP ground words skipped: {error}", file=sys.stderr)
        return None


def _toml(m: dict) -> str:
    """A small TOML writer for manifests (scalars, lists of scalars, tables, arrays of tables), used when tomli_w is
    not installed: the generator runs in the pipeline's Python, which the app's own extra doesn't reach (first places
    batch, 18 Oct: every generated job failed on 'No module named tomli_w')."""
    import json as _json

    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return _json.dumps(v, ensure_ascii=False)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(value(x) for x in v) + "]"
        raise TypeError(f"cannot write {type(v).__name__} to TOML")

    def key(k):
        return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _json.dumps(k)

    out = []

    def table(d, path):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict) and not (
            isinstance(v, list) and v and all(isinstance(x, dict) for x in v))}
        if path and scalars:
            out.append(f"\n[{'.'.join(map(key, path))}]")
        elif path and not any(isinstance(v, dict) for v in d.values()):
            out.append(f"\n[{'.'.join(map(key, path))}]")
        for k, v in scalars.items():
            out.append(f"{key(k)} = {value(v)}")
        for k, v in d.items():
            if isinstance(v, dict):
                table(v, path + [k])
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                for item in v:
                    out.append(f"\n[[{'.'.join(map(key, path + [k]))}]]")
                    for kk, vv in item.items():
                        out.append(f"{key(kk)} = {value(vv)}")
    table(m, [])
    return "\n".join(out).lstrip("\n") + "\n"


def dumps(m: dict, note: str = "") -> str:
    head = "# Generated by touchdown.manifest_gen from the place catalogue" + (f" ({note})" if note else "") + \
           "; edit the catalogue or the templates, not this file.\n"
    try:
        import tomli_w
    except ImportError:
        return head + _toml(m)
    return head + tomli_w.dumps(m)


def catalogue_status() -> list[dict]:
    from geoblock_core.places import places
    return [{"id": p["id"], "name": p["name"], "country": p.get("country", ""), "ok": support(p)[0],
             "why": support(p)[1]} for p in places()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("place", nargs="?", help="catalogue id (or a unique name match)")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--name", default="", help="custom point: its name (default 'Custom place')")
    p.add_argument("--project", help="custom point: the 3DEP 1 m project (default: probe with geoblock_core.lidar)")
    p.add_argument("--zone", type=int, help="custom point: the project's UTM zone")
    p.add_argument("--region", default="", help="custom point: its state or region, for the words")
    p.add_argument("--subject", default="../subjects/test-subject-red-coat.jpg")
    p.add_argument("--person", default="a person in everyday outdoor clothes")
    p.add_argument("--shot-name", help="[shot] name (default auto-<place id>)")
    p.add_argument("--naip", action="store_true", help="measure the ground underfoot on NAIP (network)")
    p.add_argument("--cache", type=Path, default=Path("outputs/touchdown"), help="cache for --naip / --project probes")
    p.add_argument("-o", "--output", type=Path, help="write here (default: print)")
    p.add_argument("--list", action="store_true", help="list catalogue places and whether they can be generated")
    a = p.parse_args(argv)
    if a.list:
        for row in catalogue_status():
            print(f"{'ok ' if row['ok'] else '-- '} {row['id']:<34} {row['name']}" + (f"  ({row['why']})" if row['why'] else ""))
        return 0
    if a.place:
        place = find(a.place)
    elif a.lat is not None and a.lon is not None:
        project, zone = a.project, a.zone
        if not project:
            from geoblock_core.lidar import dem1m_projects_at
            found = dem1m_projects_at(a.lat, a.lon, a.cache)
            if not found:
                p.error("no USGS 3DEP 1 m lidar DEM project covers this point")
            project, zone = found[0]["project"], found[0]["zone"]
        place = place_from_point(a.lat, a.lon, name=a.name, project=project, region=a.region,
                                 zone=zone or int((a.lon + 180) // 6) + 1)
    else:
        p.error("give a catalogue place id, or --lat and --lon")
    ground = naip_words(place, a.cache) if a.naip else None
    text = dumps(generate(place, a.subject, a.person, name=a.shot_name, ground=ground), place["id"])
    if a.output:
        a.output.write_text(text)
        print(f"wrote {a.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
