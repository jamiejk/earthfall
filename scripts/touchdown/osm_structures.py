"""OpenStreetMap structures with mapped heights, for the Blender map ([terrain] osm_structures, 18 Oct).

A bare-earth DEM with a flat aerial photograph has no bridges or towers: at Battery Spencer the photo end still
showed the Golden Gate's north tower filling a third of the frame, the map had nothing there, and the swoop grew
the tower out of nothing (a semi-transparent fade-in at f142, v10). OSM maps many landmarks as 3D parts: the
Golden Gate's towers are stacked ``building:part`` pieces with ``min_height``/``height`` up to 225 m. Everything
here comes from tags; nothing is shaped by hand:

- ``building`` / ``building:part`` with ``height`` (or ``building:levels`` x 3 m) and optional ``min_height``,
  heights above the ground under the outline;
- ``man_made=bridge`` outlines (ways, and the outer ways of multipolygon relations) as a 3 m deck slab. Its level is
  the ``min_height`` of the non-tower parts mapped inside the outline (the deck pieces), else the ground at the
  bridge's two ends.

Parts inside a bridge outline take the bridge's ``colour`` tag; others take the mean colour of the aerial photograph
under them. Output: JSON rings in the map's local metres (relative to the DEM origin), counter-clockwise.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OVERPASS = "https://overpass-api.de/api/interpreter"
AGENT = "geoblock-touchdown/1.0 (OSM structures for a terrain fly-in)"


def _query(lat: float, lon: float, radius_m: float) -> dict:
    q = (f"[out:json][timeout:120];("
         f'way(around:{radius_m:.0f},{lat},{lon})["building"]["height"];'
         f'way(around:{radius_m:.0f},{lat},{lon})["building:part"]["height"];'
         f'way(around:{radius_m:.0f},{lat},{lon})["building"]["building:levels"];'
         f'way(around:{radius_m:.0f},{lat},{lon})["man_made"="bridge"];'
         f'relation(around:{radius_m:.0f},{lat},{lon})["man_made"="bridge"];'
         f");out body geom;")
    data = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)


def _number(value) -> float | None:
    try:
        return float(str(value).split()[0].replace("m", ""))
    except (TypeError, ValueError, IndexError):
        return None


def _colour(tag: str | None):
    if tag and tag.startswith("#") and len(tag) == 7:
        return tuple(int(tag[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return None


def _inside(point, ring) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def prepare(lat: float, lon: float, zone: int, dem: Path, imagery: Path, out: Path, radius_m: float = 2500) -> dict:
    from pyproj import Transformer
    import cv2
    meta = json.loads((dem / "metadata.json").read_text())
    near, far = np.load(dem / "height.npy"), np.load(dem / "height_far.npy")
    from geoblock_core.utm import projected_crs
    to_utm = Transformer.from_crs("EPSG:4326", projected_crs(zone), always_xy=True)

    def local(geometry):
        pts = np.array([to_utm.transform(p["lon"], p["lat"]) for p in geometry], float)
        pts -= (meta["x"], meta["y"])
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
        return pts if area > 0 else pts[::-1]

    def ground(x, y):
        if abs(x) <= 396 and abs(y) <= 396:
            return float(near[round(200 - y / 2), round(200 + x / 2)])
        return float(far[int(np.clip(round(340 - y / 50), 0, 680)), int(np.clip(round(340 + x / 50), 0, 680))])

    squares = {w: cv2.imread(str(imagery / f"naip_{w:05d}.png")) for w in (900, 2700, 8100)}

    def aerial_tint(ring):
        cx, cy = ring.mean(axis=0)
        for w, img in squares.items():
            if img is None or max(abs(cx), abs(cy)) > w * 0.46:
                continue
            n = img.shape[0]
            px = np.clip(np.round(n / 2 + ring[:, 0] / w * n).astype(int), 0, n - 1)
            py = np.clip(np.round(n / 2 - ring[:, 1] / w * n).astype(int), 0, n - 1)
            x0, x1, y0, y1 = px.min(), px.max() + 1, py.min(), py.max() + 1
            b, g, r = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0) / 255
            return (float(r), float(g), float(b))
        return (0.5, 0.5, 0.5)

    raw = _query(lat, lon, radius_m)
    bridges, parts = [], []
    for e in raw.get("elements", []):
        tags = e.get("tags", {})
        rings = []
        if e["type"] == "way" and e.get("geometry") and e["geometry"][0] == e["geometry"][-1]:
            rings = [e["geometry"]]
        elif e["type"] == "relation":
            rings = [m["geometry"] for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")
                     and m["geometry"][0] == m["geometry"][-1]]
        for geometry in rings:
            ring = local(geometry)
            if len(ring) < 3:
                continue
            if tags.get("man_made") == "bridge" and not tags.get("height"):
                bridges.append({"ring": ring, "tags": tags, "id": f"{e['type']}/{e['id']}"})
                continue
            height = _number(tags.get("height"))
            if height is None and tags.get("building:levels"):
                height = 3.0 * (_number(tags.get("building:levels")) or 0)
            if not height or height <= 0.5:
                continue
            parts.append({"ring": ring, "min": _number(tags.get("min_height")) or 0.0, "max": height, "tags": tags,
                          "id": f"{e['type']}/{e['id']}"})
    structures = []
    for b in bridges:
        inside = [p for p in parts if _inside(p["ring"].mean(axis=0), b["ring"])]
        decks = [p["min"] for p in inside if p["tags"].get("man_made") != "tower" and p["min"] > 2]
        colour = _colour(b["tags"].get("colour"))
        for p in inside:
            p["bridge"] = b["id"]
            p["colour"] = p.get("colour") or colour
        if decks:
            level, source = float(np.median(decks)), "deck parts inside the outline"
        else:  # the ground at the two farthest-apart vertices: a bridge joins its ends
            d = np.linalg.norm(b["ring"][:, None] - b["ring"][None], axis=2)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            level, source = max(ground(*b["ring"][i]), ground(*b["ring"][j])), "ground at the bridge ends"
            level -= min(ground(*v) for v in b["ring"])  # as a height above the lowest ground under it
        structures.append({"id": b["id"], "kind": "deck", "ring": b["ring"].round(2).tolist(), "min": level - 3.0,
                           "max": level, "rgb": colour or aerial_tint(b["ring"]), "deck_level_source": source,
                           "name": b["tags"].get("name")})
    for p in parts:
        structures.append({"id": p["id"], "kind": "tower" if p["tags"].get("man_made") == "tower" else "part",
                           "ring": p["ring"].round(2).tolist(), "min": p["min"], "max": p["max"],
                           "rgb": p.get("colour") or aerial_tint(p["ring"]), "bridge": p.get("bridge")})
    result = {"source": "OpenStreetMap contributors (ODbL), via the Overpass API", "radius_m": radius_m,
              "count": len(structures), "bridges": len(bridges),
              "tallest_m": max((s["max"] for s in structures), default=None), "structures": structures}
    out.write_text(json.dumps(result) + "\n")
    return {k: v for k, v in result.items() if k != "structures"}
