"""Check catalog coordinates against named features in Overture Maps, which is reachable on AWS where
Wikidata and Nominatim are not.

Reads only the row groups whose bounding boxes overlap a catalog place (land features, infrastructure and
places themes), then prints, for each place, the nearest features sharing a word with its name and their
distance from the curated point. A best match hundreds of metres away needs a look; the name may also differ.

    uv run --no-project --with pyarrow --with fsspec --with aiohttp --with requests \\
        python scripts/check_places_overture.py [--release 2026-09-23.1]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import unicodedata
import urllib.parse
import urllib.request

import fsspec
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src/geoblock_core/data/places.json"
BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
THEMES = ("theme=base/type=land/", "theme=base/type=infrastructure/", "theme=places/type=place/")
WINDOW_DEG = 0.03  # about 3 km around each landing and view
COMMON = set("the of a at view point summit mount mountain viewpoint overlook rim village national park area "
             "upper lower de di du la le".split())


def parquet_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}" + (
            f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        listing = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]*\.parquet)</Key>", listing)
        more = re.search(r"<NextContinuationToken>([^<]*)<", listing)
        if not more:
            return keys
        token = more.group(1)


def column_range(row_group, name: str):
    for i in range(row_group.num_columns):
        column = row_group.column(i)
        if column.path_in_schema == name and column.statistics and column.statistics.has_min_max:
            return column.statistics.min, column.statistics.max
    return None


def named_features(key: str, boxes: list[tuple]) -> list[dict]:
    """Named features in one Overture file whose bounding-box centre is inside a place's window."""
    with fsspec.filesystem("https").open(f"{BUCKET}/{key}", block_size=1 << 20) as handle:
        parquet = pq.ParquetFile(handle)
        wanted = []
        for i in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(i)
            ranges = [column_range(group, f"bbox.{edge}") for edge in ("xmin", "ymin", "xmax", "ymax")]
            if None in ranges:
                continue
            (west, _), (south, _), (_, east), (_, north) = ranges
            if any(west <= b[3] and east >= b[1] and south <= b[4] and north >= b[2] for b in boxes):
                wanted.append(i)
        if not wanted:
            return []
        columns = [c for c in ("names", "subtype", "class", "categories", "bbox", "basic_category")
                   if c in parquet.schema_arrow.names]
        table = parquet.read_row_groups(wanted, columns=columns)
    found = []
    for row in table.to_pylist():
        box = row["bbox"]
        x, y = (box["xmin"] + box["xmax"]) / 2, (box["ymin"] + box["ymax"]) / 2
        hits = [b[0] for b in boxes if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
        name = (row.get("names") or {}).get("primary")
        if hits and name:
            kind = row.get("class") or row.get("basic_category") or (row.get("categories") or {}).get("primary")
            found.append({"places": hits, "name": name, "longitude": x, "latitude": y,
                          "kind": kind or row.get("subtype")})
    return found


def words(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.replace("'", "").replace("’", ""))
    text = re.sub(r"[^a-z0-9 ]", " ", text.encode("ascii", "ignore").decode().lower())
    return {word for word in text.split() if word not in COMMON and len(word) > 2}


def metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.hypot((lon2 - lon1) * 111_320 * math.cos(math.radians(lat1)), (lat2 - lat1) * 110_574)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default="2026-09-23.1")
    args = parser.parse_args(argv)
    places = json.loads(CATALOG.read_text())["places"]
    boxes = []
    for place in places:
        points = [(place["latitude"], place["longitude"])]
        if place.get("view_toward"):
            points.append((place["view_toward"]["latitude"], place["view_toward"]["longitude"]))
        boxes += [(place["id"], lon - WINDOW_DEG, lat - WINDOW_DEG, lon + WINDOW_DEG, lat + WINDOW_DEG)
                  for lat, lon in points]
    keys = [key for theme in THEMES for key in parquet_keys(f"release/{args.release}/{theme}")]
    near = defaultdict(list)
    with ThreadPoolExecutor(12) as pool:
        for features in pool.map(lambda key: named_features(key, boxes), keys):
            for feature in features:
                for place_id in feature["places"]:
                    near[place_id].append(feature)
    for place in places:
        wanted = words(place["name"].split(",")[0])
        matches = sorted(
            ((len(wanted & words(f["name"])) / len(wanted),
              metres(place["latitude"], place["longitude"], f["latitude"], f["longitude"]), f)
             for f in near[place["id"]] if wanted & words(f["name"])),
            key=lambda match: (-match[0], match[1]))
        best = "; ".join(f"{f['name']} [{f['kind']}] {distance:.0f} m" for _, distance, f in matches[:3])
        print(f"{place['id']:<32} {best or 'no feature with a matching name'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
