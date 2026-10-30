"""Probe the non-US terrain and imagery sources: a small DEM patch and an aerial crop per catalogue place.

  cd scripts && python -m touchdown.global_data_check --place gornergrat-matterhorn
  cd scripts && python -m touchdown.global_data_check --all              # every non-US catalogue place
  cd scripts && python -m touchdown.global_data_check --endpoints        # one request per server: reachable?

Per place it fetches the landing patch (800 m) from the national DEM that geoblock_core.global_dem picks for the
country (GLO-30 where there is none, or when that source fails), and the 900 m and 300 m aerial squares from the
touchdown.aerial chain (national orthophoto, Sentinel-2 L2A fill). It writes ``<out>/<place>.jpg`` (hillshade,
900 m imagery, 300 m imagery at full resolution, numbers underneath), ``<out>/summary.json`` and
``<out>/all_sheets.jpg``. Network and CPU only, a minute or so per place; caches go under ``<out>/cache``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geoblock_core import global_dem  # noqa: E402
from touchdown import aerial  # noqa: E402

PANEL = 512

# One cheap request per server, at a point it covers: (label, method, url).
ENDPOINTS = [
    ("swissalti3d STAC (CH DEM)", "GET", f"{global_dem.SwissALTI3DProvider.stac}?bbox=7.78,45.98,7.79,45.99&limit=1"),
    ("IGN wms-r (FR DEM)", "GET", global_dem.IGNRGEAltiProvider().url((1005000, 6535000, 1005100, 6535100), 100, 100)),
    ("IGN LiDAR HD (FR DEM)", "GET", global_dem.IGNLidarHDProvider().url((1001300, 6538400, 1001400, 6538500), 200,
                                                                          200)),
    ("IDEE MDT05 WCS (ES DEM)", "GET", global_dem.SpainMDTProvider().url((903100, 4615800, 903300, 4616000), 40, 40)),
    ("Tirol WCS M28 (AT DEM)", "GET", global_dem.TirolDTMProvider().zone_provider(11.38).url(
        (79100, 241700, 79200, 241800), 200, 200)),
    ("Tirol WCS M31 (AT DEM)", "GET", global_dem.TirolDTMProvider().zone_provider(12.69).url(
        (-48550, 215150, -48450, 215250), 200, 200)),
    ("Kartverket WCS (NO DEM)", "GET", global_dem.KartverketDTMProvider().url((-30000, 6580000, -29900, 6580100), 100,
                                                                              100)),
    ("LINZ nz-elevation (NZ DEM)", "HEAD", f"{global_dem.LINZDEMProvider.bucket}/{global_dem.LINZDEMProvider.lidar}/CB08.tiff"),
    ("GSI dem5a (JP DEM)", "GET", global_dem.GSIDEMProvider().url("dem5a_png", 15, 29011, 12938)),
    ("Copernicus GLO-30 bucket", "HEAD", "https://copernicus-dem-30m.s3.amazonaws.com/tileList.txt"),
    ("Sentinel-2 COG bucket (fill)", "GET", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=1"
                                            "&prefix=sentinel-s2-l2a-cogs/32/T/LS/2023/7/"),
    ("Earth Search STAC (S2 search)", "GET", "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"),
    ("LINZ nz-imagery (NZ photos)", "HEAD", "https://nz-imagery.s3.ap-southeast-2.amazonaws.com/catalog.json"),
]


def endpoints() -> list[dict]:
    """Every DEM endpoint above, then one tile from each touchdown.aerial tile source."""
    import httpx
    rows = list(ENDPOINTS)
    for s in aerial.SOURCES.values():
        if s.template and s.available():
            z = min(s.max_zoom, 15)
            w, so, e, n = s.areas[0]
            x, y = aerial._merc((w + e) / 2, (so + n) / 2, z)
            rows.append((f"{s.id} tiles (imagery)", "GET", aerial.tile_url(s, z, int(x // 256), int(y // 256))))
    out = []
    with httpx.Client(headers={"User-Agent": global_dem.USER_AGENT}, timeout=20, follow_redirects=True) as client:
        for label, method, url in rows:
            t = time.time()
            try:
                r = client.request(method, url)
                status = f"HTTP {r.status_code} {r.headers.get('content-type', '')}"
                ok = r.status_code < 400
            except Exception as exc:  # noqa: BLE001
                status, ok = f"{type(exc).__name__}: {str(exc)[:120]}", False
            out.append({"endpoint": label, "reachable": ok, "status": status, "seconds": round(time.time() - t, 1),
                        "url": url})
            print(f"{'OK  ' if ok else 'FAIL'}  {label:34} {status}", flush=True)
    return out


def hillshade(height: np.ndarray, step: float, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    gy, gx = np.gradient(height, step)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(azimuth), math.radians(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return (np.clip(shade, 0, 1) * 255).astype(np.uint8)


def sample(path: Path, crs: str, x: float, y: float, radius: float, step: float) -> np.ndarray:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    n = int(2 * radius / step) + 1
    out = np.full((n, n), np.nan, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=from_origin(x - radius, y + radius, step, step),
                  dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def check(place: dict, out: Path, cache: Path) -> dict:
    import nadir_dive
    from geoblock_core.places import landing
    lat, lon = landing(place)
    country = place.get("country", "")
    p = global_dem.plan(lat, lon, country)
    row: dict = {"place": place["id"], "country": country, "lat": lat, "lon": lon, "crs": p.projected_crs}
    spec = global_dem.spec_for(lat, lon, p.zone, global_dem.NEAR_RADIUS_M)

    t = time.time()
    dem_error = None
    try:
        near, record = global_dem.provider(p.near).acquire(spec, cache)
        dem_source = p.near
    except Exception as exc:  # noqa: BLE001
        dem_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"  DEM {p.near} failed: {dem_error}", flush=True)
        near, record = global_dem.GLO30Provider().acquire(spec, cache)
        dem_source = "glo30"
    height = sample(near, p.projected_crs, spec.origin_easting_m, spec.origin_northing_m, 400, 2)
    row["dem"] = {"source": dem_source, "planned": p.near, "error": dem_error, "dataset": record.dataset,
                  "nominal_resolution_m": record.nominal_resolution_m, "native_pixel_m": record.native_pixel_spacing,
                  "vertical_crs": record.vertical_crs, "coverage": round(float(np.isfinite(height).mean()), 4),
                  "landing_height_m": None if not np.isfinite(height[200, 200]) else round(float(height[200, 200]), 1),
                  "relief_m": round(float(np.nanmax(height) - np.nanmin(height)), 1), "seconds": round(time.time() - t, 1)}

    source = aerial.pick(country, lat, lon)
    imagery = {}
    crops = {}
    for width, pixels in ((900, 1024), (300, 1024)):
        t = time.time()
        try:
            rgba, cover = aerial.square_rgba(source, nadir_dive._square(lat, lon, width), pixels)
            crops[width] = Image.fromarray(rgba[..., :3]).resize((PANEL, PANEL), Image.LANCZOS)
            imagery[str(width)] = {"cover": cover, "metres_per_px": round(width / pixels, 3),
                                   "seconds": round(time.time() - t, 1)}
        except Exception as exc:  # noqa: BLE001
            imagery[str(width)] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(f"  imagery {width} m failed: {imagery[str(width)]['error']}", flush=True)
    finest = [k for c in imagery.values() for k, v in c.get("cover", {}).items() if v > 0.5]
    row["imagery"] = {"source": source, "native_m": aerial.SOURCES[source].metres, "squares": imagery,
                      "main_source_300m": max(imagery.get("300", {}).get("cover", {"none": 1}).items(),
                                              key=lambda kv: kv[1])[0] if "cover" in imagery.get("300", {}) else None}
    row["imagery"]["sources_used"] = sorted(set(finest))

    shade = Image.fromarray(hillshade(np.nan_to_num(height, nan=float(np.nanmin(height))), 2.0)).convert("RGB")
    panels = [shade.resize((PANEL, PANEL)), crops.get(900), crops.get(300)]
    sheet = Image.new("RGB", (3 * PANEL, PANEL + 120), "white")
    for i, im in enumerate(panels):
        if im is not None:
            sheet.paste(im, (i * PANEL, 0))
    d, dem = ImageDraw.Draw(sheet), row["dem"]
    lines = [f"{place['id']} ({country})  {lat:.5f}, {lon:.5f}  {p.projected_crs}",
             f"DEM 800 m: {dem['dataset']}  ~{dem['nominal_resolution_m']} m  cover {dem['coverage']:.0%}  "
             f"relief {dem['relief_m']} m  heights {dem['vertical_crs']}" + (f"  [fallback: {dem_error[:60]}]" if dem_error else ""),
             f"imagery: {source} ({aerial.SOURCES[source].metres} m native)  900 m cover "
             f"{imagery.get('900', {}).get('cover')}  300 m cover {imagery.get('300', {}).get('cover')}",
             "panels: hillshade 800 m | imagery 900 m | imagery 300 m (0.29 m/px)"]
    for i, text in enumerate(lines):
        d.text((8, PANEL + 8 + 26 * i), text, fill="black")
    sheet.save(out / f"{place['id']}.jpg", quality=88)
    return row


def main() -> int:
    from geoblock_core.places import find_place, places as load_places
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", action="append", default=[], help="catalogue id (repeatable)")
    ap.add_argument("--all", action="store_true", help="every non-US catalogue place")
    ap.add_argument("--endpoints", action="store_true", help="only try one request per server")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "global-data-check")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.endpoints:
        rows = endpoints()
        (a.out / "endpoints.json").write_text(json.dumps(rows, indent=2) + "\n")
        return 0 if all(r["reachable"] for r in rows) else 1
    places = [find_place(p) for p in a.place]
    if a.all:
        places += [p for p in load_places() if p.get("country") != "US" and p not in places]
    if not places:
        ap.error("give --place <id>, --all or --endpoints")
    summary = []
    for place in places:
        print(f"== {place['id']}", flush=True)
        try:
            summary.append(check(place, a.out, a.out / "cache"))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            summary.append({"place": place["id"], "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    sheets = [Image.open(a.out / f"{s['place']}.jpg") for s in summary if "error" not in s]
    if sheets:
        big = Image.new("RGB", (sheets[0].width, sum(s.height for s in sheets)), "white")
        y = 0
        for s in sheets:
            big.paste(s, (0, y))
            y += s.height
        big.save(a.out / "all_sheets.jpg", quality=85)
    return 0 if all("error" not in s and not s["dem"]["error"] for s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
