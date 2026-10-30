"""Check the touchdown.aerial sources from the desktop: a one-tile probe per source, then the 45-degree dive per place.

  cd scripts && python -m touchdown.aerial_check --probe                # does every server answer?
  cd scripts && python -m touchdown.aerial_check                        # dives for the default test places
  cd scripts && python -m touchdown.aerial_check gornergrat-matterhorn "kinderdijk:51.8850,4.6390:NL"

Each dive uses the Yosemite 45-v2 [dive.oblique] settings and is written to outputs/aerial-check/<name>/ with a
contact sheet (four frames across the dive, the last one the hand-off to the swoop) and imagery.json (which source
covered how much of each square). No GPU, no ComfyUI: network and CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from touchdown import aerial  # noqa: E402
from touchdown import dive as divemod  # noqa: E402

DEFAULT = ["aiguille-du-midi", "ronda:36.7406,-5.1658:ES", "gornergrat-matterhorn", "kinderdijk:51.8850,4.6390:NL",
           "mount-fuji", "milford-sound", "tre-cime-di-lavaredo"]
OBLIQUE = dict(from_height_m=7000, to_height_m=175, frames=64, start_pitch_deg=70, pitch_deg=45, turn_frames=28,
               clear=16, aim=15)
PROBE = (46.0, 7.75)  # fallback probe point when a source's first area centre is sea


def probe() -> int:
    import httpx
    bad = 0
    with httpx.Client(headers={"User-Agent": aerial.USER_AGENT}, timeout=30, follow_redirects=True) as client:
        for s in aerial.SOURCES.values():
            if s.fetch:
                lat, lon = 37.7156, -119.6772
                try:
                    t = time.time()
                    rgba = s.fetch(divemod_square(lat, lon, 300), 64)
                    print(f"OK    {s.id:14} {time.time() - t:5.1f} s  cover {float((rgba[..., 3] > 127).mean()):.0%}")
                except Exception as exc:  # noqa: BLE001
                    bad += 1
                    print(f"FAIL  {s.id:14} {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if not s.available():
                print(f"SKIP  {s.id:14} no {s.key_env} set")
                continue
            lat, lon = {"CH": (46.0, 7.75), "FR": (45.88, 6.89), "ES": (36.74, -5.17), "NL": (51.885, 4.639),
                        "JP": (35.36, 138.73), "NZ": (-44.67, 167.93)}.get(s.country, PROBE)
            z = min(s.max_zoom, 17)
            x, y = aerial._merc(lon, lat, z)
            url = aerial.tile_url(s, z, int(x // 256), int(y // 256), __import__("os").environ.get(s.key_env, ""))
            t = time.time()
            try:
                r = client.get(url)
                info = f"HTTP {r.status_code} {r.headers.get('content-type', '')} {len(r.content)} B"
                ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image/")
                if ok:
                    im = np.asarray(Image.open(__import__("io").BytesIO(r.content)).convert("RGB"))
                    info += f", mean {im.mean(axis=(0, 1)).round().astype(int).tolist()}, std {im.std():.1f}"
            except Exception as exc:  # noqa: BLE001
                ok, info = False, f"{type(exc).__name__}: {str(exc)[:160]}"
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'}  {s.id:14} {time.time() - t:5.1f} s  z{z}  {info}")
    return bad


def divemod_square(lat, lon, w):
    import nadir_dive
    return nadir_dive._square(lat, lon, w)


def place_args(spec: str) -> tuple[str, float, float, str, float]:
    """``catalog-id`` or ``name:lat,lon:CC`` -> (name, lat, lon, country, heading)."""
    if ":" in spec:
        name, ll, cc = spec.split(":")
        lat, lon = map(float, ll.split(","))
        return name, lat, lon, cc, 0.0
    from geoblock_core.places import find_place, landing, view_heading
    p = find_place(spec)
    lat, lon = landing(p)
    return p["id"], lat, lon, p.get("country", ""), max(view_heading(p), 0.0)


def sheet(clip: Path, out: Path, title: str, picks=(16, 32, 48, 63)) -> None:
    import cv2
    cap, frames, i = cv2.VideoCapture(str(clip)), {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in picks:
            frames[i] = Image.fromarray(f[..., ::-1]).resize((640, 360))
        i += 1
    im = Image.new("RGB", (640 * len(picks), 390), "white")
    for k, p in enumerate(picks):
        if p in frames:
            im.paste(frames[p], (640 * k, 30))
    ImageDraw.Draw(im).text((8, 8), title, fill="black")
    im.save(out, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("places", nargs="*", default=DEFAULT)
    ap.add_argument("--probe", action="store_true", help="only fetch one tile from every source")
    ap.add_argument("--source", default="auto", help="force a touchdown.aerial source id")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "aerial-check")
    a = ap.parse_args()
    if a.probe:
        return 1 if probe() else 0
    summary = []
    for spec in a.places:
        name, lat, lon, cc, heading = place_args(spec)
        source = a.source if a.source != "auto" else aerial.pick(cc, lat, lon)
        folder = a.out / name
        print(f"== {name} ({cc}) {lat:.4f},{lon:.4f} via {source}", flush=True)
        t = time.time()
        try:
            paths = divemod.fetch(lat, lon, folder / "imagery", divemod.FAR_LEVELS, source=source)
            fetched = time.time() - t
            facts = divemod.render_oblique(paths, folder / "dive.mp4", heading=heading, **OBLIQUE)
            cover = json.loads((folder / "imagery" / "imagery.json").read_text())["cover"]
            sheet(folder / "dive.mp4", folder / "sheet.jpg",
                  f"{name} via {source}: frames 16 / 32 / 48 / 63 (hand-off). cover 300 m square: {cover.get('300')}")
            summary.append({"place": name, "source": source, "fetch_s": round(fetched), "total_s": round(time.time() - t),
                            "cover": cover, **facts})
        except Exception as exc:  # noqa: BLE001
            summary.append({"place": name, "source": source, "error": f"{type(exc).__name__}: {exc}"[:300]})
        print(json.dumps(summary[-1]), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if summary:  # all sheets stacked, for one look
        ims = [Image.open(a.out / s["place"] / "sheet.jpg") for s in summary if "error" not in s]
        if ims:
            big = Image.new("RGB", (ims[0].width, sum(i.height for i in ims)), "white")
            y = 0
            for i in ims:
                big.paste(i, (0, y))
                y += i.height
            big.save(a.out / "all_sheets.jpg", quality=85)
    return 0 if all("error" not in s for s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
