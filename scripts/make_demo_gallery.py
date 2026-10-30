"""Build the app's Examples gallery: a short clip per place, with a title and a poster.

The gallery is the first thing a judge sees, so it holds finished Touchdown clips and nothing else: the
clay *blocking* renders and the old experiment clips that were in docs/review/place-catalog are not examples of
the product.

    python scripts/make_demo_gallery.py --out app/demo-clips --clip "Monument Valley=path.mp4" ...

Each clip is copied at 1280x720 (the gallery is a preview panel; the full-resolution clips are what the demo
video is cut from) with a sidecar .json holding its title and a .jpg poster taken from a frame that shows the
place rather than the cloud.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

POSTER_AT = 0.55  # a third of the way in: past the orbit, over the place


def clip(src: Path, out: Path, title: str, width: int = 1280) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", f"scale={width}:-2",
                    "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{duration * POSTER_AT:.2f}", "-i", str(out),
                    "-frames:v", "1", "-vf", "scale=960:-2", str(out.with_suffix(".jpg"))], check=True)
    out.with_suffix(".json").write_text(json.dumps({"title": title}, indent=2) + "\n")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)  {title}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("app/demo-clips"))
    p.add_argument("--clip", action="append", default=[], metavar="TITLE=PATH",
                   help="a clip to include, with the title the gallery shows")
    args = p.parse_args()
    for item in args.clip:
        title, _, path = item.partition("=")
        src = Path(path)
        if not src.is_file():
            print(f"skipping {src}: not there")
            continue
        name = "-".join(w.lower() for w in title.split() if w.isalnum())[:40] or src.stem
        clip(src, args.out / f"{name}.mp4", title)


if __name__ == "__main__":
    main()
