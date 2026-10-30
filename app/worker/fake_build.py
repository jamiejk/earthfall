"""A stand-in for ``python -m touchdown.build`` with no GPU: same folders, same log lines, a tiny video.

For tests and for trying the app's interface without ComfyUI:
    TOUCHDOWN_BUILD_CMD="{python} {app}/worker/fake_build.py {manifest}" touchdown-app

Env: FAKE_STAGE_SECONDS (default 0.5) per stage; FAKE_FAIL_AT=<stage> exits 1 when that stage starts.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import tomllib
import zlib
from pathlib import Path

STAGES = ("terrain", "stills", "opener", "dive", "cloud", "swoop", "body", "end", "face", "sound")
COLOURS = {"orbit": (20, 40, 90), "cloud": (235, 238, 242), "end": (196, 110, 70)}


def png(path: Path, rgb: tuple[int, int, int], w: int = 64, h: int = 36) -> None:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def video(path: Path) -> None:
    if not shutil.which("ffmpeg"):
        # A 74-byte non-playable file would be served as a finished Touchdown. Fail loudly instead.
        print("fake_build needs ffmpeg on PATH to make its stand-in clip", file=sys.stderr, flush=True)
        raise SystemExit(2)
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24",
                           "-t", "2", "-pix_fmt", "yuv420p", str(path)], capture_output=True)
    if done.returncode != 0:
        print(f"fake_build could not run ffmpeg: {done.stderr.decode(errors='replace')[-300:]}", file=sys.stderr,
              flush=True)
        raise SystemExit(2)


def main() -> None:
    manifest = Path(sys.argv[1])
    m = tomllib.loads(manifest.read_text())
    out = Path(m.get("output", "outputs/touchdown"))
    root = out / m["shot"]["name"]
    delay = float(os.environ.get("FAKE_STAGE_SECONDS", "0.5"))
    fail_at = os.environ.get("FAKE_FAIL_AT", "")
    print(f"fake build of {manifest} (person: {m['shot']['person']!r})", flush=True)
    last = None
    for stage in STAGES:
        if stage == "terrain" and not m.get("terrain", {}).get("enabled", False):
            continue
        if stage == "face" and not m.get("face", {}).get("enabled", False):
            continue
        if stage == "sound" and not m.get("sound", {}).get("enabled", True):
            continue
        key = hashlib.sha1(json.dumps([stage, m["shot"]], sort_keys=True).encode()).hexdigest()[:12]
        folder = root / stage / key
        if (folder / "done.json").is_file():
            print(f"[{stage}] cached {folder}", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if stage == fail_at:
            print(f"Traceback (most recent call last):\nRuntimeError: fake failure in {stage}", flush=True)
            sys.exit(1)
        time.sleep(delay)
        if stage == "stills":
            for name, rgb in COLOURS.items():
                png(folder / f"{name}.png", rgb)
            last = folder / "end.png"
        elif stage == "sound":
            last = folder / "touchdown.mp4"
            video(last)
        else:
            last = folder / "output.txt"
            last.write_text(stage)
        (folder / "done.json").write_text(json.dumps({"output": last.name, "seconds": delay, "facts": {}}))
        print(json.dumps({"stage": stage, "key": key, "seconds": delay, "facts": {}}), flush=True)
    print(f"\nBuilt {last}\nReport {root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
