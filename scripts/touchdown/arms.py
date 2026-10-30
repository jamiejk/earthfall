"""Run a set of Touchdown arms (shots/arms/*.toml) one after another, unattended, one seed per stage.

The arms share shot.name, so stages they have in common render once. Each arm's finished cut, audit and report are
copied to ``outputs/touchdown/arms/<arm>/`` with a contact sheet; an arm that stops on a check is recorded and the
next one runs. Results go to ``outputs/touchdown/arms/results.md``.

    PYTHONPATH=scripts:src <python-with-opencv> -m touchdown.arms shots/arms/*.toml \\
        --url http://127.0.0.1:8191 --comfy-python <project .venv>/bin/python
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(manifest: Path, out_root: Path, extra: list[str]) -> dict:
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "touchdown.build", str(manifest), "--unattended", *extra],
                          capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    folder = out_root / manifest.stem
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "build.log").write_text(log)
    built = [line.split(" ", 1)[1] for line in log.splitlines() if line.startswith("Built ")]
    result = {"arm": manifest.stem, "ok": proc.returncode == 0 and bool(built),
              "minutes": round((time.monotonic() - t0) / 60, 1)}
    if built:
        cut = Path(built[-1])
        shutil.copy(cut, folder / "touchdown.mp4")
        for name in ("audit.json", "report.md"):
            src = cut.parents[2] / name  # outputs/touchdown/<shot>/<stage>/<key>/touchdown.mp4
            if src.is_file():
                shutil.copy(src, folder / name)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(folder / "touchdown.mp4"), "-vf",
                        "select=not(mod(n\\,12)),scale=320:-1,tile=6x4", "-frames:v", "1",
                        str(folder / "contact.jpg")], check=False)
    else:
        tail = [line for line in log.splitlines() if line.strip()][-3:]
        result["stopped"] = " | ".join(tail)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifests", type=Path, nargs="+")
    p.add_argument("--output", type=Path, default=Path("outputs/touchdown/arms"))
    p.add_argument("--url")
    p.add_argument("--comfy-python")
    a = p.parse_args()
    extra = [*(["--url", a.url] if a.url else []), *(["--comfy-python", a.comfy_python] if a.comfy_python else [])]
    a.output.mkdir(parents=True, exist_ok=True)
    lines = ["| arm | result | minutes |", "|---|---|---|"]
    for m in sorted(a.manifests):
        r = run(m, a.output, extra)
        lines.append(f"| {r['arm']} | {'built' if r['ok'] else 'stopped: ' + r.get('stopped', '')} | {r['minutes']} |")
        (a.output / "results.md").write_text("\n".join(lines) + "\n")
        print(lines[-1], flush=True)


if __name__ == "__main__":
    main()
