"""Print each manifest's terrain cache key, as JSON {manifest: {"key": ..., "neutral": ...} or null}.

``key`` is the folder name the build uses. ``neutral`` is the same hash with the person-specific [shot] fields
(subject, person, name) blanked: the terrain stage never reads them, so two manifests with the same ``neutral``
key have identical terrain inputs, and one's finished terrain folder is valid for the other.

Run by the worker in the pipeline's own Python (cwd and PYTHONPATH as for a build), so the key is computed by
touchdown.build itself and can never drift from what a build would use:
    python app/worker/terrain_key.py --output <dir> --descent 45 a.toml b.toml
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

PERSONAL = ("subject", "person", "name")  # [shot] fields a job changes; the terrain stage reads none of them


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifests", nargs="+", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--descent", default="45")
    a = p.parse_args()
    from touchdown.build import Build

    keys = {}
    for manifest in a.manifests:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                b = Build(manifest, a.output, None, True, descent=a.descent)
                if not b.terrain_mode:
                    keys[str(manifest)] = None
                    continue
                key = b.key("terrain", ())
                for k in PERSONAL:
                    b.m["shot"].pop(k, None)
                keys[str(manifest)] = {"key": key, "neutral": b.key("terrain", ())}
        except Exception as e:  # one unreadable manifest must not stop the others
            keys[str(manifest)] = None
            keys.setdefault("_errors", {})[str(manifest)] = repr(e)
    print(json.dumps(keys))


if __name__ == "__main__":
    main()
