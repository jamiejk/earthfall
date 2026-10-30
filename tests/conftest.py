"""Make the repository's own packages importable, however pytest is invoked.

The pipeline lives in ``scripts/`` (``touchdown.*``) and the library in ``src/`` (``geoblock_core``); the test
modules each put them on ``sys.path`` themselves, which only works when pytest is started from the repository
root. This conftest does it for every run: ``pytest``, ``pytest tests/test_app.py``, ``pytest ../tests`` from
another directory, or an editor's test runner. The repository root goes on the path too, so ``app.server``
(the web app, a namespace package) can be imported as well.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _path in (ROOT / "scripts", ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
