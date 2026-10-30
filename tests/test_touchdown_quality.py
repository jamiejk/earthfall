import tomllib
from pathlib import Path

import pytest
import tomli_w

from touchdown.build import Build

ROOT = Path(__file__).resolve().parents[1]


def build(tmp_path, quality=None):
    m = tomllib.loads((ROOT / "shots/flux3/monument-valley.toml").read_text())
    base = ROOT / "shots/flux3"
    for k in ("end_still", "end_reference", "subject"):
        if m["shot"].get(k):
            m["shot"][k] = str((base / m["shot"][k]).resolve())
    if quality:
        m.setdefault("video", {})["quality"] = quality
    path = tmp_path / f"{quality or 'default'}.toml"
    path.write_text(tomli_w.dumps(m))
    return Build(path, tmp_path / "out", None, True)


def test_draft_quality_skips_the_2x_pass_and_keys_the_video_stages(tmp_path):
    full, draft = build(tmp_path), build(tmp_path, "draft")
    assert full.full_res() and not draft.full_res()
    for stage in ("opener", "swoop"):
        assert full.key(stage, ()) != draft.key(stage, ()), stage
    assert full.key("stills", ()) == draft.key("stills", ())
    with pytest.raises(ValueError):
        build(tmp_path, "8k").full_res()
