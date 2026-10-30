import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from touchdown.authored_scene import asset_plan, camera_pose, person_matte, read_scene


def test_every_wide_stage_inherits_master_but_not_identity(tmp_path):
    cfg = dict(place="Paris", character="Eiffel Tower and river", look="warm left light",
               references=[dict(local_path="tower.jpg"), dict(local_path="river.jpg")],
               subject="identity.jpg", seed=12)
    stages = {s["name"]: s for s in asset_plan(cfg, tmp_path)}
    for name in ("ground", "person", "under_cloud", "orbit"):
        assert stages[name]["context_parent"] == str(tmp_path / "master.png")
    assert stages["person"]["references"][0] == str(tmp_path / "master.png")
    for name in ("master", "ground", "under_cloud", "orbit"):
        assert "identity.jpg" not in stages[name]["references"]
    for name in ("orbit", "under_cloud"):
        assert str(tmp_path / "master_look_reference.png") in stages[name]["references"]
    assert "400 kilometres" in stages["orbit"]["prompt"]


def test_camera_moves_forward_downward_and_settles():
    poses = [camera_pose(t)["position"] for t in np.linspace(0,1,96)]
    p = np.array(poses)
    assert np.all(np.diff(p[:,1]) > 0)
    assert np.all(np.diff(p[:,2]) < 0)
    assert np.all(p[:,2] >= 1.5)
    assert np.linalg.norm(p[-1]-p[-2]) < .02
    with pytest.raises(ValueError):
        camera_pose(1.1)


def test_matte_retains_legs_and_clear_gap_without_a_coat_colour_assumption(tmp_path):
    im = np.full((300,200,3), (0,255,255), dtype=np.uint8)
    im[30:70,80:120] = (130,85,60)
    im[70:180,60:140] = (30,40,90)
    im[180:270,65:90] = (20,20,20)
    im[180:270,110:135] = (20,20,20)
    source = tmp_path / "person.png"; Image.fromarray(im).save(source)
    result = person_matte(source, tmp_path / "rgba.png")
    rgba = np.asarray(Image.open(tmp_path / "rgba.png"))
    assert result["bounds"] == [60,30,140,270]
    assert rgba[220,15,3] == 255
    assert rgba[220,40,3] == 0
    assert rgba[220,65,3] == 255


def test_matte_rejects_a_clipped_subject(tmp_path):
    im = np.full((300,200,3), (0,255,255), dtype=np.uint8)
    im[0:280,50:150] = (100,80,70)
    Image.fromarray(im).save(tmp_path / "clipped.png")
    with pytest.raises(ValueError):
        person_matte(tmp_path / "clipped.png", tmp_path / "out.png")


def test_references_resolve_locally_after_moving_a_checkout(tmp_path):
    for name in ("a.jpg", "b.jpg", "subject.jpg"):
        (tmp_path / name).write_bytes(b"image")
    records = [dict(page_id=i, local_path=f"/old/checkout/{name}") for i,name in enumerate(("a.jpg","b.jpg"))]
    (tmp_path / "refs.json").write_text(json.dumps(dict(selected=records)))
    (tmp_path / "scene.toml").write_text('subject="subject.jpg"\nreference_manifest="refs.json"\nreference_ids=[0,1]\n')
    cfg = read_scene(tmp_path / "scene.toml")
    assert cfg["references"][0]["local_path"] == str(tmp_path / "a.jpg")
