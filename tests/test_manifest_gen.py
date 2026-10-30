"""touchdown.manifest_gen: FLUX 3 Touchdown manifests generated from the place catalogue (no network, no GPU).

    PYTHONPATH=scripts:. python3 -m pytest -q tests/test_manifest_gen.py
"""
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from touchdown import manifest_gen as mg  # noqa: E402

HAND = {"monument-valley": "monument-valley", "grand-canyon-mather-point": "grand-canyon-mather",
        "yosemite-tunnel-view": "yosemite-tunnel-view"}
PLACE_WORDS = ("scene", "region", "approach", "swoop_view", "swoop_toward", "ambient")


def supported_ids():
    return [row["id"] for row in mg.catalogue_status() if row["ok"] and not row["id"].startswith("custom-")]


def test_catalogue_support_is_the_us_lidar_and_naip_places_and_fine_places_abroad():
    ids = supported_ids()
    assert len(ids) == 29  # 24 US lidar + NAIP, 5 abroad with fine national terrain and photos
    assert {"monument-valley", "grand-canyon-mather-point", "yosemite-tunnel-view", "bryant-park"} <= set(ids)
    rows = {row["id"]: row for row in mg.catalogue_status()}
    assert rows["mount-fuji"]["ok"]
    assert not rows["ben-nevis"]["ok"] and rows["ben-nevis"]["why"].startswith("coming soon: only 30 m terrain")
    assert not rows["zion-angels-landing"]["ok"]  # US, NAIP, but no 1 m lidar project listed
    assert not rows["haleakala-summit"]["ok"]     # Hawaii: no NAIP
    with pytest.raises(ValueError):
        mg.generate("ben-nevis", "s.jpg", "p")


@pytest.mark.parametrize("place_id", sorted(HAND))
def test_generated_manifest_has_every_hand_written_field(place_id):
    m = mg.generate(place_id, "subject.jpg", "a person in a red coat")
    hand = tomllib.loads((ROOT / "shots" / "flux3" / f"{HAND[place_id]}.toml").read_text())
    assert m["shot"]["place"] == place_id
    for key in PLACE_WORDS:
        assert m["shot"][key] and key in hand["shot"], key
    for key in ("orbit_region", "end_prompt", "opener_end_prompt", "flux3", "flux3_url", "seed"):
        assert m["stills"][key], key
    assert "{person}" in m["stills"]["end_prompt"] and "{scene}" in m["stills"]["opener_end_prompt"]
    assert "preserve the shape" not in m["stills"]["end_prompt"].lower()
    assert "end_conditioning" not in m["stills"] and "end" in m["stills"]["flux3"]  # outline-map default
    assert "lock_pin" not in m["end"] and "keep_after_pin" not in m["end"]          # auto_switch defaults
    # not per place: the newest hand-tuned FLUX 3 settings, unchanged
    base = mg.base_manifest()
    for table in ("opener", "cloud", "body", "face", "sound"):
        assert m[table] == base[table], table
    assert {k: v for k, v in m["end"].items()} == {k: v for k, v in base["end"].items() if k != "lock_pin"}
    assert m["swoop"]["frames"] == hand["swoop"]["frames"] and m["swoop"]["seeds"] == hand["swoop"]["seeds"]
    assert m["dive"]["oblique"] == hand["dive"]["oblique"] and m["dive"]["sea"] is False


def test_terrain_follows_the_catalogue_and_the_automatic_search():
    gc = mg.generate("grand-canyon-mather-point", "s.jpg", "p")["terrain"]
    hand = tomllib.loads((ROOT / "shots/flux3/grand-canyon-mather.toml").read_text())["terrain"]
    for key in ("project", "utm_zone", "heading", "embedding_rank", "view_previews", "search_radius_m",
                "avoid_pavement", "surface"):
        assert gc[key] == hand[key], key
    mv = mg.generate("monument-valley", "s.jpg", "p")["terrain"]
    assert mv["heading"] == 58.0 and mv["project"] == "UT_WestEast_B22" and "latitude" not in mv
    assert "embedding_rank" not in mv and "view_query" not in mv
    yo = mg.generate("yosemite-tunnel-view", "s.jpg", "p")["terrain"]
    assert yo["heading"] == 75.1 and yo["utm_zone"] == 11


def test_prompts_format_with_the_build_words():
    for place_id in supported_ids():
        m = mg.generate(place_id, "s.jpg", "a person")
        words = {"ambient": "", **{k: v for k, v in m["shot"].items() if isinstance(v, str)}, "scene_ambient": "x"}
        for text in (m["stills"]["end_prompt"], m["stills"]["opener_end_prompt"], m["swoop"]["prompt"],
                     m["end"]["prompt"], m["end"]["lock_prompt"]):
            text.format(**words)
        for key in PLACE_WORDS:
            assert "{" not in m["shot"][key] and "  " not in m["shot"][key], (place_id, key)
        assert m["stills"]["scene_type"] in ("landmark", "urban", "nature")


def test_scene_types_and_families():
    def kind(place_id):
        m = mg.generate(place_id, "s.jpg", "p")
        return m["generated"]["family"], m["stills"]["scene_type"]
    assert kind("monument-valley") == ("desert", "nature")
    assert kind("grand-canyon-mather-point") == ("canyon", "nature")
    assert kind("golden-gate-battery-spencer") == ("coast", "landmark")
    assert kind("bryant-park")[1] == "urban"
    bixby = mg.generate("big-sur-bixby-bridge", "s.jpg", "p")
    assert bixby["dive"]["sea"] is True


def test_naip_ground_words_decide_the_surface():
    paved = {"shares": {"paved": 0.55, "trees": 0.3, "rock": 0.15}, "words": "mostly paved paths and roads, with dark conifers"}
    m = mg.generate("yosemite-tunnel-view", "s.jpg", "p", ground=paved)
    assert "surface" not in m["terrain"] and "avoid_pavement" not in m["terrain"]
    assert m["stills"]["layout_gate"] == "landmarks" and "paved" in m["shot"]["scene"]
    assert "mostly paved paths and roads" in m["stills"]["end_prompt"] and m["generated"]["naip_ground"]
    rock = {"shares": {"rock": 0.7, "trees": 0.2}, "words": "mostly bare pale rock slabs and boulders"}
    m = mg.generate("yosemite-glacier-point", "s.jpg", "p", ground=rock)
    assert m["terrain"]["surface"] == "natural" and "bare rock slabs" in m["shot"]["scene"]


def test_custom_point_round_trips_through_the_build(tmp_path):
    pytest.importorskip("numpy")
    from touchdown.build import Build
    from geoblock_core.places import find_place

    place = mg.place_from_point(36.8619, -111.3743, name="Somewhere {odd} \"quoted\"", project="AZ_NorthEast_D23",
                                zone=12, region="Arizona")
    assert place["id"] == "custom-36.86190n-111.37430w" and place["name"] == "Somewhere odd quoted"
    subject = tmp_path / "s.jpg"
    subject.write_bytes(b"jpeg")
    m = mg.generate(place, str(subject), "a person")
    assert m["place"]["id"] == place["id"] and m["terrain"]["project"] == "AZ_NorthEast_D23"
    assert m["terrain"]["heading"] == -1.0 and m["terrain"]["embedding_rank"] is True
    path = tmp_path / "custom.toml"
    path.write_text(mg.dumps(m))
    b = Build(path, tmp_path / "out", None, True)
    assert find_place(place["id"])["name"] == "Somewhere odd quoted"
    assert b.m["end"]["lock_pin"] == "48" and b.m["end"]["keep_after_pin"] == 8  # auto_switch defaults
    assert b.end_conditioning() == "structure"
    assert len(b.key("terrain", ())) == 12
    with pytest.raises(ValueError):
        mg.place_from_point(21.3, -157.8, project="X", zone=4)  # Hawaii: no NAIP


def test_generated_manifests_dry_run(tmp_path):
    pytest.importorskip("numpy")
    from touchdown.build import Build
    subject = tmp_path / "s.jpg"
    subject.write_bytes(b"jpeg")
    for place_id in HAND:
        path = tmp_path / f"{place_id}.toml"
        path.write_text(mg.dumps(mg.generate(place_id, str(subject), "a person")))
        b = Build(path, tmp_path / "out", None, True)
        assert b.build(None) is not None


def test_cli(tmp_path, capsys):
    assert mg.main(["--list"]) == 0
    assert "coming soon" in capsys.readouterr().out
    out = tmp_path / "hb.toml"
    assert mg.main(["horseshoe-bend", "--subject", "s.jpg", "--person", "p", "-o", str(out)]) == 0
    assert tomllib.loads(out.read_text())["shot"]["place"] == "horseshoe-bend"


def test_badlands_get_their_own_words():
    w = mg.words(mg.find("death-valley-zabriskie"))
    assert w["family"] == "badlands" and "mudstone" in w["end_prompt"] and "trees" not in w["backdrop"]
    assert mg.generate("death-valley-zabriskie", "s.jpg", "p")["stills"]["layout_gate"] == "landmarks"
