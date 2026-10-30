"""A saved setup must satisfy the worker that consumes it, not just look complete.

The first version of this builder stripped references to three keys while hashing the id and then
saved the stripped copy. Every file looked present and the build died at its first input check with
KeyError: 'name'. These tests pin the contract the worker actually reads.
"""
import hashlib

from touchdown.studio_setup import setup_from, slug

SUBJECT = {"name": "subject.jpg", "sha256": "a" * 64}
DESTINATION = {"id": "teotihuacan", "title": "Teotihuacan", "region": "State of Mexico",
               "country": "Mexico", "latitude": 19.6925, "longitude": -98.8438}


def manifest(page_ids=(42551786, 42551788)):
    selected = []
    for i, page_id in enumerate(page_ids):
        selected.append({
            "page_id": page_id, "title": f"Teotihuacan {i}.jpg", "artist": "Ovedc",
            "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
            "source": "Wikimedia Commons", "commons_page": f"https://commons.wikimedia.org/?curid={page_id}",
            "source_sha256": str(page_id) * 2, "local_path": f"/tmp/location-{i}.jpg",
            "sha256": hashlib.sha256(str(page_id).encode()).hexdigest(),
        })
    return {"selected": selected, "candidates": []}


def test_every_reference_carries_what_the_worker_reads():
    setup = setup_from(manifest(), SUBJECT, DESTINATION, ending="arms-open", framing="three-quarter")
    for index, reference in enumerate(setup["references"]):
        assert reference["name"] == f"location-{index}.jpg"
        assert reference["sha256"] and len(reference["sha256"]) == 64
        assert reference["page_id"] and reference["title"]
        assert reference["license"] and reference["url"].startswith("https://")


def test_the_id_tracks_content_not_the_download():
    first = setup_from(manifest(), SUBJECT, DESTINATION, ending="arms-open", framing="three-quarter")
    moved = manifest()
    for item in moved["selected"]:
        item["local_path"] = "/somewhere/else.jpg"      # a re-download at another path
    second = setup_from(moved, SUBJECT, DESTINATION, ending="arms-open", framing="three-quarter")
    assert first["id"] == second["id"]

    different = setup_from(manifest(page_ids=(1, 2)), SUBJECT, DESTINATION,
                           ending="arms-open", framing="three-quarter")
    assert different["id"] != first["id"]


def test_fewer_than_two_references_is_refused():
    try:
        setup_from(manifest(page_ids=(42551786,)), SUBJECT, DESTINATION, ending="arms-open",
                   framing="three-quarter")
    except ValueError as exc:
        assert "two" in str(exc)
    else:
        raise AssertionError("a one-reference setup must be refused")


def test_slug_is_safe_for_a_place_name():
    assert slug("Teotihuacan") == "teotihuacan"
    assert slug("Aït Benhaddou") == "ait-benhaddou"
    assert slug("Chichén Itzá") == "chichen-itza"
    assert slug("Mont Saint-Michel") == "mont-saint-michel"


def test_vision_screen_skips_objects_and_crowds(tmp_path, monkeypatch):
    from touchdown import studio_setup
    candidates = [{"page_id": i, "title": f"Chambord {i}.jpg", "artist": f"a{i}"} for i in range(6)]
    monkeypatch.setattr(studio_setup, "find_place_photos", lambda *a, **k: candidates)

    def download(client, item, folder, limit):
        path = folder / f"commons_{item['page_id']}.jpg"
        path.write_bytes(b"x")
        return {**item, "local_path": str(path), "sha256": "0" * 64, "license": "CC0", "license_url": ""}
    monkeypatch.setattr(studio_setup, "_download", download)
    monkeypatch.setattr(studio_setup, "attribution", lambda selected: "")
    views = {0: ("detail_or_object", False), 1: ("landmark_exterior", True), 2: ("aerial_of_landmark", False),
             3: ("interior", False), 4: ("landmark_exterior", False), 5: ("landmark_exterior", False)}
    night = {4}
    seen = []

    def judge(path, prompt, output):
        page = int(path.stem.split("_")[1])
        seen.append(page)
        view, people = views[page]
        return {"shows_named_landmark": True, "view": view, "people_prominent": people, "natural_daylight": page not in night, "fisheye_or_distorted": False, "evidence": ""}
    manifest = studio_setup.screened_photos("Chambord", "Loir-et-Cher", "France", 47.616, 1.517, 2, tmp_path, judge)
    assert [p["page_id"] for p in manifest["selected"]] == [2, 5]  # 4 is a night light show
    assert seen == [0, 1, 2, 3, 4, 5]  # stops once two are accepted
    assert not (tmp_path / "commons_0.jpg").exists()  # rejected downloads are not left as references
    assert manifest["picked_by"] == "vision screen"


def test_vision_screen_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    import pytest
    from touchdown import studio_setup
    monkeypatch.setattr(studio_setup, "find_place_photos", lambda *a, **k: [{"page_id": 1, "title": "t", "artist": "a"}])
    monkeypatch.setattr(studio_setup, "_download", lambda c, item, f, l: {**item, "local_path": str(f / "x.jpg")})
    with pytest.raises(ValueError, match="accepted 0 of 1"):
        studio_setup.screened_photos("X", "", "", 0, 0, 2, tmp_path,
                                     lambda *a: {"shows_named_landmark": True, "view": "interior", "people_prominent": False, "natural_daylight": True, "fisheye_or_distorted": False, "evidence": ""})


def test_vision_screen_rejects_a_namesake_building(tmp_path, monkeypatch):
    import pytest
    from touchdown import studio_setup
    candidates = [{"page_id": i, "title": f"Ávila {i}.jpg", "artist": f"a{i}"} for i in range(2)]
    monkeypatch.setattr(studio_setup, "find_place_photos", lambda *a, **k: candidates)
    monkeypatch.setattr(studio_setup, "_download", lambda c, item, f, l: {**item, "local_path": str(f / f"commons_{item['page_id']}.jpg")})
    # Page 1 is the Biblioteca Pública de Ávila: a clean daylight exterior, but not the walls.
    judge = lambda path, prompt, output: {"shows_named_landmark": path.stem != "commons_1", "view": "landmark_exterior",
                                          "people_prominent": False, "natural_daylight": True,
                                          "fisheye_or_distorted": False, "evidence": ""}
    with pytest.raises(ValueError, match="accepted 1 of 2"):
        studio_setup.screened_photos("Walls of Ávila", "", "", 40.66, -4.70, 2, tmp_path, judge)
