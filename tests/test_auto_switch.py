import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from touchdown.auto_switch import (NO_CORRIDOR, flux3_defaults, geotagged, osm_heights,  # noqa: E402
                                   search_relaxing_pavement, use_photo_end)


def test_flux3_defaults_fill_pin_and_tail_only_when_unset():
    m = {"stills": {"flux3": ["end"]}, "end": {"frames": 73}}
    assert flux3_defaults(m) == {"lock_pin": "48", "keep_after_pin": 8}
    assert m["end"]["lock_pin"] == "48" and m["end"]["keep_after_pin"] == 8
    own = {"stills": {"flux3": ["end"]}, "end": {"lock_pin": "none"}}
    assert flux3_defaults(own) == {"keep_after_pin": 8}
    assert own["end"]["lock_pin"] == "none"


def test_flux3_defaults_leave_other_manifests_alone():
    m = {"stills": {}, "end": {}}
    assert flux3_defaults(m) == {}
    assert m["end"] == {}


def test_pavement_relaxed_only_when_no_corridor():
    calls = []

    def search(avoid_pavement, **kw):
        calls.append(avoid_pavement)
        if avoid_pavement:
            raise ValueError(NO_CORRIDOR)
        return {"x": 1}

    result, relaxed = search_relaxing_pavement(search, {"avoid_pavement": True}, radius_m=150)
    assert result == {"x": 1} and relaxed and calls == [True, False]


def test_pavement_other_errors_and_opt_out_raise():
    def other(avoid_pavement, **kw):
        raise ValueError("No level, image-supported straight approach near the catalogue coordinate")

    with pytest.raises(ValueError):
        search_relaxing_pavement(other, {"avoid_pavement": True})

    def no_corridor(avoid_pavement, **kw):
        raise ValueError(NO_CORRIDOR)

    with pytest.raises(ValueError):
        search_relaxing_pavement(no_corridor, {"avoid_pavement": True, "auto_relax_pavement": False})


def test_pavement_untouched_when_it_finds_a_view():
    result, relaxed = search_relaxing_pavement(lambda avoid_pavement: avoid_pavement, {"avoid_pavement": True})
    assert result is True and not relaxed


def test_geotagged_by_camera_template_or_exif(tmp_path):
    from PIL import Image
    assert geotagged({"camera_location": {"latitude": 1, "longitude": 2}})
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(plain)
    assert not geotagged({"local_path": str(plain)})
    tagged = tmp_path / "gps.jpg"
    im = Image.new("RGB", (8, 8))
    exif = im.getexif()
    exif.get_ifd(0x8825).update({1: "N", 2: (37.0, 49.0, 39.0), 3: "W", 4: (122.0, 28.0, 53.0)})
    im.save(tagged, exif=exif)
    assert geotagged({"local_path": str(tagged)})


def test_osm_heights_needs_a_tall_structure():
    assert not osm_heights(None)
    assert not osm_heights({"structures": [{"max": 6}]})
    assert osm_heights({"structures": [{"max": 6}, {"max": 225}]})


def test_photo_end_switch():
    photo = [{"camera_location": {"latitude": 1, "longitude": 2}}]
    osm = {"structures": [{"max": 225, "kind": "tower", "bridge": "golden-gate"}]}
    assert use_photo_end({}, 2, photo, osm)[0]
    arch = {"structures": [{"max": 85, "kind": "deck", "bridge": "bixby"}]}
    assert use_photo_end({}, 3, photo, arch) == (False, "OSM maps no bridge tower for the photo camera solve")
    assert not use_photo_end({}, 1, photo, osm)[0]           # one failure is not enough
    assert not use_photo_end({}, 3, [{}], osm)[0]            # no geotagged photo
    assert not use_photo_end({}, 3, photo, {"structures": []})[0]  # no OSM heights
    assert not use_photo_end({"auto_photo_end": False}, 3, photo, osm)[0]
    assert use_photo_end({"end_from_photo": True}, 0, [], None) == (True, "set in the manifest")
    assert use_photo_end({"end_from_photo": False}, 9, photo, osm) == (False, "set in the manifest")
