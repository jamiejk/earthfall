from geoblock_core.place_photos import camera_location, pick_diverse, route_waypoint, score_candidate


def test_camera_metadata_never_promotes_subject_coordinates():
    assert camera_location("{{Object location|36.1|-112.1}}") is None
    assert camera_location("{{Location|36.1|-112.1|heading:90}}")['heading_deg'] == 90
    dms = camera_location("{{Camera location|36|3|0|N|112|6|0|W|heading:NE}}")
    assert abs(dms['latitude'] - 36.05) < 1e-6
    assert abs(dms['longitude'] + 112.1) < 1e-6
    assert dms['heading_deg'] == 45


def test_photo_op_discovery_keeps_pedestrian_reference_and_camera_role():
    p = page("Visitors on Mather Point viewing platform.jpg")
    p['revisions'] = [{'slots': {'main': {'content': '{{Location|36.0617|-112.1078|heading:30}}'}}}]
    assert score(p) is None
    candidate = score_candidate(p, 'Mather Point', 36.0617, -112.1078, 2, purpose='photo_op')
    assert candidate['coordinate_role'] == 'camera'
    assert candidate['camera_location']['heading_deg'] == 30


def test_conflicting_photo_heading_is_not_used_as_a_route_bearing():
    p = page('View north from Mather Point.jpg')
    p['revisions'] = [{'slots': {'main': {'content': '{{Location|36.0617|-112.1078|heading:180}}'}}}]
    assert score(p)['camera_location']['heading_deg'] is None


def page(title, *, licence="CC BY-SA 4.0", width=4000, height=3000, lat=36.062, lon=-112.108,
         artist="Ann", mime="image/jpeg", url="https://upload.wikimedia.org/a.jpg", categories=()):
    return {"pageid": abs(hash(title)) % 10**8, "title": f"File:{title}",
            "coordinates": [{"lat": lat, "lon": lon}] if lat is not None else [],
            "categories": [{"title": c} for c in categories],
            "imageinfo": [{"mime": mime, "width": width, "height": height, "thumburl": url,
                           "descriptionurl": "https://commons.wikimedia.org/wiki/File:x",
                           "extmetadata": {"LicenseShortName": {"value": licence},
                                           "Artist": {"value": f"<a href='u'>{artist}</a>"}}}]}


def score(p, heading=-1):
    return score_candidate(p, "Mather Point, Grand Canyon", 36.0617, -112.1078, 2.0, heading)


def test_only_open_licences_and_landscape_jpegs_pass():
    assert score(page("Mather Point view.jpg"))["artist"] == "Ann"
    assert score(page("Mather Point view.jpg", licence="CC BY-NC 2.0")) is None
    assert score(page("Mather Point view.jpg", licence="All rights reserved")) is None
    assert score(page("Mather Point view.jpg", mime="image/png")) is None
    assert score(page("Mather Point view.jpg", width=2000, height=3000)) is None  # portrait
    assert score(page("Mather Point view.jpg", width=9000, height=2000)) is None  # wide panorama
    assert score(page("Mather Point view.jpg", url="https://example.com/a.jpg")) is None


def test_off_subject_and_distant_photos_are_dropped():
    assert score(page("Ranger program at Mather Point.jpg")) is None
    assert score(page("Corvus corax Mather Point.jpg")) is None
    assert score(page("Mather Point view.jpg", lat=36.2)) is None


def test_ranking_prefers_name_quality_mark_and_matching_view_direction():
    plain = score(page("Canyon 12.jpg"))["score"]
    named = score(page("Mather Point Grand Canyon.jpg"))["score"]
    marked = score(page("Canyon 12.jpg", categories=["Category:Quality images"]))["score"]
    foggy = score(page("Canyon fog 12.jpg"))["score"]
    assert named > plain and marked > plain and foggy < plain
    north = score(page("View north from Mather Point.jpg"), heading=5)["score"]
    south = score(page("View south from Mather Point.jpg"), heading=5)["score"]
    assert north > south


def test_pick_diverse_keeps_one_photo_per_author():
    items = [{"artist": a, "page_id": i} for i, a in enumerate(["Ann", "ann", "Bo", "Cy", "Bo"])]
    assert [i["page_id"] for i in pick_diverse(items, 3)] == [0, 2, 3]


def test_edited_versions_are_not_independent_reference_views():
    items = [{"artist": a, "page_id": i} for i, a in enumerate([
        "Yann (talk)", "Yann; edited by King of Hearts", "Yann; edited by Jim Carter", "Ann"])]
    assert [i["page_id"] for i in pick_diverse(items, 3)] == [0, 3]


def test_route_waypoint_gives_camera_position_and_heading():
    scene = {"projected_crs": "EPSG:26912", "origin_projected_m": [400232.5, 3991359.7]}
    route = {"positions": [[0.0, 0.0, 2200.0], [0.0, 1000.0, 2000.0]],
             "look_at": [[0.0, 500.0, 2100.0], [1000.0, 1000.0, 1900.0]]}
    start, end = route_waypoint(scene, route, 0), route_waypoint(scene, route, 99)
    assert abs(start["latitude"] - 36.0617) < 0.001 and abs(start["longitude"] + 112.1078) < 0.001
    assert start["heading_deg"] == 0.0 and end["heading_deg"] == 90.0
    assert 0.008 < end["latitude"] - start["latitude"] < 0.01  # 1 km north


def test_campsites_are_dropped_and_river_level_views_rank_lower():
    assert score(page("Mather Campground group site tents")) is None
    assert score(page("Colorado River view from the Point"))["score"] < score(page("Grand Canyon view from the Point"))["score"]


def test_commons_thumbnails_on_the_thumb_host_are_accepted_and_other_hosts_are_not():
    assert score(page("Mather Point view.jpg", url="https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ab/x.jpg/3840px-x.jpg"))
    assert score(page("Mather Point view.jpg", url="https://example.org/x.jpg")) is None
