import json

import numpy as np
import pytest

from geoblock_core.run import main
from geoblock_core.workflow import (CONTROL_VIDEO, H3_VIDEO, ShotRequest, blocking_graph, h3_graph)


def links(graph):
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                yield node_id, name, value


def test_blocking_graph_from_coordinates_is_closed():
    graph = blocking_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11))
    assert graph["2"]["class_type"] == "GeoBlockPlaceFromCoordinates"
    assert graph[CONTROL_VIDEO]["inputs"]["video"] == ["5", 0]
    assert all(source in graph for _, _, (source, _) in links(graph))


def test_blocking_graph_passes_the_palette():
    request = ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, palette="grand-canyon")
    assert blocking_graph(request)["5"]["inputs"]["palette"] == "grand-canyon"
    assert blocking_graph(ShotRequest(name="x", latitude=0, longitude=0))["5"]["inputs"]["palette"] == "default"


def test_blocking_graph_from_search():
    graph = blocking_graph(ShotRequest(name="", search="Half Dome", result_number=2))
    assert graph["1"]["inputs"]["query"] == "Half Dome"
    assert graph["2"]["inputs"]["result_number"] == 2


def test_h3_graph_wires_render_outputs_and_photos():
    graph = h3_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11,
                                 place_photos=["a.jpg", "b.jpg"]))
    h3 = graph["45"]["inputs"]
    assert h3["length"] == ["5", 2]
    assert h3["ref_images.ref_image_0"] == ["5", 1]
    assert h3["ref_images.ref_image_2"] == ["32", 0] and graph["32"]["inputs"]["image"] == "b.jpg"
    assert graph[H3_VIDEO]["class_type"] == "SaveVideo"
    assert all(source in graph for _, _, (source, _) in links(graph))


def test_h3_graph_rejects_clips_over_15_seconds():
    with pytest.raises(ValueError, match="15 s"):
        h3_graph(ShotRequest(name="x", latitude=36, longitude=-112, duration_s=20))


def test_cli_saves_graph_without_submitting(tmp_path):
    path = tmp_path / "graph.json"
    assert main(["--latitude", "36.06", "--longitude", "-112.11", "--name", "Rim",
                 "--preset", "Orbit", "--save-graph", str(path)]) == 0
    assert json.loads(path.read_text())["4"]["inputs"]["preset"] == "Orbit"



def test_self_hosted_target_never_receives_the_cloud_key(monkeypatch):
    pytest.importorskip("comfy_sdk")
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import os
    import threading
    from geoblock_core.run import _client

    seen = []

    class Recorder(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"

    def request_with(target):
        with _client(target) as client:
            try:
                client.jobs.get("probe")
            except Exception:
                pass

    monkeypatch.setenv("COMFY_API_KEY", "cloud-secret")
    monkeypatch.delenv("COMFY_BASE_URL", raising=False)
    monkeypatch.delenv("GEOBLOCK_PROXY_TOKEN", raising=False)
    request_with(url)
    monkeypatch.setenv("GEOBLOCK_PROXY_TOKEN", "pod-token")
    request_with(url)
    server.shutdown()
    assert seen and seen[0] is None
    assert set(seen[1:]) == {"Bearer pod-token"}
    assert os.environ["COMFY_API_KEY"] == "cloud-secret" and "COMFY_BASE_URL" not in os.environ


def test_h3_graph_turbo_depth_and_first_frame_are_wired():
    graph = h3_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, place_photos=["a.jpg"],
                                 lora="turbo.safetensors", steps=4, opening_reference=False,
                                 depth_strength=0.8, depth_end=0.6, first_frame="first.png"))
    assert graph["5"]["inputs"]["depth_pass"] is True
    assert graph["56"]["inputs"]["video"] == ["5", 4]
    assert graph["57"]["inputs"]["model"] == ["54", 0] and graph["57"]["inputs"]["end_percent"] == 0.6
    assert graph["48"]["inputs"]["model"] == ["57", 0] and graph["50"]["inputs"]["model"] == ["57", 0]
    assert graph["48"]["inputs"]["conditioning"] == ["59", 0] and graph["59"]["inputs"]["frame_idx"] == 0
    h3 = graph["45"]["inputs"]
    assert h3["ref_images.ref_image_0"] == ["31", 0] and "ref_images.ref_image_1" not in h3
    assert "<Picture 1> show only" in h3["prompt"] and "opening frame" not in h3["prompt"]
    assert all(source in graph for _, _, (source, _) in links(graph))


def test_plain_h3_graph_has_no_extra_stages():
    graph = h3_graph(ShotRequest(name="Rim", latitude=36.06, longitude=-112.11))
    assert graph["5"]["inputs"]["depth_pass"] is False
    assert not {"54", "55", "57", "59", "23"} & graph.keys()
    assert graph["48"]["inputs"]["model"] == ["42", 0]


def test_weather_reaches_the_render_and_the_prompt():
    request = ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, weather="golden hour")
    assert blocking_graph(request)["5"]["inputs"]["weather"] == "golden hour"
    prompt = next(node["inputs"]["prompt"] for node in h3_graph(request).values()
                  if node["class_type"] == "MiniMaxH3ReferenceToVideo")
    assert "Golden hour" in prompt


def test_ltx_graph_is_closed_and_pins_first_and_last_frames():
    from geoblock_core.workflow import LTX_KEYFRAMES, LTX_VIDEO, ltx_frame_count, ltx_graph
    request = ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, palette="grand-canyon",
                          weather="thunderstorm")
    graph = ltx_graph(request)
    assert all(source in graph for _, _, (source, _) in links(graph))
    assert ltx_frame_count(10) == 241 and graph["108"]["inputs"]["length"] == 241
    assert graph["74"]["inputs"]["batch_index"] == 240 and graph["66"]["inputs"]["batch_index"] == 0
    assert graph["67"]["inputs"]["prompt"] == ["73", 0] and graph["81"]["inputs"]["frame"] == 240
    assert graph["73"]["inputs"]["weather"] == "thunderstorm"
    assert graph["105"]["inputs"]["text"].startswith("3DREAL.")
    assert graph["118"]["inputs"]["samples"] == ["130", 2] and graph["150"]["inputs"]["frame_idx"] == -1
    assert {LTX_VIDEO, *LTX_KEYFRAMES[:2]} <= graph.keys() and "93" not in graph
    zoom = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, preset="Earth zoom"))
    assert all(source in zoom for _, _, (source, _) in links(zoom))
    assert [zoom[str(140 + k)]["inputs"]["frame_idx"] for k in range(8)] == [8, 32, 56, 72, 88, 112, 176, 64]
    assert zoom["111"]["inputs"]["latent"] == ["147", 2] and zoom["158"]["inputs"]["frame_idx"] == -1 and zoom["105"]["inputs"]["text"].count("trees") == 0
    assert all(node["inputs"].get("strength", 1.0) == 1.0 for node in graph.values()
               if node["class_type"] in ("LTXVAddGuide", "LTXVImgToVideoInplace"))


def test_render_command_keeps_depth_and_adds_imagery(tmp_path, monkeypatch):
    import subprocess
    from geoblock_core import blender
    seen = {}

    class Stop(Exception):
        pass

    def fake_popen(command, **kwargs):
        seen["command"] = command
        raise Stop
    monkeypatch.setattr(blender, "find_blender", lambda: ["blender"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    imagery = tmp_path / "imagery.json"
    imagery.write_text("{}")
    with pytest.raises(Stop):
        blender.render_route({"project_id": "p"}, {"route_sha256": "r", "preset": "Rim reveal"}, tmp_path,
                             depth=True, imagery=imagery)
    assert "--depth" in seen["command"] and seen["command"][-2:] == ["--imagery", str(imagery)]


def test_keyframe_prompt_describes_only_what_the_frame_shows():
    from geoblock_core.looks import keyframe_prompt
    top = keyframe_prompt("Mather Point", "grand-canyon", "Clear.", {"over": "top", "above_ground_m": 40})
    drop = keyframe_prompt("Mather Point", "grand-canyon", "Clear.", {"over": "drop", "above_ground_m": 1500})
    assert "pinyon-juniper" in top and "Supai" not in top
    assert "Supai" in drop and "pinyon-juniper" not in drop and "no foreground edge" in drop


def test_earth_zoom_lands_on_a_subject_photo_or_an_invented_figure():
    from geoblock_core.looks import keyframe_prompt
    from geoblock_core.workflow import SUBJECT, ltx_graph
    request = ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, preset="Earth zoom",
                          subject_photo="me.png")
    graph = ltx_graph(request)
    assert all(source in graph for _, _, (source, _) in links(graph))
    assert graph[SUBJECT]["inputs"]["photo"] == ["171", 0] and graph["171"]["inputs"]["image"] == "me.png"
    last = graph["81"]  # prompt node for frame 240
    assert last["inputs"]["subject"] == [SUBJECT, 0] and last["inputs"]["reference"] is True
    assert graph["75"]["inputs"]["image2"] == ["240c", 0] and graph["75"]["inputs"]["image3"] == ["171", 0]
    assert graph["241"]["inputs"]["image2"] == ["171", 0] and "image3" not in graph["241"]["inputs"]  # arrival
    assert graph["215"]["inputs"]["frame"] == 32  # the cloud whiteout keyframe
    assert "_decoded" not in graph
    assert "image2" not in graph["67"]["inputs"]  # not on the orbit frame
    assert "inside a thick layer of white cloud" in keyframe_prompt("X", "default", "", {"over": "cloud"}, None)
    situation = {"over": "top", "above_ground_m": 1.7, "subject_distance_m": 4}
    assert "Image 2" in keyframe_prompt("X", "default", "", situation, {"photo": True, "description": ""})
    assert "a hiker" in keyframe_prompt("X", "default", "", situation, {"photo": False, "description": "a hiker"})
    assert "person" not in keyframe_prompt("X", "default", "", {"over": "top", "above_ground_m": 40}, None)


def test_flux_keyframes_and_ltx25_layout_to_render_are_closed_graphs():
    from geoblock_core.looks import keyframe_prompt
    from geoblock_core.workflow import LTX_VIDEO, ltx_graph
    flux = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                 preset="Earth zoom", subject_photo="me.png"))
    assert all(source in flux for _, _, (source, _) in links(flux))
    assert flux["61"]["inputs"]["type"] == "flux2" and "64" not in flux and "TextEncodeQwenImageEditPlus" not in {
        node["class_type"] for node in flux.values()}
    assert flux["74"]["inputs"]["batch_index"] == 240 and flux["74fguider"]["inputs"]["positive"] == ["74fpos3", 0]
    # klein gets the hold's first keyframe cropped to this frame's push-in framing as Image 2, the subject as Image 3
    crop = flux["74zcrop"]
    assert crop["class_type"] == "GeoBlockPushInCrop" and (crop["inputs"]["from_frame"], crop["inputs"]["to_frame"]) == (
        112, 240) and crop["inputs"]["image"] == ["240c", 0]
    assert flux["74fimg2"]["inputs"]["image"] == ["74zcrop", 0] and flux["74fimg3"]["inputs"]["image"] == ["172", 0]
    assert flux["81"]["inputs"]["reference"] is True and "240zcrop" not in flux and "248zcrop" in flux  # not on the hold's first keyframe
    # Every keyframe is colour-matched to its blocking frame before it is saved, pinned or cropped from.
    assert flux["74c"] == {"class_type": "GeoBlockColourMatch", "inputs": {
        "image": ["79", 0], "reference": ["74", 0], "strength": 0.6}}
    assert flux["92"]["inputs"]["images"] == ["74c", 0] and flux["80"]["inputs"]["image"] == ["74c", 0]
    assert flux["248zcrop"]["inputs"]["image"] == ["240c", 0]
    free = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                 ltx_version="2.5", preset="Earth zoom", pin_last_keyframe=False))
    pinned = [node["inputs"]["frame_idx"] for key, node in free.items()
              if node["class_type"] == "LTXVAddGuide" and key.isdigit() and 312 <= int(key) < 320]
    assert pinned == [8, 32, 56, 72, 88, 112, 176, 64] and "turns back toward the camera" in free["305"]["inputs"]["text"]
    assert free["380"]["class_type"] == "GeoBlockColourSmooth" and free["380"]["inputs"]["sigma_s"] == 0.75
    assert free["371"]["inputs"]["images"] == ["380", 0] and free["380"]["inputs"]["images"] == ["370", 0]
    # The descent keyframe (f56) is painted from the f72 keyframe below it, uncropped, with the descent sentence.
    assert free["223"]["inputs"]["reference"] is True and free["223"]["inputs"]["reference_kind"] == "descent"
    assert free["216fimg2"]["inputs"]["image"] == ["224c", 0] and "216zcrop" not in free
    assert "ground this camera reaches" in keyframe_prompt("X", "default", "", {"over": "top", "above_ground_m": 200,
                                                                              "reference": "descent"}, None)
    plain = ltx_graph(ShotRequest(name="X", latitude=0, longitude=0, colour_match=0))
    assert "380" not in plain  # rim reveal: no temporal smoothing by default
    assert not any(node["class_type"] == "GeoBlockColourMatch" for node in plain.values())
    assert flux["66fguider"]["inputs"]["positive"] == ["66fpos", 0]  # no subject on the orbit frame
    assert flux["115"]["class_type"] == "KSampler"  # still LTX 2.3 + 3DREAL

    both = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                 ltx_version="2.5", preset="Earth zoom"))
    assert all(source in both for _, _, (source, _) in links(both))
    assert "100" not in both and both["310"]["inputs"]["image"] == ["60", 0]
    assert both["310"]["inputs"]["latent_downscale_factor"] == ["301", 1]
    assert both["311"]["inputs"]["frame_idx"] == -1 and both["311"]["inputs"]["crop"] == "center"
    assert [both[str(312 + k)]["inputs"]["frame_idx"] for k in range(8)] == [8, 32, 56, 72, 88, 112, 176, 64]
    assert "320" not in both  # the Earth zoom leaves its last seconds to the layout (no pinned last frame)
    assert both["395"]["inputs"]["positive"] == ["319", 0]  # guides dropped after stage 1
    # Stage 2 starts again from the text conditioning and re-adds the guides at full size; a 10 s clip
    # leaves out the layout guide there (memory), a 5 s clip keeps it.
    assert "340" not in both and both["341"]["inputs"]["positive"] == ["307", 0]
    assert both["341"]["inputs"]["latent"] == ["331", 0] and both["341"]["inputs"]["latent_downscale_factor"] == 1.0
    assert "dive_tempo" not in both["4"]["inputs"]
    low = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                ltx_version="2.5", preset="Earth zoom", dive_tempo="low-swoop"))
    assert low["4"]["inputs"]["dive_tempo"] == "low-swoop"
    low_guides = [low[str(312 + k)]["inputs"]["frame_idx"] for k in range(8)]
    assert low_guides[0] < 8 or low_guides[1] < 32  # the blocking pin and cloud keyframe come sooner
    assert low_guides[2:] == [56, 72, 88, 112, 176, 64]
    short = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, ltx_version="2.5",
                                  duration_s=5))
    assert short["340"]["inputs"]["image"] == ["60", 0] and short["341"]["inputs"]["positive"] == ["340", 0]
    assert both["370"]["inputs"]["samples"] == ["365", 2] and both[LTX_VIDEO]["inputs"]["video"] == ["371", 0]
    assert not any("Audio" in node["class_type"] or "AV" in node["class_type"] for key, node in both.items()
                   if key.isdigit() and 300 <= int(key) < 380)
    assert not both["305"]["inputs"]["text"].startswith("3DREAL")
    with pytest.raises(ValueError):
        ltx_graph(ShotRequest(name="X", latitude=0, longitude=0, ltx_version="3"))


def test_subject_sheet_feeds_every_person_keyframe():
    from geoblock_core.workflow import SUBJECT_SHEET, ltx_graph
    base = dict(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                preset="Earth zoom")
    photo = ltx_graph(ShotRequest(**base, subject_photo="me.png"))
    # The sheet is a klein edit of the photo, prompted by the Subject node's sheet prompt, and saved for review
    assert photo["172text"]["inputs"]["text"] == ["170", 2] and photo["172scale"]["inputs"]["image"] == ["171", 0]
    assert photo["172guider"]["inputs"]["positive"] == ["172pos", 0] and photo[SUBJECT_SHEET]["class_type"] == "SaveImage"
    assert photo["170"]["inputs"]["subject_sheet"] is True and photo["170"]["inputs"]["weather"] == "clear"
    person = [key for key, node in photo.items() if (key.endswith("fimg2") or key.endswith("fimg3"))
              and photo[key]["inputs"]["image"] != ["224c", 0]]  # not the descent keyframe's ground reference
    assert person and all(photo[key]["inputs"]["image"] == ["172", 0] or photo[key]["inputs"]["image"][0].endswith(
        "zcrop") for key in person)
    assert not any(node["inputs"].get("image") == ["171", 0] for key, node in photo.items()
                   if key.endswith(("fimg2", "fimg3")))
    # Without a photo the sheet is painted from the description, and still pins the person
    invented = ltx_graph(ShotRequest(**base, weather="snow"))
    assert "171" not in invented and "172scale" not in invented and invented["172guider"]["inputs"]["positive"] == [
        "172text", 0]
    assert invented["74fimg3"]["inputs"]["image"] == ["172", 0]
    # Opting out, or Qwen, keeps the raw photo
    raw = ltx_graph(ShotRequest(**base, subject_photo="me.png", subject_sheet=False))
    assert "172" not in raw and raw["74fimg3"]["inputs"]["image"] == ["171", 0]
    qwen = ltx_graph(ShotRequest(**(base | {"keyframe_model": "qwen"}), subject_photo="me.png"))
    assert "172" not in qwen and qwen["170"]["inputs"]["subject_sheet"] is False
    # Keyframes only: the paint stage, no LTX
    quick = ltx_graph(ShotRequest(**base, keyframes_only=True))
    assert all(source in quick for _, _, (source, _) in links(quick))
    assert "90" not in quick and "100" not in quick and SUBJECT_SHEET in quick and "91" in quick


def test_weather_presets_and_prompts():
    from geoblock_core.looks import default_subject, keyframe_prompt, subject_sentence, subject_sheet_prompt
    from geoblock_core.weather import WEATHER, weather_prompt
    for name, weather in WEATHER.items():
        assert {"horizon", "zenith", "sun_energy", "sun_elevation", "haze", "deck", "prompt", "aloft"} <= set(weather)
    assert {"snow", "fog", "dusk", "night", "sandstorm"} <= set(WEATHER)
    # From orbit the weather is what is seen from above, not lightning on the far rim
    assert "lightning" in weather_prompt("thunderstorm") and "lightning" not in weather_prompt(
        "thunderstorm", {"over": "orbit"})
    assert weather_prompt("snow", {"over": "top"}) == WEATHER["snow"]["prompt"]
    assert weather_prompt("clear", {"over": "cloud"}) == "" and "pink" in weather_prompt("dusk", {"over": "cloud"})
    assert "puffer" in default_subject("snow") and default_subject("clear").startswith("a hiker in a light jacket")
    orbit = keyframe_prompt("Mather Point", "grand-canyon", weather_prompt("night", {"over": "orbit"}),
                            {"over": "orbit", "above_ground_m": 420000})
    assert "night side" in orbit
    # A subject sheet painted from a description is still "the person shown in" an image
    situation = {"subject_distance_m": 8.0, "reference": True}
    sheet = subject_sentence({"photo": False, "image": True, "description": "a woman in a red coat"}, situation)
    assert "the person shown in Image 3 (a woman in a red coat)" in sheet
    plain = subject_sentence({"photo": False, "image": False, "description": "a woman in a red coat"}, situation)
    assert "Image 3" not in plain and "natural relaxed pose: a woman in a red coat." in plain
    assert "Image 1 shows a person" in subject_sheet_prompt("", True)
    assert subject_sheet_prompt("a man in a blue shirt", False).endswith(
        "a man in a blue shirt. Sharp natural photographic detail, real skin and fabric.")


def test_night_and_dust_reach_the_top_of_the_earth_zoom():
    from geoblock_core.looks import keyframe_prompt, render_to_real_prompt
    from geoblock_core.weather import weather_prompt
    for weather, word in (("night", "night side"), ("sandstorm", "dust storm")):
        orbit = {"over": "orbit", "above_ground_m": 420000}
        prompt = keyframe_prompt("Mather Point", "grand-canyon", weather_prompt(weather, orbit), orbit, weather=weather)
        assert word in prompt and "420 km" in prompt and "white cloud systems, a thin blue" not in prompt.split(
            "black space")[0] or weather == "sandstorm"
        cloud = keyframe_prompt("X", "default", weather_prompt(weather, {"over": "cloud"}), {"over": "cloud"},
                                weather=weather)
        assert "thick layer of white cloud" not in cloud
        ltx = render_to_real_prompt("Mather Point", "grand-canyon", "", "", "Earth zoom", weather=weather)
        assert "whole way down" in ltx and "layer of white clouds" not in ltx
    assert "layer of white clouds" in render_to_real_prompt("X", "default", "", "", "Earth zoom", weather="clear")
    assert "thick layer of white cloud" in keyframe_prompt("X", "default", "", {"over": "cloud"}, weather="dusk")


def test_hold_keyframes_sharp_and_drop_and_no_glare():
    from geoblock_core.looks import keyframe_prompt
    from geoblock_core.workflow import ltx_graph
    base = dict(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                ltx_version="2.5", preset="Earth zoom", pin_last_keyframe=False)
    sharp = ltx_graph(ShotRequest(**base, hold_keyframes="sharp"))
    # f176 is repainted from the f112 keyframe upscaled by Real-ESRGAN, cropped, then scaled back to size
    assert sharp["248ux"]["class_type"] == "ImageUpscaleWithModel" and sharp["248ux"]["inputs"]["image"] == ["240c", 0]
    assert sharp["248zcrop"]["inputs"]["image"] == ["248ux", 0] and sharp["248zs"]["inputs"]["scale_by"] == 0.5
    assert sharp["248fimg2"]["inputs"]["image"] == ["248zs", 0] and "240ux" not in sharp
    drop = ltx_graph(ShotRequest(**base, hold_keyframes="drop"))
    pinned = [node["inputs"]["frame_idx"] for key, node in drop.items()
              if node["class_type"] == "LTXVAddGuide" and key.isdigit() and 312 <= int(key) < 320]
    assert pinned == [8, 32, 56, 72, 88, 112, 64]
    for over in ("orbit", "high", "cloud"):
        text = keyframe_prompt("X", "default", "", {"over": over, "above_ground_m": 5000}, None)
        assert "no lens flare" in text and "toward the sun" not in text
    assert "no lens flare" in ltx_graph(ShotRequest(**base))["305"]["inputs"]["text"]


def test_dense_hold_keyframes_every_32_frames():
    from geoblock_core.workflow import LTX_KEYFRAMES, ltx_graph
    dense = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", pin_last_keyframe=False,
                                  hold_keyframes="dense"))
    pinned = sorted(node["inputs"]["frame_idx"] for key, node in dense.items()
                    if node["class_type"] == "LTXVAddGuide" and key.isdigit() and 312 <= int(key) < 330)
    assert pinned == [8, 32, 56, 64, 72, 88, 112, 144, 176, 208]
    assert len(set(LTX_KEYFRAMES)) == len(LTX_KEYFRAMES)
    # every later hold keyframe is a sharpened crop of the f112 one
    crops = [node["inputs"] for node in dense.values() if node["class_type"] == "GeoBlockPushInCrop"]
    assert sorted(c["to_frame"] for c in crops) == [144, 176, 208, 240] and {c["from_frame"] for c in crops} == {112}


def test_direct_hold_keyframe_is_the_upscaled_crop_not_a_repaint():
    from geoblock_core.workflow import ltx_graph
    request = dict(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                   ltx_version="2.5", preset="Earth zoom", pin_last_keyframe=False)
    direct, sharp = ltx_graph(ShotRequest(**request, hold_keyframes="direct")), ltx_graph(ShotRequest(**request,
                                                                                                    hold_keyframes="sharp"))
    samplers = lambda g: sum(n["class_type"] == "SamplerCustomAdvanced" for n in g.values())  # noqa: E731
    assert samplers(direct) == samplers(sharp) - 1  # f176 is not repainted
    crop = next(n for n in direct.values() if n["class_type"] == "GeoBlockPushInCrop" and n["inputs"]["to_frame"] == 176)
    assert crop["inputs"]["from_frame"] == 112
    guide = next(n for n in direct.values() if n["class_type"] == "LTXVAddGuide" and n["inputs"]["frame_idx"] == 176)
    preprocess = direct[guide["inputs"]["image"][0]]
    assert preprocess["class_type"] == "LTXVPreprocess" and direct[preprocess["inputs"]["image"][0]]["class_type"] == "ImageScaleBy"


def test_orbit_from_blocking_pins_frame_zero_to_the_blocking():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", pin_last_keyframe=False,
                                  orbit_from_blocking=True))
    guides = {node["inputs"]["frame_idx"]: node["inputs"]["image"] for key, node in graph.items()
              if node["class_type"] == "LTXVAddGuide" and key.isdigit() and 312 <= int(key) < 330}
    assert guides[0] == ["199", 0] and graph["198"]["inputs"]["batch_index"] == 0
    assert graph["311"]["inputs"]["frame_idx"] == -1  # klein's f0 is still the look reference


def test_place_subject_pastes_the_person_into_eye_level_keyframes_only():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", pin_last_keyframe=False, place_subject=True))
    placed = sorted(node["inputs"]["frame"] for node in graph.values() if node["class_type"] == "GeoBlockPlaceSubject")
    assert placed == [112, 176, 240]
    prompts = {node["inputs"]["frame"]: node["inputs"].get("placed", False) for node in graph.values()
               if node["class_type"] == "GeoBlockKeyframePrompt"}
    assert prompts[112] and prompts[240] and not prompts[88]
    sheet = next(node for node in graph.values() if node["class_type"] == "GeoBlockSubject")
    assert sheet["inputs"]["chroma"]  # a green wall, so the person can be cut out


def test_keyframe_denoise_starts_klein_from_the_frame_itself_at_eye_level_only():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", keyframe_denoise=0.5))
    saved = {node["inputs"]["filename_prefix"].rsplit("_", 1)[1]: node["inputs"]["images"] for node in graph.values()
             if node["class_type"] == "SaveImage" and "_keyframe_" in node["inputs"]["filename_prefix"]}
    def start(frame):
        decoded = graph[graph[saved[frame][0]]["inputs"]["image"][0]]  # through the colour match
        sampler = graph[decoded["inputs"]["samples"][0]]
        return graph[sampler["inputs"]["latent_image"][0]]["class_type"], graph[sampler["inputs"]["sigmas"][0]]
    kind, split = start("0112")
    assert kind == "VAEEncode" and split["class_type"] == "SplitSigmas"
    assert graph[split["inputs"]["sigmas"][0]]["inputs"]["steps"] - split["inputs"]["step"] == 10
    assert start("0088")[0] == "EmptyFlux2LatentImage"  # higher frames still paint from scratch
    plain = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom"))
    assert not any(node["class_type"] == "SplitSigmas" for node in plain.values())


def test_keyframe_refine_repaints_eye_level_keyframes_twice():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", keyframe_denoise=0.8, keyframe_refine=0.6,
                                  place_subject=True))
    save = next(node for node in graph.values() if node["class_type"] == "SaveImage"
                and node["inputs"]["filename_prefix"].endswith("_keyframe_0176"))
    decoded = graph[graph[save["inputs"]["images"][0]]["inputs"]["image"][0]]  # through the colour match
    second = graph[decoded["inputs"]["samples"][0]]
    encoded = graph[second["inputs"]["latent_image"][0]]
    first = graph[graph[encoded["inputs"]["pixels"][0]]["inputs"]["samples"][0]]
    assert first["class_type"] == second["class_type"] == "SamplerCustomAdvanced"
    assert graph[first["inputs"]["latent_image"][0]]["class_type"] == "VAEEncode"
    placed = next(node for node in graph.values() if node["class_type"] == "GeoBlockPlaceSubject")
    assert placed["inputs"]["clean_plates"] == ["5", 5]


def test_composite_subject_paints_the_empty_plate_then_blends_the_pasted_person():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True))
    save = next(node for node in graph.values() if node["class_type"] == "SaveImage"
                and node["inputs"]["filename_prefix"].endswith("_keyframe_0176"))
    decoded = graph[graph[save["inputs"]["images"][0]]["inputs"]["image"][0]]  # through the colour match
    blend = graph[decoded["inputs"]["samples"][0]]
    split = graph[blend["inputs"]["sigmas"][0]]
    assert split["class_type"] == "SplitSigmas"
    assert graph[split["inputs"]["sigmas"][0]]["inputs"]["steps"] - split["inputs"]["step"] == 10  # 0.5 of 20
    masked = graph[blend["inputs"]["latent_image"][0]]
    assert masked["class_type"] == "SetLatentNoiseMask" and masked["inputs"]["mask"][1] == 1
    pasted = graph[graph[graph[masked["inputs"]["samples"][0]]["inputs"]["pixels"][0]]["inputs"]["image"][0]]
    assert pasted["class_type"] == "GeoBlockPlaceSubject" and pasted["inputs"]["erase"] is False
    plate = graph[graph[pasted["inputs"]["image"][0]]["inputs"]["samples"][0]]  # from the first plate's crop
    assert graph[plate["inputs"]["latent_image"][0]]["class_type"] == "VAEEncode"
    first = next(node for node in graph.values() if node["class_type"] == "SaveImage"
                 and node["inputs"]["filename_prefix"].endswith("_plate_0112"))
    sampler = graph[graph[first["inputs"]["images"][0]]["inputs"]["samples"][0]]
    assert graph[sampler["inputs"]["latent_image"][0]]["class_type"] == "EmptyFlux2LatentImage"  # painted afresh
    prompts = [node for node in graph.values() if node["class_type"] == "GeoBlockKeyframePrompt"
               and node["inputs"]["frame"] == 176]
    assert sorted("subject" in node["inputs"] for node in prompts) == [False, True]  # plate: nobody; blend: them
    assert [node["inputs"].get("plate", False) for node in prompts if "subject" not in node["inputs"]] == [True]
    plates = sorted(node["inputs"]["frame"] for node in graph.values() if node["class_type"] == "GeoBlockCleanPlate")
    assert plates == [88, 112, 176, 240]  # the descent keyframe before the hold sees the stand-in too
    f88 = next(node for node in graph.values() if node["class_type"] == "GeoBlockKeyframePrompt"
               and node["inputs"]["frame"] == 88 and "subject" not in node["inputs"])
    assert f88["inputs"]["reference"] and f88["inputs"].get("reference_kind") == "descent"  # painted from f112


def test_composite_repaints_the_closing_plate_from_the_first_plate_further():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True))
    plate = next(node for node in graph.values() if node["class_type"] == "SaveImage"
                 and node["inputs"]["filename_prefix"].endswith("_plate_0240"))
    sampler = graph[graph[plate["inputs"]["images"][0]]["inputs"]["samples"][0]]
    assert graph[sampler["inputs"]["latent_image"][0]]["class_type"] == "VAEEncode"
    split = graph[sampler["inputs"]["sigmas"][0]]
    assert graph[split["inputs"]["sigmas"][0]]["inputs"]["steps"] - split["inputs"]["step"] == 18  # 0.9 of 20


def test_cloud_keyframe_can_be_left_to_the_blocking():
    from geoblock_core.workflow import ltx_graph
    def frames(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", **kw))
        return sorted(node["inputs"]["frame"] for node in graph.values()
                      if node["class_type"] == "GeoBlockKeyframePrompt")
    assert 32 in frames() and 32 not in frames(cloud_keyframe="none") and 32 not in frames(cloud_keyframe="blocking")


def test_person_detector_repaints_people_off_the_plate_before_the_paste():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                  person_detector="/models/Florence-2-base"))
    pasted = next(node for node in graph.values() if node["class_type"] == "GeoBlockPlaceSubject"
                  and node["inputs"]["frame"] == 112)
    image, rounds = pasted["inputs"]["image"], 0
    while graph[image[0]]["class_type"] == "ImageCompositeMasked":  # each round pasted back through its mask
        paste = graph[image[0]]["inputs"]
        repaint = graph[graph[paste["source"][0]]["inputs"]["samples"][0]]
        masked = graph[repaint["inputs"]["latent_image"][0]]
        assert masked["class_type"] == "SetLatentNoiseMask" and masked["inputs"]["mask"] == paste["mask"]
        assert graph[paste["mask"][0]]["class_type"] == "GeoBlockFindPeople"
        # it starts from the person filled in from their surroundings, not from noise
        fill = graph[graph[masked["inputs"]["samples"][0]]["inputs"]["pixels"][0]]
        assert fill["class_type"] == "GeoBlockFillPeople" and fill["inputs"]["image"] == paste["destination"]
        split = graph[repaint["inputs"]["sigmas"][0]]
        assert split["class_type"] == "SplitSigmas" and split["inputs"]["step"] > 0
        image, rounds = paste["destination"], rounds + 1
    assert rounds == 2


def test_fill_people_fills_the_box_from_around_it():
    from geoblock_core.people import fill_people
    image = np.zeros((40, 40, 3))
    image[:, :, 1] = 0.5
    image[10:20, 10:20] = 1.0  # a bright person
    mask = np.zeros((40, 40), np.float32)
    mask[8:22, 8:22] = 1.0
    filled = fill_people(image, mask)
    assert np.allclose(filled[15, 15], [0, 0.5, 0], atol=0.01) and np.array_equal(filled[0, 0], image[0, 0])


def test_fill_people_is_smooth_not_blocks_and_skips_huge_masks():
    from geoblock_core.people import fill_people
    rows, cols = np.mgrid[0:120, 0:200] / 200.0
    image = np.dstack([rows, cols, 0.5 * (rows + cols)])  # sky-to-ground and left-to-right gradients
    image[30:100, 80:110] = [0.1, 0.1, 0.1]  # a dark walker
    mask = np.zeros((120, 200), np.float32)
    mask[20:110, 70:120] = 1.0
    filled = fill_people(image, mask)
    inside = filled[20:110, 70:120]
    assert np.abs(np.diff(inside, axis=0)).max() < 0.02 and np.abs(np.diff(inside, axis=1)).max() < 0.02  # r17k: blocks
    assert inside.mean() > 0.3  # the walker's dark is gone
    huge = np.ones((120, 200), np.float32)
    huge[:, :50] = 0
    assert np.array_equal(fill_people(image, huge), image)  # r17k: an oak called a person, 59% of the frame


def test_person_boxes_are_taller_than_wide_and_not_huge():
    from geoblock_core.people import _person_sized
    assert _person_sized((617, 411, 758, 767), 1360, 768)  # the r17k walker at f176
    assert not _person_sized((1, 0, 505, 559), 1360, 768)  # r17k's oak
    assert not _person_sized((100, 300, 400, 380), 1360, 768)  # wider than tall


def test_people_repaint_has_no_reference_image():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                  person_detector="/models/Florence-2-base"))
    fills = {node_id for node_id, node in graph.items() if node["class_type"] == "GeoBlockFillPeople"}
    encodes = {node_id for node_id, node in graph.items() if node["class_type"] == "VAEEncode"
               and node["inputs"]["pixels"][0] in fills}
    assert fills and not any(node["class_type"] == "ReferenceLatent" and node["inputs"]["latent"][0] in encodes
                             for node in graph.values())  # klein copied the fill's blocks back (r17k)


def test_people_mask_grows_each_box_more_below():
    from geoblock_core.people import people_mask
    mask = people_mask((100, 100), [(40, 40, 50, 60)], grow=0.25)
    assert mask[60, 45] == 1 and mask[69, 45] == 1 and mask[72, 45] == 0 and mask[34, 45] == 0


def test_plate_foreground_is_the_measured_ground_not_fixed_words():
    from geoblock_core.looks import subject_sentence
    from geoblock_core.workflow import ltx_graph
    plate = {"plate": True, "subject_distance_m": 20.0}
    road = subject_sentence(None, plate | {"foreground": "mostly paved paths and roads, with green grass"})
    assert "paved paths and roads" in road and "a road stays a road" in road and "soil" not in road
    assert "the ground Image 1 shows there" in subject_sentence(None, plate)
    from geoblock_core.looks import keyframe_prompt
    situation = plate | {"frame": 112, "over": "top", "above_ground_m": 1.7, "ground_m": 280.0}
    assert "except the person" not in keyframe_prompt("Twin Peaks", "default", "", situation)  # invited a walker
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                  foreground="mostly green grass"))
    plates = [node["inputs"] for node in graph.values() if node["class_type"] == "GeoBlockKeyframePrompt"
              and node["inputs"].get("plate")]
    assert plates and all(inputs["foreground"] == "mostly green grass" for inputs in plates)


def test_chroma_sheet_is_lit_like_the_landing():
    from geoblock_core.looks import subject_sheet_prompt
    sunny = subject_sheet_prompt("a hiker", False, chroma=True, weather="clear")
    assert "hard direct sunlight from the right" in sunny and "soft even" not in sunny and "no shadow on it" in sunny
    assert "cloudy sky" in subject_sheet_prompt("a hiker", False, chroma=True, weather="overcast")
    assert "soft even light" in subject_sheet_prompt("a hiker", False)  # the plain sheet stays a studio sheet


def test_swoop_keyframes_can_start_from_the_blocking_frame():
    from geoblock_core.workflow import ltx_graph
    def starts(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", **kw))
        out = {}
        for key, node in graph.items():
            if node["class_type"] == "GeoBlockKeyframePrompt" and key.isdigit():
                sampler = graph[str(int(key) - 3)]  # prompt at base+7, sampler at base+4
                out[node["inputs"]["frame"]] = graph[sampler["inputs"]["latent_image"][0]]["class_type"]
        return out
    plain, swoop = starts(), starts(swoop_denoise=0.85)
    assert set(plain.values()) == {"EmptyFlux2LatentImage"}
    assert {f for f, kind in swoop.items() if kind == "VAEEncode"} == {56, 64, 72}  # the swoop, not cloud or landing
    landing = starts(swoop_denoise=0.85, composite_subject=True)
    assert landing[88] == "VAEEncode" and starts(composite_subject=True)[88] == "EmptyFlux2LatentImage"


def test_stage1_can_be_saved_for_diagnosis():
    from geoblock_core.workflow import ltx_graph
    request = dict(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                   ltx_version="2.5", preset="Earth zoom")
    assert "377" not in ltx_graph(ShotRequest(**request))
    graph = ltx_graph(ShotRequest(**request, save_stage1=True))
    assert graph["377"]["inputs"]["filename_prefix"].endswith("_stage1")
    assert graph["375"]["inputs"]["samples"] == graph["331"]["inputs"]["samples"]  # the latent stage 2 upscales
    assert all(source in graph for _, _, (source, _) in links(graph))


def test_hold_keyframes_can_follow_by_reprojection():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                  hold_keyframes="sharp", hold_follow="reproject", ground_repaint=0.85))
    moves = [node["inputs"] for node in graph.values() if node["class_type"] == "GeoBlockReprojectHold"]
    assert sorted(m["to_frame"] for m in moves) == [176, 240] and all(m["from_frame"] == 112 for m in moves)
    # the stretched near ground is repainted through the reprojection's mask
    masks = [node["inputs"]["mask"] for node in graph.values() if node["class_type"] == "SetLatentNoiseMask"
             and graph[node["inputs"]["mask"][0]]["class_type"] == "GeoBlockReprojectHold"]
    assert len(masks) == 2 and all(m[1] == 1 for m in masks)
    assert not any(node["class_type"] in ("GeoBlockPushInCrop", "ImageUpscaleWithModel") for node in graph.values())
    assert all(source in graph for _, _, (source, _) in links(graph))
    chain = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                  hold_keyframes="dense", hold_follow="chain"))
    moves = sorted((node["inputs"]["from_frame"], node["inputs"]["to_frame"]) for node in chain.values()
                   if node["class_type"] == "GeoBlockReprojectHold")
    assert moves == [(112, 144), (144, 176), (176, 208), (208, 240)]
    assert all(source in chain for _, _, (source, _) in links(chain))


def test_deck_exit_keyframe_guides_the_cloud_break_out():
    from geoblock_core.workflow import ltx_graph

    def guides(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", cloud_keyframe="blocking", **kw))
        return graph, sorted({n["inputs"]["frame_idx"] for n in graph.values() if n["class_type"] == "LTXVAddGuide"})

    plain, frames = guides()
    assert 48 not in frames
    pinned, frames = guides(exit_keyframe="blocking")
    assert 48 in frames and 32 in frames  # on the cleared frame after the thinning (f41), not inside it
    painted, frames = guides(exit_keyframe="repaint", composite_subject=True, hold_keyframes="dense",
                             subject_description="a hiker")
    assert 48 in frames and {144, 176, 208} <= set(frames)
    assert all(source in painted for _, _, (source, _) in links(painted))
    low, frames = guides(exit_keyframe="blocking", dive_tempo="low-swoop")
    assert 16 in frames and 32 in frames  # the solid whiteout's middle and the cleared frame


def test_punch_through_sits_before_the_video_and_the_upscale():
    from geoblock_core.workflow import LTX_VIDEO, ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", punch_through=True))
    node = graph["381"]
    assert node["class_type"] == "GeoBlockPunchThrough" and node["inputs"]["images"] == ["380", 0]
    assert graph[graph[LTX_VIDEO]["inputs"]["video"][0]]["inputs"]["images"] == ["381", 0]
    assert node["inputs"]["start_frame"] < 32 < node["inputs"]["end_frame"] <= 48


def test_swoop_gap_fills_the_wide_guide_gaps_of_the_swoop():
    from geoblock_core.workflow import ltx_graph

    def guides(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", cloud_keyframe="blocking",
                                      exit_keyframe="repaint", composite_subject=True, hold_keyframes="dense",
                                      subject_description="a hiker", dive_tempo="low-swoop", **kw))
        return graph, sorted({n["inputs"]["frame_idx"] for n in graph.values() if n["class_type"] == "LTXVAddGuide"})

    plain, before = guides()
    filled, after = guides(swoop_gap=16)
    assert 48 not in before and 96 not in before
    assert set(after) - set(before) == {48}  # the r32 smear f40-56 (32 to 56); not 96 (r33: a near oak whip-pan)
    swoop = [f for f in after if 32 <= f <= 76]
    assert max(b - a for a, b in zip(swoop, swoop[1:])) <= 16
    assert all(source in filled for _, _, (source, _) in links(filled))
    assert not any(520 <= int(k) < 600 for k in plain if k.isdigit())


def test_stretch_grain_reaches_the_reprojection_nodes():
    from geoblock_core.workflow import ltx_graph

    def reprojections(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                      hold_keyframes="dense", hold_follow="reproject", subject_description="a hiker", **kw))
        return [n["inputs"] for n in graph.values() if n["class_type"] == "GeoBlockReprojectHold"]

    assert all("grain" not in inputs for inputs in reprojections())
    grained = reprojections(stretch_grain=0.08)
    assert grained and all(inputs["grain"] == 0.08 for inputs in grained)


def test_landing_plate_denoise_starts_only_the_descent_plate_from_its_own_latent():
    from geoblock_core.workflow import ltx_graph

    def starts(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                      hold_keyframes="dense", subject_description="a hiker", dive_tempo="low-swoop", **kw))
        return sum(1 for n in graph.values() if n["class_type"] == "EmptyFlux2LatentImage")

    blank = starts()
    assert starts(landing_plate_denoise=0.85) == blank - 1
    assert starts(swoop_denoise=0.85) < blank - 1


def test_landing_from_hold_paints_the_landing_plate_from_the_hold_plate_moved_back():
    from geoblock_core.workflow import ltx_graph

    def moves(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                      ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                      hold_keyframes="dense", hold_follow="reproject", subject_description="a hiker",
                                      dive_tempo="low-swoop", **kw))
        found = sorted((n["inputs"]["from_frame"], n["inputs"]["to_frame"]) for n in graph.values()
                       if n["class_type"] == "GeoBlockReprojectHold")
        return graph, found

    _, plain = moves()
    graph, landing = moves(landing_from_hold=True, landing_plate_denoise=0.8, stretch_grain=0.08)
    assert set(landing) - set(plain) == {(112, 88)}
    assert all(source in graph for _, _, (source, _) in links(graph))
    filled, _ = moves(landing_from_hold=True, landing_fill="blocking")
    node = next(n for n in filled.values() if n["class_type"] == "GeoBlockReprojectHold"
                and n["inputs"]["to_frame"] == 88)
    plate = filled[node["inputs"]["fill"][0]]
    assert plate["class_type"] == "GeoBlockCleanPlate" and plate["inputs"]["frame"] == 88
    assert all(source in filled for _, _, (source, _) in links(filled))


def test_hold_keep_pastes_the_moved_first_plate_back_over_each_later_hold_plate():
    from geoblock_core.workflow import ltx_graph

    def build(**kw):
        return ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                     ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                     hold_keyframes="dense", hold_follow="reproject", subject_description="a hiker",
                                     dive_tempo="smooth-swoop", **kw))

    assert not any(n["class_type"] == "ImageCompositeMasked" and n["inputs"]["mask"][1] == 2
                   for n in build().values())
    graph = build(hold_keep=0.9)
    holds = {name: n for name, n in graph.items() if n["class_type"] == "GeoBlockReprojectHold"}
    assert holds and all(n["inputs"]["keep"] == 0.9 for n in holds.values())
    pastes = [n for n in graph.values() if n["class_type"] == "ImageCompositeMasked"
              and n["inputs"]["mask"][0] in holds and n["inputs"]["mask"][1] == 2]
    assert len(pastes) == len(holds)  # every later hold plate, the closing close-up included
    assert all(p["inputs"]["source"] == [p["inputs"]["mask"][0], 0] for p in pastes)
    assert all(source in graph for _, _, (source, _) in links(graph))
    chained = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                    ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                    hold_keyframes="dense", hold_follow="chain", subject_description="a hiker",
                                    dive_tempo="smooth-swoop", hold_keep=1.0))
    steps = sorted((n["inputs"]["from_frame"], n["inputs"]["to_frame"]) for n in chained.values()
                   if n["class_type"] == "GeoBlockReprojectHold")
    assert all(b == c for (_, b), (c, _) in zip(steps, steps[1:]))  # each from the one before, kept on the way
    assert all(source in chained for _, _, (source, _) in links(chained))


def test_landing_repaint_and_hold_near_repaint_wire_the_reprojection_masks():
    from geoblock_core.workflow import ltx_graph
    graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11, keyframe_model="flux-klein",
                                  ltx_version="2.5", preset="Earth zoom", composite_subject=True,
                                  hold_keyframes="dense", hold_follow="reproject", subject_description="a hiker",
                                  dive_tempo="smooth-swoop", landing_from_hold=True, landing_fill="blocking",
                                  landing_repaint=0.95, hold_near_repaint=8.0, ground_repaint=0.85))
    holds = {name: n for name, n in graph.items() if n["class_type"] == "GeoBlockReprojectHold"}
    landing = next(name for name, n in holds.items() if n["inputs"]["to_frame"] == 88)
    assert "near_m" not in holds[landing]["inputs"]
    assert all(n["inputs"]["near_m"] == 8.0 for name, n in holds.items() if name != landing)
    masked = [n for n in graph.values() if n["class_type"] == "SetLatentNoiseMask" and n["inputs"]["mask"] == [landing, 3]]
    assert len(masked) == 1
    assert all(source in graph for _, _, (source, _) in links(graph))


def test_exit_denoise_sets_how_far_the_deck_exit_keyframe_is_repainted():
    from geoblock_core.workflow import ltx_graph

    def split_steps(**kw):
        graph = ltx_graph(ShotRequest(name="Mather Point", latitude=36.06, longitude=-112.11,
                                      keyframe_model="flux-klein", ltx_version="2.5", preset="Earth zoom",
                                      dive_tempo="smooth-swoop", exit_keyframe="repaint", **kw))
        return sorted(n["inputs"]["step"] for n in graph.values() if n["class_type"] == "SplitSigmas")

    plain, higher = split_steps(), split_steps(exit_denoise=0.85)
    assert plain != higher and sum(higher) < sum(plain)  # fewer schedule steps skipped: repainted further
