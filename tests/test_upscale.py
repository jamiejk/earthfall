import json
import shutil
import subprocess

import pytest

from geoblock_core.run import main
from geoblock_core.upscale import FINAL_VIDEO, UPSCALERS, add_final_upscale, prescale, upscale_file_graph
from geoblock_core.workflow import LTX_VIDEO, ShotRequest, ltx_graph


def closed(graph):
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                assert value[0] in graph, value


@pytest.mark.parametrize("method", UPSCALERS)
def test_every_upscaler_builds_a_closed_graph(method):
    graph = upscale_file_graph("clip.mp4", method=method, target="4k", prefix="t", start=120, frames=121)
    closed(graph)
    assert graph["403"]["inputs"] == {"image": ["402", 0], "batch_index": 120, "length": 121}
    assert graph["400"]["class_type"] == "GeoBlockFinalUpscalePrep"
    assert graph[FINAL_VIDEO]["class_type"] == "SaveVideo"


def test_seedvr2_takes_the_output_height_from_the_prep_node():
    graph = upscale_file_graph("clip.mp4", method="seedvr2-7b", target="1080p", prefix="t")
    upscaler = graph["432"]["inputs"]
    assert upscaler["resolution"] == ["400", 1] and upscaler["image"] == ["400", 0]
    assert upscaler["batch_size"] % 4 == 1 and upscaler["temporal_overlap"] < upscaler["batch_size"]
    assert graph["430"]["inputs"]["model"].startswith("seedvr2_ema_7b")


def test_half_size_rebuild_only_where_it_applies():
    assert prescale("seedvr2", "1080p") == 0.5 and prescale("seedvr2", "4k") == 1
    assert prescale("flashvsr", "4k") == 0.5 and prescale("ltx-detail", "1080p") == 1
    with pytest.raises(ValueError, match="flashvsr"):
        add_final_upscale({}, ["1", 0], method="flashvsr", target="1080p", prefix="t")


def test_ltx_graph_upscales_the_decoded_frames_before_saving():
    request = ShotRequest(name="Rim", latitude=36.06, longitude=-112.11, ltx_version="2.5",
                          keyframe_model="flux-klein", final_upscale="seedvr2", final_target="4k")
    graph = ltx_graph(request)
    closed(graph)
    decoded = graph[graph[LTX_VIDEO]["inputs"]["video"][0]]["inputs"]["images"]
    assert graph["400"]["inputs"]["images"] == decoded
    assert graph[FINAL_VIDEO]["inputs"]["filename_prefix"].endswith("_final_seedvr2_4k")
    assert FINAL_VIDEO not in ltx_graph(ShotRequest(name="Rim", latitude=36.06, longitude=-112.11))


def test_ltx_detail_follows_lightricks_tiled_upscale():
    graph = upscale_file_graph("clip.mp4", method="ltx-detail", target="4k", prefix="t")
    assert graph["450"]["inputs"]["output_size"] == "4K" and graph["450"]["inputs"]["tile_size"] == "HD"
    assert graph["456"]["inputs"]["use_streaming"] is True
    assert graph["459"]["inputs"]["latents"] == ["456", 2]


def test_cli_upscale_video_saves_graph(tmp_path):
    path = tmp_path / "g.json"
    assert main(["--upscale-video", "shot.mp4", "--final-upscale", "esrgan", "--final-target", "1080p",
                 "--save-graph", str(path)]) == 0
    graph = json.loads(path.read_text())
    assert graph["401"]["inputs"]["file"] == "shot.mp4" and graph["410"]["class_type"] == "UpscaleModelLoader"


def test_cli_final_upscale_needs_ltx(tmp_path):
    with pytest.raises(SystemExit):
        main(["--latitude", "36", "--longitude", "-112", "--final-upscale", "seedvr2",
              "--save-graph", str(tmp_path / "g.json")])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_chunked_upscale_parts_join_back_to_every_frame(tmp_path):
    from geoblock_core.run import _frame_count, _join_videos
    parts = []
    for n, frames in enumerate((64, 64, 17)):
        part = tmp_path / f"part{n}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=24",
                        "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(part)], check=True)
        parts.append(part)
    joined = tmp_path / "joined.mp4"
    _join_videos(parts, joined)
    assert _frame_count(joined) == 145 and not joined.with_suffix(".txt").exists()
