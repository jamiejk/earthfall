"""The exported graphs are real ComfyUI graphs, and the files in the repository are the ones the export
writes today. Nothing here needs a ComfyUI server, a GPU or a network: the export builds the graphs with
the pipeline's own code and a stub in place of the server.
"""
import json
from pathlib import Path
import socket

import pytest

import export_workflows
from export_workflows import export

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/comfyui-workflows"
# The node classes that end a branch: their output goes nowhere on purpose.
OUTPUT_CLASSES = {"SaveImage", "SaveVideo", "SaveAudio", "PreviewImage", "PreviewAudio"}


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a connection fails the test: the export must not talk to a server."""
    def blocked(*args, **kwargs):
        raise AssertionError("the export must not touch the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def links(node) -> list:
    """The node references in one node: ["<node id>", <output index>]."""
    return [value for value in node["inputs"].values()
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)]


def graphs_of(folder: Path) -> dict:
    """The exported graphs, by file name."""
    return {path.name: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))
            if path.name != "index.json"}


def test_export_writes_a_file_per_graph_and_an_index(tmp_path, no_network):
    exported = export(tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    assert {item["file"] for item in index["graphs"]} == {item.file for item in exported}
    for item in exported:
        assert (tmp_path / item.file).is_file()
        assert item.stage and item.description and item.source
    assert index["graphs"] and all(item["nodes"] > 0 for item in index["graphs"])
    assert all(item["comfyui"] and item["custom_nodes"] for item in index["graphs"])


def test_every_node_has_a_class_type_and_its_links_point_at_nodes(tmp_path):
    export(tmp_path)
    graphs = graphs_of(tmp_path)
    assert graphs
    for name, graph in graphs.items():
        for node_id, node in graph.items():
            assert node.get("class_type"), (name, node_id)
            assert isinstance(node.get("inputs"), dict), (name, node_id)
            for link in links(node):
                assert link[0] in graph, (name, node_id, link)


def test_no_node_is_orphaned(tmp_path):
    """Every node is used by another node or ends a branch on purpose (a save)."""
    export(tmp_path)
    for name, graph in graphs_of(tmp_path).items():
        referenced = {link[0] for node in graph.values() for link in links(node)}
        for node_id, node in graph.items():
            if node_id in referenced:
                continue
            assert node["class_type"] in OUTPUT_CLASSES, (name, node_id, node["class_type"])


def test_the_exported_graphs_are_the_ones_the_pipeline_builds(tmp_path):
    """The graphs are the pipeline's own: the stills stage, the LTX clip, the measured-terrain swoop."""
    export(tmp_path)
    graphs = graphs_of(tmp_path)

    stills = graphs["touchdown-stills-flux3.json"]
    saves = [node for node in stills.values() if node["class_type"] == "SaveImage"]
    # the end, the orbit, the cloud, the opener end, and the end still again at its full 2K size
    assert len(saves) == 5
    assert any(node["inputs"]["filename_prefix"].endswith("_end_full") for node in saves)
    assert stills["f3u"]["inputs"]["unet_name"].startswith("flux3")
    assert stills["42"]["inputs"]["image"] == "guide_frame_0081.png"  # the guide's last frame
    assert stills["41"]["inputs"]["image"] == "subject.jpg"  # the person's photo
    assert all(node["class_type"] != "UNETLoader" or node["inputs"]["unet_name"].startswith("flux-2")
               for node in graphs["touchdown-stills-klein.json"].values())

    clip = graphs["touchdown-ltx25-clip.json"]
    assert clip["44"]["inputs"]["image"] == "orbit.png" and clip["46"]["inputs"]["image"] == "opener_end.png"
    assert [node["inputs"]["frame_idx"] for node in clip.values() if node["class_type"] == "LTXVAddGuide"
            and node["inputs"]["frame_idx"] == 32] == [32, 32]  # the cloud still, pinned on both stages
    assert clip["90"]["class_type"] == "SaveVideo"

    swoop = graphs["touchdown-terrain-v2v.json"]
    assert swoop["50"]["class_type"] == "LoadVideo" and swoop["50"]["inputs"]["file"] == "guide.mp4"
    assert swoop["301"]["class_type"] == "LTXICLoRALoaderModelOnly"  # the Layout To Render IC-LoRA
    assert swoop["310"]["class_type"] == "LTXAddVideoICLoRAGuide"

    blocking = graphs["geoblock-blocking-earth-zoom.json"]
    assert blocking["5"]["class_type"] == "GeoBlockRenderBlockingVideo"
    assert blocking["4"]["inputs"]["preset"] == "Earth zoom" and blocking["4"]["inputs"]["duration_seconds"] == 10
    assert blocking["22"]["class_type"] == "SaveImage"  # the opening frame

    library = graphs["geoblock-ltx25-earth-zoom.json"]
    assert library["60"]["class_type"] == "GetVideoComponents"  # the blocking video
    assert library["300"]["class_type"] == "UNETLoader" and library["301"]["class_type"] == "LTXICLoRALoaderModelOnly"
    assert any(node["class_type"] == "GeoBlockKeyframePrompt" for node in library.values())


def test_the_committed_graphs_are_what_the_export_writes_today(tmp_path, monkeypatch, no_network):
    """The graphs in the repository are regenerated from the pipeline's code, not hand-edited."""
    for name in ("TOUCHDOWN_FLUX3_UNET", "TOUCHDOWN_FLUX3_CLIP", "TOUCHDOWN_FLUX3_VAE"):
        monkeypatch.delenv(name, raising=False)
    export(tmp_path)
    committed = graphs_of(EXAMPLES)
    assert committed, f"no exported graphs in {EXAMPLES}"
    assert committed == graphs_of(tmp_path)


def test_the_export_uses_the_pipeline_code():
    """Every graph is built by a function of the pipeline, not by a copy of it in the export script."""
    sources = {item.source for item in export_workflows.build()}
    assert sources == {
        "geoblock_core.workflow.blocking_graph",
        "geoblock_core.workflow.ltx_graph",
        "touchdown.comfy_job.stills",
        "touchdown.comfy_job.ltx",
        "touchdown.comfy_job.terrain_v2v (touchdown.terrain_v2v.render)",
        "touchdown.comfy_job.face_plate",
        "touchdown.flux3.graph",
    }
