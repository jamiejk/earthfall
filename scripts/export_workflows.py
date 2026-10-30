"""Export the pipeline's ComfyUI graphs as API-format workflow JSON.

    uv run python scripts/export_workflows.py
    uv run python scripts/export_workflows.py --out examples/comfyui-workflows

The exported files are the graphs the pipeline actually submits. They are built by calling the code that
builds them (``geoblock_core.workflow``, ``touchdown.comfy_job``, ``fakeout_earth_zoom``) with a stub in
place of the ComfyUI server, so an exported graph cannot drift from what a run would send. The stub only
replaces the network: the SDK's asset upload becomes the file name the server would need in its ``input/``
folder, and the submit is captured instead of sent.

The output is ComfyUI's "Save (API Format)" shape: a dict of node id -> ``{"class_type", "inputs", "_meta"}``.
The ComfyUI frontend opens that shape directly, and ``POST /prompt`` takes it unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fakeout_earth_zoom as fakeout  # noqa: E402
from geoblock_core.workflow import ShotRequest, blocking_graph, ltx_graph  # noqa: E402
from touchdown import comfy_job, flux3  # noqa: E402

DEFAULT_OUT = ROOT / "examples/comfyui-workflows"

# The place the pipeline is built around: Mather Point, Grand Canyon South Rim.
PLACE = {"latitude": 36.0617, "longitude": -112.1078}
# The prompt fields the Touchdown manifests carry ([shot] person/scene/region/approach).
WORDS = {
    "person": "a young woman with short curly dark hair, wearing a long red raincoat over a grey hoodie, "
              "black jeans and white trainers",
    "scene": "on the South Rim of the Grand Canyon at Mather Point, with the canyon and its distant buttes "
             "behind her",
    "region": "the desert of northern Arizona and southern Utah",
    "approach": "the rim of the Grand Canyon, looking forward and down across scrub and rock toward the canyon",
    "ambient": "thin dust drifts across the rim and the clouds move slowly over the canyon",
}
END_PROMPT = ("Make a real eye-level photograph from exactly the same camera viewpoint and compass direction "
              "as image 1. Keep the outlines and positions of the distant buttes and mesas, the horizon and the "
              "sky where image 1 has them; do not rotate or move them. Draw every butte, mesa and slope as real "
              "weathered sandstone with horizontal rock layers and desert varnish. In the centre, about ten "
              "metres from the camera, add {person}, standing still facing the camera, shown full length. The "
              "near ground is natural rock and sparse desert scrub. Natural daylight and realistic human scale. "
              "No text.")
OPENER_END_PROMPT = ("A real photograph taken at eye level about fifteen metres away: {person} stands {scene}, "
                     "seen full length, small in the middle of the frame, facing the camera, on the rim. The wide "
                     "canyon view fills most of the frame. Natural daylight, a few soft clouds, 35mm lens, sharp, "
                     "real photo, no text.")
SWOOP_PROMPT = ("One continuous shot with no cuts. The camera is already swooping forward and down over "
                "{approach}, about a hundred and seventy metres up, looking forward and down. It flies straight "
                "ahead, tilting up toward the horizon in one smooth, quick, steady move as it descends, levels "
                "out at eye level, and glides toward {person} standing {scene}, who stays still and looks at the "
                "camera. Steady camera, smooth continuous motion, real aerial photography, natural colour, sharp "
                "photographic detail, photorealistic.")
FACE_PROMPT = ("Turn the blurry approach crop into a sharp, detailed close-up photograph of the same woman's "
               "face, at the same angle and with the same expression. Keep her identity, hair and clothing. "
               "Real skin texture, sharp eyes, no smoothing, no plastic skin. Natural daylight, no text.")


@dataclass
class Exported:
    """One exported graph: the file, what it is, and what a server needs to run it."""
    file: str
    title: str
    stage: str
    description: str
    source: str
    graph: dict
    comfyui: str
    custom_nodes: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    notes: str = ""


class _StubAssets:
    """Stands in for the SDK's asset uploader: the file name is what the server needs in its input folder."""

    def from_file(self, path) -> str:
        return Path(path).name


class _StubClient:
    def __init__(self):
        self.assets = _StubAssets()


@contextmanager
def _stub_client(url):
    yield _StubClient()


@contextmanager
def captured():
    """Capture the graph a job would submit, instead of submitting it. Returns the captured graphs."""
    graphs: list[dict] = []
    real_client, real_submit = fakeout._client, fakeout._submit

    def submit(client, graph, wanted, destination):
        graphs.append(graph)
        return {node: f"{node}.png" for node in wanted}

    fakeout._client = _stub_client
    fakeout._submit = submit
    try:
        yield graphs
    finally:
        fakeout._client, fakeout._submit = real_client, real_submit


def _request(**kw) -> ShotRequest:
    """The GeoBlock library's request for the Touchdown shot: the Earth zoom, LTX 2.5, klein keyframes.
    ``pin_last_keyframe`` is the library's ``--pin-ending``: the Earth zoom otherwise leaves the last
    seconds to the layout and its last keyframe is painted and saved, but never pinned."""
    return ShotRequest(name="touchdown", duration_s=10, preset="Earth zoom", keyframe_model="flux-klein",
                        ltx_version="2.5", subject_description="a hiker in a light jacket",
                        pin_last_keyframe=True, **PLACE, **kw)


def _geoblock_blocking() -> Exported:
    """The measured-terrain blocking graph: place -> landform scene -> camera preset -> Blender render."""
    request = _request()
    return Exported(
        file="geoblock-blocking-earth-zoom.json",
        title="GeoBlock blocking (Earth zoom)",
        stage="terrain",
        description="The measured part of the shot: the place becomes a landform scene from survey data, the "
                    "camera preset plans the route over it, and Blender renders the blocking video the "
                    "generative stages follow.",
        source="geoblock_core.workflow.blocking_graph",
        graph=blocking_graph(request),
        comfyui="v0.4+ with the GeoBlock nodes (comfy_geoblock_prototype/) and Blender 4.2+ on PATH",
        custom_nodes=["GeoBlock nodes (comfy_geoblock_prototype/ in this repository)"],
        models=[],
        assets=[],
        notes="The blocking graph needs no models: it renders terrain, not pictures.")


def _geoblock_ltx() -> Exported:
    """The whole GeoBlock graph: blocking plus photoreal keyframes and the LTX 2.5 render-to-real stage."""
    return Exported(
        file="geoblock-ltx25-earth-zoom.json",
        title="GeoBlock render-to-real (LTX 2.5)",
        stage="whole shot",
        description="The library's single-graph version of the shot: the blocking video, the photoreal "
                    "keyframes painted from it, and LTX 2.5 with Lightricks' Layout To Render IC-LoRA "
                    "following the blocking. One graph, one submit.",
        source="geoblock_core.workflow.ltx_graph",
        graph=ltx_graph(_request()),
        comfyui="v0.32+ with ComfyUI-LTXVideo, and the GeoBlock nodes for the blocking half",
        custom_nodes=["GeoBlock nodes (comfy_geoblock_prototype/ in this repository)",
                      "ComfyUI-LTXVideo"],
        models=[],
        assets=[],
        notes="The Touchdown build does not submit this graph: it splits the work into stages so each one can "
              "be cached, checked and picked. It is here to show the same models wired as one graph.")


def _touchdown_stills(flux3_stills: list[str], name: str, title: str, description: str) -> Exported:
    """The stills stage: the end, orbit, cloud and opener-end stills, painted by FLUX in one submit."""
    with captured() as graphs:
        comfy_job.stills(url="http://127.0.0.1:8191", folder="outputs/touchdown/touchdown/stills", seed=20261009,
                         name="touchdown", words=WORDS, flux3=flux3_stills,
                         end_reference="guide_frame_0081.png", subject="subject.jpg", scene_reference="scene.jpg",
                         end_prompt=END_PROMPT.format(**WORDS), opener_end_prompt=OPENER_END_PROMPT.format(**WORDS))
    graph = graphs[0]
    return Exported(
        file=name,
        title=title,
        stage="stills",
        description=description,
        source="touchdown.comfy_job.stills",
        graph=graph,
        comfyui="v0.41.3+ for the FLUX 3 stills; v0.32+ for the FLUX.2 ones",
        custom_nodes=["ComfyUI core (FLUX.2 / FLUX 3 nodes)"],
        models=[],
        assets=["guide_frame_0081.png", "subject.jpg", "scene.jpg"],
        notes="One submit paints all the stills; the four SaveImage nodes are the end, the orbit, the cloud and "
              "the opener end.")


def _touchdown_flux3_still() -> Exported:
    """The FLUX 3 still on its own, as scripts/touchdown/flux3.py builds it (plain HTTP, no SDK)."""
    graph = flux3.graph(END_PROMPT.format(**WORDS), 20261009, ["guide_frame_0081.png", "subject.jpg"],
                        "geoblock/touchdown_end")
    return Exported(
        file="touchdown-flux3-still.json",
        title="FLUX 3 Image Klein still",
        stage="stills",
        description="One FLUX 3 Image Klein still at 1344x768 with two reference images, the same graph "
                    "scripts/touchdown/flux3.py submits to a plain ComfyUI HTTP API (no SDK proxy).",
        source="touchdown.flux3.graph",
        graph=graph,
        comfyui="v0.41.3+ (FLUX 3 Image Klein)",
        custom_nodes=["ComfyUI core (FLUX 3 nodes)"],
        models=[],
        assets=["guide_frame_0081.png", "subject.jpg"])


def _touchdown_ltx_clip() -> Exported:
    """The LTX 2.5 clip: the opener, the cloud passage, the swoop and the living end are all this graph."""
    with captured() as graphs:
        comfy_job.ltx(url="http://127.0.0.1:8191", folder="outputs/touchdown/touchdown/opener", frames=241,
                      pins=[[0, "orbit.png"], [32, "cloud.png"], [-1, "opener_end.png"]],
                      prompt=fakeout.ZOOM_IN.format(**WORDS), seed=20261009,
                      prefix="geoblock/touchdown_opener_s20261009", ltx_upscale=True)
    graph = graphs[0]
    return Exported(
        file="touchdown-ltx25-clip.json",
        title="Touchdown LTX 2.5 clip",
        stage="opener, cloud, swoop, end",
        description="Plain LTX 2.5 keyframe image-to-video with the stills pinned as guides: the orbit still at "
                    "frame 0, the cloud still in the whiteout, the end still at the last frame. The opener, the "
                    "cloud passage, the swoop and the living end are this graph with different pins.",
        source="touchdown.comfy_job.ltx",
        graph=graph,
        comfyui="v0.32+ with ComfyUI-LTXVideo",
        custom_nodes=["ComfyUI-LTXVideo"],
        models=[],
        assets=["orbit.png", "cloud.png", "opener_end.png"])


def _touchdown_terrain_v2v() -> Exported:
    """The measured-terrain swoop: the guide video in, the LTX 2.5 Layout To Render out."""
    with captured() as graphs:
        comfy_job.terrain_v2v(url="http://127.0.0.1:8191", guide="guide.mp4", first="guide_frame_0001.png",
                              last="end.png", output="outputs/touchdown/touchdown/swoop", seed=20261009, frames=81,
                              prompt=SWOOP_PROMPT.format(**WORDS), ltx_upscale=True)
    graph = graphs[0]
    return Exported(
        file="touchdown-terrain-v2v.json",
        title="Touchdown terrain guide to LTX 2.5",
        stage="swoop",
        description="The measured route as a guide: the Blender guide video goes in as the Layout To Render "
                    "IC-LoRA guide, the first frame as a one-frame reference, the end still is pinned, and LTX "
                    "2.5 renders the approach over the real terrain.",
        source="touchdown.comfy_job.terrain_v2v (touchdown.terrain_v2v.render)",
        graph=graph,
        comfyui="v0.32+ with ComfyUI-LTXVideo",
        custom_nodes=["ComfyUI-LTXVideo"],
        models=[],
        assets=["guide.mp4", "guide_frame_0001.png", "end.png"])


def _touchdown_face_plate() -> Exported:
    """The face stage: one klein edit of a late-approach crop into a detailed face reference."""
    with captured() as graphs:
        comfy_job.face_plate(url="http://127.0.0.1:8191", folder="outputs/touchdown/touchdown/face",
                             reference="approach_frame.png", seed=20261009, prompt=FACE_PROMPT)
    graph = graphs[0]
    return Exported(
        file="touchdown-face-plate.json",
        title="Touchdown face plate",
        stage="face",
        description="The optional face stage: the blurry face in the late approach is edited into a detailed, "
                    "square face reference, which is then tracked onto the living end.",
        source="touchdown.comfy_job.face_plate",
        graph=graph,
        comfyui="v0.32+ (FLUX.2 klein nodes are core)",
        custom_nodes=["ComfyUI core (FLUX.2 nodes)"],
        models=[],
        assets=["approach_frame.png"])


def build() -> list[Exported]:
    """Every graph the pipeline submits, built by calling the pipeline's own code."""
    return [
        _geoblock_blocking(),
        _geoblock_ltx(),
        _touchdown_stills(["end", "orbit", "cloud", "opener_end"], "touchdown-stills-flux3.json",
                          "Touchdown stills (FLUX 3)",
                          "The stills stage with FLUX 3 Image Klein at 2K for all four stills: the end still "
                          "painted from the guide's last frame and the person's photo, the orbit still, the "
                          "whiteout still and the separate opener end pin."),
        _touchdown_stills([], "touchdown-stills-klein.json", "Touchdown stills (FLUX.2 klein)",
                          "The same stills stage with FLUX.2 [klein] 9B distilled (4 steps, CFG 1) for all four "
                          "stills. This is the default: FLUX 3 needs ComfyUI v0.41.3+, and klein kept the "
                          "flat whiteout better."),
        _touchdown_flux3_still(),
        _touchdown_ltx_clip(),
        _touchdown_terrain_v2v(),
        _touchdown_face_plate(),
    ]


NODE_ROLES = {
    "SaveImage": "save image",
    "SaveVideo": "save video",
    "LoadImage": "load image",
    "LoadVideo": "load video",
    "UNETLoader": "model",
    "VAELoader": "VAE",
    "CLIPLoader": "text encoder",
    "CLIPTextEncode": "prompt",
    "KSampler": "sample",
    "SamplerCustomAdvanced": "sample",
    "VAEDecode": "decode",
    "VAEDecodeTiled": "decode",
    "GetVideoComponents": "video",
    "LTXVAddGuide": "guide",
    "LTXAddVideoICLoRAGuide": "guide",
    "LTXVPreprocess": "preprocess",
    "LTXVLatentUpsampler": "upscale",
    "LatentUpscaleModelLoader": "upscale model",
    "LTXICLoRALoaderModelOnly": "IC-LoRA",
    "EmptyLTXVLatentVideo": "latent",
    "EmptyFlux2LatentImage": "latent",
    "LTXVConditioning": "conditioning",
    "CFGGuider": "guide",
    "ReferenceLatent": "reference",
    "ReferenceMultiLatent": "reference",
    "ImageScale": "scale",
    "ImageScaleBy": "scale",
    "ImageScaleToTotalPixels": "scale",
    "GeoBlockSubject": "subject",
    "GeoBlockKeyframePrompt": "keyframe prompt",
    "GeoBlockColourMatch": "colour match",
    "GeoBlockColourSmooth": "colour smooth",
    "GeoBlockPushInCrop": "push-in crop",
    "GeoBlockRenderBlockingVideo": "Blender render",
    "GeoBlockCameraPreset": "camera preset",
    "GeoBlockBuildLandformScene": "landform scene",
    "GeoBlockPlaceFromCoordinates": "place",
}
MODEL_INPUTS = ("unet_name", "vae_name", "clip_name", "ckpt_name", "lora_name", "model_name", "text_encoder")


def node_titles(graph: dict, title: str) -> dict:
    """Give every node a title, as ComfyUI's own "Save (API Format)" does. The title names the graph and
    what the node does, so a file opened on its own says what it is."""
    for node in graph.values():
        role = NODE_ROLES.get(node["class_type"], node["class_type"])
        node["_meta"] = {"title": f"{title} · {role}"}
    return graph


def models_of(graph: dict) -> list[str]:
    """The model files the graph asks for, read out of the graph itself."""
    found = []
    for node in graph.values():
        for name, value in node["inputs"].items():
            if name in MODEL_INPUTS and isinstance(value, str) and value and value not in found:
                found.append(value)
    return sorted(found)


def custom_nodes_of(graph: dict) -> list[str]:
    """The custom node packs the graph needs, from the class types it uses."""
    types = {node["class_type"] for node in graph.values()}
    packs = []
    if any(t.startswith("GeoBlock") for t in types):
        packs.append("GeoBlock nodes (comfy_geoblock_prototype/ in this repository)")
    if any(t.startswith("LTX") for t in types):
        packs.append("ComfyUI-LTXVideo")
    return packs


def export(out: Path = DEFAULT_OUT) -> list[Exported]:
    """Write every graph as API-format JSON, with an index that says what each file is."""
    out.mkdir(parents=True, exist_ok=True)
    exported = build()
    for item in exported:
        item.models = models_of(item.graph)
        item.custom_nodes = item.custom_nodes or custom_nodes_of(item.graph)
        graph = node_titles(item.graph, item.title)
        (out / item.file).write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
    index = {
        "format": "ComfyUI API format (the shape of the frontend's \"Save (API Format)\")",
        "generated_by": "scripts/export_workflows.py",
        "comfyui": {
            "blocking": "v0.4+ with the GeoBlock nodes and Blender 4.2+",
            "ltx25": "v0.32+ with ComfyUI-LTXVideo",
            "flux3": "v0.41.3+ (FLUX 3 Image Klein)",
        },
        "graphs": [{
            "file": item.file,
            "title": item.title,
            "stage": item.stage,
            "description": item.description,
            "built_by": item.source,
            "nodes": len(item.graph),
            "comfyui": item.comfyui,
            "custom_nodes": item.custom_nodes,
            "models": item.models,
            "assets": item.assets,
            "notes": item.notes,
        } for item in exported],
    }
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the graphs "
                        "(default: examples/comfyui-workflows/)")
    args = parser.parse_args()
    for item in export(args.out):
        print(f"{item.file}: {len(item.graph)} nodes ({item.stage})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
