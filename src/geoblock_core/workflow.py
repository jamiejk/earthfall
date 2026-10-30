"""API-format ComfyUI graphs for the GeoBlock pipeline.

The same graph runs in a local ComfyUI (through comfy-api-proxy) or on Comfy
Cloud via the Comfy SDK. Node IDs are fixed so callers can address outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .camera import PRESETS, h3_frame_count
from .looks import render_to_real_prompt
from .upscale import FINAL_VIDEO, add_final_upscale  # noqa: F401  (FINAL_VIDEO is re-exported)
from .weather import DEFAULT_WEATHER, weather_prompt
from .zoom import CLOUD_BASE_M, cloud_entry_fraction, cloud_middle_fraction, height_fraction

# Output node IDs, stable across graph variants.
CONTROL_VIDEO = "20"
ROUTE_MAP = "21"
OPENING_FRAME = "22"
DEPTH_VIDEO = "23"
H3_VIDEO = "46"
PLACE_PHOTO_NODES = ("31", "32", "33", "34", "35", "36", "37", "38")
LTX_VIDEO = "90"
STAGE1_VIDEO = "377"  # LTX 2.5 stage 1 decoded on its own (--save-stage1)
SUBJECT = "170"
SUBJECT_SHEET = "173"  # the subject sheet (one clean full-length picture of the person), saved for review
LTX_KEYFRAMES = ("91", "92", "93", "94", "95", "96", "97", "98", "99", "82", "83", "84", "85", "86", "87", "88", "89", "80")  # photoreal keyframes, saved
# Where to pin extra photoreal keyframes, as fractions of the shot. A scale change as large as the Earth
# zoom drifts without anchors at each scale; the in-cloud frames are left to the blocking.
# (fraction, source): "repaint" asks Qwen for a photograph of the blocking frame; "blocking" pins the blocking
# frame itself, which for the high part of the Earth zoom already is real satellite imagery (Qwen redraws
# those nadir views as oblique landscapes).
# The cloud whiteout gets its own repainted keyframe: pinned only by the blocking, LTX dropped the cloud sheets
# and flew through clear air. So does the last of the swoop (0.37), where the near ground is the coarse,
# untextured 20 m mesh at eye height and LTX invented a white rock slab between the 0.3 and 0.45 keyframes.
# 0.23 is the descent below the cloud: with no keyframe between the cloud and 0.3, LTX painted orange sand there
# and then snapped to the 0.3 keyframe's pale limestone (about 20 dE of ground colour in 6 frames). Painted on
# its own it chose dry brown ground, so it is a "descent" keyframe: painted from the 0.3 one below it.
# The pinned blocking frame is taken at 100 km, above where the cloud deck's sheets fade in from 60 km (the dive now
# lingers in the deck, so a fixed 0.08 fell at 18 km, among the sheets), and the cloud keyframe in the middle of
# the whiteout. 0.265 halves the 16-frame gap of the fast swoop between 0.23 and 0.3: with only the blocking
# there, LTX smeared the terraced house rows into pink, white and green bands (Twin Peaks r16/r18 f60-77). It is a
# "descent" keyframe too, painted from 0.3, so the two show the same houses.
MID_KEYFRAMES = {"Earth zoom": ((round(height_fraction(100_000), 3), "blocking"), (round(cloud_middle_fraction(), 3), "repaint"),
                                (0.23, "descent"),
                                (0.3, "repaint"), (0.37, "repaint"), (0.45, "repaint"), (0.72, "repaint"),
                                (0.265, "descent"))}  # last, so the earlier keyframes keep their node ids

HOLD_SPACING = 32  # frames between Earth zoom hold keyframes with hold_keyframes="dense"

H3_TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
H3_CONTROL_PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
H3_MODELS = {
    "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
}


# Render-to-real: Qwen-Image-Edit repaints the first and last blocking frames as photographs, then LTX 2.3
# with the 3DREAL IC-LoRA follows the whole blocking video between those two pinned frames.
LTX_MODELS = {
    "checkpoint": "ltx-2.3-22b-dev-fp8.safetensors",
    "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
    "ic_lora": "3DREAL-strong-v2.safetensors",
    "text_encoder": "gemma_3_12B_it_fp8_scaled.safetensors",
    "upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "qwen_unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "qwen_lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen_clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_vae": "qwen_image_vae.safetensors",
    # FLUX.2 [klein] 9B distilled (4 steps), the alternative keyframe painter (keyframe_model="flux-klein").
    "flux_unet": "flux-2-klein-9b-fp8.safetensors",
    "flux_clip": "qwen_3_8b_fp8mixed.safetensors",
    "flux_vae": "flux2-vae.safetensors",
}

# LTX 2.5 with Lightricks' Layout To Render IC-LoRA (ltx_version="2.5"): the same recipe as 3DREAL (clay
# playblast + photoreal first frame, two-stage distilled, 2x upscale), but trained for exactly this step.
# The int8 convrot ComfyUI builds are the ones that fit a 24 GB card.
LTX25_MODELS = {
    "transformer": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "ic_lora": "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors",
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}
KEYFRAME_MODELS = ("qwen", "flux-klein")
LTX_VERSIONS = ("2.3", "2.5")


EXIT_DENOISE = 0.6  # the deck-exit keyframe, painted from the blocking's half-cleared haze so the haze stays
# (on the cleared frame since r30, 0.6 kept the blocking's grey, map-like city: the review of 9 Oct found f30-48
# desaturated from r30 on, where r23-r26's exit in the haze came out with a blue bay; see exit_denoise)


@dataclass
class ShotRequest:
    name: str
    latitude: float | None = None
    longitude: float | None = None
    search: str | None = None
    result_number: int = 1
    radius_km: float = 12
    preset: str = "Rim reveal"
    duration_s: float = 10
    heading_deg: float = -1
    shot_scale_m: float = 0
    lens_mm: float = 24
    clearance_m: float = 25
    orbit_degrees: float = 120
    engine: str = "eevee"
    palette: str = "default"      # Blender terrain colours: default or grand-canyon
    weather: str = DEFAULT_WEATHER  # sky, sun and haze in the render and the prompt
    imagery: bool = False         # drape NASA HLS satellite imagery instead of palette colours
    trees: str = "off"            # measured tree proxies: off, auto, lidar or meta
    buildings: bool = False       # Earth zoom: Overture building blocks around the landing
    canopy: str = ""              # looks.canopy_text of the measured trees, for the keyframe and LTX prompts
    width: int = 1280
    height: int = 720
    samples: int = 16
    output_prefix: str = "geoblock/shot"
    prompt: str = ""
    place_photos: list = field(default_factory=list)  # LoadImage values (file names or SDK assets)
    seed: int = 20261005
    steps: int = 12
    h3_models: dict = field(default_factory=lambda: dict(H3_MODELS))
    lora: str = ""                # e.g. H3_TURBO_LORA with steps=4
    opening_reference: bool = True  # pass the clay opening frame as <Picture 1>
    depth_strength: float = 0.0   # > 0 renders a depth pass and drives H3 Fun ControlNet with it
    depth_end: float = 0.6        # stop the control at this fraction of the schedule
    control_patch: str = H3_CONTROL_PATCH
    first_frame: object = None    # LoadImage value anchored at frame 0 with MiniMaxH3AddGuide
    ltx_models: dict = field(default_factory=lambda: dict(LTX_MODELS))
    ltx_upscale: bool = True      # 2x latent upscale + 3-step refine (about 1920x1088)
    ground_repaint: float = 0.0   # reproject: repaint the stretched near ground through its mask at this denoise (0 = off)
    dive_tempo: str = "standard"  # Earth zoom: zoom.DIVE_TEMPOS ("low-swoop" slows the swoop below the cloud)
    hold_follow: str = "crop"     # Earth zoom: later hold keyframes start from the first one "crop"ped or "reproject"ed,
    #                               or "chain": each reprojected from the hold keyframe before it
    save_stage1: bool = False     # LTX 2.5: also decode and save stage 1 (before the refine), for diagnosis
    keyframe_model: str = "qwen"  # photoreal keyframe painter: qwen (Qwen-Image-Edit 2511) or flux-klein
    colour_match: float = 0.6     # pull each keyframe's colour cast toward its blocking frame (0 = off)
    colour_smooth_s: float | None = None  # temporal smoothing of the video's colour cast; None = 0.75 s for the Earth zoom
    ltx_version: str = "2.3"      # 2.3: LTX 2.3 + 3DREAL; 2.5: LTX 2.5 + Layout To Render
    ltx25_models: dict = field(default_factory=lambda: dict(LTX25_MODELS))
    # The LTX 2.5 text encoder is 15.4 GB and the transformer 21.5 GB: together they do not fit on a 24 GB card,
    # and the transformer needs the card. The encoder runs once per job, so it belongs on the CPU.
    ltx_encoder_device: str = "cpu"
    subject_photo: object = None  # LoadImage value: the person the Earth zoom lands on
    subject_description: str = ""  # used without a photo (an invented figure)
    pin_last_keyframe: bool | None = None  # LTX 2.5: False lets the end follow the layout; None = not for the Earth zoom
    subject_sheet: bool = True    # klein: paint one clean full-length picture of the person for every person keyframe
    keyframes_only: bool = False  # stop after the photoreal keyframes (fast look checks, no LTX)
    orbit_from_blocking: bool = False  # Earth zoom: pin frame 0 to the blocking frame (klein's stays the look reference)
    keyframe_denoise: float = 1.0  # klein, eye-level frames: below 1 starts from the frame itself (layout, person's size)
    landing_repaint: float = 0.0  # landing_from_hold: repaint where f112's plate doesn't reach, through its mask (0 = off)
    hold_near_repaint: float = 0.0  # reprojected hold plates: also repaint the ground this close to the camera (m)
    hold_keep: float = 0.0  # reprojected hold plates: paste the first plate back over the repaint this strongly (kept mask)
    landing_fill: str = "around"  # landing_from_hold: where f112's plate doesn't reach, fill from around or "blocking"
    landing_from_hold: bool = False  # Earth zoom composite: paint the landing plate from the hold's first, reprojected
    landing_plate_denoise: float = 1.0  # klein, Earth zoom: the same for the landing's descent plate alone (f88)
    swoop_denoise: float = 1.0  # klein, Earth zoom swoop keyframes (below the cloud, before the composite lead-in)
    keyframe_refine: float = 0.0   # klein, eye-level frames: > 0 repaints that result once more at this denoise
    composite_subject: bool = False  # klein, eye-level frames: paint the clean plate, paste the person, harmonise
    person_detector: str = ""  # composite: local Florence-2 folder; people found on a plate are painted out
    foreground: str = ""  # composite: measured ground between the hold camera and the person (geoblock-run fills it)
    punch_through: bool = False  # Earth zoom: dissolve LTX's cut from cloud to clear ground (zoom.punch_through)
    exit_denoise: float = EXIT_DENOISE  # Earth zoom exit_keyframe "repaint": how far klein repaints the blocking
    exit_keyframe: str = "none"  # Earth zoom: a keyframe where the camera leaves the cloud deck: none, blocking or repaint
    stretch_grain: float = 0.0  # Earth zoom reprojected hold: noise on the blurred stretched ground, for klein to texture
    swoop_gap: int = 0  # Earth zoom: > 0 adds "descent" keyframes between the deck exit and the hold until no gap is wider
    cloud_keyframe: str = "repaint"  # Earth zoom cloud-dive keyframe: repaint, blocking or none (geoblock-run: none)
    hold_plate_denoise: float = 0.6  # composite: later hold plates repaint the first plate's crop this far (1: afresh)
    closing_plate_denoise: float = 0.9  # composite: the same for the closing close-up, cropped furthest
    place_subject: bool = False   # Earth zoom: paste the subject sheet's person over the stand-in before klein paints
    hold_keyframes: str = "crop"  # Earth zoom hold keyframes after the first: crop (of the first), sharp (Real-ESRGAN
                                  # before the crop), drop (only the first is pinned), dense (sharp,
                                  # every 32 frames) or direct (the upscaled crop itself, not repainted)
    final_upscale: str = "none"   # optional last pass on the LTX frames: one of upscale.UPSCALERS
    final_target: str = "4k"      # 4k (3840x2176) or 1080p (same size, detail rebuilt from half size)
    place_id: str | None = None   # a places.json landing place; replaces search and coordinates
    look: dict | None = None      # measured look of the place (landcover.look_from_landcover) for prompts
    landing_lidar: str = "none"   # "auto" or a 3DEP 1 m project: 1 m lidar terrain where the Earth zoom lands


def blocking_graph(request: ShotRequest) -> dict:
    """Place → landform scene → camera preset → Blender blocking video."""
    if request.place_id:
        from .places import catalog_option
        graph = {"2": {"class_type": "GeoBlockPlaceFromCatalog", "inputs": {
            "place": catalog_option(request.place_id), "coverage_radius_km": request.radius_km}}}
    elif request.search:
        graph = {
            "1": {"class_type": "GeoBlockSearchLandmarks", "inputs": {"query": request.search}},
            "2": {"class_type": "GeoBlockChooseLandmark", "inputs": {
                "landmark_matches": ["1", 0], "result_number": request.result_number,
                "coverage_radius_km": request.radius_km}},
        }
    else:
        if request.latitude is None or request.longitude is None:
            raise ValueError("Give a search query or both latitude and longitude.")
        graph = {"2": {"class_type": "GeoBlockPlaceFromCoordinates", "inputs": {
            "name": request.name, "latitude": request.latitude, "longitude": request.longitude,
            "coverage_radius_km": request.radius_km}}}
    graph.update({
        "3": {"class_type": "GeoBlockBuildLandformScene", "inputs": {"place": ["2", 0], "radius_km_override": 0,
                                                             "landing_lidar": request.landing_lidar}},
        "4": {"class_type": "GeoBlockCameraPreset", "inputs": {
            "terrain_scene": ["3", 0], "preset": request.preset, "duration_seconds": request.duration_s,
            "heading_degrees": request.heading_deg, "shot_scale_m": request.shot_scale_m,
            "lens_mm": request.lens_mm, "min_clearance_m": request.clearance_m,
            "orbit_degrees": request.orbit_degrees,
            **({"dive_tempo": request.dive_tempo} if request.dive_tempo != "standard" else {})}},
        "5": {"class_type": "GeoBlockRenderBlockingVideo", "inputs": {
            "terrain_scene": ["3", 0], "shot": ["4", 0], "engine": request.engine,
            "width": request.width, "height": request.height, "samples": request.samples,
            "depth_pass": request.depth_strength > 0, "palette": request.palette, "weather": request.weather,
            "satellite_imagery": request.imagery, **({"trees": request.trees} if request.trees != "off" else {}),
            **({"buildings": True} if request.buildings else {})}},
        CONTROL_VIDEO: {"class_type": "SaveVideo", "inputs": {
            "video": ["5", 0], "filename_prefix": request.output_prefix + "_control",
            "format": "auto", "codec": "auto"}},
        ROUTE_MAP: {"class_type": "SaveImage", "inputs": {
            "images": ["4", 1], "filename_prefix": request.output_prefix + "_route"}},
        OPENING_FRAME: {"class_type": "SaveImage", "inputs": {
            "images": ["5", 1], "filename_prefix": request.output_prefix + "_opening"}},
    })
    if request.depth_strength > 0:
        graph[DEPTH_VIDEO] = {"class_type": "SaveVideo", "inputs": {
            "video": ["5", 4], "filename_prefix": request.output_prefix + "_depth", "format": "auto", "codec": "auto"}}
    return graph


def default_prompt(request: ShotRequest) -> str:
    photos = len(request.place_photos)
    first = 2 if request.opening_reference else 1
    lines = []
    if request.opening_reference:
        lines.append("<Picture 1> is the opening frame of the camera move.")
    lines.append(f"<Video 1> fixes the continuous camera route, timing and the measured shape of the terrain "
                 f"at {request.name}: keep every ridge, cliff and valley exactly where <Video 1> puts them, "
                 "frame by frame.")
    if photos:
        tags = (f"<Picture {first}>" if photos == 1
                else f"<Picture {first}> to <Picture {first + photos - 1}>")
        lines.append(f"{tags} show only what the real place looks like: rock colour and banding, vegetation, "
                     "light and atmosphere. Do not copy their viewpoint, framing, railings, buildings or people.")
    lines.append("A single uninterrupted aerial drone shot with realistic rock, vegetation and atmosphere. "
                 + weather_prompt(request.weather) + " No cuts, no morphing, no people, no text.")
    return " ".join(lines)


def _check_h3(request: ShotRequest) -> None:
    if request.duration_s > 15:
        raise ValueError("H3 takes reference videos of at most 15 s; chain shorter shots for longer moves.")
    if len(request.place_photos) + request.opening_reference > 9:
        raise ValueError("H3 takes at most 9 reference images.")


def _add_h3(graph: dict, request: ShotRequest, video: list, opening_frame: list, length,
            depth_video: list | None = None) -> dict:
    """Append the Ref2VA stage, reading the control video, opening frame and length from the given sources."""
    references = {"ref_videos.ref_video_0": ["41", 0]}
    first_photo = 0
    if request.opening_reference:
        references["ref_images.ref_image_0"] = opening_frame
        first_photo = 1
    for index, (node, photo) in enumerate(zip(PLACE_PHOTO_NODES, request.place_photos), start=first_photo):
        graph[node] = {"class_type": "LoadImage", "inputs": {"image": photo}}
        references[f"ref_images.ref_image_{index}"] = [node, 0]
    models = request.h3_models
    model, positive = ["42", 0], ["45", 0]
    if request.lora:
        graph["54"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": request.lora, "strength_model": 1.0}}
        model = ["54", 0]
    if request.depth_strength > 0:
        if depth_video is None:
            raise ValueError("Depth control needs the blocking render's depth pass.")
        graph.update({
            "55": {"class_type": "ModelPatchLoader", "inputs": {"name": request.control_patch}},
            "56": {"class_type": "GetVideoComponents", "inputs": {"video": depth_video}},
            "57": {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {
                "model": model, "model_patch": ["55", 0], "vae": ["44", 0], "strength": request.depth_strength,
                "start_percent": 0.0, "end_percent": request.depth_end, "control_video": ["56", 0]}},
        })
        model = ["57", 0]
    if request.first_frame is not None:
        graph.update({
            "58": {"class_type": "LoadImage", "inputs": {"image": request.first_frame}},
            "59": {"class_type": "MiniMaxH3AddGuide", "inputs": {
                "positive": positive, "vae": ["44", 0], "latent": ["45", 1], "image": ["58", 0], "frame_idx": 0}},
        })
        positive = ["59", 0]
    graph.update({
        "41": {"class_type": "GetVideoComponents", "inputs": {"video": video}},
        "42": {"class_type": "UNETLoader", "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "43": {"class_type": "CLIPLoader", "inputs": {"clip_name": models["clip"], "type": "minimax", "device": "default"}},
        "44": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "45": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["43", 0], "vae": ["44", 0], "prompt": request.prompt or default_prompt(request),
            "width": 864, "height": 480, "length": length, "ref_image_size": "match", **references}},
        "47": {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed}},
        "48": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": positive}},
        "49": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "50": {"class_type": "BasicScheduler", "inputs": {
            "model": model, "scheduler": "beta", "steps": request.steps, "denoise": 1.0}},
        "51": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["47", 0], "guider": ["48", 0], "sampler": ["49", 0], "sigmas": ["50", 0],
            "latent_image": ["45", 1]}},
        "52": {"class_type": "VAEDecode", "inputs": {"samples": ["51", 0], "vae": ["44", 0]}},
        "53": {"class_type": "CreateVideo", "inputs": {"images": ["52", 0], "fps": 24}},
        H3_VIDEO: {"class_type": "SaveVideo", "inputs": {
            "video": ["53", 0], "filename_prefix": request.output_prefix + "_h3", "format": "auto", "codec": "auto"}},
    })
    return graph


def h3_graph(request: ShotRequest) -> dict:
    """The blocking graph plus MiniMax H3 Reference-to-Video on its outputs."""
    _check_h3(request)
    return _add_h3(blocking_graph(request), request, ["5", 0], ["5", 1], ["5", 2], depth_video=["5", 4])


def h3_from_files_graph(request: ShotRequest, control_video, opening_frame, frames: int) -> dict:
    """H3 only, for a server without GeoBlock or Blender (e.g. Comfy Cloud).

    ``control_video`` and ``opening_frame`` are LoadVideo/LoadImage values:
    file names on that server, or Comfy SDK assets uploaded at submit time.
    """
    _check_h3(request)
    graph = {
        "39": {"class_type": "LoadImage", "inputs": {"image": opening_frame}},
        "40": {"class_type": "LoadVideo", "inputs": {"file": control_video}},
    }
    return _add_h3(graph, request, ["40", 0], ["39", 0], frames)


def ltx_frame_count(duration_s: float) -> int:
    """LTX length for the route: the blocking's frame count rounded down to 8n + 1."""
    return (h3_frame_count(duration_s) - 1) // 8 * 8 + 1


DECK_EXIT_FRAMES = 8  # blender_earth_zoom.DECK_EXIT_FRAMES: the deck thins out over this many frames before its base


def _cloud_frame(request: ShotRequest) -> int:
    """The Earth zoom's cloud keyframe frame: the guide slot nearest the middle of the solid whiteout, from where
    the camera enters the deck to where it starts thinning out. (The middle in log height fell at f24 with
    low-swoop, already half-cleared, and pinned there LTX skipped the white phase: r23 f15 cut to haze.)"""
    last = h3_frame_count(request.duration_s) - 1
    entry = cloud_entry_fraction(tempo=request.dive_tempo) * last
    thinning = height_fraction(CLOUD_BASE_M, tempo=request.dive_tempo) * last - DECK_EXIT_FRAMES
    return 8 * round((entry + thinning) / 2 / 8)


def _mid_keyframes(request: ShotRequest) -> tuple:
    """MID_KEYFRAMES for the request, the blocking pin and cloud keyframe moved with its dive tempo."""
    keyframes = MID_KEYFRAMES.get(request.preset, ())
    if request.preset != "Earth zoom" or request.dive_tempo == "standard":
        return keyframes
    moved = ((round(height_fraction(100_000, tempo=request.dive_tempo), 3), "blocking"),
             (_cloud_frame(request) / (h3_frame_count(request.duration_s) - 1), "repaint"))
    return moved + keyframes[2:]




def _exit_frame(request: ShotRequest) -> int:
    """The first guide slot (multiple of 8) at or after the frame where the blocking has cleared the deck, and
    after the cloud keyframe. Guided inside the thinning haze (r24, r26, r27: f24, f24, f40), LTX snapped from
    cloud to clear ground in a frame or two; guided on the cleared frame (r28: f32 after the blocking cleared at
    f28), it followed the blocking's 8 frames of haze with no cut."""
    exit_at = height_fraction(CLOUD_BASE_M, tempo=request.dive_tempo) * (h3_frame_count(request.duration_s) - 1)
    slot = 8 * math.ceil(exit_at / 8)
    return slot if slot > _cloud_frame(request) else slot + 8


def _keyframe_base(k: int) -> int:
    """First node id of mid keyframe k: 200 + 8k, then from 520 past the LTX 2.5 nodes at 300-377."""
    return 200 + 8 * k if k < 12 else 520 + 8 * (k - 12)


def _swoop_fill(request: ShotRequest, guides: list) -> list:
    """Guide slots to add between the deck exit and the hold so no gap is wider than request.swoop_gap. In the
    swoop LTX smears the terraced house rows into bands across the widest gaps (Twin Peaks r29-r32: f40-56
    between the f32 exit and f56 keyframes), as the 0.265 keyframe fixed for r18. r33's f48 fill turned them into
    streets of roofed houses."""
    start = _exit_frame(request) if request.exit_keyframe != "none" else 8 * round(
        height_fraction(CLOUD_BASE_M, tempo=request.dive_tempo) * (h3_frame_count(request.duration_s) - 1) / 8)
    # Only over the rooftops, up to the composite lead-in: lower down a fill keyframe at f96 (r33) painted a big
    # near oak that whipped past the frame edge over f91-100, worse than r32's smear there.
    end = _arrive_frame(request) - COMPOSITE_LEAD_FRAMES
    slots = sorted({g for g in guides if start <= g <= end} | {start})
    added = []
    for a, b in zip(slots, slots[1:]):
        while b - a > request.swoop_gap:
            middle = 8 * round((a + b) / 16)
            if middle in (a, b):
                break
            added.append(middle)
            a = middle
    return added


def _arrive_frame(request: ShotRequest) -> int:
    """First LTX frame of the Earth zoom's eye-level hold (matches zoom.plan_earth_zoom's arrive_at)."""
    return int(0.45 * (h3_frame_count(request.duration_s) - 1))


def _keyframe(graph: dict, request: ShotRequest, frames: list, index: int, base: int, save: str,
              reference: tuple | None = None) -> list:
    """Repaint one blocking frame as a photograph with the request's keyframe model. ``reference`` is
    (image, frame, kind) of another keyframe: kind "hold", an earlier keyframe of the same spot from further
    back, or "descent", the next keyframe down, whose ground this one should match."""
    image_ref, frame_ref, kind = reference or (None, None, None)
    if request.keyframe_model == "flux-klein":
        # klein copies a reference image wholesale, framing included, so a hold keyframe gets the earlier one
        # zoomed to its framing; given it whole, it froze the Earth zoom's push-in on the first person keyframe.
        if kind == "landing" or (kind == "hold" and request.hold_follow in ("reproject", "chain")):
            # A dolly, not a zoom: the crop scaled the whole picture by the person's growth, ~6x by f240, so the
            # distant skyline came out blown up and soft (second-opinion review of r18). Moved through the terrain
            # to the later camera, the city keeps its size and only the near ground grows.
            graph[f"{base}zr"] = {"class_type": "GeoBlockReprojectHold", "inputs": {
                "terrain_scene": ["3", 0], "shot": ["4", 0], "image": image_ref, "from_frame": frame_ref,
                "to_frame": index}}
            if request.stretch_grain > 0:
                graph[f"{base}zr"]["inputs"]["grain"] = request.stretch_grain
            if kind == "landing" and request.landing_fill == "blocking":
                graph[f"{base}zp"] = {"class_type": "GeoBlockCleanPlate", "inputs": {
                    "clean_plates": ["5", 5], "image": frames, "frame": index}}
                graph[f"{base}zr"]["inputs"]["fill"] = [f"{base}zp", 0]
            if kind == "landing" and request.landing_repaint > 0:
                graph["_unseen"] = graph.get("_unseen", {}) | {index: [f"{base}zr", 3]}
            if kind == "hold" and request.hold_near_repaint > 0:
                graph[f"{base}zr"]["inputs"]["near_m"] = request.hold_near_repaint
            if kind == "hold" and request.hold_keep > 0:
                graph[f"{base}zr"]["inputs"]["keep"] = request.hold_keep
                graph["_kept"] = graph.get("_kept", {}) | {index: ([f"{base}zr", 0], [f"{base}zr", 2])}
            image_ref = [f"{base}zr", 0]
            graph["_stretch"] = graph.get("_stretch", {}) | {index: [f"{base}zr", 1]}
        elif kind == "hold":
            if request.hold_keyframes in ("sharp", "dense"):
                # The crop is enlarged about 2x from a 960 px keyframe: too little detail is left in the distance,
                # so klein invented a new city behind the hiker at f176. Upscaled first, the crop keeps it.
                image_ref = _esrgan(graph, image_ref, f"{base}u")
            image_ref = _push_in_crop(graph, request, image_ref, frame_ref, index, f"{base}z")
            if request.hold_keyframes in ("sharp", "dense"):  # back to keyframe size: the crop keeps its input's size
                graph[f"{base}zs"] = {"class_type": "ImageScaleBy", "inputs": {
                    "image": image_ref, "upscale_method": "lanczos", "scale_by": 0.5}}
                image_ref = [f"{base}zs", 0]
        image = _flux_keyframe(graph, request, frames, index, base, save, image_ref, kind)
    else:
        image = _qwen_keyframe(graph, request, frames, index, base, save, image_ref)
    if kind == "descent":
        graph[str(base + 7)]["inputs"]["reference_kind"] = "descent"
    if request.colour_match > 0:
        # Each keyframe is painted on its own and drifts in white balance and saturation (the Earth zoom went warm,
        # then blue at the canyon, then warm again); pulling every one toward its own blocking frame gives them the
        # blocking's continuous colour, so LTX has no colour jump to interpolate across.
        decoded, matched = graph["_decoded"][index], f"{base}c"  # the last pass, when klein paints twice
        graph[matched] = {"class_type": "GeoBlockColourMatch", "inputs": {
            "image": decoded, "reference": [str(base), 0], "strength": request.colour_match}}
        graph[save]["inputs"]["images"] = graph[str(base + 6)]["inputs"]["image"] = [matched, 0]
        graph["_decoded"][index] = [matched, 0]
    return image


def _cropped_hold(graph: dict, request: ShotRequest, image: list, from_frame: int, index: int, base: int,
                  save: str) -> list:
    """A hold keyframe that is the first hold keyframe itself, Real-ESRGAN upscaled and cropped to ``index``'s
    framing, with no repaint: each klein repaint drew its own skyline (Twin Peaks r3: towers grew and shrank
    over f176-192) and lost the push-in framing (f208 came out wide again)."""
    cropped = _push_in_crop(graph, request, _esrgan(graph, image, f"{base}u"), from_frame, index, f"{base}z")
    graph.update({
        f"{base}zs": {"class_type": "ImageScaleBy", "inputs": {  # back to keyframe size
            "image": cropped, "upscale_method": "lanczos", "scale_by": 0.5}},
        save: {"class_type": "SaveImage", "inputs": {
            "images": [f"{base}zs", 0], "filename_prefix": f"{request.output_prefix}_keyframe_{index:04d}"}},
        str(base + 6): {"class_type": "LTXVPreprocess", "inputs": {"image": [f"{base}zs", 0], "img_compression": 18}},
    })
    return [str(base + 6), 0]


def _esrgan(graph: dict, image: list, prefix: str) -> list:
    """Real-ESRGAN x2 (the final upscaler's model) on one image."""
    from .upscale import ESRGAN_MODEL
    graph[f"{prefix}m"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": ESRGAN_MODEL}}
    graph[f"{prefix}x"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": [f"{prefix}m", 0],
                                                                           "image": image}}
    return [f"{prefix}x", 0]


def _push_in_crop(graph: dict, request: ShotRequest, image: list, from_frame: int, to_frame: int, prefix: str) -> list:
    """An Earth zoom hold keyframe cropped (GeoBlockPushInCrop) to how a later frame frames the stand-in."""
    graph[f"{prefix}crop"] = {"class_type": "GeoBlockPushInCrop", "inputs": {
        "terrain_scene": ["3", 0], "shot": ["4", 0], "image": image, "from_frame": from_frame, "to_frame": to_frame}}
    return [f"{prefix}crop", 0]


def _flux_keyframe(graph: dict, request: ShotRequest, frames: list, index: int, base: int, save: str,
                   reference: list | None = None, kind: str | None = None) -> list:
    """FLUX.2 [klein] edit of one blocking frame (ComfyUI's klein edit template): the frame, then the earlier
    keyframe ``reference`` and the subject's photo on eye-level frames, go in as reference latents; 4 steps at CFG 1 from an empty latent. Uses node IDs
    base..base+7 like the Qwen version (same roles for prompt, decode, save and preprocess) plus "<base>f*"."""
    n = [str(base + i) for i in range(8)]
    f = lambda name: f"{base}f{name}"  # noqa: E731
    eye_level = "_subject_image" in graph and index >= _arrive_frame(request)
    # The descent keyframe just before the hold sees the stand-in too: painted on its own it turned it into a
    # tree, which LTX then morphed into the hiker (r5, 0:03-0:04). It is composited like the hold.
    composite = request.composite_subject and "_subject_image" in graph and (
        index >= _arrive_frame(request) - COMPOSITE_LEAD_FRAMES)
    graph.update({
        n[7]: {"class_type": "GeoBlockKeyframePrompt", "inputs": {
            "terrain_scene": ["3", 0], "shot": ["4", 0], "frame": index, "palette": request.palette,
            "weather": request.weather, "reference": reference is not None,
            **({"subject": [SUBJECT, 0]} if SUBJECT in graph and not composite else {}),
            **({"canopy": request.canopy} if request.canopy else {})}},
        n[0]: {"class_type": "ImageFromBatch", "inputs": {"image": frames, "batch_index": index, "length": 1}},
        f("scale"): {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [n[0], 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
        f("size"): {"class_type": "GetImageSize", "inputs": {"image": [f("scale"), 0]}},
        n[1]: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["61", 0], "text": [n[7], 0]}},
        n[2]: {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [n[1], 0]}},
        n[3]: {"class_type": "VAEEncode", "inputs": {"pixels": [f("scale"), 0], "vae": ["62", 0]}},
        f("pos"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": [n[1], 0], "latent": [n[3], 0]}},
        f("neg"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": [n[2], 0], "latent": [n[3], 0]}},
    })
    if composite:  # the place with nobody in it: the blocking frame rendered without the stand-in
        graph[f("plate")] = {"class_type": "GeoBlockCleanPlate", "inputs": {
            "clean_plates": ["5", 5], "image": frames, "frame": index}}
        graph[f("scale")]["inputs"]["image"] = [f("plate"), 0]
        graph[n[7]]["inputs"]["plate"] = True
        if request.foreground:
            graph[n[7]]["inputs"]["foreground"] = request.foreground
        # Painted afresh, each hold plate grew its own trees (r9k: oaks at f112/f176, stone pines at f240; r11k:
        # a different place at f240 even with the crop as Image 2). Later ones start from the first plate cropped
        # to their framing instead. The closing close-up is cropped ~6x: at 0.6 its city stayed mush (r10k), so
        # it is repainted further: 0.8 stayed soft, 1.0 painted a different place, 0.9 is crisp and the same (r19k).
        follow_denoise = request.closing_plate_denoise if save == LTX_KEYFRAMES[1] else request.hold_plate_denoise
        if kind == "landing":
            follow_denoise = request.landing_plate_denoise
        if follow := (reference is not None and kind in ("hold", "landing") and follow_denoise < 1.0):
            graph[f("scale")]["inputs"]["image"] = reference
    elif request.place_subject and eye_level:
        # klein redraws the person full-length and centred whatever the stand-in's size; pasted in at that size
        # first, they stay there, and the push-in's framing with them.
        graph[f("place")] = {"class_type": "GeoBlockPlaceSubject", "inputs": {
            "terrain_scene": ["3", 0], "shot": ["4", 0], "image": [n[0], 0],
            "subject_image": graph["_subject_image"], "frame": index, "clean_plates": ["5", 5]}}
        graph[f("scale")]["inputs"]["image"] = [f("place"), 0]
        graph[f("placed")] = {"class_type": "SaveImage", "inputs": {  # what klein is given, for review
            "images": [f("place"), 0], "filename_prefix": f"{request.output_prefix}_placed_{index:04d}"}}
        graph[n[7]]["inputs"]["placed"] = True
    positive, negative = [f("pos"), 0], [f("neg"), 0]
    extra = ([reference] if reference is not None else []) + (
        [graph["_subject_image"]] if eye_level and not composite else [])
    for k, image in enumerate(extra, start=2):  # Image 2 (earlier keyframe or subject), Image 3 (subject)
        graph.update({
            f(f"img{k}"): {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": image, "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
            f(f"lat{k}"): {"class_type": "VAEEncode", "inputs": {"pixels": [f(f"img{k}"), 0], "vae": ["62", 0]}},
            f(f"pos{k}"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive,
                                                                     "latent": [f(f"lat{k}"), 0]}},
            f(f"neg{k}"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative,
                                                                     "latent": [f(f"lat{k}"), 0]}},
        })
        positive, negative = [f(f"pos{k}"), 0], [f(f"neg{k}"), 0]
    size = [f("size"), 0], [f("size"), 1]
    graph.update({
        f("latent"): {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1],
                                                                       "batch_size": 1}},
        f("sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}},
        f("guider"): {"class_type": "CFGGuider", "inputs": {"model": ["63", 0], "positive": positive,
                                                            "negative": negative, "cfg": 1.0}},
        f("sampler"): {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        f("noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + index}},
        n[4]: {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [f("noise"), 0], "guider": [f("guider"), 0], "sampler": [f("sampler"), 0],
            "sigmas": [f("sigmas"), 0], "latent_image": [f("latent"), 0]}},
        n[5]: {"class_type": "VAEDecode", "inputs": {"samples": [n[4], 0], "vae": ["62", 0]}},
        save: {"class_type": "SaveImage", "inputs": {"images": [n[5], 0],
                                                     "filename_prefix": f"{request.output_prefix}_keyframe_{index:04d}"}},
        n[6]: {"class_type": "LTXVPreprocess", "inputs": {"image": [n[5], 0], "img_compression": 18}},
    })
    if composite:
        if follow:
            _klein_from(graph, f(""), [f("sigmas"), 0], n[4], [n[3], 0], follow_denoise)
            del graph[f("latent")]
        elif kind == "descent" and (plate_denoise := min(request.swoop_denoise, request.landing_plate_denoise)) < 1.0:
            # The landing's descent plate (f88) drifts like the swoop keyframes: painted from a blank it framed a
            # dirt clearing and an umbrella pine the blocking doesn't have, and LTX swapped scenery at f88-100
            # (r18, r19). From its own clean plate it keeps the blocking's framing. landing_plate_denoise does this
            # for it alone: from a blank it also painted a big oak that whipped past the frame edge (r33-r37).
            _klein_from(graph, f(""), [f("sigmas"), 0], n[4], [n[3], 0], plate_denoise)
            del graph[f("latent")]
        return _composite_subject(graph, request, index, base, save, size)
    deck_exit = (request.preset == "Earth zoom" and request.exit_keyframe == "repaint"
                 and index == _exit_frame(request))
    swoop = (request.preset == "Earth zoom" and not eye_level and request.swoop_denoise < 1.0
             and _cloud_frame(request) < index < _arrive_frame(request) - COMPOSITE_LEAD_FRAMES)
    if deck_exit:
        _klein_from(graph, f(""), [f("sigmas"), 0], n[4], [n[3], 0], request.exit_denoise)
        del graph[f("latent")]
    elif swoop:
        # Painted from a blank, the f64 swoop keyframe came out wider than its blocking frame (r19: rows of houses
        # where the blocking had the summit), and LTX, held to the blocking's layout, honoured neither and smeared
        # the rows. Started from the frame's own latent, the keyframe keeps the blocking's framing.
        _klein_from(graph, f(""), [f("sigmas"), 0], n[4], [n[3], 0], request.swoop_denoise)
        del graph[f("latent")]
    if eye_level and request.keyframe_denoise < 1.0:
        # From a blank latent klein recomposes freely: Twin Peaks r5k/r6k drew the hiker 2.5x the pasted person
        # and reframed the city. Starting from the frame's own latent, partly noised, keeps its layout (r7k: the
        # size held at 0.6 and 0.8, but the higher frames stayed CG, so only eye-level frames). The 4 steps are
        # the tail of a longer schedule, as KSampler's denoise does.
        _klein_from(graph, f(""), [f("sigmas"), 0], n[4], [n[3], 0], request.keyframe_denoise)
        del graph[f("latent")]
    if eye_level and request.keyframe_denoise < 1.0 and 0 < request.keyframe_refine < 1.0:
        # A second pass on the first one's photo-ish result: more realism, with the layout already set.
        graph.update({
            f("r_lat"): {"class_type": "VAEEncode", "inputs": {"pixels": [n[5], 0], "vae": ["62", 0]}},
            f("r_sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0],
                                                                       "height": size[1]}},
            f("r_noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + index + 1}},
            f("r_sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": [f("r_noise"), 0], "guider": [f("guider"), 0], "sampler": [f("sampler"), 0],
                "sigmas": [f("r_sigmas"), 0], "latent_image": [f("r_lat"), 0]}},
            f("r_decode"): {"class_type": "VAEDecode", "inputs": {"samples": [f("r_sample"), 0], "vae": ["62", 0]}},
        })
        _klein_from(graph, f("r_"), [f("r_sigmas"), 0], f("r_sample"), [f("r_lat"), 0], request.keyframe_refine)
        graph[save]["inputs"]["images"] = [f("r_decode"), 0]
        graph[n[6]]["inputs"]["image"] = [f("r_decode"), 0]
        graph["_decoded"] = graph.get("_decoded", {}) | {index: [f("r_decode"), 0]}
        return [n[6], 0]
    graph["_decoded"] = graph.get("_decoded", {}) | {index: [n[5], 0]}
    return [n[6], 0]


# The blend pass over the pasted person, when --keyframe-denoise is left at 1. It is masked to the person and a
# ring around them, so it can relight them and seat their feet without repainting the plate (r9k: unmasked at
# 0.35-0.5 the person stayed flat and crisp, an obvious paste). Their light is matched to the scene in the
# paste itself (zoom._match_light), since the pass does little.
COMPOSITE_DENOISE = 0.5  # r10k: 0.65 changed almost nothing, 0.8 swapped the person for a tree


# Taking a stranger off a painted plate (--person-detector): the box grown well past the person so the patch
# isn't person-shaped, filled from around it and repainted part-way with this prompt, twice at most.
PEOPLE_REPAINT_PROMPT = ("The same empty ground as all around it, its surface continuing unchanged (road, path or "
                         "grass, whichever surrounds it), in sunlight, with nobody on it.")
PEOPLE_MASK_GROW, PEOPLE_REPAINT_DENOISE, PEOPLE_REPAINT_ROUNDS = 0.6, 0.75, 2
# The near ground a reprojected hold plate stretched and blurred (zoom.stretch_weight) can be repainted through
# its mask (ShotRequest.ground_repaint): at the hold plate's denoise klein kept the blur as defocus (r23k
# f176/f208). Asked for a "close-up", klein put a blurred plant right in front of the lens (r24k f208).
GROUND_REPAINT_PROMPT = ("The ground in front of the camera, seen from standing height{ground}: plain and flat, "
                         "in sharp focus and even sunlight, continuing the surface around it, with no plant, rock "
                         "or object close to the camera and nobody on it.")


def _composite_subject(graph: dict, request: ShotRequest, index: int, base: int, save: str, size: tuple) -> list:
    """Eye-level keyframe in three steps. Painting the blocking frame with the person in it, klein drew them 2.5x
    the stand-in (Twin Peaks r5k/r6k), and starting from the frame to stop that left everything CG (r7k/r8k).
    So klein paints the clean plate at full strength with nobody in it, the subject sheet's person is pasted onto
    that photo at the stand-in's place and size over the stand-in's rendered shadow, and a klein pass masked to
    them and a ring around them blends their light and edges."""
    n = [str(base + i) for i in range(8)]
    f = lambda name: f"{base}f{name}"  # noqa: E731
    plate = [n[5], 0]
    if kept := graph.get("_kept", {}).get(index):
        # The first plate moved to this camera, pasted back where it covers the view unstretched (zoom.kept_mask):
        # repainted whole, each hold plate grew its own midground and LTX popped it in (r44 f192-208).
        graph[f("keep")] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": plate, "source": kept[0], "x": 0, "y": 0, "resize_source": True, "mask": kept[1]}}
        plate = [f("keep"), 0]
    if unseen := graph.get("_unseen", {}).get(index):
        # Where the hold plate doesn't reach the landing camera (57% of f88) the blocking went in, and klein at the
        # plate's 0.8 kept it CG: a plain grey road (r44-r48 f88-98), and at 0.9 everywhere it invented trees (r45).
        # Repainted through that mask only, with the plate's own prompt and references, at landing_repaint.
        r = lambda name: f(f"u_{name}")  # noqa: E731
        graph.update({
            r("lat"): {"class_type": "VAEEncode", "inputs": {"pixels": plate, "vae": ["62", 0]}},
            r("mask"): {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [r("lat"), 0], "mask": unseen}},
            r("noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + index + 11}},
            r("sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}},
            r("sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": [r("noise"), 0], "guider": [f("guider"), 0], "sampler": [f("sampler"), 0],
                "sigmas": [r("sigmas"), 0], "latent_image": [r("mask"), 0]}},
            r("decode"): {"class_type": "VAEDecode", "inputs": {"samples": [r("sample"), 0], "vae": ["62", 0]}},
            r("paste"): {"class_type": "ImageCompositeMasked", "inputs": {
                "destination": plate, "source": [r("decode"), 0], "x": 0, "y": 0, "resize_source": True,
                "mask": unseen}},
        })
        _klein_from(graph, r(""), [r("sigmas"), 0], r("sample"), [r("mask"), 0], request.landing_repaint)
        plate = [r("paste"), 0]
    stretched = graph.get("_stretch", {}).get(index)
    if stretched and request.ground_repaint > 0:
        ground = f": {request.foreground}" if request.foreground else ""
        g = lambda name: f(f"g_{name}")  # noqa: E731
        graph.update({
            g("text"): {"class_type": "CLIPTextEncode", "inputs": {
                "clip": ["61", 0], "text": GROUND_REPAINT_PROMPT.format(ground=ground)}},
            g("zero"): {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [g("text"), 0]}},
            g("lat"): {"class_type": "VAEEncode", "inputs": {"pixels": plate, "vae": ["62", 0]}},
            g("guider"): {"class_type": "CFGGuider", "inputs": {"model": ["63", 0], "positive": [g("text"), 0],
                                                                "negative": [g("zero"), 0], "cfg": 1.0}},
            g("mask"): {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [g("lat"), 0], "mask": stretched}},
            g("noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + index + 7}},
            g("sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}},
            g("sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": [g("noise"), 0], "guider": [g("guider"), 0], "sampler": [f("sampler"), 0],
                "sigmas": [g("sigmas"), 0], "latent_image": [g("mask"), 0]}},
            g("decode"): {"class_type": "VAEDecode", "inputs": {"samples": [g("sample"), 0], "vae": ["62", 0]}},
            g("paste"): {"class_type": "ImageCompositeMasked", "inputs": {
                "destination": plate, "source": [g("decode"), 0], "x": 0, "y": 0, "resize_source": False,
                "mask": stretched}},
        })
        _klein_from(graph, g(""), [g("sigmas"), 0], g("sample"), [g("mask"), 0], request.ground_repaint)
        plate = [g("paste"), 0]
    if request.person_detector:
        # klein put a walker on the path below the hiker's spot whatever the plate prompt said (r10k-r14k). Found
        # people are filled in from the ground around them and that patch is repainted with an empty-path prompt,
        # partly: repainted from noise with the plate prompt, klein drew a new walker in the same spot (r16k).
        # A second round catches one it draws anyway. Each round is pasted back through its mask only, so a
        # round that finds nobody leaves the plate as it was.
        graph[f("p_raw")] = {"class_type": "SaveImage", "inputs": {
            "images": plate, "filename_prefix": f"{request.output_prefix}_plate_raw_{index:04d}"}}
        graph.update({
            f("p_text"): {"class_type": "CLIPTextEncode", "inputs": {"clip": ["61", 0], "text": PEOPLE_REPAINT_PROMPT}},
            f("p_zero"): {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [f("p_text"), 0]}},
        })
        for r in range(1, PEOPLE_REPAINT_ROUNDS + 1):
            p = lambda name: f(f"p{r}_{name}")  # noqa: E731,B023
            graph.update({
                p("find"): {"class_type": "GeoBlockFindPeople", "inputs": {
                    "image": plate, "model_path": request.person_detector, "grow": PEOPLE_MASK_GROW}},
                p("fill"): {"class_type": "GeoBlockFillPeople", "inputs": {"image": plate, "mask": [p("find"), 0]}},
                p("lat"): {"class_type": "VAEEncode", "inputs": {"pixels": [p("fill"), 0], "vae": ["62", 0]}},
                # No reference image: given the fill as Image 1, klein copied its blocks back (r17k).
                p("guider"): {"class_type": "CFGGuider", "inputs": {"model": ["63", 0], "positive": [f("p_text"), 0],
                                                                    "negative": [f("p_zero"), 0], "cfg": 1.0}},
                p("mask"): {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [p("lat"), 0],
                                                                          "mask": [p("find"), 0]}},
                p("noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + index + 2 + r}},
                p("sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0],
                                                                         "height": size[1]}},
                p("sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
                    "noise": [p("noise"), 0], "guider": [p("guider"), 0], "sampler": [f("sampler"), 0],
                    "sigmas": [p("sigmas"), 0], "latent_image": [p("mask"), 0]}},
                p("decode"): {"class_type": "VAEDecode", "inputs": {"samples": [p("sample"), 0], "vae": ["62", 0]}},
                p("paste"): {"class_type": "ImageCompositeMasked", "inputs": {
                    "destination": plate, "source": [p("decode"), 0], "x": 0, "y": 0, "resize_source": False,
                    "mask": [p("find"), 0]}},
            })
            _klein_from(graph, p(""), [p("sigmas"), 0], p("sample"), [p("mask"), 0], PEOPLE_REPAINT_DENOISE)
            plate = [p("paste"), 0]
    graph["_plates"] = graph.get("_plates", {}) | {index: plate}
    graph.update({
        f("h_place"): {"class_type": "GeoBlockPlaceSubject", "inputs": {
            "terrain_scene": ["3", 0], "shot": ["4", 0], "image": plate,
            "subject_image": graph["_subject_image"], "frame": index, "erase": False,
            "clean_plates": ["5", 5]}},  # for the stand-in's shadow
        f("h_placed"): {"class_type": "SaveImage", "inputs": {
            "images": [f("h_place"), 0], "filename_prefix": f"{request.output_prefix}_placed_{index:04d}"}},
        f("h_plate"): {"class_type": "SaveImage", "inputs": {
            "images": plate, "filename_prefix": f"{request.output_prefix}_plate_{index:04d}"}},
        f("h_prompt"): {"class_type": "GeoBlockKeyframePrompt", "inputs": {
            **graph[n[7]]["inputs"], "subject": [SUBJECT, 0], "reference": False, "placed": True}},
        f("h_text"): {"class_type": "CLIPTextEncode", "inputs": {"clip": ["61", 0], "text": [f("h_prompt"), 0]}},
        f("h_zero"): {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [f("h_text"), 0]}},
        f("h_scale"): {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [f("h_place"), 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
        f("h_lat"): {"class_type": "VAEEncode", "inputs": {"pixels": [f("h_scale"), 0], "vae": ["62", 0]}},
        f("h_img2"): {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": graph["_subject_image"], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
        f("h_lat2"): {"class_type": "VAEEncode", "inputs": {"pixels": [f("h_img2"), 0], "vae": ["62", 0]}},
    })
    positive, negative = [f("h_text"), 0], [f("h_zero"), 0]
    for k, latent in enumerate((f("h_lat"), f("h_lat2")), start=1):  # Image 1 the composite, Image 2 the sheet
        graph[f(f"h_pos{k}")] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive,
                                                                            "latent": [latent, 0]}}
        graph[f(f"h_neg{k}")] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative,
                                                                            "latent": [latent, 0]}}
        positive, negative = [f(f"h_pos{k}"), 0], [f(f"h_neg{k}"), 0]
    graph.update({
        f("h_guider"): {"class_type": "CFGGuider", "inputs": {"model": ["63", 0], "positive": positive,
                                                              "negative": negative, "cfg": 1.0}},
        f("h_sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}},
        f("h_noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + index + 1}},
        f("h_mask"): {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [f("h_lat"), 0],
                                                                    "mask": [f("h_place"), 1]}},
        f("h_sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [f("h_noise"), 0], "guider": [f("h_guider"), 0], "sampler": [f("sampler"), 0],
            "sigmas": [f("h_sigmas"), 0], "latent_image": [f("h_mask"), 0]}},
        f("h_decode"): {"class_type": "VAEDecode", "inputs": {"samples": [f("h_sample"), 0], "vae": ["62", 0]}},
    })
    denoise = request.keyframe_denoise if request.keyframe_denoise < 1.0 else COMPOSITE_DENOISE
    _klein_from(graph, f("h_"), [f("h_sigmas"), 0], f("h_sample"), [f("h_mask"), 0], denoise)
    graph[save]["inputs"]["images"] = [f("h_decode"), 0]
    graph[n[6]]["inputs"]["image"] = [f("h_decode"), 0]
    graph["_decoded"] = graph.get("_decoded", {}) | {index: [f("h_decode"), 0]}
    return [n[6], 0]


COMPOSITE_LEAD_FRAMES = 32  # composite keyframes start this far before the hold (blender_earth_zoom plates too)

KLEIN_DENOISE_STEPS = 20  # schedule length for a partial klein pass: denoise in steps of 0.05


def _klein_from(graph: dict, prefix: str, sigmas: list, sampler: str, latent: list, denoise: float) -> None:
    """Make klein's sampler start from ``latent`` partly noised instead of a blank: the last ``denoise`` of a
    KLEIN_DENOISE_STEPS schedule, as KSampler's denoise does. Cutting a 4-step schedule instead made 0.8, 0.9 and
    0.95 the same run (r8k)."""
    total = KLEIN_DENOISE_STEPS
    keep = min(total, max(1, round(total * denoise)))
    graph[sigmas[0]]["inputs"]["steps"] = total
    graph[f"{prefix}split"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": sigmas, "step": total - keep}}
    graph[sampler]["inputs"]["sigmas"] = [f"{prefix}split", 1]
    graph[sampler]["inputs"]["latent_image"] = latent


def _subject_sheet(graph: dict, request: ShotRequest) -> list:
    """FLUX.2 [klein]: one full-length picture of the person on a plain grey background (GeoBlockSubject's sheet
    prompt), edited from the user's photo or painted from the description. Every eye-level keyframe gets this
    instead of the raw photo: the photo's background and framing stay out of the scene, a head-and-shoulders photo
    still gives a whole person, and an invented person stays the same person across keyframes."""
    f = lambda name: f"172{name}"  # noqa: E731
    graph.update({
        f("text"): {"class_type": "CLIPTextEncode", "inputs": {"clip": ["61", 0], "text": [SUBJECT, 2]}},
        f("zero"): {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [f("text"), 0]}},
    })
    positive, negative = [f("text"), 0], [f("zero"), 0]
    if "171" in graph:  # the photo as the reference latent
        graph.update({
            f("scale"): {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": ["171", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
            f("lat"): {"class_type": "VAEEncode", "inputs": {"pixels": [f("scale"), 0], "vae": ["62", 0]}},
            f("pos"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [f("lat"), 0]}},
            f("neg"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [f("lat"), 0]}},
        })
        positive, negative = [f("pos"), 0], [f("neg"), 0]
    width, height = 832, 1248  # portrait, about 1 MP
    graph.update({
        f("latent"): {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height,
                                                                       "batch_size": 1}},
        f("sigmas"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        f("guider"): {"class_type": "CFGGuider", "inputs": {"model": ["63", 0], "positive": positive,
                                                            "negative": negative, "cfg": 1.0}},
        f("sampler"): {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        f("noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + 7}},
        f("sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [f("noise"), 0], "guider": [f("guider"), 0], "sampler": [f("sampler"), 0],
            "sigmas": [f("sigmas"), 0], "latent_image": [f("latent"), 0]}},
        "172": {"class_type": "VAEDecode", "inputs": {"samples": [f("sample"), 0], "vae": ["62", 0]}},
        SUBJECT_SHEET: {"class_type": "SaveImage", "inputs": {"images": ["172", 0],
                                                              "filename_prefix": f"{request.output_prefix}_subject"}},
    })
    return ["172", 0]


def _qwen_keyframe(graph: dict, request: ShotRequest, frames: list, index: int, base: int, save: str,
                   reference: list | None = None) -> list:
    """Repaint one blocking frame as a photograph; returns the IMAGE output. Uses node IDs base..base+7.

    ``reference`` is an earlier repainted keyframe of the same spot, given as Image 2 so trees, ground,
    person and background match it instead of being reinvented (the subject photo then becomes Image 3).
    """
    n = [str(base + i) for i in range(8)]
    graph.update({
        n[7]: {"class_type": "GeoBlockKeyframePrompt", "inputs": {
            "terrain_scene": ["3", 0], "shot": ["4", 0], "frame": index, "palette": request.palette,
            "weather": request.weather, "reference": reference is not None,
            **({"subject": [SUBJECT, 0]} if SUBJECT in graph else {}),
            **({"canopy": request.canopy} if request.canopy else {})}},
        n[0]: {"class_type": "ImageFromBatch", "inputs": {"image": frames, "batch_index": index, "length": 1}},
        n[1]: {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["61", 0], "prompt": [n[7], 0], "vae": ["62", 0], "image1": [n[0], 0]}},
        n[2]: {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["61", 0], "prompt": "", "vae": ["62", 0], "image1": [n[0], 0]}},
        n[3]: {"class_type": "VAEEncode", "inputs": {"pixels": [n[0], 0], "vae": ["62", 0]}},
    })
    extra = ([reference] if reference is not None else []) + (
        [graph["_subject_image"]] if "_subject_image" in graph and index >= _arrive_frame(request) else [])
    for slot, image in zip(("image2", "image3"), extra):
        for node in (n[1], n[2]):
            graph[node]["inputs"][slot] = image
    graph.update({
        n[4]: {"class_type": "KSampler", "inputs": {
            "model": ["64", 0], "positive": [n[1], 0], "negative": [n[2], 0], "latent_image": [n[3], 0],
            "seed": request.seed + index, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0}},
        n[5]: {"class_type": "VAEDecode", "inputs": {"samples": [n[4], 0], "vae": ["62", 0]}},
        save: {"class_type": "SaveImage", "inputs": {"images": [n[5], 0],
                                                     "filename_prefix": f"{request.output_prefix}_keyframe_{index:04d}"}},
        n[6]: {"class_type": "LTXVPreprocess", "inputs": {"image": [n[5], 0], "img_compression": 18}},
    })
    graph["_decoded"] = graph.get("_decoded", {}) | {index: [n[5], 0]}
    return [n[6], 0]


def _add_ltx(graph: dict, request: ShotRequest, video: list) -> dict:
    """Append photoreal keyframes and the LTX render-to-real stage (2.3 + 3DREAL, or 2.5 + Layout To Render)."""
    if request.keyframe_model not in KEYFRAME_MODELS or request.ltx_version not in LTX_VERSIONS:
        raise ValueError(f"keyframe_model must be one of {KEYFRAME_MODELS}, ltx_version one of {LTX_VERSIONS}.")
    m, length = request.ltx_models, ltx_frame_count(request.duration_s)
    ckpt = m["checkpoint"]
    sheet = request.subject_sheet and request.keyframe_model == "flux-klein"
    if request.preset == "Earth zoom":  # it lands on a person: the user's photo, or an invented figure
        graph[SUBJECT] = {"class_type": "GeoBlockSubject", "inputs": {
            "description": request.subject_description, "weather": request.weather, "subject_sheet": sheet,
            **({"chroma": True} if sheet and (request.place_subject or request.composite_subject) else {})}}
        if request.subject_photo is not None:
            graph["171"] = {"class_type": "LoadImage", "inputs": {"image": request.subject_photo}}
            graph[SUBJECT]["inputs"]["photo"] = ["171", 0]
            graph["_subject_image"] = ["171", 0]
    graph["60"] = {"class_type": "GetVideoComponents", "inputs": {"video": video}}
    if request.keyframe_model == "flux-klein":
        graph.update({
            "61": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2",
                                                         "device": "default"}},
            "62": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
            "63": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        })
    else:
        graph.update({
            # Qwen-Image-Edit 2511 with the 4-step Lightning LoRA: it keeps the frame's geometry; 24-step CFG
            # sampling looks sharper but moves the horizon and invents landforms.
            "61": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image",
                                                         "device": "default"}},
            "62": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
            "63": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
            "64": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["65", 0], "shift": 3.1}},
            "65": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["63", 0], "lora_name": m["qwen_lora"],
                                                                  "strength_model": 1.0}},
        })
    if sheet and SUBJECT in graph:
        graph["_subject_image"] = _subject_sheet(graph, request)
    first = _keyframe(graph, request, ["60", 0], 0, 66, LTX_KEYFRAMES[0])
    plan = [(k, 8 * round(fraction * (length - 1) / 8), source)
            for k, (fraction, source) in enumerate(_mid_keyframes(request))]
    if request.preset == "Earth zoom" and request.exit_keyframe != "none":
        # r22: once the blocking stopped being solid white (f36-37), LTX cut to a clear oblique view like the f56
        # keyframe, 2-3 frames ahead of the blocking's haze clearing, in stage 1 already. A guide at the deck's
        # exit holds it to the half-cleared view. Appended, so the other keyframes keep their node ids.
        plan.append((len(plan), _exit_frame(request), request.exit_keyframe))
    if request.preset == "Earth zoom" and request.cloud_keyframe != "repaint":
        # r5: pinned to klein's one cloud picture at f32, LTX froze the dive for most of a second (frame-to-frame
        # change ~0.2 against ~20 in the blocking), then cut to clear air. Left to the blocking it flies through.
        cloud = _cloud_frame(request)
        plan = [(k, index, "blocking" if index == cloud else source) for k, index, source in plan
                if index != cloud or request.cloud_keyframe == "blocking"]
    if request.composite_subject and SUBJECT in graph:
        # The keyframe just before the hold is painted after it, from it ("descent"), so the two agree.
        plan = [(k, index, "descent" if source == "repaint" and
                 _arrive_frame(request) - COMPOSITE_LEAD_FRAMES <= index < _arrive_frame(request) else source)
                for k, index, source in plan]
    if request.hold_keyframes == "drop" and SUBJECT in graph:  # pin only the first keyframe of the hold
        held = [index for _, index, source in plan if source == "repaint" and index >= _arrive_frame(request)]
        plan = [item for item in plan if not held or item[1] <= held[0]]
    if request.hold_keyframes == "dense" and SUBJECT in graph:
        # Between f112 and f176 LTX drifted away from the keyframes' city and snapped back at f170-176, sharp
        # crop or not; dropping f176 let the hiker walk off. Hold keyframes every HOLD_SPACING frames (each a
        # sharpened crop of the first) leave it no room to drift.
        held = [index for _, index, source in plan if source == "repaint" and index >= _arrive_frame(request)]
        if held:
            extra = range(held[0] + HOLD_SPACING, length - HOLD_SPACING // 2, HOLD_SPACING)
            plan = [item for item in plan if item[1] <= held[0]] + [
                (len(plan) + n, index, "repaint") for n, index in enumerate(extra)]
    if request.preset == "Earth zoom" and request.swoop_gap > 0:
        plan += [(max(k for k, _, _ in plan) + 1 + n, index, "descent")
                 for n, index in enumerate(_swoop_fill(request, [index for _, index, _ in plan]))]
    images, anchor = {}, None  # anchor: the first repainted keyframe of the eye-level hold
    # "descent" keyframes are painted last, from the next keyframe down, so the two show the same ground.
    for k, index, source in sorted(plan, key=lambda item: item[2] == "descent"):
        if source in ("repaint", "descent"):
            held = SUBJECT in graph and index >= _arrive_frame(request)
            reference = (*anchor, "hold") if held and anchor else None
            if source == "descent":
                below = next(i for _, i, s in plan if i > index and s == "repaint")
                reference = (graph["_decoded"][below], below, "descent")
                if request.landing_from_hold and request.composite_subject and SUBJECT in graph and (
                        _arrive_frame(request) - COMPOSITE_LEAD_FRAMES <= index < _arrive_frame(request)):
                    # The landing plate (f88) given the hold's first plate whole put that plate's oak right of
                    # centre; the hold has it on the left, so LTX swept it across the frame (r33-r41 f91-100). Moved
                    # back through the terrain to this camera, it is the same place from where this camera stands.
                    reference = (graph.get("_plates", {}).get(below, graph["_decoded"][below]), below, "landing")
            if held and anchor and request.hold_keyframes == "direct":
                images[k] = _cropped_hold(graph, request, graph["_decoded"][anchor[1]], anchor[1], index, _keyframe_base(k),
                                          LTX_KEYFRAMES[2 + k])
                continue
            images[k] = _keyframe(graph, request, ["60", 0], index, _keyframe_base(k), LTX_KEYFRAMES[2 + k], reference)
            if held and (anchor is None or request.hold_follow == "chain"):
                # later hold keyframes follow it (its empty plate, when compositing); "chain": each follows the one
                # before, so every step moves the camera only a few metres and the painter cleans up as it goes
                anchor = (graph.get("_plates", {}).get(index, graph["_decoded"][index]), index)
        else:
            base = _keyframe_base(k)
            graph[str(base)] = {"class_type": "ImageFromBatch", "inputs": {
                "image": ["60", 0], "batch_index": index, "length": 1}}
            graph[str(base + 1)] = {"class_type": "LTXVPreprocess", "inputs": {
                "image": [str(base), 0], "img_compression": 18}}
            images[k] = [str(base + 1), 0]
    mids = [(index, images[k]) for k, index, _ in plan]
    if request.orbit_from_blocking and request.preset == "Earth zoom":
        # klein drew a thin straight line across the clouds of the f0 orbit keyframe (Twin Peaks), which the
        # blocking frame, real Blue Marble imagery, doesn't have; LTX copied it. klein's frame stays the
        # look reference, and the blocking frame pins frame 0 itself.
        graph["198"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["60", 0], "batch_index": 0, "length": 1}}
        graph["199"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["198", 0], "img_compression": 18}}
        mids = [(0, ["199", 0]), *mids]
    last = _keyframe(graph, request, ["60", 0], length - 1, 74, LTX_KEYFRAMES[1], anchor and (*anchor, "hold"))
    graph.pop("_decoded", None)
    graph.pop("_plates", None)
    graph.pop("_stretch", None)
    graph.pop("_kept", None)
    graph.pop("_unseen", None)
    graph.pop("_subject_image", None)
    if request.keyframes_only:
        return graph
    prompt = request.prompt or render_to_real_prompt(
        request.name, request.palette, PRESETS[request.preset], weather_prompt(request.weather), request.preset,
        look=request.look, weather=request.weather, canopy=request.canopy)
    if request.ltx_version == "2.5":
        return _add_ltx25(graph, request, length, first, last, mids, prompt.removeprefix("3DREAL. "))
    # Guides and keyframes stay at strength 1.0: below that LTX's masked attention fails in cuDNN.
    graph.update({
        "100": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "101": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["100", 0], "lora_name": m["distilled_lora"],
                                                               "strength_model": 1.0}},
        "102": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["101", 0], "lora_name": m["ic_lora"],
                                                               "strength_model": 1.0}},
        "103": {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": ["102", 0]}},
        "104": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": m["text_encoder"],
                                                                  "ckpt_name": ckpt, "device": "default"}},
        "105": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["104", 0], "text": prompt}},
        "106": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["104", 0], "text": (
            "blurry, low quality, flicker, morphing, cuts, watermark, text, cartoon, CGI, clay, plastic, untextured")}},
        "107": {"class_type": "LTXVConditioning", "inputs": {"positive": ["105", 0], "negative": ["106", 0],
                                                            "frame_rate": 24}},
        "108": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 960, "height": 544, "length": length,
                                                                "batch_size": 1}},
        "109": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["100", 2], "image": first,
                                                                 "latent": ["108", 0], "strength": 1.0, "bypass": False}},
        "110": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["107", 0], "negative": ["107", 1], "vae": ["100", 2], "latent": ["109", 0],
            "image": ["60", 0], "frame_idx": 0, "strength": 1.0, "iclora_parameters": ["103", 0]}},
        "111": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["110", 0], "negative": ["110", 1], "vae": ["100", 2], "latent": ["110", 2],
            "image": last, "frame_idx": -1, "strength": 1.0}},
        "112": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": ckpt}},
        "113": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": length, "frame_rate": 24,
                                                                "batch_size": 1, "audio_vae": ["112", 0]}},
        "114": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["111", 2], "audio_latent": ["113", 0]}},
        "115": {"class_type": "KSampler", "inputs": {
            "model": ["102", 0], "positive": ["111", 0], "negative": ["111", 1], "latent_image": ["114", 0],
            "seed": request.seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler_ancestral",
            "scheduler": "linear_quadratic", "denoise": 1.0}},
        "116": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["115", 0]}},
        "117": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["111", 0], "negative": ["111", 1],
                                                          "latent": ["116", 0]}},
        "118": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["117", 2], "vae": ["100", 2], "tile_size": 768,
                                                          "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
        "119": {"class_type": "CreateVideo", "inputs": {"images": ["118", 0], "fps": 24}},
        LTX_VIDEO: {"class_type": "SaveVideo", "inputs": {"video": ["119", 0], "filename_prefix": request.output_prefix
                                                          + "_ltx", "format": "auto", "codec": "auto"}},
    })
    guide = "110"
    for k, (index, image) in enumerate(mids):
        node = str(140 + k)
        graph[node] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": [guide, 0], "negative": [guide, 1], "vae": ["100", 2], "latent": [guide, 2],
            "image": image, "frame_idx": index, "strength": 1.0}}
        guide = node
    for key in ("positive", "negative", "latent"):
        graph["111"]["inputs"][key] = [guide, {"positive": 0, "negative": 1, "latent": 2}[key]]
    if request.ltx_upscale:  # fal's two-stage recipe: 2x latent upscale, re-pin frame 0, 3-step refine
        graph.update({
            "120": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": m["upscaler"]}},
            "121": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["117", 2], "upscale_model": ["120", 0],
                                                                   "vae": ["100", 2]}},
            "122": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["100", 2], "image": first,
                                                                     "latent": ["121", 0], "strength": 1.0,
                                                                     "bypass": False}},
            "123": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["122", 0],
                                                                  "audio_latent": ["116", 1]}},
            "124": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}},
            "125": {"class_type": "CFGGuider", "inputs": {"model": ["102", 0], "positive": ["117", 0],
                                                         "negative": ["117", 1], "cfg": 1.0}},
            "126": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "127": {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed + 1}},
            "128": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["127", 0], "guider": ["125", 0], "sampler": ["126", 0], "sigmas": ["124", 0],
                "latent_image": ["123", 0]}},
            "129": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["128", 0]}},
        })
        # Re-pin the last and mid keyframes at the new size: without them the refine pass drifts back to
        # whatever the prompt suggests (bushes on land seen from orbit in the Earth zoom).
        cond, latent = ["117", 0], ["122", 0]
        negative = ["117", 1]
        for k, (index, image) in enumerate([*mids, (-1, last)]):
            node = str(150 + k)
            graph[node] = {"class_type": "LTXVAddGuide", "inputs": {
                "positive": cond, "negative": negative, "vae": ["100", 2], "latent": latent,
                "image": image, "frame_idx": index, "strength": 1.0}}
            cond, negative, latent = [node, 0], [node, 1], [node, 2]
        graph["123"]["inputs"]["video_latent"] = latent
        graph["125"]["inputs"]["positive"], graph["125"]["inputs"]["negative"] = cond, negative
        graph["130"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": cond, "negative": negative,
                                                                  "latent": ["129", 0]}}
        graph["118"]["inputs"]["samples"] = ["130", 2]
    return graph


def _add_ltx25(graph: dict, request: ShotRequest, length: int, first: list | None, last: list, mids: list,
               prompt: str, layout: bool = True) -> dict:
    """LTX 2.5 + Layout To Render on the blocking video, wired like Lightricks'
    LTX-2.5_ICLoRA_Layout_To_Render_Two_Stage_Distilled.json: a video-only latent; the blocking goes in as the
    IC-LoRA guide at frame 0 and the photoreal first frame as a one-frame IC-LoRA reference at RoPE position -1
    (centre crop); stage 2 starts again from the text conditioning and re-adds both guides at full size on the
    upscaled latent. The mid and last keyframes are pinned with LTXVAddGuide on both stages ("any keyframe
    provided is used on both stages"). Node IDs 300-379.

    layout=False is plain keyframe image-to-video on the same models (no IC-LoRA, no blocking video): only the
    ``mids`` are pinned, frame 0 included, and ``first`` is unused (scripts/fakeout_earth_zoom.py)."""
    m = request.ltx25_models
    vae = ["302", 0]
    use_layout = layout
    model = ["301", 0] if layout else ["300", 0]

    def guides(cond: list, negative: list, latent: list, factor, base: int, layout_guide: bool = True):
        """Layout guide, first-frame reference, then the other keyframes; returns the last node's outputs."""
        layout, look = str(base), str(base + 1)
        common = {"vae": vae, "strength": 1.0, "latent_downscale_factor": factor, "use_tiled_encode": False,
                  "tile_size": 256, "tile_overlap": 64}
        if layout_guide and use_layout:
            graph[layout] = {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
                "positive": cond, "negative": negative, "latent": latent, "image": ["60", 0], "frame_idx": 0,
                "crop": "disabled", **common}}
            cond, negative, latent = [layout, 0], [layout, 1], [layout, 2]
        if not use_layout:
            return pins(cond, negative, latent, base)
        graph[look] = {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": cond, "negative": negative, "latent": latent, "image": first,
            "frame_idx": -1, "crop": "center", **common}}
        return pins([look, 0], [look, 1], [look, 2], base)

    def pins(cond: list, negative: list, latent: list, base: int):
        # pin_last_keyframe=False leaves the end to the layout: a pinned last keyframe holds the person in the
        # pose the keyframes share, so the Earth zoom's hiker stopped moving over the last seconds.
        pin_last = request.pin_last_keyframe if request.pin_last_keyframe is not None else request.preset != "Earth zoom"
        for k, (index, image) in enumerate([*mids, *([(-1, last)] if pin_last else [])]):
            node = str(base + 2 + k)
            graph[node] = {"class_type": "LTXVAddGuide", "inputs": {
                "positive": cond, "negative": negative, "vae": vae, "latent": latent, "image": image,
                "frame_idx": index, "strength": 1.0}}
            cond, negative, latent = [node, 0], [node, 1], [node, 2]
        return cond, negative, latent

    def sample(cond: list, negative: list, latent: list, sigmas: str, seed: int, base: int) -> list:
        """Distilled sampling (CFG 1, euler_ancestral), then drop the guide frames; returns the latent."""
        n = [str(base + i) for i in range(6)]
        graph.update({
            n[0]: {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            n[1]: {"class_type": "CFGGuider", "inputs": {"model": model, "positive": cond,
                                                        "negative": negative, "cfg": 1.0}},
            n[2]: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
            n[3]: {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}},
            n[4]: {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": [n[0], 0], "guider": [n[1], 0], "sampler": [n[2], 0], "sigmas": [n[3], 0],
                "latent_image": latent}},
            n[5]: {"class_type": "LTXVCropGuides", "inputs": {"positive": cond, "negative": negative,
                                                             "latent": [n[4], 0]}},
        })
        return [n[5], 2]

    graph.update({
        "300": {"class_type": "UNETLoader", "inputs": {"unet_name": m["transformer"], "weight_dtype": "default"}},
        **({"301": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
            "model": ["300", 0], "lora_name": m["ic_lora"], "strength_model": 1.0}}} if layout else {}),
        "302": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "304": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "ltxv",
                                                      "device": getattr(request, "ltx_encoder_device", "cpu")}},
        "305": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["304", 0], "text": prompt}},
        "306": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["304", 0], "text": (
            "blurry, low quality, flicker, morphing, cuts, watermark, text, cartoon, CGI, clay, plastic, untextured")}},
        "307": {"class_type": "LTXVConditioning", "inputs": {"positive": ["305", 0], "negative": ["306", 0],
                                                            "frame_rate": 24}},
        "308": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 960, "height": 544, "length": length,
                                                                "batch_size": 1}},
    })
    video = sample(*guides(["307", 0], ["307", 1], ["308", 0], ["301", 1] if layout else 1, 310), (
        "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"), request.seed, 390)  # clear of up to 18 guides from 312
    if request.save_stage1 and request.ltx_upscale:
        # Is a smear or scenery swap made in stage 1 or by the refine (which runs without the layout guide past
        # 121 frames)? The half-size stage 1, decoded on its own.
        graph.update({
            "375": {"class_type": "VAEDecodeTiled", "inputs": {"samples": video, "vae": vae, "tile_size": 512,
                                                              "overlap": 64, "temporal_size": 128,
                                                              "temporal_overlap": 32}},
            "376": {"class_type": "CreateVideo", "inputs": {"images": ["375", 0], "fps": 24}},
            STAGE1_VIDEO: {"class_type": "SaveVideo", "inputs": {"video": ["376", 0], "format": "auto", "codec": "auto",
                                                          "filename_prefix": request.output_prefix + "_stage1"}},
        })
    if request.ltx_upscale:  # stage 2: 2x latent upscale, guides again at full size, 3 refine steps
        graph.update({
            "330": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": m["upscaler"]}},
            "331": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": video, "upscale_model": ["330", 0],
                                                                   "vae": vae}},
        })
        # The full-size layout guide doubles stage 2's tokens: past 5 s (121 frames) it runs a 24 GB card out of
        # memory, so longer clips refine without it, as ltx_pipelines.ic_lora does by default (no
        # --stage-2-ic-lora); the first-frame reference and keyframes still hold the look.
        video = sample(*guides(["307", 0], ["307", 1], ["331", 0], 1.0, 340, layout_guide=length <= 121),
                       "0.909375, 0.725, 0.421875, 0.0", request.seed + 1, 360)
    # Lightricks' low-VRAM decode settings (512 px tiles; 768 reportedly stalls 10 s clips on 24 GB).
    graph.update({
        "370": {"class_type": "VAEDecodeTiled", "inputs": {"samples": video, "vae": vae, "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 128, "temporal_overlap": 32}},
        "371": {"class_type": "CreateVideo", "inputs": {"images": ["370", 0], "fps": 24}},
        LTX_VIDEO: {"class_type": "SaveVideo", "inputs": {"video": ["371", 0], "filename_prefix": request.output_prefix
                                                          + "_ltx25", "format": "auto", "codec": "auto"}},
    })
    return graph


def ltx_graph(request: ShotRequest) -> dict:
    """The blocking graph plus photoreal keyframes and LTX 3DREAL render-to-real on its outputs, then the
    optional final upscale on the decoded LTX frames (before any video compression)."""
    graph = _add_ltx(blocking_graph(request), request, ["5", 0])
    smooth = request.colour_smooth_s if request.colour_smooth_s is not None else (
        0.75 if request.preset == "Earth zoom" else 0.0)
    if smooth > 0 and LTX_VIDEO in graph:  # even out what white-balance drift the keyframe colour match leaves
        video = graph[graph[LTX_VIDEO]["inputs"]["video"][0]]
        graph["380"] = {"class_type": "GeoBlockColourSmooth", "inputs": {
            "images": video["inputs"]["images"], "sigma_s": smooth, "fps": video["inputs"]["fps"]}}
        video["inputs"]["images"] = ["380", 0]
    if request.punch_through and request.preset == "Earth zoom" and LTX_VIDEO in graph:
        video = graph[graph[LTX_VIDEO]["inputs"]["video"][0]]
        last = h3_frame_count(request.duration_s) - 1
        entry = round(cloud_entry_fraction(tempo=request.dive_tempo) * last)
        leave = round(height_fraction(CLOUD_BASE_M, tempo=request.dive_tempo) * last)
        graph["381"] = {"class_type": "GeoBlockPunchThrough", "inputs": {
            "images": video["inputs"]["images"], "start_frame": entry, "end_frame": leave + 4, "span": 6}}
        video["inputs"]["images"] = ["381", 0]
    if request.final_upscale != "none":
        frames = graph[graph[LTX_VIDEO]["inputs"]["video"][0]]["inputs"]["images"]
        add_final_upscale(graph, frames, method=request.final_upscale, target=request.final_target,
                          seed=request.seed,
                          prefix=f"{request.output_prefix}_final_{request.final_upscale}_{request.final_target}",
                          ltx_models=request.ltx25_models)
    return graph
