"""Optional final upscale / detail pass on a finished GeoBlock video.

The LTX stage ends at 1920x1088. This stage takes those frames (straight from the decode in the same graph, or
from a saved video) and either rebuilds detail at the same size ("1080p") or doubles it ("4k", 3840x2176).

Upscalers, all local and openly licensed:

- ``seedvr2``: ByteDance SeedVR2 3B (fp8), one-step diffusion video restoration (Apache 2.0), through
  numz/ComfyUI-SeedVR2_VideoUpscaler (Apache 2.0). Frames go through in overlapping temporal batches.
- ``seedvr2-7b``: the same with the 7B model (fp8 with the last block in fp16).
- ``flashvsr``: FlashVSR v1.1, one-step streaming diffusion VSR on Wan 2.1 1.3B (Apache 2.0), through
  lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast (GPL-3.0). 4x only, so the frames are halved first; 4k target only
  (its 128 px alignment would crop 1088 rows to 1024).
- ``ltx-detail``: LTX 2.5 with Lightricks' Refine Details IC-LoRA on their tiled-fusion sampler, as in
  ComfyUI-LTXVideo's LTX-2.5_V2V_TiledFusion_Upscale.json (LTX-2 Community License). Re-renders the clip with
  8 distilled steps in overlapping tiles; the slowest by far.
- ``esrgan``: Real-ESRGAN x2plus frame by frame (BSD-3), the classic baseline. No temporal model.

The "1080p" target halves the frames first and upscales 2x back, so the model rebuilds the texture instead of
magnifying the softness LTX left in it. Node IDs 400-499.
"""

from __future__ import annotations

UPSCALERS = ("seedvr2", "seedvr2-7b", "flashvsr", "ltx-detail", "esrgan")
TARGETS = ("1080p", "4k")
FINAL_VIDEO = "490"

SEEDVR2_MODELS = {
    "seedvr2": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2-7b": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "vae": "ema_vae_fp16.safetensors",
}
ESRGAN_MODEL = "RealESRGAN_x2plus.pth"
LTX_DETAIL_LORA = "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"
# Look and style only: anything named here can stamp into every tile (Lightricks' note on the upscale graph).
LTX_DETAIL_PROMPT = ("High resolution aerial photograph in motion, crisp natural fine texture, clean sharp edges, "
                     "natural colour and contrast, clear air, real camera detail.")
LTX_DETAIL_NEGATIVE = ("blurry, soft, smeared, oversharpened, halos, ringing, noise, grain, flicker, watermark, "
                       "text, cartoon, CGI, plastic")
# Lightricks' distilled schedule, 8 steps from pure noise (the source video is the IC-LoRA guide).
LTX_DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def seedvr2_batch(target: str) -> int:
    """Frames per SeedVR2 batch (4n+1). Bigger batches flicker less, but on a 24 GB card shared with a desktop
    (about 14 GB usable) 21 frames at 4k and 61 at 1080p ran out of memory."""
    return 9 if target == "4k" else 21


def prescale(method: str, target: str) -> float:
    """How much to shrink the frames before the upscaler: half for the 1080p detail rebuild and for FlashVSR
    (always 4x); ltx-detail re-renders at the size it is given."""
    return 0.5 if method != "ltx-detail" and (target == "1080p" or method == "flashvsr") else 1.0


def output_height(source_height: int, target: str) -> int:
    return source_height * 2 if target == "4k" else source_height


def check(method: str, target: str) -> None:
    if method not in UPSCALERS:
        raise ValueError(f"Unknown upscaler {method!r}; choose from {', '.join(UPSCALERS)}.")
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; choose from {', '.join(TARGETS)}.")
    if method == "flashvsr" and target != "4k":
        raise ValueError("flashvsr only runs to 4k (it crops 1088-row frames to 1024 at 1080p).")


def add_final_upscale(graph: dict, images: list, *, method: str, target: str, prefix: str, seed: int = 20261005,
                      fps: float = 24, batch_size: int = 0, blocks_to_swap: int = 0,
                      ltx_models: dict | None = None) -> dict:
    """Upscale the IMAGE batch ``images`` and save it as FINAL_VIDEO."""
    check(method, target)
    # GeoBlock 8 frees the LTX weights from VRAM (SeedVR2 and FlashVSR load outside ComfyUI's model manager)
    # and halves the frames when the upscaler should rebuild them from half size.
    graph["400"] = {"class_type": "GeoBlockFinalUpscalePrep", "inputs": {
        "images": images, "upscaler": method, "target": target, "free_vram": True}}
    images, resolution = ["400", 0], ["400", 1]
    if method == "ltx-detail":
        frames = _ltx_detail(graph, images, target, seed, ltx_models)
    else:
        if method == "esrgan":
            graph.update({
                "410": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": ESRGAN_MODEL}},
                "411": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["410", 0],
                                                                           "image": images}},
            })
            frames = ["411", 0]
        elif method == "flashvsr":
            graph["420"] = {"class_type": "FlashVSRNode", "inputs": {
                "frames": images, "model": "FlashVSR-v1.1", "mode": "tiny-long", "scale": 4, "tiled_vae": True,
                "tiled_dit": True, "unload_dit": True, "seed": seed % 2**32}}
            frames = ["420", 0]
        else:
            batch = batch_size or seedvr2_batch(target)
            graph.update({
                "430": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
                    "model": SEEDVR2_MODELS[method], "device": "cuda:0", "blocks_to_swap": blocks_to_swap,
                    "swap_io_components": False, "offload_device": "cpu" if blocks_to_swap else "none",
                    "cache_model": False, "attention_mode": "sdpa"}},
                "431": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
                    "model": SEEDVR2_MODELS["vae"], "device": "cuda:0", "encode_tiled": True,
                    "encode_tile_size": 512, "encode_tile_overlap": 64, "decode_tiled": True,
                    "decode_tile_size": 512, "decode_tile_overlap": 64, "tile_debug": "false",
                    "offload_device": "none", "cache_model": False}},
                "432": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
                    "image": images, "dit": ["430", 0], "vae": ["431", 0], "seed": seed % 2**32,
                    "resolution": resolution, "max_resolution": 0, "batch_size": batch, "uniform_batch_size": False,
                    "temporal_overlap": min(4, batch - 1), "prepend_frames": 0, "color_correction": "lab",
                    "input_noise_scale": 0.0, "latent_noise_scale": 0.0, "offload_device": "cpu",
                    "enable_debug": False}},
            })
            frames = ["432", 0]
    graph.update({
        "480": {"class_type": "CreateVideo", "inputs": {"images": frames, "fps": fps}},
        FINAL_VIDEO: {"class_type": "SaveVideo", "inputs": {"video": ["480", 0], "filename_prefix": prefix,
                                                             "format": "auto", "codec": "auto"}},
    })
    return graph


def _ltx_detail(graph: dict, images: list, target: str, seed: int, ltx_models: dict | None) -> list:
    """Lightricks' LTX-2.5_V2V_TiledFusion_Upscale graph with HD (1280 px) tiles; returns the decoded frames."""
    from .workflow import LTX25_MODELS
    m = ltx_models or LTX25_MODELS
    sizes, guide = ["450", 0], "456"
    graph.update({
        "440": {"class_type": "UNETLoader", "inputs": {"unet_name": m["transformer"], "weight_dtype": "default"}},
        "441": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"model": ["440", 0], "lora_name": LTX_DETAIL_LORA,
                                                                    "strength_model": 1.0}},
        "442": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "443": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "ltxv",
                                                      "device": "default"}},
        "444": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["443", 0], "text": LTX_DETAIL_PROMPT}},
        "445": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["443", 0], "text": LTX_DETAIL_NEGATIVE}},
        "446": {"class_type": "LTXVConditioning", "inputs": {"positive": ["444", 0], "negative": ["445", 0],
                                                            "frame_rate": 24}},
        "449": {"class_type": "GetImageSize", "inputs": {"image": images}},
        "450": {"class_type": "LTXVGetTilingSizes", "inputs": {
            "width": ["449", 0], "height": ["449", 1], "frame_count": ["449", 2], "tile_size": "HD",
            "initial_canvas_size": "FullHD", "output_size": "4K" if target == "4k" else "FullHD", "image": images}},
        "451": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": [sizes[0], 4], "height": [sizes[0], 5],
                                                                "length": ["449", 2], "batch_size": 1}},
        "452": {"class_type": "ImageScale", "inputs": {"image": images, "upscale_method": "lanczos",
                                                      "width": [sizes[0], 4], "height": [sizes[0], 5],
                                                      "crop": "disabled"}},
        guide: {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": ["446", 0], "negative": ["446", 1], "vae": ["442", 0], "latent": ["451", 0],
            "image": ["452", 0], "frame_idx": 0, "strength": 1.0, "latent_downscale_factor": ["441", 1],
            "crop": "disabled", "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64,
            "use_streaming": True, "tile_frames": [sizes[0], 7]}},
        "457": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "458": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_DISTILLED_SIGMAS}},
        "459": {"class_type": "LTXVTiledFusionSampler", "inputs": {
            "model": ["441", 0], "positive": [guide, 0], "negative": [guide, 1], "latents": [guide, 2],
            "sigmas": ["458", 0], "sampler": ["457", 0], "seed": seed, "cfg": 1.0,
            "tile_width": [sizes[0], 2], "tile_height": [sizes[0], 3], "overlap_frac": [sizes[0], 6],
            "blend_var": 0.05, "grid_cycle": 1, "tile_frames": [sizes[0], 7], "vae": ["442", 0],
            "canvas_device": "auto"}},
        "460": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["459", 0], "vae": ["442", 0], "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 128, "temporal_overlap": 32}},
    })
    return ["460", 0]


def upscale_file_graph(video, *, method: str, target: str, prefix: str, start: int = 0, frames: int = 0,
                       **options) -> dict:
    """Upscale a saved video (a LoadVideo value), optionally only ``frames`` frames from ``start``."""
    graph = {
        "401": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "402": {"class_type": "GetVideoComponents", "inputs": {"video": ["401", 0]}},
    }
    images = ["402", 0]
    if frames:
        graph["403"] = {"class_type": "ImageFromBatch", "inputs": {"image": images, "batch_index": start,
                                                                  "length": frames}}
        images = ["403", 0]
    return add_final_upscale(graph, images, method=method, target=target, prefix=prefix, **options)
