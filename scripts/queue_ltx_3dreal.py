"""Queue an LTX 2.3 + 3DREAL IC-LoRA render-to-real job on local ComfyUI using stock nodes only.

usage: python scripts/queue_ltx_3dreal.py <clay video in Comfy input dir> <prefix> [--frames 121] [--width 960] [--height 544]
       [--keyframe img.png] [--lora 3DREAL-strong-v2.safetensors] [--strength 1.0] [--steps 8] [--seed 20261005]

Guides and keyframes stay at strength 1.0: below that, LTX's masked attention fails in cuDNN ("No valid execution
plans built") and can take ComfyUI down with it (docs/render-to-real-tests.md). The strength flags are kept for
experiments; 1.0 is the default on purpose.
"""
import argparse, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoblock_core.weather import WEATHER, weather_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("prefix")
ap.add_argument("--frames", type=int, default=121)
ap.add_argument("--width", type=int, default=960); ap.add_argument("--height", type=int, default=544)
ap.add_argument("--upscale", action="store_true", help="2x latent upscale + 3-step refine (fal two-stage)"); ap.add_argument("--keyframe"); ap.add_argument("--end-keyframe"); ap.add_argument("--end-strength", type=float, default=1.0); ap.add_argument("--keyframe-strength", type=float, default=1.0)
ap.add_argument("--lora", default="3DREAL-strong-v2.safetensors")
ap.add_argument("--strength", type=float, default=1.0)
ap.add_argument("--guide-strength", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=8); ap.add_argument("--cfg", type=float, default=1.0); ap.add_argument("--distilled", type=float, default=1.0); ap.add_argument("--sampler", default="euler_ancestral"); ap.add_argument("--seed", type=int, default=20261005)
ap.add_argument("--prompt", default=(
    "3DREAL. Make it photorealistic. A drone flies steadily forward over the South Rim of the Grand Canyon at "
    "Mather Point, Arizona, then glides out past the rim edge as the ground drops away to reveal the canyon. "
    "In the first two seconds the camera skims low over flat rim plateau covered in dark green pinyon pine and "
    "juniper trees with pale limestone soil between them. At the rim the cream Kaibab limestone cliff falls away "
    "and the view opens onto vast eroded buttes and temples banded in red Supai sandstone, buff Coconino "
    "sandstone and grey-green slopes, with deep shadowed side canyons between them. The flat forested North Rim "
    "runs level along the far horizon. Real photographic detail, "
    "natural colour. No water, no snow, no people, no buildings, no text."))
ap.add_argument("--weather", choices=tuple(WEATHER), default="clear", help="Append this weather to the prompt; match the blocking render")
a = ap.parse_args()
if a.weather:
    a.prompt = a.prompt.replace(" {photo}", " " + weather_prompt(a.weather) + " {photo}") if "{photo}" in a.prompt else a.prompt + " " + weather_prompt(a.weather)

CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
g = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors", "strength_model": a.distilled}},
    "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": a.lora, "strength_model": a.strength}},
    "4": {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": ["3", 0]}},
    "5": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp8_scaled.safetensors", "ckpt_name": CKPT, "device": "default"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": a.prompt}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": "blurry, low quality, flicker, morphing, cuts, watermark, text, cartoon, CGI, clay, plastic, untextured, snow"}},
    "8": {"class_type": "LTXVConditioning", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "frame_rate": 24}},
    "9": {"class_type": "LoadVideo", "inputs": {"file": a.video}},
    "10": {"class_type": "GetVideoComponents", "inputs": {"video": ["9", 0]}},
    "11": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": a.width, "height": a.height, "length": a.frames, "batch_size": 1}},
}
latent = ["11", 0]
if a.keyframe:
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": a.keyframe}}
    g["21"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 18}}
    g["22"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["21", 0], "latent": latent, "strength": a.keyframe_strength, "bypass": False}}
    latent = ["22", 0]
g.update({
    "12": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["8", 0], "negative": ["8", 1], "vae": ["1", 2], "latent": latent,
                                                     "image": ["10", 0], "frame_idx": 0, "strength": a.guide_strength, "iclora_parameters": ["4", 0]}},
    "13": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "14": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": a.frames, "frame_rate": 24, "batch_size": 1, "audio_vae": ["13", 0]}},
    "15": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["12", 2], "audio_latent": ["14", 0]}},
    "16": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["15", 0],
                                                "seed": a.seed, "steps": a.steps, "cfg": a.cfg, "sampler_name": a.sampler, "scheduler": "linear_quadratic", "denoise": 1.0}},
    "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
    "18": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["12", 0], "negative": ["12", 1], "latent": ["17", 0]}},
    "19": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["18", 2], "vae": ["1", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
    "23": {"class_type": "CreateVideo", "inputs": {"images": ["19", 0], "fps": 24}},
    "24": {"class_type": "SaveVideo", "inputs": {"video": ["23", 0], "filename_prefix": a.prefix, "format": "auto", "codec": "auto"}},
})
if a.end_keyframe:
    g["25"] = {"class_type": "LoadImage", "inputs": {"image": a.end_keyframe}}
    g["26"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["25", 0], "img_compression": 18}}
    g["27"] = {"class_type": "LTXVAddGuide", "inputs": {"positive": ["12", 0], "negative": ["12", 1], "vae": ["1", 2],
               "latent": ["12", 2], "image": ["26", 0], "frame_idx": -1, "strength": a.end_strength}}
    g["15"]["inputs"]["video_latent"] = ["27", 2]
    for k, i in (("positive", 0), ("negative", 1)):
        g["16"]["inputs"][k] = ["27", i]; g["18"]["inputs"][k] = ["27", i]
if a.upscale:
    g.update({
        "30": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}},
        "31": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["18", 2], "upscale_model": ["30", 0], "vae": ["1", 2]}},
        "33": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["31", 0], "audio_latent": ["17", 1]}},
        "34": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}},
        "35": {"class_type": "CFGGuider", "inputs": {"model": ["3", 0], "positive": ["18", 0], "negative": ["18", 1], "cfg": 1.0}},
        "36": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "37": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed + 1}},
        "38": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["37", 0], "guider": ["35", 0], "sampler": ["36", 0],
                                                                 "sigmas": ["34", 0], "latent_image": ["33", 0]}},
        "39": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["38", 0]}},
    })
    if a.keyframe:  # re-anchor the first frame at the new size before refining
        g["32"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["21", 0], "latent": ["31", 0], "strength": 1.0, "bypass": False}}
        g["33"]["inputs"]["video_latent"] = ["32", 0]
    g["19"]["inputs"]["samples"] = ["39", 0]
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
try:
    print(urllib.request.urlopen(req).read().decode())
except urllib.error.HTTPError as e:
    print(e.read().decode()); raise SystemExit(1)
