"""Repaint a clay frame photoreal with Qwen-Image-Edit 2511 (lightning, 4 steps), keeping its geometry."""
import argparse, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoblock_core.weather import WEATHER, weather_prompt  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("clay"); ap.add_argument("prefix"); ap.add_argument("--photo"); ap.add_argument("--photo2"); ap.add_argument("--scene-text", default="")
ap.add_argument("--quality", action="store_true", help="No Lightning LoRA: 24 steps, CFG 4, 1792x1008"); ap.add_argument("--seed", type=int, default=20261005); ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--prompt", default=(
    "Image 1 is the only source of truth for geometry, composition and camera. Repaint its surfaces only as a real "
    "photograph; do not reframe, recenter, zoom or move the horizon. The flat foreground is the South Rim plateau "
    "at Mather Point, Grand Canyon: dark green pinyon pine and juniper trees scattered over pale limestone soil. The "
    "distant flat-topped buttes on the horizon are the canyon's far side, banded red and cream rock. {photo}"
    "Sharp natural photographic detail."))
ap.add_argument("--weather", choices=tuple(WEATHER), default="clear", help="Append this weather to the prompt; match the blocking render")
a = ap.parse_args()
if a.weather:
    a.prompt = a.prompt.replace(" {photo}", " " + weather_prompt(a.weather) + " {photo}") if "{photo}" in a.prompt else a.prompt + " " + weather_prompt(a.weather)
photo = "Use the other images only for the real colours of the rock, vegetation and light, not its viewpoint. " if a.photo else ""
g = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "4": {"class_type": "LoadImage", "inputs": {"image": a.clay}},
 "13": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
 "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["13", 0], "shift": 3.1}},
 "6": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": a.prompt.format(photo=photo), "vae": ["3", 0], "image1": ["4", 0]}},
 "7": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["4", 0]}},
 "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
 "9": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0],
       "seed": a.seed, "steps": a.steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
 "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": a.prefix}},
}
if a.photo:
    g["12"] = {"class_type": "LoadImage", "inputs": {"image": a.photo}}
    for n in ("6", "7"): g[n]["inputs"]["image2"] = ["12", 0]
if a.photo2:
    g["14"] = {"class_type": "LoadImage", "inputs": {"image": a.photo2}}
    for n in ("6", "7"): g[n]["inputs"]["image3"] = ["14", 0]
if a.quality:
    g["5"]["inputs"]["model"] = ["1", 0]
    del g["13"]
    g["15"] = {"class_type": "ImageScale", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": 1792, "height": 1008, "crop": "center"}}
    for n in ("6", "7"): g[n]["inputs"]["image1"] = ["15", 0]
    g["8"]["inputs"]["pixels"] = ["15", 0]
    g["9"]["inputs"].update(steps=max(a.steps, 24), cfg=4.0)
    g["7"]["inputs"]["prompt"] = "blurry, painting, illustration, CGI, 3D render, clay, plastic, low detail, oversaturated"
r = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
try: print(urllib.request.urlopen(r).read().decode())
except urllib.error.HTTPError as e: print(e.read().decode()); raise SystemExit(1)
