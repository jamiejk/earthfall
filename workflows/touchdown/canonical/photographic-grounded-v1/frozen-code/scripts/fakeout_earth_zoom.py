"""Fake-out Earth zoom: the Higgsfield recipe on our local models, with no geodata at all.

Higgsfield's Earth Zoom (docs/review/higgsfield-how-it-works.md) is one image-to-video generation from one photo
of a person: the model invents the orbit, cloud, city and street. This reproduces that with FLUX.2 [klein] for
the stills and plain LTX 2.5 image-to-video (no Layout To Render, no blocking), or Wan 2.2 with an Earth
zoom-out camera LoRA for comparison. Once this looks right, measured data can be added back step by step.

  1. klein paints the end still: the person, waist-up, standing in the scene (or --end-still / --subject photo).
  2. The video, one of three ways (--mode):
     reverse     LTX starts on the end still and pulls back up to orbit; the clip is played backwards (the
                 creators' "Earth Zoom Out, reversed" trick). The sharp, coherent frame is generated first.
     first-last  klein also paints an orbit still; LTX pins orbit at frame 0 and the end still at the last frame.
     end-only    Only the end still is pinned (last frame); the prompt alone makes the orbit start.
     nadir       Real imagery straight down from deep space through a cloud deck to ~400 m (no model,
                 scripts/nadir_dive.py), then one LTX first-last swoop from its last frame to the end still.
     chain       Two first-last clips joined on one klein still ~6 km up, just below the cloud (orbit -> whiteout
                 -> high still, then high still -> person), so the shot comes out of the cloud far above the city.
     --engine wan always runs reverse (an image-to-video zoom-out LoRA, start frame only).
  3. ffmpeg reverses the clip where needed and writes <name>_zoomin.mp4 next to the download.

  uv run python scripts/fakeout_earth_zoom.py --mode reverse --seed 1 --output-dir outputs/runs/fakeout
  uv run python scripts/fakeout_earth_zoom.py --stills-only            # look at the stills first (~1 min)
  uv run python scripts/fakeout_earth_zoom.py --engine wan --frames 81 # needs the Wan 2.2 models (--wan-*)

Prints the wall time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # nadir_dive
from geoblock_core.run import _client, _submit
from geoblock_core.workflow import (
    LTX25_MODELS,
    LTX_MODELS,
    LTX_VIDEO,
    STAGE1_VIDEO,
    _add_ltx25,
)

PERSON = ("a young woman with short curly dark hair, wearing a bright red raincoat over a grey hoodie, "
          "black jeans and a small backpack")
SCENE = ("on the grassy summit of Twin Peaks in San Francisco, the city's houses running down the hills behind "
         "her to the downtown towers and the bay")
REGION = "the coast of northern California and San Francisco Bay"
APPROACH = "the rooftops and streets of the city and up the green hillside"

# klein drew "a medium shot from the waist up ... standing" full-length (run 1), so the framing is spelled out
# and the word "standing" left out.
END_STILL = ("A real portrait photograph, eye level, cropped at the waist: the frame shows only the head, "
             "shoulders and upper body of {person}, from the top of her head down to her belt, filling the middle "
             "of the frame, {scene}. She is still, facing the camera, her face turned slightly. Natural daylight, "
             "a few soft clouds, 50mm lens, sharp, natural colour, real photo, no text.")
END_STILL_SUBJECT = ("A real portrait photograph, eye level, cropped at the waist: the frame shows only the head, "
                     "shoulders and upper body of the person from image 2 (same face, hair and clothes), from the top "
                     "of their head down to their belt, filling the middle of the frame, {scene}. They are still, "
                     "facing the camera, face turned slightly. Natural daylight, 50mm lens, sharp, natural colour, "
                     "real photo, no text.")
# Grounding step 1: the end still painted from a real view of the place (e.g. a GeoBlock render or a frame of an
# earlier run), the place kept and the person put in front of it.
END_STILL_PLACE = ("Turn image 1 into a real photograph of exactly the same place from the same viewpoint: keep the "
                   "skyline, the hills, the streets, the buildings and the trees where they are, as real photography. "
                   "Remove anyone already there. In the centre, close to the camera at eye level, add {who}, framed "
                   "from the top of the head down to the belt, still and facing the camera, face turned slightly. "
                   "Natural daylight, 50mm lens, sharp, natural colour, no text.")

ORBIT_STILL = ("A real photograph taken from low Earth orbit, looking down at an angle at {region}: the curved "
               "blue limb of the Earth with a thin glowing atmosphere near the top of the frame, black space above "
               "it, white cloud systems and land below, daylight, natural colour, no text.")

CLOUD_STILL = ("A real photograph from inside a thick white cloud: soft white and pale grey mist fills the whole "
               "frame evenly, slightly brighter towards the top, nothing else visible, no text.")
# FLUX 3 versions (17 Oct): FLUX 3 follows the words more literally than klein. With ORBIT_STILL and a "cloud deck
# directly below" region it painted a high view with a cloud bank in the near foreground, which LTX lifted up the
# frame into a vertical sink; and its flat whiteout gave the cloud stage no direction. These keep klein's low oblique
# composition (cloud field receding to the horizon, open foreground) and give the whiteout streaks toward the middle.
ORBIT_STILL_FLUX3 = ("A real photograph from a spacecraft in low Earth orbit, looking forward at a shallow oblique angle "
                     "toward the horizon over {region}. The curved blue limb of the Earth with a thin glowing band of "
                     "atmosphere crosses the upper third of the frame, black space above it. Below the horizon a wide "
                     "field of small white cumulus clouds is scattered over the land and recedes in perspective toward "
                     "the horizon, the clouds smaller and denser with distance. The bottom of the frame is open ground "
                     "between small, flat, distant clouds: no large cloud, cloud bank or towering cloud in the near "
                     "foreground. Daylight, natural colour, sharp, no spacecraft parts, no text.")
CLOUD_STILL_FLUX3 = ("A real photograph from an aircraft flying forward through a thick, bright white cloud: almost "
                     "even bright white mist fills the whole frame, only faint soft wisps of slightly denser vapour "
                     "drift slowly in from the edges of the frame toward a brighter glow just above the centre. Very low "
                     "contrast, no dark grey, no ground, no sky, no horizon, nothing else visible, no text.")
AERIAL_STILL = ("A real aerial photograph from a helicopter a few hundred metres up, flying forward and looking "
                "ahead and down over {approach}, high above {region}, the horizon near the top of the frame, a "
                "little haze, daylight, natural colour, no text.")

ZOOM_IN = ("One continuous shot with no cuts. The camera starts in low Earth orbit above the curved blue limb of "
           "the Earth, then dives straight down at a steady, unhurried pace into a thick white cloud layer that "
           "fills the whole frame, bursts out below the clouds high above {region}, swoops fast and low over "
           "{approach} with strong motion blur, levels out at eye level, and spends the second half of the shot on a slow, steady push-in, ending on a "
           "waist-up shot of {person} standing {scene}, who stays still and looks at the camera.{ambient} Real aerial "
           "photography, natural colour, photorealistic.")
# chain: segment A ends on a high still just below the cloud (~20,000 ft), segment B starts on that same still
HIGH_STILL = ("A real aerial photograph from about 6,000 metres up, just below a broken cloud deck whose ragged grey "
              "underside fills the top edge of the frame, looking forward and down at an angle over {region}, the "
              "city, hills and coastline small and sharp far below, a little haze, daylight, natural colour, no text.")
CHAIN_A = ("One continuous shot with no cuts. The camera starts in low Earth orbit above the curved blue limb of the "
           "Earth, then dives straight down, accelerating, into a thick white cloud layer that fills the whole frame, "
           "then breaks out below the clouds very high above {region}, about 20,000 feet up, and keeps gliding "
           "forward and down with the whole city spread out far below. Real aerial photography, natural colour, "
           "photorealistic.")
CHAIN_B = ("One continuous shot with no cuts. The camera starts very high above {region}, just below the clouds, "
           "already gliding forward and down, descends steadily and then faster over {approach}, levels out at eye "
           "level, and spends the second half of the shot on a slow, steady push-in, ending on a waist-up shot of "
           "{person} standing {scene}, who stays still and looks at the camera.{ambient} Real aerial photography, "
           "natural colour, photorealistic.")
# --a-reverse: segment A generated upwards from the high still (and played backwards). A first-last A with
# both ends pinned dissolved instead of flying (high-merge, 9 Oct: optical flow ~0.1 px/frame over f0-119).
CHAIN_A_UP = ("One continuous shot with no cuts. It starts just below a grey cloud deck very high above {region}, "
              "then the camera rises straight up, faster and faster, into the cloud deck, which fills the whole frame "
              "white, bursts out above the clouds and keeps climbing into space, until the curved blue limb of the "
              "Earth shows against black space. Real aerial photography, natural colour, photorealistic.")
# ladder (Jamie, 9 Oct): stills first, from the end shot upwards, each a klein edit of the rung below so they stay
# the same place; then one first-last segment per adjacent pair. Pins only fly when they are close in scale.
RUNG_STILL = ("A real aerial photograph of exactly the same place as image 1, seen from about {feet} feet ({metres} m) "
              "up and further back, looking down at an angle toward the same spot: the same hills, roads, trees and "
              "buildings, now smaller, with more of {region} visible around them. {figure}Daylight as in image 1, "
              "natural colour, sharp, no text.")
RUNG_FIGURE = "The person from image 1 is still standing on the same spot, now a small figure near the centre. "
RUNG_UNDER_CLOUD = "A grey cloud deck's ragged underside fills the top edge of the frame. "
RUNG_SEGMENT = ("One continuous shot with no cuts. The camera flies steadily down and forward from about {feet_hi} "
                "feet to about {feet_lo} feet above {region}, toward the same spot, the ground growing closer. "
                "Real aerial photography, natural colour, photorealistic.")
LAST_SEGMENT = ("One continuous shot with no cuts. The camera glides down and forward to eye level and slows into a "
                "gentle push-in, ending on a waist-up shot of {person} standing {scene}, who stays still and looks at "
                "the camera.{ambient} Real photography, natural colour, photorealistic.")
# --hold-frames: the ambient sentence alone moved nothing (r4-ambient: the pinned last frame freezes the end), so
# the end hold is its own image-to-video clip that starts on the main clip's last frame and is pinned nowhere else.
HOLD = ("One continuous shot with no cuts, a real photograph come to life: a locked-off camera holds the same "
        "waist-up framing of {person} standing {scene}. She does not walk or step; she stands on the same spot "
        "and looks at the camera, breathing, blinking. Behind her clouds drift slowly across the sky, a few birds fly past in the "
        "distance, and the grass, her hair and her coat move in the wind. Natural colour, photorealistic.")
# zoom: the painted ladder's klein edits stayed near 200 m (9 Oct), so the upper rungs are outpainted instead: each
# is the rung below shrunk into its centre and klein fills the border, which pins the scale exactly. The dive
# between them is then an exact digital zoom (no LTX, so nothing to dissolve), and LTX only does the swoop.
NADIR_STILL = ("A real aerial photograph looking straight down from about 600 metres above the spot where the person "
               "in image 1 stands: the same hill with its paths, grass and trees seen from directly above, the houses "
               "and streets of the city spreading out all around it, sharp, natural colour, daylight, no text.")
OUTPAINT_STILL = ("A real aerial photograph looking straight down from about {metres:,} metres. The picture in the "
                  "middle is the same ground seen from lower down; continue it seamlessly outwards in every "
                  "direction with more of {region}{extra}. Sharp, natural colour, daylight, no text.")
SWOOP = ("One continuous shot with no cuts. The camera starts looking straight down from about 2,000 feet above the "
         "hill, then dives and tilts up as it swoops down over {approach}, levels out at eye level, and spends the "
         "second half of the shot on a slow, steady push-in, ending on a waist-up shot of {person} standing {scene}, "
         "who stays still and looks at the camera.{ambient} Real aerial photography, natural colour, photorealistic.")
NADIR_SWOOP = ("One continuous shot with no cuts. The camera starts looking straight down from about 400 metres above "
               "the hill, still descending, then tilts up as it swoops down over {approach}, levels out at eye "
               "level, and spends the second half of the shot on a slow, steady push-in, ending on a waist-up shot "
               "of {person} standing {scene}, who stays still and looks at the camera.{ambient} Real aerial "
               "photography, natural colour, photorealistic.")
TOP_UP = ("One continuous shot with no cuts. It starts looking straight down from very high above {region}, then "
          "the camera rises straight up, faster and faster, past thin white clouds, until the curved blue limb of "
          "the Earth and black space appear. Real photography from space, natural colour, photorealistic.")
AMBIENT = (" Around her the scene stays alive: clouds drift slowly across the sky, a few birds fly past in the "
           "distance, and the grass and her hair move in the wind.")
ZOOM_OUT = ("One continuous shot with no cuts. It starts on a waist-up shot of {person} standing {scene}, who "
            "stays completely still. The camera pulls back slowly, then lifts off and rises faster and faster, "
            "rushing up and away over {approach} with strong motion blur, high above {region}, up through a thick "
            "white cloud layer that fills the whole frame, and out to low Earth orbit, showing the curved blue limb "
            "of the Earth against black space. Real aerial photography, natural colour, photorealistic.")

# Wan 2.2 image-to-video A14B (ComfyUI's repackaged fp8 files) and the Earth zoom-out camera LoRA from
# huggingface.co/wangkanai/wan22-fp16-i2v-loras (loras/wan/wan22-camera-earthzoomout.safetensors). The LoRA card
# does not say which expert it was trained for, so it goes on both.
WAN_MODELS = {
    "high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan_2.1_vae.safetensors",
    "lora": "wan22-camera-earthzoomout.safetensors",
}

STILL_W, STILL_H = 1344, 768  # klein; LTX and Wan centre-crop to their own 16:9-ish sizes


def _klein(graph: dict, base: int, text: str, seed: int, prefix: str, references: tuple = ()) -> list:
    """One FLUX.2 [klein] still (4 steps, CFG 1), with reference images (image 1, image 2, ...) as in
    ComfyUI's klein edit template; returns its image output."""
    n = [str(base + i) for i in range(12)]
    graph.setdefault("1", {"class_type": "UNETLoader", "inputs": {"unet_name": LTX_MODELS["flux_unet"],
                                                                  "weight_dtype": "default"}})
    graph.setdefault("2", {"class_type": "CLIPLoader", "inputs": {"clip_name": LTX_MODELS["flux_clip"],
                                                                  "type": "flux2", "device": "default"}})
    graph.setdefault("3", {"class_type": "VAELoader", "inputs": {"vae_name": LTX_MODELS["flux_vae"]}})
    positive, negative = [n[0], 0], [n[1], 0]
    graph.update({
        n[0]: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": text}},
        n[1]: {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [n[0], 0]}},
    })
    for k, reference in enumerate(references):
        r = [f"{base}r{k}{x}" for x in "sepn"]
        graph.update({
            r[0]: {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": reference, "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
            r[1]: {"class_type": "VAEEncode", "inputs": {"pixels": [r[0], 0], "vae": ["3", 0]}},
            r[2]: {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [r[1], 0]}},
            r[3]: {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [r[1], 0]}},
        })
        positive, negative = [r[2], 0], [r[3], 0]
    graph.update({
        n[2]: {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": STILL_W, "height": STILL_H,
                                                                "batch_size": 1}},
        n[3]: {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": STILL_W, "height": STILL_H}},
        n[4]: {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": positive, "negative": negative,
                                                    "cfg": 1.0}},
        n[5]: {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [f"{base}n", 0], "guider": [n[4], 0], "sampler": [f"{base}s", 0], "sigmas": [n[3], 0],
            "latent_image": [n[2], 0]}},
        f"{base}n": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        f"{base}s": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        n[6]: {"class_type": "VAEDecode", "inputs": {"samples": [n[5], 0], "vae": ["3", 0]}},
        n[7]: {"class_type": "SaveImage", "inputs": {"images": [n[6], 0], "filename_prefix": prefix}},
    })
    return [n[6], 0]


FLUX3_MODELS = {  # black-forest-labs/flux-3-image-klein-comfy; needs ComfyUI v0.41.3+
    # The int8-convrot pair is the one for a 24 GB card (5.8 GB + 4.9 GB rather than 11.1 + 8.9); the names
    # can be set so the same code runs either set of weights.
    "unet": os.environ.get("TOUCHDOWN_FLUX3_UNET", "flux3-image-klein-bf16.safetensors"),
    "clip": os.environ.get("TOUCHDOWN_FLUX3_CLIP", "qwen3vl_4b_bf16.safetensors"),
    "vae": os.environ.get("TOUCHDOWN_FLUX3_VAE", "flux2-vae.safetensors"),
}
FLUX3_W, FLUX3_H = 2688, 1536  # BFL recommend 2K; at 1344x768 FLUX 3 kept the layout guide's render smear (17 Oct)


def _flux3(graph: dict, base: int, text: str, seed: int, prefix: str, references: tuple = (),
            full: bool = False) -> list:
    """One FLUX 3 Image Klein still (distilled, 16 steps, no guidance, BFL's timestep shift), as BFL's
    flux3_klein_bf16_t2i / _i2i workflows without the prompt upsampler (our prompts are already full captions;
    the upsampler made the end still worse). Rendered at 2K, saved scaled to STILL_W x STILL_H so every later
    stage sees the same size as with klein. Same node layout as ``_klein``: image at base+6, save at base+7.

    ``full`` also saves the frame at its full 2K size (node base+5): the video stages work at 960x544, so that
    copy is the only one of the face sharp enough to hold the last seconds of the shot on."""
    n = [str(base + i) for i in range(12)]
    graph.setdefault("f3u", {"class_type": "UNETLoader", "inputs": {"unet_name": FLUX3_MODELS["unet"],
                                                                    "weight_dtype": "default"}})
    graph.setdefault("f3c", {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX3_MODELS["clip"],
                                                                    "type": "flux3", "device": "default"}})
    graph.setdefault("f3v", {"class_type": "VAELoader", "inputs": {"vae_name": FLUX3_MODELS["vae"]}})
    graph.setdefault("f3m", {"class_type": "ModelSamplingFlux", "inputs": {  # mu = ln(16), as in BFL's workflows
        "model": ["f3u", 0], "max_shift": 2.77259, "base_shift": 2.77259, "width": 1024, "height": 1024}})
    graph.update({
        n[0]: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["f3c", 0], "text": text}},
        n[1]: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["f3c", 0], "text": ""}},
    })
    positive, negative = [n[0], 0], [n[1], 0]
    if references:
        graph[n[8]] = {"class_type": "ReferenceMultiLatent", "inputs": {
            "positive": positive, "negative": negative, "vae": ["f3v", 0], "resolution": "resize",
            "resolution.megapixels": 1.0, **{f"images.image{k}": r for k, r in enumerate(references)}}}
        positive, negative = [n[8], 0], [n[8], 1]
    graph.update({
        n[2]: {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": FLUX3_W, "height": FLUX3_H,
                                                                "batch_size": 1}},
        n[3]: {"class_type": "KSampler", "inputs": {
            "model": ["f3m", 0], "seed": seed, "steps": 16, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0, "positive": positive, "negative": negative,
            "latent_image": [n[2], 0]}},
        n[4]: {"class_type": "VAEDecode", "inputs": {"samples": [n[3], 0], "vae": ["f3v", 0]}},
        n[6]: {"class_type": "ImageScale", "inputs": {"image": [n[4], 0], "upscale_method": "lanczos",
                                                      "width": STILL_W, "height": STILL_H, "crop": "center"}},
        n[7]: {"class_type": "SaveImage", "inputs": {"images": [n[6], 0], "filename_prefix": prefix}},
    })
    if full:
        graph.update({
            # the same frame at its full size: the video stages work at 960x544, so this is the only copy of the
            # face that is sharp enough to hold the last seconds of the shot on
            n[5]: {"class_type": "SaveImage", "inputs": {"images": [n[4], 0], "filename_prefix": prefix + "_full"}},
        })
    return [n[6], 0]


def _klein_outpaint(graph: dict, base: int, inner: list, text: str, seed: int, prefix: str,
                    factor: int = 3) -> tuple[list, str]:
    """klein fills the border around ``inner`` shrunk by ``factor`` into the middle of a same-size canvas (latent
    noise mask on the border only), then the shrunk original is pasted back over the middle with a feathered edge,
    so the middle is exactly the rung below. Node IDs "<base>o*"; returns (image, save node)."""
    o = lambda name: f"{base}o{name}"
    w, h = STILL_W // factor, STILL_H // factor
    px, py = (STILL_W - w) // 2, (STILL_H - h) // 2
    graph.update({
        o("fit"): {"class_type": "ImageScale", "inputs": {"image": inner, "upscale_method": "lanczos",
                                                         "width": STILL_W, "height": STILL_H, "crop": "center"}},
        o("small"): {"class_type": "ImageScale", "inputs": {"image": [o("fit"), 0], "upscale_method": "lanczos",
                                                           "width": w, "height": h, "crop": "disabled"}},
        o("pad"): {"class_type": "ImagePadForOutpaint", "inputs": {
            "image": [o("small"), 0], "left": px, "top": py, "right": STILL_W - w - px, "bottom": STILL_H - h - py,
            "feathering": 0}},
        o("enc"): {"class_type": "VAEEncode", "inputs": {"pixels": [o("pad"), 0], "vae": ["3", 0]}},
        o("mask"): {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [o("enc"), 0], "mask": [o("pad"), 1]}},
        o("txt"): {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": text}},
        o("zero"): {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [o("txt"), 0]}},
        o("pos"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": [o("txt"), 0], "latent": [o("enc"), 0]}},
        o("neg"): {"class_type": "ReferenceLatent", "inputs": {"conditioning": [o("zero"), 0], "latent": [o("enc"), 0]}},
        o("sched"): {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": STILL_W, "height": STILL_H}},
        o("guide"): {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": [o("pos"), 0],
                                                          "negative": [o("neg"), 0], "cfg": 1.0}},
        o("noise"): {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        o("sampler"): {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        o("sample"): {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [o("noise"), 0], "guider": [o("guide"), 0], "sampler": [o("sampler"), 0],
            "sigmas": [o("sched"), 0], "latent_image": [o("mask"), 0]}},
        o("dec"): {"class_type": "VAEDecode", "inputs": {"samples": [o("sample"), 0], "vae": ["3", 0]}},
        o("solid"): {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": w, "height": h}},
        o("feather"): {"class_type": "FeatherMask", "inputs": {"mask": [o("solid"), 0], "left": 12, "top": 12,
                                                              "right": 12, "bottom": 12}},
        o("paste"): {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": [o("dec"), 0], "source": [o("small"), 0], "x": px, "y": py, "resize_source": False,
            "mask": [o("feather"), 0]}},
        o("save"): {"class_type": "SaveImage", "inputs": {"images": [o("paste"), 0], "filename_prefix": prefix}},
    })
    return [o("paste"), 0], o("save")


def _zoom_video(rungs: list[Path], frames: int, out: str, size: tuple[int, int] = (1920, 1088),
                factor: int = 3) -> str:
    """Exact digital dive from rungs[0] (highest) to rungs[-1], each rung the middle 1/factor of the one above:
    constant zoom speed in log scale, every frame drawn from the sharpest rung that covers it. 24 fps."""
    from PIL import Image
    images = [Image.open(p).convert("RGB").resize(size, Image.LANCZOS) for p in rungs]
    width, height = size
    span = len(images) - 1
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frames):
            t = span * i / (frames - 1)
            k = min(int(t), span - 1) if span else 0
            zoom = factor ** (t - k)  # how far into rung k
            frame = Image.new("RGB", size)
            for j in range(k, len(images)):  # paint rung k, then each sharper rung inside it
                scale = zoom / factor ** (j - k)  # rung j drawn at this size relative to the frame
                w, h = width * scale, height * scale
                if w < 4:
                    break
                left, top = (width - w) / 2, (height - h) / 2
                # the part of rung j that lands inside the frame
                box = (max(0, -left) / scale, max(0, -top) / scale, min(w, width - left) / scale,
                       min(h, height - top) / scale)
                crop = images[j].crop(tuple(round(v) for v in box))
                dest = (round(max(0, left)), round(max(0, top)))
                dw, dh = round(min(w, width - left) - max(0, -left)), round(min(h, height - top) - max(0, -top))
                frame.paste(crop.resize((max(dw, 1), max(dh, 1)), Image.LANCZOS), dest)
            frame.save(f"{tmp}/{i:05d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", out], check=True)
    return out


def _zoom(args, words: dict) -> None:
    """End shot -> straight-down still ~600 m above her (klein edit) -> outpainted rungs x3 each up to
    --zoom-steps; then [LTX rise from the top rung to orbit, reversed] + exact digital dive down the rungs + an LTX
    first-last swoop from the 600 m still to the end shot. Writes fakeout_zoom_s<seed>_joined.mp4."""
    prefix = f"geoblock/fakeout_zoom_s{args.seed}"
    started = time.monotonic()
    heights = [600 * 3 ** k for k in range(args.zoom_steps + 1)]
    with _client(args.url) as client:
        graph: dict = {}
        wanted: list[str] = []
        end = _end_still(graph, client, args, words, prefix, wanted)
        rung = _klein(graph, 400, NADIR_STILL, args.seed + 400, prefix + "_rung600m", (end,))
        wanted.append("407")
        for k, metres in enumerate(heights[1:], start=1):
            extra = (", thin white clouds drifting between the camera and the ground" if metres >= 5000 else "")
            extra += ", the coastline and the ocean" if metres >= 15000 else ""
            rung, save = _klein_outpaint(graph, 400 + 20 * k, rung, OUTPAINT_STILL.format(
                metres=metres, region=args.region, extra=extra), args.seed + 400 + k, f"{prefix}_rung{metres}m")
            wanted.append(save)
        stills = _submit(client, graph, tuple(wanted), args.output_dir)
        end_path = Path(stills["107"]) if "107" in stills else args.end_still
        rungs = [Path(stills[n]) for n in reversed(wanted[-len(heights):])]  # highest first
        print("Rungs (top first):", *rungs, sep="\n  ")
        if args.stills_only:
            return
        clips = []

        def segment(name: str, frames: int, pins: list, prompt: str, seed: int) -> str:
            graph: dict = {}
            loaded = [(index, _load(graph, client, str(41 + j), path)) for j, (index, path) in enumerate(pins)]
            return _submit(client, graph, tuple(_ltx(graph, args, frames, loaded, prompt, seed, f"{prefix}_{name}")),
                           args.output_dir)[LTX_VIDEO]

        if args.top == "rise":
            clips.append(_reverse(segment("top", args.top_frames, [(0, rungs[0])], TOP_UP.format(**words), args.seed)))
        clips.append(_zoom_video(rungs, args.zoom_frames, str(rungs[-1].with_name(f"fakeout_zoom_s{args.seed}_dive.mp4"))))
        clips.append(segment("swoop", args.frames or 121, [(0, rungs[-1]), (-1, end_path)], SWOOP.format(**words),
                             args.seed + 1))
        if args.hold_frames:
            clips.append(_hold(client, args, words, clips[-1], prefix))
    joined = str(Path(clips[-1]).with_name(f"fakeout_zoom_s{args.seed}_joined.mp4"))
    _join(clips, joined)
    print(f"Elapsed {time.monotonic() - started:.0f} s, zoom rungs {heights} m")
    print("Clips:", *clips, f"Joined: {joined}", sep="\n  ")


def _frames_small(path: str, size: tuple[int, int] = (160, 90)):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={size[0]}:{size[1]},format=gray", "-f",
                          "rawvideo", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0]).astype(np.float32)


def _match_speed(lead: str, clip: str, ramp: int = 24) -> str:
    """Drop frames from the start of ``clip`` so its motion starts at the end speed of ``lead``: LTX starts the
    swoop almost still and hovers for ~2 s after a dive that arrives fast (dada893: a hard brake at the join).
    A frame is kept once the motion since the last kept frame reaches a threshold that starts at the lead's final
    speed and eases (smoothstep) to 0 over ``ramp`` output frames, so the clip decelerates to its own speed and
    every frame after the ramp is kept, even when the clip never moves as fast as the lead (241-frame swoop:
    peak 16.4 against 21.4). From the first frame that moves at 0.8x the lead's speed by itself, every frame is
    kept. Returns the retimed file."""
    import numpy as np
    tail = _frames_small(lead)[-4:]
    target = float(np.abs(np.diff(tail, axis=0)).mean())
    frames = _frames_small(clip)
    motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))  # motion[i]: frame i -> i + 1
    keep, acc = [0], 0.0
    for i in range(1, len(frames)):
        acc += motion[i - 1]
        t = min(len(keep) / ramp, 1.0)
        if motion[i - 1] >= 0.8 * target or t >= 1.0:
            keep += list(range(i, len(frames)))
            break
        if acc >= target * (1 - t * t * (3 - 2 * t)):
            keep.append(i)
            acc = 0.0
    print(f"Swoop retimed to the dive's speed ({target:.1f}): kept {len(keep)} of {len(frames)} frames, "
          f"first {keep[:12]}")
    out = str(Path(clip).with_name(Path(clip).stem + "_retimed.mp4"))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", clip, f"{tmp}/%05d.png"], check=True)
        for j, i in enumerate(keep):
            Path(f"{tmp}/{i + 1:05d}.png").rename(f"{tmp}/k{j:05d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/k%05d.png", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", out], check=True)
    return out


def _grade_ramp(clip: str, first: Path, last: Path) -> str:
    """Hold the swoop's colour on a straight line from the dive's last frame to the end still: LTX drifts warm
    and brown in between (naip-zoom, dada893). Per frame, a white balance only: each channel is scaled so the
    frame's colour balance (channel means over their average) is the blend of the two stills' at that point.
    Brightness and contrast stay LTX's (matching mean and spread as well flattened the swoop to grey haze);
    the brightness step at the join is _lift_join's, after the retime. Returns the graded file."""
    import numpy as np
    from PIL import Image

    def balance(image) -> np.ndarray:
        m = np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0)
        return m / max(float(m.mean()), 1.0)

    start = np.asarray(Image.open(first).convert("RGB"), dtype=np.float32)
    b0, b1 = balance(start), balance(Image.open(last).convert("RGB"))
    out = str(Path(clip).with_name(Path(clip).stem + "_graded.mp4"))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", clip, f"{tmp}/%05d.png"], check=True)
        files = sorted(Path(tmp).glob("*.png"))
        for i, path in enumerate(files):
            t = i / max(len(files) - 1, 1)
            frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            graded = frame * ((b0 * (1 - t) + b1 * t) / balance(frame))
            Image.fromarray(np.clip(graded, 0, 255).astype(np.uint8)).save(path)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", out], check=True)
    return out


def _lift_join(lead_last: Path, clip: str, span: int = 36) -> str:
    """Remove the brightness step where ``clip`` follows ``lead_last``: LTX darkens over its first frames and
    _match_speed squeezes that into 1-2 frames (hybrid: L 42.4 -> 37.8 across the join). The offset (lead's last
    frame minus the mean of the retimed clip's frames 1-4) is added to every channel, tapered to 0 over ``span``
    frames by smoothstep, and frame 0 is only lifted as far as it was below (the desktop's lramp, without cv2:
    an equal offset on R, G and B moves lightness and leaves the colour balance). Returns the lifted file."""
    import numpy as np
    from PIL import Image

    def lum(a: np.ndarray) -> float:
        return float((a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)).mean())

    target = lum(np.asarray(Image.open(lead_last).convert("RGB"), dtype=np.float32))
    out = str(Path(clip).with_name(Path(clip).stem + "_lifted.mp4"))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", clip, f"{tmp}/%05d.png"], check=True)
        files = sorted(Path(tmp).glob("*.png"))
        frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) for p in files[:span]]
        offset = target - float(np.mean([lum(f) for f in frames[1:5]]))
        first = target - lum(frames[0])
        print(f"Join lift: {offset:+.1f} (frame 0 {first:+.1f}), tapered over {span} frames")
        for k, (path, frame) in enumerate(zip(files, frames, strict=False)):
            t = k / span
            shift = offset * (1 - t * t * (3 - 2 * t))
            if k == 0:
                shift = min(shift, first) if offset > 0 else max(shift, first)
            Image.fromarray(np.clip(frame + shift, 0, 255).astype(np.uint8)).save(path)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%05d.png", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", out], check=True)
    return out


def _nadir(args, words: dict) -> None:
    """Grounded dive (scripts/nadir_dive.py: real imagery, straight down from deep space through a cloud deck to
    ~400 m, no model) + one LTX first-last swoop from its last frame to the end shot (193 frames by default: the
    desktop's NAIP prototype flew at 193 and crammed the tilt into ~15 frames with smear at 121). Writes
    fakeout_nadir_s<seed>_joined.mp4."""
    import nadir_dive

    from geoblock_core.places import find_place, landing, view_heading
    prefix = f"geoblock/fakeout_nadir_s{args.seed}"
    started = time.monotonic()
    place = find_place(args.place)
    lat, lon = landing(place)
    folder = args.output_dir / "nadir"
    levels = nadir_dive.load_levels(nadir_dive.fetch_imagery(lat, lon, folder / f"imagery_{args.place}"))
    bearing = args.dive_bearing if args.dive_bearing is not None else max(view_heading(place), 0.0)
    dive = nadir_dive.render_dive(levels, lat, lon, bearing, folder / f"fakeout_nadir_s{args.seed}_dive.mp4",
                                  frames=args.dive_frames, coverage=args.cloud_cover, seed=args.seed,
                                  end_m=args.dive_end_m, spin=args.dive_spin)
    last = dive.with_name(dive.stem + "_last.png")
    print(f"Dive: {dive} ({time.monotonic() - started:.0f} s)")
    if args.stills_only:
        return
    if args.swoop:  # an existing swoop whose first frame the dive was framed to land on (--dive-end-m/-bearing)
        swoop = _grade_ramp(str(args.swoop), last, args.end_still)
        joined = str(dive.with_name(f"fakeout_nadir_s{args.seed}_{args.swoop.stem}_joined.mp4"))
        _join([str(dive), _lift_join(last, _match_speed(str(dive), swoop))], joined)
        print(f"Joined: {joined}")
        return
    with _client(args.url) as client:
        graph: dict = {}
        wanted: list[str] = []
        end = _end_still(graph, client, args, words, prefix, wanted)
        pins = [(0, _load(graph, client, "44", last)), (-1, end)]
        wanted += _ltx(graph, args, args.frames or 193, pins, args.prompt or NADIR_SWOOP.format(**words),
                       args.seed + 1, prefix + "_swoop")
        saved = _submit(client, graph, tuple(wanted), args.output_dir)
        swoop = _grade_ramp(saved[LTX_VIDEO], last, Path(saved["107"]) if "107" in saved else args.end_still)
        clips = [str(dive), _lift_join(last, _match_speed(str(dive), swoop))]
        if args.hold_frames:
            clips.append(_hold(client, args, words, clips[-1], prefix))
    joined = str(Path(saved[LTX_VIDEO]).with_name(f"fakeout_nadir_s{args.seed}_joined.mp4"))
    _join(clips, joined)
    print(f"Elapsed {time.monotonic() - started:.0f} s")
    print("Clips:", *clips, f"Joined: {joined}", sep="\n  ")


def _wan(graph: dict, start: list, length: int, prompt: str, seed: int, prefix: str, args) -> str:
    """Wan 2.2 A14B image-to-video with the camera LoRA on both experts: 20 steps (10 high-noise, 10 low-noise),
    CFG 3.5, shift 8, at 832x480 and 16 fps. Node IDs 500-529; returns the SaveVideo node."""
    def expert(base: int, name: str) -> list:
        graph.update({
            str(base): {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}},
            str(base + 1): {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": [str(base), 0], "lora_name": args.wan_lora, "strength_model": args.wan_lora_strength}},
            str(base + 2): {"class_type": "ModelSamplingSD3", "inputs": {"model": [str(base + 1), 0], "shift": 8.0}},
        })
        return [str(base + 2), 0]

    high, low = expert(500, args.wan_high), expert(503, args.wan_low)
    graph.update({
        "506": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.wan_text_encoder, "type": "wan",
                                                      "device": "default"}},
        "507": {"class_type": "VAELoader", "inputs": {"vae_name": args.wan_vae}},
        "508": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["506", 0], "text": prompt}},
        "509": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["506", 0], "text": (
            "blurry, low quality, flicker, morphing, cuts, watermark, text, cartoon, CGI, static, still frame")}},
        "510": {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["508", 0], "negative": ["509", 0], "vae": ["507", 0], "width": 832, "height": 480,
            "length": length, "batch_size": 1, "start_image": start}},
        "511": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": high, "add_noise": "enable", "noise_seed": seed, "steps": 20, "cfg": 3.5,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["510", 0], "negative": ["510", 1],
            "latent_image": ["510", 2], "start_at_step": 0, "end_at_step": 10,
            "return_with_leftover_noise": "enable"}},
        "512": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": low, "add_noise": "disable", "noise_seed": seed, "steps": 20, "cfg": 3.5,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["510", 0], "negative": ["510", 1],
            "latent_image": ["511", 0], "start_at_step": 10, "end_at_step": 10000,
            "return_with_leftover_noise": "disable"}},
        "513": {"class_type": "VAEDecode", "inputs": {"samples": ["512", 0], "vae": ["507", 0]}},
        "514": {"class_type": "CreateVideo", "inputs": {"images": ["513", 0], "fps": 16}},
        "515": {"class_type": "SaveVideo", "inputs": {"video": ["514", 0], "format": "auto", "codec": "auto",
                                                      "filename_prefix": prefix + "_wan"}},
    })
    return "515"


def _reverse(path: str) -> str:
    """Play the clip backwards (the zoom-out becomes the zoom-in); returns the new file."""
    out = str(Path(path).with_name(Path(path).stem + "_zoomin.mp4"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", "reverse", "-an", "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", out], check=True)
    return out


def _load(graph: dict, client, node: str, path: Path) -> list:
    graph[node] = {"class_type": "LoadImage", "inputs": {"image": client.assets.from_file(str(path))}}
    return [node, 0]


def _end_still(graph: dict, client, args, words: dict, prefix: str, wanted: list) -> list:
    """The last frame: --end-still as given, or klein from the prompt / --end-reference place / --subject photo."""
    if args.end_still:
        return _load(graph, client, "40", args.end_still)
    references = []
    if args.end_reference:
        references.append(_load(graph, client, "42", args.end_reference))
    if args.subject:
        references.append(_load(graph, client, "41", args.subject))
    if args.end_reference:
        who = "the person from image 2 (same face, hair and clothes)" if args.subject else args.person
        text = END_STILL_PLACE.format(who=who)
    else:
        text = (END_STILL_SUBJECT if args.subject else END_STILL).format(**words)
    wanted.append("107")
    return _klein(graph, 100, text, args.seed, prefix + "_end", tuple(references))


def _orbit_still(graph: dict, client, args, words: dict, prefix: str, wanted: list) -> list:
    if args.orbit_still:
        return _load(graph, client, "43", args.orbit_still)
    wanted.append("127")
    return _klein(graph, 120, ORBIT_STILL.format(**words), args.seed + 1, prefix + "_orbit")


def _ltx(graph: dict, args, frames: int, pins: list, prompt: str, seed: int, prefix: str) -> list:
    """Plain LTX 2.5 keyframe image-to-video; returns the nodes to download."""
    request = SimpleNamespace(ltx25_models=dict(LTX25_MODELS), pin_last_keyframe=False, preset="fake-out",
                              seed=seed, save_stage1=not args.no_upscale, ltx_upscale=not args.no_upscale,
                              output_prefix=prefix, ltx_encoder_device="cpu")
    _add_ltx25(graph, request, frames, None, pins[-1][1], pins, prompt, layout=False)
    return [LTX_VIDEO] + ([] if args.no_upscale else [STAGE1_VIDEO])


def _chain(args, words: dict, frames: int) -> None:
    """Two first-last clips joined on one shared still about 6 km up, just below the cloud: A runs orbit ->
    whiteout -> high still, B runs high still -> person. Both are pinned to the same still, so the join frame
    matches exactly; the motion either side of it is the open question."""
    if frames % 8 != 1 or args.high_frames % 8 != 1:
        raise SystemExit("both segments need 8k+1 frames")
    prefix = f"geoblock/fakeout_ltx_chain_s{args.seed}"
    started = time.monotonic()
    with _client(args.url) as client:
        graph: dict = {}
        wanted: list[str] = []
        _end_still(graph, client, args, words, prefix, wanted)
        orbit = None if args.a_reverse else _orbit_still(graph, client, args, words, prefix, wanted)
        if args.high_still:
            high = _load(graph, client, "44", args.high_still)
        else:
            high = _klein(graph, 180, HIGH_STILL.format(**words), args.seed + 180, prefix + "_high")
            wanted.append("187")
        pins = [(0, orbit)]
        if args.a_reverse:
            pins = []
        elif args.cloud_at is not None:
            cloud = _klein(graph, 140, CLOUD_STILL, args.seed + 140, prefix + "_cloud")
            wanted.append("147")
            pins.append((round(args.cloud_at * (args.high_frames - 1) / 8) * 8, cloud))
        pins = [(0, high)] if args.a_reverse else [*pins, (-1, high)]
        if args.low_at is not None and not args.low_still:  # painted now, used by segment B
            _klein(graph, 160, AERIAL_STILL.format(**words), args.seed + 160, prefix + "_low")
            wanted.append("167")
        if args.stills_only:
            print(_submit(client, graph, tuple(wanted), args.output_dir))
            return
        prompt_a = args.prompt or (CHAIN_A_UP if args.a_reverse else CHAIN_A).format(**words)
        wanted += _ltx(graph, args, args.high_frames, pins, prompt_a, args.seed, prefix + "_a")
        first = _submit(client, graph, tuple(wanted), args.output_dir)
        # segment B starts from the stills segment A actually used, read back from its outputs
        graph = {}
        high_b = _load(graph, client, "44", Path(first["187"])) if "187" in first else _load(
            graph, client, "44", args.high_still)
        end_b = _load(graph, client, "40", Path(first["107"])) if "107" in first else _load(
            graph, client, "40", args.end_still)
        pins_b = [(0, high_b)]
        if args.low_at is not None:
            low = _load(graph, client, "45", Path(first["167"]) if "167" in first else args.low_still)
            pins_b.append((round(args.low_at * (frames - 1) / 8) * 8, low))
        second = _submit(client, graph, tuple(_ltx(graph, args, frames, [*pins_b, (-1, end_b)],
                                                   CHAIN_B.format(**words), args.seed + 1, prefix + "_b")),
                         args.output_dir)
    a, b = first[LTX_VIDEO], second[LTX_VIDEO]
    if args.a_reverse:
        a = _reverse(a)
    joined = str(Path(b).with_name(f"fakeout_chain_s{args.seed}_joined.mp4"))
    # drop B's first frame: it is the same pinned still as A's last
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-i", b, "-filter_complex",
                    "[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[b];[0:v][b]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", joined], check=True)
    print(f"Elapsed {time.monotonic() - started:.0f} s, chain {args.high_frames} + {frames} frames, seed {args.seed}")
    print(f"A: {a}\nB: {b}\nJoined: {joined}")


def _hold(client, args, words: dict, clip: str, prefix: str) -> str:
    """An ambient end hold: image-to-video from the clip's last frame (pinned at frame 0 only); returns its path."""
    still = str(Path(clip).with_name(Path(clip).stem + "_last.png"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", clip, "-update", "1", "-frames:v", "1",
                    still], check=True)
    graph: dict = {}
    pins = [(0, _load(graph, client, "46", Path(still)))]
    saved = _submit(client, graph, tuple(_ltx(graph, args, args.hold_frames, pins, HOLD.format(**words),
                                              args.seed + 99, prefix + "_hold")), args.output_dir)
    return saved[LTX_VIDEO]


def _join(clips: list[str], out: str) -> None:
    """Concatenate segments that share their boundary frame (each later clip's first frame is dropped)."""
    inputs = [x for c in clips for x in ("-i", c)]
    parts = "".join(f"[{k}:v]trim=start_frame={0 if k == 0 else 1},setpts=PTS-STARTPTS[v{k}];" for k in range(len(clips)))
    chain = "".join(f"[v{k}]" for k in range(len(clips)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
                    f"{parts}{chain}concat=n={len(clips)}:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-crf",
                    "16", "-pix_fmt", "yuv420p", out], check=True)


def _ladder(args, words: dict) -> None:
    """Stills from the end shot upwards (each a klein edit of the one below), then one first-last LTX segment per
    adjacent pair, top first, joined on the shared stills. Writes fakeout_ladder_s<seed>_joined.mp4."""
    rungs = [int(h) for h in args.rungs.split(",")]
    for n in (args.rung_frames, args.top_frames, args.last_frames):
        if n % 8 != 1:
            raise SystemExit(f"segments need 8k+1 frames, not {n}")
    prefix = f"geoblock/fakeout_ltx_ladder_s{args.seed}"
    started = time.monotonic()
    with _client(args.url) as client:
        graph: dict = {}
        wanted: list[str] = []
        below = _end_still(graph, client, args, words, prefix, wanted)
        nodes = ["107" if "107" in wanted else None]  # saved still per rung, end first
        for k, metres in enumerate(rungs):
            base = 200 + 20 * k
            text = RUNG_STILL.format(feet=f"{round(metres * 3.281, -1):,.0f}", metres=metres, region=args.region,
                                     figure=RUNG_FIGURE if metres <= 300 else "")
            if k == len(rungs) - 1:
                text = text.replace("Daylight", RUNG_UNDER_CLOUD + "Daylight")
            below = _klein(graph, base, text, args.seed + base, f"{prefix}_rung{metres}m", (below,))
            nodes.append(str(base + 7))
            wanted.append(str(base + 7))
        if args.top == "orbit":
            _orbit_still(graph, client, args, words, prefix, wanted)
            _klein(graph, 140, CLOUD_STILL, args.seed + 140, prefix + "_cloud")
            wanted.append("147")
        stills = _submit(client, graph, tuple(wanted), args.output_dir)
        if args.end_still:
            stills["end"] = str(args.end_still)
        paths = [Path(stills[n] if n else stills["end"]) for n in nodes]  # end, rung 1, ..., top rung
        print("Stills:", *paths, sep="\n  ")
        if args.stills_only:
            return
        clips = []

        def segment(name: str, frames: int, pins: list, prompt: str, seed: int) -> str:
            graph: dict = {}
            loaded = [(index, _load(graph, client, str(41 + j), path)) for j, (index, path) in enumerate(pins)]
            saved = _submit(client, graph, tuple(_ltx(graph, args, frames, loaded, prompt, seed,
                                                      f"{prefix}_{name}")), args.output_dir)
            return saved[LTX_VIDEO]

        if args.top == "rise":
            clips.append(_reverse(segment("top", args.top_frames, [(0, paths[-1])],
                                          CHAIN_A_UP.format(**words), args.seed)))
        elif args.top == "orbit":
            orbit = Path(stills["127"]) if "127" in stills else args.orbit_still
            cloud = round(0.4 * (args.top_frames - 1) / 8) * 8
            clips.append(segment("top", args.top_frames, [(0, orbit), (cloud, Path(stills["147"])),
                                                           (-1, paths[-1])], CHAIN_A.format(**words), args.seed))
        for k in range(len(rungs), 0, -1):  # top rung down to the end shot
            hi, lo = rungs[k - 1], (rungs[k - 2] if k > 1 else 0)
            prompt = (LAST_SEGMENT.format(**words) if k == 1 else RUNG_SEGMENT.format(
                feet_hi=f"{round(hi * 3.281, -1):,.0f}", feet_lo=f"{round(lo * 3.281, -1):,.0f}", region=args.region))
            clips.append(segment(f"seg{hi}m", args.last_frames if k == 1 else args.rung_frames,
                                 [(0, paths[k]), (-1, paths[k - 1])], prompt,
                                 args.seed + k))
        if args.hold_frames:
            clips.append(_hold(client, args, words, clips[-1], prefix))
    joined = str(Path(clips[-1]).with_name(f"fakeout_ladder_s{args.seed}_joined.mp4"))
    _join(clips, joined)
    print(f"Elapsed {time.monotonic() - started:.0f} s, ladder {rungs}, top {args.top}")
    print("Segments:", *clips, f"Joined: {joined}", sep="\n  ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("ltx", "wan"), default="ltx")
    parser.add_argument("--mode", choices=("reverse", "first-last", "end-only", "chain", "ladder", "zoom", "nadir"), default="reverse",
                        help="LTX only; Wan always runs reverse")
    parser.add_argument("--person", default=PERSON, help="who stands at the end (ignored with --subject)")
    parser.add_argument("--subject", type=Path, help="a photo of the person; klein keeps their look")
    parser.add_argument("--scene", default=SCENE, help="where they stand, as a phrase after 'standing'")
    parser.add_argument("--region", default=REGION, help="what the orbit and the cloud exit look down on")
    parser.add_argument("--approach", default=APPROACH, help="what the low swoop passes over")
    parser.add_argument("--end-reference", type=Path, help="a real view of the place (a GeoBlock render or run "
                        "frame): klein paints the end still from it, keeping the place")
    parser.add_argument("--orbit-still", type=Path, help="first-last: use this image (e.g. our measured orbit "
                        "render) as frame 0 instead of a klein orbit still")
    parser.add_argument("--end-still", type=Path, help="use this image as the end still instead of painting one")
    parser.add_argument("--cloud-at", type=float, help="first-last/end-only: also pin a klein whiteout still at "
                        "this fraction of the clip (e.g. 0.15)")
    parser.add_argument("--aerial-at", type=float, help="first-last/end-only: also pin a klein low aerial still at "
                        "this fraction of the clip (e.g. 0.4), to steady the swoop")
    parser.add_argument("--high-still", type=Path, help="chain: use this image as the high (~6 km) still")
    parser.add_argument("--high-frames", type=int, default=97, help="chain: length of segment A (orbit to high "
                        "still), 8k+1")
    parser.add_argument("--low-at", type=float, help="chain: pin a low aerial still at this fraction of segment B "
                        "(e.g. 0.6), so it flies the last stretch instead of dissolving into the end still")
    parser.add_argument("--low-still", type=Path, help="chain: the low aerial still for --low-at (default: klein)")
    parser.add_argument("--a-reverse", action="store_true", help="chain: make segment A by rising from the high "
                        "still to orbit (only the high still pinned) and playing it backwards, instead of first-last")
    parser.add_argument("--rungs", default="30,180,900,3000,6000", help="ladder: heights in metres above the "
                        "end shot, lowest first; the top one sits just under the cloud deck")
    parser.add_argument("--rung-frames", type=int, default=41, help="ladder: frames per segment (8k+1)")
    parser.add_argument("--last-frames", type=int, default=97, help="ladder: frames of the last segment, lowest "
                        "rung to the end shot, which carries the slow push-in (8k+1)")
    parser.add_argument("--top", choices=("orbit", "rise", "none"), default="rise", help="ladder: above the top "
                        "rung, first-last from the orbit still with a whiteout pin, a rise-and-reverse to orbit, "
                        "or nothing")
    parser.add_argument("--top-frames", type=int, default=73, help="ladder: frames of the top segment (8k+1)")
    parser.add_argument("--hold-frames", type=int, help="append an ambient end hold of this many frames (8k+1, "
                        "e.g. 49), its own image-to-video clip from the last frame")
    parser.add_argument("--zoom-steps", type=int, default=3, help="zoom: outpainted rungs above the 600 m still, "
                        "x3 each (3 = 1.8, 5.4, 16.2 km)")
    parser.add_argument("--zoom-frames", type=int, default=96, help="zoom: frames of the digital dive")
    parser.add_argument("--place", default="san-francisco-twin-peaks", help="nadir: catalog place to dive onto")
    parser.add_argument("--dive-frames", type=int, default=111, help="nadir: frames of the real-imagery dive")
    parser.add_argument("--dive-end-m", type=float, help="nadir: altitude of the dive's last frame (default 400)")
    parser.add_argument("--dive-bearing", type=float, help="nadir: screen-up bearing at the end of the dive "
                        "(default: the way the person looks)")
    parser.add_argument("--dive-spin", type=float, default=0.0, help="nadir: degrees the view turns over the dive")
    parser.add_argument("--swoop", type=Path, help="nadir: join this existing swoop clip (with --end-still) instead "
                        "of making one with LTX; frame the dive to land on its first frame")
    parser.add_argument("--cloud-cover", type=float, default=0.35, help="nadir: regional cloud cover 0..1 "
                        "(always thick over the place, so the dive punches through)")
    parser.add_argument("--no-ambient", action="store_true", help="leave the clouds/birds/wind sentence out")
    parser.add_argument("--stills-only", action="store_true", help="paint the stills, no video")
    parser.add_argument("--frames", type=int, help="8k+1 for LTX (default 121 = 5 s at 24 fps), 4k+1 for Wan "
                                                   "(default 81 = 5 s at 16 fps)")
    parser.add_argument("--prompt", help="override the video prompt")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--no-upscale", action="store_true", help="LTX stage 1 only (960x544)")
    parser.add_argument("--wan-high", default=WAN_MODELS["high"])
    parser.add_argument("--wan-low", default=WAN_MODELS["low"])
    parser.add_argument("--wan-text-encoder", default=WAN_MODELS["text_encoder"])
    parser.add_argument("--wan-vae", default=WAN_MODELS["vae"])
    parser.add_argument("--wan-lora", default=WAN_MODELS["lora"])
    parser.add_argument("--wan-lora-strength", type=float, default=0.85)
    parser.add_argument("--url", help="ComfyUI server (default: COMFY_BASE_URL)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/fakeout"))
    args = parser.parse_args()

    mode = "reverse" if args.engine == "wan" else args.mode
    frames = args.frames or (81 if args.engine == "wan" else 121)
    if args.engine == "ltx" and frames % 8 != 1:
        raise SystemExit(f"LTX needs 8k+1 frames (e.g. 73, 121, 241), not {frames}")
    if args.engine == "wan" and frames % 4 != 1:
        raise SystemExit(f"Wan needs 4k+1 frames (e.g. 49, 81), not {frames}")
    words = {"person": "the person" if args.subject else args.person, "scene": args.scene, "region": args.region,
             "approach": args.approach, "ambient": "" if args.no_ambient else AMBIENT}
    if mode == "chain":
        return _chain(args, words, frames)
    if mode == "ladder":
        return _ladder(args, words)
    if mode == "zoom":
        return _zoom(args, words)
    if mode == "nadir":
        return _nadir(args, words)
    prefix = f"geoblock/fakeout_{args.engine}_{mode}_s{args.seed}"

    with _client(args.url) as client:
        graph: dict = {}
        wanted: list[str] = []
        end = _end_still(graph, client, args, words, prefix, wanted)
        orbit = _orbit_still(graph, client, args, words, prefix, wanted) if mode == "first-last" else None
        mids = []  # LTX guides land on latent frames: keep them on multiples of 8
        for base, fraction, text in ((140, args.cloud_at, CLOUD_STILL), (160, args.aerial_at, AERIAL_STILL)):
            if fraction is not None and mode != "reverse":
                image = _klein(graph, base, text.format(**words), args.seed + base, f"{prefix}_mid{base}")
                mids.append((round(fraction * (frames - 1) / 8) * 8, image))
                wanted.append(str(base + 7))
        if not args.stills_only:
            prompt = args.prompt or (ZOOM_OUT if mode == "reverse" else ZOOM_IN).format(**words)
            if args.engine == "wan":
                wanted.append(_wan(graph, end, frames, prompt, args.seed, prefix, args))
            else:
                pins = {"reverse": [(0, end)], "first-last": [(0, orbit), *mids, (-1, end)],
                        "end-only": [*mids, (-1, end)]}[mode]
                wanted += _ltx(graph, args, frames, pins, prompt, args.seed, prefix)
        print(f"Video prompt: {graph.get('305', graph.get('508', {})).get('inputs', {}).get('text', '(none)')}")
        started = time.monotonic()
        saved = _submit(client, graph, tuple(wanted), args.output_dir)
        if args.hold_frames and LTX_VIDEO in saved and mode != "reverse":
            hold = _hold(client, args, words, saved[LTX_VIDEO], prefix)
            joined = str(Path(hold).with_name(Path(saved[LTX_VIDEO]).stem + "_with_hold.mp4"))
            _join([saved[LTX_VIDEO], hold], joined)
            print(f"With hold: {joined}")
    elapsed = time.monotonic() - started
    video = saved.get(LTX_VIDEO) or saved.get("515")
    if video and mode == "reverse":
        print(f"Zoom-in: {_reverse(video)}")
    print(f"Elapsed {elapsed:.0f} s, {args.engine} {mode}, {frames} frames, seed {args.seed}: {saved}")


if __name__ == "__main__":
    main()
