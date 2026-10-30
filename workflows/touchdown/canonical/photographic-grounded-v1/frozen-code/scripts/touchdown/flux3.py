"""FLUX 3 Image Klein stills on the side-by-side ComfyUI v0.41 install (plain ComfyUI HTTP API, no SDK proxy).

The graph is BFL's distilled bf16 image-edit workflow flattened to API form, as in the 17 Oct trial
(ComfyUI-flux3/trial_2026-10-17/flux3_trial.py): 16 steps, no guidance, references through ReferenceMultiLatent,
the prompt used as typed (no upsampler). Standard library only, so it runs in the build's own Python.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

W, H = 1344, 768


def _call(url: str, path: str, data: dict | None = None):
    req = urllib.request.Request(url + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _upload(url: str, path: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)["name"]


def graph(prompt: str, seed: int, refs: list[str], prefix: str, width: int = W, height: int = H,
          inpaint: str | None = None) -> dict:
    """FLUX 3 Klein text/reference-to-image. With ``inpaint`` (an uploaded RGBA image of exactly width x height)
    the sampler starts from that image and repaints only where its alpha is 0 (ComfyUI's LoadImage mask)."""
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": os.environ.get("TOUCHDOWN_FLUX3_UNET", "flux3-image-klein-bf16.safetensors"),
                                                      "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": os.environ.get("TOUCHDOWN_FLUX3_CLIP", "qwen3vl_4b_bf16.safetensors"), "type": "flux3",
                                                      "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": os.environ.get("TOUCHDOWN_FLUX3_VAE", "flux2-vae.safetensors")}},
         "4": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0], "max_shift": 2.77259,
                                                             "base_shift": 2.77259, "width": 1024, "height": 1024}},
         "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
         "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}}}
    pos, neg = ["10", 0], ["11", 0]
    if refs:
        rm = {"positive": pos, "negative": neg, "vae": ["3", 0], "resolution": "resize", "resolution.megapixels": 1.0}
        for k, name in enumerate(refs):
            g[f"r{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            rm[f"images.image{k}"] = [f"r{k}", 0]
        g["12"] = {"class_type": "ReferenceMultiLatent", "inputs": rm}
        pos, neg = ["12", 0], ["12", 1]
    if inpaint:
        g.update({"m0": {"class_type": "LoadImage", "inputs": {"image": inpaint}},
                  "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["m0", 0], "vae": ["3", 0]}},
                  "17": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["m0", 1]}}})
        latent = ["17", 0]
    else:
        g["13"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["13", 0]
    g.update({
        "14": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": seed, "steps": 16, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                                                    "positive": pos, "negative": neg, "latent_image": latent}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}}})
    return g


def still(url: str, refs: list[Path], prompt: str, seed: int, prefix: str, out: Path, timeout_s: int = 900) -> Path:
    """Render one FLUX 3 Klein still with ``refs`` as image 1, image 2, ... and save it to ``out``."""
    names = [_upload(url, Path(p)) for p in refs]
    pid = _call(url, "/prompt", {"prompt": graph(prompt, seed, names, prefix), "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while True:
        h = _call(url, f"/history/{pid}")
        status = h.get(pid, {}).get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError("FLUX 3 still failed: " + json.dumps(status)[:2000])
        if status.get("completed"):
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError("FLUX 3 still timed out")
        time.sleep(1)
    image = h[pid]["outputs"]["16"]["images"][0]
    query = urllib.parse.urlencode({"filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]})
    with urllib.request.urlopen(f"{url}/view?{query}") as r:
        out.write_bytes(r.read())
    return out
