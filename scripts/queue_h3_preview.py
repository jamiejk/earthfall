"""Queue a five-second Worm's Head MiniMax H3 reference-to-video preview."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_pack_geometry

PACK = ROOT / "data/reference_images/worms_head/h3_reference_pack.json"
WORKFLOW = ROOT / "outputs/h3_preview_workflow_api.json"
PREVIEW_VIDEO = "flyover_reference_h3_preview.mp4"
FRAME_COUNT = 124
NEXT_SOURCE_VIDEO = ROOT / "outputs/B_lidar_full_2m/flyover_reference_v2_h3.mp4"
NEXT_PREVIEW_VIDEO = "flyover_reference_h3_next.mp4"
NEXT_FRAME_COUNT = 175  # 17 * 10 + 5 frames, about 7.3 seconds at 24 fps
NEXT_WORKFLOW = ROOT / "outputs/h3_next_workflow_api.json"
FIXED_SOURCE_VIDEO = ROOT / "outputs/B_lidar_full_2m/flyover_reference_v3_h3.mp4"
FIXED_PREVIEW_VIDEO = "flyover_reference_h3_fixed.mp4"
FIXED_WORKFLOW = ROOT / "outputs/h3_fixed_workflow_api.json"
VEDA_PREDICTOR = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"


def graph(pack: dict, *, next_take: bool = False, fixed_take: bool = False,
          veda: bool = False) -> dict:
    next_take = next_take or fixed_take
    if next_take:
        prompt = (
            "<Picture 1> is the opening Rhossili cliff-edge view. "
            "<Picture 2> shows the full connected Worm's Head promontory and tidal causeway. "
            "<Video 1> supplies the continuous out-and-back camera route. "
            "A single, slower, steady aerial flyover of the real Worm's Head in Rhossili, Wales. "
            "Begin at the headland edge, fly west beside the connected rocky causeway, pass Inner and Outer Head, "
            "make a broad, smooth 180-degree turn over open water beyond Outer Head, then fly east toward Rhossili. "
            "Keep the coastline, cliffs, and causeway stable and recognizable throughout the turn; no jump, cut, "
            "morph, or sudden camera whip. Retain the simple clay-like landform geometry from the motion reference. "
            "The sea is lively: rolling swells break repeatedly against the rocky shore, with irregular crashing "
            "whitewater, spreading foam, and light spray along both sides of Worm's Head. The breakers move and "
            "dissipate naturally rather than forming a static white seam. Daylight, no people, no text."
        )
        if fixed_take:
            prompt += (
                " Show distinct wave fronts travelling in from open sea, cresting, striking the rocks, "
                "throwing up white spray, then receding; several impacts happen during the shot."
                " Finish while still flying back toward the headland; keep the same uninterrupted viewpoint "
                "through the last frame. Do not reset to the opening view or switch shots near the end."
            )
    else:
        prompt = (
            "<Picture 1> is the opening Rhossili cliff-edge view. "
            "<Picture 2> shows the full connected Worm's Head promontory and tidal causeway. "
            "<Video 1> supplies the exact out-and-back camera route. "
            + pack["prompt"]
        )
    frame_count = NEXT_FRAME_COUNT if next_take else FRAME_COUNT
    reference_video = FIXED_PREVIEW_VIDEO if fixed_take else (NEXT_PREVIEW_VIDEO if next_take else PREVIEW_VIDEO)
    output_prefix = "video/WormsHead_H3_fixed" if fixed_take else ("video/WormsHead_H3_next" if next_take else "video/WormsHead_H3_preview")
    folder = "geoblock_worms_head/"
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": folder + "commons_115697390.jpg"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": folder + "commons_133768741.jpg"}},
        "6": {"class_type": "LoadVideo", "inputs": {"file": folder + reference_video}},
        "7": {"class_type": "GetVideoComponents", "inputs": {"video": ["6", 0]}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
            "width": 864, "height": 480, "length": frame_count, "ref_image_size": "match",
            "ref_images.ref_image_0": ["4", 0], "ref_images.ref_image_1": ["5", 0],
            "ref_videos.ref_video_0": ["7", 0],
        }},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": 20261003}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["8", 0]}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "beta", "steps": 12, "denoise": 1.0}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["10", 0], "sampler": ["11", 0],
            "sigmas": ["12", 0], "latent_image": ["8", 1],
        }},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {
            "video": ["15", 0], "filename_prefix": output_prefix, "format": "auto", "codec": "auto",
        }},
    }
    if veda:
        # Keep reference attention dense: this predictor was trained on T2VA,
        # while our geographic identity and camera route come from references.
        workflow["20"] = {"class_type": "VedaSparseAttention", "inputs": {
            "model": ["1", 0], "predictor": VEDA_PREDICTOR,
            "generated_sparsity": "90%", "reference_sparsity": "0%",
            "verbose": True,
        }}
        workflow["10"]["inputs"]["model"] = ["20", 0]
        workflow["12"]["inputs"]["model"] = ["20", 0]
        workflow["16"]["inputs"]["filename_prefix"] += "_veda"
    return workflow


def stage_inputs(pack: dict, comfy_root: Path, *, next_take: bool = False, fixed_take: bool = False) -> None:
    next_take = next_take or fixed_take
    target = comfy_root / "input/geoblock_worms_head"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("commons_115697390.jpg", "commons_133768741.jpg"):
        item = next(image for image in pack["image_references"] if Path(image["path"]).name == name)
        source = Path(item["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Reference photo changed: {source}")
        destination = target / name
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            shutil.copy2(source, destination)

    source = FIXED_SOURCE_VIDEO if fixed_take else (NEXT_SOURCE_VIDEO if next_take else Path(pack["motion_reference"]))
    destination = target / (FIXED_PREVIEW_VIDEO if fixed_take else (NEXT_PREVIEW_VIDEO if next_take else PREVIEW_VIDEO))
    frame_count = NEXT_FRAME_COUNT if next_take else FRAME_COUNT
    if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
        return
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
         "-of", "json", str(source)], check=True, capture_output=True, text=True,
    )
    source_frames = int(json.loads(probe.stdout)["streams"][0]["nb_frames"])
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-vf", f"setpts=(PTS-STARTPTS)*{frame_count}/{source_frames},fps=24,scale=864:480:flags=lanczos",
         "-frames:v", str(frame_count), "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", str(destination)], check=True,
    )
    print(f"Prepared {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, default=Path("/run/media/jamie/Data/Image Creation/Comfy"))
    parser.add_argument("--take", choices=("preview", "next", "fixed"), default="preview")
    parser.add_argument("--veda", action="store_true",
                        help="Use optional Veda sparse attention; keep reference attention dense")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    pack = json.loads(PACK.read_text())
    print(f"3D height evidence: {qualify_pack_geometry(pack)}")
    if args.take == "preview":
        video = Path(pack["motion_reference"])
        if hashlib.sha256(video.read_bytes()).hexdigest() != pack.get("motion_reference_sha256"):
            raise ValueError("Motion reference differs from the qualified 3D blocking video.")
    next_take = args.take in ("next", "fixed")
    fixed_take = args.take == "fixed"
    if args.submit:
        stage_inputs(pack, args.comfy_root, next_take=next_take, fixed_take=fixed_take)
    workflow = graph(pack, next_take=next_take, fixed_take=fixed_take, veda=args.veda)
    workflow_path = FIXED_WORKFLOW if fixed_take else (NEXT_WORKFLOW if next_take else WORKFLOW)
    if args.veda:
        workflow_path = workflow_path.with_name(workflow_path.stem + "_veda.json")
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(workflow, indent=2) + "\n")
    print(f"Saved API workflow to {workflow_path}")
    if args.submit:
        if args.veda:
            with urlopen(args.url + "/object_info/VedaSparseAttention", timeout=10) as response:
                if "VedaSparseAttention" not in json.load(response):
                    raise RuntimeError("VedaSparseAttention is not loaded in this ComfyUI instance")
        request = Request(args.url + "/prompt", data=json.dumps({"prompt": workflow}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            print(response.read().decode())


if __name__ == "__main__":
    main()
