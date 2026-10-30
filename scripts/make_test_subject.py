"""Paint a stand-in "user photo" with FLUX.2 [klein] to test --subject without a real person's photo.

It is deliberately a casual phone photo (busy street behind, cut off at the knees), the hard case the subject
sheet has to clean up. Saves the image and prints its path, to pass as ``geoblock-run --subject``.

    uv run python scripts/make_test_subject.py --output-dir outputs/test_subject
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoblock_core.run import _client, _submit  # noqa: E402
from geoblock_core.workflow import LTX_MODELS  # noqa: E402

PEOPLE = {
    "red-raincoat": ("A casual smartphone photo of a young woman with short curly dark hair, wearing a bright red "
                     "raincoat over a grey hoodie, black jeans and white trainers, standing on a busy city street "
                     "with shops, cars and passers-by behind her, framed from the knees up, slightly off centre, "
                     "daylight."),
    "denim-jacket": ("A casual smartphone selfie-style photo of a middle-aged man with a grey beard and glasses, "
                     "wearing a faded denim jacket over a mustard yellow t-shirt, head and shoulders, in a cafe "
                     "with warm lights behind him."),
}


def graph(text: str, seed: int, prefix: str) -> dict:
    width, height = 832, 1248
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": LTX_MODELS["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": LTX_MODELS["flux_clip"], "type": "flux2",
                                                    "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": LTX_MODELS["flux_vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": text}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        "8": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                                                   "cfg": 1.0}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["8", 0], "sampler": ["9", 0], "sigmas": ["7", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", choices=tuple(PEOPLE), default="red-raincoat")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--url", help="ComfyUI behind comfy-api-proxy (default: COMFY_BASE_URL)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_subject"))
    args = parser.parse_args()
    with _client(args.url) as client:
        saved = _submit(client, graph(PEOPLE[args.person], args.seed, f"geoblock/test_subject_{args.person}"),
                        ("13",), args.output_dir)
    print(saved["13"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
