#!/usr/bin/env bash
set -euo pipefail

# FLUX 3 Image Klein, the ComfyUI package: the distilled model's weights in the layout ComfyUI expects, plus
# the Qwen3-VL text encoder it needs.
#
# Accept access at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy and log in first
# (``hf auth login`` with the account that accepted it). The original release,
# black-forest-labs/flux-3-image-klein-base, is the same model in the diffusers layout; this package is the
# one ComfyUI loads directly.
#
# Usage: bash scripts/download_flux3_klein.sh [destination]      (default: models/FLUX-3-Klein)
#
# The int8-convrot weights are the ones to use on a 24 GB card: the transformer is 5.8 GB rather than 11.1 GB and
# the text encoder 4 GB rather than 8.9 GB. The bf16 pair is fetched as well, for a card with room, and for
# ComfyUI v0.41.3+ (the release that loads FLUX 3 weights directly).
repo="black-forest-labs/flux-3-image-klein-comfy"
destination="${1:-models/FLUX-3-Klein}"

hf download "$repo" \
  diffusion_models/flux3-image-klein-int8-convrot.safetensors \
  text_encoders/qwen3vl_4b_int8-convrot.safetensors \
  diffusion_models/flux3-image-klein-bf16.safetensors \
  text_encoders/qwen3vl_4b_bf16.safetensors \
  vae/flux2-vae.safetensors \
  workflows/flux3_klein_bf16_i2i.json \
  workflows/flux3_klein_bf16_mi2i.json \
  workflows/README.md workflows/LICENSE.upstream \
  README.md LICENSE.md NOTICE.md \
  --local-dir "$destination"

echo
echo "Now point ComfyUI at them. The pipeline asks ComfyUI for these names:"
echo "  diffusion_models/flux3-image-klein-bf16.safetensors   (or flux3-image-klein-int8-convrot.safetensors)"
echo "  text_encoders/qwen3vl_4b_bf16.safetensors              (or qwen3vl_4b_int8-convrot.safetensors)"
echo "  vae/flux2-vae.safetensors"
echo "They are in $destination in the same folders; copy or symlink them into ComfyUI's models/ directory,"
echo "or point ComfyUI's extra_model_paths.yaml at $destination."
echo "FLUX 3 needs ComfyUI v0.41.3 or newer; on an older ComfyUI use FLUX.2 [klein] instead"
echo "(TOUCHDOWN_STILLS=klein, or flux3 = [] in the manifest)."
