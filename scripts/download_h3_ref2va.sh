#!/usr/bin/env bash
set -euo pipefail

# ComfyUI's Ref2VA set for an RTX 4090. Run only where you have H3 rights.
repo="Comfy-Org/MiniMax-H3"
revision="e5eb578a89295337b8ff433a035929ce0279e0b6"
destination="${1:-models/MiniMax-H3}"

hf download "$repo" \
  diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  --revision "$revision" --local-dir "$destination"

hf cache verify "$repo" --revision "$revision" --local-dir "$destination"
