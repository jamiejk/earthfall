#!/usr/bin/env bash
# Install the final-upscaler custom nodes and weights for scripts/upscaler_shootout.py and --final-upscale.
#
# usage: scripts/setup_upscalers.sh <ComfyUI dir> <live models dir> [<python of the ComfyUI venv>]
#   e.g. scripts/setup_upscalers.sh "/run/media/jamie/Data/Image Creation/Comfy" \
#            "/run/media/jamie/Image Creation/Live-Models"
# Weights go into the live models folder (keep them off NTFS) and are symlinked into ComfyUI/models.
# The LTX Refine Details IC-LoRA is gated: accept the terms on Hugging Face and export HF_TOKEN first.
# Pinned node versions: SeedVR2 v2.5.24 (2025-12-24), FlashVSR Ultra Fast (2025-11-20).
set -euo pipefail
comfy=${1:?ComfyUI dir}; live=${2:?live models dir}; py=${3:-"$comfy/venv/bin/python"}
[ -x "$py" ] || py=$(command -v python3)
fetch() {  # repo file dest_dir
  mkdir -p "$3"
  [ -s "$3/$(basename "$2")" ] && return 0
  "$py" - "$1" "$2" "$3" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo, name, dest = sys.argv[1:]
print("ok", hf_hub_download(repo, name, local_dir=dest))
EOF
}
link() {  # live_path comfy_path
  mkdir -p "$(dirname "$2")"
  if [ "$(realpath "$(dirname "$1")")" = "$(realpath "$(dirname "$2")")" ]; then return 0; fi  # same folder already
  if [ -e "$2" ] && [ ! -L "$2" ]; then echo "leaving existing $2"; else ln -sfn "$1" "$2"; fi
}

# Custom nodes
cd "$comfy/custom_nodes"
[ -d ComfyUI-SeedVR2_VideoUpscaler ] || git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
git -C ComfyUI-SeedVR2_VideoUpscaler checkout -q 4490bd1
[ -d ComfyUI-FlashVSR_Ultra_Fast ] || git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast
git -C ComfyUI-FlashVSR_Ultra_Fast checkout -q 4820b3f
# Their requirements list torch unpinned; keep the installed torch.
"$py" -m pip install -q omegaconf "diffusers>=0.33.1" "peft>=0.17.0" "rotary_embedding_torch>=0.5.3" gguf einops psutil

# SeedVR2: 3B fp8, 7B fp8 (last block fp16), VAE
fetch numz/SeedVR2_comfyUI seedvr2_ema_3b_fp8_e4m3fn.safetensors "$live/SEEDVR2"
fetch AInVFX/SeedVR2_comfyUI seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors "$live/SEEDVR2"
fetch numz/SeedVR2_comfyUI ema_vae_fp16.safetensors "$live/SEEDVR2"
link "$live/SEEDVR2" "$comfy/models/SEEDVR2"

# FlashVSR v1.1 (tiny decoder; the Wan VAE is only for "full" mode)
for f in diffusion_pytorch_model_streaming_dmd.safetensors LQ_proj_in.ckpt TCDecoder.ckpt; do
  fetch JunhaoZhuang/FlashVSR-v1.1 "$f" "$live/FlashVSR-v1.1"
done
link "$live/FlashVSR-v1.1" "$comfy/models/FlashVSR-v1.1"

# Real-ESRGAN x2plus (BSD-3), the frame-by-frame baseline
mkdir -p "$live/upscale_models"
[ -s "$live/upscale_models/RealESRGAN_x2plus.pth" ] || curl -fL -o "$live/upscale_models/RealESRGAN_x2plus.pth" \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
link "$live/upscale_models/RealESRGAN_x2plus.pth" "$comfy/models/upscale_models/RealESRGAN_x2plus.pth"

# LTX 2.5 Refine Details IC-LoRA (gated; LTX 2.5 transformer, VAE and Gemma are already installed)
fetch Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors "$live/loras" \
  || echo "Refine Details LoRA not downloaded (gated?): ltx-detail will fail, the others still run"
link "$live/loras/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors" \
  "$comfy/models/loras/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"

echo "Done. Restart ComfyUI so it loads the new nodes."
