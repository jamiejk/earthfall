#!/usr/bin/env bash
# Set up and start the full GeoBlock graph on a GPU pod (RunPod or any Linux box):
# ComfyUI + GeoBlock nodes + Blender (bpy module) + MiniMax H3 Ref2VA + comfy-api-proxy.
#
# Re-runnable: finished steps are skipped. Put WORK on a network volume so the
# ~42 GB of H3 weights survive pod restarts.
#
#   GEOBLOCK_PROXY_TOKEN=<random secret> bash scripts/runpod/bootstrap.sh
#
# Then, from your machine (RunPod exposes HTTP port 8189 as below):
#   export GEOBLOCK_PROXY_TOKEN=<same secret>
#   uv run geoblock-run --blocking-url https://<pod-id>-8189.proxy.runpod.net ...
#
# Environment:
#   GEOBLOCK_PROXY_TOKEN  required; bearer token for the public v2 API
#   WORK                  install root (default /workspace)
#   GEOBLOCK_REPO/_REF    GeoBlock source (a private repo needs a token in the URL)
#   SKIP_MODELS=1         skip the H3 download (blocking videos only)
#   COMFY_ARGS            extra ComfyUI flags, e.g. --cpu for a GPU-less test
#   HF_TOKEN              only if Hugging Face asks for it
set -euo pipefail

: "${GEOBLOCK_PROXY_TOKEN:?Set GEOBLOCK_PROXY_TOKEN so the public API requires a token}"
WORK="${WORK:-/workspace}"
GEOBLOCK_REPO="${GEOBLOCK_REPO:-https://github.com/jamiejk/geoblock-scenecompiler.git}"
GEOBLOCK_REF="${GEOBLOCK_REF:-claude/pensive-euler-du646r}"
COMFY_COMMIT="5c460d8172fe30761ff67c0df3d5643bb74e0d70"  # tested with GeoBlock on 5 Oct 2026
BPY_VERSION="5.2.2"
H3_REPO="Comfy-Org/MiniMax-H3"
H3_REVISION="e5eb578a89295337b8ff433a035929ce0279e0b6"
COMFY="$WORK/ComfyUI"
LOGS="$WORK/logs"
mkdir -p "$WORK" "$LOGS"

step() { printf '\n== %s\n' "$*"; }

step "System packages"
missing=()
for pkg in git ffmpeg libgl1 libegl1 libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if ((${#missing[@]})); then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

step "ComfyUI $COMFY_COMMIT"
if [[ ! -d "$COMFY/.git" ]]; then
  git clone -q https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
git -C "$COMFY" fetch -q --depth 1 origin "$COMFY_COMMIT" && git -C "$COMFY" checkout -q "$COMFY_COMMIT"
[[ -x "$WORK/comfy-venv/bin/python" ]] || uv venv -q -p 3.12 "$WORK/comfy-venv"

step "GeoBlock ($GEOBLOCK_REF)"
if [[ ! -d "$WORK/geoblock/.git" ]]; then
  git clone -q "$GEOBLOCK_REPO" "$WORK/geoblock"
fi
git -C "$WORK/geoblock" fetch -q origin "$GEOBLOCK_REF" && git -C "$WORK/geoblock" checkout -q FETCH_HEAD
VIRTUAL_ENV="$WORK/comfy-venv" uv pip install -q -r "$COMFY/requirements.txt" \
  -e "$WORK/geoblock" comfy-api-proxy "huggingface_hub[cli]"
ln -sfn "$WORK/geoblock/comfy_geoblock_prototype" "$COMFY/custom_nodes/comfy_geoblock_prototype"

step "Blender as the bpy module ($BPY_VERSION)"
if [[ ! -x "$WORK/bpy-venv/bin/python" ]]; then
  uv venv -q -p 3.13 "$WORK/bpy-venv"
  VIRTUAL_ENV="$WORK/bpy-venv" uv pip install -q "bpy==$BPY_VERSION"
fi
export GEOBLOCK_BLENDER="$WORK/bpy-venv/bin/python"
"$GEOBLOCK_BLENDER" -c "import bpy; print('bpy', bpy.app.version_string)"

if [[ "${SKIP_MODELS:-0}" != 1 ]]; then
  step "MiniMax H3 Ref2VA weights (~42 GB, pinned revision)"
  "$WORK/comfy-venv/bin/hf" download "$H3_REPO" \
    diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors \
    vae/minimax_h3_video_vae_fp16.safetensors \
    --revision "$H3_REVISION" --local-dir "$COMFY/models"
fi

step "Start ComfyUI (local only) and comfy-api-proxy (public, token required)"
pkill -f "$COMFY/main.py" 2>/dev/null || true
pkill -f "comfy-api-proxy run" 2>/dev/null || true
cd "$COMFY"
# ComfyUI itself has no authentication, so it stays on loopback. Reach its UI
# through an SSH tunnel: ssh -L 8188:127.0.0.1:8188 <pod ssh address>
nohup env GEOBLOCK_BLENDER="$GEOBLOCK_BLENDER" GEOBLOCK_DATA_DIR="$WORK/geoblock-data" \
  "$WORK/comfy-venv/bin/python" "$COMFY/main.py" --listen 127.0.0.1 --port 8188 ${COMFY_ARGS:-} \
  > "$LOGS/comfyui.log" 2>&1 &
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8188/object_info/GeoBlockRenderBlockingVideo >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:8188/object_info/GeoBlockRenderBlockingVideo >/dev/null \
  || { echo "ComfyUI did not load the GeoBlock nodes; see $LOGS/comfyui.log"; exit 1; }
nohup "$WORK/comfy-venv/bin/comfy-api-proxy" run --comfyui http://127.0.0.1:8188 \
  --host 0.0.0.0 --port 8189 --token "$GEOBLOCK_PROXY_TOKEN" --comfyui-base-dir "$COMFY" \
  --max-upload-mb 500 > "$LOGS/proxy.log" 2>&1 &
for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:8189/api/v2/health >/dev/null && break
  sleep 1
done
curl -sf http://127.0.0.1:8189/api/v2/health >/dev/null || { echo "Proxy failed; see $LOGS/proxy.log"; exit 1; }
echo "Ready. Comfy API v2 on port 8189 (token required). Logs: $LOGS"
