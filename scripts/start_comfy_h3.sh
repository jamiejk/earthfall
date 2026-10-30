#!/usr/bin/env bash
set -euo pipefail

comfy_root="${COMFY_ROOT:-/run/media/jamie/Data/Image Creation/Comfy}"
comfy_python="${COMFY_PYTHON:-/home/jamie/comfyui-venv/bin/python}"
cd "$comfy_root"
exec "$comfy_python" main.py --listen 127.0.0.1 --port "${COMFY_PORT:-8188}" "$@"
