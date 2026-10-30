#!/usr/bin/env bash
# Isolated local runtime for photographic checks. Downloads are explicit.
set -euo pipefail
task_vision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_HOST="127.0.0.1:${TOUCHDOWN_VISION_PORT:-11439}"
export OLLAMA_MODELS="${TOUCHDOWN_VISION_MODELS:-$task_vision_root/models/vision-checker/ollama}"
export OLLAMA_NO_CLOUD=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_LOAD_TIMEOUT=10m
exec ollama serve
