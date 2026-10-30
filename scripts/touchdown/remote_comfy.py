"""Optional remote ComfyUI for Create's generation stages (FLUX 3 stills and LTX video).

Off unless TOUCHDOWN_REMOTE_COMFY=1. Then:
- TOUCHDOWN_FLUX3_FREE_URL is the remote ComfyUI's plain HTTP API (stills, /free), and TOUCHDOWN_COMFY_URL its
  comfy-api-proxy (the Comfy SDK the video stages use). Both may be the same host behind one reverse proxy.
- TOUCHDOWN_REMOTE_COMFY_TOKEN, when set, is sent as `Authorization: Bearer <token>` to those two URLs only: the
  stills' urllib calls through a process-wide opener, the SDK through GEOBLOCK_PROXY_TOKEN (geoblock_core.run._client).
- The build no longer starts or stops a ComfyUI of its own: studio_build.runtime('comfy', ...) waits for the remote
  to answer and leaves it running. Qwen checks, OpenCV and assembly stay on this machine.

The remote must already hold what a local launch would load: ComfyUI 0.41.3, the pinned TouchdownPromptRelay node
(workflows/touchdown/canonical/photographic-grounded-v1/relay), and the same model files under the same names.
"""
from __future__ import annotations

import os
import time
import urllib.request
from contextlib import contextmanager

ENABLED = os.environ.get("TOUCHDOWN_REMOTE_COMFY") == "1"
TOKEN = os.environ.get("TOUCHDOWN_REMOTE_COMFY_TOKEN", "")
WAIT = float(os.environ.get("TOUCHDOWN_REMOTE_COMFY_WAIT", "180"))


def urls():
    return [u.rstrip("/") for u in (os.environ.get("TOUCHDOWN_FLUX3_FREE_URL", ""), os.environ.get("TOUCHDOWN_COMFY_URL", "")) if u]


class Bearer(urllib.request.BaseHandler):
    """Adds the token to requests for the remote ComfyUI URLs, and to nothing else (Qwen, Overpass, Commons)."""
    handler_order = 400

    def __init__(self, token, prefixes):
        self.token, self.prefixes = token, tuple(prefixes)

    def _add(self, request):
        if self.token and request.full_url.startswith(self.prefixes):
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return request

    http_request = https_request = _add


def install(*modules):
    """Point this process at the remote ComfyUI; `modules` are studio_build modules (live and frozen copies)."""
    if not ENABLED:
        return False
    if TOKEN:
        os.environ["GEOBLOCK_PROXY_TOKEN"] = TOKEN
        urllib.request.install_opener(urllib.request.build_opener(Bearer(TOKEN, urls())))
    # canonical_stage and enable_prompt_relay add flags to the launch command; nothing is launched in this mode.
    os.environ.setdefault("TOUCHDOWN_COMFY_SERVER_CMD", "remote")
    os.environ.setdefault("TOUCHDOWN_COMFY_SERVER_CWD", ".")
    for module in modules:
        module._runtime = _wrap(module, module._runtime)
    return True


def _wrap(module, local):
    @contextmanager
    def runtime(kind, root):
        if kind == "vision":
            with local(kind, root) as metadata:
                yield metadata
            return
        deadline = time.monotonic() + WAIT
        while not module.reachable(module.COMFY_URL, "/system_stats"):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"The remote ComfyUI at {module.COMFY_URL} did not answer")
            time.sleep(2)
        yield None
    return runtime
