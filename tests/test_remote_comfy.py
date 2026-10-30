"""Optional remote ComfyUI: the token goes only to the remote URLs, and the build no longer launches ComfyUI."""
import importlib
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest


def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

        def log_message(self, *a):
            pass
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


@pytest.fixture
def remote(monkeypatch):
    httpd, seen = server()
    url = f"http://127.0.0.1:{httpd.server_port}"
    other, other_seen = server()
    monkeypatch.setenv("TOUCHDOWN_REMOTE_COMFY", "1")
    monkeypatch.setenv("TOUCHDOWN_REMOTE_COMFY_TOKEN", "secret")
    monkeypatch.setenv("TOUCHDOWN_FLUX3_FREE_URL", url)
    monkeypatch.setenv("TOUCHDOWN_COMFY_URL", url + "/proxy")
    monkeypatch.delenv("GEOBLOCK_PROXY_TOKEN", raising=False)
    monkeypatch.delenv("TOUCHDOWN_COMFY_SERVER_CMD", raising=False)
    from touchdown import remote_comfy
    r = importlib.reload(remote_comfy)
    yield r, url, seen, f"http://127.0.0.1:{other.server_port}", other_seen
    urllib.request.install_opener(None)
    httpd.shutdown(); other.shutdown()


def test_token_only_reaches_the_remote_comfyui(remote):
    r, url, seen, other, other_seen = remote
    launched = []

    @contextmanager
    def local(kind, root):
        launched.append(kind); yield "vision-metadata"
    module = SimpleNamespace(_runtime=local, COMFY_URL=url,
                             reachable=lambda u, e: urllib.request.urlopen(u + e, timeout=2) is not None)
    assert r.install(module)
    import os
    assert os.environ["GEOBLOCK_PROXY_TOKEN"] == "secret" and os.environ["TOUCHDOWN_COMFY_SERVER_CMD"] == "remote"
    with module._runtime("comfy", None) as meta:
        assert meta is None
    with module._runtime("vision", None) as meta:
        assert meta == "vision-metadata"
    urllib.request.urlopen(other + "/api/version").read()
    assert launched == ["vision"]                                  # ComfyUI is never launched locally
    assert seen[0] == ("/system_stats", "Bearer secret")
    assert other_seen == [("/api/version", None)]                   # Qwen and other hosts never see the token


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TOUCHDOWN_REMOTE_COMFY", raising=False)
    from touchdown import remote_comfy
    r = importlib.reload(remote_comfy)
    module = SimpleNamespace(_runtime=object())
    assert not r.install(module) and not isinstance(module._runtime, type(lambda: 0))
