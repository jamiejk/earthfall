"""Settings, all from environment variables (see app/README.md)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]


def _repo_default() -> Path:
    """TOUCHDOWN_REPO, else this checkout (app/ sits in the repo), else the current directory (installed wheel)."""
    if os.environ.get("TOUCHDOWN_REPO"):
        return Path(os.environ["TOUCHDOWN_REPO"]).resolve()
    return APP_DIR.parent if (APP_DIR.parent / "shots").is_dir() else Path.cwd().resolve()


def _web_default() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "web", APP_DIR / "web"):  # installed wheel / source checkout
        if (candidate / "index.html").is_file():
            return candidate
    return APP_DIR / "web"


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("TOUCHDOWN_DATA", "touchdown-data")).resolve())
    repo: Path = field(default_factory=_repo_default)
    web_dir: Path = field(default_factory=_web_default)
    passcode: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_PASSCODE", ""))
    worker_token: str = field(default_factory=lambda: os.environ.get("TOUCHDOWN_WORKER_TOKEN", ""))
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("TOUCHDOWN_MAX_UPLOAD_MB", "12")))
    max_queue: int = field(default_factory=lambda: int(os.environ.get("TOUCHDOWN_MAX_QUEUE", "10")))
    # A job is swept as failed when its heartbeat is older than this. It must be longer than the worker's longest
    # phase (manifest generation is allowed 900 s, the terrain keys 600 s), or the server kills a healthy build.
    worker_timeout_s: float = field(default_factory=lambda: float(os.environ.get("TOUCHDOWN_WORKER_TIMEOUT", "1800")))
    # extra mp4s to copy into the gallery on start (os.pathsep-separated dirs); the example films come from
    # app/examples through scripts/install_earthfall_examples.py
    gallery_seed: list[Path] = field(default_factory=lambda: [
        Path(p) for p in os.environ.get("TOUCHDOWN_GALLERY_SEED", "").split(os.pathsep) if p])

    def __post_init__(self):
        self.cookie_secret = secrets.token_hex(16)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "touchdown.sqlite3"

    @property
    def uploads(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def gallery_dir(self) -> Path:
        return self.data_dir / "gallery"
