"""The job queue: one SQLite table. One job runs at a time; the rest wait in creation order."""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    status TEXT NOT NULL,             -- queued | running | done | failed | cancelled
    preset TEXT NOT NULL,
    person TEXT NOT NULL,             -- what the user typed (may be empty: the worker fills the default)
    photo TEXT NOT NULL,              -- file name under uploads/
    stages TEXT NOT NULL DEFAULT '{}',   -- stage -> pending | running | done | cached
    previews TEXT NOT NULL DEFAULT '[]', -- preview names, in arrival order
    message TEXT NOT NULL DEFAULT '',    -- short human status from the worker
    error TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',     -- file name under jobs/<id>/
    started REAL,
    finished REAL,
    heartbeat REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    place TEXT NOT NULL DEFAULT '{}',    -- catalogue or custom place for generated manifests (JSON)
    quality TEXT NOT NULL DEFAULT 'full' -- full (LTX 2x pass, 1920x1088) | draft (960x544, faster)
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, created);
"""

ACTIVE = ("queued", "running")


class Jobs:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(jobs)")}
        if "place" not in columns:  # a queue database from before generated places
            self.conn.execute("ALTER TABLE jobs ADD COLUMN place TEXT NOT NULL DEFAULT '{}'")
        if "quality" not in columns:  # a queue database from before the quality choice
            self.conn.execute("ALTER TABLE jobs ADD COLUMN quality TEXT NOT NULL DEFAULT 'full'")
        if "workflow" not in columns:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN workflow TEXT NOT NULL DEFAULT 'terrain'")
        if "setup_id" not in columns:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN setup_id TEXT NOT NULL DEFAULT ''")
        for name in ("parent_id", "source_sha256"):
            if name not in columns:
                self.conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        self.conn.execute("CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id, created)")
        # FastAPI handles worker updates and browser polling on different threads.
        # Serialize reads as well as writes on this shared SQLite connection.
        # Reentrant because claim/cancel/add_preview also call get().
        self.lock = threading.RLock()

    def _row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["stages"] = json.loads(job["stages"])
        job["previews"] = json.loads(job["previews"])
        job["cancel_requested"] = bool(job["cancel_requested"])
        job["place"] = json.loads(job.get("place") or "{}")
        return job

    def create(self, preset: str, person: str, photo: str, place: dict | None = None, quality: str = "full",
               workflow: str = "terrain", setup_id: str = "", parent_id: str = "", source_sha256: str = "") -> dict:
        job_id = secrets.token_urlsafe(9).replace("-", "x").replace("_", "y")
        with self.lock:
            self.conn.execute("INSERT INTO jobs (id, created, status, preset, person, photo, place, quality, workflow, setup_id, parent_id, source_sha256) "
                              "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                              (job_id, time.time(), "queued", preset, person, photo, json.dumps(place or {}), quality, workflow, setup_id, parent_id, source_sha256))
        return self.get(job_id)

    def latest_setup(self, setup_id: str) -> dict | None:
        with self.lock:
            return self._row(self.conn.execute("SELECT * FROM jobs WHERE setup_id=? ORDER BY created DESC LIMIT 1",
                                              (setup_id,)).fetchone())

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            return self._row(self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def active_count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]

    def position(self, job: dict) -> int:
        """1 = next to run. Counts the running job and the queued jobs created before this one."""
        if job["status"] != "queued":
            return 0
        with self.lock:
            ahead = self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running' OR (status = 'queued' AND "
                                      "(created < ? OR (created = ? AND id < ?)))",
                                      (job["created"], job["created"], job["id"])).fetchone()[0]
        return ahead + 1

    def latest_enhancement(self, parent_id: str) -> dict | None:
        with self.lock:
            return self._row(self.conn.execute("SELECT * FROM jobs WHERE parent_id=? AND workflow='refine-details-v1' ORDER BY created DESC LIMIT 1", (parent_id,)).fetchone())

    def claim(self, photographic: bool = False, enhancement: bool = False, canonical: bool = False) -> dict | None:
        """Hand the oldest queued job to the worker, unless one is already running."""
        with self.lock:
            if self.conn.execute("SELECT 1 FROM jobs WHERE status = 'running'").fetchone():
                return None
            workflows = ([] if (enhancement or canonical) and not photographic else ["terrain"]) + (["photographic-v1"] if photographic else []) + (["refine-details-v1"] if enhancement else []) + (["photographic-grounded-v1"] if canonical else [])
            placeholders = ','.join('?' for _ in workflows)
            row = self.conn.execute(f"SELECT id FROM jobs WHERE status = 'queued' AND workflow IN ({placeholders}) ORDER BY created, id LIMIT 1", workflows).fetchone()
            if row is None:
                return None
            now = time.time()
            self.conn.execute("UPDATE jobs SET status='running', started=?, heartbeat=?, message='starting' "
                              "WHERE id=?", (now, now, row["id"]))
        return self.get(row["id"])

    def update(self, job_id: str, **fields) -> None:
        for k in ("stages", "previews"):
            if k in fields:
                fields[k] = json.dumps(fields[k])
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.lock:
            self.conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def add_preview(self, job_id: str, name: str) -> None:
        with self.lock:
            job = self.get(job_id)
            if job and name not in job["previews"]:
                self.conn.execute("UPDATE jobs SET previews = ? WHERE id = ?",
                                  (json.dumps(job["previews"] + [name]), job_id))

    def cancel(self, job_id: str) -> str | None:
        """Queued: cancelled at once. Running: flagged; the worker stops its build and reports back."""
        with self.lock:
            job = self.get(job_id)
            if job is None:
                return None
            if job["status"] == "queued":
                self.conn.execute("UPDATE jobs SET status='cancelled', finished=? WHERE id=?", (time.time(), job_id))
                return "cancelled"
            if job["status"] == "running":
                self.conn.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
                return "cancelling"
            return job["status"]

    def finish(self, job_id: str, status: str, error: str = "", message: str = "") -> None:
        self.update(job_id, status=status, error=error[-8000:], message=message, finished=time.time())

    def sweep(self, timeout_s: float) -> None:
        """A running job whose worker went quiet for ``timeout_s`` is failed, so the queue moves on."""
        with self.lock:
            self.conn.execute("UPDATE jobs SET status='failed', finished=?, error='the worker stopped reporting' "
                              "WHERE status='running' AND heartbeat < ?", (time.time(), time.time() - timeout_s))
