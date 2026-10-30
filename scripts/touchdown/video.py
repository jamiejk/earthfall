"""Frame I/O and measurements for the Touchdown build. Clips are streamed through ffmpeg pipes one frame at a time,
so no stage holds a whole 1080p clip in RAM (the desktop has run out of system RAM twice). Measurements run on a
small grey copy of the clip, which does fit (240x136 x 300 frames is about 10 MB)."""
from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Self

import numpy as np

W, H, FPS = 1920, 1088, 24


def count(path: Path | str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return int(out.strip().split(",")[0])


def frames(path: Path | str, size: tuple[int, int] = (W, H), gray: bool = False) -> Iterator[np.ndarray]:
    """Stream a clip's frames as uint8 arrays (RGB, or grey with ``gray``), scaled to ``size``."""
    w, h = size
    fmt, depth = ("gray", 1) if gray else ("rgb24", 3)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo",
                             "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE)
    n = w * h * depth
    try:
        while chunk := proc.stdout.read(n):
            if len(chunk) < n:
                break
            yield np.frombuffer(chunk, np.uint8).reshape((h, w) if gray else (h, w, 3))
    finally:
        if proc.poll() is None:  # the caller stopped early: stop ffmpeg quietly instead of breaking its pipe
            proc.kill()
        proc.stdout.close()
        proc.wait()


def frame(path: Path | str, index: int) -> np.ndarray:
    """One full-size RGB frame (negative index counts from the end)."""
    if index < 0:
        index += count(path)
    for k, im in enumerate(frames(path)):
        if k == index:
            return im.copy()
    raise IndexError(f"{path} has no frame {index}")


def small(path: Path | str, size: tuple[int, int] = (240, 136)) -> np.ndarray:
    """The whole clip, small and grey, as float32 (n, h, w)."""
    return np.stack([f.astype(np.float32) for f in frames(path, size, gray=True)])


def save_png(image: np.ndarray, path: Path) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


class Writer:
    """Write RGB frames to an H.264 file as they are produced."""

    def __init__(self, path: Path | str, crf: int = 14):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                      f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                                      "-pix_fmt", "yuv420p", str(self.path)], stdin=subprocess.PIPE)

    def write(self, image: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
        self.count += 1

    def extend(self, images: Iterable[np.ndarray]) -> None:
        for im in images:
            self.write(im)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


# --- measurements ------------------------------------------------------------------------------------------

def diff(small_frames: np.ndarray) -> np.ndarray:
    """Mean absolute grey difference per step (step i: frame i -> i+1). Texture and contrast raise it as well as
    speed, so it is a diagnostic, not a camera speed (Astra's review)."""
    return np.abs(np.diff(small_frames, axis=0)).mean(axis=(1, 2))


def flow(small_frames: np.ndarray, lo: int = 0, hi: int | None = None, win: int = 15) -> np.ndarray:
    """Mean dense optical-flow magnitude (px per step at the small size) for steps lo..hi-1; NaN elsewhere. The
    desktop's join scripts (build2, build_cloud) measured at 480x272 with ``win`` 21; their thresholds assume that."""
    import cv2
    n = len(small_frames)
    hi = n - 1 if hi is None else min(hi, n - 1)
    out = np.full(n - 1, np.nan, np.float32)
    g = np.clip(small_frames, 0, 255).astype(np.uint8)
    for i in range(max(lo, 0), hi):
        out[i] = np.linalg.norm(cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, 0.5, 3, win, 3, 5, 1.1, 0),
                                axis=2).mean()
    return out


def motion(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Camera-like motion between two small grey frames as a similarity fit: (zoom % per step, roll degrees per
    step, shift px per step). Separates a zoom from a slide, which mean flow or pixel difference cannot."""
    import cv2
    g0, g1 = (np.clip(x, 0, 255).astype(np.uint8) for x in (a, b))
    pts = cv2.goodFeaturesToTrack(g0, 300, 0.01, 6)
    if pts is None or len(pts) < 8:
        return float("nan"), float("nan"), float("nan")
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None)
    good = ok.ravel() == 1
    if good.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    m, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC)
    if m is None:
        return float("nan"), float("nan"), float("nan")
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    return (scale - 1) * 100, float(np.degrees(np.arctan2(m[1, 0], m[0, 0]))), float(np.hypot(m[0, 2], m[1, 2]))


# --- colour ------------------------------------------------------------------------------------------------

def lab(image: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)


def rgb(lab_image: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.clip(lab_image, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def lab_mean(image: np.ndarray) -> np.ndarray:
    return lab(image).reshape(-1, 3).mean(axis=0)


def smoothstep(t: float | np.ndarray) -> float | np.ndarray:
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def fit_frame(image: np.ndarray, width: int = W, height: int = H) -> np.ndarray:
    """Crop an image to the frame's shape and resize it to the frame."""
    import cv2
    h, w = image.shape[:2]
    if w * height > h * width:  # too wide: crop the sides
        new_w = int(round(h * width / height))
        left = (w - new_w) // 2
        image = image[:, left:left + new_w]
    else:  # too tall: crop the top and bottom
        new_h = int(round(w * height / width))
        top = (h - new_h) // 2
        image = image[top:top + new_h]
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def match_lab(patch: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Match a patch's colour to the frame it is going into (mean and spread in Lab).

    A face taken from another picture carries that picture's light; without this the seam across the cheeks is
    what makes a pasted face look pasted.
    """
    import cv2
    a = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    ma, sa = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)
    mb, sb = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0)
    matched = (a - ma) / np.maximum(sa, 1e-3) * sb + mb
    return cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """``a`` at t=0, ``b`` at t=1."""
    t = float(np.clip(t, 0.0, 1.0))
    return a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t


def push_in_zoom(frame_index: int, total: int, prefix: int, final_zoom: float,
                 focus_x: float = 0.5, focus_y: float = 0.46) -> tuple[float, float, float]:
    """The zoom and centre the push-in uses at ``frame_index``: (z, cx, cy).

    The end hold starts from where the video stopped, so the cut from one to the other does not change the framing.
    """
    t = smoothstep((frame_index - (prefix - 1)) / max(1, total - prefix))
    z = 1 + (final_zoom - 1) * t
    cx, cy = W * (0.5 + (focus_x - 0.5) * t), H * (0.5 + (focus_y - 0.5) * t)
    return z, cx, cy


def push_in(image: np.ndarray, frame_index: int, total: int, prefix: int,
            final_zoom: float, focus_x: float = 0.5, focus_y: float = 0.46) -> np.ndarray:
    """Smooth camera push on a stable living-end clip, starting after the video prefix.

    The pinned prefix reproduces the preceding approach. A geometric push avoids asking the model to
    reimagine the landscape and person during the hand-off.
    """
    import cv2
    if final_zoom < 1:
        raise ValueError("end zoom must be >= 1")
    z, cx, cy = push_in_zoom(frame_index, total, prefix, final_zoom, focus_x, focus_y)
    if z == 1:
        return image
    matrix = np.float32([[z, 0, W / 2 - z * cx], [0, z, H / 2 - z * cy]])
    return cv2.warpAffine(image, matrix, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def shift_lab(image: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return rgb(lab(image) + offset)
