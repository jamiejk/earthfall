"""Final-upscaler shoot-out on a local ComfyUI: run every upscaler on the same clips, then measure and compare.

Each job is the graph `geoblock-run --upscale-video` builds (geoblock_core.upscale), on a stretch of a finished
1920x1088 LTX render. The script waits for an empty queue before each job (the GPU is shared), samples GPU memory
and free RAM while it runs, downloads the result, and then scores every output against its source:

- fidelity: PSNR of the output shrunk back to the source size, and the mean colour shift (0-255);
- detail: high-frequency energy relative to a plain bicubic upscale (1.0 = no added detail);
- flicker: frame-to-frame change of the high-frequency band, flow-compensated with OpenCV when it is installed,
  relative to bicubic (1.0 = as steady as the source; above 1 = added shimmer).

It writes crop sheets (still detail), a tiled crop video per clip (play it to judge shimmer), results.json and
results.md into --out. Run it with ComfyUI's own Python (it needs numpy, Pillow and PyAV; OpenCV is optional).

usage: python scripts/upscaler_shootout.py --clip rim=ups_rim.mp4:0 --clip ez=ups_ez.mp4:120 \\
           [--methods seedvr2,seedvr2-7b,flashvsr,esrgan,ltx-detail] [--targets 4k,1080p] [--frames 121] \\
           [--comfy http://127.0.0.1:8190] [--out docs/review/upscaler] [--analyse-only]
Clip files are names in ComfyUI's input folder; the number after the colon is the first frame.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoblock_core.upscale import FINAL_VIDEO, UPSCALERS, check, upscale_file_graph  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------------------------- running jobs

def _http(comfy: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(comfy + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body[:1] in (b"{", b"[") else body


def _wait_for_idle_gpu(comfy: str) -> None:
    announced = False
    while True:
        queue = _http(comfy, "/queue")
        if not queue["queue_running"] and not queue["queue_pending"]:
            return
        if not announced:
            print("  waiting for the ComfyUI queue to empty (shared GPU)", flush=True)
            announced = True
        time.sleep(30)


class _Sampler(threading.Thread):
    """Peak GPU memory (nvidia-smi, whole card) and lowest free RAM while a job runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak_vram_mb = 0
        self.min_ram_free_mb = _ram_free_mb()

    def run(self):
        while not self.stop.wait(1.0):
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=10).stdout.split()[0]
                self.peak_vram_mb = max(self.peak_vram_mb, int(used))
            except Exception:
                pass
            self.min_ram_free_mb = min(self.min_ram_free_mb, _ram_free_mb())


def _ram_free_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def run_job(comfy: str, graph: dict, destination: Path) -> dict:
    _wait_for_idle_gpu(comfy)
    sampler = _Sampler()
    sampler.start()
    started = time.time()
    prompt_id = _http(comfy, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        time.sleep(5)
        history = _http(comfy, f"/history/{prompt_id}").get(prompt_id)
        if history and history.get("status", {}).get("completed") is not None:
            if history["status"].get("status_str") in ("success", "error") or history.get("outputs"):
                break
    sampler.stop.set()
    result = {"wall_s": round(time.time() - started, 1), "peak_vram_gb": round(sampler.peak_vram_mb / 1024, 1),
              "ram_free_low_gb": round(sampler.min_ram_free_mb / 1024, 1), "status": history["status"]["status_str"]}
    if result["status"] != "success":
        errors = [m[1] for m in history["status"].get("messages", []) if m[0] == "execution_error"]
        result["error"] = (errors[0].get("exception_message", "") if errors else "unknown")[:500]
        return result
    files = history["outputs"].get(FINAL_VIDEO, {}).get("images", [])
    if not files:
        result.update(status="error", error="no output video")
        return result
    item = files[0]
    query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_http(comfy, f"/view?{query}"))
    result["file"] = str(destination)
    return result


# ------------------------------------------------------------------------------------------------- analysis

def read_frames(path, start: int = 0, count: int = 0):
    """Yield RGB uint8 frames start..start+count-1 (all if count is 0)."""
    import av
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index < start:
                continue
            if count and index >= start + count:
                return
            yield frame.to_ndarray(format="rgb24")


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _resize(rgb: np.ndarray, width: int, height: int, method) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((width, height), method))


def _highpass(gray: np.ndarray, sigma: float) -> np.ndarray:
    picture = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    return gray - np.asarray(picture.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


class _Steadiness:
    """Frame-to-frame change of the high-pass band, as a fraction of its strength."""

    def __init__(self):
        self.previous = None
        self.change = self.energy = 0.0
        self.warped_change = 0.0

    def add(self, highpass: np.ndarray, flow: np.ndarray | None):
        if self.previous is not None:
            self.change += float(np.abs(highpass - self.previous).mean())
            if flow is not None:  # pull frame t+1 back onto frame t along the source motion
                self.warped_change += float(np.abs(self.previous - _warp(highpass, flow)).mean())
        self.energy += float(np.abs(highpass).mean())
        self.previous = highpass

    def index(self, frames: int) -> tuple[float, float | None]:
        mean_energy = self.energy / frames
        plain = self.change / (frames - 1) / mean_energy
        warped = self.warped_change / (frames - 1) / mean_energy if self.warped_change else None
        return plain, warped


def score(source_path, start: int, frames: int, output_path) -> dict:
    """Fidelity, detail and flicker of one output against its source stretch (and bicubic as the baseline)."""
    out_steady, bicubic_steady = _Steadiness(), _Steadiness()
    psnr, shift, detail_out, detail_bicubic = [], [], [], []
    previous_gray, count = None, 0
    for src, out in zip(read_frames(source_path, start, frames), read_frames(output_path)):
        h, w = src.shape[:2]
        H, W = out.shape[:2]
        down = _resize(out, w, h, Image.BOX)
        mse = float(((_luma(down) - _luma(src)) ** 2).mean())
        psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
        shift.append(float(np.abs(down.reshape(-1, 3).mean(0) - src.reshape(-1, 3).mean(0)).mean()))
        bicubic = _resize(src, W, H, Image.BICUBIC)
        sigma = 1.5 * W / 1920
        hf_out, hf_bicubic = _highpass(_luma(out), sigma), _highpass(_luma(bicubic), sigma)
        detail_out.append(float(np.abs(hf_out).mean()))
        detail_bicubic.append(float(np.abs(hf_bicubic).mean()))
        flow = None
        gray = _luma(src)
        if cv2 is not None and previous_gray is not None:
            # flow from frame t to t+1 on the source, scaled to the output size
            flow = cv2.calcOpticalFlowFarneback(previous_gray.astype(np.uint8), gray.astype(np.uint8), None,
                                                0.5, 4, 21, 3, 5, 1.1, 0)
            flow = cv2.resize(flow, (W, H)) * (W / w)
        previous_gray = gray
        out_steady.add(hf_out, flow)
        bicubic_steady.add(hf_bicubic, flow)
        count += 1
    if count < 2:
        raise ValueError(f"{output_path}: fewer than two frames")
    out_plain, out_warped = out_steady.index(count)
    ref_plain, ref_warped = bicubic_steady.index(count)
    result = {"frames": count, "psnr_db": round(float(np.mean(psnr)), 2),
              "colour_shift": round(float(np.mean(shift)), 2),
              "detail_gain": round(float(np.mean(detail_out) / np.mean(detail_bicubic)), 2),
              "flicker_ratio": round(out_plain / ref_plain, 3)}
    if out_warped is not None:
        result["warp_flicker_ratio"] = round(out_warped / ref_warped, 3)
    return result


# ------------------------------------------------------------------------------------------- review media

CROPS = {"near": (0.22, 0.78), "middle": (0.5, 0.55), "far": (0.6, 0.4)}  # crop centres, fractions of the frame


def _crop(rgb: np.ndarray, centre, size) -> np.ndarray:
    h, w = rgb.shape[:2]
    cw, ch = size
    left = int(np.clip(centre[0] * w - cw / 2, 0, w - cw))
    top = int(np.clip(centre[1] * h - ch / 2, 0, h - ch))
    return rgb[top:top + ch, left:left + cw]


def _frame(path, index: int) -> np.ndarray:
    return next(read_frames(path, index, 1))


def crop_sheet(source, start: int, outputs: dict, picks, path: Path, crop_px=(288, 162), cell=(576, 324)):
    """Rows: picked frames x crop regions. Columns: bicubic source, then each upscaler."""
    columns = ["source (bicubic)", *outputs]
    rows = [(pick, name) for pick in picks for name in CROPS]
    sheet = Image.new("RGB", (cell[0] * len(columns), (cell[1] + 18) * len(rows) + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(columns):
        draw.text((c * cell[0] + 6, 4), label, fill="black")
    frames = {name: {pick: _frame(file, pick) for pick in picks} for name, file in outputs.items()}
    for r, (pick, region) in enumerate(rows):
        src = _frame(source, start + pick)
        y = 22 + r * (cell[1] + 18)
        draw.text((6, y + cell[1] + 2), f"frame {start + pick}, {region}", fill="black")
        for c, label in enumerate(columns):
            if c == 0:
                crop = _crop(src, CROPS[region], crop_px)
            else:
                out = frames[label][pick]
                k = out.shape[0] / src.shape[0]
                crop = _crop(out, CROPS[region], (round(crop_px[0] * k), round(crop_px[1] * k)))
            sheet.paste(Image.fromarray(crop).resize(cell, Image.BICUBIC if c == 0 else Image.LANCZOS),
                        (c * cell[0], y))
    sheet.save(path, quality=90)


def crop_video(source, start: int, frames: int, outputs: dict, path: Path, region="middle", crop_px=(240, 135)):
    """The same crop from every output side by side (source bicubic first), to judge shimmer in motion."""
    import av
    names = ["source (bicubic)", *outputs]
    cell = (crop_px[0] * 2, crop_px[1] * 2)
    columns = min(3, len(names))
    rows = -(-len(names) // columns)
    readers = [read_frames(source, start, frames)] + [read_frames(f) for f in outputs.values()]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height = cell[0] * columns, cell[1] * rows
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16"}
        for pictures in zip(*readers):
            canvas = Image.new("RGB", (stream.width, stream.height), "black")
            draw = ImageDraw.Draw(canvas)
            src_h = pictures[0].shape[0]
            for i, (name, rgb) in enumerate(zip(names, pictures)):
                k = rgb.shape[0] / src_h
                crop = _crop(rgb, CROPS[region], (round(crop_px[0] * k), round(crop_px[1] * k)))
                x, y = (i % columns) * cell[0], (i // columns) * cell[1]
                canvas.paste(Image.fromarray(crop).resize(cell, Image.BICUBIC if i == 0 else Image.LANCZOS), (x, y))
                draw.text((x + 6, y + 4), name, fill="yellow")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# ---------------------------------------------------------------------------------------------------- verdict

# Pass marks, set before any GPU result (so the numbers can't be tuned to a favourite). Calibration on a 960 px
# preview: bicubic 1.00, unsharp mask 1.10, per-frame noise 2.15, real Real-ESRGAN x2plus (CPU) 1.77 flow-warped flicker.
MAX_FLICKER = 1.15        # flow-warped flicker vs bicubic (plain ratio when OpenCV was missing)
MAX_COLOUR_SHIFT = 3.0    # mean 0-255 per-channel shift after shrinking back (CPU Real-ESRGAN gave 1.8)
MIN_PSNR_DB = 30.0        # output shrunk back vs source; lower = the content changed
MAX_MINUTES_10S = 45.0    # extrapolated wall time for a 241-frame (10 s) clip


def verdict(results: dict, frames: int) -> list[str]:
    """Per target: every upscaler that ran on all clips, checked against the pass marks; the pick is the one with
    the most added detail among those that pass. Worst clip counts. Watch the shimmer videos before trusting it."""
    lines = ["## Provisional pick", "",
             f"Pass marks: flow-warped flicker <= {MAX_FLICKER}, colour shift <= {MAX_COLOUR_SHIFT}, "
             f"PSNR >= {MIN_PSNR_DB} dB, <= {MAX_MINUTES_10S:.0f} min for a 10 s clip. Worst clip counts.", ""]
    clips = sorted({key.split("/")[0] for key in results})
    for target in sorted({key.split("/")[2] for key in results}):
        scored = []
        for method in UPSCALERS:
            entries = [results.get(f"{clip}/{method}/{target}") for clip in clips]
            entries = [e for e in entries if e]
            if not entries:
                continue
            if any("psnr_db" not in e for e in entries) or len(entries) < len(clips):
                lines.append(f"- {target} {method}: incomplete ({next((e.get('error') for e in entries if e.get('error')), 'not run on every clip')})")
                continue
            flicker = max(e.get("warp_flicker_ratio", e["flicker_ratio"]) for e in entries)
            shift = max(e["colour_shift"] for e in entries)
            psnr = min(e["psnr_db"] for e in entries)
            detail = min(e["detail_gain"] for e in entries)
            minutes = max(e.get("wall_s", 0) for e in entries) / 60 * 241 / frames
            failed = [name for name, bad in (("flicker", flicker > MAX_FLICKER), ("colour", shift > MAX_COLOUR_SHIFT),
                                              ("fidelity", psnr < MIN_PSNR_DB), ("time", minutes > MAX_MINUTES_10S)) if bad]
            lines.append(f"- {target} {method}: flicker {flicker:.2f}, colour {shift:.1f}, PSNR {psnr:.1f} dB, "
                         f"detail {detail:.2f}x, ~{minutes:.0f} min per 10 s"
                         + (f" -> fails {', '.join(failed)}" if failed else " -> passes"))
            if not failed:
                scored.append((detail, -minutes, method))
        pick = max(scored)[2] if scored else "none passes"
        lines.append(f"- **{target} pick: {pick}**")
        lines.append("")
    return lines


# ------------------------------------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", required=True, help="name=file_in_comfy_input.mp4:first_frame")
    parser.add_argument("--methods", default="esrgan,seedvr2,seedvr2-7b,flashvsr,ltx-detail")
    parser.add_argument("--targets", default="4k,1080p")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--comfy", default="http://127.0.0.1:8190")
    parser.add_argument("--comfy-input", type=Path, help="ComfyUI input folder (to read sources for scoring); "
                                                         "default: ask the server's /view for them")
    parser.add_argument("--out", type=Path, default=Path("docs/review/upscaler"))
    parser.add_argument("--videos", type=Path, default=Path("outputs/upscaler"), help="Where full outputs go")
    parser.add_argument("--analyse-only", action="store_true", help="Score what is already in --videos")
    args = parser.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    targets = [t for t in args.targets.split(",") if t]
    for method in methods:
        if method not in UPSCALERS:
            parser.error(f"unknown method {method}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.videos.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    clips = {}
    for spec in args.clip:
        name, rest = spec.split("=", 1)
        file, _, first = rest.partition(":")
        clips[name] = (file, int(first or 0))

    for name, (file, start) in clips.items():
        source = args.videos / f"source_{name}.mp4"
        if not source.exists():
            if args.comfy_input:
                source.write_bytes((args.comfy_input / file).read_bytes())
            else:
                query = urllib.parse.urlencode({"filename": file, "type": "input"})
                source.write_bytes(_http(args.comfy, f"/view?{query}"))
        for target in targets:
            for method in methods:
                try:
                    check(method, target)
                except ValueError:
                    continue
                key = f"{name}/{method}/{target}"
                video = args.videos / f"{name}_{method}_{target}.mp4"
                entry = results.get(key, {})
                if not args.analyse_only and not video.exists():
                    print(f"{key}: running", flush=True)
                    graph = upscale_file_graph(file, method=method, target=target, start=start,
                                               frames=args.frames, prefix=f"upscaler_shootout/{name}_{method}_{target}")
                    entry = run_job(args.comfy, graph, video)
                    if entry["status"] != "success" and "memory" in entry.get("error", "").lower() \
                            and method.startswith("seedvr2"):
                        print(f"{key}: out of memory, retrying with smaller batches", flush=True)
                        graph = upscale_file_graph(file, method=method, target=target, start=start,
                                                   frames=args.frames, batch_size=9,
                                                   blocks_to_swap=32 if method == "seedvr2-7b" else 16,
                                                   prefix=f"upscaler_shootout/{name}_{method}_{target}_b9")
                        entry = {**run_job(args.comfy, graph, video), "retry": "batch 9"}
                    print(f"{key}: {entry}", flush=True)
                if video.exists() and "psnr_db" not in entry:
                    entry.update(score(source, start, args.frames, video))
                    print(f"{key}: scored {entry}", flush=True)
                results[key] = entry
                results_path.write_text(json.dumps(results, indent=2) + "\n")

        for target in targets:
            done = {m: args.videos / f"{name}_{m}_{target}.mp4" for m in UPSCALERS  # earlier runs' outputs too
                    if (args.videos / f"{name}_{m}_{target}.mp4").exists()}
            if done:
                picks = [10, args.frames // 2, args.frames - 11]
                crop_sheet(source, start, done, picks, args.out / f"{name}_{target}_crops.jpg")
                crop_video(source, start, args.frames, done, args.out / f"{name}_{target}_shimmer.mp4")

    lines = ["| Clip | Upscaler | Target | Wall time | Peak VRAM (card) | Lowest free RAM | PSNR vs source | "
             "Colour shift | Detail vs bicubic | Flicker vs bicubic | Flow-warped flicker | Notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, entry in results.items():
        clip, method, target = key.split("/")
        wall = f"{entry['wall_s'] / 60:.1f} min" if "wall_s" in entry else ""
        lines.append("| " + " | ".join([
            clip, method, target, wall, f"{entry.get('peak_vram_gb', '')} GB", f"{entry.get('ram_free_low_gb', '')} GB",
            str(entry.get("psnr_db", "")), str(entry.get("colour_shift", "")), str(entry.get("detail_gain", "")),
            str(entry.get("flicker_ratio", "")), str(entry.get("warp_flicker_ratio", "")),
            (entry.get("error", "") or entry.get("retry", "")).replace("|", "/").replace("\n", " ")[:120]]) + " |")
    lines += ["", *verdict(results, args.frames)]
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
