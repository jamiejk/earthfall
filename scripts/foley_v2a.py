#!/usr/bin/env python3
"""Add sound effects to a finished cut with the LTX-2.3 Foley V2A LoRA.

The video stays frozen; LTX only generates the audio latents. The LoRA was
trained on ~3.7 s clips and its workflow handles up to ~7 s, so the cut is
split into overlapping shot windows, each with its own sound prompt. The
windows are crossfaded (equal power over the overlap), loudness-normalised
and muxed back onto the untouched full-resolution video.

Needs, on the ComfyUI side (port 8190):
  - ComfyUI-LTXVideo (audio / V2A nodes); VideoHelperSuite is not needed (to_core swaps in core nodes)
  - ltx-2.3-22b-dev-fp8 + gemma_3_12B_it_fp8_scaled (already used by the 2.3 graph)
  - loras/ltx-2.3-22b-lora-foley-v2a-1.0.safetensors and the API-format
    workflow ltx-2.3-foley-v2a.json, both from the gated HF repo
    Lightricks/LTX-2.3-22b-LoRA-Foley-V2A

The official workflow is patched by node type, not node id: VHS_LoadVideo gets
the window file and frame count, RandomNoise the seed, the text node feeding
the guider's positive/negative input gets the prompts, and checkpoint / LoRA
loaders get the names below if they differ. Run with --dry-run first to see
what would be patched.

Example:
  python scripts/foley_v2a.py --video a7wb_refined_final.mp4 \
      --workflow ltx-2.3-foley-v2a.json --seeds 42,7,1 --out-dir foley_out
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

FPS = 24
WIDTH, HEIGHT = 960, 544  # the LoRA's training resolution

NO_SPEECH = " No speech is present. No music is present."
NEGATIVE = ("music, singing, melody, instruments, speech, talking, voice, narration, "
            "low quality, distorted, clipping, muffled, static noise, silence")

# Shot windows for the Touchdown cut (337 frames, 14 s). Frame counts must be
# 1 + 8n. Neighbouring windows overlap by 17 frames (~0.7 s) for the crossfade.
DEFAULT_SHOTS = [
    {
        "name": "orbit_dive",
        "start": 0, "frames": 121,  # f0-120: orbit, cloud whiteout, aerial dive with shake
        "prompt": ("A deep, low rumble of high-altitude wind builds into a roaring rush of air "
                   "as the camera plunges through thick cloud, buffeting turbulence, air tearing "
                   "past at high speed, a rising whoosh."),
    },
    {
        "name": "swoop",
        "start": 104, "frames": 121,  # f104-224: top-down descent, swoop over the hillside
        "prompt": ("A fast whoosh of air as the camera sweeps low over a grassy hillside, wind "
                   "roar easing as it slows, dry grass and shrubs rustling below, a faint distant "
                   "city hum."),
    },
    {
        "name": "hilltop",
        "start": 208, "frames": 129,  # f208-336: push-in to the woman on the windy hilltop
        "prompt": ("Strong gusty wind on an exposed hilltop, a nylon raincoat flapping and "
                   "rustling, hair blowing, dry grass swishing, a seagull calls once overhead, "
                   "faint distant city traffic far below."),
    },
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_frames(video: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True).stdout.strip()
    return int(out)


def cut_window(video: Path, start: int, frames: int, dst: Path) -> None:
    """Silent 960x544 24 fps excerpt of exactly `frames` frames."""
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", f"select='between(n,{start},{start + frames - 1})',setpts=N/{FPS}/TB,"
                f"scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={FPS}",
         "-frames:v", str(frames), "-an", "-c:v", "libx264", "-crf", "14",
         "-pix_fmt", "yuv420p", str(dst)])


# --- ComfyUI API -------------------------------------------------------------

def api(server: str, path: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(server + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def upload(server: str, path: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
    body += path.read_bytes()
    body += (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n"
             f"true\r\n--{boundary}--\r\n").encode()
    res = json.loads(api(server, "/upload/image", body,
                         {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    return res["name"] if not res.get("subfolder") else f"{res['subfolder']}/{res['name']}"


def text_source(graph: dict, link) -> str | None:
    """Follow a conditioning link upstream to the node that holds the prompt text."""
    seen = set()
    stack = [link]
    while stack:
        cur = stack.pop()
        if not (isinstance(cur, list) and len(cur) == 2):
            continue
        nid = str(cur[0])
        if nid in seen or nid not in graph:
            continue
        seen.add(nid)
        inputs = graph[nid].get("inputs", {})
        if isinstance(inputs.get("text"), str):
            return nid
        if "positive" in inputs and "negative" in inputs:
            # Pass-through nodes (LTXVConditioning etc.) output positive on 0, negative on 1.
            stack.append(inputs["negative"] if cur[1] == 1 else inputs["positive"])
            continue
        for key in ("conditioning", "conditioning_1", "cond", "positive", "negative"):
            if key in inputs:
                stack.append(inputs[key])
    return None


def patch(graph: dict, video_name: str, frames: int, seed: int, prompt: str,
          negative: str, names: dict) -> list[str]:
    g = copy.deepcopy(graph)
    log = []
    pos_ids, neg_ids = set(), set()
    for nid, node in g.items():
        ins = node.get("inputs", {})
        ct = node.get("class_type", "")
        if ct.startswith("VHS_LoadVideo"):
            ins["video"] = video_name
            if "frame_load_cap" in ins:
                ins["frame_load_cap"] = frames
            if "skip_first_frames" in ins:
                ins["skip_first_frames"] = 0
            if "force_rate" in ins:
                ins["force_rate"] = FPS
            log.append(f"{nid} {ct}: video={video_name} frames={frames}")
        if ct == "RandomNoise" and "noise_seed" in ins:
            ins["noise_seed"] = seed
            log.append(f"{nid} RandomNoise: seed={seed}")
        for key, val in list(ins.items()):
            if key in ("ckpt_name",) and names.get("checkpoint") and isinstance(val, str):
                ins[key] = names["checkpoint"]
                log.append(f"{nid} {ct}: {key}={ins[key]}")
            if key == "lora_name" and isinstance(val, str) and "foley" in val.lower() and names.get("lora"):
                ins[key] = names["lora"]
                log.append(f"{nid} {ct}: lora_name={ins[key]}")
            if key in ("text_encoder", "clip_name", "clip_name1") and names.get("text_encoder") \
                    and isinstance(val, str) and "gemma" in val.lower():
                ins[key] = names["text_encoder"]
                log.append(f"{nid} {ct}: {key}={ins[key]}")
        if "positive" in ins:
            src = text_source(g, ins["positive"])
            if src:
                pos_ids.add(src)
        if "negative" in ins:
            src = text_source(g, ins["negative"])
            if src:
                neg_ids.add(src)
    for nid in pos_ids - neg_ids:
        g[nid]["inputs"]["text"] = prompt
        log.append(f"{nid} positive text set")
    for nid in neg_ids - pos_ids:
        g[nid]["inputs"]["text"] = negative
        log.append(f"{nid} negative text set")
    if not pos_ids:
        log.append("WARNING: no positive prompt node found; prompt not applied")
    log += to_core(g, frames, names)
    return g, log


def to_core(graph: dict, frames: int, names: dict) -> list[str]:
    """Run the official graph without ComfyUI-VideoHelperSuite or the ArbitraryCode / VideoSave_LTX helpers
    (not installed on the desktop's 8190): VHS_LoadVideo -> core LoadVideo + GetVideoComponents; the two
    ArbitraryCode nodes (frame count rounded down to 1 + 8n, and the batch cut to it) are dropped, since every window
    is already cut to exactly 1 + 8n frames; VideoSave_LTX -> core SaveAudio. The Gemma loader is swapped for
    LTXAVTextEncoderLoader when a ComfyUI-format text encoder file is given (--text-encoder)."""
    log = []
    for nid, node in list(graph.items()):
        ct = node.get("class_type", "")
        if ct.startswith("VHS_LoadVideo"):
            video = node["inputs"]["video"]
            graph[nid] = {"class_type": "LoadVideo", "inputs": {"file": video}}
            graph[nid + "_c"] = {"class_type": "GetVideoComponents", "inputs": {"video": [nid, 0]}}
            for oid, other in graph.items():                 # IMAGE consumers of the loader
                if oid == nid + "_c":
                    continue
                for k, v in other.get("inputs", {}).items():
                    if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid and v[1] == 0:
                        other["inputs"][k] = [nid + "_c", 0]
            log.append(f"{nid} VHS_LoadVideo -> LoadVideo + GetVideoComponents")
    for nid, node in list(graph.items()):
        if node.get("class_type") != "ArbitraryCode":
            continue
        a = node["inputs"].get("a")
        for other in graph.values():
            for k, v in other.get("inputs", {}).items():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid:
                    # the frame-count node feeds an INT; the batch-cut node passes its images through
                    other["inputs"][k] = frames if "b" not in node["inputs"] else a
        del graph[nid]
        log.append(f"{nid} ArbitraryCode removed")
    for nid, node in list(graph.items()):
        if node.get("class_type") == "VideoSave_LTX":
            graph[nid] = {"class_type": "SaveAudio", "inputs": {"audio": node["inputs"]["audio"],
                                                                 "filename_prefix": "foley/foley_v2a"}}
            log.append(f"{nid} VideoSave_LTX -> SaveAudio")
        if node.get("class_type") == "LTXVGemmaCLIPModelLoader" and names.get("text_encoder"):
            graph[nid] = {"class_type": "LTXAVTextEncoderLoader", "inputs": {
                "text_encoder": names["text_encoder"], "ckpt_name": names.get("checkpoint") or node["inputs"]["ltxv_path"],
                "device": "default"}}
            log.append(f"{nid} LTXVGemmaCLIPModelLoader -> LTXAVTextEncoderLoader({names['text_encoder']})")
    return log


def queue_and_wait(server: str, graph: dict, timeout: float) -> dict:
    client = uuid.uuid4().hex
    body = json.dumps({"prompt": graph, "client_id": client}).encode()
    pid = json.loads(api(server, "/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(api(server, f"/history/{pid}"))
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status.get("messages", []))[:2000])
            if status.get("completed", True):
                return entry.get("outputs", {})
        time.sleep(3)
    raise TimeoutError(f"prompt {pid} not done after {timeout:.0f} s")


def fetch_audio(server: str, outputs: dict, dst: Path) -> None:
    """Pull the generated audio (SaveAudio or a VHS video with audio) to a 48 kHz wav."""
    files = []
    for out in outputs.values():
        for key in ("audio", "gifs", "videos"):
            files += out.get(key, [])
    if not files:
        raise RuntimeError(f"no audio/video outputs in {list(outputs)}")
    f = files[0]
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                "type": f.get("type", "output")})
    raw = dst.with_suffix(Path(f["filename"]).suffix)
    raw.write_bytes(api(server, f"/view?{q}"))
    run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-vn", "-ac", "2", "-ar", "48000", str(dst)])


# --- stitching -----------------------------------------------------------------

def fade_filter(fade: tuple[int, int, float] | None) -> str:
    """Volume ramp from 0 dB at frame f0 to `db` at frame f1, held to the end."""
    if not fade:
        return ""
    f0, f1, db = fade
    t0, t1 = f0 / FPS, f1 / FPS
    expr = (f"if(lt(t,{t0:.4f}),1,if(lt(t,{t1:.4f}),"
            f"pow(10,{db}*(t-{t0:.4f})/{t1 - t0:.4f}/20),pow(10,{db}/20)))")
    return f",volume='{expr}':eval=frame"


def granular_bed(src: Path, a: float, b: float, dur: float, dst: Path, grain: float = 0.45,
                 xfade: float = 0.15, seed: int = 0) -> None:
    """Fill `dur` seconds with random grains taken from [a, b) s of `src`,
    joined by equal-power crossfades. Holds a wind bed the model let die
    without the periodic repeat of a loop or the echo of a mirrored one."""
    import numpy as np  # only needed for this step

    sr = 48000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sr),
                          "-f", "f64le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(raw, np.float64)[int(a * sr):int(b * sr)]
    g, f = int(grain * sr), int(xfade * sr)
    rng = np.random.default_rng(seed)
    fade_in = np.sin(np.linspace(0, np.pi / 2, f)) if f else np.ones(0)
    out = np.zeros(int(dur * sr) + g)
    pos, last = 0, -1
    while pos < len(out) - g:
        st = int(rng.integers(0, max(1, len(x) - g)))
        if abs(st - last) < g // 2:  # avoid near-repeats of the same grain
            st = (st + g) % max(1, len(x) - g)
        last = st
        chunk = x[st:st + g].copy()
        if pos:
            chunk[:f] *= fade_in
        chunk[-f:] *= fade_in[::-1]
        out[pos:pos + g] += chunk
        pos += g - f
    out = out[:int(dur * sr)]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f64le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-ac", "2", str(dst)], input=out.tobytes(), check=True)


def integrated_lufs(wav: Path) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af", "ebur128=framelog=quiet",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    vals = [ln.split("I:")[1].split("LUFS")[0] for ln in err.splitlines() if " I:" in ln]
    return float(vals[-1]) if vals else -70.0


def stitch(wavs: list[Path], shots: list[dict], total_frames: int, dst: Path,
           fade: tuple[int, int, float] | None = None, target_lufs: float = -16.0) -> None:
    """Trim each window to its length (or fill it with a granular bed when the
    shot has "loop": [window, a_s, b_s]), apply per-shot filters, crossfade over
    the overlaps, set one static gain so the whole mix sits at `target_lufs`
    (static, so per-shot gains survive), then apply the optional fade."""
    by_name = {s["name"]: i for i, s in enumerate(shots)}
    inputs, filters = [], []
    for i, (w, s) in enumerate(zip(wavs, shots)):
        dur = s["frames"] / FPS
        if s.get("loop"):
            name, a, b = s["loop"]
            bed = dst.with_name(f"bed_{s['name']}_{dst.stem}.wav")
            granular_bed(wavs[by_name[name]], float(a), float(b), dur, bed, seed=i)
            w = bed
        inputs += ["-i", str(w)]
        extra = ""
        if s.get("highpass"):
            extra += f",highpass=f={s['highpass']}"
        if s.get("lowpass"):
            extra += f",lowpass=f={s['lowpass']}"
        if s.get("gain_db"):
            extra += f",volume={s['gain_db']}dB"
        filters.append(f"[{i}:a]aresample=48000,atrim=0:{dur:.4f}{extra},"
                       f"apad=whole_dur={dur:.4f},asetpts=N/SR/TB[w{i}]")
    cur = "w0"
    for i in range(1, len(shots)):
        prev_end = shots[i - 1]["start"] + shots[i - 1]["frames"]
        overlap = max(prev_end - shots[i]["start"], 1) / FPS
        nxt = f"x{i}"
        filters.append(f"[{cur}][w{i}]acrossfade=d={overlap:.4f}:c1=qsin:c2=qsin[{nxt}]")
        cur = nxt
    total = total_frames / FPS
    filters.append(f"[{cur}]atrim=0:{total:.4f},apad=whole_dur={total:.4f}[out]")
    pre = dst.with_name(dst.stem + "_pre.wav")
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters),
         "-map", "[out]", "-ac", "2", str(pre)])
    gain = target_lufs - integrated_lufs(pre)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(pre), "-af",
         f"volume={gain:.2f}dB,alimiter=limit=0.84:level=false{fade_filter(fade)}", "-ac", "2", str(dst)])


def decomb(src: Path, dst: Path, block_s: float = 0.5, passes: int = 8) -> None:
    """Remove the slapback/comb echo the Foley LoRA bakes into its audio.

    The raw windows carry a repeating echo at ~33 ms (and 66/100 ms), heard as
    reverb or slapback on the wind. Per half-second block, find the strongest
    autocorrelation peak between 26 and 110 ms and cancel it with the inverse
    of a feedback comb (s = x - g*x[n-d]); repeat a few passes. Levels are kept.
    """
    import numpy as np  # only needed for this step

    sr = 48000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sr),
                          "-f", "f64le", "-"], check=True, capture_output=True).stdout
    y = np.frombuffer(raw, np.float64).copy()
    block = int(block_s * sr)
    hop, win = block // 2, np.hanning(block)
    lo, hi = int(0.026 * sr), int(0.110 * sr)
    for _ in range(passes):
        out, wsum = np.zeros_like(y), np.zeros_like(y)
        for st in range(0, len(y) - block + 1, hop):
            seg = y[st:st + block]
            c = seg - seg.mean()
            spec = np.fft.rfft(c, 2 * block)
            ac = np.fft.irfft(spec * np.conj(spec))[:block]
            ac /= ac[0] + 1e-12
            d = lo + int(ac[lo:hi].argmax())
            g = ac[d] if ac[d] >= 0.03 else 0.0
            prev = y[max(0, st - d):st + block - d]
            prev = np.concatenate([np.zeros(block - len(prev)), prev])
            out[st:st + block] += (seg - g * prev) * win
            wsum[st:st + block] += win
        y = np.where(wsum > 1e-6, out / np.maximum(wsum, 1e-6), y)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f64le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-ac", "2", str(dst)], input=y.tobytes(), check=True)


def mux(video: Path, wav: Path, dst: Path) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(wav),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
         "-shortest", str(dst)])


def _decombed(wav: Path) -> Path:
    dst = wav.with_name(wav.stem + "_dc.wav")
    decomb(wav, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--workflow", type=Path,
                    help="API-format ltx-2.3-foley-v2a.json from the LoRA repo")
    ap.add_argument("--shots", type=Path, help="JSON list of {name,start,frames,prompt[,negative,gain_db,lowpass,highpass,loop]}; "
                         "loop = [window_name, a_s, b_s] (mix-time only); default: Touchdown cut")
    ap.add_argument("--seeds", default="42", help="comma-separated; one full mix per seed")
    ap.add_argument("--server", default="http://127.0.0.1:8190")
    ap.add_argument("--out-dir", type=Path, default=Path("foley_out"))
    ap.add_argument("--checkpoint", default="", help="override ckpt_name (e.g. ltx-2.3-22b-dev-fp8.safetensors)")
    ap.add_argument("--lora", default="", help="override the Foley LoRA file name")
    ap.add_argument("--text-encoder", default="", help="override the Gemma text encoder file name")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--fade-out", default="",
                    help="F0:F1:DB, ramp the mix from 0 dB at frame F0 to DB at frame F1 and hold "
                         "(e.g. 160:215:-26 to let the wind die away on the push-in)")
    ap.add_argument("--remix", action="store_true",
                    help="skip ComfyUI; re-stitch existing {name}_s{seed}.wav files in --out-dir")
    ap.add_argument("--decomb", action="store_true",
                    help="remove the model's ~33 ms slapback/comb echo from each window before mixing (needs numpy)")
    ap.add_argument("--dry-run", action="store_true", help="print the patches for the first window and exit")
    args = ap.parse_args()

    shots = json.loads(args.shots.read_text()) if args.shots else DEFAULT_SHOTS
    for s in shots:
        if (s["frames"] - 1) % 8 and not s.get("loop"):
            sys.exit(f"shot {s['name']}: frames must be 1 + 8n, got {s['frames']}")
    fade = None
    if args.fade_out:
        f0, f1, db = args.fade_out.split(":")
        fade = (int(f0), int(f1), float(db))
        if fade[1] <= fade[0]:
            sys.exit("--fade-out: F1 must be after F0")
    if not args.remix and not args.workflow:
        sys.exit("--workflow is required unless --remix")
    graph = json.loads(args.workflow.read_text()) if args.workflow else {}
    if "nodes" in graph:
        sys.exit("workflow is in UI format; export it with 'Save (API)' first")
    names = {"checkpoint": args.checkpoint, "lora": args.lora, "text_encoder": args.text_encoder}
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    total = probe_frames(args.video)
    end = max(s["start"] + s["frames"] for s in shots)
    if end > total:
        sys.exit(f"shots run to frame {end} but the video has {total}")

    if args.dry_run:
        _, log = patch(graph, "window.mp4", shots[0]["frames"], seeds[0],
                       shots[0]["prompt"] + NO_SPEECH, NEGATIVE, names)
        print("\n".join(log))
        return 0

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = ("_dc" if args.decomb else "") + ("_fade" if fade else "")

    if args.remix:
        for seed in seeds:
            wavs = [out / f"{s['name']}_s{seed}.wav" for s in shots]
            missing = [str(w) for w in wavs if not w.exists()]
            if missing:
                sys.exit(f"missing: {', '.join(missing)}")
            if args.decomb:
                wavs = [_decombed(w) for w in wavs]
            mix = out / f"mix_s{seed}{tag}.wav"
            stitch(wavs, shots, total, mix, fade)
            final = out / f"{args.video.stem}_foley_s{seed}{tag}.mp4"
            mux(args.video, mix, final)
            print(f"[seed {seed}] {final}", flush=True)
        return 0

    win_names = []
    for s in shots:
        clip = out / f"win_{s['name']}.mp4"
        cut_window(args.video, s["start"], s["frames"], clip)
        win_names.append(upload(args.server, clip))

    for seed in seeds:
        wavs = []
        for s, name in zip(shots, win_names):
            g, log = patch(graph, name, s["frames"], seed, s["prompt"] + NO_SPEECH, s.get("negative", NEGATIVE), names)
            t0 = time.time()
            print(f"[seed {seed}] {s['name']}: f{s['start']}-{s['start'] + s['frames'] - 1}", flush=True)
            outputs = queue_and_wait(args.server, g, args.timeout)
            wav = out / f"{s['name']}_s{seed}.wav"
            fetch_audio(args.server, outputs, wav)
            print(f"  done in {time.time() - t0:.0f} s -> {wav}", flush=True)
            wavs.append(wav)
        if args.decomb:
            wavs = [_decombed(w) for w in wavs]
        mix = out / f"mix_s{seed}{tag}.wav"
        stitch(wavs, shots, total, mix, fade)
        final = out / f"{args.video.stem}_foley_s{seed}{tag}.mp4"
        mux(args.video, mix, final)
        print(f"[seed {seed}] {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
