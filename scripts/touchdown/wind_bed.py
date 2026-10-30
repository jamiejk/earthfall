#!/usr/bin/env python3
"""Procedural wind soundtrack for a Touchdown cut, driven by segments.json.

Jamie's brief for the sound (10 Oct): wind only, no reverb or slapback, a
soft opening, more turbulence through the cloud, fading to near silence as
we move in on the figure. The LTX Foley V2A runs kept baking a ~33 ms comb
echo into the wind, so this builds the bed from shaped noise instead: random
phase every STFT frame, so there is nothing that can echo or ring.

How it works:
  - an intensity curve (dB) per video frame comes from the segments: soft
    opener, ramp and heavy buffeting through the cloud, steady roar in the
    dive, easing through the swoop, fade to near silence on the end;
  - the curve is nudged by measured picture motion (mean frame difference),
    so faster stretches roar a little louder and slower ones ease off;
  - each STFT frame is white noise shaped by a wind spectrum whose brightness
    follows the intensity; cloud turbulence adds random gusts and low thumps;
  - two partly correlated channels give some width; one static gain sets the
    loudness, so the shape is kept.

segments.json: {"opener": [a, b], "cloud": [a, b], "dive": [a, b],
"swoop": [a, b], "end": [a, b]} in video frames (end exclusive). Missing
segments are skipped; "end" may overlap "swoop" (the end wins from its start).

Example:
  python scripts/touchdown/wind_bed.py --video touchdown_silent.mp4 \\
      --segments segments.json --out touchdown.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

SR = 48000
NFFT = 2048
HOP = 512

# Per-segment look: start/end level (dB), turbulence depth, brightness (0-1),
# stereo width (0 = enclosed/mono, 1 = wide). Jamie (10 Oct, 19:42Z): no wind
# until we hit the atmosphere, a distinct texture inside the cloud, quieter
# once we're down near the ground.
LOOK = {
    "opener": dict(db=(-90.0, -90.0), turb=0.0, bright=(0.3, 0.3), width=0.8),   # orbit: silence
    "cloud": dict(db=(-7.0, -3.0), turb=0.9, bright=(0.12, 0.2), width=0.3),    # muffled, buffeting, enclosed
    "dive": dict(db=(-3.0, -5.0), turb=0.3, bright=(0.8, 0.7), width=0.85),      # open air: bright roar
    "swoop": dict(db=(-8.0, -18.0), turb=0.2, bright=(0.6, 0.35), width=0.8),    # down at ground level: quieter
}
ENTRY = 8  # frames at the end of the opener where the wind fades in (hitting the atmosphere)


def probe(video: Path) -> tuple[int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", str(video)],
                         check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["nb_read_packets"]), float(num) / float(den)


def motion(video: Path, frames: int) -> np.ndarray:
    """Mean absolute frame difference on a 96x54 grey copy, one value per frame."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=96:54,format=gray",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, 54, 96).astype(np.float32)
    m = np.zeros(frames, np.float32)
    d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))
    m[1:len(d) + 1] = d[:frames - 1]
    m[0] = m[1] if frames > 1 else 0
    return m


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.hanning(n)
    k /= k.sum()
    pad = np.pad(x, (n, n), mode="edge")
    return np.convolve(pad, k, mode="same")[n:-n]


def curves(seg: dict, frames: int, fps: float, mot: np.ndarray | None,
           fade_lead: int, fade_len: int, fade_db: float):
    """Per-frame level (dB), turbulence depth and brightness."""
    db = np.full(frames, -30.0)
    turb = np.zeros(frames)
    bright = np.full(frames, 0.3)
    width = np.full(frames, 0.8)
    for name in ("opener", "cloud", "dive", "swoop"):
        if name not in seg:
            continue
        a, b = (max(0, min(frames, int(v))) for v in seg[name])
        if b <= a:
            continue
        look = LOOK[name]
        t = np.linspace(0, 1, b - a)
        db[a:b] = look["db"][0] + (look["db"][1] - look["db"][0]) * t
        turb[a:b] = look["turb"]
        bright[a:b] = look["bright"][0] + (look["bright"][1] - look["bright"][0]) * t
        width[a:b] = look["width"]
    if "opener" in seg and "cloud" in seg:
        b = int(seg["opener"][1])
        a = max(int(seg["opener"][0]), b - ENTRY)
        if b > a:
            db[a:b] = np.linspace(-50.0, LOOK["cloud"]["db"][0], b - a)
            bright[a:b] = LOOK["cloud"]["bright"][0]
            width[a:b] = LOOK["cloud"]["width"]
    # Anything after the last segment holds the last level until the fade.
    last = max((int(v[1]) for k, v in seg.items() if k in LOOK), default=0)
    if 0 < last < frames:
        db[last:] = db[last - 1]
        turb[last:] = turb[last - 1]
        bright[last:] = bright[last - 1]
        width[last:] = width[last - 1]

    if mot is not None and "dive" in seg:
        a, b = (int(v) for v in seg["dive"])
        ref = np.median(mot[a:b]) if b > a else np.median(mot)
        ratio = np.clip(smooth(mot, int(fps * 0.5)) / max(ref, 1e-3), 0.4, 2.0)
        nudge = np.clip(6.0 * np.log2(ratio), -3.0, 3.0)
        lo = int(seg.get("cloud", seg.get("dive"))[0])  # leave the opener's soft start alone
        db[lo:] += nudge[lo:]

    if "end" in seg:
        e = int(seg["end"][0])
        f0, f1 = max(0, e - fade_lead), min(frames, e - fade_lead + fade_len)
        if f1 > f0:
            ramp = np.linspace(0, 1, f1 - f0)
            start = db[f0]
            db[f0:f1] = start + (fade_db - start) * ramp
            db[f1:] = fade_db
            turb[f0:f1] *= 1 - ramp
            turb[f1:] = 0
            bright[f0:] = np.minimum(bright[f0:], 0.3)

    n = max(1, int(fps * 0.15))
    return smooth(db, n), smooth(turb, n), smooth(bright, n), smooth(width, n)


def wind(frames: int, fps: float, db: np.ndarray, turb: np.ndarray, bright: np.ndarray,
         width: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dur = frames / fps
    n_out = int(dur * SR)
    n_frames = n_out // HOP + 4
    t_stft = np.arange(n_frames) * HOP / SR
    t_vid = np.arange(frames) / fps
    lvl = np.interp(t_stft, t_vid, db)
    tb = np.interp(t_stft, t_vid, turb)
    br = np.interp(t_stft, t_vid, bright)
    wd = np.interp(t_stft, t_vid, width)

    # Gusts: low-passed random walk at a few Hz, deeper where turbulence is high.
    def lowpass_noise(rate_hz: float) -> np.ndarray:
        g = rng.standard_normal(n_frames)
        return smooth(g, max(3, int(SR / HOP / rate_hz)))
    gust = lowpass_noise(2.5)
    gust /= np.abs(gust).max() + 1e-9
    flutter = lowpass_noise(9.0)
    flutter /= np.abs(flutter).max() + 1e-9
    gust_db = (0.25 + tb) * (7.0 * gust + 3.0 * flutter)  # some gusting everywhere, more in turbulence
    # Buffeting thumps: short low-frequency swells, only where turbulence is high.
    thump = np.clip(lowpass_noise(4.0) * 3 - 1.2, 0, None) * tb

    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    hp = (freqs / 90.0) ** 2 / (1 + (freqs / 90.0) ** 2)  # roll off rumble below ~90 Hz
    win = np.hanning(NFFT)
    out = np.zeros((2, n_out + NFFT))
    wsum = np.zeros(n_out + NFFT)
    for i in range(n_frames):
        fc = (250.0 + 3200.0 * br[i] ** 1.5) * 2 ** (0.7 * gust[i])  # brightness follows gusts, like real wind
        env = hp / np.sqrt(1 + (freqs / fc) ** 3)
        env *= 1 + 2.5 * thump[i] * np.exp(-((freqs - 70.0) / 60.0) ** 2)
        gain = 10 ** ((lvl[i] + gust_db[i]) / 20)
        common = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
        st = i * HOP
        if st + NFFT > out.shape[1]:
            break
        for ch in range(2):
            own = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
            spec = (np.sqrt(1 - wd[i] ** 2) * common + wd[i] * own) * env * gain
            frame = np.fft.irfft(spec, NFFT) * win
            out[ch, st:st + NFFT] += frame
        wsum[st:st + NFFT] += win ** 2
    out = out[:, :n_out] / np.sqrt(np.maximum(wsum[:n_out], 1e-6))  # independent frames add in power
    return out


def lufs(wav: Path) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af", "ebur128=framelog=quiet",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    vals = [ln.split("I:")[1].split("LUFS")[0] for ln in err.splitlines() if " I:" in ln]
    return float(vals[-1]) if vals else -70.0


def write_wav(x: np.ndarray, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "pcm_s24le", str(dst)], input=x.T.astype(np.float32).tobytes(), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="muxed mp4 (video stream copied)")
    ap.add_argument("--wav", type=Path, help="also keep the soundtrack here")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lufs", type=float, default=-18.0, help="integrated loudness of the whole track")
    ap.add_argument("--fade", default="14:70:-34",
                    help="LEAD:LEN:DB, start the fade LEAD frames before the end segment, reach DB after LEN frames")
    ap.add_argument("--no-motion", action="store_true", help="don't nudge the level by picture motion")
    args = ap.parse_args()

    frames, fps = probe(args.video)
    seg = json.loads(args.segments.read_text())
    lead, length, fdb = args.fade.split(":")
    mot = None if args.no_motion else motion(args.video, frames)
    db, turb, bright, width = curves(seg, frames, fps, mot, int(lead), int(length), float(fdb))
    x = wind(frames, fps, db, turb, bright, width, args.seed)

    wav = args.wav or args.out.with_suffix(".wind.wav")
    pre = wav.with_name(wav.stem + "_pre.wav")
    peak = np.abs(x).max()
    write_wav(x / max(peak, 1e-9) * 0.5, pre)
    gain = args.lufs - lufs(pre)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(pre), "-af",
                    f"volume={gain:.2f}dB,alimiter=limit=0.89:level=false", "-c:a", "pcm_s24le", str(wav)],
                   check=True)
    pre.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video), "-i", str(wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", str(args.out)], check=True)
    print(json.dumps({"out": str(args.out), "wav": str(wav), "frames": frames, "fps": fps,
                      "gain_db": round(gain, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
