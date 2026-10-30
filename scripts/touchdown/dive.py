"""The computed middle of Touchdown: a dive over real aerial photos, no model.

A port of the desktop's tiltdive2.py as run for the current cut (v5: ``8100 800 52 41 32 46.9 16``, AIM 15):

- Phase 1 (nadir): looking straight down, the footprint shrinks at a constant log rate from ``from_m`` to ``to_m``
  over nested NAIP squares (each a third of the one above, centred on the landing), so the zoom never slows before
  the hand-off. The view turns 0 -> ``spin`` degrees (smoothstep) to face the end shot while still looking down (a
  turn after the tilt reads as banking round a bend), and the crop centre slides along that heading so that the
  pitched phase ends looking ``aim`` metres past the landing. The first ``clear`` frames thin out of cloud mist.
- Phase 2 (pitch): a virtual pinhole camera (60 degree FOV) whose height falls at the log rate easing to half,
  drifting forward, pitching up 0 -> ``pitch`` degrees with a smoothstep in pitch *rate*, so it is still turning on
  its last frame and LTX can carry the motion on. Each pixel samples the square whose resolution matches its ground
  footprint (blended between neighbours in log space), never one that does not contain it.

The squares come from NAIP in the US and from national open orthophotos elsewhere (touchdown.aerial, chosen by the
place's country); a square with nodata (sea, borders) is filled from the next coarser square.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .video import H, W, Writer, save_png, smoothstep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
LEVELS = (24300, 8100, 2700, 900, 300)  # square widths in metres, each a third of the one above
PIXELS = 2048
HFOV = math.radians(60.0)
TAN = math.tan(HFOV / 2)
MIST = (212, 214, 216)
HAZE = (196, 206, 218)   # far-distance haze (oblique dive)
SKY = (120, 160, 210)
HAZE_M = 30000.0         # haze builds over this many metres of sight line
HAZE_MAX = 0.9
HORIZON_BAND = 0.12      # rays within this of grazing (cos of the angle from straight down) fade into haze
NODATA = (28, 52, 78)    # the flat fill fetch() and the older naipzoom squares put in nodata (sea, borders)
FAR_LEVELS = (72900,) + LEVELS
EDGE_FROM, EDGE_TO = 0.34, 0.46  # share of a square's width (from its centre) over which the oblique view eases
                                 # to the next coarser square; was a hard switch at 0.46 (a visible line, 17 Oct)


def _match_levels(imgs: dict[int, Image.Image]) -> dict[int, Image.Image]:
    """Colour-match each square to the finer one inside it, finest first, so the landing keeps its colours.

    NAIP squares of different sizes come from different mosaics (years, seasons): in the FLUX 3 Monument Valley
    cut (17 Oct) the 8.1 km square was redder than the 2.7 km one and the oblique view showed a red band above a
    hard line. Per channel, the coarser square's centre third is scaled to the finer square's mean and spread."""
    out, ws = dict(imgs), sorted(imgs)
    for f, w in zip(ws, ws[1:]):
        fine = np.asarray(out[f].resize((256, 256), Image.BILINEAR), np.float32).reshape(-1, 3)
        n = out[w].size[0]
        a, b = round(n * (0.5 - f / w / 2)), round(n * (0.5 + f / w / 2))
        centre = np.asarray(out[w].crop((a, a, b, b)).resize((256, 256), Image.BILINEAR), np.float32).reshape(-1, 3)
        gain = fine.std(0) / np.maximum(centre.std(0), 1.0)
        arr = (np.asarray(out[w], np.float32) - centre.mean(0)) * np.clip(gain, 0.5, 2.0) + fine.mean(0)
        out[w] = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def fetch(lat: float, lon: float, folder: Path, levels: tuple[int, ...] = LEVELS, source: str = "naip") -> dict[int, Path]:
    """The aerial squares around (lat, lon), cached; nodata filled from the next coarser square. ``source`` is a
    touchdown.aerial source id (the place's own; other countries and Sentinel-2 cloudless fill what it lacks). The
    files keep their naip_ names whatever the source: touchdown.sea keys its masks on them. ``imagery.json`` in
    ``folder`` records which source covered how much of each square, with their credits."""
    import json

    import nadir_dive
    from . import aerial
    folder.mkdir(parents=True, exist_ok=True)
    paths = {w: folder / f"naip_{w:05d}.png" for w in levels}
    record = folder / "imagery.json"
    facts = json.loads(record.read_text()) if record.is_file() else {}
    if facts.get("source", "naip") != source:  # cached squares from another source: fetch again
        for p in paths.values():
            p.unlink(missing_ok=True)
        facts = {}
    facts["source"] = source
    cover = facts.setdefault("cover", {})
    coarser = None
    for w in levels:
        if not paths[w].is_file():
            print(f"Fetching {source} {w} m square", flush=True)
            if source == "naip":  # as before: NAIP alone, nodata from the coarser square
                rgba = nadir_dive._naip(nadir_dive._square(lat, lon, w), PIXELS)
                cover[str(w)] = {"naip": round(float((rgba[..., 3] > 127).mean()), 4)}
            else:
                rgba, cover[str(w)] = aerial.square_rgba(source, nadir_dive._square(lat, lon, w), PIXELS)
                print(f"  cover {cover[str(w)]}", flush=True)
            im = Image.fromarray(rgba).resize((PIXELS, PIXELS), Image.LANCZOS)  # plate carree -> metres square
            arr = np.asarray(im).copy()
            hole = arr[..., 3] < 128
            if hole.any():
                fill = (np.asarray(_centre_crop(coarser, 1 / 3).resize((PIXELS, PIXELS), Image.LANCZOS))
                        if coarser is not None else np.zeros((PIXELS, PIXELS, 3), np.uint8) + np.uint8([28, 52, 78]))
                arr[..., :3][hole] = fill[hole]
                print(f"  {hole.mean():.1%} nodata filled", flush=True)
            Image.fromarray(arr[..., :3]).save(paths[w])
        coarser = Image.open(paths[w]).convert("RGB")
    used = sorted({i for c in cover.values() for i, f in c.items() if f > 0})
    facts["credits"] = aerial.credits(used)
    record.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
    return paths


def _refill(im: Image.Image) -> Image.Image:
    """Paint over the flat dark nodata fill with colours grown in from the surrounding imagery (inpainted at 256
    px, then blurred and scaled up): open sea becomes sea-coloured instead of a navy wedge."""
    import cv2
    arr = np.asarray(im)
    hole = (np.abs(arr.astype(np.int16) - np.int16(NODATA)).max(axis=2) <= 6) | (arr.max(axis=2) < 8)  # or black
    if hole.mean() < 0.001:
        return im
    n = arr.shape[0]
    small = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)
    m = cv2.resize(hole.astype(np.uint8) * 255, (256, 256), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    fill = cv2.inpaint(small, m, 9, cv2.INPAINT_TELEA)
    fill = cv2.GaussianBlur(cv2.resize(fill, (n, n), interpolation=cv2.INTER_CUBIC), (0, 0), 6)
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 4)[..., None]
    soft = np.maximum(soft, hole[..., None])
    return Image.fromarray((arr * (1 - soft) + fill * soft).astype(np.uint8))


def _centre_crop(im: Image.Image, frac: float) -> Image.Image:
    n = im.size[0]
    a = n * (1 - frac) / 2
    return im.crop((round(a), round(a), round(n - a), round(n - a)))


class Dive:
    def __init__(self, paths: dict[int, Path], start: float, far: bool = False):
        """``far``: the pinhole uses every square (up to 72.9 km) and draws haze and sky, for views that reach the
        horizon (the oblique dive); otherwise the pyramid is 8.1 km down to 300 m, as for the v5 cut."""
        self.imgs = {w: Image.open(p).convert("RGB") for w, p in paths.items()}
        self.far = far
        if far:  # the oblique view reaches open sea, where the squares carry a flat dark nodata fill: paint it over
            self.imgs = {w: _refill(im) for w, im in self.imgs.items()}
            self.imgs = _match_levels(self.imgs)
        self.arr = {w: np.asarray(self.imgs[w]) for w in (sorted(paths) if far else LEVELS[1:])}
        self.start = start
        u = (np.arange(W) + 0.5 - W / 2) / (W / 2) * TAN
        v = (np.arange(H) + 0.5 - H / 2) / (W / 2) * TAN
        self.uu, self.vv = np.meshgrid(u, v)

    def _cut(self, level: int, foot: float, rot: float, off_m: float, heading: float) -> Image.Image:
        """A ``foot``-metre-wide straight-down view from square ``level``, centred ``off_m`` metres along
        ``heading`` from the landing, rotated ``rot`` degrees."""
        im = self.imgs[level]
        n = im.size[0]
        s = n / level  # px per metre
        cw, ch = foot * s, foot * s * H / W
        big = max(cw, ch) * 1.5  # room to rotate
        b = math.radians(heading)
        cx, cy = n / 2 + off_m * s * math.sin(b), n / 2 - off_m * s * math.cos(b)
        sub = im.crop(tuple(round(x) for x in (cx - big / 2, cy - big / 2, cx + big / 2, cy + big / 2)))
        sub = sub.rotate(rot, resample=Image.BICUBIC)
        k = sub.size[0] / big
        c = sub.size[0] / 2
        return sub.crop((int(c - cw * k / 2), int(c - ch * k / 2), int(c + cw * k / 2),
                         int(c + ch * k / 2))).resize((W, H), Image.LANCZOS)

    def nadir(self, foot: float, rot: float, off_m: float, heading: float) -> Image.Image:
        """Cut from the finest square that covers the view (and its slide); near the top of a square, blended
        toward the next coarser one, but never toward one coarser than the dive's start."""
        level = min(w for w in LEVELS if w >= foot + 2.2 * abs(off_m))
        up = [w for w in LEVELS if level < w <= self.start]
        a = self._cut(level, foot, rot, off_m, heading)
        t = float(np.clip((foot / level - 0.8) / 0.2, 0, 1))
        if up and t > 0:
            a = Image.blend(a, self._cut(min(up), foot, rot, off_m, heading), t)
        return a

    def pitched(self, h: float, fwd: float, pitch: float, rot: float) -> Image.Image:
        """Camera ``h`` m up over the ground point ``fwd`` m ahead (screen-up) of the landing, pitched ``pitch`` rad."""
        import cv2
        th = math.radians(rot)
        r_hat = np.array([math.cos(th), math.sin(th)])  # screen-right in image axes (x right, y down)
        f_hat = -np.array([-math.sin(th), math.cos(th)])  # screen-up
        den = self.vv * math.sin(pitch) + math.cos(pitch)
        t = h / den
        off_r = t * self.uu
        off_f = t * (math.sin(pitch) - self.vv * math.cos(pitch))
        gx = off_r * r_hat[0] + (off_f + fwd) * f_hat[0]
        gy = off_r * r_hat[1] + (off_f + fwd) * f_hat[1]
        foot = h * (2 * TAN / W) / den ** 2  # metres per output pixel
        pyramid = sorted(self.arr)  # 300, 900, 2700, 8100
        finest = pyramid[0]
        li = np.log(np.maximum(foot, 1e-3) / (finest / PIXELS)) / math.log(3)  # 0 = the finest square's resolution
        reach = np.maximum(np.abs(gx), np.abs(gy))
        for k, w in enumerate(pyramid[:-1]):  # toward a square's edge -> ease into the coarser one, so no hard line
            li = np.maximum(li, k + smoothstep((reach / w - EDGE_FROM) / (EDGE_TO - EDGE_FROM)))
        li = np.clip(li, 0, len(pyramid) - 1.001)
        lo = np.floor(li).astype(int)
        fr = (li - lo)[..., None]
        acc = np.zeros(gx.shape + (3,), np.float32)
        for k, w in enumerate(pyramid):
            wk = np.where((lo == k)[..., None], 1 - fr, 0) + np.where((lo == k - 1)[..., None], fr, 0)
            if wk.max() <= 0:
                continue
            mpp = w / PIXELS
            sample = cv2.remap(self.arr[w], (PIXELS / 2 + gx / mpp).astype(np.float32),
                               (PIXELS / 2 + gy / mpp).astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            acc += sample.astype(np.float32) * wk
        if self.far:
            # haze: with distance; toward the edge of the largest square, so its edge never shows; and in a band
            # just below the horizon, so the ground melts into a soft glow instead of ending in a hard line
            dist = np.where(den > 1e-3, h / np.maximum(den, 1e-3), 1e9)
            edge = np.maximum(np.abs(gx), np.abs(gy)) / (pyramid[-1] / 2)
            a = np.maximum.reduce([HAZE_MAX * (1 - np.exp(-dist / HAZE_M)),
                                   smoothstep((edge - 0.55) / 0.4),
                                   smoothstep(1 - den / HORIZON_BAND)])
            a = np.where(den <= 1e-3, 1.0, a)[..., None]
            acc = acc * (1 - a) + np.float32(HAZE) * a
            up = np.clip(-den / 0.6, 0, 1) ** 0.7  # sky: a pale glow at the horizon deepening to blue upward
            sky = np.float32(HAZE) * (1 - up[..., None]) + np.float32(SKY) * up[..., None]
            acc = np.where((den <= 1e-3)[..., None], sky, acc)
        return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))


def _mist(im: Image.Image, i: int, clear: int) -> Image.Image:
    if i >= clear:
        return im
    a = 1 - i / clear
    soft = im.convert("L").filter(ImageFilter.GaussianBlur(60)).convert("RGB")
    return Image.blend(im, Image.blend(Image.new("RGB", im.size, MIST), soft, 0.15), a ** 1.5 * 0.97)


def render(paths: dict[int, Path], out: Path, *, from_m: float, to_m: float, nadir_frames: int, pitch_frames: int,
           pitch_deg: float, spin: float, clear: int, aim: float = 15.0) -> dict:
    """Write the dive to ``out`` (and ``<out>_last.png``); returns facts for the build log."""
    d = Dive(paths, from_m)
    k = math.log(from_m / to_m) / (nadir_frames - 1)  # log zoom per frame
    h0 = to_m / (2 * TAN)

    def pitch_path():
        h, drift = h0, 0.0
        for j in range(1, pitch_frames + 1):
            t = j / pitch_frames
            h *= math.exp(-k * (1 - 0.5 * t))
            drift += h * k * 0.6 * t * t  # the forward drift eases in from 0
            yield t, h, drift

    *_, (_, h_end, drift_end) = pitch_path()
    # slide the nadir crop along the heading so the last pitched frame's centre looks ``aim`` m past the landing
    slide = aim - h_end * math.tan(math.radians(pitch_deg)) - drift_end
    last = None
    with Writer(out) as writer:
        for i in range(nadir_frames):
            e = float(smoothstep(i / (nadir_frames - 1)))  # spin and slide ease in and out while straight down
            last = _mist(d.nadir(from_m * math.exp(-k * i), spin * e, slide * e, spin), i, clear)
            writer.write(np.asarray(last))
        switch = float(np.abs(np.asarray(d.pitched(h0, slide, 0.0, spin)).astype(np.float32)
                              - np.asarray(last).astype(np.float32)).mean())
        for t, h, drift in pitch_path():
            last = d.pitched(h, slide + drift, math.radians(pitch_deg) * (2 * t ** 3 - t ** 4), spin)
            writer.write(np.asarray(last))
    save_png(np.asarray(last), out.with_name(out.stem + "_last.png"))
    return {"frames": nadir_frames + pitch_frames, "end_height_m": round(h_end, 1), "slide_m": round(slide, 1),
            "switch_mismatch": round(switch, 2)}


def render_oblique(paths: dict[int, Path], out: Path, *, from_height_m: float, to_height_m: float, frames: int,
                   start_pitch_deg: float, pitch_deg: float, turn_frames: int, heading: float, clear: int,
                   aim: float = 15.0) -> dict:
    """An angled dive (Jamie, 10 Oct: the straight-down view after a near-level cloud exit was unexplained).

    The camera flies straight at a point ``aim`` m past the landing, along heading ``heading``. It leaves the cloud
    looking forward and down at ``start_pitch_deg`` from straight down (70 puts the horizon near the top, as in the
    cloud shot) and over the first ``turn_frames`` frames tips down to ``pitch_deg`` (about 45), which it then holds
    to the hand-off, ``to_height_m`` up. Distance to the aim point falls at a constant log rate, so the approach
    reads as a steady dive. The ground is flat (no terrain): fine from altitude, flatter than life near hills."""
    d = Dive(paths, from_height_m, far=True)
    p0, p1 = math.radians(start_pitch_deg), math.radians(pitch_deg)

    def pitch(i: int) -> float:
        return p1 + (p0 - p1) * (1 - float(smoothstep(i / max(turn_frames, 1))))

    d0 = from_height_m / math.cos(p0)
    d1 = to_height_m / math.cos(p1)
    k = math.log(d0 / d1) / (frames - 1)
    last = None
    with Writer(out) as writer:
        for i in range(frames):
            p = pitch(i)
            dist = d0 * math.exp(-k * i)
            last = _mist(d.pitched(dist * math.cos(p), aim - dist * math.sin(p), p, heading), i, clear)
            writer.write(np.asarray(last))
    save_png(np.asarray(last), out.with_name(out.stem + "_last.png"))
    return {"frames": frames, "mode": "oblique", "end_height_m": round(to_height_m, 1),
            "start_distance_m": round(d0), "end_distance_m": round(d1)}
