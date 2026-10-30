"""The generic-orbit opening: a cached orbit photograph, a dive into cumulus and a join inside the flight's own cloud.

The opening no longer depicts the destination from orbit. A generic low-orbit photograph of a mostly cloud-covered
Earth (assets/touchdown/opening, made once with FLUX 3 Klein from a fixed seed) is approached slowly for ~16 frames,
then the camera dives into a field of sunlit cumulus (a second cached still) and continues into the flight's first
full-cloud frame J, where the film cuts to the flight. LTX 2.5 renders the 49-frame retreat on Layout To Render from
a camera-only proxy built here, with flight frames J+8..J pinned at frame 0 and the orbit still as look reference and
end pin. Stages: prepare (join, zoom rate, proxy, hard checks), worker (generation), assemble (40 generated frames +
the flight from J).
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'assets/touchdown/opening'
TIMING = REPO / 'workflows/touchdown/canonical/photographic-grounded-v1/opening-timing.json'
WIDTH, HEIGHT, FPS = 1280, 736, 24
GENERATED, PREFIX = 49, 9                 # generation frames; the first PREFIX are the pinned flight frames J+8..J
LAYOUT_STRENGTH = .5
# How the opening is guided (TOUCHDOWN_OPENING_MODE, default 'camera'): 'camera' = a globe-only proxy at Layout To
# Render 0.3, no cloud added, so LTX grows one cloud bank from the orbit's own clouds into the pinned flight frames;
# 'layers' = the v10 proxy with cumulus shells and a deck easing into flight frame J (three visible cloud phases);
# 'keyframes' = no proxy, the earlier canonical opening's pins (flight frames, a 1.8x closer orbit crop at generation
# frame 40 at 0.5, the orbit still at the end) with Prompt Relay.
MODES = ('layers', 'camera', 'keyframes')
MODE_STRENGTH = dict(layers=.5, camera=.3)
CLOSER_FRAME, CLOSER_SCALE, CLOSER_STRENGTH = 40, 1.8, .5
GUIDE_TIMES = [0, 4.5, 40, 48]             # Prompt Relay clock of those pins (split_flight.guide_times)
# 'camera': log zoom per frame from CAMERA_START, rising (smoothstep^CAMERA_CURVE, so ~2.5-3.5% to play 16) to a peak
# by CAMERA_PEAK_BY, then easing to the flight's rate at J over the last frames; the peak is set so the whole approach
# zooms CAMERA_TOTAL times. Zooming into the orbit still's cloudy centre hides the ground by ~play 20 even at 2.5%, so
# this mode checks ground >= CAMERA_GROUND through play 20.
CAMERA_START, CAMERA_PEAK_BY, CAMERA_TOTAL, CAMERA_CURVE, CAMERA_GROUND = .025, 34, 12., 4, .04
SEED_OFFSET = 32                          # the opening's seed is the scene seed + 32, as the grounded opening
# Join: among flight frames after JOIN_MIN with no blue sky in its top quarter and cloud nearly everywhere, whose next
# PREFIX - 1 frames are also cloud (the pinned prefix must be cloud throughout), the densest white one within
# DENSE_WINDOW frames of the first: brightest and least textured, so the opening's white bank meets a white bank and
# not the thin streaky cloud at the edge of the passage. Joins keeping the film within FILM_SECONDS are preferred.
JOIN_MIN, SKY_MAX, CLOUD_MIN, TOP_CLOUD_MIN, PREFIX_CLOUD_MIN = 8, .01, .97, .97, .9
DENSE_WINDOW, TEXTURE_WEIGHT, FILM_SECONDS = 12, 20., (8.1, 9.1)
RATE_BOUNDS = (.04, .2)                   # log zoom per frame the flight may have at J
# The frozen delivery ends the portrait as soon as the push reaches the requested crop, about 7 native frames in
# (Colosseum: native 24-30 of 73), which cuts the figure's action as it starts. The film continues the same native
# clip for up to ENDING_HOLD more frames, still moving, so the action plays out (the wave runs to about native 56).
ENDING_HOLD = 24
# The opening's free frames drift from the pinned flight frames (Sydney: generation frame 9 is bright cumulus, flight
# frame J=36 dimmer fog; the cut stepped 15 against ~6 either side). Assembly cuts from one of the last SEAM_GEN
# delivered frames to whichever full-cloud flight frame within SEAM_BACK before J to J+SEAM_AHEAD looks most alike.
SEAM_GEN, SEAM_BACK, SEAM_AHEAD = 3, 12, 8
# The descent's last frames retrace the portrait clip's own start, where its camera is nearly at rest, so the flight
# slows before the push-in (Sydney 6.3-7.0 s at a third of the median motion, after the frozen hold drop). In the
# RUN_IN frames before the portrait, a frame that moved less than RUN_IN_FLOOR x the flight's median is dropped, never
# two in a row (at most double speed), without blending.
RUN_IN, RUN_IN_FLOOR = 48, .6
# Those last frames are descent frames guided by native portrait frames 0..PORTRAIT_PREFIX-1, so the person already
# moves in them: dropping there sped her up just as the wave starts (Jamie 30 Oct). The run-in stops before them.
PORTRAIT_PREFIX = 25
# 'camera' mode with FOG frames (TOUCHDOWN_OPENING_FOG, 0 turns it off): the proxy's last FOG play frames before the pinned prefix
# ease from the globe into the flight's own full-cloud frames J-1, J-2, ..., so LTX's free frames next to the prefix
# start from the same fog instead of the globe's bright cumulus (the cause of the cut's step, see SEAM_GEN).
FOG = int(os.environ.get('TOUCHDOWN_OPENING_FOG', '12'))
# The camera: log distance to the cloud deck per play frame, ORBIT_RATE until ORBIT, eased up to the flight's own
# zoom rate at J by DIVE and held to the join. At orbit height the deck lies just above the ground, so the ground
# closes in at nearly the same rate. The orbit lasts ~1.2 s, as in the example films; the cloud closes over ~28-33.
ORBIT, DIVE, ORBIT_RATE = 28, 34, .016
DECK_DEPTH = .5                           # deck below the play-end camera, as a fraction of its distance to the ground
EASE = 14                                 # last moving frames over which the deck's cumulus eases into flight frame J
MAGNIFIED = (1.6, 3.)                     # a shell fades out between these screen magnifications of its texture
MAX_DIFF, FLIGHT_DIFF = 10.5, 1.25        # proxy gate on the last 16 play diffs: at most the larger of MAX_DIFF and
                                          # 1.25x the flight's own largest step among the pinned frames
WHITE_ENTRY = 3                         # camera proxy frames allowed to be plain white just before the fog
GROUND_MIN = .1                           # ground/limb visible through play 24 (the orbit must not be clouded over)
# Cumulus crops (x, y, width, flip) of the 1920x1088 still for the top shell, the middle shell and the deck (finest
# first, each continued by the next at 2.5x).
CROPS = dict(top=((0, 0, 1920, 0), (500, 300, 1100, 1), (100, 500, 1000, 0), (820, 40, 1100, 1)),
             mid=((300, 200, 1300, 1), (0, 0, 1920, 0), (900, 380, 1000, 0), (60, 60, 1200, 1)),
             deck=((700, 560, 960, 0), (300, 300, 1400, 1), (0, 0, 1920, 0), (1000, 500, 900, 1), (0, 0, 1920, 1)))
LAYERS = dict(top=30., mid=32., opacity=(.97, .9), ramps=((25, 30), (26, 31), (27, 33)), gap=(-16, -4),
              radius=(.25, .2, .2), width=(.02, .014))   # extents reach the horizon from the end camera
DECK_FRAMES = (8, 8, 3, 0)                # prefix generation frames continuing flight frame J outward, at 2.5x steps
EDGE, LOD_BIAS = .4, .3
SUN_RIM = np.array([255., 228., 205.])    # BGR pale blue of the atmosphere at the limb


class Held(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float32) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# The join and the flight's zoom rate.

def cloudiness(image):
    """0..1 per pixel: bright, unsaturated pixels are cloud."""
    hsv = cv2.cvtColor(image.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    return smoothstep(120, 175, hsv[..., 2]) * smoothstep(80, 45, hsv[..., 1])


def frame_stats(image):
    """Cloud fraction, cloud fraction of the top band and blue-sky fraction of the top quarter of one frame."""
    small = cv2.resize(image, (320, 184), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    cloud = cloudiness(small)
    sky = (hsv[..., 0] > 90) & (hsv[..., 0] < 130) & (hsv[..., 1] > 40)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return dict(cloud=round(float(cloud.mean()), 4), top_cloud=round(float(cloud[:28].mean()), 4),
                sky=round(float(sky[:46].mean()), 4), bright=round(float(hsv[..., 2].mean()), 2),
                texture=round(float(np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 3)).mean()), 3))


def read_frames(video, count=None):
    cap, out = cv2.VideoCapture(str(video)), []
    while count is None or len(out) < count:
        ok, image = cap.read()
        if not ok:
            break
        out.append(image)
    cap.release()
    return out


def detect_join(frames, extra=0):
    """The densest white full-cloud frame of the flight's cloud passage, and the per-frame statistics. `extra` frames
    follow the flight in the film (the ending hold) and count towards its length."""
    stats = [frame_stats(f) for f in frames]
    delivered = GENERATED - PREFIX
    full = [j for j in range(JOIN_MIN, len(frames) - GENERATED + 1)
            if stats[j]['sky'] < SKY_MAX and stats[j]['cloud'] >= CLOUD_MIN and stats[j]['top_cloud'] >= TOP_CLOUD_MIN
            and all(t['cloud'] >= PREFIX_CLOUD_MIN for t in stats[j:j + PREFIX])]
    if not full:
        raise Held('The flight has no full-frame cloud passage to join the opening to.')
    # Prefer joins that keep the film within FILM_SECONDS; a short or long flight keeps its own length.
    fits = [j for j in full if FILM_SECONDS[0] <= (delivered + len(frames) + extra - j) / 24 <= FILM_SECONDS[1]]
    full = fits or full
    window = [j for j in full if j <= full[0] + DENSE_WINDOW]
    return max(window, key=lambda j: stats[j]['bright'] - TEXTURE_WEIGHT * stats[j]['texture']), stats


def zoom_rate(frames, join, pairs=4):
    """The flight's log zoom per frame at the join: ECC affine scale on high-passed 640x368 greys, J..J+pairs."""
    grey = [cv2.cvtColor(cv2.resize(f, (640, 368), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            .astype(np.float32) for f in frames[join:join + pairs + 1]]
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 25)
    scales = []
    for a, b in zip(grey, grey[1:]):
        try:
            _, m = cv2.findTransformECC(hp(a), hp(b), np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                                        (3, 200, 1e-6), None, 5)
        except cv2.error:
            continue
        scales.append(float(np.sqrt(abs(np.linalg.det(m[:, :2])))))
    if not scales:
        raise Held('Could not measure the flight\'s zoom inside the cloud.')
    rate = float(np.log(np.median(scales)))
    return float(np.clip(rate, *RATE_BOUNDS)), scales


# The proxy: a sphere with the generic orbit projected on it, two cumulus shells and a cumulus deck.

def basis(pitch):
    return (np.array([0., -np.sin(pitch), np.cos(pitch)]), np.array([0., np.cos(pitch), np.sin(pitch)]),
            np.array([1., 0., 0.]))


def rays(pitch, f):
    fwd, up, right = basis(pitch)
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    d = fwd * f + right * (u - WIDTH / 2)[..., None] - up * (v - HEIGHT / 2)[..., None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def sphere_hit(cam, d, radius):
    """Nearest hit of a sphere seen from outside it; a shell the camera is already below is never hit."""
    b = d @ cam
    disc = b * b - (cam @ cam - radius * radius)
    hit = (b < 0) & (disc >= 0) & (cam @ cam > radius * radius)
    t = -b - np.sqrt(np.maximum(disc, 0))
    return np.where(hit, t, np.inf), cam + d * t[..., None]


def project(cam, point, pitch, f):
    fwd, up, right = basis(pitch)
    q = point - cam
    z = q @ fwd
    z = np.where(z > 1e-9, z, 1e-9)
    return ((f * (q @ right) / z + WIDTH / 2 - .5).astype(np.float32),
            (-f * (q @ up) / z + HEIGHT / 2 - .5).astype(np.float32))


def shell_point(cam0, fwd, radius):
    """Where the descent line crosses a shell (camera above it), with its tangent axes."""
    b = fwd @ cam0
    c = cam0 + fwd * (-b - np.sqrt(b * b - (cam0 @ cam0 - radius * radius)))
    n = c / np.linalg.norm(c)
    e2 = fwd - (fwd @ n) * n
    return c, np.array([1., 0., 0.]), e2 / np.linalg.norm(e2)


def pyramid(image):
    out = [image.astype(np.float32)]
    while min(out[-1].shape[:2]) > 16:
        out.append(cv2.pyrDown(out[-1]))
    return out


def mirror(x, n):
    x = np.mod(x, 2 * (n - 1))
    return np.where(x > n - 1, 2 * (n - 1) - x, x).astype(np.float32)


def bilinear(image, x, y):
    """Bilinear lookup in float: cv2.remap's 1/32-pixel table shows as steps where a texel spans tens of pixels."""
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, image.shape[1] - 1), np.minimum(y0 + 1, image.shape[0] - 1)
    fx, fy = x - x0, y - y0
    if image.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    return top * (1 - fy) + (image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy


def sample(levels, x, y, lod):
    """Trilinear lookup at level-0 pixel coordinates (mirrored beyond the edges) and a per-pixel level of detail."""
    lod = np.clip(np.nan_to_num(lod, nan=len(levels) - 1.), 0, len(levels) - 1)
    out = 0.
    for level in range(int(lod.min()), int(np.ceil(lod.max())) + 1):
        weight = np.clip(1 - np.abs(lod - level), 0, 1)
        if weight.any():
            image, scale = levels[level], 2. ** level
            h, w = image.shape[:2]
            got = bilinear(image, mirror((x + .5) / scale - .5, w), mirror((y + .5) / scale - .5, h))
            out = out + (weight[..., None] if got.ndim == 3 else weight) * got
    return out


def nested(stack, x, y, lod, ratio=2.5):
    """An image continued beyond its edges by the next images at ratio, ratio^2... its size (feathered, finest on top)."""
    h, w = stack[0][0].shape[:2]
    out = None
    for j in range(len(stack) - 1, -1, -1):
        k = ratio ** j
        xj, yj = (x - w / 2) / k + w / 2, (y - h / 2) / k + h / 2
        got = sample(stack[j], xj, yj, lod - np.log2(k))
        if out is None:
            out = got
            continue
        inside = smoothstep(0, .12 * w, np.minimum(np.minimum(xj, w - 1 - xj), np.minimum(yj, h - 1 - yj)))[..., None]
        out = out * (1 - inside) + got * inside
    return out


def density(x, y):
    """Texels per screen pixel (the larger axis) of a coordinate field."""
    (xr, xc), (yr, yc) = np.gradient(x), np.gradient(y)
    return np.maximum(np.maximum(np.hypot(xc, yc), np.hypot(xr, yr)), 1e-6)


def cloud_texture(frame, gap):
    """A cloud image as a layer: its colour, alpha from local luminance so the darker hollows become gaps."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = grey - cv2.GaussianBlur(grey, (0, 0), 40)
    return np.dstack([frame.astype(np.float32), 255 * smoothstep(*gap, local)])


def crops(still, rows):
    """Cumulus crops at the frame's aspect, resized to 1280x736."""
    h, w = still.shape[:2]
    out = []
    for x, y, width, flip in rows:
        width = min(width, round(h * WIDTH / HEIGHT))
        height = round(width * HEIGHT / WIDTH)
        x, y = max(0, min(x, w - width)), max(0, min(y, h - height))
        image = cv2.resize(still[y:y + height, x:x + width], (WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA if width >= WIDTH else cv2.INTER_CUBIC)
        out.append(cv2.flip(image, 1) if flip else image)
    return out


def schedule(dive_rate, play=GENERATED - PREFIX):
    """Cumulative log distance to the deck at play 0..play: ORBIT_RATE, smoothly rising to dive_rate by DIVE."""
    k = np.arange(play)
    t = np.clip((k + .5 - ORBIT) / (DIVE - ORBIT), 0, 1)
    rate = ORBIT_RATE + (dive_rate - ORBIT_RATE) * t * t * (3 - 2 * t)
    return np.concatenate([[0.], np.cumsum(rate)]), rate


def camera_schedule(dive_rate, play=GENERATED - PREFIX):
    """'camera' mode: cumulative log distance to the ground, starting at CAMERA_START per frame and accelerating
    smoothly to a peak, then easing to the flight's own rate at J, totalling log(CAMERA_TOTAL)."""
    k = np.arange(play) + .5
    up, down = smoothstep(0, CAMERA_PEAK_BY, k) ** CAMERA_CURVE, smoothstep(CAMERA_PEAK_BY, play - 1, k)
    fixed = np.sum(CAMERA_START * (1 - up) * (1 - down) + dive_rate * down)
    peak = max((np.log(CAMERA_TOTAL) - fixed) / np.sum(up * (1 - down)), CAMERA_START, dive_rate)
    rate = (CAMERA_START + (peak - CAMERA_START) * up) * (1 - down) + dive_rate * down
    return np.concatenate([[0.], np.cumsum(rate)]), rate


def camera_frames(orbit, pinned, fit, dive_rate, fog=()):
    """'camera' mode: the orbit still on the globe only, the camera diving toward the image centre; no cloud is
    added, so LTX grows the cloud from the pinned flight frames. Generation order, as proxy_frames."""
    h, pitch, f = fit['altitude_radii'], np.radians(fit['pitch_deg']), fit['focal_px']
    fwd, _, _ = basis(pitch)
    cam0 = np.array([0., 1 + h, 0.])
    b = fwd @ cam0
    distance = -b - np.sqrt(b * b - (cam0 @ cam0 - 1))
    play = GENERATED - PREFIX
    log_distance, rate = camera_schedule(dive_rate, play)
    d = rays(pitch, f)
    out = []
    for g in range(GENERATED):
        p = GENERATED - 1 - g
        if p >= play:
            out.append(pinned[g])
            continue
        cam = cam0 + fwd * distance * (1 - np.exp(-log_distance[p]))
        t_ground, n = sphere_hit(cam, d, 1.)
        hit = np.isfinite(t_ground)
        mx, my = project(cam0, n, pitch, f)
        image = np.where(hit[..., None], cv2.remap(orbit, mx, my, cv2.INTER_CUBIC,
                                                   borderMode=cv2.BORDER_REFLECT).astype(np.float32), 0.)
        b = d @ cam
        miss = np.sqrt(np.maximum(cam @ cam - b * b, 0)) - 1
        rim = max(.004, .02 * (np.sqrt(cam @ cam) - 1))
        glow = np.where(~hit & (b < 0) & (miss < 3 * rim), np.exp(-np.maximum(miss, 0) / rim), 0.)
        image = image + glow[..., None] * .8 * (SUN_RIM - image).clip(0)
        k = g - PREFIX                                     # fog[k] is flight frame J-1-k
        if k < len(fog):
            t = 1 - k / len(fog)
            w = t * t * (3 - 2 * t)
            image = w * fog[k].astype(np.float32) + (1 - w) * image
        out.append(image.clip(0, 255).astype(np.uint8))
    record = dict(mode='camera', fog_frames=len(fog), fit={k: fit[k] for k in ('altitude_radii', 'pitch_deg', 'focal_px')},
                  dive_rate=dive_rate, start_rate=CAMERA_START, total_zoom=CAMERA_TOTAL,
                  ground_scale_per_play=[round(float(np.exp(r)), 4) for r in rate])
    return out, record


def proxy_frames(orbit, still, pinned, fit, dive_rate, plays=None):
    """The proxy in generation (retreat) order: frames 0..PREFIX-1 are the pinned flight frames, the rest the camera
    path from the orbit still (play 0) to one step before flight frame J. Returns the frames and a record. With
    `plays`, only those play frames are rendered (the rest are None)."""
    h, pitch, f = fit['altitude_radii'], np.radians(fit['pitch_deg']), fit['focal_px']
    fwd, _, _ = basis(pitch)
    cam0 = np.array([0., 1 + h, 0.])
    b = fwd @ cam0
    distance = -b - np.sqrt(b * b - (cam0 @ cam0 - 1))
    play = GENERATED - PREFIX
    log_distance, rate = schedule(dive_rate, play)
    e = np.exp(-log_distance[-1])
    reach0 = DECK_DEPTH * distance / (DECK_DEPTH + e * (1 - DECK_DEPTH))   # the deck lies DECK_DEPTH of the way
    camera = lambda k: cam0 + fwd * reach0 * (1 - np.exp(-np.interp(k, np.arange(play + 1), log_distance)))
    altitude = lambda k: float(np.linalg.norm(camera(k)) - 1)
    deck_centre = cam0 + fwd * reach0
    texture_of = lambda image: cloud_texture(image, LAYERS['gap'])
    shells = [(1 + altitude(LAYERS[name]), [pyramid(texture_of(i)) for i in crops(still, CROPS[name])],
               LAYERS['width'][k], LAYERS['radius'][k], LAYERS['ramps'][k]) for k, name in enumerate(('top', 'mid'))]
    deck_cumulus = [pyramid(i) for i in crops(still, CROPS['deck'])]
    deck_flight = [pyramid(pinned[i]) for i in DECK_FRAMES]
    deck_radius, deck_extent, deck_ramp = float(np.linalg.norm(deck_centre)), LAYERS['radius'][2], LAYERS['ramps'][2]
    _, *deck_axes = shell_point(cam0, fwd, deck_radius)
    deck_scale = f / ((deck_centre - camera(play)) @ fwd)       # one texel per pixel at the centre from the end camera
    d = rays(pitch, f)
    still_cloud = cloudiness(orbit.astype(np.float32))
    out, exposed = [], {}
    for g in range(GENERATED):
        p = GENERATED - 1 - g
        if p >= play:
            out.append(pinned[g])
            continue
        if plays is not None and p not in plays:
            out.append(None)
            continue
        cam = camera(p)
        t_ground, n = sphere_hit(cam, d, 1.)
        hit = np.isfinite(t_ground)
        mx, my = project(cam0, n, pitch, f)                       # the orbit still, projected from the start camera
        image = np.where(hit[..., None], cv2.remap(orbit, mx, my, cv2.INTER_CUBIC,
                                                   borderMode=cv2.BORDER_REFLECT).astype(np.float32), 0.)
        bare = hit * (1 - cv2.remap(still_cloud, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        # Deck: cumulus on the sphere's tangent axes, easing over the last EASE moving frames into flight frame J
        # projected from the end camera where that projection is well-conditioned.
        t, s = sphere_hit(cam, d, deck_radius)
        front = t < t_ground
        q = s - deck_centre
        tx, ty = (q @ deck_axes[0]) * deck_scale + WIDTH / 2, (q @ deck_axes[1]) * deck_scale + HEIGHT / 2
        colour = nested(deck_cumulus, tx, ty, np.log2(density(tx, ty)))
        w = float(smoothstep(play - EASE - 1, play - 1, p))
        if w > 0:
            cx, cy = project(camera(play), s, pitch, f)
            flight_colour = nested(deck_flight, cx, cy, np.log2(density(cx, cy)))
            bright = flight_colour.mean(-1) / 255 - .7
            depth = smoothstep(.1, .5, (s - camera(play)) @ fwd / ((deck_centre - camera(play)) @ fwd) + EDGE * bright)
            colour = colour + (flight_colour - colour) * (w * depth)[..., None]
        reach = np.linalg.norm(q, axis=-1) * (1 - EDGE * (colour.mean(-1) / 255 - .7))
        alpha = front * (1 - smoothstep(.6 * deck_extent, deck_extent, reach)) * smoothstep(*deck_ramp, p)
        image = image * (1 - alpha[..., None]) + colour * alpha[..., None]
        bare = bare * (1 - alpha)
        for index, (radius, levels, width, extent, ramp) in enumerate(shells):
            t, s = sphere_hit(cam, d, radius)
            front = t < t_ground
            c, e1, e2 = shell_point(cam0, fwd, radius)
            q = s - c
            sx, sy = (q @ e1) * WIDTH / width + WIDTH / 2, (q @ e2) * WIDTH / width + HEIGHT / 2
            layer = nested(levels, sx, sy, np.log2(density(sx, sy)) + LOD_BIAS)
            reach = np.linalg.norm(q, axis=-1) * (1 + EDGE * (.5 - layer[..., 3] / 255))
            alpha = front * (1 - smoothstep(.55 * extent, extent, reach)) * smoothstep(*ramp, p)
            alpha = alpha * layer[..., 3] / 255 * LAYERS['opacity'][index]
            alpha = alpha * (1 - smoothstep(*MAGNIFIED, 1 / density(sx, sy)))
            image = image * (1 - alpha[..., None]) + layer[..., :3] * alpha[..., None]
            bare = bare * (1 - alpha)
        b = d @ cam
        miss = np.sqrt(np.maximum(cam @ cam - b * b, 0)) - 1
        rim = max(.004, .02 * (np.sqrt(cam @ cam) - 1))
        glow = np.where(~hit & (b < 0) & (miss < 3 * rim), np.exp(-np.maximum(miss, 0) / rim), 0.)
        image = image + glow[..., None] * .8 * (SUN_RIM - image).clip(0)
        exposed[p] = round(float(bare.mean()), 4)
        out.append(image.clip(0, 255).astype(np.uint8))
    record = dict(fit={k: fit[k] for k in ('altitude_radii', 'pitch_deg', 'focal_px')}, dive_rate=dive_rate,
                  orbit_rate=ORBIT_RATE, orbit_frames=ORBIT, dive_by=DIVE, deck_depth=DECK_DEPTH, ease_frames=EASE,
                  deck_scale_per_play=[round(float(np.exp(r)), 4) for r in rate], layers=LAYERS, crops=CROPS,
                  camera_altitude_by_play={k: round(altitude(k), 5) for k in range(0, play + 1, 4)},
                  ground_exposed_by_play=exposed)
    return out, record


def ground_visible(image):
    """Fraction of the frame that is lit surface other than cloud (ground, sea, the limb)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] > 40) & (cloudiness(image) < .5)).mean())


def checks(images):
    """Play-order checks: 640x368 grey mean |diff| per pair (gated against the pinned flight frames' own steps),
    flat-white frames, ground visible through play 24, and the strongest off-centre autocorrelation peak of each
    frame's high-pass detail (a repeated tile shows as a peak)."""
    grey = [cv2.cvtColor(cv2.resize(i, (640, 368), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for i in images[::-1]]
    diffs = [round(float(np.abs(a - b).mean()), 2) for a, b in zip(grey, grey[1:])]
    flat = [k for k, x in enumerate(grey) if x.mean() > 225 and cv2.Laplacian(x, cv2.CV_32F).std() < 3]
    peaks = []
    for x in grey:
        hp = x - cv2.GaussianBlur(x, (0, 0), 8)
        spec = np.fft.fft2(hp)
        ac = np.fft.fftshift(np.real(np.fft.ifft2(spec * np.conj(spec))))
        ac /= max(float(ac.max()), 1e-9)
        cy, cx = np.array(ac.shape) // 2
        ac[cy - 12:cy + 13, cx - 12:cx + 13] = 0
        peaks.append(round(float(ac.max()), 3))
    play = len(images) - PREFIX
    ground = [round(ground_visible(i), 4) for i in images[::-1][:play]]
    gate = max(MAX_DIFF, FLIGHT_DIFF * max(diffs[play:] or [0]))
    return dict(play_diffs=diffs, max_diff_last16=max(diffs[-16:]), diff_gate=round(gate, 2),
                flat_white_play_frames=flat, ground_visible_by_play=ground,
                min_ground_to_24=min(ground[:25]), max_repeat_peak=max(peaks))


def encode(images, target, crf=14, size=(WIDTH, HEIGHT)):
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                '-s', f'{size[0]}x{size[1]}', '-r', str(FPS), '-i', 'pipe:0', '-an',
                                '-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', str(target)],
                               stdin=subprocess.PIPE)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('Could not encode ' + str(target))


def sheet(images, target):
    tiles = [cv2.copyMakeBorder(cv2.resize(images[g], (320, 184)), 1, 1, 1, 1, cv2.BORDER_CONSTANT)
             for g in range(len(images) - 1, -1, -4)]                         # play order, every 4th
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(target), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def seam_grey(image):
    return cv2.cvtColor(cv2.resize(image, (320, 184), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def best_seam(generated, flight, prefix, join):
    """(generation frame, flight frame, difference) of the closest cut from the opening into the flight's cloud:
    generation frames prefix..prefix+SEAM_GEN-1 (later ones drop that many opening frames) against full-cloud flight
    frames join-SEAM_BACK..join+SEAM_AHEAD. Mean absolute difference of 320x184 greys."""
    candidates = [j for j in range(max(JOIN_MIN, join - SEAM_BACK), min(len(flight), join + SEAM_AHEAD + 1))
                  if (lambda t: t['sky'] < SKY_MAX and t['cloud'] >= CLOUD_MIN)(frame_stats(flight[j]))] or [join]
    cloud = {j: seam_grey(flight[j]) for j in candidates}
    best = None
    for g in range(prefix, min(len(generated), prefix + SEAM_GEN)):
        a = seam_grey(generated[g])
        for j in candidates:
            d = float(np.abs(a - cloud[j]).mean())
            if best is None or d < best[2]:
                best = (g, j, round(d, 2))
    return best


def guided_start(pacing):
    """Flight frame where the descent frames guided by the native portrait begin (she is already moving there)."""
    mapping = pacing.get('source_frame_mapping') or []
    last = PORTRAIT_PREFIX - 1
    return next((i for i, m in enumerate(mapping) if m.startswith('native:')
                 or (m.startswith('descent:') and int(m.split(':')[1]) <= last)), None)


def run_in_drops(flight_video, join, native_start, guided_from=None):
    """Flight frames to drop from the slow run-in to the portrait (indices into the flight video)."""
    cap = cv2.VideoCapture(str(flight_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, join)
    greys = []
    for _ in range(join, native_start):
        ok, image = cap.read()
        if not ok:
            break
        greys.append(seam_grey(image))
    cap.release()
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(greys, greys[1:])]
    if guided_from is not None:                    # never drop a frame that shows the person moving (see RUN_IN)
        diffs = diffs[:max(0, guided_from - join - 1)]
    if len(diffs) < RUN_IN:
        return []
    step, drops, moved, last = RUN_IN_FLOOR * float(np.median(diffs)), [], 0., False
    for i in range(len(diffs) - RUN_IN + 1, len(diffs)):          # frame i + 1 follows diffs[i]; the last stays
        moved += diffs[i - 1]
        if moved < step and not last:
            drops.append(join + i); last = True
        else:
            moved, last = 0., False
    return drops


def ending_hold(pacing, size, hold=ENDING_HOLD):
    """The native portrait frames after the delivered ending, as delivery treats them: the frozen sharpness ramp
    continued from the portrait seam, then resized to the film. Empty when the clip has nothing left."""
    ending, ramp = pacing.get('ending') or {}, pacing.get('portrait_sharpness_ramp') or {}
    last, available = ending.get('native_last_frame'), ending.get('native_available', 0)
    if last is None:
        return [], []
    indices = list(range(last + 1, min(available, last + 1 + hold)))
    if not indices:
        return [], []
    cap = cv2.VideoCapture(pacing['native_portrait'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    sigma, frames, first = ramp.get('sigma', 0.), ramp.get('frames', 1), pacing['native_first_frame']
    images = []
    for k in indices:
        ok, image = cap.read()
        if not ok:
            break
        weight = 1 - (k - first) / frames
        if sigma and weight > 0:
            image = cv2.GaussianBlur(image, (0, 0), sigma * weight)
        if image.shape[1::-1] != tuple(size):
            image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LANCZOS4)
        images.append(image)
    cap.release()
    return indices[:len(images)], images


# Stages, each run in its own process by canonical_stage.py.

def prepare(root, mode=None, strength=None):
    mode = mode or os.environ.get('TOUCHDOWN_OPENING_MODE', 'camera')
    if mode not in MODES:
        raise ValueError('Unknown opening mode: ' + mode)
    if strength is None and os.environ.get('TOUCHDOWN_OPENING_STRENGTH'):
        strength = float(os.environ['TOUCHDOWN_OPENING_STRENGTH'])
    strength = MODE_STRENGTH.get(mode) if strength is None else strength
    folder = root / 'opening'
    if folder.exists():
        raise ValueError('Fresh opening root required')
    folder.mkdir()
    source = root / 'flight/run/motion/earth_zoom.mp4'
    frames = read_frames(source)
    cap = cv2.VideoCapture(str(source))
    info = [len(frames), int(cap.get(3)), int(cap.get(4)), cap.get(5)]
    cap.release()
    if info[3] != 24 or len(frames) < GENERATED + JOIN_MIN:
        raise ValueError('The flight is too short or not 24 fps')
    pacing = read(root / 'flight/run/motion/pacing.json')
    held, tail = ending_hold(pacing, info[1:3])
    hold = folder / 'ending-hold.mp4'
    if tail:
        encode(tail, hold, crf=12, size=info[1:3])
    join, stats = detect_join(frames, len(tail))
    rate, scales = zoom_rate(frames, join)
    shutil.copytree(root / 'flight/frozen-code', folder / 'frozen-code')
    shutil.copy2(root / 'flight/pinned-model-paths.json', folder / 'pinned-model-paths.json')
    shutil.copy2(TIMING, folder / 'timing.json')
    for name in ('generic-orbit.png', 'generic-orbit.json', 'cumulus.png', 'cumulus.json', 'limb-fit.json'):
        shutil.copy2(ASSETS / name, folder / name)
    pinned = [cv2.resize(frames[i], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
              for i in range(join + PREFIX - 1, join - 1, -1)]
    prefix = folder / 'cloud-retreat-prefix.mp4'
    encode(pinned, prefix, crf=12)
    pinned = read_frames(prefix)              # the proxy carries exactly the frames LTX is given
    orbit, still = cv2.imread(str(folder / 'generic-orbit.png')), cv2.imread(str(folder / 'cumulus.png'))
    guides, c = [], None
    if mode == 'keyframes':
        zw, zh = round(WIDTH / CLOSER_SCALE), round(HEIGHT / CLOSER_SCALE)
        x, y = (WIDTH - zw) // 2, (HEIGHT - zh) // 2
        cv2.imwrite(str(folder / 'closer-earth.png'),
                    cv2.resize(orbit[y:y + zh, x:x + zw], (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4))
        guides = [folder / 'closer-earth.png']
    else:
        if mode == 'camera':
            fog = []
            for j in range(join - 1, max(join - 1 - FOG, -1), -1):
                if stats[j]['sky'] >= SKY_MAX or stats[j]['cloud'] < CLOUD_MIN:
                    break
                fog.append(cv2.resize(frames[j], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            images, record = camera_frames(orbit, pinned, read(folder / 'limb-fit.json'), rate, fog)
        else:
            images, record = proxy_frames(orbit, still, pinned, read(folder / 'limb-fit.json'), rate)
        record['checks'] = c = checks(images)
        encode(images, folder / 'globe-proxy.mp4')
        sheet(images, folder / 'globe-proxy-sheet.jpg')
        write(folder / 'globe-proxy.json', record)
        guides = [folder / 'globe-proxy.mp4']
        # A globe-only proxy cuts from globe to the pinned cloud at J by design: only its moving frames are gated.
        jump = max(c['play_diffs'][:GENERATED - PREFIX - 1]) if mode == 'camera' else c['max_diff_last16']
        ground_ok = (min(c['ground_visible_by_play'][:21]) >= CAMERA_GROUND if mode == 'camera'
                     else c['min_ground_to_24'] >= GROUND_MIN)
        # Fog frames are the flight's own cloud and may be plain white (Highclere, 29 Oct); only the
        # globe part of the proxy must never wash out.
        # The camera proxy itself goes white as it enters the cloud, so its last few frames before the fog
        # may be plain white too (Longleat: one fog frame, frame 38 white).
        cloud_from = len(images) - PREFIX - (len(fog) + WHITE_ENTRY if mode == 'camera' else 0)
        flat = [k for k in c['flat_white_play_frames'] if k < cloud_from]
        if flat or (mode == 'layers' and jump >= c['diff_gate']) or not ground_ok:
            raise Held(f'The opening proxy failed its checks (flat white frames {c["flat_white_play_frames"]}, '
                       f'last-16 diff {c["max_diff_last16"]} against {c["diff_gate"]}, '
                       f'ground through play 24 {c["min_ground_to_24"]} against {GROUND_MIN}).')
    seed = read(root / 'prepared/scene-config.json')['seed'] + SEED_OFFSET
    files = [p for p in (folder / 'frozen-code').rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.name != 'extra-paths.json']
    files += [source, prefix, *guides, *([hold] if tail else []), folder / 'pinned-model-paths.json', folder / 'timing.json',
              *(folder / n for n in ('generic-orbit.png', 'cumulus.png', 'limb-fit.json')), Path(__file__).resolve()]
    write(folder / 'experiment.json', dict(
        recipe='generic-orbit-opening-v1', source=str(source), source_info=info, parent_seed=seed,
        generated_frames=GENERATED, prefix_frames=PREFIX, generated_size=[WIDTH, HEIGHT], join_source_frame=join,
        join_stats=stats[max(0, join - 4):join + PREFIX], flight_zoom_rate=rate, flight_zoom_scales=scales,
        ending_hold=dict(video=str(hold) if tail else None, native_frames=held, frames=len(tail)),
        mode=mode, layout_strength=strength, source_prefix_frames=list(range(join + PREFIX - 1, join - 1, -1)),
        generated_delivery_frames=list(range(GENERATED - 1, PREFIX - 1, -1)), film_frames=GENERATED - PREFIX
        + info[0] - join + len(tail), proxy_checks=c, crossfade=False, retiming=False,
        hashes={str(p.resolve()): sha(p) for p in files}))
    write(root / 'status.json', dict(stage='opening', status='prepared'))


def verify(folder):
    for path, expected in read(folder / 'experiment.json')['hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Changed opening input: ' + path)


def worker(root):
    """Generate the 49-frame retreat with the job's frozen inference code (Layout To Render, or keyframes only)."""
    runtime = Path('/tmp/touchdown-generation-runtime.env')   # fills in a hand-run stage; an app worker's own
    if runtime.is_file():                                     # environment wins
        for line in runtime.read_text().splitlines():
            key, value = line.split('=', 1)
            if key.startswith('TOUCHDOWN_'):
                os.environ.setdefault(key, json.loads(value))
    folder = root / 'opening'
    verify(folder)
    plan = read(folder / 'experiment.json')
    code = folder / 'frozen-code'
    sys.path[:0] = [str(code / 'scripts'), str(code / 'src')]
    os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:2])
    from touchdown import split_flight as flight, studio_build as studio
    import fakeout_earth_zoom as f
    assert Path(flight.__file__).is_relative_to(code)
    studio.REPO = REPO
    command = os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--deterministic' not in shlex.split(command):
        command += ' --deterministic'
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD'] = (command + ' --extra-model-paths-config '
                                                + shlex.quote(str(folder / 'pinned-model-paths.json')))
    flight.enable_prompt_relay()
    part = folder / 'generation'
    part.mkdir()
    write(root / 'status.json', dict(stage='opening', status='generating'))
    with studio.runtime('comfy', folder):
        with f._client(studio.VIDEO_URL) as client:
            graph = {'50': {'class_type': 'LoadVideo',
                            'inputs': {'file': client.assets.from_file(str(folder / 'cloud-retreat-prefix.mp4'))}},
                     '51': {'class_type': 'GetVideoComponents', 'inputs': {'video': ['50', 0]}}}
            orbit = f._load(graph, client, '46', folder / 'generic-orbit.png')
            if plan.get('mode', 'layers') == 'keyframes':
                # The canonical opening's call (scene_grounded_workflow.opening_worker) with the generic orbit.
                closer = f._load(graph, client, '47', folder / 'closer-earth.png')
                saved, _ = flight._generate(
                    f, client, graph, plan['generated_frames'], [(0, ['51', 0]), (CLOSER_FRAME, closer), (-1, orbit)],
                    {CLOSER_FRAME: CLOSER_STRENGTH}, read(folder / 'timing.json'), GUIDE_TIMES, plan['parent_seed'],
                    'touchdown/generic_orbit_opening', part, dict(width=WIDTH, height=HEIGHT))
            else:
                graph.update({'59': {'class_type': 'LoadVideo',
                                     'inputs': {'file': client.assets.from_file(str(folder / 'globe-proxy.mp4'))}},
                              '60': {'class_type': 'GetVideoComponents', 'inputs': {'video': ['59', 0]}}})
                saved, _ = flight._generate_layout(
                    client, graph, plan['generated_frames'], [(0, ['51', 0]), (-1, orbit)], {},
                    read(folder / 'timing.json'), plan['parent_seed'], 'touchdown/generic_orbit_opening', part,
                    dict(width=WIDTH, height=HEIGHT, layout_strength=plan['layout_strength']), orbit)
            write(folder / 'generation-result.json', flight._record(saved, f))
    verify(folder)
    write(root / 'status.json', dict(stage='opening', status='generated'))


def assemble(root):
    """The 40 delivered generated frames in play order, the flight from frame J, then the ending hold (the rest of
    the native portrait's action). No crossfade, no retiming."""
    folder = root / 'opening'
    verify(folder)
    plan, result = read(folder / 'experiment.json'), read(folder / 'generation-result.json')
    if sha(Path(result['video'])) != result['sha256']:
        raise ValueError('The generated opening changed')
    count, w, h, _ = plan['source_info']
    n, prefix, join, dest = plan['generated_frames'], plan['prefix_frames'], plan['join_source_frame'], root / 'complete-flight.mp4'
    flight = read_frames(plan['source'], join + SEAM_AHEAD + 1)
    generated = read_frames(result['video'], prefix + SEAM_GEN)
    planned = round(float(np.abs(seam_grey(generated[prefix]) - seam_grey(flight[join])).mean()), 2)
    first, join, seam = best_seam(generated, flight, prefix, join)
    pacing = read(Path(plan['source']).parent / 'pacing.json')
    drops = run_in_drops(plan['source'], join, pacing['native_start_frame'], guided_start(pacing))
    hold = plan.get('ending_hold') or {}
    held = hold.get('frames', 0)
    inputs = ['-i', result['video'], '-i', plan['source']] + (['-i', hold['video']] if held else [])
    graph = (f'[0:v]trim=start_frame={first}:end_frame={n},reverse,scale={w}:{h},setpts=PTS-STARTPTS[a];'
             f'[1:v]trim=start_frame={join},setpts=PTS-STARTPTS'
             + (",select='not(" + '+'.join(f'eq(n\\,{d - join})' for d in drops) + ")',setpts=N/(24*TB)" if drops else '')
             + '[b];'
             + ('[2:v]setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0[v]' if held else '[a][b]concat=n=2:v=1:a=0[v]'))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', graph, '-map', '[v]', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(dest)], check=True)
    frames = n - first + count - join - len(drops) + held
    cap = cv2.VideoCapture(str(dest))
    got = (int(cap.get(7)), cap.get(5))
    cap.release()
    if got != (frames, 24):
        raise ValueError(f'Assembled film has {got}, expected {frames} frames at 24 fps')
    write(root / 'assembly.json', dict(video=str(dest), sha256=sha(dest), frames=frames, fps=24,
                                       replacement_frames=n - first, join_source_frame=join,
                                       planned_join_source_frame=plan['join_source_frame'], run_in_dropped_frames=drops,
                                       seam=dict(generated_frame=first, flight_frame=join, difference=seam,
                                                 planned_difference=planned),
                                       retained_source_frames=count - join, ending_hold_frames=held,
                                       ending_hold_native_frames=hold.get('native_frames', []), duration_seconds=frames / 24,
                                       crossfade=False, retiming=False,
                                       export='H.264 CRF16; source frames from the join re-encoded'))
    write(root / 'status.json', dict(stage='assembled', status='awaiting_review', app_promoted=False))
