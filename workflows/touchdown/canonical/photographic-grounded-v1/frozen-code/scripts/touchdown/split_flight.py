"""Split photographic flight for any place: the Stonehenge anchored approach, generalised.

One generation from the portrait to orbit with pins only at its ends holds a view and dissolves
(Teotihuacan, 24 Oct). Three independently drawn references (a regional high still, an edited
low aerial, the portrait) do not lie on one camera path either, and LTX cuts between them
(Teotihuacan split3, 24 Oct). The Stonehenge film that worked took every reference from one
continuous generated flight, so they shared a path. This module does that for any place:

1. Parent: one timed (Prompt Relay) retreat from the reversed portrait to orbit.png, with 48 extra
   orbital frames that are discarded so the film opens moving. Its delivered orbit and first part of
   the cloud (to delivery 40) are kept. Its frame at 5 s is repaired by FLUX into anchor.png, with the
   arrival as the landmark reference.
2. The anchor is checked by the vision model between GPU stages (studio_build).
3. Descent: 153 frames, generated as a retreat and reversed, with the reversed portrait at frame 0,
   anchor.png at frame 72, the checked high.png at frame 128 (delivery 64, just below the cloud) and the
   parent's own cloud frames as a moving guide at its end, with timed beats.

Delivery joins whole frames: the parent opening, the descent, then the native portrait. No dissolve,
interpolation or face patch. Motion is measured and reported, not gated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np

from touchdown.authored_scene import digest, generate_asset
from touchdown.photo_delivery import JoinMismatch, frame_at, match_metrics, video_info
from touchdown.photo_gate import require_arrival_gate
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.relay_timing import wire_relay

FPS = 24
# Delivery beats of the parent, as in relay_timing: orbit, clouds, high aerial, approach, settle.
BOUNDS = (0, 29, 58, 108, 156, 192)
# 0.75 stalled the camera for 1-1.5 s at the anchor in all three option-B films (Machu Picchu, Hagia
# Sophia, Petra); with the raised arrival the parent climbs on its own, so the anchor only steers.
# The parent climbed straight from a low oblique into cloud on most places (Angkor, Mont-Saint-Michel,
# Alhambra, Neuschwanstein). Pinning the checked high.png into the parent at half strength only nudged it
# (Chambord broad, Chartres medium-high, Carcassonne low, 25 Oct). So the parent is now kept only to deep
# cloud (delivery 40) and the descent starts there: it comes out of the cloud onto high.png, pinned at
# delivery 64 just after the cloud beat, then flies down to the anchor. The join is inside the cloud.
# At 0.6 the camera passed the high still by at medium height (Fontainebleau f1); at 0.8 it reached it and
# froze for 1.1 s (Versailles e1, whose join failed before the hold trim ran). 0.8 with the hold trim.
# Over five places that cloud join reached the high view once (frozen), cut twice where high.png's terrain
# differed from the arrival (Segovia, Avila) and failed to join twice (Versailles, Avila). Jamie found the
# parent-pin film (Chambord, ae15da1) the best, so the parent pin is the default again: high.png at half
# strength just below the cloud exit inside the kept opening, the join below the cloud. high_join="cloud"
# keeps the descent-from-cloud variant: 
#   split=dict(high_join="cloud", opening_frames=40, descent_frames=153, high_strength=.8)
# Dubrovnik d1 (26 Oct): with the moving high guide the broad high view held ~2 s, but a 6 % pull-back ran
# at 0.65x the flight's speed (near-hold), and with the anchor at delivery 120 the descent had 48 frames to
# come down 2 km and dissolved the walls in over the high view. 25 % over the guide matches the flight's
# speed there; the anchor at 136 gives the descent 64 frames for that drop.
# Chillon c1: the layout proxy at strength 1.0 gave the first continuous dive, but Layout To Render copied the
# proxy faithfully, magnified blur and pasted portrait included. At 0.6 (Bodiam b1) it still drew the
# magnified still as a soft map from ~4.3 s; 0.4 asks it to repaint more while keeping the camera path.
DEFAULTS = dict(parent_handles=48, high_join="parent", opening_frames=72, anchor_delivery_frame=136,
                anchor_strength=.5, high_delivery_frame=64, high_strength=.7, high_motion_frames=24, high_zoom=1.25,
                parent_layout=False, layout_strength=.4, layout_nest=.012, layout_rungs=6, ladder_drift=.01,
                seam_carry_frames=24, seam_carry_strength=.7, seam_carry_accel=4.0, seam_bridge_frames=32, seam_carry_source="parent",
                descent_frames=121, prefix_frames=25, width=1280, height=736)


# Kept-opening lengths the descent may start from (descent = 193 - opening frames, 8n+1). The join is
# invisible only in near-featureless whiteout; at delivery 40 Versailles' parent still showed bright cloud
# edges and terrain, and its descent's cloud differed by 19 grey levels. 32 (a 161-frame descent) would pass
# the 24 GB card, which peaked at 23.9 GB on 153 frames.
CLOUD_OPENINGS = (40, 48)


def cloud_opening(parent_at, candidates=CLOUD_OPENINGS):
    """The kept opening whose seam frames (parent delivery O..O+7, the descent's moving guide) are the most
    featureless cloud, with every candidate's mean texture."""
    scores = {O: round(float(np.mean([_sharpness(parent_at(d)) for d in range(O, O + SEAM_FRAMES)])), 2)
              for O in candidates}
    return min(scores, key=scores.get), scores


def with_opening(o, opening):
    if opening not in CLOUD_OPENINGS:
        raise ValueError("The kept opening must be one of the cloud openings")
    return {**o, "opening_frames": opening, "descent_frames": BOUNDS[-1] - opening + 1}


def options(cfg):
    # TOUCHDOWN_SPLIT (JSON) overrides the defaults for one build, e.g. '{"parent_layout": true}'.
    o = {**DEFAULTS, **cfg.get("video", {}).get("split", {}), **json.loads(os.environ.get("TOUCHDOWN_SPLIT") or "{}")}
    if o["parent_handles"] % 8 or o["parent_handles"] < 0:
        raise ValueError("parent_handles must be a multiple of 8")
    for key in ("descent_frames", "prefix_frames"):
        if o[key] < 9 or o[key] % 8 != 1:
            raise ValueError(f"{key} must be 8n+1 frames")
    if o["descent_frames"] != BOUNDS[-1] - o["opening_frames"] + 1:
        raise ValueError("The descent must cover the parent delivery after the kept opening")
    if not o["opening_frames"] < o["anchor_delivery_frame"] < BOUNDS[-1] - o["prefix_frames"]:
        raise ValueError("The opening must precede the anchor and the portrait")
    if not 0 < o["high_strength"] < 1:
        raise ValueError("A full-strength high still stalls the camera; use a strength below 1")
    if o["high_motion_frames"] and (o["high_motion_frames"] % 8 or o["high_delivery_frame"] % 8):
        raise ValueError("The high view's moving guide needs 8k frames ending on a delivery frame that is a multiple of 8")
    if o["high_motion_frames"] and o["high_motion_frames"] + o["high_delivery_frame"] > BOUNDS[3] + 1:
        raise ValueError("The high view's moving guide must stay inside the high aerial beat")
    if o["seam_carry_frames"] % 8 or (o["seam_carry_frames"] and not 0 < o["seam_carry_strength"] < 1):
        raise ValueError("The seam carry needs 8k frames at a strength below 1")
    if o["seam_bridge_frames"] % 8 or (o["seam_bridge_frames"] and not o["seam_carry_frames"]):
        raise ValueError("The bridge needs 8k frames and a seam carry")
    if o["seam_carry_source"] not in ("parent", "zoom"):
        raise ValueError("seam_carry_source is 'parent' (the parent's own frames) or 'zoom' (a synthetic crop zoom)")
    if not 1 <= o["seam_carry_accel"] <= 6:
        raise ValueError("seam_carry_accel is how many times faster the carry ends than it starts (1-6)")
    if o["parent_layout"] and o["high_join"] != "parent":
        raise ValueError("The layout proxy flies the parent, so it needs the parent join")
    if not 0 <= o["layout_rungs"] <= 8 or not 0 <= o["ladder_drift"] < .05:
        raise ValueError("layout_rungs is 0 (nest the portrait) to 8 doublings; ladder_drift is ln-scale per frame")
    if not 0 < o["layout_strength"] <= 1 or not 0 < o["layout_nest"] < .2:
        raise ValueError("layout_strength is in (0, 1]; layout_nest is the portrait's width inside the high still")
    if not 1 < o["high_zoom"] < 1.5:
        raise ValueError("high_zoom is the pull-back over the moving guide; keep it a few percent")
    if o["high_join"] == "cloud":
        if not BOUNDS[1] < o["opening_frames"] < BOUNDS[2]:
            raise ValueError("The cloud join's opening must end inside the cloud")
        if not BOUNDS[2] <= o["high_delivery_frame"] < BOUNDS[3]:
            raise ValueError("The high still belongs just below the cloud, in the descent's high aerial")
    elif o["high_join"] == "parent":
        if not BOUNDS[2] <= o["opening_frames"]:
            raise ValueError("The parent join's opening must end below the cloud")
        if not BOUNDS[2] < o["high_delivery_frame"] < o["opening_frames"]:
            raise ValueError("The high still belongs just below the cloud, inside the kept opening")
    else:
        raise ValueError("high_join is parent or cloud")
    if not 0 < o["anchor_strength"] < 1:
        raise ValueError("A full-strength anchor stalls the camera; use a strength below 1")
    o.setdefault("end_face_ratio", FACE_STOP.get(cfg.get("video", {}).get("framing")))
    return o


# Face height / frame height at which the push-in has reached each requested crop. LTX does not stop a
# push where the prompt says, so delivery ends the portrait at the first frame that reaches the crop.
FACE_STOP = {"full": .105, "three-quarter": .15, "medium": .21, "close": .36}
# Counted from the portrait seam. The portrait seam sits late in the native clip (Hagia Sophia: native 24
# of 73), and the requested crop arrives about 7 frames after it, so a longer minimum overshoots.
MIN_NATIVE_FRAMES = 6


def face_ratios(video, first=0):
    """YuNet face height over frame height for each frame from `first`, None where no centred face is found."""
    from touchdown.face_refine import _detector, _face
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    ratios, detector = [], None
    while True:
        ok, image = cap.read()
        if not ok:
            break
        detector = detector or _detector((image.shape[1], image.shape[0]))
        face = _face(detector, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ratios.append(None if face is None else round(float(face[3]) / image.shape[0], 4))
    cap.release()
    return ratios


def ending_frame(ratios, stop, first, minimum=MIN_NATIVE_FRAMES):
    """Last native frame to deliver: the first at or past `first+minimum` whose face reaches `stop`."""
    if stop is None:
        return len(ratios) - 1
    for k in range(first + minimum, len(ratios)):
        if ratios[k] is not None and ratios[k] >= stop:
            return k
    return len(ratios) - 1


def _native_time(pacing, k):
    return (pacing["native_start_frame"] + k - pacing["native_first_frame"]) / FPS


def feet_visible_until(pacing):
    """Delivered time after which the push has cropped the feet (the face passes the full-body ratio)."""
    ending = pacing.get("ending") or {}
    ratios, n = ending.get("face_ratios") or [], pacing.get("native_first_frame", 0)
    for k in range(n, ending.get("native_last_frame", -1) + 1):
        if ratios[k] is not None and ratios[k] >= FACE_STOP["full"]:
            return _native_time(pacing, k)
    return None


def measured_crop(pacing, framing):
    """The requested crop, judged by face height on the delivered last frame rather than by the small vision model."""
    ending = pacing.get("ending") or {}
    stop, last = ending.get("stop_face_ratio"), ending.get("native_last_frame")
    if stop is None or last is None or ending["face_ratios"][last] is None:
        return None
    ratio = ending["face_ratios"][last]
    order = list(FACE_STOP)
    closer = order[order.index(framing) + 1] if order.index(framing) + 1 < len(order) else None
    ok = ratio >= stop and (closer is None or ratio < FACE_STOP[closer])
    return dict(status="pass" if ok else "fail", evidence=f"Last-frame face height is {ratio:.3f} of the frame; "
                f"{framing} needs {stop:.3f}" + (f" to {FACE_STOP[closer]:.3f}." if closer else " or more."))


def parent_frames(o):
    return BOUNDS[-1] + 1 + o["parent_handles"]


def guide_times(frame, count):
    """Pixel-time centres of the latent planes a frame-0 guide appends: first frame, then 8-frame groups."""
    if count == 1:
        return [frame]
    if frame == 0:
        if count % 8 != 1:
            raise ValueError("A guide at frame 0 must have 8n+1 frames")
        return [0] + [8*i - 3.5 for i in range(1, (count-1)//8 + 1)]
    # Mid-clip, LTXVAddGuide prepends the first image and drops that latent: N images (N = 8k) at frame 8j+1
    # become N/8 planes covering frame..frame+N-1.
    if frame % 8 != 1 or count % 8:
        raise ValueError("A mid-clip video guide needs 8k frames starting at frame 8j+1")
    return [frame + 8*i + 3.5 for i in range(count//8)]


def words(cfg):
    w = dict(place=cfg.get("place", "the location"), region="the surrounding region",
             pattern="a broad regional mosaic of connected terrain, vegetation, small roads and drainage",
             palette="natural", light="Soft natural daylight", landscape="", support="the ground",
             appearance="the same person")
    w.update({k: v for k, v in cfg.get("words", {}).items() if v})
    return w


def _global(w):
    return (f"One uninterrupted live-action photographic camera move. Natural muted {w['palette']} land, "
            f"{w['light']}, consistent illumination and exposure. Real irregular surface detail, atmospheric "
            "depth and one fixed rectilinear 35 mm lens with no fisheye or barrel distortion. Maintain one heading "
            "and continuous movement at every moment. "
            "Physical perspective and parallax, no illustration, no CGI, no cuts, wipes, dissolves or pauses.")


def _retreat_texts(cfg, final_high):
    """Retreat prompts per beat. Only the approach names the place; wider views never show it large."""
    w = words(cfg)
    surroundings = f" {w['landscape']}" if w["landscape"] else ""
    return dict(
        settle=("Begin with the exact supplied moving photographic portrait, already moving. Retreat smoothly "
                "away from the same person along the supplied reversed camera motion, BACKWARD AND UPWARD, accelerating. Preserve the "
                f"person, clothes, natural light and the same {w['support']} beneath them, with natural body and "
                "fabric motion. Never replace the moving portrait with a still."),
        approach=(f"Keep retreating BACKWARD AND UPWARD over the SAME connected ground at {w['place']}. The person "
                  "shrinks into a tiny figure on the same spot while the landmark stays large, sharp and in the same "
                  f"place in its wider setting, never moving to a new platform.{surroundings} Strong parallax, no "
                  "hovering or pause. By the END of this interval reach a broad high oblique aerial."),
        high_aerial=(f"Climb fast at about 45 degrees, about two kilometres above {w['pattern']}. The landmark "
                     "becomes microscopic and people cannot be resolved. " + final_high),
        clouds=("Continue ascending rapidly into a deep substantial cloud layer. Billowing rounded cloud masses "
                "with illuminated tops and shaded recesses sweep close past the lens. In the middle of this interval "
                "opaque white cloud fills the WHOLE frame: no ground, roads or rivers show through it. Then the camera "
                "breaks out above the cloud tops. Always move through real depth; never a transparent veil over the "
                "ground, no warping or stretching landscape, no drawn streaks."),
        orbit=(f"The retreat finishes close above the broad curved horizon of Earth over {w['region']}. A thin "
               "blue atmospheric limb and a narrow band of black space above. Thick white weather systems cover "
               "broad regional terrain. Keep physically moving above the cloud tops. No individual structures "
               "or people are visible."))


def _timing(name, frames, global_prompt, beats):
    total = 0
    for beat in beats:
        if beat["start_frame"] != total or beat["end_frame"] <= beat["start_frame"]:
            raise ValueError("Beats must tile the generation clock")
        total = beat["end_frame"]
    if total != frames:
        raise ValueError("Beats must cover every generated frame")
    return dict(version=name, fps=FPS, source_frames=frames, global_prompt=global_prompt,
                generation_beats=beats, single_prompt=global_prompt + " " + " ".join(b["prompt"] for b in beats),
                epsilon=.1, temporal_stride=8)


def parent_timing(cfg, o):
    """Retreat clock: generation frame g is delivery frame 192-g; frames past 192 are discarded orbit."""
    texts = _retreat_texts(cfg, "Keep the region clearly visible BELOW the clouds; only at the END of this "
                                "interval reach the underside of the cloud layer and begin entering it.")
    names = ("orbit", "clouds", "high_aerial", "approach", "settle")
    beats, cursor = [], 0
    for i in reversed(range(len(names))):
        length = BOUNDS[i+1] - BOUNDS[i] + (names[i] == "settle") + (o["parent_handles"] if names[i] == "orbit" else 0)
        beats.append(dict(name=names[i], start_frame=cursor, end_frame=cursor+length, prompt=texts[names[i]]))
        cursor += length
    return _timing("split-parent-v2", parent_frames(o), _global(words(cfg)), beats)


def descent_timing(cfg, o):
    """The parent's beats from the portrait to the kept opening in the cloud, on the descent's own clock."""
    texts = _retreat_texts(cfg, "Pass exactly through the supplied high aerial photograph, still climbing. "
                                "Only at the END of this interval reach the underside of the cloud layer."
                           if o["high_join"] == "cloud" else
                           "Finish exactly in the supplied high aerial photograph, still climbing. "
                           "No cloud passage, no orbit or space in this part.")
    texts["clouds"] = ("Climb straight up into a deep substantial cloud layer. Billowing rounded cloud masses sweep "
                       "close past the lens until white cloud fills the whole view. The camera stays inside the "
                       "cloud; no orbit, space or cloud tops in this part.")
    n = o["descent_frames"]
    beats, cursor = [], 0
    for name, lo, hi in (("settle", BOUNDS[4], BOUNDS[5] + 1), ("approach", BOUNDS[3], BOUNDS[4]),
                         ("high_aerial", BOUNDS[2], BOUNDS[3]), ("clouds", BOUNDS[1], BOUNDS[2])):
        length = min(hi - lo, n - cursor)
        if length <= 0:
            break
        beats.append(dict(name=name, start_frame=cursor, end_frame=cursor+length, prompt=texts[name]))
        cursor += length
    return _timing("split-descent-v2", n, _global(words(cfg)), beats)


def enable_prompt_relay():
    """Install the pinned Prompt Relay node for this ComfyUI launch; call before the server starts."""
    from touchdown.relay_experiment import REPO, RUNTIME, write
    node = RUNTIME / "custom_nodes/TouchdownPromptRelay"
    upstream_file = node / "upstream.json"
    if not upstream_file.is_file():
        raise ValueError(f"The pinned Prompt Relay source is missing: {upstream_file}")
    upstream = json.loads(upstream_file.read_text())
    for name, meta in upstream["files"].items():
        if digest(node / name) != meta["sha256"]:
            raise ValueError("Changed pinned Prompt Relay source")
    shutil.copyfile(REPO / "scripts/touchdown/relay_node.py", node / "__init__.py")
    config = RUNTIME / "extra-paths.json"
    write(config, dict(touchdown_relay=dict(custom_nodes=str(node.parent))))
    flag = " --extra-model-paths-config " + shlex.quote(str(config))
    if flag not in os.environ["TOUCHDOWN_COMFY_SERVER_CMD"]:
        os.environ["TOUCHDOWN_COMFY_SERVER_CMD"] += flag
    return upstream


def set_strength(graph, frame, strength):
    nodes = [k for k, v in graph.items()
             if v.get("class_type") == "LTXVAddGuide" and v["inputs"].get("frame_idx") == frame]
    if not nodes:
        raise ValueError(f"No guide at frame {frame}")
    for k in nodes:
        graph[k]["inputs"]["strength"] = strength
    return sorted(nodes, key=int)


SEAM_FRAMES = 8


def _clip(source, target, start, count):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf",
                    f"trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(target)], check=True)


def _reversed_clip(source, target, count):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf",
                    f"trim=end_frame={count},reverse,setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(target)], check=True)


def high_motion_start(o):
    """Generation frame where the high view's moving guide starts; it ends on the pinned high frame."""
    return BOUNDS[-1] - o["high_delivery_frame"] - o["high_motion_frames"] + 1


def high_motion(still, target, frames, zoom, width, height):
    """A pull-back of `frames` over the high still, in generation (retreat) order: the window grows by a
    constant factor per frame from 1/zoom of the still to the whole still, so played forwards it is a steady
    zoom in. A still conditions one latent plane and is only a soft prior at strength below 1 (Chambord
    aside, 25 Oct: 2 high views in 11 places); frames past the first latent of a video guide are written in
    place, so the guide carries the camera's motion, as the parent's cloud frames do at the join."""
    image = cv2.imread(str(still))
    h, w = image.shape[:2]
    aspect = width / height
    full_w, full_h = (w, w / aspect) if w / h <= aspect else (h * aspect, h)
    out = []
    for i in range(frames):
        window = zoom ** (i / (frames - 1) - 1)
        cw, ch = full_w * window, full_h * window
        x0, y0 = (w - cw) / 2, (h - ch) / 2
        matrix = np.array([[width / cw, 0, -x0 * width / cw], [0, height / ch, -y0 * height / ch]], np.float32)
        out.append(cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_AREA))
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an",
                                "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(target)], stdin=subprocess.PIPE)
    for frame in out:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError("Could not encode the high view's moving guide")
    return out


def _cloud(frame, i, rng_seed=11):
    """Bright, slowly drifting cloud texture (no ground) for the proxy's whiteout."""
    h, w = frame.shape[:2]
    rng = np.random.default_rng(rng_seed)
    base = cv2.resize(rng.random((h // 32 + 4, w // 32 + 4)).astype(np.float32), (w + 128, h + 128), interpolation=cv2.INTER_CUBIC)
    shift = i % 128
    tex = base[shift // 2:shift // 2 + h, shift:shift + w]
    tex = 215 + 40 * (tex - tex.min()) / max(1e-6, float(tex.max() - tex.min()))
    return np.dstack([tex, tex, np.minimum(255, tex + 6)]).clip(0, 255).astype(np.uint8)


def _window(image, scale, width, height):
    """The centred `scale` part of `image` (fitted to the output aspect), resized to the output. Above 1 the
    edges are mirrored, so a pull-back past the whole still keeps moving instead of smearing."""
    h, w = image.shape[:2]
    aspect = width / height
    full_w, full_h = (w, w / aspect) if w / h <= aspect else (h * aspect, h)
    cw, ch = full_w * scale, full_h * scale
    x0, y0 = (w - cw) / 2, (h - ch) / 2
    matrix = np.array([[width / cw, 0, -x0 * width / cw], [0, height / ch, -y0 * height / ch]], np.float32)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REFLECT_101 if scale > 1 else cv2.BORDER_REPLICATE)


def proxy_window(o, count, bottom=None, low=None):
    """ln of the proxy's window on the high still at each generation frame from the prefix's end to the cloud.

    It never stands still. Over the moving high guide it matches that guide exactly (constant factor per frame,
    ending on the whole still); above it the pull-back carries on at the same rate into the cloud; below it the
    rate swells smoothly from the guide's rate into the swoop and back, so the drop from the nested portrait to
    the high view is one continuous zoom. Eilean Donan e1 (26 Oct): the parent itself held 20 frames past the
    end of its high guide and iris-wiped to the ground, because nothing told it to keep flying.

    By default the swoop ends at the prefix on the nested portrait (`layout_nest`). With a ladder it ends at
    `bottom` on ln-window `low` (the lowest rung filling the frame) and drifts on into that rung below it."""
    high_g = BOUNDS[-1] - o["high_delivery_frame"]
    if o["high_motion_frames"]:
        top, lift = high_motion_start(o), float(np.log(o["high_zoom"]))
        rate = lift / (high_g - top)
    else:
        top, lift, rate = high_g, 0.0, .0097
    n = BOUNDS[-1] - BOUNDS[1]
    bottom = count - 1 if bottom is None else bottom
    low = float(np.log(o["layout_nest"])) if low is None else low
    span = top - bottom
    swell = (-lift - low - span * rate) * np.pi / (2 * span)
    if swell <= 0:
        raise ValueError("The nested portrait must be deeper in the high still than the high guide's zoom")
    out = {}
    for g in range(count, n + 1):
        k = g - bottom
        out[g] = (low - o["ladder_drift"] * (bottom - g) if g < bottom
                  else low + rate * k + swell * span / np.pi * (1 - np.cos(np.pi * k / span)) if g <= top
                  else -lift + rate * (g - top))
    return out


def layout_proxy(prefix_frames, high, orbit, target, o):
    """A crude fly-through in generation (retreat) order for Layout To Render: the reversed portrait prefix,
    then a pull-back from its last frame nested at the centre of the high still (proxy_window: constant zoom
    factor per frame reads as constant speed, and it never stops), on past the whole high still into a
    drifting whiteout that hides the swap, and a slow pull-back over the orbit still. LTX follows the proxy's
    camera path and repaints the look; any hard swap it would copy sits inside the opaque cloud. Returns the
    frames written."""
    width, height = o["width"], o["height"]
    n, count = parent_frames(o), len(prefix_frames)
    cloud_in, cloud_full = BOUNDS[-1] - BOUNDS[2] + 2, BOUNDS[-1] - 50
    cloud_out, orbit_g = BOUNDS[-1] - 36, BOUNDS[-1] - BOUNDS[1]
    windows = proxy_window(o, count)
    inner = cv2.resize(prefix_frames[-1], (width, height), interpolation=cv2.INTER_AREA)
    # Chillon c1 (26 Oct): a rectangular feather (and its own sky and horizon) came out as a literal growing
    # picture-in-picture. The portrait now melts in as a soft round blob with no edges, and becomes the whole
    # frame only as it fills it.
    y, x = np.mgrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    radius = np.hypot(x, y).astype(np.float32)
    r, out = o["layout_nest"], []
    for g in range(n):
        if g < count:
            frame = cv2.resize(prefix_frames[g], (width, height), interpolation=cv2.INTER_AREA)
        elif g < cloud_out:
            window = float(np.exp(windows[g]))
            frame = _window(high, window, width, height).astype(np.float32)
            size = r / window
            iw, ih = int(width * size), int(height * size)
            if iw > 2 and ih > 2:
                x0, y0 = (width - iw) // 2, (height - ih) // 2
                small = cv2.resize(inner, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32)
                # Bodiam b1: blob + (1 - blob) * size ** 2 was already a near-rectangle at 15-60 % of the frame.
                # The blob's radius now shrinks only as the portrait nears full frame, so it stays round until then
                # and reaches the corners exactly when it fills the frame.
                reach = np.minimum(1.0, cv2.resize(radius, (iw, ih)) * (1 - min(1.0, size) ** 4))
                mask = (np.cos(reach * np.pi / 2) ** 2)[..., None]
                frame[y0:y0 + ih, x0:x0 + iw] = small * mask + frame[y0:y0 + ih, x0:x0 + iw] * (1 - mask)
            cover = 0 if g < cloud_in else min(1.0, (g - cloud_in) / max(1, cloud_full - cloud_in))
            frame = (frame * (1 - cover) + _cloud(frame, g) * cover).clip(0, 255).astype(np.uint8)
        else:
            pull = min(1.0, (g - cloud_out) / max(1, n - 1 - cloud_out))
            frame = _window(orbit, 1 / 1.25 ** (1 - pull), width, height)
            cover = max(0.0, 1 - (g - cloud_out) / max(1, orbit_g - cloud_out))
            frame = (frame * (1 - cover) + _cloud(frame, g) * cover).astype(np.uint8)
        out.append(frame)
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an",
                                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(target)], stdin=subprocess.PIPE)
    for frame in out:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError("Could not encode the layout proxy")
    return out


def _encode(frames, target, width, height, what):
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an",
                                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(target)], stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"Could not encode the {what}")


def rung_prompt(cfg):
    from touchdown.studio_brief import LENS
    w = words(cfg)
    return (f"Create ONE live-action NADIR drone photograph of {w['place']}: the camera about 30 m directly above "
            "the ground shown in image 1, pointing STRAIGHT DOWN (90 degrees, top-down), so the landmark from image 1 "
            "and image 2 is seen from above as a roof plan with its grounds, paths and nearby land around it. The "
            "top of the frame faces the way image 1 looks. No sky, no horizon, no side view. "
            f"{w['light']}. REMOVE every person. One camera view filling the entire image: no montage, grid, inset, "
            f"border or text. No illustration, map, miniature model or CGI. {LENS}")


def outpaint_prompt(cfg, altitude):
    from touchdown.studio_brief import LENS
    w = words(cfg)
    # Bran ladder check (9000d74): naming the place here made FLUX add more palaces in the new ring.
    return (f"ONE live-action NADIR aerial photograph of {w['region']}, taken from about {altitude} m straight up, "
            "pointing straight down like a satellite orthophoto: every roof, tree and hill seen from directly above, "
            "never side-on, with no slopes or ridgelines seen from the side and no haze band. "
            f"{w['pattern']}, at one uniform scale across the whole frame, the ground "
            "continuing seamlessly out from the middle in every direction. The only landmark building is the one "
            "already at the very centre: around it only ordinary houses, roads, fields, woods and water, no other "
            "castle, palace, church or monument. No sky, no horizon, no seam, frame, border, inset or vignette. "
            f"No person, no text, no map or CGI. {LENS}")


# Where FLUX repaints around the pasted rung, as a fraction of the pasted picture's size: the outer band of the
# picture is repainted too, so the new land grows out of it rather than meeting it at an edge.
NEST_BAND = .08
NEST_FEATHER = .06


def _feather(width, height, fraction=.06):
    """1 in the middle, easing to 0 over `fraction` of each side's length at the edges."""
    mask = np.zeros((height, width), np.float32)
    bx, by = max(2, int(width * fraction)), max(2, int(height * fraction))
    mask[by:-by, bx:-bx] = 1
    return cv2.GaussianBlur(mask, (0, 0), min(bx, by) / 3)[..., None]


def outpaint_canvas(previous):
    """Rung i's inpainting input (BGRA): rung i - 1 at half size in the centre, neutral grain around it, and alpha 0
    (repaint) everywhere but the half's core. FLUX then keeps the centre where it is, so every rung zooms out
    exactly 2x and keeps the landmark. Bran b1: an unmasked FLUX edit redrew the whole picture (a different
    castle, barely zoomed out) and turned a mirrored placeholder into a kaleidoscope."""
    h, w = previous.shape[:2]
    rng = np.random.default_rng(0)
    canvas = np.full((h, w, 4), 0, np.uint8)
    canvas[..., :3] = (int(previous.mean()) + rng.normal(0, 12, (h, w, 3))).clip(0, 255).astype(np.uint8)
    top, left = h // 4, w // 4
    canvas[top:top + h // 2, left:left + w // 2, :3] = cv2.resize(previous, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    by, bx = int(h // 2 * NEST_BAND), int(w // 2 * NEST_BAND)
    canvas[top + by:top + h // 2 - by, left + bx:left + w // 2 - bx, 3] = 255
    return canvas


def match_colour(painted, previous):
    """Shift the outpaint's Lab mean and spread onto the rung below's (Pena p1: FLUX painted dry brown hills
    around a green forest core, and the core read as a green square)."""
    src = cv2.cvtColor(painted, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(previous, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_s, std_s = src.reshape(-1, 3).mean(0), src.reshape(-1, 3).std(0) + 1e-3
    mean_r, std_r = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0) + 1e-3
    out = (src - mean_s) * (std_r / std_s) + mean_r
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def nest_back(painted, previous, fraction=NEST_FEATHER):
    """Paste rung i - 1 back over the centre of the outpaint with a feathered edge inside the repainted band:
    the rung's core is the rung below pixel for pixel."""
    h, w = previous.shape[:2]
    painted = cv2.resize(painted, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    inner = cv2.resize(previous, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = _feather(w // 2, h // 2, fraction + NEST_BAND / 2)
    y0, x0 = h // 4, w // 4
    region = painted[y0:y0 + h // 2, x0:x0 + w // 2]
    painted[y0:y0 + h // 2, x0:x0 + w // 2] = inner * mask + region * (1 - mask)
    return painted.clip(0, 255).astype(np.uint8)


def ladder(cfg, scene, folder, comfy_url, o, ground_frame):
    """The zoom ladder: a low drone still rising from the portrait's ground view (rung 0), then each rung
    FLUX-inpainted around the one below at twice the coverage (and twice the height), with the rung below kept
    at its core, up to roughly the high view's scale. A zoom through the rungs then matches at every scale.

    Chillon, Bodiam and Caerphilly (26 Oct): nesting the ground-level portrait in a magnified high still came out
    as a literal picture-in-picture over a blurred, map-like town, at every strength and mask shape tried,
    because nothing showed the ground between 2 km and 50 m. Returns the rung paths, lowest first."""
    rungs_dir = folder / "ladder"; rungs_dir.mkdir(parents=True, exist_ok=True)
    # Hluboká h1: a 60 m, 45-degree rung 0 drawn from the arrival was a different viewpoint from the portrait,
    # and the descent dissolved (the parent cut) from it to eye level. Rung 0 now rises from the prefix's last
    # (widest) ground frame on the same heading, looking straight down, so the last move to her is a crane down
    # and a tilt up.
    ground = rungs_dir / "ground.png"
    cv2.imwrite(str(ground), cv2.resize(ground_frame, (2560, 1472), interpolation=cv2.INTER_CUBIC))
    spec = dict(name="rung0", references=[str(ground), str(scene / "arrival.png")], prompt=rung_prompt(cfg),
                models=cfg["models"], seed=cfg["seed"] + 601, width=2560, height=1472)
    paths = [generate_asset(comfy_url, spec, rungs_dir)]
    for i in range(1, o["layout_rungs"] + 1):
        previous = cv2.imread(str(paths[-1]))
        canvas = rungs_dir / f"canvas{i}.png"
        cv2.imwrite(str(canvas), outpaint_canvas(previous))
        # Bran ladder check: with the rung below as a reference Klein redrew it at full size around the kept core
        # (a Droste stack of boxes), and an eye-level rung only widens, it never climbs. Rungs are nadir views,
        # where 2x coverage is 2x height, and the border is inpainted from the prompt and the kept core alone.
        spec = dict(name=f"rung{i}-painted", references=[], inpaint=str(canvas),
                    prompt=outpaint_prompt(cfg, 30 * 2 ** i), models=cfg["models"], seed=cfg["seed"] + 601 + i,
                    width=2560, height=1472)
        path = rungs_dir / f"rung{i}.png"
        painted = cv2.resize(cv2.imread(str(generate_asset(comfy_url, spec, rungs_dir))), previous.shape[1::-1],
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), nest_back(match_colour(painted, previous), previous))
        paths.append(path)
    (rungs_dir / "ladder.json").write_text(json.dumps([dict(path=str(p), sha256=digest(p)) for p in paths], indent=2))
    return paths


def ladder_frame(rungs, log_window, width, height):
    """One proxy frame at ln-window `log_window` on the top rung: the smallest rung that covers it, with the rung
    below pasted back at its centre for detail (the two agree there, so no edge shows)."""
    top = len(rungs) - 1
    if log_window >= 0:
        return _window(rungs[top], float(np.exp(log_window)), width, height)
    j = max(0, top - int(np.floor(-log_window / np.log(2))))
    local = float(np.exp(log_window)) * 2 ** (top - j)
    frame = _window(rungs[j], local, width, height).astype(np.float32)
    if j > 0:
        size = .5 / local
        iw, ih = int(round(width * size)), int(round(height * size))
        x0, y0 = (width - iw) // 2, (height - ih) // 2
        inner = cv2.resize(rungs[j - 1], (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32)
        mask = _feather(iw, ih, NEST_FEATHER + NEST_BAND / 2)
        frame[y0:y0 + ih, x0:x0 + iw] = inner * mask + frame[y0:y0 + ih, x0:x0 + iw] * (1 - mask)
    return frame.clip(0, 255).astype(np.uint8)


# The ladder's camera: rung 0 is a nadir view from RUNG0_HEIGHT through the 35 mm lens every still asks for.
LADDER_FOV = 2 * np.arctan(36 / 70)
RUNG0_HEIGHT = 30.0
SKY = np.array((232, 214, 196), np.float32)  # pale haze blue (BGR)


def ground_view(rungs, height_m, pitch_deg, back_m, width, height):
    """Render the ladder as a flat textured ground seen by a camera `height_m` up, `back_m` behind the rungs'
    centre, looking forward (towards the top of the rungs) and `pitch_deg` down (90 is nadir). Each ground point
    is taken from the finest rung that covers it; rays at or above the horizon are haze sky."""
    f = width / 2 / np.tan(LADDER_FOV / 2)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    a, b = (xx - width / 2) / f, -(yy - height / 2) / f
    p = np.radians(pitch_deg)
    dy, dz = np.cos(p) + b * np.sin(p), -np.sin(p) + b * np.cos(p)
    ground = dz < -1e-3
    t = np.where(ground, height_m / np.maximum(-dz, 1e-3), 0)
    x, y = t * a, -back_m + t * dy
    rh, rw = rungs[0].shape[:2]
    half = RUNG0_HEIGHT * np.tan(LADDER_FOV / 2)
    out = np.broadcast_to(SKY, (height, width, 3)).copy()
    for i in range(len(rungs) - 1, -1, -1):
        ex = half * 2 ** i; ey = ex * rh / rw
        u = ((x / ex + 1) / 2 * rw).astype(np.float32); v = ((1 - y / ey) / 2 * rh).astype(np.float32)
        inside = ground if i == len(rungs) - 1 else ground & (np.abs(x) < ex * .97) & (np.abs(y) < ey * .97)
        sample = cv2.remap(rungs[i], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        out[inside] = sample[inside]
    # Distance haze, so the far ground meets the sky without a line.
    distance = np.hypot(x, y + back_m)
    fog = np.where(ground, 1 - np.exp(-distance / (half * 2 ** len(rungs))), 1.0)[..., None]
    return (out * (1 - fog) + SKY * fog).clip(0, 255).astype(np.uint8)


def crane(k, n):
    """Height (m), pitch (degrees) and setback (m) of the crane from her eye-level view (k = 0) up to the nadir
    of rung 0 (k = n), eased at both ends."""
    s = k / n
    e = s * s * (3 - 2 * s)
    return 1.7 * (RUNG0_HEIGHT / 1.7) ** e, 6 + 84 * e, 15 * (1 - e)


def ladder_proxy(prefix_frames, rungs, orbit, target, o):
    """layout_proxy on the ladder: the prefix, then a crane up from her view to the nadir of rung 0 at the anchor
    frame, the swoop up the ladder to the top rung (which the high guide is cut from), on into the cloud and the orbit."""
    width, height = o["width"], o["height"]
    n, count = parent_frames(o), len(prefix_frames)
    cloud_in, cloud_full = BOUNDS[-1] - BOUNDS[2] + 2, BOUNDS[-1] - 50
    cloud_out, orbit_g = BOUNDS[-1] - 36, BOUNDS[-1] - BOUNDS[1]
    anchor = BOUNDS[-1] - o["anchor_delivery_frame"]
    windows = proxy_window(o, count, bottom=anchor, low=-(len(rungs) - 1) * float(np.log(2)))
    last = cv2.resize(prefix_frames[-1], (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for g in range(n):
        if g < count:
            frame = cv2.resize(prefix_frames[g], (width, height), interpolation=cv2.INTER_AREA)
        elif g < anchor:
            # Pena p1: from the nadir drone view the parent cut, and the descent irised, to her eye level. The
            # proxy now cranes: down from the nadir of rung 0, tilting up, onto her view, rendered from the rungs
            # as ground, and settles into the prefix's last frame over its first 8 frames.
            frame = ground_view(rungs, *crane(g - count + 1, anchor - count + 1), width, height).astype(np.float32)
            settle = max(0.0, 1 - (g - count + 1) / 9)
            frame = (last * settle + frame * (1 - settle)).clip(0, 255).astype(np.uint8)
        elif g < cloud_out:
            frame = ladder_frame(rungs, windows[g], width, height)
            cover = 0 if g < cloud_in else min(1.0, (g - cloud_in) / max(1, cloud_full - cloud_in))
            frame = (frame * (1 - cover) + _cloud(frame, g) * cover).clip(0, 255).astype(np.uint8)
        else:
            pull = min(1.0, (g - cloud_out) / max(1, n - 1 - cloud_out))
            frame = _window(orbit, 1 / 1.25 ** (1 - pull), width, height)
            cover = max(0.0, 1 - (g - cloud_out) / max(1, orbit_g - cloud_out))
            frame = (frame * (1 - cover) + _cloud(frame, g) * cover).astype(np.uint8)
        out.append(frame)
    _encode(out, target, width, height, "ladder proxy")
    return out


def _read_frames(video):
    cap = cv2.VideoCapture(str(video)); frames = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                return frames
            frames.append(image)
    finally:
        cap.release()


def _generate_layout(client, graph, frames, pins, strengths, timing, seed, prefix, folder, o, look):
    """The parent on Layout To Render: the proxy at graph node "60" is the IC-LoRA layout guide, `look` the
    photographic reference; pins and strengths as in the plain parent. Prompt Relay is not wired: its
    clock assumes full-size guide planes, and the layout carries the timing."""
    from geoblock_core.workflow import LTX25_MODELS, LTX_VIDEO, _add_ltx25
    import fakeout_earth_zoom as f
    request = SimpleNamespace(ltx25_models=dict(LTX25_MODELS), pin_last_keyframe=False, preset="fake-out",
                              seed=seed, save_stage1=False, ltx_upscale=False, output_prefix=prefix,
                              ltx_encoder_device="cpu")
    _add_ltx25(graph, request, frames, look, pins[-1][1], pins, timing["single_prompt"], layout=True)
    for node in graph.values():
        if node["class_type"] == "LTXAddVideoICLoRAGuide":
            node["inputs"]["strength"] = o["layout_strength"]
    nodes = {frame: set_strength(graph, frame, s) for frame, s in strengths.items()}
    graph["308"]["inputs"].update(width=o["width"], height=o["height"])
    return f._submit(client, graph, (LTX_VIDEO,), folder), nodes


def _record(saved, f, **extra):
    video = Path(saved[f.LTX_VIDEO]); workflow = video.parent / "graph.json"
    return dict(video=str(video), sha256=digest(video), workflow=str(workflow),
                workflow_sha256=digest(workflow), **extra)


def _checked(record):
    if digest(Path(record["video"])) != record["sha256"] or digest(Path(record["workflow"])) != record["workflow_sha256"]:
        raise ValueError("A photographic flight segment changed")
    return record


def _generate(f, client, graph, frames, pins, strengths, timing, guides, seed, prefix, folder, o):
    wanted = f._ltx(graph, SimpleNamespace(no_upscale=True), frames, pins, timing["single_prompt"], seed, prefix)
    nodes = {frame: set_strength(graph, frame, s) for frame, s in strengths.items()}
    graph["308"]["inputs"].update(width=o["width"], height=o["height"])
    if any(n["class_type"] in {"LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide"} for n in graph.values()):
        raise ValueError("3D/layout conditioning is forbidden")
    wire_relay(graph, timing, folder / "attention-audit.json", guides, enabled=True)
    return f._submit(client, graph, tuple(wanted), folder), nodes


def _inputs(cfg, scene, output, near_folder):
    ancestry = verify_lineage(cfg, scene, asset_plan(cfg, scene))
    missing = {"arrival", "orbit"} - set(ancestry["generated"])
    if missing:
        raise ValueError("Generate and verify these photographs first: " + ", ".join(sorted(missing)))
    require_arrival_gate(cfg, scene, output)
    near = json.loads((near_folder / "result.json").read_text())
    prior = json.loads((near_folder / "provenance.json").read_text())
    if (digest(Path(near["video"])) != near["sha256"] or digest(Path(near["workflow"])) != near["workflow_sha256"]
            or prior["parent"]["arrival_sha256"] != digest(scene / "arrival.png")):
        raise ValueError("The moving portrait or its parent changed")
    return ancestry, near


def anchor_prompt(cfg):
    w = words(cfg)
    return ("Edit image 1 into ONE single full-frame live-action aerial photograph. KEEP its exact camera position, "
            "perspective, heading, composition, landmark footprint and scale, paths and ground. Repair any blur, "
            "ghosting, double exposure or smeared structure into one physically solid scene with sharp photographic "
            f"surface detail. Use image 2 ONLY as a visual reference for {w['place']}: the same landmark's shapes, "
            "materials and natural light. Image 2 must NOT appear as another picture or panel. REMOVE every person, "
            "filling the place with the same ground; the view shows no person at all. Output exactly one camera view filling the entire image: no montage, grid, "
            "multiple views, split-screen or borders. No illustration, miniature model, CGI or text.")


def parent(cfg, scene, output, url, comfy_url, near_folder):
    """Generate the parent flight, take far.png and the repaired anchor from it. Comfy runtime only."""
    import fakeout_earth_zoom as f
    o = options(cfg)
    ancestry, near = _inputs(cfg, scene, output, near_folder)
    folder = output / "flight"
    part = folder / "parent"; part.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"] + int(cfg["video"].get("flight_seed_offset", 32))
    # The prefix must fit inside the settle beat. A 73-frame prefix pinned the portrait for three
    # seconds, so the parent rushed portrait to orbit in the remaining three (Teotihuacan split4).
    count = o["prefix_frames"]
    timing = parent_timing(cfg, o); (part / "timing.json").write_text(json.dumps(timing, indent=2))
    prefix = part / "near-retreat-prefix.mp4"
    _reversed_clip(near["video"], prefix, count)
    frames = parent_frames(o)
    with f._client(url) as client:
        graph = {"50": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(prefix))}},
                 "51": {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}}
        pins, strengths, guides = [(0, ["51", 0])], {}, guide_times(0, count)
        rungs = (ladder(cfg, scene, part, comfy_url, o, _read_frames(prefix)[-1])
                 if o["parent_layout"] and o["layout_rungs"] else None)
        if o["high_join"] == "parent" and o["high_motion_frames"]:
            high_frame = high_motion_start(o)
            motion = part / "high-motion.mp4"
            # With a ladder the high guide is cut from its top rung, so the guide and the proxy are one picture.
            high_motion(rungs[-1] if rungs else scene / "high.png", motion, o["high_motion_frames"], o["high_zoom"],
                        o["width"], o["height"])
            graph.update({"47": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(motion))}},
                          "48": {"class_type": "GetVideoComponents", "inputs": {"video": ["47", 0]}}})
            pins.append((high_frame, ["48", 0]))
            strengths[high_frame] = o["high_strength"]; guides += guide_times(high_frame, o["high_motion_frames"])
        elif o["high_join"] == "parent":
            high_frame = BOUNDS[-1] - o["high_delivery_frame"]
            pins.append((high_frame, f._load(graph, client, "45", scene / "high.png")))
            strengths[high_frame] = o["high_strength"]; guides.append(high_frame)
        pins.append((-1, f._load(graph, client, "46", scene / "orbit.png"))); guides.append(frames - 1)
        if o["parent_layout"]:
            proxy = part / "layout-proxy.mp4"
            if rungs:
                ladder_proxy(_read_frames(prefix), [cv2.imread(str(p)) for p in rungs], cv2.imread(str(scene / "orbit.png")), proxy, o)
            else:
                layout_proxy(_read_frames(prefix), cv2.imread(str(scene / "high.png")), cv2.imread(str(scene / "orbit.png")), proxy, o)
            graph.update({"59": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(proxy))}},
                          "60": {"class_type": "GetVideoComponents", "inputs": {"video": ["59", 0]}}})
            saved, _ = _generate_layout(client, graph, frames, pins, strengths, timing, seed, "touchdown/split_parent",
                                        part, o, f._load(graph, client, "44", scene / "arrival.png"))
        else:
            saved, _ = _generate(f, client, graph, frames, pins, strengths, timing, guides, seed,
                                 "touchdown/split_parent", part, o)
    record = _record(saved, f, prefix_frames=count, prefix_sha256=digest(prefix), high_join=o["high_join"],
                     **({"high_sha256": digest(scene / "high.png"), "high_frame": high_frame,
                         "high_strength": o["high_strength"], "high_motion_frames": o["high_motion_frames"],
                         "high_zoom": o["high_zoom"]} if o["high_join"] == "parent" else {}),
                     **({"layout_proxy": str(part / "layout-proxy.mp4"), "layout_proxy_sha256": digest(part / "layout-proxy.mp4"),
                         "layout_strength": o["layout_strength"], "layout_nest": o["layout_nest"],
                         "ladder": [dict(path=str(p), sha256=digest(p)) for p in rungs or []]} if o["parent_layout"] else {}))
    (part / "result.json").write_text(json.dumps(record, indent=2))
    cap = cv2.VideoCapture(record["video"])
    try:
        if video_info(cap)[0] != frames:
            raise ValueError("Unexpected parent clock")
        if o["high_join"] == "cloud":
            opening, cloud_texture = cloud_opening(lambda d: frame_at(cap, BOUNDS[-1] - d))
        else:
            opening, cloud_texture = o["opening_frames"], None
        cv2.imwrite(str(folder / "far.png"), frame_at(cap, BOUNDS[-1] - opening))
        cv2.imwrite(str(folder / "composition.png"), frame_at(cap, BOUNDS[-1] - o["anchor_delivery_frame"]))
    finally:
        cap.release()
    spec = dict(name="anchor", references=[str(folder / "composition.png"), str(scene / "arrival.png")],
                prompt=anchor_prompt(cfg), models=cfg["models"], seed=cfg["seed"] + 502, width=2560, height=1472)
    generate_asset(comfy_url, spec, folder)
    binding = dict(parent=record, far_sha256=digest(folder / "far.png"), composition_sha256=digest(folder / "composition.png"),
                   anchor_sha256=digest(folder / "anchor.png"), arrival_sha256=digest(scene / "arrival.png"),
                   anchor_delivery_frame=o["anchor_delivery_frame"], photographic_roots=ancestry,
                   opening_frames=opening, cloud_texture=cloud_texture)
    (folder / "anchor-binding.json").write_text(json.dumps(binding, indent=2))
    return binding


def descent(cfg, scene, output, url, near_folder):
    """Generate the anchored descent and deliver the film. Needs the anchor checked first."""
    import fakeout_earth_zoom as f
    _, near = _inputs(cfg, scene, output, near_folder)
    folder = output / "flight"
    binding = json.loads((folder / "anchor-binding.json").read_text())
    o = options(cfg)
    if o["high_join"] == "cloud":
        o = with_opening(o, binding["opening_frames"])
    upper = _checked(binding["parent"])
    for name in ("far", "composition", "anchor"):
        if digest(folder / f"{name}.png") != binding[f"{name}_sha256"]:
            raise ValueError(f"The parent-derived {name} still changed")
    review = json.loads((folder / "anchor-review.json").read_text())
    if review.get("status") != "pass" or review.get("sha256") != binding["anchor_sha256"]:
        raise ValueError("The anchor must pass its review before the descent")
    part = folder / "descent"; part.mkdir(parents=True, exist_ok=True)
    timing = descent_timing(cfg, o); (part / "timing.json").write_text(json.dumps(timing, indent=2))
    prefix = part / "near-retreat-prefix.mp4"
    _reversed_clip(near["video"], prefix, o["prefix_frames"])
    n = o["descent_frames"]
    anchor_frame = BOUNDS[-1] - o["anchor_delivery_frame"]
    seed = cfg["seed"] + int(cfg["video"].get("flight_seed_offset", 32)) + 1
    with f._client(url) as client:
        graph = {"50": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(prefix))}},
                 "51": {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}}
        # The descent ends on the parent's own last 8 frames before the kept opening (inside the cloud), as a moving guide, not on
        # one still: with only far.png LTX arrived at its own speed and cloud cover, and the join hitched or
        # jumped (Barcelona 2.6 s). Descent frames 113..120 are parent generation frames 113..120.
        seam = part / "parent-seam.mp4"
        _clip(upper["video"], seam, n - SEAM_FRAMES, SEAM_FRAMES)
        graph.update({"52": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(seam))}},
                      "53": {"class_type": "GetVideoComponents", "inputs": {"video": ["52", 0]}}})
        if ladder_mode(o):
            pins, guides, strengths, extra = [(0, ["51", 0])], guide_times(0, o["prefix_frames"]), {}, {}
        else:
            pins = [(0, ["51", 0]), (anchor_frame, f._load(graph, client, "45", folder / "anchor.png"))]
            guides = guide_times(0, o["prefix_frames"]) + [anchor_frame]
            strengths, extra = {anchor_frame: o["anchor_strength"]}, {}
        if o["high_join"] == "cloud":
            high_frame = BOUNDS[-1] - o["high_delivery_frame"]
            pins.append((high_frame, f._load(graph, client, "46", scene / "high.png")))
            guides.append(high_frame); strengths[high_frame] = o["high_strength"]
            extra = dict(high_sha256=digest(scene / "high.png"), high_frame=high_frame, high_strength=o["high_strength"])
        if o["seam_carry_frames"]:
            bridge = o["seam_bridge_frames"]
            carry_frame, carry_count = carry_guide(o)
            if bridge and not ladder_mode(o) and carry_frame != anchor_frame + 1:
                raise ValueError("The bridge must end next to the anchor frame")
            carry = part / "seam-carry.mp4"
            if o["seam_carry_source"] == "parent":
                # Eilean Donan e1 (26 Oct): the synthetic bridge nested anchor.png in the seam's crop zoom and LTX
                # drew it literally, a growing picture-in-picture inset that popped to the full anchor. The parent
                # already flew this stretch in one generation, and anchor.png is a repair of its frame at the
                # anchor's delivery time, so its own frames from just below the anchor to the seam are a guide
                # that runs on one camera path into the anchor.
                _clip(upper["video"], carry, carry_frame, carry_count)
                measured = [-v for v in _log_scales(carry, carry_count) if v is not None]
                speed = round(float(np.median(measured)), 4) if measured else None
            else:
                _, speed = seam_carry(seam, carry, o["seam_carry_frames"], o["width"], o["height"], o["seam_carry_accel"],
                                      anchor=folder / "anchor.png", bridge=bridge)
            graph.update({"54": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(carry))}},
                          "55": {"class_type": "GetVideoComponents", "inputs": {"video": ["54", 0]}}})
            pins.append((carry_frame, ["55", 0])); strengths[carry_frame] = o["seam_carry_strength"]
            guides += guide_times(carry_frame, carry_count)
            extra.update(seam_carry_frame=carry_frame, seam_carry_frames=o["seam_carry_frames"],
                         seam_carry_strength=o["seam_carry_strength"], seam_carry_speed=speed,
                         seam_carry_accel=o["seam_carry_accel"], seam_bridge_frames=bridge,
                         seam_carry_source=o["seam_carry_source"],
                         seam_carry_sha256=digest(carry))
        pins.append((n - SEAM_FRAMES, ["53", 0])); guides += guide_times(n - SEAM_FRAMES, SEAM_FRAMES)
        saved, nodes = _generate(f, client, graph, n, pins, strengths, timing, guides,
                                 seed, "touchdown/split_descent", part, o)
    lower = _record(saved, f, anchor_guide_nodes=nodes.get(anchor_frame), prefix_sha256=digest(prefix),
                    seam_clip_sha256=digest(seam), **extra)
    (part / "result.json").write_text(json.dumps(lower, indent=2))
    (folder / "provenance.json").write_text(json.dumps(dict(
        parent=dict(near_video_sha256=near["sha256"]), options=o, anchor=binding, source_segments=[upper, lower],
        geometry_guides=False), indent=2))
    return assemble(output, near, upper, lower, o)


def _log_scales(video, end, width=480):
    """ln of the similarity-transform scale between consecutive frames (ORB + RANSAC), None where untracked."""
    cap = cv2.VideoCapture(str(video))
    orb = cv2.ORB_create(1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    scales, prev = [], None
    try:
        for i in range(end):
            ok, image = cap.read()
            if not ok:
                break
            h = int(image.shape[0] * width / image.shape[1])
            grey = cv2.cvtColor(cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(grey, None)
            if prev is not None:
                scale = None
                if des is not None and prev[1] is not None and len(kp) >= 8 and len(prev[0]) >= 8:
                    matches = matcher.match(prev[1], des)
                    if len(matches) >= 8:
                        a = np.float32([prev[0][m.queryIdx].pt for m in matches])
                        b = np.float32([kp[m.trainIdx].pt for m in matches])
                        matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3)
                        if matrix is not None and inliers is not None and int(inliers.sum()) >= 8:
                            scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
                scales.append(round(float(np.log(scale)), 4) if scale and scale > 0 else None)
            prev = (kp, des)
    finally:
        cap.release()
    return scales


def zoom_profile(video, end, width=480):
    """Advisory: the camera's apparent speed on a pure descent is one number per frame, the scale of the
    similarity transform between consecutive frames (ln scale, positive = zooming in). It tells "the move
    finished early" (the curve goes flat before the portrait) from "the move was truncated" (it never
    reaches the ground) and shows a hold (near zero) or a pop (a spike) at the join; frame differences
    cannot tell those apart. Measured on the delivered flight up to `end` (the portrait seam)."""
    scales = _log_scales(video, end, width)
    known = [v for v in scales if v is not None]
    if len(known) < FPS:
        return dict(log_scale=scales, summary="too few tracked frames")
    half = FPS // 2
    smooth = [float(np.mean([v for v in scales[i:i + half] if v is not None] or [0])) for i in range(len(scales) - half + 1)]
    peak = max(smooth)
    flat_from = next((i for i in range(len(smooth)) if all(v < peak * .15 for v in smooth[i:])), None)
    return dict(log_scale=scales, window_frames=half, peak_log_scale=round(peak, 4),
                peak_frame=int(np.argmax(smooth)), flat_from_frame=flat_from,
                slowest_frame=int(np.argmin(smooth)), slowest_log_scale=round(min(smooth), 4),
                summary=("the zoom stops %d frames before the portrait" % (len(scales) - flat_from)
                         if flat_from is not None and flat_from < len(scales) - FPS else "the zoom runs to the portrait"))


def carry_guide(o):
    """Descent generation frame and length of the seam carry (with its bridge): it ends next to the seam.
    Prompt Relay needs one guide time per latent plane of it (Bled l1 listed only the carry's 3 of 7 and
    the relay refused the query geometry)."""
    if ladder_mode(o):
        # The parent flew the whole drop on the ladder, crane included: its own frames guide the descent from just
        # above the prefix to the seam, and the anchor still is not pinned (Pena p1: with the anchor pinned 31
        # frames above the prefix the descent drew the top-down drone view shrinking as a disc over her).
        return o["prefix_frames"], o["descent_frames"] - SEAM_FRAMES - o["prefix_frames"]
    count = o["seam_carry_frames"] + o["seam_bridge_frames"]
    return o["descent_frames"] - SEAM_FRAMES - count, count


def ladder_mode(o):
    return bool(o["parent_layout"] and o["layout_rungs"] and o["seam_carry_source"] == "parent")


SEAM_SPEED = (.004, .03)  # ln scale per frame the carried zoom may take


def seam_carry(seam, target, frames, width, height, accel=1.0, anchor=None, bridge=0):
    """The zoom continued past the join, as a moving guide for the descent frames just below the seam.

    Castel del Monte c1 (26 Oct): the parent flew through the high view at a steady 0.010 ln-scale per frame,
    the 8 seam frames carried that across the join, and then the descent nearly stopped for 17 frames
    (0.001 per frame) before rushing down in a 6-frame misty morph. These frames crop into the seam's
    lowest frame at the speed the parent was flying there, in retreat order (they end next to the seam),
    so LTX is told to keep going at that speed rather than settle.

    Bamburgh b1: a 16-frame carry at constant speed kept the camera moving through it, then the descent held
    and dissolved from the high coast to ground level (2 km in one cut). The real swoop runs at about 0.03-0.05
    per frame (Castel del Monte after its morph), so the carry eases in from the parent's speed to `accel`
    times it, guiding the camera into the swoop instead of leaving LTX to jump there.

    Montségur m1: the accelerating carry gave a real dive into the mountains, then the descent held for 8
    frames and dissolved to the castle, the anchor still: LTX dives only as far as it is guided. With `anchor`
    and `bridge`, the guide runs `bridge` frames further at the final speed while the anchor, nested at the
    centre and faded in from half way through the carry, grows until it fills the frame on the descent's
    anchor frame. The whole drop from the seam to the anchor is then one guided zoom. Returns (frames, speed)."""
    measured = [-v for v in _log_scales(seam, SEAM_FRAMES) if v is not None]
    speed = float(np.clip(np.median(measured), *SEAM_SPEED)) if measured else SEAM_SPEED[0]
    cap = cv2.VideoCapture(str(seam))
    try:
        ok, lowest = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("Could not read the parent seam")
    # Delivered offset k = 1..frames below the seam flies at speed * accel ** ((k - 1) / (frames - 1)), then
    # the bridge holds the final speed.
    steps = [speed * accel ** ((k - 1) / max(1, frames - 1)) for k in range(1, frames + 1)] + [speed * accel] * bridge
    depth = np.cumsum(steps)
    total = frames + bridge
    inner = cv2.resize(cv2.imread(str(anchor)), (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) if bridge else None
    feather = np.zeros((height, width), np.float32)
    border = max(8, width // 10)
    feather[border:-border, border:-border] = 1
    feather = cv2.GaussianBlur(feather, (0, 0), border / 2)
    appear, ramp = frames // 2, 8
    delivered = []
    for k in range(1, total + 1):
        frame = _window(lowest, float(np.exp(-depth[k - 1])), width, height)
        if bridge and k > appear:
            size = float(np.exp(-(depth[-1] - depth[k - 1])))
            iw, ih = max(2, int(round(width * size))), max(2, int(round(height * size)))
            x0, y0 = (width - iw) // 2, (height - ih) // 2
            alpha = min(1.0, (k - appear) / ramp)
            mask = (cv2.resize(feather, (iw, ih)) if k < total else np.ones((ih, iw), np.float32))[..., None] * alpha
            region = frame[y0:y0 + ih, x0:x0 + iw].astype(np.float32)
            frame[y0:y0 + ih, x0:x0 + iw] = (cv2.resize(inner, (iw, ih), interpolation=cv2.INTER_AREA) * mask
                                              + region * (1 - mask)).astype(np.uint8)
        delivered.append(frame)
    out = delivered[::-1]
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an",
                                "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(target)], stdin=subprocess.PIPE)
    for frame in out:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError("Could not encode the seam carry")
    return out, round(speed, 4)


def _cost(m):
    return m["mean_pixel_change"] + m["max_displacement_at_640px"] * .5


def motion_report(diffs, generated_end, window=FPS // 2):
    """Advisory: the slowest half second of the generated part against its median motion."""
    part = diffs[:max(0, generated_end-1)]
    if len(part) < window * 2:
        return None
    median = float(np.median(part))
    means = [float(np.mean(part[i:i+window])) for i in range(len(part)-window+1)]
    i = int(np.argmin(means))
    return dict(median_difference=round(median, 3), slowest_window_start_frame=i, window_frames=window,
                slowest_window_ratio=round(means[i]/median, 3) if median > 0 else None, gate=False)


HOLD_FLOOR = .35
HOLD_MAX_DROP = 12


def collapse_holds(diffs, end, floor=HOLD_FLOOR, max_drop=HOLD_MAX_DROP):
    """Delivered frames to keep so the flight never hangs: inside [0, end) a frame is dropped while the camera
    has moved less than `floor` x the flight's median since the last kept frame (at most `max_drop` in a row).

    LTX parks the camera for up to a second where a stage settles onto a pinned or guided frame. Dropping
    near-identical frames there shortens the hang without interpolating or blending anything; the portrait
    from `end` on is kept whole.
    """
    total = len(diffs) + 1
    part = diffs[:max(0, end-1)]
    if len(part) < FPS:
        return list(range(total))
    step = floor * float(np.median(part))
    keep, moved, dropped = [0], 0.0, 0
    for i in range(1, total):
        moved += diffs[i-1]
        if i >= end - 1 or moved >= step or dropped >= max_drop:
            keep.append(i); moved, dropped = 0.0, 0
        else:
            dropped += 1
    return keep


def _rewrite(target, keep, width, height):
    """Re-encode `target` with only the `keep` frames (no blending, no interpolation)."""
    cap = cv2.VideoCapture(str(target)); frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(image)
    cap.release()
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        stdin=subprocess.PIPE)
    try:
        for i in keep:
            encoder.stdin.write(frames[i].tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Photographic delivery encoding failed")


def _seam(a, b):
    """match_metrics, plus a featureless-haze case: inside cloud or haze SIFT finds too few features to register,
    but two frames that are nearly identical there join invisibly (Edinburgh held on 12 features in haze)."""
    m = match_metrics(a, b)
    if m["acceptable"] or m.get("matches") is None:
        return m
    small = [cv2.cvtColor(cv2.resize(x, (640, 368)), cv2.COLOR_BGR2GRAY) for x in (a, b)]
    change = float(np.abs(small[0].astype(float) - small[1]).mean())
    texture = max(float(cv2.Laplacian(x, cv2.CV_64F).var()) for x in small)
    haze = m["matches"] < 30 and change <= 8 and texture <= 60
    return {**m, "acceptable": haze, "haze": haze, "mean_pixel_change": round(change, 3),
            "max_displacement_at_640px": m.get("max_displacement_at_640px", 0.0), "texture": round(texture, 1)}


def _grey(image):
    return cv2.cvtColor(cv2.resize(image, (320, 184)), cv2.COLOR_BGR2GRAY)


def far_seam(parent_at, lower_frames, opening, reach=8, depth=12):
    """The parent/descent join: the closest pair near the pin, (parent delivery frame P, descent frame q).

    The pin binds descent[-1] to parent delivery `opening`, but LTX lands a few frames off (Mont-Saint-Michel was
    12 px out at the pinned pair), so nearby pairs on the same diagonal are measured too.
    """
    dc = len(lower_frames)
    pairs = []
    for P in range(max(1, opening - reach), opening + reach + 1):
        image = parent_at(P)
        for q in range(dc - 1, max(0, dc - 1 - depth), -1):
            m = _seam(image, lower_frames[q])
            if m["acceptable"]:
                pairs.append(dict(P=P, q=q, **m))
    if not pairs:
        return None
    return min(pairs, key=lambda m: _cost(m) + .05 * (abs(m["P"] - opening) + (dc - 1 - m["q"])))


SHARPNESS_RAMP = 12


def _sharpness(image):
    return float(cv2.Laplacian(cv2.cvtColor(cv2.resize(image, (640, 368)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def sharpness_match(soft, sharp):
    """Gaussian sigma that brings the sharp native portrait down to the softer descent at the portrait seam.

    The descent is a 1280x736 generation enlarged to the portrait's size, so the cut to the native portrait was a
    one-frame jump in detail (Barcelona: Laplacian 44 -> 100) though it matched to 0.2 px. Delivery eases the
    native frames from that sigma to none over half a second; frames are filtered one at a time, never mixed.
    """
    target = _sharpness(soft)
    if _sharpness(sharp) <= target * 1.2:
        return 0.0
    sigmas = np.arange(.25, 4.01, .25)
    return float(min(sigmas, key=lambda g: abs(_sharpness(cv2.GaussianBlur(sharp, (0, 0), g)) - target)))


def assemble(output, near, upper, lower, o):
    """Parent delivery frames [0, opening), the descent from its far pin down, then the native portrait."""
    for part in (upper, lower, near):
        _checked(part)
    caps = [cv2.VideoCapture(p["video"]) for p in (upper, lower, near)]
    try:
        (uc, uw, uh, uf), (dc, dw, dh, df), (nc, nw, nh, nf) = [video_info(c) for c in caps]
        if (uf, df, nf) != (FPS,)*3 or (uw, uh) != (dw, dh) or abs(uw/uh - nw/nh) > .002:
            raise ValueError("Segments must share 24 fps, size and the portrait's aspect")
        if uc != parent_frames(o) or dc != o["descent_frames"]:
            raise ValueError("Unexpected segment clock")
        up, down, native = caps
        lower_frames = [frame_at(down, i) for i in range(dc)]
        # descent[dc-1] was pinned to parent delivery frame `opening` (generation frame 192-opening)
        opening = o["opening_frames"]
        parent_at = lambda d: frame_at(up, BOUNDS[-1] - d)
        far = far_seam(parent_at, lower_frames, opening)
        if far is None:
            raise JoinMismatch("The descent does not start on the kept opening's frame; no dissolve was added")
        # descent[prefix-1-n] was guided by native[n]
        last = o["prefix_frames"] - 1
        pairs = [dict(n=k, **match_metrics(lower_frames[last-k], frame_at(native, k)))
                 for k in range(min(o["prefix_frames"], nc-1))]
        viable = [p for p in pairs if p["acceptable"]]
        if not viable:
            raise JoinMismatch("The moving portrait does not join closely enough; no dissolve was added")
        bottom = min(viable, key=_cost)
        n = bottom["n"]
        ratios = face_ratios(near["video"]) if o.get("end_face_ratio") else [None] * nc
        end = ending_frame(ratios, o.get("end_face_ratio"), n)
        nc = end + 1
        sigma = sharpness_match(lower_frames[last - n], frame_at(native, n))
        mapping = ([("parent", BOUNDS[-1] - d) for d in range(far["P"])]
                   + [("descent", g) for g in range(far["q"], last - n, -1)]
                   + [("native", j) for j in range(n, nc)])
        target = output / "earth_zoom.mp4"
        encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{nw}x{nh}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            stdin=subprocess.PIPE)
        try:
            native.set(cv2.CAP_PROP_POS_FRAMES, n)
            for source, index in mapping:
                if source == "parent":
                    image = frame_at(up, index)
                elif source == "descent":
                    image = lower_frames[index]
                else:
                    ok, image = native.read()
                    if not ok:
                        raise ValueError("Short native portrait")
                    weight = 1 - (index - n) / SHARPNESS_RAMP
                    if sigma and weight > 0:
                        image = cv2.GaussianBlur(image, (0, 0), sigma * weight)
                if image.shape[1::-1] != (nw, nh):
                    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close(); code = encoder.wait()
        if code:
            raise RuntimeError("Photographic delivery encoding failed")
    finally:
        for c in caps:
            c.release()
    from touchdown.surface_stability import frame_differences, jump_cuts
    native_start = len(mapping) - (nc - n)
    diffs = frame_differences(target)
    keep = collapse_holds(diffs, native_start)
    held = [i for i in range(len(mapping)) if i not in set(keep)]
    if held:
        _rewrite(target, keep, nw, nh)
        mapping = [mapping[i] for i in keep]
        native_start -= len(held)
        diffs = frame_differences(target)
    record = dict(output_sha256=digest(target), output_frames=len(mapping), fps=FPS,
                  source_segments=[upper, lower], native_portrait=near["video"], native_portrait_sha256=near["sha256"],
                  target_high_frame=(next(i for i, (s, g) in enumerate(mapping)
                                          if s != "parent" and (s == "native" or g <= BOUNDS[-1] - o["high_delivery_frame"]))
                                     if o["high_join"] == "cloud" else BOUNDS[2] + (far["P"] - BOUNDS[2]) // 2),
                  seam_frame=far["P"],
                  native_start_frame=native_start, native_first_frame=n, native_frames=nc - n,
                  high_seam=far, portrait_seam=bottom, source_frame_mapping=[f"{s}:{i}" for s, i in mapping],
                  generated_cuts=jump_cuts(diffs[:native_start]), motion=motion_report(diffs, native_start),
                  zoom=zoom_profile(target, native_start), flight_size=[uw, uh], delivered_size=[nw, nh], interpolation=False, crossfade=False,
                  face_composite=False, final_portrait_speed=1.0,
                  dropped_hold_frames=held, hold_floor=HOLD_FLOOR, portrait_sharpness_ramp=dict(sigma=sigma, frames=SHARPNESS_RAMP),
                  ending=dict(stop_face_ratio=o.get("end_face_ratio"), native_last_frame=end,
                              native_available=len(ratios), face_ratios=ratios))
    (output / "pacing.json").write_text(json.dumps(record, indent=2))
    return record
