"""Anchor an Earth flight with a moving photographic close approach.

Generate a gentle near-camera push-in, reverse it as a retreat prefix, and extend
its photographic frames into the flight. Reverse the full shot for descent.
No 3D/layout conditioning.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.authored_scene import digest, read_scene
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.photo_gate import require_arrival_gate


def run(cfg, scene, output, stage, url, near_source=None):
    import fakeout_earth_zoom as f
    ancestry = verify_lineage(cfg, scene, asset_plan(cfg, scene))
    if "arrival" not in ancestry["generated"]:
        raise ValueError("Generate and verify the photographic arrival first")
    require_arrival_gate(cfg, scene, output)
    folder = output / stage
    folder.mkdir(parents=True, exist_ok=True)
    pins = []
    with f._client(url) as client:
        if stage == "near":
            graph = {}
            image = f._load(graph, client, "44", scene / "arrival.png")
            frames, width, height, seed = int(cfg.get("video", {}).get("near_frames", 49)), 1280, 736, cfg["seed"] + 31
            prompt = cfg.get("prompts", {}).get("near") or (
                "One continuous sharp live-action environmental portrait of this same woman at this "
                "same location in this photograph. Over these two seconds, the camera gently dollies forward "
                "about one metre, changing from the waist-up portrait to a closer head-and-shoulders "
                "portrait. A constant lens, slow smooth real camera movement. Her face "
                "remains sharply focused with crisp eyes, brows, lips, natural skin and individual "
                "dark curls throughout. Preserve her likeness, red raincoat, grey hoodie and black "
                "trousers. Her arms stay down and relaxed by her sides. She breathes quietly and "
                "blinks once, with a small relaxed closed-mouth smile. A light breeze moves a few "
                "curls and coat folds. Stable photographic setting and architecture, consistent "
                "daylight and depth. Short shutter exposure, deep focus. No speaking, no opening "
                "her mouth, no raising hands, no gestures, no large head turn, no blur, no morphing, "
                "no frozen still, no zoom, no cut."
            )
            parent = dict(arrival_sha256=digest(scene / "arrival.png"), photographic_roots=ancestry)
        else:
            near_folder = near_source if near_source is not None else output / "near"
            record = json.loads((near_folder / "result.json").read_text())
            near = Path(record["video"])
            prior = json.loads((near_folder / "provenance.json").read_text())
            if (digest(near) != record["sha256"]
                    or digest(Path(record["workflow"])) != record["workflow_sha256"]
                    or prior["parent"]["arrival_sha256"] != digest(scene / "arrival.png")):
                raise ValueError("The close photographic sequence or its parent changed")
            # The near generation produced a clear forward portrait push-in.
            # Reverse that as the retreat prefix; reversing the full flight
            # later restores the near sequence's natural forward motion.
            prefix = folder / "near-retreat-prefix.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(near),
                            "-vf", "reverse", "-an", "-c:v", "libx264", "-crf", "12",
                            "-pix_fmt", "yuv420p", str(prefix)], check=True)
            graph = {
                "50": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(prefix))}},
                "51": {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}}
            image = ["51", 0]
            options = cfg.get("video", {})
            # The offset is separable from cfg["seed"] so a flight can be re-rolled without touching
            # the seed the arrival was generated with, which the lineage check pins.
            frames, width, height, seed = (int(options.get("frames", 217)), int(options.get("flight_width", 960)),
                                           int(options.get("flight_height", 544)),
                                           cfg["seed"] + int(options.get("flight_seed_offset", 32)))
            if options.get("scale_anchors"):
                high_index = int(options["high_frame"])
                if high_index <= int(options.get("near_frames", 49)) or high_index >= frames-24:
                    raise ValueError("The high aerial anchor must follow the portrait prefix and precede the orbital climb")
                if options.get("pin_high", True):
                    pins.append((high_index, f._load(graph, client, "45", scene / "high.png")))
                endpoint = options.get("flight_endpoint", "orbit")
                if endpoint not in ("orbit", "high"):
                    raise ValueError("Use a photographic orbit or high endpoint")
                pins.append((-1, f._load(graph, client, "46", scene / f"{endpoint}.png")))
            prompt = cfg.get("prompts", {}).get("prefix_flight") or (
                "Continue the supplied gentle photographic camera retreat into one uninterrupted "
                "fast aerial flight. The first two seconds are the supplied near-portrait sequence: "
                "preserve those moving frames, the woman's clear face and relaxed arms. After that "
                "sequence the camera accelerates backward and upward at 45 degrees over the same "
                "Trocadéro terrace. The woman becomes smaller on the same paving. Keep the Eiffel "
                "Tower recognisable in the same urban setting beyond her. Connected Parisian cream "
                "limestone city blocks, grey zinc roofs, tree-lined avenues and the Seine expand "
                "around the scene with real perspective and strong parallax. Fly back and up high "
                "over this coherent city, then briefly through a dense white cloud bank that fills "
                "the lens. Emerge above the clouds and keep climbing rapidly until the curved blue "
                "limb of Earth appears against black space over western Europe. Natural photographic "
                "daylight, consistent sunlight from the left, green-brown land and white weather "
                "systems. Fixed focal length and one camera heading. A physical flight, no cuts, "
                "no dissolves, no lens zoom, no camera orbit, no morphing architecture, no plastic "
                "surfaces, no repeated landmarks. The woman stays natural, does not speak or gesture."
            )
            parent = dict(near_source=str(near_folder), near_video=str(near), near_video_sha256=digest(near),
                          prefix=str(prefix), prefix_sha256=digest(prefix), prefix_reversed=True,
                          near_workflow=record["workflow"], near_workflow_sha256=record["workflow_sha256"],
                          photographic_roots=ancestry)
        if frames < 9 or frames % 8 != 1:
            raise ValueError("Photographic motion requires 8n+1 frames")
        refine = stage == "near" or cfg.get("video", {}).get("flight_refine", True)
        wanted = f._ltx(graph, SimpleNamespace(no_upscale=not refine), frames, [(0, image), *pins], prompt, seed,
                        f"touchdown/portrait_{stage}")
        graph["308"]["inputs"].update(width=width, height=height)
        forbidden = {"LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide"}
        if any(n["class_type"] in forbidden for n in graph.values()):
            raise ValueError("3D/layout conditioning is forbidden")
        provenance = dict(parent=parent, frames=frames, base_width=width, base_height=height,
                          seed=seed, prompt=prompt, scale_anchor_frames=[i for i, _ in pins], latent_refinement=refine,
                          conditioning="photographic near-motion prefix" if stage == "flight" else "photographic arrival image")
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
        saved = f._submit(client, graph, tuple(wanted), folder)
    generation = Path(saved[f.LTX_VIDEO])
    workflow = generation.parent / "graph.json"
    record = dict(video=str(generation), sha256=digest(generation), workflow=str(workflow),
                  workflow_sha256=digest(workflow))
    if stage == "flight":
        destination = folder / "earth_zoom.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(generation),
                        "-vf", "reverse", "-an", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(destination)], check=True)
        record.update(descent=str(destination), descent_sha256=digest(destination))
    (folder / "result.json").write_text(json.dumps(record, indent=2))
    print(record.get("descent", record["video"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("shots/authored/paris-photo-only.toml"))
    p.add_argument("--scene", type=Path, default=Path("outputs/touchdown/photo-only-paris"))
    p.add_argument("--output", type=Path, default=Path("outputs/touchdown/photo-only-paris/portrait-controlled"))
    p.add_argument("--stage", choices=["near", "flight", "pace"], required=True)
    p.add_argument("--near-source", type=Path, help="Reuse a verified photographic near sequence for a flight variant")
    p.add_argument("--reveal-frame", type=int, help="Source frame at which the city is clearly revealed")
    p.add_argument("--reveal-time", type=float, default=2.6)
    p.add_argument("--url", default="http://127.0.0.1:8193")
    a = p.parse_args()
    if a.stage == "pace":
        if a.reveal_frame is None:
            p.error("--stage pace requires --reveal-frame from visual inspection")
        pace(a.output.resolve(), a.reveal_frame, a.reveal_time)
    else:
        run(read_scene(a.manifest), a.scene.resolve(), a.output.resolve(), a.stage, a.url,
            a.near_source.resolve() if a.near_source else None)


def pace(output, reveal_frame, reveal_time):
    """Shorten only the orbital/cloud section; retain the clear close sequence."""
    import cv2
    source = output / "flight/earth_zoom.mp4"
    record = json.loads((output / "flight/result.json").read_text())
    if digest(source) != record["descent_sha256"]:
        raise ValueError("The source flight no longer matches its record")
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_frame = round(reveal_time*24)
    if not 0 < reveal_frame < total or target_frame <= 0:
        raise ValueError("Invalid reveal timing")
    target = output / "earth_zoom.mp4"
    if reveal_frame <= target_frame:
        cap.release(); shutil.copy2(source, target)
        mapping = list(range(total))
    else:
        encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target)], stdin=subprocess.PIPE)
        mapping = []
        for i in range(total-(reveal_frame-target_frame)):
            if i <= target_frame:
                t = i/target_frame
                position = (-t**3+t**2+t)*reveal_frame + (t**3-t**2)*target_frame
            else:
                position = reveal_frame+i-target_frame
            mapping.append(round(position))
        if mapping[0] != 0 or mapping[-1] != total-1 or not all(a<b for a,b in zip(mapping,mapping[1:])):
            raise ValueError("Pacing would skip an endpoint or repeat/reverse a frame")
        index = -1
        try:
            for frame in mapping:
                while index < frame:
                    ok, image = cap.read(); index += 1
                    if not ok:
                        raise ValueError("Short source flight")
                encoder.stdin.write(image.tobytes())
        finally:
            cap.release(); encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoder failed")
    (output / "pacing.json").write_text(json.dumps(dict(
        source=str(source), source_sha256=digest(source), output_sha256=digest(target),
        source_reveal_frame=reveal_frame, target_reveal_frame=min(reveal_frame,target_frame),
        output_frames=len(mapping), fps=24, source_frame_mapping=mapping,
        interpolation=False, crossfade=False, final_portrait_speed=1.0), indent=2))
    print(target, flush=True)


if __name__ == "__main__":
    main()
