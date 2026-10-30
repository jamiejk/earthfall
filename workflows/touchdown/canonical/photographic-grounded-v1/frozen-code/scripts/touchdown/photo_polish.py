"""Pace the approved photographic flight and extend its living arrival.

The only video conditioning is a nine-frame prefix from the existing photo-
generated film. No terrain renderer, depth map or layout model enters this path.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.authored_scene import digest


def video_info(path):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)]))["streams"][0]
    result["nb_frames"] = int(result["nb_frames"])
    if result["r_frame_rate"] != "24/1":
        raise ValueError("Expected 24 fps input")
    return result


def extend(source, output, url):
    import fakeout_earth_zoom as f
    folder = output / "arrival-extension"
    folder.mkdir(parents=True, exist_ok=True)
    info = video_info(source)
    # Check the actual source workflow rather than accepting a layout-rendered
    # clip merely because it was given a photographic filename.
    source_graphs = list(source.parent.glob("*/graph.json"))
    result_record = source.parent / "result.json"
    if result_record.exists():
        selected = json.loads(result_record.read_text())
        if selected["sha256"] != digest(source):
            raise ValueError("The source no longer matches its generation record")
        graph_record = Path(selected["workflow"])
        if graph_record not in source_graphs or digest(graph_record) != selected["workflow_sha256"]:
            raise ValueError("The source workflow record changed")
    elif len(source_graphs) == 1:
        graph_record = source_graphs[0]
    else:
        raise ValueError("Use a flight with a result.json identifying its source generation")
    original = json.loads(graph_record.read_text())
    forbidden = {"LoadVideo", "LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide"}
    if any(n["class_type"] in forbidden for n in original.values()):
        raise ValueError("Flight source contains video/layout conditioning")
    if sum(n["class_type"] == "LoadImage" for n in original.values()) != 1:
        raise ValueError("Expected the single-arrival-photograph flight")
    prefix = folder / "photographic-prefix.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-vf", f"trim=start_frame={info['nb_frames']-9},setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(prefix)], check=True)
    prompt = (
        "Continue this same live-action environmental portrait seamlessly. The camera completes "
        "its very gentle settling motion and stays with the woman at the same framing. Preserve "
        "her exact face, short dark curls, red raincoat, grey hoodie and black trousers. She quietly "
        "breathes, blinks naturally once, and gives a small relaxed smile. A light breeze subtly "
        "moves individual curls, the coat hem and sleeve folds. Her body remains naturally "
        "balanced in the same place on the same pale sandstone rim. The canyon and broad copper "
        "mesa remain stable behind her, soft storm light from the left. Clouds drift gently. "
        "Natural unretouched travel footage. No speaking, no large gestures, no camera orbit, "
        "no zoom, no landscape changes, no cuts, no dissolve, no frozen photograph."
    )
    with f._client(url) as client:
        graph = {
            "50": {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(prefix))}},
            "51": {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}}
        wanted = f._ltx(graph, SimpleNamespace(no_upscale=False), 57,
                        [(0, ["51", 0])], prompt, 2026102262, "touchdown/photo_arrival_extension")
        if any(n["class_type"] in forbidden - {"LoadVideo"} for n in graph.values()):
            raise ValueError("Layout conditioning is forbidden")
        provenance = dict(source=str(source), source_sha256=digest(source),
                          source_workflow=str(graph_record), source_workflow_sha256=digest(graph_record),
                          source_lineage=json.loads((source.parent / "lineage.json").read_text()),
                          prefix_sha256=digest(prefix), prefix_frames=9, total_frames=57,
                          appended_frames=48, prompt=prompt, seed=2026102262,
                          conditioning="Photographic generated-video prefix; no 3D/layout guide")
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
        saved = f._submit(client, graph, tuple(wanted), folder)
    result = Path(saved[f.LTX_VIDEO])
    (folder / "result.json").write_text(json.dumps(dict(video=str(result), sha256=digest(result)), indent=2))
    print(result, flush=True)


def source_position(frame):
    """C1 monotonic map: reveal terrain sooner, then gently slow the arrival.

    Output frames 0/61/114/162 correspond to original frames 0/90/144/176.
    The ending remains the actual last frame, permitting a matching continuation.
    """
    knots = [(0., 0., 1.5), (61., 90., 1.3), (114., 144., .85), (162., 176., .4)]
    if not 0 <= frame <= 162:
        raise ValueError("Frame outside the retimed flight")
    for (x0, y0, d0), (x1, y1, d1) in zip(knots, knots[1:]):
        if frame <= x1:
            t = (frame-x0)/(x1-x0)
            return ((2*t**3-3*t**2+1)*y0 + (t**3-2*t**2+t)*(x1-x0)*d0
                    + (-2*t**3+3*t**2)*y1 + (t**3-t**2)*(x1-x0)*d1)
    raise AssertionError("Unreachable")


def interpolate(a, b, amount, flow):
    """Optical-flow interpolation is confined to the slowed arrival."""
    import cv2
    import numpy as np
    height, width = a.shape[:2]
    scale = min(1., 960 / width)
    size = (round(width*scale), round(height*scale))
    if flow is None:
        engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), size)
        gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), size)
        forward = engine.calc(ga, gb, None)
        backward = engine.calc(gb, ga, None)
        flow = [cv2.resize(f, (width, height)) / scale for f in (forward, backward)]
    grid = np.stack(np.meshgrid(np.arange(width, dtype=np.float32),
                                np.arange(height, dtype=np.float32)), axis=-1)
    warped = []
    for image, movement, fraction in ((a, flow[0], amount), (b, flow[1], 1-amount)):
        mapping = grid.copy()
        for _ in range(3):
            sampled = cv2.remap(movement, mapping, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mapping = grid-fraction*sampled
        warped.append(cv2.remap(image, mapping, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return cv2.addWeighted(warped[0], 1-amount, warped[1], amount, 0), flow


def compose(source, output):
    import cv2
    from PIL import Image, ImageDraw
    folder = output / "arrival-extension"
    provenance = json.loads((folder / "provenance.json").read_text())
    if digest(source) != provenance["source_sha256"]:
        raise ValueError("The flight changed after extension generation")
    result = json.loads((folder / "result.json").read_text())
    ending = Path(result["video"])
    if digest(ending) != result["sha256"]:
        raise ValueError("The generated ending changed")
    info, end_info = video_info(source), video_info(ending)
    if info["nb_frames"] != 177 or end_info["nb_frames"] != 57:
        raise ValueError("Expected 177-frame flight and 57-frame continuation")
    if (info["width"], info["height"]) != (end_info["width"], end_info["height"]):
        raise ValueError("Flight and ending dimensions must agree")
    width, height = info["width"], info["height"]
    target = output / "earth_zoom.mp4"
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(source))
    _, a = cap.read(); _, b = cap.read()
    index = 0; flow = None
    sampling = []; previews = []
    try:
        for frame in range(163):
            position = min(176., max(0., source_position(frame)))
            integer = int(position)
            while index < integer:
                a = b; ok, b = cap.read(); index += 1; flow = None
                if not ok:
                    b = a
            fraction = position-integer
            if integer >= 144 and fraction > .02 and integer < 176:
                image, flow = interpolate(a, b, fraction, flow)
            else:
                image = b if fraction >= .5 else a
            encoder.stdin.write(image.tobytes())
            sampling.append(dict(output_frame=frame, source_frame=position,
                                 method="optical_flow" if integer >= 144 and fraction > .02 and integer < 176 else "nearest"))
            if frame % 6 == 0:
                previews.append((frame, cv2.resize(image, (480, 272))))
        cap.release()
        cap = cv2.VideoCapture(str(ending))
        for i in range(57):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Short ending at {i}")
            if i >= 9:
                frame = 163+i-9
                encoder.stdin.write(image.tobytes())
                if frame % 6 == 0:
                    previews.append((frame, cv2.resize(image, (480, 272))))
    finally:
        cap.release(); encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("Video encoder failed")
    board = Image.new("RGB", (2400, 300*((len(previews)+4)//5)), "#181818")
    for i, (frame, image) in enumerate(previews):
        x, y = i%5*480, i//5*300
        board.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (x, y+24))
        ImageDraw.Draw(board).text((x+8, y+5), f"{frame/24:.2f}s", fill="white")
    board.save(output / "motion-board.jpg", quality=93)
    evidence = dict(source=str(source), source_sha256=digest(source), ending=str(ending),
                    ending_sha256=digest(ending), output_sha256=digest(target),
                    output_info=video_info(target), flight_frames=163, appended_frames=48,
                    cloud_clear_seconds=61/24, interpolation="Optical flow only from source frame 144",
                    crossfade=False, frame_mapping=sampling)
    (output / "edit.json").write_text(json.dumps(evidence, indent=2))
    print(target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("outputs/touchdown/photo-only-canyonlands/video-upscaled-retreat/earth_zoom.mp4"))
    p.add_argument("--output", type=Path, default=Path("outputs/touchdown/photo-only-canyonlands/polish-v2"))
    p.add_argument("--stage", choices=["extension", "compose"], required=True)
    p.add_argument("--url", default="http://127.0.0.1:8193")
    a = p.parse_args(); output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if a.stage == "extension":
        extend(a.source.resolve(), output, a.url)
    else:
        compose(a.source.resolve(), output)


if __name__ == "__main__":
    main()
