"""Stage Woodside NAIP references and queue a five-second MiniMax H3 test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

from queue_h3_preview import graph


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_structure_heights

COMFY_ROOT = Path("/run/media/jamie/Data/Image Creation/Comfy")
CLAY_FRAMES = ROOT / "outputs/woodside_library_3dep/flyover_frames"
TEXTURED_FRAMES = ROOT / "outputs/woodside_library_3dep/textured_frames"
FLAT_GUIDE_FRAMES = ROOT / "outputs/woodside_library_3dep/flat_guide_frames"
CLOSE_MEASURED_FRAMES = ROOT / "outputs/woodside_library_3dep/blocking_measured_frames"
BUILDING_MANIFEST = ROOT / "outputs/woodside_library_3dep/measured_buildings.json"
CLOSE_OPENING = ROOT / "outputs/woodside_library_3dep/blocking_measured_storyboard/frame_0001.png"
CLOSE_NAIP = ROOT / "data/woodside_library_naip_close_220m.png"
CLOSE_REFERENCE_MANIFEST = ROOT / "data/woodside_library_close_reference_manifest.json"
SOURCE_IMAGES = (
    ROOT / "data/woodside_library_naip_660m.png",
    ROOT / "data/woodside_library_naip_2400m.png",
)
STYLE_IMAGES = (
    ROOT / "assets/style_references/woodside_clearline/woodside_clearline_wide.png",
    ROOT / "assets/style_references/woodside_clearline/woodside_clearline_near.png",
)
STYLE_MANIFEST = ROOT / "assets/style_references/woodside_clearline/manifest.json"
FRAME_COUNT = 124


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", help="GPU-enabled ComfyUI base URL, required with --submit")
    p.add_argument("--comfy-root", type=Path, default=COMFY_ROOT)
    p.add_argument("--take", choices=("closemeasured", "closemeasuredtight"), default="closemeasuredtight",
                   help="Only measured-building 3D controls are eligible for new H3 jobs")
    p.add_argument("--style", choices=("clear-line",), default="clear-line")
    p.add_argument("--clearline-prompt", choices=("auto", "legacy", "positive"), default="auto")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--veda", action="store_true",
                   help="Use optional Veda sparse attention; keep reference attention dense")
    a = p.parse_args()
    prompt_mode = ("positive" if a.take in {"flatguide", "closemeasured", "closemeasuredtight"} else "legacy") if a.clearline_prompt == "auto" else a.clearline_prompt
    if a.submit and not a.url:
        p.error("--submit requires --url for a GPU-enabled ComfyUI instance")
    if a.style == "clear-line" and a.take == "clay":
        p.error("--style clear-line requires --take textured or flatguide")
    if a.take == "flatguide" and a.style != "clear-line":
        p.error("--take flatguide requires --style clear-line")
    if a.take in {"closemeasured", "closemeasuredtight"} and a.style != "clear-line":
        p.error("Close measured takes require --style clear-line")
    if a.take in {"closemeasured", "closemeasuredtight"}:
        qualify_structure_heights(json.loads(BUILDING_MANIFEST.read_text()), "382886306", 200)
    source_frames = {"clay": CLAY_FRAMES, "textured": TEXTURED_FRAMES,
                     "flatguide": FLAT_GUIDE_FRAMES,
                     "closemeasured": CLOSE_MEASURED_FRAMES,
                     "closemeasuredtight": CLOSE_MEASURED_FRAMES}[a.take]
    place_images = (CLOSE_OPENING, CLOSE_NAIP) if a.take == "closemeasuredtight" else SOURCE_IMAGES
    references = place_images + (STYLE_IMAGES if a.style == "clear-line" else ())
    if not all(path.is_file() for path in references):
        raise FileNotFoundError("Woodside place or style reference images are missing")
    if a.take == "closemeasuredtight":
        recorded_close = json.loads(CLOSE_REFERENCE_MANIFEST.read_text())["images"]
        if [Path(item["path"]) for item in recorded_close] != list(place_images):
            raise ValueError("Close reference manifest does not match the selected images")
        for item in recorded_close:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Close reference checksum changed: {item['path']}")
    if a.style == "clear-line":
        recorded = {item["file"]: item["sha256"] for item in json.loads(STYLE_MANIFEST.read_text())["images"]}
        for path in STYLE_IMAGES:
            if sha256(path) != recorded[path.name]:
                raise ValueError(f"Style reference checksum changed: {path}")
    if len(list(source_frames.glob("frame_*.png"))) != 192:
        raise RuntimeError(f"Woodside {a.take} control must have exactly 192 source frames")

    target = a.comfy_root / "input/geoblock_woodside"
    target.mkdir(parents=True, exist_ok=True)
    staged_images = []
    for source in references:
        dest = target / source.name
        if not dest.exists() or sha256(dest) != sha256(source):
            shutil.copy2(source, dest)
        staged_images.append(dest)
    control = target / f"woodside_library_{a.take}_control_124f.mp4"
    if not control.exists() or control.stat().st_mtime < max(p.stat().st_mtime for p in source_frames.glob("frame_*.png")):
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "24",
            "-start_number", "1", "-i", str(source_frames / "frame_%04d.png"),
            "-vf", "setpts=(PTS-STARTPTS)*124/192,fps=24,scale=864:480:flags=lanczos",
            "-frames:v", str(FRAME_COUNT), "-an", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", str(control),
        ], check=True)

    workflow = graph({"prompt": ""}, veda=a.veda)
    workflow["4"]["inputs"]["image"] = "geoblock_woodside/" + staged_images[0].name
    workflow["5"]["inputs"]["image"] = "geoblock_woodside/" + staged_images[1].name
    workflow["6"]["inputs"]["file"] = "geoblock_woodside/" + control.name
    clay_prompt = (
        "<Picture 1> is a 2022 NAIP aerial crop of Woodside Library at 3140 Woodside Road, "
        "Woodside, California. <Picture 2> shows the wider Woodside terrain and street layout. "
        "<Video 1> supplies the smooth eight-second clay flyover camera motion, retimed for this shot. "
        "Create one continuous five-second oblique aerial flyover around this address, keeping the "
        "wooded suburban terrain and road layout consistent with the aerial images. Replace the gold "
        "schematic location pin and grey clay surface with believable vegetation, roads and buildings. "
        "Natural California daylight. Smooth camera movement without cuts, morphs or abrupt turns. "
        "No labels, captions, people, or text. This is a stylized geographic reconstruction."
    )
    textured_prompt = (
        "<Picture 1> is a 2022 NAIP aerial crop around Woodside Library, 3140 Woodside Road, "
        "Woodside California. <Picture 2> is the wider NAIP aerial survey. <Video 1> is a "
        "georeferenced aerial texture draped on measured USGS 3DEP terrain; use its continuous "
        "smooth camera arc and broad terrain layout only. The video is a planning model, not the "
        "desired final appearance: do not copy flat aerial texture, mesh facets, clay shading or "
        "the schematic gold locator pin. Keep the roads, building placements and wooded pattern "
        "guided by the aerial images. Give trees and buildings natural volume and California daylight. "
        "One uninterrupted five-second oblique aerial shot, no cut, jump, morph, text, people "
        "or labels. Stylized geographic reconstruction."
    )
    clearline_prompt = (
        "<Picture 1> is the close 2022 NAIP aerial survey of Woodside Library, California; "
        "<Picture 2> is the wider aerial survey. They define geography and road layout, not style. "
        "<Picture 3> and <Picture 4> are original Tintin-era ligne claire comic illustrations. They define "
        "the final visual style: uniform crisp ink outlines, flat restrained daylight colours, "
        "legible individual tree canopies, simple roof volumes and minimal shadows. "
        "<Video 1> is a rough 3D planning model. Follow its smooth camera arc, timing and broad "
        "terrain only. Transform every frame into a coherent hand-drawn ligne claire landscape; "
        "do not retain aerial-photo texture, model-rendered surfaces, mesh facets, clay shading "
        "or the gold locator pin. Keep the recognizable roads and wooded suburban layout. "
        "One continuous five-second shot with consistent linework and no cuts, jumps, morphs, "
        "text, people, speech balloons or labels."
    )
    positive_clearline_prompt = (
        "<Picture 1> is a close aerial survey of the Woodside, California landscape. "
        "<Picture 2> shows its wider road network and wooded hills. Use these for geographic layout. "
        "<Picture 3> and <Picture 4> are original clear-line illustrations; use them as the "
        "visual target for every frame: crisp even dark ink contours, flat restrained daylight "
        "colours, distinct rounded tree canopies, simple low roof shapes and open blue sky. "
        "<Video 1> supplies only the smooth camera arc, timing and broad terrain composition. "
        "Draw a coherent 2D ligne claire landscape from the opening to the final frame. "
        "Keep roads and wooded blocks in their surveyed positions while expressing houses and "
        "trees as clean illustrated forms. One continuous oblique aerial shot, stable line weight "
        "and stable colours throughout. The frame contains only the landscape and sky."
    )
    close_measured_prompt = (
        "<Picture 1> is a close aerial survey of Woodside Library, California. "
        "<Picture 2> shows the wider Woodside road network and wooded hills. Use these for "
        "geographic layout. <Picture 3> and <Picture 4> are original clear-line illustrations; "
        "use their crisp even ink contours, flat restrained daylight colours, legible tree "
        "canopies and simple low roof shapes throughout the shot. <Video 1> supplies a close "
        "smooth camera arc and approximate building footprints and heights measured from USGS "
        "lidar. Preserve their relative placement, depth and scale while drawing an illustrated "
        "landscape. The block geometry is a spatial guide; give each structure a coherent "
        "illustrated roof and facade. One continuous oblique flyover with stable line weight and "
        "colour, no cuts, jumps, morphs, text, people, speech balloons or labels."
    )
    close_tight_prompt = (
        "<Picture 1> is the exact opening camera composition: low oblique view of Woodside "
        "Library and its immediate neighbouring buildings. Keep this close camera scale through "
        "the entire shot. <Picture 2> is the address-centred 220-metre NAIP crop showing the "
        "real footprints, roads and trees; use it for place details, not an overhead camera angle. "
        "<Picture 3> and <Picture 4> are original clear-line illustrations defining crisp even "
        "ink contours and flat daylight colours. <Video 1> is the required close continuous "
        "oblique camera path around building volumes whose heights were estimated from lidar. "
        "Match its framing, parallax, building scale and movement frame by frame. Draw finished "
        "illustrated roofs, facades, trees and roads in that same 3D arrangement. One close "
        "five-second flyover with stable line weight; no zoom out, distant panorama, overhead "
        "map view, cuts, morphing, text, people or labels."
    )
    if a.style == "clear-line":
        for node_id, slot, path in zip(("17", "18"), (2, 3), staged_images[2:]):
            workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": "geoblock_woodside/" + path.name}}
            workflow["8"]["inputs"][f"ref_images.ref_image_{slot}"] = [node_id, 0]
    workflow["8"]["inputs"]["prompt"] = ((close_tight_prompt if a.take == "closemeasuredtight" else
                                              close_measured_prompt if a.take == "closemeasured" else
                                              positive_clearline_prompt if prompt_mode == "positive" else clearline_prompt)
                                            if a.style == "clear-line" else
                                            textured_prompt if a.take == "textured" else clay_prompt)
    workflow["9"]["inputs"]["noise_seed"] = 20261004
    workflow["16"]["inputs"]["filename_prefix"] = (
        "video/WoodsideLibrary_H3_clearline_closemeasured_tight" if a.take == "closemeasuredtight" else
        "video/WoodsideLibrary_H3_clearline_closemeasured" if a.take == "closemeasured" else
        ("video/WoodsideLibrary_H3_clearline_flatguide_positive" if prompt_mode == "positive" else
         "video/WoodsideLibrary_H3_clearline_flatguide") if a.take == "flatguide" else
        "video/WoodsideLibrary_H3_clearline" if a.style == "clear-line" else
        "video/WoodsideLibrary_H3_textured" if a.take == "textured" else "video/WoodsideLibrary_H3_test")
    if a.veda:
        workflow["16"]["inputs"]["filename_prefix"] += "_veda"
    suffix = "clearline_closemeasured_tight" if a.take == "closemeasuredtight" else "clearline_closemeasured" if a.take == "closemeasured" else ("clearline_flatguide_positive" if prompt_mode == "positive" else "clearline_flatguide") if a.take == "flatguide" else (
        "clearline" if a.style == "clear-line" else a.take)
    if a.veda:
        suffix += "_veda"
    out = ROOT / f"outputs/woodside_library_3dep/h3_{suffix}_workflow_api.json"
    out.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "address": "3140 Woodside Road, Woodside, CA 94062",
        "address_geocode": "OpenStreetMap node 368169993, 37.4291056, -122.2580225",
        "terrain": "USGS 3DEP 1 m bare-earth DEM, CA_CaliforniaGaps_B23; NAVD88",
        "images": [{"path": str(path), "sha256": sha256(path)} for path in place_images],
        "style": a.style,
        "clearline_prompt_mode": prompt_mode if a.style == "clear-line" else None,
        "style_images": [{"path": str(path), "sha256": sha256(path),
                          "role": "style only; not geographic ground truth"} for path in
                         (STYLE_IMAGES if a.style == "clear-line" else ())],
        "imagery_source": "USGS NAIP Imagery ImageServer; USDA NAIP, 2022-05-18, 0.6 m source; public domain",
        "motion_reference": {"path": str(control), "sha256": sha256(control), "frames": FRAME_COUNT},
        "building_heights": str(BUILDING_MANIFEST) if a.take in {"closemeasured", "closemeasuredtight"} else None,
        "source_frames": str(source_frames),
        "h3_workflow": str(out),
        "attention": {"method": "Veda", "generated_sparsity": "90%", "reference_sparsity": "0%"} if a.veda else {"method": "full"},
    }
    (out.parent / f"h3_{suffix}_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")
    if a.submit:
        if a.veda:
            with urlopen(a.url + "/object_info/VedaSparseAttention", timeout=10) as response:
                if "VedaSparseAttention" not in json.load(response):
                    raise RuntimeError("VedaSparseAttention is not loaded in this ComfyUI instance")
        request = Request(a.url + "/prompt", data=json.dumps({"prompt": workflow}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            print(response.read().decode())


if __name__ == "__main__":
    main()
