"""Reference-conditioned assets for a bounded, authored photographic set.

This experimental CLI is separate from the measured-terrain production build.
Generate the master first; later assets inherit its appearance through references.
Altitude views combine separate layout guides with crops of the master's appearance.
The near approach is rendered from persistent geometry/textures, not repainted frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown import flux3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scene(path: Path) -> dict:
    cfg = tomllib.loads(path.read_text())
    root = path.resolve().parent
    if cfg.get("gate", {}).get("arrival_report"):
        cfg["gate"]["arrival_report"] = str((root / cfg["gate"]["arrival_report"]).resolve())
    for key in ("subject", "reference_manifest"):
        cfg[key] = str((root / cfg[key]).resolve())
        if not Path(cfg[key]).is_file():
            raise ValueError(f"Missing {key}: {cfg[key]}")
    for key in ("aerial_layout", "orbit_layout"):
        if key in cfg:
            cfg[key] = str((root / cfg[key]).resolve())
            if not Path(cfg[key]).is_file():
                raise ValueError(f"Missing {key}: {cfg[key]}")
    metadata = json.loads(Path(cfg["reference_manifest"]).read_text())
    records = {r["page_id"]: r for r in metadata["selected"]}
    cfg["references"] = []
    for page_id in cfg["reference_ids"]:
        record = records[page_id].copy()
        original = Path(record["local_path"])
        local = Path(cfg["reference_manifest"]).parent / original.name
        if not local.is_file():
            raise ValueError(f"Reference is unavailable: {local}")
        record.update(local_path=str(local), sha256=digest(local))
        cfg["references"].append(record)
    if len(cfg["references"]) < 2:
        raise ValueError("Use at least two reviewed location references")
    return cfg


def asset_plan(cfg: dict, output: Path) -> list[dict]:
    """Roles and scale restrictions are explicit; identity never conditions orbit."""
    refs = [r["local_path"] for r in cfg["references"]]
    master = str(output / "master.png")
    context = (f"Location inspiration: {cfg['place']}. {cfg['character']} "
               f"Shared art direction: {cfg['look']} "
               "This is a plausible cinematic interpretation, not an exact topographic reconstruction. ")
    # Full-scene editing overconstrains camera scale. Inherit appearance through
    # crops of the master; use separate grayscale references for camera layout.
    ground_ref = str(output / "master_ground_reference.png")
    look_ref = str(output / "master_look_reference.png")
    definitions = [
        ("master", refs, (2688, 1536), context +
         "Use the supplied real location photographs to create one beautiful, coherent photographic scene. "
         "A wide cinematic view from a natural sandstone overlook. The foreground is a continuous pale warm "
         "sandstone shelf across the bottom quarter of the image, broad enough to stand on. Beyond the shelf "
         "is a vast, deeply eroded canyon with distinctive layered flat-topped mesas, strong middle-distance "
         "forms, and softer distant horizons. The skyline is about one third down the image. Camera level, "
         "35mm lens, natural photographic textures and convincing depth. Entirely empty, uninhabited landscape. "
         "An unretouched National Geographic landscape photograph, with irregular broken weathered rock, "
         "fine cracks and sparse scrub, natural ledges and talus, subtle colours, realistic atmospheric depth. "
         "The broad foreground shelf is uninterrupted bare stone. No human figures anywhere in the image. "
         "No buildings, cars, signs, paths or railings."),
        ("ground", [ground_ref], (1536,1536),
        "Create a flat overhead sandstone material texture filling the entire square image. "
        "The reference is a close crop of the approved scene's foreground rock: match only its pale "
        "warm colour and mineral character. Fine grains and subtle irregular hairline fissures, "
        "even diffuse light, no shading gradient, seamless continuous stone, no horizon or scenery. "
        "A real close-up photograph looking straight down at stone, not a landscape photograph."),
        ("person", [master, cfg["subject"]], (1344, 2048), context +
         "Image 1 supplies only the light and colour environment. Image 2 is the identity reference: "
         "preserve that person's exact face, hair, age, skin tone and clothing. Show just this one person "
         "standing naturally, facing the camera with a slight three-quarter body turn, arms relaxed, "
         "both feet planted. Full body including complete shoes; leave a clear margin on every side. "
         "Match image 1's soft directional daylight on their skin and clothing. Studio cutout on an "
         "absolutely uniform vivid cyan background, no scenery, no ground, no cast shadow, no text. "
         "The person occupies about 85 percent of the height; detailed face and fabric."),
        ("under_cloud", [str(output / "aerial_layout.png"),look_ref], (2688,1536),
        "A cinematic aerial photograph looking steeply down from 1500 metres over a vast canyon region. "
        "Use image 1 only for the aerial camera angle and broad distribution of mesas and ravines. "
        "Image 2 contains three material and atmosphere samples from the final destination: carry their "
        "cream and muted copper rock colours, storm-light and cool shadow into this new aerial view. "
        "Generate a single continuous photograph, not a collage. Tiny trees, broad sandstone benches, "
        "branching deep ravines, a few wisps of cloud passing close to the camera. Warm sun from upper left. "
        "No eye-level foreground, no enormous cliff facing the camera, no people. " + context),
        ("orbit", [str(output / "orbit_layout.png"),look_ref], (2688,1536),
        "A photograph from the International Space Station, 400 kilometres above Earth. Image 1 sets "
        "the planetary scale and curved horizon. Image 2 supplies only the final destination's colour "
        "and atmosphere samples; its local rock shapes must not be enlarged into the orbital view. "
        "Black space, delicate blue curved atmospheric limb, vast mostly flat arid southwestern US "
        "below a broad system of tiny white clouds, subtle cream and muted rusty regional colours. "
        "Fine river drainage barely visible through blue atmospheric haze, no individual mountains "
        "or cliff faces discernible. Sun from the left, a single photograph, no panels or text. " + context)]
    return [dict(name=name, references=references, width=size[0], height=size[1], models=cfg.get("models", {}),
                 context_parent=master if name != "master" else None,
                 prompt=prompt, seed=cfg["seed"] + i) for i, (name, references, size, prompt) in enumerate(definitions)]


def prepare_context(cfg: dict, output: Path):
    """Explicit, recorded crops convey appearance without imposing eye-level scale."""
    from PIL import Image, ImageOps
    master = Image.open(output / "master.png").convert("RGB")
    w,h = master.size
    crops = [(0.1,0.05,0.5,0.25), (.17,.32,.5,.55), (.3,.86,.62,.99)]
    board = Image.new("RGB", (1536,512))
    for i, box in enumerate(crops):
        region = master.crop(tuple(round(v*(w if k%2 == 0 else h)) for k,v in enumerate(box)))
        board.paste(ImageOps.fit(region,(512,512)),(i*512,0))
        if i == 2:
            region.save(output / "master_ground_reference.png")
    board.save(output / "master_look_reference.png")
    sources = {}
    for name,key in (("aerial", "aerial_layout"),("orbit", "orbit_layout")):
        if key not in cfg:
            raise ValueError(f"Set {key} to a reviewed camera-scale reference")
        image = Image.open(cfg[key]).convert("RGB")
        # Strip colour so the guide supplies camera/scale, not a second palette.
        ImageOps.grayscale(image).convert("RGB").save(output / f"{name}_layout.png")
        sources[name] = dict(path=cfg[key], sha256=digest(Path(cfg[key])), role="camera and scale only")
    (output / "context.json").write_text(json.dumps(dict(master_sha256=digest(output/"master.png"),
                crops=crops, layouts=sources, scope="Appearance inherited; geographic registration not asserted"), indent=2))


def generate_asset(url: str, spec: dict, output: Path) -> Path:
    """Cache only complete results, keyed by actual references, prompt and graph."""
    refs = [Path(p) for p in spec["references"]]
    graph = flux3.graph(spec["prompt"], spec["seed"], [p.name for p in refs],
                        f"touchdown/authored_{spec['name']}", spec["width"], spec["height"],
                        inpaint=Path(spec["inpaint"]).name if spec.get("inpaint") else None)
    for key, (node, field) in {"unet": ("1", "unet_name"), "clip": ("2", "clip_name"), "vae": ("3", "vae_name")}.items():
        if key in spec.get("models", {}):
            graph[node]["inputs"][field] = spec["models"][key]
    evidence = dict(spec=spec, graph=graph, references=[dict(path=str(p), sha256=digest(p)) for p in refs])
    if spec.get("inpaint"):
        evidence["inpaint_sha256"] = digest(Path(spec["inpaint"]))
    if spec.get("context_parent"):
        evidence["context_parent_sha256"] = digest(Path(spec["context_parent"]))
    key = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    record = output / f"{spec['name']}.json"
    target = output / f"{spec['name']}.png"
    if record.exists() and target.exists():
        cached = json.loads(record.read_text())
        if cached.get("key") == key and cached.get("output_sha256") == digest(target):
            print(f"[{spec['name']}] cached", flush=True)
            return target
        archive = output / "history" / f"{spec['name']}-{cached.get('key', 'unknown')[:12]}"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, archive / target.name)
        shutil.copy2(record, archive / record.name)
    for i, ref in enumerate(refs):
        # Content-address upload names: distinct assets with equal basenames cannot alias.
        upload = output / "uploads" / f"{digest(ref)[:20]}{ref.suffix}"
        upload.parent.mkdir(exist_ok=True)
        if not upload.exists():
            upload.write_bytes(ref.read_bytes())
        graph[f"r{i}"]["inputs"]["image"] = flux3._upload(url, upload)
    if spec.get("inpaint"):
        source = Path(spec["inpaint"])
        upload = output / "uploads" / f"{digest(source)[:20]}{source.suffix}"
        upload.parent.mkdir(exist_ok=True)
        if not upload.exists():
            upload.write_bytes(source.read_bytes())
        graph["m0"]["inputs"]["image"] = flux3._upload(url, upload)
    (output / f"{spec['name']}.workflow.json").write_text(json.dumps(graph, indent=2))
    started = time.monotonic()
    prompt_id = flux3._call(url, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    print(f"[{spec['name']}] queued {prompt_id}", flush=True)
    while time.monotonic() - started < 1200:
        result = flux3._call(url, f"/history/{prompt_id}").get(prompt_id, {})
        status = result.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"{spec['name']}: {json.dumps(status)}")
        if status.get("completed"):
            break
        time.sleep(1)
    else:
        raise TimeoutError(f"Timed out waiting for {spec['name']} ({prompt_id}); server job may still be running")
    item = result["outputs"]["16"]["images"][0]
    query = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
    with urllib.request.urlopen(f"{url}/view?{query}", timeout=60) as response:
        content = response.read()
    from PIL import Image
    import io
    with Image.open(io.BytesIO(content)) as im:
        if im.size != (spec["width"], spec["height"]):
            raise ValueError(f"Unexpected asset dimensions: {im.size}")
        im.verify()
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(target)
    record.write_text(json.dumps(dict(evidence, key=key, prompt_id=prompt_id,
                                     seconds=time.monotonic() - started, output_sha256=digest(target)), indent=2))
    print(f"[{spec['name']}] done in {time.monotonic()-started:.1f}s", flush=True)
    return target


def person_matte(source: Path, output: Path) -> dict:
    """Remove the deliberately uniform cyan backing, independent of coat colour.

    Require a background majority and an isolated central subject. Preserve holes
    between limbs; reject clipped assets rather than silently fixing anatomy.
    """
    import cv2
    import numpy as np
    from PIL import Image
    image = np.asarray(Image.open(source).convert("RGB"))
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cyan = ((hsv[:, :, 0] > 75) & (hsv[:, :, 0] < 105) &
            (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 100))
    border = np.concatenate([cyan[:10].ravel(), cyan[-10:].ravel(), cyan[:, :10].ravel(), cyan[:, -10:].ravel()])
    if border.mean() < .95:
        raise ValueError("Person plate is not isolated on a cyan background; inspect before rendering")
    mask = np.where(cyan, cv2.GC_BGD, cv2.GC_PR_FGD).astype(np.uint8)
    core = cv2.erode((~cyan).astype(np.uint8), np.ones((5, 5), np.uint8))
    mask[core.astype(bool)] = cv2.GC_FGD
    cv2.grabCut(image, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
    if n < 2:
        raise ValueError("No person in plate")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = (labels == component).astype(np.uint8) * 255
    ys, xs = np.where(alpha)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
    if min(x0, y0, w-x1, h-y1) < 8 or y1-y0 < .5*h:
        raise ValueError("Person is clipped or too small in the plate")
    alpha = cv2.GaussianBlur(alpha, (3, 3), .5)
    # The synthetic backing can contaminate hair-edge pixels. Despill only the
    # boundary, retaining the reference's interior skin and clothing colours.
    edge = cv2.dilate((alpha < 255).astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
    corrected = image.copy().astype(np.float32)
    spill = np.maximum(0, np.minimum(corrected[:,:,1],corrected[:,:,2])-corrected[:,:,0]) * edge
    corrected[:,:,1] -= spill
    corrected[:,:,2] -= spill
    rgba = np.dstack([np.clip(corrected,0,255).astype(np.uint8), alpha])[y0:y1, x0:x1]
    Image.fromarray(rgba).save(output)
    return dict(source=str(source), bounds=[x0, y0, x1, y1], size=[x1-x0, y1-y0],
                background_border_fraction=float(border.mean()), subject_height_m=1.72,
                limitation="Static photographic person on a plane; no side/rear view or body animation")


def camera_pose(t: float) -> dict:
    """A bounded descending dolly with zero velocity at the finish."""
    if not 0 <= t <= 1:
        raise ValueError("Camera progress must be in [0, 1]")
    remaining = (1-t)**2.3
    distance = 2.6 + 35.4*remaining
    return dict(position=[0.0, -distance, 1.52 + 7.0*remaining], target=[0.0, 0.0, 1.16],
                focal_length_mm=35.0)


def write_attribution(cfg: dict, output: Path):
    rows = ["# Location references", "", "Authored cinematic interpretation; exact topography is not claimed.", ""]
    for r in cfg["references"]:
        rows.append(f"- [{r['title']}]({r['commons_page']}) — {r['artist']}; {r['license']}. SHA256: `{r['sha256']}`")
    (output / "ATTRIBUTION.md").write_text("\n".join(rows) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["plan", "assets", "render", "all"], default="all")
    p.add_argument("--url", default="http://127.0.0.1:8192")
    p.add_argument("--only", choices=["master", "ground", "person", "under_cloud", "orbit"])
    p.add_argument("--blender", default="blender")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--preview", action="store_true", help="render five camera samples rather than a clip")
    a = p.parse_args()
    if a.frames < 2 or a.width < 16 or a.height < 16 or a.width % 2 or a.height % 2:
        p.error("Use at least two frames and positive even output dimensions")
    cfg = read_scene(a.manifest)
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    plan = asset_plan(cfg, output)
    (output / "scene_plan.json").write_text(json.dumps(dict(config=cfg, assets=plan), indent=2))
    write_attribution(cfg, output)
    if a.stage in ("assets", "all"):
        for spec in plan:
            if a.only is None or spec["name"] == a.only:
                if spec["name"] != "master":
                    prepare_context(cfg, output)
                generate_asset(a.url, spec, output)
    if a.stage in ("render", "all"):
        for name in ("master", "ground", "person"):
            if not (output / f"{name}.png").is_file():
                raise ValueError(f"Generate {name} before rendering")
        matte = person_matte(output / "person.png", output / "person_rgba.png")
        poses = [camera_pose(i/(a.frames-1)) for i in range(a.frames)]
        render = dict(width=a.width, height=a.height, frames=a.frames, fps=24, poses=poses,
                      matte=matte, preview=a.preview, assets={name: digest(output/f"{name}.png")
                                                           for name in ("master", "ground", "person")})
        (output / "render.json").write_text(json.dumps(render, indent=2))
        script = Path(__file__).with_name("authored_scene_blender.py")
        subprocess.run([a.blender, "-b", "-t", "8", "--factory-startup", "--python", str(script), "--", str(output)], check=True)
        if not a.preview:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "24",
                            "-start_number", "0", "-i", str(output / "frames" / "%04d.png"),
                            "-frames:v", str(a.frames), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(output / "approach.mp4")], check=True)
            print(output / "approach.mp4", flush=True)


if __name__ == "__main__":
    main()
