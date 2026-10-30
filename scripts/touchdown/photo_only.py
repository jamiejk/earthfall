"""Photo-referenced FLUX scenes and plain image-conditioned LTX, without 3D guides.

The approved input boundary is the reviewed Commons photographs and the subject
photo. Every generated image must descend from those roots within this output
directory; earlier rendered or render-conditioned experiments are not inputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.authored_scene import digest, generate_asset, read_scene, write_attribution


def asset_plan(cfg: dict, output: Path) -> list[dict]:
    photos = [r["local_path"] for r in cfg["references"]]
    # A new location can supply its own photographed arrival and flight brief.
    # The single-photo route needs only this arrival, not invented aerial pins.
    if "arrival" in cfg.get("prompts", {}):
        plan = [dict(name="arrival", references=[*photos, cfg["subject"]], width=2688, height=1536,
                     models=cfg["models"], seed=cfg["seed"], prompt=cfg["prompts"]["arrival"])]
        if cfg.get("video", {}).get("scale_anchors"):
            # High is photo-informed text generation: close-up landmark pixels
            # copied their camera distance despite explicit altitude prompts.
            plan += [dict(name="high", references=[], width=2688, height=1536,
                          **({"context_parent": cfg["regional_context"]} if cfg.get("regional_context") else {}),
                          models=cfg["models"], seed=cfg["seed"]+1, prompt=cfg["prompts"]["high"]),
                     # Orbit is text-only too: from high.png it kept the 30-60 km oblique and only added a limb.
                     dict(name="orbit", references=[], width=2688, height=1536,
                          **({"context_parent": cfg["regional_context"]} if cfg.get("regional_context") else {}),
                          models=cfg["models"], seed=cfg["seed"]+2, prompt=cfg["prompts"]["orbit"])]
        return plan
    final = str(output / "arrival.png")
    approach = str(output / "approach.png")
    high = str(output / "high.png")
    look = cfg["look"]
    definitions = [
        ("arrival", [*photos, cfg["subject"]],
         "Create a single beautiful real travel photograph at a Canyonlands overlook. "
         "Images 1 to 3 are actual location references: draw on their natural rock erosion, broad "
         "connected mesas, talus slopes and scrub, without copying their exact composition. "
         "Image 4 identifies the woman: preserve her face, short dark curls, red raincoat, grey hoodie "
         "and black trousers. Photograph her naturally within the landscape, with the same light "
         "on her and the rocks, realistic contact and depth. A relaxed three-quarter-length portrait "
         "from the knees up, woman centred slightly right, head near the top quarter of the frame. "
         "Behind her a pale irregular sandstone ridge curves from lower centre toward the right, "
         "with sparse low scrub and a deep canyon to the left. A broad layered copper mesa beyond "
         "the canyon, fading distant ridges, broken moving storm clouds with blue openings. "
         "A natural, inviting expression, a slight turn of her shoulders, loose coat and curls "
         "caught by a breeze. A 35mm environmental portrait with clear surroundings and detailed "
         "skin. One integrated photograph, no composited cutout, no isolated rock pillars. " + look),
        ("approach", [final],
         "Photograph the setting of image 1 from a new camera position far above and in front of "
         "the woman, retreating diagonally along the pale ridge. A drone camera is now 180 metres "
         "above the ridge looking DOWN at 45 degrees. The same woman is a tiny red figure near the "
         "centre-right on the pale ridge; her entire body occupies less than one percent of the "
         "picture height. Reveal the full extent of her connected sandstone promontory, with "
         "irregular benches, talus and scrub, and a deep branching canyon to the left. The large "
         "copper mesa belongs in the far background. The entire image is landscape seen from "
         "above; terrain reaches the top edge, no sky. This is an aerial establishing photograph, "
         "with hundreds of metres of ground visible, not a portrait or a close-up of a cliff. "
         "Preserve the reference's geological character, direction of sunlight and natural colour. "
         "Invent plausible surroundings to form one coherent landscape. " + look),
        ("high", [approach],
         "Move the aerial camera of image 1 much farther back and higher: 1500 metres above the "
         "same canyon region, looking down at 45 degrees, a real high-altitude oblique aerial "
         "photograph. The reference's pale curved promontory is now a small feature at centre-right "
         "among a much larger branching system of mesas, canyons and drainage. Several kilometres "
         "of continuous connected landscape. No human figure can be resolved at this altitude. "
         "Cloud wisps pass very close to the camera at the upper and left edges, revealing the "
         "terrain far below through an opening. Sky is outside the frame. Atmospheric blue-grey "
         "depth softens the distant ground. Preserve the geological language and sunlight of the "
         "reference while freely expanding its surroundings. " + look),
        ("cloud", [high],
         "A real aerial photograph above the weather system over the desert region in image 1. "
         "The camera looks diagonally down at 45 degrees from eight kilometres altitude. Large "
         "soft three-dimensional white cloud billows pass close to the lens and fill most of the "
         "frame; through a small central opening we glimpse the reference's muted copper and pale "
         "desert far, far below, softened by blue haze. This is a cloudscape, with the land a tiny "
         "distant hint, not a canyon at eye level. Warm white sunlit cloud edges on the left, cool "
         "blue-grey shadow underneath. Natural photographic texture and translucent vapour, "
         "one coherent photograph, no text, no collage."),
        ("orbit", [high],
         "A real photograph from low Earth orbit, 400 kilometres above the southwestern United "
         "States, looking obliquely toward the curved blue atmospheric limb. Black space fills "
         "the top quarter. A vast, mostly flat arid continent fills the rest, crossed by a broad "
         "weather system of white clouds. The reference image supplies ONLY regional colour and "
         "sunlight: muted cream and copper land under blue haze, warm light from the left. Its "
         "local cliffs and landforms are microscopic at this planetary scale and cannot be "
         "seen individually. Delicate drainage patterns at continental scale, photographic "
         "subtlety, natural atmosphere, no towering cliffs, no giant mountains, no text.")]
    return [dict(name=name, references=refs, width=2688, height=1536,
                 models=cfg["models"], seed=cfg["seed"] + i, prompt=prompt)
            for i, (name, refs, prompt) in enumerate(definitions)]


def verify_lineage(cfg: dict, output: Path, plan: list[dict]) -> dict:
    """Verify ancestry, including cached parent pixels, before using any asset."""
    allowed = {str(Path(p).resolve()) for p in
               [cfg["subject"], *(r["local_path"] for r in cfg["references"])]}
    roots = {p: digest(Path(p)) for p in sorted(allowed)}
    declared = set(allowed)
    lineage = {}
    for spec in plan:
        refs = [str(Path(p).resolve()) for p in spec["references"]]
        if not set(refs) <= declared:
            raise ValueError(f"Unapproved reference ancestry: {spec['name']}")
        record = output / f"{spec['name']}.json"
        image = output / f"{spec['name']}.png"
        declared.add(str(image.resolve()))
        if not record.exists() or not image.exists():
            continue
        if not set(refs) <= allowed:
            raise ValueError(f"Unverified parent image: {spec['name']}")
        evidence = json.loads(record.read_text())
        if evidence["spec"] != spec:
            raise ValueError(f"Stale generation specification: {spec['name']}")
        if evidence["output_sha256"] != digest(image):
            raise ValueError(f"Changed generated image: {image}")
        expected = [{"path": p, "sha256": digest(Path(p))} for p in spec["references"]]
        if evidence["references"] != expected:
            raise ValueError(f"Changed reference pixels: {spec['name']}")
        lineage[spec["name"]] = dict(sha256=digest(image), references=expected)
        if spec.get("context_parent"):
            context_hash = digest(Path(spec["context_parent"]))
            if evidence.get("context_parent_sha256") != context_hash:
                raise ValueError(f"Changed photographic context: {spec['name']}")
            lineage[spec["name"]]["context_parent_sha256"] = context_hash
        allowed.add(str(image.resolve()))
    return dict(root_photographs=roots, generated=lineage, geometry_guides=False)


def board(output: Path):
    from PIL import Image, ImageDraw, ImageOps
    names = ["orbit", "cloud", "high", "approach", "arrival"]
    canvas = Image.new("RGB", (960, 5 * 580), "#191919")
    for i, name in enumerate(names):
        with Image.open(output / f"{name}.png") as im:
            canvas.paste(ImageOps.fit(im.convert("RGB"), (960, 548)), (0, i * 580 + 28))
        ImageDraw.Draw(canvas).text((12, i * 580 + 8), name, fill="white")
    canvas.save(output / "photo-only-board.jpg", quality=92)


def video(cfg, output, plan, url, upscale, pin_high=False, retreat=False):
    # No layout renderer, depth, mesh, video guide or IC-LoRA reference enters
    # this graph. Only FLUX photographs condition the video. The approach image
    # informs the high view's ancestry rather than imposing another pose pin.
    import fakeout_earth_zoom as f
    from types import SimpleNamespace
    lineage = verify_lineage(cfg, output, plan)
    if len(lineage["generated"]) != len(plan):
        raise ValueError("Generate and verify every photo first")
    options = cfg.get("video", {})
    frames = int(options.get("frames", 177))
    width, height = int(options.get("width", 960)), int(options.get("height", 544))
    if frames < 9 or frames % 8 != 1 or min(width, height) < 512 or width % 32 or height % 32:
        raise ValueError("Use 8n+1 frames and video dimensions divisible by 32, at least 512 pixels")
    if not retreat and not {"orbit", "cloud", "high", "arrival"} <= {s["name"] for s in plan}:
        raise ValueError("This scene has a single photographic arrival; use --route retreat")
    pins = [(0, "orbit"), (32, "cloud"), *([(64, "high")] if pin_high else []), (144, "arrival")]
    prompt = (
        "One continuous fast cinematic camera flight from low Earth orbit down to a woman at a "
        "Canyonlands overlook. Real live-action aerial footage, constant focal length. The CAMERA "
        "travels physically forward and downward; cloud masses rush past the lens with strong "
        "foreground parallax. The opening plunges rapidly into dense rolling white cloud, which "
        "briefly obscures the land completely. Then the camera emerges well above the canyon and "
        "continues a fast forward and downward 45-degree descent along one heading toward the pale "
        "ridge. The same naturally eroded cream sandstone promontory stays connected to its canyon "
        "and the muted copper mesas. Nearby ground flows rapidly down and out of frame, distant "
        "mesas move more slowly. Moving wisps and billowing clouds, warm light from the left. The "
        "camera slows smoothly only near the woman and levels out into her environmental portrait. "
        "She breathes, blinks, and turns her head slightly toward the arriving camera; a breeze "
        "visibly moves her curls and red raincoat. Preserve her likeness and natural integration "
        "with the light and landscape. Stable coherent geography throughout the visible approach. "
        "No cuts, no crossfades, no dissolves, no camera orbit, no lens zoom, no frozen person."
    )
    if retreat:
        pins = [(0, "arrival")]
        prompt = (
            "One continuous live-action aerial camera shot beginning with this environmental portrait "
            "of the woman in a red raincoat on a Canyonlands overlook. The camera immediately flies "
            "BACKWARD AND UPWARD at 45 degrees, rapidly accelerating, with a fixed focal length. "
            "Strong true perspective and foreground parallax: the pale rock rim and scrub move past "
            "the lens as hundreds of metres of naturally eroded canyon landscape are revealed. "
            "The woman becomes a tiny red speck on the same connected sandstone promontory. She "
            "breathes quietly, her curls and loose red coat moving gently in the breeze. The camera "
            "continues its fast diagonal retreat several kilometres up over muted copper mesas "
            "and deep ravines, through a dense rolling cloud bank which completely surrounds the "
            "lens briefly. Still travelling rapidly backward and upward, emerge above the clouds "
            "and keep climbing until the curved blue limb of Earth is revealed against black space. "
            "The final view is low Earth orbit over the arid southwestern United States. Warm sun "
            "from the left, cool blue-grey clouds and shadows, restrained natural colour. One "
            "uninterrupted physical flight through one coherent photographic world. No cuts, no "
            "dissolves, no morphing rocks, no hovering, no scene changes, no lens zoom."
        )
        prompt = cfg.get("prompts", {}).get("retreat", prompt)
    suffix = "-retreat" if retreat else "-high-pin" if pin_high else ""
    folder = output / (("video-upscaled" if upscale else "video-draft") + suffix)
    folder.mkdir(exist_ok=True)
    args = SimpleNamespace(no_upscale=not upscale)
    with f._client(url) as client:
        graph = {}
        loaded = [(frame, f._load(graph, client, str(44 + i), output / f"{name}.png"))
                  for i, (frame, name) in enumerate(pins)]
        seed = cfg["seed"] + (21 if retreat else 20)
        wanted = f._ltx(graph, args, frames, loaded, prompt, seed, "touchdown/photo_only")
        graph["308"]["inputs"].update(width=width, height=height)
        forbidden = {"LoadVideo", "LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide"}
        if any(n["class_type"] in forbidden for n in graph.values()):
            raise ValueError("Non-photographic conditioning entered the graph")
        (folder / "lineage.json").write_text(json.dumps(lineage, indent=2))
        (folder / "parameters.json").write_text(json.dumps(dict(
            frames=frames, fps=24, pins=pins, prompt=prompt, final_free_frames=0 if retreat else 32,
            seed=seed, upscale=upscale, reversed_for_descent=retreat,
            base_width=width, base_height=height), indent=2))
        saved = f._submit(client, graph, tuple(wanted), folder)
    destination = folder / "earth_zoom.mp4"
    if retreat:
        import subprocess
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", saved[f.LTX_VIDEO],
                        "-vf", "reverse", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(destination)], check=True)
    else:
        shutil.copy2(saved[f.LTX_VIDEO], destination)
    workflow = Path(saved[f.LTX_VIDEO]).parent / "graph.json"
    (folder / "result.json").write_text(json.dumps(dict(
        video=str(destination), sha256=digest(destination), generation=str(saved[f.LTX_VIDEO]),
        workflow=str(workflow), workflow_sha256=digest(workflow)), indent=2))
    print(destination, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("shots/authored/canyonlands-photo-only.toml"))
    p.add_argument("--output", type=Path, default=Path("outputs/touchdown/photo-only-canyonlands"))
    p.add_argument("--stage", choices=["stills", "audit", "video"], default="stills")
    p.add_argument("--only", choices=["arrival", "approach", "high", "cloud", "orbit"])
    p.add_argument("--url")
    p.add_argument("--upscale", action="store_true")
    p.add_argument("--route", choices=["retreat", "forward", "high-pin"], default="retreat",
                   help="Use one photo-conditioned flight, or reproduce a rejected multi-image comparison")
    a = p.parse_args()
    cfg = read_scene(a.manifest)
    if any(k in cfg for k in ("aerial_layout", "orbit_layout")):
        p.error("Photo-only manifests cannot contain rendered layout inputs")
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = asset_plan(cfg, output)
    if a.stage == "stills":
        write_attribution(cfg, output)
        for i, spec in enumerate(plan):
            if a.only and spec["name"] != a.only:
                continue
            verify_lineage(cfg, output, plan[:i])
            generate_asset(a.url or "http://127.0.0.1:8192", spec, output)
        if len(plan) > 1 and all((output / f"{s['name']}.png").exists() for s in plan):
            board(output)
    elif a.stage == "video":
        video(cfg, output, plan, a.url or "http://127.0.0.1:8193", a.upscale,
              pin_high=a.route == "high-pin", retreat=a.route == "retreat")
    audit = verify_lineage(cfg, output, plan)
    (output / "lineage.json").write_text(json.dumps(audit, indent=2))
    print(f"Verified {len(audit['generated'])} generated images from {len(audit['root_photographs'])} photographic roots", flush=True)


if __name__ == "__main__":
    main()
