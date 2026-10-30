"""Run a GeoBlock shot through the Comfy SDK and download its outputs.

By default the target is chosen by the SDK's own environment: ``COMFY_BASE_URL``
for a self-hosted ComfyUI behind comfy-api-proxy (e.g. http://127.0.0.1:8189)
or a serverless deployment, unset for Comfy Cloud with ``COMFY_API_KEY``.

``--blocking-url`` and ``--h3-url`` split the shot across two servers: the
blocking video renders where GeoBlock and Blender are installed, then only the
H3 stage runs on the second target (``cloud`` means Comfy Cloud), with the
control video, opening frame and photos uploaded as SDK assets.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys

from .camera import PRESETS
from .camera import h3_frame_count
from .workflow import (KEYFRAME_MODELS, LTX_VERSIONS, CONTROL_VIDEO, DEPTH_VIDEO, LTX_KEYFRAMES, LTX_VIDEO, STAGE1_VIDEO, SUBJECT_SHEET, ltx_graph, H3_MODELS, H3_TURBO_LORA, H3_VIDEO, OPENING_FRAME, ROUTE_MAP, ShotRequest,
                       blocking_graph, h3_from_files_graph, h3_graph)
from .upscale import FINAL_VIDEO, TARGETS, UPSCALERS, upscale_file_graph
from .weather import DEFAULT_WEATHER, WEATHER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoblock-run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    place = parser.add_argument_group("place (catalog, search, or name + coordinates)")
    place.add_argument("--place", metavar="ID", help="Landing place from the catalog (see --list-places): sets the "
                       "coordinates, the view heading, palette and the measured look")
    place.add_argument("--list-places", nargs="?", const="", metavar="FILTER",
                       help="List catalog places (optionally matching FILTER, e.g. 'utah' or 'summit') and exit")
    place.add_argument("--place-info", metavar="ID", help="Show one catalog place's terrain, lidar, imagery and "
                       "licences and exit")
    place.add_argument("--name", default="", help="Landmark name used in reports and prompts")
    place.add_argument("--search", help="Search text; picks --result-number from the matches")
    place.add_argument("--result-number", type=int, default=1)
    place.add_argument("--latitude", type=float)
    place.add_argument("--longitude", type=float)
    place.add_argument("--radius-km", type=float, default=12)
    place.add_argument("--landing-lidar", default="none", metavar="auto|none|PROJECT",
                       help="US only: an 800 m square of USGS 3DEP 1 m lidar terrain where the Earth zoom lands "
                            "('auto' picks the newest project). Catalog overlooks then stand on the brink, which the "
                            "10 m DEM rounds off")
    shot = parser.add_argument_group("shot")
    shot.add_argument("--preset", choices=list(PRESETS), default="Rim reveal")
    shot.add_argument("--duration", type=float, default=10)
    shot.add_argument("--heading", type=float, default=-1, help="Degrees from north; -1 = toward the drop")
    shot.add_argument("--scale-m", type=float, default=0)
    shot.add_argument("--lens-mm", type=float, default=24)
    shot.add_argument("--engine", choices=("eevee", "workbench", "cycles"), default="eevee")
    shot.add_argument("--palette", choices=("default", "grand-canyon"), default="default",
                      help="Blender terrain colours (grand-canyon follows the canyon rock bands)")
    shot.add_argument("--weather", choices=tuple(WEATHER), default=DEFAULT_WEATHER,
                      help="Sky, sun and haze in the blocking render and the H3 prompt")
    shot.add_argument("--imagery", action="store_true",
                      help="Drape NASA HLS 30 m satellite imagery (public domain, via GIBS) on the terrain")
    shot.add_argument("--trees", choices=("off", "auto", "lidar", "meta"), default="off", nargs="?", const="auto",
                      help="Stand a proxy on every measured tree near the camera path (USGS 3DEP lidar where "
                           "surveyed, else Meta/WRI canopy height). --trees alone means auto.")
    shot.add_argument("--buildings", choices=("auto", "on", "off"), default="auto", nargs="?", const="on",
                      help="Earth zoom: a block on every Overture Maps building footprint within 7 km, at its "
                           "measured height. auto (default): on for catalog places measured as cities. "
                           "--buildings alone means on.")
    shot.add_argument("--no-tree-prefetch", action="store_true",
                      help="Let ComfyUI fetch the trees itself instead of this environment")
    shot.add_argument("--width", type=int, default=1280)
    shot.add_argument("--height", type=int, default=720)
    shot.add_argument("--samples", type=int, default=16)
    ltx = parser.add_argument_group("LTX render-to-real (photoreal keyframes + LTX 2.3 3DREAL)")
    ltx.add_argument("--ltx", action="store_true",
                     help="Repaint the first and last blocking frames with Qwen-Image-Edit, then follow the "
                          "blocking with LTX 2.3 + 3DREAL between them")
    ltx.add_argument("--keyframes", choices=KEYFRAME_MODELS, default="qwen",
                     help="Keyframe painter: Qwen-Image-Edit 2511 + Lightning, or FLUX.2 [klein] 9B distilled")
    ltx.add_argument("--colour-match", type=float, default=0.6, metavar="0-1",
                     help="Pull each keyframe's colour toward its blocking frame so the grade stays continuous (0 = off)")
    ltx.add_argument("--colour-smooth", type=float, default=None, metavar="SECONDS",
                     help="Smooth the video's colour cast over this many seconds (default 0.75 for the Earth zoom, else off)")
    ltx.add_argument("--ltx-version", choices=LTX_VERSIONS, default="2.3",
                     help="2.3: LTX 2.3 + 3DREAL; 2.5: LTX 2.5 + Lightricks' Layout To Render IC-LoRA")
    ltx.add_argument("--no-upscale", action="store_true", help="Skip the 2x latent upscale and refine pass")
    ltx.add_argument("--subject", type=Path, help="Photo of the person the Earth zoom lands on (uploaded)")
    ltx.add_argument("--free-ending", action="store_true",
                     help="LTX 2.5: don't pin the last frame to a keyframe, so the end follows the layout's motion "
                          "(the Earth zoom already doesn't)")
    ltx.add_argument("--pin-ending", action="store_true",
                     help="LTX 2.5: pin the last frame to its keyframe even for the Earth zoom")
    ltx.add_argument("--subject-description", default="",
                     help="Describe an invented person when there is no --subject photo (default: a hiker "
                          "dressed for the --weather)")
    ltx.add_argument("--no-subject-sheet", action="store_true",
                     help="klein: give the person keyframes the raw --subject photo instead of first painting one "
                          "clean full-length picture of the person")
    ltx.add_argument("--orbit-from-blocking", action=argparse.BooleanOptionalAction, default=True,
                     help="Earth zoom: pin frame 0 to the blocking frame (real satellite imagery) as well as klein's "
                          "repaint, which stays the look reference; on by default (klein drew a line across the globe)")
    ltx.add_argument("--place-subject", action="store_true",
                     help="Earth zoom: paste the person over the grey stand-in, at its size, before the keyframe "
                          "painter repaints eye-level frames (keeps the push-in's framing)")
    ltx.add_argument("--hold-follow", choices=("crop", "reproject", "chain"), default="crop",
                     help="Earth zoom: later hold keyframes start from the first one cropped to the person's growth "
                          "(crop, a zoom) or moved through the terrain to their camera (reproject, a dolly); chain "
                          "reprojects each from the hold keyframe before it (shorter moves, less stretched ground)")
    ltx.add_argument("--ground-repaint", type=float, default=0.0,
                     help="with --hold-follow reproject: repaint the near ground the reprojection stretched, through "
                          "its mask, at this denoise (e.g. 0.85); 0 leaves it blurred (r23)")
    ltx.add_argument("--save-stage1", action="store_true",
                     help="LTX 2.5: also save stage 1 decoded (<prefix>_stage1), to tell stage-1 faults from the "
                          "refine's")
    ltx.add_argument("--dive-tempo", choices=["standard", "low-swoop", "smooth-swoop"], default="standard",
                     help="Earth zoom: low-swoop crosses orbit and cloud sooner and spends the frames on the low "
                          "swoop (~8%% picture growth a frame instead of ~11%%), which LTX smears when fast; "
                          "smooth-swoop is low-swoop braking into a push-in that keeps moving, riding a smoothed "
                          "envelope of the ground instead of every bump")
    ltx.add_argument("--swoop-keyframe-denoise", type=float, default=1.0,
                     help="klein, Earth zoom swoop keyframes (below the cloud, above the landing): below 1 they start "
                          "from the blocking frame, keeping its framing (e.g. 0.85); 1 paints from scratch")
    ltx.add_argument("--landing-from-hold", action="store_true",
                     help="Earth zoom with --composite-subject: paint the landing plate (f88) from the hold's first "
                          "plate moved back through the terrain to its camera, at --landing-plate-denoise (0.8 "
                          "suggested), so the two show the same trees in the same places")
    ltx.add_argument("--landing-fill", choices=("around", "blocking"), default="around",
                     help="with --landing-from-hold: what goes where the hold plate doesn't reach the landing "
                          "camera's view: a fill from around it, or the landing frame's own blocking (its road)")
    ltx.add_argument("--exit-keyframe-denoise", type=float, default=None,
                     help="with --exit-keyframe repaint: how far klein repaints the blocking's deck-exit frame "
                          "(default 0.6, which keeps the cleared frame's grey CG city)")
    ltx.add_argument("--landing-repaint", type=float, default=0.0,
                     help="with --landing-from-hold: repaint the part of the landing plate the hold plate doesn't "
                          "reach (the --landing-fill), through its mask, at this denoise (e.g. 0.95; 0: off)")
    ltx.add_argument("--hold-near-repaint", type=float, default=0.0, metavar="METRES",
                     help="Earth zoom with --hold-follow reproject: later hold plates also blur and (with "
                          "--ground-repaint) repaint the ground within this many metres of their camera, where near "
                          "bushes moved as flat ground shear into streaks (0: off)")
    ltx.add_argument("--hold-keep", type=float, default=0.0,
                     help="Earth zoom with --hold-follow reproject and --composite-subject: paste the first hold "
                          "plate, moved to each later hold keyframe's camera, back over its repaint this strongly "
                          "where it covers the view unstretched, so the midground doesn't change between keyframes "
                          "(0: off, 1: fully)")
    ltx.add_argument("--landing-plate-denoise", type=float, default=1.0,
                     help="klein, Earth zoom with --composite-subject: the same for the landing's descent plate only "
                          "(f88), leaving the swoop keyframes painted from scratch")
    ltx.add_argument("--keyframe-denoise", type=float, default=1.0,
                     help="klein, Earth zoom eye-level keyframes: below 1 the painter starts from the blocking frame "
                          "(with the pasted person) instead of a blank, so it keeps the frame's layout and the "
                          "person's size; 1 paints from scratch with the frame only as a reference")
    ltx.add_argument("--composite-subject", action="store_true",
                     help="Earth zoom, klein: paint each eye-level keyframe's clean plate (no stand-in) at full "
                          "strength, paste the person at the stand-in's place and size, then blend them in with a "
                          "klein pass masked to them (--keyframe-denoise, or 0.65 if left at 1)")
    ltx.add_argument("--hold-plate-denoise", type=float, default=0.6,
                     help="with --composite-subject: later hold plates repaint the first plate, cropped to their "
                          "framing, this far (1 paints each afresh)")
    ltx.add_argument("--closing-plate-denoise", type=float, default=0.9,
                     help="with --composite-subject: the same for the closing close-up's plate")
    ltx.add_argument("--person-detector", default="",
                     help="with --composite-subject: a local Florence-2 folder; people it finds on a painted plate "
                          "are repainted away before the subject is pasted in")
    ltx.add_argument("--punch-through", action="store_true",
                     help="Earth zoom: turn LTX's cut from cloud to clear ground into a fast bright dissolve (it "
                          "will not paint the part-cleared frames between)")
    ltx.add_argument("--stretch-grain", type=float, default=0.0, metavar="STD",
                     help="Earth zoom, --hold-follow reproject: add noise of this strength (0.08 suggested) to the "
                          "blurred near ground of later hold plates; klein kept the plain blur as defocus (r34)")
    ltx.add_argument("--swoop-gap", type=int, default=0, metavar="FRAMES",
                     help="Earth zoom: add keyframes between the cloud exit and the hold until no two guides are "
                          "more than FRAMES apart (16 suggested; the terraces smeared across 24-frame gaps)")
    ltx.add_argument("--exit-keyframe", choices=("none", "blocking", "repaint"), default="none",
                     help="Earth zoom: a keyframe where the camera leaves the cloud deck, pinned to the blocking's "
                          "half-cleared haze or repainted from it (r22: LTX cut to clear air there ahead of the haze)")
    ltx.add_argument("--cloud-keyframe", choices=("none", "blocking", "repaint"), default="blocking",
                     help="Earth zoom: the cloud-dive keyframe pinned to the blocking frame (default), left out "
                          "(r15: LTX skipped the cloud with a cut at f23/24), or repainted (froze the dive in r5)")
    ltx.add_argument("--keyframe-refine", type=float, default=0.0,
                     help="with --keyframe-denoise below 1: repaint each eye-level keyframe once more at this "
                          "denoise (e.g. 0.6), for realism on top of the kept layout; 0 = one pass")
    ltx.add_argument("--hold-keyframes", choices=("crop", "sharp", "drop", "dense", "direct"), default="crop",
                     help="Earth zoom keyframes after the first one at eye level: repainted from a crop of the first "
                          "(crop), from a Real-ESRGAN upscaled crop (sharp), left out (drop), sharp crops every 32 frames (dense), or the upscaled crop "
                          "itself with no repaint (direct)")
    ltx.add_argument("--keyframes-only", action="store_true",
                     help="Stop after the photoreal keyframes (and subject sheet): a quick look check, no LTX")
    final = parser.add_argument_group("final upscale (after --ltx, or on a saved video with --upscale-video)")
    final.add_argument("--final-upscale", choices=("none", *UPSCALERS), default="none",
                       help="Last pass on the LTX frames: seedvr2 (SeedVR2 3B), seedvr2-7b, flashvsr (FlashVSR "
                            "v1.1), ltx-detail (LTX 2.5 Refine Details, tiled; slow) or esrgan (Real-ESRGAN, "
                            "frame by frame)")
    final.add_argument("--final-target", choices=TARGETS, default="4k",
                       help="4k doubles the frames (3840x2176); 1080p keeps the size and rebuilds the detail")
    final.add_argument("--upscale-video", type=Path,
                       help="Only run the final upscale on this video (uploaded); no place or shot needed")
    final.add_argument("--motion-blur", choices=("none", "targeted"), default="none",
                       help="Earth zoom: after --ltx, also write <video>_mblur.mp4 with motion blur over the swoop "
                            "and landing only (motion_blur.py), for the upscale to start from")
    final.add_argument("--motion-blur-video", type=Path,
                       help="Only apply --motion-blur targeted to this saved Earth zoom video (uses --duration and "
                            "--dive-tempo for its frames); no server needed")
    final.add_argument("--upscale-chunk", type=int, default=64, metavar="FRAMES",
                       help="--upscale-video in jobs of this many frames, joined with ffmpeg (0: one job). The "
                            "Real-ESRGAN 4K of all 241 frames went over the 8190 service's 45 GB memory cap")
    h3 = parser.add_argument_group("MiniMax H3 (omit --h3 to stop at the blocking video)")
    h3.add_argument("--h3", action="store_true", help="Generate the final video with H3 Ref2VA")
    h3.add_argument("--photo", type=Path, action="append", default=[],
                    help="Licensed place photo to upload as an H3 reference (repeatable, up to 8)")
    h3.add_argument("--auto-photos", type=int, default=0, metavar="N",
                    help="Also fetch N openly licensed Wikimedia Commons photos near the coordinates")
    h3.add_argument("--photo-radius-km", type=float, default=2)
    h3.add_argument("--exclude-photo", type=int, action="append", default=[], metavar="PAGE_ID",
                    help="Commons page ID to skip in --auto-photos (repeatable)")
    h3.add_argument("--prompt", default="")
    h3.add_argument("--seed", type=int, default=20261005)
    h3.add_argument("--steps", type=int, help="Sampling steps (default 12, or 4 with --turbo)")
    h3.add_argument("--turbo", action="store_true", help=f"Use the 4-step Ref2VA LoRA ({H3_TURBO_LORA})")
    h3.add_argument("--lora", default="", help="Other model-only LoRA file name on the H3 server")
    h3.add_argument("--no-opening-reference", action="store_true",
                    help="Do not pass the clay opening frame as a reference image")
    h3.add_argument("--depth-strength", type=float, default=0.0,
                    help="Render a depth pass and drive H3 Fun ControlNet with it (try 0.6-1.0)")
    h3.add_argument("--depth-end", type=float, default=0.6, help="Fraction of the schedule the control covers")
    h3.add_argument("--first-frame", type=Path, help="Image anchored at frame 0 (MiniMaxH3AddGuide)")
    for key, value in H3_MODELS.items():
        h3.add_argument(f"--h3-{key}", default=value, help=f"H3 {key} file name on the H3 server")
    targets = parser.add_argument_group("targets (default: COMFY_BASE_URL, or Comfy Cloud when unset)")
    targets.add_argument("--blocking-url", help="Server with GeoBlock nodes and Blender")
    targets.add_argument("--h3-url", help="Server for the H3 stage; 'cloud' for Comfy Cloud. Implies two stages.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--save-graph", type=Path, help="Write the API-format graph here and do not submit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.upscale_video:
        return _upscale_video(args)
    if args.motion_blur_video:
        request = ShotRequest(name="motion blur", latitude=0.0, longitude=0.0, preset="Earth zoom",
                              duration_s=args.duration, dive_tempo=args.dive_tempo)
        print(f"Saved {_motion_blur(request, args.motion_blur_video)}")
        return 0
    if args.final_upscale != "none" and not args.ltx:
        raise SystemExit("--final-upscale applies to --ltx shots, or to a saved video with --upscale-video.")
    if args.list_places is not None or args.place_info:
        from .places import catalog_report, find_place, place_report
        print(place_report(find_place(args.place_info)) if args.place_info else catalog_report(args.list_places))
        return 0
    look, city = None, False
    if args.place:
        from .places import find_place, place_for_pipeline
        try:
            chosen = place_for_pipeline(find_place(args.place))
        except KeyError as exc:
            raise SystemExit(exc.args[0]) from None
        if args.search or args.latitude is not None:
            raise SystemExit("--place replaces --search and --latitude/--longitude; give one of them.")
        args.place = chosen["catalog_id"]
        args.name = args.name or chosen["display_name"]
        args.latitude, args.longitude = chosen["latitude"], chosen["longitude"]
        if args.palette == "default":
            args.palette = chosen["palette"]
        brink = chosen["lidar_landing"] if args.landing_lidar.lower() != "none" else None
        if args.heading < 0:
            args.heading = (brink or chosen)["heading_deg"]
        look = chosen["look"]
        city = bool(find_place(args.place).get("coverage", {}).get("city"))
        if look and args.preset == "Earth zoom" and args.landing_lidar.lower() != "none":
            look = _ground_underfoot(look, brink or chosen)
    if not args.search and not args.name:
        args.name = "Chosen landmark"
    request = ShotRequest(
        name=args.name or args.search, latitude=args.latitude, longitude=args.longitude,
        search=args.search, result_number=args.result_number, radius_km=args.radius_km,
        preset=args.preset, duration_s=args.duration, heading_deg=args.heading,
        shot_scale_m=args.scale_m, lens_mm=args.lens_mm, engine=args.engine, palette=args.palette, weather=args.weather, imagery=args.imagery, trees=args.trees,
        buildings=args.preset == "Earth zoom" and (args.buildings == "on" or (args.buildings == "auto" and city)),
        width=args.width,
        height=args.height, samples=args.samples, prompt=args.prompt, seed=args.seed,
        steps=args.steps or (4 if args.turbo else 12), lora=args.lora or (H3_TURBO_LORA if args.turbo else ""),
        opening_reference=not args.no_opening_reference, ltx_upscale=not args.no_upscale,
        keyframe_model=args.keyframes, colour_match=args.colour_match, colour_smooth_s=args.colour_smooth, ltx_version=args.ltx_version,
        final_upscale=args.final_upscale, final_target=args.final_target,
        subject_description=args.subject_description, subject_sheet=not args.no_subject_sheet,
        keyframes_only=args.keyframes_only, hold_keyframes=args.hold_keyframes, place_subject=args.place_subject,
        keyframe_denoise=args.keyframe_denoise, keyframe_refine=args.keyframe_refine,
        swoop_denoise=args.swoop_keyframe_denoise, landing_plate_denoise=args.landing_plate_denoise,
        landing_from_hold=args.landing_from_hold, landing_fill=args.landing_fill, save_stage1=args.save_stage1,
        hold_keep=args.hold_keep, hold_near_repaint=args.hold_near_repaint,
        landing_repaint=args.landing_repaint,
        **({"exit_denoise": args.exit_keyframe_denoise} if args.exit_keyframe_denoise is not None else {}),
        hold_follow=args.hold_follow, dive_tempo=args.dive_tempo, ground_repaint=args.ground_repaint,
        composite_subject=args.composite_subject, hold_plate_denoise=args.hold_plate_denoise,
        closing_plate_denoise=args.closing_plate_denoise, cloud_keyframe=args.cloud_keyframe,
        exit_keyframe=args.exit_keyframe, punch_through=args.punch_through, swoop_gap=args.swoop_gap,
        stretch_grain=args.stretch_grain,
        person_detector=args.person_detector,
        orbit_from_blocking=args.orbit_from_blocking,
        pin_last_keyframe=False if args.free_ending else True if args.pin_ending else None, depth_strength=args.depth_strength,
        depth_end=args.depth_end, place_id=args.place, look=look, landing_lidar=args.landing_lidar)
    if args.save_graph:
        request.place_photos = [photo.name for photo in args.photo]
        request.first_frame = args.first_frame.name if args.first_frame else None
        request.subject_photo = args.subject.name if args.subject else None
        graph = ltx_graph(request) if args.ltx else h3_graph(request) if args.h3 else blocking_graph(request)
        args.save_graph.parent.mkdir(parents=True, exist_ok=True)
        args.save_graph.write_text(json.dumps(graph, indent=2) + "\n")
        print(f"Wrote {args.save_graph}")
        return 0
    try:
        import comfy_sdk  # noqa: F401
    except ImportError:
        print("geoblock-run needs the Comfy SDK: uv sync --extra comfy  (or pip install comfy-sdk)", file=sys.stderr)
        return 2
    request.h3_models = {key: getattr(args, f"h3_{key}") for key in H3_MODELS}
    buildings = None  # before the trees: no tree stands on a building footprint
    if request.buildings and not args.blocking_url and not args.no_tree_prefetch:
        buildings = _prefetch_buildings(request)
    if request.trees != "off" and not args.blocking_url and not args.no_tree_prefetch:
        _prefetch_trees(request, buildings)
    if (request.composite_subject and request.preset == "Earth zoom" and not request.prompt
            and not args.blocking_url and not args.no_tree_prefetch):
        request.foreground = _foreground_underfoot(request)
    if args.ltx and args.h3:
        raise SystemExit("Choose --ltx or --h3, not both.")
    if args.auto_photos:
        args.photo += _auto_photos(args)
    destination = args.output_dir
    if args.h3 and args.h3_url:
        if args.depth_strength > 0:
            raise SystemExit("--depth-strength needs the H3 stage on the server that renders the blocking video.")
        blocking = _run(args.blocking_url, blocking_graph(request), (CONTROL_VIDEO, ROUTE_MAP, OPENING_FRAME),
                        destination)
        with _client(args.h3_url) as client:
            request.place_photos = [client.assets.from_file(str(photo)) for photo in args.photo]
            if args.first_frame:
                request.first_frame = client.assets.from_file(str(args.first_frame))
            graph = h3_from_files_graph(
                request, client.assets.from_file(blocking[CONTROL_VIDEO]),
                client.assets.from_file(blocking[OPENING_FRAME]), h3_frame_count(request.duration_s))
            _submit(client, graph, (H3_VIDEO,), destination)
        return 0
    with _client(args.blocking_url) as client:
        request.place_photos = [client.assets.from_file(str(photo)) for photo in args.photo]
        if args.first_frame:
            request.first_frame = client.assets.from_file(str(args.first_frame))
        if args.subject:
            request.subject_photo = client.assets.from_file(str(args.subject))
        graph = ltx_graph(request) if args.ltx else h3_graph(request) if args.h3 else blocking_graph(request)
        saved = _submit(client, graph, (CONTROL_VIDEO, ROUTE_MAP, OPENING_FRAME) + ((H3_VIDEO,) if args.h3 else ())
                + ((*((LTX_VIDEO,) if LTX_VIDEO in graph else ()), *(k for k in (*LTX_KEYFRAMES, SUBJECT_SHEET)
                                                                  if k in graph)) if args.ltx else ())
                + ((FINAL_VIDEO,) if FINAL_VIDEO in graph else ())
                + ((STAGE1_VIDEO,) if STAGE1_VIDEO in graph else ())
                + ((DEPTH_VIDEO,) if args.depth_strength > 0 else ()),
                destination)
    if args.motion_blur == "targeted" and LTX_VIDEO in saved and request.preset == "Earth zoom":
        print(f"Saved {_motion_blur(request, Path(saved[LTX_VIDEO]))}")
    return 0


def _motion_blur(request: ShotRequest, video: Path) -> Path:
    """--motion-blur targeted on a saved Earth zoom video, its windows from the shot's deck exit and arrival."""
    from . import motion_blur
    from .workflow import _arrive_frame
    from .zoom import CLOUD_BASE_M, height_fraction
    last = h3_frame_count(request.duration_s) - 1
    exit_frame = height_fraction(CLOUD_BASE_M, tempo=request.dive_tempo) * last
    return motion_blur.apply(video, video.with_name(f"{video.stem}_mblur.mp4"), _frame_count(video), exit_frame,
                             _arrive_frame(request))


def _upscale_video(args) -> int:
    """--upscale-video: the final upscale alone, on a video rendered earlier."""
    method = "seedvr2" if args.final_upscale == "none" else args.final_upscale
    prefix = f"geoblock/{args.upscale_video.stem}_{method}_{args.final_target}"
    if args.save_graph:
        graph = upscale_file_graph(args.upscale_video.name, method=method, target=args.final_target,
                                   prefix=prefix, seed=args.seed)
        args.save_graph.parent.mkdir(parents=True, exist_ok=True)
        args.save_graph.write_text(json.dumps(graph, indent=2) + "\n")
        print(f"Wrote {args.save_graph}")
        return 0
    count = _frame_count(args.upscale_video) if args.upscale_chunk else 0
    with _client(args.blocking_url) as client:
        video = client.assets.from_file(str(args.upscale_video))
        if count <= args.upscale_chunk:
            _submit(client, upscale_file_graph(video, method=method, target=args.final_target, prefix=prefix,
                                               seed=args.seed), (FINAL_VIDEO,), args.output_dir)
            return 0
        parts = []
        for start in range(0, count, args.upscale_chunk):
            frames = min(args.upscale_chunk, count - start)
            print(f"Frames {start}-{start + frames - 1} of {count}", flush=True)
            saved = _submit(client, upscale_file_graph(video, method=method, target=args.final_target,
                                                       prefix=f"{prefix}_{start:04d}", start=start, frames=frames,
                                                       seed=args.seed), (FINAL_VIDEO,), args.output_dir)
            parts.append(Path(saved[FINAL_VIDEO]))
    joined = args.output_dir / f"{args.upscale_video.stem}_{method}_{args.final_target}.mp4"
    _join_videos(parts, joined)
    print(f"Saved {joined}")
    return 0


def _frame_count(video: Path) -> int:
    """Frames in a video file (ffprobe)."""
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
                         check=True, capture_output=True, text=True).stdout
    return int(out.strip().split(",")[0])


def _join_videos(parts: list, destination: Path) -> None:
    """Concatenate same-format videos without re-encoding (ffmpeg's concat demuxer)."""
    import subprocess
    listing = destination.with_suffix(".txt")
    listing.write_text("".join(f"file '{part.resolve()}'\n" for part in parts))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    str(destination)], check=True)
    listing.unlink()


def _ground_underfoot(look: dict, spot: dict) -> dict:
    """With the 1 m landing patch (US only), the prompt words for the ground underfoot come from NAIP aerial
    photos around where the person stands, not WorldCover's 600 m share. Unchanged where NAIP fails."""
    import httpx
    from .imagery import describe_ground, naip_ground
    from .landcover import with_ground_underfoot
    root = Path(__file__).resolve().parents[2]
    try:
        ground = naip_ground(spot["latitude"], spot["longitude"], root / "data/cache/raw")
    except (httpx.HTTPError, ValueError, OSError) as exc:
        print(f"Ground underfoot from NAIP failed ({exc}); the prompts keep the WorldCover words.")
        return look
    print(f"Ground underfoot (NAIP, {ground['radius_m']:g} m): {ground['words']}")
    open_ground = {name: share for name, share in ground["shares"].items() if name != "trees"}  # between trees
    total = sum(open_ground.values())
    between = describe_ground({name: share / total for name, share in open_ground.items()}) if total else ""
    return with_ground_underfoot(look, ground["words"], between)


def _foreground_underfoot(request: ShotRequest) -> str:
    """The Earth zoom's hold camera looks at the person across a few metres of ground: what that ground is (NAIP
    and Overture roads, imagery.naip_ground), for the plate prompt's near foreground. Twin Peaks' camera stands
    on the summit road. Empty where it can't be measured (no coordinates, outside NAIP)."""
    if request.latitude is None or request.longitude is None:
        return ""
    import httpx
    from pyproj import Transformer
    from .camera import Terrain, plan_route
    from .imagery import naip_ground
    from .scene import build_landform_scene
    root = Path(__file__).resolve().parents[2]
    data = Path(os.environ.get("GEOBLOCK_DATA_DIR") or root / "outputs").expanduser()
    try:
        scene = build_landform_scene(name=request.name.strip(), latitude=request.latitude,
                                     longitude=request.longitude, radius_m=1000 * request.radius_km,
                                     output_root=data / "scenes", raw_cache=root / "data/cache/raw",
                                     location_source="user-supplied coordinates")
        route = plan_route(scene, request.preset, request.duration_s, heading_deg=request.heading_deg,
                           scale_m=request.shot_scale_m or None, lens_mm=request.lens_mm,
                           clearance_m=request.clearance_m, orbit_degrees=request.orbit_degrees,
                           terrain=Terrain.from_scene(scene), dive_tempo=request.dive_tempo)
        camera = route["positions"][route["arrive_frame"]][:2]
        subject = route.get("subject_local_m", (0.0, 0.0))
        middle = [(c + s) / 2 for c, s in zip(camera, subject)]
        radius = max(6.0, round(math.dist(camera, subject) / 2 + 3))
        east, north = (o + m for o, m in zip(scene["origin_projected_m"], middle))
        longitude, latitude = Transformer.from_crs(scene["projected_crs"], "EPSG:4326",
                                                   always_xy=True).transform(east, north)
        ground = naip_ground(latitude, longitude, root / "data/cache/raw", radius_m=radius)
    except (httpx.HTTPError, ValueError, OSError, KeyError) as exc:
        print(f"Foreground ground from NAIP failed ({exc}); the plate prompt describes Image 1's ground.")
        return ""
    print(f"Foreground between camera and person (NAIP, {radius:g} m): {ground['words']}")
    return ground["words"]


def _prefetch_trees(request: ShotRequest, buildings: dict | None = None) -> None:
    """Measure the trees in this environment, before ComfyUI renders the blocking.

    The tree source needs laspy and shapely (``--extra buildings``). Fetching here lets a ComfyUI
    without them reuse the result: the scene, route and tree files are keyed the same way, and the
    Render Blocking Video node finds them in the shared data folder (GEOBLOCK_DATA_DIR, else
    ``outputs/`` in this checkout). Needs coordinates rather than a search.
    """
    if request.latitude is None or request.longitude is None:
        print("--trees: no coordinates, so ComfyUI fetches the trees itself (needs laspy and shapely there).")
        return
    from .camera import Terrain, plan_route
    from .scene import build_landform_scene
    from .looks import canopy_text, scene_palette
    from .trees import canopy_summary, fetch_trees, tree_report
    root = Path(__file__).resolve().parents[2]
    data = Path(os.environ.get("GEOBLOCK_DATA_DIR") or root / "outputs").expanduser()
    name = request.name.strip()
    scene = build_landform_scene(name=name, latitude=request.latitude, longitude=request.longitude,
                                 radius_m=1000 * request.radius_km, output_root=data / "scenes",
                                 raw_cache=root / "data/cache/raw", location_source="user-supplied coordinates")
    route = plan_route(scene, request.preset, request.duration_s, heading_deg=request.heading_deg,
                       scale_m=request.shot_scale_m or None, lens_mm=request.lens_mm,
                       clearance_m=request.clearance_m, orbit_degrees=request.orbit_degrees,
                       terrain=Terrain.from_scene(scene), dive_tempo=request.dive_tempo)
    manifest = fetch_trees(scene, route, Path(scene["terrain_dir"]) / "trees", cache_dir=root / "data/cache/raw",
                           source=request.trees, buildings=buildings)
    print(tree_report(manifest))
    print(f"Trees cached for ComfyUI at {manifest['path']}")
    if manifest["count"] and not request.prompt:  # the prompts describe the measured trees, not stock woodland
        # An Earth zoom's keyframes that use these words are the landing: describe the trees around the person.
        around = tuple(route.get("subject_local_m", (0.0, 0.0))) if request.preset == "Earth zoom" else None
        request.canopy = canopy_text(canopy_summary(manifest, around), scene_palette(request.palette, scene),
                                     (request.look or {}).get("between_trees", ""))
        print(f"Canopy words: {request.canopy}")


def _prefetch_buildings(request: ShotRequest) -> dict | None:
    """Read the Overture buildings in this environment (pyarrow and shapely: ``--extra buildings``) and cache
    them where the Render Blocking Video node looks, as ``_prefetch_trees`` does for trees."""
    if request.latitude is None or request.longitude is None:
        print("--buildings: no coordinates, so ComfyUI fetches the buildings itself (needs pyarrow and shapely).")
        return None
    from .buildings import building_report, fetch_buildings
    from .scene import build_landform_scene
    root = Path(__file__).resolve().parents[2]
    data = Path(os.environ.get("GEOBLOCK_DATA_DIR") or root / "outputs").expanduser()
    scene = build_landform_scene(name=request.name.strip(), latitude=request.latitude, longitude=request.longitude,
                                 radius_m=1000 * request.radius_km, output_root=data / "scenes",
                                 raw_cache=root / "data/cache/raw", location_source="user-supplied coordinates")
    manifest = fetch_buildings(scene, Path(scene["terrain_dir"]) / "buildings", cache_dir=root / "data/cache/raw")
    print(building_report(manifest))
    print(f"Buildings cached for ComfyUI at {manifest['path']}")
    return manifest


def _auto_photos(args) -> list[Path]:
    """Fetch Commons photos on this machine, so any target (local, RunPod, Comfy Cloud) gets them as uploads."""
    from .place_photos import fetch_place_photos, photo_report
    if not args.h3:
        raise SystemExit("--auto-photos only applies with --h3.")
    if args.latitude is None or args.longitude is None:
        raise SystemExit("--auto-photos needs --latitude and --longitude.")
    folder = args.output_dir.parent / "place_photos" / f"{args.latitude:.4f}_{args.longitude:.4f}"
    manifest = fetch_place_photos(args.name, args.latitude, args.longitude, args.auto_photos, folder,
                                  radius_km=args.photo_radius_km, heading_deg=args.heading,
                                  exclude=tuple(args.exclude_photo))
    print(photo_report(manifest))
    print(f"Attribution: {folder / 'ATTRIBUTION.md'}")
    return [Path(item["local_path"]) for item in manifest["selected"]]


@contextmanager
def _client(target: str | None):
    """A Comfy SDK client for ``target``: a URL, 'cloud', or None for the environment's choice.

    Comfy Cloud authenticates with COMFY_API_KEY. A self-hosted target named
    here (e.g. a RunPod pod's comfy-api-proxy) uses GEOBLOCK_PROXY_TOKEN
    instead, so the Comfy Cloud key is never sent to it.
    """
    from comfy_sdk import Comfy
    saved = os.environ.get("COMFY_BASE_URL")
    api_key = None
    if target == "cloud":
        os.environ.pop("COMFY_BASE_URL", None)
    elif target:
        os.environ["COMFY_BASE_URL"] = target
        api_key = os.environ.get("GEOBLOCK_PROXY_TOKEN") or ""
    try:
        if api_key == "":
            cloud_key = os.environ.pop("COMFY_API_KEY", None)
            try:
                client = Comfy(client_info="geoblock")
            finally:
                if cloud_key is not None:
                    os.environ["COMFY_API_KEY"] = cloud_key
        else:
            client = Comfy(api_key=api_key, client_info="geoblock")  # None: COMFY_API_KEY
    finally:
        if saved is None:
            os.environ.pop("COMFY_BASE_URL", None)
        else:
            os.environ["COMFY_BASE_URL"] = saved
    with client:
        yield client


def _run(target: str | None, graph: dict, wanted: tuple[str, ...], destination: Path) -> dict:
    with _client(target) as client:
        return _submit(client, graph, wanted, destination)


def _submit(client, graph: dict, wanted: tuple[str, ...], destination: Path) -> dict:
    """Submit, report progress, and download the wanted output nodes."""
    from comfy_sdk import InvalidWorkflow, OutputReady, Progress, StatusChange
    try:
        job = client.submit(client.workflows.from_json(graph))
    except InvalidWorkflow as exc:
        # Usually a missing model file or custom node on that server; show which.
        details = json.dumps(getattr(exc, "details", None), indent=2, default=str)
        raise SystemExit(f"{client!r} rejected the graph: {exc}\n{details}") from None
    print(f"Submitted job {job.id} to {client!r}", flush=True)
    for event in job.events():
        if isinstance(event, Progress):
            print(f"  {event.value:.0%} {event.message or ''}".rstrip(), flush=True)
        elif isinstance(event, OutputReady):
            print(f"  output ready: {event.output.name}", flush=True)
        elif isinstance(event, StatusChange) and event.status in ("succeeded", "failed", "cancelled"):
            break
    job = job.result()
    folder = destination / job.id
    folder.mkdir(parents=True, exist_ok=True)
    saved = {}
    for node in wanted:
        for output in job.get_outputs(node):
            saved[node] = str(output.to_file(folder / output.name))
            print(f"Saved {saved[node]}")
    (folder / "graph.json").write_text(json.dumps(graph, indent=2, default=str) + "\n")
    (folder / "outputs.json").write_text(json.dumps(saved, indent=2) + "\n")
    return saved

if __name__ == "__main__":
    raise SystemExit(main())
