"""GeoBlock's first ComfyUI scene-to-H3 prototype.

Install by linking this directory into ComfyUI/custom_nodes. The GIS compiler
stays outside ComfyUI; this extension exposes a small, verified demo slice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, io, ui

from .core import build_pack, load_demo_terrain, load_style_refs, plan_shot, resolve_place
from .landform import (build_scene, place_from_catalog, place_from_coordinates, place_photos,
                       plan_shot as plan_preset_shot, render_shot)
from .landmark_search import choose_landmark, search_landmarks
from geoblock_core.blender import ENGINES, PALETTES, TREE_SOURCES
from geoblock_core.weather import DEFAULT_WEATHER, WEATHER
from geoblock_core.camera import PRESETS
from .resource_catalog import CATEGORIES, discover_resources


Place = io.Custom("GEOBLOCK_PLACE")
Candidates = io.Custom("GEOBLOCK_CANDIDATES")
Scene = io.Custom("GEOBLOCK_SCENE")
Shot = io.Custom("GEOBLOCK_SHOT")
Subject = io.Custom("GEOBLOCK_SUBJECT")


def _image(path: str):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


class GeoBlockDiscoverSources(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockDiscoverSources",
            display_name="GeoBlock 0 · Discover Geometry Sources",
            category="GeoBlock/Prototype",
            description="Browse curated catalogues and datasets alphabetically. A listing is only a lead; the selected shot still needs local geometry validation.",
            inputs=[
                io.Combo.Input("category", options=list(CATEGORIES), default="All"),
                io.String.Input("place_or_region", default=""),
            ],
            outputs=[io.String.Output(display_name="source report")],
        )

    @classmethod
    def execute(cls, category, place_or_region):
        _, report = discover_resources(category, place_or_region)
        return io.NodeOutput(report, ui=ui.PreviewText(report))


class GeoBlockSearchLandmarks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockSearchLandmarks",
            display_name="GeoBlock 1 · Search Landmarks",
            category="GeoBlock/Place",
            description="Search for a user-chosen landmark. Review the numbered matches before selecting one; search does not imply 3D coverage.",
            inputs=[io.String.Input("query", default="")],
            outputs=[Candidates.Output(display_name="landmark matches"),
                     io.String.Output(display_name="search report")],
        )

    @classmethod
    def execute(cls, query):
        matches, report = search_landmarks(query)
        return io.NodeOutput(matches, report, ui=ui.PreviewText(report))


class GeoBlockChooseLandmark(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockChooseLandmark",
            display_name="GeoBlock 2 · Choose Landmark",
            category="GeoBlock/Place",
            description="Pick one numbered search result and the area the eventual shot must cover.",
            inputs=[Candidates.Input("landmark_matches"),
                    io.Int.Input("result_number", default=1, min=1, max=5),
                    io.Float.Input("coverage_radius_km", default=1.0, min=0.1, max=20, step=0.1)],
            outputs=[Place.Output(display_name="chosen place"),
                     io.String.Output(display_name="selection report")],
        )

    @classmethod
    def execute(cls, landmark_matches, result_number, coverage_radius_km):
        place, report = choose_landmark(landmark_matches, result_number, coverage_radius_km)
        return io.NodeOutput(place, report, ui=ui.PreviewText(report))


class GeoBlockPlaceFromCoordinates(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPlaceFromCoordinates",
            display_name="GeoBlock 2 · Place From Coordinates",
            category="GeoBlock/Place",
            description="Use exact coordinates instead of a search, e.g. a viewpoint you have already located.",
            inputs=[io.String.Input("name", default="Mather Point, Grand Canyon"),
                    io.Float.Input("latitude", default=36.0617, min=-90, max=90, step=0.000001),
                    io.Float.Input("longitude", default=-112.1078, min=-180, max=180, step=0.000001),
                    io.Float.Input("coverage_radius_km", default=12, min=0.5, max=30, step=0.5)],
            outputs=[Place.Output(display_name="chosen place"),
                     io.String.Output(display_name="selection report")],
        )

    @classmethod
    def execute(cls, name, latitude, longitude, coverage_radius_km):
        place, report = place_from_coordinates(name, latitude, longitude, coverage_radius_km)
        return io.NodeOutput(place, report, ui=ui.PreviewText(report))


def _catalog_options() -> list[str]:
    from geoblock_core.places import places
    return [f"{place['id']} · {place['name']}" for place in places()]


class GeoBlockPlaceFromCatalog(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPlaceFromCatalog",
            display_name="GeoBlock 2 · Place From Catalog",
            category="GeoBlock/Place",
            description="Pick a fly-down landing place from the GeoBlock catalog. Each place lists its measured "
                        "terrain, lidar, imagery and land cover; data is fetched only for the place you pick. "
                        "The camera faces the place's view (e.g. the Matterhorn) unless a heading is set.",
            inputs=[io.Combo.Input("place", options=_catalog_options(),
                                   default=_catalog_options()[0]),
                    io.Float.Input("coverage_radius_km", default=0, min=0, max=30, step=0.5,
                                   tooltip="0 uses the catalog's radius (12 km).")],
            outputs=[Place.Output(display_name="chosen place"),
                     io.Float.Output(display_name="view heading", tooltip="-1 means toward the deepest drop."),
                     io.String.Output(display_name="place report")],
        )

    @classmethod
    def execute(cls, place, coverage_radius_km):
        chosen, report = place_from_catalog(place, coverage_radius_km)
        return io.NodeOutput(chosen, chosen["heading_deg"], report, ui=ui.PreviewText(report))


class GeoBlockBuildLandformScene(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockBuildLandformScene",
            display_name="GeoBlock 3 · Build Landform Scene",
            category="GeoBlock/Scene",
            description="Fetch measured terrain around the chosen place (USGS 3DEP 10 m in the conterminous "
                        "US, Copernicus GLO-30 elsewhere), compile it, and check there is real relief at the landmark.",
            inputs=[Place.Input("place"),
                    io.Float.Input("radius_km_override", default=0, min=0, max=30, step=0.5,
                                   tooltip="0 uses the radius chosen with the place."),
                    io.String.Input("landing_lidar", default="none", optional=True,
                                    tooltip="US only: 'auto' adds an 800 m square of USGS 3DEP 1 m lidar terrain "
                                            "where the Earth zoom lands (newest project), or name a 3DEP 1 m "
                                            "project. Catalog overlooks then stand on the brink. 'none' keeps 10 m.")],
            outputs=[Scene.Output(display_name="terrain scene"),
                     io.Image.Output(display_name="scene preview"),
                     io.String.Output(display_name="scene report")],
        )

    @classmethod
    def execute(cls, place, radius_km_override, landing_lidar="none"):
        scene, report = build_scene(place, radius_km_override, landing_lidar)
        return io.NodeOutput(scene, _image(scene["preview_path"]), report, ui=ui.PreviewText(report))


class GeoBlockCameraPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockCameraPreset",
            display_name="GeoBlock 4 · Camera Preset",
            category="GeoBlock/Shot",
            description="Plan a drone-style camera move through the measured terrain. Routes are lifted to "
                        "clear the ground and timed to MiniMax H3 frame lengths.",
            inputs=[Scene.Input("terrain_scene"),
                    io.Combo.Input("preset", options=list(PRESETS), default="Rim reveal"),
                    io.Float.Input("duration_seconds", default=10, min=2, max=60, step=0.1,
                                   tooltip="Over 15 s is split into overlapping H3 segments for clip chaining."),
                    io.Float.Input("heading_degrees", default=-1, min=-1, max=359, step=1,
                                   tooltip="-1 aims toward the deepest measured drop."),
                    io.Float.Input("shot_scale_m", default=0, min=0, max=20000, step=50,
                                   tooltip="Distance scale of the move; 0 picks one from the scene size."),
                    io.Float.Input("lens_mm", default=24, min=12, max=120, step=1),
                    io.Float.Input("min_clearance_m", default=25, min=5, max=500, step=5),
                    io.Float.Input("orbit_degrees", default=120, min=30, max=360, step=5),
                    io.Combo.Input("dive_tempo", options=["standard", "low-swoop", "smooth-swoop"], default="standard",
                                   optional=True,
                                   tooltip="Earth zoom: low-swoop crosses orbit and cloud sooner and spends the "
                                           "time on the low swoop, which LTX smears when it is fast; "
                                           "smooth-swoop also brakes into the push-in and rides over bumps.")],
            outputs=[Shot.Output(display_name="shot"),
                     io.Image.Output(display_name="route map"),
                     io.Int.Output(display_name="H3 frames"),
                     io.String.Output(display_name="shot report")],
        )

    @classmethod
    def execute(cls, terrain_scene, preset, duration_seconds, heading_degrees, shot_scale_m,
                lens_mm, min_clearance_m, orbit_degrees, dive_tempo="standard"):
        route, preview, report = plan_preset_shot(terrain_scene, preset, duration_seconds, heading_degrees,
                                                  shot_scale_m, lens_mm, min_clearance_m, orbit_degrees,
                                                  dive_tempo)
        return io.NodeOutput(route, _image(preview), route["frames"], report, ui=ui.PreviewText(report))


class GeoBlockRenderBlockingVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockRenderBlockingVideo",
            display_name="GeoBlock 5 · Render Blocking Video",
            category="GeoBlock/Shot",
            description="Render the camera route over the measured terrain with headless Blender "
                        "(PATH or GEOBLOCK_BLENDER). Identical renders are reused.",
            is_output_node=True,
            inputs=[Scene.Input("terrain_scene"), Shot.Input("shot"),
                    io.Combo.Input("engine", options=list(ENGINES), default="eevee",
                                   tooltip="EEVEE and Workbench need a GPU; Cycles also runs on CPU."),
                    io.Int.Input("width", default=1280, min=256, max=3840, step=16),
                    io.Int.Input("height", default=720, min=144, max=2160, step=16),
                    io.Int.Input("samples", default=16, min=1, max=1024),
                    io.Boolean.Input("depth_pass", default=False, optional=True,
                                     tooltip="Also render a depth video (near white, far black) for "
                                             "MiniMax H3 Fun ControlNet."),
                    io.Combo.Input("palette", options=list(PALETTES), default="default", optional=True,
                                   tooltip="Terrain height colours. grand-canyon follows the canyon's rock "
                                           "bands so video models do not read the clay as snow."),
                    io.Combo.Input("weather", options=list(WEATHER), default=DEFAULT_WEATHER, optional=True,
                                   tooltip="Sky, sun and haze of the blocking render. Video models copy its "
                                           "light, so pick the weather here, not only in the prompt."),
                    io.Boolean.Input("satellite_imagery", default=False, optional=True,
                                     tooltip="Drape public-domain NASA HLS 30 m imagery (via GIBS) on the "
                                             "terrain instead of the palette colours."),
                    io.Combo.Input("trees", options=list(TREE_SOURCES), default="off", optional=True,
                                   tooltip="Stand a low-poly proxy on every measured tree near the camera path, so "
                                           "video models keep trees in place. auto: USGS 3DEP lidar where surveyed, "
                                           "else Meta/WRI 1 m canopy height (CC BY 4.0)."),
                    io.Boolean.Input("buildings", default=False, optional=True,
                                     tooltip="Earth zoom: stand a flat-roofed block on every Overture Maps building "
                                             "footprint within 7 km, at its measured height, so video models keep "
                                             "a city's buildings in place (OpenStreetMap ODbL and others).")],
            outputs=[io.Video.Output(display_name="control video"),
                     io.Image.Output(display_name="opening frame"),
                     io.Int.Output(display_name="H3 frames"),
                     io.String.Output(display_name="render report"),
                     io.Video.Output(display_name="depth video",
                                     tooltip="The control video again when depth_pass is off."),
                     io.String.Output(display_name="clean plates",
                                      tooltip="Earth zoom: folder of hold frames rendered without the stand-in "
                                              "(empty otherwise), for Place Subject.")],
        )

    @classmethod
    def execute(cls, terrain_scene, shot, engine, width, height, samples, depth_pass=False, palette="default",
                weather=DEFAULT_WEATHER, satellite_imagery=False, trees="off", buildings=False):
        import comfy.utils
        bar = comfy.utils.ProgressBar(shot["frames"])
        output_dir = Path(folder_paths.get_output_directory())
        manifest, report = render_shot(terrain_scene, shot, output_dir / "geoblock", engine, width, height,
                                       samples, depth=depth_pass, palette=palette, weather=weather, imagery=satellite_imagery,
                                       trees=trees, buildings=buildings,
                                       progress=lambda done, total: bar.update_absolute(done, total))
        video = Path(manifest["video"])
        depth = Path(manifest.get("depth_video") or video)
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(video)), _image(manifest["first_frame"]), shot["frames"], report,
            InputImpl.VideoFromFile(str(depth)), manifest.get("clean_plates") or "",
            ui=ui.PreviewVideo([ui.SavedResult(video.name, str(video.parent.relative_to(output_dir)),
                                               io.FolderType.output)]),
        )


class GeoBlockKeyframePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockKeyframePrompt",
            display_name="GeoBlock 6 · Keyframe Prompt",
            category="GeoBlock/Shot",
            description="Prompt for repainting one blocking frame as a photograph. It describes only what that "
                        "frame can show, from where the camera is: over the landmark's own level or out over a drop.",
            inputs=[Scene.Input("terrain_scene"), Shot.Input("shot"),
                    io.Int.Input("frame", default=0, min=0, max=100000),
                    io.Combo.Input("palette", options=list(PALETTES), default="default"),
                    io.Combo.Input("weather", options=list(WEATHER), default=DEFAULT_WEATHER),
                    Subject.Input("subject", optional=True,
                                  tooltip="The person the Earth zoom lands on (GeoBlock Subject node)."),
                    io.Boolean.Input("reference", default=False, optional=True,
                                     tooltip="An earlier repainted keyframe is given as Image 2: keep it consistent."),
                    io.Combo.Input("reference_kind", options=["hold", "descent"], default="hold", optional=True,
                                   tooltip="hold: Image 2 is the same spot from further back. descent: Image 2 is "
                                           "the ground a moment later, from lower down."),
                    io.String.Input("canopy", default="", optional=True, multiline=True,
                                    tooltip="Measured trees in words (geoblock-run --trees fills this), used instead "
                                            "of the place's stock vegetation words for the ground near the camera."),
                    io.Boolean.Input("placed", default=False, optional=True,
                                     tooltip="The person is already pasted into Image 1 (GeoBlock Place Subject)."),
                    io.Boolean.Input("plate", default=False, optional=True,
                                     tooltip="An empty plate the person is pasted onto later: keep their spot clear."),
                    io.String.Input("foreground", default="", optional=True,
                                    tooltip="With plate: the measured ground between the camera and the person "
                                            "(geoblock-run fills it), for the plate's near foreground.")],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, terrain_scene, shot, frame, palette, weather, subject=None, reference=False,
                reference_kind="hold", canopy="", placed=False, plate=False, foreground=""):
        from geoblock_core.looks import frame_situation, keyframe_prompt
        from geoblock_core.looks import default_subject
        from geoblock_core.weather import weather_prompt
        from geoblock_core.looks import scene_palette
        situation = frame_situation(terrain_scene, shot, frame) | {
            "reference": (reference_kind if reference_kind == "descent" else True) if reference else False,
            "placed": placed, "plate": plate, "foreground": foreground}
        if subject and not subject.get("description") and not subject.get("photo"):
            subject = subject | {"description": default_subject(weather)}  # dressed for the weather
        prompt = keyframe_prompt(terrain_scene["name"], scene_palette(palette, terrain_scene),
                                 weather_prompt(weather, situation), situation, subject, look=terrain_scene.get("look"),
                                 weather=weather, canopy=canopy)
        return io.NodeOutput(prompt, ui=ui.PreviewText(f"{situation}\n{prompt}"))


class GeoBlockPushInCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPushInCrop",
            display_name="GeoBlock · Push-in Crop",
            category="GeoBlock/Shot",
            description="Crops a repainted Earth zoom keyframe to how a later frame of the eye-level push-in "
                        "frames the stand-in figure (scaled and shifted from the camera path), so the next "
                        "keyframe can be repainted from it without losing the framing.",
            inputs=[Scene.Input("terrain_scene"), Shot.Input("shot"), io.Image.Input("image"),
                    io.Int.Input("from_frame", default=0, min=0, max=100000),
                    io.Int.Input("to_frame", default=0, min=0, max=100000)],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, terrain_scene, shot, image, from_frame, to_frame):
        import comfy.utils
        from geoblock_core.zoom import push_in_crop
        _, height, width, _ = image.shape
        x, y, w, h = push_in_crop(terrain_scene, shot, from_frame, to_frame, width / height)
        left, top = round(x * width), round(y * height)
        right, bottom = left + max(1, round(w * width)), top + max(1, round(h * height))
        pad = (max(0, -left), max(0, right - width), max(0, -top), max(0, bottom - height))
        picture = torch.nn.functional.pad(image.movedim(-1, 1), pad, mode="replicate")  # edges where it reaches out
        picture = picture[:, :, top + pad[2]:bottom + pad[2], left + pad[0]:right + pad[0]]
        picture = comfy.utils.common_upscale(picture, width, height, "lanczos", "disabled").movedim(1, -1)
        return io.NodeOutput(picture.clamp(0, 1),
                             ui=ui.PreviewText(f"frame {from_frame} -> {to_frame}: crop x {x:.3f} y {y:.3f} "
                                               f"w {w:.3f} h {h:.3f}"))


class GeoBlockReprojectHold(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockReprojectHold",
            display_name="GeoBlock · Reproject Hold",
            category="GeoBlock/Shot",
            description="Moves a repainted Earth zoom hold keyframe to a later frame's camera through the scene's "
                        "terrain: the near ground grows with the dolly while the distant city and sky only turn "
                        "with the camera (Push-in Crop zooms the whole picture). Areas the earlier view doesn't "
                        "cover are filled from around them, for the painter to redo.",
            inputs=[Scene.Input("terrain_scene"), Shot.Input("shot"), io.Image.Input("image"),
                    io.Int.Input("from_frame", default=0, min=0, max=100000),
                    io.Int.Input("to_frame", default=0, min=0, max=100000),
                    io.Image.Input("fill", optional=True,
                                   tooltip="The later frame's blocking (clean plate), used where the earlier plate "
                                           "doesn't reach instead of a fill from around it."),
                    io.Float.Input("grain", default=0.0, min=0.0, max=0.5, step=0.01, optional=True,
                                   tooltip="Noise added to the blurred stretched ground, for the painter to turn "
                                           "into texture (it keeps plain blur as defocus)."),
                    io.Float.Input("near_m", default=0.0, min=0.0, max=50.0, step=0.5, optional=True,
                                   tooltip="Also hand the ground within this many metres of the later camera to "
                                           "the painter (blurred, in the stretched-ground mask): near bushes moved "
                                           "as ground shear into streaks. 0: only the stretched ground."),
                    io.Float.Input("keep", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                                   tooltip="Strength of the kept mask: how much of the earlier plate to paste back "
                                           "over the repaint where it covers the view unstretched.")],
            outputs=[io.Image.Output(display_name="image"),
                     io.Mask.Output(display_name="stretched ground"),
                     io.Mask.Output(display_name="kept"),
                     io.Mask.Output(display_name="unseen")],
        )

    @classmethod
    def execute(cls, terrain_scene, shot, image, from_frame, to_frame, fill=None, grain=0.0, keep=1.0, near_m=0.0):
        from geoblock_core.zoom import reproject_hold, unseen_mask
        out, seen, stretched, kept = reproject_hold(
            terrain_scene, shot, from_frame, to_frame, image[0].cpu().numpy().astype(np.float64), grain=grain,
            fill=None if fill is None else fill[0].cpu().numpy().astype(np.float64), near_m=near_m)
        kept = keep * kept
        return io.NodeOutput(torch.from_numpy(out).float()[None].clamp(0, 1),
                             torch.from_numpy(stretched).float()[None].clamp(0, 1),
                             torch.from_numpy(kept).float()[None].clamp(0, 1),
                             torch.from_numpy(unseen_mask(seen)).float()[None].clamp(0, 1),
                             ui=ui.PreviewText(f"frame {from_frame} -> {to_frame}: {seen.mean():.0%} of the view "
                                               f"comes from the earlier plate, {(stretched > 0.5).mean():.0%} "
                                               "stretched (repainted)"))


class GeoBlockPlaceSubject(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPlaceSubject",
            display_name="GeoBlock · Place Subject",
            category="GeoBlock/Shot",
            description="Pastes the person from the subject sheet over an Earth zoom blocking frame, at the grey "
                        "stand-in's place and size (from the camera path), so the keyframe painter keeps the "
                        "push-in's framing instead of redrawing the person full-length and centred.",
            inputs=[Scene.Input("terrain_scene"), Shot.Input("shot"), io.Image.Input("image"),
                    io.Image.Input("subject_image"), io.Int.Input("frame", default=0, min=0, max=100000),
                    io.String.Input("clean_plates", default="", optional=True,
                                    tooltip="Render Blocking Video's clean plates: the frame without the stand-in, "
                                            "so nothing has to be painted out behind the person."),
                    io.Boolean.Input("erase", default=True, optional=True,
                                     tooltip="Paint the grey stand-in out first. Off for an image without one "
                                             "(a painted clean plate).")],
            outputs=[io.Image.Output(display_name="image"),
                     io.Mask.Output(display_name="mask", tooltip="The person, grown by a ring around them, for a "
                                                                 "blend pass that relights only them.")],
        )

    @classmethod
    def execute(cls, terrain_scene, shot, image, subject_image, frame, clean_plates="", erase=True):
        from geoblock_core.zoom import place_subject, stand_in_on_screen
        _, height, width, _ = image.shape
        x, y, h = stand_in_on_screen(terrain_scene, shot, frame, width / height)
        frame_image, clean = image[0].cpu().numpy().astype(np.float64), None
        plate = Path(clean_plates) / f"clean_{frame:04d}.png" if clean_plates else None
        if erase and plate is not None and plate.is_file():  # without a stand-in to remove, the image stays
            from PIL import Image
            clean = np.asarray(Image.open(plate).convert("RGB").resize((width, height), Image.LANCZOS)) / 255.0
        shadow, shade = Path(clean_plates) / f"shadow_{frame:04d}.png" if clean_plates else None, None
        if shadow is not None and shadow.is_file() and plate is not None and plate.is_file():
            from PIL import Image
            from geoblock_core.zoom import shadow_factor, shadow_near
            lit, shaded = (np.asarray(Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)) / 255.0
                           for path in (plate, shadow))
            shade = shadow_near(shadow_factor(lit, shaded), x, y + h / 2, h)
            if not (shade < 0.97).any():  # the shadow pass cast nothing (r11k): no false ", shadow"
                shade = None
        placed, mask = place_subject(frame_image, subject_image[0].cpu().numpy().astype(np.float64), x, y + h / 2,
                                     h, clean=clean, erase=erase, shade=shade, with_mask=True)
        source = "clean plate" if clean is not None else "stand-in painted out" if erase else "no stand-in"
        source += ", shadow" if shade is not None else ", contact shadow only"
        return io.NodeOutput(torch.from_numpy(placed).float()[None], torch.from_numpy(mask).float()[None],
                             ui=ui.PreviewText(f"frame {frame}: feet x {x:.3f} y {y + h / 2:.3f}, height {h:.3f}; "
                                               f"{source}"))


class GeoBlockFindPeople(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockFindPeople",
            display_name="GeoBlock · Find People",
            category="GeoBlock/Shot",
            description="Masks any person on a painted empty plate (Florence-2 object detection from a local "
                        "folder), so a masked repaint can take out a stranger before the subject is pasted in. An "
                        "empty mask if none is found or the detector can't load.",
            inputs=[io.Image.Input("image"), io.String.Input("model_path", default=""),
                    io.Float.Input("grow", default=0.25, min=0.0, max=2.0, step=0.05, optional=True)],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, image, model_path, grow=0.25):
        from geoblock_core.people import find_people, people_mask
        _, height, width, _ = image.shape
        try:
            boxes = find_people(image[0].cpu().numpy(), model_path) if model_path else []
            note = f"{len(boxes)} people: {[[round(v) for v in box] for box in boxes]}"
        except Exception as exc:  # noqa: BLE001 - a missing or incompatible detector must not stop the render
            boxes, note = [], f"detector unavailable: {type(exc).__name__}: {exc}"
        mask = people_mask((height, width), boxes, grow)
        return io.NodeOutput(torch.from_numpy(mask)[None], ui=ui.PreviewText(note))


class GeoBlockFillPeople(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockFillPeople",
            display_name="GeoBlock · Fill People",
            category="GeoBlock/Shot",
            description="Fills the masked area (from Find People) smoothly from the scene around it, so a masked "
                        "repaint starts from empty ground rather than a person-shaped hole. Unchanged if the mask is "
                        "empty.",
            inputs=[io.Image.Input("image"), io.Mask.Input("mask")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, mask):
        from geoblock_core.people import fill_people
        from geoblock_core.people import MAX_FILL_SHARE
        hole = mask[0].cpu().numpy()
        if hole.shape != tuple(image.shape[1:3]) or not (hole > 0.5).any():
            return io.NodeOutput(image, ui=ui.PreviewText("nothing to fill"))
        if (hole > 0.5).mean() > MAX_FILL_SHARE:
            return io.NodeOutput(image, ui=ui.PreviewText(f"mask covers {(hole > 0.5).mean():.0%}: left as is"))
        filled = fill_people(image[0].cpu().numpy().astype(np.float64), hole)
        return io.NodeOutput(torch.from_numpy(filled).float()[None],
                             ui=ui.PreviewText(f"filled {(hole > 0.5).mean():.1%} of the frame"))


class GeoBlockCleanPlate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockCleanPlate",
            display_name="GeoBlock · Clean Plate",
            category="GeoBlock/Shot",
            description="One Earth zoom blocking frame rendered without the grey stand-in (from Render Blocking "
                        "Video's clean plates), at the blocking video's size; the blocking frame itself if there "
                        "is no plate for that frame.",
            inputs=[io.String.Input("clean_plates", default=""), io.Image.Input("image"),
                    io.Int.Input("frame", default=0, min=0, max=100000)],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, clean_plates, image, frame):
        _, height, width, _ = image.shape
        plate = Path(clean_plates) / f"clean_{frame:04d}.png" if clean_plates else None
        if plate is None or not plate.is_file():
            return io.NodeOutput(image[frame:frame + 1], ui=ui.PreviewText(f"frame {frame}: no clean plate"))
        from PIL import Image
        clean = np.asarray(Image.open(plate).convert("RGB").resize((width, height), Image.LANCZOS)) / 255.0
        return io.NodeOutput(torch.from_numpy(clean).float()[None], ui=ui.PreviewText(f"frame {frame}: clean plate"))


class GeoBlockColourMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockColourMatch",
            display_name="GeoBlock · Colour Match",
            category="GeoBlock/Shot",
            description="Moves an image's colour cast toward a reference (its blocking frame): the mean of the two "
                        "colour axes in CIELAB, by strength. Lightness and contrast stay as painted (the blocking's "
                        "shading is not photographic). Keeps every repainted keyframe on the blocking's continuous "
                        "colour, so the video has no white-balance jumps between them.",
            inputs=[io.Image.Input("image"), io.Image.Input("reference"),
                    io.Float.Input("strength", default=0.6, min=0.0, max=1.0, step=0.05)],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, reference, strength):
        import comfy.utils
        import kornia
        lab = kornia.color.rgb_to_lab(image.movedim(-1, 1)[:, :3].clamp(0, 1))
        ref = comfy.utils.common_upscale(reference.movedim(-1, 1)[:, :3], lab.shape[3], lab.shape[2], "bilinear",
                                         "disabled")
        shift = strength * (kornia.color.rgb_to_lab(ref.clamp(0, 1)).mean((2, 3), keepdim=True)
                            - lab.mean((2, 3), keepdim=True))
        shift[:, 0] = 0  # colour axes only
        rgb = kornia.color.lab_to_rgb(lab + shift).clamp(0, 1).movedim(1, -1)
        a, b = shift.flatten(1)[0, 1:].tolist()
        return io.NodeOutput(rgb, ui=ui.PreviewText(f"colour cast shift a {a:+.1f} b {b:+.1f}"))

class GeoBlockColourSmooth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockColourSmooth",
            display_name="GeoBlock · Temporal Colour Smooth",
            category="GeoBlock/Shot",
            description="Evens out the colour cast over time: each frame's mean CIELAB a/b is moved onto a Gaussian "
                        "average of its neighbours (sigma in seconds), so there are no quick white-balance shifts "
                        "between keyframes. Lightness, contrast and detail are untouched.",
            inputs=[io.Image.Input("images"), io.Float.Input("sigma_s", default=0.75, min=0.0, max=5.0, step=0.05),
                    io.Float.Input("fps", default=24.0, min=1.0, max=120.0)],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, sigma_s, fps):
        import kornia
        if sigma_s <= 0 or len(images) < 3:
            return io.NodeOutput(images)
        chunk = 16
        means = torch.cat([kornia.color.rgb_to_lab(images[i:i + chunk].movedim(-1, 1)[:, :3].clamp(0, 1)).mean((2, 3))
                           for i in range(0, len(images), chunk)])
        t = torch.arange(len(means), dtype=torch.float32)
        weights = torch.exp(-0.5 * ((t[:, None] - t[None, :]) / (sigma_s * fps)) ** 2)
        shift = weights / weights.sum(1, keepdim=True) @ means - means
        shift[:, 0] = 0  # colour axes only
        out = torch.empty_like(images)
        for i in range(0, len(images), chunk):
            lab = kornia.color.rgb_to_lab(images[i:i + chunk].movedim(-1, 1)[:, :3].clamp(0, 1))
            out[i:i + chunk, ..., :3] = kornia.color.lab_to_rgb(lab + shift[i:i + chunk, :, None, None]).clamp(0, 1).movedim(1, -1)
        if images.shape[-1] > 3:
            out[..., 3:] = images[..., 3:]
        return io.NodeOutput(out, ui=ui.PreviewText(f"largest cast shift {shift.abs().max():.1f}"))


class GeoBlockPunchThrough(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPunchThrough",
            display_name="GeoBlock · Cloud Punch-Through",
            category="GeoBlock/Shot",
            description="Turns the Earth zoom's cut from cloud to clear ground into a fast bright dissolve: the "
                        "largest frame change between the two frames given is taken as the cut, and the frames "
                        "around it cross-fade from cloud to ground, lifted toward white in the middle.",
            inputs=[io.Image.Input("images"), io.Int.Input("start_frame", default=0, min=0, max=100000),
                    io.Int.Input("end_frame", default=0, min=0, max=100000),
                    io.Int.Input("span", default=6, min=2, max=24)],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, start_frame, end_frame, span):
        from geoblock_core.zoom import punch_through
        first, last = max(0, start_frame - span), min(len(images), end_frame + span + 1)
        window = images[first:last].cpu().numpy()
        out, cut = punch_through(window, start_frame - first, end_frame - first, span)
        if cut is None:
            return io.NodeOutput(images, ui=ui.PreviewText("no cloud cut in range"))
        result = images.clone()
        result[first:last] = torch.from_numpy(out)
        return io.NodeOutput(result, ui=ui.PreviewText(f"cloud cut at frame {first + cut}, dissolved over {span}"))


class GeoBlockSubject(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockSubject",
            display_name="GeoBlock 7 · Subject",
            category="GeoBlock/Shot",
            description="The person the Earth zoom lands on. Connect a photo of your own figure, or leave it "
                        "empty and describe one. With subject_sheet on, the keyframe painter first makes one clean "
                        "full-length picture of the person (from the photo, or from the description) with "
                        "'sheet prompt', and every eye-level keyframe is given that picture.",
            inputs=[io.Image.Input("photo", optional=True,
                                   tooltip="A clear photo of the person; full length is best. Leave empty to "
                                           "invent one."),
                    io.String.Input("description", default="", multiline=True,
                                    tooltip="Used when there is no photo (and as a hint with one). Empty: a "
                                            "hiker dressed for the weather."),
                    io.Combo.Input("weather", options=list(WEATHER), default=DEFAULT_WEATHER, optional=True,
                                   tooltip="Dresses an invented person for the weather."),
                    io.Boolean.Input("subject_sheet", default=True, optional=True,
                                     tooltip="A subject sheet goes to the keyframes instead of the raw photo."),
                    io.Boolean.Input("chroma", default=False, optional=True,
                                     tooltip="Paint the sheet on a green chroma-key wall, so 'Place Subject' can "
                                             "cut the person out cleanly.")],
            outputs=[Subject.Output(display_name="subject"),
                     io.Image.Output(display_name="photo", tooltip="The photo, or a blank image without one."),
                     io.String.Output(display_name="sheet prompt",
                                      tooltip="klein prompt for the subject sheet (photo as Image 1, if any).")],
        )

    @classmethod
    def execute(cls, description, photo=None, weather=DEFAULT_WEATHER, subject_sheet=True, chroma=False):
        from geoblock_core.looks import default_subject, subject_sheet_prompt
        description = description.strip()
        has_photo = photo is not None
        subject = {"photo": has_photo, "image": has_photo or subject_sheet,
                   "description": description or ("" if has_photo else default_subject(weather))}
        image = photo if has_photo else torch.zeros((1, 64, 64, 3))
        return io.NodeOutput(subject, image, subject_sheet_prompt(subject["description"], has_photo, chroma,
                                                                         weather))


class GeoBlockFinalUpscalePrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from geoblock_core.upscale import TARGETS, UPSCALERS
        return io.Schema(
            node_id="GeoBlockFinalUpscalePrep",
            display_name="GeoBlock 8 · Final Upscale Prep",
            category="GeoBlock/Shot",
            description="Put between the LTX decode and the final upscaler. Frees the video model from VRAM "
                        "(SeedVR2 and FlashVSR load their weights outside ComfyUI's model manager), halves the "
                        "frames when the upscaler should rebuild detail from half size, and gives the output "
                        "height to wire into SeedVR2's resolution.",
            inputs=[io.Image.Input("images"),
                    io.Combo.Input("upscaler", options=list(UPSCALERS), default="seedvr2"),
                    io.Combo.Input("target", options=list(TARGETS), default="4k",
                                   tooltip="4k doubles the frames; 1080p keeps the size and rebuilds the detail."),
                    io.Boolean.Input("free_vram", default=True)],
            outputs=[io.Image.Output(display_name="images"),
                     io.Int.Output(display_name="resolution", tooltip="Output height (short edge) in pixels.")],
        )

    @classmethod
    def execute(cls, images, upscaler, target, free_vram):
        from geoblock_core.upscale import check, output_height, prescale
        check(upscaler, target)
        if free_vram:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        frames, height, width, _ = images.shape
        factor = prescale(upscaler, target)
        if factor != 1:
            size = (round(height * factor / 8) * 8, round(width * factor / 8) * 8)
            images = torch.cat([torch.nn.functional.interpolate(chunk.movedim(-1, 1), size=size, mode="area")
                                .movedim(1, -1) for chunk in images.split(16)])
        resolution = output_height(height, target)
        return io.NodeOutput(images, resolution,
                             ui=ui.PreviewText(f"{upscaler} to {target}: {frames} frames {width}x{height} -> "
                                               f"{images.shape[2]}x{images.shape[1]} in, {resolution} px tall out"))


class GeoBlockPlacePhotos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPlacePhotos",
            display_name="GeoBlock · Place Photos (Wikimedia Commons)",
            category="GeoBlock/Place",
            description="Find openly licensed Commons photos near the place (CC0, CC BY, CC BY-SA, public "
                        "domain) for H3's reference images. Review the picks; exclude any by page ID.",
            inputs=[Place.Input("place"),
                    io.Int.Input("count", default=3, min=1, max=4),
                    io.Float.Input("radius_km", default=2, min=0.1, max=10, step=0.1),
                    io.Float.Input("heading_degrees", default=-1, min=-1, max=359, step=1,
                                   tooltip="Favour photos titled as views this way; -1 ignores direction."),
                    io.String.Input("exclude_page_ids", default="",
                                    tooltip="Commons page IDs to skip, separated by spaces or commas.")],
            outputs=[io.Image.Output(display_name="photo 1"), io.Image.Output(display_name="photo 2"),
                     io.Image.Output(display_name="photo 3"), io.Image.Output(display_name="photo 4"),
                     io.String.Output(display_name="photo report")],
        )

    @classmethod
    def execute(cls, place, count, radius_km, heading_degrees, exclude_page_ids):
        paths, report = place_photos(place, count, radius_km, heading_degrees, exclude_page_ids)
        if len(paths) < count:
            report += f"\nOnly {len(paths)} of {count} found; the remaining outputs repeat the last photo."
        images = [_image(path) for path in paths]
        images += [images[-1]] * (4 - len(images))
        return io.NodeOutput(*images, report, ui=ui.PreviewText(report))


class GeoBlockResolvePlace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockResolvePlace",
            display_name="GeoBlock Example · Resolve Fixture",
            category="GeoBlock/Examples",
            description="Legacy prepared examples only. Use Search Landmarks and Choose Landmark for user-selected places.",
            inputs=[
                io.String.Input("place", default="Worm's Head, Rhossili, Wales"),
                io.Combo.Input("location_mode", options=["Search place name", "Exact coordinates"], default="Search place name"),
                io.Float.Input("latitude", default=51.5643, min=-90, max=90, step=0.000001),
                io.Float.Input("longitude", default=-4.3241, min=-180, max=180, step=0.000001),
                io.Float.Input("coverage_radius_km", default=2.5, min=0.1, max=20, step=0.1),
            ],
            outputs=[Place.Output(display_name="place"), io.String.Output(display_name="location report")],
        )

    @classmethod
    def execute(cls, place, location_mode, latitude, longitude, coverage_radius_km):
        resolved, report = resolve_place(place, location_mode, latitude, longitude, coverage_radius_km)
        return io.NodeOutput(resolved, report, ui=ui.PreviewText(report))


class GeoBlockTerrainDemo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockTerrainDemo",
            display_name="GeoBlock Example · Prepared Scene",
            category="GeoBlock/Examples",
            description="Prepared example scenes only. User-selected landmarks require their own validated geometry source.",
            inputs=[Place.Input("place")],
            outputs=[
                Scene.Output(display_name="terrain scene"),
                io.Image.Output(display_name="scene preview"),
                io.String.Output(display_name="terrain report"),
            ],
        )

    @classmethod
    def execute(cls, place):
        scene, report = load_demo_terrain(place)
        return io.NodeOutput(scene, _image(scene["preview_path"]), report, ui=ui.PreviewText(report))


class GeoBlockPlanShot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockPlanShot",
            display_name="GeoBlock Example · Plan Fixture Shot",
            category="GeoBlock/Prototype",
            description="Choose the camera path and target duration; the H3 frame count is derived automatically.",
            inputs=[
                Scene.Input("terrain_scene"),
                io.Combo.Input("camera_route", options=["Auto route", "Wide turn and return", "Wide turn, slower return", "Close measured-building orbit"], default="Auto route"),
                io.Float.Input("duration_seconds", default=7.3, min=5, max=15, step=0.1),
            ],
            outputs=[
                Shot.Output(display_name="shot"),
                io.Int.Output(display_name="H3 frames"),
                io.String.Output(display_name="shot report"),
            ],
        )

    @classmethod
    def execute(cls, terrain_scene, camera_route, duration_seconds):
        shot, report = plan_shot(terrain_scene, camera_route, duration_seconds)
        return io.NodeOutput(shot, shot["frame_count"], report, ui=ui.PreviewText(report))


class GeoBlockBuildReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockBuildReferences",
            display_name="GeoBlock Example · Build Fixture References",
            category="GeoBlock/Prototype",
            description="Retimes the control clip and loads checked place references, including close-shot opening and NAIP crops.",
            inputs=[
                Scene.Input("terrain_scene"),
                Shot.Input("shot"),
                io.Combo.Input("visual_style", options=["Photographic", "Clay study", "Clear-line comic"], default="Photographic"),
                io.Combo.Input("sea_motion", options=["Not applicable", "Breaking waves", "Calm sea"], default="Not applicable"),
            ],
            outputs=[
                io.Video.Output(display_name="control video"),
                io.Image.Output(display_name="place photo 1"),
                io.Image.Output(display_name="place photo 2"),
                io.String.Output(display_name="H3 prompt"),
                io.Int.Output(display_name="H3 frames"),
                io.String.Output(display_name="reference report"),
            ],
        )

    @classmethod
    def execute(cls, terrain_scene, shot, visual_style, sea_motion):
        pack, report = build_pack(
            terrain_scene, shot, visual_style, sea_motion, Path(folder_paths.get_input_directory())
        )
        video = InputImpl.VideoFromFile(pack["video_path"])
        return io.NodeOutput(
            video, _image(pack["photo_paths"][0]), _image(pack["photo_paths"][1]),
            pack["prompt"], pack["frame_count"], report,
            ui=ui.PreviewVideo([
                ui.SavedResult(
                    Path(pack["video_path"]).name,
                    "geoblock_prototype", io.FolderType.input,
                )
            ]),
        )


class GeoBlockShowReport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockShowReport",
            display_name="GeoBlock · Show Report",
            category="GeoBlock/Prototype",
            is_output_node=True,
            inputs=[io.String.Input("report", force_input=True)],
            outputs=[],
        )

    @classmethod
    def execute(cls, report):
        return io.NodeOutput(ui=ui.PreviewText(report))


class GeoBlockClearLineStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeoBlockClearLineStyle",
            display_name="GeoBlock · Clear-line Style",
            category="GeoBlock/Prototype",
            description="Load two original Woodside style frames for H3 image slots 3 and 4.",
            inputs=[Scene.Input("terrain_scene")],
            outputs=[io.Image.Output(display_name="style image 1"),
                     io.Image.Output(display_name="style image 2"),
                     io.String.Output(display_name="style report")],
        )

    @classmethod
    def execute(cls, terrain_scene):
        paths, report = load_style_refs(terrain_scene)
        return io.NodeOutput(_image(paths[0]), _image(paths[1]), report, ui=ui.PreviewText(report))


class GeoBlockPrototypeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeoBlockSearchLandmarks, GeoBlockChooseLandmark, GeoBlockPlaceFromCoordinates, GeoBlockPlaceFromCatalog,
            GeoBlockBuildLandformScene, GeoBlockCameraPreset, GeoBlockRenderBlockingVideo, GeoBlockKeyframePrompt, GeoBlockPushInCrop, GeoBlockReprojectHold, GeoBlockPlaceSubject, GeoBlockCleanPlate, GeoBlockFindPeople, GeoBlockFillPeople, GeoBlockColourMatch, GeoBlockColourSmooth, GeoBlockPunchThrough, GeoBlockSubject,
            GeoBlockFinalUpscalePrep,
            GeoBlockPlacePhotos,
            GeoBlockDiscoverSources, GeoBlockResolvePlace, GeoBlockTerrainDemo, GeoBlockPlanShot,
            GeoBlockBuildReferences, GeoBlockClearLineStyle, GeoBlockShowReport,
        ]


async def comfy_entrypoint() -> GeoBlockPrototypeExtension:
    return GeoBlockPrototypeExtension()
