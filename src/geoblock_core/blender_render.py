"""Render a GeoBlock camera route over its measured terrain as a blocking video.

Runs under Blender or the ``bpy`` Python module, with options after ``--``.
Normally launched by ``geoblock_core.blender.render_route``:

    blender -b --factory-startup --python src/geoblock_core/blender_render.py -- \
        --scene outputs/scenes/<id>/scene.json --route route.json --output-dir out/

Writes PNG frames, ``control.mp4`` (when ffmpeg is on PATH), ``first_frame.png``
and ``render.json``; with ``--depth`` also ``depth.mp4``, a near-white/far-black
depth video for H3's Fun ControlNet. The terrain is shaded by height with distance haze so the
scene edge fades instead of ending abruptly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import bpy
from mathutils import Vector
import numpy as np


# Blender's Python does not have GeoBlock installed; import it from beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoblock_core.height_policy import qualify_landform_heights
from geoblock_core.buildings import clear_of_path, extrude, load_buildings
from geoblock_core.trees import clear_of, feather_edges, proxy_mesh
from geoblock_core.weather import DEFAULT_WEATHER, WEATHER

ENGINES = {"eevee": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"), "workbench": ("BLENDER_WORKBENCH",),
           "cycles": ("CYCLES",)}
SKY = (0.45, 0.58, 0.78)
# Height-fraction colour stops. "default" suits any landform; "grand-canyon" follows the South Rim strata
# (Vishnu schist, Tonto platform, Redwall, Supai, Coconino, Kaibab) between the river and the forested rims.
PALETTES = {
    "default": ((0.0, (0.16, 0.11, 0.08)), (0.30, (0.42, 0.17, 0.08)), (0.55, (0.55, 0.28, 0.13)),
                (0.80, (0.66, 0.55, 0.40)), (1.0, (0.30, 0.34, 0.24))),
    "grand-canyon": ((0.0, (0.07, 0.06, 0.06)), (0.20, (0.14, 0.10, 0.09)), (0.24, (0.36, 0.20, 0.10)),
                     (0.30, (0.34, 0.36, 0.20)), (0.40, (0.48, 0.28, 0.16)), (0.44, (0.62, 0.26, 0.12)),
                     (0.55, (0.58, 0.16, 0.06)), (0.65, (0.66, 0.24, 0.09)), (0.70, (0.78, 0.62, 0.40)),
                     (0.78, (0.62, 0.56, 0.44)), (0.815, (0.20, 0.26, 0.10)), (1.0, (0.10, 0.18, 0.06))),
}
HAZE = {"default": 0.85, "grand-canyon": 0.4}  # how far distant ground fades to the sky
TREE_GREEN = (0.05, 0.10, 0.035)  # dark enough that keyframe painters read crowns, not moss or clay
BUILDING_WALL = (0.50, 0.48, 0.45)  # pale concrete: walls read as built, not as rock or clay
# Muted facade colours, one per building: pale enough to read as painted render or brick, not as rock or clay.
FACADES = (BUILDING_WALL, (0.62, 0.58, 0.50), (0.56, 0.55, 0.53), (0.58, 0.50, 0.41), (0.57, 0.45, 0.39),
           (0.47, 0.51, 0.55), (0.66, 0.65, 0.62), (0.50, 0.53, 0.46))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=tuple(ENGINES), default="eevee")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=1, help="Render every Nth frame (previews only)")
    parser.add_argument("--sun-azimuth", type=float, help="Degrees from north; default side-lights the route")
    parser.add_argument("--sun-elevation", type=float, help="Degrees; default comes from --weather")
    parser.add_argument("--depth", action="store_true", help="Also render depth.mp4 for depth control")
    parser.add_argument("--palette", choices=tuple(PALETTES), default="default", help="Terrain height colours")
    parser.add_argument("--weather", choices=tuple(WEATHER), default=DEFAULT_WEATHER, help="Sky, sun and haze")
    parser.add_argument("--imagery", type=Path, help="imagery.json from geoblock_core.imagery; overrides the palette")
    parser.add_argument("--trees", type=Path, help="trees-*.json from geoblock_core.trees: measured tree proxies")
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(cli)


def terrain_material(low: float, high: float, haze_start: float, haze_end: float,
                     stops=PALETTES["default"], haze_max: float = 0.85,
                     haze_colour=WEATHER[DEFAULT_WEATHER]["horizon"], imagery: dict | None = None,
                     snow: float = 0.0) -> bpy.types.Material:
    """Banded height colours, or draped satellite imagery, with aerial-perspective haze toward the sky colour.

    ``imagery`` is {"path": png, "bounds": (xmin, ymin, xmax, ymax)} in scene metres, north up. ``snow`` (0..1)
    whitens flat ground above the middle of the height range, leaving steep faces bare.
    """
    material = bpy.data.materials.new("GeoBlock terrain")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.9
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    height = nodes.new("ShaderNodeMapRange")
    height.inputs["From Min"].default_value = low
    height.inputs["From Max"].default_value = high
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = stops[0][0], (*stops[0][1], 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = stops[-1][0], (*stops[-1][1], 1)
    for position, rgb in stops[1:-1]:
        element = ramp.color_ramp.elements.new(position)
        element.color = (*rgb, 1)
    camera = nodes.new("ShaderNodeCameraData")
    haze = nodes.new("ShaderNodeMapRange")
    haze.name = "Haze"  # the Earth zoom rescales it with altitude
    haze.inputs["From Min"].default_value = haze_start
    haze.inputs["From Max"].default_value = haze_end
    haze.inputs["To Max"].default_value = haze_max
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["B"].default_value = (*haze_colour, 1)
    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], height.inputs["Value"])
    links.new(height.outputs["Result"], ramp.inputs["Fac"])
    links.new(camera.outputs["View Distance"], haze.inputs["Value"])
    links.new(haze.outputs["Result"], mix.inputs["Factor"])
    colour = ramp.outputs["Color"]
    if imagery:
        xmin, ymin, xmax, ymax = imagery["bounds"]
        u, v = nodes.new("ShaderNodeMapRange"), nodes.new("ShaderNodeMapRange")
        for node, low_, high_ in ((u, xmin, xmax), (v, ymin, ymax)):
            node.inputs["From Min"].default_value, node.inputs["From Max"].default_value = low_, high_
        combine = nodes.new("ShaderNodeCombineXYZ")
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(imagery["path"]))
        texture.extension = "EXTEND"
        links.new(separate.outputs["X"], u.inputs["Value"])
        links.new(separate.outputs["Y"], v.inputs["Value"])
        links.new(u.outputs["Result"], combine.inputs["X"])
        links.new(v.outputs["Result"], combine.inputs["Y"])
        links.new(combine.outputs["Vector"], texture.inputs["Vector"])
        # Satellite reflectance looks pale and flat next to how the place reads from the ground.
        grade = nodes.new("ShaderNodeHueSaturation")
        grade.inputs["Saturation"].default_value, grade.inputs["Value"].default_value = 1.5, 0.8
        links.new(texture.outputs["Color"], grade.inputs["Color"])
        colour = grade.outputs["Color"]
    if snow > 0:
        colour = snow_cover(nodes, links, colour, geometry, height.outputs["Result"], snow)
    links.new(colour, mix.inputs["A"])
    links.new(mix.outputs["Result"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def snow_cover(nodes, links, colour, geometry, height_fraction, amount: float, line: float = 0.5):
    """Snow on ground flatter than about 40 degrees and above ``line`` of the height range, broken up by noise so
    it reads as drifts and patches rather than paint."""
    flat = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Normal"], flat.inputs["Vector"])
    slope = nodes.new("ShaderNodeMapRange")
    slope.inputs["From Min"].default_value, slope.inputs["From Max"].default_value = 0.72, 0.9
    links.new(flat.outputs["Z"], slope.inputs["Value"])
    above = nodes.new("ShaderNodeMapRange")
    above.inputs["From Min"].default_value, above.inputs["From Max"].default_value = line - 0.12, line + 0.08
    links.new(height_fraction, above.inputs["Value"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.004  # about 250 m patches, in metres of position
    noise.inputs["Detail"].default_value = 6.0
    links.new(geometry.outputs["Position"], noise.inputs["Vector"])
    patches = nodes.new("ShaderNodeMapRange")
    patches.inputs["From Min"].default_value, patches.inputs["From Max"].default_value = 0.3, 0.55
    links.new(noise.outputs["Fac"], patches.inputs["Value"])
    factor = None
    for value in (slope.outputs["Result"], above.outputs["Result"], patches.outputs["Result"]):
        if factor is None:
            factor = value
            continue
        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        links.new(factor, multiply.inputs[0])
        links.new(value, multiply.inputs[1])
        factor = multiply.outputs["Value"]
    scaled = nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = amount
    links.new(factor, scaled.inputs[0])
    white = nodes.new("ShaderNodeMix")
    white.data_type = "RGBA"
    white.inputs["B"].default_value = (0.90, 0.92, 0.96, 1)
    links.new(scaled.outputs["Value"], white.inputs["Factor"])
    links.new(colour, white.inputs["A"])
    return white.outputs["Result"]


def fog_material(index: int, density: float, scale_m: float) -> bpy.types.Material:
    """A sheet of low cloud: noise thresholded to ``density`` with soft billows about ``scale_m`` across."""
    material = bpy.data.materials.new(f"Fog {index}")
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["W"].default_value = 1.3 + 2.1 * index
    noise.inputs["Scale"].default_value = 1.0 / scale_m
    noise.inputs["Detail"].default_value = 7.0
    noise.inputs["Roughness"].default_value = 0.55
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = max(0.0, 0.62 - 0.4 * density)
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = min(1.0, ramp.color_ramp.elements[0].position + 0.22)
    ramp.color_ramp.elements[1].color = (0.92, 0.92, 0.92, 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.93, 0.93, 0.95, 1)
    shader.inputs["Roughness"].default_value = 1.0
    links.new(ramp.outputs["Color"], shader.inputs["Alpha"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def add_fog(scene_info: dict, fog: dict, size_m: float) -> list:
    """Two sheets of low cloud lying in the low ground, ``fog["level"]`` of the way from the lowest ground up to
    the landmark's own level, so the landmark and the high ridges stand clear of it (a cloud inversion)."""
    low = scene_info["elevation_range_m"][0]
    top = low + fog["level"] * (float(scene_info["target_height_m"]) - low)
    sheets = []
    for index, (dz, density) in enumerate(((0.0, fog["density"]), (45.0, 0.55 * fog["density"]))):
        bpy.ops.mesh.primitive_plane_add(size=size_m, location=(0, 0, top + dz))
        sheet = bpy.context.active_object
        sheet.name = f"Fog sheet {index}"
        sheet.data.materials.append(fog_material(index, density, 0.04 * float(scene_info["radius_m"])))
        sheets.append(sheet)
    return sheets


def _imagery(manifest_path: Path | None, scene_info: dict) -> dict | None:
    if not manifest_path:
        return None
    manifest = json.loads(manifest_path.read_text())
    east, north = scene_info["origin_projected_m"]
    xmin, ymin, xmax, ymax = manifest["bounds_projected_m"]
    return {"path": manifest["texture"], "bounds": (xmin - east, ymin - north, xmax - east, ymax - north)}


def add_trees(terrain, manifest_path: Path, material: bpy.types.Material,
              keep_clear=(), path_clear=(), path_margin_m: float = 1.5) -> bpy.types.Object | None:
    """One mesh of low-poly tree proxies standing on the terrain mesh as rendered (not the lidar ground).

    ``terrain`` is one mesh object or several (the highest surface under a tree wins). Trees around any
    ``keep_clear`` point (camera positions, the stand-in) are left out, and those within ``path_margin_m`` of a
    ``path_clear`` point (a fast camera's path, which a tree beside it whips across the frame).
    """
    from mathutils.bvhtree import BVHTree

    manifest = json.loads(manifest_path.read_text())
    depsgraph = bpy.context.evaluated_depsgraph_get()
    terrains = terrain if isinstance(terrain, (list, tuple)) else [terrain]
    # BVHTree.FromObject works in object space; the terrains are placed with identity transforms or baked.
    trees_bvh = [BVHTree.FromObject(obj, depsgraph) for obj in terrains]
    top = max((obj.matrix_world @ Vector(corner)).z for obj in terrains for corner in obj.bound_box) + 100
    standing = []
    for east, north, height, radius in manifest["trees"]:
        hits = [hit[0].z for bvh in trees_bvh
                if (hit := bvh.ray_cast(Vector((east, north, top)), Vector((0, 0, -1))))[0] is not None]
        if hits:
            standing.append((east, north, max(hits) - 0.2, height, radius))
    standing = feather_edges(standing)
    if len(keep_clear):
        standing = clear_of(standing, keep_clear)
    if len(path_clear):
        standing = clear_of(standing, path_clear, margin_m=path_margin_m)
    vertices, faces = proxy_mesh(standing)
    if not len(faces):
        return None
    mesh = bpy.data.meshes.new("Tree proxies")  # foreach_set: from_pydata is slow for millions of vertices
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.astype("float32").ravel())
    mesh.loops.add(3 * len(faces))
    mesh.loops.foreach_set("vertex_index", faces.astype("int32").ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", (3 * np.arange(len(faces))).astype("int32"))
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype="int32"))
    mesh.update()
    mesh.validate()
    trees = bpy.data.objects.new("Tree proxies", mesh)
    bpy.context.scene.collection.objects.link(trees)
    trees.data.materials.append(material)
    print(f"GEOBLOCK_TREES {len(standing)} of {len(manifest['trees'])} from {manifest['dataset']}", flush=True)
    return trees


def add_buildings(terrain, manifest_path: Path, material: bpy.types.Material,
                  keep_clear=()) -> bpy.types.Object | None:
    """One mesh of flat-roofed building blocks standing on the terrain mesh as rendered.

    Each block's walls start a metre under the lowest ground at its corners, so none floats on a slope; its
    roof is its measured height over the middle of that ground. Blocks that would reach a ``keep_clear``
    point (camera positions, the stand-in) are left out.
    """
    from mathutils.bvhtree import BVHTree

    manifest = json.loads(manifest_path.read_text())
    rings, bases, heights = load_buildings(manifest)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    terrains = terrain if isinstance(terrain, (list, tuple)) else [terrain]
    bvhs = [BVHTree.FromObject(obj, depsgraph) for obj in terrains]
    top = max((obj.matrix_world @ Vector(corner)).z for obj in terrains for corner in obj.bound_box) + 100
    down = Vector((0, 0, -1))
    standing, bottoms, tops, shades = [], [], [], []
    # Each building its own facade colour (a fixed draw per building), so terraced rows of attached houses read
    # as separate houses and not as one long slab.
    for ring, base, height, shade in zip(rings, bases, heights, np.random.default_rng(7).random(len(rings))):
        ground = []
        for x, y in ring:
            hits = [hit[0].z for bvh in bvhs if (hit := bvh.ray_cast(Vector((x, y, top)), down))[0] is not None]
            if hits:
                ground.append(max(hits))
        if len(ground) < len(ring):  # off the terrain
            continue
        middle = float(np.median(ground))
        standing.append(ring)
        shades.append(shade)
        bottoms.append(min(ground) - 1.0 if base <= 0 else middle + base)
        tops.append(middle + height)
    bottoms, tops = np.array(bottoms), np.array(tops)
    if len(keep_clear) and standing:
        keep = clear_of_path(standing, tops, keep_clear)
        standing = [ring for ring, k in zip(standing, keep) if k]
        shades = [shade for shade, k in zip(shades, keep) if k]
        bottoms, tops = bottoms[keep], tops[keep]
    vertices, loops, loop_starts, totals = extrude(standing, bottoms, tops)
    if not len(totals):
        return None
    mesh = bpy.data.meshes.new("Building blocks")
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.astype("float32").ravel())
    mesh.loops.add(len(loops))
    mesh.loops.foreach_set("vertex_index", loops.astype("int32"))
    mesh.polygons.add(len(totals))
    mesh.polygons.foreach_set("loop_start", loop_starts.astype("int32"))
    mesh.polygons.foreach_set("loop_total", totals.astype("int32"))
    mesh.polygons.foreach_set("use_smooth", np.zeros(len(totals), dtype=bool))  # crisp edges, not rounded tanks
    shade = mesh.attributes.new("building_shade", "FLOAT", "FACE")  # one roof, then one face per wall
    shade.data.foreach_set("value", np.repeat(shades, [1 + len(ring) for ring in standing]).astype("float32"))
    mesh.update()
    mesh.validate()
    if os.environ.get("GEOBLOCK_BUILDING_ROOFS") == "tri":  # A/B: triangulate the n-gon roofs once, up front
        import bmesh
        work = bmesh.new()
        work.from_mesh(mesh)
        bmesh.ops.triangulate(work, faces=[f for f in work.faces if len(f.verts) > 4])
        work.to_mesh(mesh)
        work.free()
    buildings = bpy.data.objects.new("Building blocks", mesh)
    bpy.context.scene.collection.objects.link(buildings)
    buildings.data.materials.append(material)
    if os.environ.get("GEOBLOCK_BUILDING_SHADOWS") == "0":  # A/B: blocks cast no sun shadows
        buildings.visible_shadow = False
    print(f"GEOBLOCK_BUILDINGS {len(standing)} of {len(rings)} from Overture {manifest['release']}", flush=True)
    return buildings


def building_walls(material: bpy.types.Material) -> bpy.types.Material:
    """Turn a terrain material into the buildings' one: roofs keep the draped aerial imagery (or palette), so
    the city reads the same from above with or without blocks; walls are one of a few muted facade colours per
    building (``building_shade``), with a little of the imagery."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    haze_mix = next(n for n in nodes if n.type == "MIX" and n.inputs["Factor"].is_linked
                    and n.inputs["Factor"].links[0].from_node.name == "Haze")
    colour = haze_mix.inputs["A"].links[0].from_socket
    geometry = nodes.new("ShaderNodeNewGeometry")
    normal = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Normal"], normal.inputs["Vector"])
    roof = nodes.new("ShaderNodeMath")
    roof.operation = "GREATER_THAN"
    roof.inputs[1].default_value = 0.7
    links.new(normal.outputs["Z"], roof.inputs[0])
    wall = nodes.new("ShaderNodeMix")
    wall.data_type = "RGBA"
    wall.inputs["Factor"].default_value = 0.75
    shade = nodes.new("ShaderNodeAttribute")
    shade.attribute_name = "building_shade"
    facades = nodes.new("ShaderNodeValToRGB")
    facades.color_ramp.interpolation = "CONSTANT"
    elements = facades.color_ramp.elements
    for _ in range(len(FACADES) - len(elements)):
        elements.new(0.0)
    for element, (position, facade) in zip(elements, enumerate(FACADES)):
        element.position, element.color = position / len(FACADES), (*facade, 1)
    links.new(shade.outputs["Fac"], facades.inputs["Fac"])
    links.new(facades.outputs["Color"], wall.inputs["B"])
    links.new(colour, wall.inputs["A"])
    pick = nodes.new("ShaderNodeMix")
    pick.data_type = "RGBA"
    links.new(roof.outputs["Value"], pick.inputs["Factor"])
    links.new(wall.outputs["Result"], pick.inputs["A"])
    links.new(colour, pick.inputs["B"])
    links.new(pick.outputs["Result"], haze_mix.inputs["A"])
    material.name = "GeoBlock buildings"
    return material


def sky_world(weather: dict) -> bpy.types.World:
    """Flat haze light for the terrain; the camera sees a horizon-to-zenith blue gradient instead."""
    world = bpy.data.worlds.new("Haze sky")
    world.use_nodes = True
    world.color = SKY
    nodes, links = world.node_tree.nodes, world.node_tree.links
    light = nodes["Background"]
    light.inputs["Color"].default_value = (*weather.get("ambient_colour", SKY), 1)
    # Less sun means more of the light comes from the sky (except at dusk and night, when there is little of either).
    light.inputs["Strength"].default_value = (0.35 * (1 + max(0.0, 4.5 - weather["sun_energy"]) / 3)
                                              * weather.get("ambient", 1.0))
    coords = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (*weather["horizon"], 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.35, (*weather["zenith"], 1)
    seen = nodes.new("ShaderNodeBackground")
    path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(coords.outputs["Generated"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], ramp.inputs["Fac"])
    sky = ramp.outputs["Color"]
    if weather.get("sky_clouds"):
        sky = cloud_layer(nodes, links, coords, separate, sky, *weather["sky_clouds"])
    if weather.get("stars"):
        sky = star_field(nodes, links, coords, separate, sky)
    links.new(sky, seen.inputs["Color"])
    links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(light.outputs["Background"], mix.inputs[1])
    links.new(seen.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], nodes["World Output"].inputs["Surface"])
    return world


def cloud_layer(nodes, links, coords, separate, sky, cover: float, colour):
    """Cumulus on a flat cloud ceiling: noise on the view direction divided by its height, so clouds shrink and
    crowd toward the horizon as real ones do. ``cover`` 0..1."""
    lift = nodes.new("ShaderNodeMath")
    lift.operation = "MAXIMUM"
    lift.inputs[1].default_value = 0.03
    links.new(separate.outputs["Z"], lift.inputs[0])
    plane = nodes.new("ShaderNodeVectorMath")
    plane.operation = "DIVIDE"
    links.new(coords.outputs["Generated"], plane.inputs[0])
    links.new(lift.outputs["Value"], plane.inputs[1])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.9
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    links.new(plane.outputs["Vector"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeMapRange")
    start = 0.62 - 0.3 * cover
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = start, start + 0.12
    links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    up = nodes.new("ShaderNodeMapRange")  # thin out right at the horizon, where the haze is
    up.inputs["From Min"].default_value, up.inputs["From Max"].default_value = 0.0, 0.08
    links.new(separate.outputs["Z"], up.inputs["Value"])
    amount = nodes.new("ShaderNodeMath")
    amount.operation = "MULTIPLY"
    links.new(ramp.outputs["Result"], amount.inputs[0])
    links.new(up.outputs["Result"], amount.inputs[1])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["B"].default_value = (*colour, 1)
    links.new(amount.outputs["Value"], mix.inputs["Factor"])
    links.new(sky, mix.inputs["A"])
    return mix.outputs["Result"]


def star_field(nodes, links, coords, separate, sky):
    """Small bright points over the camera-visible sky above the horizon, added to its colour."""
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 140.0  # cells about half a degree across: a few pixels at 1280 wide
    voronoi.inputs["Randomness"].default_value = 1.0
    links.new(coords.outputs["Generated"], voronoi.inputs["Vector"])
    point = nodes.new("ShaderNodeMapRange")
    point.inputs["From Min"].default_value, point.inputs["From Max"].default_value = 0.0, 0.12
    point.inputs["To Min"].default_value, point.inputs["To Max"].default_value = 1.0, 0.0
    links.new(voronoi.outputs["Distance"], point.inputs["Value"])
    sharp = nodes.new("ShaderNodeMath")
    sharp.operation = "POWER"
    sharp.inputs[1].default_value = 2.0
    links.new(point.outputs["Result"], sharp.inputs[0])
    level = nodes.new("ShaderNodeSeparateColor")  # a random brightness per star; most stay faint
    links.new(voronoi.outputs["Color"], level.inputs["Color"])
    cube = nodes.new("ShaderNodeMath")
    cube.operation = "POWER"
    cube.inputs[1].default_value = 3.0
    links.new(level.outputs[0], cube.inputs[0])
    bright = nodes.new("ShaderNodeMath")
    bright.operation = "MULTIPLY"
    links.new(sharp.outputs["Value"], bright.inputs[0])
    links.new(cube.outputs["Value"], bright.inputs[1])
    boost = nodes.new("ShaderNodeMath")
    boost.operation = "MULTIPLY"
    boost.inputs[1].default_value = 4.0
    links.new(bright.outputs["Value"], boost.inputs[0])
    bright = boost
    up = nodes.new("ShaderNodeMapRange")
    up.inputs["From Min"].default_value, up.inputs["From Max"].default_value = 0.0, 0.05
    links.new(separate.outputs["Z"], up.inputs["Value"])
    masked = nodes.new("ShaderNodeMath")
    masked.operation = "MULTIPLY"
    links.new(bright.outputs["Value"], masked.inputs[0])
    links.new(up.outputs["Result"], masked.inputs[1])
    add = nodes.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.inputs["Factor"].default_value = 1.0
    links.new(sky, add.inputs["A"])
    links.new(masked.outputs["Value"], add.inputs["B"])
    return add.outputs["Result"]


def sun_light(weather: dict) -> bpy.types.Object:
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy, sun_data.angle = weather["sun_energy"], math.radians(1.5)
    sun_data.color = weather.get("sun_colour", (1.0, 1.0, 1.0))
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    return sun


def depth_material(near: float, far: float) -> bpy.types.Material:
    """Unlit log-distance shade: white at ``near`` metres, black at ``far`` and beyond (the sky)."""
    material = bpy.data.materials.new("GeoBlock depth")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    camera = nodes.new("ShaderNodeCameraData")
    log = nodes.new("ShaderNodeMath")
    log.operation = "LOGARITHM"
    log.inputs[1].default_value = math.e
    shade = nodes.new("ShaderNodeMapRange")
    shade.inputs["From Min"].default_value = math.log(near)
    shade.inputs["From Max"].default_value = math.log(far)
    shade.inputs["To Min"].default_value = 1.0
    shade.inputs["To Max"].default_value = 0.0
    links.new(camera.outputs["View Distance"], log.inputs[0])
    links.new(log.outputs["Value"], shade.inputs["Value"])
    links.new(shade.outputs["Result"], emission.inputs["Strength"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def encode(frames_dir: Path, video: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:g}",
                        "-i", str(frames_dir / "frame_%04d.png"), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
    return bool(ffmpeg)


def render_frames(scene: bpy.types.Scene, frames: list[int], frames_dir: Path, label: str) -> None:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for index, frame in enumerate(frames, start=1):
        scene.frame_set(frame)
        scene.render.filepath = str((frames_dir / f"frame_{index:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        if index == 1 or index % 24 == 0 or index == len(frames):
            print(f"GEOBLOCK_PROGRESS{label} {index}/{len(frames)}", flush=True)


def set_engine(scene: bpy.types.Scene, name: str, samples: int) -> str:
    for engine in ENGINES[name]:
        try:
            scene.render.engine = engine  # EEVEE's identifier differs between Blender 4 and 5
            break
        except TypeError:
            continue
    else:
        raise ValueError(f"This Blender has no {name} render engine.")
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.transparent_max_bounces = 64  # the Earth zoom's stacked cloud sheets go black at 8
        scene.cycles.use_denoising = True
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.type == backend for d in prefs.devices):
                    for device in prefs.devices:
                        device.use = device.type == backend
                    scene.cycles.device = "GPU"
                    break
            except TypeError:
                continue
    elif engine == "BLENDER_WORKBENCH":
        shading = scene.display.shading
        shading.light, shading.color_type = "STUDIO", "MATERIAL"
        shading.show_cavity, shading.show_shadows = True, True
    else:
        scene.eevee.taa_render_samples = samples
    return engine


def main() -> None:
    args = options()
    weather = WEATHER[args.weather]
    scene_info = json.loads(args.scene.read_text())
    route = json.loads(args.route.read_text())
    terrain_path = Path(scene_info["terrain_path"])
    if Path(route["terrain_path"]).resolve() != terrain_path.resolve():
        raise ValueError("The route was planned for a different terrain mesh.")
    evidence = json.loads(Path(scene_info["geometry_evidence_path"]).read_text())
    qualify_landform_heights(evidence, terrain_path, tuple(scene_info["origin_projected_m"]))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(terrain_path.resolve()))
    terrain = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
    terrain.name = "Measured terrain"
    for polygon in terrain.data.polygons:
        polygon.use_smooth = True
    low, high = scene_info["elevation_range_m"]
    radius = float(scene_info["radius_m"])
    terrain.data.materials.clear()
    visibility = weather.get("visibility", 1.0)
    terrain.data.materials.append(terrain_material(low, high, 0.35 * radius * visibility, 1.6 * radius * visibility,
                                                      PALETTES[args.palette],
                                                      min(0.95, HAZE[args.palette] * weather["haze"]),
                                                      weather["horizon"], _imagery(args.imagery, scene_info),
                                                      weather.get("snow", 0.0)))
    terrain.active_material.diffuse_color = (0.72, 0.58, 0.45, 1)  # Workbench colour
    trees = None
    if args.trees:
        tree_material = terrain_material(low, high, 0.35 * radius, 1.6 * radius, ((0.0, TREE_GREEN), (1.0, TREE_GREEN)),
                                         min(0.95, HAZE[args.palette] * weather["haze"]), weather["horizon"])
        tree_material.diffuse_color = (*TREE_GREEN, 1)
        trees = add_trees(terrain, args.trees, tree_material, keep_clear=route["positions"])

    scene = bpy.context.scene
    engine = set_engine(scene, args.engine, args.samples)
    scene.world = sky_world(weather)
    fog = add_fog(scene_info, weather["fog"], 2.4 * radius) if weather.get("fog") else []
    sun = sun_light(weather)
    azimuth = args.sun_azimuth if args.sun_azimuth is not None else route["heading_deg"] + 100
    az, el = math.radians(azimuth), math.radians(args.sun_elevation if args.sun_elevation is not None
                                                          else weather["sun_elevation"])
    toward_sun = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
    sun.rotation_euler = (-toward_sun).to_track_quat("-Z", "Y").to_euler()

    camera_data = bpy.data.cameras.new("Route camera")
    camera_data.lens = route["lens_mm"]
    camera_data.clip_start, camera_data.clip_end = 1.0, 8 * radius
    camera = bpy.data.objects.new("Route camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.rotation_mode = "QUATERNION"
    for frame, (position, look) in enumerate(zip(route["positions"], route["look_at"]), start=1):
        camera.location = position
        camera.rotation_quaternion = (Vector(look) - Vector(position)).to_track_quat("-Z", "Y")
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)

    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = route["fps"]
    scene.frame_start, scene.frame_end = 1, route["frames"]
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.file_format = "PNG"
    rendered = list(range(1, route["frames"] + 1, max(1, args.frame_step)))
    frames_dir = args.output_dir / "frames"
    render_frames(scene, rendered, frames_dir, "")
    shutil.copyfile(frames_dir / "frame_0001.png", args.output_dir / "first_frame.png")
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output_dir / "blocking.blend").resolve()))

    fps = route["fps"] / max(1, args.frame_step)
    video = args.output_dir / "control.mp4"
    ffmpeg = encode(frames_dir, video, fps)
    depth_video = None
    if args.depth:
        terrain.data.materials.clear()
        terrain.data.materials.append(depth_material(near=30.0, far=1.6 * radius))
        if trees:
            trees.data.materials.clear()
            trees.data.materials.append(terrain.data.materials[0])
        scene.world = bpy.data.worlds.new("Black")
        scene.world.use_nodes = True
        scene.world.color = (0, 0, 0)
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        sun.hide_render = True
        for sheet in fog:
            sheet.hide_render = True  # depth shows the ground under the fog
        scene.view_settings.view_transform = "Standard"
        if engine == "BLENDER_EEVEE" or engine == "BLENDER_EEVEE_NEXT":
            scene.eevee.taa_render_samples = 1
        render_frames(scene, rendered, args.output_dir / "depth_frames", "_DEPTH")
        depth_video = args.output_dir / "depth.mp4"
        if not encode(args.output_dir / "depth_frames", depth_video, fps):
            depth_video = None
    manifest = {
        "route_sha256": route["route_sha256"], "preset": route["preset"], "engine": engine,
        "blender_version": bpy.app.version_string, "resolution": [args.width, args.height],
        "frames": len(rendered), "frame_step": args.frame_step, "fps": route["fps"],
        "video": str(video) if ffmpeg else None, "first_frame": str(args.output_dir / "first_frame.png"),
        "depth_video": str(depth_video) if depth_video else None,
        "terrain_path": str(terrain_path), "attribution": scene_info.get("attribution"),
        "trees": str(args.trees) if args.trees else None,
    }
    (args.output_dir / "render.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GEOBLOCK_RENDERED " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
