"""Headless Blender renderer for the Earth Zoom: orbit, through the clouds, down to the landmark.

One scene in metres with the landmark at the origin on the local tangent plane:

- a globe (radius 6 371 km, centre straight below) textured with Blue Marble, plus a sharper
  cap around the landmark, and a cloud shell from one real day of VIIRS imagery;
- a thin limb-glow atmosphere;
- the regional terrain (about 100 km radius) and the local terrain, both bent to the
  Earth's curvature and draped with NASA HLS imagery (or the height palette);
- a stack of cloud sheets a few kilometres up that the camera dives through, which hides
  the change from globe texture to measured terrain.

Run with ``blender -b --factory-startup --python blender_earth_zoom.py -- --zoom zoom.json ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geoblock_core.weather import DEFAULT_WEATHER, WEATHER  # noqa: E402
from geoblock_core.ride import RIDE_FLOOR_FRAMES, SMOOTH_TEMPOS, ride_floor  # noqa: E402
import blender_render as base  # noqa: E402

EARTH_RADIUS = 6_371_000.0
CLOUD_SHEET_SPACING_M = 500  # stacked cloud deck sheets, base to top
DECK_EXIT_FRAMES = 8  # the deck thins out over this many frames before the camera reaches its base
DECK_EXIT_FADE_M = 800.0  # and over this height above the base, whichever ends it sooner
CLOUD_SHEET_ALPHA = 0.85  # one sheet's opacity where it is solid (the colour ramp's top)


def deck_thinning(share: float, sheets_below: int) -> float:
    """Each sheet's fade for the deck below the camera to let ``share`` (0..1) of its full cover through: the
    sheets stack, so fading each one linearly left the whiteout solid until the last frame of the exit and the
    ground appeared in one step (r16 blocking: frame change 24 at f39 against 11-13 around it, and LTX cut there).
    Faded so the stack's opacity falls linearly, the ground shows through steadily."""
    n = max(1, sheets_below)
    full = 1.0 - (1.0 - CLOUD_SHEET_ALPHA) ** n
    return (1.0 - (1.0 - share * full) ** (1.0 / n)) / CLOUD_SHEET_ALPHA


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoom", type=Path, required=True, help="zoom.json: route, scenes, globe, imagery")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-range", help="FIRST:LAST (1-based, inclusive): render only these frames, e.g. "
                                              "for timing; the video and manifest then cover only them")
    parser.add_argument("--palette", choices=tuple(base.PALETTES), default="default")
    parser.add_argument("--weather", choices=tuple(WEATHER), default=DEFAULT_WEATHER)
    parser.add_argument("--engine", choices=tuple(base.ENGINES), default="eevee",
                        help="cycles runs without a GPU (slow previews)")
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(cli)


def globe_frame(latitude: float, longitude: float) -> Matrix:
    """Object-to-world matrix that puts (latitude, longitude) at the top of a globe below the origin,
    with geographic north along world +Y."""
    phi, lam = math.radians(latitude), math.radians(longitude)
    up = Vector((math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)))
    east = Vector((-math.sin(lam), math.cos(lam), 0.0))
    north = up.cross(east)
    rotation = Matrix((east, north, up))  # rows: world X, Y, Z expressed in globe object axes
    return Matrix.Translation((0, 0, -EARTH_RADIUS)) @ rotation.to_4x4()


def lonlat_uv(nodes, links):
    """Shader nodes giving equirectangular (u, v) from the object-space direction of a globe point."""
    coords = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    norm = nodes.new("ShaderNodeVectorMath")
    norm.operation = "NORMALIZE"
    links.new(coords.outputs["Object"], norm.inputs[0])
    links.new(norm.outputs["Vector"], sep.inputs["Vector"])
    lon = nodes.new("ShaderNodeMath")
    lon.operation = "ARCTAN2"
    links.new(sep.outputs["Y"], lon.inputs[0])
    links.new(sep.outputs["X"], lon.inputs[1])
    lat = nodes.new("ShaderNodeMath")
    lat.operation = "ARCSINE"
    links.new(sep.outputs["Z"], lat.inputs[0])
    return lon.outputs["Value"], lat.outputs["Value"]


def remap(nodes, links, value, low, high):
    node = nodes.new("ShaderNodeMapRange")
    node.clamp = False
    node.inputs["From Min"].default_value, node.inputs["From Max"].default_value = low, high
    links.new(value, node.inputs["Value"])
    return node.outputs["Result"]


def image_lookup(nodes, links, path, u, v, *, colour=True):
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(u, combine.inputs["X"])
    links.new(v, combine.inputs["Y"])
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path))
    if not colour:
        tex.image.colorspace_settings.name = "Non-Color"
    tex.extension = "EXTEND"
    tex.interpolation = "Cubic"
    links.new(combine.outputs["Vector"], tex.inputs["Vector"])
    return tex.outputs["Color"]


def cap_mask(nodes, links, lon_deg_u, lat_deg_v):
    """1 inside the cap's (u, v) unit square, with a soft edge, else 0."""
    total = None
    for value in (lon_deg_u, lat_deg_v):
        for edge, op in ((0.04, "GREATER_THAN"), (0.96, "LESS_THAN")):
            node = nodes.new("ShaderNodeMath")
            node.operation = op
            node.inputs[1].default_value = edge
            links.new(value, node.inputs[0])
            if total is None:
                total = node.outputs["Value"]
            else:
                mul = nodes.new("ShaderNodeMath")
                mul.operation = "MULTIPLY"
                links.new(total, mul.inputs[0])
                links.new(node.outputs["Value"], mul.inputs[1])
                total = mul.outputs["Value"]
    return total


def globe_material(globe: dict, *, clouds: bool, weather: dict | None = None) -> bpy.types.Material:
    material = bpy.data.materials.new("Globe clouds" if clouds else "Globe ground")
    if clouds:
        blended(material)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    lon, lat = lonlat_uv(nodes, links)
    u = remap(nodes, links, lon, -math.pi, math.pi)
    v = remap(nodes, links, lat, -math.pi / 2, math.pi / 2)
    west, south, east, north = globe["cap_bounds_deg"]
    cu = remap(nodes, links, lon, math.radians(west), math.radians(east))
    cv = remap(nodes, links, lat, math.radians(south), math.radians(north))
    files = globe["files"]
    name = "clouds.png" if clouds else "ground.jpg"
    world_tex = image_lookup(nodes, links, files[f"globe_{name}"], u, v, colour=not clouds)
    cap_tex = image_lookup(nodes, links, files[f"cap_{name}"], cu, cv, colour=not clouds)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(cap_mask(nodes, links, cu, cv), mix.inputs["Factor"])
    links.new(world_tex, mix.inputs["A"])
    links.new(cap_tex, mix.inputs["B"])
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.95
    if clouds:
        shader.inputs["Base Color"].default_value = (*(weather or {}).get("shell_colour", (0.95, 0.95, 0.97)), 1)
        to_bw = nodes.new("ShaderNodeRGBToBW")
        links.new(mix.outputs["Result"], to_bw.inputs["Color"])
        fade = nodes.new("ShaderNodeValue")
        fade.name = "Fade"
        fade.outputs[0].default_value = 1.0
        faded = nodes.new("ShaderNodeMath")
        faded.operation = "MULTIPLY"
        links.new(to_bw.outputs["Val"], faded.inputs[0])
        links.new(fade.outputs[0], faded.inputs[1])
        links.new(faded.outputs["Value"], shader.inputs["Alpha"])
    else:  # night: the globe is dimmed, not just lit less, so the orbit reads as the night side
        dim = nodes.new("ShaderNodeMix")
        dim.data_type = "RGBA"
        dim.blend_type = "MULTIPLY"
        dim.inputs["Factor"].default_value = 1.0
        light = (weather or {}).get("globe_light", 1.0)
        dim.inputs["B"].default_value = (light, light, light, 1)
        links.new(mix.outputs["Result"], dim.inputs["A"])
        links.new(dim.outputs["Result"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def atmosphere_material(glow: float = 1.0) -> bpy.types.Material:
    """Blue limb glow: emission and alpha grow toward grazing angles."""
    material = blended(bpy.data.materials.new("Atmosphere"))
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    layer = nodes.new("ShaderNodeLayerWeight")
    layer.inputs["Blend"].default_value = 0.35
    power = nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.inputs[1].default_value = 3.0
    links.new(layer.outputs["Facing"], power.inputs[0])
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.35, 0.6, 1.0, 1)
    emission.inputs["Strength"].default_value = 2.0 * glow
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(power.outputs["Value"], mix.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def blended(material: bpy.types.Material) -> bpy.types.Material:
    """Sorted alpha blending: dithered transparency turns thin cloud into grain at low sample counts."""
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    return material


def cloud_sheet_material(index: int, cover: float, darkness: float, radius: float,
                         colour: tuple | None = None, dive: tuple | None = None) -> bpy.types.Material:
    """Procedural cumulus sheet: noise thresholded to the weather's cover, fading out toward its edge.

    ``dive`` is (x, y, width) of where the camera enters the deck: along the line from there to the landmark
    (the origin), within ``width``, the sheet is solid whatever the weather's cover, so the dive is a whiteout
    instead of a flight through gaps between cloud patches."""
    material = blended(bpy.data.materials.new(f"Cloud deck {index}"))
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.00012
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    noise.noise_dimensions = "4D"  # a different slice of the same noise per sheet
    noise.inputs["W"].default_value = index * 3.7
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    fac = noise.outputs["Fac"]
    if dive is not None:
        entry, width = Vector((dive[0], dive[1], 0.0)), dive[2]
        length = max(entry.length, 1.0)
        along_axis = nodes.new("ShaderNodeVectorMath")  # how far along the line from the landmark to the entry
        along_axis.operation = "DOT_PRODUCT"
        along_axis.inputs[1].default_value = entry / length
        links.new(coords.outputs["Object"], along_axis.inputs[0])
        along = nodes.new("ShaderNodeClamp")
        along.inputs["Min"].default_value, along.inputs["Max"].default_value = 0.0, length
        links.new(along_axis.outputs["Value"], along.inputs["Value"])
        nearest = nodes.new("ShaderNodeVectorMath")
        nearest.operation = "SCALE"
        nearest.inputs[0].default_value = entry / length
        links.new(along.outputs["Result"], nearest.inputs["Scale"])
        offset = nodes.new("ShaderNodeVectorMath")
        offset.operation = "DISTANCE"
        links.new(coords.outputs["Object"], offset.inputs[0])
        links.new(nearest.outputs["Vector"], offset.inputs[1])
        # A coarse noise roughens the corridor's outline, so from above it is a towering cloud bank, not a pill.
        rough = nodes.new("ShaderNodeTexNoise")
        rough.inputs["Scale"].default_value = 1.5 / width
        rough.inputs["Detail"].default_value = 4.0
        links.new(coords.outputs["Object"], rough.inputs["Vector"])
        jitter = nodes.new("ShaderNodeMath")
        jitter.operation = "MULTIPLY_ADD"
        jitter.inputs[1].default_value, jitter.inputs[2].default_value = 2.0 * width, -1.0 * width
        links.new(rough.outputs["Fac"], jitter.inputs[0])
        rough_offset = nodes.new("ShaderNodeMath")
        rough_offset.operation = "ADD"
        links.new(offset.outputs["Value"], rough_offset.inputs[0])
        links.new(jitter.outputs["Value"], rough_offset.inputs[1])
        solid = nodes.new("ShaderNodeMapRange")
        solid.interpolation_type = "SMOOTHSTEP"
        solid.inputs["From Min"].default_value, solid.inputs["From Max"].default_value = 0.2 * width, 1.2 * width
        solid.inputs["To Min"].default_value, solid.inputs["To Max"].default_value = 0.7, 0.0
        links.new(rough_offset.outputs["Value"], solid.inputs["Value"])
        boosted = nodes.new("ShaderNodeMath")
        boosted.operation = "ADD"
        links.new(fac, boosted.inputs[0])
        links.new(solid.outputs["Result"], boosted.inputs[1])
        fac = boosted.outputs["Value"]
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = max(0.0, 0.62 - 0.32 * cover)
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = min(1.0, ramp.color_ramp.elements[0].position + 0.18)
    ramp.color_ramp.elements[1].color = (0.85, 0.85, 0.85, 1)
    links.new(fac, ramp.inputs["Fac"])
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    grey = 0.95 - 0.7 * darkness
    shader.inputs["Base Color"].default_value = (*colour, 1) if colour else (grey, grey, grey * 1.02, 1)
    shader.inputs["Roughness"].default_value = 1.0
    edge = nodes.new("ShaderNodeVectorMath")
    edge.operation = "LENGTH"
    links.new(coords.outputs["Object"], edge.inputs[0])
    falloff = nodes.new("ShaderNodeMapRange")
    falloff.inputs["From Min"].default_value, falloff.inputs["From Max"].default_value = 0.4 * radius, radius
    falloff.inputs["To Min"].default_value, falloff.inputs["To Max"].default_value = 1.0, 0.0
    links.new(edge.outputs["Value"], falloff.inputs["Value"])
    alpha = nodes.new("ShaderNodeMath")
    alpha.operation = "MULTIPLY"
    links.new(ramp.outputs["Color"], alpha.inputs[0])
    links.new(falloff.outputs["Result"], alpha.inputs[1])
    fade = nodes.new("ShaderNodeValue")
    fade.name = "Fade"
    fade.outputs[0].default_value = 1.0
    faded = nodes.new("ShaderNodeMath")
    faded.operation = "MULTIPLY"
    links.new(alpha.outputs["Value"], faded.inputs[0])
    links.new(fade.outputs[0], faded.inputs[1])
    links.new(faded.outputs["Value"], shader.inputs["Alpha"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def import_terrain(path: str, name: str) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    obj = next(o for o in bpy.context.scene.objects if o not in before and o.type == "MESH")
    obj.name = name
    return obj


def taper_edges(obj: bpy.types.Object, radius: float, start: float = 0.8) -> None:
    """Bring heights down to the globe's sea level toward the terrain's edge, so a wide scene meets the
    globe instead of ending in a wall that casts a dark outline seen from high up."""
    for vertex in obj.data.vertices:
        edge = max(abs(vertex.co.x), abs(vertex.co.y)) / radius
        if edge > start:
            keep = max(0.0, 1 - (edge - start) / (1 - start))
            vertex.co.z *= keep * keep * (3 - 2 * keep)


def bend_to_globe(obj: bpy.types.Object, offset: Vector, sink_m: float = 0.0) -> None:
    """Move a tangent-plane terrain by ``offset`` (its origin relative to the landmark) and drop each
    vertex by the Earth's curvature, so wide terrain meets the globe instead of floating above it."""
    obj.data.transform(Matrix.Translation(offset) @ obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    for vertex in obj.data.vertices:
        x, y, z = vertex.co
        vertex.co.z = z - (x * x + y * y) / (2 * EARTH_RADIUS) - sink_m
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def cut_out(obj: bpy.types.Object, half_size: float) -> None:
    """Delete regional faces under the sharper local terrain."""
    import bmesh
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    edge = half_size + 1e-3  # faces touching the boundary belong to the inner mesh, which reaches it exactly
    inside = [f for f in mesh.faces if all(abs(v.co.x) <= edge and abs(v.co.y) <= edge for v in f.verts)]
    bmesh.ops.delete(mesh, geom=inside, context="FACES")
    mesh.to_mesh(obj.data)
    mesh.free()


def feather_into(inner: bpy.types.Object, outer: bpy.types.Object, radius: float, start: float = 0.85) -> None:
    """Blend the sharper inner terrain's heights into the outer terrain's toward its edge, so the nested
    patch meets the coarser mesh without a step or overlapping faces. Call before cutting the outer mesh."""
    for vertex in inner.data.vertices:
        edge = max(abs(vertex.co.x), abs(vertex.co.y)) / radius
        if edge <= start:
            continue
        hit, location, *_ = outer.ray_cast(Vector((vertex.co.x, vertex.co.y, 20_000.0)), Vector((0, 0, -1)))
        if hit:
            t = min(1.0, (edge - start) / (1 - start))
            t = t * t * (3 - 2 * t)
            vertex.co.z = (1 - t) * vertex.co.z + t * location.z


def feather_to_cap(material: bpy.types.Material, globe: dict, radius: float) -> None:
    """Blend the regional imagery into the globe's Blue Marble cap toward the terrain's edge, so the
    200 km square does not show as a patch of different colour from high up."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    mix = next(n for n in nodes if n.type == "MIX" and n.inputs["A"].is_linked and n.inputs["Factor"].is_linked)
    colour_link = next(l for l in links if l.to_socket == mix.inputs["A"])
    geometry = nodes.new("ShaderNodeNewGeometry")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Position"], sep.inputs["Vector"])
    lat0 = math.radians(globe["latitude"])
    west, south, east, north = globe["cap_bounds_deg"]
    # Local metres to cap (u, v): small-area equirectangular approximation around the landmark.
    u = remap(nodes, links, sep.outputs["X"], (west - globe["longitude"]) * 111_320 * math.cos(lat0),
              (east - globe["longitude"]) * 111_320 * math.cos(lat0))
    v = remap(nodes, links, sep.outputs["Y"], (south - globe["latitude"]) * 110_540,
              (north - globe["latitude"]) * 110_540)
    cap = image_lookup(nodes, links, globe["files"]["cap_ground.jpg"], u, v)
    edge = nodes.new("ShaderNodeMath")
    edge.operation = "MAXIMUM"
    ax, ay = nodes.new("ShaderNodeMath"), nodes.new("ShaderNodeMath")
    ax.operation = ay.operation = "ABSOLUTE"
    links.new(sep.outputs["X"], ax.inputs[0])
    links.new(sep.outputs["Y"], ay.inputs[0])
    links.new(ax.outputs["Value"], edge.inputs[0])
    links.new(ay.outputs["Value"], edge.inputs[1])
    fade = nodes.new("ShaderNodeMapRange")
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = 0.45 * radius, 0.95 * radius
    links.new(edge.outputs["Value"], fade.inputs["Value"])
    blend = nodes.new("ShaderNodeMix")
    blend.data_type = "RGBA"
    links.new(fade.outputs["Result"], blend.inputs["Factor"])
    links.new(colour_link.from_socket, blend.inputs["A"])
    links.new(cap, blend.inputs["B"])
    links.new(blend.outputs["Result"], mix.inputs["A"])


def terrain_material_for(scene_info: dict, imagery: str | None, palette: str, weather: dict, haze_end: float):
    low, high = scene_info["elevation_range_m"]
    manifest = Path(imagery) if imagery else None
    haze_end *= weather.get("visibility", 1.0)
    return base.terrain_material(low, high, 0.15 * haze_end, haze_end, base.PALETTES[palette],
                                 min(0.9, base.HAZE[palette] * weather["haze"]), weather["horizon"],
                                 base._imagery(manifest, scene_info), weather.get("snow", 0.0))


def night_dim(material: bpy.types.Material) -> bpy.types.NodeSocket:
    """A multiplier on the terrain colour, keyframed with altitude: from orbit the terrain patches get the globe's
    night dimming (else they show as a lit square around the landing), near the ground they get full moonlight."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    colour_link = next(l for l in links if l.to_socket == shader.inputs["Base Color"])
    value = nodes.new("ShaderNodeValue")
    value.name = "Night dim"
    value.outputs[0].default_value = 1.0
    dim = nodes.new("ShaderNodeMix")
    dim.data_type = "RGBA"
    dim.blend_type = "MULTIPLY"
    dim.inputs["Factor"].default_value = 1.0
    combine = nodes.new("ShaderNodeCombineColor")
    for channel in ("Red", "Green", "Blue"):
        links.new(value.outputs[0], combine.inputs[channel])
    links.new(colour_link.from_socket, dim.inputs["A"])
    links.new(combine.outputs["Color"], dim.inputs["B"])
    links.new(dim.outputs["Result"], shader.inputs["Base Color"])
    return value.outputs[0]


# The measured trees cover only a few km around the landing. From higher up that area reads as one dark
# patch on the imagery, so the proxies fade in on the way down instead.
TREES_FROM_M = 1000.0  # above ground: no trees above this
TREES_FULL_M = 450.0   # all trees below this


BUILDINGS_UP_TO_M = 20_000.0  # camera altitude above which the building blocks are hidden


# Trees whose crown comes within this of the camera path (below 300 m) are left out. At 1.5 m a measured oak
# beside the summit road swept across the right edge of the frame in a few frames as the swoop passed it (Twin
# Peaks r33/r35 f91-100), a whip-pan LTX smears; a drone flying this line would give it room too.
TREE_PATH_CLEARANCE_M = 8.0


def tree_opacity(above_ground_m: float) -> float:
    u = min(1.0, max(0.0, (TREES_FROM_M - above_ground_m) / (TREES_FROM_M - TREES_FULL_M)))
    return u * u * (3 - 2 * u)


def fade_in(material: bpy.types.Material) -> bpy.types.NodeSocket:
    """Put a transparency mix in front of the material output; returns the socket that sets its opacity."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
    surface = output.inputs["Surface"].links[0].from_socket
    mix = nodes.new("ShaderNodeMixShader")
    links.new(nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    links.new(surface, mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    opacity = nodes.new("ShaderNodeValue")
    opacity.name = "Tree fade"
    links.new(opacity.outputs[0], mix.inputs["Fac"])
    return opacity.outputs[0]


def space_world(weather: dict) -> bpy.types.World:
    """Black space for the camera; the ground is lit by the sun and a little sky light."""
    world = base.sky_world(weather)
    nodes, links = world.node_tree.nodes, world.node_tree.links
    # Fade the camera-visible sky from blue (near the ground) to black (high up) by the camera's altitude,
    # driven per frame through this node's value.
    altitude = nodes.new("ShaderNodeValue")
    altitude.name = "Altitude fade"
    altitude.outputs[0].default_value = 0.0
    seen = next(n for n in nodes if n.type == "BACKGROUND" and n.name != "Background")
    black = nodes.new("ShaderNodeMix")
    black.data_type = "RGBA"
    black.inputs["B"].default_value = (0, 0, 0, 1)
    link = next(l for l in links if l.to_node == seen and l.to_socket.name == "Color")
    links.new(link.from_socket, black.inputs["A"])
    links.new(altitude.outputs[0], black.inputs["Factor"])
    links.new(black.outputs["Result"], seen.inputs["Color"])
    return world


def landing_imagery_credit(zoom: dict) -> str | None:
    """NAIP's credit when the landing patch wears it (HLS shares the globe's GIBS credit)."""
    path = zoom.get("landing_imagery")
    credit = json.loads(Path(path).read_text()).get("credit") if path and Path(path).is_file() else None
    return credit if credit and "GIBS" not in credit else None


def scene_ground_z(x: float = 0.0, y: float = 0.0) -> float:
    """Height of the rendered terrain at (x, y), by casting a ray straight down onto the terrain meshes only
    (both have identity transforms after bending, so world and object space agree)."""
    for name in ("Landing terrain", "Local terrain", "Regional terrain"):
        obj = bpy.data.objects.get(name)
        if obj:
            hit, location, *_ = obj.ray_cast(Vector((x, y, 20_000.0)), Vector((0, 0, -1)))
            if hit:
                return location.z
    return 0.0


def add_stand_in(spot: tuple[float, float], heading_deg: float, hold: tuple[int, int], height: float = 1.75) -> None:
    """A plain person-sized figure on the landmark, facing the camera's approach. It gives the video model
    something real to keep in frame, at the right scale, for the eye-level hold. Over the hold it shifts its
    weight from side to side, and late in the hold (after the last pinned person keyframe, about 0.72 of the
    shot) it glances out to one side and back: the head turns 50 degrees, the body follows by 40%. A nose
    shows which way the head points, because a turning grey sphere reads as still in the layout video."""
    ground_z = scene_ground_z(*spot)
    material = bpy.data.materials.new("Stand-in")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.55, 0.5, 0.45, 1)
    bsdf.inputs["Roughness"].default_value = 0.8
    neck = height - 0.24
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(radius=0.19, depth=neck - 0.04, location=(0, 0, ground_z + (neck - 0.04) / 2))
    parts.append(bpy.context.active_object)
    for side in (-1, 1):  # arms, so it reads as a person rather than a post
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.62, location=(0.26 * side, 0, ground_z + height - 0.62))
        parts.append(bpy.context.active_object)
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.data.materials.append(material)
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    body = bpy.context.active_object
    body.name = "Stand-in figure"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0, 0, ground_z + height - 0.12))
    head = bpy.context.active_object
    head.name = "Stand-in head"
    head.data.materials.append(material)
    head.scale = (0.85, 1.0, 1.1)  # a face direction to turn
    bpy.ops.mesh.primitive_cone_add(radius1=0.035, depth=0.08, location=(0, -0.13, ground_z + height - 0.13),
                                    rotation=(math.radians(90), 0, 0))  # the nose, on the side facing the camera
    nose = bpy.context.active_object
    nose.data.materials.append(material)
    nose.parent = head
    nose.matrix_parent_inverse = head.matrix_world.inverted()
    for part in (body, head):
        part.location.x += spot[0]
        part.location.y += spot[1]
    facing = -math.radians(heading_deg)  # shoulders square to the approach
    right = (math.cos(facing), math.sin(facing))
    first, last = hold
    base = {part: part.location.copy() for part in (body, head)}

    def ease(t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return t * t * (3 - 2 * t)

    for frame in range(first, last + 1, 3):
        phase = (frame - first) / max(1, last - first)
        glance = ease((phase - 0.55) / 0.13) * (1 - ease((phase - 0.8) / 0.15))  # out, hold, back
        head.rotation_euler.z = facing + math.radians(50) * glance
        body.rotation_euler.z = facing + math.radians(50) * 0.4 * glance
        shift = 0.05 * math.sin(2 * math.pi * 0.7 * phase)  # weight from foot to foot
        for part in (body, head):
            part.location.x = base[part].x + shift * right[0]
            part.location.y = base[part].y + shift * right[1]
            part.keyframe_insert("location", frame=frame)
            part.keyframe_insert("rotation_euler", frame=frame)


def main() -> None:
    args = options()
    zoom = json.loads(args.zoom.read_text())
    weather = WEATHER[args.weather]
    route = zoom["route"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    engine = base.set_engine(scene, args.engine, args.samples)
    if hasattr(scene.eevee, "use_shadows"):
        scene.eevee.use_shadows = True

    # Globe, clouds and atmosphere.
    globe = zoom["globe"]
    frame = globe_frame(globe["latitude"], globe["longitude"])
    for name, radius, material, segments in (
            ("Earth", EARTH_RADIUS - 30.0, globe_material(globe, clouds=False, weather=weather), (512, 256)),
            ("Cloud shell", EARTH_RADIUS + 9_000.0, globe_material(globe, clouds=True, weather=weather), (512, 256)),
            ("Atmosphere", EARTH_RADIUS + 80_000.0,
             atmosphere_material(0.35 if weather.get("globe_light", 1.0) < 0.5 else 1.0), (256, 128))):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segments[0], ring_count=segments[1], radius=radius)
        obj = bpy.context.active_object
        obj.name = name
        obj.matrix_world = frame
        obj.data.materials.append(material)
        bpy.ops.object.shade_smooth()
    cloud_shell, atmosphere = bpy.data.objects["Cloud shell"], bpy.data.objects["Atmosphere"]

    # Regional and local terrain on the curved Earth.
    # One aerial-perspective range for all of them: the local scene's own (1.6x its 12 km radius) hazed it from
    # 5 km out while the regional terrain around it stayed clear to 37 km, so below the cloud the local square
    # showed as a pale semi-transparent sheet over the city (Twin Peaks r44, 0:01).
    regional_info, local_info = zoom["regional_scene"], zoom["local_scene"]
    haze_end = 2.5 * regional_info["radius_m"]
    regional = import_terrain(regional_info["terrain_path"], "Regional terrain")
    re_east, re_north = regional_info["origin_projected_m"]
    lo_east, lo_north = local_info["origin_projected_m"]
    regional.data.transform(regional.matrix_world)
    regional.matrix_world = Matrix.Identity(4)
    taper_edges(regional, regional_info["radius_m"])
    bend_to_globe(regional, Vector((re_east - lo_east, re_north - lo_north, 0)), sink_m=4.0)
    cut_out(regional, local_info["radius_m"] - 2 * regional_info["mesh_spacing_m"])
    regional.data.materials.clear()
    regional.data.materials.append(terrain_material_for(regional_info, zoom.get("regional_imagery"), args.palette,
                                                        weather, haze_end))
    feather_to_cap(regional.data.materials[0], globe, regional_info["radius_m"])
    local = import_terrain(local_info["terrain_path"], "Local terrain")
    bend_to_globe(local, Vector((0, 0, 0)))
    local.data.materials.clear()
    local.data.materials.append(terrain_material_for(local_info, zoom.get("local_imagery"), args.palette,
                                                     weather, haze_end))
    terrains = [regional, local]
    landing_info = zoom.get("landing_scene")
    if landing_info:  # 1 m lidar patch where the camera comes down to eye level
        landing = import_terrain(landing_info["terrain_path"], "Landing terrain")
        bend_to_globe(landing, Vector((0, 0, 0)))
        feather_into(landing, local, landing_info["radius_m"])
        cut_out(local, landing_info["radius_m"])
        landing.data.materials.clear()
        # Colour by the local scene's height range, so the palette matches across the seam.
        landing.data.materials.append(terrain_material_for(
            {**landing_info, "elevation_range_m": local_info["elevation_range_m"]}, zoom.get("landing_imagery"),
            args.palette, weather, haze_end))
        terrains.append(landing)

    subject = tuple(route.get("subject_local_m", (0.0, 0.0)))  # zoom.standing_spot on a 1 m landing patch
    if route.get("subject", True):
        add_stand_in(subject, route["heading_deg"], (route["arrive_frame"] + 1, route["frames"]))
    trees = tree_fade = None
    if zoom.get("trees"):  # measured tree proxies near the landing, kept off the camera path and the person
        low, high = local_info["elevation_range_m"]
        tree_material = terrain_material_for(local_info, None, args.palette, weather, haze_end)
        green = base.TREE_GREEN
        ramp = next(node for node in tree_material.node_tree.nodes if node.type == "VALTORGB")
        for element in ramp.color_ramp.elements:
            element.color = (*green, 1)
        clear = [p for p, h in zip(route["positions"], route["above_ground_m"]) if h < 300]
        trees = base.add_trees(terrains[1:], Path(zoom["trees"]), tree_material,  # local and the 1 m landing
                               keep_clear=[(*subject, scene_ground_z(*subject) + 2.0)], path_clear=clear,
                               path_margin_m=TREE_PATH_CLEARANCE_M)
        tree_fade = fade_in(tree_material) if trees else None

    buildings = None
    if zoom.get("buildings"):  # Overture footprints and heights, roofs wearing the same imagery as the ground
        building_material = base.building_walls(terrain_material_for(
            local_info, zoom.get("local_imagery"), args.palette, weather, haze_end))
        clear = [p for p, h in zip(route["positions"], route["above_ground_m"]) if h < 1000]
        buildings = base.add_buildings(terrains[1:], Path(zoom["buildings"]), building_material,
                                       keep_clear=clear + [(*subject, scene_ground_z(*subject) + 2.0)])

    if weather.get("fog"):
        base.add_fog(local_info, weather["fog"], 2.0 * local_info["radius_m"])
    # The cloud deck the camera dives through.
    cover, darkness = weather["deck"]
    deck = zoom["cloud_deck_m"]
    sheets = []
    # Where the camera enters the deck, and a solid corridor from there to the landmark about a third as wide.
    inside = [p for p in route["positions"] if deck[0] <= p[2] <= deck[1]]
    dive = (inside[0][0], inside[0][1], max(3000.0, 0.35 * math.hypot(inside[0][0], inside[0][1]))) if inside else None
    # A/B timing switches (scripts/time_building_render.py --variants clouds): sheet spacing, shadows, dithering.
    spacing = int(os.environ.get("GEOBLOCK_CLOUD_SPACING", CLOUD_SHEET_SPACING_M))
    for index, height in enumerate(range(int(deck[0]), int(deck[1]) + 1, spacing) if spacing > 0 else ()):
        bpy.ops.mesh.primitive_plane_add(size=2 * regional_info["radius_m"], location=(0, 0, height))
        sheet = bpy.context.active_object
        sheet.name = f"Cloud sheet {index}"
        material = cloud_sheet_material(index, cover, darkness, regional_info["radius_m"],
                                        weather.get("deck_colour"), dive)
        if os.environ.get("GEOBLOCK_CLOUD_DITHER") == "1":
            if hasattr(material, "surface_render_method"):
                material.surface_render_method = "DITHERED"
            else:
                material.blend_method = "HASHED"
        sheet.data.materials.append(material)
        # No sun shadows from the sheets: they fell on the sheets below and on the ground as hard-edged grey
        # slabs, which the painters kept as flat grey planes (Twin Peaks r2, f28-36 and f47-52).
        sheet.visible_shadow = os.environ.get("GEOBLOCK_CLOUD_SHADOWS") == "1"
        sheets.append(sheet)
    sheet_heights = list(range(int(deck[0]), int(deck[1]) + 1, spacing)) if spacing > 0 else []

    globe_light = weather.get("globe_light", 1.0)
    shaded = terrains + ([buildings] if buildings else [])
    dims = [night_dim(terrain.data.materials[0]) for terrain in shaded] if globe_light < 1.0 else []
    hazes = []
    for terrain in shaded + ([trees] if trees else []):
        haze = terrain.data.materials[0].node_tree.nodes["Haze"]
        hazes.append((haze, haze.inputs["From Min"].default_value, haze.inputs["From Max"].default_value))
    scene.world = space_world(weather)
    fade = scene.world.node_tree.nodes["Altitude fade"]
    sun = base.sun_light(weather)
    az, el = math.radians(route["heading_deg"] + 100), math.radians(weather["sun_elevation"])
    toward_sun = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
    sun.rotation_euler = (-toward_sun).to_track_quat("-Z", "Y").to_euler()

    camera_data = bpy.data.cameras.new("Zoom camera")
    camera_data.lens = route["lens_mm"]
    camera = bpy.data.objects.new("Zoom camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.rotation_mode = "QUATERNION"
    landmark_ground = scene_ground_z(*subject)
    shell_fade = cloud_shell.data.materials[0].node_tree.nodes["Fade"].outputs[0]
    sheet_fades = [sheet.data.materials[0].node_tree.nodes["Fade"].outputs[0] for sheet in sheets]
    # The camera leaves the deck's base within a frame or two on a fast descent, which cut straight from whiteout
    # to clear ground. Thin the deck out over DECK_EXIT_FRAMES before that frame instead: a punch-through.
    exit_frame = next((i for i, a in enumerate(route["altitude_m"], start=1) if a < deck[0]), None)
    # Near the ground, follow the rendered mesh rather than the raster the route was planned on: the camera's
    # height over the mesh (eye level of the figure, over any rise) instead of over the raster.
    shifts = []
    for position, height in zip(route["positions"], route["above_ground_m"]):
        ground = max(scene_ground_z(position[0], position[1]), landmark_ground) if height < 200 else None
        shifts.append(0.0 if ground is None else ground + height - position[2])
    if route.get("dive_tempo") in SMOOTH_TEMPOS:
        # Followed frame by frame the camera bobbed over every bump of the mesh (r44 read as bumpy): ride a smooth
        # upper envelope of it instead, from the frame it starts following it.
        low = next((i for i, h in enumerate(route["above_ground_m"]) if h < 200), len(shifts))
        shifts[low:] = ride_floor(shifts[low:], RIDE_FLOOR_FRAMES)
    for index, (position, look, up) in enumerate(zip(route["positions"], route["look_at"], route["up"]), start=1):
        p, forward = Vector(position), (Vector(look) - Vector(position)).normalized()
        shift = shifts[index - 1]
        if shift:
            p.z += shift
            look = Vector(look) + Vector((0, 0, shift))
        right = forward.cross(Vector(up)).normalized()
        true_up = right.cross(forward)
        camera.matrix_world = Matrix((right, true_up, -forward)).transposed().to_4x4()
        camera.location = p
        camera.keyframe_insert("location", frame=index)
        camera.keyframe_insert("rotation_quaternion", frame=index)
        altitude = route["altitude_m"][index - 1]
        # Depth precision: clip planes follow the altitude.
        camera_data.clip_start = max(0.1, route["above_ground_m"][index - 1] * 0.002)
        camera_data.clip_end = altitude + 2.5 * EARTH_RADIUS
        camera_data.keyframe_insert("clip_start", frame=index)
        camera_data.keyframe_insert("clip_end", frame=index)
        fade.outputs[0].default_value = min(1.0, max(0.0, (math.log10(max(altitude, 1)) - 4.3) / 0.9))
        fade.outputs[0].keyframe_insert("default_value", frame=index)
        # Cross-fade the real-day cloud shell (seen from space) into the procedural deck (flown through), and
        # thin the deck out below its base, so cloud cover changes smoothly instead of switching.
        shell = min(1.0, max(0.0, (altitude - deck[1]) / (2.0 * deck[1])))
        deck_in = min(1.0, max(0.0, (60_000 - altitude) / 40_000))
        # Gone by the time the camera reaches the deck's base: it faded out below the base before, so the
        # bottom sheet was still opaque when the camera passed through it, a one-frame cut (r5/r15 f46->47).
        deck_out = min(1.0, max(0.0, (altitude - deck[0]) / DECK_EXIT_FADE_M))
        if exit_frame is not None:
            deck_out = min(deck_out, max(0.0, (exit_frame - 1 - index) / DECK_EXIT_FRAMES))
        shell_fade.default_value = shell
        shell_fade.keyframe_insert("default_value", frame=index)
        cloud_shell.hide_render = shell == 0
        cloud_shell.keyframe_insert("hide_render", frame=index)
        atmosphere.hide_render = altitude < 120_000  # inside the glow shell it would tint the whole sky
        atmosphere.keyframe_insert("hide_render", frame=index)
        thinned = deck_thinning(deck_out, sum(h < altitude for h in sheet_heights))
        for sheet, sheet_fade in zip(sheets, sheet_fades):
            sheet_fade.default_value = deck_in * thinned
            sheet_fade.keyframe_insert("default_value", frame=index)
            sheet.hide_render = deck_in * deck_out == 0
            sheet.keyframe_insert("hide_render", frame=index)
        # Night: terrain matches the dimmed globe from 60 km up, full moonlight below 8 km.
        up = min(1.0, max(0.0, (math.log10(max(altitude, 1)) - math.log10(8_000)) / math.log10(60_000 / 8_000)))
        for dim in dims:
            dim.default_value = 1.0 + (globe_light - 1.0) * up
            dim.keyframe_insert("default_value", frame=index)
        if tree_fade is not None:
            tree_fade.default_value = tree_opacity(route["above_ground_m"][index - 1])
            tree_fade.keyframe_insert("default_value", frame=index)
            trees.hide_render = tree_fade.default_value == 0
            trees.keyframe_insert("hide_render", frame=index)
        if buildings:  # under a pixel, and under the cloud deck, from higher up
            buildings.hide_render = altitude > BUILDINGS_UP_TO_M
            buildings.keyframe_insert("hide_render", frame=index)
        for haze, near, far in hazes:  # aerial perspective scales with how high the camera is
            height = route["above_ground_m"][index - 1]  # oblique views see far: scale with height
            haze.inputs["From Min"].default_value = max(near, 2.0 * height)
            haze.inputs["From Max"].default_value = max(far, 15.0 * height)
            haze.inputs["From Min"].keyframe_insert("default_value", frame=index)
            haze.inputs["From Max"].keyframe_insert("default_value", frame=index)

    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = route["fps"]
    scene.frame_start, scene.frame_end = 1, route["frames"]
    scene.view_settings.view_transform = "AgX"
    scene.render.image_settings.file_format = "PNG"
    first, last = map(int, args.frame_range.split(":")) if args.frame_range else (1, route["frames"])
    rendered = list(range(first, last + 1, max(1, args.frame_step)))
    frames_dir = args.output_dir / "frames"
    base.render_frames(scene, rendered, frames_dir, "")
    import shutil
    shutil.copyfile(frames_dir / "frame_0001.png", args.output_dir / "first_frame.png")
    clean_dir = None
    stand_in = [obj for obj in scene.objects if any(slot.material and slot.material.name.startswith("Stand-in")
                                                    for slot in obj.material_slots)]
    if stand_in and args.frame_step == 1 and not args.frame_range:
        # Clean plates: the eye-level hold without the stand-in, every 8th frame (all hold keyframes fall on
        # them) and the last, named by video frame index. GeoBlockPlaceSubject pastes the person onto these;
        # painting the stand-in out of the blocking frame left a smeared box behind the head (Twin Peaks r6k/r7k).
        clean_dir = args.output_dir / "clean"
        clean_dir.mkdir(exist_ok=True)
        start = max(0, route["arrive_frame"] - 32)  # workflow.COMPOSITE_LEAD_FRAMES: the keyframe before the hold too
        hold = [i for i in range(start, route["frames"]) if i % 8 == 0 or i == route["frames"] - 1]
        # EEVEE drops the blocking's shadows near the landing ("Shadow buffer full", 3600+ of 2048 pages even
        # with the biggest pool), so the shadow plates came out identical to the clean ones (r11k-r13k). Both
        # plate passes render in Cycles, at low samples with denoising: a few frames, and real shadows.
        base.set_engine(scene, "cycles", 32)
        # Clean plates first, with the stand-in fully out of the render. Rendered after a shadow-only pass they
        # still carried its shadow, a dark figure on the path that klein painted as a stranger (r10k/r11k).
        for obj in stand_in:
            obj.hide_render = True
        for index in hold:
            scene.frame_set(index + 1)
            scene.render.filepath = str((clean_dir / f"clean_{index:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
        if all(hasattr(obj, "visible_camera") for obj in stand_in):
            # Shadow plates: the stand-in hidden from the camera but still casting its shadow. Over the clean
            # plate this gives the shadow a pasted person should cast, with this render's sun.
            for obj in stand_in:
                obj.hide_render = False
                obj.visible_camera = False
            for index in hold:
                scene.frame_set(index + 1)
                scene.render.filepath = str((clean_dir / f"shadow_{index:04d}.png").resolve())
                bpy.ops.render.render(write_still=True)
    video = args.output_dir / "control.mp4"
    ffmpeg = base.encode(frames_dir, video, route["fps"] / max(1, args.frame_step))
    manifest = {"route_sha256": route["route_sha256"], "preset": route["preset"], "engine": engine,
                "blender_version": bpy.app.version_string, "resolution": [args.width, args.height],
                "frames": len(rendered), "frame_step": args.frame_step, "fps": route["fps"],
                "video": str(video) if ffmpeg else None, "first_frame": str(args.output_dir / "first_frame.png"),
                "depth_video": None, "terrain_path": local_info["terrain_path"],
                "clean_plates": str(clean_dir) if clean_dir else None,
                "attribution": "; ".join(filter(None, (local_info.get("attribution"),
                                                        (landing_info or {}).get("attribution"),
                                                        zoom.get("buildings_attribution"),
                                                        landing_imagery_credit(zoom),
                                                        globe.get("credit"))))}
    (args.output_dir / "render.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GEOBLOCK_RENDERED " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
