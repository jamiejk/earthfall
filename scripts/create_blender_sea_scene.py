"""Build an exploratory Blender scene from a GeoBlock terrain GLB.

Run with Blender's Python, passing options after ``--``. The water elevation is a
visual staging value in the source raster's unverified height frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_landform_heights


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain",
        type=Path,
        default=ROOT / "outputs/B_lidar_1m_zoom/terrain.glb",
    )
    parser.add_argument(
        "--blend",
        type=Path,
        default=ROOT / "outputs/B_lidar_1m_zoom/sea_scene.blend",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=ROOT / "outputs/B_lidar_1m_zoom/sea_preview.png",
    )
    parser.add_argument("--water-height", type=float, default=0.0)
    parser.add_argument("--view", choices=("detail", "full", "lookout"), default="detail")
    cli = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(cli)


def make_terrain_material(max_height: float) -> bpy.types.Material:
    material = bpy.data.materials.new("Terrain | coastal rock and grass")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.88
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    height = nodes.new("ShaderNodeMapRange")
    height.inputs["From Min"].default_value = -1.0
    height.inputs["From Max"].default_value = max_height
    height.clamp = True
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
    for position, rgba in (
        (0.0, (0.17, 0.16, 0.14, 1)),
        (0.07, (0.36, 0.34, 0.29, 1)),
        (0.24, (0.48, 0.44, 0.32, 1)),
        (0.52, (0.33, 0.38, 0.26, 1)),
        (0.82, (0.40, 0.41, 0.33, 1)),
        (1.0, (0.50, 0.49, 0.41, 1)),
    ):
        element = ramp.color_ramp.elements[0] if position == 0 else ramp.color_ramp.elements.new(position)
        element.position = position
        element.color = rgba
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.15
    noise.inputs["Detail"].default_value = 3.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.28

    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], height.inputs["Value"])
    links.new(height.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(geometry.outputs["Position"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_sea_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Sea | blue-green procedural ripples")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.028, 0.20, 0.26, 1)
    shader.inputs["Metallic"].default_value = 0.15
    shader.inputs["Roughness"].default_value = 0.24
    geometry = nodes.new("ShaderNodeNewGeometry")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.075
    noise.inputs["Detail"].default_value = 2.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.35
    links.new(geometry.outputs["Position"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def point_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    options = args()
    if not options.terrain.is_file():
        raise FileNotFoundError(options.terrain)
    evidence_path = options.terrain.with_name("landform_height_evidence.json")
    if not evidence_path.is_file():
        raise ValueError("A measured landmark-height manifest is required before building this scene")
    qualify_landform_heights(json.loads(evidence_path.read_text()), options.terrain)
    options.blend.parent.mkdir(parents=True, exist_ok=True)
    options.preview.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(options.terrain.resolve()))
    terrain = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
    terrain.name = "LiDAR terrain | " + ("2 m full headland" if options.view != "detail" else "1 m western ridge")
    terrain.data.materials.clear()
    terrain.data.materials.append(make_terrain_material(130.0 if options.view != "detail" else 48.0))
    for polygon in terrain.data.polygons:
        polygon.use_smooth = True

    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, options.water_height))
    sea = bpy.context.object
    sea.name = f"SEA | provisional {options.water_height:+.2f} m"
    sea.scale = (7500, 7500, 1) if options.view != "detail" else (1200, 1200, 1)
    sea.data.materials.append(make_sea_material())
    sea["height_in_source_metres"] = options.water_height
    sea["vertical_reference"] = "Unverified Welsh LiDAR height frame; visual staging only"

    camera_data = bpy.data.cameras.new("Camera | " + ("Rhossili lookout" if options.view == "lookout" else "coast overview"))
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.collection.objects.link(camera)
    if options.view == "lookout":
        metadata = json.loads(options.terrain.with_name("metadata.json").read_text())
        origin_x = metadata["origin_easting_m"]
        origin_y = metadata["origin_northing_m"]
        camera.location = (240320 - origin_x, 187520 - origin_y, 72.0)
        point_at(camera, (239260 - origin_x, 187410 - origin_y, 22))
        camera_data.type = "PERSP"
        camera_data.lens = 38
    else:
        camera.location = (0, -2600, 4700) if options.view == "full" else (345, 410, 285)
        point_at(camera, (0, 0, 15) if options.view == "full" else (0, 0, 12))
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = 4700 if options.view == "full" else 690
    camera_data.clip_end = 50000
    bpy.context.scene.camera = camera

    sun_data = bpy.data.lights.new("Sun | late afternoon", "SUN")
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    bpy.context.collection.objects.link(sun)
    sun.location = (-250, -300, 500)
    point_at(sun, (0, 0, 0))
    sun_data.energy = 2.0
    sun_data.angle = 0.055

    world = bpy.data.worlds.new("Clear coastal sky")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.52, 0.68, 0.83, 1)
    background.inputs["Strength"].default_value = 0.65
    bpy.context.scene.world = world

    scene = bpy.context.scene
    scene["source_terrain"] = str(options.terrain.resolve())
    scene["height_alignment"] = "exploratory_unaligned"
    scene["water_height_status"] = "Provisional visual level; not a measured tide or certified ODN height"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64 if options.view == "lookout" else 32
    scene.render.resolution_x = 1920 if options.view == "lookout" else 1400
    scene.render.resolution_y = 1080 if options.view == "lookout" else 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(options.preview.resolve())
    scene.view_settings.view_transform = "AgX"
    if options.view == "lookout":
        scene["lookout_projected_east_north_m"] = "240320, 187520"
        scene["lookout_camera_height_m"] = 72.0
        scene["lookout_note"] = "Elevated visual viewpoint above the Rhossili headland edge"

    bpy.ops.wm.save_as_mainfile(filepath=str(options.blend.resolve()))
    bpy.ops.render.render(write_still=True)
    print(f"Saved {options.blend} and {options.preview}")


if __name__ == "__main__":
    main()
