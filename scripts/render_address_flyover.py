"""Render an address flyover with measured building heights over ground terrain.

Bare-earth terrain and aerial imagery alone cannot become a 3D blocking shot.
Run with Blender in background mode, passing arguments after ``--``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoblock_core.height_policy import qualify_structure_heights


FPS = 24
DURATION = 8
FRAMES = FPS * DURATION


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terrain", type=Path, required=True)
    p.add_argument("--texture", type=Path, help="North-up aerial crop matching the scene's square")
    p.add_argument("--height-m", type=float, required=True, help="Address-point DEM height in metres")
    p.add_argument("--address", required=True)
    p.add_argument("--blend", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--storyboard", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-marker", action="store_true", help="Omit the schematic address pin and halo")
    p.add_argument("--flat-guide", action="store_true", help="Use flat colour without cavity or cast shadows")
    p.add_argument("--buildings", type=Path, required=True,
                   help="Measured-height building manifest; unknown heights stay absent")
    p.add_argument("--close-shot", action="store_true", help="Closer camera requires verified landmark building height")
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    return mat


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_building(block, origin, wall_material, roof_material):
    outline = block["footprint_utm_m"][:-1]
    if len(outline) < 3:
        return
    base = block["ground_z_m"] - 0.25
    top = block["roof_z_m"]
    xy = [(x - origin[0], y - origin[1]) for x, y in outline]
    count = len(xy)
    verts = [(x, y, base) for x, y in xy] + [(x, y, top) for x, y in xy]
    faces = [tuple(range(count-1, -1, -1)), tuple(range(count, count*2))]
    faces.extend((i, (i+1) % count, (i+1) % count + count, i + count)
                 for i in range(count))
    mesh = bpy.data.meshes.new(f"OSM {block['osm_way_id']} measured block")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(wall_material)
    mesh.materials.append(roof_material)
    mesh.polygons[1].material_index = 1


def main():
    a = arguments()
    if not a.terrain.exists():
        raise FileNotFoundError(a.terrain)
    building_manifest = json.loads(a.buildings.read_text())
    buildings = building_manifest.get("buildings", [])
    required_radius = 200 if a.close_shot else 700
    qualify_structure_heights(building_manifest, building_manifest["target_osm_way_id"],
                              required_radius)
    metadata = json.loads(a.terrain.with_name("metadata.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(a.terrain.resolve()))
    terrain = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
    terrain.name = "USGS 3DEP bare-earth terrain"
    terrain.data.materials.clear()
    if a.texture:
        if not a.texture.is_file():
            raise FileNotFoundError(a.texture)
        radius = float(metadata["bounds_local_m"][2])
        uv_layer = terrain.data.uv_layers.new(name="Geographic image UV")
        for poly in terrain.data.polygons:
            for loop_index in poly.loop_indices:
                vertex = terrain.data.vertices[terrain.data.loops[loop_index].vertex_index]
                uv_layer.data[loop_index].uv = (
                    (vertex.co.x + radius) / (2 * radius),
                    (vertex.co.y + radius) / (2 * radius),
                )
        image = bpy.data.images.load(str(a.texture.resolve()))
        texmat = material("Terrain | USGS NAIP aerial drape", (1, 1, 1))
        texmat.use_nodes = True
        nodes = texmat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        image_node.interpolation = "Linear"
        nodes.active = image_node
        texmat.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
        terrain.data.materials.append(texmat)
    else:
        terrain.data.materials.append(material("Terrain | warm clay", (0.52, 0.51, 0.46)))

    origin = building_manifest["origin_utm_m"]
    wall = material("Buildings | measured pale walls", (0.89, 0.84, 0.71))
    roof = material("Buildings | measured warm roof", (0.78, 0.50, 0.33))
    for block in buildings:
        if block["status"] == "accepted":
            add_building(block, origin, wall, roof)

    # The geocoded address point is the origin in the compiled local XY frame.
    if not a.no_marker:
        pin_z = a.height_m + 2
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=10, depth=3, location=(0, 0, pin_z))
        ring = bpy.context.object
        ring.name = "Address point | OSM geocode (not a building footprint)"
        ring.data.materials.append(material("Address marker | gold", (0.95, 0.47, 0.08)))
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=9, radius2=0, depth=30,
                                       location=(0, 0, a.height_m + 20))
        pin = bpy.context.object
        pin.name = "Address locator"
        pin.data.materials.append(ring.data.materials[0])
        bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=12,
                                        location=(0, 0, a.height_m + 2),
                                        major_radius=28, minor_radius=1.6)
        footprint = bpy.context.object
        footprint.name = "Address point halo | schematic"
        footprint.data.materials.append(ring.data.materials[0])

    camera_data = bpy.data.cameras.new("Camera | Woodside address flyover")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.collection.objects.link(camera)
    scene = bpy.context.scene
    scene.camera = camera
    camera_data.lens = 42 if a.close_shot else 38
    camera_data.clip_end = 10000
    for frame in range(1, FRAMES + 1):
        t = (frame - 1) / (FRAMES - 1)
        # Sweep around the address without a hairpin; the geocoded point stays
        # in view while the terrain context changes continuously.
        u = t * t * (3 - 2 * t)
        angle = math.radians(-130 + 180 * u)
        radius = (200 - 95 * math.sin(math.pi * u)) if a.close_shot else (690 - 320 * math.sin(math.pi * u))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = (a.height_m + 90 - 35 * math.sin(math.pi * u)) if a.close_shot else (a.height_m + 280 - 100 * math.sin(math.pi * u))
        camera.location = (x, y, z)
        point_at(camera, (0, 0, a.height_m + 8))
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)

    scene["address"] = a.address
    scene["address_point_source"] = metadata["source"].get("provider", "OSM geocode")
    scene["terrain_source"] = metadata["source"]["dataset"]
    scene["terrain_vertical_crs"] = metadata["common_vertical_crs"]
    scene["building_model_status"] = (
        f"{building_manifest['accepted_buildings']} approximate blocks with measured heights")
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT" if a.flat_guide else "STUDIO"
    scene.display.shading.color_type = "TEXTURE" if a.texture else "MATERIAL"
    scene.display.shading.show_shadows = not a.flat_guide
    scene.display.shading.show_cavity = not a.flat_guide
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.background_type = "WORLD"
    scene.world = bpy.data.worlds.new("Clay sky")
    scene.world.color = (0.62, 0.70, 0.76)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAMES
    scene.frame_set(1)
    a.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.blend.resolve()))
    if a.storyboard or a.render:
        a.frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        frames = (1, 48, 96, 144, 192) if a.storyboard else range(1, FRAMES + 1)
        for frame in frames:
            scene.frame_set(frame)
            scene.render.filepath = str((a.frames_dir / f"frame_{frame:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
            if frame == 1 or frame % 24 == 0 or frame == FRAMES:
                print(f"Rendered {frame}/{FRAMES}", flush=True)
    print(f"Saved {a.blend}", flush=True)


if __name__ == "__main__":
    main()
