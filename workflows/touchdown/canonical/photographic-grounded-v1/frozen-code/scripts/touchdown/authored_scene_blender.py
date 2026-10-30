"""Persistent photographic set for the bounded authored approach (run in Blender).

An actual textured ground mesh and rocks supply near parallax; a distant matte
supplies the canyon. The isolated person is a stationary card, deliberately seen
only from the front. This is an authored set, not recovered terrain geometry.
"""
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def image_material(name, path, emission=False, alpha=False):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(path), check_existing=True)
    texture.interpolation = "Linear"
    if emission:
        shader = nodes.new("ShaderNodeEmission")
        links.new(texture.outputs["Color"], shader.inputs["Color"])
    else:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = .88
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 130
        noise.inputs["Detail"].default_value = 3
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = .16
        bump.inputs["Distance"].default_value = .02
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    if alpha:
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        links.new(texture.outputs["Alpha"], mix.inputs[0])
        links.new(transparent.outputs[0], mix.inputs[1])
        links.new(shader.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs["Surface"])
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
    else:
        links.new(shader.outputs[0], output.inputs["Surface"])
    return material


def mesh_object(name, vertices, faces, material, uv=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    if uv is not None:
        layer = mesh.uv_layers.new(name="Photographic coordinates")
        for loop in mesh.loops:
            layer.data[loop.index].uv = uv[loop.vertex_index]
    return obj


def main():
    folder = Path(sys.argv[sys.argv.index("--") + 1])
    cfg = json.loads((folder / "render.json").read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = cfg["width"], cfg["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = cfg["fps"]
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.62, .72, .83, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .65

    ground = image_material("Reference-conditioned sandstone", folder / "ground.png")
    # A broad shelf, with centimetres of surface relief and a naturally broken rim.
    # Its colour/detail are generated once and remain attached to the same vertices.
    vertices, uvs, faces = [], [], []
    nx, ny = 100, 140
    for j in range(ny + 1):
        v = j / ny
        for i in range(nx + 1):
            x = -65 + 130*i/nx
            rim = 2 + .45*math.sin(x*.19) + .2*math.sin(x*.83)
            y = -115 + (rim + 115)*v
            z = .035*math.sin(x*2.1)*math.sin(y*1.7) + .015*math.cos(y*4+x)
            # Keep the planted feet on an exactly level patch, without a colour patch.
            z *= min(1, (x*x+y*y)/2)
            vertices.append((x,y,z)); uvs.append((x/7,y/7))
            if i < nx and j < ny:
                k = j*(nx+1)+i
                faces.append((k,k+1,k+nx+2,k+nx+1))
    mesh_object("Continuous landing shelf", vertices, faces, ground, uvs)

    backdrop = image_material("Approved canyon matte", folder / "master.png", emission=True)
    # The master photograph is placed far enough away for a restrained perspective
    # change; it is never regenerated or crossfaded during the approach.
    distance, width = 600.0, 850.0
    height = width * 1536/2688
    centre_z = -110.0
    mesh_object("Distant canyon and sky", [(-width/2,distance,centre_z-height/2),
                (width/2,distance,centre_z-height/2),(width/2,distance,centre_z+height/2),
                (-width/2,distance,centre_z+height/2)], [(0,1,2,3)], backdrop,
                [(0,0),(1,0),(1,1),(0,1)])

    rng = random.Random(20261021)
    for i in range(28):
        x = rng.choice([-1,1])*rng.uniform(4,26)
        y = rng.uniform(-45,1)
        radius = rng.uniform(.1,.55)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1, location=(x,y,radius*.25))
        rock = bpy.context.object
        rock.name = f"Persistent foreground stone {i:02d}"
        rock.scale = (radius*1.6,radius,radius*.6)
        rock.rotation_euler = (rng.uniform(-.2,.2),rng.uniform(-.2,.2),rng.uniform(0,6.28))
        rock.data.materials.append(ground)
        # Give rocks object-local UVs instead of relying on nonexistent sphere UVs.
        layer = rock.data.uv_layers.new()
        for loop in rock.data.loops:
            co = rock.data.vertices[loop.vertex_index].co
            layer.data[loop.index].uv = (co.x*.3+.5, co.y*.3+.5)

    person = image_material("Photographic subject", folder / "person_rgba.png", emission=True, alpha=True)
    w, h = cfg["matte"]["size"]
    person_height = cfg["matte"]["subject_height_m"]
    half_width = person_height*w/h/2
    mesh_object("Stationary reference subject", [(-half_width,0,0),(half_width,0,0),
                (half_width,0,person_height),(-half_width,0,person_height)],
                [(0,1,2,3)], person, [(0,0),(1,0),(1,1),(0,1)])

    bpy.ops.object.light_add(type="SUN", location=(-20,-10,35))
    sunlight = bpy.context.object
    sunlight.name = "Shared camera-left sunlight"
    sunlight.rotation_euler = (math.radians(28),math.radians(-32),math.radians(-30))
    sunlight.data.energy = 1.3
    sunlight.data.angle = .14

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.sensor_width = 36
    camera.data.clip_start = .05
    camera.data.clip_end = 3000
    camera.data.dof.use_dof = False
    frames = folder / ("preview" if cfg["preview"] else "frames")
    frames.mkdir(exist_ok=True)
    indices = sorted(set([0,cfg["frames"]//4,cfg["frames"]//2,3*cfg["frames"]//4,cfg["frames"]-1])) if cfg["preview"] else range(cfg["frames"])
    for i in indices:
        pose = cfg["poses"][i]
        camera.location = pose["position"]
        direction = Vector(pose["target"])-camera.location
        camera.rotation_euler = direction.to_track_quat("-Z","Y").to_euler()
        camera.data.lens = pose["focal_length_mm"]
        scene.render.filepath = str(frames / f"{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "authored_set.blend"))


if __name__ == "__main__":
    main()
