"""One neutral, target-locked local dolly; no photographs baked into the guide."""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.clay_layout_blender import ellipsoid, rod
from touchdown.clay_layout import adult_landmarks, HEIGHT, HEAD_HEIGHT


def main():
    root = Path(sys.argv[sys.argv.index('--')+1])
    cfg = json.loads((root/'camera-plan.json').read_text())
    preview = '--preview' in sys.argv
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = 960, 544
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.fps = 24
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.45,.45,.45,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .8
    bpy.ops.object.light_add(type='SUN', location=(-10,-10,20))
    bpy.context.object.data.energy = 1.3
    bpy.context.object.data.angle = .5
    bpy.context.object.rotation_euler = (.4,-.5,-.4)

    def material(name, value):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        p = m.node_tree.nodes['Principled BSDF']
        p.inputs['Base Color'].default_value = (value,value,value,1)
        p.inputs['Roughness'].default_value = 1
        return m

    ground, water, path, adult = [material(n,v) for n,v in
        [('grey ground',.48),('grey water',.25),('grey route',.66),('grey adult',.56)]]

    def plane(name, x0, x1, z, mat):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(x0,-150,z),(x1,-150,z),(x1,150,z),(x0,150,z)], [], [(0,1,2,3)])
        obj = bpy.data.objects.new(name,mesh)
        scene.collection.objects.link(obj)
        obj.data.materials.append(mat)

    plane('untextured connected ground',cfg['shore_x'],100,0,ground)
    plane('untextured lake',-150,cfg['shore_x'],0,water)
    plane('flat route',cfg['path_x']-.65,cfg['path_x']+.65,.015,path)
    points = {k:(v[0],v[1],v[2]-adult_landmarks(156)['left_foot'][2]+.04)
              for k,v in adult_landmarks(156).items()}
    for name,radii in [('head',(.105,.10,HEAD_HEIGHT/2)),('neck',(.055,.055,.065)),
                       ('torso',(.19,.105,.265)),('pelvis',(.16,.105,.12))]:
        ellipsoid('adult '+name,points[name],radii,adult)
    for side in ['left','right']:
        for a,b,r in [('shoulder','elbow',.043),('elbow','wrist',.035),
                       ('hip','knee',.064),('knee','ankle',.046)]:
            rod('adult '+side+' '+a,points[f'{side}_{a}'],points[f'{side}_{b}'],r,adult)
        for name,radii in [('hand',(.035,.025,.067)),('foot',(.055,.13,.04))]:
            ellipsoid('adult '+side+' '+name,points[f'{side}_{name}'],radii,adult)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    scene.camera = cam
    cam.data.lens = 35
    cam.data.sensor_width = 36
    cam.data.clip_end = 1000
    folder = root/('preview' if preview else 'frames')
    folder.mkdir(exist_ok=True)
    witnesses = []
    for i in (list(range(0,49,6)) if preview else range(49)):
        pose = cfg['poses'][i]
        cam.location = pose['position']
        cam.rotation_euler = (Vector(pose['target'])-cam.location).to_track_quat('-Z','Y').to_euler()
        bpy.context.view_layer.update()
        witness = {}
        for name,p in [('head',(0,0,1.78)),('foot',(0,0,0)),
                       ('near_route',(cfg['path_x'], -3,0)),('far_route',(cfg['path_x'],8,0))]:
            q = world_to_camera_view(scene,cam,Vector(p))
            witness[name] = [q.x,1-q.y,q.z]
        witnesses.append(dict(frame=i,points=witness))
        scene.render.filepath = str(folder/f'{i:04d}.png')
        bpy.ops.render.render(write_still=True)
    (root/('preview-audit.json' if preview else 'geometry-audit.json')).write_text(json.dumps(dict(
        witnesses=witnesses,head_units=HEIGHT/HEAD_HEIGHT,adult_height_m=HEIGHT,
        camera_lens_mm=35,grey_materials=4,textures=0,trees=0,buildings=0,image_planes=0,
        pose='neutral, feet fixed, no articulation',camera_plan=cfg),indent=2)+'\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(root/'local-descent.blend'))


if __name__ == '__main__':
    main()
