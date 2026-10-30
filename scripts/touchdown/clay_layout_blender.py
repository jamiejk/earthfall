"""Render an untextured neutral-grey camera/layout signal, with no tree models."""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touchdown.clay_layout import adult_landmarks, ground_height, shore, HEIGHT, HEAD_HEIGHT
from touchdown.authored_scene_blender import mesh_object


def ellipsoid(name, point, radii, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1, location=point)
    obj=bpy.context.object; obj.name=name; obj.scale=radii; obj.data.materials.append(material)
    for poly in obj.data.polygons: poly.use_smooth=True
    return obj


def rod(name, a, b, radius, material):
    delta=Vector(b)-Vector(a)
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=delta.length, location=(Vector(a)+Vector(b))/2)
    obj=bpy.context.object; obj.name=name; obj.rotation_euler=delta.to_track_quat('Z','Y').to_euler()
    obj.data.materials.append(material)
    return obj


def main():
    root=Path(sys.argv[sys.argv.index('--')+1]); cfg=json.loads((root/'render.json').read_text())
    preview='--preview' in sys.argv
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=cfg['width']; scene.render.resolution_y=cfg['height'];scene.render.resolution_percentage=100
    scene.render.fps=24; scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'
    if hasattr(scene,'eevee'): scene.eevee.taa_render_samples=16
    scene.world.use_nodes=True
    bg=scene.world.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(.4,.4,.4,1);bg.inputs['Strength'].default_value=.75
    bpy.ops.object.light_add(type='SUN',location=(-80,-80,300))
    sun=bpy.context.object;sun.data.energy=2.;sun.data.color=(1,1,1);sun.data.angle=.16;sun.rotation_euler=(.45,-.5,-.7)
    clay=bpy.data.materials.new('Uniform neutral grey clay');clay.use_nodes=True
    shader=clay.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.5,.5,.5,1)
    shader.inputs['Roughness'].default_value=1.
    mesh_object('Single flat lake',[(-50000,-40000,0),(50000,-40000,0),(50000,50000,0),(-50000,50000,0)],[(0,1,2,3)],clay)
    for name,ys,width,nx in [('Regional ground',[-18000+i*300 for i in range(151)],28000,95),
                            ('Connected landing ground',[-500+i*6 for i in range(251)],850,80)]:
        vs=[];fs=[]
        for j,y in enumerate(ys):
            for i in range(nx+1):
                x=shore(y)+width*(i/nx)**1.8
                vs.append((x,y,ground_height(x+.001,y)-(.1 if name=='Regional ground' else 0)))
                if i<nx and j<len(ys)-1:
                    k=j*(nx+1)+i;fs.append((k,k+1,k+nx+2,k+nx+1))
        obj=mesh_object(name,vs,fs,clay)
        for p in obj.data.polygons:p.use_smooth=True
    # Close the terrain at the waterline; this grey bank supplies a readable
    # shore edge through lighting, without a water/grass colour distinction.
    vs=[];fs=[]
    for i in range(501):
        y=-18000+i*90;x=shore(y)
        vs.extend([(x,y,0),(x,y,ground_height(x+.001,y))])
        if i<500:fs.append((i*2,i*2+2,i*2+3,i*2+1))
    mesh_object('Continuous shore bank',vs,fs,clay)
    vs=[];fs=[]
    for i in range(280):
        y=-500+i*2;cx=2.3*math.sin(y*.016)-2.2*math.exp(-(y/15)**2)
        for x in [cx-1.5,cx+1.5]:vs.append((x,y,ground_height(x,y)+.10))
        if i<279:fs.append((i*2,i*2+1,i*2+3,i*2+2))
    mesh_object('Route ribbon',vs,fs,clay)
    # A single mass marks broad occupancy; it does not prescribe toy turrets/roofs.
    bpy.ops.mesh.primitive_cube_add(size=1,location=(-43,182,13)); obj=bpy.context.object
    obj.name='Broad landmark occupancy only';obj.dimensions=(36,85,26);obj.data.materials.append(clay)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    landmarks=adult_landmarks(156); actors=[]
    for name,radii in [('head',(.105,.10,HEAD_HEIGHT/2)),('neck',(.055,.055,.065)),
                        ('torso',(.19,.105,.265)),('pelvis',(.16,.105,.12))]:
        actors.append((ellipsoid('Adult '+name,landmarks[name],radii,clay),name,None))
    for side in ['left','right']:
        for a,b,radius in [('shoulder','elbow',.043),('elbow','wrist',.035),('hip','knee',.064),('knee','ankle',.046)]:
            ka,kb=f'{side}_{a}',f'{side}_{b}'
            actors.append((rod('Adult '+ka,landmarks[ka],landmarks[kb],radius,clay),ka,kb))
        for name,radii in [('hand',(.035,.025,.067)),('foot',(.055,.13,.04))]:
            k=f'{side}_{name}';actors.append((ellipsoid('Adult '+k,landmarks[k],radii,clay),k,None))
    bpy.ops.object.camera_add();cam=bpy.context.object;scene.camera=cam;cam.name='Unchanged 35mm physical camera'
    cam.data.lens=35;cam.data.sensor_width=36;cam.data.clip_start=.05;cam.data.clip_end=150000
    # Only articulation changes; world geometry stays fixed for the whole shot.
    folder=root/('preview' if preview else 'frames');folder.mkdir(exist_ok=True)
    selected=[0,12,24,36,48,60,72,84,96] if preview else list(range(cfg['frames']))
    projections=[]
    for i in selected:
        pose=cfg['poses'][i]; points=adult_landmarks(pose['delivery_frame'])
        cam.location=pose['position'];cam.rotation_euler=(Vector(pose['target'])-cam.location).to_track_quat('-Z','Y').to_euler()
        for obj,a,b in actors:
            if b:
                delta=Vector(points[b])-Vector(points[a]);obj.location=(Vector(points[a])+Vector(points[b]))/2
                obj.rotation_euler=delta.to_track_quat('Z','Y').to_euler()
            else:obj.location=points[a]
        bpy.context.view_layer.update()
        witness=dict(head=points['head'],foot=points['left_foot'],knee=points['left_knee'],
                     path_point=(-2.5,-12,ground_height(-2.5,-12)+.1),castle=(-43,182,13.5))
        projections.append(dict(delivery_frame=pose['delivery_frame'],points={n:list(world_to_camera_view(scene,cam,Vector(p))) for n,p in witness.items()}))
        scene.render.filepath=str(folder/f'{i:04d}.png');bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(root/'minimal-clay.blend'))
    materials={slot.material for obj in scene.objects for slot in obj.material_slots if slot.material}
    nodes=[node for mat in materials if mat.use_nodes for node in mat.node_tree.nodes]
    audit=dict(version='minimal-neutral-clay-v1',camera_poses=cfg['poses'],witness_projection=projections,
        mesh_objects=[o.name for o in scene.objects if o.type=='MESH'],tree_meshes=0,image_planes=0,view_crossfades=0,
        material_count=len(materials),
        material_base_colours=[list(m.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value) for m in materials if m.use_nodes],
        texture_nodes=[n.bl_idname for n in nodes if 'Tex' in n.bl_idname],
        human_height_m=HEIGHT,head_height_m=HEAD_HEIGHT,head_units=HEIGHT/HEAD_HEIGHT,
        scope='Camera/layout clay only. No foliage, surface colour, photographic textures, windows, roofs, props or facial features.')
    (root/('preview-audit.json' if preview else 'geometry-audit.json')).write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
