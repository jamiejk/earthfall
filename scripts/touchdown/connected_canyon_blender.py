"""Reference-textured continuous canyon, with shared lighting and person contact."""
import json
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from connected_canyon import CanyonSurface
from authored_scene_blender import image_material,mesh_object


def canyon_material(folder):
    mat=bpy.data.materials.new('Continuous sandstone photographic material');mat.use_nodes=True
    n=mat.node_tree.nodes;l=mat.node_tree.links;n.clear()
    out=n.new('ShaderNodeOutputMaterial');shader=n.new('ShaderNodeBsdfPrincipled');shader.inputs['Roughness'].default_value=.95
    geo=n.new('ShaderNodeNewGeometry')
    scale=n.new('ShaderNodeVectorMath');scale.operation='SCALE';scale.inputs['Scale'].default_value=1/18
    warp=n.new('ShaderNodeTexNoise');warp.inputs['Scale'].default_value=.055;warp.inputs['Detail'].default_value=3
    l.new(geo.outputs['Position'],warp.inputs[0])
    displacement=n.new('ShaderNodeVectorMath');displacement.operation='SCALE';displacement.inputs['Scale'].default_value=12
    l.new(warp.outputs['Color'],displacement.inputs[0])
    warped=n.new('ShaderNodeVectorMath');warped.operation='ADD'
    l.new(geo.outputs['Position'],warped.inputs[0]);l.new(displacement.outputs[0],warped.inputs[1]);l.new(warped.outputs[0],scale.inputs[0])
    textures=[]
    for path in [folder/'ground.png',folder/'cliff_texture.png']:
        tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True)
        tex.projection='BOX';tex.projection_blend=.35;tex.extension='REPEAT'
        l.new(scale.outputs[0],tex.inputs[0]);textures.append(tex)
    sep=n.new('ShaderNodeSeparateXYZ');l.new(geo.outputs['Normal'],sep.inputs[0])
    slope=n.new('ShaderNodeMapRange');slope.clamp=True;slope.inputs['From Min'].default_value=.45;slope.inputs['From Max'].default_value=.9
    l.new(sep.outputs['Z'],slope.inputs['Value'])
    mix=n.new('ShaderNodeMixRGB');l.new(slope.outputs[0],mix.inputs[0]);l.new(textures[1].outputs['Color'],mix.inputs[1]);l.new(textures[0].outputs['Color'],mix.inputs[2])
    broad=n.new('ShaderNodeTexNoise');broad.inputs['Scale'].default_value=.009;broad.inputs['Detail'].default_value=3
    l.new(geo.outputs['Position'],broad.inputs[0])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(.48,.33,.24,1);ramp.color_ramp.elements[1].color=(1,.94,.82,1)
    l.new(broad.outputs['Fac'],ramp.inputs[0])
    variation=n.new('ShaderNodeMixRGB');variation.blend_type='MULTIPLY';variation.inputs[0].default_value=.45
    l.new(mix.outputs[0],variation.inputs[1]);l.new(ramp.outputs[0],variation.inputs[2])
    noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=3;noise.inputs['Detail'].default_value=4
    l.new(geo.outputs['Position'],noise.inputs[0])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.3;bump.inputs['Distance'].default_value=.055
    l.new(noise.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],shader.inputs['Normal'])
    # Haze belongs to distance, not to a photograph pasted below the mesh.
    camera=n.new('ShaderNodeCameraData');haze=n.new('ShaderNodeMapRange');haze.clamp=True
    haze.inputs['From Min'].default_value=600;haze.inputs['From Max'].default_value=8500
    l.new(camera.outputs['View Distance'],haze.inputs['Value'])
    tint=n.new('ShaderNodeMixRGB');tint.inputs[2].default_value=(.48,.59,.69,1)
    l.new(haze.outputs[0],tint.inputs[0]);l.new(variation.outputs[0],tint.inputs[1]);l.new(tint.outputs[0],shader.inputs['Base Color'])
    l.new(shader.outputs[0],out.inputs[0]);return mat


def main():
    folder=Path(sys.argv[sys.argv.index('--')+1]);cfg=json.loads((folder/'render.json').read_text())
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=cfg['width'];scene.render.resolution_y=cfg['height'];scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.fps=24
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.60,.69,.80,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.58
    surface=CanyonSurface(Path(cfg['terrain_cache']));mat=canyon_material(folder)
    # Nested samples use the same height function, material and world coordinates.
    for radius,step,cut in [(6000,25,400),(400,2,60),(60,.5,0)]:
        axis=np.arange(-radius,radius+step*.5,step);xx,yy=np.meshgrid(axis,axis);zz=surface.height(xx,yy)
        count=len(axis);vertices=np.stack([xx,yy,zz],axis=-1).reshape(-1,3).tolist();faces=[]
        for j in range(count-1):
            for i in range(count-1):
                if cut and abs((axis[i]+axis[i+1])/2)<cut and abs((axis[j]+axis[j+1])/2)<cut:continue
                k=j*count+i;faces.append((k,k+1,k+count+1,k+count))
        obj=mesh_object(f'Connected canyon surface {radius}m',vertices,faces,mat)
        for poly in obj.data.polygons:poly.use_smooth=True
    # Sky-only photography; every visible rock belongs to the connected mesh.
    sky=image_material('Storm sky reference',folder/'sky_texture.png',emission=True)
    mesh_object('Distant storm sky',[(-15000,16000,-1000),(15000,16000,-1000),(15000,16000,8500),(-15000,16000,8500)],[(0,1,2,3)],sky,[(0,0),(1,0),(1,1),(0,1)])
    person=image_material('Reference person under scene lighting',folder/'person_rgba.png',emission=False,alpha=True)
    w,h=cfg['matte']['size'];ph=cfg['matte']['subject_height_m'];hw=ph*w/h/2
    mesh_object('Person temporary guide',[(-hw,0,0),(hw,0,0),(hw,0,ph),(-hw,0,ph)],[(0,1,2,3)],person,[(0,0),(1,0),(1,1),(0,1)])
    bpy.ops.object.light_add(type='SUN');sun=bpy.context.object
    sun.rotation_euler=(math.radians(28),math.radians(-32),math.radians(-30));sun.data.energy=1.25;sun.data.angle=.2
    bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera
    camera.data.sensor_width=36;camera.data.clip_start=.05;camera.data.clip_end=24000
    out=folder/('preview' if cfg['preview'] else 'frames');out.mkdir(exist_ok=True)
    indices=sorted(set(round(t*(cfg['frames']-1)) for t in [0,.1,.2,.3,.4,.5,.65,.8,1])) if cfg['preview'] else range(cfg['frames'])
    for i in indices:
        scene.frame_set(i+1);pose=cfg['poses'][i];camera.location=pose['position']
        camera.rotation_euler=(Vector(pose['target'])-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.lens=35
        scene.render.filepath=str(out/f'{i:04d}.png');bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'connected_canyon.blend'))

if __name__=='__main__':main()
