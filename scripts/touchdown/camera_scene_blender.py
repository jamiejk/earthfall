"""One persistent, photo-informed Chillon shoreline scene for the camera pilot."""
import json
import math
import random
import sys
from pathlib import Path
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0,str(Path(__file__).resolve().parent))
from shoreline_camera import shore,ground_height
from authored_scene_blender import mesh_object


def material(name,colour,scale=1.,rough=.85,noise=.15):
    m=bpy.data.materials.new(name);m.use_nodes=True
    ns=m.node_tree.nodes;ls=m.node_tree.links;s=ns.get('Principled BSDF')
    s.inputs['Base Color'].default_value=(*colour,1);s.inputs['Roughness'].default_value=rough
    tex=ns.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=scale;tex.inputs['Detail'].default_value=4
    coord=ns.new('ShaderNodeTexCoord');ls.new(coord.outputs['Object'],tex.inputs['Vector'])
    ramp=ns.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color=(*(c*(1-noise) for c in colour),1)
    ramp.color_ramp.elements[1].color=(*(min(1,c*(1+noise)) for c in colour),1)
    ls.new(tex.outputs['Fac'],ramp.inputs[0]);ls.new(ramp.outputs[0],s.inputs['Base Color'])
    bump=ns.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.2;bump.inputs['Distance'].default_value=.035
    ls.new(tex.outputs['Fac'],bump.inputs['Height']);ls.new(bump.outputs[0],s.inputs['Normal'])
    return m


def box(name,loc,size,mat):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
    o=bpy.context.object;o.name=name;o.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    o.data.materials.append(mat)
    bevel=o.modifiers.new('Weathered edge','BEVEL');bevel.width=min(size)*.025;bevel.segments=2
    return o


def cylinder(name,x,y,z,r,h,mat,vertices=24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=h,location=(x,y,z+h/2))
    o=bpy.context.object;o.name=name;o.data.materials.append(mat);return o


def roof(name,x,y,z,rx,ry,height,mat):
    v=[(x-rx,y-ry,z),(x+rx,y-ry,z),(x+rx,y+ry,z),(x-rx,y+ry,z),(x,y-ry,z+height),(x,y+ry,z+height)]
    return mesh_object(name,v,[(0,1,4),(3,5,2),(0,4,5,3),(1,2,5,4),(0,3,2,1)],mat)


def rock(name,loc,scale,mat):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1,location=loc)
    o=bpy.context.object;o.name=name;o.scale=scale;o.data.materials.append(mat);return o


def main():
    root=Path(sys.argv[sys.argv.index('--')+1]);cfg=json.loads((root/'render.json').read_text())
    preview='--preview' in sys.argv
    rng=random.Random(cfg['seed'])
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=cfg['width'];scene.render.resolution_y=cfg['height'];scene.render.resolution_percentage=100
    scene.render.fps=24;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX'
    if hasattr(scene,'eevee'):scene.eevee.taa_render_samples=24
    scene.world.use_nodes=True
    ns=scene.world.node_tree.nodes;ls=scene.world.node_tree.links
    sky=ns.new('ShaderNodeTexSky')
    available=sky.bl_rna.properties['sky_type'].enum_items.keys()
    sky.sky_type='NISHITA' if 'NISHITA' in available else 'MULTIPLE_SCATTERING'
    sky.sun_elevation=.65;sky.sun_rotation=-1.8
    ls.new(sky.outputs[0],ns['Background'].inputs[0]);ns['Background'].inputs[1].default_value=.22
    bpy.ops.object.light_add(type='SUN',location=(-100,-100,300));sun=bpy.context.object;sun.data.energy=2.2
    sun.rotation_euler=(.45,-.5,-.7);sun.data.angle=.06
    grass=material('Persistent connected grass',(.17,.24,.055),3,.95,.45)
    stone=material('Pale weathered masonry',(.48,.45,.36),7,.93,.28)
    slate=material('Slate roof',(.12,.13,.115),9,.8,.35)
    ochre=material('Ochre turret roof',(.32,.24,.12),8,.85,.3)
    gravel=material('Shore path gravel',(.43,.40,.30),12,.95,.35)
    leaf=material('Broadleaf foliage',(.06,.13,.025),1,.95,.5)
    bark=material('Bark',(.17,.1,.045),9,1,.3)
    dark=material('Window recess',(.035,.033,.027),1,1,.1)
    water=material('Lake surface',(.025,.20,.23),.35,.2,.15)
    water.node_tree.nodes.get('Principled BSDF').inputs['Metallic'].default_value=.25
    mesh_object('One continuous lake',[(-50000,-40000,0),(50000,-40000,0),(50000,50000,0),(-50000,50000,0)],[(0,1,2,3)],water)
    # Continuous inland height field; only one lake/shoreline, no image planes.
    for name,ys,width,nx in [('Regional terrain',[-18000+i*250 for i in range(181)],28000,130),
                              ('Connected shore ground',[-500+i*6 for i in range(251)],850,100)]:
        vs=[];fs=[]
        for j,y in enumerate(ys):
            for i in range(nx+1):
                x=shore(y)+width*(i/nx)**1.8
                z=ground_height(x+.001,y)-(.1 if name=='Regional terrain' else 0)
                vs.append((x,y,z))
                if i<nx and j<len(ys)-1:
                    a=j*(nx+1)+i;fs.append((a,a+1,a+nx+2,a+nx+1))
        o=mesh_object(name,vs,fs,grass)
        for p in o.data.polygons:p.use_smooth=True
    # A persistent gravel corridor from the regional shore to the person.
    vs=[];fs=[]
    for i in range(280):
        y=-500+i*2;cx=2.3*math.sin(y*.016)-2.2*math.exp(-(y/15)**2)
        for x in [cx-1.5,cx+1.5]:vs.append((x,y,ground_height(x,y)+.15))
        if i<279:fs.append((i*2,i*2+1,i*2+3,i*2+2))
    mesh_object('Connected approach path',vs,fs,gravel)
    # Low wall follows the shore; person remains on broad connected inland ground.
    for i in range(75):
        y=-90+i*2.5;x=shore(y)+.6;z=ground_height(x,y)
        box(f'Shore retaining wall {i}',(x,y,z+.32),(1,2.5,.64),stone)
    # Castle island and masonry silhouette interpreted from the two source photos.
    box('Rock island',(-43,62,.6),(40,94,1.8),stone)
    for x,y,w,l,h in [(-42,65,15,17,29),(-46,38,25,19,16),(-43,86,23,21,17),(-57,62,10,35,13),(-29,62,9,34,15)]:
        box('Castle masonry',(x,y,1.5+h/2),(w,l,h),stone)
        roof('Persistent pitched roof',x,y,1.5+h,w/2+1,l/2+1,6,ochre if h>25 else slate)
        for dx in [-.3, .3]:
            for k in [0,1]:box('Recessed window',(x+dx*w,y-l/2-.04,5+k*5),(.8,.08,1.3),dark)
    for i,(x,y,r,h) in enumerate([(-60,27,4.5,18),(-26,29,4.2,19),(-61,88,3.8,17),(-25,88,3.8,18)]):
        cylinder('Round castle turret',x,y,1.5,r,h,stone)
        bpy.ops.mesh.primitive_cone_add(vertices=32,radius1=r+1,depth=7,location=(x,y,1.5+h+3.5))
        bpy.context.object.name='Conical turret roof';bpy.context.object.data.materials.append(ochre)
    box('Bridge deck',(-18,35,7),(20,4,1.2),stone)
    for x in [-24,-18,-12]:box('Bridge support',(x,35,3.5),(1.4,2.5,7),stone)
    # Set the castle farther behind the arrival so its complete silhouette fits
    # the fixed 35 mm portrait view; every castle component moves together.
    castle_names=('Rock island','Castle masonry','Persistent pitched roof','Recessed window',
                  'Round castle turret','Conical turret roof','Bridge deck','Bridge support')
    for obj in list(scene.objects):
        if obj.type=='MESH' and any(obj.name.startswith(n) for n in castle_names):obj.location.y+=120
    # Scale-bearing settlement/tree masses stay in world space.
    for i in range(85):
        x=rng.uniform(30,180);y=rng.uniform(-90,650);z=ground_height(x,y);h=rng.uniform(4,8)
        cylinder('Persistent tree trunk',x,y,z,.2,h*.65,bark,10)
        rock('Persistent tree canopy',(x,y,z+h*.75),(h*.35,h*.32,h*.5),leaf)
    for x,y in [(-6,-28),(6,-17),(9,10),(-11,18)]:
        z=ground_height(x,y);cylinder('Near tree trunk',x,y,z,.18,3,bark,12)
        rock('Near foliage',(x,y,z+4),(1.9,2.3,2.2),leaf)
    for i in range(45):
        x=rng.uniform(80,500);y=rng.uniform(150,1100);z=ground_height(x,y)
        box('Distant shore house',(x,y,z+4),(8,11,8),stone);roof('Distant roof',x,y,z+8,4.5,6,3,slate)
    witness=cfg['witness_points']['foreground_stone']
    rock('Foreground witness stone',witness,(1.15,1.8,.8),stone)
    for i in range(18):
        x=rng.choice([-1,1])*rng.uniform(3,8);y=rng.uniform(-50,8);r=rng.uniform(.12,.45)
        rock('Near ground stone',(x,y,ground_height(x,y)+r*.4),(r*1.3,r,r*.7),stone)
    # Articulated physical proxy: no photographic billboard or terrain projection.
    coat=material('Beige trench coat',(.49,.35,.20),30,.8,.08)
    skin=material('Skin',(.55,.34,.22),10,.7,.05)
    hair=material('Brown hair',(.065,.031,.013),20,.9,.2)
    shoe=material('Dark shoes',(.025,.022,.019),5,.5,.1)
    z=ground_height(0,0)
    box('Left shoe',(-.15,-.08,z+.06),(.18,.32,.12),shoe)
    box('Right shoe',(.15,-.08,z+.06),(.18,.32,.12),shoe)
    cylinder('Left leg',-.14,0,z+.1,.085,.62,skin,14);cylinder('Right leg',.14,0,z+.1,.085,.62,skin,14)
    bpy.ops.mesh.primitive_cone_add(vertices=24,radius1=.32,radius2=.22,depth=.88,location=(0,0,z+1.02))
    bpy.context.object.name='Coat torso';bpy.context.object.data.materials.append(coat)
    rock('Head',(0,0,z+1.59),(.115,.105,.15),skin)
    rock('Hair',(0,.035,z+1.65),(.13,.12,.13),hair)
    arms=[]
    for sign in [-1,1]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=14,radius=.073,depth=.55,location=(sign*.28,0,z+1.15))
        o=bpy.context.object;o.name='Articulated coat sleeve';o.data.materials.append(coat);arms.append((sign,o))
    for k in range(5):rock('Coat button',(.06,-.25,z+.76+k*.13),(.015,.008,.015),ochre)
    # One fixed 35 mm camera, moving through the same scene on the product clock.
    bpy.ops.object.camera_add();cam=bpy.context.object;cam.name='Fixed-lens physical camera'
    scene.camera=cam;cam.data.lens=35;cam.data.sensor_width=36;cam.data.clip_start=.05;cam.data.clip_end=150000
    cfg['witness_projection']=[]
    folder=root/('preview' if preview else 'frames');folder.mkdir(exist_ok=True)
    selected=[0,12,24,36,48,60,72,84,96] if preview else list(range(cfg['frames']))
    for i in selected:
        p=cfg['poses'][i];cam.location=p['position'];cam.rotation_euler=(Vector(p['target'])-cam.location).to_track_quat('-Z','Y').to_euler()
        for sign,arm in arms:
            u=max(0,min(1,(p['delivery_frame']-156)/36))
            angle=sign*(.08+1.15*u)
            arm.rotation_euler[1]=angle
            arm.location=(sign*(.25+.25*math.sin(abs(angle))),0,z+1.38-.25*math.cos(angle))
        bpy.context.view_layer.update()
        projected={name:list(world_to_camera_view(scene,cam,Vector(point))) for name,point in cfg['witness_points'].items()}
        cfg['witness_projection'].append(dict(delivery_frame=p['delivery_frame'],points=projected))
        scene.render.filepath=str(folder/f'{i:04d}.png');bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(root/'shoreline-scene.blend'))
    (root/('preview-audit.json' if preview else 'geometry-audit.json')).write_text(json.dumps(dict(
        version='persistent-shoreline-v1',camera_poses=cfg['poses'],witness_projection=cfg['witness_projection'],
        static_meshes=len([o for o in scene.objects if o.type=='MESH'])-len(arms),
        dynamic_objects='Two articulated sleeve proxies only',image_planes=0,view_crossfades=0,
        scope='Authored interpretation from location photographs; not surveyed geometry'),indent=2))


if __name__=='__main__':main()
