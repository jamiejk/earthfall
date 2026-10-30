"""A sculpted sandstone approach with persistent route, cliffs and foreground."""
import json
import math
import random
import sys
from pathlib import Path
import bpy
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from authored_scene_blender import image_material,mesh_object
from authored_film import centre,surface


def project_route_photo(folder, objects, poses, aspect):
    """Two fixed photographic projections, blended by world location.

    The wide view covers the start; the closer view supplies the destination.
    A local material restores fine ground detail and contact shadows at arrival.
    """
    photo=folder/'route_photo.png'
    if not photo.is_file():return
    wide=folder/'route_wide_photo.png'
    if not wide.is_file():wide=photo
    material=bpy.data.materials.new('Fixed photographic route projections');material.use_nodes=True
    ns=material.node_tree.nodes;ls=material.node_tree.links;ns.clear()
    output=ns.new('ShaderNodeOutputMaterial')
    geometry=ns.new('ShaderNodeNewGeometry');sep=ns.new('ShaderNodeSeparateXYZ')
    ls.new(geometry.outputs['Position'],sep.inputs[0])
    projected=[];coverage=[]
    for path,pose in [(photo,poses[48]),(wide,poses[0])]:
        origin=Vector(pose['position']);forward=(Vector(pose['target'])-origin).normalized()
        right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward).normalized()
        delta=ns.new('ShaderNodeVectorMath');delta.operation='SUBTRACT';delta.inputs[1].default_value=origin
        ls.new(geometry.outputs['Position'],delta.inputs[0])
        def dot(axis):
            node=ns.new('ShaderNodeVectorMath');node.operation='DOT_PRODUCT';node.inputs[1].default_value=axis
            ls.new(delta.outputs[0],node.inputs[0]);return node.outputs['Value']
        depth=ns.new('ShaderNodeMath');depth.operation='MAXIMUM';depth.inputs[1].default_value=.05
        ls.new(dot(forward),depth.inputs[0])
        combine=ns.new('ShaderNodeCombineXYZ')
        for index,(axis,scale) in enumerate([(right,pose['focal_length_mm']/36),(up,pose['focal_length_mm']/36*aspect)]):
            divide=ns.new('ShaderNodeMath');divide.operation='DIVIDE';ls.new(dot(axis),divide.inputs[0]);ls.new(depth.outputs[0],divide.inputs[1])
            transform=ns.new('ShaderNodeMath');transform.operation='MULTIPLY_ADD';transform.inputs[1].default_value=scale;transform.inputs[2].default_value=.5
            ls.new(divide.outputs[0],transform.inputs[0]);ls.new(transform.outputs[0],combine.inputs[index])
        tex=ns.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True);tex.extension='EXTEND'
        ls.new(combine.outputs[0],tex.inputs[0]);projected.append(tex)
        xy=ns.new('ShaderNodeSeparateXYZ');ls.new(combine.outputs[0],xy.inputs[0])
        edges=[]
        for component in ['X','Y']:
            edges.append(xy.outputs[component])
            inv=ns.new('ShaderNodeMath');inv.operation='SUBTRACT';inv.inputs[0].default_value=1
            ls.new(xy.outputs[component],inv.inputs[1]);edges.append(inv.outputs[0])
        nearest=edges[0]
        for edge in edges[1:]:
            minimum=ns.new('ShaderNodeMath');minimum.operation='MINIMUM'
            ls.new(nearest,minimum.inputs[0]);ls.new(edge,minimum.inputs[1]);nearest=minimum.outputs[0]
        valid=ns.new('ShaderNodeMapRange');valid.clamp=True;valid.inputs['From Min'].default_value=0
        valid.inputs['From Max'].default_value=.035;ls.new(nearest,valid.inputs['Value']);coverage.append(valid.outputs[0])
    def world_ramp(start,end):
        ramp=ns.new('ShaderNodeMapRange');ramp.clamp=True
        ramp.inputs['From Min'].default_value=start;ramp.inputs['From Max'].default_value=end
        ls.new(sep.outputs['Y'],ramp.inputs['Value']);return ramp.outputs['Result']
    near_weight=ns.new('ShaderNodeMath');near_weight.operation='MULTIPLY'
    ls.new(world_ramp(-25,-15),near_weight.inputs[0]);ls.new(coverage[0],near_weight.inputs[1])
    colour=ns.new('ShaderNodeMixRGB');ls.new(near_weight.outputs[0],colour.inputs[0])
    ls.new(projected[1].outputs['Color'],colour.inputs[1]);ls.new(projected[0].outputs['Color'],colour.inputs[2])
    em=ns.new('ShaderNodeEmission');ls.new(colour.outputs[0],em.inputs[0])
    localuv=ns.new('ShaderNodeUVMap');localuv.uv_map='Photographic coordinates'
    stone=ns.new('ShaderNodeTexImage');stone.image=bpy.data.images.load(str(folder/'ground.png'),check_existing=True)
    ls.new(localuv.outputs[0],stone.inputs[0])
    bsdf=ns.new('ShaderNodeBsdfPrincipled');bsdf.inputs['Roughness'].default_value=.9
    ls.new(stone.outputs['Color'],bsdf.inputs['Base Color'])
    mix=ns.new('ShaderNodeMixShader');ls.new(world_ramp(-10,-3),mix.inputs[0])
    ls.new(em.outputs[0],mix.inputs[1]);ls.new(bsdf.outputs[0],mix.inputs[2]);ls.new(mix.outputs[0],output.inputs[0])
    for obj in objects:
        chosen=material
        if obj.name!='Unbroken winding sandstone spur' and not obj.name.startswith('Weathered rim fragment'):
            chosen=material.copy()
            copied_mix=next(n for n in chosen.node_tree.nodes if n.type=='MIX_SHADER')
            for link in list(copied_mix.inputs[0].links):chosen.node_tree.links.remove(link)
            copied_mix.inputs[0].default_value=0
        obj.data.materials.clear();obj.data.materials.append(chosen)


def cliff_material(folder):
    mat=image_material('Reference sandstone with horizontal strata',folder/'ground.png')
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    tex=next(n for n in nodes if n.type=='TEX_IMAGE')
    coord=nodes.new('ShaderNodeTexCoord'); sep=nodes.new('ShaderNodeSeparateXYZ')
    links.new(coord.outputs['Geometry'] if 'Geometry' in coord.outputs else coord.outputs['Generated'],sep.inputs[0])
    # World position gives every exposed rock the same sedimentary bedding scale.
    geometry=nodes.new('ShaderNodeNewGeometry');links.new(geometry.outputs['Position'],sep.inputs[0])
    wave=nodes.new('ShaderNodeMath');wave.operation='MULTIPLY';wave.inputs[1].default_value=5.8
    links.new(sep.outputs['Z'],wave.inputs[0])
    sin=nodes.new('ShaderNodeMath');sin.operation='SINE';links.new(wave.outputs[0],sin.inputs[0])
    ramp=nodes.new('ShaderNodeValToRGB');links.new(sin.outputs[0],ramp.inputs[0])
    ramp.color_ramp.elements[0].position=0
    ramp.color_ramp.elements[0].color=(.24,.12,.07,1)
    ramp.color_ramp.elements[1].position=1
    ramp.color_ramp.elements[1].color=(.60,.43,.29,1)
    mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.68
    links.new(tex.outputs['Color'],mix.inputs[1]);links.new(ramp.outputs['Color'],mix.inputs[2])
    shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    links.new(mix.outputs[0],shader.inputs['Base Color'])
    return mat


def ledge(name,x,y,z,rx,ry,height,ground,cliff,seed):
    """Stratified irregular rock mass with a textured cap, not a smooth boulder."""
    rng=random.Random(seed); sides=28;levels=12
    rim=[1+rng.uniform(-.13,.13) for _ in range(sides)]
    verts=[];uv=[];faces=[]
    for j in range(levels+1):
        f=j/levels
        ledge_scale=.85+.15*f+.045*math.sin(j*2.8)+.018*math.sin(j*7.1)
        for i in range(sides):
            angle=2*math.pi*i/sides
            xx=x+rx*rim[i]*math.cos(angle)*ledge_scale
            yy=y+ry*rim[i]*math.sin(angle)*ledge_scale
            zz=z-height*(1-f)+.04*math.sin(angle*5)
            verts.append((xx,yy,zz));uv.append((i/sides*2,f*height/5))
            if j<levels:
                a=j*sides+i;b=j*sides+(i+1)%sides
                faces.append((a,b,b+sides,a+sides))
    obj=mesh_object(name+' exposed strata',verts,faces,cliff,uv)
    cap=verts[-sides:]+[(x,y,z+.035)]
    capuv=[((v[0]-x)/7,(v[1]-y)/7) for v in cap]
    mesh_object(name+' weathered cap',cap,[(i,(i+1)%sides,sides) for i in range(sides)],ground,capuv)
    return obj


def main():
    folder=Path(sys.argv[sys.argv.index('--')+1]);cfg=json.loads((folder/'render.json').read_text())
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=cfg['width'];scene.render.resolution_y=cfg['height'];scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.fps=24
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.56,.66,.78,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.58
    ground=image_material('Continuous reference-conditioned sandstone',folder/'ground.png')
    cliff=cliff_material(folder)

    # A curved promontory with broken sides. The person is planted in this mesh.
    vertices=[];uvs=[];faces=[];nx=48;ny=260
    left=[];right=[]
    for j in range(ny+1):
        y=-145+153*j/ny
        width=4.8+max(0,-y)*.055+1.1*math.sin(y*.15)+.45*math.sin(y*.51)
        for i in range(nx+1):
            u=i/nx; x=centre(y)+(2*u-1)*width
            z=surface(x,y)+(.11*math.sin(y*1.4+x*3))*abs(2*u-1)**3
            vertices.append((x,y,z));uvs.append((x/6,y/6))
            if i==0:left.append((x,y,z))
            if i==nx:right.append((x,y,z))
            if i<nx and j<ny:
                k=j*(nx+1)+i;faces.append((k,k+1,k+nx+2,k+nx+1))
    mesh_object('Unbroken winding sandstone spur',vertices,faces,ground,uvs)
    # Two stepped cliff walls join the cap at exactly its edge.
    for side,points in [('Left',left),('Right',right)]:
        vs=[];uv=[];fs=[];levels=18
        for j,(x,y,z) in enumerate(points):
            for k in range(levels+1):
                f=k/levels;outward=(-1 if side=='Left' else 1)
                xx=x+outward*(f*7+.28*math.sin(k*2.4)+.1*math.sin(y*.5))*f
                wall_depth=140 if cfg.get('regional_ground') else 38
                vs.append((xx,y,z-f*wall_depth));uv.append((y/7,-f*wall_depth/7))
                if j<len(points)-1 and k<levels:
                    a=j*(levels+1)+k;fs.append((a,a+1,a+levels+2,a+levels+1))
        mesh_object(side+' continuous broken cliff',vs,fs,cliff,uv)
    # Foreground rock shoulders beside the route. Their tops are below the initial
    # flight, then cross the frame as the camera descends toward the portrait.
    for i,(y,side,rx,ry,h) in enumerate([(-76,-1,3.0,6,3.8),(-55,1,3.0,5,3.0),
                (-34,-1,2.1,4,2.4),(-21,1,1.9,3.2,1.8),(-10,-1,1.35,2,1.1)]):
        x=centre(y)+side*(4.5+max(0,-y)*.025)
        ledge('Route shoulder '+str(i),x,y,surface(x,y)+h,rx,ry,h+2,ground,cliff,100+i)
    # Canyon forms between the near spur and distant photograph supply parallax
    # and cover the straight lower edge of the photographic backdrop.
    for i,(x,y,rx,ry,z) in enumerate([(-65,20,30,42,-12),(62,38,28,40,-17),
            (-95,120,45,60,-5),(105,180,50,68,-14),(-12,240,32,48,-28)]):
        ledge('Middle distance mesa '+str(i),x,y,z,rx,ry,z+145 if cfg.get('regional_ground') else 65,ground,cliff,200+i)
    rng=random.Random(730)
    for i in range(95):
        y=rng.uniform(-110,6);side=rng.choice([-1,1]);x=centre(y)+side*rng.uniform(2.3,4.2)
        r=rng.uniform(.06,.23)
        ledge('Weathered rim fragment '+str(i),x,y,surface(x,y)+r,r*1.5,r,r*1.1,ground,cliff,500+i)

    project_route_photo(folder,[o for o in bpy.context.scene.objects if o.type=='MESH'],
                        cfg.get('projection_poses',cfg['poses']),cfg['width']/cfg['height'])
    backdrop=image_material('Distant canyon sky photographic backdrop',folder/'master.png',emission=True)
    distance=750;width=1100;height=width*1536/2688;cz=-115
    mesh_object('Far atmospheric canyon', [(-width/2,distance,cz-height/2),(width/2,distance,cz-height/2),
                (width/2,distance,cz+height/2),(-width/2,distance,cz+height/2)],[(0,1,2,3)],backdrop,
                [(0,0),(1,0),(1,1),(0,1)])
    for side in [-1,1]:
        x=side*width
        mesh_object('Peripheral canyon continuation '+str(side),[(x-width/2,distance,cz-height/2),
                    (x+width/2,distance,cz-height/2),(x+width/2,distance,cz+height/2),
                    (x-width/2,distance,cz+height/2)],[(0,1,2,3)],backdrop,
                    [(1,0),(0,0),(0,1),(1,1)])
    floor=image_material('Distant pale canyon floor',folder/'ground.png')
    floor_nodes=floor.node_tree.nodes;floor_links=floor.node_tree.links
    floor_tex=next(n for n in floor_nodes if n.type=='TEX_IMAGE')
    floor_shader=next(n for n in floor_nodes if n.type=='BSDF_PRINCIPLED')
    tint=floor_nodes.new('ShaderNodeMixRGB');tint.blend_type='MULTIPLY';tint.inputs[0].default_value=1
    tint.inputs[2].default_value=(.48,.32,.23,1)
    floor_links.new(floor_tex.outputs['Color'],tint.inputs[1]);floor_links.new(tint.outputs[0],floor_shader.inputs['Base Color'])
    if cfg.get('regional_ground'):
        regional=image_material('Regional canyon aerial reference',folder/'under_cloud.png',emission=True)
        vs=[];uv=[];fs=[];count=120
        for j in range(count+1):
            y=-1800+4000*j/count
            for i in range(count+1):
                x=-2000+4000*i/count
                z=-125+22*math.sin(x*.005)*math.sin(y*.007)+12*math.sin(x*.019+y*.013)
                vs.append((x,y,z));uv.append((i/count,j/count))
                if i<count and j<count:
                    k=j*(count+1)+i;fs.append((k,k+1,k+count+2,k+count+1))
        obj=mesh_object('Broad regional canyon surface',vs,fs,regional,uv)
        for polygon in obj.data.polygons:polygon.use_smooth=True
    else:
        mesh_object('Deep canyon floor',[(-1800,-800,-105),(1800,-800,-105),(1800,1700,-105),(-1800,1700,-105)],
                    [(0,1,2,3)],floor,[(0,0),(40,0),(40,30),(0,30)])
    person=image_material('Reference person photographic card',folder/'person_rgba.png',emission=True,alpha=True)
    w,h=cfg['matte']['size'];ph=cfg['matte']['subject_height_m'];hw=ph*w/h/2
    mesh_object('Person at end of route',[(-hw,0,0),(hw,0,0),(hw,0,ph),(-hw,0,ph)],[(0,1,2,3)],person,
                [(0,0),(1,0),(1,1),(0,1)])
    bpy.ops.object.light_add(type='SUN',location=(-20,-10,35));sun=bpy.context.object
    sun.rotation_euler=(math.radians(28),math.radians(-32),math.radians(-30));sun.data.energy=1.3;sun.data.angle=.18
    bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera
    camera.data.sensor_width=36;camera.data.clip_start=.05;camera.data.clip_end=3000
    frames=folder/('preview' if cfg['preview'] else 'frames');frames.mkdir(exist_ok=True)
    indices=sorted(set(round(t*(cfg['frames']-1)) for t in [0,.1,.2,.3,.4,.5,.65,.8,1])) if cfg['preview'] else range(cfg['frames'])
    for i in indices:
        pose=cfg['poses'][i];camera.location=pose['position'];direction=Vector(pose['target'])-camera.location
        camera.rotation_euler=direction.to_track_quat('-Z','Y').to_euler();camera.data.lens=pose['focal_length_mm']
        scene.render.filepath=str(frames/f'{i:04d}.png');bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'sandstone_spur.blend'))

if __name__=='__main__':main()
