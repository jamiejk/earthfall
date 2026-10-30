"""Blender-only terrain ray depths for a bounded backward-ramp experiment."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

p=argparse.ArgumentParser();p.add_argument('--dem',type=Path,required=True);p.add_argument('--poses',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=True)
identity={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
          (a.poses,a.dem/'metadata.json',a.dem/'height_detail.npy',a.dem/'height_far.npy')}
identity['renderer']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest=a.output/'inputs.json'
if not manifest.exists() or json.loads(manifest.read_text()) != identity:
 for path in a.output.glob('*.npy'): path.unlink()
 manifest.write_text(json.dumps(identity,indent=2))
m=json.loads((a.dem/'metadata.json').read_text());trees=[]
for name,step in [('height_detail.npy',1),('height_far.npy',50)]:
 h=np.load(a.dem/name)-m['landing'];n=len(h)
 vertices=[((j-(n-1)/2)*step,((n-1)/2-i)*step,float(h[i,j])) for i in range(n) for j in range(n)]
 faces=[(i*n+j,(i+1)*n+j,(i+1)*n+j+1,i*n+j+1) for i in range(n-1) for j in range(n-1)]
 trees.append(BVHTree.FromPolygons(vertices,faces))
poses=json.loads(a.poses.read_text())['guide_poses'];w,ht=480,272;f=w/(2*np.tan(np.pi/6))
for index in range(56,81):
 if (a.output/f'{index:04d}.npy').exists(): continue
 pose=poses[index];camera=Vector(pose['camera_xyz_m']);forward=(Vector(pose['aim_xyz_m'])-camera).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward).normalized();depth=np.full((ht,w),np.nan,np.float32)
 for y in range(ht):
  for x in range(w):
   direction=(forward+right*((x+.5-w/2)/f)+up*((ht/2-y-.5)/f)).normalized()
   hits=[tree.ray_cast(camera,direction,40000) for tree in trees]
   distances=[hit[3] for hit in hits if hit[0] is not None]
   if distances: depth[y,x]=min(distances)*direction.dot(forward)
 np.save(a.output/f'{index:04d}.npy',depth);print(index,flush=True)
