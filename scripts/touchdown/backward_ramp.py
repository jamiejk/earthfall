"""Fixed-seed backwards FLUX conditioning trial, followed by one LTX pass per arm.

Experimental runner; not enabled in the production build. Mesh depth is a proxy
for photographed structures, not measured depth of the inserted person/railings.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from . import video as V


def basis(pose):
    camera=np.asarray(pose['camera_xyz_m'],float)
    forward=np.asarray(pose['aim_xyz_m'],float)-camera;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    return camera,forward,right,np.cross(right,forward)


def reproject(image, guide, depth, current_pose, later_pose, later_depth=None):
    """Inverse warp using terrain axial depth and exact camera poses; fill holes from guide."""
    h,w=guide.shape[:2];image=cv2.resize(image,(w,h));depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
    c,f,r,u=basis(current_pose);lc,lf,lr,lu=basis(later_pose)
    yy,xx=np.mgrid[:h,:w];focal=w/(2*np.tan(np.pi/6))
    rays=f+((xx+.5-w/2)/focal)[...,None]*r+((h/2-yy-.5)/focal)[...,None]*u
    finite=np.isfinite(depth);rel=c-lc+np.nan_to_num(depth,nan=40000)[...,None]*rays
    # Sky uses only camera rotation, rather than an invented ground intersection.
    rel[~finite]=rays[~finite]*40000
    z=rel@lf;mx=(w/2-.5+focal*(rel@lr)/np.maximum(z,.01)).astype(np.float32)
    my=(h/2-.5-focal*(rel@lu)/np.maximum(z,.01)).astype(np.float32)
    valid=(z>0)&(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
    if later_depth is not None:
        ld=cv2.resize(later_depth,(w,h),interpolation=cv2.INTER_NEAREST)
        expected=cv2.remap(ld,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
        valid &= (~finite & ~np.isfinite(expected)) | (finite & np.isfinite(expected) & (np.abs(expected-z)<np.maximum(1,.03*z)))
    warped=cv2.remap(image,mx,my,cv2.INTER_LINEAR)
    warped[~valid]=guide[~valid]
    return warped, float(valid.mean())


def flow_warp(image, guide, later_guide):
    h,w=guide.shape[:2];a=cv2.cvtColor(guide,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(later_guide,cv2.COLOR_RGB2GRAY)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a,b,None)
    yy,xx=np.mgrid[:h,:w];mx=(xx+flow[:,:,0]).astype(np.float32);my=(yy+flow[:,:,1]).astype(np.float32)
    valid=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
    warped=cv2.remap(cv2.resize(image,(w,h)),mx,my,cv2.INTER_LINEAR)
    warped[~valid]=guide[~valid]
    return warped,float(valid.mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--endpoint',type=Path,required=True)
    p.add_argument('--depth',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comfy-python',required=True);p.add_argument('--url',default='http://127.0.0.1:8191')
    p.add_argument('--mode',choices=['plain','depth'],required=True);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--appearance-only',action='store_true',help='Appearance branch references only the later image')
    p.add_argument('--denoise',type=float,help='Hold denoise fixed; otherwise use the partial-denoise ramp')
    a=p.parse_args()
    if a.denoise is not None and not 0 < a.denoise <= 1: p.error('--denoise must be in (0,1]')
    a.output.mkdir(parents=True,exist_ok=True)
    poses=json.loads((a.scene/'terrain_guide.json').read_text())['guide_poses']
    guides={i:np.asarray(Image.open(a.scene/'guide_frames'/f'{i+1:04d}.png').convert('RGB').resize((960,544))) for i in range(81)}
    anchors=list(range(56,81,3));edited={80:a.endpoint};records=[]
    prompt=('Make image 1 a real photograph from exactly its existing camera viewpoint. '
      'Preserve its camera angle, landscape silhouette and scale. Use the later photograph '
      'for the same person, materials and connected pedestrian setting, adapting them to image 1 viewpoint. '
      'One coherent physical place, realistic perspective and parallax, natural daylight. '
      'The woman wears a mustard yellow jacket, dark green trousers and brown boots. '
      'Do not copy a person at the later image size; place her at the destination at correct perspective scale.')
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'src')])}
    def job(name,kw):
        result=subprocess.run([a.comfy_python,'-m','touchdown.comfy_job',name,json.dumps({'url':a.url,**kw})],env=env,text=True,capture_output=True)
        if result.returncode: raise RuntimeError(result.stderr[-4000:])
        return json.loads(result.stdout.strip().splitlines()[-1])['result']
    for index,later_index in zip(anchors[-2::-1],anchors[:0:-1]):
        folder=a.output/f'frame_{index:04d}';folder.mkdir(exist_ok=True)
        guide=folder/'guide.png';Image.fromarray(guides[index]).save(guide)
        later=edited[later_index];coverage=None
        if a.mode=='depth':
            warped,coverage=reproject(np.asarray(Image.open(later).convert('RGB')),guides[index],np.load(a.depth/f'{index:04d}.npy'),poses[index],poses[later_index],np.load(a.depth/f'{later_index:04d}.npy'))
            later=folder/'projected.png';Image.fromarray(warped).save(later)
        t=(index-56)/24;strength=.2+.7*t;denoise=a.denoise if a.denoise is not None else .6+.25*t
        kw={'folder':str(folder),'guide':str(guide),'later':str(later),'strength':strength,'denoise':denoise,'seed':a.seed,'prompt':prompt,'appearance_only':a.appearance_only}
        identity={**kw,'guide_sha':hashlib.sha256(guide.read_bytes()).hexdigest(),'later_sha':hashlib.sha256(later.read_bytes()).hexdigest(), 'job_code':hashlib.sha256(Path(__file__).with_name('comfy_job.py').read_bytes()).hexdigest()}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();done=folder/'done.json'
        if done.exists() and json.loads(done.read_text())['key']==key:
            out=json.loads(done.read_text())['output']
        else:
            out=job('backward_edit',kw);done.write_text(json.dumps({'key':key,'output':out},indent=2))
        edited[index]=Path(out);record={'frame':index,'later_frame':later_index,'strength':strength,'denoise':denoise,'seed':a.seed,'appearance_only':a.appearance_only,'coverage':coverage,'output':str(out)};records.append(record)
        print(json.dumps(record),flush=True)
        (a.output/'ramp.json').write_text(json.dumps({'mode':a.mode,'records':records,'limitations':['Terrain depth does not reconstruct invented person or railings','Sparse edits; intermediate views are geometry-warped without crossfades']},indent=2))
    with V.Writer(a.output/'guide.mp4') as writer:
        for index in range(81):
            if index<56: image=guides[index]
            elif index in edited: image=np.asarray(Image.open(edited[index]).convert('RGB').resize((960,544)))
            else:
                later_index=min(i for i in edited if i>index);later_image=np.asarray(Image.open(edited[later_index]).convert('RGB'))
                if a.mode=='depth': image,_=reproject(later_image,guides[index],np.load(a.depth/f'{index:04d}.npy'),poses[index],poses[later_index],np.load(a.depth/f'{later_index:04d}.npy'))
                else: image,_=flow_warp(later_image,guides[index],guides[later_index])
            writer.write(cv2.resize(image,(V.W,V.H)))
    out=job('terrain_v2v',{'guide':str(a.output/'guide.mp4'),'first':str(a.scene/'guide_frames'/'0001.png'),'last':str(a.endpoint),'output':str(a.output/'ltx'),'seed':7,'frames':81,
        'prompt':'One continuous straight camera descent following the guide. Preserve the same connected pedestrian platform and canyon, person identity and viewpoint. Natural daylight, realistic perspective and smooth parallax, no dissolves.'})
    shutil.copy(out,a.output/'swoop.mp4')
    from .terrain_quality import assess,assess_surface_sequence
    quality=assess(a.output/'swoop.mp4',a.endpoint)
    masks=a.scene/'guide_frames_surface_masks'
    if masks.is_dir(): quality['surface_sequence']=assess_surface_sequence(a.output/'swoop.mp4',masks)
    (a.output/'quality.json').write_text(json.dumps(quality,indent=2))
    print(str(a.output/'swoop.mp4'),flush=True)

if __name__=='__main__':main()
