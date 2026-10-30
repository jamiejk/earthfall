"""Assemble a computed descent and fixed reversed landing, with optional opaque mist.

The mist conceals a geometry change; it does not reconstruct matching terrain.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from . import video as V


def fog_alpha(distance,noise,radius=8):
    if abs(distance)<=.5:return np.ones_like(noise,np.float32)
    if abs(distance)>=radius:return np.zeros_like(noise,np.float32)
    x=(radius-abs(distance))/(radius-.5);weight=x*x*(3-2*x)
    return np.clip(weight+(.5-noise)*weight*(1-weight)*.7,0,1).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prefix','descent','landing','living','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prefix-frames',type=int,required=True);p.add_argument('--mist',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    prefix_n=a.prefix_frames;descent_n=V.count(a.descent);landing_n=V.count(a.landing)
    join=prefix_n+descent_n;end=join+landing_n
    bounds={'opener':[0,22],'cloud':[22,prefix_n],'dive':[prefix_n,join],'swoop':[join,end],'end':[end,end+V.count(a.living)-9]}
    (a.output/'segments.json').write_text(json.dumps(bounds,indent=2))
    white=V.frame(a.prefix,prefix_n-1)
    rng=np.random.default_rng(20261013)
    noise=cv2.GaussianBlur(cv2.resize(rng.random((32,64)).astype(np.float32),(V.W,V.H)),(0,0),20)
    def apply(image,index):
        if not a.mist:return image
        distance=index-(join-.5)
        field=np.roll(noise,int(distance*6),axis=1)
        alpha=fog_alpha(distance,field)
        cloud=np.repeat((248-18*field)[:,:,None],3,axis=2)
        return np.clip(image.astype(np.float32)*(1-alpha[:,:,None])+cloud*alpha[:,:,None],0,255).astype(np.uint8)
    out=a.output/'silent.mp4'
    with V.Writer(out) as writer:
        index=0
        for i,image in enumerate(V.frames(a.prefix)):
            if i>=prefix_n:break
            writer.write(image);index+=1
        for i,image in enumerate(V.frames(a.descent)):
            if i<10:
                veil=1-V.smoothstep(i/10)
                image=np.clip(image.astype(float)*(1-veil)+white.astype(float)*veil,0,255).astype(np.uint8)
            # Lens blur during the dive; no frame blends or digital push.
            if i<descent_n-12:
                image=cv2.GaussianBlur(image,(0,0),.6)
            writer.write(apply(image,index));index+=1
        for image in V.frames(a.landing):
            writer.write(apply(image,index));index+=1
        for i,image in enumerate(V.frames(a.living)):
            if i<9:continue
            writer.write(image);index+=1
    # CLI keeps procedural wind's existing timeline/fade implementation.
    import subprocess,sys
    subprocess.run([sys.executable,'-m','touchdown.wind_bed','--video',str(out),
        '--segments',str(a.output/'segments.json'),'--out',str(a.output/'touchdown.mp4'),'--seed','7'],check=True)
    (a.output/'join.json').write_text(json.dumps({'mist':a.mist,'cut_frame':join,
        'fully_occluded_frames':[join-1,join] if a.mist else [],
        'mist_radius_frames':8,'landing_retimed':False,'living_prefix_skipped':9,
        'sources':{k:str(getattr(a,k)) for k in ['prefix','descent','landing','living']},
        'limitations':['Different invented and measured geometry; occlusion is concealment, not reconstruction']},indent=2))
    print(a.output/'touchdown.mp4',flush=True)

if __name__=='__main__':main()
