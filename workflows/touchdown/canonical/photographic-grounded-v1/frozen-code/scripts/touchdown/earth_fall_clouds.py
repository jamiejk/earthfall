"""Fixed-lens ray-cast Earth and spatial cloud layers, with no image dissolves."""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from touchdown.proc_cloud import fbm

RADIUS=6_371_000.
SUN=np.array([246,247,245],np.float32)
SHADE=np.array([172,185,204],np.float32)


def rays(width,height,pitch_deg,lens=35):
    x=(np.arange(width,dtype=np.float32)+.5)/width-.5
    y=.5-(np.arange(height,dtype=np.float32)+.5)/height
    xx,yy=np.meshgrid(x*36/lens,y*36/lens*height/width)
    p=math.radians(pitch_deg)
    forward=np.array([0,math.cos(p),-math.sin(p)],np.float32)
    up=np.array([0,math.sin(p),math.cos(p)],np.float32)
    result=forward+xx[...,None]*np.array([1,0,0],np.float32)+yy[...,None]*up
    return result/np.linalg.norm(result,axis=-1,keepdims=True)


def sphere_hit(origin,directions,radius):
    o=np.array(origin,dtype=np.float64)+[0,0,RADIUS]
    direction64=directions.astype(np.float64)
    a=np.einsum('...i,...i->...',direction64,direction64)
    b=np.einsum('...i,i->...',direction64,o)
    c=float(o@o-radius*radius)
    disc=b*b-a*c
    root=np.sqrt(np.maximum(disc,0))
    near=(-b-root)/a;far=(-b+root)/a
    distance=np.where(near>0,near,np.where(far>0,far,np.inf))
    return np.where(disc>=0,distance,np.inf).astype(np.float32)


def opening_pose(t):
    if not 0<=t<=1:raise ValueError('progress must be in [0,1]')
    altitude=2000+348000*(1-t)**3
    pitch=27+18*min(1,t/.65)**2*(3-2*min(1,t/.65))
    return dict(position=[0.,-altitude,altitude],pitch_deg=pitch,focal_length_mm=35.)


class CloudField:
    def __init__(self,size=768):
        self.noise=fbm(size,17,base=12).astype(np.float32)
        self.size=size

    @staticmethod
    def value_noise(x,y,scale):
        """Unbounded world-space value noise: no repeating cloud texture tile."""
        gx=x/scale;gy=y/scale
        ix=np.floor(gx).astype(np.int64);iy=np.floor(gy).astype(np.int64)
        fx=gx-ix;fy=gy-iy;fx=fx*fx*(3-2*fx);fy=fy*fy*(3-2*fy)
        def hashed(a,b):
            n=a.astype(np.uint32)*np.uint32(374761393)+b.astype(np.uint32)*np.uint32(668265263)
            n=(n^(n>>np.uint32(13)))*np.uint32(1274126177)
            return ((n^(n>>np.uint32(16)))&np.uint32(0xffffff)).astype(np.float32)/16777215
        a=hashed(ix,iy);b=hashed(ix+1,iy);c=hashed(ix,iy+1);d=hashed(ix+1,iy+1)
        return ((a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy).astype(np.float32)

    def transport(self,origin,directions,base,top,*,spherical=False,layers=24,tile_m=40000.,ragged_base=None,time_s=None):
        """Spatial scattering, with optional wind advection and billow evolution."""
        h,w=directions.shape[:2]
        colour=np.zeros((h,w,3),np.float32);trans=np.ones((h,w),np.float32)
        origin=np.asarray(origin,np.float32)
        altitude=float(np.linalg.norm(origin.astype(np.float64)+[0,0,RADIUS])-RADIUS) if spherical else float(origin[2])
        # Only shells below a downward-facing camera can lie on the forward ray.
        heights=np.linspace(top,base,layers)
        for height in heights:
            if height>altitude:continue
            if spherical:
                distance=sphere_hit(origin,directions,RADIUS+height)
            else:
                distance=(height-origin[2])/np.minimum(directions[:,:,2],-1e-5)
            valid=(distance>0)&np.isfinite(distance)
            distance=np.where(valid,distance,0)
            px=origin[0]+directions[:,:,0]*distance+(height-base)*.27
            py=origin[1]+directions[:,:,1]*distance+(height-base)*.13
            if time_s is not None:
                # Wind carries the density field; turbulence evolves its shape.
                # Orbital billows use an accelerated weather timescale for the
                # short shot. This is density motion, not image warping.
                speed=300 if spherical else 35
                px=px-speed*time_s;py=py-speed*.42*time_s
                amplitude=500 if spherical else 55
                phase=py/(7000 if spherical else 250)+height*.003
                px+=amplitude*(np.sin(phase+time_s*1.1)-np.sin(phase))
                py+=amplitude*.6*(np.cos(phase+time_s*.8)-np.cos(phase))
            fine_scale=4000 if spherical else 300
            noise=(.55*self.value_noise(px,py,fine_scale)+.30*self.value_noise(px,py,fine_scale*.33)+
                   .15*self.value_noise(px,py,fine_scale*.09))
            macro=.65*self.value_noise(px+32871,py-16783,90000 if spherical else 2300)+.35*self.value_noise(px,py,34000 if spherical else 900)
            cover=np.clip((macro-.42)/.20,0,1)
            relative=(height-base)/(top-base)
            threshold=.18+.58*relative**1.4
            # A spatial cloud bank lies across the flight corridor. Its density is
            # fixed in world space; it does not animate to conceal an edit.
            bank=np.exp(-(px*px+(py+1000)**2)/(16000. if spherical else 2200.)**2)
            density=np.clip((noise+.65*bank*(1-relative)**2-threshold)/.15,0,1)*np.maximum(cover,bank)
            if ragged_base is not None:
                # Billows have different bottom heights across the flight path.
                # They clear at different pixels as the camera passes beneath,
                # instead of behaving like a uniform translucent sheet.
                billow=.6*self.value_noise(px+173,py-281,120)+.4*self.value_noise(px,py,45)
                local_bottom=ragged_base+220*(2*billow-1)
                edge=np.clip((height-local_bottom)/45,0,1)
                density*=edge*edge*(3-2*edge)
            alpha=density*.62*valid
            light=.2+.8*relative
            col=SHADE+(SUN-SHADE)*light
            colour+=(trans*alpha)[...,None]*col
            trans*=1-alpha
        return colour,trans


class Earth:
    def __init__(self,folder,latitude=38.31,longitude=-109.86):
        import json
        self.metadata=json.loads((folder/'globe.json').read_text())
        self.ground=np.asarray(Image.open(folder/'globe_ground.jpg').convert('RGB'),np.float32)
        self.cap=np.asarray(Image.open(folder/'cap_ground.jpg').convert('RGB'),np.float32)
        phi,lam=map(math.radians,(latitude,longitude))
        up=np.array([math.cos(phi)*math.cos(lam),math.cos(phi)*math.sin(lam),math.sin(phi)])
        east=np.array([-math.sin(lam),math.cos(lam),0]);north=np.cross(up,east)
        self.frame=np.stack([east,north,up])

    def render(self,origin,directions):
        distance=sphere_hit(origin,directions,RADIUS)
        hit=np.isfinite(distance)
        safe=np.where(hit,distance,0)
        points=np.asarray(origin)+directions*safe[...,None]+[0,0,RADIUS]
        unit=points/np.linalg.norm(points,axis=-1,keepdims=True)
        ecef=unit@self.frame
        lon=np.arctan2(ecef[:,:,1],ecef[:,:,0]);lat=np.arcsin(np.clip(ecef[:,:,2],-1,1))
        ih,iw=self.ground.shape[:2]
        ground=cv2.remap(self.ground,((lon+math.pi)/(2*math.pi)*iw).astype(np.float32),
                         ((math.pi/2-lat)/math.pi*ih).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        west,south,east,north=self.metadata['cap_bounds_deg']
        u=(np.degrees(lon)-west)/(east-west);v=(north-np.degrees(lat))/(north-south)
        ch,cw=self.cap.shape[:2]
        cap=cv2.remap(self.cap,(u*cw).astype(np.float32),(v*ch).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        edge=np.minimum.reduce([u,1-u,v,1-v]);weight=np.clip(edge/.04,0,1)*hit
        ground=ground*(1-weight[...,None])+cap*weight[...,None]
        # Tangent rays have more blue atmospheric path length. The glow is tied
        # to sphere/ray geometry, so it changes with camera position and pitch.
        normal=unit
        facing=np.abs(np.sum(normal*directions,axis=-1))
        altitude=max(0,float(np.linalg.norm(np.asarray(origin)+[0,0,RADIUS])-RADIUS))
        haze=np.clip((1-facing)**3*.75+min(.25,altitude/1_400_000),0,.83)
        blue=np.array([126,169,207],np.float32)
        ground=ground*(1-haze[...,None])+blue*haze[...,None]
        sky=np.zeros_like(ground)+np.array([1,3,7],np.float32)
        o=np.asarray(origin,dtype=np.float64)+[0,0,RADIUS]
        b=np.einsum('...i,i->...',directions.astype(np.float64),o)
        closest=np.sqrt(np.maximum(0,float(o@o)-b*b))-RADIUS
        glow=np.clip(1-closest/18000,0,1)**2
        sky+=glow[...,None]*np.array([38,92,164],np.float32)
        return np.where(hit[...,None],ground,sky)
