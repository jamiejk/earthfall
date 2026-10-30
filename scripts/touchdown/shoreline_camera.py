"""Pure camera/terrain math shared by the pilot runner and Blender."""
import math

START,END=96,192
FRAMES=END-START+1
FPS=24


def shore(y):
    return -14+5*math.sin(y/1400)+.001*y


def ground_height(x,y):
    if x<shore(y):return 0.
    foothill=max(0.,x-180.)
    mountain=math.exp(-((x-5200)/6000)**2)*max(0.,x-500.)*.29
    relief=(.6+.22*math.sin(x*.0013+y*.0007)+.18*math.sin(y*.0021))*mountain
    near=.045*math.sin(x*.7)*math.sin(y*.65)*min(1.,(x*x+y*y)/6.)
    return 8.+relief+foothill*.003+near


def camera_pose(frame):
    if not START<=frame<=END:raise ValueError('Frame outside the bounded original delivery interval')
    if frame<108:
        u=(frame-START)/(108-START)
        distance=12500*(9500/12500)**u
        altitude=9144.
    elif frame<=156:
        u=(frame-108)/48
        r=(math.exp(-8.6*u)-math.exp(-8.6))/(1-math.exp(-8.6))
        distance=8+(9500-8)*r
        altitude=1.62+(9144-1.62)*r
    else:
        u=(frame-156)/36
        distance=3.15+4.85*(1-u)**2
        altitude=1.62
    x=.08*(distance-3.15)
    y=-distance
    z=ground_height(x,y)+altitude
    # The same physical destination is the aim throughout; no changing focal length.
    return dict(delivery_frame=frame,time_seconds=frame/FPS,
                position=[x,y,z],target=[-.35,0,ground_height(0,0)+1.35],
                focal_length_mm=35.,altitude_above_local_ground_m=altitude)


def project(point,pose,aspect=1280/736):
    """Normalized image coordinates, also used to audit persistent witnesses."""
    import numpy as np
    eye=np.array(pose['position']);forward=np.array(pose['target'])-eye
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward);delta=np.array(point)-eye
    depth=float(delta@forward)
    if depth<=0:return None
    return [.5+float(delta@right)/depth*35/36,
            .5+float(delta@up)/depth*35/36*aspect,depth]
