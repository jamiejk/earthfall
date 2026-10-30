"""Project a sky-free crop of a failed flat-surface seed; diagnostic, not an accepted orbit."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from touchdown.orbital_reference import sha, verify, write


def project(root):
    verify(root)
    source = root / 'expand-1/raw-outpaint.png'
    texture = cv2.imread(str(source))[192:1088, 169:1751]
    assert texture.shape == (896, 1582, 3)
    cv2.imwrite(str(root/'diagnostic-surface-crop.png'), texture)
    w, h = 1920, 1088
    origin = np.array([0., 0., 1.08])
    pitch = np.deg2rad(45)
    forward, up = np.array([0., np.cos(pitch), -np.sin(pitch)]), np.array([0., np.sin(pitch), np.cos(pitch)])
    xx, yy = np.meshgrid((np.arange(w)+.5-w/2)/(w/2), -(np.arange(h)+.5-h/2)/(w/2))
    ray = forward + np.tan(np.deg2rad(40))*(xx[...,None]*np.array([1.,0.,0.])+yy[...,None]*up)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    b = ray@origin
    discriminant = b*b-(origin@origin-1)
    valid = (discriminant>=0)&(b<0)
    point = origin + ray*(-b-np.sqrt(np.maximum(0, discriminant)))[...,None]
    longitude, latitude = np.arctan2(point[...,0],point[...,2]), np.arcsin(np.clip(point[...,1],-1,1))
    center_t = 1.08*np.sin(pitch)-np.sqrt((1.08*np.sin(pitch))**2-(1.08**2-1))
    center_lat = np.arcsin(center_t*np.cos(pitch))
    u, v = .5+longitude/1.6, .5-(latitude-center_lat)/.9
    assert np.all(((u>=0)&(u<=1)&(v>=0)&(v<=1))[valid]), 'Missing texture coverage'
    mapped = cv2.remap(texture,(u*(texture.shape[1]-1)).astype(np.float32),
        (v*(texture.shape[0]-1)).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    normal = point/np.linalg.norm(point,axis=-1,keepdims=True)
    cosine = np.clip(np.sum(normal*(-ray),axis=-1),0,1)
    haze = np.clip((1-cosine)**3*.65,0,.65)[...,None]
    result = mapped*(1-haze)+np.array([205.,157.,105.])*haze
    result[~valid] = [8,5,3]
    cv2.imwrite(str(root/'globe-projection-diagnostic.png'),result.clip(0,255).astype(np.uint8))
    audit = dict(status='failed_surface_reference_diagnostic_only',
        source=str(source),source_sha256=sha(source),crop=[169,192,1751,1088],
        globe_sha256=sha(root/'globe-projection-diagnostic.png'),
        successful_outpaint_stages=1,second_expansion_submitted=False,flux_finish_submitted=False,
        radius=1,camera_height_in_radii=.08,pitch_degrees=45,horizontal_fov_degrees=80,
        texture_longitude_span_radians=1.6,texture_latitude_span_radians=.9,
        source_region_nominal_uv=[(480-169)/1582,(272-192)/896,(1440-169)/1582,(816-192)/896],
        limitations=['Oblique foreground relief baked into the texture',
                    'Local features are too large for a believable orbital photograph',
                    'Camera position is defined but ground scale and geography are not registered'],
        video_submissions=0,app_promoted=False,runner_sha256=sha(Path(__file__)))
    write(root/'diagnostic-projection.json',audit)
    write(root/'status.json',dict(status='held_failed_flat_surface_and_scale',video_submissions=0,
        reason='Overhead conversion remained oblique; native Fill added sky. Projection retained as a diagnostic only.',
        second_expansion_submitted=False,flux_finish_submitted=False,app_promoted=False))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    project(p.parse_args().root.resolve())
