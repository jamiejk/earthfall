import cv2
import numpy as np
from touchdown.opening_motion import pair_motion, opening_motion


def texture():
    rng=np.random.default_rng(42)
    image=rng.integers(30,220,(368,640,3),dtype=np.uint8)
    return cv2.GaussianBlur(image,(5,5),0)


def test_changing_noise_on_stationary_picture_is_a_hold():
    a=texture();noise=np.random.default_rng(9).normal(0,2,a.shape)
    b=np.clip(a.astype(float)+noise,0,255).astype(np.uint8)
    r=pair_motion(a,b)
    assert r["status"]=="hold" and r["median_camera_travel_px"]<.1


def test_coherent_translation_moves_and_textureless_frame_is_unclear():
    a=texture();b=cv2.warpAffine(a,np.float32([[1,0,4],[0,1,2]]),(640,368))
    r=pair_motion(a,b)
    assert r["status"]=="moving" and 4<r["median_camera_travel_px"]<5
    assert pair_motion(np.zeros_like(a),np.zeros_like(a))["status"]=="unclear"


def test_motion_begins_late_cannot_pass_opening_check(tmp_path):
    path=tmp_path/"late.mp4";writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),24,(640,368))
    a=texture()
    for i in range(49):
        x=max(0,i-24)*2
        writer.write(cv2.warpAffine(a,np.float32([[1,0,x],[0,1,0]]),(640,368)))
    writer.release()
    report=opening_motion(path)
    assert report["status"]=="fail"
    assert all(i["status"]=="hold" for i in report["intervals"])
