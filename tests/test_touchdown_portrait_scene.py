import cv2
import numpy as np
from touchdown.portrait_scene import dolly_maps,ground_maps,extract_person


def test_zero_camera_travel_preserves_pixel_coordinates():
    depth=np.full((40,60),8.,np.float32);yy,xx=np.mgrid[:40,:60]
    mx,my=dolly_maps(depth,0)
    assert np.allclose(mx,xx) and np.allclose(my,yy)
    mx,my=ground_maps((40,60),1.5,0)
    assert np.allclose(mx,xx) and np.allclose(my,yy)


def test_ground_and_person_projection_agree_at_planted_feet():
    h,w=80,120;focal=w/(2*np.tan(np.pi/6));depth=8.;foot_y=65.;height=depth*(foot_y-(h-1)/2)/focal
    distance=1.;current_y=(h-1)/2+(foot_y-(h-1)/2)/(1+distance/depth)
    _,ground_y=ground_maps((h,w),height,distance)
    _,person_y=dolly_maps(np.full((h,w),depth),distance)
    ground=cv2.getRectSubPix(ground_y,(1,1),((w-1)/2,current_y))[0,0]
    person=cv2.getRectSubPix(person_y,(1,1),((w-1)/2,current_y))[0,0]
    assert abs(ground-foot_y)<.05 and abs(person-foot_y)<.05


def test_person_extraction_does_not_clip_boots_below_coat():
    image=np.full((600,1000,3),130,np.uint8)
    image[125:161,480:520]=(180,130,110)
    image[160:290,460:540]=(220,180,30)
    image[290:460,465:495]=(20,40,20);image[290:460,505:535]=(20,40,20)
    image[450:477,460:495]=(80,45,25);image[450:477,505:540]=(80,45,25)
    mask,box=extract_person(image)
    assert mask[475,475] and mask[475,520]
    assert box[3]>=476


def test_short_trial_diagnostics_keep_production_frame_count_strict(monkeypatch,tmp_path):
    import pytest
    import fade_check
    from PIL import Image
    from touchdown import terrain_quality as quality
    rows=np.tile([1.,2.,.8,200.],(72,1))
    monkeypatch.setattr(fade_check,'measure',lambda clip:rows)
    yy,xx=np.mgrid[:136,:240];rgb=np.stack([xx,yy,xx],axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path/'end.png')
    monkeypatch.setattr(quality.V,'frame',lambda clip,index:rgb)
    with pytest.raises(ValueError,match='81 frames'):
        quality.assess(tmp_path/'clip.mp4',tmp_path/'end.png')
    result=quality.assess(tmp_path/'clip.mp4',tmp_path/'end.png',expected_frames=73)
    assert result['end_correlation']==1 and not result['warnings']
