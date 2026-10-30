import numpy as np
from touchdown.retreat_join import fog_alpha


def test_mist_fully_hides_both_sides_of_the_scene_cut():
    noise=np.linspace(0,1,100,dtype=np.float32).reshape(10,10)
    cloud=np.full((10,10,3),245.,np.float32)
    for distance,scene in [(-.5,np.zeros_like(cloud)),(.5,np.full_like(cloud,255))]:
        alpha=fog_alpha(distance,noise)[:,:,None]
        composite=scene*(1-alpha)+cloud*alpha
        assert np.array_equal(composite,cloud)


def test_mist_does_not_alter_landing_outside_the_handoff():
    noise=np.full((10,10),.5,np.float32)
    assert not fog_alpha(-8,noise).any()
    assert not fog_alpha(8,noise).any()
    weights=[float(fog_alpha(d,noise).mean()) for d in [.5,1,2,4,6,8]]
    assert weights==sorted(weights,reverse=True)


def test_entry_coat_search_can_exclude_foreground_stonework():
    from touchdown.portrait_scene import extract_person
    image=np.full((600,1000,3),130,np.uint8)
    image[125:161,480:520]=(180,130,110);image[160:290,460:540]=(220,180,30)
    image[290:460,465:495]=(20,40,20);image[290:460,505:535]=(20,40,20)
    image[450:477,460:495]=(80,45,25);image[450:477,505:540]=(80,45,25)
    image[520:600,400:600]=(220,180,30)
    mask,box=extract_person(image,coat_max_y_fraction=.52)
    assert box[1]<=125 and box[3]>=476 and box[3]<520
    assert mask[475,475]
