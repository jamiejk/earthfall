import pytest
from PIL import Image

from touchdown.scene_retreat_build import crop_portrait


@pytest.mark.parametrize('box', [(1, 2, 41, 182), (600, 180, 639, 359)])
def test_closeup_near_image_edge_keeps_subject_head_and_stays_in_bounds(box):
    image = Image.new('RGB', (640, 360))
    x0, y0, x1, y1 = box
    head = ((x0+x1)//2, y0+5)
    image.putpixel(head, (255, 0, 0))
    crop, rect = crop_portrait(image, box)
    assert 0 <= rect[0] < rect[2] <= image.width
    assert 0 <= rect[1] < rect[3] <= image.height
    assert rect[0] <= head[0] < rect[2]
    assert rect[1] <= head[1] < rect[3]
    assert crop.getpixel((head[0]-rect[0], head[1]-rect[1])) == (255, 0, 0)
    assert crop.width*9 == crop.height*16


def test_closeup_rejects_too_small_person_before_model_run():
    with pytest.raises(ValueError, match='cannot support'):
        crop_portrait(Image.new('RGB', (640, 360)), (300, 150, 310, 170))
