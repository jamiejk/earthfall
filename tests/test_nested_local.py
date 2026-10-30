import copy

import numpy as np
import pytest
from PIL import Image

from touchdown.nested_local import expansion_geometry,outpaint_canvas,restore_native_centre,pair_graph,validate_pair_graph


def test_twofold_expansion_protects_native_centre_and_records_mapping():
    original=Image.fromarray(np.random.default_rng(7).integers(0,256,(32,64,3),dtype='uint8'))
    g=expansion_geometry(original.size);canvas=np.asarray(outpaint_canvas(original))
    assert g['working_centre']==[16,8,32,16]
    assert np.all(canvas[8:24,16:48,3]==255) and np.count_nonzero(canvas[:,:,3])==32*16
    full=restore_native_centre(original,Image.new('RGB',(64,32),(11,22,33)))
    assert full.size==(128,64)
    assert np.array_equal(np.asarray(full)[16:48,32:96],np.asarray(original))
    assert np.array_equal(np.asarray(full)[0,0],[11,22,33])


def test_plain_adjacent_pair_has_first_last_images_on_both_stages_without_viewport():
    g=pair_graph(3,2,17,'higher','wider');validate_pair_graph(g,17)
    assert sum(n['class_type']=='LTXVAddGuide' for n in g.values())==4


@pytest.mark.parametrize('node,key,value',[
    ('312','frame_idx',12),('343','image',['41',0]),('308','length',97),
    ('307','frame_rate',30),('360','noise_seed',18),('370','samples',['395',2]),
])
def test_pair_rejects_silent_reference_or_clock_changes(node,key,value):
    g=copy.deepcopy(pair_graph(2,1,17,'wide','close'));g[node]['inputs'][key]=value
    with pytest.raises(ValueError):validate_pair_graph(g,17)


def test_pair_rejects_viewport_conditioning():
    g=pair_graph(2,1,17,'wide','close');g['rogue']=dict(class_type='LoadVideo',inputs={})
    with pytest.raises(ValueError):validate_pair_graph(g,17)
