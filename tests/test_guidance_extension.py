import pytest
from touchdown.guidance_extension import graph,validate


def test_validation_preserves_noncopyable_sdk_assets_and_length():
    from threading import RLock
    guide,photo=RLock(),RLock()
    g=graph(guide,photo)
    assert g['50']['inputs']['file'] is guide
    assert g['40']['inputs']['image'] is photo
    assert g['308']['inputs']['length']==97


@pytest.mark.parametrize('change',['length','stage2_guide','pin','seed'])
def test_extension_keeps_original_clock_and_guides(change):
    g=graph('guide','photo')
    if change=='length':g['308']['inputs']['length']=49
    elif change=='stage2_guide':g['340']['inputs']['image']=['41',0]
    elif change=='pin':g['999']={'class_type':'LTXVAddGuide','inputs':{}}
    elif change=='seed':g['360']['inputs']['noise_seed']+=1
    with pytest.raises((ValueError,AssertionError)):validate(g)
