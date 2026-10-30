"""Protect the bounded diagnostic's clock and persistent motion conditioning."""
import pytest
from touchdown.local_descent import graph, validate_graph


@pytest.mark.parametrize('change', ['clock','missing_motion','timed_pin','refine_seed','adapter'])
def test_recipe_rejects_lost_constraints(change):
    g=graph('guide.mp4','appearance.png')
    if change=='clock':g['308']['inputs']['length']=193
    elif change=='missing_motion':g['340']['inputs']['image']=['41',0]
    elif change=='timed_pin':g['999']={'class_type':'LTXVAddGuide','inputs':{}}
    elif change=='refine_seed':g['360']['inputs']['noise_seed']+=1
    elif change=='adapter':g['301']['inputs']['strength_model']=.5
    with pytest.raises(AssertionError):validate_graph(g)
