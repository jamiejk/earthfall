import copy

import pytest

from touchdown.layout_recipe_trial import build_graph, validate_graph, FRAMES


def test_published_two_stage_recipe_keeps_both_guide_sets_and_original_clock():
    graph=build_graph(1399629552,'guide','first','last')
    validate_graph(graph,1399629552)
    assert FRAMES==97 and (FRAMES-1)%8==0
    assert graph['331']['inputs']['samples']==['395',2]
    assert graph['370']['inputs']['samples']==['365',2]


@pytest.mark.parametrize('node,key,value',[
    ('308','length',89),('308','height',736),('307','frame_rate',30),
    ('340','strength',.7),('341','image',['40',0]),('342','frame_idx',0),
    ('360','noise_seed',1399629553),('361','model',['300',0]),
    ('393','sigmas','1,0'),('370','samples',['395',2]),
])
def test_recipe_guard_rejects_silent_timing_or_conditioning_regressions(node,key,value):
    graph=copy.deepcopy(build_graph(1399629552,'guide','first','last'))
    graph[node]['inputs'][key]=value
    with pytest.raises(ValueError):validate_graph(graph,1399629552)
