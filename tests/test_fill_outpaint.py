"""Guard the distinction between real Fill conditioning and the rejected generic edit."""
import copy
import json

import pytest

from touchdown.fill_outpaint import API, WORKFLOW, validate


@pytest.mark.parametrize('variant', ['', '-preserve-source', '-hard-protect'])
def test_ui_and_api_connect_identical_ports(variant):
    api = API.with_name('flux-fill-outpaint' + variant + '.api.json')
    workflow = WORKFLOW.with_name('flux-fill-outpaint' + variant + '.json')
    graph = json.loads(api.read_text())
    ui = json.loads(workflow.read_text())
    base_nodes = set(json.loads(API.read_text()))
    hard = variant == '-hard-protect'
    validate(graph if hard else {k: v for k, v in graph.items() if k in base_nodes}, hard)
    nodes = {n['id']: n for n in ui['nodes']}
    assert set(map(int, graph)) == set(nodes)
    for _, source, slot, target, target_slot, kind in ui['links']:
        name = nodes[target]['inputs'][target_slot]['name']
        assert graph[str(target)]['inputs'][name] == [str(source), slot]
    assert nodes[3]['widgets_values'] == [1399629552, 'fixed', 20, 1, 'euler', 'normal', 1]
    assert nodes[38]['widgets_values'] == [hard]
    assert nodes[31]['widgets_values'] == ['flux1-fill-dev.safetensors', 'fp8_e4m3fn']
    if variant:
        assert graph['50']['inputs'] == dict(destination=['49', 0], source=['17', 0],
                                             x=960, y=544, resize_source=False)
        assert graph['51']['inputs']['images'] == ['50', 0]
        assert graph['9']['inputs']['images'] == ['8', 0]


@pytest.mark.parametrize('mutation', ['generic_model', 'missing_mask', 'composited_output'])
def test_rejects_the_previous_shortcuts(mutation):
    g = copy.deepcopy(json.loads(API.read_text()))
    if mutation == 'generic_model':
        g['31']['inputs']['unet_name'] = 'flux3-image-klein-bf16.safetensors'
    elif mutation == 'missing_mask':
        g['38']['inputs'].pop('mask')
    else:
        g['9']['inputs']['images'] = ['paste', 0]
        g['paste'] = dict(class_type='ImageCompositeMasked', inputs={})
    with pytest.raises(ValueError):
        validate(g)
