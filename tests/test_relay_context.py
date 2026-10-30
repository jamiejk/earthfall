import pytest
from touchdown.relay_context import validate_partition


@pytest.mark.parametrize('ranges,count', [([(0,4),(5,8)],8), ([(0,4),(4,8)],10), ([(0,5),(4,8)],8)])
def test_unassigned_or_overlapping_connector_context_is_refused(ranges,count):
    with pytest.raises(ValueError):validate_partition(ranges,count,2)


def test_registers_remain_independent_and_receive_their_phase_mask():
    torch=pytest.importorskip('torch')
    from touchdown.relay_context import connect_phases
    from touchdown.relay_timing import temporal_mask
    calls=[]
    def connector(x,unprocessed):
        assert unprocessed
        calls.append(x.clone())
        # A mixing connector writes the complete input mean into every register.
        return x.mean(dim=1,keepdim=True).expand(-1,8,-1).clone()
    raw=torch.tensor([[[1.],[1.],[9.],[9.],[9.]]])
    out,ranges=connect_phases(raw,[2,3],connector)
    assert len(calls)==2 and ranges==[(0,8),(8,16)]
    assert torch.all(out[:,:8]==1) and torch.all(out[:,8:]==9)
    validate_partition(ranges,16,2)
    beats=[dict(start_frame=0,end_frame=24),dict(start_frame=24,end_frame=48)]
    mask=temporal_mask([0,48],1,ranges,beats,16,.1,torch.float32,'cpu')
    assert torch.all(mask[0,:8]==0) and torch.all(mask[0,8:]<-10)
    assert torch.all(mask[1,8:]==0) and torch.all(mask[1,:8]<-10)
    # Change only phase two: phase one's connector outputs must be identical.
    raw[:,2:]=99
    changed,_=connect_phases(raw,[2,3],connector)
    assert torch.equal(changed[:,:8],out[:,:8])


def test_each_phase_is_text_encoded_separately_with_global_motion():
    torch=pytest.importorskip('torch')
    from touchdown.relay_context import encode_phases
    class Clip:
        seen=[]
        def tokenize(self,text):self.seen.append(text);return text
        def encode_from_tokens_scheduled(self,text):
            return [[torch.ones(1,len(text),2),{'unprocessed_ltxav_embeds':True}]]
    clip=Clip();positive,lengths=encode_phases(clip,'motion',[{'prompt':'orbit'},{'prompt':'ground'}])
    assert clip.seen==['motion orbit','motion ground']
    assert positive[0][0].shape[1]==sum(lengths)


def test_recipe_fingerprints_include_live_adapter_and_review_policy():
    from touchdown.flight_runtime import recipe_hashes
    paths=recipe_hashes()
    for name in ['relay_node.py','relay_context.py','flight_contract.py','phase_review.py','flight-review-v1.json']:
        assert any(p.endswith('/'+name) for p in paths)
