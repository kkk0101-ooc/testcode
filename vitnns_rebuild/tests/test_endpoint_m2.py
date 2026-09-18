import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from vitnns.endpoint_m2_regions import slices, aggregate_map, rank, region_orders, auc, COUNTS
from vitnns.endpoint_m2_compute import clean_state, isolated_suffix
from vitnns.endpoint_e4_intervention import intervention_scores

def test_regions_scores_random_and_auc():
    for geometry,n in [('patch',14),('legacy_grid',4)]:
        masks=np.zeros((224,224),int)
        for ys,xs in slices(geometry): masks[ys,xs]+=1
        assert np.all(masks==1)
        a=np.ones((1,3,224,224)); a[:,:,0,:]=-1
        assert sum(aggregate_map(a,geometry))==3*224*224
        assert sum(aggregate_map(a,geometry,False))==3*222*224
        scores={m:list(range(n*n)) for m in ('endpoint_absolute','endpoint_signed',
            'regional_taylor','gradient_only','displacement_only','attention')}
        orders=region_orders(scores,geometry,'fixture')
        assert orders==region_orders(scores,geometry,'fixture') and len(orders)==27
        assert len({r['seed'] for r in orders if r['method']=='random'})==20
        for r in orders: assert sorted(r['order'])==list(range(n*n))
        counts=COUNTS[geometry]
        assert auc([c/(n*n) for c in counts],counts,geometry)==pytest.approx(counts[-1]/(2*n*n))
    assert rank([1,3,3,0])==[1,2,0,3]

@pytest.mark.parametrize('block',[2,5,10])
def test_suffix_preserves_clean_bypass_and_matches_native(block):
    torch.set_num_threads(2); torch.manual_seed(219)
    model=VisionTransformer(img_size=224,patch_size=112,embed_dim=12,depth=12,
        num_heads=3,mlp_ratio=2,num_classes=5).double().eval()
    for b in model.blocks: b.attn.fused_attn=False
    image=torch.randn(1,3,224,224,dtype=torch.float64)
    state=clean_state(model,image,block,2)
    assert state['closure_error']<1e-10
    for head in range(3):
        replacement=state['edge']*.75
        observed=isolated_suffix(model,state,replacement,block,head,2)[0]
        row=dict(component_id='r',block=block,head=head,num_heads=3,operation='head')
        _,native,_=intervention_scores(model,image,class_id=2,candidates=[row],
            replacements={'r':replacement[:,:,head*4:(head+1)*4]})
        assert observed==pytest.approx(native[0],abs=1e-10)
        assert isolated_suffix(model,state,state['edge'],block,head,2)[0]==pytest.approx(state['score'],abs=1e-10)
    # Check a component VJP using an independently finite-differenced scalar
    # with the clean downstream cotangent held fixed.
    direction=torch.randn_like(image); direction/=direction.norm(); eps=1e-4
    def scalar(x):
        observed={}
        def hook(_m,args): observed['edge']=args[0]
        handle=model.blocks[block].attn.proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad(): model(x)
            return (observed['edge'][:,:,:4]*state['gradient'][:,:,:4]).sum().item()
        finally: handle.remove()
    fd=(scalar(image+eps*direction)-scalar(image-eps*direction))/(2*eps)
    vjp=float((state['gradients'][0]*direction.numpy()).sum())
    assert fd==pytest.approx(vjp,abs=1e-9,rel=1e-5)

def test_complete_synthetic_scan_and_image_aggregation(monkeypatch):
    from vitnns.endpoint_m2_compute import scan
    from vitnns.endpoint_m2_regions import HEADS
    from vitnns.endpoint_m2_statistics import image_values
    monkeypatch.setitem(HEADS,'deit_s16',2)
    torch.set_num_threads(2);torch.manual_seed(291)
    model=VisionTransformer(img_size=224,patch_size=16,embed_dim=4,depth=12,
        num_heads=2,mlp_ratio=1,num_classes=5).float().eval()
    for b in model.blocks:b.attn.fused_attn=False
    image=torch.randn(1,3,224,224)
    result=scan(model,image,{'normalized_zero':torch.zeros_like(image),'blur_rgb':image*.8},
        dict(block=5,class_id=2,model_id='deit_s16',sample_id='synthetic'))
    result['model_id']='deit_s16'
    assert len(result['curves'])==2*2*2*27
    heads,block=image_values(result)
    assert len(block)==512 and len(heads)==1024
    random_curve=[r for r in result['curves'] if r['method']=='random' and r['geometry']=='patch' and r['reference']=='blur_rgb']
    expected=np.mean([r['auc']['full'] for r in random_curve])
    assert block[('patch','blur_rgb','random','full','auc')]==pytest.approx(expected)
    for row in result['curves']:
        for point in row['points']:
            for outcome in ('full','isolated'):
                assert point['regret'][outcome]>=-1e-12
                assert point['absolute'][outcome]==abs(point['signed'][outcome])

def test_inference_scope_and_image_pairing():
    from vitnns.endpoint_m2_regions import METHODS,HEADS,BLOCKS,GRIDS,REFERENCES
    from vitnns.endpoint_m2_statistics import summarize
    groups={}
    for model in HEADS:
        for block in BLOCKS:
            for geometry in GRIDS:
                for ref in REFERENCES:
                    for outcome in ('full','isolated'):
                        groups[(model,block,geometry,ref,outcome,'auc')]={
                            str(i):[2.]+[1.]*(len(METHODS)-1) for i in range(400)}
    summary,paired,tests=summarize(groups)
    assert len(tests)==672
    assert all(t['reject']==(t['direction']=='endpoint_greater') for t in tests)
    assert all(r['median']==1 and r['ci95']==[1,1] for r in paired)
