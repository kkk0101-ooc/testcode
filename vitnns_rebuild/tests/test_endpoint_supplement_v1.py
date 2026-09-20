import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer
from vitnns.endpoint_m4 import scan_endpoint_maps as original_scan, _summary
from vitnns.endpoint_supplement_v1 import scan_endpoint_maps, feature_distributions

def model(patch=224):
    torch.manual_seed(3920);torch.set_num_threads(2)
    m=VisionTransformer(img_size=224,patch_size=patch,embed_dim=12,depth=12,num_heads=3,mlp_ratio=2,num_classes=1000).eval()
    for b in m.blocks:b.attn.fused_attn=False
    return m

def test_margin_native_gradient_detach_and_original_parity():
    m=model();x=torch.randn(1,3,224,224)
    old=original_scan(m,x,class_id=17)
    new=scan_endpoint_maps(m,x,class_id=17,blocks=(0,5,11))
    for r in new['attention']:
        q=next(v for v in old['attention'] if (v['block'],v['head'])==(r['block'],r['head']))
        assert r['routing_share']==q['routing_share']
        assert r['frozen_attention_endpoint_error']==q['frozen_attention_endpoint_error']
    margin=scan_endpoint_maps(m,x,class_id=17,runnerup=9,blocks=(0,5,11),detach_validation=True)
    clean=x.clone().requires_grad_();z=m(clean);a=torch.autograd.grad(z[0,17],clean,retain_graph=True)[0];b=torch.autograd.grad(z[0,9],clean)[0]
    expected=_summary((clean.detach()*(a-b))[0])
    for k in ('absolute_mass','signed_sum','l2_norm'):
        assert np.isclose(margin['full_endpoint'][k],expected[k],atol=1e-6,rtol=1e-5)
    assert all(r['closure']['pass'] for r in margin['residual'])
    assert all(r['attention_detach_value_gradient']['pass'] for r in margin['attention_detach_validation'])
    assert [r['block'] for r in margin['block_validation']]==[0,5,11]

def test_feature_energy_native_and_negative_control_pairing():
    m=model(16);x=torch.randn(1,3,224,224)
    rows=feature_distributions(m,x,{str(b):[2,7] for b in range(12)},'test')
    captured={};hook=m.blocks[0].register_forward_hook(lambda _m,_a,o:captured.update(h=o))
    clean=x.clone().requires_grad_();m(clean);h=captured['h'][0,1:,2];energy=.5*h.square().sum();g=torch.autograd.grad(energy,clean)[0]
    mass=(clean.detach()*g).abs().double().reshape(3,14,16,14,16).sum((0,2,4)).flatten();p=(mass/mass.sum()).numpy()
    assert np.allclose(rows[0]['p'],p,rtol=2e-5,atol=1e-7)
    q=h.detach().double().square();assert np.allclose(rows[0]['q'],(q/q.sum()).numpy())
    assert len(rows)==24
    assert all(r['wrong_channel']!=r['channel'] and r['random_channel']!=r['channel'] for r in rows)
    for r in rows:
        assert np.allclose(sorted(r['q']),sorted(r['q_permuted']))
        for key in ('p','q','q_wrong_channel','q_random_channel','q_cross_layer'):assert np.isclose(sum(r[key]),1)
    hook.remove()

def test_random_auc_absolute_before_repeat_mean_cancellation():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/analyze_endpoint_supplement_baseline.py'
    spec=importlib.util.spec_from_file_location('baseline',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    positive=[(0.,0.),(.125,2.),(.25,4.)];negative=[(x,-y) for x,y in positive]
    corrected=np.mean([module.auc([(x,abs(y)) for x,y in curve]) for curve in (positive,negative)])
    legacy=module.auc([(x,abs((y+negative[i][1])/2)) for i,(x,y) in enumerate(positive)])
    assert corrected==2.0 and legacy==0.
