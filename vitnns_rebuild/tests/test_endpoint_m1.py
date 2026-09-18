"""Independent skip-edge semantics and numerical fixtures for M1."""
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from vitnns.endpoint_m1_compute import capture, replace, missing_rows

def test_skip_only_replacement_and_gradient_match_manual_graph():
    torch.manual_seed(751)
    torch.set_num_threads(2)
    model = VisionTransformer(img_size=224,patch_size=224,embed_dim=12,depth=12,
                              num_heads=3,mlp_ratio=2,num_classes=1000).float().eval()
    for block in model.blocks:
        block.attn.fused_attn=False
    x=torch.randn(1,3,224,224); ref=x*0.3
    score,edges,grads=capture(model,x,7,grad=True)
    _,reference,_=capture(model,ref,7)
    assert score == pytest.approx(float(model(x)[0,7]),abs=1e-6)
    for edge in ('identity1','identity2'):
        row=next(r for r in missing_rows(3) if r['block']==5 and r['operation']==edge)
        baseline,replaced=replace(model,x,7,[row],{row['component_id']:reference[5][edge]})[row['component_id']]
        block=model.blocks[5]; original=block.forward
        # Manual counterfactual uses frozen reference skip and the clean sibling input.
        def manual(u):
            a=block.drop_path1(block.ls1(block.attn(block.norm1(u))))
            r=(reference[5][edge] if edge=='identity1' else u)+a
            m=block.drop_path2(block.ls2(block.mlp(block.norm2(r))))
            return (reference[5][edge] if edge=='identity2' else r)+m
        block.forward=manual
        try:
            expected=float(model(x)[0,7])
        finally:
            block.forward=original
        assert replaced == pytest.approx(expected,abs=2e-6)
        noop=replace(model,x,7,[row],{row['component_id']:edges[5][edge]})[row['component_id']]
        assert noop[0] == pytest.approx(noop[1],abs=2e-6)
        direction=torch.randn_like(edges[5][edge]); h=0.0003
        plus=replace(model,x,7,[row],{row['component_id']:edges[5][edge]+h*direction})[row['component_id']][1]
        minus=replace(model,x,7,[row],{row['component_id']:edges[5][edge]-h*direction})[row['component_id']][1]
        analytic=float((grads[5][edge]*direction).sum())
        assert (plus-minus)/(2*h) == pytest.approx(analytic,rel=0.03,abs=2e-4)

def test_missing_registry_is_exact():
    assert len(missing_rows(12,False))==24
    assert len(missing_rows(12,True))==48
    assert len({r['component_id'] for r in missing_rows(6,True)})==48
