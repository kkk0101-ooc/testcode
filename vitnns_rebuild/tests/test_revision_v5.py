"""Independent native-forward and analytic controls for new neuron experiments."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from vitnns.revision_v5_extended import capture,suffix,unit_maps,grad,vjp,family_maps,checked,replace_many
from vitnns.revision_v5_scan import statistics

@pytest.fixture(scope='module')
def fixture():
 torch.set_num_threads(2);torch.manual_seed(419)
 model=VisionTransformer(img_size=224,patch_size=16,embed_dim=24,depth=12,num_heads=3,num_classes=1000).eval().double()
 for b in model.blocks:b.attn.fused_attn=False
 for p in model.parameters():p.requires_grad_(False)
 image=torch.randn(1,3,224,224,dtype=torch.double)
 return model,image

@pytest.mark.parametrize('kind',['post','mlp','family','head'])
def test_suffix_matches_native_replacement(fixture,kind):
 model,image=fixture;b=3;clean,logits,obs=capture(model,image,[b]);key='mlp' if kind=='family' else kind
 z=obs[b,key].detach();replacement=z.clone();replacement[:,1,2]+=.7
 expected=suffix(model,obs,b,kind,replacement)[0]
 module={'post':model.blocks[b],'mlp':model.blocks[b].mlp.act,'family':model.blocks[b].mlp.act,'head':model.blocks[b].attn.proj}[kind]
 if kind=='head':handle=module.register_forward_pre_hook(lambda _m,args:(replacement,))
 else:handle=module.register_forward_hook(lambda _m,args,y:replacement)
 try:
  with torch.no_grad():native=model(image)
 finally:handle.remove()
 torch.testing.assert_close(expected,native,atol=1e-10,rtol=1e-10)
 torch.testing.assert_close(suffix(model,obs,b,kind,z)[0],logits,atol=1e-10,rtol=1e-10)

def test_mlp_partition_and_independent_direction(fixture):
 model,image=fixture;b=3;clean,logits,obs=capture(model,image,[b]);z=obs[b,'mlp'];units=[(0,1),(17,9),(120,15)]
 maps=unit_maps(z,clean,units);g=grad(logits[0,12],z);parts=maps*torch.stack([g[0,t,c] for t,c in units])[:,None,None,None]
 remaining=g.clone()
 for t,c in units:remaining[0,t,c]=0
 branch=vjp(obs[b,'branch'],clean,grad(logits[0,12],obs[b,'post']))
 torch.testing.assert_close(parts.sum(0)+vjp(z,clean,remaining),branch,atol=1e-10,rtol=1e-9)
 channels=[1,8,19];fm=family_maps(z,clean,g,channels);remain=g.clone();remain[:,1:,channels]=0
 torch.testing.assert_close(fm.sum(0)+vjp(z,clean,remain),branch,atol=1e-10,rtol=1e-9)
 with pytest.raises(AssertionError):checked(parts.sum(0)+vjp(z,clean,remaining)+.01,branch)
 direction=torch.randn_like(image);eps=1e-5;values=[]
 for x in (image+eps*direction,image-eps*direction):
  _,_,o=capture(model,x,[b]);values.append(o[b,'mlp'][0,17,9].detach())
 analytic=(torch.autograd.grad(z[0,17,9],clean)[0]*direction).sum()
 torch.testing.assert_close((values[0]-values[1])/(2*eps),analytic,atol=1e-7,rtol=1e-5)

def test_zero_mass_and_rgb_cancellation():
 maps=torch.zeros(2,3,224,224);maps[1,0,0,0]=1;maps[1,1,0,0]=-1
 s=statistics(maps)
 assert np.isnan(s['entropy'][0]) and np.isnan(s['cancellation'][0])
 assert s['absolute_mass'][1]==2 and s['signed_sum'][1]==0 and s['cancellation'][1]==1
 assert s['patch_mass'][1,0]==2 and s['patch_signed'][1,0]==0
