"""V5 MLP cuts, isolated neuron replacement, and independent-checkpoint scans."""
import hashlib,math
import numpy as np
import torch
from .revision_v5_scan import statistics,neuron_scan
from .endpoint_m3_atlas import endpoint_chunk

METHODS=['endpoint','activation','gradient','activation_gradient','displacement','taylor']+[f'random_{i:02d}' for i in range(20)]+['activation_matched','high_non_target_endpoint','random_non_target_endpoint']
def seed(*parts):return int.from_bytes(hashlib.sha256(('NNS_VIT_REVISION_V5_20260921_R1|'+'|'.join(map(str,parts))).encode()).digest()[:8],'big')
def store(out,prefix,values):
 for k,v in values.items():out[prefix+k]=v

def capture(model,image,blocks):
 obs={};handles=[]
 for b in blocks:
  block=model.blocks[b]
  for key,module,pre in [('post',block,False),('mlp',block.mlp.act,False),('head',block.attn.proj,True),('qkv',block.attn.qkv,False),('u',block.norm1,True),('r1',block.norm2,True),('branch',block.drop_path2,False),('attn',block.drop_path1,False)]:
   if pre:
    def hook(_m,args,key=(b,key)):obs[key]=args[0]
    handles.append(module.register_forward_pre_hook(hook))
   else:
    def hook(_m,args,y,key=(b,key)):obs[key]=y
    handles.append(module.register_forward_hook(hook))
 clean=image.detach().clone().requires_grad_(True)
 try:logits=model(clean)
 finally:
  for h in handles:h.remove()
 return clean,logits,obs

def vjp(tensor,clean,g):
 return (clean.detach()*torch.autograd.grad(tensor,clean,grad_outputs=g.detach(),retain_graph=True)[0])[0].detach()
def grad(score,tensor):return torch.autograd.grad(score,tensor,retain_graph=True)[0].detach()
def unit_maps(tensor,clean,units):
 return torch.cat([endpoint_chunk(tensor,clean,tuple(units[i:i+8])).detach() for i in range(0,len(units),8)])
def family_maps(tensor,clean,g,channels):
 maps=[]
 for c in channels:
  s=torch.zeros_like(g);s[:,1:,c]=g[:,1:,c];maps.append(vjp(tensor,clean,s))
 return torch.stack(maps)
def checked(a,b):
 torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
 return float((a-b).abs().max())

def structure(model,clean,logits,obs,blocks,class_id,out):
 checks=[];calls=0
 full=vjp(logits,clean,torch.nn.functional.one_hot(torch.tensor([class_id],device=clean.device),1000).to(clean.dtype));calls+=1
 store(out,'total_',statistics(full[None]))
 for b in blocks:
  o=lambda k:obs[b,k]
  gs={k:grad(logits[0,class_id],o(k)) for k in ('post','r1','head','qkv')};calls+=4
  maps={}
  for name,tensor,g in [('identity',o('u'),gs['r1']),('attention',o('attn'),gs['r1']),('carry',o('r1'),gs['post']),('mlp_branch',o('branch'),gs['post'])]:
   maps[name]=vjp(tensor,clean,g);calls+=1
  for i,k in enumerate(('Q','K','V')):
   s=torch.zeros_like(gs['qkv']).reshape(1,197,3,-1);s[:,:,i]=gs['qkv'].reshape_as(s)[:,:,i]
   maps[k]=vjp(o('qkv'),clean,s.reshape_as(gs['qkv']));calls+=1
  maps['routing']=maps['Q']+maps['K']
  width=model.embed_dim//model.blocks[b].attn.num_heads;hm=[]
  for h in range(model.blocks[b].attn.num_heads):
   s=torch.zeros_like(gs['head']);s[:,:,h*width:(h+1)*width]=gs['head'][:,:,h*width:(h+1)*width]
   hm.append(vjp(o('head'),clean,s));calls+=1
  checks.extend([checked(maps['identity']+maps['attention'],full),checked(maps['carry']+maps['mlp_branch'],full),checked(maps['Q']+maps['K']+maps['V'],maps['attention']),checked(torch.stack(hm).sum(0),maps['attention'])])
  for k,m in maps.items():store(out,f'b{b:02d}_{k}_',statistics(m[None]))
  store(out,f'b{b:02d}_heads_',statistics(torch.stack(hm)))
  out[f'b{b:02d}_head_activation_gradient']=(gs['head']*o('head').detach()).reshape(1,197,-1,width).sum((0,1,3)).abs().cpu().numpy()
 return checks,calls

def mlp_study(model,image,targets,item):
 blocks=range(12);clean,logits,obs=capture(model,image,blocks);out={};closures=[];backward=0
 for b in blocks:
  tensor=obs[b,'mlp'];units=[(u['token'],u['channel']) for u in targets[str(b)]['mlp']];channels=targets[str(b)]['family_channels']
  own=unit_maps(tensor,clean,units);g=grad(logits[0,item['class_id']],tensor);backward+=5
  weights=torch.stack([g[0,t,c] for t,c in units]);parts=own*weights[:,None,None,None]
  store(out,f'b{b:02d}_mlp_own_',statistics(own));store(out,f'b{b:02d}_mlp_scalar_',statistics(parts))
  store(out,f'b{b:02d}_mlp_family_',statistics(family_maps(tensor,clean,g,channels)));backward+=len(channels)
  downstream=grad(logits[0,item['class_id']],obs[b,'post']);branch=vjp(obs[b,'branch'],clean,downstream);backward+=2
  remaining=g.clone()
  for t,c in units:remaining[0,t,c]=0
  rem=vjp(tensor,clean,remaining);backward+=1
  closures.append(checked(parts.sum(0)+rem,branch))
  selected=torch.zeros_like(g);selected[:,1:,channels]=g[:,1:,channels]
  family_sum=vjp(tensor,clean,selected);remainder=vjp(tensor,clean,g-selected);backward+=2
  closures.append(checked(family_sum+remainder,branch))
  store(out,f'b{b:02d}_mlp_remainder_',statistics(rem[None]));store(out,f'b{b:02d}_mlp_branch_',statistics(branch[None]))
  out[f'b{b:02d}_mlp_activation']=np.asarray([float(tensor[0,t,c].detach()) for t,c in units])
  out[f'b{b:02d}_mlp_units']=np.asarray(units);out[f'b{b:02d}_family_channels']=np.asarray(channels)
 c,n=structure(model,clean,logits,obs,blocks,item['class_id'],out);closures+=c;backward+=n
 out['logits']=logits.detach().cpu().numpy()[0]
 return out,dict(forward_calls=1,backward_calls=backward,closure_max=max(closures),closure_checks=len(closures))

@torch.no_grad()
def suffix(model,obs,b,kind,replacements,batch_size=32):
 outputs=[];batch_calls=0
 for start in range(0,len(replacements),batch_size):
  z=replacements[start:start+batch_size];n=len(z)
  if kind=='post':x=z
  elif kind in ('mlp','family'):x=obs[b,'r1'].detach().expand(n,-1,-1)+model.blocks[b].mlp.drop2(model.blocks[b].mlp.fc2(z))
  elif kind=='head':
   cur=model.blocks[b];x=obs[b,'u'].detach().expand(n,-1,-1)+cur.attn.proj_drop(cur.attn.proj(z));x=x+cur.mlp(cur.norm2(x))
  else:raise ValueError(kind)
  for block in model.blocks[b+1:]:x=block(x)
  outputs.append(model.head(model.fc_norm(model.norm(x)[:,0])).detach());batch_calls+=1
 return torch.cat(outputs),batch_calls

def response(logits,gt,high,random):
 other=logits.clone();other[:,gt]=-torch.inf
 return torch.stack((logits[:,gt],logits[:,high],logits[:,random],logits[:,gt]-other.max(1).values),1)

def scalar_scores(tensor,g,reference,units,own,kind,channels=None,clean=None):
 if kind!='family':
  act=torch.stack([tensor[0,t,c].detach() for t,c in units]);gr=torch.stack([g[0,t,c] for t,c in units]);delta=torch.stack([tensor[0,t,c].detach()-reference[0,t,c] for t,c in units])
  m=statistics(own)['absolute_mass']*gr.abs().cpu().numpy()
  return np.stack((m,act.abs().cpu().numpy(),gr.abs().cpu().numpy(),(act*gr).abs().cpu().numpy(),delta.abs().cpu().numpy(),(delta*gr).abs().cpu().numpy())),(delta*gr).cpu().numpy()
 a=tensor.detach()[0,1:,channels];gr=g[0,1:,channels];d=a-reference[0,1:,channels]
 maps=family_maps(tensor,clean,g,channels)
 return np.stack((statistics(maps)['absolute_mass'],a.norm(dim=0).cpu().numpy(),gr.norm(dim=0).cpu().numpy(),(a*gr).sum(0).abs().cpu().numpy(),d.norm(dim=0).cpu().numpy(),(d*gr).sum(0).abs().cpu().numpy())),(d*gr).sum(0).cpu().numpy()

def replace_many(tensor,reference,selections,units,kind,channels,alphas):
 out=tensor.detach().expand(len(selections),-1,-1).clone()
 for i,(selection,alpha) in enumerate(zip(selections,alphas)):
  for j in selection:
   if kind=='family':
    c=channels[j];out[i,1:,c]=tensor.detach()[0,1:,c]+alpha*(reference[0,1:,c]-tensor.detach()[0,1:,c])
   else:
    t,c=units[j];out[i,t,c]=tensor.detach()[0,t,c]+alpha*(reference[0,t,c]-tensor.detach()[0,t,c])
 return out

def intervention_study(model,image,refs,targets,item):
 blocks=[0,3,7,11];clean,logits,obs=capture(model,image,blocks);out={};backward=0;forward=1;suffix_calls=0;suffix_examples=0;controls=[]
 gt=item['class_id'];other=logits.detach()[0].clone();other[gt]=-torch.inf;high=int(other.argmax());random=min((i for i in range(1000) if i not in (gt,high)),key=lambda i:seed('non_target',item['sample_id'],i));classes=[gt,high,random]
 base=response(logits.detach(),gt,high,random)[0];out['classes']=np.asarray(classes);out['clean_response']=base.cpu().numpy();out['methods']=np.asarray(METHODS)
 refobs={}
 for name,x in refs.items():
  _,_,o=capture(model,x,blocks);refobs[name]={k:v.detach() for k,v in o.items()};forward+=1
 for b in blocks:
  for kind in ('post','mlp','family'):
   tensor=obs[b,'mlp' if kind=='family' else kind];units=[(u['token'],u['channel']) for u in targets[str(b)]['post' if kind=='post' else 'mlp']];channels=targets[str(b)]['family_channels'];n=len(channels) if kind=='family' else len(units)
   own=None if kind=='family' else unit_maps(tensor,clean,units);backward+=0 if kind=='family' else 4
   gradients=[grad(logits[0,c],tensor) for c in classes];backward+=3
   control,nb=suffix(model,obs,b,kind,tensor.detach());suffix_calls+=nb;suffix_examples+=1;controls.append(checked(control,logits.detach()))
   for name in refs:
    ref=refobs[name][b,'mlp' if kind=='family' else kind];scores=[];predictions=[]
    for g in gradients:
     s,p=scalar_scores(tensor,g,ref,units,own,kind,channels,clean);scores.append(s);predictions.append(p);backward+=len(channels) if kind=='family' else 0
    prefix=f'b{b:02d}_{kind}_{name}_';out[prefix+'scores']=np.asarray(scores);out[prefix+'taylor_signed']=np.asarray(predictions)
    selections=[[j] for a in (.1,1.) for j in range(n)];alphas=[a for a in (.1,1.) for j in range(n)]
    replacements=replace_many(tensor,ref,selections,units,kind,channels,alphas)
    y,nb=suffix(model,obs,b,kind,replacements);suffix_calls+=nb;suffix_examples+=len(replacements)
    out[prefix+'single_effect']=(base[None]-response(y,gt,high,random)).cpu().numpy().reshape(2,n,4)
    orders=[np.argsort(-s,kind='stable') for s in scores[0]]
    rng=np.random.default_rng(seed(item['sample_id'],b,kind,name,'random'))
    orders.extend([rng.permutation(n) for _ in range(20)])
    selected=list(orders[0][:min(8,n)]);matched=[];available=set(range(n))-set(selected)
    # With only eight family candidates, match disjointly at each budget below.
    for j in selected:
     if available:
      k=min(available,key=lambda k:(abs(math.log(max(scores[0][1,j],1e-30))-math.log(max(scores[0][1,k],1e-30))),k));matched.append(k);available.remove(k)
    orders.extend([np.asarray(matched,dtype=int),np.argsort(-scores[1][0],kind='stable'),np.argsort(-scores[2][0],kind='stable')])
    top=[];match_distance=[]
    for mi,order in enumerate(orders):
     for budget in (1,2,4,8):
      k=min(budget,n)
      if mi==26:
       chosen=list(orders[0][:k]);available=set(range(n))-set(chosen);control=[];dist=[]
       for j in chosen:
        # At k > n/2, a disjoint equal-budget set cannot exist: report NA, not reuse selected units.
        if not available:break
        q=min(available,key=lambda q:(abs(math.log(max(scores[0][1,j],1e-30))-math.log(max(scores[0][1,q],1e-30))),q));control.append(q);available.remove(q);dist.append(abs(math.log(max(scores[0][1,j],1e-30))-math.log(max(scores[0][1,q],1e-30))))
       top.append(control if len(control)==k else []);match_distance.append(np.mean(dist) if len(control)==k else np.nan)
      else:top.append(list(order[:k]))
    repl=replace_many(tensor,ref,top,units,kind,channels,[1.]*len(top));y,nb=suffix(model,obs,b,kind,repl);suffix_calls+=nb;suffix_examples+=len(repl)
    effect=(base[None]-response(y,gt,high,random)).cpu().numpy().reshape(len(METHODS),4,4)
    for bi,v in enumerate(match_distance):
     if not np.isfinite(v):effect[26,bi]=np.nan
    out[prefix+'topk_effect']=effect;out[prefix+'match_log_activation_distance']=np.asarray(match_distance)
    out[prefix+'topk_indices']=np.asarray([s+[-1]*(8-len(s)) for s in top],dtype=np.int16).reshape(len(METHODS),4,8)
 return out,dict(forward_calls=forward,backward_calls=backward,suffix_forward_batches=suffix_calls,suffix_forward_examples=suffix_examples,clean_suffix_max=max(controls),controls=len(controls))

def external_study(model,image,refs,targets,item):
 blocks=[0,3,7,11];out,checks=neuron_scan(model,image,targets,blocks);clean,logits,obs=capture(model,image,blocks)
 closures,n=structure(model,clean,logits,obs,blocks,item['class_id'],out);checks['forward_calls']+=1;checks['backward_calls']+=n
 checks.update(closure_max=max(closures),closure_checks=len(closures));calls=0;examples=0
 gt=item['class_id'];base=logits.detach()[0,gt]
 for name,x in refs.items():
  _,_,ro=capture(model,x,blocks);checks['forward_calls']+=1
  for b in blocks:
   tensor=obs[b,'head'];ref=ro[b,'head'].detach();heads=model.blocks[b].attn.num_heads;w=model.embed_dim//heads
   replacements=tensor.detach().expand(heads,-1,-1).clone()
   for h in range(heads):replacements[h,:,h*w:(h+1)*w]=ref[0,:,h*w:(h+1)*w]
   y,nb=suffix(model,obs,b,'head',replacements);calls+=nb;examples+=heads
   out[f'b{b:02d}_{name}_head_effect']=(base-y[:,gt]).cpu().numpy()
 checks.update(suffix_forward_batches=calls,suffix_forward_examples=examples)
 return out,checks

def extended_scan(kind,model,image,refs,targets,item):
 return {'e2':lambda:mlp_study(model,image,targets,item),'e3':lambda:intervention_study(model,image,refs,targets,item),'e5':lambda:external_study(model,image,refs,targets,item)}[kind]()
