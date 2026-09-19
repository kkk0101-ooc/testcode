#!/usr/bin/env python3
"""Six CPU display tasks: real activations, Endpoint, Grad-CAM and attention flow."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from contextlib import ExitStack

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from run_endpoint_e4_intervention_worker import load_model,load_inputs
from vitnns.endpoint_vit import capture_unfused_attention_edges

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+chr(10))

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    tasks=json.loads((REPO/'reports/nns_vit_endpoint_e4_intervention_r2_20260918/task_manifest.json').read_text())['tasks']
    specs={t['task_id']:t['payload'] for t in tasks};manifest=[]
    for model_id in ('deit_s16','vit_b16'):
        model,binding=load_model(specs[f'{model_id}__014'],torch.device('cpu'))
        if model_id=='vit_b16':model=model.double()
        for order in (4,14,15):
            taskid=f'{model_id}__{order:03d}';out=ROOT/'assets'/f'{taskid}.npz';meta=out.with_suffix('.json')
            if out.exists() and meta.exists():
                r=json.loads(meta.read_text());assert sha(out)==r['array_sha256'];manifest.append(r);print('reuse',taskid,flush=True);continue
            start=time.monotonic();spec=specs[taskid];x,_,_=load_inputs(spec,torch.device('cpu'));x=x.to(next(model.parameters()).dtype).detach().requires_grad_(True)
            observed={};edges=[{} for _ in model.blocks];handles=[]
            def hook(name):
                def fn(m,args,output):observed[name]=output
                return fn
            for b,block in enumerate(model.blocks):
                handles.append(block.register_forward_hook(hook(f'b{b}')))
                if b in (0,5,11):
                    for label,module in [('norm1',block.norm1),('attn',block.attn.proj),('norm2',block.norm2),('fc1',block.mlp.fc1),('gelu',block.mlp.act),('fc2',block.mlp.fc2)]:
                        handles.append(module.register_forward_hook(hook(f'b{b}.{label}')))
            handles.append(model.patch_embed.register_forward_hook(hook('patch')))
            with ExitStack() as stack:
                for block,e in zip(model.blocks,edges):stack.enter_context(capture_unfused_attention_edges(block.attn,e))
                logits=model(x);score=logits[0,spec['class_id']]
                gradient=torch.autograd.grad(score,x,retain_graph=True)[0]
                raw=x.detach()*gradient
                result={'input_rgb':np.rint(((x.detach()*torch.tensor(spec['std']).view(1,3,1,1))+torch.tensor(spec['mean']).view(1,3,1,1))[0].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8'),
                    'class_endpoint':raw[0].numpy(),'class_gradient':gradient[0].numpy()}
                att=[e['attention_softmax'][0].detach().double().mean(0) for e in edges]
                roll=torch.eye(197,dtype=torch.float64)
                for a in att:
                    a=a+torch.eye(197,dtype=torch.float64);a=a/a.sum(1,keepdim=True);roll=a@roll
                result['raw_attention']=att[-1][0,1:].reshape(14,14).numpy();result['rollout']=roll[0,1:].reshape(14,14).numpy()
                activation=observed['b11.norm1'];g=torch.autograd.grad(score,activation,retain_graph=True)[0]
                alpha=g[:,1:].mean(1);cam=(activation[:,1:]*alpha[:,None,:]).sum(-1).clamp_min(0)
                result['gradcam']=cam[0].reshape(14,14).detach().numpy()
                records=[]
                # All layers only for the shared main example; selected modules at B0/5/11.
                if order in (4,14):
                    for name,h in observed.items():
                        patch=h if name=='patch' else h[:,1:]
                        channel=int(patch.detach().var(dim=1,unbiased=False).mean(0).argmax())
                        target=.5*patch[:,:,channel].square().sum()
                        gx=torch.autograd.grad(target,x,retain_graph=True)[0]
                        result[name+'.energy.endpoint']=(x.detach()*gx)[0].numpy()
                        result[name+'.energy.feature']=patch[0,:,channel].detach().reshape(14,14).numpy()
                        records.append(dict(name=name,channel=channel,target='half_patch_channel_squared_sum',value=float(target.detach())))
                        if name=='patch' or (name.startswith('b') and '.' not in name):
                            target0=patch[:,:,0].sum();g0=torch.autograd.grad(target0,x,retain_graph=True)[0]
                            result[name+'.c0.endpoint']=(x.detach()*g0)[0].numpy();result[name+'.c0.feature']=patch[0,:,0].detach().reshape(14,14).numpy()
                for handle in handles:handle.remove()
                clean_score=float(score.detach());predicted=int(logits.argmax())
                # Source parity against the original retained Endpoint class map.
                e2=json.loads((REPO/f'reports/nns_vit_endpoint_e2_atlas_r1_20260918/formal_results/attempts/{taskid}/attempt-01.json').read_text())['result']
                ar=next(r for r in e2['artifacts'] if r['kind']=='full_raw_atlas');assert sha(REPO/ar['path'])==ar['sha256']
                atlas=np.load(REPO/ar['path'],mmap_mode='r');idx=next(i for i,r in enumerate(e2['summaries']) if r['target_id']=='class.gt')
                delta=np.asarray(result['class_endpoint'],float)-np.asarray(atlas[idx],float)
                rel=float(np.linalg.norm(delta)/np.linalg.norm(np.asarray(atlas[idx],float)))
                # CPU versus archived GPU is a precision diagnostic. Independently
                # verify this implementation against the unmodified native CPU graph.
                native_x=x.detach().clone().requires_grad_(True)
                for h in handles:h.remove()
            native_score=model(native_x)[0,spec['class_id']]
            native_map=(native_x*torch.autograd.grad(native_score,native_x)[0])[0].detach().numpy()
            native_error=float(np.linalg.norm(native_map-result['class_endpoint'])/np.linalg.norm(native_map))
            assert native_error<1e-7,('NATIVE_EXPLICIT_PARITY',taskid,native_error)
            # Save detached outputs; release graph before next task.
            assert all(np.isfinite(a).all() for a in result.values())
            np.savez_compressed(out,**result)
            row=dict(task_id=taskid,model_id=model_id,sample_id=spec['sample_id'],class_id=spec['class_id'],wnid=spec['wnid'],
                image_sha256=spec['image_sha256'],model_binding=binding,clean_logit=clean_score,predicted=predicted,
                class_endpoint_relative_l2_vs_sealed=rel,native_explicit_relative_l2=native_error,selected_channels=records,array_sha256=sha(out),seconds=time.monotonic()-start,
                array_keys={k:list(v.shape) for k,v in result.items()},device='cpu',dtype=str(next(model.parameters()).dtype),new_population_tests=0)
            write(meta,row);manifest.append(row);print('done',taskid,round(row['seconds'],2),rel,flush=True)
            del logits,score,observed,edges,result,x
        del model
    write(ROOT/'evidence/compute_receipt.json',dict(status='PASS_DISPLAY_COMPUTATION',tasks=manifest,task_count=6,cpu_only=True,method_comparison='qualitative_same_input',torch_version=torch.__version__))

if __name__=='__main__':main()
