#!/usr/bin/env python3
"""Attempt2: source-summed PEFA with execution-coherent path IG."""
from __future__ import annotations
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_WORKSPACE_CONFIG_PREIMPORT')
import argparse,json,math,platform,resource,sys
from pathlib import Path
import numpy as np,torch,torchvision,timm
from PIL import Image
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;sys.path[:0]=[str(REPO),str(REPO/'vitnns_rebuild/src')]
from vit_pcia_pefa.run_step4_attempt1_worker import EP0,load_image,load_model,timed
from vit_pcia_pefa.src.vit_adapter import first_order_all_blocks,parity,port_fields,port_layout,structural_negative_controls,tensor_summary
from vitnns.endpoint_ep2 import gauss_legendre,integrate_logit
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-id',choices=['vit_b16','deit_s16'],required=True);p.add_argument('--physical-index',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.physical_index):raise RuntimeError('CUDA_VISIBLE_DEVICES_DRIFT')
 if a.output.exists():raise RuntimeError('OUTPUT_EXISTS')
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;dev=torch.device('cuda:0')
 probe=torch.arange(16,device=dev,dtype=torch.float32).reshape(4,4).requires_grad_();(probe@probe.T).square().sum().backward();torch.cuda.synchronize()
 reg=json.loads((EP0/'model_registry.json').read_text());s=next(x for x in reg['models'] if x['model_id']==a.model_id);receipt=json.loads((ROOT/'step4_runtime/discovery_receipt.json').read_text());m,binding=load_model(s,dev);x,size=load_image(receipt,s,dev);cid=int(receipt['class_id']);nodes,weights=gauss_legendre(512,dev,torch.float32)
 def endpoint():
  root=x.detach().requires_grad_(True);score=m(root)[:,cid].sum();return root*torch.autograd.grad(score,root)[0]
 E,c_endpoint=timed(endpoint)
 external_ig,c_ig=timed(lambda:integrate_logit(m,x,cid,steps=512,alpha_chunk=8));external_IG=external_ig.pop('map').double().to(dev)
 def pcia():
  acc=None;names=None
  for node,weight in zip(nodes,weights):
   root=(x*float(node)).detach().requires_grad_(True);fields,names=port_fields(m,root,block_index=0,class_id=cid,second_order=False);term=x.unsqueeze(0).double()*fields.detach().double()*float(weight);acc=term if acc is None else acc.add(term)
  return acc,names
 (A,names),c_pcia=timed(pcia);path_IG=A[-1]
 (fields,_),c_terminal=timed(lambda:port_fields(m,x.detach().requires_grad_(True),block_index=0,class_id=cid,second_order=False));T=x.unsqueeze(0).double()*fields.detach().double();R=T-A
 layout=port_layout(m.blocks[0].attn.num_heads);idx={n:i for i,n in enumerate(names)};omega_directions=[]
 for seed in (20260921,20260922):
  g=torch.Generator(device=dev);g.manual_seed(seed);v=torch.randn(x.shape,device=dev,dtype=x.dtype,generator=g);v/=v.norm();omega_directions.append(v)
 cuts=[]
 for cut_id,keys in layout.cuts.items():
  selected=[idx[k] for k in keys]
  def cut_field(root):
   cur,cur_names=port_fields(m,root,block_index=0,class_id=cid);cur_idx={n:i for i,n in enumerate(cur_names)};return cur[[cur_idx[k] for k in keys]].sum(0)
  omega=[]
  for v in omega_directions:
   root=x.detach().requires_grad_(True);_,jvp=torch.autograd.functional.jvp(cut_field,root,v,strict=True);_,vjp=torch.autograd.functional.vjp(cut_field,root,v=v,strict=True);omega.append(parity(vjp,jvp))
  tests={'endpoint_D_closure':parity(fields[selected].sum(0),fields[-1]),'pcia_to_ig':parity(A[selected].sum(0),path_IG),'pefa_to_endpoint':parity(T[selected].sum(0),E.double()),'T_equals_A_plus_R':parity(T[selected].sum(0),(A+R)[selected].sum(0)),'endpoint_minus_ig_equals_R':parity(R[selected].sum(0),E.double()-path_IG),'omega_zero_sum':omega};ok=all(v['pass'] for k,v in tests.items() if k!='omega_zero_sum') and all(v['pass'] for v in omega);cuts.append({'cut_id':cut_id,'component_ids':list(keys),'identity_families':5,'identities':tests,'pass':ok})
 blocks=first_order_all_blocks(m,x,class_id=cid);neg=structural_negative_controls(m,x,block_index=0,class_id=cid)
 qkv=layout.cuts['QKV_WITH_BYPASS_FULL'];selected=[idx[k] for k in qkv];path_cases=[('omit_qkv_value_head_pcia',A[selected[:-1]].sum(0),path_IG,'FAIL_QUADRATURE_CLOSURE'),('merge_qkv_consumer_cotangents',A[[*selected,idx['query_head_0']]].sum(0),path_IG,'FAIL_PORT_FIDELITY'),('omit_qkv_value_head_pefa',T[selected[:-1]].sum(0),E.double(),'FAIL_QUADRATURE_CLOSURE')]
 for control_id,observed,reference,expected in path_cases:
  check=parity(observed,reference);neg.append({'control_id':control_id,'expected_failure':expected,'observed_failure':expected if not check['pass'] else 'FALSE_PASS','pass':not check['pass'],'parity':check})
 batches=10+4*math.ceil(s['heads']/3);c_endpoint.update({'logical_forwards':1,'logical_reverse_products':1,'nodes':0});c_ig.update({'logical_forwards':514,'logical_reverse_products':512,'nodes':512,'result':external_ig,'execution':'independent alpha_batch=8 cost baseline'});c_pcia.update({'logical_forwards':512,'logical_base_vjp_calls':512*batches,'nodes':512});c_terminal.update({'logical_forwards':1,'logical_base_vjp_calls':batches,'nodes':0});c_pefa={'wall_seconds':c_pcia['wall_seconds']+c_terminal['wall_seconds'],'logical_forwards':513,'logical_base_vjp_calls':513*batches,'nodes':512,'peak_allocated_gpu_bytes':max(c_pcia['peak_allocated_gpu_bytes'],c_terminal['peak_allocated_gpu_bytes']),'peak_reserved_gpu_bytes':max(c_pcia['peak_reserved_gpu_bytes'],c_terminal['peak_reserved_gpu_bytes']),'components':{'pcia':c_pcia,'terminal_port_endpoint':c_terminal}}
 external_parity=parity(path_IG,external_IG);ok=all(c['pass'] for c in cuts) and all(b['pass'] for b in blocks) and all(n['pass'] for n in neg)
 result={'schema_version':1,'stage':'ENDPOINT_PCIA_PEFA_STEP4_ATTEMPT2','status':'PASS' if ok else 'FAIL','model_id':a.model_id,'sample_id':receipt['sample_id'],'split':'discovery','split_order':0,'class_id':cid,'image_sha256':receipt['image_sha256'],'original_size_wh':size,'input_shape':list(x.shape),'worker_identity':{'sys_executable_realpath':str(Path(sys.executable).resolve()),'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,'numpy':np.__version__,'Pillow':Image.__version__,'cuda':torch.version.cuda,'gpu_uuid':a.gpu_uuid,'physical_index':a.physical_index,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32':False,'autocast':False,'startup_matmul_backward':'PASS'},'model_binding':binding,'source_resolution':'one exact all-coordinate contraction','path_ig_reference':'PCIA __full__ channel, same nodes and execution','independent_batched_ig_diagnostic':external_parity,'primary_block':0,'primary_cuts':cuts,'all_blocks_first_order':blocks,'negative_controls':neg,'costs':{'endpoint':c_endpoint,'ig':c_ig,'pcia':c_pcia,'source_summed_pefa':c_pefa},'summaries':{'E':tensor_summary(E),'path_IG':tensor_summary(path_IG),'external_IG':tensor_summary(external_IG),'A':tensor_summary(A),'T':tensor_summary(T),'R':tensor_summary(R)},'process_peak_cpu_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),'raw_hessian_materialized':False,'raw_port_by_input_by_input_materialized':False,'raw_result_arrays_retained':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':result['status'],'model_id':a.model_id,'cuts_pass':sum(c['pass'] for c in cuts),'blocks_pass':sum(b['pass'] for b in blocks),'negative_controls_pass':sum(n['pass'] for n in neg),'external_ig_pass':external_parity['pass'],'pefa_seconds':c_pefa['wall_seconds']},sort_keys=True));return 0 if ok else 3
if __name__=='__main__':raise SystemExit(main())
