from pathlib import Path
import json,collections
paper=Path('papers/nns_vit_endpoint');reg=json.loads(Path('manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json').read_text()); rows=[]
for model,D,H in [('vit_b16',768,12),('deit_s16',384,6)]:
 targets=reg['targets_by_model'][model]
 predicates=[('post-block token-channel','12*197*D',12*197*D,lambda r:r['kind']=='token_channel' and r['module'].startswith('blocks.'),'12 blocks x 4 tokens x 4 channels','yes'),('post-block patch sum/energy','2*12*D',2*12*D,lambda r:r['kind'] in ['patch_channel_sum','patch_channel_energy'] and r['module'].count('.')==1,'12 blocks x 4 channels x 2 aggregates','no'),('attention-head coordinate','12*H*197*64',12*H*197*64,lambda r:r['kind']=='head_channel','all heads; CLS token; within-head channel 0','yes'),('module patch aggregate','3*(4D+2*4D)',36*D,lambda r:r['kind']=='patch_channel_sum' and r['module'].count('.')>1,'blocks 0/5/11; six modules; channel 0','no'),('patch embedding aggregate','D',D,lambda r:r['module']=='patch_embed','four fixed channels; sum over patches','no'),('final CLS coordinate','D',D,lambda r:r['module']=='norm','channel 0','yes'),('ground-truth logit','1000',1000,lambda r:r['kind']=='ground_truth_logit','frozen image label','output'),('clean-top1 logit','1000',1000,lambda r:r['kind']=='frozen_clean_top1_logit','frozen clean prediction','output')]
 for name,formula,pop,pred,strategy,neuron in predicates:
  count=sum(pred(r) for r in targets); rows.append(dict(model=model,object=name,population_formula=formula,population=pop,registered=count,coverage=count/pop,strategy=strategy,scalar_neuron=neuron))
 assert sum(r['registered'] for r in rows if r['model']==model)==len(targets)
Path('reports/nns_vit_endpoint_m1_r1_20260918/registry_coverage.json').write_text(json.dumps(rows,indent=2)+'\n')
header='| Model | Object | Candidate definition | Candidates | Registered/image | Coverage | Strategy | Scalar neuron |\n|---|---|---|---:|---:|---:|---|---|'
table=header+'\n'+'\n'.join(f"| {r['model']} | {r['object']} | {r['population_formula']} | {r['population']:,} | {r['registered']} | {100*r['coverage']:.5f}% | {r['strategy']} | {r['scalar_neuron']} |" for r in rows)
for lang in ['zh','en']:
 p=paper/lang/'full_draft.md';s=p.read_text();marker='### 4.2'
 if lang=='zh':
  intro='**Table 6. 原 registry 的实际覆盖率。** B=12，T=197（CLS + 196 patches），D=768/384，head width=64。候选总体按表中对象定义；module aggregate 总体固定为 blocks 0/5/11 的六个模块，每层四个 D 宽度模块和两个 4D 宽度模块。\n\n'
  tail='\n\n“完整扫描”仅指 registry-complete，即所有预注册条目均执行；post-block 坐标只覆盖 0.01058%/0.02115%，并非全部 ViT 神经元。head coordinate 是 CLS 的单一 head-channel 坐标，不代表该 head 全部 token-channel 坐标。ground-truth 与 clean-top1 是两个注册条目，在预测正确时对应相同 logit；条目数不等于不同标量函数数。Endpoint 是 normalized-input Gradient × Input；结构记账分解局部 Jacobian，不是函数值的因果分解。\n\n'
 else:
  intro='**Table 6. Actual coverage of the original registry.** B=12, T=197 (CLS + 196 patches), D=768/384, and head width=64. Candidate populations follow the object definitions below. Module aggregates use the six modules at blocks 0/5/11: four D-wide modules and two 4D-wide modules per selected block.\n\n'
  tail='\n\nA complete scan is registry-complete: every preregistered entry was executed. Post-block coordinates cover 0.01058%/0.02115% of their candidate populations. Head coordinates select one CLS head-channel scalar. Ground-truth and clean-top1 remain two registry entries but select the same logit on correctly classified images; registry-entry counts differ from distinct scalar-function counts. Endpoint is normalized-input Gradient × Input. Structural accounting decomposes the local Jacobian, rather than a causal decomposition of function values.\n\n'
 if '**Table 6.' in s:
  begin=s.index('**Table 6.');end=s.index(marker,begin);s=s[:begin]+s[end:]
 index=s.index(marker);s=s[:index]+intro+table+tail+s[index:];s=s.replace('成分级有限效应预测整体成立并保留一个边界例外','成分级有限效应关联呈现广泛但非普遍的 family-level 支持');p.write_text(s)
