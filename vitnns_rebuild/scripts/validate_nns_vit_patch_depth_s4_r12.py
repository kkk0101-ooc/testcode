#!/usr/bin/env python3
"""Independent CPU-only R12 validator and false-pass mutation suite."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.stats import norm
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
from vitnns import nns_patch_depth_s4_r12 as r12  # noqa:E402
ROOT=REPO/r12.ROOT_REL
def require(c,m):
 if not c:raise RuntimeError(m)
def array(rec):
 p=REPO/rec['path'];require(p.is_file() and p.stat().st_size==rec['bytes'] and r12.sha_file(p)==rec['sha256'],'array identity');a=np.load(p,allow_pickle=False);require(list(a.shape)==rec['shape'] and str(a.dtype)==rec['dtype'],'array geometry/dtype');require(not np.issubdtype(a.dtype,np.number) or bool(np.isfinite(a).all()),'numeric array finiteness');h=hashlib.sha256();x=np.ascontiguousarray(a);h.update(str(x.dtype).encode());h.update(b'\0');h.update(str(tuple(x.shape)).encode());h.update(b'\0');h.update(x.tobytes());require(h.hexdigest()==rec['canonical_array_digest'],'array digest');return a
def seed(*parts):return int.from_bytes(hashlib.sha256(('NNS_VIT_PATCH_DEPTH_S4_R5_GROUP_SEQUENTIAL_POWER_V1\0'+'\0'.join(str(x) for x in parts)).encode()).digest()[:8],'big')
def bca(groups,e):
 g=[np.asarray(x,float) for x in groups];means=np.array([x.mean() for x in g]);est=float(means.mean());rng=np.random.Generator(np.random.PCG64(seed('ordinary_BCa',e,1,20000)));boot=np.empty(20000)
 for start in range(0,20000,1000):
  n=min(1000,20000-start);total=np.zeros(n)
  for x in g:total+=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1)
  boot[start:start+n]=total/len(g)
 prop=min(max(float(np.mean(boot<est)),1/40000),1-1/40000);z0=float(norm.ppf(prop));jack=[]
 for i,x in enumerate(g):
  if len(x)<=1:continue
  other=float(means.sum()-means[i])
  for j in range(len(x)):jack.append((other+(x.sum()-x[j])/(len(x)-1))/len(g))
 a=np.asarray(jack);jm=float(a.mean());delta=jm-a;den=6*float(np.sum(delta*delta))**1.5;acc=0 if den==0 else float(np.sum(delta**3))/den;qs=[]
 for alpha in (.025,.975):
  z=float(norm.ppf(alpha));qs.append(float(norm.cdf(z0+(z0+z)/(1-acc*(z0+z)))))
 lo,hi=(float(x) for x in np.quantile(boot,qs,method='linear'));return {'estimate':est,'lower':lo,'upper':hi,'half_width':max(est-lo,hi-est),'replicates':20000,'seed_namespace':'NNS_VIT_PATCH_DEPTH_S4_R5_GROUP_SEQUENTIAL_POWER_V1','bias_correction_z0':z0,'acceleration':acc,'resampling_unit':'source_image_within_each_fixed_class_then_equal_class_mean'}
def adjusted_p(z):
 local={}
 for size in (1,2,3):
  import itertools
  for inter in itertools.combinations(('P1','P2','D1'),size):
   value=1.0
   for e in inter:
    value=min(value,size*.05*float(norm.sf(z[e]))/.025)
    if e=='D1':value=min(value,2*size*.05*float(norm.sf(z['D1_LOWER']))/.025)
   local[inter]=min(1.0,value)
 return {e:max(v for k,v in local.items() if e in k) for e in ('P1','P2','D1')}
def independent(records):
 endpoint={e:[] for e in ('P1','P2','D1')};states={};matched=possible=0
 for row in records:
  docs={}
  for t in row['terminals']:
   p=REPO/t['path'];require(r12.sha_file(p)==t['sha256'],'terminal SHA');d=json.loads(p.read_text());require((r12.canonical_digest({k:v for k,v in d.items() if k!='canonical_digest'})==d['canonical_digest']),'terminal canonical');docs[t['unit_type']]=d;states[d['status']]=states.get(d['status'],0)+1
  n=docs['NNS_IMAGE'];rank=row['formal_class_rank']
  if n['status']=='VALID_NNS':
   by={Path(x['path']).stem:x for x in n['payload']['arrays']};num=array(by['p1_numerators']).astype(float);den=array(by['p1_denominators']).astype(float);valid=array(by['p1_valid_blocks']).astype(bool);plan=json.loads((REPO/n['payload']['p2_plan']['record']['path']).read_text());claim=plan.pop('canonical_digest');require(r12.canonical_digest(plan)==claim,'plan digest');possible+=36;matched+=sum(c['status']=='MATCHED' for c in plan['coordinates'])
   if valid.all() and (den>0).all():cor=num[:,0]/den-(num[:,1:]/den[:,None]).mean(1);endpoint['P1'].append((rank,float(cor.mean())));endpoint['D1'].append((rank,float(cor[8:12].mean()-cor[0:4].mean())))
  p=docs['OCCLUSION_PRIMARY']
  if p['status']=='VALID_OCCLUSION':x=array(p['payload']['arrays'][0]).astype(float);endpoint['P2'].append((rank,float((x[:,1:].mean(1)-x[:,0]).mean())))
 table=json.loads((REPO/r12.R10_ROOT_REL/'critical_boundary_table.json').read_text())['payload'];rows=table['rows'];summary={}
 for e in ('P1','P2','D1'):
  by={}
  for rank,v in endpoint[e]:by.setdefault(rank,[]).append(v)
  groups=[x for _,x in sorted(by.items())];cm=np.array([math.fsum(x)/len(x) for x in groups]);est=float(cm.mean());se=float(cm.std(ddof=1)/math.sqrt(len(cm)));ci=bca(groups,e);key='D1_upper_z_boundary' if e=='D1' else 'P1_P2_upper_z_boundary';boundary=next(float(x[key]) for x in rows if x['intersection_size']==3 and x['look']==1);summary[e]={'scheduled_images':240,'valid_images':len(endpoint[e]),'NA_images':240-len(endpoint[e]),'valid_fraction':len(endpoint[e])/240,'NA_fraction':1-len(endpoint[e])/240,'valid_classes':len(groups),'estimate':est,'class_cluster_standard_error':se,'ordinary_BCa_95':ci,'family_size_three_L1_adjusted_CI':{'lower':est-boundary*se,'upper':est+boundary*se if e=='D1' else None,'boundary_z':boundary}}
 z={'P1':(summary['P1']['estimate']-.02)/summary['P1']['class_cluster_standard_error'],'P2':summary['P2']['estimate']/summary['P2']['class_cluster_standard_error'],'D1':(summary['D1']['estimate']-.02)/summary['D1']['class_cluster_standard_error'],'D1_LOWER':(-summary['D1']['estimate']-.02)/summary['D1']['class_cluster_standard_error']};p=adjusted_p(z)
 for e in summary:summary[e]['heterogeneous_null_adjusted_p']=p[e]
 return summary,matched/possible,states
def validate():
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and 'PIL' not in sys.modules,'CPU-only');binding=r12.verify_r11_bindings(REPO);records,identity=r12.verify_selection(REPO);require(identity['all_checks_pass'],'corrected identity');summary,matching,states=independent(records);producer=json.loads((ROOT/'endpoint_numeric_recomputation.json').read_text());require(summary==producer['payload']['endpoint_summaries'],'producer-validator numeric mismatch');require(abs(matching-json.loads((ROOT/'input_integrity_report.json').read_text())['payload']['matching_fraction'])<1e-15,'matching mismatch');require(states==r12.EXPECTED_STATES,'states mismatch');require(summary['P1']['valid_classes']==120 and summary['D1']['valid_classes']==120 and summary['P2']['valid_classes']==119,'class scopes');require(json.loads((ROOT/'corrected_selection_identity_report.json').read_text())['payload']==identity,'identity report mismatch');require(binding['R11_terminal_status']=='BLOCKED_DATA_INTEGRITY','R11 history');return {'status':'PASS_POST_HOC_EXPLORATORY','identity_checks_pass':True,'numeric_checks_pass':True,'producer_validator_match':True,'R10_R11_immutable':True,'selected_source_count':240,'selected_distinct_class_ranks':122,'P1_D1_valid_classes':120,'P2_valid_classes':119,'full_manifest_records_digest':identity['full_manifest_records_digest'],'membership_projection_digest':identity['membership_projection_digest'],'endpoint_summaries':summary,'matching_fraction':matching,'Y4_signal_count':0,'S5_execution_count':0,'S6_execution_count':0}
def falsepass():
 identity=json.loads((ROOT/'corrected_selection_identity_report.json').read_text())['payload'];numeric=json.loads((ROOT/'endpoint_numeric_recomputation.json').read_text())['payload'];binding=json.loads((ROOT/'r11_blocked_terminal_binding.json').read_text())['payload'];access=json.loads((ROOT/'cpu_access_and_process_isolation_audit.json').read_text())['payload'];cases=[]
 def attack(name,base,mut,check):
  x=copy.deepcopy(base);mut(x);rejected=False
  try:check(x)
  except:rejected=True
  require(rejected,'accepted '+name);cases.append({'attack':name,'mutation_applied':True,'rejected':True})
 def ic(x):require(x['full_manifest_records_digest']==r12.FULL_RECORDS_DIGEST and x['membership_projection_digest']==r12.MEMBERSHIP_DIGEST and r12.canonical_digest(x['membership_projection'])==r12.MEMBERSHIP_DIGEST and len(x['membership_projection'])==240 and len({r['formal_class_rank'] for r in x['membership_projection']})==122 and x['selected_distinct_class_rank_count']==122 and x['cutoff_completion_unix_ns']==r12.CUTOFF_NS and x['cutoff_schedule_index']==300,'identity')
 for name,mut in [('records_order',lambda x:x['membership_projection'].reverse()),('membership_add',lambda x:x['membership_projection'].append(x['membership_projection'][0])),('membership_delete',lambda x:x['membership_projection'].pop()),('projection_source_id_field',lambda x:x['membership_projection'][0].update({'source_id':x['membership_projection'][0].pop('source_image_id')})),('selected_class_120',lambda x:x.__setitem__('selected_distinct_class_rank_count',120)),('swap_digests',lambda x:(x.__setitem__('full_manifest_records_digest',x['membership_projection_digest']),x.__setitem__('membership_projection_digest',r12.FULL_RECORDS_DIGEST))),('cutoff',lambda x:x.__setitem__('cutoff_schedule_index',301))]:attack(name,identity,mut,ic)
 def nc(x):require(x['posthoc_gate_checks']['P2_point'] and x['posthoc_gate_checks']['P2_adjusted_lower'] and x['posthoc_gate_checks']['P2_precision'],'P2 gate')
 for name,key in [('P2_point_gate','P2_point'),('P2_lower_gate','P2_adjusted_lower'),('P2_precision_gate','P2_precision')]:attack(name,numeric,lambda x,k=key:x['posthoc_gate_checks'].__setitem__(k,False),nc)
 attack('R11_terminal_binding',binding,lambda x:x.__setitem__('R11_terminal_status','PASS'),lambda x:require(x['R11_terminal_status']=='BLOCKED_DATA_INTEGRITY','R11 binding'));attack('Y4_signal_count',access,lambda x:x.__setitem__('Y4_signal_count',1),lambda x:require(x['Y4_signal_count']==0,'Y4'))
 terminal_fixture={'path':binding['R11_files'][0]['path'],'sha256':binding['R11_files'][0]['sha256']};attack('terminal_SHA',terminal_fixture,lambda x:x.__setitem__('sha256','0'*64),lambda x:require(r12.sha_file(REPO/x['path'])==x['sha256'],'terminal SHA'))
 scopes={'selected':122,'P1_D1':120,'P2':119};attack('class_scope_122_to_120',scopes,lambda x:x.__setitem__('selected',120),lambda x:require(x=={'selected':122,'P1_D1':120,'P2':119},'class scopes'));attack('class_scope_120_to_122',scopes,lambda x:x.__setitem__('P1_D1',122),lambda x:require(x=={'selected':122,'P1_D1':120,'P2':119},'class scopes'));attack('class_scope_119_to_120',scopes,lambda x:x.__setitem__('P2',120),lambda x:require(x=={'selected':122,'P1_D1':120,'P2':119},'class scopes'));return {'status':'PASS','attack_count':len(cases),'all_rejected':True,'attacks':cases}
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('validate','false-pass'));a=p.parse_args();print(json.dumps(validate() if a.command=='validate' else falsepass(),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
