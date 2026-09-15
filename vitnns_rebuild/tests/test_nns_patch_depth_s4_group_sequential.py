from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from vitnns import nns_patch_depth_s4_group_sequential as gs
from vitnns import nns_patch_depth_s4_group_queue as queue
ROOT=Path(__file__).resolve().parents[1]
def test_look_manifests_are_exact_nested_class_rank_prefixes():
 previous=set()
 for look,c,n in ((1,200,400),(2,384,768),(3,512,1024),(4,640,1280),(5,872,1744)):
  d=json.loads((ROOT/gs.ROOT_REL/f'look_{look}_{c}_classes_{n}_images_manifest.json').read_text());rows=d['payload']['records'];ids={r['source_image_id'] for r in rows};assert len(rows)==n and len({r['annotation_wnid'] for r in rows})==c;assert sorted(set(r['formal_class_rank'] for r in rows))==list(range(c));assert not previous or previous<ids;previous=ids
def test_boundaries_use_incremental_spending_not_cumulative_raw_p():
 table=gs.critical_boundary_table();inc=[.025,.015,.005,.003,.002]
 from scipy.stats import norm
 for row in table['rows']:
  i=row['look']-1;m=row['intersection_size'];assert np.isclose(row['P1_P2_upper_z_boundary'],norm.isf(inc[i]/m));assert np.isclose(row['D1_upper_z_boundary'],norm.isf(inc[i]/(2*m)));assert row['cumulative_budget_is_not_ordinary_p_threshold']
def test_closed_testing_requires_all_containing_intersections():
 table=gs.critical_boundary_table();z={e:[10.0] for e in ('P1','P2','D1')};z['D1_LOWER']=[-10.0];result=gs.crossed_by_look(z,table,0);assert all(result['elementary_rejections'].values())
 z={'P1':[2.0],'P2':[-10.0],'D1':[-10.0],'D1_LOWER':[-10.0]};assert not gs.crossed_by_look(z,table,0)['elementary_rejections']['P1']
def test_repeated_adjusted_p_values_are_closed_and_bounded():
 z={'P1':[10.0],'P2':[10.0],'D1':[10.0],'D1_LOWER':[-10.0]};p=gs.repeated_adjusted_p_values(z,0);assert set(p)==set(gs.ENDPOINTS) and all(0<=v<=1 for v in p.values()) and all(v<.05 for v in p.values())
def test_group_queue_forbids_future_tranche(tmp_path):
 schedule=[{'schedule_index':0,'source_image_id':'a','formal_class_rank':0,'first_authorized_look':1},{'schedule_index':1,'source_image_id':'b','formal_class_rank':200,'first_authorized_look':2}];worker={'worker_id':'w','worker_pid':1,'physical_gpu_index':0,'gpu_uuid':'GPU'};claim=queue.claim_next(tmp_path/'launches/x',tmp_path,schedule,1,worker);assert claim['schedule_index']==0 and claim['look']==1;assert all(json.loads(p.read_text())['payload']['schedule_index']!=1 for p in tmp_path.glob('launches/x/claims/**/*.json'))
def test_preseal_access_audit_is_zero():
 d=json.loads((ROOT/gs.ROOT_REL/'phased_formal_access_audit_preseal.json').read_text())['payload'];assert d['efficacy_aggregate_access_count']==0 and d['D2_formal_access_count']==0 and d['S5_execution_count']==0
def test_bca_interval_is_deterministic_and_contains_mean():
 values=np.linspace(-.1,.2,40).reshape(20,2).tolist();a=gs.bca_mean_interval(values,'P1',1);b=gs.bca_mean_interval(values,'P1',1);assert a==b and a['lower']<a['estimate']<a['upper'] and a['replicates']==20000 and 'within_each_fixed_class' in a['resampling_unit']
def test_rebind_terminal_is_resolved_and_never_reclaims_nns(tmp_path):
 from vitnns import nns_patch_depth_s4 as s4
 root=tmp_path/'recovery';source=tmp_path/'source.json';source_doc=s4.signed('s4_unit_terminal','VALID_NNS',{'unit_type':'NNS_IMAGE','schedule_index':0});source.write_text(json.dumps(source_doc));binding=queue.rebind_path(root,'NNS_IMAGE',0);binding.parent.mkdir(parents=True,exist_ok=True);binding.write_text(json.dumps(gs.signed('s4_r4_completed_unit_rebind','PASS',{'source_terminal_path':str(source),'source_terminal_sha256':gs.sha_file(source)})));schedule=[{'schedule_index':0,'source_image_id':'a','formal_class_rank':0,'first_authorized_look':1}];worker={'worker_id':'w','worker_pid':1,'physical_gpu_index':0,'gpu_uuid':'GPU'};claim=queue.claim_next(root/'launches/x',root,schedule,1,worker);assert queue.resolved_terminal(root,'NNS_IMAGE',0)[2]=='REBIND';assert claim['unit_type']=='OCCLUSION_PRIMARY'
def test_fresh_launch_ignores_old_stale_claim(tmp_path):
 root=tmp_path/'stage';schedule=[{'schedule_index':0,'source_image_id':'a','formal_class_rank':0,'first_authorized_look':1}];worker={'worker_id':'w','worker_pid':1,'physical_gpu_index':0,'gpu_uuid':'GPU'};assert queue.claim_next(root/'launches/old',root,schedule,1,worker)['unit_type']=='NNS_IMAGE';assert queue.claim_next(root/'launches/new',root,schedule,1,worker)['unit_type']=='NNS_IMAGE'
