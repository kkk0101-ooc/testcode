#!/usr/bin/env python3
"""Standard-library EP0 verifier. Reads only registered metadata/source/checkpoint hashes."""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, os, re, subprocess, sys
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'manifests/nns_vit_endpoint/ep0_r1_20260918'
REPORT=REPO/'reports/nns_vit_endpoint_ep0_r1_20260918'
BUNDLE=('access_contract.json','cohort_registry.json','evaluation_registry.json','implementation_contract.json','intervention_registry.json','literature_registry.json','model_registry.json','protocol.json','resource_contract.json','statistical_analysis_plan.json','target_registry.json')
NAMESPACE='NNS_VIT_ENDPOINT_EP0_R1_20260918'
PARENT='a0386ee9a1aea8f78fd8dfc39d8725d7639c8a63'

def sha(data):return hashlib.sha256(data).hexdigest()
def cd(value):return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def readj(p):return json.loads(p.read_text())
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def require(condition,name):
    if not condition:raise ValueError(name)

def closed(value,schema,path='bundle'):
    if 'const' in schema:
        require(type(value)==type(schema['const']) and value==schema['const'],'SCHEMA_CONST:'+path);return
    if schema.get('type')=='object':
        require(isinstance(value,dict) and schema.get('additionalProperties') is False,'SCHEMA_OBJECT:'+path)
        require(set(value)==set(schema['properties'])==set(schema['required']),'SCHEMA_KEYS:'+path)
        for k,v in value.items():closed(v,schema['properties'][k],path+'.'+k)
    elif schema.get('type')=='array':
        require(isinstance(value,list) and schema.get('items') is False,'SCHEMA_ARRAY:'+path)
        require(len(value)==schema['minItems']==schema['maxItems']==len(schema['prefixItems']),'SCHEMA_LENGTH:'+path)
        for i,(v,s) in enumerate(zip(value,schema['prefixItems'])):closed(v,s,path+f'[{i}]')
    else:raise ValueError('OPEN_SCHEMA:'+path)

def semantic(bundle):
    p,c,m,t,s,e,i,r,a=(bundle[x] for x in ['protocol.json','cohort_registry.json','model_registry.json','target_registry.json','statistical_analysis_plan.json','evaluation_registry.json','intervention_registry.json','resource_contract.json','access_contract.json'])
    require(p['status']=='EP0_PROTOCOL_FROZEN' and p['TBD_count']==0 and not p['production_experiments_authorized'] and not p['EP1_started'],'AUTHORITY_OR_PLACEHOLDER')
    require(a['authorized_stage']=='EP0_ONLY' and not a['production_authorized'],'EP0_ONLY')
    rows=c['rows'];counts=dict(Counter(x['split'] for x in rows))
    require(counts=={'discovery':64,'calibration':64,'evaluation':400} and c['counts']==counts,'COHORT_COUNTS')
    require(len(rows)==len({x['sample_id'] for x in rows})==len({x['class_id'] for x in rows})==528,'COHORT_DUPLICATE_OR_CLASS_LEAK')
    excluded=set(c['excluded_union']);sources=c['source_exclusions']
    require(excluded==set().union(*(set(v) for v in sources.values())) and len(excluded)==c['excluded_union_count'],'EXCLUSION_UNION')
    require(not excluded.intersection(x['sample_id'] for x in rows),'PRIOR_EXPOSURE_LEAK')
    require(not c['selected_image_bytes_opened'] and not c['selection_uses_outcomes'],'COHORT_ACCESS')
    require(all(c['legacy_endpoint_overlap'][k]==0 for k in counts),'LEGACY_OVERLAP')
    require(p['cohort']['rows_digest']==cd(rows),'COHORT_DIGEST')
    require({x['model_id'] for x in m['models']}=={'vit_b16','deit_s16'} and all(x['evaluation_n']==400 and x['depth']==12 for x in m['models']),'MODEL_SCOPE')
    require(not m['ep0_checkpoint_deserialized'],'CHECKPOINT_DESERIALIZED')
    require(t['target_count_by_model']=={'vit_b16':457,'deit_s16':385},'TARGET_COUNT')
    for mid,n in t['target_count_by_model'].items():
        units=t['targets_by_model'][mid];require(len(units)==len({u['target_id'] for u in units})==n,'TARGET_REGISTRY')
        require(sum(u['kind']=='patch_channel_energy' for u in units)==48,'ENERGY_OMISSION')
        require(set(range(12)).issubset({int(u['module'].split('.')[1]) for u in units if u['module'].startswith('blocks.')}),'BLOCK_OMISSION')
    cut={x['cut_id']:x for x in t['cuts']}
    require(cut['heads_branch']['reconstruction']=='A_F' and not cut['heads_branch']['bypass'],'HEAD_SCOPE')
    require(cut['heads_total']['reconstruction']=='L_F' and cut['heads_total']['bypass'],'BYPASS_OMISSION')
    require(cut['qkv_branch']['components']=='Q+K+V' and cut['routing_value_branch']['reconstruction']=='A_F','ROUTING_OMISSION')
    require('before RGB sum' in t['mass']['absolute'],'MASS_SCALE')
    require(len(s['families'])==16 and len(s['tests'])==28 and len({x['test_id'] for x in s['tests']})==28,'STAT_FAMILY')
    require(set(s['holm']['test_ids'])=={x['test_id'] for x in s['tests']} and s['holm']['alpha']==.05,'HOLM_SCOPE')
    require(s['bootstrap']['replicates']==10000 and s['bootstrap']['level']==.95,'BOOTSTRAP')
    for f in s['families']:
        require(f['n_valid_min']==360 and f['delta_min']>0 and f['ci_width_max']>0,'DECISION_THRESHOLDS')
        require(len(f['tests'])==(4 if f['claim']=='C4b' else 1),'MANDATORY_BASELINE_TESTS')
    require(e['fidelity']['replacements']==['normalized_zero','blur_rgb'],'REPLACEMENT_SELECTION')
    require(i['C4b']['status']=='ENABLED_CONFIRMATORY' and i['EP5A']['purpose']=='correctness_only','C4_CLAIM_CONFLATION')
    require(i['C4b']['mandatory_rankers']==['total_gxi','raw_gradient','regional_taylor','random_regions'],'BASELINE_OMISSION')
    require(e['ig']['candidate_nodes']==[32,64,128,256,512] and e['ig']['check_nodes']==[64,128,256,512,1024],'PATH_GRID')
    require(r['aggregate_gpu_hours']==sum(x['gpu_hours'] for x in r['stages'])==320,'GPU_BUDGET')
    require(r['max_retained_GiB']==sum(x['incremental_retained_GiB'] for x in r['stages'])==256,'STORAGE_BUDGET')
    acc=0
    for x in r['stages']:
        acc+=x['incremental_retained_GiB'];require(x['cumulative_retained_GiB']==acc,'STORAGE_CUMULATIVE')
    require(r['queue']['dynamic_global'] and not r['queue']['static_modulo_sharding'] and r['queue']['one_worker_per_gpu'],'QUEUE_CONTRACT')
    require(not r['current_stage_gpu_permission'],'GPU_PERMISSION')
    require(bundle['literature_registry.json']['positioning_decision']=='REPRODUCTION_UNIFIED_IMPLEMENTATION_AND_SYSTEMATIC_EMPIRICAL_AUDIT','NOVELTY_SCOPE')
    for v in bundle.values():
        txt=json.dumps(v)
        require('TBD_EP0' not in txt and 'PLACEHOLDER' not in txt,'UNRESOLVED_FREEZE')

def mutation_tests(bundle,schema):
    tests=[
      ('grant_production',lambda b:b['protocol.json'].__setitem__('production_experiments_authorized',True)),
      ('cohort_duplicate',lambda b:b['cohort_registry.json']['rows'].__setitem__(1,copy.deepcopy(b['cohort_registry.json']['rows'][0]))),
      ('class_target_top1_primary',lambda b:b['protocol.json']['targets'].__setitem__('primary','clean_top1')),
      ('mass_after_rgb',lambda b:b['target_registry.json']['mass'].__setitem__('absolute','abs after RGB sum')),
      ('head_missing_bypass',lambda b:b['target_registry.json']['cuts'][1].__setitem__('bypass',False)),
      ('remove_query',lambda b:b['target_registry.json']['cuts'][2].__setitem__('components','V')),
      ('ep5a_as_utility',lambda b:b['intervention_registry.json']['EP5A'].__setitem__('purpose','utility')),
      ('remove_taylor',lambda b:b['intervention_registry.json']['C4b']['mandatory_rankers'].remove('regional_taylor')),
      ('shorten_holm',lambda b:b['statistical_analysis_plan.json']['holm']['test_ids'].pop()),
      ('lower_effect',lambda b:b['statistical_analysis_plan.json']['families'][0].__setitem__('delta_min',0)),
      ('change_ci',lambda b:b['statistical_analysis_plan.json']['bootstrap'].__setitem__('level',.8)),
      ('drop_blur',lambda b:b['evaluation_registry.json']['fidelity']['replacements'].pop()),
      ('static_shard',lambda b:b['resource_contract.json']['queue'].__setitem__('static_modulo_sharding',True)),
      ('increase_cap',lambda b:b['resource_contract.json'].__setitem__('aggregate_gpu_hours',999)),
      ('open_key',lambda b:b['protocol.json'].__setitem__('payload',{})),
      ('defer_required',lambda b:b['protocol.json'].__setitem__('statistics','TBD_EP0'))]
    records=[]
    for name,mutate in tests:
        changed=copy.deepcopy(bundle);mutate(changed)
        try:semantic(changed);closed(changed,schema)
        except (ValueError,KeyError):records.append({'case':name,'rejected':True})
        else:raise ValueError('FALSE_PASS:'+name)
    return records

def verify_sources(bundle):
    src=readj(ROOT/'source_bindings.json');drift=[]
    expected_sources={
        'docs/NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md',
        'docs/NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md',
        'src/vitnns/endpoint_vit.py','src/vitnns/endpoint_nns.py','src/vitnns/modeling.py',
        'src/vitnns/labels.py','src/vitnns/constants.py','requirements.lock.txt',
        'config/stage_x_x2.json','scripts/run_endpoint_nns_vit_level4.py','scripts/build_endpoint_ep0.py',
        'assets/imagenet/imagenet_2012_validation_synset_labels.txt',
        'manifests/stage_c_sample_registry.csv','manifests/stage_d_fidelity_v1/registry.json',
        'manifests/stage_x_prior_access_ledger_v1.json',
        'evidence/stage_y/y2_clean_formal_20260910/formal/cohort_registry.json',
        'reports/nns_vit_endpoint_level4_r1_20img_20260917/schedule.json',
        'assets/models/vit_base_patch16_224_weights.semantic_candidate.pth',
        'assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth',
        '../相关文章/Understanding neural network through neuron level visualization/en/00-front-matter.md',
        '../相关文章/Understanding neural network through neuron level visualization/en/03-proposed-method.md',
        '../相关文章/Understanding neural network through neuron level visualization/en/04-experiments.md',
    }
    require(len(src['files'])==22 and {x['path'] for x in src['files']}==expected_sources,'SOURCE_READ_WHITELIST')
    for item in src['files']:
        path=(REPO/item['path']).resolve()
        # No raw image, mask or array read is permitted by this verifier.
        require(path.suffix.lower() not in {'.jpeg','.jpg','.png','.npy','.npz','.tar','.gz'},'FORBIDDEN_SOURCE_SUFFIX')
        require(path.is_file(),'SOURCE_MISSING:'+item['path'])
        require(file_sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'SOURCE_HASH:'+item['path'])
    require(not any(src[x] for x in ['torch_imported','timm_imported','PIL_imported']),'PRODUCER_IMPORT')
    require(src['formal_image_payload_reads']==src['new_model_calls']==0 and src['cuda_visible_devices']=='','ACCESS_RECEIPT')
    c=bundle['cohort_registry.json']
    source_specs=[('stage_c','manifests/stage_c_sample_registry.csv'),('stage_d_e_f','manifests/stage_d_fidelity_v1/registry.json'),('stage_x_prior','manifests/stage_x_prior_access_ledger_v1.json'),('stage_y','evidence/stage_y/y2_clean_formal_20260910/formal/cohort_registry.json'),('legacy_endpoint_20','reports/nns_vit_endpoint_level4_r1_20img_20260917/schedule.json')]
    for key,relative in source_specs:
        extracted=sorted(set(re.findall(r'ILSVRC2012_val_\d{8}\.JPEG',(REPO/relative).read_text())))
        require(extracted==c['source_exclusions'][key],'EXCLUSION_SOURCE_RECOMPUTE:'+key)
    local_names=sorted(p.name for p in (REPO/'assets/data/ILSVRC2012_val').glob('ILSVRC2012_val_*.JPEG'))
    # Additional future materialization is allowed; the 1000 originally excluded names remain frozen.
    require(set(c['source_exclusions']['all_preexisting_local_materialized_val']).issubset(local_names),'LOCAL_FILENAME_SNAPSHOT')
    # Reconstruct the cohort from the bound label text and exclusion metadata.
    label_path=REPO/'assets/imagenet/imagenet_2012_validation_synset_labels.txt'
    labels=label_path.read_text().splitlines();require(len(labels)==50000,'LABEL_COUNT')
    wnids=sorted(set(labels));require(wnids==c['class_mapping'],'LABEL_MAPPING')
    require(sha(('\n'.join(wnids)+'\n').encode())=='70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15','LABEL_ORDER_HASH')
    excluded=set(c['excluded_union']);candidates={w:[] for w in wnids}
    for n,w in enumerate(labels,1):
        f=f'ILSVRC2012_val_{n:08d}.JPEG'
        if f not in excluded:candidates[w].append(f)
    eligible=[w for w in wnids if candidates[w]];require(len(eligible)==c['eligible_class_count'],'ELIGIBLE_CLASSES')
    classes=sorted(eligible,key=lambda w:(sha((NAMESPACE+'|class|'+w).encode()),w))[:528]
    for n,(w,row) in enumerate(zip(classes,c['rows'])):
        expected=min(candidates[w],key=lambda f:(sha((NAMESPACE+'|image|'+f).encode()),f))
        split='discovery' if n<64 else 'calibration' if n<128 else 'evaluation'
        require(row['sample_id']==expected and row['wnid']==w and row['class_id']==wnids.index(w) and row['split']==split,'COHORT_RECOMPUTE')
        require(row['payload_hash_state']=='DEFERRED_UNTIL_AUTHORIZED_MATERIALIZATION','EARLY_PAYLOAD')
    require(subprocess.run(['git','merge-base','--is-ancestor',PARENT,'HEAD'],cwd=REPO).returncode==0,'PARENT_ANCESTRY')
    return {'bound_sources':len(src['files']),'cohort_recomputed':528,'checkpoint_hashes_verified':2,'new_image_reads':0,'model_deserialization':False,'GPU_used':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');args=p.parse_args()
    bundle={name:readj(ROOT/name) for name in BUNDLE};schema=readj(ROOT/'protocol_bundle.schema.json')
    semantic(bundle);closed(bundle,schema);sources=verify_sources(bundle);attacks=mutation_tests(bundle,schema)
    checklist=readj(ROOT/'ep0_checklist.json')
    require(set(checklist)=={'stage','planning_checklist_count','resolutions','ep1_execution_status','authority_scope'},'CHECKLIST_SCHEMA')
    require(checklist['planning_checklist_count']==len(checklist['resolutions'])==17 and checklist['ep1_execution_status']=='NOT_STARTED','CHECKLIST_COUNT')
    for item in checklist['resolutions']:
        require(set(item)=={'item','file','key','status'} and item['status']=='FROZEN','CHECKLIST_ENTRY')
        require(item['key'] in bundle[item['file']],'CHECKLIST_RESOLUTION')
    environment=readj(ROOT/'environment_contract.json')
    require(set(environment)=={'python','versions','mode','CUDA_VISIBLE_DEVICES','runtime_experiment_identity'},'ENVIRONMENT_SCHEMA')
    require(environment['versions']=={'torch':'2.2.2+cu121','timm':'0.9.16','torchvision':'0.17.2+cu121','numpy':'1.26.4','Pillow':'10.2.0'} and environment['CUDA_VISIBLE_DEVICES']=='','ENVIRONMENT_IDENTITY')
    # Ensure package code itself imports no accelerator or image/model libraries.
    for file in [Path(__file__),REPO/'scripts/build_endpoint_ep0.py']:
        tree=ast.parse(file.read_text())
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        require(not any(x.split('.')[0] in {'torch','timm','PIL','numpy','scipy'} for x in names),'FORBIDDEN_IMPORT')
    summary={'status':'PASS_EP0_PROTOCOL_FREEZE','schema_closed':True,'source_read_whitelist':True,'checklist_resolutions':17,'semantic_checks':'PASS','mutation_tests':attacks,'sources':sources,'cohort_counts':bundle['cohort_registry.json']['counts'],'target_counts':bundle['target_registry.json']['target_count_by_model'],'test_count':28,'family_count':16,'production_authorized':False,'EP1_executed':False,'next_authorization':'NONE; EP1 requires explicit user request','protocol_digest':cd(bundle['protocol.json'])}
    if args.seal:
        require(not REPORT.exists(),'REPORT_EXISTS_USE_FRESH_ROOT')
        REPORT.mkdir(parents=True)
        files=[{'path':q.relative_to(REPO).as_posix(),'sha256':file_sha(q),'bytes':q.stat().st_size} for q in sorted(ROOT.rglob('*')) if q.is_file()]
        for q in [Path(__file__),REPO/'scripts/build_endpoint_ep0.py']:
            files.append({'path':q.relative_to(REPO).as_posix(),'sha256':file_sha(q),'bytes':q.stat().st_size})
        for name in ['docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md','docs/NNS_ViT_Endpoint_Scanner_EP0相关工作审计_20260918.md']:
            q=REPO/name;files.append({'path':name,'sha256':file_sha(q),'bytes':q.stat().st_size})
        for name,obj in [('validation_report.json',summary),('artifact_manifest.json',{'files':files,'canonical_digest':cd(files)}),('terminal_receipt.json',{'status':'EP0_PROTOCOL_FROZEN','scientific_results':'NOT_RUN','production_experiments_authorized':False,'EP1_started':False,'parent_commit':PARENT,'next_step':'EP1_IMPLEMENTATION_AND_REGRESSION','next_authorization':'REQUIRES_EXPLICIT_USER_REQUEST','validation_digest':cd(summary),'manifest_digest':cd(files)})]:
            (REPORT/name).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    elif REPORT.exists():
        manifest=readj(REPORT/'artifact_manifest.json');require(manifest['canonical_digest']==cd(manifest['files']),'MANIFEST_DIGEST')
        for item in manifest['files']:
            q=REPO/item['path'];require(file_sha(q)==item['sha256'] and q.stat().st_size==item['bytes'],'ARTIFACT_HASH:'+item['path'])
        terminal=readj(REPORT/'terminal_receipt.json')
        require(terminal['validation_digest']==cd(readj(REPORT/'validation_report.json')),'TERMINAL_BINDING')
        require(terminal['manifest_digest']==cd(manifest['files']) and not terminal['production_experiments_authorized'],'TERMINAL_AUTHORITY')
        require(readj(REPORT/'validation_report.json')==summary,'REPORT_RECOMPUTE')
    print(json.dumps({k:v for k,v in summary.items() if k!='mutation_tests'},ensure_ascii=False))
    print('negative mutations rejected:',len(attacks))
if __name__=='__main__':main()
