#!/usr/bin/env python3
"""EP0 only: freeze metadata and public literature; never decode images or load models."""
from __future__ import annotations
import argparse, ast, csv, hashlib, html, json, math, os, re, subprocess, sys
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'manifests/nns_vit_endpoint/ep0_r1_20260918'
PARENT = 'a0386ee9a1aea8f78fd8dfc39d8725d7639c8a63'
PLAN = 'docs/NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md'
NAMESPACE = 'NNS_VIT_ENDPOINT_EP0_R1_20260918'
READS = {}


def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return digest(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
def read(path):
    path=Path(path);data=path.read_bytes()
    READS[os.path.relpath(path,REPO)]={'path':os.path.relpath(path,REPO),'sha256':digest(data),'bytes':len(data),'mode':'metadata_or_source_text'}
    return data

def readj(relative): return json.loads(read(REPO/relative))
def streamhash(relative):
    path=REPO/relative;h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    row={'path':relative,'sha256':h.hexdigest(),'bytes':path.stat().st_size,'mode':'checkpoint_stream_hash_only_no_deserialization'}
    READS[relative]=row;return row

def write(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')

def image_ids(obj): return set(re.findall(r'ILSVRC2012_val_\d{8}\.JPEG',json.dumps(obj)))

def make_cohort():
    labels=read(REPO/'assets/imagenet/imagenet_2012_validation_synset_labels.txt').decode().splitlines()
    assert len(labels)==50000 and len(set(labels))==1000 and set(Counter(labels).values())=={50}
    wnids=sorted(set(labels));assert digest(('\n'.join(wnids)+'\n').encode())=='70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15'
    sources={}
    local=sorted(p.name for p in (REPO/'assets/data/ILSVRC2012_val').glob('ILSVRC2012_val_*.JPEG'))
    assert len(local)==1000
    sources['all_preexisting_local_materialized_val']=local
    for key,path in [
        ('stage_c','manifests/stage_c_sample_registry.csv'),
        ('stage_d_e_f','manifests/stage_d_fidelity_v1/registry.json'),
        ('stage_x_prior','manifests/stage_x_prior_access_ledger_v1.json'),
        ('stage_y','evidence/stage_y/y2_clean_formal_20260910/formal/cohort_registry.json'),
        ('legacy_endpoint_20','reports/nns_vit_endpoint_level4_r1_20img_20260917/schedule.json')]:
        sources[key]=sorted(set(re.findall(r'ILSVRC2012_val_\d{8}\.JPEG',read(REPO/path).decode())))
    sources['scanner_demo']=['ILSVRC2012_val_00007922.JPEG']
    excluded=set().union(*(set(v) for v in sources.values()))
    candidates={w:[] for w in wnids}
    for i,w in enumerate(labels,1):
        filename=f'ILSVRC2012_val_{i:08d}.JPEG'
        if filename not in excluded:candidates[w].append(filename)
    eligible=[w for w in wnids if candidates[w]]
    assert len(eligible)>=528
    classes=sorted(eligible,key=lambda w:(digest((NAMESPACE+'|class|'+w).encode()),w))[:528]
    rows=[]
    for i,w in enumerate(classes):
        filename=min(candidates[w],key=lambda f:(digest((NAMESPACE+'|image|'+f).encode()),f))
        split='discovery' if i<64 else 'calibration' if i<128 else 'evaluation'
        order=i if i<64 else i-64 if i<128 else i-128
        rows.append({'sample_id':filename,'source_object_id':'ILSVRC2012_VAL_OFFICIAL:'+filename,'class_id':wnids.index(w),'wnid':w,'split':split,'split_order':order,'class_selection_sha256':digest((NAMESPACE+'|class|'+w).encode()),'image_selection_sha256':digest((NAMESPACE+'|image|'+filename).encode()),'payload_hash_state':'DEFERRED_UNTIL_AUTHORIZED_MATERIALIZATION','payload_source':'assets/imagenet/ILSVRC2012_img_val.tar::'+filename})
    archive=REPO/'assets/imagenet/ILSVRC2012_img_val.tar'
    assert archive.is_file()
    return {'release':'ILSVRC2012_VAL_OFFICIAL','namespace':NAMESPACE,'algorithm':'retain WNIDs with at least one nonexcluded filename, rank by SHA256(namespace|class|WNID), first528; within class rank nonexcluded filename by SHA256(namespace|image|filename); first64 discovery, next64 calibration, remaining400 evaluation','eligible_class_count':len(eligible),'class_mapping':wnids,'official_label_file_sha256':READS['assets/imagenet/imagenet_2012_validation_synset_labels.txt']['sha256'],'counts':{'discovery':64,'calibration':64,'evaluation':400},'models_share_exact_rows':True,'source_exclusions':sources,'excluded_union':sorted(excluded),'excluded_union_count':len(excluded),'legacy_endpoint_overlap':{s:0 for s in ['discovery','calibration','evaluation']},'previous_exposure_scope':'all local 1000 plus named historical metadata; no claim of globally never-seen ImageNet','selection_uses_outcomes':False,'selected_image_bytes_opened':False,'archive_container':{'path':'assets/imagenet/ILSVRC2012_img_val.tar','stat_bytes':archive.stat().st_size,'read_mode':'stat_only','image_payload_verification':'authorized later, no automatic replacement if absent'},'rows':rows}

def make_models():
    definitions=[('vit_b16','vit_base_patch16_224.augreg2_in21k_ft_in1k','assets/models/vit_base_patch16_224_weights.semantic_candidate.pth',768,12,[.5]*3,[.5]*3,'489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d'),('deit_s16','deit_small_patch16_224.fb_in1k','assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth',384,6,[.485,.456,.406],[.229,.224,.225],'cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f')]
    rows=[]
    for mid,name,path,dim,heads,mean,std,expected in definitions:
        bound=streamhash(path);assert bound['sha256']==expected
        rows.append({'model_id':mid,'timm_name':name,'checkpoint':bound,'role':'primary' if mid=='vit_b16' else 'secondary_replication','depth':12,'embedding_dim':dim,'heads':heads,'head_dim':64,'patch_size':16,'prefix_tokens':1,'patch_tokens':196,'evaluation_n':400,'discovery_n':64,'calibration_n':64,'preprocessing':{'decode':'Pillow 10.2.0 RGB, explicit no EXIF transpose','resize':'torchvision 0.17.2 Resize(248,bicubic,antialias=True), shorter side; center crop224','crop':[224,224],'input_shape':[1,3,224,224],'mean':mean,'std':std,'unit_rgb_scale':255},'load_contract':'EP1 only after new authorization; strict keys and architecture, float32 weights; double on CPU fixtures','backend':'timm0.9.16 standard Block/Attention unfused eval','extra_input_dependent_bias':False,'constant_seed':'none; parameters including CLS position and biases fixed','tensor_digest':'EP1 recomputes after authorized deserialization; file SHA256 is EP0 byte identity'})
    return {'models':rows,'ep0_checkpoint_deserialized':False,'cross_model_scope':'two registered standard blocks; no all-ViT claim','non_registered_architectures':'not supported by this protocol'}

def make_targets(models):
    out={}
    for m in models['models']:
        dim=m['embedding_dim'];channels=[0,dim//4,dim//2,dim-1];units=[]
        def add(i,module,kind,token=-1,channel=-1,head=-1,location='output'):
            units.append({'target_id':i,'module':module,'kind':kind,'token':token,'channel':channel,'head':head,'location':location})
        for block in range(12):
            for t in [0,1,98,196]:
                for c in channels:add(f'b{block}.t{t}.c{c}',f'blocks.{block}','token_channel',t,c)
            for c in channels:
                for kind in ['patch_channel_sum','patch_channel_energy']:add(f'b{block}.{kind}.c{c}',f'blocks.{block}',kind,channel=c)
            for h in range(m['heads']):add(f'b{block}.head{h}.cls.c0',f'blocks.{block}.attn.proj','head_channel',0,0,h,'input')
        for block in [0,5,11]:
            for module in ['norm1','attn.proj','norm2','mlp.fc1','mlp.act','mlp.fc2']:
                add(f'b{block}.{module}.sum.c0',f'blocks.{block}.{module}','patch_channel_sum',channel=0)
        for c in channels:add(f'patch.sum.c{c}','patch_embed','patch_channel_sum',channel=c)
        add('final.cls.c0','norm','token_channel',0,0)
        add('class.gt','','ground_truth_logit')
        add('class.clean_top1','','frozen_clean_top1_logit')
        out[m['model_id']]=units
    cuts=[
      {'cut_id':'heads_branch','components':'head_0..H-1','reconstruction':'A_F','bypass':False},
      {'cut_id':'heads_total','components':'identity + head_0..H-1','reconstruction':'L_F','bypass':True},
      {'cut_id':'qkv_branch','components':'Q+K+V','reconstruction':'A_F','bypass':False},
      {'cut_id':'routing_value_branch','components':'routing+V','reconstruction':'A_F','bypass':False},
      {'cut_id':'qkv_total','components':'identity+Q+K+V','reconstruction':'L_F','bypass':True},
      {'cut_id':'routing_value_total','components':'identity+routing+V','reconstruction':'L_F','bypass':True},
      {'cut_id':'residual1','components':'identity+attention','reconstruction':'L_F','bypass':True},
      {'cut_id':'residual2','components':'carry+MLP','reconstruction':'L_F','bypass':True}]
    return {'main_formula':'E(s,x)=x*grad_x(s)','input_coordinates':'normalized RGB [c,h,w]','reference_parameter_in_canonical_endpoint':False,'delta_extension':'(x-x0)*grad, secondary and separately named','path_identity':'IG on same scalar target; full Jacobian at each alpha','primary_class':'ground_truth raw logit; all correctness retained','secondary_class':'clean top1, lowest class-index argmax tie, frozen across conditions','margin':'NOT_PLANNED','head_direction':'fixed within-head channel0, not learned per sample','energy':'0.5*sum over patch tokens of same post-block channel squared; CLS excluded','target_count_by_model':{k:len(v) for k,v in out.items()},'targets_by_model':out,'cuts':cuts,'cut_blocks':list(range(12)),'cut_target':'ground_truth_logit','cut_scopes':{'A_F':'attention input-edge component using fixed downstream cotangent for F','L_F':'complete input endpoint of F only if all input paths pass registered add'},'hook_edges':['QKV projection output split before attention reshape','row-wise softmax output','AV output before concat','attn.proj input','attn.proj output','drop_path1 output','norm1 input identity edge copy','norm2 input first add output and carry edge copy','drop_path2 output','block output second add'],'backend_contract':'explicit unfused; no attention/projection dropout, stochastic depth, checkpointing; unsupported extra branches fail','mass':{'signed':'fsum(E[c,h,w])','absolute':'fsum(abs(E[c,h,w])) before RGB sum or pooling','signed_pixel':'sum_c E','absolute_pixel':'sum_c abs(E)','absolute_cell':'sum over pixels and channels of abs(E), then rank','display_only':True}}

LITERATURE=[
 ('ig','https://proceedings.mlr.press/v70/sundararajan17a.html','Axiomatic Attribution for Deep Networks','Sundararajan, Taly, Yan',2017,'ICML, PMLR70:3319-3328','Abstract and bibliographic record',['Abstract'],'IG is established; our endpoint replaces the path average, not a new IG formulation'),
 ('conductance','https://arxiv.org/abs/1805.12233','How Important Is a Neuron?','Dhamdhere, Sundararajan, Yan',2018,'arXiv:1805.12233','Abstract',['Abstract:'],'Hidden-unit conductance and its ablation evaluation pre-exist; input-resolved path objects are not claimed new'),
 ('neuron_gradient','https://raw.githubusercontent.com/meta-pytorch/captum/413e888e1e8f22cf8da1631cc9ca2133ce30a9db/captum/attr/_core/neuron/neuron_gradient.py','Captum NeuronGradient','Captum contributors',2026,'source snapshot 413e888e1e8f22cf8da1631cc9ca2133ce30a9db','class NeuronGradient; attribute selector',['class NeuronGradient','a callable'],'Internal scalar/sum selectors already exist; our delta is ViT registry and explicit reconstruction contracts'),
 ('input_x_gradient','https://captum.ai/api/input_x_gradient.html','Captum InputXGradient','Captum contributors',2026,'official API snapshot','class InputXGradient',['A baseline approach'],'Input times gradient is an established baseline'),
 ('neuron_conductance','https://captum.ai/api/neuron.html','Captum neuron attribution API','Captum contributors',2026,'official API snapshot','Neuron Conductance',['Computes conductance with respect'],'API returns input-shaped conductance for hidden neurons; input resolution alone is not novelty'),
 ('chefer','https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html','Transformer Interpretability Beyond Attention Visualization','Chefer, Gur, Wolf',2021,'CVPR:782-791','Abstract',['Abstract'],'Attention and skip conservation already studied; relevance rules differ from the true endpoint Jacobian'),
 ('attnlrp','https://proceedings.mlr.press/v235/achtibat24a.html','AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers','Achtibat et al.',2024,'ICML, PMLR235:135-168','Abstract',['Abstract'],'Input and latent transformer attribution already studied; custom relevance propagation differs from GxI'),
 ('rollout','https://aclanthology.org/2020.acl-main.385/','Quantifying Attention Flow in Transformers','Abnar, Zuidema',2020,'ACL:4190-4197','Abstract',['In the Transformer model'],'Attention rollout/flow is established; class-agnostic attention mixing does not equal full QKV Jacobian'),
 ('atp','https://arxiv.org/html/2403.00745v1','AtP*: An efficient and scalable method for localizing LLM behaviour to components','Kramar, Lieberum, Shah, Nanda',2024,'arXiv:2403.00745v1','2.2 eq4-5,3.1.1,C.2',['first-order Taylor expansion','False negatives from attention saturation','C.2 Edge AtP'],'Activation-displacement Taylor, QK saturation fixes and edge patching already exist; p_ref is mandatory prior-art baseline'),
 ('patching_practice','https://arxiv.org/html/2309.16042v2','Towards Best Practices of Activation Patching in Language Models: Metrics and Methods','Zhang, Nanda',2024,'arXiv:2309.16042v2','Abstract; Sections3-4',['we systematically examine'],'Corruption and metric choices can change localization; freeze replacements and metrics'),
 ('libragrad','https://openaccess.thecvf.com/content/CVPR2025/html/Mehri_LibraGrad_Balancing_Gradient_Flow_for_Universally_Better_Vision_Transformer_Attributions_CVPR_2025_paper.html','LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions','Mehri, Baghshah, Pilehvar',2025,'CVPR:67-78','Abstract',['gradient flow imbalances'],'Transformer gradient path analysis pre-exists; backward pruning/scaling differs from true Jacobian')]

def fetch_literature(row):
    key,url,title,authors,year,venue,location,terms,delta=row
    last=None
    for attempt in range(2):
        try:
            raw=urlopen(Request(url,headers={'User-Agent':'NNS-EP0-metadata-audit/1'}),timeout=25).read();break
        except Exception as exc:last=exc
    else:raise RuntimeError(f'{key}: {last}')
    src=raw.decode(errors='replace')
    if '<html' in src.lower() or '<!doctype' in src.lower():
        src=re.sub(r'<script.*?</script>|<style.*?</style>',' ',src,flags=re.S)
        src=html.unescape(re.sub('<[^>]+>',' ',src))
    text=re.sub(r'\s+',' ',src)
    excerpts=[]
    for term in terms:
        pos=text.lower().find(term.lower())
        if pos>=0:excerpts.append(text[max(0,pos-90):pos+1500])
    assert excerpts,key
    return {'id':key,'url':url,'title':title,'authors':authors,'year':year,'venue_or_version':venue,'location_reviewed':location,'fetched_sha256':digest(raw),'fetched_bytes':len(raw),'excerpts':excerpts,'technical_difference':delta,'evidence_level':'source/API excerpt reviewed; not an exhaustive absence proof','code_binding':'413e888e1e8f22cf8da1631cc9ca2133ce30a9db' if key=='neuron_gradient' else 'reference only; no code imported by EP0','checked_date':'2026-09-18'}

def make_literature():
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch_literature,LITERATURE))
    for suffix in ['00-front-matter.md','03-proposed-method.md','04-experiments.md']:
        read(REPO.parent/'相关文章/Understanding neural network through neuron level visualization/en'/suffix)
    records.insert(0,{'id':'nns_cnn','url':'https://doi.org/10.1016/j.neunet.2023.09.030','title':'Understanding neural network through neuron level visualization','authors':'Hui Dou, Furao Shen, Jian Zhao, Xinyu Mu','year':2023,'venue_or_version':'Neural Networks168:484-495','location_reviewed':'local English transcription sections3 eq1-14 and4.2.1 filter aggregate','fetched_sha256':READS['../相关文章/Understanding neural network through neuron level visualization/en/03-proposed-method.md']['sha256'],'fetched_bytes':READS['../相关文章/Understanding neural network through neuron level visualization/en/03-proposed-method.md']['bytes'],'excerpts':['Input coordinate initialization and current ReLU/winner propagation; filter map activation-weighted neuron-image sum. Terminal gate notation must be interpreted consistently.'],'technical_difference':'Endpoint principle and energy target are inherited; no official-code parity claim','evidence_level':'local paper transcription, not official code','code_binding':'official repository not located; paper-formula regression only','checked_date':'2026-09-18'})
    return {'cutoff_date':'2026-09-18','queries':['internal neuron gradient attribution atlas','input resolved head query key value Jacobian decomposition','residual edge attribution gradient activation','Attribution Patching Taylor finite effect','NNS filter energy aggregation'],'search_scope':'targeted prior-art audit; official venues, arXiv and pinned Captum source; no exhaustive novelty absence claim','retrieval_notes':['generic search initially returned no results; direct official-source retrieval used','incorrect candidate PMLR conductance URL returned404; corrected to arXiv1805.12233','NeuronConductance raw source timed out; official neuron API read instead'],'records':records,'positioning_decision':'REPRODUCTION_UNIFIED_IMPLEMENTATION_AND_SYSTEMATIC_EMPIRICAL_AUDIT','new_formula_claim':False,'spatial_information_advantage_claim':'UNTESTED_REQUIRES_C4B','comparison_axes':['internal scalar/aggregate selectors','input-resolved hidden contribution','head/QKV/residual accounting','energy target','finite and spatial intervention','cost and contract verification']}


def make_statistics():
    tests=[];families=[]
    for model in ['vit_b16','deit_s16']:
        for ref in ['normalized_zero','blur_rgb']:
            for structure in ['heads','routing_value','residual_branches']:
                key=f'C4a.{model}.{ref}.{structure}'
                families.append({'family_id':key,'claim':'C4a','model':model,'replacement':ref,'component_family':structure,'statistic':'median over images of within-image Spearman(m_k,abs(e_k))','delta_min':.10,'n_valid_min':360,'ci_width_max':.30,'tests':[key]})
                tests.append({'test_id':key,'type':'exact_two_sided_sign','value':'image Spearman','null_median':0.,'alternative_reported':'two-sided, support requires positive direction','min_nonzero_n':300})
            key=f'C4b.{model}.{ref}.head_regions'
            ids=[]
            for baseline in ['total_gxi','raw_gradient','regional_taylor','random_regions']:
                tid=key+'.vs_'+baseline;ids.append(tid)
                tests.append({'test_id':tid,'type':'exact_two_sided_sign','value':'paired image-level S_endpoint minus S_baseline','null_median':0.,'alternative_reported':'two-sided, support requires positive direction','min_nonzero_n':300})
            families.append({'family_id':key,'claim':'C4b','model':model,'replacement':ref,'component_family':'block5_heads_0_mid_last','statistic':'per-image mean across3 heads of trapezoidal abs effect AUC over q=0,1/16,2/16,4/16 divided by0.25; median paired difference','delta_min':.05,'units':'raw logit','n_valid_min':360,'ci_width_max':.50,'tests':ids})
    assert len(tests)==28 and len(families)==16
    return {'fixed_n':400,'sample_size_rationale':{'design':'fixed precision, no power simulation or claimed power','independent_units':'one image in each of400 distinct true classes, shared across2 models','binary_sign_worst_case_95_halfwidth':1.96*math.sqrt(.25/400),'description':'approx 0.049 sign-proportion halfwidth; correlation/logit precision decided by frozen CI-width gates, not assumed','no_optional_stopping':True,'no_replacement_or_expansion_after_outcomes':True},'primary_predictor':'m_k for C4a; E-derived candidate-region ranking for C4b','signed_secondary':'a_k vs e_k and p_ref vs e_k; descriptive only, sign agreement excludes abs(effect)<=1e-6','families':families,'tests':tests,'holm':{'alpha':.05,'scope':'all28 test IDs across both models/replacements and C4a/C4b','test_ids':sorted(t['test_id'] for t in tests),'algorithm':'sort(p,test_id); adj_i=max(previous,min(1,(28-i)*p_i)), i zero-based'},'sign_test':{'near_zero_difference':1e-12,'zero_rule':'abs(value-null)<=1e-12 contributes tie, excluded from binomial count but retained in N/CI','p_formula':'min(1,2*sum(comb(n,j) for j=0..min(n_positive,n_negative))/2**n)','assumption':'independent selected images; null equal positive/negative sign probability conditional on nonzero, not exchangeable heads','min_nonzero_n':300},'bootstrap':{'level':.95,'replicates':10000,'algorithm':'resample image indices with replacement; paired methods/references/model clusters preserved; medians; NumPy1.26.4 PCG64','seed':'SHA256(namespace|bootstrap|family_id) first8 bytes big endian','quantile':'linear','ci_type':'percentile two-sided, precision interval not familywise CI','holm_controls_tests_not_CIs':True},'NA':{'rank_ties':'average rank, exact numeric equal','min_components':8,'constant_vector':'NA_CONSTANT_VECTOR','near_zero_mass':1e-12,'zero_mass':'retained; undefined metric NA, never impute0','nonfinite_raw':'INVALID_NUMERIC for affected required family','nonfinite_metric':'NA only if mathematical undefined reason registered','paired_set':'intersection across all mandatory methods and both required inputs for family, no method-specific advantage','required_raw_coverage':'all400 attempted; invalid/missing paths prevent support except registered NA','missing_due_cap':'PAUSED_RESOURCE_CAP and INCONCLUSIVE, not a favorable partial-cohort claim'},'decision_order':['INVALID if cohort/hook/protocol/numeric violation','INCONCLUSIVE if exact coverage incomplete, Nvalid<360, nnonzero<300, metric missing, CI width above family max','SUPPORTED if median>=delta, positive direction, all relevant Holm p<=.05; C4b all4 mandatory differences pass','NOT_SUPPORTED if valid sufficient precise data but support criteria fail'],'aggregate_claims':{'C4a':'all12 required association families supported for two-model two-reference association statement','C4b':'all4 spatial families and all16 baseline tests supported for joint spatial gain statement','C1':'EP1 proof/regression, not inferred from utility','C2':'registered target completion and resource measurements; no universal speed claim','C3':'descriptive tradeoff, no superiority/noninferiority confirmatory claim'},'secondary_not_in_holm':['clean_top1','signed association','zeroing','Q/K separate finite effects','joint routing/value interaction','flip/noise','internal unit fidelity','GradCAM/rollout descriptive comparisons']}


def make_evaluation():
    return {'methods_primary':['endpoint_gxi_same_identity_row','full_target_IG_zero','raw_gradient'],'enhanced_methods':{'gradcam':'descriptive, last-block norm1, class target only, token map explicitly labeled; no universal comparator requirement','smoothgrad_ig':'NOT_PLANNED','expected_gradients':'NOT_PLANNED'},'ig':{'baseline':'normalized_zero','path':'straight','quadrature':'Gauss-Legendre','candidate_nodes':[32,64,128,256,512],'check_nodes':[64,128,256,512,1024],'alpha_chunk_candidates':[8,4,2,1],'rule':'EP2 all64 calibration images and two frozen class targets; select smallest M whose relative L2 map difference to2M<=0.02 and normalized completeness residual<=0.01 on at least61/64 images per target and model; finite required on64/64; no pass -> EP2_NUMERICAL_CALIBRATION_FAILED, no new nodes','completeness_denominator':'max(abs(F(x)-F(0)),1e-6)','near_zero_delta_rule':'abs delta<1e-6 requires abs completeness residual<=1e-5','selection':'one fixed M per model shared all formal images; no per-image result-driven integration adaptation'},'fidelity':{'targets':['ground_truth_logit','frozen_clean_top1_secondary'],'scales':[4,16],'cell_counts':[3136,196],'ranking':'sum abs RGB coordinate attribution within cell; decreasing score then row-major cell ID','signed_positive_secondary':'sum RGB then sum cell then max(score,0)','replacements':['normalized_zero','blur_rgb'],'blur':{'source':'clean224crop RGB float32 in[0,1], then normalize per model','kernel':[31,31],'sigma':[10.,10.],'padding':'reflect','operator':'torchvision.functional.gaussian_blur with explicit kernel/sigma; frozen clean blur once'},'fractions':[i/20 for i in range(21)],'select_count':'floor(q*Ncells+0.5), include0 andall; x axis actual selected area/Ncells','deletion':'replace selected cell RGB by same-location baseline b starting clean','insertion':'start b, restore same cells from clean','primary_ordinate':'raw frozen class logit, trapezoid; insertion and deletion separate','secondary_absolute_effect':'abs(F(x)-F(perturbed)), direction explicit','normalized_auc':'secondary only divide by F(x)-F(b); abs denominator<1e-6 -> NA, raw preserved','ood_conditions':'both zero and blur mandatory; no noise/inpainting replacement','pixel_shuffle':'NOT_PLANNED'},'stability':{'primary_target':'ground_truth_logit','flip':'horizontal flip in cropped RGB before normalization; exact inverse flip on maps; compare signed RGB-sum pixel Spearman and4x4 absolute-mass Spearman','model_response':'abs(F(x)-F(Tx)), clean/flipped predicted class, correctness','main_population':'all400, no filter','controlled_strata_logit_delta':[0.1,0.5],'strata_role':'descriptive cumulative subsets, coverage reported, not confirmation','internal_flip':'NOT_PLANNED due no fixed semantic correspondence','noise':{'rgb_sigma':[.01,.03],'seeds_per_sigma':3,'clip':[0.,1.],'seed':'SHA256(namespace|sample_id|noise|sigma|repeat) first8 bytes','reuse':'same physical RGB perturbation across models'},'round_trip':'T_inverse(T(x)) equality is implementation check only'},'atlas':{'selection':'all registered target summaries on all400 evaluation images; retain full raw atlas on first16 evaluation IDs in frozen order; fixed gallery first8; no hand-picked image claims','maps':'one signed RGB-sum map, linear symmetric own max scale with scale saved; no smoothing/clipping; positive/negative/mass derivatives labeled','near_zero':'max abs<=1e-12 => zero display flag; retain numeric row','failures':'all failure/near-zero index rows retained; gallery failure panels clearly marked','cross_model_channels':'coordinate alignment only, no semantic identity'},'runtime':{'K':[1,16,64,256,'full_registered_atlas'],'M':'frozen EP2 selection','cut_C':'all registered components/12blocks','regimes':['single-target backward','one shared forward multiple backwards','batched VJP only if implemented and validated','cut incremental','complete atlas amortized'],'measurements':['latency p50,p95,max','targets/sec','images/sec','peak allocated/reserved bytes','stored bytes/target','total wall','GPU hours'],'warmups':3,'repeats':10,'sync':'CUDA synchronize before/after timers','load_io':'load once; exclusive compute timing and end-to-end timing separately','no_speed_threshold':True}}


def make_interventions(models):
    return {'target':'ground_truth_logit','blocks':list(range(12)),'references':['normalized_zero','blur_rgb'],'donors':'same-image deterministic replacement forward; no other-image donor','canonical_endpoint_unchanged_by_reference':True,'operations':[
        {'id':'head','edge':'AV before concat, all tokens of selected head','keep_clean':'other heads and residual bypass; recompute W_O and downstream'},
        {'id':'routing','edge':'rowwise softmax A, all heads/tokens at registered block','keep_clean':'V and residual bypass'},
        {'id':'value','edge':'V after projection reshape, all heads/tokens','keep_clean':'clean A and bypass'},
        {'id':'query','edge':'Q after projection','keep_clean':'K,V,bypass; recompute scores,softmax,AV'},
        {'id':'key','edge':'K after projection','keep_clean':'Q,V,bypass; recompute scores,softmax,AV'},
        {'id':'attention_branch','edge':'drop_path1 output/add input','keep_clean':'identity'},
        {'id':'mlp_branch','edge':'drop_path2 output/add input','keep_clean':'carry'},
        {'id':'joint_routing_value','edge':'A and V jointly','keep_clean':'bypass; interaction e_joint-e_R-e_V descriptive'}],
      'C4a':{'component_sets':{'heads':'all12blocks x Hheads; no bypass in ranked components','routing_value':'all12blocks x [R,V],24 components','residual_branches':'all12blocks x [attention,MLP],24 components'},'effect':'F(clean)-F(replace), positive is score drop','predictors':{'m':'sum abs raw E','a':'sum signed E','p_ref':'dot(detached downstream gradient,z_clean-z_reference)','output_norm':'L2(z_clean)','gradient_norm':'L2(g)','gradient_activation':'dot(g,z_clean), abs for abs effects','random':'hash-based component ranking,20replicates average per image'},'attention_references':['raw CLS attention entropy and concentration descriptive','attention rollout separately labeled class-agnostic']},
      'EP5A':{'purpose':'correctness_only','images':'first16 calibration for EP1, first16 evaluation after separate EP5 authorization','direction':'v=J_zx*x at endpoint','eta':[.01,.003,.001,.0003],'all_other_cut_edges':'clean; cotangent detached','check':'effect/eta versus a; minimum abs error across fixed eta <=1e-4+0.01*abs(a), all finite; report entire curve','negative_evidence':'not C4 utility'},
      'C4b':{'status':'ENABLED_CONFIRMATORY','heads_by_model':{m['model_id']:[0,m['heads']//2,m['heads']-1] for m in models['models']},'block':5,'candidate_regions':{'count':16,'layout':'4x4 nonoverlapping56x56 windows, row-major, entire224crop','budget_counts':[0,1,2,4],'actual_area_fractions':[0.,.0625,.125,.25]},'score_endpoint':'sum abs E RGB coordinates in each candidate','mandatory_rankers':['total_gxi','raw_gradient','regional_taylor','random_regions'],'score_total_gxi':'sum abs full target GxI over candidate','score_raw_gradient':'sum abs grad_x F over candidate','score_regional_taylor':'abs(dot(g_head,z_head(clean)-z_head(x_with_single_candidate_replaced)))','candidate_taylor_cache':'one perturbed forward per candidate/ref/image/model shared all selected heads, cost included','ranking':'descending candidate score then candidate index; top k union, no effect-based selection','random_repeats':20,'random_aggregation':'mean effect curve across20 fixed random permutations per image/head/ref before image-level statistic','isolated_effect':'F(clean)-F(z_head(x_region_replaced),z_other_heads(clean),identity(clean)); recompute native projection/downstream','full_forward_effect':'F(clean)-F(x_region_replaced), descriptive alongside isolated effect','primary_score':'mean3heads[trapezoid(abs(isolated_effect),actual_area)/0.25]','secondary':'signed effects and full-forward curves','nonlinear_interactions':'Taylor ranks single windows; evaluate finite unions, no additivity assumed'},
      'zeroing_secondary':{'enabled':True,'images':'first64 evaluation','operations':['head','attention_branch','mlp_branch'],'meaning':'activation zeros not baseline-input activations'},'qk_secondary_images':'first64 evaluation','region_ref_hash':'record perturbation masks and replacement tensor hashes on authorized execution','mixed_state_scope':'finite operational effect, not full natural causal mechanism'}


def make_resources():
    caps=[('EP0',0,6,2,2),('EP1',8,12,14,16),('EP2',16,24,24,40),('EP3',64,72,128,168),('EP4',96,96,24,192),('EP5A',8,12,8,200),('EP5B',128,120,48,248),('EP6',0,24,8,256)]
    return {'basis':'prospective resource allocation frozen by EP0 scope; not a runtime prediction or authorization to consume later budgets','stages':[{'stage':s,'gpu_hours':g,'wall_hours':w,'incremental_retained_GiB':disk,'cumulative_retained_GiB':cum} for s,g,w,disk,cum in caps],'aggregate_gpu_hours':320,'max_retained_GiB':256,'worker':{'gpu_memory_cap_GiB':24,'cpu_rss_cap_GiB':32,'task_timeout_seconds':1800,'max_attempts':2,'minimum_free_disk_GiB':50,'cpu_threads':4},'gpu_qualification':{'type':'NVIDIA H20','min_devices':1,'max_devices':8,'min_free_MiB':32768,'max_utilization_percent':10,'record_existing_pids':True,'do_not_kill_other_processes':True},'queue':{'implementation':'minimal SQLite3 WAL transaction claim','one_worker_per_gpu':True,'dynamic_global':True,'static_modulo_sharding':False,'claim_lease_seconds':180,'heartbeat_seconds':30,'fencing_token':'monotonic per attempt; expired worker commit rejected','result':'immutable validated file then transactional complete','seed':'SHA256(namespace|task_id) first8 bytes; independent of worker count','stop':'before new claim reserve full task_timeout GPU/wall/storage bound; if insufficient -> no dispatch','resume':'validate hashes, reset expired claims only, exact task coverage','oom':'halve target batch then alpha chunk in fixed candidates; no dtype/M/model/target reduction'},'runtime_identity':{'CUDA_before_import':'CUBLAS_WORKSPACE_CONFIG=:4096:8','deterministic_algorithms':True,'tf32':False,'autocast':False,'production_dtype':'float32','fixture_dtype':'float64 CPU and float32 CPU/GPU','sys_executable_realpath_required':True,'cuda_uuid_and_versions_required':True},'cap_status':'PAUSED_RESOURCE_CAP separate from scientific outcome; preserve roots; explicit prospective authorization required for any increase','missing_tasks':'no favorable partial-cohort support','current_stage_gpu_permission':False,'retention':{'all_images':'raw endpoint/IG/gradient class maps; all unit/component summary rows; masks/effects/predictions/resources','raw_atlas_and_component_maps':'first16 frozen evaluation IDs only, all models; other raw maps computed for scalar/spatial statistics then discarded after validated per-task summary','format':'float32 signed RGB arrays, no display transform, lossless compression optional','no_rendering_data_in_metrics':True}}


def make_contracts():
    return {'float64':{'atol':1e-9,'rtol':1e-7,'near_zero_norm':1e-12,'finite_difference_steps':[1e-4,3e-5,1e-5],'fd_atol':1e-6,'fd_rtol':1e-4},'float32':{'atol':1e-5,'rtol':1e-4,'near_zero_norm':1e-6,'finite_difference_steps':[.01,.003,.001],'fd_atol':.002,'fd_rtol':.02},'acceptance':'max_abs<=atol+rtol*max_abs(reference); relativeL2 reported, check<=rtol when norm>near_zero or max_abs<=atol; no nonfinite','fused_unfused_logits':{'float64':{'atol':1e-8,'rtol':1e-7},'float32':{'atol':.0001,'rtol':.0001}},'backend_support':'canonical unfused; fused incapable of hooks unsupported for decomposition; fused logits parity mandatory on supported GPU path','required_regressions':['paper_formula_CNN_GxI','CNN_relu_off_unique_winner','energy_weight_target_same_tensor','all_block_each_target_kind','residual_wrong_total_gradient_false_pass','nonzero_head_omission','nonzero_routing_omission','QKV_branch_vs_total_scope','head_projection_bias_downstream','forward_no_hook_noop_hook_parity','finite_difference_JVP','local_direction_correctness','single_multi_worker_equivalence','expired_attempt_reject','crash_exact_coverage','P95_single_gpu_and_multi_gpu_throughput'],'failed_regression':'EP1_FAIL fresh recovery required; no EP2 posthoc tolerance widening','EP2_allowed_adaptation':{'target_batch_candidates':[16,8,4,1],'alpha_batch_candidates':[8,4,2,1],'selection':'largest fits within caps and fixed numerical contracts on all64 calibration inputs; deterministic order','disallowed':['sample count','class target','primary family','effect threshold','CI/test','backend semantics','resource cap increase','outcome-dependent per-image integration count']}}


def make_access():
    return {'authorization_text':'执行EP0','authorized_stage':'EP0_ONLY','parent_commit':PARENT,'branch':'stage-d-d3-pass','prior_plan':PLAN,'allowed_current':['metadata JSON/CSV/text','source code without importing torch/timm','public literature/API text','checkpoint byte streaming for SHA256 only','archive stat only','filename listing without opening image payloads','EP0 protocol validation and git publication'],'forbidden_current':['JPEG/PNG image or mask payloads','archive member payload read','model deserialization/forward/backward','Torch/timm/PIL imports','CUDA initialization or GPU jobs','formal prediction/logit/attribution/intervention outcomes','EP1 execution','new power simulation'],'source_access_caveat':'metadata may contain historical hashes/status but no new formal outcomes used','future_permissions':[{'stage':'EP1','eligible':'synthetic fixtures and first8 discovery IDs; checkpoint loading','requires':'new explicit EP1 user authorization and published EP0 PASS','evaluation_access':False},{'stage':'EP2','eligible':'64 calibration IDs and model hashes','requires':'new EP2 authorization and EP1 PASS','evaluation_access':False},{'stage':'EP3-EP5','eligible':'400 sealed evaluation IDs and zero/blur reference transforms only','requires':'per-stage explicit authorization, published prior PASS, code/data binding and archive/image SHA before decode','evaluation_access':True}],'ep0_pass_next_step':'EP1_IMPLEMENTATION_AND_REGRESSION_REQUIRES_USER_AUTHORIZATION','production_authorized':False,'unrelated_untracked_cut_intervention_files':'not authority, not read for protocol, not modified'}


def schema_for(value):
    if isinstance(value,dict):return {'type':'object','additionalProperties':False,'required':sorted(value),'properties':{k:schema_for(v) for k,v in value.items()}}
    if isinstance(value,list):
        # Position-bound immutable protocol schema, exact arrays are intentional at freeze.
        return {'type':'array','minItems':len(value),'maxItems':len(value),'prefixItems':[schema_for(v) for v in value],'items':False}
    return {'const':value}


def main():
    if ROOT.exists():raise FileExistsError('Use a fresh EP0 root; never overwrite terminal roots')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==PARENT
    for p in [PLAN,'docs/NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md','src/vitnns/endpoint_vit.py','src/vitnns/endpoint_nns.py','src/vitnns/modeling.py','src/vitnns/labels.py','src/vitnns/constants.py','requirements.lock.txt','config/stage_x_x2.json','scripts/run_endpoint_nns_vit_level4.py', 'scripts/build_endpoint_ep0.py']:
        read(REPO/p)
    cohort=make_cohort();models=make_models();targets=make_targets(models)
    print('metadata cohort',cohort['counts'],'excluded',cohort['excluded_union_count'],'targets',targets['target_count_by_model'],flush=True)
    literature=make_literature();print('public source records',len(literature['records']),flush=True)
    protocol={'schema_version':1,'protocol_id':NAMESPACE,'stage':'EP0','status':'EP0_PROTOCOL_FROZEN','study_positioning':literature['positioning_decision'],'scope':'two registered standard ViT blocks, original-NNS-style endpoint adaptation and empirical audit','cohort':{'counts':cohort['counts'],'rows_digest':canonical(cohort['rows']),'same400_each_model':True,'known_legacy_endpoint_overlap':0},'targets':{'primary':'ground_truth_logit','secondary':'frozen_clean_top1','retain_misclassified':True,'internal_registry':'target_registry.json'},'statistics':'statistical_analysis_plan.json','evaluations':'evaluation_registry.json','interventions':'intervention_registry.json','resources':'resource_contract.json','numerical_contract':'implementation_contract.json','novelty':'literature_registry.json','access':'access_contract.json','TBD_count':0,'production_experiments_authorized':False,'EP1_started':False}
    items={'protocol.json':protocol,'cohort_registry.json':cohort,'model_registry.json':models,'target_registry.json':targets,'statistical_analysis_plan.json':make_statistics(),'evaluation_registry.json':make_evaluation(),'intervention_registry.json':make_interventions(models),'resource_contract.json':make_resources(),'implementation_contract.json':make_contracts(),'literature_registry.json':literature,'access_contract.json':make_access()}
    assert items['resource_contract.json']['aggregate_gpu_hours']==sum(r['gpu_hours'] for r in items['resource_contract.json']['stages'])
    ROOT.mkdir(parents=True)
    for name,value in items.items():write(name,value)
    write('protocol_bundle.schema.json',{'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Immutable EP0 protocol bundle, closed and value-bound','type':'object','additionalProperties':False,'required':sorted(items),'properties':{k:schema_for(v) for k,v in items.items()}})
    write('source_bindings.json',{'files':list(READS.values()),'sys_executable_realpath':str(Path(sys.executable).resolve()),'torch_imported':'torch' in sys.modules,'timm_imported':'timm' in sys.modules,'PIL_imported':'PIL' in sys.modules,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'new_model_calls':0,'formal_image_payload_reads':0,'cohort_source_method':'official WNID text and filenames only; no tar content opened','public_sources':'literature registry fetch hashes/excerpts; live HTML may change','dirty_user_files_preserved':True})
    resolutions=[('class_target','target_registry.json','primary_class'),('unit_tensor_positions','target_registry.json','targets_by_model'),('reconstruction_scopes','target_registry.json','cuts'),('method_names','target_registry.json','main_formula'),('RGB_mass','target_registry.json','mass'),('atlas_selection','evaluation_registry.json','atlas'),('scan_scale_cost','evaluation_registry.json','runtime'),('fidelity','evaluation_registry.json','fidelity'),('stability','evaluation_registry.json','stability'),('replacement_zeroing','intervention_registry.json','zeroing_secondary'),('EP5A_EP5B_C4','intervention_registry.json','C4b'),('cohort_models','cohort_registry.json','rows'),('statistics_exit','statistical_analysis_plan.json','decision_order'),('model_backend','model_registry.json','models'),('novelty','literature_registry.json','positioning_decision'),('resource_caps','resource_contract.json','stages'),('outcome_access','access_contract.json','future_permissions')]
    write('ep0_checklist.json',{'stage':'EP0','planning_checklist_count':17,'resolutions':[{'item':item,'file':file,'key':key,'status':'FROZEN'} for item,file,key in resolutions],'ep1_execution_status':'NOT_STARTED','authority_scope':'EP0_ONLY'})
    write('environment_contract.json',{'python':str(Path(sys.executable).resolve()),'versions':{'torch':'2.2.2+cu121','timm':'0.9.16','torchvision':'0.17.2+cu121','numpy':'1.26.4','Pillow':'10.2.0'},'mode':'package metadata only, no torch/timm/PIL import','CUDA_VISIBLE_DEVICES':'','runtime_experiment_identity':'future workers additionally record CUDA/runtime/UUID before execution; not queried in EP0'})
    print('EP0 files created:',ROOT,flush=True)

if __name__=='__main__':main()
