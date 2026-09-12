from __future__ import annotations
import copy,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from jsonschema import ValidationError
from vitnns import stage_y_y3 as y3

REPO=Path(__file__).resolve().parents[1]

def validator():
    spec=importlib.util.spec_from_file_location("validate_y3",REPO/"scripts/validate_stage_y_y3.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_roster_coordinate_and_source_correction():
    assert y3.CONDITIONS==("horizontal_flip","translate_x_-8","translate_x_8","rotate_-10","rotate_10","brightness_0.9","brightness_1.1","gaussian_noise_0.01_seed_0")
    assert [y3.unit_index(r,c) for r in range(20) for c in range(8)]==list(range(160))
    assert y3.COMPUTED_ORDERS==(77,84,93) and 93 in y3.SELECTION_ORDER

def test_subset_and_transform_binding_metadata_only():
    subset=y3.subset_records(REPO);transforms=y3.transform_records(REPO)
    assert len(subset)==20 and len(transforms)==8 and [x["selection_order"] for x in subset]==list(y3.SELECTION_ORDER)

def test_noise_seed_exact_pcg64_formula():
    source="ILSVRC2012_VAL_OFFICIAL:ILSVRC2012_val/example.JPEG";condition=y3.CONDITIONS[-1]
    assert y3.noise_seed(source,condition)==int.from_bytes(__import__("hashlib").sha256(("STAGE_Y_NOISE_V1_20260910\0"+source+"\0"+condition).encode()).digest()[:8],"big")
    a=np.random.Generator(np.random.PCG64(y3.noise_seed(source,condition))).normal(0,.01,(3,4,4)).astype(np.float32);b=np.random.Generator(np.random.PCG64(y3.noise_seed(source,condition))).normal(0,.01,(3,4,4)).astype(np.float32);np.testing.assert_array_equal(a,b)

def test_transform_inverse_mask_fixtures(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8");import torch
    from vitnns import stage_y_y1 as y1
    image=torch.linspace(-1,1,3*224*224,dtype=torch.float32).reshape(1,3,224,224)
    for condition in y3.CONDITIONS:
        transformed,meta=y1.transform_image(image,condition,"source");assert transformed.shape==image.shape and transformed.dtype==torch.float32
        _,mask=y1.inverse_map(torch.ones((224,224),dtype=torch.float32),meta);assert mask.dtype==np.bool_ and mask.shape==(224,224)
        if condition=="translate_x_-8":assert not mask[:,:8].any() and mask[:,8:].all()
        elif condition=="translate_x_8":assert mask[:,:216].all() and not mask[:,216:].any()
        elif condition.startswith("rotate_"):assert 0<mask.mean()<1
        else:assert mask.all()

def test_fixed_target_and_prediction_flip_are_independent():
    fixed={"class_id":7,"rule":"clean predicted-class raw logit"};pred={"fixed_target":fixed,"clean_target_logit":2.,"clean_baseline_logit":0.,"transformed_predicted_class":8,"transformed_fixed_target_logit":1.,"baseline_target_logit":0.,"transformed_target_delta":1.,"prediction_changed":True,"target_reselected":False,"clean_forward_executed":False};y3.validate_document(y3.signed("prediction",pred));assert pred["fixed_target"]["class_id"]==7

def test_array_rosters_and_inverse_direction():
    assert len(y3.expected_array_names(13))==265 and len(y3.expected_array_names(17))>265
    w=np.zeros((196,196),np.float32);w[1,0]=1;t=np.zeros((196,196),np.float32);t[0,0]=2
    expected=np.asarray(w,dtype=np.float64)@np.asarray(t,dtype=np.float64)@np.asarray(w,dtype=np.float64).T
    np.testing.assert_array_equal(y3.inverse_matrix(w,t),expected.astype(np.float32))

def test_all_nested_schemas_closed_and_run_spec_valid():
    y3.validate_schema_files(REPO)
    def walk(x):
        if isinstance(x,dict):
            if x.get("type")=="object":assert x.get("additionalProperties") is False
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(y3.artifact_schema())

@pytest.mark.parametrize("attack",["unknown_array","wrong_shape","wrong_dtype","wrong_axes","nonfinite","wrong_digest"])
def test_array_false_pass_rejections(tmp_path,attack):
    name="input_attribution";a=np.zeros((3,224,224),np.float32)
    if attack=="unknown_array":name="extra"
    if attack=="wrong_shape":a=a.reshape(3,112,448)
    if attack=="wrong_dtype":a=a.astype(np.float64)
    if attack=="nonfinite":a[0,0,0]=np.nan
    p=tmp_path/(name+".npy");np.save(p,a,allow_pickle=False);row={**y3.file_record(p,tmp_path),"canonical_digest":y3.array_digest(a),"dtype":str(a.dtype),"shape":list(a.shape),"axes":["rgb","pixel_y","pixel_x"]}
    if attack=="wrong_axes":row["axes"]=["bad"]
    if attack=="wrong_digest":row["canonical_digest"]="f"*64
    with pytest.raises(Exception):y3.validate_array(tmp_path,row)

def test_cublas_preimport_guard():
    env=dict(os.environ);env.pop("CUBLAS_WORKSPACE_CONFIG",None);env["PYTHONPATH"]=str(REPO/"src");p=subprocess.run([str(y3.PYTHON),"-c","import vitnns.stage_y_y3_worker"],cwd=REPO,env=env,text=True,capture_output=True);assert p.returncode!=0 and "before Torch import" in p.stderr
    env["CUBLAS_WORKSPACE_CONFIG"]=":16:8";p=subprocess.run([str(y3.PYTHON),"-c","import vitnns.stage_y_y3_worker"],cwd=REPO,env=env,text=True,capture_output=True);assert p.returncode!=0

def test_startup_probe_source_compiles_and_contains_no_nul():
    spec=importlib.util.spec_from_file_location("run_y3",REPO/"scripts/run_stage_y_y3.py");runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);source=runner.probe_script();assert "\0" not in source;compile(source,"<y3-probe>","exec")

def test_worker_imports_y2_contract_before_use():
    source=(REPO/"src/vitnns/stage_y_y3_worker.py").read_text();assert "from . import stage_y_y2_r2 as r2" in source;compile(source,"<y3-worker>","exec")

def test_false_pass_audit_has_all_required_attacks():
    result=validator().false_pass_audit();assert result["attack_count"]==result["rejected_count"]>=40

def test_schema_rejects_open_nested_payload():
    doc=y3.signed("prediction",{"fixed_target":{"class_id":1,"rule":"clean predicted-class raw logit"},"clean_target_logit":1.,"clean_baseline_logit":0.,"transformed_predicted_class":1,"transformed_fixed_target_logit":1.,"baseline_target_logit":0.,"transformed_target_delta":1.,"prediction_changed":False,"target_reselected":False,"clean_forward_executed":False});doc["payload"]["extra"]={};doc["canonical_digest"]=y3.canonical({k:v for k,v in doc.items() if k!="canonical_digest"})
    with pytest.raises(ValidationError):y3.validate_document(doc)

def test_synthetic_worker_terminal_reconciliation_completion(tmp_path):
    root=tmp_path/"formal";pending=root/"completion/y3.pending.json"
    y3.write(pending,"pending",{"started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"},"PENDING")
    counts={state:0 for state in y3.TERMINAL_STATES};counts["VALID"]=20
    reconciliation={"denominator_images":20,"expected_nonclean_units":160,"clean_reference_count":20,"terminal_count":160,"valid_count":160,"complete_image_count":20,"prediction_flip_count":3,"condition_counts":{condition:{**{state:0 for state in y3.TERMINAL_STATES},"VALID":20} for condition in y3.CONDITIONS},"shard_counts":[{"shard_id":0,"terminal_counts":counts}],"unified_condition_count":180,"blocking_states":[],"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"next_authorization":y3.NEXT}
    rec=y3.write(root/"reconciliation.json","reconciliation",reconciliation);dummy=y3.file_record(root/"reconciliation.json",tmp_path)
    completion={"workflow_status":"PASS","denominator_images":20,"expected_nonclean_units":160,"clean_reference_count":20,"terminal_count":160,"complete_image_count":20,"formal_outcome_access":False,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"next_authorization":y3.NEXT,"reconciliation":dummy,"validation_report":dummy,"whole_root_inventory":dummy,"payload_manifest":dummy,"report":dummy,"failure":None}
    y3.publish_terminal(pending,root/"completion/y3.json",y3.signed("completion",completion));assert not pending.exists() and y3.read(root/"completion/y3.json")["status"]=="PASS"
