from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3c4 as f3c4
from vitnns import stage_f_f3c4_validate as validator

REPO=Path(__file__).resolve().parents[1]
def _synthetic_terminal(tmp_path: Path) -> Path:
    root=tmp_path/"terminal"; root.mkdir()
    sample_id="SYNTHETIC_NOT_FORMAL"; method=f3c4.METHODS[0]; patch=np.arange(1,197,dtype=np.float32).reshape(14,14); mass=np.asarray(patch/float(patch.sum(dtype=np.float64)),dtype=np.float32)
    rankings=[]; priorities=[]; seeds=[]
    for repeat in range(5):
        seed=f3c4._tie_seed(sample_id,repeat); ranking,priority=f3c4._method_ranking(patch,1,seed); seeds.append(seed); rankings.append(ranking.astype(np.float32)); priorities.append(f3c4._array_digest(priority))
    actual=np.asarray(f3c4.ACTUAL_FRACTIONS,dtype=np.float32); insertion=np.stack([actual+np.float32(repeat)*np.float32(.001) for repeat in range(5)]); deletion=np.stack([1-actual+np.float32(repeat)*np.float32(.001) for repeat in range(5)]); ins=np.trapz(insertion.astype(np.float64),actual.astype(np.float64),axis=1); dele=np.trapz(deletion.astype(np.float64),actual.astype(np.float64),axis=1); scores=0.5*(ins-dele)
    arrays={"native.npy":np.ones((3,224,224),dtype=np.float32),"patch_mass.npy":patch,"target_oriented_mass.npy":mass,"rankings.npy":np.stack(rankings),"insertion_curves.npy":insertion,"deletion_curves.npy":deletion,"insertion_aucs.npy":ins.astype(np.float32),"deletion_aucs.npy":dele.astype(np.float32),"fidelity_scores.npy":scores.astype(np.float32),"fractions.npy":np.asarray(f3c4.FRACTIONS,dtype=np.float32),"actual_fractions.npy":actual}
    records=[]
    for name,value in arrays.items():
        with (root/name).open("wb") as stream: np.save(stream,value,allow_pickle=False)
        records.append({"path":name,"bytes":(root/name).stat().st_size,"sha256":f3c4._sha_file(root/name),"array_digest":f3c4._array_digest(value),"shape":list(value.shape),"dtype":"float32"})
    target={"target_ref":"clean_predicted_class_raw_logit","class_index":7,"clean_input_sha256":"c"*64,"official_truth_index":7,"clean_top1_correct":True,"resolved_once_from_same_clean_forward":True,"frozen_target":{"synthetic":True}}; target_digest=f3c4.canonical_sha256(target)
    identity={"protocol_id":"STAGE_F_F3C_CLASS_TARGETED_V1_20260907","sample_id":sample_id,"formal_position":1,"selection_order":179,"jpeg_sha256":"a"*64,"official_imagenet1k_index":7,"method_id":method,"model_sha256":f3c4.MODEL_SHA256,"model_canonical_tensor_digest":f3c4.MODEL_TENSOR_DIGEST,"preprocessing_identity":"p"*64,"clean_input_sha256":"c"*64,"frozen_target_class_index":7,"target_identity_digest":target_digest,"source_config_code_digest":f3c4.F3C3_SOURCE_DIGEST,"source_config_code_hashes":{"synthetic":"s"*64},"attempt_id":"attempt-000001"}
    payload={"native_array_digest":f3c4._array_digest(arrays["native.npy"]),"patch_mass_array_digest":f3c4._array_digest(patch),"target_oriented_mass_array_digest":f3c4._array_digest(mass),"ranking_array_digest":f3c4._array_digest(arrays["rankings.npy"]),"tie_priority_sha256":priorities,"insertion_curve_array_digest":f3c4._array_digest(insertion),"deletion_curve_array_digest":f3c4._array_digest(deletion)}
    curves=[{"fidelity_score":float(value)} for value in scores]
    result=f3c4._signed({"schema_version":1,"document_type":"synthetic_f3c4_terminal_fixture","status":"VALID","attempt_id":"attempt-000001","unit_identity":identity,"unit_canonical_digest":f3c4.canonical_sha256(identity),"target":target,"target_identity_digest":target_digest,"method_id":method,"class_targeted":True,"class_agnostic":False,"attention_hashes":["h"*64]*12,"endpoint_parity":{"status":"PASS","max_absolute_error":0.0},"ig_conservation":{"selected_steps":512,"passed":True,"attempts":[{"steps":512,"pass":[True]}]},"payload_identities":payload,"fidelity_contract":{"fractions":list(f3c4.FRACTIONS),"actual_fractions":list(f3c4.ACTUAL_FRACTIONS),"tie_seeds":seeds,"tie_priority_sha256":priorities,"target":"unmodified_frozen_clean_predicted_class_raw_logit","score":"0.5*(insertion_auc-deletion_auc)"},"fidelity_numeric_payload":{"sigma":1,"curves":curves,"fidelity_score_mean_over_five_ties":float(np.mean(scores))}}); (root/"result.json").write_bytes(f3c4.canonical_json_bytes(result)+b"\n")
    status=f3c4._signed({"schema_version":1,"document_type":"synthetic_status","status":"VALID","terminal":True}); (root/"status.json").write_bytes(f3c4.canonical_json_bytes(status)+b"\n")
    manifest=f3c4._signed({"schema_version":1,"document_type":"synthetic_manifest","status":"VALID","result":f3c4._file_record(root,"result.json"),"status_record":f3c4._record(root,"status.json"),"arrays":records}); (root/"manifest.json").write_bytes(f3c4.canonical_json_bytes(manifest)+b"\n")
    blind=f3c4._signed({"schema_version":1,"document_type":"synthetic_blind","status":"VALID","result_sha256":f3c4._sha_file(root/"result.json"),"manifest_sha256":f3c4._sha_file(root/"manifest.json")}); (root/"blind.json").write_bytes(f3c4.canonical_json_bytes(blind)+b"\n")
    return root

def _expected(root: Path):
    result=json.loads((root/"result.json").read_text()); identity=result["unit_identity"]
    return {key:identity[key] for key in ("formal_position","selection_order","sample_id","method_id","attempt_id","jpeg_sha256","official_imagenet1k_index","preprocessing_identity","source_config_code_digest","source_config_code_hashes")}

def test_synthetic_terminal_reconciles_and_scores_five_ties(tmp_path: Path):
    root=_synthetic_terminal(tmp_path); item=f3c4.reconcile_terminal(root,_expected(root)); assert item["repeat_scores"].shape==(5,) and item["arrays"]["target_oriented_mass.npy"].flags.writeable is False

def test_terminal_array_mutation_is_rejected(tmp_path: Path):
    root=_synthetic_terminal(tmp_path); path=root/"fidelity_scores.npy"; value=np.load(path,allow_pickle=False); value[0]+=np.float32(.2)
    with path.open("wb") as stream: np.save(stream,value,allow_pickle=False)
    with pytest.raises(f3c4.StageFF3C4Error,match="hash"): f3c4.reconcile_terminal(root,_expected(root))

def test_terminal_identity_mutation_is_rejected(tmp_path: Path):
    root=_synthetic_terminal(tmp_path); expected=_expected(root); path=root/"result.json"; value=json.loads(path.read_text()); value.pop("canonical_digest"); value["unit_identity"]["selection_order"]=44; value["unit_canonical_digest"]=f3c4.canonical_sha256(value["unit_identity"]); value=f3c4._signed(value); path.write_bytes(f3c4.canonical_json_bytes(value)+b"\n")
    with pytest.raises(f3c4.StageFF3C4Error): f3c4.reconcile_terminal(root,expected)

def test_csv_json_parity_mutation_is_rejected(tmp_path: Path):
    root=tmp_path; paper=root/f3c4.ROOT_PATH/"paper_data"; paper.mkdir(parents=True); rows=[{"a":1,"b":2.0} for _ in range(250)]; document=f3c4._json_rows("stage_f_f3c4_per_image_method_scores",rows); (paper/"per_image_method_scores.json").write_bytes(f3c4.canonical_json_bytes(document)+b"\n")
    with (paper/"per_image_method_scores.csv").open("w",newline="") as stream: writer=csv.DictWriter(stream,fieldnames=["a","b"],lineterminator="\n"); writer.writeheader(); writer.writerows(rows[:-1]+[{"a":1,"b":3.0}])
    with pytest.raises(f3c4.StageFF3C4Error,match="parity"): validator.validate_table_parity(root,"per_image_method_scores")

def test_validator_source_has_no_torch_timm_or_model_runtime():
    source=(REPO/"src/vitnns/stage_f_f3c4_validate.py").read_text(); assert not any(token in source for token in ("import torch","import timm","load_canonical_model","torch.load","cuda:"))
