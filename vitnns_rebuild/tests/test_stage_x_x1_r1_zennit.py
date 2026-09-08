import hashlib,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];CONFIG=REPO/"config/stage_x_x1_r1_zennit.json"
def test_zennit_addendum_is_signed_and_exact():
 value=json.loads(CONFIG.read_text());observed=value.pop("canonical_digest");actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert observed==actual;assert value["package"]=="zennit" and value["version"]=="0.5.1";assert value["wheel_sha256"]=="a49be81f34ccb851633fea08a07bba2d04e73b00c25ffe54ac0ec822d0a3b096";assert value["wheel_bytes"]==54629;assert value["other_dependency_download_allowed"] is False;assert value["dataset_or_checkpoint_download_allowed"] is False
def test_gamma_grid_is_deferred_to_x2_not_result_adaptive():
 value=json.loads(CONFIG.read_text());grid=value["attnlrp_gamma_grid"];assert grid["conv"]==[0.1,0.25,100.0];assert grid["linear"]==[0.0,0.01,0.05,0.1,1.0];assert grid["selection_stage"]=="X2_BEFORE_X3_OUTCOME_ACCESS";assert grid["x1_role"]=="forward_parity_and_adapter_compatibility_only"
