from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from vitnns import nns_patch_depth_s2_r4 as s2
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_pre_power_contract():
    v=json.loads((ROOT/"config/nns_vit_patch_depth/s2_r4_pre_power_protocol.json").read_text()); assert v["pre_power"]["effect_thresholds"]=={"P1":.02,"P2":.1,"D1":.02,"D2":.01}; assert v["execution_contract"]["static_index_mod_K_full_production"]=="fail_closed_disabled"
def test_holm_fixed_slots_and_tie_rule():
    r=s2.holm({"P1":.01,"P2":.01,"D1":.2,"D2":1.0}); assert [x["endpoint"] for x in r["rows"]]==["P1","P2","D1","D2"] and r["family_size"]==4
def test_atomic_claim_one_winner(tmp_path):
    workers=[{"worker_id":f"w{i}","worker_pid":i+1,"physical_gpu_index":i,"gpu_uuid":f"g{i}"} for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as p: values=list(p.map(lambda x:s2.atomic_claim(tmp_path,3,"s3",x,9),workers))
    assert sum(x is not None for x in values)==1
def test_terminal_skip_stale_claim_and_missing_only(tmp_path):
    stage=tmp_path/"stage"; old=stage/"launches/old"; new=stage/"launches/new"; w={"worker_id":"w","worker_pid":1,"physical_gpu_index":0,"gpu_uuid":"g"}; schedule=[{"schedule_index":i,"source_image_id":f"s{i}"} for i in range(3)]
    assert s2.atomic_claim(old,1,"s1",w,1); p=old/"units/unit-0000/completion/unit.json"; p.parent.mkdir(parents=True); p.write_text(json.dumps({"payload":{"schedule_index":0,"terminal":{"complete":True,"immutable":True}}})); assert s2.claim_next(stage,new,schedule,w,2)["schedule_index"]==1
def test_seed_worker_independent_resources_and_static_disabled():
    assert s2.seed64("n",1,"s")==s2.seed64("n",1,"s"); rows=[{"schedule_index":2,"launch_id":"b","gpu_seconds":1e16,"cpu_seconds":1,"wall_seconds":2,"retained_bytes":3},{"schedule_index":1,"launch_id":"a","gpu_seconds":1,"cpu_seconds":4,"wall_seconds":5,"retained_bytes":6},{"schedule_index":3,"launch_id":"c","gpu_seconds":-1e16,"cpu_seconds":7,"wall_seconds":8,"retained_bytes":9}]; r=s2.aggregate_resources(rows); assert r["gpu_seconds"]==1.0 and not r["divide_GPU_hours_by_K"]
