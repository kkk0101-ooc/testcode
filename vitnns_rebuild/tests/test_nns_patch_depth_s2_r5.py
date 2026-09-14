import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=importlib.util.spec_from_file_location("v",ROOT/"scripts/validate_nns_vit_patch_depth_s2_r5.py");v=importlib.util.module_from_spec(S);S.loader.exec_module(v)
def test_final_blocked_root_recomputes():
    r=v.validate();assert r["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and r["D2_power_at_872"]==.6934 and r["next_authorization"]=="NONE"
def test_final_false_pass_rejected():
    r=v.false_pass();assert r["status"]=="PASS" and r["attack_count"]==8
