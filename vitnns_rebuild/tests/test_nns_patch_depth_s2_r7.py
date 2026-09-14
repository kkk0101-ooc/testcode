import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=importlib.util.spec_from_file_location("v",ROOT/"scripts/validate_nns_vit_patch_depth_s2_r7.py");v=importlib.util.module_from_spec(S);S.loader.exec_module(v)
def test_r7_source_recomputes():
    r=v.validate();assert r["terminal_status"]=="BLOCKED_SCIENTIFIC_VALIDITY" and r["D2_images"]==11557 and r["D2_power"]==.82161 and r["next_authorization"]=="NONE"
def test_r7_false_pass():
    r=v.false_pass();assert r["status"]=="PASS" and r["attack_count"]==22
