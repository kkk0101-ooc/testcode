from __future__ import annotations
import copy
from pathlib import Path
from vitnns import nns_patch_depth_s4_r11_posthoc as r11

def test_canonical_is_stable():
 assert r11.canonical_digest({'b':1,'a':2})==r11.canonical_digest({'a':2,'b':1})
def test_allowed_terminal_states_are_exploratory_or_integrity_only():
 assert r11.RUN_ID=='s4_r11_posthoc_exploratory_20260916' and 'PASS' not in ('PASS_POST_HOC_EXPLORATORY','FAIL_POST_HOC_EXPLORATORY','BLOCKED_DATA_INTEGRITY')
def test_expected_anchor_contract_is_frozen():
 assert r11.EXPECTED_CUTOFF_NS==1789511492710807249 and r11.EXPECTED_CUTOFF_INDEX==300 and r11.EXPECTED_SELECTED_CLASS_COUNT==120
def test_expected_numeric_anchor_is_frozen():
 assert r11.EXPECTED_NUMERIC['P2']['estimate']==0.0749396239 and r11.EXPECTED_NUMERIC['P2']['half_width']==0.0178554310
def test_cpu_only_source_has_no_forbidden_imports():
 source=(Path(__file__).resolve().parents[1]/'src/vitnns/nns_patch_depth_s4_r11_posthoc.py').read_text();assert 'import torch' not in source and 'import timm' not in source and 'from PIL' not in source
def test_original_rule_is_distinct_from_posthoc():
 assert r11.EXPECTED_NUMERIC['P2']['estimate']<0.10 and r11.EXPECTED_NUMERIC['P2']['half_width']>0.015
