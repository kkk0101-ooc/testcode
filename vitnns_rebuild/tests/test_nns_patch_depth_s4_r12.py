from __future__ import annotations
import json
from pathlib import Path
from vitnns import nns_patch_depth_s4_r12 as r12
ROOT=Path(__file__).resolve().parents[1]
def test_corrected_constants():
 assert r12.FULL_RECORDS_DIGEST=='3bb68cb6bb627f4a84f1a6d7bd54e26d430541b4ae4cd565a6fe3eb2d46863e9' and r12.MEMBERSHIP_DIGEST=='ad0c10e5057d767ef4329f81b22c5f1569afb7e820c7f88bb5199da7e03cb9b2'
def test_corrected_class_scopes():
 assert r12.EXPECTED_COUNTS=={'selected_sources':240,'selected_terminals':720,'selected_distinct_class_ranks':122,'P1_D1_valid_classes':120,'P2_valid_classes':119}
def test_membership_projection_field_name():
 rows=[{'schedule_index':3,'formal_class_rank':2,'source_id':'x'}];x=r12.corrected_identity(rows);assert x['membership_projection']==[{'schedule_index':3,'formal_class_rank':2,'source_image_id':'x'}]
def test_r11_binding_is_exact():
 assert r12.R11_BINDINGS['terminal_receipt.json'][0]=='eaabb5cc0912ea7b6bc632300e1214bcedbc16febe7c2fc1b11e2d71e6efa6a4'
def test_cpu_only_source():
 s=(ROOT/'src/vitnns/nns_patch_depth_s4_r12.py').read_text();assert 'import torch' not in s and 'import timm' not in s and 'from PIL' not in s
def test_original_P2_not_redefined():
 assert r12.EXPECTED_NUMERIC['P2']['estimate']<.10 and r12.EXPECTED_NUMERIC['P2']['half_width']>.015
def test_actual_r11_records_have_both_corrected_digests_and_122_ranks():
 records=json.loads((ROOT/'evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/exploratory_240_source_manifest.json').read_text())['payload']['records'];identity=r12.corrected_identity(records);assert len(records)==240 and len({x['formal_class_rank'] for x in records})==122;assert identity['full_manifest_records_digest']==r12.FULL_RECORDS_DIGEST and identity['membership_projection_digest']==r12.MEMBERSHIP_DIGEST
def test_string_seed_array_uses_hash_shape_dtype_not_isfinite():
 source=(ROOT/'src/vitnns/nns_patch_depth_s4_r12.py').read_text();assert 'not np.issubdtype(a.dtype,np.number) or bool(np.isfinite(a).all())' in source
