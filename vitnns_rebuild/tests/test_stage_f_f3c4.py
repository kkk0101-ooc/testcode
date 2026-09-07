from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_f_f3c4 as f3c4

REPO=Path(__file__).resolve().parents[1]

def test_f3c4_config_schema_and_exact_frozen_contract():
    config=f3c4.load_config(REPO); schema=json.loads((REPO/f3c4.SCHEMA_PATH).read_text())
    assert not list(Draft202012Validator(schema).iter_errors(config))
    assert tuple(x["id"] for x in config["methods"])==f3c4.METHODS
    assert config["formal_sample_count"]==50 and config["formal_statistical_unit"]=="image"
    assert config["statistics"]["primary_contrast"]==f3c4.PRIMARY_CONTRAST
    assert config["statistics"]["secondary_holm_family"]==list(f3c4.SECONDARY_CONTRASTS)
    assert config["rendering"]["overlay_alpha"]==0.55 and config["rendering"]["clipping"] is False

def test_false_pass_roster_is_exactly_47_unique():
    assert len(f3c4.FALSE_PASS_ATTACKS)==47 and len(set(f3c4.FALSE_PASS_ATTACKS))==47

@pytest.mark.parametrize("attack",f3c4.FALSE_PASS_ATTACKS)
def test_every_false_pass_attack_is_independently_rejected(attack: str):
    state={name:True for name in f3c4.FALSE_PASS_ATTACKS}; state[attack]=False
    assert f3c4.false_pass_reasons(state)==(attack,)

def test_numpy_statistics_match_frozen_reference_values():
    values=np.linspace(-0.25,0.4,50,dtype=np.float64); contrast=f3c4.PRIMARY_CONTRAST
    first=f3c4.bootstrap_mean_ci(values,contrast); second=f3c4.bootstrap_mean_ci(values,contrast)
    assert first==second and first["resamples"]==10000 and first["draw_shape"]==[10000,50] and first["quantile_method"]=="linear"
    p=f3c4.paired_sign_flip_p_value(values,contrast); assert p["resamples"]==100000 and p["two_sided"] is True and p["add_one_correction"] is True and 0<p["p_value"]<=1

def test_statistics_reject_wrong_n_resamples_and_contrast():
    with pytest.raises(f3c4.StageFF3C4Error): f3c4.bootstrap_mean_ci(np.ones(250),f3c4.PRIMARY_CONTRAST)
    with pytest.raises(f3c4.StageFF3C4Error): f3c4.bootstrap_mean_ci(np.ones(50),f3c4.PRIMARY_CONTRAST,resamples=9999)
    with pytest.raises(f3c4.StageFF3C4Error): f3c4.paired_sign_flip_p_value(np.ones(50),"posthoc")

def test_holm_uses_frozen_family_order_and_tie_break():
    raw={name:0.02 for name in f3c4.SECONDARY_CONTRASTS}; result=f3c4.holm_adjust(raw)
    assert [x["contrast"] for x in result]==list(f3c4.SECONDARY_CONTRASTS)
    assert [x["family_order"] for x in result]==[0,1,2]
    assert [x["adjusted_p_value"] for x in result]==[0.06,0.06,0.06]
    with pytest.raises(f3c4.StageFF3C4Error): f3c4.holm_adjust(dict(reversed(list(raw.items()))))

def test_image_summary_requires_50_and_ddof_one():
    values=np.arange(50,dtype=np.float64); result=f3c4._summary(values)
    assert result["n"]==50 and result["sample_sd_ddof_1"]==float(np.std(values,ddof=1))
    with pytest.raises(f3c4.StageFF3C4Error): f3c4._summary(values[:-1])

def test_cpu_source_has_no_execution_import_or_call_surface():
    source=(REPO/"src/vitnns/stage_f_f3c4.py").read_text()
    for pattern in ("from . import stage_f_f3c1","stage_f_f3c1_kernels","load_canonical_model","torch.load","model(","backward(","integrated_grad","evaluate_common_fidelity"):
        assert pattern not in source

def test_fine_grid_is_explicitly_not_computed():
    assert f3c4.load_config(REPO)["fine_grid_fidelity"]==f3c4.FINE_GRID_STATUS

def test_formal_rows_are_exactly_50_and_exclude_pilot_order44():
    rows=f3c4._formal_rows(REPO); assert len(rows)==50 and [x["formal_position"] for x in rows]==list(range(1,51))
    assert not ({x["selection_order"] for x in rows}&{44,416,350,438,411,337})

def test_plan_seal_or_outcome_access_guard_is_explicit():
    source=(REPO/"src/vitnns/stage_f_f3c4.py").read_text(); assert source.index("verify_plan_sealed(repo)",source.index("def reconcile_and_analyze"))<source.index("reconcile_terminal(terminal,expected)",source.index("def reconcile_and_analyze")); assert source.index("verify_plan_sealed(repo)",source.index("def render_formal"))<source.index("_image_crop_224(jpeg)",source.index("def render_formal"))

def test_atomic_completion_uses_replace_and_no_f4_surface():
    source=(REPO/"src/vitnns/stage_f_f3c4.py").read_text(); assert "os.replace(temporary,pending)" in source and "os.replace(pending,repo/COMPLETION_PATH)" in source and "run_f4" not in source
