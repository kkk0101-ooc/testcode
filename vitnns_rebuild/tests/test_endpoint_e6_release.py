from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from vitnns.endpoint_e6_release import (
    EndpointE6Error, aggregate_decision, average_ranks, bootstrap_median_ci,
    decide_family, exact_two_sided_sign, holm_adjust, normalized_auc, spearman,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = json.loads((REPO / "config/nns_vit_endpoint/endpoint_e6_release_run_spec.json").read_text())
PLAN = json.loads((REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json").read_text())


def test_contract_counts_scope_and_authorization_are_exact():
    assert SPEC["authorization_token"] == "ENDPOINT_E6_R3_RECOVERY_AFTER_RELEASE_FORMAT_FAILURE_20260918"
    assert SPEC["stage"] == "ENDPOINT-E6-R3-RECOVERY"
    assert SPEC["outputs"]["formal_root"] == "reports/nns_vit_endpoint_e6_release_r3_20260918"
    assert SPEC["predecessors"]["endpoint_e6_r1_failure_sha256"] == "be15f9995ba7832a4aabcc1aed025f97be28d76558af28ba300fde4349f789fa"
    assert SPEC["execution"] == {
        "cpu_only": True, "cuda_visible_devices": "", "torch_timm_pil_import_allowed": False,
        "image_decode_allowed": False, "model_load_allowed": False,
        "new_empirical_compute_allowed": False, "source_recomputing_statistics": True,
        "output_root_must_not_exist": True,
    }
    assert len(PLAN["families"]) == SPEC["statistics"]["family_count"] == 16
    assert len(PLAN["tests"]) == len(PLAN["holm"]["test_ids"]) == SPEC["statistics"]["test_count"] == 28
    assert len({row["family_id"] for row in PLAN["families"]}) == 16
    assert {row["test_id"] for row in PLAN["tests"]} == set(PLAN["holm"]["test_ids"])


def test_average_ranks_and_spearman_handle_exact_ties():
    assert average_ranks([3, 1, 1, 2]).tolist() == [3.0, 0.5, 0.5, 2.0]
    assert spearman(range(8), range(8))["value"] == pytest.approx(1.0)
    assert spearman(range(8), list(reversed(range(8))))["value"] == pytest.approx(-1.0)
    assert spearman([1] * 8, range(8)) == {"value": None, "n": 8, "reason": "NA_CONSTANT_VECTOR"}
    with pytest.raises(EndpointE6Error, match="COMPONENT_COUNT"): spearman(range(7), range(7))


def test_auc_uses_frozen_actual_areas():
    assert normalized_auc([0, 1, 2, 4]) == pytest.approx(2.0)
    assert normalized_auc([0, -1, -2, -4]) == pytest.approx(2.0)


def test_bootstrap_is_deterministic_and_paired():
    columns = {"a": np.arange(40), "b": np.arange(40) + 3}
    first = bootstrap_median_ci(columns, namespace="ns", family_id="family")
    second = bootstrap_median_ci(columns, namespace="ns", family_id="family")
    assert first == second and first["results"]["b"]["estimate"] - first["results"]["a"]["estimate"] == 3


def test_exact_sign_and_holm_are_frozen_and_tie_deterministic():
    assert exact_two_sided_sign([1] * 10)["raw_p_value"] == pytest.approx(2 / 2**10)
    assert exact_two_sided_sign([1, -1, 0])["n_tie"] == 1
    ids = list(PLAN["holm"]["test_ids"]); raw = {key: 0.01 for key in ids}
    result = holm_adjust(raw, ids)
    assert [row["test_id"] for row in result] == sorted(ids)
    assert result[0]["adjusted_p_value"] == pytest.approx(0.28)
    with pytest.raises(EndpointE6Error, match="HOLM_FAMILY"): holm_adjust(dict(list(raw.items())[:-1]), ids)


def test_decision_order_is_invalid_inconclusive_supported_not_supported():
    family = {"exact_coverage": True, "n_valid": 400, "n_valid_min": 360, "ci_width_max": .3, "delta_min": .1}
    test = {"n_nonzero": 400, "min_nonzero_n": 300, "estimate": .2, "ci_lower": .1, "ci_upper": .25, "ci_width": .15, "holm_reject": True}
    assert decide_family(family, [test], invalid_reasons=["BROKEN"])[0] == "INVALID"
    low = deepcopy(family); low["n_valid"] = 359
    assert decide_family(low, [test])[0] == "INCONCLUSIVE"
    assert decide_family(family, [test])[0] == "SUPPORTED"
    failed = deepcopy(test); failed["estimate"] = .09
    assert decide_family(family, [failed])[0] == "NOT_SUPPORTED"
    assert aggregate_decision(["SUPPORTED"] * 4) == "SUPPORTED"
    assert aggregate_decision(["SUPPORTED", "NOT_SUPPORTED"]) == "NOT_SUPPORTED"
    assert aggregate_decision(["NOT_SUPPORTED", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert aggregate_decision(["INVALID", "SUPPORTED"]) == "INVALID"
