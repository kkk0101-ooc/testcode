from __future__ import annotations

import hashlib
import inspect
import json
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vitnns import stage_c_c10 as c10
from vitnns import stage_c_c10_stats as stats
from vitnns.stage_c_config import load_stage_c_config


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/阶段C_protocol.md"
PERTURBATIONS = (
    "translate_right_down_4",
    "center_crop_216_resize_224",
    "gaussian_rgb_std_0_01",
)
EXPLORATORY_WINNERS = ((4, 2), (6, 10), (11, 2))


def _status(value):
    return value["status"] if isinstance(value, dict) else value.status


def _summary(
    *, median=0.60, positive_count=15, lower=0.20, upper=0.80, n=15
):
    return {
        "scope": "holdout",
        "n": n,
        "median": median,
        "positive_count": positive_count,
        "bootstrap_95_percent_ci": {"lower": lower, "upper": upper},
    }


def _spatial_rows(*, passing=15):
    rows = []
    for index in range(15):
        good = index < passing
        rows.append({
            "sample_id": f"holdout-{index:02d}",
            "s_patch": 0.01 if good else 0.009,
            "positive_patch_evidence_fraction_of_token_l1": (
                0.01 if good else 0.009
            ),
            "rho": 0.05 if good else 0.049,
        })
    return rows


def _stability(**overrides):
    result = {name: _summary() for name in PERTURBATIONS}
    for name, summary in overrides.items():
        result[name] = summary
    return result


def _row(sample_id, role, *, value=1.0, status="OK", **extra):
    return {
        "sample_id": sample_id,
        "role": role,
        "metric_id": "synthetic_metric",
        "method_id": "synthetic_method",
        "config_id": "block_0_head_9",
        "value": value,
        "status": status,
        **extra,
    }


# ---------------------------------------------------------------------------
# Frozen image-cluster bootstrap and summaries
# ---------------------------------------------------------------------------


def test_bootstrap_seed_has_frozen_sha256_golden_and_namespace():
    metric_id = "delta_fid_fine4"
    expected = int.from_bytes(
        hashlib.sha256(
            b"stage-c-bootstrap|20260821|delta_fid_fine4"
        ).digest()[:8],
        "big",
        signed=False,
    )

    assert expected == 5_888_273_262_938_456_719
    assert stats.bootstrap_seed(metric_id) == expected
    assert stats.metric_seed(metric_id) == expected
    assert stats.bootstrap_seed(metric_id) == stats.bootstrap_seed(metric_id)
    assert stats.bootstrap_seed("delta_fid_patch") != expected


@pytest.mark.parametrize("bad", ["", None, 1, b"metric", True])
def test_bootstrap_seed_rejects_noncanonical_metric_ids(bad):
    with pytest.raises((TypeError, ValueError)):
        stats.bootstrap_seed(bad)


def test_paired_median_bootstrap_is_exact_15_by_10000_and_linear_quantile():
    metric_id = "delta_fid_fine4"
    values = np.arange(15, dtype=np.float64) / 100.0
    sample_ids = [f"holdout-{index:02d}" for index in range(15)]

    result = stats.paired_median_bootstrap(
        metric_id, values, sample_ids=sample_ids, return_distribution=True
    )

    distribution = np.asarray(result["distribution"], dtype=np.float64)
    assert result["metric_id"] == metric_id
    assert result["seed"] == stats.bootstrap_seed(metric_id)
    assert result["resamples"] == 10_000
    assert result["cluster_size"] == 15
    assert result["cluster_unit"] == "image"
    assert result["ordered_sample_ids_sha256"] == hashlib.sha256(
        json.dumps(
            sample_ids, sort_keys=False, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert result["statistic"] == "paired_median"
    assert result["quantile_method"] == "linear"
    assert result["confidence_level"] == 0.95
    assert distribution.shape == (10_000,)

    expected_ci = np.quantile(distribution, [0.025, 0.975], method="linear")
    assert distribution[:12].tolist() == pytest.approx([
        0.06, 0.06, 0.09, 0.07, 0.10, 0.07,
        0.06, 0.06, 0.10, 0.07, 0.04, 0.05,
    ])
    assert expected_ci.tolist() == pytest.approx([0.04, 0.11])
    assert result["ci_lower"] == pytest.approx(expected_ci[0])
    assert result["ci_upper"] == pytest.approx(expected_ci[1])
    expected_bytes = np.ascontiguousarray(distribution, dtype="<f8").tobytes()
    assert result["distribution_dtype"] == "float64_le"
    assert result["distribution_sha256"] == hashlib.sha256(
        expected_bytes
    ).hexdigest()
    assert result["distribution_sha256"] == (
        "0105f4130ea75383045ee7cf0ca9b007a87e9377fb39cb0a7d415dc3e17778dd"
    )


def test_bootstrap_uses_one_paired_draw_matrix_not_independent_sides():
    metric_id = "paired-side-golden"
    left = np.linspace(0.10, 0.24, 15)
    right = left - np.linspace(-0.03, 0.04, 15)
    paired_values = left - right
    result = stats.paired_median_bootstrap(
        metric_id, paired_values, return_distribution=True
    )

    rng = np.random.Generator(np.random.PCG64(stats.bootstrap_seed(metric_id)))
    draws = rng.integers(0, 15, size=(10_000, 15), endpoint=False)
    expected = np.median(left[draws] - right[draws], axis=1)
    observed = np.asarray(result["distribution"])

    assert observed == pytest.approx(expected)
    assert hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()
    independently_resampled = (
        np.median(left[draws], axis=1)
        - np.median(right[np.roll(draws, 1, axis=0)], axis=1)
    )
    assert not np.array_equal(observed, independently_resampled)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resamples": 9_999}, "10000"),
        ({"cluster_size": 14}, "15"),
        ({"sample_ids": ["same"] * 15}, "unique"),
    ],
)
def test_bootstrap_rejects_contract_drift(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        stats.paired_median_bootstrap(
            "metric", np.arange(15, dtype=float), **kwargs
        )


@pytest.mark.parametrize(
    "values",
    [
        np.arange(14, dtype=float),
        np.arange(16, dtype=float),
        [*map(float, range(14)), None],
        [*map(float, range(14)), float("nan")],
        [*map(float, range(14)), float("inf")],
        [*map(float, range(14)), True],
    ],
)
def test_bootstrap_never_drops_or_imputes_invalid_image_clusters(values):
    with pytest.raises((TypeError, ValueError)):
        stats.paired_median_bootstrap("metric", values)


def test_summary_uses_sample_sd_ddof_1_and_bootstraps_holdout_only():
    holdout_values = np.arange(1.0, 16.0)
    holdout_ids = [f"holdout-{index:02d}" for index in range(15)]
    holdout = stats.summarize_values(
        "metric", holdout_values, scope="holdout", sample_ids=holdout_ids
    )

    assert holdout["unit"] == "image"
    assert holdout["n"] == 15
    assert holdout["sample_ids"] == holdout_ids
    assert holdout["mean"] == pytest.approx(float(holdout_values.mean()))
    assert holdout["median"] == pytest.approx(float(np.median(holdout_values)))
    assert holdout["standard_deviation_sample_ddof_1"] == pytest.approx(
        float(holdout_values.std(ddof=1))
    )
    assert holdout["standard_deviation_sample_ddof_1"] != pytest.approx(
        float(holdout_values.std(ddof=0))
    )
    assert holdout["bootstrap"]["resamples"] == 10_000
    assert holdout["bootstrap"]["cluster_size"] == 15
    assert set(holdout["bootstrap_95_percent_ci"]) == {"lower", "upper"}

    calibration = stats.summarize_values(
        "metric", range(5), scope="calibration"
    )
    all_20 = stats.summarize_values(
        "metric", range(20), scope="all_20"
    )
    assert calibration["n"] == 5
    assert all_20["n"] == 20
    assert calibration["bootstrap"] is None
    assert calibration["bootstrap_95_percent_ci"] is None
    assert all_20["bootstrap"] is None
    assert all_20["bootstrap_95_percent_ci"] is None


@pytest.mark.parametrize(
    ("scope", "values"),
    [
        ("calibration", range(4)),
        ("calibration", range(6)),
        ("holdout", range(14)),
        ("holdout", range(16)),
        ("all_20", range(19)),
        ("all_20", range(21)),
    ],
)
def test_summary_enforces_exact_5_15_20_role_denominators(scope, values):
    with pytest.raises((TypeError, ValueError)):
        stats.summarize_values("metric", values, scope=scope)


@pytest.mark.parametrize("scope", ["audit", "all", "exploratory", ""])
def test_summary_rejects_unregistered_scope(scope):
    with pytest.raises((TypeError, ValueError)):
        stats.summarize_values("metric", range(15), scope=scope)


# ---------------------------------------------------------------------------
# Effect, spatial/stability, and unique five-level decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (_summary(median=0.02, positive_count=12, lower=1e-12, upper=0.02),
         "PASS_EFFECT"),
        (_summary(median=0.50, positive_count=11, lower=0.10, upper=0.90),
         "FAIL_EFFECT"),
        (_summary(median=0.50, positive_count=15, lower=-0.01, upper=0.019999),
         "FAIL_EFFECT"),
        (_summary(median=0.019999, positive_count=15, lower=0.01, upper=0.02),
         "UNCERTAIN_EFFECT"),
        (_summary(median=0.02, positive_count=12, lower=0.0, upper=0.02),
         "UNCERTAIN_EFFECT"),
        (_summary(median=0.02, positive_count=12, lower=-0.01, upper=0.02),
         "UNCERTAIN_EFFECT"),
    ],
)
def test_effect_pass_fail_uncertain_and_exact_boundaries(summary, expected):
    result = stats.evaluate_effect(summary)
    assert _status(result) == expected
    if isinstance(result, dict):
        assert result["median"] == summary["median"]
        assert result["positive_count"] == summary["positive_count"]


def test_effect_rejects_incomplete_or_wrong_scope_summary():
    with pytest.raises((TypeError, ValueError)):
        stats.evaluate_effect(_summary(n=14))
    wrong_scope = _summary()
    wrong_scope["scope"] = "calibration"
    with pytest.raises((TypeError, ValueError)):
        stats.evaluate_effect(wrong_scope)


def test_spatial_pass_includes_all_three_registered_perturbations():
    result = stats.evaluate_spatial(_spatial_rows(passing=12), _stability())

    assert _status(result) == "PASS_SPATIAL"
    assert result["joint_pass_count"] == 12
    assert set(result["stability"]) == set(PERTURBATIONS)
    assert {
        _status(value) for value in result["stability"].values()
    } == {"PASS_STABILITY"}


def test_spatial_joint_support_uses_same_images_and_exact_inclusive_thresholds():
    rows = _spatial_rows(passing=15)
    # Each individual condition still has 14 passes, but their intersection has 12.
    rows[0]["s_patch"] = 0.009
    rows[1]["positive_patch_evidence_fraction_of_token_l1"] = 0.009
    rows[2]["rho"] = 0.049
    result = stats.evaluate_spatial(rows, _stability())
    assert result["joint_pass_count"] == 12
    assert _status(result) == "PASS_SPATIAL"

    rows[3]["rho"] = 0.049
    failed = stats.evaluate_spatial(rows, _stability())
    assert failed["joint_pass_count"] == 11
    assert _status(failed) == "FAIL_SPATIAL"


def test_spatial_retains_na_image_in_denominator_as_joint_failure():
    rows = _spatial_rows(passing=15)
    rows[0]["rho"] = None

    result = stats.evaluate_spatial(rows, _stability())

    assert result["sample_count"] == 15
    assert result["joint_pass_count"] == 14
    assert result["per_sample"][0]["contains_na"] is True
    assert result["per_sample"][0]["passes_joint_spatial_evidence"] is False


@pytest.mark.parametrize(
    ("positive_count", "expected"),
    [(11, "FAIL_STABILITY"), (12, "UNCERTAIN_STABILITY")],
)
def test_residual_stability_na_never_drops_or_imputes_images(
    positive_count, expected
):
    sample_ids = [f"holdout-{index:02d}" for index in range(15)]
    incomplete = {
        "metric_id": "residual", "scope": "holdout",
        "unit": "image", "confirmatory": True, "n": 15,
        "sample_ids": sample_ids, "status": "INCOMPLETE_VECTOR",
        "positive_count": positive_count, "na_count": 1,
        "median": None, "bootstrap_95_percent_ci": None,
        "bootstrap": None,
    }
    result = stats.evaluate_spatial(
        _spatial_rows(), _stability(translate_right_down_4=incomplete)
    )

    assert result["stability"]["translate_right_down_4"]["status"] == expected
    assert result["stability"]["translate_right_down_4"]["na_count"] == 1
    assert result["status"] == (
        "FAIL_SPATIAL" if expected == "FAIL_STABILITY"
        else "UNCERTAIN_SPATIAL"
    )


@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_each_registered_perturbation_can_fail_the_overall_spatial_gate(perturbation):
    summaries = _stability(
        **{perturbation: _summary(positive_count=11, median=0.9, lower=0.8, upper=1.0)}
    )
    result = stats.evaluate_spatial(_spatial_rows(), summaries)

    assert _status(result) == "FAIL_SPATIAL"
    assert _status(result["stability"][perturbation]) == "FAIL_STABILITY"
    assert set(result["stability"]) == set(PERTURBATIONS)


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (_summary(median=0.50, positive_count=12, lower=1e-12, upper=0.50),
         "PASS_STABILITY"),
        (_summary(median=0.9, positive_count=11, lower=0.8, upper=1.0),
         "FAIL_STABILITY"),
        (_summary(median=0.499, positive_count=15, lower=0.1, upper=0.499),
         "FAIL_STABILITY"),
        (_summary(median=0.499, positive_count=15, lower=0.1, upper=0.50),
         "UNCERTAIN_STABILITY"),
        (_summary(median=0.50, positive_count=12, lower=0.0, upper=0.50),
         "UNCERTAIN_STABILITY"),
        (_summary(median=0.50, positive_count=12, lower=-0.1, upper=0.50),
         "UNCERTAIN_STABILITY"),
    ],
)
def test_stability_pass_fail_uncertain_and_exact_boundaries(summary, expected):
    summaries = _stability(translate_right_down_4=summary)
    result = stats.evaluate_spatial(_spatial_rows(), summaries)
    observed = _status(result["stability"]["translate_right_down_4"] )
    assert observed == expected
    overall = {
        "PASS_STABILITY": "PASS_SPATIAL",
        "FAIL_STABILITY": "FAIL_SPATIAL",
        "UNCERTAIN_STABILITY": "UNCERTAIN_SPATIAL",
    }[expected]
    assert _status(result) == overall


def test_spatial_requires_exactly_three_registered_perturbations_and_15_rows():
    with pytest.raises((TypeError, ValueError)):
        stats.evaluate_spatial(_spatial_rows()[:-1], _stability())
    incomplete = _stability()
    incomplete.pop(PERTURBATIONS[-1])
    with pytest.raises((TypeError, ValueError)):
        stats.evaluate_spatial(_spatial_rows(), incomplete)
    extra = _stability(extra=_summary())
    with pytest.raises((TypeError, ValueError)):
        stats.evaluate_spatial(_spatial_rows(), extra)


@pytest.mark.parametrize(
    ("validity", "effect", "spatial", "cost", "expected"),
    [
        ("FAIL_VALIDITY", "PASS_EFFECT", "PASS_SPATIAL", "PASS_COST",
         "NO_GO_VALIDITY"),
        ("PASS_VALIDITY", "FAIL_EFFECT", "PASS_SPATIAL", "PASS_COST",
         "NO_GO_NOVELTY"),
        ("PASS_VALIDITY", "PASS_EFFECT", "FAIL_SPATIAL", "PASS_COST",
         "NO_GO_NOVELTY"),
        ("PASS_VALIDITY", "UNCERTAIN_EFFECT", "PASS_SPATIAL", "PASS_COST",
         "INCONCLUSIVE"),
        ("PASS_VALIDITY", "PASS_EFFECT", "UNCERTAIN_SPATIAL", "PASS_COST",
         "INCONCLUSIVE"),
        ("PASS_VALIDITY", "PASS_EFFECT", "PASS_SPATIAL", "FAIL_COST",
         "NO_GO_COST"),
        ("PASS_VALIDITY", "PASS_EFFECT", "PASS_SPATIAL", "PASS_COST",
         "GO"),
    ],
)
def test_exact_five_level_decision_truth_table(
    validity, effect, spatial, cost, expected
):
    assert stats.decide(validity, effect, spatial, cost) == expected


@pytest.mark.parametrize(
    ("validity", "effect", "spatial", "cost", "expected"),
    [
        ("FAIL_VALIDITY", "FAIL_EFFECT", "FAIL_SPATIAL", "FAIL_COST",
         "NO_GO_VALIDITY"),
        ("PASS_VALIDITY", "FAIL_EFFECT", "UNCERTAIN_SPATIAL", "FAIL_COST",
         "NO_GO_NOVELTY"),
        ("PASS_VALIDITY", "UNCERTAIN_EFFECT", "PASS_SPATIAL", "FAIL_COST",
         "INCONCLUSIVE"),
    ],
)
def test_decision_priority_dominates_lower_priority_flags(
    validity, effect, spatial, cost, expected
):
    assert stats.decide(validity, effect, spatial, cost) == expected


def test_exploratory_results_cannot_enter_primary_decision_api():
    signature = inspect.signature(stats.decide)
    assert tuple(signature.parameters) == (
        "validity_status", "effect_status", "spatial_status", "cost_status"
    )
    assert all(
        token not in signature.parameters
        for token in ("exploratory", "holm", "p_values", "winner")
    )
    primary = stats.decide(
        "PASS_VALIDITY", "PASS_EFFECT", "PASS_SPATIAL", "PASS_COST"
    )
    assert primary == "GO"


# ---------------------------------------------------------------------------
# Holm contract audit and exact three-member step-down
# ---------------------------------------------------------------------------


def test_frozen_holm_contract_is_fail_closed_and_does_not_invent_p_values():
    config = load_stage_c_config().to_dict()
    protocol_text = PROTOCOL.read_text(encoding="utf-8")

    result = stats.audit_holm_contract(config, protocol_text)

    assert result["status"] == "HOLM_BLOCKED_UNDERSPECIFIED"
    assert result["method_registration"] == (
        "holm_within_metric_family_alpha_0_05"
    )
    assert result["missing_definitions"]
    assert "family_members" in result["missing_definitions"]
    assert "p_value_method" in result["missing_definitions"]
    assert result["can_compute_raw_p_values"] is False
    assert result["can_perform_holm"] is False
    assert result["holm_performed"] is False
    assert result.get("raw_p_values", []) == []
    assert result.get("adjusted_p_values", []) == []
    assert result.get("reject", []) == []
    assert result.get("affects_primary_decision", False) is False
    assert result.get("acceptance_blocking", True) is True


def test_complete_holm_contract_requires_exact_family_three_winners_and_alpha():
    config = load_stage_c_config().to_dict()
    config["statistics"]["exploratory_holm_contract"] = {
        "test_statistic": "registered_synthetic_statistic",
        "null_hypothesis": "registered_synthetic_null",
        "alternative": "greater",
        "sidedness": "one_sided",
        "direction": "greater",
        "exact_family_members": [list(item) for item in EXPLORATORY_WINNERS],
        "p_value_method": "registered_synthetic_exact_method",
        "alpha": 0.05,
    }
    protocol = json.dumps(
        config["statistics"]["exploratory_holm_contract"], sort_keys=True
    )

    result = stats.audit_holm_contract(config, protocol)

    assert result["status"] == "HOLM_CONTRACT_COMPLETE"
    assert result["definitions"]["alpha"] == 0.05
    assert tuple(map(tuple, result["definitions"]["family_members"])) == (
        EXPLORATORY_WINNERS
    )
    assert len(result["definitions"]["family_members"]) == 3
    assert not result["missing_definitions"]
    assert not result["ambiguous_definitions"]
    assert result["can_compute_raw_p_values"] is True
    assert result["can_perform_holm"] is True


def test_holm_step_down_orders_three_members_and_stops_after_first_nonrejection():
    raw = {
        "block_4_head_2": 0.010,
        "block_6_head_10": 0.020,
        "block_11_head_2": 0.040,
    }

    records = stats.holm_step_down(raw, alpha=0.05)

    assert len(records) == 3
    assert [item["member"] for item in records] == [
        "block_4_head_2", "block_6_head_10", "block_11_head_2"
    ]
    assert [item["rank"] for item in records] == [1, 2, 3]
    assert [item["threshold"] for item in records] == pytest.approx(
        [0.05 / 3, 0.05 / 2, 0.05]
    )
    assert [item["adjusted_p_value"] for item in records] == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    assert [item["reject"] for item in records] == [True, True, True]

    stopped = stats.holm_step_down(
        {"block_4_head_2": 0.010, "block_6_head_10": 0.030,
         "block_11_head_2": 0.031},
        alpha=0.05,
    )
    assert [item["reject"] for item in stopped] == [True, False, False]


@pytest.mark.parametrize(
    "raw",
    [
        {"a": 0.01, "b": 0.02},
        {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04},
        {"a": 0.01, "b": -0.1, "c": 0.03},
        {"a": 0.01, "b": 1.1, "c": 0.03},
        {"a": 0.01, "b": float("nan"), "c": 0.03},
    ],
)
def test_holm_rejects_non_three_member_or_invalid_families(raw):
    with pytest.raises((TypeError, ValueError)):
        stats.holm_step_down(raw, alpha=0.05)


# ---------------------------------------------------------------------------
# Long-table preservation, role separation, source boundary, determinism
# ---------------------------------------------------------------------------


def test_partition_rows_keeps_exact_5_15_20_and_misclassified_rows():
    calibration = [
        _row(f"cal-{index}", "calibration", misclassified=index == 0)
        for index in range(5)
    ]
    holdout = [
        _row(f"hold-{index}", "holdout", misclassified=index in {2, 9})
        for index in range(15)
    ]

    scopes = c10.partition_rows_by_scope([*calibration, *holdout])

    assert set(scopes) == {"calibration", "holdout", "all_20"}
    assert len(scopes["calibration"]) == 5
    assert len(scopes["holdout"]) == 15
    assert len(scopes["all_20"]) == 20
    assert {row["sample_id"] for row in scopes["calibration"]} == {
        row["sample_id"] for row in calibration
    }
    assert {row["sample_id"] for row in scopes["holdout"]} == {
        row["sample_id"] for row in holdout
    }
    assert sum(row["misclassified"] for row in scopes["all_20"]) == 3


def test_reconcile_is_expected_left_join_and_never_drops_na_failure_or_missing():
    expected = ["ok", "legitimate-na", "misclassified", "failed", "missing"]
    observed = [
        {"row_key": "ok", "status": "OK", "value": 1.0},
        {"row_key": "legitimate-na", "status": "LEGITIMATE_NA",
         "value": None, "na_reason": "constant_left"},
        {"row_key": "misclassified", "status": "OK", "value": 2.0,
         "misclassified": True},
        {"row_key": "failed", "status": "UNIT_FAILED", "value": None},
    ]

    reconciled = c10.reconcile_expected_rows(expected, observed)

    assert [row["row_key"] for row in reconciled] == expected
    assert len(reconciled) == len(expected)
    assert reconciled[1]["status"] == "LEGITIMATE_NA"
    assert reconciled[1]["value"] is None
    assert reconciled[2]["misclassified"] is True
    assert reconciled[3]["status"] == "UNIT_FAILED"
    assert reconciled[4]["status"] == "SOURCE_MISSING"
    assert reconciled[4]["value"] is None


def test_reconcile_rejects_duplicate_and_extra_observations():
    with pytest.raises((TypeError, ValueError)):
        c10.reconcile_expected_rows(
            ["a"],
            [
                {"row_key": "a", "status": "OK", "value": 1.0},
                {"row_key": "a", "status": "OK", "value": 2.0},
            ],
        )
    with pytest.raises((TypeError, ValueError)):
        c10.reconcile_expected_rows(
            ["a"],
            [{"row_key": "b", "status": "OK", "value": 1.0}],
        )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("L0/result.json", "L0/result.json"),
        ("L1/clean.npy", "L1/clean.npy"),
        ("L1/maps/residual.npy", "L1/maps/residual.npy"),
    ],
)
def test_metric_source_path_accepts_only_l0_l1_numeric_evidence(path, expected):
    assert c10.validate_metric_source_path(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "L2/tensor.npy",
        "L3/display.png",
        "L1/display.npy",
        "L1/heatmap.png",
        "L1/source.JPEG",
        "../L1/clean.npy",
        "L1/../L0/result.json",
        "/tmp/L1/clean.npy",
        "C:\\tmp\\L1\\clean.npy",
    ],
)
def test_metric_source_path_rejects_l2_l3_display_images_and_escape(path):
    with pytest.raises((TypeError, ValueError)):
        c10.validate_metric_source_path(path)


def test_all_public_statistical_outputs_are_deterministic():
    values = np.linspace(-0.02, 0.08, 15)
    ids = [f"holdout-{index:02d}" for index in range(15)]

    first_bootstrap = stats.paired_median_bootstrap(
        "determinism", values, sample_ids=ids, return_distribution=True
    )
    second_bootstrap = stats.paired_median_bootstrap(
        "determinism", values, sample_ids=ids, return_distribution=True
    )
    first_summary = stats.summarize_values(
        "determinism", values, scope="holdout", sample_ids=ids
    )
    second_summary = stats.summarize_values(
        "determinism", values, scope="holdout", sample_ids=ids
    )
    first_holm = stats.holm_step_down({"c": 0.03, "a": 0.01, "b": 0.02})
    second_holm = stats.holm_step_down({"b": 0.02, "c": 0.03, "a": 0.01})

    assert first_bootstrap == second_bootstrap
    assert first_summary == second_summary
    assert first_holm == second_holm


def test_compact_summary_and_bootstrap_retain_hash_identity_without_values():
    sample_ids = [f"holdout-{index:02d}" for index in range(15)]
    values = np.linspace(-0.02, 0.12, 15)
    summary = stats.summarize_values(
        "compact-identity", values, scope="holdout", sample_ids=sample_ids
    )
    selected = [
        {
            "sample_id": sample_id,
            "row_key_sha256": hashlib.sha256(
                f"row-{index}".encode("ascii")
            ).hexdigest(),
            "status": "OK",
            "value": float(value),
            "na_reason": None,
        }
        for index, (sample_id, value) in enumerate(zip(sample_ids, values))
    ]

    compact, bootstrap = c10._compact_summary_record(
        summary, selected, {"metric_family": "synthetic"},
        Counter({"OK": 15}), Counter(),
    )

    assert bootstrap is not None
    assert compact["metric_id"] == bootstrap["metric_id"]
    assert compact["bootstrap_record_id"] == bootstrap["metric_id"]
    assert bootstrap["ordered_sample_ids_sha256"] == (
        summary["bootstrap"]["ordered_sample_ids_sha256"]
    )
    assert compact["ordered_sample_ids_sha256"] == (
        bootstrap["ordered_sample_ids_sha256"]
    )
    assert bootstrap["distribution_sha256"] == (
        summary["bootstrap"]["distribution_sha256"]
    )
    assert bootstrap["draw_indices_sha256"] == (
        summary["bootstrap"]["draw_indices_sha256"]
    )
    assert compact["per_sample_values_ref"] == "per_sample.csv"
    assert compact["sample_registry_ref"] == "sample_registry.holdout"
    assert bootstrap["sample_registry_ref"] == "sample_registry.holdout"
    assert all(name not in compact for name in ("sample_ids", "values", "per_sample", "bootstrap"))
    assert "sample_ids" not in bootstrap


def test_l1_applicability_retains_all_seven_negative_view_na_rows():
    def plan_unit(method_id, digest):
        unit = {
            "key": {
                "sample_id": "holdout-00",
                "method_id": method_id,
                "config_id": "block_00_head_09",
                "block_index": 0,
                "head_index": 9,
                "target_ref": "clean_predicted_class_raw_logit",
                "canonical_digest": digest,
            },
            "sample": {
                "sample_id": "holdout-00",
                "role": "holdout",
                "audit": False,
            },
            "comparison_line": "value_path",
            "fidelity_eligible": True,
            "structural_reference_only": False,
        }
        return unit

    def record(unit, index, array, semantics, marker):
        key = unit["key"]
        return {
            "plan_index": index,
            "unit_key": dict(key),
            "state": "succeeded",
            "trusted_by_batch_validator": True,
            "attempt_path": (
                f"units/{key['canonical_digest']}/attempts/attempt-000001"
            ),
            "hashes": {
                "status_sha256": marker * 64,
                "selected_status_sha256": marker * 64,
                "manifest_sha256": "2" * 64,
                "provenance_sha256": "3" * 64,
                "result_sha256": "4" * 64,
                "unit_identity_sha256": key["canonical_digest"],
            },
            "scalars": {
                "sign_semantics": semantics,
                "prediction": {"clean_top1_correct": True},
            },
            "resources": {},
            "audit": False,
            "failure": None,
            "validation_errors": [],
            "l1_clean_array": np.asarray(array, dtype=np.float32),
            "l1_clean_binding": {"sha256": marker * 64},
        }

    nns = plan_unit(c10.NNS_METHOD_ID, "a" * 64)
    direct = plan_unit(c10.DIRECT_METHOD_ID, "b" * 64)
    records = [
        record(
            nns, 0, np.linspace(-1.0, 1.0, 196).reshape(14, 14),
            "signed", "a",
        ),
        record(direct, 1, np.ones((14, 14)), "nonnegative", "b"),
    ]

    rows = c10._l1_derived_rows(records, [nns, direct])
    negative = [
        row for row in rows
        if row["method_id"] == c10.NNS_METHOD_ID
        and row["metric_family"] == "clean_spatial_structure_similarity"
        and row["view"] == "negative_magnitude"
    ]

    assert len(rows) == c10._expected_l1_derived_row_count(records) == 19
    assert len(negative) == 7
    assert Counter(row["metric_id"].split("|", 1)[0] for row in negative) == {
        "pearson": 1,
        "spearman": 1,
        "jensen_shannon_divergence": 1,
        "hotspot_jaccard_top_10_percent": 1,
        "hotspot_jaccard_top_20_percent": 1,
        "kl": 2,
    }
    assert {row["status"] for row in negative} == {"NA"}
    assert {row["na_reason"] for row in negative} == {
        "unsupported_sign_semantics"
    }
    assert {row["invalidating"] for row in negative} == {False}


def test_required_metric_coverage_audit_is_dynamic_and_sample_complete():
    sample_ids = [f"sample-{index:02d}" for index in range(20)]
    rows = []
    for metric_name, patterns in c10._REQUIRED_METRIC_COVERAGE.items():
        for pattern_index, pattern in enumerate(patterns):
            family, metric_spec = pattern.split("/", 1)
            metric_base, separator, qualifier = metric_spec.partition(":")
            if metric_base == "*":
                metric_base = "endpoint_parity_error"
            for sample_index, sample_id in enumerate(sample_ids):
                rows.append({
                    "sample_id": sample_id,
                    "metric_family": family,
                    "metric_id": metric_base,
                    "curve_direction": qualifier if separator else None,
                    "component": None,
                    "status": "OK",
                    "row_key_sha256": hashlib.sha256(
                        f"{metric_name}|{pattern_index}|{sample_index}".encode("ascii")
                    ).hexdigest(),
                })
    config = SimpleNamespace(
        metrics={"required": list(c10._REQUIRED_METRIC_COVERAGE)}
    )
    registry = {
        "calibration": sample_ids[:5],
        "holdout": sample_ids[5:],
        "all_20": sample_ids,
    }

    audit = c10._metric_coverage_audit(rows, config, registry)

    assert audit["status"] == "PASS"
    assert audit["required_metric_count"] == len(c10._REQUIRED_METRIC_COVERAGE)
    assert {record["required_metric"] for record in audit["records"]} == set(
        c10._REQUIRED_METRIC_COVERAGE
    )

    with pytest.raises(c10.StageC10IntegrityError, match="coverage is incomplete"):
        c10._metric_coverage_audit(rows[1:], config, registry)

    exact_rows = [
        {**row, "plan_index": index, "method_id": "method",
         "reference_method_id": None, "config_id": "config",
         "source_unit_key_sha256": hashlib.sha256(
             f"unit-{index}".encode("ascii")
         ).hexdigest(), "source_kind": "c9_l0_scalar",
         "source_field_path": f"/metric/{index}"}
        for index, row in enumerate(rows)
    ]
    exact_coordinates = [c10._applicability_coordinate(row) for row in exact_rows]
    exact_audit = c10._metric_coverage_audit(
        exact_rows, config, registry, exact_coordinates
    )
    assert exact_audit["exact_applicability"]["coordinate_count"] == len(
        exact_rows
    )
    tampered = list(exact_rows)
    tampered[0] = {**tampered[0], "config_id": "wrong-config"}
    with pytest.raises(c10.StageC10IntegrityError, match="applicability coordinate"):
        c10._metric_coverage_audit(
            tampered, config, registry, exact_coordinates
        )


def test_c10_modules_do_not_import_or_call_model_image_gpu_backends():
    forbidden_modules = {"torch", "torchvision", "timm", "PIL"}
    forbidden_calls = {
        "load_model", "load_state_dict", "run_execution_unit",
        "run_model", "forward", "backward", "Image.open",
    }

    for module in (stats, c10):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    calls.add(function.id)
                elif isinstance(function, ast.Attribute):
                    parts = [function.attr]
                    value = function.value
                    while isinstance(value, ast.Attribute):
                        parts.append(value.attr)
                        value = value.value
                    if isinstance(value, ast.Name):
                        parts.append(value.id)
                    calls.add(".".join(reversed(parts)))

        assert imported.isdisjoint(forbidden_modules)
        assert calls.isdisjoint(forbidden_calls)
        assert not any(
            call.split(".", 1)[0] in forbidden_modules for call in calls
        )
