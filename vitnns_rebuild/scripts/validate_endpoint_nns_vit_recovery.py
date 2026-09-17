#!/usr/bin/env python3
"""Validate endpoint NNS-ViT method identity and exploratory outputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MVP = REPO / "reports/nns_vit_endpoint_mvp_r2_20260917"
LEVEL4 = REPO / "reports/nns_vit_endpoint_level4_r1_20img_20260917"
R9 = REPO / "reports/nns_vit_unified_paper_figures_r9_20260917"
R9_RECEIPT_SHA256 = "319c69ec6aa080fadf82b658e4825aa2952d4bb4fa99a28c7fcad4954ab46683"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def signed(path: Path) -> dict:
    value = json.loads(path.read_text())
    digest = value.pop("canonical_digest")
    if digest != canonical(value):
        raise AssertionError(f"canonical digest mismatch: {path}")
    value["canonical_digest"] = digest
    return value


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    assert sha256(R9 / "terminal_receipt.json") == R9_RECEIPT_SHA256

    mvp = signed(MVP / "pilot_report.json")
    assert mvp["status"] == "PASS_LEVEL_3"
    assert mvp["decision"]["endpoint_equals_gradient_times_input_max_abs"] == 0.0
    assert mvp["decision"]["five_hundred_image_rerun_authorized"] is False
    assert max(
        mvp["decision"]["module_additivity"][method][key]
        for method in ("endpoint", "gradient_times_input", "path")
        for key in ("first_residual_max_abs", "second_residual_max_abs")
    ) < 1e-6
    mvp_manifest = signed(MVP / "artifact_manifest.json")
    for record in mvp_manifest["files"]:
        path = MVP / record["path"]
        assert path.is_file() and path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]

    protocol = signed(LEVEL4 / "protocol.json")
    schedule = signed(LEVEL4 / "schedule.json")
    summary = signed(LEVEL4 / "summary.json")
    assert protocol["image_count"] == 20
    assert protocol["gpu_execution"]["worker_count"] == 8
    assert protocol["gpu_execution"]["physical_indices"] == list(range(8))
    assert protocol["gpu_execution"]["dynamic_atomic_claims"] is True
    assert protocol["five_hundred_image_rerun_authorized"] is False
    assert len(schedule["records"]) == 20
    assert schedule["records_digest"] == canonical(schedule["records"])
    for record in schedule["records"]:
        path = REPO / record["path"]
        assert path.is_file() and sha256(path) == record["jpeg_sha256"]

    results = [signed(path) for path in sorted((LEVEL4 / "results").glob("*.json"))]
    assert len(results) == 20 and not list((LEVEL4 / "failures").glob("*.json"))
    assert sorted(row["schedule_index"] for row in results) == list(range(20))
    assert {row["status"] for row in results} == {"VALID"}
    assert set(Counter(row["physical_gpu_index"] for row in results)) == set(range(8))
    assert all(row["gpu_name"] == "NVIDIA H20" for row in results)
    for row in results:
        array_path = LEVEL4 / row["arrays"]["path"]
        assert sha256(array_path) == row["arrays"]["sha256"]
        with np.load(array_path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "endpoint_zero", "path_zero", "endpoint_black", "path_black",
                "endpoint_flip_inverse", "path_flip_inverse",
            }
            assert all(arrays[name].shape == (1, 3, 224, 224) for name in arrays.files)
            assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
        assert row["methods"]["ordinary_gradient"]["endpoint_identity_max_abs"] == 0.0
        assert row["methods"]["zero"]["path_steps"] in protocol["path_schedule"]
        assert row["methods"]["black"]["path_steps"] in protocol["path_schedule"]
        for method in ("endpoint_zero", "path_zero"):
            for unit in ("4", "16"):
                fidelity = row["fidelity"][method][unit]
                assert fidelity["status"] == "VALID"
                assert len(fidelity["insertion_curve"]) == len(fidelity["deletion_curve"]) == 21

    metrics = {
        "valid_count": len(results), "failure_count": 0,
        "endpoint_identity_max_abs": max(row["methods"]["ordinary_gradient"]["endpoint_identity_max_abs"] for row in results),
        "endpoint_path_zero_pearson_median": median([row["methods"]["zero"]["endpoint_vs_path"]["pearson"] for row in results]),
        "endpoint_path_zero_sign_median": median([row["methods"]["zero"]["endpoint_vs_path"]["sign_agreement"] for row in results]),
        "endpoint_reference_pearson_median": median([row["methods"]["reference_comparison"]["endpoint_zero_vs_black"]["pearson"] for row in results]),
        "path_reference_pearson_median": median([row["methods"]["reference_comparison"]["path_zero_vs_black"]["pearson"] for row in results]),
        "endpoint_flip_spearman_median": median([row["methods"]["flip_stability"]["endpoint"]["spearman"] for row in results]),
        "path_flip_spearman_median": median([row["methods"]["flip_stability"]["path"]["spearman"] for row in results]),
        "endpoint_to_path_runtime_ratio_median": median([row["runtime_seconds"]["endpoint_zero"] / row["runtime_seconds"]["path_zero"] for row in results]),
    }
    for unit in (4, 16):
        endpoint = [row["fidelity"]["endpoint_zero"][str(unit)]["combined_fidelity"] for row in results]
        path = [row["fidelity"]["path_zero"][str(unit)]["combined_fidelity"] for row in results]
        metrics[f"fidelity_{unit}"] = {
            "valid_pairs": 20, "endpoint_median": median(endpoint),
            "path_median": median(path),
            "endpoint_minus_path_median": median([a - b for a, b in zip(endpoint, path)]),
        }
    for key, value in metrics.items():
        recorded = summary["metrics"][key]
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                assert math.isclose(subvalue, recorded[subkey], abs_tol=1e-12, rel_tol=1e-12)
        elif isinstance(value, int):
            assert value == recorded
        else:
            assert math.isclose(value, recorded, abs_tol=1e-12, rel_tol=1e-12)
    assert summary["status"] == "PASS_EXPLORATORY_ANALYSIS"
    assert summary["decision"] == "NO_GO_CURRENT_ENDPOINT_PROTOCOL"
    assert summary["five_hundred_image_rerun_authorized"] is False
    assert summary["gate_checks"] == {
        "all_20_valid": True, "endpoint_identity": True,
        "fine_fidelity_positive": True, "fine_noninferiority_heuristic": False,
        "flip_stability": False, "reference_stability": False, "runtime": True,
    }
    print(json.dumps({
        "status": "PASS", "mvp_status": mvp["status"],
        "level4_status": summary["status"], "decision": summary["decision"],
        "valid_images": len(results), "gpu_indices": sorted(set(row["physical_gpu_index"] for row in results)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
