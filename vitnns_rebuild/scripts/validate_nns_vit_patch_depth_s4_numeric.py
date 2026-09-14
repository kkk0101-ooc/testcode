#!/usr/bin/env python3
"""Frozen-environment NumPy/Torch numeric validator for S4 arrays."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s4_r3_20260915"
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s4 as s4  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def record_by_name(terminal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {Path(item["path"]).stem: item for item in terminal["payload"].get("arrays", [])}


def load_array(record: dict[str, Any], shape: tuple[int, ...], dtype: str) -> np.ndarray:
    path = REPO / record["path"]
    require(path.is_file(), "missing large array: " + record["path"])
    require(s4.sha_file(path) == record["sha256"], "large array SHA drift: " + record["path"])
    value = np.load(path, allow_pickle=False)
    require(value.shape == shape and str(value.dtype) == dtype, "large array shape/dtype drift: " + record["path"])
    if np.issubdtype(value.dtype, np.floating):
        require(np.isfinite(value).all(), "large array nonfinite: " + record["path"])
    require(s4.array_digest(value) == record["canonical_array_digest"], "large array canonical digest drift: " + record["path"])
    return value


def validate_final() -> dict[str, Any]:
    import torch
    require(os.path.realpath(sys.executable) == "/data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns/bin/python", "numeric validator interpreter drift")
    schedule = json.loads((REPO / s4.SCHEDULE_REL).read_text())["payload"]["records"]
    audit_indices = set(json.loads((REPO / s4.SCHEDULE_REL).read_text())["payload"]["audit_subset_schedule_indices"])
    nns_terminals = sorted((ROOT / "unit_terminals/nns_image").glob("unit-*.json"))
    require(len(nns_terminals) == 1744, "numeric validator requires 1744 NNS terminals")
    array_count = 0
    valid_nns = 0
    quadrature_counts: dict[str, int] = {}
    p1_valid_blocks = 0
    p1_raw_draw_count = 0
    p2_matched_coordinates = 0
    audit_count = 0
    max_head_sum_error = 0.0
    max_residual_error = 0.0
    max_scalar_error = 0.0
    for path in nns_terminals:
        terminal = json.loads(path.read_text())
        require(s4.verify_signed(terminal), "NNS terminal digest drift")
        if terminal["status"] != "VALID_NNS":
            continue
        valid_nns += 1
        payload = terminal["payload"]
        index = int(payload["schedule_index"])
        records = record_by_name(terminal)
        scores = load_array(records["source_scores"], (4, 12, 196), "float32"); array_count += 1
        occupancy = load_array(records["occupancy"], (4, 196), "float32"); array_count += 1
        _ = load_array(records["block_features"], (12, 196, 768), "float32"); array_count += 1
        permutations = load_array(records["p1_permutation_indices"], (12, 64, 196), "int16"); array_count += 1
        seed_digests = load_array(records["p1_seed_digests"], (12, 64), "|S64"); array_count += 1
        numerators = load_array(records["p1_numerators"], (12, 65), "float64"); array_count += 1
        denominators = load_array(records["p1_denominators"], (12,), "float64"); array_count += 1
        positive_mass = load_array(records["p1_positive_mass"], (12,), "float64"); array_count += 1
        raw = load_array(records["p1_raw_alignment"], (12,), "float64"); array_count += 1
        draws = load_array(records["p1_permutation_alignment_draws"], (12, 64), "float64"); array_count += 1
        flags = load_array(records["p1_valid_blocks"], (12,), "uint8"); array_count += 1
        require(np.array_equal(scores[1], np.maximum(scores[0], 0.0)) and np.array_equal(scores[2], np.minimum(scores[0], 0.0)) and np.array_equal(scores[3], np.abs(scores[0])), "source score signed bookkeeping drift")
        np.testing.assert_allclose(occupancy.astype(np.float64).sum(0), np.ones(196), atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(positive_mass, denominators)
        np.testing.assert_allclose(denominators, scores[1].astype(np.float64).sum(1), rtol=1e-6, atol=1e-8)
        for block in range(12):
            for repeat in range(64):
                permutation = permutations[block, repeat]
                require(hashlib_sha(permutation.tobytes()) == seed_digests[block, repeat].decode(), "P1 stored permutation digest drift")
                for stratum in range(7):
                    members = np.flatnonzero(s4.POSITION_STRATA == stratum)
                    require(sorted(permutation[members].tolist()) == members.tolist(), "P1 permutation crossed strata")
            if flags[block]:
                expected_observed = float(np.dot(scores[1, block].astype(np.float64), occupancy[3].astype(np.float64)))
                require(math.isclose(numerators[block, 0], expected_observed, rel_tol=1e-10, abs_tol=1e-10), "P1 numerator drift")
                np.testing.assert_allclose(raw[block], numerators[block, 0] / denominators[block], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(draws[block], numerators[block, 1:] / denominators[block], rtol=1e-12, atol=1e-12)
                p1_valid_blocks += 1
                p1_raw_draw_count += 64
        plan_record = payload["p2_plan"]["record"]
        require(s4.sha_file(REPO / plan_record["path"]) == plan_record["sha256"], "P2 plan SHA drift")
        plan = json.loads((REPO / plan_record["path"]).read_text())
        claimed = plan.pop("canonical_digest")
        require(claimed == s4.canonical_digest(plan), "P2 plan canonical digest drift")
        require(len(plan["coordinates"]) == 36, "P2 plan coordinate count drift")
        for coordinate in plan["coordinates"]:
            if coordinate["status"] != "MATCHED":
                continue
            selected_counts = np.bincount(s4.POSITION_STRATA[coordinate["selected"]], minlength=7)
            tolerance = max(4, math.ceil(0.25 * coordinate["selected_boundary_complexity"]))
            require(len(coordinate["controls"]) == 4, "P2 control count drift")
            for control in coordinate["controls"]:
                require(np.array_equal(np.bincount(s4.POSITION_STRATA[control["indices"]], minlength=7), selected_counts), "P2 strata mismatch")
                require(control["boundary_absolute_error"] <= tolerance, "P2 boundary mismatch")
            p2_matched_coordinates += 1
        nns = payload["nns"]
        require(nns["closure"]["status"] == "PASS" and nns["blocks"] == list(range(12)), "NNS closure/depth drift")
        require(nns["block11_matrix_zero_mass_explicit"] is True and nns["block11_cls_routing_preserved"] is True, "block 11 propagation drift")
        selected_steps = str(nns["selected_steps"]); quadrature_counts[selected_steps] = quadrature_counts.get(selected_steps, 0) + 1
        max_head_sum_error = max(max_head_sum_error, float(nns["closure"]["head_sum_max_abs_error"]))
        max_residual_error = max(max_residual_error, float(nns["closure"]["branch_cell_max_abs_error"]))
        max_scalar_error = max(max_scalar_error, float(nns["closure"]["scalar_map_matrix_max_abs_error"]))
        if index in audit_indices:
            matrices = load_array(records["audit_first_attention_T"], (12, 196, 196), "float32"); array_count += 1
            classes = load_array(records["audit_first_attention_C"], (12, 196), "float32"); array_count += 1
            recomputed = matrices.astype(np.float64).sum(2) + classes.astype(np.float64)
            np.testing.assert_allclose(recomputed, scores[0].astype(np.float64), rtol=1e-5, atol=1e-5)
            require(float(np.abs(matrices[11]).sum(dtype=np.float64)) == 0.0, "audit block 11 matrix not exact zero")
            audit_count += 1
    occlusion_terminals = sorted(ROOT.glob("unit_terminals/occlusion_*/*.json"))
    require(len(occlusion_terminals) == 3488, "occlusion terminal count drift")
    valid_occlusion = 0
    for path in occlusion_terminals:
        terminal = json.loads(path.read_text())
        require(s4.verify_signed(terminal), "occlusion terminal digest drift")
        if terminal["status"] != "VALID_OCCLUSION":
            continue
        valid_occlusion += 1
        record = record_by_name(terminal)
        require(len(record) == 1, "occlusion array record count drift")
        _ = load_array(next(iter(record.values())), (36, 5), "float32"); array_count += 1
        require(terminal["payload"]["coordinate_plan_shared_between_operators"] is True and terminal["payload"]["operator_selected_from_logit_drop"] is False, "occlusion outcome-independent contract drift")
    repeat = json.loads((ROOT / "deterministic_single_multi_gpu_repeatability_report.json").read_text())
    require(repeat["status"] == "PASS" and repeat["payload"]["all_byte_identical"] is True, "repeatability evidence drift")
    return {
        "status": "PASS", "python_executable_realpath": os.path.realpath(sys.executable),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "NNS_terminal_count": len(nns_terminals), "VALID_NNS_count": valid_nns,
        "validated_array_count": array_count, "quadrature_selected_step_counts": dict(sorted(quadrature_counts.items())),
        "P1_valid_block_count": p1_valid_blocks, "P1_raw_permutation_draw_count": p1_raw_draw_count,
        "P2_matched_coordinate_count": p2_matched_coordinates, "VALID_occlusion_count": valid_occlusion,
        "audit_subset_validated_count": audit_count, "audit_subset_required_count": 64,
        "max_head_sum_abs_error": max_head_sum_error, "max_residual_pair_abs_error": max_residual_error,
        "max_matrix_cell_scalar_abs_error": max_scalar_error, "single_multi_repeatability": "PASS",
        "cohort_level_inference_count": 0, "scientific_claim_decision": None,
    }


def hashlib_sha(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("final",))
    args = parser.parse_args()
    del args
    print(json.dumps(validate_final(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
