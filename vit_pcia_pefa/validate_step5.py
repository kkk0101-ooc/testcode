#!/usr/bin/env python3
"""Independent prompt-to-artifact validator for Step 5."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PILOT = Path(os.environ.get("STEP5_PILOT_ROOT", ROOT / "step5_pilot_r8_20260921")).resolve()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest", None)
    need(isinstance(claimed, str) and digest(value) == claimed, "canonical digest: " + str(path))
    value["canonical_digest"] = claimed
    return value


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values); begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + end - 1) / 2
        for position in range(begin, end):
            result[order[position]] = rank
        begin = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    x, y = ranks(left), ranks(right)
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    x = [value - mx for value in x]; y = [value - my for value in y]
    denominator = math.sqrt(math.fsum(value * value for value in x) * math.fsum(value * value for value in y))
    return None if denominator == 0 else math.fsum(a * b for a, b in zip(x, y)) / denominator


def same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def same_value(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same_value(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return same_number(left, right)
    return left == right


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values); position = probability * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    return ordered[lower] if lower == upper else (
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )


def summary(values: list[float | None], namespace: str) -> dict[str, Any]:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return {"n": 0, "mean": None, "median": None, "bootstrap_95": None}
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
    generator = random.Random(seed)
    boot = [math.fsum(generator.choice(kept) for _ in kept) / len(kept) for _ in range(10000)]
    return {
        "n": len(kept), "mean": math.fsum(kept) / len(kept),
        "median": statistics.median(kept),
        "bootstrap_95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "bootstrap_replicates": 10000, "bootstrap_seed": seed,
    }


def main() -> int:
    required = {
        "protocol.json", "source_binding.json", "task_manifest.json", "freeze_receipt.json",
        "startup_snapshot.json", "launch_terminal.json", "resource_ledger.json",
        "validation_report.json", "claim_decisions.json", "terminal_receipt.json",
        "artifact_manifest.json",
        "claim_evidence_index.json", "reproduction_commands.json", "inheritance_receipt.json",
        "omega_audit_startup_snapshot.json", "omega_audit_terminal.json",
    }
    need(PILOT.is_dir() and all((PILOT / name).is_file() for name in required), "required deliverables")
    protocol = verify_signed(PILOT / "protocol.json")
    binding = verify_signed(PILOT / "source_binding.json")
    manifest = verify_signed(PILOT / "task_manifest.json")
    freeze = verify_signed(PILOT / "freeze_receipt.json")
    startup = verify_signed(PILOT / "startup_snapshot.json")
    launch = verify_signed(PILOT / "launch_terminal.json")
    resource = verify_signed(PILOT / "resource_ledger.json")
    validation = verify_signed(PILOT / "validation_report.json")
    claims = verify_signed(PILOT / "claim_decisions.json")
    terminal = verify_signed(PILOT / "terminal_receipt.json")
    artifacts = verify_signed(PILOT / "artifact_manifest.json")
    evidence_index = verify_signed(PILOT / "claim_evidence_index.json")
    reproduction = verify_signed(PILOT / "reproduction_commands.json")
    inheritance = verify_signed(PILOT / "inheritance_receipt.json")
    predecessor_binding = verify_signed(ROOT / "step5_pilot_r7_20260921/source_binding.json")
    omega_startup = verify_signed(PILOT / "omega_audit_startup_snapshot.json")
    omega_terminal = verify_signed(PILOT / "omega_audit_terminal.json")

    need(protocol["status"] == "FROZEN_PROSPECTIVE" and protocol["task_count"] == 16, "protocol identity")
    need(protocol["predecessor_run_id"] == "step5_pilot_r7_20260921"
         and protocol["predecessor_status"] == "TERMINAL_MATHEMATICAL_CLOSURE_FAIL",
         "recovery provenance")
    need(protocol["prior_outcome_known"] and protocol["recovery_trigger_outcome_independent"]
         and protocol["inherited_tasks"].startswith("14 canonical-valid"), "honest recovery")
    need(protocol["seed_namespace"] == "ENDPOINT_PCIA_PEFA|step5", "seed namespace")
    need(protocol["split"] == "calibration" and protocol["split_order_prefix"] == [0, 8], "cohort")
    need(protocol["models"] == ["deit_s16", "vit_b16"] and protocol["blocks"] == [0, 5, 11], "models blocks")
    need(protocol["integration_nodes"] == 512 and protocol["ih_nodes"] == 16, "quadrature")
    need(protocol["forward_dtype"] == "float32" and protocol["reduction_dtype"] == "float64", "precision")
    need(protocol["omega_scientific_score_dtype"] == "float32"
         and protocol["omega_zero_sum_gate_source"] == "independent_float64_complete_cut"
         and protocol["omega_zero_sum_audit_dtype"] == "float64"
         and protocol["omega_zero_sum_evaluation_components"] == ["identity", "attention"],
         "Omega precision responsibility")
    need(protocol["references"] == ["normalized_zero_primary", "raw_rgb_black_sensitivity"], "references")
    need(protocol["evaluation_access"] is False and protocol["step6_authorized"] is False, "stage scope")
    need(freeze["pilot_outcome_access_before_freeze"] is True
         and freeze["prior_outcome_known"] is True
         and freeze["recovery_trigger_outcome_independent"] is True
         and freeze["outcome_access_during_execution"] is False
         and freeze["step6_started"] is False, "honest prospective recovery freeze")
    need(freeze["protocol_digest"] == protocol["canonical_digest"]
         and freeze["task_manifest_digest"] == manifest["canonical_digest"]
         and freeze["source_binding_digest"] == binding["canonical_digest"], "freeze links")
    need(binding["pilot_outcome_access_before_freeze"] is True
         and binding["outcome_access_during_execution"] is False
         and binding["forbidden_evaluation_access_count"] == 0, "binding access")
    dirty_text = "\n".join(binding["producer_dirty_status"]) + ("\n" if binding["producer_dirty_status"] else "")
    need(binding["producer_dirty"] == bool(binding["producer_dirty_status"])
         and hashlib.sha256(dirty_text.encode()).hexdigest() == binding["producer_dirty_status_sha256"],
         "freeze-time dirty status")
    need(" passed in " in binding["preaccess_tests"] and "failed" not in binding["preaccess_tests"],
         "preaccess tests")
    for row in (*binding["files"], *binding["checkpoints"], *binding["pilot_images"]):
        path = REPO / row["path"]
        need(path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
             "source drift: " + row["path"])

    need(manifest["task_count"] == len(manifest["tasks"]) == 16, "task manifest count")
    identities = {(row["payload"]["model_id"], row["payload"]["split_order"]) for row in manifest["tasks"]}
    need(identities == {(model, index) for model in protocol["models"] for index in range(8)}, "task coverage")
    need(all(row["payload"]["split"] == "calibration" for row in manifest["tasks"]), "task split")
    need(manifest["evaluation_access"] is False and manifest["pilot_outcome_access_before_freeze"] is True
         and manifest["outcome_access_during_execution"] is False, "manifest access")
    need(startup["status"] == "PASS_STARTUP_GATE" and 1 <= startup["selected_gpu_count"] <= 4, "startup")
    need(startup["dynamic_global_queue"] and not startup["static_modulo_sharding"], "queue scheduling")
    need(startup["existing_pids_recorded"] and not startup["other_processes_terminated"], "shared host safety")
    need(all(row["name"].startswith("NVIDIA H20") and row["memory_free_mib"] >= 32768
             and row["utilization_percent"] <= 10 for row in startup["selected_gpus"]), "gpu qualification")
    need(launch["status"] == "PASS_TASK_EXECUTION" and launch["coverage"]["status"] == "PASS", "launch terminal")
    need(launch["coverage"]["task_count"] == launch["coverage"]["completed_count"] == 16, "launch coverage")
    need(omega_startup["status"] == "PASS_OMEGA_AUDIT_STARTUP_GATE"
         and omega_startup["dtype"] == "float64", "Omega audit startup")
    need(omega_terminal["status"] == "PASS_OMEGA_AUDIT_EXECUTION"
         and omega_terminal["coverage"]["status"] == "PASS"
         and omega_terminal["coverage"]["completed_count"] == 16, "Omega audit terminal")
    need(inheritance["status"] == "PASS_INHERITANCE"
         and len(inheritance["inherited_task_ids"]) == 14
         and inheritance["missing_tasks"] == ["vit_b16__02", "vit_b16__05"]
         and inheritance["scientific_values_changed"] is False, "r7 inheritance")
    inheritance_records = {row["task_id"]: row for row in inheritance["records"]}
    need(set(inheritance_records) == set(inheritance["inherited_task_ids"]), "inheritance records")

    connection = sqlite3.connect(PILOT / "queue.sqlite3")
    try:
        rows = connection.execute("SELECT task_id,state,attempt,result_path,result_digest,seed FROM tasks ORDER BY task_id").fetchall()
        seed_namespace = connection.execute("SELECT value FROM metadata WHERE key='seed_namespace'").fetchone()
    finally:
        connection.close()
    need(seed_namespace == (protocol["seed_namespace"],), "queue seed namespace")
    need(len(rows) == 16 and all(row[1] == "completed" and 1 <= row[2] <= 2 for row in rows), "queue terminals")
    need(all(int(row[5]) == int.from_bytes(
        hashlib.sha256(f"{protocol['seed_namespace']}|{row[0]}".encode()).digest()[:8], "big"
    ) for row in rows), "queue task seeds")
    documents = []
    expected_categories = {"residual", "attention_branch", "query", "key", "value", "head", "projection", "mlp"}
    expected_controls = set(protocol["negative_controls"])
    input_methods = set(protocol["contribution"]["methods"])
    for task_id, state, attempt, result_path, result_digest, task_seed in rows:
        result_file = PILOT / "results" / result_path
        document = verify_signed(result_file)
        need(document["canonical_digest"] == result_digest and document["task_id"] == task_id, "result identity")
        if task_id in inheritance_records:
            record = inheritance_records[task_id]
            need(sha256(result_file) == record["source_sha256"]
                 and document["canonical_digest"] == record["canonical_digest"], "inherited result digest")
        result = document["result"]; documents.append(result)
        need(result["status"] == "PASS_TASK" and result["split"] == "calibration", "task status scope")
        need(set(result["block_results"]) == {"0", "5", "11"}, "block result coverage")
        for block in result["block_results"].values():
            need(block["closure"]["pass"] and all(row["pass"] for row in block["closure"]["checks"].values()), "closure")
            need(set(block["closure"]["checks"]) == {"port_D_to_full_gradient", "pcia_to_ig",
                 "pefa_to_endpoint", "T_equals_A_plus_R", "endpoint_minus_ig_equals_R"},
                 "five closure identities")
            for quantity in ("D", "T", "R"):
                representation = block["representation"][quantity]
                need(set(representation["registered_categories"]) == expected_categories, "representation categories")
                need(representation["overall"]["finite"], "representation finite")
            omega = block["omega_directional"]
            need(omega["representation"]["overall"]["finite"], "omega scientific score finite")
            need(omega["complete_cut_component_ids"] == block["representation"]["D"]["component_ids"],
                 "omega cut identity")
        controls = result["negative_controls"]
        need({row["control_id"] for row in controls} == expected_controls and all(row["pass"] for row in controls), "negative controls")
        need(set(result["input_intervention"]["methods"]) == input_methods, "input baselines")
        need(set(result["stability_checks"]) == {"reference_black_rgb", "path_quadrants_0_3_first",
             "small_noise", "target_clean_top1", "randomized_classifier"}, "stability coverage")
        need(all(row["closure"]["pass"] for row in result["stability_checks"].values()), "stability closure")
        matrix = result["integrated_hessians"]["matrix"]
        need(len(matrix) == 4 and all(len(row) == 4 and all(math.isfinite(value) for value in row) for row in matrix), "IH matrix")
        for family in ("input_intervention", "head_ablation"):
            methods = result[family]["methods"]; effects = result[family]["effects"]
            for method, method_scores in methods.items():
                for effect, observed in result[family]["correlations"][method].items():
                    recomputed = spearman(method_scores, effects[effect])
                    need(same_number(observed, recomputed), f"correlation recompute:{task_id}:{family}:{method}:{effect}")
        identity = result["worker_identity"]
        need(identity["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and identity["startup_matmul_backward"] == "PASS", "runtime identity")
        need(identity["deterministic_algorithms"] and not identity["tf32"] and not identity["autocast"], "determinism")
        need(identity["driver_version"] and identity["model_training"] is False
             and identity["training_module_count"] == 0 and identity["unfused_attention_all"]
             and identity["gradient_checkpointing"] is False, "runtime model identity")
        reproducibility = result["reproducibility"]
        need(reproducibility["seed_namespace"] == protocol["seed_namespace"]
             and reproducibility["task_seed"] == int(document["seed"]) == int(task_seed), "task seed")
        need(set(reproducibility["rng_state_before"]) == {"python", "numpy", "torch_cpu_base64", "torch_cuda_base64"}
             and set(reproducibility["rng_state_after"]) == {"python", "numpy", "torch_cpu_base64", "torch_cuda_base64"},
             "rng states")
        producer_binding = predecessor_binding if task_id in inheritance_records else binding
        need(reproducibility["git_commit"] == producer_binding["producer_commit"]
             and reproducibility["git_dirty"] == producer_binding["producer_dirty"]
             and reproducibility["git_dirty_status_sha256"] == producer_binding["producer_dirty_status_sha256"],
             "git runtime identity")
        expected_checkpoint = next(row["sha256"] for row in binding["checkpoints"]
                                   if row["model_id"] == result["model_id"])
        need(reproducibility["checkpoint_sha256"] == expected_checkpoint
             and reproducibility["input_sha256"] == result["image_sha256"], "model input hashes")
        need(not result["raw_hessian_materialized"] and not result["full_port_input_input_tensor_materialized"], "materialization boundary")

    omega_connection = sqlite3.connect(PILOT / "omega_audit_queue.sqlite3")
    try:
        omega_rows = omega_connection.execute(
            "SELECT task_id,state,attempt,result_path,result_digest,seed FROM tasks ORDER BY task_id"
        ).fetchall()
    finally:
        omega_connection.close()
    need(len(omega_rows) == 16 and all(row[1] == "completed" for row in omega_rows), "Omega audit queue")
    omega_audits = {}
    for task_id, _state, _attempt, result_path, result_digest, seed in omega_rows:
        document = verify_signed(PILOT / "omega_audit_results" / result_path)
        need(document["canonical_digest"] == result_digest and document["task_id"] == task_id,
             "Omega audit result identity")
        audit = document["result"]; omega_audits[task_id] = audit
        need(audit["status"] == "PASS_OMEGA_FLOAT64_AUDIT"
             and audit["model_dtype"] == "float64"
             and audit["task_seed"] == int(seed)
             and audit["scientific_omega_scores_recomputed"] is False
             and audit["tolerance_changed"] is False, "Omega audit contract")
        need(set(audit["blocks"]) == {"0", "5", "11"}, "Omega audit blocks")
        for block in audit["blocks"].values():
            need(block["pass"] and block["dtype"] == "float64"
                 and block["evaluation_component_ids"] == ["identity", "attention"]
                 and block["complete_cut_zero_sum"]["pass"]
                 and block["full_gradient_zero_sum"]["pass"]
                 and block["complete_cut_vs_full_gradient"]["pass"], "Omega audit closure")

    need(validation["task_count"] == len(documents) == 16 and validation["models"] == {"deit_s16": 8, "vit_b16": 8}, "validation coverage")
    need(validation["closure_pass"] and validation["scientific_closure_pass"]
         and validation["omega_float64_audit_pass"]
         and validation["negative_controls_pass"], "validation mathematical gates")
    need(validation["statistics_unit"] == "image" and validation["reference_conditions_reported_separately"], "statistics contract")
    need(validation["correlation_and_causal_intervention_separated"] and validation["evaluation_access"] is False, "causal/access boundary")
    input_effects = list(documents[0]["input_intervention"]["effects"]); head_effects = list(documents[0]["head_ablation"]["effects"])
    recomputed = {"input": {}, "head": {}, "stability": {}}
    for model in protocol["models"]:
        model_rows = [row for row in documents if row["model_id"] == model]
        recomputed["input"][model] = {effect: {method: summary(
            [row["input_intervention"]["correlations"][method][effect] for row in model_rows],
            f"{protocol['run_id']}|input|{model}|{effect}|{method}",
        ) for method in protocol["contribution"]["methods"]} for effect in input_effects}
        recomputed["head"][model] = {effect: {method: summary(
            [row["head_ablation"]["correlations"][method][effect] for row in model_rows],
            f"{protocol['run_id']}|head|{model}|{effect}|{method}",
        ) for method in model_rows[0]["head_ablation"]["methods"]} for effect in head_effects}
        recomputed["stability"][model] = {check: {
            "R_rank_spearman": summary(
                [row["stability_checks"][check]["spearman_vs_primary"] for row in model_rows],
                f"{protocol['run_id']}|stability|R|{model}|{check}"),
            "endpoint_rank_spearman": summary(
                [row["stability_checks"][check]["endpoint_map_spearman_vs_primary"] for row in model_rows],
                f"{protocol['run_id']}|stability|E|{model}|{check}"),
            "reference_condition": model_rows[0]["stability_checks"][check]["reference_condition"],
        } for check in model_rows[0]["stability_checks"]}
    need(same_value(validation["aggregates"], recomputed), "independent aggregate recompute")
    simple_input = ["endpoint", "ig", "conductance_pcia", "integrated_hessians", "random", "input_norm"]
    contribution_supported = False
    for effect in input_effects:
        passes = []
        for model in protocol["models"]:
            table = recomputed["input"][model][effect]; candidate = table["pefa_R"]["mean"]
            baseline = max(table[name]["mean"] for name in simple_input)
            passes.append(candidate - baseline >= 0.05)
        contribution_supported = contribution_supported or all(passes)
    prediction_supported = contribution_supported
    head_baselines = ["endpoint_port_Q", "conductance_pcia", "pefa_T", "pefa_R", "activation_norm", "random"]
    for effect in head_effects:
        passes = []
        for model in protocol["models"]:
            table = recomputed["head"][model][effect]; candidate = table["omega"]["mean"]
            passes.append(candidate - max(table[name]["mean"] for name in head_baselines) >= 0.05)
        prediction_supported = prediction_supported or all(passes)
    representation = validation["representation"]
    need(representation["status"] == "DESCRIPTIVE" and set(representation["quantity_semantics"]) == {"D", "Q", "R", "Omega"},
         "representation quantities")
    need(len(representation["worst_closure_by_model_block_check"]) == 36, "worst closure coverage")
    for model in protocol["models"]:
        need(set(representation["models"][model]) == {"0", "5", "11"}, "representation depths")
        for block in representation["models"][model].values():
            need(set(block) == {"D", "Q", "R", "Omega"}, "representation DQROmega")
            for quantity in block.values():
                need(set(quantity["registered_categories"]) == expected_categories, "aggregated categories")
    need(resource["task_count"] == 16 and resource["aggregate_h20_hours"] <= 64, "gpu budget")
    need(resource["launch_elapsed_seconds"] <= 72 * 3600 and resource["peak_reserved_gpu_bytes"] <= 40 * 1024**3, "runtime caps")
    need(resource["predecessor_resource"]["through_run_id"] == "step5_r7_omega_diagnosis_20260921"
         and same_number(resource["predecessor_resource"]["h20_hours_upper_bound"],
                         10.087391376632073), "cumulative predecessor resource")
    need(same_number(resource["current_launch_worker_seconds_upper_bound"],
                     launch["elapsed_seconds"] * launch["selected_gpu_count"]), "current resource accounting")
    expected_h20 = (resource["predecessor_resource"]["h20_hours_upper_bound"]
                    + resource["current_launch_worker_seconds_upper_bound"] / 3600.0
                    + resource["omega_audit_launch_worker_seconds_upper_bound"] / 3600.0)
    need(same_number(resource["aggregate_h20_hours"], expected_h20), "aggregate GPU-hour recompute")
    need(resource["peak_cpu_rss_bytes"] <= 32 * 1024**3 and resource["retained_bytes_before_finalize"] <= 128 * 1024**3, "memory storage caps")

    claim_ids = {row["claim_id"] for row in claims["decisions"]}
    need(claim_ids == {"MATHEMATICAL_CLOSURE", "NEGATIVE_CONTROL_DETECTION", "COST_AFFORDABLE",
                       "INCREMENTAL_OVER_SIMPLE_BASELINES", "PREREGISTERED_INTERVENTION_PREDICTION"}, "claim table")
    need(all(row["decision"] in {"SUPPORTED", "NOT_SUPPORTED", "INVALID"} for row in claims["decisions"]), "claim vocabulary")
    decision_map = {row["claim_id"]: row["decision"] for row in claims["decisions"]}
    need(decision_map == {
        "MATHEMATICAL_CLOSURE": "SUPPORTED",
        "NEGATIVE_CONTROL_DETECTION": "SUPPORTED",
        "COST_AFFORDABLE": "SUPPORTED" if resource["status"] == "SUPPORTED" else "INVALID",
        "INCREMENTAL_OVER_SIMPLE_BASELINES": "SUPPORTED" if contribution_supported else "NOT_SUPPORTED",
        "PREREGISTERED_INTERVENTION_PREDICTION": "SUPPORTED" if prediction_supported else "NOT_SUPPORTED",
    }, "independent claim recompute")
    promoted = all(row["decision"] == "SUPPORTED" for row in claims["decisions"])
    need(claims["promotion_gate_pass"] == promoted and claims["step6_production_authorized"] is False, "promotion boundary")
    expected_status = "PASS_PEFA_PILOT_PROMOTION_GATE" if promoted else (
        "NOT_SUPPORTED_EMPIRICAL_VALUE" if all(row["decision"] != "INVALID" for row in claims["decisions"][:]) else "INVALID_STEP5_PILOT"
    )
    need(claims["status"] == terminal["status"] == validation["status"] == expected_status, "terminal status")
    need(terminal["step6_started"] is False, "step6 not started")
    need(terminal["claim_decisions_digest"] == claims["canonical_digest"]
         and terminal["validation_report_digest"] == validation["canonical_digest"]
         and terminal["resource_ledger_digest"] == resource["canonical_digest"], "terminal links")
    need({row["claim_id"] for row in evidence_index["claims"]} == claim_ids
         and evidence_index["step6_started"] is False, "claim evidence index")
    need(any("freeze_step5.py" in command for command in reproduction["commands"])
         and any("validate_step5.py" in command for command in reproduction["commands"]),
         "reproduction commands")

    manifest_paths = {row["path"] for row in artifacts["files"]}
    need(artifacts["file_count"] == len(artifacts["files"]), "artifact count")
    need({"protocol.json", "source_binding.json", "task_manifest.json", "freeze_receipt.json",
          "startup_snapshot.json", "launch_terminal.json", "resource_ledger.json",
          "validation_report.json", "claim_decisions.json", "terminal_receipt.json",
          "claim_evidence_index.json", "reproduction_commands.json", "inheritance_receipt.json",
          "omega_audit_startup_snapshot.json", "omega_audit_terminal.json"} <= manifest_paths,
         "artifact manifest coverage")
    for row in artifacts["files"]:
        path = PILOT / row["path"]
        need(path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
             "artifact drift: " + row["path"])
    tests = subprocess.run(
        [str(REPO / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q",
         "vit_pcia_pefa/tests/test_step5_pilot.py", "vit_pcia_pefa/tests/test_step5_queue.py",
         "vit_pcia_pefa/tests/test_step4_vit_adapter.py"],
        cwd=REPO, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                       "PYTHONPATH": ".:vitnns_rebuild/src"}, capture_output=True, text=True,
    )
    need(tests.returncode == 0, "unit tests: " + tests.stdout + tests.stderr)
    report = {
        "status": "PASS_STEP5_PROMPT_TO_ARTIFACT_AUDIT", "terminal_status": terminal["status"],
        "tasks_verified": 16, "models_verified": 2, "images_verified": 8,
        "blocks_per_task": 3, "input_methods_verified": len(input_methods),
        "negative_controls_per_task": len(expected_controls), "stability_checks_per_task": 5,
        "claim_decisions_verified": 5, "artifacts_verified": artifacts["file_count"],
        "unit_tests": tests.stdout.strip(), "step6_started": False,
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
