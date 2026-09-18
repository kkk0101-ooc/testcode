#!/usr/bin/env python3
"""Diagnose the sealed E4 R2 analytic-JVP failure without utility inference."""
from __future__ import annotations

import argparse
import json
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")
from pathlib import Path
import sqlite3
import subprocess
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_endpoint_e4_intervention_worker import load_inputs, load_model  # noqa: E402
from vitnns.endpoint_e2_atlas import file_sha256, write_json_new  # noqa: E402
from vitnns.endpoint_e4_intervention import (  # noqa: E402
    _component_summaries, capture_registered_edges, component_rows, component_tensor,
    directional_edges, dot64, edge_gradients,
)

SOURCE = REPO / "reports/nns_vit_endpoint_e4_intervention_r2_20260918"
TASK_ID = "vit_b16__009"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--physical-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("DIAGNOSTIC_OUTPUT_EXISTS")
    terminal = json.loads((SOURCE / "terminal_receipt.json").read_text())
    if terminal["status"] != "ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT":
        raise RuntimeError("R2_NOT_SEALED_FAILED")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=REPO, text=True).split()[0]
    if head != remote:
        raise RuntimeError("DIAGNOSTIC_SOURCE_NOT_PUBLISHED")
    source_sha = file_sha256(Path(__file__))
    blob = subprocess.check_output(["git", "show", f"{head}:vitnns_rebuild/scripts/{Path(__file__).name}"], cwd=REPO)
    if __import__("hashlib").sha256(blob).hexdigest() != source_sha:
        raise RuntimeError("DIAGNOSTIC_SOURCE_DRIFT")
    with sqlite3.connect(SOURCE / "formal_queue.sqlite3") as connection:
        row = connection.execute("SELECT payload_json,state,attempt FROM tasks WHERE task_id=?", (TASK_ID,)).fetchone()
    if row is None or row[1:] != ("failed", 2):
        raise RuntimeError("FAILED_TASK_IDENTITY")
    task = json.loads(row[0]); os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_index)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0"); model, binding = load_model(task, device)
    image, _references, _original = load_inputs(task, device)
    rows = component_rows(int(model.blocks[0].attn.num_heads), include_secondary=True)
    summaries = _component_summaries(model, image, class_id=task["class_id"], rows=rows)
    capture = capture_registered_edges(model, image, class_id=task["class_id"], enable_grad=True)
    gradients = edge_gradients(capture); directions = directional_edges(model, image, class_id=task["class_id"]); del capture
    components = []
    for item in rows:
        block = int(item["block"]); component_id = item["component_id"]
        gradient = component_tensor(gradients[block], item); direction = component_tensor(directions[block], item)
        prediction = dot64(gradient, direction); a_value = summaries[component_id]["a"]
        if isinstance(gradient, dict):
            absolute_dot_mass = float(sum((gradient[key].double() * direction[key].double()).abs().sum() for key in sorted(gradient)))
            element_count = sum(gradient[key].numel() for key in gradient)
        else:
            absolute_dot_mass = float((gradient.double() * direction.double()).abs().sum())
            element_count = gradient.numel()
        error = abs(prediction - a_value)
        components.append({
            "component_id": component_id, "family": item["family"], "block": block,
            "a": a_value, "directional_prediction": prediction, "absolute_error": error,
            "absolute_dot_mass": absolute_dot_mass, "element_count": element_count,
            "r2_threshold": 1e-5 + 1e-4 * abs(a_value),
            "condition_scaled_threshold": 1e-5 + 1e-4 * absolute_dot_mass,
            "r2_pass": error <= 1e-5 + 1e-4 * abs(a_value),
            "condition_scaled_pass": error <= 1e-5 + 1e-4 * absolute_dot_mass,
        })
    failed = [row for row in components if not row["r2_pass"]]
    output = {
        "status": "DIAGNOSTIC_ONLY_NOT_STAGE_RESULT", "task_id": TASK_ID,
        "source_r2_terminal_sha256": file_sha256(SOURCE / "terminal_receipt.json"),
        "diagnostic_commit": head, "diagnostic_script_sha256": source_sha,
        "model_binding": binding, "component_count": len(components),
        "r2_failed_component_count": len(failed),
        "condition_scaled_failed_component_count": sum(not row["condition_scaled_pass"] for row in components),
        "maximum_absolute_error": max(row["absolute_error"] for row in components),
        "maximum_error_over_absolute_dot_mass": max(row["absolute_error"] / max(row["absolute_dot_mass"], 1e-30) for row in components),
        "failed_components": failed, "components": components,
        "scientific_utility_interpretation": "NOT_PERFORMED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=False); write_json_new(args.output, output)
    print(json.dumps({key: output[key] for key in ("status", "task_id", "component_count", "r2_failed_component_count", "condition_scaled_failed_component_count", "maximum_absolute_error", "maximum_error_over_absolute_dot_mass")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
