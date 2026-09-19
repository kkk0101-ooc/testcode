#!/usr/bin/env python3
"""Recompute raw maps only for cases frozen by Endpoint M6 selection."""
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"
M3_ROOT = REPO / "reports/nns_vit_endpoint_m3_r1_20260919"
M4_ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from run_endpoint_e4_intervention_worker import load_inputs, load_model  # noqa: E402
from vitnns.endpoint_m6 import extract_selected_maps, summarize_map  # noqa: E402
from vitnns.endpoint_vit import EndpointTarget, scan_endpoint_vit  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def close_summary(observed, expected, code):
    for field in ("signed_sum", "absolute_mass", "l2_norm"):
        difference = abs(float(observed[field]) - float(expected[field]))
        limit = 2e-5 + 2e-4 * abs(float(expected[field]))
        require(difference <= limit, code + ":" + field)


def display_image(clean, mean, std):
    mean = torch.tensor(mean, device=clean.device).reshape(1, 3, 1, 1)
    std = torch.tensor(std, device=clean.device).reshape(1, 3, 1, 1)
    return ((clean * std + mean).clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def main():
    freeze = load(ROOT / "freeze.json"); selection_path = ROOT / "selection_manifest.json"
    require(freeze["selection_before_image_access"] and selection_path.is_file(), "SELECTION_NOT_FROZEN")
    for relative, expected in freeze["sources"].items(): require(sha(REPO / relative) == expected, "FROZEN_SOURCE:" + relative)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=REPO, text=True).split()[0]
    require(head == remote == freeze["producer_head"], "PUBLISHED_PRODUCER")
    selection = load(selection_path); m3_tasks = {row["task_id"]: row["payload"] for row in load(M3_ROOT / "task_manifest.json")["tasks"]}
    m4_tasks = {row["task_id"]: row["payload"] for row in load(M4_ROOT / "task_manifest.json")["tasks"]}
    cases = {row["case_id"]: row for row in selection["cases"]}; arrays = {}; checks = []; image_keys = set(); models = {}
    device = torch.device("cuda:0"); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    started = time.perf_counter()

    # M3 selected token-channel targets, grouped by source task to share forwards.
    m3_groups = {}
    for case in cases.values():
        for unit in case.get("selected_units", []):
            key = (unit["task_id"], unit["block"], unit["token"], unit["channel"])
            m3_groups.setdefault(unit["task_id"], {})[key] = unit
    for task_id in sorted(m3_groups):
        task = m3_tasks[task_id]
        if task["model_id"] not in models: models[task["model_id"]] = load_model(task, device)
        model, _binding = models[task["model_id"]]; image, _, _ = load_inputs(task, device)
        targets, ordered = [], []
        for index, (key, unit) in enumerate(sorted(m3_groups[task_id].items())):
            name = "m3__" + "__".join(map(str, key)); ordered.append((name, unit))
            targets.append(EndpointTarget(name=name, module="blocks.%d" % unit["block"], kind="token_channel", token=unit["token"], channel=unit["channel"]))
        scan = scan_endpoint_vit(model, image, targets)
        for name, unit in ordered:
            array = scan.units[name].attribution[0].numpy().astype(np.float32); arrays[name] = array
            observed = summarize_map(array); close_summary(observed, unit, "M3_RECOMPUTE:" + name)
            checks.append({"kind": "M3", "array_key": name, "source": unit["source_path"], "status": "PASS", "summary": observed})
        image_key = "image__" + task_id; arrays[image_key] = display_image(image, task["mean"], task["std"]); image_keys.add(image_key)
        torch.cuda.empty_cache()

    # M4 selected heads/residuals, grouped by task to share one graph per image.
    m4_groups = {}
    for case in cases.values():
        for row in case.get("selected_heads", []): m4_groups.setdefault(row["task_id"], {"attention": [], "residual": []})["attention"].append({**row, "case_id": case["case_id"]})
        for row in case.get("selected_residuals", []): m4_groups.setdefault(row["task_id"], {"attention": [], "residual": []})["residual"].append({**row, "case_id": case["case_id"]})
    for task_id in sorted(m4_groups):
        task = m4_tasks[task_id]
        if task["model_id"] not in models: models[task["model_id"]] = load_model(task, device)
        model, _binding = models[task["model_id"]]; image, _, _ = load_inputs(task, device)
        group = m4_groups[task_id]; result = extract_selected_maps(model, image, task["class_id"], group["attention"], group["residual"])
        for key, array in result["arrays"].items():
            if key != "full_endpoint": arrays[key] = array
        for row in group["attention"]:
            for component in ("q", "k", "routing", "value", "full"):
                key = row["case_id"] + "__" + component; close_summary(result["summaries"][key], row[component], "M4_ATTENTION_RECOMPUTE:" + key)
                checks.append({"kind": "M4_ATTENTION", "array_key": key, "source": row["source_path"], "status": "PASS", "summary": result["summaries"][key]})
        for row in group["residual"]:
            for component in ("identity", "branch", "output"):
                key = row["case_id"] + "__" + component; close_summary(result["summaries"][key], row[component], "M4_RESIDUAL_RECOMPUTE:" + key)
                checks.append({"kind": "M4_RESIDUAL", "array_key": key, "source": row["source_path"], "status": "PASS", "summary": result["summaries"][key]})
        image_key = "image__" + task_id; arrays[image_key] = display_image(image, task["mean"], task["std"]); image_keys.add(image_key)
        torch.cuda.empty_cache()

    # M2 visual values were already frozen; image decoding occurs only after selection.
    boundary = cases["m2_semantic_boundary"]["selected_boundary"]
    task = next(row["payload"] for row in load(REPO / "reports/nns_vit_endpoint_m2_r1_20260919/task_manifest.json")["tasks"] if row["task_id"] == boundary["task_id"])
    image, _, _ = load_inputs(task, device); arrays["image__" + boundary["task_id"]] = display_image(image, task["mean"], task["std"]); image_keys.add("image__" + boundary["task_id"])
    arrays["m2_semantic_boundary__endpoint_absolute"] = np.asarray(boundary["endpoint_absolute_196"], np.float32).reshape(14, 14)
    arrays["m2_semantic_boundary__regional_taylor"] = np.asarray(boundary["regional_taylor_196"], np.float32).reshape(14, 14)
    raw_path = ROOT / "raw_arrays.npz"; np.savez_compressed(raw_path, **arrays)
    receipt = {"status": "PASS", "selection_manifest_sha256": sha(selection_path), "raw_arrays_sha256": sha(raw_path), "array_count": len(arrays),
               "recomputed_map_checks": len(checks), "display_images": len(image_keys), "checks": checks, "task_seconds": time.perf_counter() - started,
               "worker_identity": {"sys_executable_realpath": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
                 "torch_version": torch.__version__, "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
                 "physical_gpu_index": int(os.environ["CUDA_VISIBLE_DEVICES"]), "visible_device_name": torch.cuda.get_device_name(0),
                 "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"], "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                 "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "autocast": False}}
    (ROOT / "map_recompute_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "M6_SELECTED_MAPS_PASS", "arrays": len(arrays), "checks": len(checks), "seconds": receipt["task_seconds"]}))


if __name__ == "__main__":
    main()
