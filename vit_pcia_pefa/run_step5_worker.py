#!/usr/bin/env python3
"""Execute frozen Step-5 pilot tasks from the dynamic SQLite queue."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_PREIMPORT")

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import math
import platform
import random
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator

import numpy as np
import torch
import torchvision
import timm
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]

from vit_pcia_pefa.run_step4_attempt1_worker import load_image, load_model  # noqa: E402
from vit_pcia_pefa.src.pilot import (  # noqa: E402
    closure_summary, cut_representation_summary, endpoint_and_logit, formation_fields,
    integrated_hessians_groups, integrated_hessians_scores,
    omega_directional_analysis, port_categories, projection_head_norms,
    quadrant_masks, quadrant_scores, randomized_classifier,
    representation_summary, spearman, stable_random_scores,
)
from vit_pcia_pefa.src.vit_adapter import (  # noqa: E402
    port_fields, port_layout, structural_negative_controls,
)
from vit_pcia_pefa.src.step5_queue import (  # noqa: E402
    claim_task, complete_task, heartbeat,
)
from vitnns.endpoint_ep2 import gauss_legendre  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def rng_state_snapshot(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"bit_generator": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu_base64": base64.b64encode(torch.get_rng_state().cpu().numpy().tobytes()).decode(),
        "torch_cuda_base64": base64.b64encode(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).decode(),
    }


def verify_binding(binding_path: Path) -> dict[str, Any]:
    binding = json_load(binding_path)
    for row in (*binding["files"], *binding["checkpoints"], *binding["pilot_images"]):
        path = REPO / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("FAIL_SOURCE_BINDING:" + row["path"])
    return binding


def distribution(value: torch.Tensor, indexes: list[int]) -> list[float]:
    masses = value[indexes].detach().double().flatten(1).abs().sum(1)
    total = masses.sum()
    if not bool(torch.isfinite(masses).all()) or float(total) <= 0:
        return [0.0 for _ in indexes]
    return [float(item) for item in (masses / total)]


def total_variation(left: list[float], right: list[float]) -> float:
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(left, right))


def correlations(methods: dict[str, list[float]], effects: dict[str, list[float]]) -> dict[str, dict[str, float | None]]:
    return {method: {effect: spearman(scores, values) for effect, values in effects.items()}
            for method, scores in methods.items()}


@contextmanager
def head_ablation(model, block: int, head: int) -> Iterator[None]:
    from vit_pcia_pefa.src.pilot import ablate_projection_head
    with ablate_projection_head(model, block, head):
        yield


def abandon(queue: Path, claim: dict[str, Any], error: dict[str, Any]) -> None:
    now = time.time_ns()
    connection = sqlite3.connect(queue, timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT state,attempt,fencing_token,worker_id,device_uuid FROM tasks WHERE task_id=?",
            (claim["task_id"],),
        ).fetchone()
        valid = current is not None and current[0] == "claimed" and tuple(current[1:]) == (
            claim["attempt"], claim["fencing_token"], claim["worker_id"], claim["device_uuid"],
        )
        if not valid:
            raise RuntimeError("STALE_FENCING_TOKEN")
        terminal = claim["attempt"] >= 2
        connection.execute(
            "UPDATE attempts SET status=?,completed_ns=? WHERE task_id=? AND attempt=?",
            ("FAILED" if terminal else "RETRYABLE", now, claim["task_id"], claim["attempt"]),
        )
        connection.execute(
            "UPDATE tasks SET state=?,fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
            "lease_until_ns=NULL,heartbeat_ns=NULL WHERE task_id=?",
            ("failed" if terminal else "pending", claim["task_id"]),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (f"failure:{claim['task_id']}:{claim['attempt']}", json.dumps(error, sort_keys=True)),
        )
        connection.commit()
    finally:
        connection.close()


def execute(model, model_spec: dict[str, Any], task: dict[str, Any], claim: dict[str, Any],
            queue: Path, protocol: dict[str, Any], binding: dict[str, Any],
            device: torch.device) -> dict[str, Any]:
    started = time.monotonic()
    last_heartbeat = started
    task_seed = int(claim["seed"])
    random.seed(task_seed); np.random.seed(task_seed % (2**32))
    torch.manual_seed(task_seed); torch.cuda.manual_seed_all(task_seed)
    rng_before = rng_state_snapshot(device)

    def progress() -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - started > protocol["resources"]["task_timeout_seconds"]:
            raise RuntimeError("PAUSED_RESOURCE_CAP:TASK_TIMEOUT")
        if now - last_heartbeat >= protocol["resources"]["heartbeat_seconds"]:
            heartbeat(queue, claim, lease_seconds=protocol["resources"]["lease_seconds"])
            last_heartbeat = now

    image_receipt = {
        "cache_path": task["image_path"], "image_bytes": task["image_bytes"],
        "image_sha256": task["image_sha256"],
    }
    image, original_size = load_image(image_receipt, model_spec, device)
    baseline = torch.zeros_like(image)
    mean = torch.tensor(model_spec["preprocessing"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model_spec["preprocessing"]["std"], device=device).view(1, 3, 1, 1)
    black_rgb = -mean / std
    masks = quadrant_masks(image)
    directions = [image * mask for mask in masks]
    waypoint = directions[0] + directions[3]
    nodes_tensor, weights_tensor = gauss_legendre(protocol["integration_nodes"], device, torch.float32)
    nodes = [float(item) for item in nodes_tensor]
    weights = [float(item) for item in weights_tensor]
    class_id = int(task["class_id"])
    with torch.no_grad():
        clean_logits = model(image).detach()
        clean_top1 = int(clean_logits.argmax(1).item())
    clean_endpoint, clean_logit = endpoint_and_logit(
        model, image, class_id, baseline=baseline,
    )
    with randomized_classifier(model, int(claim["seed"])):
        randomized_endpoint, _ = endpoint_and_logit(model, image, class_id, baseline=baseline)
    negative_controls = structural_negative_controls(
        model, image, block_index=protocol["primary_intervention_block"], class_id=class_id,
    )
    if not all(row["pass"] for row in negative_controls):
        raise RuntimeError("FAIL_NEGATIVE_CONTROL")

    block_results = {}
    retained_fields = {}
    for block in protocol["blocks"]:
        fields = formation_fields(
            model, image=image, baseline=baseline, waypoints=(), block_index=block,
            class_id=class_id, nodes=nodes, weights=weights, progress=progress,
        )
        cut_id = protocol["primary_cut"]
        closure = closure_summary(model, block, fields, cut_id)
        indexes = {name: index for index, name in enumerate(fields["names"])}
        cut_names = port_layout(model.blocks[block].attn.num_heads).cuts[cut_id]
        selected = [indexes[name] for name in cut_names]
        categories = {}
        for quantity in ("D", "T", "R"):
            registered = representation_summary(fields[quantity], fields["names"])
            categories[quantity] = cut_representation_summary(
                fields[quantity], fields["names"], cut_names,
            )
            categories[quantity]["registered_categories"] = registered["categories"]
            categories[quantity]["registered_per_port"] = registered["per_port"]
        omega_block = omega_directional_analysis(
            model, image, block_index=block, class_id=class_id,
            component_names=fields["names"][:-1], cut_component_names=cut_names,
            seed=task_seed + block,
        )
        if (protocol.get("omega_zero_sum_gate_source", "task_float32_streamed")
                == "task_float32_streamed"
                and not omega_block["complete_cut_zero_sum"]["pass"]):
            raise RuntimeError("FAIL_QUADRATURE_CLOSURE:OMEGA_ZERO_SUM")
        block_results[str(block)] = {
            "closure": closure, "representation": categories,
            "omega_directional": omega_block,
            "A_cut_absolute_mass": float(fields["A"][selected].abs().sum()),
            "T_cut_absolute_mass": float(fields["T"][selected].abs().sum()),
            "R_cut_absolute_mass": float(fields["R"][selected].abs().sum()),
        }
        retained_fields[block] = (fields, selected)
        progress()

    primary_block = protocol["primary_intervention_block"]
    main, selected = retained_fields[primary_block]
    # The IH rule has its own frozen Gauss--Legendre quadrature.
    ih_nodes_t, ih_weights_t = gauss_legendre(protocol["ih_nodes"], device, torch.float32)
    ih = integrated_hessians_groups(
        model, image=image, baseline=baseline, class_id=class_id,
        nodes=[float(item) for item in ih_nodes_t], weights=[float(item) for item in ih_weights_t],
        masks=masks, progress=progress,
    )
    input_methods = {
        "endpoint": quadrant_scores(main["T"][-1], masks, port_absolute=False),
        "ig": quadrant_scores(main["A"][-1], masks, port_absolute=False),
        "conductance_pcia": quadrant_scores(main["A"][selected], masks, port_absolute=True),
        "integrated_hessians": integrated_hessians_scores(ih),
        "pefa_T": quadrant_scores(main["T"][selected], masks, port_absolute=True),
        "pefa_R": quadrant_scores(main["R"][selected], masks, port_absolute=True),
        "random": stable_random_scores(str(task_seed) + "|input", 4),
        "input_norm": [float((image * mask).double().norm()) for mask in masks],
    }
    clean_names = main["names"]
    clean_distribution = distribution(main["T"], selected)
    input_effects = {"endpoint_change_l1": [], "target_logit_abs_change": [], "routing_total_variation": []}
    for mask in masks:
        perturbed = torch.where(mask, baseline, image)
        endpoint, logit = endpoint_and_logit(model, perturbed, class_id, baseline=baseline)
        port_q, names = port_fields(
            model, perturbed.detach().requires_grad_(True), block_index=primary_block,
            class_id=class_id, second_order=False,
        )
        if names != clean_names:
            raise RuntimeError("INTERVENTION_PORT_ORDER_DRIFT")
        port_q = (perturbed - baseline).unsqueeze(0).double() * port_q.detach().double()
        input_effects["endpoint_change_l1"].append(float((endpoint - clean_endpoint).double().abs().sum()))
        input_effects["target_logit_abs_change"].append(abs(logit - clean_logit))
        input_effects["routing_total_variation"].append(
            total_variation(clean_distribution, distribution(port_q, selected))
        )
        progress()

    head_names = [name for name in clean_names if name.startswith("head_")]
    head_indexes = [{name: index for index, name in enumerate(clean_names)}[name] for name in head_names]
    omega_analysis = block_results[str(primary_block)]["omega_directional"]
    omega = omega_analysis["scores_l2"]
    head_methods = {
        "endpoint_port_Q": [float(main["T"][index].abs().sum()) for index in head_indexes],
        "conductance_pcia": [float(main["A"][index].abs().sum()) for index in head_indexes],
        "pefa_T": [float(main["T"][index].abs().sum()) for index in head_indexes],
        "pefa_R": [float(main["R"][index].abs().sum()) for index in head_indexes],
        "omega": [omega[name] for name in head_names],
        "activation_norm": projection_head_norms(model, image, primary_block),
        "random": stable_random_scores(str(task_seed) + "|head", len(head_names)),
    }
    head_effects = {"endpoint_change_l1": [], "target_logit_abs_change": []}
    for index in range(len(head_names)):
        endpoint, logit = endpoint_and_logit(
            model, image, class_id, baseline=baseline,
            block_index=primary_block, head_index=index,
        )
        head_effects["endpoint_change_l1"].append(float((endpoint - clean_endpoint).double().abs().sum()))
        head_effects["target_logit_abs_change"].append(abs(logit - clean_logit))
        progress()

    reference_fields = formation_fields(
        model, image=image, baseline=black_rgb.expand_as(image), waypoints=(),
        block_index=primary_block, class_id=class_id, nodes=nodes, weights=weights, progress=progress,
    )
    path_fields = formation_fields(
        model, image=image, baseline=baseline, waypoints=(waypoint,),
        block_index=primary_block, class_id=class_id, nodes=nodes, weights=weights, progress=progress,
    )
    raw = (image * std + mean).clamp(0, 1)
    generator = torch.Generator(device=device); generator.manual_seed(int(claim["seed"]))
    noisy_raw = (raw + torch.randn(raw.shape, device=device, dtype=raw.dtype, generator=generator) * protocol["stability"]["noise_sigma_rgb"]).clamp(0, 1)
    noisy = (noisy_raw - mean) / std
    noise_fields = formation_fields(
        model, image=noisy, baseline=baseline, waypoints=(), block_index=primary_block,
        class_id=class_id, nodes=nodes, weights=weights, progress=progress,
    )
    target_fields = formation_fields(
        model, image=image, baseline=baseline, waypoints=(), block_index=primary_block,
        class_id=clean_top1, nodes=nodes, weights=weights, progress=progress,
    )
    with randomized_classifier(model, int(claim["seed"])):
        randomized_fields = formation_fields(
            model, image=image, baseline=baseline, waypoints=(), block_index=primary_block,
            class_id=class_id, nodes=nodes, weights=weights, progress=progress,
        )
    clean_reference_endpoint, _ = endpoint_and_logit(
        model, image, class_id, baseline=black_rgb.expand_as(image),
    )
    noise_endpoint, _ = endpoint_and_logit(model, noisy, class_id, baseline=baseline)
    top1_endpoint, _ = endpoint_and_logit(model, image, clean_top1, baseline=baseline)
    main_scores = quadrant_scores(main["R"][selected], masks, port_absolute=True)
    checks = {}
    for key, value in (
        ("reference_black_rgb", reference_fields), ("path_quadrants_0_3_first", path_fields),
        ("small_noise", noise_fields), ("target_clean_top1", target_fields),
        ("randomized_classifier", randomized_fields),
    ):
        value_selected = [
            {name: index for index, name in enumerate(value["names"])}[name]
            for name in port_layout(model.blocks[primary_block].attn.num_heads).cuts[protocol["primary_cut"]]
        ]
        scores = quadrant_scores(value["R"][value_selected], masks, port_absolute=True)
        checks[key] = {
            "reference_condition": "black_rgb" if key == "reference_black_rgb" else "normalized_zero",
            "scores": scores, "spearman_vs_primary": spearman(main_scores, scores),
            "closure": closure_summary(model, primary_block, value, protocol["primary_cut"]),
        }
    checks["reference_black_rgb"]["endpoint_map_spearman_vs_primary"] = spearman(
        quadrant_scores(clean_endpoint, masks, port_absolute=False),
        quadrant_scores(clean_reference_endpoint, masks, port_absolute=False),
    )
    checks["path_quadrants_0_3_first"]["endpoint_map_spearman_vs_primary"] = 1.0
    checks["small_noise"]["endpoint_map_spearman_vs_primary"] = spearman(
        quadrant_scores(clean_endpoint, masks, port_absolute=False),
        quadrant_scores(noise_endpoint, masks, port_absolute=False),
    )
    checks["target_clean_top1"]["endpoint_map_spearman_vs_primary"] = spearman(
        quadrant_scores(clean_endpoint, masks, port_absolute=False),
        quadrant_scores(top1_endpoint, masks, port_absolute=False),
    )
    checks["randomized_classifier"]["endpoint_map_spearman_vs_primary"] = spearman(
        quadrant_scores(clean_endpoint, masks, port_absolute=False),
        quadrant_scores(randomized_endpoint, masks, port_absolute=False),
    )

    elapsed = time.monotonic() - started
    if not all(row["closure"]["pass"] for row in block_results.values()):
        raise RuntimeError("FAIL_QUADRATURE_CLOSURE")
    if not all(row["closure"]["pass"] for row in checks.values()):
        raise RuntimeError("FAIL_QUADRATURE_CLOSURE:STABILITY")
    return {
        "status": "PASS_TASK", "task_id": claim["task_id"],
        "model_id": task["model_id"], "sample_id": task["sample_id"],
        "split": task["split"], "split_order": task["split_order"],
        "class_id": class_id, "clean_top1_class_id": clean_top1,
        "image_sha256": task["image_sha256"], "original_size_wh": original_size,
        "block_results": block_results, "negative_controls": negative_controls,
        "input_intervention": {
            "methods": input_methods, "effects": input_effects,
            "correlations": correlations(input_methods, input_effects),
        },
        "head_ablation": {
            "block": primary_block, "component_ids": head_names, "methods": head_methods,
            "effects": head_effects, "correlations": correlations(head_methods, head_effects),
        },
        "stability_checks": checks,
        "integrated_hessians": {
            "definition": "recursive_IG_4x4_group_matrix", "nodes_per_integral": protocol["ih_nodes"],
            "matrix": [[float(item) for item in row] for row in ih],
        },
        "worker_identity": {
            "sys_executable_realpath": str(Path(sys.executable).resolve()),
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__,
            "numpy": np.__version__, "Pillow": Image.__version__, "cuda": torch.version.cuda,
            "gpu_uuid": claim["device_uuid"], "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32": False, "autocast": False, "startup_matmul_backward": "PASS",
            "driver_version": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
            ).splitlines()[0].strip(),
            "model_training": bool(model.training),
            "training_module_count": sum(int(module.training) for module in model.modules()),
            "unfused_attention_all": all(not block.attn.fused_attn for block in model.blocks),
            "gradient_checkpointing": bool(getattr(model, "grad_checkpointing", False)),
        },
        "reproducibility": {
            "seed_namespace": protocol["seed_namespace"], "task_seed": task_seed,
            "rng_state_before": rng_before, "rng_state_after": rng_state_snapshot(device),
            "git_commit": binding["producer_commit"],
            "git_dirty": binding["producer_dirty"],
            "git_dirty_status_sha256": binding["producer_dirty_status_sha256"],
            "checkpoint_sha256": next(
                row["sha256"] for row in binding["checkpoints"] if row["model_id"] == task["model_id"]
            ),
            "input_sha256": task["image_sha256"],
        },
        "resources": {
            "wall_seconds": elapsed, "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "raw_hessian_materialized": False, "full_port_input_input_tensor_materialized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_DRIFT")
    binding = verify_binding(args.root / "source_binding.json")
    protocol = json_load(args.root / "protocol.json")
    queue = args.root / "queue.sqlite3"
    results = args.root / "results"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    probe = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).requires_grad_()
    (probe @ probe.T).square().sum().backward(); torch.cuda.synchronize()
    model_registry = json_load(REPO / protocol["model_registry_path"])
    loaded: dict[str, tuple[Any, dict[str, Any]]] = {}
    while True:
        claim = claim_task(
            queue, worker_id=args.worker_id, device_uuid=args.gpu_uuid,
            lease_seconds=protocol["resources"]["lease_seconds"],
            max_attempts=protocol["resources"]["max_attempts"],
        )
        if claim is None:
            break
        task = claim["payload"]
        try:
            model_id = task["model_id"]
            if model_id not in loaded:
                specification = next(row for row in model_registry["models"] if row["model_id"] == model_id)
                loaded[model_id] = (load_model(specification, device)[0], specification)
            model, specification = loaded[model_id]
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            result = execute(model, specification, task, claim, queue, protocol, binding, device)
            if result["resources"]["peak_reserved_gpu_bytes"] > protocol["resources"]["gpu_memory_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:GPU_MEMORY")
            if result["resources"]["peak_cpu_rss_bytes"] > protocol["resources"]["cpu_rss_cap_bytes"]:
                raise RuntimeError("PAUSED_RESOURCE_CAP:CPU_RSS")
            complete_task(queue, results, claim, result)
            print(json.dumps({"status": result["status"], "task_id": claim["task_id"],
                              "seconds": result["resources"]["wall_seconds"]}), flush=True)
        except Exception as error:
            failure = {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "time_ns": time.time_ns(),
            }
            abandon(queue, claim, failure)
            print(json.dumps({"status": "TASK_FAILED", "task_id": claim["task_id"],
                              "attempt": claim["attempt"], **failure}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
