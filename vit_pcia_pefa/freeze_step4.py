#!/usr/bin/env python3
"""Freeze the Step-4 source, ports, cuts, quadrature and cost gates."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def port(model_id: str, block: int, producer: str, consumer: str, slot: int, slice_spec: str = "ALL") -> dict:
    return {
        "model_id": model_id, "graph_revision": "timm-0.9.16-standard-unfused-eval",
        "block_index": block, "producer_tensor_id": f"blocks.{block}.{producer}#0",
        "consumer_operator_id": f"blocks.{block}.{consumer}", "input_slot": slot,
        "invocation_index": 0, "slice_spec": slice_spec,
    }


def main() -> int:
    for target in (ROOT / "step4_source_binding.json", ROOT / "vit_port_registry.json"):
        if target.exists():
            raise RuntimeError("STEP4_FREEZE_TARGET_EXISTS:" + target.name)
    models = json.loads((EP0 / "model_registry.json").read_text())["models"]
    source_paths = [
        ROOT / "执行规划.md", ROOT / "method_contract.md", ROOT / "step3_decision.md",
        ROOT / "quadrature_registry.json", ROOT / "src/vit_adapter.py",
        ROOT / "tests/test_step4_vit_adapter.py",
        ROOT / "freeze_step4.py",
        ROOT / "materialize_step4_discovery.py",
        ROOT / "run_step4_worker.py",
        ROOT / "run_step4_quadrature_diagnostic.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_vit.py",
        REPO / "vitnns_rebuild/tests/test_endpoint_vit.py", EP0 / "model_registry.json",
        EP0 / "cohort_registry.json",
    ]
    files = [{
        "path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256(path), "mode": "source_or_registry",
    } for path in source_paths]
    for model in models:
        path = REPO / "vitnns_rebuild" / model["checkpoint"]["path"]
        if path.stat().st_size != model["checkpoint"]["bytes"] or sha256(path) != model["checkpoint"]["sha256"]:
            raise RuntimeError("CHECKPOINT_IDENTITY:" + model["model_id"])
        files.append({
            "path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "mode": "checkpoint_stream_hash_no_deserialization",
        })
    archive = REPO / "vitnns_rebuild/assets/imagenet/ILSVRC2012_img_val.tar"
    archive_binding = {"path": archive.relative_to(REPO).as_posix(), "bytes": archive.stat().st_size,
                       "sha256": sha256(archive)}
    if archive_binding["sha256"] != "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0":
        raise RuntimeError("ARCHIVE_IDENTITY")
    nodes256, weights256 = np.polynomial.legendre.leggauss(256)
    nodes512, weights512 = np.polynomial.legendre.leggauss(512)
    quadrature = {}
    for count, nodes, weights in ((256, nodes256, weights256), (512, nodes512, weights512)):
        nodes = np.ascontiguousarray((nodes + 1.0) / 2.0, dtype="<f8")
        weights = np.ascontiguousarray(weights / 2.0, dtype="<f8")
        quadrature[str(count)] = {
            "count": count, "generation": "numpy.polynomial.legendre.leggauss; t=(s+1)/2; w=w/2",
            "nodes_sha256": hashlib.sha256(nodes.tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "weight_sum": float(weights.sum()),
        }
    source = {
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP4",
        "status": "FROZEN_BEFORE_VIT_OUTCOME",
        "authorization": "USER_REQUEST_EXECUTE_STEP4_20260921",
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "dirty_tree_policy": "preserve unrelated user changes; bind every reused and new Step-4 file by bytes",
        "files": files, "archive": archive_binding, "quadrature": quadrature,
        "outcome_access_before_freeze": {"discovery_jpeg_reads": 0, "checkpoint_deserializations_for_outcome": 0},
        "numeric": {"forward_dtype": "float32", "reduction_dtype": "float64", "atol": 1e-5, "rtol": 1e-4, "near_zero": 1e-6},
        "compute": {
            "path": "normalized-zero straight line", "primary_nodes": 512, "diagnostic_nodes": 256,
            "patch_group_partition": "four exact 7x7 patch quadrants over 14x14 patches",
            "materialize_input_hessian": False, "materialize_port_by_input_by_input": False,
        },
        "validation_scope": {
            "models": ["vit_b16", "deit_s16"], "discovery_rows_per_model": 1,
            "primary_block": 0, "full_vit_check": "all 12 blocks first-order raw-coordinate cuts",
            "target": "frozen EP0 ground-truth raw logit",
            "omega_directional_probes": 2,
        },
        "cost_gate": {
            "aggregate_h20_hours_cap": 16, "wall_hours_cap": 24, "retained_gib_cap": 32,
            "worker_gpu_memory_gib_cap": 24, "worker_cpu_rss_gib_cap": 32,
            "task_seconds_cap": 1800, "max_attempts": 2, "minimum_free_disk_gib": 50,
            "production_projection_images": 64, "production_projection_models": 2,
            "production_projection_blocks": 12, "production_projection_targets": 1,
            "projection_rule": "observed primary 512-node block seconds times 64 images times 12 blocks for each model",
        },
        "failure_codes": ["FAIL_SOURCE_BINDING", "FAIL_PORT_ID_AMBIGUOUS", "FAIL_CUT_NOT_LEGAL", "FAIL_SECOND_ORDER_GRAPH_TRUNCATED", "FAIL_QUADRATURE_CLOSURE", "FAIL_NEGATIVE_CONTROL", "FAIL_PORT_FIDELITY", "NO_GO_COST", "PAUSED_RESOURCE_CAP"],
        "step5_authorized": False,
    }
    source["canonical_digest"] = canonical(source)
    write_new(ROOT / "step4_source_binding.json", source)

    registries = []
    for model in models:
        heads = int(model["heads"])
        for block in range(int(model["depth"])):
            identity = port(model["model_id"], block, "input", "add1", 0)
            attention = port(model["model_id"], block, "attn.proj_drop.output", "add1", 1)
            projection = port(model["model_id"], block, "attn.av_concat", "attn.proj", 0)
            head_ports = [port(model["model_id"], block, "attn.av_concat", "attn.proj", 0, f"HEAD[{head}]") for head in range(heads)]
            q_ports = [port(model["model_id"], block, "attn.q_norm.output", "attn.score_matmul", 0, f"HEAD[{head}]") for head in range(heads)]
            k_ports = [port(model["model_id"], block, "attn.k_norm.output", "attn.score_matmul", 1, f"HEAD[{head}]") for head in range(heads)]
            v_ports = [port(model["model_id"], block, "attn.qkv.v", "attn.av_matmul", 1, f"HEAD[{head}]") for head in range(heads)]
            carry = port(model["model_id"], block, "add1.output", "add2", 0)
            mlp = port(model["model_id"], block, "mlp.drop2.output", "add2", 1)
            fc1 = port(model["model_id"], block, "norm2.output", "mlp.fc1", 0)
            gelu = port(model["model_id"], block, "mlp.fc1.output", "mlp.gelu", 0)
            fc2 = port(model["model_id"], block, "mlp.gelu.output", "mlp.fc2", 0)
            cuts = [
                ("RESIDUAL1_FULL", [identity, attention]),
                ("HEADS_WITH_BYPASS_FULL", [identity, *head_ports]),
                ("QKV_WITH_BYPASS_FULL", [identity, *q_ports, *k_ports, *v_ports]),
                ("ROUTING_VALUE_WITH_BYPASS_FULL", [identity, *q_ports, *k_ports, *v_ports]),
                ("PROJECTION_WITH_BYPASS_FULL", [identity, projection]),
                ("RESIDUAL2_FULL", [carry, mlp]),
                ("MLP_FC1_WITH_CARRY_FULL", [carry, fc1]),
                ("MLP_GELU_WITH_CARRY_FULL", [carry, gelu]),
                ("MLP_FC2_WITH_CARRY_FULL", [carry, fc2]),
            ]
            for cut_id, ports in cuts:
                registries.append({
                    "cut_id": cut_id, "model_id": model["model_id"], "block_index": block,
                    "reconstructs": "full ground-truth logit input gradient", "legal_path_intersection_count": 1,
                    "port_ids": ports, "port_count": len(ports),
                })
    registry = {
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP4",
        "status": "FROZEN_MACHINE_READABLE_LEGAL_CUTS",
        "port_id_fields": ["model_id", "graph_revision", "block_index", "producer_tensor_id", "consumer_operator_id", "input_slot", "invocation_index", "slice_spec"],
        "semantics": "each entry is one runtime producer-to-consumer use; head slices are mutually exclusive and exhaustive",
        "cuts": registries, "cut_count": len(registries),
    }
    registry["canonical_digest"] = canonical(registry)
    write_new(ROOT / "vit_port_registry.json", registry)
    print(json.dumps({"status": "PASS_STEP4_FREEZE", "source_digest": source["canonical_digest"], "cut_count": len(registries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
