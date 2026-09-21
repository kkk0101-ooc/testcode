#!/usr/bin/env python3
"""Execute the frozen CPU-float64 A0 fixture."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.pilot import closure_summary  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import (  # noqa: E402
    capture_projection_input, clean_parity, cut_formation_fields, difference_direction,
    endpoint_and_logit, localization_scores, projection_head_wrapper, rank_metrics,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value)
    body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_step5_1_a0_fixture.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve()
    protocol = verify_signed(run / "protocol.json")
    output = run / "a0_fixture_result.json"
    torch.manual_seed(5101)
    model = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3, mlp_ratio=2,
        num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    ).double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks:
        block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(protocol["integration"]["fixture_nodes"])
    nodes, weights = (raw_nodes + 1) / 2, raw_weights / 2
    block, class_id, true_head = protocol["a0"]["fixture_block"], 17, 1
    activation = capture_projection_input(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    clean = cut_formation_fields(
        model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights,
    )
    rows = []
    for strength in protocol["wrapper"]["a0_strengths"]:
        with projection_head_wrapper(model, block, true_head, activation, strength=float(strength)):
            changed_activation = capture_projection_input(model, image, block)
            changed_endpoint, changed_logit, changed_prediction = endpoint_and_logit(model, image, class_id)
            changed = cut_formation_fields(
                model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights,
            )
        parity_row = clean_parity(
            clean, changed, clean_endpoint=clean_endpoint, modified_endpoint=changed_endpoint,
            clean_logit=clean_logit, modified_logit=changed_logit,
            clean_prediction=clean_prediction, modified_prediction=changed_prediction,
            **{k: protocol["numeric"]["fixture"][k] for k in ("atol", "rtol", "near_zero")},
        )
        methods = localization_scores(
            clean, changed, clean_activation=activation, modified_activation=changed_activation,
            random_namespace=f"fixture|{strength}",
        )
        rows.append({
            "strength": strength, "clean_parity": parity_row,
            "clean_closure": closure_summary(model, block, clean, protocol["cut_id"]),
            "modified_closure": closure_summary(model, block, changed, protocol["cut_id"]),
            "pefa_R_rank": rank_metrics(methods["pefa_R_delta"], true_head, 1),
            "true_head_pefa_R_delta": methods["pefa_R_delta"][true_head],
            "difference_direction": difference_direction(clean, changed),
        })
    controls = []
    for mode, strength in (("bounded_quadratic", 0.0), ("linear_tangent", 1.0)):
        with projection_head_wrapper(model, block, true_head, activation, strength=strength, mode=mode):
            changed_activation = capture_projection_input(model, image, block)
            changed = cut_formation_fields(
                model, image, block_index=block, class_id=class_id, nodes=nodes, weights=weights,
            )
        methods = localization_scores(
            clean, changed, clean_activation=activation, modified_activation=changed_activation,
            random_namespace=mode,
        )
        controls.append({
            "mode": mode, "strength": strength,
            "maximum_change_score": max(max(methods[key]) for key in
                ("pefa_R_delta", "pcia_delta", "endpoint_delta", "gradient_delta", "activation_delta")),
            "closure": closure_summary(model, block, changed, protocol["cut_id"]),
        })
    signals = [row["true_head_pefa_R_delta"] for row in rows]
    tolerance = protocol["numeric"]["fixture"]["atol"]
    passed = (
        all(row["clean_parity"]["pass"] and row["clean_closure"]["pass"]
            and row["modified_closure"]["pass"] and row["difference_direction"]["pass"]
            for row in rows)
        and signals[0] > tolerance and all(a <= b + tolerance for a, b in zip(signals, signals[1:]))
        and all(row["maximum_change_score"] <= tolerance and row["closure"]["pass"] for row in controls)
    )
    write_new(output, {
        "schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_A0_FIXTURE" if passed else "FAIL_A0_FIXTURE",
        "dtype": "float64", "block_index": block, "class_id": class_id,
        "true_head": true_head, "rows": rows, "controls": controls,
        "dose_nondecreasing": all(a <= b + tolerance for a, b in zip(signals, signals[1:])),
        "scientific_claim": False, "completed_unix_ns": time.time_ns(),
    })
    print(json.dumps({"status": "PASS_A0_FIXTURE" if passed else "FAIL_A0_FIXTURE",
                      "signals": signals}, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
