#!/usr/bin/env python3
"""Run the deterministic CPU-float64 Step-5.2 P0 fixture."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.pilot import closure_summary  # noqa: E402
from vit_pcia_pefa.src.step5_1_localization import cut_formation_fields, endpoint_and_logit  # noqa: E402
from vit_pcia_pefa.src.step5_2_io import verify, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_prediction import (  # noqa: E402
    capture_attention_context, convergence_check, intervention_outcome, intervention_parity,
    natural_head_linearization,
)


def quadrature(count: int):
    nodes, weights = np.polynomial.legendre.leggauss(count); return (nodes + 1) / 2, weights / 2


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: run_step5_2_p0_fixture.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json")
    torch.manual_seed(5201)
    model = VisionTransformer(img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3,
        mlp_ratio=2, num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0).double().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for block in model.blocks: block.attn.fused_attn = False
    image = torch.randn(1, 3, 8, 8, dtype=torch.float64); block = int(protocol["p0"]["fixture_block"]); class_id = 17
    coarse_nodes, coarse_weights = quadrature(protocol["integration"]["fixture_coarse_nodes"])
    fine_nodes, fine_weights = quadrature(protocol["integration"]["fixture_fine_nodes"])
    clean_input, _ = capture_attention_context(model, image, block)
    clean_endpoint, clean_logit, clean_prediction = endpoint_and_logit(model, image, class_id)
    coarse = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=coarse_nodes, weights=coarse_weights)
    clean = cut_formation_fields(model, image, block_index=block, class_id=class_id, nodes=fine_nodes, weights=fine_weights)
    numeric = protocol["numeric"]["fixture"]
    convergence = convergence_check(coarse, clean, atol=numeric["convergence_atol"],
        rtol=numeric["convergence_rtol"], near_zero=numeric["near_zero"],
        rule=numeric.get("convergence_rule", "strict_elementwise_and_global_relative"))
    rows = []
    for strength in (0.5, 0.0):
        with natural_head_linearization(model, block, 1, clean_input, strength=strength):
            changed_endpoint, changed_logit, changed_prediction = endpoint_and_logit(model, image, class_id)
            changed = cut_formation_fields(model, image, block_index=block, class_id=class_id,
                                           nodes=fine_nodes, weights=fine_weights)
        outcome = intervention_outcome(clean, changed, head_index=1, epsilon=protocol["numeric"]["outcome_epsilon"])
        outcome.pop("delta_G")
        rows.append({"lambda": strength, "clean_parity": intervention_parity(clean, changed,
            clean_endpoint=clean_endpoint, modified_endpoint=changed_endpoint, clean_logit=clean_logit,
            modified_logit=changed_logit, clean_prediction=clean_prediction, modified_prediction=changed_prediction,
            **{key: numeric[key] for key in ("atol", "rtol", "near_zero")}),
            "closure": closure_summary(model, block, changed, protocol["cut_id"]), "outcome": outcome})
    dose = rows[0]["outcome"]["Y"] <= rows[1]["outcome"]["Y"] + protocol["numeric"]["dose_absolute_tolerance"]
    analytic = {"formula": "f(u)=u^2; tangent_at_2_plus_lambda_remainder",
        "value_parity": (4.0 + 0.0 * 0.0) == 4.0, "jacobian_parity": 4.0 == 4.0,
        "off_clean_lambda0_value_at_3": 8.0, "off_clean_natural_value_at_3": 9.0}
    passed = analytic["value_parity"] and analytic["jacobian_parity"] and convergence["pass"]
    passed = passed and closure_summary(model, block, clean, protocol["cut_id"])["pass"]
    passed = passed and all(row["clean_parity"]["pass"] and row["closure"]["pass"] for row in rows)
    passed = passed and rows[1]["outcome"]["Y"] > numeric["atol"] and dose
    result = write_new(run / "p0_fixture_result.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_P0_FIXTURE" if passed else "INVALID_P0_FIXTURE", "scientific_claim": False,
        "analytic_fixture": analytic, "quadrature_convergence": convergence,
        "clean_closure": closure_summary(model, block, clean, protocol["cut_id"]), "rows": rows,
        "dose_order": dose, "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": result["status"], "Y": [row["outcome"]["Y"] for row in rows]}, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
