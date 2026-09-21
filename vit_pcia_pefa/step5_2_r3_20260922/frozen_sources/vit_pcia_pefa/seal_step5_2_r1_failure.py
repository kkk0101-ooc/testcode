#!/usr/bin/env python3
"""Seal the immutable Step-5.2 r1 P0 convergence failure."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import verify, write_new  # noqa: E402


def main() -> int:
    run = ROOT / "step5_2_r1_20260922"
    protocol = verify(run / "protocol.json"); decision = verify(run / "p0_decision.json")
    results = {model: verify(run / "p0" / f"{model}.json") for model in protocol["models"]}
    if decision["status"] != "INVALID_NUMERICAL_OR_INTERVENTION": raise RuntimeError("R1_NOT_INVALID")
    receipt = write_new(run / "failure_receipt.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "INVALID_NUMERICAL_OR_INTERVENTION", "failure_code": "FAIL_QUADRATURE_CONVERGENCE",
        "failed_gate": "P0_REAL_256_TO_512_CONVERGENCE",
        "model_convergence": {model: row["quadrature_convergence"] for model, row in results.items()},
        "clean_parity_all_pass": all(item["clean_parity"]["pass"] for row in results.values() for item in row["rows"]),
        "complete_cut_closure_all_pass": all(row["clean_closure"]["pass"] and all(item["closure"]["pass"] for item in row["rows"]) for row in results.values()),
        "dose_order_all_pass": all(row["dose_order"] for row in results.values()),
        "scientific_decision": "NONE_NUMERICALLY_INVALID", "prediction_started": False,
        "intervention_started": False, "evaluation_access": False, "step6_started": False,
        "p0_decision_digest": decision["canonical_digest"], "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": receipt["status"], "failure_code": receipt["failure_code"],
                      "digest": receipt["canonical_digest"]}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
