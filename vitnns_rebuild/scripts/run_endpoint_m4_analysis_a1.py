#!/usr/bin/env python3
"""Pre-result frozen CPU-only descriptive M4-A1 analysis."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values(observed):
    return [float(value) for value in observed if value is not None and math.isfinite(float(value))]


def median(observed):
    observed = values(observed)
    return None if not observed else float(np.median(observed))


def spearman(left, right):
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 4:
        return None
    a, b = (np.asarray(value) for value in zip(*pairs))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def accepted_results():
    manifest = load(ROOT / "task_manifest.json")["tasks"]
    output = []
    for task in manifest:
        task_id = task["task_id"]
        attempts = sorted((ROOT / "results/attempts" / task_id).glob("attempt-*.json"))
        completed = [load(path) for path in attempts]
        completed = [row for row in completed if row["task_id"] == task_id]
        if len(completed) != 1:
            raise ValueError("ACCEPTED_RESULT:" + task_id)
        output.append(completed[0]["result"])
    return output


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    if any(name in sys.modules for name in ("torch", "timm", "PIL")):
        raise ValueError("CPU_ONLY_IMPORTS")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=REPO, text=True).split()[0]
    if head != remote:
        raise ValueError("PUBLISHED_ANALYSIS_AMENDMENT_REQUIRED")
    sources = ["docs/endpoint_m4_analysis_amendment_20260919.md", "scripts/run_endpoint_m4_analysis_a1.py"]
    bindings = {}
    for relative in sources:
        local = sha(REPO / relative)
        published = hashlib.sha256(subprocess.check_output(["git", "show", f"{head}:vitnns_rebuild/{relative}"], cwd=REPO)).hexdigest()
        if local != published:
            raise ValueError("PUBLISHED_ANALYSIS_SOURCE_DRIFT:" + relative)
        bindings[relative] = local
    freeze = {"stage": "ENDPOINT-M4-A1", "time_ns": time.time_ns(), "producer_commit": head,
              "remote_commit": remote, "sources": bindings, "new_inferential_tests": 0}
    (ROOT / "m4_a1_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    results = accepted_results(); summary_rows = []; image_rows = []
    phases = {"early": range(4), "middle": range(4, 8), "late": range(8, 12)}
    for result in results:
        base = {"model_id": result["model_id"], "sample_id": result["sample_id"]}
        for block in range(12):
            heads = [row for row in result["attention"] if row["block"] == block]
            for edge in ("q", "k", "routing", "value", "full"):
                for metric, getter in (
                    ("signed_magnitude", lambda row, edge=edge: abs(row[edge]["signed_sum"])),
                    ("absolute_mass", lambda row, edge=edge: row[edge]["absolute_mass"]),
                    ("map_cancellation", lambda row, edge=edge: row[edge]["map_cancellation"]),
                ):
                    image_rows.append({**base, "domain": "attention", "group": edge, "block": block,
                                       "metric": metric, "value": median(getter(row) for row in heads)})
            for kind in ("attention", "mlp"):
                row = next(item for item in result["residual"] if item["block"] == block and item["residual_type"] == kind)
                for edge in ("identity", "branch", "output"):
                    image_rows.extend((
                        {**base, "domain": "residual", "group": f"{kind}_{edge}", "block": block,
                         "metric": "signed_magnitude", "value": abs(row[edge]["signed_sum"])},
                        {**base, "domain": "residual", "group": f"{kind}_{edge}", "block": block,
                         "metric": "absolute_mass", "value": row[edge]["absolute_mass"]},
                        {**base, "domain": "residual", "group": f"{kind}_{edge}", "block": block,
                         "metric": "map_cancellation", "value": row[edge]["map_cancellation"]},
                    ))
        for kind in ("attention", "mlp"):
            selected = [next(item for item in result["residual"] if item["block"] == block and item["residual_type"] == kind) for block in range(12)]
            image_rows.append({**base, "domain": "residual_relation", "group": kind, "block": "",
                               "metric": "branch_share_path_cancellation_spearman",
                               "value": spearman([row["branch_share"] for row in selected], [row["path_cancellation"] for row in selected])})
        for phase, blocks in phases.items():
            for metric in ("identity_share", "path_cancellation", "output_identity_patch_tv"):
                by_kind = {}
                for kind in ("attention", "mlp"):
                    by_kind[kind] = median(next(item for item in result["residual"] if item["block"] == block and item["residual_type"] == kind)[metric] for block in blocks)
                image_rows.append({**base, "domain": "residual_type_contrast", "group": phase, "block": "",
                                   "metric": "attention_minus_mlp_" + metric,
                                   "value": None if None in by_kind.values() else by_kind["attention"] - by_kind["mlp"]})
    groups = sorted({(row["model_id"], row["domain"], row["group"], row["block"], row["metric"]) for row in image_rows})
    for key in groups:
        observed = values(row["value"] for row in image_rows if (row["model_id"], row["domain"], row["group"], row["block"], row["metric"]) == key)
        summary_rows.append({"model_id": key[0], "domain": key[1], "group": key[2], "block": key[3],
                             "metric": key[4], "n": len(observed), "na": 400 - len(observed), "median": median(observed)})
    write_csv(ROOT / "m4_a1_image_values.csv", image_rows)
    write_csv(ROOT / "m4_a1_descriptive_summary.csv", summary_rows)
    receipt = {"status": "PASS", "mode": "CPU_ONLY", "task_count": len(results),
               "image_value_rows": len(image_rows), "summary_rows": len(summary_rows),
               "new_inferential_tests": 0, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    (ROOT / "m4_a1_analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
