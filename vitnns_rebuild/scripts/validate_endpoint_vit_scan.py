#!/usr/bin/env python3
"""Recompute a saved endpoint atlas's hashes and numeric identities on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_json(path):
    value = json.loads(path.read_text())
    digest = value.pop("canonical_digest")
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    require(hashlib.sha256(body.encode()).hexdigest() == digest, f"canonical digest: {path}")
    return value


def validate(root):
    root = root.resolve()
    report = signed_json(root / "scan_report.json")
    manifest = signed_json(root / "artifact_manifest.json")
    require(report["status"] == "PASS_ENDPOINT_IMPLEMENTATION_DEMO", "report not PASS")
    require(not (root / "failure.json").exists(), "root has failure receipt")
    paths = [row["path"] for row in manifest["files"]]
    require(len(set(paths)) == len(paths), "duplicate manifest path")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root), "manifest path escapes root")
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"], f"artifact hash: {path}")
    for record in [report["source"], report["model"]]:
        require(sha256(Path(record["path"])) == record["sha256"], "source/model hash mismatch")
    for record in report["code"]:
        require(sha256(REPO / record["path"]) == record["sha256"], "producer code changed")
    require(report["scan_forward_count"] == 1, "scan used extra forward")
    require(report["baseline_forward_count"] == report["integration_steps"] == 0, "scan used baseline/path")
    require(report["runtime"]["device"] == "cpu" and report["runtime"]["CUDA_VISIBLE_DEVICES"] == "", "runtime device")
    require(report["formula"] == "L_s[i] = x[i] * d s(x) / d x[i]", "method formula")
    target_count = 0
    with np.load(root / "raw_arrays.npz", allow_pickle=False) as arrays:
        require(all(np.isfinite(arrays[key]).all() for key in arrays.files), "nonfinite raw data")
        x = arrays["input"]
        require(x.shape == (1, 3, 224, 224) and x.dtype == np.float64, "input contract")
        mean = np.array(report["preprocessing"]["mean"])[None, :, None, None]
        std = np.array(report["preprocessing"]["std"])[None, :, None, None]
        rgb = arrays["original_rgb"].transpose(2, 0, 1)[None] / 255
        np.testing.assert_allclose(x, (rgb - mean) / std, atol=0, rtol=0)
        expected_keys = {"input", "original_rgb", "logits", "independent_logits", "independent_class_endpoint"}
        for name, row in report["targets"].items():
            a = arrays[name + "__endpoint"]
            gradient = arrays[name + "__gradient"]
            expected_keys.update((name + "__endpoint", name + "__gradient"))
            require(a.shape == gradient.shape == x.shape, f"map shape: {name}")
            np.testing.assert_array_equal(a, x * gradient)
            np.testing.assert_allclose(a.sum(), row["attribution_sum"], atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(np.abs(a).sum(), row["absolute_mass"], atol=1e-9, rtol=1e-10)
            require(np.count_nonzero(a) == row["nonzero_count"] and row["absolute_mass"] > 0, f"degenerate map: {name}")
            np.testing.assert_allclose(row["value"] - a.sum(), row["local_affine_intercept"], atol=1e-9, rtol=1e-10)
            spec = row["definition"]
            if spec["kind"] in {"logit", "margin"}:
                score = arrays["logits"][0, spec["class_id"]]
                if spec["kind"] == "margin":
                    score -= arrays["logits"][0, spec["other_class_id"]]
            else:
                expected_keys.add(name + "__tokens")
                tokens = arrays[name + "__tokens"][0]
                if spec["kind"] == "patch_channel_sum":
                    score = tokens[0 if spec["module"] == "patch_embed" else 1:].sum()
                else:
                    score = tokens[spec["token"]]
            np.testing.assert_allclose(score, row["value"], atol=1e-9, rtol=1e-10)
            target_count += 1
        checks = {
            "native_logits_max_abs": float(np.max(np.abs(arrays["logits"] - arrays["independent_logits"]))),
            "independent_class_endpoint_max_abs": float(np.max(np.abs(arrays["independent_class_endpoint"] - arrays["class_logit__endpoint"]))),
            "qkv_closure_max_abs": float(np.max(np.abs(arrays["attention_component__full"] - arrays["attention_component__query"] - arrays["attention_component__key"] - arrays["attention_component__value"]))),
            "head_component_identity_max_abs": float(np.max(np.abs(arrays["attention_component__full"] - arrays[f"b{report['head_block']}_head0__endpoint"]))),
        }
        expected_keys.update("attention_component__" + key for key in ("full", "query", "key", "value"))
        for block in report["blocks"]:
            for name, left, right in (("residual1", "input", "attention"), ("residual2", "residual1", "mlp")):
                error = arrays[f"b{block}_{name}__endpoint"] - arrays[f"b{block}_{left}__endpoint"] - arrays[f"b{block}_{right}__endpoint"]
                checks[f"b{block}_{name}_closure_max_abs"] = float(np.abs(error).max())
            decomposition = report["decompositions"][str(block)]
            full_key = f"cut_b{block}__full"
            expected_keys.add(full_key)
            full = arrays[full_key]
            checks[f"b{block}_terminal_cut_identity_max_abs"] = float(np.abs(full - arrays["class_logit__endpoint"]).max())
            require(decomposition["target"] == report["targets"]["class_logit"]["definition"], "cut target differs from class target")
            components = {}
            for key, record in decomposition["components"].items():
                expected_keys.add(record["array_key"])
                component = arrays[record["array_key"]]
                require(component.shape == x.shape, "component shape")
                np.testing.assert_allclose(component.sum(), record["signed_endpoint_contribution"], atol=1e-9, rtol=1e-10)
                np.testing.assert_allclose(np.abs(component).sum(), record["absolute_endpoint_mass"], atol=1e-9, rtol=1e-10)
                components[key] = component
            np.testing.assert_array_equal(components["routing"], components["query"] + components["key"])
            heads = tuple(key for key in components if key.startswith("head_"))
            expected_cuts = {
                "residual1": ("identity", "attention"),
                "residual2": ("carry", "mlp"),
                "heads_branch": heads,
                "heads_with_bypass": ("identity",) + heads,
                "qkv_branch": ("query", "key", "value"),
                "routing_value_branch": ("routing", "value"),
                "qkv_with_bypass": ("identity", "query", "key", "value"),
                "routing_value_with_bypass": ("identity", "routing", "value"),
            }
            require(set(decomposition["cuts"]) == set(expected_cuts), "incomplete cut registry")
            for name, keys in decomposition["cuts"].items():
                require(set(keys) == set(expected_cuts[name]) and len(keys) == len(set(keys)), "missing or overlapping cut component")
                reconstruction = (
                    sum((components[key] for key in expected_cuts["heads_branch"]), np.zeros_like(full))
                    if name in {"heads_branch", "qkv_branch", "routing_value_branch"}
                    else full
                )
                error = sum((components[key] for key in keys), np.zeros_like(full)) - reconstruction
                maximum = float(np.abs(error).max())
                relative = float(np.linalg.norm(error) / max(np.linalg.norm(full), 1e-30))
                require(maximum == decomposition["closure_max_abs"][name], "cut max error mismatch")
                np.testing.assert_allclose(relative, decomposition["closure_relative_l2"][name], atol=1e-15, rtol=1e-8)
                checks[f"b{block}_terminal_{name}_closure_max_abs"] = maximum
        require(set(arrays.files) == expected_keys, "raw array key mismatch")
        for key, error in checks.items():
            require(error <= 1e-9, f"numeric identity: {key}")
            require(error == report["checks"][key], f"recorded error mismatch: {key}")
    return {"status": "PASS", "target_count": target_count, "max_recomputed_identity_error": max(checks.values()), "files_hashed": len(paths)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    print(json.dumps(validate(parser.parse_args().output_dir), indent=2))
