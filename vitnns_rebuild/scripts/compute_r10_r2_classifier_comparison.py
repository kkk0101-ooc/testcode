#!/usr/bin/env python3
"""Compute runner-up, low-score, and margin classifier learning images."""
from __future__ import annotations

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r2_20260917"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r10_r1_20260917"
CORRECT = 235
COMPETITOR = 174
LOW = 171
LABELS = {235: "German shepherd", 174: "Norwegian elkhound", 171: "Italian greyhound"}
WNIDS = {235: "n02106662", 174: "n02091467", 171: "n02091032"}
ORDERS = [2048, 4096, 8192]


def checked(path: Path) -> dict:
    value = common.read(path)
    if not common.verify(value):
        raise ValueError(f"Bad canonical digest: {path}")
    return value


def main() -> None:
    started = time.monotonic()
    if (ROOT / "terminal_receipt.json").exists():
        raise RuntimeError("Sealed R10-R2 root")
    old_terminal = checked(OLD / "terminal_receipt.json")
    if old_terminal["status"] != "PASS_PRESENTATION_RENDER" or old_terminal["revision"] != "R10-R1":
        raise RuntimeError("R10-R1 predecessor is not passed")
    old_receipt = checked(OLD / "module_compute_receipt.json")
    old_arrays = checked(OLD / "source_arrays_manifest.json")
    if old_arrays["files"] != common.inventory(OLD / "source_arrays"):
        raise RuntimeError("R10-R1 source arrays changed")

    data = ROOT / "source_arrays"
    data.mkdir(parents=True, exist_ok=True)
    for row in old_arrays["files"]:
        source = OLD / "source_arrays" / row["path"]
        if common.sha(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise RuntimeError(f"R10-R1 source array drift: {source}")
        shutil.copyfile(source, data / row["path"])

    logits_clean = np.load(data / "r10_logits_clean.npy", allow_pickle=False)
    logits_base = np.load(data / "r10_logits_baseline.npy", allow_pickle=False)
    order = np.argsort(logits_clean)[::-1]
    if int(order[0]) != CORRECT or int(order[1]) != COMPETITOR or int(np.argmin(logits_clean)) != LOW:
        raise RuntimeError("Frozen classifier selection rules no longer match")

    validation_labels = REPO / "assets/imagenet/imagenet_2012_validation_synset_labels.txt"
    official_wnid = validation_labels.read_text(encoding="utf-8").splitlines()[7921].strip()
    if official_wnid != WNIDS[CORRECT]:
        raise RuntimeError(f"Official label mismatch: {official_wnid}")

    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    for class_id in [CORRECT, COMPETITOR, LOW]:
        if info.label_names()[class_id] != WNIDS[class_id] or LABELS[class_id] not in info.index_to_description(class_id):
            raise RuntimeError(f"ImageNet class mapping mismatch: {class_id}")

    sys.path.insert(0, str(REPO / "src"))
    from vitnns.modeling import load_canonical_model
    model_path = REPO / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = load_canonical_model(
        model_path, device="cpu", unfused_attention=True,
        expected_state_dict_sha256=old_receipt["checkpoint_sha256"],
        expected_model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        expected_model_identity="F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES",
        expected_canonical_tensor_digest="150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
    ).eval().double()
    model.requires_grad_(False)
    photo = np.load(data / "original_rgb.npy", allow_pickle=False)
    x = ((torch.from_numpy(photo.copy()).permute(2, 0, 1).double()[None] / 255 - 0.5) / 0.5)
    with torch.no_grad():
        parity_logits = model(x)[0].numpy()
    parity_error = float(np.max(np.abs(parity_logits - logits_clean)))
    if parity_error >= 1e-4 or int(parity_logits.argmax()) != CORRECT:
        raise RuntimeError(f"Full-model parity failed: {parity_error}")

    target_ids = [COMPETITOR, LOW]
    deltas = {class_id: float(logits_clean[class_id] - logits_base[class_id]) for class_id in [CORRECT, COMPETITOR, LOW]}
    margin_delta = deltas[CORRECT] - deltas[COMPETITOR]
    previous = None
    attempts = []
    selected = None
    final = None
    for quadrature_order in ORDERS:
        cache = data / f"classifier_comparison_rgb_ig_order_{quadrature_order}.npy"
        if cache.exists():
            current = np.load(cache, allow_pickle=False)
            if current.shape != (2, 3, 224, 224) or not np.isfinite(current).all():
                raise RuntimeError(f"Invalid cached comparison map: {cache}")
            print(f"Classifier comparison {quadrature_order}: using validated cache", flush=True)
        else:
            nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
            alphas, weights = (nodes + 1) / 2, weights / 2
            maps = torch.zeros((2, 3, 224, 224), dtype=torch.float64)
            for begin in range(0, quadrature_order, 8):
                alpha = torch.from_numpy(alphas[begin:begin + 8])[:, None, None, None]
                weight = torch.from_numpy(weights[begin:begin + 8])[:, None, None, None]
                z = (alpha * x).requires_grad_(True)
                logits = model(z)
                for index, class_id in enumerate(target_ids):
                    grad = torch.autograd.grad(
                        logits[:, class_id].sum(), z, retain_graph=index < len(target_ids) - 1
                    )[0]
                    maps[index] += (weight * grad * x).sum(0).detach()
                if begin % 128 == 0:
                    print(f"Classifier comparison {quadrature_order}: {min(begin + 8, quadrature_order)}/{quadrature_order}", flush=True)
            current = maps.numpy()
            np.save(cache, current, allow_pickle=False)

        correct = np.load(data / f"classifier_logit_rgb_ig_order_{quadrature_order}.npy", allow_pickle=False)
        competitor, low = current
        margin = correct - competitor
        combined = np.stack([correct, competitor, margin, low])
        change = None if previous is None else (
            np.sum(np.abs(combined - previous), axis=(1, 2, 3)) /
            np.maximum(np.sum(np.abs(combined), axis=(1, 2, 3)), 1e-12)
        )
        closures = {
            "correct": abs(float(correct.sum()) - deltas[CORRECT]),
            "competitor": abs(float(competitor.sum()) - deltas[COMPETITOR]),
            "margin": abs(float(margin.sum()) - margin_delta),
            "low": abs(float(low.sum()) - deltas[LOW]),
        }
        margin_tolerance = 1e-4 + 0.001 * (abs(deltas[CORRECT]) + abs(deltas[COMPETITOR]))
        tolerances = {
            "correct": 1e-4 + 0.001 * abs(deltas[CORRECT]),
            "competitor": 1e-4 + 0.001 * abs(deltas[COMPETITOR]),
            "margin": margin_tolerance,
            "low": 1e-4 + 0.001 * abs(deltas[LOW]),
        }
        report = {
            "order": quadrature_order, "completeness_abs": closures,
            "completeness_tolerance": tolerances,
            "map_L1_relative_change": None if change is None else change.tolist(),
            "margin_identity_max_abs": float(np.max(np.abs(margin - (correct - competitor)))),
        }
        attempts.append(report)
        np.save(data / f"classifier_four_targets_rgb_ig_order_{quadrature_order}.npy", combined, allow_pickle=False)
        print(json.dumps(report), flush=True)
        if (change is not None and np.all(change < 0.005) and
                all(closures[key] < tolerances[key] for key in closures)):
            selected = quadrature_order
            final = combined
            break
        previous = combined

    if final is None or selected is None:
        common.write(ROOT / "classifier_comparison_failed.json", common.signed({
            "status": "FAIL_CLASSIFIER_COMPARISON_CONVERGENCE", "attempts": attempts,
        }))
        raise RuntimeError("Classifier comparison convergence failed")

    np.save(data / "classifier_four_targets_rgb_ig.npy", final, allow_pickle=False)
    np.save(data / "classifier_four_targets_maps.npy", final.sum(1), allow_pickle=False)
    sources = []
    for path in [
        OLD / "terminal_receipt.json", OLD / "module_compute_receipt.json",
        OLD / "source_arrays_manifest.json", validation_labels, model_path, Path(__file__),
        REPO / "src/vitnns/modeling.py", REPO / "docs/unified_paper_figures_r10_r2_design_20260917.md",
    ]:
        sources.append({
            "path": os.path.relpath(path, REPO), "bytes": path.stat().st_size,
            "sha256": common.sha(path),
        })
    targets = [
        {"id": "correct", "class_id": CORRECT, "wnid": WNIDS[CORRECT], "label": LABELS[CORRECT],
         "selection_rule": "official ground-truth class and clean argmax",
         "clean_logit": float(logits_clean[CORRECT]), "baseline_logit": float(logits_base[CORRECT]),
         "delta": deltas[CORRECT]},
        {"id": "competitor", "class_id": COMPETITOR, "wnid": WNIDS[COMPETITOR], "label": LABELS[COMPETITOR],
         "selection_rule": "highest clean logit after excluding correct class",
         "clean_logit": float(logits_clean[COMPETITOR]), "baseline_logit": float(logits_base[COMPETITOR]),
         "delta": deltas[COMPETITOR]},
        {"id": "margin", "positive_class_id": CORRECT, "negative_class_id": COMPETITOR,
         "selection_rule": "correct minus strongest competitor",
         "clean_logit": float(logits_clean[CORRECT] - logits_clean[COMPETITOR]),
         "baseline_logit": float(logits_base[CORRECT] - logits_base[COMPETITOR]), "delta": margin_delta},
        {"id": "low", "class_id": LOW, "wnid": WNIDS[LOW], "label": LABELS[LOW],
         "selection_rule": "global argmin of 1000 clean logits",
         "clean_logit": float(logits_clean[LOW]), "baseline_logit": float(logits_base[LOW]),
         "delta": deltas[LOW]},
    ]
    common.write(ROOT / "classifier_comparison_receipt.json", common.signed({
        "status": "PASS_SINGLE_IMAGE_CLASSIFIER_COMPARISON",
        "source_object_id": old_receipt["source_object_id"],
        "official_ground_truth": {"validation_image_number": 7922, "wnid": official_wnid,
                                    "model_class_index": CORRECT, "label": LABELS[CORRECT]},
        "targets": targets, "target_order": ["correct", "competitor", "margin", "low"],
        "reference": "zero in normalized input; RGB gray 0.5",
        "path": "straight line from reference to input; complete native 12-block forward graph",
        "orders": ORDERS, "attempts": attempts, "selected_order": selected,
        "raw_shape": list(final.shape), "map_shape": list(final.sum(1).shape),
        "shared_display_scale_required": True, "float64_forward_max_abs": parity_error,
        "checkpoint_sha256": old_receipt["checkpoint_sha256"], "source_files": sources,
        "runtime": {"device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "torch_version": torch.__version__, "threads": 4, "dtype": "float64",
                    "sys_executable": str(Path(sys.executable).resolve()),
                    "wall_seconds": time.monotonic() - started},
        "scope": "single-input classifier-output comparison; no population inference",
    }))
    print("PASS: classifier comparison maps saved", flush=True)


if __name__ == "__main__":
    main()
