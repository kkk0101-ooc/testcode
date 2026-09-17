#!/usr/bin/env python3
"""Compute real B0 MLP, second-residual, and class-logit learning images."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_20260917"
R9 = REPO / "reports/nns_vit_unified_paper_figures_r9_20260917"
CHANNELS = [0, 255, 511, 767]
MODULES = ["first_residual", "mlp_branch", "block_output"]
TARGET_CLASS = 235


def checked(path: Path) -> dict:
    value = common.read(path)
    if not common.verify(value):
        raise ValueError(f"Bad canonical digest: {path}")
    return value


def main() -> None:
    started = time.monotonic()
    if (ROOT / "terminal_receipt.json").exists():
        raise RuntimeError("Sealed R10 root")

    r9_terminal = checked(R9 / "terminal_receipt.json")
    if r9_terminal["status"] != "PASS_PRESENTATION_RENDER":
        raise RuntimeError("R9 predecessor is not passed")
    r9_arrays = checked(R9 / "source_arrays_manifest.json")
    if r9_arrays["files"] != common.inventory(R9 / "source_arrays"):
        raise RuntimeError("R9 source arrays changed")
    r9_receipt = checked(R9 / "module_compute_receipt.json")
    if r9_receipt["status"] != "PASS_SINGLE_IMAGE_RESPONSE_MAPS":
        raise RuntimeError("R9 module computation is not passed")

    data = ROOT / "source_arrays"
    data.mkdir(parents=True, exist_ok=True)
    for row in r9_arrays["files"]:
        source = R9 / "source_arrays" / row["path"]
        if common.sha(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise RuntimeError(f"R9 array drift: {source}")
        shutil.copyfile(source, data / row["path"])

    photo = np.load(data / "original_rgb.npy", allow_pickle=False)
    sys.path.insert(0, str(REPO / "src"))
    from vitnns.modeling import load_canonical_model

    model_path = REPO / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = load_canonical_model(
        model_path,
        device="cpu",
        unfused_attention=True,
        expected_state_dict_sha256=r9_receipt["checkpoint_sha256"],
        expected_model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        expected_model_identity="F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES",
        expected_canonical_tensor_digest="150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
    ).eval()
    model.requires_grad_(False)

    architecture = {
        "blocks": len(model.blocks),
        "embedding_width": model.embed_dim,
        "patch_tokens": model.patch_embed.num_patches,
        "b0_mlp": [
            model.blocks[0].mlp.fc1.in_features,
            model.blocks[0].mlp.fc1.out_features,
            model.blocks[0].mlp.fc2.out_features,
        ],
        "mlp_activation": type(model.blocks[0].mlp.act).__name__,
        "classifier": [model.head.in_features, model.head.out_features],
        "global_pool": model.global_pool,
        "num_prefix_tokens": model.num_prefix_tokens,
    }
    expected_architecture = {
        "blocks": 12, "embedding_width": 768, "patch_tokens": 196,
        "b0_mlp": [768, 3072, 768], "mlp_activation": "GELU",
        "classifier": [768, 1000], "global_pool": "token",
        "num_prefix_tokens": 1,
    }
    if architecture != expected_architecture or model.attn_pool is not None:
        raise RuntimeError(f"Unexpected model architecture: {architecture}")

    x32 = (torch.from_numpy(photo.copy()).permute(2, 0, 1).float()[None] / 255 - 0.5) / 0.5
    captured: dict[str, np.ndarray] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().double().numpy().copy()
        return hook

    def capture_classifier_input(_module, inputs):
        captured["classifier_input"] = inputs[0].detach().double().numpy().copy()

    handles = [
        model.blocks[0].drop_path2.register_forward_hook(capture("mlp_branch")),
        model.blocks[0].register_forward_hook(capture("block_output")),
        model.norm.register_forward_hook(capture("final_norm")),
        model.head.register_forward_pre_hook(capture_classifier_input),
    ]
    with torch.no_grad():
        logits32 = model(x32)[0].double().numpy()
    for handle in handles:
        handle.remove()
    r9_logits = np.load(data / "logits.npy", allow_pickle=False)
    logit_error = abs(float(logits32[TARGET_CLASS]) - float(r9_logits[TARGET_CLASS]))
    if int(logits32.argmax()) != TARGET_CLASS or logit_error >= 1e-4:
        raise RuntimeError(f"Full-model parity failed: argmax={logits32.argmax()}, error={logit_error}")

    model.double()
    x = x32.double()

    def forward(z: torch.Tensor) -> dict[str, torch.Tensor]:
        patch = model.patch_embed(z)
        h = model.norm_pre(model.patch_drop(model._pos_embed(patch)))
        block0 = model.blocks[0]
        attention = block0.drop_path1(block0.ls1(block0.attn(block0.norm1(h))))
        first_residual = h + attention
        mlp_branch = block0.drop_path2(block0.ls2(block0.mlp(block0.norm2(first_residual))))
        block_output = first_residual + mlp_branch
        h = block_output
        for block in model.blocks[1:]:
            h = block(h)
        final_norm = model.norm(h)
        logits = model.forward_head(final_norm)
        return {
            "first_residual": first_residual[:, 1:],
            "mlp_branch": mlp_branch[:, 1:],
            "block_output": block_output[:, 1:],
            "final_norm": final_norm,
            "classifier_input": final_norm[:, 0],
            "logits": logits,
        }

    with torch.no_grad():
        clean = forward(x)
        baseline = forward(torch.zeros_like(x))

    parity = {
        "mlp_branch": float(np.max(np.abs(clean["mlp_branch"].numpy() - captured["mlp_branch"][:, 1:]))),
        "block_output": float(np.max(np.abs(clean["block_output"].numpy() - captured["block_output"][:, 1:]))),
        "final_norm": float(np.max(np.abs(clean["final_norm"].numpy() - captured["final_norm"]))),
        "classifier_input": float(np.max(np.abs(clean["classifier_input"].numpy() - captured["classifier_input"]))),
        "logits": float(np.max(np.abs(clean["logits"].numpy()[0] - logits32))),
    }
    if any(not np.isfinite(value) or value >= 1e-4 for value in parity.values()):
        raise RuntimeError(f"Float32/float64 forward parity failed: {parity}")

    for name in ["mlp_branch", "block_output"]:
        np.save(data / f"{name}_clean_tokens.npy", clean[name].numpy()[0], allow_pickle=False)
        np.save(data / f"{name}_baseline_tokens.npy", baseline[name].numpy()[0], allow_pickle=False)
    np.save(data / "final_norm_clean_tokens.npy", clean["final_norm"].numpy()[0], allow_pickle=False)
    np.save(data / "final_norm_baseline_tokens.npy", baseline["final_norm"].numpy()[0], allow_pickle=False)
    np.save(data / "classifier_input_clean.npy", clean["classifier_input"].numpy()[0], allow_pickle=False)
    np.save(data / "classifier_input_baseline.npy", baseline["classifier_input"].numpy()[0], allow_pickle=False)
    np.save(data / "r10_logits_clean.npy", clean["logits"].numpy()[0], allow_pickle=False)
    np.save(data / "r10_logits_baseline.npy", baseline["logits"].numpy()[0], allow_pickle=False)

    module_clean = np.stack([clean[name][0, :, CHANNELS].sum(0).numpy() for name in MODULES])
    module_base = np.stack([baseline[name][0, :, CHANNELS].sum(0).numpy() for name in MODULES])
    module_delta = module_clean - module_base
    classifier_clean = float(clean["logits"][0, TARGET_CLASS])
    classifier_base = float(baseline["logits"][0, TARGET_CLASS])
    classifier_delta = classifier_clean - classifier_base

    print("Model and forward parity passed; computing R10 extension maps", flush=True)
    module_attempts = []
    previous_modules = None
    final_modules = None
    module_order = None
    for order in [64, 128, 256, 512]:
        cache = data / f"b0_extension_rgb_ig_order_{order}.npy"
        if cache.exists():
            current_modules = np.load(cache, allow_pickle=False)
            if current_modules.shape != (3, 4, 3, 224, 224) or not np.isfinite(current_modules).all():
                raise RuntimeError(f"Invalid cached module map: {cache}")
            print(f"Module quadrature {order}: using validated cache", flush=True)
        else:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            alphas, weights = (nodes + 1) / 2, weights / 2
            module_maps = torch.zeros((3, 4, 3, 224, 224), dtype=torch.float64)
            for begin in range(0, order, 8):
                alpha = torch.from_numpy(alphas[begin:begin + 8])[:, None, None, None]
                weight = torch.from_numpy(weights[begin:begin + 8])[:, None, None, None]
                z = (alpha * x).requires_grad_(True)
                values = forward(z)
                targets = [
                    values[name][:, :, channel].sum()
                    for name in MODULES for channel in CHANNELS
                ]
                for target_index, target in enumerate(targets):
                    grad = torch.autograd.grad(target, z, retain_graph=target_index < len(targets) - 1)[0]
                    module_index, channel_index = divmod(target_index, 4)
                    module_maps[module_index, channel_index] += (weight * grad * x).sum(0).detach()
                if begin % 32 == 0:
                    print(f"Module quadrature {order}: {min(begin + 8, order)}/{order}", flush=True)
            current_modules = module_maps.numpy()
            np.save(cache, current_modules, allow_pickle=False)

        module_closure = np.abs(current_modules.sum((2, 3, 4)) - module_delta)
        residual_error = float(np.max(np.abs(current_modules[2] - current_modules[0] - current_modules[1])))
        module_change = None if previous_modules is None else (
            np.sum(np.abs(current_modules - previous_modules), axis=(2, 3, 4)) /
            np.maximum(np.sum(np.abs(current_modules), axis=(2, 3, 4)), 1e-12)
        )
        report = {
            "order": order,
            "completeness_abs": module_closure.tolist(),
            "map_L1_relative_change": None if module_change is None else module_change.tolist(),
            "second_residual_additivity_max_abs": residual_error,
        }
        module_attempts.append(report)
        print(json.dumps({"module_attempt": report}), flush=True)
        if (
            module_change is not None
            and np.all(module_change < 0.005)
            and np.all(module_closure < 1e-4 + 0.001 * np.abs(module_delta))
            and residual_error < 1e-9
        ):
            final_modules = current_modules
            module_order = order
            break
        previous_modules = current_modules

    if final_modules is None or module_order is None:
        common.write(ROOT / "module_compute_failed.json", common.signed({
            "status": "FAIL_MODULE_CONVERGENCE", "module_attempts": module_attempts,
        }))
        raise RuntimeError("R10 B0 module-map convergence failed")

    classifier_attempts = []
    previous_classifier = None
    final_classifier = None
    classifier_order = None
    for order in [64, 128, 256, 512, 1024, 2048, 4096]:
        cache = data / f"classifier_logit_rgb_ig_order_{order}.npy"
        if cache.exists():
            current_classifier = np.load(cache, allow_pickle=False)
            if current_classifier.shape != (3, 224, 224) or not np.isfinite(current_classifier).all():
                raise RuntimeError(f"Invalid cached classifier map: {cache}")
            print(f"Classifier quadrature {order}: using validated cache", flush=True)
        else:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            alphas, weights = (nodes + 1) / 2, weights / 2
            classifier_map = torch.zeros((3, 224, 224), dtype=torch.float64)
            for begin in range(0, order, 8):
                alpha = torch.from_numpy(alphas[begin:begin + 8])[:, None, None, None]
                weight = torch.from_numpy(weights[begin:begin + 8])[:, None, None, None]
                z = (alpha * x).requires_grad_(True)
                target = forward(z)["logits"][:, TARGET_CLASS].sum()
                grad = torch.autograd.grad(target, z)[0]
                classifier_map += (weight * grad * x).sum(0).detach()
                if begin % 128 == 0:
                    print(f"Classifier quadrature {order}: {min(begin + 8, order)}/{order}", flush=True)
            current_classifier = classifier_map.numpy()
            np.save(cache, current_classifier, allow_pickle=False)

        classifier_closure = abs(float(current_classifier.sum()) - classifier_delta)
        classifier_change = None if previous_classifier is None else (
            float(np.sum(np.abs(current_classifier - previous_classifier))) /
            max(float(np.sum(np.abs(current_classifier))), 1e-12)
        )
        report = {
            "order": order,
            "completeness_abs": classifier_closure,
            "map_L1_relative_change": classifier_change,
        }
        classifier_attempts.append(report)
        print(json.dumps({"classifier_attempt": report}), flush=True)
        if (
            classifier_change is not None
            and classifier_change < 0.005
            and classifier_closure < 1e-4 + 0.001 * abs(classifier_delta)
        ):
            final_classifier = current_classifier
            classifier_order = order
            break
        previous_classifier = current_classifier

    if final_classifier is None or classifier_order is None:
        common.write(ROOT / "module_compute_failed.json", common.signed({
            "status": "FAIL_CLASSIFIER_CONVERGENCE",
            "module_attempts": module_attempts, "classifier_attempts": classifier_attempts,
        }))
        raise RuntimeError("R10 classifier-map convergence failed")

    r9_rgb = np.load(data / "module_response_rgb_ig.npy", allow_pickle=False)
    r9_first_residual_error = float(np.max(np.abs(final_modules[0] - r9_rgb[2])))
    if r9_first_residual_error >= 1e-9:
        raise RuntimeError(f"R9 first-residual map mismatch: {r9_first_residual_error}")

    np.save(data / "b0_extension_rgb_ig.npy", final_modules, allow_pickle=False)
    np.save(data / "b0_extension_maps.npy", final_modules.sum(2), allow_pickle=False)
    np.save(data / "classifier_logit_rgb_ig.npy", final_classifier, allow_pickle=False)
    np.save(data / "classifier_logit_map.npy", final_classifier.sum(0), allow_pickle=False)

    binding_paths = [
        R9 / "terminal_receipt.json", R9 / "render_manifest.json",
        R9 / "source_arrays_manifest.json", R9 / "module_compute_receipt.json",
        model_path, Path(__file__), REPO / "src/vitnns/modeling.py",
        REPO / "src/vitnns/stage_x_nns_v2.py",
        REPO / "docs/unified_paper_figures_r10_design_20260917.md",
    ]
    sources = [
        {"path": os.path.relpath(path, REPO), "sha256": common.sha(path), "bytes": path.stat().st_size}
        for path in binding_paths
    ]
    common.write(ROOT / "module_compute_receipt.json", common.signed({
        "status": "PASS_SINGLE_IMAGE_VIT_EXTENSION_MAPS",
        "source_object_id": r9_receipt["source_object_id"],
        "selection": "first_frozen_Y2_cohort_sample",
        "channels": CHANNELS,
        "module_targets": MODULES,
        "classifier_target": {
            "class_id": TARGET_CLASS,
            "selection_rule": "clean argmax already fixed by predecessor source record",
            "clean_logit": classifier_clean,
            "baseline_logit": classifier_base,
            "delta": classifier_delta,
        },
        "block": 0,
        "architecture": architecture,
        "module_target_definition": "sum of one fixed channel over 196 patch-token outputs",
        "classifier_target_definition": "full model class-235 logit from final-normalized CLS through Linear(768,1000)",
        "reference": "zero in normalized input; RGB gray 0.5",
        "path": "straight line from reference to input; complete native forward graph",
        "module_raw_shape": list(final_modules.shape),
        "classifier_raw_shape": list(final_classifier.shape),
        "module_clean_values": module_clean.tolist(),
        "module_baseline_values": module_base.tolist(),
        "module_target_deltas": module_delta.tolist(),
        "module_attempts": module_attempts,
        "classifier_attempts": classifier_attempts,
        "selected_orders": {"module": module_order, "classifier": classifier_order},
        "second_residual_additivity_max_abs": module_attempts[-1]["second_residual_additivity_max_abs"],
        "r9_first_residual_map_max_abs": r9_first_residual_error,
        "float32_float64_forward_max_abs": parity,
        "full_model_logit_abs_error": logit_error,
        "checkpoint_sha256": r9_receipt["checkpoint_sha256"],
        "source_files": sources,
        "runtime": {
            "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_version": torch.__version__, "threads": 4, "dtype": "float64",
            "sys_executable": str(Path(sys.executable).resolve()),
            "wall_seconds": time.monotonic() - started,
        },
        "scope": "single-input method illustration; no new cohort or population inference",
    }))
    print("PASS: R10 MLP, second-residual and classifier maps saved", flush=True)


if __name__ == "__main__":
    main()
