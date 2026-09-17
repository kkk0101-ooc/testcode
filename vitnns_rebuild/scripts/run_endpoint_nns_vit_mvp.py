#!/usr/bin/env python3
"""Run the one-image Endpoint NNS-ViT method-identity pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/endpoint-nns-vit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_mvp_r2_20260917"
R9 = REPO / "reports/nns_vit_unified_paper_figures_r9_20260917"
SOURCE = R9 / "source_arrays/original_rgb.npy"
MODEL = REPO / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
CHANNEL = 0
BLOCK = 0
HEAD = 9

sys.path.insert(0, str(REPO / "src"))
from vitnns.attribution import integrated_gradients
from vitnns.endpoint_nns import endpoint_learning_image
from vitnns.modeling import load_canonical_model

ScalarCallback = Callable[[torch.Tensor], torch.Tensor]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value: dict) -> None:
    payload = dict(value)
    payload["canonical_digest"] = canonical(value)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_model() -> torch.nn.Module:
    model = load_canonical_model(
        MODEL,
        device="cpu",
        unfused_attention=True,
        expected_state_dict_sha256="489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
        expected_model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        expected_model_identity="F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES",
        expected_canonical_tensor_digest="150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
    ).eval().double()
    model.requires_grad_(False)
    return model


def block0_values(model: torch.nn.Module, image: torch.Tensor) -> dict[str, torch.Tensor]:
    patch = model.patch_embed(image)
    embedded = model.norm_pre(model.patch_drop(model._pos_embed(patch)))
    block = model.blocks[0]
    attention = block.drop_path1(block.ls1(block.attn(block.norm1(embedded))))
    residual1 = embedded + attention
    mlp = block.drop_path2(block.ls2(block.mlp(block.norm2(residual1))))
    residual2 = residual1 + mlp
    return {
        "patch_embedding": patch,
        "attention_branch": attention[:, 1:],
        "first_residual": residual1[:, 1:],
        "mlp_branch": mlp[:, 1:],
        "second_residual": residual2[:, 1:],
    }


def module_callback(model: torch.nn.Module, name: str) -> ScalarCallback:
    def callback(image: torch.Tensor) -> torch.Tensor:
        value = block0_values(model, image)[name]
        return value[:, :, CHANNEL].sum(1)
    return callback


def direct_gradient_times_input(
    callback: ScalarCallback, image: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    value = image.detach().clone().requires_grad_(True)
    target = callback(value)
    gradient = torch.autograd.grad(target.sum(), value)[0]
    return gradient.detach(), ((image - baseline) * gradient).detach()


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a_center = a - a.mean()
    b_center = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    centered_denominator = float(np.linalg.norm(a_center) * np.linalg.norm(b_center))
    nonzero = (a != 0) | (b != 0)
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "l1_relative_to_right": float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30)),
        "cosine": float(np.dot(a, b) / denominator) if denominator else 1.0,
        "pearson": float(np.dot(a_center, b_center) / centered_denominator) if centered_denominator else 1.0,
        "sign_agreement_nonzero_union": float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else 1.0,
    }


def compute_pair(
    callback: ScalarCallback,
    image: torch.Tensor,
    baseline: torch.Tensor,
    model: torch.nn.Module,
    *,
    steps: tuple[int, ...],
) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.monotonic()
    endpoint = endpoint_learning_image(
        callback, image, reference=baseline, parameters=model
    )
    endpoint_seconds = time.monotonic() - started
    gradient, independent = direct_gradient_times_input(callback, image, baseline)
    endpoint_identity = compare(
        endpoint.attribution.numpy(), independent.numpy()
    )
    started = time.monotonic()
    path = integrated_gradients(
        callback, image, baseline, steps=steps, step_chunk_size=4,
        atol=1e-4, rtol=0.01, parameters=model,
        metadata={"scope": "endpoint_nns_vit_single_image_mvp"},
    )
    path_seconds = time.monotonic() - started
    arrays = {
        "gradient": gradient.numpy(),
        "gradient_times_input": independent.numpy(),
        "endpoint": endpoint.attribution.numpy(),
        "path": path.attribution.double().numpy(),
    }
    report = {
        "endpoint_target": endpoint.target.tolist(),
        "endpoint_sum": endpoint.attribution_sum.tolist(),
        "response_delta": endpoint.response_delta.tolist(),
        "endpoint_response_reconstruction_error": endpoint.response_reconstruction_error.tolist(),
        "endpoint_vs_independent_gradient_times_input": endpoint_identity,
        "endpoint_vs_path": compare(arrays["endpoint"], arrays["path"]),
        "path_converged": bool(path.converged),
        "path_selected_steps": int(path.steps),
        "path_attempts": [dict(attempt) for attempt in path.attempts],
        "path_conservation": {
            key: value.detach().cpu().tolist() for key, value in path.conservation.items()
        },
        "endpoint_wall_seconds": endpoint_seconds,
        "path_wall_seconds": path_seconds,
    }
    return report, arrays


def fixed_attention_target(
    model: torch.nn.Module, image: torch.Tensor, baseline: torch.Tensor
) -> tuple[ScalarCallback, dict]:
    block = model.blocks[BLOCK]

    def qkv_values(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch = model.patch_embed(value)
        embedded = model.norm_pre(model.patch_drop(model._pos_embed(patch)))
        normalized = block.norm1(embedded)
        batch, tokens, width3 = block.attn.qkv(normalized).shape
        qkv = block.attn.qkv(normalized).reshape(
            batch, tokens, 3, block.attn.num_heads, width3 // 3 // block.attn.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q @ k.transpose(-2, -1) * block.attn.scale).softmax(-1)
        return attention, v

    with torch.no_grad():
        clean_attention, clean_value = qkv_values(image)
        _, baseline_value = qkv_values(baseline)
        fixed = clean_attention[:, HEAD].detach().clone()
        clean_output = (fixed @ clean_value[:, HEAD])[:, 0]
        baseline_output = (fixed @ baseline_value[:, HEAD])[:, 0]
        direction_delta = clean_output - baseline_output
        direction = direction_delta / direction_delta.norm(dim=1, keepdim=True)

    def callback(value: torch.Tensor) -> torch.Tensor:
        _, current_value = qkv_values(value)
        output = fixed.expand(value.shape[0], -1, -1) @ current_value[:, HEAD]
        return (output[:, 0] * direction.expand(value.shape[0], -1)).sum(1)

    return callback, {
        "block": BLOCK,
        "head": HEAD,
        "query_token": 0,
        "fixed_clean_attention": True,
        "direction_norm": float(direction_delta.norm().item()),
        "target_kind": "fixed_clean_attention_cls_head_direction",
    }


def spatial(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 4:
        value = value[0]
    return value.sum(0)


def signed_image(value: np.ndarray) -> tuple[np.ndarray, float]:
    limit = float(np.max(np.abs(value)))
    if limit == 0:
        return np.zeros_like(value), 0.0
    return np.sign(value) * np.log1p(9 * np.abs(value) / limit) / math.log(10), limit


def save_figure(path: Path, rows: list[tuple[str, np.ndarray, np.ndarray, dict]]) -> None:
    figure, axes = plt.subplots(len(rows), 3, figsize=(10.5, 2.8 * len(rows)), squeeze=False)
    for row, (label, endpoint, path_map, metrics) in enumerate(rows):
        difference = endpoint - path_map
        for column, (title, value) in enumerate((
            ("Endpoint NNS-ViT", endpoint),
            ("Path-integrated", path_map),
            ("Endpoint - path", difference),
        )):
            shown, _ = signed_image(value)
            axes[row, column].imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title, fontsize=11, weight="bold")
        axes[row, 0].text(
            -0.04, 0.5, label, transform=axes[row, 0].transAxes, ha="right", va="center",
            fontsize=9, rotation=90,
        )
        axes[row, 2].text(
            1.02, 0.5,
            f"r={metrics['pearson']:.3f}\ncos={metrics['cosine']:.3f}\nsign={metrics['sign_agreement_nonzero_union']:.3f}",
            transform=axes[row, 2].transAxes, ha="left", va="center", fontsize=8,
        )
    figure.suptitle("Endpoint and path-integrated learning images on one frozen input", fontsize=15, weight="bold")
    figure.tight_layout(rect=(0.05, 0.02, 0.92, 0.95))
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.monotonic()
    if ROOT.exists():
        raise RuntimeError(f"MVP root already exists: {ROOT}")
    r9_before = inventory(R9)
    ROOT.mkdir(parents=True)
    arrays_root = ROOT / "arrays"
    figures_root = ROOT / "figures"
    arrays_root.mkdir()
    figures_root.mkdir()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    photo = np.load(SOURCE, allow_pickle=False)
    image = (torch.from_numpy(photo.copy()).permute(2, 0, 1).double()[None] / 255 - 0.5) / 0.5
    baseline = torch.zeros_like(image)
    model = load_model()
    with torch.no_grad():
        logits = model(image)
    class_id = int(logits.argmax(1).item())

    module_names = (
        "patch_embedding", "attention_branch", "first_residual",
        "mlp_branch", "second_residual",
    )
    target_reports = {}
    all_arrays: dict[str, dict[str, np.ndarray]] = {}
    module_rows = []
    for name in module_names:
        print(f"module {name}", flush=True)
        report, arrays = compute_pair(
            module_callback(model, name), image, baseline, model, steps=(16, 32, 64, 128)
        )
        if not report["path_converged"]:
            raise RuntimeError(f"path IG failed for {name}")
        target_reports[name] = {
            "target_kind": "sum_fixed_channel_over_196_patch_tokens",
            "channel": CHANNEL,
            **report,
        }
        all_arrays[name] = arrays
        module_rows.append((
            name.replace("_", " "), spatial(arrays["endpoint"]), spatial(arrays["path"]),
            report["endpoint_vs_path"],
        ))

    class_callback = lambda value: model(value)[:, class_id]
    print("class logit", flush=True)
    class_report, class_arrays = compute_pair(
        class_callback, image, baseline, model, steps=(512, 1024)
    )
    if not class_report["path_converged"]:
        raise RuntimeError("class-logit path IG failed")
    target_reports["classifier_logit"] = {
        "target_kind": "clean_predicted_class_raw_logit",
        "class_id": class_id,
        **class_report,
    }
    all_arrays["classifier_logit"] = class_arrays

    print("conditional B0/H9 value path", flush=True)
    conditional_callback, conditional_definition = fixed_attention_target(model, image, baseline)
    conditional_report, conditional_arrays = compute_pair(
        conditional_callback, image, baseline, model, steps=(128, 256, 512)
    )
    if not conditional_report["path_converged"]:
        raise RuntimeError("conditional Value-path IG failed")
    target_reports["conditional_value_path"] = {
        **conditional_definition, **conditional_report,
    }
    all_arrays["conditional_value_path"] = conditional_arrays

    for target_name, arrays in all_arrays.items():
        for array_name, value in arrays.items():
            np.save(arrays_root / f"{target_name}__{array_name}.npy", value, allow_pickle=False)

    first = all_arrays["first_residual"]
    patch = all_arrays["patch_embedding"]
    attention = all_arrays["attention_branch"]
    second = all_arrays["second_residual"]
    mlp = all_arrays["mlp_branch"]
    additivity = {}
    for method in ("endpoint", "path", "gradient_times_input"):
        additivity[method] = {
            "first_residual_max_abs": float(np.max(np.abs(first[method] - patch[method] - attention[method]))),
            "second_residual_max_abs": float(np.max(np.abs(second[method] - first[method] - mlp[method]))),
        }

    save_figure(figures_root / "01_module_chain_endpoint_vs_path.png", module_rows)
    save_figure(figures_root / "02_terminal_targets_endpoint_vs_path.png", [
        ("classifier logit", spatial(class_arrays["endpoint"]), spatial(class_arrays["path"]), class_report["endpoint_vs_path"]),
        ("conditional B0 H9", spatial(conditional_arrays["endpoint"]), spatial(conditional_arrays["path"]), conditional_report["endpoint_vs_path"]),
    ])

    endpoint_identity_max = max(
        report["endpoint_vs_independent_gradient_times_input"]["max_abs"]
        for report in target_reports.values()
    )
    decision = {
        "status": "PASS_LEVEL_3" if endpoint_identity_max < 1e-10 else "FAIL_LEVEL_3",
        "endpoint_equals_gradient_times_input_max_abs": endpoint_identity_max,
        "all_path_integrals_converged": all(report["path_converged"] for report in target_reports.values()),
        "module_additivity": additivity,
        "level_4_authorization": "AUTHORIZED_20_IMAGE_EXPLORATORY_PILOT" if endpoint_identity_max < 1e-10 else "NOT_AUTHORIZED",
        "level_4_not_executed_in_this_script": True,
        "five_hundred_image_rerun_authorized": False,
    }
    report = {
        "status": decision["status"],
        "scope": "single frozen input method identity and full-chain pilot",
        "source_object_id": "ILSVRC2012_VAL_OFFICIAL:ILSVRC2012_val/ILSVRC2012_val_00007922.JPEG",
        "input_sha256": sha256(SOURCE),
        "checkpoint_sha256": sha256(MODEL),
        "class_id": class_id,
        "clean_class_logit": float(logits[0, class_id].item()),
        "reference": "normalized zero, equivalent to RGB 0.5",
        "module_channel": CHANNEL,
        "targets": target_reports,
        "decision": decision,
        "runtime": {
            "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "dtype": "float64", "torch_version": torch.__version__,
            "sys_executable": str(Path(sys.executable).resolve()),
            "wall_seconds": time.monotonic() - started,
        },
    }
    write_json(ROOT / "pilot_report.json", report)
    file_manifest = [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(ROOT.rglob("*")) if path.is_file()
    ]
    write_json(ROOT / "artifact_manifest.json", {"files": file_manifest})
    if inventory(R9) != r9_before:
        raise RuntimeError("sealed R9 changed during MVP")
    print(json.dumps({"status": decision["status"], "root": str(ROOT), "class_id": class_id}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
