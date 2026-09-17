#!/usr/bin/env python3
"""Generate a CPU original-input Endpoint NNS-ViT atlas for one RGB crop."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/endpoint-vit-scan-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns.endpoint_vit import (
    attention_endpoint_components, decompose_endpoint_vit, default_endpoint_targets, scan_endpoint_vit,
)
from vitnns.modeling import canonical_preprocess_spec, load_canonical_model

DEFAULT_SOURCE = REPO / "reports/nns_vit_unified_paper_figures_r9_20260917/source_arrays/original_rgb.npy"
DEFAULT_SOURCE_SHA = "25f9343d5042a801080fbea7a637464f47cfbada9553a3073606bc78182b2f60"
MODEL_SHA = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
TENSOR_SHA = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    value = {**value, "canonical_digest": hashlib.sha256(body.encode()).hexdigest()}
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def panel(axis, value: np.ndarray, title: str, *, learning: bool) -> float:
    limit = float(np.abs(value).max())
    scaled = value / limit if limit else np.zeros_like(value)
    if learning:
        scaled = np.sign(scaled) * np.log1p(9 * np.abs(scaled)) / np.log(10)
    axis.imshow(scaled, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel(f"max |value| = {limit:.3g}", fontsize=8)
    return limit


def render(root: Path, photo: np.ndarray, scan, blocks: list[int], head_block: int, patch_grid: tuple[int, int], prefix: int, decompositions) -> list[dict]:
    figures = root / "figures"
    figures.mkdir()
    records = []

    def atlas(names, title, filename, columns):
        groups = (len(names) + columns - 1) // columns
        figure, axes = plt.subplots(groups * 2, columns, figsize=(2.5 * columns, 6.0 * groups), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        limits = {}
        for index, name in enumerate(names):
            row, column = divmod(index, columns)
            top, bottom = axes[2 * row, column], axes[2 * row + 1, column]
            top.axis("on")
            bottom.axis("on")
            unit = scan.units[name]
            offset = 0 if unit.target.module == "patch_embed" else prefix
            values = unit.token_values[0, offset:].numpy().reshape(patch_grid)
            limits[name] = {
                "token_abs_max": panel(top, values, name + " / token values", learning=False),
                "map_abs_max": panel(bottom, unit.attribution[0].sum(0).numpy(), "Endpoint learning image", learning=True),
            }
            if column == 0:
                top.set_ylabel("Native forward", fontsize=10)
                bottom.set_ylabel("NNS endpoint", fontsize=10)
        figure.suptitle(title, fontsize=14)
        figure.text(0.5, 0.006, "Red: positive; blue: negative. Each panel uses its own symmetric scale; learning images use signed log display.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=2.4)
        figure.savefig(figures / filename, dpi=150)
        plt.close(figure)
        records.append({"path": "figures/" + filename, "targets": names, "display_limits": limits})

    for index, block in enumerate(blocks):
        names = (["patch_embedding"] if index == 0 else []) + [
            f"b{block}_{key}" for key in ("input", "attention", "residual1", "mlp_fc1", "mlp_gelu", "mlp", "residual2")
        ]
        atlas(names, f"Endpoint NNS-ViT / block {block} / channel 0", f"{index + 1:02d}_block_{block:02d}.png", len(names))
    names = [name for name in scan.units if name.startswith(f"b{head_block}_head")]
    atlas(names, f"Block {head_block} heads / top: patch O_h values; bottom: CLS O_h target / channel 0", f"{len(blocks)+1:02d}_head_atlas.png", min(6, len(names)))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(photo)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    unit = scan.units["class_logit"]
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, f"Full ViT -> final CLS -> logit\n\nclass {unit.target.class_id}\nlogit = {float(unit.value.item()):.4f}\n\nL = x * gradient(logit)", ha="center", va="center", fontsize=13)
    limit = panel(axes[2], unit.attribution[0].sum(0).numpy(), "Signed endpoint learning image", learning=True)
    figure.tight_layout()
    class_file = f"{len(blocks)+2:02d}_classifier.png"
    figure.savefig(figures / class_file, dpi=150)
    plt.close(figure)
    records.append({"path": "figures/" + class_file, "targets": ["class_logit"], "display_limits": {"map_abs_max": limit}})
    atlas([f"b{b}_unit_t1_c0" for b in blocks], "Single internal units / target: first patch token, channel 0", f"{len(blocks)+3:02d}_single_units.png", len(blocks))
    for index, (block, decomposition) in enumerate(decompositions.items()):
        keys = ["identity", "attention", "carry", "mlp", "query", "key", "value", "routing"]
        keys += [key for key in decomposition.components if key.startswith("head_")]
        maps = {"full": decomposition.full, **decomposition.components}
        names = ["full"] + keys
        columns = 7
        rows = (len(names) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.8 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        limits = {}
        for axis, name in zip(axes.flat, names):
            axis.axis("on")
            limits[name] = panel(axis, maps[name][0].sum(0).numpy(), name, learning=True)
        figure.suptitle(f"Class {decomposition.target.class_id} endpoint edge decomposition / block {block}", fontsize=14)
        figure.text(0.5, 0.006, "Separate cuts: identity + attention; carry + MLP; identity + all heads; identity + Q + K + V. Routing = Q + K. Per-panel scales.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=2.0)
        filename = f"{len(blocks)+4+index:02d}_block_{block:02d}_edge_decomposition.png"
        figure.savefig(figures / filename, dpi=150)
        plt.close(figure)
        records.append({"path": "figures/" + filename, "targets": [decomposition.target.name], "display_limits": limits})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rgb-npy", type=Path, default=DEFAULT_SOURCE, help="uint8 [224,224,3] already-resized RGB crop")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--head-block", type=int, default=0)
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(f"choose a fresh output directory: {root}")
    source = args.input_rgb_npy.resolve()
    source_sha = sha256(source)
    expected = args.expected_input_sha256 or (DEFAULT_SOURCE_SHA if source == DEFAULT_SOURCE.resolve() else None)
    if expected is not None and source_sha != expected:
        raise ValueError("input SHA-256 mismatch")
    photo = np.load(source, allow_pickle=False)
    if photo.shape != (224, 224, 3) or photo.dtype != np.uint8:
        raise ValueError("provide a uint8 RGB crop with shape [224,224,3]")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    model_path = REPO / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
    model = load_canonical_model(
        model_path, device="cpu", unfused_attention=True,
        expected_state_dict_sha256=MODEL_SHA,
        expected_model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        expected_model_identity="F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES",
        expected_canonical_tensor_digest=TENSOR_SHA,
    ).eval().double()
    model.requires_grad_(False)
    preprocess = canonical_preprocess_spec(model)
    mean = torch.tensor(preprocess.mean, dtype=torch.float64)[None, :, None, None]
    std = torch.tensor(preprocess.std, dtype=torch.float64)[None, :, None, None]
    image = (torch.from_numpy(photo.copy()).permute(2, 0, 1)[None].double() / 255 - mean) / std
    targets = default_endpoint_targets(model, blocks=args.blocks, head_block=args.head_block, class_id=args.class_id)
    root.mkdir(parents=True)
    try:
        print(f"Scanning {len(targets)} native targets on CPU", flush=True)
        begin_scan = time.monotonic()
        scan = scan_endpoint_vit(model, image, targets)
        scan_seconds = time.monotonic() - begin_scan
        arrays = {"input": image.numpy(), "original_rgb": photo, "logits": scan.logits.numpy()}
        target_records = {}
        for name, unit in scan.units.items():
            arrays[name + "__gradient"] = unit.gradient.numpy()
            arrays[name + "__endpoint"] = unit.attribution.numpy()
            if unit.token_values is not None:
                arrays[name + "__tokens"] = unit.token_values.numpy()
            target_records[name] = {
                "definition": asdict(unit.target), "value": float(unit.value.item()),
                "attribution_sum": float(unit.attribution_sum.item()),
                "local_affine_intercept": float(unit.local_affine_intercept.item()),
                "absolute_mass": float(unit.attribution.abs().sum()),
                "nonzero_count": int(torch.count_nonzero(unit.attribution)),
            }
        print("Checking native class gradient, residual sums and Q/K/V edge closure", flush=True)
        direct_input = image.clone().requires_grad_(True)
        direct_logits = model(direct_input)
        class_id = scan.units["class_logit"].target.class_id
        direct_gradient = torch.autograd.grad(direct_logits[:, class_id].sum(), direct_input)[0]
        arrays["independent_logits"] = direct_logits.detach().numpy()
        arrays["independent_class_endpoint"] = (image * direct_gradient).detach().numpy()
        components = attention_endpoint_components(model, image, block=args.head_block, head=0)
        for key, value in components.items():
            arrays["attention_component__" + key] = value.numpy()
        checks = {
            "native_logits_max_abs": float((scan.logits - direct_logits.detach()).abs().max()),
            "independent_class_endpoint_max_abs": float(np.abs(arrays["independent_class_endpoint"] - arrays["class_logit__endpoint"]).max()),
            "qkv_closure_max_abs": float((components["full"] - components["query"] - components["key"] - components["value"]).abs().max()),
            "head_component_identity_max_abs": float((components["full"] - scan.units[f"b{args.head_block}_head0"].attribution).abs().max()),
        }
        for block in args.blocks:
            for name, left, right in (("residual1", "input", "attention"), ("residual2", "residual1", "mlp")):
                error = arrays[f"b{block}_{name}__endpoint"] - arrays[f"b{block}_{left}__endpoint"] - arrays[f"b{block}_{right}__endpoint"]
                checks[f"b{block}_{name}_closure_max_abs"] = float(np.abs(error).max())
        decompositions = {}
        decomposition_records = {}
        for block in args.blocks:
            print(f"Decomposing class {class_id} at block {block} input edges", flush=True)
            decomposition = decompose_endpoint_vit(model, image, block=block, target=scan.units["class_logit"].target)
            decompositions[block] = decomposition
            arrays[f"cut_b{block}__full"] = decomposition.full.numpy()
            component_records = {}
            for key, value in decomposition.components.items():
                array_key = f"cut_b{block}__{key}"
                arrays[array_key] = value.numpy()
                component_records[key] = {
                    "component_id": key, "target_id": decomposition.target.name, "array_key": array_key,
                    "signed_endpoint_contribution": float(value.sum()), "absolute_endpoint_mass": float(value.abs().sum()),
                }
            decomposition_records[str(block)] = {
                "target": asdict(decomposition.target), "value": float(decomposition.value.item()),
                "components": component_records, "cuts": decomposition.cuts,
                "closure_max_abs": decomposition.closure_max_abs,
                "closure_relative_l2": decomposition.closure_relative_l2,
            }
            checks[f"b{block}_terminal_cut_identity_max_abs"] = float(np.abs(arrays[f"cut_b{block}__full"] - arrays["class_logit__endpoint"]).max())
            for key, error in decomposition.closure_max_abs.items():
                checks[f"b{block}_terminal_{key}_closure_max_abs"] = error
        np.savez_compressed(root / "raw_arrays.npz", **arrays)
        figures = render(root, photo, scan, args.blocks, args.head_block, tuple(model.patch_embed.grid_size), model.num_prefix_tokens, decompositions)
        passed = all(error <= 1e-9 for error in checks.values()) and all(row["absolute_mass"] > 0 for row in target_records.values())
        code_paths = [Path(__file__).resolve(), REPO / "src/vitnns/endpoint_vit.py", REPO / "src/vitnns/endpoint_nns.py", REPO / "src/vitnns/modeling.py"]
        report = {
            "status": "PASS_ENDPOINT_IMPLEMENTATION_DEMO" if passed else "FAIL_ENDPOINT_IMPLEMENTATION_DEMO",
            "scope": "one-image original-input internal scanner demonstration",
            "method": "Endpoint NNS-ViT (Gradient x Input)",
            "formula": "L_s[i] = x[i] * d s(x) / d x[i]",
            "input_coordinates": "normalized model input; seeds are the input values themselves",
            "preprocessing": preprocess.to_dict(),
            "baseline_forward_count": 0, "integration_steps": 0,
            "scan_forward_count": scan.forward_count, "validation_forward_count": 2,
            "decomposition_forward_count": len(decompositions),
            "model_constants": "held fixed in forward and Jacobian; no spatial constant allocation",
            "local_affine_intercept_meaning": "s(x) - sum_i L_s[i]; local Taylor intercept, not a complete bias decomposition",
            "source": {"path": str(source), "sha256": source_sha},
            "model": {"path": str(model_path), "sha256": sha256(model_path), "canonical_tensor_digest": TENSOR_SHA},
            "code": [{"path": str(path.relative_to(REPO)), "sha256": sha256(path)} for path in code_paths],
            "git_head_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "git_status_at_execution": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines(),
            "blocks": args.blocks, "head_block": args.head_block,
            "targets": target_records, "checks": checks,
            "decompositions": decomposition_records,
            "display": {"rgb_reduction": "signed sum for display only", "map_transform": "sign(a)*log1p(9*abs(a)/max_abs)/log(10)", "smoothing": False, "token_values": "same channel over patch tokens; head target is CLS, whose token is excluded from the spatial grid"},
            "figures": figures,
            "runtime": {"device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "dtype": "float64", "torch_version": torch.__version__, "sys_executable": str(Path(sys.executable).resolve()), "threads": args.threads, "scan_seconds": scan_seconds, "total_seconds": time.monotonic() - started},
        }
        write_json(root / "scan_report.json", report)
        write_json(root / "artifact_manifest.json", {"files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file()
        ]})
        print(json.dumps({"status": report["status"], "targets": len(targets), "scan_seconds": scan_seconds, "checks": checks}, indent=2), flush=True)
        if not passed:
            raise RuntimeError("implementation validation failed; preserve this root")
    except Exception as error:
        write_json(root / "failure.json", {"status": "FAIL", "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
