from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy
import timm
import torch
import torchvision

from .constants import MODEL_NAME, MODEL_SHA256
from .dataset import ManifestRow
from .io_utils import sha256_file
from .modeling import PreprocessSpec
from .provenance import git_state, topk_indices


STAGE_B_PROVENANCE_SCHEMA_VERSION = 1


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact logical tensor value in a device-independent representation."""
    value = tensor.detach().to(device="cpu").contiguous()
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{context} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{context} is not hexadecimal") from error


def _validate_layer_record(record: Mapping[str, Any], expected_block: int) -> None:
    required = {
        "block_index",
        "endpoint_parity",
        "conservation",
        "repeatability",
        "sanity",
        "signed_rgb_shape",
        "rgb_sum_shape",
        "fixed_attention_sha256",
        "direct_attention_recomputation_max_abs_error",
        "native_resolution",
        "resize_applied",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(
            f"Stage B layer {expected_block} provenance is missing {sorted(missing)}"
        )
    if record["block_index"] != expected_block:
        raise ValueError("Stage B layer provenance order does not match configured blocks")
    _require_sha256(
        record.get("fixed_attention_sha256"),
        f"layers[{expected_block}].fixed_attention_sha256",
    )
    if record.get("direct_attention_recomputation_max_abs_error", float("inf")) >= 1e-6:
        raise ValueError(f"Block {expected_block} direct attention parity did not pass")
    if record["endpoint_parity"].get("pass") is not True:
        raise ValueError(f"Block {expected_block} endpoint parity did not pass")
    if record["endpoint_parity"].get("max_abs_error", float("inf")) >= 1e-6:
        raise ValueError(f"Block {expected_block} endpoint parity is not < 1e-6")
    if record["conservation"].get("pass") is not True:
        raise ValueError(f"Block {expected_block} conservation did not pass")
    if record["conservation"].get("native_pass") is not True:
        raise ValueError(f"Block {expected_block} native signed-map conservation did not pass")
    if record["repeatability"].get("enabled") and record["repeatability"].get("pass") is not True:
        raise ValueError(f"Block {expected_block} repeatability did not pass")
    if record["sanity"].get("enabled") and record["sanity"].get("pass") is not True:
        raise ValueError(f"Block {expected_block} sanity test did not pass")
    if record["signed_rgb_shape"] != [3, 224, 224]:
        raise ValueError(f"Block {expected_block} signed RGB map is not native [3,224,224]")
    if record["rgb_sum_shape"] != [224, 224]:
        raise ValueError(f"Block {expected_block} RGB-sum map is not native [224,224]")
    if record["native_resolution"] is not True or record["resize_applied"] is not False:
        raise ValueError(f"Block {expected_block} provenance must state native, non-resized maps")


def build_stage_b_provenance(
    *,
    repo_root: Path,
    state_dict_path: Path,
    image_path: Path,
    config_path: Path,
    lock_path: Path,
    manifest_row: ManifestRow,
    preprocess: PreprocessSpec,
    direct_logits: torch.Tensor,
    direct_input: torch.Tensor,
    baseline: torch.Tensor,
    direct_input_sha256: str,
    device: torch.device,
    seed: int,
    config: Mapping[str, Any],
    resolved_class_index: int,
    layer_records: Sequence[Mapping[str, Any]],
    retention: Mapping[str, Any],
    code_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    state_dict_path = Path(state_dict_path).resolve()
    image_path = Path(image_path).resolve()
    config_path = Path(config_path).resolve()
    lock_path = Path(lock_path).resolve()
    if direct_logits.ndim != 2 or direct_logits.shape != (1, 1000):
        raise ValueError(f"Expected direct logits [1,1000], got {tuple(direct_logits.shape)}")
    if direct_input.shape != (1, 3, 224, 224):
        raise ValueError(f"Expected canonical direct input [1,3,224,224], got {tuple(direct_input.shape)}")
    if baseline.shape != direct_input.shape or baseline.dtype != direct_input.dtype:
        raise ValueError("Baseline must have the exact canonical input shape and dtype")
    if tensor_sha256(direct_input) != direct_input_sha256:
        raise ValueError("Provided direct input SHA-256 does not match the tensor")
    image_hash = sha256_file(image_path)
    if image_hash != manifest_row.image_sha256:
        raise ValueError("Image changed after manifest construction")
    actual_model_hash = sha256_file(state_dict_path)
    if actual_model_hash != MODEL_SHA256:
        raise ValueError("Canonical model hash changed before provenance construction")
    blocks = list(config.get("blocks", []))
    if [record.get("block_index") for record in layer_records] != blocks:
        raise ValueError("Layer records do not exactly cover the configured blocks")
    code = dict(code_state or git_state(repo_root))
    top5 = topk_indices(direct_logits.detach().cpu(), 5)
    ground_truth = manifest_row.imagenet1k_index
    sanity = {
        str(record["block_index"]): record["sanity"]
        for record in layer_records
    }
    return {
        "schema_version": STAGE_B_PROVENANCE_SCHEMA_VERSION,
        "stage": "B",
        "method": {
            "name": config["method"],
            "boundary": (
                "For each selected block, hold its endpoint attention A fixed, cut the "
                "current block's Q/K-to-A dependency, and integrate only the V path to one "
                "pre-projection per-head scalar O_h[query_token, head_channel] = (A_h @ V_h)[...]."
            ),
            "conservation_target": "block_internal_pre_projection_O_equals_A_matmul_V_scalar",
            "class_logit_is_value_path_target": False,
            "class_context": (
                "The resolved class index describes the direct model prediction context only; "
                "it is not the scalar conserved by this Value-path method."
            ),
            "attention_conditioning": "one endpoint A captured from the same input per block",
        },
        "target": {
            "value_path_scalar": dict(config["value_target"]),
            "class_context_spec": dict(config["class_target"]),
            "resolved_class_index": resolved_class_index,
        },
        "baseline": {
            "kind": config["baseline"],
            "tensor_sha256": tensor_sha256(baseline),
            "semantics": (
                "all zero values after canonical normalization"
                if config["baseline"] == "zero_normalized"
                else "RGB=(0,0,0) transformed by the canonical normalization"
            ),
        },
        "model": {
            "name": MODEL_NAME,
            "state_dict_path": str(state_dict_path),
            "state_dict_sha256": actual_model_hash,
            "attention_implementation": "unfused_timm_attention",
        },
        "input": {
            "sample_id": manifest_row.sample_id,
            "path": str(image_path),
            "image_sha256": image_hash,
            "direct_input_sha256": direct_input_sha256,
            "direct_and_attribution_tensor_shared": True,
            "ground_truth_wnid": manifest_row.wnid,
            "ground_truth_index": ground_truth,
            "label_source": manifest_row.label_source,
        },
        "preprocessing": preprocess.to_dict(),
        "prediction": {
            "top1_index": top5[0],
            "top5_indices": top5,
            "top1_correct": None if ground_truth is None else top5[0] == ground_truth,
            "top5_correct": None if ground_truth is None else ground_truth in top5,
        },
        "configuration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "schema_version": config["schema_version"],
            "formal": config["formal"],
            "blocks": blocks,
            "ig": dict(config["ig"]),
        },
        "layers": [dict(record) for record in layer_records],
        "sanity": {
            "kind": config["sanity"]["perturbation"],
            "parameters": dict(config["sanity"]),
            "per_block": sanity,
            "all_enabled_checks_pass": all(
                not record["sanity"].get("enabled")
                or record["sanity"].get("pass") is True
                for record in layer_records
            ),
        },
        "native_resolution": {
            "signed_rgb_shape_per_block": [3, 224, 224],
            "rgb_sum_shape_per_block": [224, 224],
            "signed_contributions": True,
            "rgb_reduction": "signed sum",
            "resize_applied": False,
            "statement": "Maps are attributed at native canonical input resolution, not resized.",
        },
        "retention": dict(retention),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "numpy": numpy.__version__,
        },
        "execution": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "input_dtype": str(direct_input.dtype),
            "metric_dtype": "float32",
            "seed": seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "code": code,
        "dependencies": {
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
        },
        "artifacts": {
            "metrics": "metric_ready/metrics.json",
            "provenance": "metric_ready/provenance.json",
            "maps": "metric_ready/maps.npz",
            "manifest": "manifest.json",
            "completion_status": "status.json",
        },
    }


def validate_stage_b_provenance(provenance: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "stage",
        "method",
        "target",
        "baseline",
        "model",
        "input",
        "preprocessing",
        "prediction",
        "configuration",
        "layers",
        "sanity",
        "native_resolution",
        "retention",
        "software",
        "execution",
        "code",
        "dependencies",
        "artifacts",
    }
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"Missing Stage B provenance fields: {sorted(missing)}")
    if provenance["schema_version"] != STAGE_B_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unexpected Stage B provenance schema version")
    if provenance["stage"] != "B":
        raise ValueError("Stage B provenance has the wrong stage")
    method = provenance["method"]
    if method.get("name") != "conditional_fixed_attention_value_path_ig":
        raise ValueError("Unexpected Stage B method name")
    if method.get("class_logit_is_value_path_target") is not False:
        raise ValueError("Stage B provenance must not misname a class logit as pure Value path")
    if provenance["model"].get("state_dict_sha256") != MODEL_SHA256:
        raise ValueError("Unexpected Stage B model hash")
    _require_sha256(provenance["input"].get("image_sha256"), "input.image_sha256")
    _require_sha256(
        provenance["input"].get("direct_input_sha256"),
        "input.direct_input_sha256",
    )
    _require_sha256(provenance["baseline"].get("tensor_sha256"), "baseline.tensor_sha256")
    _require_sha256(
        provenance["configuration"].get("sha256"), "configuration.sha256"
    )
    _require_sha256(
        provenance["dependencies"].get("lock_sha256"), "dependencies.lock_sha256"
    )
    if provenance["input"].get("direct_and_attribution_tensor_shared") is not True:
        raise ValueError("Direct and attribution paths must share the same tensor")
    if provenance["configuration"].get("formal") and provenance["code"].get("dirty"):
        raise ValueError("Formal Stage B provenance requires a clean Git tree")
    if provenance["configuration"].get("formal"):
        if provenance["sanity"].get("all_enabled_checks_pass") is not True:
            raise ValueError("Formal Stage B provenance requires passing sanity checks")
        if any(layer.get("repeatability", {}).get("pass") is not True for layer in provenance["layers"]):
            raise ValueError("Formal Stage B provenance requires passing repeatability checks")
    native = provenance["native_resolution"]
    if (
        native.get("signed_rgb_shape_per_block") != [3, 224, 224]
        or native.get("rgb_sum_shape_per_block") != [224, 224]
        or native.get("resize_applied") is not False
    ):
        raise ValueError("Stage B provenance does not describe native, non-resized maps")
    retention = provenance["retention"]
    for key in ("profile", "estimated_bytes", "actual_bytes", "budget_bytes"):
        if key not in retention:
            raise ValueError(f"Stage B retention provenance is missing {key!r}")
    if retention.get("within_budget") is not True:
        raise ValueError("Stage B retention preflight did not pass")
    if retention.get("storage_within_budget") is not True:
        raise ValueError("Stage B storage preflight did not pass")
    compute = retention.get("compute")
    if not isinstance(compute, Mapping) or compute.get("within_budget") is not True:
        raise ValueError("Stage B compute preflight did not pass")
    if compute.get("estimated_forward_calls", 1) > compute.get("max_forward_calls", 0):
        raise ValueError("Stage B estimated forward calls exceed its declared budget")
    if compute.get("estimated_backward_calls", 1) > compute.get("max_backward_calls", 0):
        raise ValueError("Stage B estimated backward calls exceed its declared budget")
    if retention["estimated_bytes"] > retention["budget_bytes"]:
        raise ValueError("Stage B estimated bytes exceed the declared budget")
    if retention["actual_bytes"] > retention["estimated_bytes"]:
        raise ValueError("Stage B actual bytes exceed the preflight estimate")
    if retention["actual_bytes"] > retention["budget_bytes"]:
        raise ValueError("Stage B actual bytes exceed the declared budget")
    blocks = provenance["configuration"].get("blocks")
    layers = provenance["layers"]
    if not isinstance(blocks, list) or not isinstance(layers, list) or len(blocks) != len(layers):
        raise ValueError("Stage B provenance has inconsistent block/layer records")
    for block, layer in zip(blocks, layers):
        if not isinstance(layer, Mapping):
            raise ValueError("Stage B layer provenance entries must be mappings")
        _validate_layer_record(layer, block)
