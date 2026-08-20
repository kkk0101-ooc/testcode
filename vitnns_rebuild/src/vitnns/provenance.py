from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Sequence

import numpy
import timm
import torch
import torchvision

from .constants import MODEL_NAME, MODEL_SHA256
from .dataset import ManifestRow
from .extractor import ExtractionResult
from .io_utils import sha256_file
from .modeling import PreprocessSpec


def git_state(repo_root: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def topk_indices(logits: torch.Tensor, k: int = 5) -> list[int]:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("Stage A provenance currently requires one sample")
    return [int(value) for value in logits[0].topk(k).indices.cpu().tolist()]


def build_provenance(
    *,
    repo_root: Path,
    state_dict_path: Path,
    image_path: Path,
    manifest_row: ManifestRow,
    preprocess: PreprocessSpec,
    extraction: ExtractionResult,
    device: torch.device,
    lock_path: Path,
    seed: int,
    validation: dict,
) -> dict:
    git = git_state(repo_root)
    actual_image_hash = sha256_file(image_path)
    if actual_image_hash != manifest_row.image_sha256:
        raise ValueError(
            f"Image changed after manifest creation: expected {manifest_row.image_sha256}, "
            f"got {actual_image_hash}"
        )
    logits = extraction.logits.detach().cpu()
    top5 = topk_indices(logits, 5)
    ground_truth = manifest_row.imagenet1k_index
    return {
        "schema_version": 1,
        "model": {
            "name": MODEL_NAME,
            "state_dict_path": str(Path(state_dict_path).resolve()),
            "state_dict_sha256": MODEL_SHA256,
        },
        "input": {
            "sample_id": manifest_row.sample_id,
            "path": str(Path(image_path).resolve()),
            "sha256": actual_image_hash,
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
            "logits_artifact": "extraction.pt",
        },
        "extractor": {
            "method": "native_timm_forward_hooks_unfused_attention",
            "captured_layers": len(extraction.layers),
            "max_attention_abs_error": extraction.max_attention_abs_error,
        },
        "validation": validation,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "numpy": numpy.__version__,
        },
        "execution": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": str(extraction.logits.dtype),
            "seed": seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "code": git,
        "dependencies": {
            "lock_path": str(Path(lock_path).resolve()),
            "lock_sha256": sha256_file(lock_path),
        },
    }


def validate_provenance(provenance: dict, logits: torch.Tensor) -> None:
    required = {
        "schema_version", "model", "input", "preprocessing", "prediction",
        "extractor", "validation", "software", "execution", "code", "dependencies",
    }
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"Missing provenance fields: {sorted(missing)}")
    if provenance["model"]["state_dict_sha256"] != MODEL_SHA256:
        raise ValueError("Unexpected model hash in provenance")
    if provenance["code"]["dirty"]:
        raise ValueError("Formal provenance requires a clean Git tree")
    expected = topk_indices(logits.detach().cpu(), 5)
    if provenance["prediction"]["top5_indices"] != expected:
        raise ValueError("Saved top-5 does not match raw logits")

