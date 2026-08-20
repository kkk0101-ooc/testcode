from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .constants import DEFAULT_STATE_DICT
from .dataset import ManifestRow
from .extractor import NativeViTExtractor, capture_block_io
from .io_utils import write_json
from .legacy import load_trusted_legacy_full_model
from .modeling import canonical_preprocess_spec, canonical_transform, load_canonical_model
from .provenance import build_provenance, validate_provenance


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def run_stage_a_sample(
    *,
    repo_root: Path,
    image_path: Path,
    manifest_row: ManifestRow,
    output_dir: Path,
    device: str = "cpu",
    state_dict_path: Path = DEFAULT_STATE_DICT,
    lock_path: Path | None = None,
    seed: int = 20260820,
    compare_legacy_pickle: bool = False,
) -> dict:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path or repo_root / "requirements.lock.txt").resolve()
    torch_device = torch.device(device)
    seed_everything(seed)

    reference = load_canonical_model(state_dict_path, device=torch_device, unfused_attention=False)
    extractor_model = load_canonical_model(state_dict_path, device=torch_device, unfused_attention=True)
    transform = canonical_transform(reference)
    with Image.open(image_path) as source:
        image_tensor = transform(source.convert("RGB")).unsqueeze(0).to(torch_device)

    with torch.inference_mode():
        reference_logits = reference(image_tensor).detach().clone()
    unfused_plain_logits, reference_inputs, reference_outputs = capture_block_io(
        extractor_model, image_tensor
    )
    extraction = NativeViTExtractor(extractor_model).extract(image_tensor)
    logits_error = float((reference_logits - extraction.logits).abs().max().item())
    hook_logits_error = float((unfused_plain_logits - extraction.logits).abs().max().item())
    block_input_error = max(
        float((expected - actual.block_input).abs().max().item())
        for expected, actual in zip(reference_inputs, extraction.layers)
    )
    block_output_error = max(
        float((expected - actual.block_output).abs().max().item())
        for expected, actual in zip(reference_outputs, extraction.layers)
    )
    validation = {
        "reference_vs_extractor_logits_max_abs_error": logits_error,
        "unfused_plain_vs_hooked_logits_max_abs_error": hook_logits_error,
        "reference_vs_extractor_block_input_max_abs_error": block_input_error,
        "reference_vs_extractor_block_output_max_abs_error": block_output_error,
        "attention_recomputation_max_abs_error": extraction.max_attention_abs_error,
        "logits_threshold": 1e-5,
        "attention_threshold": 1e-6,
        "logits_pass": logits_error < 1e-5,
        "hook_noninvasive_pass": hook_logits_error == 0.0
        and block_input_error == 0.0
        and block_output_error == 0.0,
        "attention_pass": extraction.max_attention_abs_error < 1e-6,
    }

    if compare_legacy_pickle:
        legacy = load_trusted_legacy_full_model(device=torch_device, allow_unsafe_pickle=True)
        for block in legacy.blocks:
            block.attn.fused_attn = False
        legacy_logits, legacy_inputs, legacy_outputs = capture_block_io(legacy, image_tensor)
        legacy_error = float((legacy_logits - extraction.logits).abs().max().item())
        legacy_input_error = max(
            float((expected.block_input - actual).abs().max().item())
            for expected, actual in zip(extraction.layers, legacy_inputs)
        )
        legacy_output_error = max(
            float((expected.block_output - actual).abs().max().item())
            for expected, actual in zip(extraction.layers, legacy_outputs)
        )
        validation.update({
            "legacy_pickle_vs_extractor_logits_max_abs_error": legacy_error,
            "legacy_pickle_vs_extractor_block_input_max_abs_error": legacy_input_error,
            "legacy_pickle_vs_extractor_block_output_max_abs_error": legacy_output_error,
            "legacy_pickle_pass": legacy_error < 1e-5
            and legacy_input_error < 1e-5
            and legacy_output_error < 1e-5,
        })

    if (
        not validation["logits_pass"]
        or not validation["hook_noninvasive_pass"]
        or not validation["attention_pass"]
    ):
        raise AssertionError(f"Stage A numerical gates failed: {validation}")
    if compare_legacy_pickle and not validation["legacy_pickle_pass"]:
        raise AssertionError(f"Legacy compatibility gate failed: {validation}")

    tensors = extraction.tensor_dict(cpu=True)
    torch.save(tensors, output_dir / "extraction.pt")
    provenance = build_provenance(
        repo_root=repo_root,
        state_dict_path=state_dict_path,
        image_path=image_path,
        manifest_row=manifest_row,
        preprocess=canonical_preprocess_spec(reference),
        extraction=extraction,
        device=torch_device,
        lock_path=lock_path,
        seed=seed,
        validation=validation,
    )
    validate_provenance(provenance, tensors["logits"])
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "validation.json", validation)
    return provenance

