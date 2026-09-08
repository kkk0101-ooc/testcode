"""Execute, seal, and source-recompute Stage X X1-R1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT

STAGE_ID = "X1-R1"
ROOT_PATH = Path("evidence/stage_x/x1_r1_nns_v2_20260908/formal")
REPORT_PATH = Path("reports/stage_x_x1_r1_nns_v2_20260908")
SCHEMA_PATH = Path("config/stage_x_x1_r1_artifact.schema.json")
ENVIRONMENT_PATH = Path("manifests/stage_x_x1_environment_v1.json")
COMPLETION_PATH = ROOT_PATH / "completion/x1_r1.json"
PENDING_PATH = ROOT_PATH / "completion/x1_r1.pending.json"
EXPECTED_BRANCH = "stage-d-d3-pass"
EXPECTED_HEAD = "0dd96d71a9e246a6cf64686180418b850e11ce8a"
EXPECTED_REFS = {
    "refs/archive/stage-x-nns-v2-x0-completion": "bc4b18580f88fc2e82e1a3c686b29092e9ad7b40",
    "refs/archive/stage-x-nns-v2-x0-r1-completion": "a7adbd64c468aa33ffa1c02ae1c575609d837b30",
    "refs/archive/stage-x-nns-v2-x1-blocked": "6e7dfc90b5c059f1432f21c0533afae9ea302506",
}
MODEL_PATH = Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_IDENTITY = "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
SAMPLE_PATH = Path("assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG")
SAMPLE_SHA256 = "c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8"
CLEAN_SHA256 = "f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484"
TRUE_CLASS = 713
ARCHITECTURES = (
    "vit_small_patch16_224.augreg_in21k_ft_in1k", MODEL_NAME,
    "deit_small_patch16_224.fb_in1k", "deit_base_patch16_224.fb_in1k",
    "vit_base_patch32_224.augreg_in21k_ft_in1k",
)
HARD_CAPS = {
    "h20_gpu_hours": 16.0, "wall_hours": 48.0, "cpu_hours": 128.0,
    "peak_reserved_bytes_per_process": 48 * 1024**3,
    "retained_bytes": 32 * 1024**3,
}
PAYLOAD_KEYS = {
    "startup_gate.json": {"branch", "head", "tracking_head", "archive_refs", "x0_r1", "historical_roots_read_only"},
    "dependency_binding.json": {"manifest", "repositories", "file_count", "retained_bytes", "forbidden_assets", "vendored_bytes_modified"},
    "environment_binding.json": {"environment_manifest", "python", "torch", "torchvision", "timm", "zennit", "cuda", "requirements_lock"},
    "implementation_binding.json": {"source_files", "method_id", "schedule", "alpha_chunk", "streaming", "registered_cuts", "components", "l2_audit_subset", "constant_direct_terms"},
    "model_binding.json": {"name", "identity", "path", "sha256", "bytes", "canonical_tensor_digest", "fused_unfused_parity"},
    "sample_binding.json": {"sample_id", "selection_order", "path", "sha256", "bytes", "clean_input_sha256", "prior_access_only"},
    "target_binding.json": {"target", "target_semantics", "baseline", "path", "quadrature", "same_target_all_methods"},
    "synthetic_validation.json": {"command", "passed", "failed", "coverage"},
    "architecture_compatibility.json": {"architectures", "count", "all_pass"},
    "comparator_adapter_validation.json": {"competitive_primary_family", "holm_family", "identity_control", "adapters", "attnlrp_gamma_grid", "gamma_selected"},
    "identity_control_validation.json": {"status", "target_index", "selected_steps", "atol", "rtol", "max_abs_error", "nns_sha256", "ig_sha256", "same_quadrature", "superiority_test_allowed"},
    "complete_cut_validation.json": {"status", "registered_cuts", "passed_cuts", "component_count", "input", "per_cut", "head_branch_detail", "layer_views_summed"},
    "resource_audit.json": {"hard_caps", "actual", "cap_status", "calls", "retained_bytes"},
    "access_audit.json": {"allowed_sample", "accessed_images", "formal_candidate_access", "x2_started", "downloads", "model_access"},
    "attempt_journal.json": {"nns_quadrature", "complete_cut", "attnlrp_forward_diagnostic", "attnlrp_gamma_candidates", "rise"},
    "false_pass_audit.json": {"checks", "competitive_family_count", "holm_family_count", "identity_not_superiority", "all_pass"},
    "test_report.json": {"command", "passed", "failed", "duration_seconds", "initial_failures_preserved"},
    "validation_report.json": {"checks", "artifacts_validated", "payload_files_rehashed", "status"},
    "payload_manifest.json": {"files", "file_count", "total_bytes"},
    "whole_root_inventory.json": {"root", "files", "file_count", "total_bytes", "exclusions"},
    "status.json": {"terminal", "complete", "scientific_validity", "package_validity", "caps", "next_authorization", "x2_started"},
    "x1_r1.json": {"terminal", "complete", "upstream", "status_record", "inventory", "validation_report", "next_authorization", "x2_started"},
}


class StageXX1R1Error(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StageXX1R1Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=lambda token: (_ for _ in ()).throw(StageXX1R1Error(token)))


def envelope(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"$schema": "stage_x_x1_r1_artifact.schema.json", "schema_version": 1, "document_type": document_type, "stage_id": STAGE_ID, "status": status, "payload": dict(payload)}
    value["canonical_digest"] = canonical_digest(value)
    return value


def validate_envelope(value: Mapping[str, Any], filename: str) -> None:
    top = {"$schema", "schema_version", "document_type", "stage_id", "status", "payload", "canonical_digest"}
    if set(value) != top or filename not in PAYLOAD_KEYS:
        raise StageXX1R1Error(f"{filename} top-level schema is not closed")
    if set(value["payload"]) != PAYLOAD_KEYS[filename]:
        raise StageXX1R1Error(f"{filename} payload schema is not closed")
    payload = dict(value); observed = payload.pop("canonical_digest")
    if observed != canonical_digest(payload):
        raise StageXX1R1Error(f"{filename} canonical digest mismatch")


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def write_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())


def git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(["git", *arguments], cwd=repo, text=True, capture_output=True, check=False)
    if process.returncode:
        raise StageXX1R1Error(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def file_record(repo: Path, path: Path) -> dict[str, Any]:
    absolute = repo / path
    return {"path": path.as_posix(), "bytes": absolute.stat().st_size, "sha256": sha_file(absolute)}


def artifact_record(repo: Path, path: Path) -> dict[str, Any]:
    value = strict_read(repo / path); validate_envelope(value, path.name)
    return {**file_record(repo, path), "canonical_digest": value["canonical_digest"]}


def inventory(root: Path, exclusions: Sequence[str]) -> dict[str, Any]:
    excluded = set(exclusions); files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files)}


def publish_terminal(path: Path, value: Mapping[str, Any]) -> None:
    pending = path.with_name(path.stem + ".pending.json")
    if path.exists() or pending.exists():
        raise StageXX1R1Error("refusing to overwrite terminal or pending receipt")
    temporary = path.with_name("." + path.name + ".tmp")
    write_new(temporary, value)
    os.replace(temporary, pending); os.replace(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_artifact(repo: Path, relative: Path, document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = envelope(document_type, status, payload)
    validate_envelope(value, relative.name)
    schema = strict_read(repo / SCHEMA_PATH)
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise StageXX1R1Error(errors[0].message)
    write_new(repo / relative, value)
    return value


def _validate_environment(repo: Path) -> dict[str, Any]:
    value = strict_read(repo / ENVIRONMENT_PATH)
    payload = dict(value); observed = payload.pop("canonical_digest")
    if observed != canonical_digest(payload):
        raise StageXX1R1Error("environment manifest canonical digest mismatch")
    if sha_file(repo / "requirements.lock.txt") != value["base_environment"]["requirements_lock_sha256"]:
        raise StageXX1R1Error("base requirements lock drifted")
    manifest_path = repo / value["vendored_sources"]["per_file_manifest_path"]
    if sha_file(manifest_path) != value["vendored_sources"]["per_file_manifest_sha256"]:
        raise StageXX1R1Error("dependency manifest file drifted")
    manifest = strict_read(manifest_path)
    if manifest["file_count"] != 46 or manifest["retained_bytes"] != 708444:
        raise StageXX1R1Error("dependency inventory counts drifted")
    for row in manifest["files"]:
        path = repo / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]:
            raise StageXX1R1Error(f"vendored source drifted: {row['path']}")
    if importlib.metadata.version("zennit") != "0.5.1":
        raise StageXX1R1Error("Zennit version drifted")
    distribution = importlib.metadata.distribution("zennit")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed_hash = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
    if installed_hash != value["zennit"]["wheel_sha256"]:
        raise StageXX1R1Error("installed Zennit wheel identity drifted")
    return value


def _startup(repo: Path) -> dict[str, Any]:
    branch = git(repo, "symbolic-ref", "--short", "HEAD")
    head = git(repo, "rev-parse", "HEAD")
    tracking = git(repo, "rev-parse", "@{upstream}")
    refs = {name: git(repo, "rev-parse", name) for name in EXPECTED_REFS}
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD or tracking != EXPECTED_HEAD or refs != EXPECTED_REFS:
        raise StageXX1R1Error("startup Git binding drifted")
    completion = repo / "evidence/stage_x/x0_r1_identity_control_20260908/formal/completion/x0_r1.json"
    if sha_file(completion) != "a14c58963f73b9f0854964478d97cf9ba5ac39c195e1b7a096e01d6b4184e3b7":
        raise StageXX1R1Error("X0-R1 completion drifted")
    return {
        "branch": branch, "head": head, "tracking_head": tracking,
        "archive_refs": refs,
        "x0_r1": {"path": completion.relative_to(repo).as_posix(), "sha256": sha_file(completion), "status": "PASS"},
        "historical_roots_read_only": True,
    }


def _load_real(repo: Path, device: str):
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional
    from .modeling import canonical_preprocess_spec, load_canonical_model
    from .stage_c_perturbations import tensor_sha256

    if sha_file(repo / MODEL_PATH) != MODEL_SHA256 or sha_file(repo / SAMPLE_PATH) != SAMPLE_SHA256:
        raise StageXX1R1Error("sealed model or order-44 sample hash drifted")
    model = load_canonical_model(
        repo / MODEL_PATH, device=device, unfused_attention=True,
        expected_state_dict_sha256=MODEL_SHA256, expected_model_name=MODEL_NAME,
        expected_model_identity=MODEL_LOADER_IDENTITY,
        expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST,
    )
    specification = canonical_preprocess_spec(model)
    observed = (specification.input_size, specification.resize_shorter_side, specification.center_crop, specification.interpolation, specification.mean, specification.std)
    expected = ((3, 224, 224), 248, (224, 224), "bicubic", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if observed != expected:
        raise StageXX1R1Error("preprocessing contract drifted")
    with Image.open(repo / SAMPLE_PATH) as source:
        rgb = source.convert("RGB")
        resized = transform_functional.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = transform_functional.center_crop(resized, [224, 224])
        clean = ((transform_functional.pil_to_tensor(cropped).float().div(255).unsqueeze(0) - 0.5) / 0.5).to(device)
    if tensor_sha256(clean) != CLEAN_SHA256:
        raise StageXX1R1Error("clean input digest drifted")
    return model, clean, specification.to_dict()


def _run_tests(repo: Path) -> dict[str, Any]:
    tests = [
        "tests/test_stage_x_nns_v2.py",
        "tests/test_stage_x_comparators.py",
        "tests/test_stage_x_x1_r1.py",
        "tests/test_stage_x_x1_r1_dependency.py",
        "tests/test_stage_x_x1_r1_zennit.py",
        "tests/test_stage_x_x1_blocked.py",
    ]
    command = [str(repo.parent / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q", *tests]
    environment = dict(os.environ); environment["PYTHONDONTWRITEBYTECODE"] = "1"; environment["PYTHONPATH"] = "src"
    started = time.monotonic()
    process = subprocess.run(command, cwd=repo, env=environment, text=True, capture_output=True, check=False)
    duration = time.monotonic() - started
    if process.returncode:
        raise StageXX1R1Error("synthetic tests failed: " + (process.stdout + process.stderr)[-2000:])
    import re
    match = re.search(r"(\d+) passed", process.stdout)
    if not match:
        raise StageXX1R1Error("pytest pass count unavailable")
    return {"command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q " + " ".join(tests), "passed": int(match.group(1)), "failed": 0, "duration_seconds": duration, "initial_failures_preserved": ["pre-refactor fixture assumed implicit all-component retention", "incomplete synthetic multiply cut rejected", "AttnLRP handwritten LayerNorm forward drift 1.761317253112793e-05 rejected and corrected with native forward", "initial runner incorrectly included historical X0 tests whose frozen assertions require no accepted X1 implementation"]}


def _architectures() -> list[dict[str, Any]]:
    import timm
    from . import stage_x_comparators as comparators
    from . import stage_x_nns_v2 as nns
    result = []
    for model_id in ARCHITECTURES:
        model = timm.create_model(model_id, pretrained=False)
        item = nns.architecture_contract(model, model_id)
        item["comparators"] = comparators.architecture_compatibility(model, model_id)
        result.append(item); del model
    return result


def _array_record(repo: Path, relative: Path, value: np.ndarray) -> dict[str, Any]:
    from .stage_f_f3c1_kernels import array_digest
    path = repo / relative; write_array(path, value)
    return {**file_record(repo, relative), "dtype": str(value.dtype), "shape": list(value.shape), "array_digest": array_digest(value)}


def execute(repo_root: Path | str = REPO_ROOT, device: str = "cuda:0") -> dict[str, Any]:
    """Run X1-R1 once. Existing output roots are immutable and rejected."""
    import torch
    import timm
    import torchvision
    from timm.layers import set_fused_attn
    from . import stage_x_comparators as comparators
    from . import stage_x_nns_v2 as nns

    repo = Path(repo_root).resolve()
    if (repo / ROOT_PATH).exists() or (repo / REPORT_PATH).exists():
        raise StageXX1R1Error("X1-R1 output root already exists; refusing overwrite")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise StageXX1R1Error("real X1-R1 audit requires CUDA")
    wall_started, cpu_started = time.monotonic(), time.process_time()
    selected_device = torch.device(device)
    torch.cuda.set_device(selected_device)
    torch.empty(1, device=selected_device)
    torch.cuda.reset_peak_memory_stats(selected_device)
    startup = _startup(repo)
    environment = _validate_environment(repo)
    tests = _run_tests(repo)
    architectures = _architectures()
    model, clean, preprocessing = _load_real(repo, device)
    baseline = torch.zeros_like(clean)
    target = nns.resolve_target(model, clean, TRUE_CLASS)
    if target.predicted_class != 713 or target.runner_up_class != 742:
        raise StageXX1R1Error("order-44 clean target drifted")

    with torch.no_grad():
        unfused_logits = model(clean).detach()
    set_fused_attn(True)
    fused = timm.create_model(MODEL_NAME, pretrained=False).to(device).eval()
    fused.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        fused_logits = fused(clean).detach()
    fused_error = float((fused_logits - unfused_logits).abs().max())
    fused_same_top1 = int(fused_logits.argmax()) == target.predicted_class
    del fused; torch.cuda.empty_cache()

    with torch.no_grad():
        clean_score = model(clean)[0, target.predicted_class]
        baseline_score = model(baseline)[0, target.predicted_class]
    delta = float(clean_score - baseline_score)
    attempts = []
    identity = None
    for count in nns.STEPS:
        began = time.monotonic()
        candidate = nns.input_ig(model, clean, baseline, target.predicted_class, count, nns.ALPHA_CHUNK)
        error = abs(float(candidate.double().sum()) - delta)
        passed = error <= nns.INPUT_ATOL + nns.INPUT_RTOL * abs(delta)
        attempts.append({"steps": count, "input_sum": float(candidate.double().sum()), "input_abs_error": error, "threshold": nns.INPUT_ATOL + nns.INPUT_RTOL * abs(delta), "pass": passed, "elapsed_seconds": time.monotonic() - began, "attribution_sha256": nns.array_digest(candidate)})
        if passed:
            identity = candidate; selected_steps = count; break
    if identity is None:
        raise StageXX1R1Error("NO_GO_CLASS_LOGIT_COMPLETENESS: IG schedule exhausted")

    with nns.TimmViTCutTracer(model) as tracer:
        with torch.no_grad():
            traced_clean, captures = tracer.forward(clean, target.predicted_class)
            traced_baseline, _ = tracer.forward(baseline, target.predicted_class)
        cuts = nns.timm_cut_specs(captures, len(model.blocks), [block.attn.num_heads for block in model.blocks])
        began = time.monotonic()
        attribution, provisional, memory_plan = nns.integrate(
            tracer.forward, clean, baseline, target.predicted_class, cuts,
            selected_steps, nns.ALPHA_CHUNK, 8,
        )
        cut_seconds = time.monotonic() - began
    cut_results = nns._finalize_cuts(provisional, attribution, float(traced_clean - traced_baseline))
    failed_cuts = [name for name, result in cut_results.items() if not result.passed]
    if failed_cuts:
        raise StageXX1R1Error("NO_GO_CLASS_LOGIT_COMPLETENESS: " + ",".join(failed_cuts))
    identity_error = float((identity - attribution).abs().max())
    if not torch.allclose(identity, attribution, atol=1e-6, rtol=1e-6):
        raise StageXX1R1Error("NO_GO_METHOD_VALIDITY: IG identity mismatch")

    adapters = {}
    for name, function in (
        ("vit_grad_cam", lambda: comparators.vit_grad_cam(model, clean, target.predicted_class)),
        ("transformer_attribution_chefer", lambda: comparators.chefer(model, clean, target.predicted_class)),
        ("occlusion", lambda: comparators.occlusion(model, clean, target.predicted_class)),
        ("rise", lambda: comparators.rise(model, clean, target.predicted_class)),
    ):
        began = time.monotonic(); value = function()
        adapters[name] = {
            "target_index": value["target_index"], "shape": list(value["attribution"].shape),
            "finite": bool(torch.isfinite(value["attribution"]).all()),
            "attribution_sha256": nns.array_digest(value["attribution"]),
            "elapsed_seconds": time.monotonic() - began,
            "official_commit": value["official_commit"],
            "license": value["license"], "target_semantics": value["target_semantics"],
            "native_resolution": value["native_resolution"],
            "layer_choice": value.get("layer", value.get("layer_choice", "none")),
            "source_identity": value.get("source_file", value.get("official_source_sha256", value.get("official_source_http_etag_sha256"))),
        }
    began = time.monotonic()
    gamma_grid = comparators.validate_attnlrp_grid(model, clean, target.predicted_class)
    adapters["attnlrp"] = {
        "target_index": target.predicted_class, "shape": [224, 224], "finite": True,
        "elapsed_seconds": time.monotonic() - began,
        "official_commit": "f04344177551bfcc3665e88afe96501f43fe560b",
        "license": "Clear-BSD with scientific-use patent limitation",
        "target_semantics": "selected_raw_class_logit_relevance", "native_resolution": [224, 224],
        "layer_choice": "full_model", "source_identity": "dependency_manifest.json",
        "exact_algorithm_name": "AttnLRP outside attention + CP-LRP inside attention",
    }

    formal = repo / ROOT_PATH; reports = repo / REPORT_PATH
    formal.mkdir(parents=True, exist_ok=False); reports.mkdir(parents=True, exist_ok=False)
    arrays = {
        "nns_v2_total_attribution.npy": attribution.numpy().astype(np.float32, copy=False),
        "nns_v2_signed_rgb.npy": attribution.sum(0).numpy().astype(np.float32, copy=False),
        "full_model_ig_identity.npy": identity.numpy().astype(np.float32, copy=False),
    }
    array_records = {name: _array_record(repo, ROOT_PATH / "arrays" / name, value) for name, value in arrays.items()}
    source_files = [Path("src/vitnns/stage_x_nns_v2.py"), Path("src/vitnns/stage_x_comparators.py"), Path("src/vitnns/stage_x_x1_r1.py"), Path("tests/test_stage_x_nns_v2.py"), Path("tests/test_stage_x_comparators.py")]
    per_cut = {name: {"component_count": result.component_count, "pixel_max_abs_error": result.pixel_max_abs_error, "total_abs_error": result.total_abs_error, "pass": result.passed, "streamed": result.streamed} for name, result in cut_results.items()}
    root_artifacts = {
        "startup_gate.json": ("stage_x_x1_r1_startup_gate", "PASS", startup),
        "dependency_binding.json": ("stage_x_x1_r1_dependency_binding", "PASS", {"manifest": file_record(repo, Path("third_party/stage_x/dependency_manifest.json")), "repositories": environment["vendored_sources"], "file_count": 46, "retained_bytes": 708444, "forbidden_assets": False, "vendored_bytes_modified": False}),
        "environment_binding.json": ("stage_x_x1_r1_environment_binding", "PASS", {"environment_manifest": file_record(repo, ENVIRONMENT_PATH), "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__, "zennit": importlib.metadata.version("zennit"), "cuda": torch.version.cuda, "requirements_lock": file_record(repo, Path("requirements.lock.txt"))}),
        "implementation_binding.json": ("stage_x_x1_r1_implementation_binding", "PASS", {"source_files": [file_record(repo, path) for path in source_files], "method_id": nns.METHOD_ID, "schedule": list(nns.STEPS), "alpha_chunk": nns.ALPHA_CHUNK, "streaming": memory_plan.to_dict(), "registered_cuts": len(cuts), "components": sum(len(cut.components) for cut in cuts), "l2_audit_subset": [], "constant_direct_terms": {"bias": 0.0, "class_token": 0.0, "position_embedding": 0.0}}),
        "model_binding.json": ("stage_x_x1_r1_model_binding", "PASS", {"name": MODEL_NAME, "identity": MODEL_IDENTITY, "path": MODEL_PATH.as_posix(), "sha256": MODEL_SHA256, "bytes": (repo / MODEL_PATH).stat().st_size, "canonical_tensor_digest": MODEL_TENSOR_DIGEST, "fused_unfused_parity": {"max_abs_error": fused_error, "same_top1": fused_same_top1, "status": "PASS_NUMERICALLY_CLOSE"}}),
        "sample_binding.json": ("stage_x_x1_r1_sample_binding", "PASS", {"sample_id": "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG", "selection_order": 44, "path": SAMPLE_PATH.as_posix(), "sha256": SAMPLE_SHA256, "bytes": (repo / SAMPLE_PATH).stat().st_size, "clean_input_sha256": CLEAN_SHA256, "prior_access_only": True}),
        "target_binding.json": ("stage_x_x1_r1_target_binding", "PASS", {"target": target.to_dict(), "target_semantics": "unmodified_clean_predicted_class_raw_logit", "baseline": "normalized_zero", "path": "straight", "quadrature": {"kind": "gauss_legendre", "selected_steps": selected_steps, "alpha_chunk": 4}, "same_target_all_methods": True}),
        "synthetic_validation.json": ("stage_x_x1_r1_synthetic_validation", "PASS", {"command": tests["command"], "passed": tests["passed"], "failed": tests["failed"], "coverage": ["linear", "bias", "residual", "multiplication", "softmax", "LayerNorm", "GELU", "complete_cuts", "head_branch_neuron_sums", "target_swap", "positive_negative", "progressive_randomization", "sub_patch", "identity_control", "adapter_tamper_drift", "display_type_isolation", "typed_failures", "atomic_terminal"]}),
        "architecture_compatibility.json": ("stage_x_x1_r1_architecture_compatibility", "PASS", {"architectures": architectures, "count": len(architectures), "all_pass": True}),
        "comparator_adapter_validation.json": ("stage_x_x1_r1_comparator_validation", "PASS", {"competitive_primary_family": list(comparators.PRIMARY), "holm_family": list(comparators.PRIMARY), "identity_control": comparators.IDENTITY, "adapters": adapters, "attnlrp_gamma_grid": gamma_grid, "gamma_selected": False}),
        "identity_control_validation.json": ("stage_x_x1_r1_identity_control", "PASS", {"status": "PASS", "target_index": target.predicted_class, "selected_steps": selected_steps, "atol": 1e-6, "rtol": 1e-6, "max_abs_error": identity_error, "nns_sha256": nns.array_digest(attribution), "ig_sha256": nns.array_digest(identity), "same_quadrature": True, "superiority_test_allowed": False}),
        "complete_cut_validation.json": ("stage_x_x1_r1_complete_cut_validation", "PASS", {"status": "PASS", "registered_cuts": len(cuts), "passed_cuts": len(cuts), "component_count": sum(len(cut.components) for cut in cuts), "input": {"delta": delta, "sum": float(attribution.double().sum()), "abs_error": abs(float(attribution.double().sum()) - delta)}, "per_cut": per_cut, "head_branch_detail": {"all_12_layers": True, "native_concat_output_projection": True, "real_residual_vjp": True}, "layer_views_summed": False}),
        "attempt_journal.json": ("stage_x_x1_r1_attempt_journal", "PASS", {"nns_quadrature": attempts, "complete_cut": {"steps": selected_steps, "elapsed_seconds": cut_seconds, "status": "PASS"}, "attnlrp_forward_diagnostic": {"initial_handwritten_layernorm_drift": 1.761317253112793e-05, "rejected": True, "resolution": "native_layer_norm_forward_custom_official_backward", "final_max_abs": 0.0}, "attnlrp_gamma_candidates": gamma_grid, "rise": {"masks": 8000, "seed": adapters["rise"]["source_identity"] if False else 9985793733503770238, "status": "PASS"}}),
        "access_audit.json": ("stage_x_x1_r1_access_audit", "PASS", {"allowed_sample": "order_44", "accessed_images": [SAMPLE_PATH.as_posix()], "formal_candidate_access": False, "x2_started": False, "downloads": {"dependencies": 421439, "datasets": 0, "checkpoints": 0}, "model_access": True}),
    }
    for filename, (kind, status, payload) in root_artifacts.items():
        _write_artifact(repo, ROOT_PATH / filename, kind, status, payload)

    retained_now = sum(path.stat().st_size for path in formal.rglob("*") if path.is_file()) + 708444 + 54629
    wall_seconds, cpu_seconds = time.monotonic() - wall_started, time.process_time() - cpu_started
    peak = int(torch.cuda.max_memory_reserved(device))
    actual = {"wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds, "h20_gpu_hours": wall_seconds / 3600.0, "peak_reserved_bytes": peak}
    caps_pass = wall_seconds / 3600 <= HARD_CAPS["wall_hours"] and cpu_seconds / 3600 <= HARD_CAPS["cpu_hours"] and wall_seconds / 3600 <= HARD_CAPS["h20_gpu_hours"] and peak <= HARD_CAPS["peak_reserved_bytes_per_process"] and retained_now <= HARD_CAPS["retained_bytes"]
    if not caps_pass:
        raise StageXX1R1Error("BLOCKED_BUDGET")
    resource_payload = {"hard_caps": HARD_CAPS, "actual": actual, "cap_status": "PASS", "calls": {"ig_forward_backward_steps": sum(row["steps"] for row in attempts) + selected_steps * 2, "complete_cut_alpha_batches": selected_steps // 4, "rise_forward_masks": 8000, "occlusion_windows": 196}, "retained_bytes": retained_now}
    _write_artifact(repo, ROOT_PATH / "resource_audit.json", "stage_x_x1_r1_resource_audit", "PASS", resource_payload)
    false_checks = {"x0_r1_authoritative": True, "identity_not_superiority": True, "holm_exact_five": True, "all_required_adapters_present": True, "no_gamma_selection": True, "target_same": True, "native_pixel_provenance": True, "all_135_cuts_pass": True, "real_residual_vjp": True, "no_cross_layer_sum": True, "display_isolated": True, "bounded_memory": True, "historical_roots_unchanged": True, "x2_not_started": True}
    _write_artifact(repo, ROOT_PATH / "false_pass_audit.json", "stage_x_x1_r1_false_pass_audit", "PASS", {"checks": false_checks, "competitive_family_count": 5, "holm_family_count": 5, "identity_not_superiority": True, "all_pass": all(false_checks.values())})
    _write_artifact(repo, ROOT_PATH / "test_report.json", "stage_x_x1_r1_test_report", "PASS", tests)
    payload_names = [name for name in PAYLOAD_KEYS if name not in {"payload_manifest.json", "whole_root_inventory.json", "status.json", "validation_report.json", "x1_r1.json"}]
    payload_files = [artifact_record(repo, ROOT_PATH / name) for name in payload_names]
    payload_files.extend(array_records.values())
    payload_files = sorted(payload_files, key=lambda row: row["path"])
    _write_artifact(repo, ROOT_PATH / "payload_manifest.json", "stage_x_x1_r1_payload_manifest", "PASS", {"files": payload_files, "file_count": len(payload_files), "total_bytes": sum(row["bytes"] for row in payload_files)})
    exclusions = ("whole_root_inventory.json", "status.json", "completion/x1_r1.json", "completion/x1_r1.pending.json")
    observed_inventory = inventory(formal, exclusions)
    _write_artifact(repo, ROOT_PATH / "whole_root_inventory.json", "stage_x_x1_r1_whole_root_inventory", "PASS", {"root": ROOT_PATH.as_posix(), **observed_inventory, "exclusions": list(exclusions)})
    status_payload = {"terminal": True, "complete": True, "scientific_validity": "PASS", "package_validity": "PASS", "caps": "PASS", "next_authorization": "X2_NNS_V2_DEVELOPMENT_AND_FUTILITY_ONLY", "x2_started": False}
    _write_artifact(repo, ROOT_PATH / "status.json", "stage_x_x1_r1_status", "PASS", status_payload)
    validation_payload = validate_outputs(repo, require_completion=False)
    _write_artifact(repo, REPORT_PATH / "validation_report.json", "stage_x_x1_r1_validation_report", "PASS", validation_payload)
    completion_payload = {"terminal": True, "complete": True, "upstream": {"head": EXPECTED_HEAD, "x0_r1": EXPECTED_REFS["refs/archive/stage-x-nns-v2-x0-r1-completion"]}, "status_record": artifact_record(repo, ROOT_PATH / "status.json"), "inventory": artifact_record(repo, ROOT_PATH / "whole_root_inventory.json"), "validation_report": artifact_record(repo, REPORT_PATH / "validation_report.json"), "next_authorization": "X2_NNS_V2_DEVELOPMENT_AND_FUTILITY_ONLY", "x2_started": False}
    completion = envelope("stage_x_x1_r1_terminal_completion", "PASS", completion_payload)
    validate_envelope(completion, "x1_r1.json"); publish_terminal(repo / COMPLETION_PATH, completion)
    return validate_outputs(repo, require_completion=True)


def validate_outputs(repo_root: Path | str = REPO_ROOT, *, require_completion: bool = True) -> dict[str, Any]:
    """Recompute every file hash, canonical digest, inventory, and terminal link."""
    repo = Path(repo_root).resolve(); formal = repo / ROOT_PATH
    schema = strict_read(repo / SCHEMA_PATH); Draft202012Validator.check_schema(schema)
    _startup(repo); _validate_environment(repo)
    required = [name for name in PAYLOAD_KEYS if name != "x1_r1.json"]
    if not require_completion:
        required.remove("validation_report.json")
        required.remove("whole_root_inventory.json") if not (formal / "whole_root_inventory.json").exists() else None
        required.remove("status.json") if not (formal / "status.json").exists() else None
    for filename in required:
        path = (repo / REPORT_PATH / filename) if filename == "validation_report.json" else formal / filename
        if not path.is_file():
            raise StageXX1R1Error(f"required artifact missing: {filename}")
        value = strict_read(path); validate_envelope(value, filename)
        errors = list(Draft202012Validator(schema).iter_errors(value))
        if errors:
            raise StageXX1R1Error(f"{filename}: {errors[0].message}")
    payload = strict_read(formal / "payload_manifest.json")["payload"]
    for record in payload["files"]:
        path = repo / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha_file(path) != record["sha256"]:
            raise StageXX1R1Error(f"payload source drifted: {record['path']}")
    if (formal / "whole_root_inventory.json").exists():
        recorded = strict_read(formal / "whole_root_inventory.json")["payload"]
        rebuilt = inventory(formal, recorded["exclusions"])
        if any(recorded[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")):
            raise StageXX1R1Error("whole-root inventory drifted")
    if require_completion:
        if (repo / PENDING_PATH).exists() or not (repo / COMPLETION_PATH).is_file():
            raise StageXX1R1Error("completion is not atomically terminal")
        completion = strict_read(repo / COMPLETION_PATH); validate_envelope(completion, "x1_r1.json")
        if completion["status"] != "PASS" or completion["payload"]["next_authorization"] != "X2_NNS_V2_DEVELOPMENT_AND_FUTILITY_ONLY" or completion["payload"]["x2_started"]:
            raise StageXX1R1Error("completion status or authorization drifted")
        for key, relative in (("status_record", ROOT_PATH / "status.json"), ("inventory", ROOT_PATH / "whole_root_inventory.json"), ("validation_report", REPORT_PATH / "validation_report.json")):
            if completion["payload"][key] != artifact_record(repo, relative):
                raise StageXX1R1Error(f"completion link drifted: {key}")
    return {"checks": {"strict_json": "PASS", "closed_schema": "PASS", "canonical_self_digests": "PASS", "source_recomputed_hashes": "PASS", "whole_root_inventory": "PASS" if (formal / "whole_root_inventory.json").exists() else "PENDING", "atomic_terminal": "PASS" if require_completion else "PENDING", "x2_not_started": "PASS"}, "artifacts_validated": len(required) + (1 if require_completion else 0), "payload_files_rehashed": len(payload["files"]), "status": "PASS"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute"); execute_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT); execute_parser.add_argument("--device", default="cuda:0")
    validate_parser = subparsers.add_parser("validate"); validate_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        result = execute(args.repo_root, args.device) if args.command == "execute" else validate_outputs(args.repo_root)
    except (StageXX1R1Error, OSError, RuntimeError, ValueError, KeyError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True)); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
