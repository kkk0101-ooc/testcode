"""Future M1 worker boundary for X2-MECH.

Importing this module is safe during P0-R1: model, image, Torch, timm, and
torchvision imports occur only after the explicit future M1 command starts.
P0-R1 never invokes that command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT

M1_AUTHORIZATION = "X2_MECH_M1_TRANSFORMED_MAP_AND_MULTISCALE_FIDELITY_ONLY"
METHODS = ("nns_v2", "vit_grad_cam")
TRANSFORMS = (
    "translate_x_-8", "translate_x_-4", "translate_x_4", "translate_x_8",
    "translate_y_-8", "translate_y_-4", "translate_y_4", "translate_y_8",
    "rotate_-10", "rotate_10", "horizontal_flip", "center_crop_216_resize_224",
)
ARRAY_SUFFIXES = (".npy",)
FORBIDDEN_OUTPUT_SUFFIXES = (".jpg", ".jpeg", ".png")
M1_ROOT = Path("evidence/stage_x/x2_mech_m1_20260910/formal")
ATTEMPT_ID = "attempt-000001"


class X2MechWorkerError(ValueError):
    pass


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"|")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii")); digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (224, 224) or not np.isfinite(array).all():
        raise X2MechWorkerError("map must be finite float32 [224,224]")
    return np.ascontiguousarray(array)


def validate_output_path(path: Path) -> None:
    if path.suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES:
        raise X2MechWorkerError("JPEG/PNG display products are forbidden")
    if path.suffix.lower() not in ARRAY_SUFFIXES:
        raise X2MechWorkerError("scientific maps must use the .npy array contract")


def save_map_new(path: Path, value: np.ndarray) -> dict[str, Any]:
    validate_output_path(path); array = validate_map(value); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha,
            "canonical_digest": array_digest(array), "dtype": "float32", "shape": [224, 224]}


def deterministic_shard(unit_digest: str) -> int:
    if len(unit_digest) != 64 or any(character not in "0123456789abcdef" for character in unit_digest):
        raise X2MechWorkerError("unit digest must be lowercase SHA-256")
    return int.from_bytes(bytes.fromhex(unit_digest), "big") % 4


def validate_route_plan(route: Mapping[str, Any], shard: int) -> None:
    if route.get("source_status") != "VALID":
        raise X2MechWorkerError("failed source route cannot enter M1")
    if int(route.get("shard", -1)) != deterministic_shard(str(route.get("unit_digest", ""))):
        raise X2MechWorkerError("route shard drifted")
    if int(route["shard"]) != shard:
        raise X2MechWorkerError("route does not belong to requested shard")
    maps = route.get("source_maps", [])
    if [row.get("method_id") for row in maps] != list(METHODS):
        raise X2MechWorkerError("source map roster drifted")


def _lazy_runtime() -> dict[str, Any]:
    """Load the future M1 runtime only after explicit authorization."""
    torch = importlib.import_module("torch")
    timm = importlib.import_module("timm")
    torchvision = importlib.import_module("torchvision")
    image_module = importlib.import_module("PIL.Image")
    return {"torch": torch, "timm": timm, "torchvision": torchvision, "Image": image_module}


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result: raise X2MechWorkerError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(X2MechWorkerError(f"non-finite JSON: {token}")))
    if not isinstance(value, dict): raise X2MechWorkerError("JSON root must be object")
    return value


def _csv_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1: raise X2MechWorkerError(f"expected one row: {path}")
    return rows[0]


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows); return stream.getvalue().encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _array_record(path: Path, relative: Path, value: np.ndarray) -> dict[str, Any]:
    array = validate_map(value); _write_bytes_new(path, _npy_bytes(array))
    return {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path),
            "canonical_digest": array_digest(array), "dtype": "float32", "shape": [224, 224]}


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO(); np.save(stream, validate_map(value), allow_pickle=False); return stream.getvalue()


def _mask_record(path: Path, relative: Path, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    if array.shape != (224, 224): raise X2MechWorkerError("mask shape drifted")
    stream = io.BytesIO(); np.save(stream, array, allow_pickle=False); _write_bytes_new(path, stream.getvalue())
    digest = hashlib.sha256(b"bool|224,224|" + array.tobytes(order="C")).hexdigest()
    return {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path),
            "canonical_digest": digest, "dtype": "bool", "shape": [224, 224]}


def _shared_mask_record(repo: Path, transform: str, mask_id: str, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    if array.shape != (224, 224): raise X2MechWorkerError("mask shape drifted")
    relative = M1_ROOT / "valid_masks" / f"{transform}.{mask_id}.npy"; path = repo / relative
    stream = io.BytesIO(); np.save(stream, array, allow_pickle=False); data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor, temporary_name = tempfile.mkstemp(prefix=".mask.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output: output.write(data); output.flush(); os.fsync(output.fileno())
            try: os.link(temporary, path)
            except FileExistsError: pass
        finally:
            temporary.unlink(missing_ok=True)
    if path.read_bytes() != data: raise X2MechWorkerError("shared mask differs across routes or workers")
    digest = hashlib.sha256(b"bool|224,224|" + array.tobytes(order="C")).hexdigest()
    return {"path": relative.as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "canonical_digest": digest, "dtype": "bool", "shape": [224, 224]}


def _source_summary(repo: Path, route: Mapping[str, Any], mech: Any) -> dict[str, str]:
    terminal = mech.strict_read(repo / route["source_terminal_path"]); mech.validate_signed(terminal, route["source_terminal_path"])
    clean_paths = [row["path"] for row in terminal["payload"]["bindings"] if "/clean/terminal.json" in row["path"]]
    if len(clean_paths) != 1: raise X2MechWorkerError("source clean terminal not unique")
    clean = mech.strict_read(repo / clean_paths[0]); mech.validate_signed(clean, clean_paths[0])
    summaries = [row for row in clean["payload"]["files"] if Path(row["path"]).name == "summary.csv"]
    if len(summaries) != 1: raise X2MechWorkerError("source summary not unique")
    record = summaries[0]; path = repo / record["path"]
    if path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]: raise X2MechWorkerError("source summary binding drifted")
    return _csv_one(path)


def _load_source_map(repo: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = repo / record["path"]
    if path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]: raise X2MechWorkerError("source map binding drifted")
    value = np.load(path, allow_pickle=False)
    if array_digest(validate_map(value)) != record["canonical_digest"]: raise X2MechWorkerError("source map canonical digest drifted")
    return value


def _sample_table(repo: Path) -> dict[str, dict[str, str]]:
    path = repo / "evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/manifests/development_samples.csv"
    with path.open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    return {row["source_object_id"]: row for row in rows}


def _exact_support(runtime: Mapping[str, Any], metadata: Mapping[str, Any], metrics: Any) -> np.ndarray:
    torch = runtime["torch"]; TF = runtime["torchvision"].transforms.functional
    interpolation = runtime["torchvision"].transforms.InterpolationMode
    kind = metadata["inverse"]
    if kind == "horizontal_flip": return np.ones((224, 224), dtype=bool)
    if kind == "translate" or kind == "resize_224_to_216_then_pad4":
        _, mask = metrics.inverse_map(torch.ones((224, 224), dtype=torch.float32), metadata); return mask.astype(bool)
    if kind == "rotate":
        ones = torch.ones((1, 224, 224), dtype=torch.float32)
        forward = TF.rotate(ones, float(metadata["angle"]), interpolation=interpolation.BILINEAR, fill=0.0)
        back = TF.rotate(forward, -float(metadata["angle"]), interpolation=interpolation.BILINEAR, fill=0.0)
        return (back.squeeze(0).numpy() >= 1.0 - 1e-6)
    raise X2MechWorkerError("unexpected inverse operator")


def _road_blocks(runtime: Mapping[str, Any], metrics: Any, model: Any, image: Any, score: np.ndarray, target: int, identity: str, block: int, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = runtime["torch"]; array = validate_map(score); rows = array.reshape(224 // block, block, 224 // block, block).sum(axis=(1, 3), dtype=np.float64)
    with torch.no_grad(): clean = float(model(image)[0, target]); baseline = float(model(torch.zeros_like(image))[0, target])
    denominator = clean - baseline
    if not math.isfinite(denominator) or abs(denominator) < 1e-12: raise X2MechWorkerError("ROAD normalization denominator is zero")
    pending = []; coordinates = []; unit_count = rows.size
    for percentile in metrics.ROAD_PERCENTILES:
        count = int(math.floor(unit_count * percentile / 100))
        for road_seed in metrics.ROAD_SEEDS:
            for direction, descending in (("morf", True), ("lerf", False)):
                order = metrics._ranking(rows, f"{identity}|b{block}|{percentile}|{road_seed}|{direction}", 0, descending=descending)
                unit_keep = np.ones(unit_count, dtype=np.float32); unit_keep[order[:count]] = 0
                pixel_keep = np.repeat(np.repeat(unit_keep.reshape(224 // block, 224 // block), block, axis=0), block, axis=1)
                seed = metrics.seed64("STAGE_X_X2_ROAD_NOISY_LINEAR_V1_20260909", identity, block, percentile, road_seed, direction)
                pending.append(metrics.noisy_linear_impute(image[0], pixel_keep, seed)); coordinates.append((percentile, road_seed, direction, seed, float(1 - pixel_keep.mean())))
    values = []
    with torch.no_grad():
        for begin in range(0, len(pending), 45):
            batch = torch.stack(pending[begin:begin + 45]).to(device=image.device, dtype=image.dtype)
            values.extend(float(item) for item in model(batch)[:, target].detach().cpu())
    detail = [{"percentile": p, "seed_index": s, "direction": d, "pcg64_seed": seed,
               "actual_pixel_fraction": fraction, "normalized_response": (value - baseline) / denominator}
              for (p, s, d, seed, fraction), value in zip(coordinates, values)]
    fractions = np.asarray(metrics.ROAD_PERCENTILES, dtype=np.float64) / 100.0
    aucs = {}
    for direction in ("morf", "lerf"):
        per_seed = []
        for road_seed in metrics.ROAD_SEEDS:
            responses = [row["normalized_response"] for row in detail if row["direction"] == direction and row["seed_index"] == road_seed]
            per_seed.append(float(np.trapz(responses, fractions)))
        aucs[direction] = float(np.mean(per_seed, dtype=np.float64))
    morf = aucs["morf"]; lerf = aucs["lerf"]
    return {"road_combined": 0.5 * (lerf - morf), "road_morf": morf, "road_lerf": lerf,
            "block": block, "unit_count": unit_count, "clean_logit": clean, "baseline_logit": baseline}, detail


def _publish_route(repo: Path, route: Mapping[str, Any], files: Mapping[str, bytes], terminal: Mapping[str, Any]) -> None:
    final = repo / M1_ROOT / "units" / route["unit_digest"] / "attempts" / ATTEMPT_ID
    if final.exists(): raise X2MechWorkerError("M1 route output already exists")
    final.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix="m1.", dir=final.parent))
    try:
        for name, data in files.items(): _write_bytes_new(temporary / name, data)
        _write_bytes_new(temporary / "terminal.json", _canonical(terminal) + b"\n")
        os.replace(temporary, final)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def _run_route(repo: Path, route: Mapping[str, Any], model: Any, spec: Mapping[str, Any], sample: Mapping[str, Any], device: str, runtime: Mapping[str, Any], modules: Mapping[str, Any]) -> dict[str, Any]:
    torch = runtime["torch"]; mech = modules["mech"]; metrics = modules["metrics"]; nns = modules["nns"]; comparators = modules["comparators"]; x2_worker = modules["x2_worker"]
    summary = _source_summary(repo, route, mech); target = int(summary["predicted_class"]); selected_steps = int(summary["selected_steps"])
    image, image_identity = x2_worker._load_image(repo, sample, spec, device)
    clean_maps = {row["method_id"]: _load_source_map(repo, row) for row in route["source_maps"]}
    started = time.monotonic(); cpu_started = time.process_time(); torch.cuda.reset_peak_memory_stats(device)
    arrays: dict[str, bytes] = {}; map_rows = []; mask_rows = []; fidelity_rows = []; fidelity_detail = []
    for transform in TRANSFORMS:
        changed, metadata = metrics.transform_image(image, transform, route["unit_digest"], spec["preprocess"]);
        with torch.no_grad(): prediction_changed = int(model(changed).argmax()) != target
        method_values = {
            "nns_v2": nns.input_ig(model, changed, torch.zeros_like(changed), target, selected_steps, 4).sum(0),
            "vit_grad_cam": comparators.vit_grad_cam(model, changed, target)["attribution"],
        }
        x2_mask = None; method_arrays = {}
        for method in METHODS:
            transformed = metrics.pixel_map(method_values[method]).numpy().astype(np.float32, copy=False)
            inverse, current_mask = metrics.inverse_map(method_values[method], metadata); inverse = inverse.astype(np.float32, copy=False)
            if x2_mask is None: x2_mask = current_mask.astype(bool)
            elif not np.array_equal(x2_mask, current_mask): raise X2MechWorkerError("method-specific mask drifted")
            method_arrays[method] = (transformed, inverse)
        exact = _exact_support(runtime, metadata, metrics)
        mask_bindings = {}
        for mask_name, value in (("x2_frozen_common_domain", x2_mask), ("exact_operator_support", exact)):
            record = _shared_mask_record(repo, transform, mask_name, value); mask_bindings[mask_name] = record
            mask_rows.append({"transform": transform, "mask_id": mask_name, "relative_path": record["path"], "sha256": record["sha256"],
                              "canonical_digest": record["canonical_digest"], "valid_pixels": int(value.sum()), "full_domain": bool(value.all()),
                              "transform_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":"))})
        for method in METHODS:
            transformed, inverse = method_arrays[method]
            source_map = next(row for row in route["source_maps"] if row["method_id"] == method)
            route_terminal = (M1_ROOT / "units" / route["unit_digest"] / "attempts" / ATTEMPT_ID / "terminal.json").as_posix()
            for coordinate, value in (("transformed", transformed), ("inverse", inverse)):
                name = f"maps/{transform}/{method}.{coordinate}.npy"; arrays[name] = _npy_bytes(value)
                map_rows.append({"unit_digest": route["unit_digest"], "model_id": route["model_id"], "sample_id": route["sample_id"],
                                 "method_id": method, "transform": transform, "coordinate": coordinate, "relative_path": name,
                                 "sha256": hashlib.sha256(arrays[name]).hexdigest(), "canonical_array_digest": array_digest(value),
                                 "dtype": "float32", "shape": "224x224", "selected_steps": selected_steps, "target": target,
                                 "prediction_changed": prediction_changed,
                                 "source_clean_map_path": source_map["path"], "source_clean_map_sha256": source_map["sha256"],
                                 "source_clean_map_digest": source_map["canonical_digest"], "source_clean_map_dtype": source_map["dtype"],
                                 "source_clean_map_shape": "224x224", "source_x2_terminal_path": route["source_terminal_path"],
                                 "source_x2_terminal_sha256": route["source_terminal_sha256"], "m1_route_terminal_path": route_terminal,
                                 "resource_telemetry_scope": "m1_route_terminal",
                                 "transform_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                                 "x2_mask_path": mask_bindings["x2_frozen_common_domain"]["path"],
                                 "x2_mask_digest": mask_bindings["x2_frozen_common_domain"]["canonical_digest"],
                                 "exact_mask_path": mask_bindings["exact_operator_support"]["path"],
                                 "exact_mask_digest": mask_bindings["exact_operator_support"]["canonical_digest"]})
    for method in METHODS:
        for block in (4, 16):
            summary_row, detail = _road_blocks(runtime, metrics, model, image, clean_maps[method], target, route["unit_digest"] + "|" + method, block, device)
            fidelity_rows.append({"unit_digest": route["unit_digest"], "model_id": route["model_id"], "sample_id": route["sample_id"], "method_id": method, **summary_row})
            fidelity_detail.extend({"unit_digest": route["unit_digest"], "method_id": method, "block": block, **row} for row in detail)
    fields_map = ("unit_digest","model_id","sample_id","method_id","transform","coordinate","relative_path","sha256","canonical_array_digest","dtype","shape","selected_steps","target","prediction_changed","source_clean_map_path","source_clean_map_sha256","source_clean_map_digest","source_clean_map_dtype","source_clean_map_shape","source_x2_terminal_path","source_x2_terminal_sha256","m1_route_terminal_path","resource_telemetry_scope","transform_metadata","x2_mask_path","x2_mask_digest","exact_mask_path","exact_mask_digest")
    fields_mask = ("transform","mask_id","relative_path","sha256","canonical_digest","valid_pixels","full_domain","transform_metadata")
    fields_fidelity = ("unit_digest","model_id","sample_id","method_id","road_combined","road_morf","road_lerf","block","unit_count","clean_logit","baseline_logit")
    fields_detail = ("unit_digest","method_id","block","percentile","seed_index","direction","pcg64_seed","actual_pixel_fraction","normalized_response")
    arrays["map_manifest.csv"] = _csv_bytes(fields_map, map_rows); arrays["valid_mask_registry.csv"] = _csv_bytes(fields_mask, mask_rows)
    arrays["multiscale_fidelity.csv"] = _csv_bytes(fields_fidelity, fidelity_rows); arrays["multiscale_fidelity_rows.csv"] = _csv_bytes(fields_detail, fidelity_detail)
    resources = {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_started,
                 "gpu_seconds": time.monotonic() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                 "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "downloaded_bytes": 0}
    terminal = {"schema_version": 1, "document_type": "stage_x_x2_mech_m1_route_terminal", "status": "PASS",
                "authorization": M1_AUTHORIZATION, "unit_digest": route["unit_digest"], "model_id": route["model_id"],
                "sample_id": route["sample_id"], "image_identity": image_identity, "target": target, "selected_steps": selected_steps,
                "counts": {"transforms": 12, "methods": 2, "transformed_maps": 24, "inverse_maps": 24, "masks": 24, "multiscale_fidelity_endpoints": 4},
                "resources": resources, "source_terminal_path": route["source_terminal_path"], "source_terminal_sha256": route["source_terminal_sha256"],
                "immutable": True}
    terminal["canonical_digest"] = hashlib.sha256(_canonical(terminal)).hexdigest(); _publish_route(repo, route, arrays, terminal)
    return {"status": "PASS", "unit_digest": route["unit_digest"], "resources": resources}


def run_m1_shard(repo_root: Path | str, shard: int, device: str, authorization: str) -> dict[str, Any]:
    if authorization != M1_AUTHORIZATION:
        raise X2MechWorkerError("explicit M1 authorization is required")
    if shard not in range(4):
        raise X2MechWorkerError("shard must be in 0..3")
    repo = Path(repo_root).resolve(); runtime = _lazy_runtime()
    mech = importlib.import_module("vitnns.stage_x_x2_mech"); x2_worker = importlib.import_module("vitnns.stage_x_x2_r2_r1_r1_worker")
    metrics = importlib.import_module("vitnns.stage_x_x2_metrics"); nns = importlib.import_module("vitnns.stage_x_nns_v2"); comparators = importlib.import_module("vitnns.stage_x_comparators")
    completion = mech.validate_completion(repo)
    if completion["next_authorization"] != M1_AUTHORIZATION: raise X2MechWorkerError("P0-R1 did not authorize M1")
    identities = {row["name"]: row["value"] for row in mech.strict_read(repo / mech.COMPLETION)["payload"]["identity"]}
    if mech._git(repo, "rev-parse", mech.PRODUCER_REF + "^{commit}") != identities["producer_commit"]:
        raise X2MechWorkerError("P0-R1 producer ref drifted")
    completion_commit = mech._git(repo, "rev-parse", mech.COMPLETION_REF + "^{commit}")
    mech._git(repo, "merge-base", "--is-ancestor", identities["producer_commit"], completion_commit)
    plan = mech._validate_artifact(repo, mech.ROOT / "typed_plan.json"); routes = [row for row in plan["payload"]["routes"] if row["source_status"] == "VALID" and row["shard"] == shard]
    samples = _sample_table(repo); modules = {"mech": mech, "x2_worker": x2_worker, "metrics": metrics, "nns": nns, "comparators": comparators}
    results = []; model = None; loaded = None
    try:
        for route in routes:
            validate_route_plan(route, shard); spec = x2_worker._model_spec(repo, route["model_id"])
            if loaded != route["model_id"]:
                if model is not None: del model
                model = x2_worker._load_model(repo, spec, device); loaded = route["model_id"]
            results.append(_run_route(repo, route, model, spec, samples[route["sample_id"]], device, runtime, modules))
    finally:
        if model is not None: del model
    return {"status": "PASS", "shard": shard, "route_count": len(results), "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("run-m1-shard",))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT); parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--device", required=True); parser.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_m1_shard(args.repo_root.resolve(), args.shard, args.device, args.authorization)
    except (X2MechWorkerError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True)); return 2
    print(json.dumps(result, sort_keys=True, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
