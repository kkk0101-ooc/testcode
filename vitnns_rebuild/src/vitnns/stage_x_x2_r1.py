"""X2-R1 closed control plane and pre-outcome recovery gates."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

RUN_ID = "STAGE_X_X2_R1_V1_20260909"
ROOT = Path("evidence/stage_x/x2_r1_nns_v2_development_20260909/formal")
REPORT_ROOT = Path("reports/stage_x_x2_r1_nns_v2_development_20260909")
CONFIG = Path("config/stage_x_x2_r1.json")
CONFIG_SCHEMA = Path("config/stage_x_x2_r1.schema.json")
ARTIFACT_SCHEMA = Path("config/stage_x_x2_r1_artifact.schema.json")
OLD_CONFIG = Path("config/stage_x_x2.json")
OLD_ROOT = Path("evidence/stage_x/x2_nns_v2_development_20260909/formal")
OLD_SAMPLE_MANIFEST = OLD_ROOT / "development_sample_manifest.json"
OLD_UNIT_MANIFEST = OLD_ROOT / "development_unit_manifest.json"
OLD_FAILURE_RECEIPT = OLD_ROOT / "post_failure/x2_no_go_method_validity.json"
AMENDMENT = Path("docs/stage_x_x2_r1_recovery_amendment_20260909.md")
MUTATION_RECEIPT = ROOT / "pre_outcome/mutation_suite.json"
RUN_SPEC = ROOT / "x2_r1_run_spec.json"
SAMPLE_CSV = ROOT / "manifests/development_samples.csv"
UNIT_CSV = ROOT / "manifests/development_units.csv"
EXPECTED_BRANCH = "stage-d-d3-pass"
BASE_COMMIT = "69fc9177491c63a21088a2092274859b00917879"
HISTORICAL_REFS = {
    "refs/archive/stage-x-nns-v2-x0-completion": "bc4b18580f88fc2e82e1a3c686b29092e9ad7b40",
    "refs/archive/stage-x-nns-v2-x0-r1-completion": "a7adbd64c468aa33ffa1c02ae1c575609d837b30",
    "refs/archive/stage-x-nns-v2-x1-blocked": "6e7dfc90b5c059f1432f21c0533afae9ea302506",
    "refs/archive/stage-x-nns-v2-x1-r1-completion": BASE_COMMIT,
}
SAMPLE_FIELDS = ("development_rank", "dataset_release_id", "class_id", "source_object_id", "file_bytes", "file_sha256", "selection_sha256")
UNIT_FIELDS = ("unit_digest", "run_id", "model_id", "sample_id", "sample_sha256", "development_rank", "shard")
MODEL_LOCAL_PATHS = {
    "vit_s16": Path("assets/models/stage_x_x2/vit_small_patch16_224.augreg_in21k_ft_in1k.npz"),
    "vit_b16": Path("assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"),
    "deit_s16": Path("assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth"),
    "deit_b16": Path("assets/models/stage_x_x2/deit_base_patch16_224.fb_in1k.pth"),
    "vit_b32": Path("assets/models/stage_x_x2/vit_base_patch32_224.augreg_in21k_ft_in1k.npz"),
}
IMPLEMENTATION_FILES = (
    "src/vitnns/stage_x_x2_r1.py", "src/vitnns/stage_x_x2_r1_worker.py",
    "src/vitnns/stage_x_x2_r1_reconcile.py",
    "src/vitnns/stage_x_x2_metrics.py", "src/vitnns/stage_x_nns_v2.py",
    "src/vitnns/stage_x_comparators.py", "src/vitnns/stage_x_x2_failure.py",
    "config/stage_x_x2_r1.json", "config/stage_x_x2_r1.schema.json",
    "config/stage_x_x2_r1_artifact.schema.json",
    "config/stage_x_x2_failure_receipt.schema.json",
    "docs/stage_x_x2_r1_recovery_amendment_20260909.md",
    "tests/test_stage_x_x2_r1.py",
)
ZERO_RESOURCES = {
    "wall_seconds": 0.0, "gpu_seconds": 0.0, "cpu_seconds": 0.0,
    "peak_reserved_bytes": 0, "peak_allocated_bytes": 0,
    "forward_calls": 0, "backward_calls": 0,
    "downloaded_bytes": 0, "retained_bytes": 0,
}


class X2R1Error(ValueError):
    pass


def _duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise X2R1Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    return json.loads(
        text, object_pairs_hook=_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            X2R1Error(f"non-finite JSON: {token}")
        ),
    )


def strict_read(path: Path) -> dict[str, Any]:
    value = strict_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise X2R1Error(f"JSON root must be object: {path}")
    return value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if process.returncode:
        raise X2R1Error(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _remote_ref(repo: Path, ref: str) -> str:
    rows = git(repo, "ls-remote", "origin", ref).splitlines()
    if len(rows) != 1 or rows[0].split()[-1] != ref:
        raise X2R1Error(f"remote ref missing or ambiguous: {ref}")
    return rows[0].split()[0]


def _self_validate(value: Mapping[str, Any], context: str) -> None:
    plain = dict(value)
    observed = plain.pop("canonical_digest", None)
    if observed != canonical_sha256(plain):
        raise X2R1Error(f"canonical digest mismatch: {context}")


def load_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    value, schema = strict_read(repo / CONFIG), strict_read(repo / CONFIG_SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: tuple(str(part) for part in item.absolute_path))
    if errors:
        raise X2R1Error(f"X2-R1 config: {errors[0].message}")
    _self_validate(value, CONFIG.as_posix())
    source = strict_read(repo / OLD_CONFIG)
    _self_validate(source, OLD_CONFIG.as_posix())
    record = value["source_contract"]
    if file_record(repo, OLD_CONFIG, canonical_digest=source["canonical_digest"]) != record:
        raise X2R1Error("frozen X2 source contract binding drifted")
    return value


def scalar_rows(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for key, value in sorted(values.items()):
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise X2R1Error(f"non-scalar payload value: {key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise X2R1Error(f"non-finite scalar payload value: {key}")
        result.append({"name": key, "value": value})
    return result


def terminal_state(
    *, terminal: bool = False, immutable: bool = False, complete: bool = False,
    next_authorization: str = "NONE",
) -> dict[str, Any]:
    return {
        "terminal": terminal, "immutable": immutable, "complete": complete,
        "next_authorization": next_authorization, "formal_superiority_claim": False,
        "x3_started": False, "x4_started": False, "x5_started": False,
    }


def payload(
    *, identity: Mapping[str, Any] | None = None, bindings: Iterable[Mapping[str, Any]] = (),
    checks: Iterable[Mapping[str, Any]] = (), metrics: Iterable[Mapping[str, Any]] = (),
    files: Iterable[Mapping[str, Any]] = (), counts: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None, resources: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "identity": scalar_rows(identity or {}), "bindings": list(bindings),
        "checks": list(checks), "metrics": list(metrics), "files": list(files),
        "counts": scalar_rows(counts or {}), "state": scalar_rows(state or {}),
        "resources": dict(resources or ZERO_RESOURCES),
        "terminal": dict(terminal or terminal_state()),
    }


def signed(document_type: str, status: str, body: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "$schema": ARTIFACT_SCHEMA.name, "schema_version": 1,
        "document_type": document_type, "stage_id": "X2-R1",
        "run_id": RUN_ID, "status": status, "payload": dict(body),
    }
    value["canonical_digest"] = canonical_sha256(value)
    return value


def _unique(rows: Sequence[Mapping[str, Any]], key: str, context: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise X2R1Error(f"duplicate {context}")


def validate_artifact(value: Mapping[str, Any], repo_root: Path | str = REPO_ROOT, *, verify_files: bool = False) -> None:
    repo = Path(repo_root).resolve()
    schema = strict_read(repo / ARTIFACT_SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: tuple(str(part) for part in item.absolute_path))
    if errors:
        raise X2R1Error(f"artifact schema: {errors[0].message}")
    _self_validate(value, str(value.get("document_type")))
    body = value["payload"]
    for name, key in (("identity", "name"), ("counts", "name"), ("state", "name"), ("checks", "check_id"), ("metrics", "metric_id"), ("bindings", "path"), ("files", "path")):
        _unique(body[name], key, name)
    if value["document_type"] == "stage_x_x2_r1_unit_manifest":
        counts = {row["name"]: row["value"] for row in body["counts"]}
        if counts.get("unit_count") != 200 or counts.get("unique_unit_count") != 200 or counts.get("sample_count") != 40 or counts.get("model_count") != 5:
            raise X2R1Error("unit manifest semantic count mismatch")
    if value["document_type"] in {"stage_x_x2_r1_run_spec", "stage_x_x2_r1_gamma_selection", "stage_x_x2_r1_completion"}:
        states = {row["name"]: row["value"] for row in body["state"]}
        if states.get("holm_family_count") != 5 or states.get("full_model_ig_in_holm") is not False:
            raise X2R1Error("Holm family must contain exactly five competitive methods")
    if verify_files:
        for record in [*body["bindings"], *body["files"]]:
            validate_file_record(repo, record)


def write_new(path: Path, value: Mapping[str, Any], repo_root: Path | str = REPO_ROOT) -> None:
    validate_artifact(value, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def write_csv_new(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise X2R1Error(f"CSV field mismatch: {path}")
        writer.writerow(row)
    write_bytes_new(path, output.getvalue().encode("utf-8"))


def read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise X2R1Error(f"CSV header mismatch: {path}")
        return [dict(row) for row in reader]


def file_record(repo: Path, relative: Path, *, canonical_digest: str | None = None) -> dict[str, Any]:
    path = repo / relative
    return {
        "kind": "file", "path": relative.as_posix(), "bytes": path.stat().st_size,
        "sha256": sha_file(path), "canonical_digest": canonical_digest,
    }


def artifact_record(repo: Path, relative: Path) -> dict[str, Any]:
    value = strict_read(repo / relative)
    validate_artifact(value, repo)
    return file_record(repo, relative, canonical_digest=value["canonical_digest"])


def array_record(repo: Path, relative: Path) -> dict[str, Any]:
    from .stage_f_f3c1_kernels import array_digest
    path = repo / relative
    value = np.load(path, allow_pickle=False)
    if value.dtype not in (np.float32, np.float64, np.int64, np.uint8) or not np.isfinite(value).all():
        raise X2R1Error(f"invalid scientific array: {relative}")
    return {
        "kind": "array", "path": relative.as_posix(), "bytes": path.stat().st_size,
        "sha256": sha_file(path), "canonical_digest": array_digest(value),
        "dtype": str(value.dtype), "shape": list(value.shape),
    }


def validate_file_record(repo: Path, record: Mapping[str, Any]) -> None:
    path = repo / str(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha_file(path) != record["sha256"]:
        raise X2R1Error(f"file record drift: {path}")
    if record["kind"] == "array":
        if array_record(repo, Path(str(record["path"]))) != record:
            raise X2R1Error(f"array record drift: {path}")
    elif record["canonical_digest"] is not None:
        if path.suffix == ".json":
            value = strict_read(path)
            _self_validate(value, path.as_posix())
            observed = value["canonical_digest"]
        elif path.name == "development_samples.csv":
            rows = read_csv(path, SAMPLE_FIELDS)
            for row in rows:
                row["development_rank"] = int(row["development_rank"]); row["file_bytes"] = int(row["file_bytes"])
            observed = canonical_sha256(rows)
        elif path.name == "development_units.csv":
            rows = read_csv(path, UNIT_FIELDS)
            for row in rows:
                row["development_rank"] = int(row["development_rank"]); row["shard"] = int(row["shard"])
            observed = canonical_sha256(rows)
        elif path.name == "model_materialization.csv":
            fields=("id","model_id","path","bytes","sha256","canonical_tensor_digest","key_count","parameter_count","serialized_key_count","strict_load","canonical_path")
            rows=read_csv(path,fields)
            for row in rows:
                for key in ("bytes","key_count","parameter_count","serialized_key_count"): row[key]=int(row[key])
            observed=canonical_sha256(rows)
        elif path.suffix == ".csv":
            with path.open("r",encoding="utf-8",newline="") as stream:
                observed=canonical_sha256([dict(row) for row in csv.DictReader(stream)])
        else:
            raise X2R1Error(f"unsupported canonical file record: {path}")
        if observed != record["canonical_digest"]:
            raise X2R1Error(f"document digest drift: {path}")


def publish_atomic(pending: Path, terminal: Path, value: Mapping[str, Any], repo: Path) -> None:
    if pending.exists() or terminal.exists():
        raise X2R1Error("pending or terminal publication already exists")
    pending_body=copy.deepcopy(value["payload"]);pending_body["terminal"]=terminal_state()
    write_new(pending, signed(value["document_type"], "PENDING", pending_body), repo)
    final = signed(value["document_type"], value["status"], value["payload"] )
    temporary = terminal.with_name(terminal.name + ".terminal.pending")
    if temporary.exists() or terminal.exists():
        raise X2R1Error("terminal overwrite forbidden")
    write_new(temporary, final, repo)
    os.replace(temporary, terminal)
    pending.unlink()


def _expect_rejected(name: str, action: Any, results: list[dict[str, Any]]) -> None:
    try:
        action()
    except Exception as error:
        results.append({"check_id": name, "status": "PASS", "detail": type(error).__name__})
    else:
        raise X2R1Error(f"false-pass mutation accepted: {name}")


def validate_unit_row(row: Mapping[str, Any], samples: Mapping[str, Mapping[str, Any]], models: set[str]) -> None:
    if set(row) != set(UNIT_FIELDS) or row["run_id"] != RUN_ID:
        raise X2R1Error("unit row shape/run binding drift")
    if row["model_id"] not in models or row["sample_id"] not in samples:
        raise X2R1Error("wrong sample/model binding")
    sample = samples[str(row["sample_id"])]
    if row["sample_sha256"] != sample["file_sha256"] or int(row["development_rank"]) != int(sample["development_rank"]):
        raise X2R1Error("wrong sample identity binding")
    identity = {
        "run_id": RUN_ID, "model_id": row["model_id"], "sample_id": row["sample_id"],
        "sample_sha256": row["sample_sha256"], "development_rank": int(row["development_rank"]),
    }
    digest = canonical_sha256(identity)
    if row["unit_digest"] != digest or int(row["shard"]) != int.from_bytes(bytes.fromhex(digest)[:8], "big") % 4:
        raise X2R1Error("wrong unit digest/shard")


def run_mutation_suite(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(repo)
    schema = strict_read(repo / ARTIFACT_SCHEMA)
    Draft202012Validator.check_schema(schema)
    if MUTATION_RECEIPT.exists() or (repo / MUTATION_RECEIPT).exists():
        existing = strict_read(repo / MUTATION_RECEIPT)
        validate_artifact(existing, repo, verify_files=True)
        return existing
    if (repo / ROOT).exists():
        raise X2R1Error("fresh X2-R1 root required for pre-outcome mutation suite")
    results: list[dict[str, Any]] = []
    base = signed("stage_x_x2_r1_phase_manifest", "PASS", payload(
        identity={"phase": "mutation_fixture"}, counts={"unit_count": 200},
        state={"holm_family_count": 5, "full_model_ig_in_holm": False},
    ))
    def rejected_artifact(mutator: Any) -> None:
        changed = copy.deepcopy(base); mutator(changed)
        if "canonical_digest" in changed:
            plain = dict(changed); plain.pop("canonical_digest"); changed["canonical_digest"] = canonical_sha256(plain)
        validate_artifact(changed, repo)
    _expect_rejected("root_unknown_field", lambda: rejected_artifact(lambda value: value.__setitem__("UNKNOWN", True)), results)
    _expect_rejected("payload_unknown_field", lambda: rejected_artifact(lambda value: value["payload"].__setitem__("UNKNOWN", True)), results)
    _expect_rejected("nested_unknown_field", lambda: rejected_artifact(lambda value: value["payload"]["terminal"].__setitem__("UNKNOWN", True)), results)
    _expect_rejected("missing_required_field", lambda: rejected_artifact(lambda value: value["payload"].pop("resources")), results)
    _expect_rejected("duplicate_json_key", lambda: strict_loads('{"a":1,"a":2}'), results)
    bad_digest = copy.deepcopy(base); bad_digest["canonical_digest"] = "0" * 64
    _expect_rejected("bad_canonical_digest", lambda: validate_artifact(bad_digest, repo), results)
    _expect_rejected("nan", lambda: strict_loads('{"x":NaN}'), results)
    _expect_rejected("infinity", lambda: strict_loads('{"x":Infinity}'), results)
    _expect_rejected("bool_as_int", lambda: rejected_artifact(lambda value: value["payload"]["metrics"].append({"metric_id":"bad","value":True,"unit":"count"})), results)
    manifest = signed("stage_x_x2_r1_unit_manifest", "PASS", payload(counts={"unit_count":200,"unique_unit_count":200,"sample_count":40,"model_count":5}))
    _expect_rejected("wrong_unit_count", lambda: validate_artifact(signed("stage_x_x2_r1_unit_manifest","PASS",payload(counts={"unit_count":199,"unique_unit_count":200,"sample_count":40,"model_count":5})), repo), results)
    sample = {"development_rank":0,"file_sha256":"a"*64}
    good_identity = {"run_id":RUN_ID,"model_id":"m","sample_id":"s","sample_sha256":"a"*64,"development_rank":0}
    digest = canonical_sha256(good_identity)
    good_row = {**good_identity,"unit_digest":digest,"shard":int.from_bytes(bytes.fromhex(digest)[:8],"big")%4}
    _expect_rejected("wrong_unit_digest", lambda: validate_unit_row({**good_row,"unit_digest":"b"*64},{"s":sample},{"m"}), results)
    _expect_rejected("wrong_sample_model_binding", lambda: validate_unit_row({**good_row,"model_id":"bad"},{"s":sample},{"m"}), results)
    spec = signed("stage_x_x2_r1_run_spec","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":False}))
    _expect_rejected("full_model_ig_in_holm", lambda: validate_artifact(signed("stage_x_x2_r1_run_spec","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":True})),repo), results)
    _expect_rejected("holm_family_not_five", lambda: validate_artifact(signed("stage_x_x2_r1_run_spec","PASS",payload(state={"holm_family_count":6,"full_model_ig_in_holm":False})),repo), results)
    with tempfile.TemporaryDirectory(prefix="x2-r1-mutations-") as name:
        temporary = Path(name); one = temporary / "one.json"
        write_new(one, base, repo)
        _expect_rejected("terminal_overwrite", lambda: write_new(one, base, repo), results)
        pending, terminal = temporary / "pending.json", temporary / "terminal.json"
        publish_atomic(pending, terminal, base, repo)
        _expect_rejected("pending_overwrite", lambda: publish_atomic(pending, terminal, base, repo), results)
        array_path = temporary / "array.npy"
        with array_path.open("wb") as stream:
            np.save(stream, np.zeros((2, 3), dtype=np.float32), allow_pickle=False)
        relative = array_path.relative_to(repo) if array_path.is_relative_to(repo) else None
        from .stage_f_f3c1_kernels import array_digest
        raw = np.load(array_path, allow_pickle=False)
        record = {"kind":"array","path":str(array_path),"bytes":array_path.stat().st_size,"sha256":sha_file(array_path),"canonical_digest":array_digest(raw),"dtype":"float32","shape":[2,3]}
        def validate_temp(rec: Mapping[str, Any]) -> None:
            value = np.load(array_path, allow_pickle=False)
            actual = {"kind":"array","path":str(array_path),"bytes":array_path.stat().st_size,"sha256":sha_file(array_path),"canonical_digest":array_digest(value),"dtype":str(value.dtype),"shape":list(value.shape)}
            if dict(rec) != actual: raise X2R1Error("array source drift")
        _expect_rejected("array_dtype_drift", lambda: validate_temp({**record,"dtype":"float64"}), results)
        _expect_rejected("array_shape_drift", lambda: validate_temp({**record,"shape":[3,2]}), results)
        _expect_rejected("array_file_sha_drift", lambda: validate_temp({**record,"sha256":"0"*64}), results)
        _expect_rejected("array_content_digest_drift", lambda: validate_temp({**record,"canonical_digest":"0"*64}), results)
    if [row["check_id"] for row in results] != config["pre_outcome_mutations"]:
        raise X2R1Error("mutation roster/order drifted")
    body = payload(
        identity={"suite":"X2_R1_PRE_OUTCOME_MUTATION_SUITE"},
        bindings=[file_record(repo, ARTIFACT_SCHEMA), file_record(repo, CONFIG, canonical_digest=config["canonical_digest"]), file_record(repo, AMENDMENT)],
        checks=results, counts={"attack_count":len(results),"rejected_count":len(results)},
        state={"outcome_access_before_suite":False,"all_mutations_rejected":True},
        terminal=terminal_state(terminal=True, immutable=True, complete=True),
    )
    receipt = signed("stage_x_x2_r1_pre_outcome_mutation_receipt", "PASS", body)
    write_new(repo / MUTATION_RECEIPT, receipt, repo)
    return receipt


def _source_samples(repo: Path) -> list[dict[str, Any]]:
    old = strict_read(repo / OLD_SAMPLE_MANIFEST)
    _self_validate(old, OLD_SAMPLE_MANIFEST.as_posix())
    rows = old["payload"]["samples"]
    if len(rows) != 40 or len({row["source_object_id"] for row in rows}) != 40:
        raise X2R1Error("old X2 sample manifest is not the exact frozen 40")
    result = []
    for row in rows:
        selected = {field: row[field] for field in SAMPLE_FIELDS}
        image = repo / "assets/data" / row["source_object_id"]
        if image.stat().st_size != row["file_bytes"] or sha_file(image) != row["file_sha256"]:
            raise X2R1Error(f"frozen development JPEG drifted: {row['source_object_id']}")
        result.append(selected)
    return result


def _unit_rows(models: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model in models:
        for sample in samples:
            identity = {
                "run_id": RUN_ID, "model_id": model["model_id"],
                "sample_id": sample["source_object_id"], "sample_sha256": sample["file_sha256"],
                "development_rank": int(sample["development_rank"]),
            }
            digest = canonical_sha256(identity)
            result.append({"unit_digest":digest,**identity,"shard":int.from_bytes(bytes.fromhex(digest)[:8],"big")%4})
    return result


def _startup(repo: Path) -> dict[str, Any]:
    branch = git(repo, "symbolic-ref", "--short", "HEAD")
    head, tracking = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "@{upstream}")
    remote = _remote_ref(repo, f"refs/heads/{EXPECTED_BRANCH}")
    if (branch, head, tracking, remote) != (EXPECTED_BRANCH, BASE_COMMIT, BASE_COMMIT, BASE_COMMIT):
        raise X2R1Error("X2-R1 startup Git identity drifted")
    for ref, expected in HISTORICAL_REFS.items():
        if git(repo, "rev-parse", ref) != expected or _remote_ref(repo, ref) != expected:
            raise X2R1Error(f"historical archive ref drifted: {ref}")
    from . import stage_x_x1_r1 as x1
    original = x1.EXPECTED_HEAD
    try:
        x1.EXPECTED_HEAD = BASE_COMMIT
        validation = x1.validate_outputs(repo)
    finally:
        x1.EXPECTED_HEAD = original
    if validation["status"] != "PASS":
        raise X2R1Error("X1-R1 source validation failed")
    lfs = subprocess.run(["git","lfs","fsck"],cwd=repo,text=True,capture_output=True,check=False)
    if lfs.returncode:
        raise X2R1Error("Git LFS fsck failed")
    return {"branch":branch,"local_head":head,"tracking_head":tracking,"remote_head":remote,"x1_artifacts_validated":validation["artifacts_validated"],"x1_payload_files_rehashed":validation["payload_files_rehashed"],"lfs_fsck":"PASS"}


def initialize(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = load_config(repo)
    mutation = strict_read(repo / MUTATION_RECEIPT); validate_artifact(mutation, repo, verify_files=True)
    from .stage_x_x2_failure import recompute as recompute_failure, validate as validate_failure
    failure = strict_read(repo / OLD_FAILURE_RECEIPT); validate_failure(failure, repo)
    expected_failure = recompute_failure(repo); expected_failure["canonical_digest"] = canonical_sha256(expected_failure)
    if failure != expected_failure:
        raise X2R1Error("failed X2 custody no longer source-recomputes")
    startup = _startup(repo)
    samples = _source_samples(repo)
    source_contract = strict_read(repo / OLD_CONFIG)
    units = _unit_rows(source_contract["models"], samples)
    if len(units) != len({row["unit_digest"] for row in units}) or len(units) != 200:
        raise X2R1Error("X2-R1 unit roster is not 200 unique routes")
    if {row["sample_id"] for row in units} != {row["source_object_id"] for row in samples}:
        raise X2R1Error("X2-R1 unit/sample roster drifted")
    if (repo / SAMPLE_CSV).exists() or (repo / UNIT_CSV).exists() or (repo / RUN_SPEC).exists():
        raise X2R1Error("X2-R1 initialization output exists; overwrite forbidden")
    write_csv_new(repo / SAMPLE_CSV, SAMPLE_FIELDS, samples)
    write_csv_new(repo / UNIT_CSV, UNIT_FIELDS, units)
    sample_digest = canonical_sha256(samples); unit_digest = canonical_sha256(units)
    common_state = {"holm_family_count":5,"full_model_ig_in_holm":False,"formal_candidate_accessed":False}
    artifacts = {
        "startup_gate.json": signed("stage_x_x2_r1_startup_gate","PASS",payload(identity=startup,state=common_state,checks=[{"check_id":"pre_outcome_mutation_suite","status":"PASS","detail":mutation["canonical_digest"]}])),
        "failed_attempt_binding.json": signed("stage_x_x2_r1_failed_attempt_binding","PASS",payload(bindings=[file_record(repo,OLD_FAILURE_RECEIPT,canonical_digest=failure["canonical_digest"])],counts=failure["counts"],state={**common_state,"old_outcomes_allowed_in_analysis":False})),
        "amendment_binding.json": signed("stage_x_x2_r1_amendment_binding","PASS",payload(bindings=[file_record(repo,AMENDMENT)],state={**common_state,"prospective_before_x2_r1_outcome":True})),
        "development_sample_manifest.json": signed("stage_x_x2_r1_sample_manifest","PASS",payload(bindings=[file_record(repo,OLD_SAMPLE_MANIFEST,canonical_digest=strict_read(repo/OLD_SAMPLE_MANIFEST)["canonical_digest"])],files=[file_record(repo,SAMPLE_CSV,canonical_digest=sample_digest)],counts={"sample_count":40,"unique_sample_count":40,"model_count":5},state={**common_state,"shared_ordered_list_across_models":True,"reselection_performed":False})),
        "development_unit_manifest.json": signed("stage_x_x2_r1_unit_manifest","PASS",payload(files=[file_record(repo,UNIT_CSV,canonical_digest=unit_digest)],counts={"unit_count":200,"unique_unit_count":200,"sample_count":40,"model_count":5},state={**common_state,"atomic_terminal_required":True})),
    }
    for name, value in artifacts.items():
        write_new(repo / ROOT / name, value, repo)
    run_bindings = [
        file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),
        file_record(repo,OLD_CONFIG,canonical_digest=source_contract["canonical_digest"]),
        file_record(repo,AMENDMENT), artifact_record(repo,MUTATION_RECEIPT),
        file_record(repo,OLD_FAILURE_RECEIPT,canonical_digest=failure["canonical_digest"]),
        artifact_record(repo,ROOT/"development_sample_manifest.json"),
        artifact_record(repo,ROOT/"development_unit_manifest.json"),
    ]
    run_spec = signed("stage_x_x2_r1_run_spec","PASS",payload(
        identity={"authorization":config["authorization"],"canonical_execution_path":source_contract["execution"]["canonical_path"],"run_id":RUN_ID},
        bindings=run_bindings, counts={"sample_count":40,"model_count":5,"unit_count":200,"complete_cut_count":135,"gamma_candidate_count":15,"competitive_comparator_count":5,"secondary_reference_count":5,"stability_transformation_count":23},
        state={**common_state,"sample_digest":sample_digest,"unit_digest":unit_digest,"pre_outcome_mutation_receipt_sha256":sha_file(repo/MUTATION_RECEIPT),"old_outcomes_allowed_in_analysis":False,"same_root_retry_allowed":False},
    ))
    write_new(repo / RUN_SPEC, run_spec, repo)
    return {"status":"PASS","samples":40,"units":200,"sample_digest":sample_digest,"unit_digest":unit_digest,"mutation_receipt_sha256":sha_file(repo/MUTATION_RECEIPT)}


def load_samples(repo: Path) -> list[dict[str, Any]]:
    manifest=strict_read(repo/ROOT/"development_sample_manifest.json");validate_artifact(manifest,repo,verify_files=True)
    rows = read_csv(repo / SAMPLE_CSV, SAMPLE_FIELDS)
    for row in rows:
        row["development_rank"] = int(row["development_rank"]); row["file_bytes"] = int(row["file_bytes"])
    return rows


def load_units(repo: Path) -> list[dict[str, Any]]:
    manifest=strict_read(repo/ROOT/"development_unit_manifest.json");validate_artifact(manifest,repo,verify_files=True)
    samples = {row["source_object_id"]: row for row in load_samples(repo)}
    models = {row["model_id"] for row in strict_read(repo / OLD_CONFIG)["models"]}
    rows = read_csv(repo / UNIT_CSV, UNIT_FIELDS)
    for row in rows:
        row["development_rank"] = int(row["development_rank"]); row["shard"] = int(row["shard"]); validate_unit_row(row,samples,models)
    if len(rows)!=200 or len({row["unit_digest"] for row in rows})!=200 or set(Counter(row["model_id"] for row in rows).values())!={40}:
        raise X2R1Error("unit CSV source reconciliation failed")
    return rows


def _load_state(repo: Path, spec: Mapping[str, Any], *, fused: bool = False) -> tuple[Any, Mapping[str, Any]]:
    import torch
    import timm
    from timm.layers import set_fused_attn
    path = repo / MODEL_LOCAL_PATHS[str(spec["id"])]
    set_fused_attn(fused)
    model = timm.create_model(str(spec["model_id"]), pretrained=False).eval()
    if spec["source_kind"] == "official_timm_0.9.16_npz":
        model.load_pretrained(str(path))
        state = model.state_dict()
        source_keys = len(np.load(path, allow_pickle=False).files)
    else:
        with path.open("rb") as stream:
            loaded = torch.load(stream, map_location="cpu", weights_only=True)
        state = loaded["model"] if spec["source_kind"] == "official_timm_0.9.16_pth" else loaded
        outcome = model.load_state_dict(state, strict=True)
        if tuple(outcome.missing_keys) or tuple(outcome.unexpected_keys):
            raise X2R1Error("strict checkpoint load drifted")
        source_keys = len(state)
    strict = model.load_state_dict(model.state_dict(), strict=True)
    if tuple(strict.missing_keys) or tuple(strict.unexpected_keys):
        raise X2R1Error("strict materialized model state reload drifted")
    for block in model.blocks:
        block.attn.fused_attn = fused
    return model, {"state": state, "source_key_count": source_keys}


def _continuity_image(repo: Path, spec: Mapping[str, Any]) -> Any:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform
    relative = Path("assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG")
    with Image.open(repo / relative) as image:
        rgb=image.convert("RGB");resized=transform.resize(rgb,spec["preprocess"]["resize_shorter_side"],interpolation=InterpolationMode.BICUBIC,antialias=True);cropped=transform.center_crop(resized,[224,224]);canonical=transform.pil_to_tensor(cropped).float().div(255).unsqueeze(0)
    mean=torch.tensor(spec["preprocess"]["mean"]).reshape(1,3,1,1);std=torch.tensor(spec["preprocess"]["std"]).reshape(1,3,1,1)
    return (canonical-mean)/std


def materialize_models(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    import torch
    from .recovery_v2 import canonical_state_dict_tensor_digest
    repo=Path(repo_root).resolve(); config=load_config(repo); source=strict_read(repo/OLD_CONFIG)
    output_csv=ROOT/"model_materialization.csv"; output_json=ROOT/"model_materialization.json"
    if (repo/output_csv).exists() or (repo/output_json).exists():
        raise X2R1Error("model materialization evidence exists; overwrite forbidden")
    old_materialization=strict_read(repo/OLD_ROOT/"model_materialization.json")
    old_models={row["id"]:row for row in old_materialization["payload"]["models"]}
    rows=[]
    for spec in source["models"]:
        relative=MODEL_LOCAL_PATHS[spec["id"]]; path=repo/relative; file_sha=sha_file(path)
        old=old_models[spec["id"]]
        if file_sha!=old["file"]["sha256"] or path.stat().st_size!=old["file"]["bytes"]:
            raise X2R1Error(f"checkpoint cache drifted: {spec['id']}")
        model, loaded=_load_state(repo,spec,fused=False); state=model.state_dict(); tensor_digest=canonical_state_dict_tensor_digest(state)
        key_count=len(state); parameter_count=sum(parameter.numel() for parameter in model.parameters())
        if (key_count,parameter_count,tensor_digest)!=(spec["expected_key_count"],spec["expected_parameter_count"],old["canonical_tensor_digest"]):
            raise X2R1Error(f"checkpoint semantic identity drifted: {spec['id']}")
        rows.append({"id":spec["id"],"model_id":spec["model_id"],"path":relative.as_posix(),"bytes":path.stat().st_size,"sha256":file_sha,"canonical_tensor_digest":tensor_digest,"key_count":key_count,"parameter_count":parameter_count,"serialized_key_count":loaded["source_key_count"],"strict_load":"PASS","canonical_path":"native_unfused_timm_0.9.16_eval"})
        del model,state,loaded
    fields=("id","model_id","path","bytes","sha256","canonical_tensor_digest","key_count","parameter_count","serialized_key_count","strict_load","canonical_path")
    write_csv_new(repo/output_csv,fields,rows)
    spec=next(row for row in source["models"] if row["id"]=="vit_b16")
    image=_continuity_image(repo,spec); unfused,_=_load_state(repo,spec,fused=False); fused,_=_load_state(repo,spec,fused=True)
    with torch.no_grad():
        unhooked=unfused(image); captured=[]; handles=[block.attn.attn_drop.register_forward_hook(lambda _m,_a,o:captured.append(o)) for block in unfused.blocks]; hooked=unfused(image)
        for handle in handles: handle.remove()
        fused_logits=fused(image)
    unhooked_error=float((unhooked-hooked).abs().max()); fused_error=float((unhooked-fused_logits).abs().max()); same_top1=int(unhooked.argmax())==int(fused_logits.argmax())
    if unhooked_error>=1e-6 or len(captured)!=12 or fused_error>2e-5 or not same_top1:
        raise X2R1Error("native/fused execution diagnostic failed")
    downloaded=sum(row["bytes"] for row in rows if row["id"]!="vit_b16")
    if downloaded!=875799674:
        raise X2R1Error("downloaded checkpoint byte accounting drifted")
    record=file_record(repo,output_csv,canonical_digest=canonical_sha256(rows))
    value=signed("stage_x_x2_r1_model_materialization","PASS",payload(
        identity={"canonical_execution_path":"native_unfused_timm_0.9.16_eval","continuity_sample":"ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"},
        bindings=[artifact_record(repo,RUN_SPEC)],files=[record],counts={"model_count":5,"all_strict_load":5,"downloaded_bytes":downloaded},
        metrics=[{"metric_id":"native_unfused_hooked_unhooked_max_abs","value":unhooked_error,"unit":"raw_logit"},{"metric_id":"fused_unfused_max_abs","value":fused_error,"unit":"raw_logit"}],
        state={"all_checkpoint_hashes_recomputed":True,"all_tensor_digests_recomputed":True,"all_key_counts_recomputed":True,"all_parameter_counts_recomputed":True,"all_strict_load":True,"fused_unfused_top1_unchanged":same_top1,"checkpoint_redownloaded":False,"holm_family_count":5,"full_model_ig_in_holm":False},
    ))
    write_new(repo/output_json,value,repo)
    return {"status":"PASS","model_count":5,"downloaded_bytes":downloaded,"native_unfused_hooked_unhooked_max_abs":unhooked_error,"fused_unfused_max_abs":fused_error,"fused_unfused_top1_unchanged":same_top1}


def gpu_execution_diagnostic(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    import torch
    repo=Path(repo_root).resolve();output=repo/ROOT/"gpu_execution_diagnostic.json"
    if output.exists():value=strict_read(output);validate_artifact(value,repo,verify_files=True);return value
    source=strict_read(repo/OLD_CONFIG);spec=next(row for row in source["models"] if row["id"]=="vit_b16");image=_continuity_image(repo,spec).to("cuda:0");unfused,_=_load_state(repo,spec,fused=False);fused,_=_load_state(repo,spec,fused=True);unfused=unfused.to("cuda:0");fused=fused.to("cuda:0")
    with torch.no_grad():
        native=unfused(image);captured=[];handles=[block.attn.attn_drop.register_forward_hook(lambda _module,_args,value:captured.append(value)) for block in unfused.blocks];hooked=unfused(image)
        for handle in handles:handle.remove()
        fused_logits=fused(image)
    native_error=float((native-hooked).abs().max());fused_error=float((native-fused_logits).abs().max());same_top1=int(native.argmax())==int(fused_logits.argmax())
    if native_error!=0.0 or len(captured)!=12 or fused_error!=1.6927719116210938e-05 or not same_top1:raise X2R1Error("H20 execution diagnostic differs from frozen X2 fact")
    value=signed("stage_x_x2_r1_model_materialization","PASS",payload(identity={"device":torch.cuda.get_device_name(0),"device_index":0,"canonical_execution_path":"native_unfused_timm_0.9.16_eval"},bindings=[artifact_record(repo,ROOT/"model_materialization.json")],metrics=[{"metric_id":"native_unfused_hooked_unhooked_max_abs","value":native_error,"unit":"raw_logit"},{"metric_id":"fused_unfused_max_abs","value":fused_error,"unit":"raw_logit"}],state={"hook_count":len(captured),"fused_unfused_top1_unchanged":same_top1,"fused_value_claimed_lt_1e_6":False,"diagnostic_only":True,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    write_new(output,value,repo);return value


def seal_implementation(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); path=repo/ROOT/"implementation_seal.json"
    if path.exists():
        value=strict_read(path);validate_artifact(value,repo,verify_files=True);return value
    for relative in IMPLEMENTATION_FILES:
        if not (repo/relative).is_file(): raise X2R1Error(f"implementation file missing: {relative}")
    if (repo/ROOT/"units").exists() and any((repo/ROOT/"units").rglob("*")):
        raise X2R1Error("cannot seal implementation after unit execution")
    bindings=[file_record(repo,Path(relative)) for relative in IMPLEMENTATION_FILES]
    bindings.extend((artifact_record(repo,RUN_SPEC),artifact_record(repo,MUTATION_RECEIPT),artifact_record(repo,ROOT/"model_materialization.json"),artifact_record(repo,ROOT/"gpu_execution_diagnostic.json")))
    value=signed("stage_x_x2_r1_implementation_seal","PASS",payload(bindings=bindings,counts={"implementation_file_count":len(IMPLEMENTATION_FILES)},state={"sealed_before_unit_outcome_access":True,"source_changes_after_seal_allowed":False,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    write_new(path,value,repo);return value


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    for command in ("mutate","initialize","materialize","diagnose-gpu","seal"):
        item=sub.add_parser(command); item.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    args=parser.parse_args(argv)
    try:
        result={"mutate":run_mutation_suite,"initialize":initialize,"materialize":materialize_models,"diagnose-gpu":gpu_execution_diagnostic,"seal":seal_implementation}[args.command](args.repo_root)
    except Exception as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__ == "__main__":
    raise SystemExit(main())
