"""CPU-only S5 post-hoc exploratory analysis primitives."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm


STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S5-CPU-ONLY-POST-HOC-EXPLORATORY"
RUN_ID = "s5_r1_posthoc_exploratory_20260916"
ANALYSIS_LABEL = "post-hoc exploratory"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s5_r1_posthoc_exploratory_20260916")
R12_TERMINAL_REL = Path("evidence/nns_vit_patch_depth/s4_r12_selection_anchor_recovery_20260916/terminal_receipt.json")
R12_TERMINAL_SHA256 = "a4726e87dc4053dd636d8096d72132550d8d7b802a3438d7463bed786366f575"
R12_TERMINAL_CANONICAL = "bba750a44315a634f66a0d9cc9b32ab506662455f3fecfbe1284fe6d2ff7345e"
R12_PUBLICATION_COMMIT = "a2c25b41918c6cc89f5b14f191ed749ff1ffe937"
R12_MANIFEST_REL = Path("evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/exploratory_240_source_manifest.json")
R12_MANIFEST_SHA256 = "81e3414cd04681bfc626b4cca836caa3cc2052b916aba18454b43eca08fcd1f5"
R12_MANIFEST_CANONICAL = "94b44455674f7fd68a85bfefc97ca1677d09beefdb7fa056f1998b1f728974f0"
FULL_RECORDS_DIGEST = "3bb68cb6bb627f4a84f1a6d7bd54e26d430541b4ae4cd565a6fe3eb2d46863e9"
MEMBERSHIP_DIGEST = "ad0c10e5057d767ef4329f81b22c5f1569afb7e820c7f88bb5199da7e03cb9b2"
S3_TERMINAL_REL = Path("evidence/nns_vit_patch_depth/s3_r7_20260915/terminal_receipt.json")
S3_TERMINAL_SHA256 = "8c1a4f82d3bb553246cc4c019e4e2286a8dfb7dfd8b058265af93ad5f1e192d4"
S3_TERMINAL_CANONICAL = "1b6f7308f135dc69ad0146635a31eb529cfac703710fe5a340c009c56ce1215d"
S3_D2_MANIFEST_REL = Path("evidence/nns_vit_patch_depth/s3_r7_20260915/test_prediction_error_manifest.json")
S3_D2_MANIFEST_SHA256 = "1559d00511a6e8dc5fef5031fd2d3f23ddb9683673167dc6e8fdedf662d7ceb1"
S3_D2_MANIFEST_CANONICAL = "5cb42d554bea12fcd9afc7013935a51289a10def94e4441a9647a301fdb66186"
S3_TERMINAL_INVENTORY_REL = Path("evidence/nns_vit_patch_depth/s3_r7_20260915/source_terminal_rebind_inventory.json")
S3_TERMINAL_INVENTORY_SHA256 = "5b2cd38d27fc7ef08ecacefc7a2d3948ffe0b3c6f418aebb5c52184c626854ae"
S3_TERMINAL_INVENTORY_CANONICAL = "abc440b95210203a1f4db1c7b2c9c9a499901a90e688e10d108c09d177f18850"
S2_SAP_REL = Path("evidence/nns_vit_patch_depth/s2_r4_20260914/statistical_analysis_plan.json")
S2_SAP_SHA256 = "f8672e9488b44f493337f2201d7be5f9ad376b352d864220dbf21536dab457a3"
S2_SAP_CANONICAL = "5b7ed5fc44c2a778cfeff6543c65a91d3ae9d48f1520b49d8c7db03f3383b437"
ENDPOINTS = ("P1", "P2", "D1")
ALL_ENDPOINTS = ("P1", "P2", "D1", "D2")
R12_STATES = {"VALID_NNS": 235, "NA_VOID_MASK": 5, "VALID_OCCLUSION": 448, "NA_MATCHED_CONTROL_UNAVAILABLE": 22, "NA_UPSTREAM_NNS": 10}
R12_BOUNDARY = {"P1": 2.3939797998185095, "P2": 2.3939797998185095, "D1": 2.63825727347675}
S4_BOOTSTRAP_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S4_R5_GROUP_SEQUENTIAL_POWER_V1"
S5_BOOTSTRAP_NAMESPACE = "NNS_VIT_PATCH_DEPTH_S5_R1_POSTHOC_EXPLORATORY_V1"
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".pth", ".pt", ".ckpt", ".safetensors"}
ALLOWED_ARRAY_NAMES = {
    "p1_numerators.npy", "p1_denominators.npy", "p1_valid_blocks.npy", "p1_raw_alignment.npy",
    "occlusion_image_channel_mean_logits.npy", "occlusion_normalized_zero_logits.npy", "feature_mse.npy", "control_mse.npy",
    "feature_prediction.npy", "control_prediction.npy", "occupancy.npy",
}


class S5Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S5Error(message)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"file changed while hashing: {path}")
    return digest.hexdigest()


def verify_signed(document: Mapping[str, Any]) -> bool:
    copy = dict(document)
    claimed = copy.pop("canonical_digest", None)
    return isinstance(claimed, str) and claimed == canonical_digest(copy)


def signed(document_type: str, status: str, payload: Mapping[str, Any], producer_commit: str | None = None) -> dict[str, Any]:
    document = {
        "schema_version": 1, "study_id": STUDY_ID, "stage_id": STAGE_ID, "run_id": RUN_ID,
        "document_type": document_type, "status": status, "payload": jsonable(dict(payload)),
    }
    if producer_commit is not None:
        document["producer_commit"] = producer_commit
    document["canonical_digest"] = canonical_digest(document)
    return document


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(jsonable(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def artifact(repo: Path, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"analysis_label": ANALYSIS_LABEL, "path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, Mapping) and isinstance(value.get("canonical_digest"), str):
            record["canonical_digest"] = value["canonical_digest"]
    return record


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def authorization_gate(repo: Path, check_remote: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repo / R12_TERMINAL_REL
    require(path.is_file(), "R12 terminal missing")
    file_sha = sha_file(path)
    terminal = json.loads(path.read_text())
    recomputed = canonical_digest({key: value for key, value in terminal.items() if key != "canonical_digest"})
    payload = terminal.get("payload", {})
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "symbolic-ref", "--short", "HEAD")
    tracking = git(repo, "rev-parse", "@{upstream}")
    remote = tracking
    if check_remote:
        text = git(repo, "ls-remote", "origin", f"refs/heads/{branch}")
        require(bool(text), "live remote branch absent")
        remote = text.split()[0]
    checks = {
        "terminal_file_sha256": file_sha == R12_TERMINAL_SHA256,
        "terminal_claimed_canonical_digest": terminal.get("canonical_digest") == R12_TERMINAL_CANONICAL,
        "terminal_recomputed_canonical_digest": recomputed == R12_TERMINAL_CANONICAL,
        "terminal_status": terminal.get("status") == "PASS_POST_HOC_EXPLORATORY",
        "scientific_status": payload.get("SCIENTIFIC_STATUS") == "PASS_POST_HOC_EXPLORATORY",
        "formal_confirmatory_gate_closed": payload.get("FORMAL_CONFIRMATORY_GATE_OPEN") is False,
        "exploratory_first_layer_gate_open": payload.get("EXPLORATORY_FIRST_LAYER_GATE_OPEN") is True,
        "next_authorization": payload.get("next_authorization") == "S5_CPU_ONLY_POST_HOC_EXPLORATORY_ANALYSIS_THEN_S6_EXPLORATORY_REPORTING",
        "producer_commit": terminal.get("producer_commit") == R12_PUBLICATION_COMMIT,
    }
    for label, commit in (("local_head_contains_R12", head), ("tracking_ref_contains_R12", tracking), ("live_remote_contains_R12", remote)):
        result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", R12_PUBLICATION_COMMIT, commit], check=False)
        checks[label] = result.returncode == 0
    require(all(checks.values()), "S5_NOT_AUTHORIZED")
    return terminal, {
        "analysis_label": ANALYSIS_LABEL, "title": "S5 post-hoc exploratory authorization gate", "checks": checks,
        "all_checks_pass": True,
        "R12_terminal": {"path": R12_TERMINAL_REL.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha, "canonical_digest": recomputed},
        "git": {"branch": branch, "local_head": head, "tracking_ref": tracking, "live_remote": remote, "required_R12_publication_commit": R12_PUBLICATION_COMMIT},
        "endpoint_access_started_after_gate": True,
    }


class ReadAudit:
    """Exact scientific-input whitelist plus immutable read ledger."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()
        self.allowed: dict[str, str] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def allow(self, relative: str | Path, role: str) -> Path:
        relative_path = Path(relative)
        require(not relative_path.is_absolute(), "absolute scientific input path forbidden")
        path = (self.repo / relative_path).resolve()
        require(path == self.repo or self.repo in path.parents, "scientific input escapes repository")
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, "forbidden image/model/checkpoint read requested")
        key = path.relative_to(self.repo).as_posix()
        previous = self.allowed.get(key)
        require(previous is None or previous == role, f"scientific input role drift: {key}")
        self.allowed[key] = role
        return path

    def _record(self, path: Path, kind: str, sha256: str, extra: Mapping[str, Any] | None = None) -> None:
        key = path.relative_to(self.repo).as_posix()
        require(key in self.allowed, f"read outside scientific whitelist: {key}")
        if key not in self.records:
            self.records[key] = {"analysis_label": ANALYSIS_LABEL, "path": key, "role": self.allowed[key], "kind": kind, "bytes": path.stat().st_size, "sha256": sha256, "read_count": 1, **dict(extra or {})}
        else:
            self.records[key]["read_count"] += 1

    def json(self, relative: str | Path, role: str, expected_sha: str | None = None, expected_canonical: str | None = None) -> dict[str, Any]:
        path = self.allow(relative, role)
        require(path.is_file() and path.suffix == ".json", f"JSON scientific input missing: {relative}")
        sha = sha_file(path)
        if expected_sha is not None:
            require(sha == expected_sha, f"JSON SHA drift: {relative}")
        value = json.loads(path.read_text())
        claimed = value.get("canonical_digest")
        if claimed is not None:
            require(verify_signed(value), f"JSON canonical drift: {relative}")
        if expected_canonical is not None:
            require(claimed == expected_canonical, f"JSON canonical binding drift: {relative}")
        self._record(path, "json", sha, {"canonical_digest": claimed} if claimed is not None else None)
        return value

    def array(self, record: Mapping[str, Any], role: str) -> np.ndarray:
        path = self.allow(str(record["path"]), role)
        require(path.is_file() and path.suffix == ".npy", "non-NPY array input forbidden")
        require(path.name in ALLOWED_ARRAY_NAMES or "corrected_control_mse" in path.as_posix(), f"array not in endpoint whitelist: {path.name}")
        sha = sha_file(path)
        require(path.stat().st_size == int(record["bytes"]) and sha == record["sha256"], f"array bytes/SHA drift: {record['path']}")
        array = np.load(path, allow_pickle=False)
        require(list(array.shape) == list(record["shape"]) and str(array.dtype) == record["dtype"], f"array geometry/dtype drift: {record['path']}")
        require(not np.issubdtype(array.dtype, np.number) or bool(np.isfinite(array).all()), f"nonfinite array: {record['path']}")
        digest = hashlib.sha256()
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode()); digest.update(b"\0")
        digest.update(str(tuple(contiguous.shape)).encode()); digest.update(b"\0"); digest.update(contiguous.tobytes())
        require(digest.hexdigest() == record["canonical_array_digest"], f"canonical array digest drift: {record['path']}")
        self._record(path, "npy", sha, {"canonical_array_digest": digest.hexdigest(), "dtype": str(array.dtype), "shape": list(array.shape)})
        return array

    def inventory(self) -> list[dict[str, Any]]:
        return [self.records[key] for key in sorted(self.records)]


def _seed(namespace: str, *parts: Any) -> int:
    material = namespace + "\0" + "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def bca(groups: Sequence[Sequence[float]], endpoint: str, namespace: str) -> dict[str, Any]:
    values = [np.asarray(group, dtype=np.float64) for group in groups]
    require(values and all(len(group) > 0 for group in values), f"empty BCa group: {endpoint}")
    means = np.asarray([group.mean() for group in values], dtype=np.float64)
    estimate = float(means.mean())
    rng = np.random.Generator(np.random.PCG64(_seed(namespace, "ordinary_BCa", endpoint, 1, 20000)))
    bootstrap = np.empty(20000, dtype=np.float64)
    for start in range(0, 20000, 1000):
        count = min(1000, 20000 - start)
        total = np.zeros(count, dtype=np.float64)
        for group in values:
            total += group[rng.integers(0, len(group), size=(count, len(group)))].mean(axis=1)
        bootstrap[start:start + count] = total / len(values)
    proportion = min(max(float(np.mean(bootstrap < estimate)), 1 / 40000), 1 - 1 / 40000)
    z0 = float(norm.ppf(proportion))
    jackknife: list[float] = []
    for index, group in enumerate(values):
        if len(group) <= 1:
            continue
        other = float(means.sum() - means[index])
        for omitted in range(len(group)):
            jackknife.append((other + (group.sum() - group[omitted]) / (len(group) - 1)) / len(values))
    jack = np.asarray(jackknife, dtype=np.float64)
    jack_mean = float(jack.mean())
    delta = jack_mean - jack
    denominator = 6 * float(np.sum(delta * delta)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(delta ** 3)) / denominator
    adjusted_quantiles = []
    for alpha in (0.025, 0.975):
        z = float(norm.ppf(alpha))
        adjusted_quantiles.append(float(norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))))
    lower, upper = (float(value) for value in np.quantile(bootstrap, adjusted_quantiles, method="linear"))
    return {
        "estimate": estimate, "lower": lower, "upper": upper,
        "half_width": max(estimate - lower, upper - estimate),
        "replicates": 20000, "seed_namespace": namespace,
        "bias_correction_z0": z0, "acceleration": acceleration,
        "resampling_unit": "source_image_within_each_fixed_class_then_equal_class_mean",
    }


def family_three_adjusted_p(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    z = {
        "P1": (float(summary["P1"]["estimate"]) - 0.02) / float(summary["P1"]["class_cluster_standard_error"]),
        "P2": float(summary["P2"]["estimate"]) / float(summary["P2"]["class_cluster_standard_error"]),
        "D1": (float(summary["D1"]["estimate"]) - 0.02) / float(summary["D1"]["class_cluster_standard_error"]),
        "D1_LOWER": (-float(summary["D1"]["estimate"]) - 0.02) / float(summary["D1"]["class_cluster_standard_error"]),
    }
    local: dict[tuple[str, ...], float] = {}
    for size in (1, 2, 3):
        for intersection in itertools.combinations(ENDPOINTS, size):
            candidate = 1.0
            for endpoint in intersection:
                candidate = min(candidate, size * 0.05 * float(norm.sf(z[endpoint])) / 0.025)
                if endpoint == "D1":
                    candidate = min(candidate, 2 * size * 0.05 * float(norm.sf(z["D1_LOWER"])) / 0.025)
            local[intersection] = min(1.0, candidate)
    return {endpoint: max(value for intersection, value in local.items() if endpoint in intersection) for endpoint in ENDPOINTS}


def _equal_class(groups: Mapping[Any, Sequence[float]]) -> tuple[list[list[float]], float, float]:
    ordered = [list(groups[key]) for key in sorted(groups)]
    means = np.asarray([math.fsum(group) / len(group) for group in ordered], dtype=np.float64)
    estimate = float(means.mean())
    standard_error = float(means.std(ddof=1) / math.sqrt(len(means)))
    return ordered, estimate, standard_error


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(x.std()) == 0 or float(y.std()) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute(repo: Path, audit: ReadAudit) -> dict[str, Any]:
    manifest = audit.json(R12_MANIFEST_REL, "R12_BOUND_240_SOURCE_MANIFEST", R12_MANIFEST_SHA256, R12_MANIFEST_CANONICAL)
    records = manifest["payload"]["records"]
    require(len(records) == 240, "R12 source count drift")
    projection = [{"schedule_index": int(row["schedule_index"]), "formal_class_rank": int(row["formal_class_rank"]), "source_image_id": row["source_id"]} for row in records]
    full_digest = canonical_digest(records)
    membership_digest = canonical_digest(projection)
    require(full_digest == FULL_RECORDS_DIGEST and membership_digest == MEMBERSHIP_DIGEST, "R12 selection identity drift")
    require(len({int(row["formal_class_rank"]) for row in records}) == 122, "R12 selected class-rank count drift")

    endpoint_values: dict[str, list[dict[str, Any]]] = {endpoint: [] for endpoint in ALL_ENDPOINTS}
    endpoint_groups: dict[str, dict[Any, list[float]]] = {endpoint: defaultdict(list) for endpoint in ALL_ENDPOINTS}
    nns_profile_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    terminal_states: Counter[str] = Counter()
    matched_coordinates = 0
    possible_coordinates = 0
    matched_controls = 0
    possible_controls = 0
    p2_valid_nns_denominator = 0
    p2_valid_image_count = 0
    d1_by_source: dict[str, float] = {}

    for row in records:
        documents: dict[str, dict[str, Any]] = {}
        terminal_records = {terminal["unit_type"]: terminal for terminal in row["terminals"]}
        require(set(terminal_records) == {"NNS_IMAGE", "OCCLUSION_PRIMARY", "OCCLUSION_SENSITIVITY"}, "R12 terminal topology drift")
        for unit_type in ("NNS_IMAGE", "OCCLUSION_PRIMARY", "OCCLUSION_SENSITIVITY"):
            terminal_record = terminal_records[unit_type]
            document = audit.json(terminal_record["path"], "R12_BOUND_SOURCE_TERMINAL", terminal_record["sha256"], terminal_record["canonical_digest"])
            require(document["status"] == terminal_record["status"], "bound terminal status drift")
            require(document["payload"]["source_id"] == row["source_id"] and int(document["payload"]["schedule_index"]) == int(row["schedule_index"]), "bound terminal identity drift")
            documents[unit_type] = document
            terminal_states[document["status"]] += 1

        rank = int(row["formal_class_rank"])
        source_id = str(row["source_id"])
        nns = documents["NNS_IMAGE"]
        p1_state = "NA"
        p2_state = "NA"
        d1_state = "NA"
        p1_reason = nns["status"] if nns["status"] != "VALID_NNS" else None
        d1_reason = p1_reason
        p2_reason = documents["OCCLUSION_PRIMARY"]["status"] if documents["OCCLUSION_PRIMARY"]["status"] != "VALID_OCCLUSION" else None

        if nns["status"] == "VALID_NNS":
            p2_valid_nns_denominator += 1
            by_name = {Path(item["path"]).name: item for item in nns["payload"]["arrays"]}
            required_names = {"p1_numerators.npy", "p1_denominators.npy", "p1_valid_blocks.npy", "p1_raw_alignment.npy"}
            require(required_names <= set(by_name), "P1 raw array set incomplete")
            numerator = audit.array(by_name["p1_numerators.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(np.float64)
            denominator = audit.array(by_name["p1_denominators.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(np.float64)
            valid_blocks = audit.array(by_name["p1_valid_blocks.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(bool)
            raw_alignment = audit.array(by_name["p1_raw_alignment.npy"], "R12_BOUND_P1_RAW_ARRAY").astype(np.float64)
            require(numerator.shape == (12, 65) and denominator.shape == (12,) and valid_blocks.shape == (12,) and raw_alignment.shape == (12,), "P1 raw geometry drift")
            require(bool(valid_blocks.all()) and bool((denominator > 0).all()), "VALID_NNS has invalid P1 denominator")
            recomputed_raw = numerator[:, 0] / denominator
            require(np.allclose(recomputed_raw, raw_alignment, rtol=1e-12, atol=1e-12), "P1 raw alignment drift")
            corrected = recomputed_raw - (numerator[:, 1:] / denominator[:, None]).mean(axis=1)
            p1_value = float(corrected.mean())
            d1_value = float(corrected[8:12].mean() - corrected[0:4].mean())
            endpoint_values["P1"].append({"analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "value": p1_value})
            endpoint_values["D1"].append({"analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "value": d1_value})
            endpoint_groups["P1"][rank].append(p1_value)
            endpoint_groups["D1"][rank].append(d1_value)
            d1_by_source[source_id] = d1_value
            p1_state = d1_state = "VALID"
            p1_reason = d1_reason = None
            for block in range(12):
                nns_profile_rows.append({
                    "analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank,
                    "block": block, "component": "first_residual_addition_all_head_aggregate_attention_branch",
                    "target": "annotation_class_raw_logit", "raw_alignment": float(recomputed_raw[block]),
                    "position_control_corrected_alignment": float(corrected[block]),
                })
            plan_record = nns["payload"]["p2_plan"]["record"]
            plan = audit.json(plan_record["path"], "R12_BOUND_P2_PLAN", plan_record["sha256"], plan_record["canonical_digest"])
            coordinates = plan["coordinates"]
            require(len(coordinates) == 36, "P2 plan coordinate count drift")
            possible_coordinates += 36
            possible_controls += 36 * 4
            matched_coordinates += sum(coordinate["status"] == "MATCHED" for coordinate in coordinates)
            matched_controls += sum(len(coordinate["controls"]) for coordinate in coordinates if coordinate["status"] == "MATCHED")

        for unit_type in ("OCCLUSION_PRIMARY", "OCCLUSION_SENSITIVITY"):
            document = documents[unit_type]
            if document["status"] != "VALID_OCCLUSION":
                continue
            require(nns["status"] == "VALID_NNS", "valid occlusion without valid NNS dependency")
            plan_record = nns["payload"]["p2_plan"]["record"]
            plan = audit.json(plan_record["path"], "R12_BOUND_P2_PLAN", plan_record["sha256"], plan_record["canonical_digest"])
            require(all(coordinate["status"] == "MATCHED" for coordinate in plan["coordinates"]), "valid occlusion has unmatched coordinate")
            array_record = document["payload"]["arrays"][0]
            values = audit.array(array_record, "R12_BOUND_P2_RAW_LOGIT_ARRAY").astype(np.float64)
            require(values.shape == (36, 5), "P2 logit geometry drift")
            clean_logit = float(document["payload"]["clean_logit"])
            operator = str(document["payload"]["operator"])
            for index, (coordinate, logits) in enumerate(zip(plan["coordinates"], values)):
                control_logits = [float(value) for value in logits[1:]]
                control_details = coordinate["controls"]
                require(len(control_details) == 4, "matched P2 coordinate lacks four controls")
                intervention_rows.append({
                    "analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank,
                    "unit_type": unit_type, "operator": operator, "coordinate_index": index,
                    "block": int(coordinate["block"]), "budget": int(coordinate["budget"]),
                    "selected_patch_ids": [int(value) for value in coordinate["selected"]],
                    "selected_stratum_counts": [int(value) for value in coordinate["stratum_counts"]],
                    "selected_boundary_complexity": int(coordinate["selected_boundary_complexity"]),
                    "control_patch_ids": [[int(value) for value in control["indices"]] for control in control_details],
                    "control_stratum_counts": [[int(value) for value in control["stratum_counts"]] for control in control_details],
                    "control_boundary_absolute_errors": [int(control["boundary_absolute_error"]) for control in control_details],
                    "control_overlap_counts": [int(control["overlap_count"]) for control in control_details],
                    "clean_logit": clean_logit, "selected_occluded_logit": float(logits[0]),
                    "control_occluded_logits": control_logits,
                    "selected_raw_logit_drop": clean_logit - float(logits[0]),
                    "control_raw_logit_drops": [clean_logit - value for value in control_logits],
                    "matched_control_corrected_logit_drop": float(np.mean(logits[1:]) - logits[0]),
                })
            if unit_type == "OCCLUSION_PRIMARY":
                p2_value = float((values[:, 1:].mean(axis=1) - values[:, 0]).mean())
                endpoint_values["P2"].append({"analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank, "value": p2_value})
                endpoint_groups["P2"][rank].append(p2_value)
                p2_valid_image_count += 1
                p2_state = "VALID"
                p2_reason = None

        validity_rows.append({
            "analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": rank,
            "scope": "R12_SELECTED_240", "unit_states": {unit_type: documents[unit_type]["status"] for unit_type in sorted(documents)},
            "endpoint_states": {
                "P1": {"state": p1_state, "reason": p1_reason}, "P2": {"state": p2_state, "reason": p2_reason},
                "D1": {"state": d1_state, "reason": d1_reason},
            },
        })

    require(dict(terminal_states) == R12_STATES, "R12 terminal-state accounting drift")
    require(p2_valid_image_count == 224 and p2_valid_nns_denominator == 235, "P2 matched-control image denominator drift")
    require(matched_coordinates == 8447 and possible_coordinates == 8460, "P2 matched-coordinate denominator drift")
    require(matched_controls == 33788 and possible_controls == 33840, "P2 matched-control denominator drift")

    s3_terminal = audit.json(S3_TERMINAL_REL, "R12_AUTHORIZED_S3_METADATA", S3_TERMINAL_SHA256, S3_TERMINAL_CANONICAL)
    require(s3_terminal["status"] == "PASS" and s3_terminal["payload"]["test_prediction_count"] == 3424, "S3 terminal binding drift")
    bound_d2 = s3_terminal["payload"]["required_evidence"]["test_prediction_error_manifest.json"]
    require(bound_d2["path"] == S3_D2_MANIFEST_REL.as_posix() and bound_d2["sha256"] == S3_D2_MANIFEST_SHA256 and bound_d2["canonical_digest"] == S3_D2_MANIFEST_CANONICAL, "S3 D2 manifest terminal binding drift")
    bound_inventory = s3_terminal["payload"]["required_evidence"]["source_terminal_rebind_inventory.json"]
    require(bound_inventory["path"] == S3_TERMINAL_INVENTORY_REL.as_posix() and bound_inventory["sha256"] == S3_TERMINAL_INVENTORY_SHA256 and bound_inventory["canonical_digest"] == S3_TERMINAL_INVENTORY_CANONICAL, "S3 source-terminal inventory binding drift")
    sap = audit.json(S2_SAP_REL, "R12_AUTHORIZED_S2_METADATA", S2_SAP_SHA256, S2_SAP_CANONICAL)
    require(sap["payload"]["contract"]["family"] == ["P1", "P2", "D1", "D2"], "S2 endpoint family metadata drift")
    d2_manifest = audit.json(S3_D2_MANIFEST_REL, "R12_AUTHORIZED_S3_D2_MANIFEST", S3_D2_MANIFEST_SHA256, S3_D2_MANIFEST_CANONICAL)
    d2_records = d2_manifest["payload"]["records"]
    require(len(d2_records) == 3482 and canonical_digest(d2_records) == d2_manifest["payload"]["records_digest"], "S3 D2 record identity drift")
    terminal_inventory = audit.json(S3_TERMINAL_INVENTORY_REL, "R12_AUTHORIZED_S3_METADATA", S3_TERMINAL_INVENTORY_SHA256, S3_TERMINAL_INVENTORY_CANONICAL)
    terminal_by_index = {int(record["schedule_index"]): record for record in terminal_inventory["payload"]["records"]}
    d2_states: Counter[str] = Counter()
    d2_by_source: dict[str, float] = {}
    d2_profile_rows: list[dict[str, Any]] = []
    for record in d2_records:
        source_id = str(record["source_image_id"])
        wnid = str(record["annotation_wnid"])
        state = str(record["terminal_state"])
        d2_states[state] += 1
        if state != "VALID_FEATURE":
            validity_rows.append({
                "analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": None,
                "scope": "S3_R7_D2_FULL_SEALED_QUEUE", "unit_states": {"D2": state},
                "endpoint_states": {"D2": {"state": "NA", "reason": state}},
            })
            continue
        arrays = record["prediction_arrays"]
        terminal_record = terminal_by_index[int(record["schedule_index"])]
        terminal = audit.json(terminal_record["path"], "R12_AUTHORIZED_S3_SOURCE_TERMINAL", terminal_record["sha256"], terminal_record["canonical_digest"])
        require(terminal["payload"]["source_image_id"] == source_id and terminal["payload"]["terminal_state"] == "VALID_FEATURE", "S3 D2 source-terminal identity drift")
        terminal_arrays = terminal["payload"]["arrays"]
        feature_records = [item for item in arrays if Path(item["path"]).name == "feature_mse.npy"]
        control_records = [item for item in arrays if Path(item["path"]).name == "control_mse.npy" or "corrected_control_mse" in item["path"]]
        feature_prediction_records = [item for item in arrays if Path(item["path"]).name == "feature_prediction.npy"]
        control_prediction_records = [item for item in arrays if Path(item["path"]).name == "control_prediction.npy"]
        occupancy_records = [item for item in terminal_arrays if Path(item["path"]).name == "occupancy.npy"]
        require(len(feature_records) == len(control_records) == len(feature_prediction_records) == len(control_prediction_records) == len(occupancy_records) == 1, "D2 prediction/error-array inventory drift")
        feature_mse = audit.array(feature_records[0], "R12_AUTHORIZED_S3_D2_ERROR_ARRAY").astype(np.float64)
        control_mse = audit.array(control_records[0], "R12_AUTHORIZED_S3_D2_ERROR_ARRAY").astype(np.float64)
        feature_prediction = audit.array(feature_prediction_records[0], "R12_AUTHORIZED_S3_D2_PREDICTION_ARRAY").astype(np.float64)
        control_prediction = audit.array(control_prediction_records[0], "R12_AUTHORIZED_S3_D2_PREDICTION_ARRAY").astype(np.float64)
        occupancy = audit.array(occupancy_records[0], "R12_AUTHORIZED_S3_D2_TARGET_ARRAY").astype(np.float64)
        require(feature_mse.shape == (12,) and control_mse.shape == (), "D2 error-array geometry drift")
        require(feature_prediction.shape == (12, 196) and control_prediction.shape == (196,) and occupancy.shape == (4, 196), "D2 prediction/target-array geometry drift")
        target = occupancy[3]
        recomputed_feature_mse = np.mean((feature_prediction - target[None, :]) ** 2, axis=1, dtype=np.float64)
        recomputed_control_mse = float(np.mean((control_prediction - target) ** 2, dtype=np.float64))
        require(np.allclose(recomputed_feature_mse, feature_mse, rtol=1e-6, atol=1e-9), "D2 feature MSE recomputation drift")
        require(np.allclose(recomputed_control_mse, control_mse, rtol=1e-6, atol=1e-9), "D2 control MSE recomputation drift")
        gain = recomputed_control_mse - recomputed_feature_mse
        d2_value = float(gain[8:12].mean() - gain[0:4].mean())
        endpoint_values["D2"].append({"analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "annotation_wnid": wnid, "value": d2_value})
        endpoint_groups["D2"][wnid].append(d2_value)
        d2_by_source[source_id] = d2_value
        for block in range(12):
            d2_profile_rows.append({
                "analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "annotation_wnid": wnid,
                "block": block, "component": "frozen_block_output_patch_features", "target": "target_patch_occupancy",
                "feature_probe_mse": float(recomputed_feature_mse[block]), "low_level_control_mse": recomputed_control_mse,
                "probe_gain_control_mse_minus_feature_mse": float(gain[block]),
            })
        validity_rows.append({
            "analysis_label": ANALYSIS_LABEL, "source_image_id": source_id, "formal_class_rank": None,
            "scope": "S3_R7_D2_FULL_SEALED_QUEUE", "unit_states": {"D2": state},
            "endpoint_states": {"D2": {"state": "VALID", "reason": None}},
        })
    require(dict(d2_states) == {"VALID_FEATURE": 3424, "NA_D2_REQUIRED_SPLIT_OR_MODEL_MISSING": 52, "NA_TARGET_DISAPPEARED_AFTER_CROP": 6}, "D2 validity accounting drift")
    require(len(endpoint_groups["D2"]) == 859, "D2 valid class count drift")

    summaries: dict[str, dict[str, Any]] = {}
    totals = {"P1": (240, 235, 5, 0), "P2": (240, 224, 16, 0), "D1": (240, 235, 5, 0), "D2": (3482, 3424, 58, 0)}
    for endpoint in ALL_ENDPOINTS:
        groups, estimate, standard_error = _equal_class(endpoint_groups[endpoint])
        interval = bca(groups, endpoint, S4_BOOTSTRAP_NAMESPACE if endpoint in ENDPOINTS else S5_BOOTSTRAP_NAMESPACE)
        total, valid, na_count, failed = totals[endpoint]
        summaries[endpoint] = {
            "analysis_label": ANALYSIS_LABEL, "endpoint": endpoint, "n_total": total, "n_valid": valid,
            "n_NA": na_count, "n_failed": failed, "valid_class_count": len(groups),
            "estimate": estimate, "class_cluster_standard_error": standard_error, "ordinary_BCa_95": interval,
        }
    adjusted_p = family_three_adjusted_p(summaries)
    for endpoint in ENDPOINTS:
        boundary = R12_BOUNDARY[endpoint]
        estimate = summaries[endpoint]["estimate"]
        se = summaries[endpoint]["class_cluster_standard_error"]
        summaries[endpoint]["adjusted_interval"] = {
            "scope": "family-size-three_L1_post-hoc_exploratory", "boundary_z": boundary,
            "lower": estimate - boundary * se, "upper": estimate + boundary * se if endpoint == "D1" else None,
        }
        summaries[endpoint]["adjusted_p"] = adjusted_p[endpoint]
        summaries[endpoint]["adjustment_scope"] = "family-size-three_L1_post-hoc_exploratory_no_strong_FWER_claim"
    d2 = summaries["D2"]
    d2_z_upper = (d2["estimate"] - 0.01) / d2["class_cluster_standard_error"]
    d2_z_lower = (-d2["estimate"] - 0.01) / d2["class_cluster_standard_error"]
    d2_p = min(1.0, 2 * min(float(norm.sf(d2_z_upper)), float(norm.sf(d2_z_lower))))
    d2["adjusted_interval"] = {
        "scope": "D2_standalone_two-direction_post-hoc_exploratory", "boundary_z": float(norm.ppf(0.975)),
        "lower": d2["estimate"] - float(norm.ppf(0.975)) * d2["class_cluster_standard_error"],
        "upper": d2["estimate"] + float(norm.ppf(0.975)) * d2["class_cluster_standard_error"],
    }
    d2["adjusted_p"] = d2_p
    d2["adjustment_scope"] = "one-slot_D2_standalone_post-hoc_exploratory_no-cross-family-or-strong-FWER-claim"

    checks_by_endpoint = {
        "P1": {"validity": summaries["P1"]["n_valid"] / summaries["P1"]["n_total"] >= 0.85, "precision": summaries["P1"]["ordinary_BCa_95"]["half_width"] <= 0.015, "adjusted_lower_above_0_02": summaries["P1"]["adjusted_interval"]["lower"] > 0.02},
        "P2": {"validity": summaries["P2"]["n_valid"] / summaries["P2"]["n_total"] >= 0.85, "matched_control_image_fraction": p2_valid_image_count / p2_valid_nns_denominator >= 0.85, "point_estimate_at_least_0_07": summaries["P2"]["estimate"] >= 0.07, "family_three_adjusted_lower_above_zero": summaries["P2"]["adjusted_interval"]["lower"] > 0.0, "ordinary_BCa_half_width_at_most_0_02": summaries["P2"]["ordinary_BCa_95"]["half_width"] <= 0.02},
        "D1": {"validity": summaries["D1"]["n_valid"] / summaries["D1"]["n_total"] >= 0.85, "precision": summaries["D1"]["ordinary_BCa_95"]["half_width"] <= 0.015, "adjusted_interval_outside_plus_or_minus_0_02": summaries["D1"]["adjusted_interval"]["lower"] > 0.02 or summaries["D1"]["adjusted_interval"]["upper"] < -0.02},
        "D2": {"validity": summaries["D2"]["n_valid"] / summaries["D2"]["n_total"] >= 0.9, "minimum_valid_images": summaries["D2"]["n_valid"] >= max(200, math.ceil(0.9 * 2 * 872)), "standalone_adjusted_interval_outside_plus_or_minus_0_01": summaries["D2"]["adjusted_interval"]["lower"] > 0.01 or summaries["D2"]["adjusted_interval"]["upper"] < -0.01, "standalone_adjusted_p_at_most_0_05": summaries["D2"]["adjusted_p"] <= 0.05},
    }
    for endpoint in ALL_ENDPOINTS:
        summaries[endpoint]["posthoc_threshold"] = {"P1": 0.02, "P2": 0.07, "D1": 0.02, "D2": 0.01}[endpoint]
        summaries[endpoint]["threshold_direction"] = "positive" if endpoint in ("P1", "P2") else "two-direction"
        summaries[endpoint]["decision_checks"] = checks_by_endpoint[endpoint]
        summaries[endpoint]["exploratory_state"] = "SUPPORTED_POST_HOC_EXPLORATORY" if all(checks_by_endpoint[endpoint].values()) else "INCONCLUSIVE_POST_HOC_EXPLORATORY"

    nns_by_block: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in nns_profile_rows:
        nns_by_block[int(row["block"])][int(row["formal_class_rank"])].append((float(row["raw_alignment"]), float(row["position_control_corrected_alignment"])))
    nns_aggregate = []
    for block in range(12):
        raw_class = [math.fsum(value[0] for value in rows) / len(rows) for _, rows in sorted(nns_by_block[block].items())]
        corrected_class = [math.fsum(value[1] for value in rows) / len(rows) for _, rows in sorted(nns_by_block[block].items())]
        nns_aggregate.append({"analysis_label": ANALYSIS_LABEL, "block": block, "valid_classes": len(raw_class), "equal_class_raw_alignment": float(np.mean(raw_class)), "equal_class_position_control_corrected_alignment": float(np.mean(corrected_class))})
    d2_by_block: dict[int, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in d2_profile_rows:
        d2_by_block[int(row["block"])][str(row["annotation_wnid"])].append((float(row["feature_probe_mse"]), float(row["low_level_control_mse"]), float(row["probe_gain_control_mse_minus_feature_mse"])))
    d2_aggregate = []
    for block in range(12):
        feature = [math.fsum(value[0] for value in rows) / len(rows) for _, rows in sorted(d2_by_block[block].items())]
        control = [math.fsum(value[1] for value in rows) / len(rows) for _, rows in sorted(d2_by_block[block].items())]
        gain = [math.fsum(value[2] for value in rows) / len(rows) for _, rows in sorted(d2_by_block[block].items())]
        d2_aggregate.append({"analysis_label": ANALYSIS_LABEL, "block": block, "valid_classes": len(feature), "equal_class_feature_probe_mse": float(np.mean(feature)), "equal_class_low_level_control_mse": float(np.mean(control)), "equal_class_probe_gain": float(np.mean(gain))})
    overlap = sorted(set(d1_by_source) & set(d2_by_source))
    d1_overlap = [d1_by_source[source] for source in overlap]
    d2_overlap = [d2_by_source[source] for source in overlap]
    relationship = {
        "analysis_label": ANALYSIS_LABEL, "paired_valid_source_count": len(overlap),
        "D1_estimate": summaries["D1"]["estimate"], "D2_estimate": summaries["D2"]["estimate"],
        "aggregate_direction_relation": "SAME_DIRECTION" if summaries["D1"]["estimate"] * summaries["D2"]["estimate"] > 0 else "DIFFERENT_OR_ZERO_DIRECTION",
        "paired_image_Pearson_correlation": _correlation(d1_overlap, d2_overlap),
        "paired_image_same_sign_fraction": float(np.mean(np.sign(d1_overlap) == np.sign(d2_overlap))) if overlap else None,
        "interpretation_boundary": "descriptive association only; neither depth correlation nor probe result establishes a causal formation mechanism or NNS significance",
    }
    by_operator: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in intervention_rows:
        by_operator[(str(row["operator"]), int(row["block"]), int(row["budget"]))].append(row)
    intervention_summary = []
    for (operator, block, budget), rows in sorted(by_operator.items()):
        corrected_values = [float(row["matched_control_corrected_logit_drop"]) for row in rows]
        boundary_values = [value for row in rows for value in row["control_boundary_absolute_errors"]]
        overlap_values = [value for row in rows for value in row["control_overlap_counts"]]
        intervention_summary.append({
            "analysis_label": ANALYSIS_LABEL, "operator": operator, "block": block, "budget": budget,
            "matched_image_count": len(rows), "matched_control_count": 4 * len(rows),
            "mean_matched_control_corrected_logit_drop": float(np.mean(corrected_values)),
            "mean_boundary_absolute_error": float(np.mean(boundary_values)), "maximum_boundary_absolute_error": int(max(boundary_values)),
            "mean_selected_control_overlap_count": float(np.mean(overlap_values)), "maximum_selected_control_overlap_count": int(max(overlap_values)),
        })

    return {
        "identity": {
            "analysis_label": ANALYSIS_LABEL, "selected_source_count": 240, "selected_terminal_count": 720,
            "selected_distinct_class_rank_count": 122, "P1_D1_valid_class_count": len(endpoint_groups["P1"]),
            "P2_valid_class_count": len(endpoint_groups["P2"]), "D2_valid_class_count": len(endpoint_groups["D2"]),
            "full_manifest_records_digest": full_digest, "membership_projection_digest": membership_digest,
            "membership_projection": projection,
        },
        "endpoint_values": endpoint_values, "endpoint_summaries": summaries,
        "nns_profile_rows": nns_profile_rows, "d2_profile_rows": d2_profile_rows,
        "nns_aggregate_profile": nns_aggregate, "d2_aggregate_profile": d2_aggregate,
        "depth_relationship": relationship, "intervention_rows": intervention_rows,
        "intervention_summary": intervention_summary,
        "matching": {
            "analysis_label": ANALYSIS_LABEL, "NNS_valid_images_denominator": p2_valid_nns_denominator,
            "P2_valid_matched_images": p2_valid_image_count, "matched_image_fraction": p2_valid_image_count / p2_valid_nns_denominator,
            "matched_coordinates": matched_coordinates, "possible_coordinates": possible_coordinates,
            "matched_coordinate_fraction": matched_coordinates / possible_coordinates,
            "matched_controls": matched_controls, "possible_controls": possible_controls,
            "matched_control_fraction": matched_controls / possible_controls,
        },
        "validity_rows": validity_rows, "R12_terminal_states": dict(terminal_states), "D2_terminal_states": dict(d2_states),
    }


PAYLOAD_KEYS: dict[str, set[str]] = {
    "s5_authorization_gate_report": {"analysis_label", "title", "checks", "all_checks_pass", "R12_terminal", "git", "endpoint_access_started_after_gate"},
    "s5_input_inventory": {"analysis_label", "title", "read_whitelist_policy", "records", "record_count", "records_digest"},
    "s5_image_level_endpoint_values": {"analysis_label", "title", "endpoint_records", "record_counts", "records_digest"},
    "s5_endpoint_decisions": {"analysis_label", "title", "endpoints", "family_three_scope", "D2_scope", "all_endpoints_retained"},
    "s5_original_vs_posthoc_P2_comparison": {"analysis_label", "title", "original_rule", "posthoc_rule", "disclosure"},
    "s5_depth_profiles_and_D1_D2_relationship": {"analysis_label", "title", "NNS_image_block_records", "D2_image_block_records", "NNS_equal_class_profile", "D2_equal_class_profile", "D1_D2_relationship", "interpretation_boundary"},
    "s5_intervention_controls_and_matching": {"analysis_label", "title", "records", "records_digest", "aggregate_by_operator_block_budget", "matching_diagnostics", "interpretation_boundary"},
    "s5_validity_accounting": {"analysis_label", "title", "records", "records_digest", "R12_terminal_states", "D2_terminal_states", "typed_NA_policy", "invalid_endpoint_family_policy"},
    "s5_claim_ledger": {"analysis_label", "title", "claims", "global_forbidden_claims", "S6_executed"},
    "s5_cpu_access_audit": {"analysis_label", "title", "CUDA_VISIBLE_DEVICES", "forbidden_import_count", "forbidden_imports_loaded", "JPEG_or_mask_decode_count", "model_or_checkpoint_load_count", "CUDA_or_GPU_initialization_count", "new_intervention_count", "scientific_read_whitelist_enforced", "scientific_read_count", "scientific_read_records_digest", "S0_to_S4_root_write_count", "S6_execution_count"},
    "s5_closed_schema_registry": {"analysis_label", "title", "envelope_additional_properties", "payload_additional_properties", "document_payload_keys", "nested_record_key_contracts", "schema_self_test"},
    "s5_independent_validation_report": {"analysis_label", "title", "producer_status_fields_read_before_recompute", "checks", "recomputed_identity", "recomputed_endpoint_summaries", "recomputed_output_digests", "scientific_read_count", "scientific_read_records_digest"},
    "s5_false_pass_mutation_audit": {"analysis_label", "title", "attack_count", "all_rejected", "attacks"},
    "s5_whole_root_inventory": {"analysis_label", "title", "files", "file_count", "files_digest", "excluded_late_bound"},
    "s5_terminal_receipt": {"analysis_label", "title", "S5_WORKFLOW_STATUS", "SCIENTIFIC_STATUS", "FORMAL_CONFIRMATORY_GATE_OPEN", "posthoc_exploratory_only", "strong_FWER_preserved_claim", "original_S4_R10_STOP_SUCCESS_claimed", "original_P2_rule_status", "endpoint_states", "R12_selection_identity", "required_evidence", "S6_execution_count", "next_authorization"},
}


NESTED_RECORD_KEYS: dict[str, set[str]] = {
    "endpoint_summary": {"analysis_label", "endpoint", "n_total", "n_valid", "n_NA", "n_failed", "valid_class_count", "estimate", "class_cluster_standard_error", "ordinary_BCa_95", "adjusted_interval", "adjusted_p", "adjustment_scope", "posthoc_threshold", "threshold_direction", "decision_checks", "exploratory_state"},
    "P1_D1_image_record": {"analysis_label", "source_image_id", "formal_class_rank", "value"},
    "D2_image_record": {"analysis_label", "source_image_id", "annotation_wnid", "value"},
    "NNS_depth_record": {"analysis_label", "source_image_id", "formal_class_rank", "block", "component", "target", "raw_alignment", "position_control_corrected_alignment"},
    "D2_depth_record": {"analysis_label", "source_image_id", "annotation_wnid", "block", "component", "target", "feature_probe_mse", "low_level_control_mse", "probe_gain_control_mse_minus_feature_mse"},
    "intervention_record": {"analysis_label", "source_image_id", "formal_class_rank", "unit_type", "operator", "coordinate_index", "block", "budget", "selected_patch_ids", "selected_stratum_counts", "selected_boundary_complexity", "control_patch_ids", "control_stratum_counts", "control_boundary_absolute_errors", "control_overlap_counts", "clean_logit", "selected_occluded_logit", "control_occluded_logits", "selected_raw_logit_drop", "control_raw_logit_drops", "matched_control_corrected_logit_drop"},
    "validity_record": {"analysis_label", "source_image_id", "formal_class_rank", "scope", "unit_states", "endpoint_states"},
    "input_JSON_inventory_record": {"analysis_label", "path", "role", "kind", "bytes", "sha256", "read_count", "canonical_digest"},
    "input_NPY_inventory_record": {"analysis_label", "path", "role", "kind", "bytes", "sha256", "read_count", "canonical_array_digest", "dtype", "shape"},
    "JSON_artifact_record": {"analysis_label", "path", "bytes", "sha256", "canonical_digest"},
}


def verify_closed_document(document: Mapping[str, Any]) -> None:
    envelope = {"schema_version", "study_id", "stage_id", "run_id", "document_type", "status", "payload", "producer_commit", "canonical_digest"}
    require(set(document) == envelope, f"open or incomplete document envelope: {document.get('document_type')}")
    require(verify_signed(document), f"document canonical digest failed: {document.get('document_type')}")
    document_type = str(document["document_type"])
    require(document_type in PAYLOAD_KEYS, f"unknown S5 document type: {document_type}")
    require(set(document["payload"]) == PAYLOAD_KEYS[document_type], f"open or incomplete payload: {document_type}")
