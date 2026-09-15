"""CPU-only S4-R11 post-hoc exploratory reconstruction primitives."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from . import nns_patch_depth_s4 as s4
from . import nns_patch_depth_s4_group_sequential as gs

STUDY_ID = "NNS_VIT_PATCH_DEPTH_SEMANTICS"
STAGE_ID = "S4-POST-HOC-EXPLORATORY"
RUN_ID = "s4_r11_posthoc_exploratory_20260916"
ROOT_REL = Path("evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916")
R10_ROOT_REL = Path("evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915")
R10_PAYLOAD_REL = Path("artifacts/nns_vit_patch_depth/s4_r10_group_sequential_20260915")
SCHEDULE_REL = Path("manifests/nns_vit_patch_depth/s4_r10_group_sequential_schedule.json")
EXPECTED_SELECTION_DIGEST = "ad0c10e5057d767ef4329f81b22c5f1569afb7e820c7f88bb5199da7e03cb9b2"
EXPECTED_CUTOFF_NS = 1789511492710807249
EXPECTED_CUTOFF_INDEX = 300
EXPECTED_SELECTED_CLASS_COUNT = 120
EXPECTED_STATES = {
    "VALID_NNS": 235, "NA_VOID_MASK": 5, "VALID_OCCLUSION": 448,
    "NA_MATCHED_CONTROL_UNAVAILABLE": 22, "NA_UPSTREAM_NNS": 10,
}
EXPECTED_NUMERIC = {
    "P1": {"estimate": 0.0626003110, "lower": 0.0573913097, "upper": 0.0678940724, "half_width": 0.0052937614, "adjusted_lower": 0.0514420599, "adjusted_p": 1.8768816516782683e-19},
    "P2": {"estimate": 0.0749396239, "lower": 0.0570841929, "upper": 0.0927266631, "half_width": 0.0178554310, "adjusted_lower": 0.0347448291, "adjusted_p": 1.613666525930498e-05},
    "D1": {"estimate": 0.0363141999, "lower": 0.0316151874, "upper": 0.0409722310, "half_width": 0.0046990126, "adjusted_lower": 0.0258156526, "adjusted_upper": 0.0468127473, "adjusted_p": 4.136584313500504e-05},
}
FAILURE_STATES = {"FAILED_IMPLEMENTATION", "BLOCKED_DATA_INTEGRITY", "FAILED_NNS", "FAILED_OCCLUSION"}

class IntegrityError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def sha_file(path: Path) -> str:
    before = path.stat(); h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed during hash: " + str(path))
    return h.hexdigest()

def signed(document_type: str, status: str, payload: Mapping[str, Any], producer_commit: str | None = None) -> dict[str, Any]:
    doc = {"schema_version": 1, "study_id": STUDY_ID, "stage_id": STAGE_ID, "run_id": RUN_ID, "document_type": document_type, "status": status, "payload": dict(payload)}
    if producer_commit is not None:
        doc["producer_commit"] = producer_commit
    doc["canonical_digest"] = canonical_digest(doc)
    return doc

def verify_signed(doc: Mapping[str, Any]) -> bool:
    value = dict(doc); claimed = value.pop("canonical_digest", None)
    return isinstance(claimed, str) and claimed == canonical_digest(value)

def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)

def artifact(repo: Path, path: Path) -> dict[str, Any]:
    result = {"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if path.suffix == ".json":
        doc = json.loads(path.read_text())
        if isinstance(doc.get("canonical_digest"), str): result["canonical_digest"] = doc["canonical_digest"]
    return result

def immutable_root_inventory(repo: Path) -> dict[str, Any]:
    records = []
    for root_rel in (R10_ROOT_REL, R10_PAYLOAD_REL):
        root = repo / root_rel
        require(root.is_dir(), "immutable R10 root absent: " + root_rel.as_posix())
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            records.append({"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)})
    return {"root_paths": [R10_ROOT_REL.as_posix(), R10_PAYLOAD_REL.as_posix()], "file_count": len(records), "total_bytes": sum(r["bytes"] for r in records), "records_digest": canonical_digest(records), "records": records}

def _terminal(repo: Path, r10_root: Path, unit_type: str, index: int) -> tuple[Path, dict[str, Any], str] | None:
    new = r10_root / "unit_terminals" / unit_type.lower() / f"unit-{index:04d}.json"
    if new.is_file():
        doc = json.loads(new.read_text()); require(gs.verify_signed(doc), "R10 terminal digest drift")
        return new, doc, "NEW"
    binding = r10_root / "rebind_terminals" / unit_type.lower() / f"unit-{index:04d}.json"
    if not binding.is_file(): return None
    record = json.loads(binding.read_text()); require(gs.verify_signed(record), "R10 rebind digest drift")
    source = repo / record["payload"]["source_terminal_path"]
    require(source.is_file() and sha_file(source) == record["payload"]["source_terminal_sha256"], "R4 rebound terminal drift")
    doc = json.loads(source.read_text()); require(s4.verify_signed(doc), "R4 source terminal digest drift")
    return source, doc, "REBIND"

def freeze_selection(repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule_doc = json.loads((repo / SCHEDULE_REL).read_text()); require(gs.verify_signed(schedule_doc), "schedule digest drift")
    rows = []
    for source in schedule_doc["payload"]["records"]:
        if int(source["formal_class_rank"]) >= 200: continue
        resolved = {u: _terminal(repo, repo / R10_ROOT_REL, u, int(source["schedule_index"])) for u in s4.UNIT_TYPES}
        if not all(resolved.values()): continue
        terminals = []
        for unit_type in s4.UNIT_TYPES:
            path, doc, provenance = resolved[unit_type]  # type: ignore[misc]
            p = doc["payload"]
            require(p["source_id"] == source["source_image_id"] and int(p["schedule_index"]) == int(source["schedule_index"]), "terminal source identity drift")
            terminals.append({"unit_type": unit_type, "path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path), "canonical_digest": doc["canonical_digest"], "status": doc["status"], "completed_unix_ns": int(p["completed_unix_ns"]), "provenance": provenance})
        rows.append({"schedule_index": int(source["schedule_index"]), "formal_class_rank": int(source["formal_class_rank"]), "source_id": source["source_image_id"], "image_sha256": source["image_sha256"], "annotation_sha256": source["annotation_sha256"], "source_completion_time_max_unix_ns": max(t["completed_unix_ns"] for t in terminals), "terminals": terminals})
    rows.sort(key=lambda r: (r["source_completion_time_max_unix_ns"], r["schedule_index"]))
    selected = rows[:240]
    selection_digest = canonical_digest(selected)
    states = dict(Counter(t["status"] for row in selected for t in row["terminals"]))
    failures = [t for row in selected for t in row["terminals"] if t["status"] in FAILURE_STATES]
    checks = {
        "source_count_240": len(selected) == 240,
        "selected_class_rank_count_120": len({r["formal_class_rank"] for r in selected}) == EXPECTED_SELECTED_CLASS_COUNT,
        "selection_identity_digest": selection_digest == EXPECTED_SELECTION_DIGEST,
        "cutoff_time": bool(selected) and selected[-1]["source_completion_time_max_unix_ns"] == EXPECTED_CUTOFF_NS,
        "cutoff_schedule_index": bool(selected) and selected[-1]["schedule_index"] == EXPECTED_CUTOFF_INDEX,
        "terminal_state_counts": states == EXPECTED_STATES,
        "unexplained_failure_count_zero": not failures,
    }
    summary = {"eligible_resolved_L1_source_count_at_freeze": len(rows), "selected_source_count": len(selected), "selected_class_rank_count": len({r["formal_class_rank"] for r in selected}), "selection_identity_records_digest": selection_digest, "expected_selection_identity_records_digest": EXPECTED_SELECTION_DIGEST, "cutoff_completion_unix_ns": selected[-1]["source_completion_time_max_unix_ns"], "cutoff_schedule_index": selected[-1]["schedule_index"], "terminal_state_counts": states, "unexplained_failure_count": len(failures), "anchor_checks": checks, "all_anchor_checks_pass": all(checks.values())}
    return selected, summary

def _array(repo: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = repo / str(record["path"]); require(path.is_file() and path.stat().st_size == int(record["bytes"]), "array file/bytes drift")
    require(sha_file(path) == record["sha256"], "array SHA drift")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"], "array shape/dtype drift")
    require(bool(np.isfinite(value).all()), "array nonfinite"); require(s4.array_digest(value) == record["canonical_array_digest"], "array canonical digest drift")
    return value

def compute(repo: Path, selected: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoints: dict[str, list[tuple[int, float]]] = {e: [] for e in gs.ENDPOINTS}; input_arrays = 0; plans = 0; matched = possible = 0; states = Counter()
    for row in selected:
        docs = {}
        terminal_records = {t["unit_type"]: t for t in row["terminals"]}
        for unit_type in s4.UNIT_TYPES:
            record = terminal_records[unit_type]; path = repo / record["path"]; require(sha_file(path) == record["sha256"], "manifest terminal SHA drift")
            doc = json.loads(path.read_text()); require((gs.verify_signed(doc) or s4.verify_signed(doc)) and doc["canonical_digest"] == record["canonical_digest"], "manifest terminal digest drift")
            require(doc["payload"]["source_id"] == row["source_id"], "manifest terminal source drift"); docs[unit_type] = doc; states[doc["status"]] += 1
        nns = docs["NNS_IMAGE"]; rank = int(row["formal_class_rank"])
        if nns["status"] == "VALID_NNS":
            by = {Path(r["path"]).stem: r for r in nns["payload"]["arrays"]}
            for required in ("p1_numerators", "p1_denominators", "p1_valid_blocks"): require(required in by, "mandatory NNS array absent")
            numerator = _array(repo, by["p1_numerators"]).astype(np.float64); denominator = _array(repo, by["p1_denominators"]).astype(np.float64); valid = _array(repo, by["p1_valid_blocks"]).astype(bool); input_arrays += 3
            plan_record = nns["payload"]["p2_plan"]["record"]; plan_path = repo / plan_record["path"]; require(sha_file(plan_path) == plan_record["sha256"], "P2 plan SHA drift")
            plan = json.loads(plan_path.read_text()); claimed = plan.pop("canonical_digest"); require(canonical_digest(plan) == claimed == plan_record["canonical_digest"], "P2 plan digest drift"); plans += 1; possible += 36; matched += sum(c["status"] == "MATCHED" for c in plan["coordinates"])
            if bool(valid.all()) and bool((denominator > 0).all()):
                corrected = numerator[:, 0] / denominator - (numerator[:, 1:] / denominator[:, None]).mean(1)
                endpoints["P1"].append((rank, float(corrected.mean()))); endpoints["D1"].append((rank, float(corrected[8:12].mean() - corrected[0:4].mean())))
        for unit_type in ("OCCLUSION_PRIMARY", "OCCLUSION_SENSITIVITY"):
            doc = docs[unit_type]
            if doc["status"] == "VALID_OCCLUSION":
                require(len(doc["payload"]["arrays"]) == 1, "occlusion array count drift"); logits = _array(repo, doc["payload"]["arrays"][0]); input_arrays += 1; require(tuple(logits.shape) == (36, 5), "occlusion geometry drift")
                if unit_type == "OCCLUSION_PRIMARY": endpoints["P2"].append((rank, float((logits.astype(np.float64)[:, 1:].mean(1) - logits.astype(np.float64)[:, 0]).mean())))
            dependency = doc["payload"]["dependency_terminal"]
            require(dependency["canonical_digest"] == nns["canonical_digest"] and sha_file(repo / dependency["path"]) == dependency["sha256"], "occlusion dependency binding drift")
    summaries = {}
    table = json.loads((repo / R10_ROOT_REL / "critical_boundary_table.json").read_text())["payload"]
    for endpoint in gs.ENDPOINTS:
        by_class: dict[int, list[float]] = {}
        for rank, value in endpoints[endpoint]: by_class.setdefault(rank, []).append(value)
        groups = [values for _, values in sorted(by_class.items())]; class_means = np.asarray([math.fsum(values) / len(values) for values in groups], dtype=np.float64)
        estimate = float(class_means.mean()); se = float(class_means.std(ddof=1) / math.sqrt(len(class_means))); bca = gs.bca_mean_interval(groups, endpoint, 1)
        boundary = float(gs.boundary_lookup(table, 3, endpoint)[0]); adjusted = {"lower": estimate - boundary * se, "upper": estimate + boundary * se if endpoint == "D1" else None, "boundary_z": boundary}
        summaries[endpoint] = {"scheduled_images": 240, "valid_images": len(endpoints[endpoint]), "NA_images": 240 - len(endpoints[endpoint]), "valid_fraction": len(endpoints[endpoint]) / 240, "NA_fraction": 1 - len(endpoints[endpoint]) / 240, "valid_classes": len(groups), "estimate": estimate, "class_cluster_standard_error": se, "ordinary_BCa_95": bca, "family_size_three_L1_adjusted_CI": adjusted}
    z = {"P1": [(summaries["P1"]["estimate"] - .02) / summaries["P1"]["class_cluster_standard_error"]], "P2": [summaries["P2"]["estimate"] / summaries["P2"]["class_cluster_standard_error"]], "D1": [(summaries["D1"]["estimate"] - .02) / summaries["D1"]["class_cluster_standard_error"]], "D1_LOWER": [(-summaries["D1"]["estimate"] - .02) / summaries["D1"]["class_cluster_standard_error"]]}
    pvalues = gs.repeated_adjusted_p_values(z, 0); closed = gs.crossed_by_look(z, table, 0)
    for endpoint in gs.ENDPOINTS: summaries[endpoint]["heterogeneous_null_adjusted_p"] = pvalues[endpoint]
    numeric_anchor_checks = {}
    for endpoint, expected in EXPECTED_NUMERIC.items():
        observed = summaries[endpoint]; values = {"estimate": observed["estimate"], "lower": observed["ordinary_BCa_95"]["lower"], "upper": observed["ordinary_BCa_95"]["upper"], "half_width": observed["ordinary_BCa_95"]["half_width"], "adjusted_lower": observed["family_size_three_L1_adjusted_CI"]["lower"], "adjusted_p": observed["heterogeneous_null_adjusted_p"]}
        if endpoint == "D1": values["adjusted_upper"] = observed["family_size_three_L1_adjusted_CI"]["upper"]
        numeric_anchor_checks[endpoint] = all(abs(values[key] - expected[key]) <= max(5e-10, abs(expected[key]) * 5e-8) for key in expected)
    validity = {e: summaries[e]["valid_fraction"] >= .85 and summaries[e]["NA_fraction"] <= .15 for e in gs.ENDPOINTS}; matching_fraction = matched / possible
    posthoc = {"P1_validity": validity["P1"], "P2_validity": validity["P2"], "D1_validity": validity["D1"], "P2_matching_fraction_at_least_0_85": matching_fraction >= .85, "P1_precision_at_most_0_015": summaries["P1"]["ordinary_BCa_95"]["half_width"] <= .015, "P2_precision_at_most_0_02": summaries["P2"]["ordinary_BCa_95"]["half_width"] <= .02, "D1_precision_at_most_0_015": summaries["D1"]["ordinary_BCa_95"]["half_width"] <= .015, "P1_adjusted_lower_above_0_02": summaries["P1"]["family_size_three_L1_adjusted_CI"]["lower"] > .02, "P2_point_at_least_0_07": summaries["P2"]["estimate"] >= .07, "P2_adjusted_lower_above_zero": summaries["P2"]["family_size_three_L1_adjusted_CI"]["lower"] > 0, "D1_adjusted_interval_outside_equivalence_band": summaries["D1"]["family_size_three_L1_adjusted_CI"]["lower"] > .02 or summaries["D1"]["family_size_three_L1_adjusted_CI"]["upper"] < -.02, "heterogeneous_null_closed_testing_all_rejected": all(closed["elementary_rejections"].values()), "numeric_anchors_all_match": all(numeric_anchor_checks.values())}
    original = {"P2_estimate_above_0_10": summaries["P2"]["estimate"] > .10, "P2_adjusted_lower_above_0_10": summaries["P2"]["family_size_three_L1_adjusted_CI"]["lower"] > .10, "P2_precision_at_most_0_015": summaries["P2"]["ordinary_BCa_95"]["half_width"] <= .015}
    integrity = {"array_records_validated": input_arrays, "P2_plans_validated": plans, "terminal_state_counts": dict(states), "matching_count": matched, "matching_possible": possible, "matching_fraction": matching_fraction}
    return {"endpoint_summaries": summaries, "heterogeneous_null_z": z, "closed_testing": closed, "numeric_anchor_checks": numeric_anchor_checks, "posthoc_gate_checks": posthoc, "all_posthoc_numeric_gates_pass": all(posthoc.values()), "original_P2_rule_checks": original, "ORIGINAL_RULE_NOT_PASSED": not all(original.values())}, integrity
