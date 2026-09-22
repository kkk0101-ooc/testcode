#!/usr/bin/env python3
"""Independently validate the fail-closed N2-V2 I0 input-only root."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i0_20260922"
CONTRACT = ROOT / "config/n2_v2/input_identifiability_contract.json"
LEDGER = ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"
ATTEMPTS = ("n2_r1_20260922", "n2_r1b_20260922", "n2_r1c_20260922")
LEGACY_BOUND_PATHS = (
    "执行规划.md", "研究目标.md", "evidence/n1_r3_20260922/terminal_receipt.json",
    "N2隔离Pilot协议.md", "N2-R1C最终生产器恢复修订.md", "config/n2_r1/pilot_contract.json",
    "evidence/n2_r1c_20260922/pair_manifest.json", "evidence/n2_r1c_20260922/pair_inputs.npz",
    "evidence/n2_r1c_20260922/blind_qc.json", "evidence/n2_r1c_20260922/validation.json",
    "N2完成审计.md", "evidence/n2_r1c_20260922/completion_audit.json",
    "schemas/n2_completion_audit.schema.json", "schemas/n2_terminal_receipt.schema.json",
    "evidence/n2_r1c_20260922/terminal_receipt.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(ROOT)).encode()); digest.update(b"\0")
        digest.update(sha(item).encode()); digest.update(b"\n")
    return {"path": str(path.relative_to(ROOT)), "file_count": len(files), "tree_sha256": digest.hexdigest()}


def all_objects_closed(node: Any) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "object" and node.get("additionalProperties") is not False:
            return False
        return all(all_objects_closed(value) for value in node.values())
    return not isinstance(node, list) or all(all_objects_closed(value) for value in node)


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    schemas = {path.name: json.loads(path.read_text()) for path in sorted((ROOT / "schemas").glob("n2_v2*.schema.json"))}
    for schema in schemas.values(): jsonschema.Draft7Validator.check_schema(schema)
    contract = json.loads(CONTRACT.read_text()); ledger = json.loads(LEDGER.read_text())
    closure = json.loads((OUT / "legacy_closure.json").read_text())
    asset = json.loads((OUT / "asset_preflight.json").read_text())
    reviewer = json.loads((OUT / "reviewer_preflight.json").read_text())
    jsonschema.validate(contract, schemas["n2_v2_input_contract.schema.json"])
    jsonschema.validate(ledger, schemas["n2_v2_lineage_exclusions.schema.json"])
    for value in (closure, asset, reviewer): jsonschema.validate(value, schemas["n2_v2_i0_preflight.schema.json"])
    terminal_path = OUT / "terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    checks = []

    def check(identifier: str, condition: bool, evidence: str) -> None:
        if not condition: raise AssertionError(identifier)
        checks.append({"check_id": identifier, "status": "PASS", "evidence": evidence})

    old_validation = json.loads(subprocess.check_output([sys.executable, str(ROOT / "validate_n2.py")], text=True, cwd=ROOT))
    check("LEGACY_TERMINAL_VALID", old_validation["validated_terminal_decision"] == "NO_GO_NOT_IDENTIFIABLE" and len(old_validation["checks"]) == 33, "legacy validator recomputed 33 checks")
    old_audit = json.loads(subprocess.check_output([sys.executable, str(ROOT / "audit_n2_completion.py")], text=True, cwd=ROOT))
    check("LEGACY_COMPLETION_AUDIT_VALID", old_audit["status"] == "PASS_N2_COMPLETION_AUDIT" and old_audit["checklist_items"] == 30, "legacy 30-item audit reran")
    bound = {row["path"]: row["sha256"] for row in closure["bound_file_hashes"]}
    check("LEGACY_BOUND_FILES_CURRENT", set(bound) == set(LEGACY_BOUND_PATHS) and all(bound[path] == sha(ROOT / path) for path in LEGACY_BOUND_PATHS), "15 legacy hashes recomputed")
    snapshots = {row["path"]: row for row in closure["immutable_root_snapshots"]}
    expected = {f"evidence/{attempt}": tree_snapshot(ROOT / "evidence" / attempt) for attempt in ATTEMPTS}
    check("LEGACY_ROOTS_UNCHANGED", snapshots == expected, "three legacy evidence trees recomputed")
    check("V2_SCHEMAS_CLOSED", all(all_objects_closed(schema) for schema in schemas.values()), "all V2 object schemas are closed")
    check("V2_AUTHORIZATION_INPUT_ONLY", not any(contract["authorization"].values()), "all model and N3 authorization flags false")
    inherited = {
        "n1_contract_sha256": contract["inheritance"]["n1_contract_path"],
        "n1_terminal_sha256": contract["inheritance"]["n1_terminal_path"],
        "model_registry_sha256": contract["inheritance"]["model_registry_path"],
        "prior_model_side_contract_sha256": contract["inheritance"]["prior_model_side_contract_path"],
    }
    check("INHERITED_IDENTITIES_BOUND", all(contract["inheritance"][key] == sha(ROOT / path) for key, path in inherited.items()), "N1 and dormant model-side identities recomputed")
    check("LEGACY_TERMINAL_HASH_BOUND", contract["legacy_n2"]["terminal_receipt_sha256"] == sha(ROOT / contract["legacy_n2"]["terminal_receipt_path"]), "legacy terminal hash recomputed")
    check("OLD_ROOTS_DECLARED_IMMUTABLE", set(contract["legacy_n2"]["immutable_attempt_roots"]) == set(expected), "three legacy roots are immutable")
    source_ids = [row["source_group_id"] for row in ledger["records"]]
    image_hashes = [row["image_sha256"] for row in ledger["records"]]
    check("LEGACY_LINEAGE_COUNT", ledger["excluded_source_count"] == len(source_ids) == 30, "30 old lineages")
    check("LEGACY_LINEAGE_UNIQUE", len(set(source_ids)) == len(source_ids) and len(set(image_hashes)) == len(image_hashes), "source and byte identities unique")
    check("LEGACY_LINEAGE_EXCLUDED", all(row["v2_use_policy"] == "REGRESSION_ONLY_EXCLUDED_FROM_ALL_SCIENTIFIC_SPLITS" for row in ledger["records"]), "all old lineages excluded")
    qc = json.loads((ROOT / "evidence/n2_r1c_20260922/blind_qc.json").read_text())
    pair = json.loads((ROOT / "evidence/n2_r1c_20260922/pair_manifest.json").read_text())
    failed = {row["pair_id"]: tuple(row["codes"]) for row in qc["pair_reviews"] if not row["artifact_free"]}
    pair_digests = {row["pair_id"]: row["arm_array_digests"] for row in pair["records"]}
    regression = {row["pair_id"]: row for row in ledger["artifact_regression_cases"]}
    check("SEVEN_REGRESSION_CASES", len(regression) == len(failed) == 7, "seven R1C artifact failures preserved")
    check("REGRESSION_BINDINGS_CURRENT", all(tuple(regression[key]["failure_codes"]) == failed[key] and regression[key]["arm_array_digests"] == pair_digests[key] for key in failed), "failure codes and arm digests recomputed")
    check("REGRESSION_NOT_SCIENTIFIC_DATA", all(row["scientific_use_forbidden"] for row in regression.values()), "regression data barred from scientific splits")
    forbidden = {"torch", "timm", "PIL", "cv2", "numpy", "scipy"}
    sources = [ROOT / "freeze_n2_v2_i0.py", ROOT / "validate_n2_v2_i0.py", ROOT / "src/nns_cst_n2_v2/input_gate.py"]
    check("INPUT_CODE_NO_MODEL_OR_IMAGE_IMPORTS", all(not (imported_roots(path) & forbidden) for path in sources), "AST import scan passes")
    check("NO_ACCEPTED_ASSET_REGISTRY", not (ROOT / asset["accepted_asset_registry_path"]).exists() and asset["accepted_aquatic_scene_count"] == asset["accepted_non_aquatic_scene_count"] == 0, "no accepted scene-only assets")
    check("NO_REVIEW_BUNDLES", reviewer["assigned_reviewer_count"] == reviewer["sealed_review_bundle_count"] == 0 and not list(OUT.glob("review_bundle_*.json")), "zero of two reviewers")
    check("NO_INPUT_PAYLOAD_ACCESS", asset["new_image_payload_reads"] == asset["new_mask_payload_reads"] == 0, "metadata-only I0")
    check("NO_MODEL_OR_GPU_ACCESS", all(value == 0 for value in closure["model_access_counts"].values()), "all model and GPU counters zero")
    check("BLOCKED_ASSETS_FAIL_CLOSED", asset["decision"] == "DO_NOT_GENERATE_PAIRS_OR_REVIEW_CARDS", "missing assets stop pair generation")
    check("BLOCKED_REVIEWERS_FAIL_CLOSED", reviewer["decision"] == "DO_NOT_RUN_BLIND_REVIEW_OR_DOWNSTREAM_GATES", "missing reviewers stop downstream gates")
    blind = contract["blind_review"]
    check("DOUBLE_REVIEW_THRESHOLDS_FROZEN", blind["required_independent_reviewers"] == 2 and blind["minimum_raw_semantic_agreement"] == 0.95 and blind["minimum_cohen_kappa"] == 0.8, "two reviewers, agreement 0.95, kappa 0.8")
    check("ARTIFACT_UNION_REJECT_FROZEN", blind["any_reviewer_artifact_flag_rejects"] and blind["adjudication_cannot_restore_rejected_item"], "artifact union-reject cannot be reversed")
    check("ALL_EIGHT_MATCH_REQUIRED", contract["matching"]["max_unmatched_rate"] == 0.0 and contract["matching"]["all_final_pairs_must_match"], "eight of eight must match")
    check("RESERVE_ORDER_OUTCOME_BLIND", contract["matching"]["outcome_guided_replacement_forbidden"] and contract["asset_policy"]["reserve_recipient_count"] == 4, "four outcome-blind reserves")
    check("NO_PAIR_OR_PASS_RECEIPT", not (OUT / "pair_manifest.json").exists() and not (OUT / "identifiability_freeze_receipt.json").exists(), "blocked I0 has no pair or PASS receipt")
    if terminal is not None:
        jsonschema.validate(terminal, schemas["n2_v2_i0_terminal.schema.json"])
        paths = {
            "protocol_sha256": ROOT / "N2-V2输入可识别性恢复协议.md", "contract_sha256": CONTRACT,
            "contract_schema_sha256": ROOT / "schemas/n2_v2_input_contract.schema.json", "lineage_ledger_sha256": LEDGER,
            "lineage_schema_sha256": ROOT / "schemas/n2_v2_lineage_exclusions.schema.json", "review_schema_sha256": ROOT / "schemas/n2_v2_review_bundle.schema.json",
            "preflight_schema_sha256": ROOT / "schemas/n2_v2_i0_preflight.schema.json", "validation_schema_sha256": ROOT / "schemas/n2_v2_i0_validation.schema.json",
            "terminal_schema_sha256": ROOT / "schemas/n2_v2_i0_terminal.schema.json", "legacy_closure_sha256": OUT / "legacy_closure.json",
            "asset_preflight_sha256": OUT / "asset_preflight.json", "reviewer_preflight_sha256": OUT / "reviewer_preflight.json",
            "validation_sha256": OUT / "validation.json", "validator_sha256": ROOT / "validate_n2_v2_i0.py",
        }
        check("I0_TERMINAL_BINDINGS", all(terminal["artifact_bindings"][key] == sha(path) for key, path in paths.items()), "14 terminal hashes recomputed")
        check("I0_TERMINAL_STOP", terminal["status"] == "BLOCKED_INPUT_ASSETS_AND_REVIEWERS" and terminal["next_gate"].startswith("STOP_I0"), "fresh-root stop enforced")
    result = {
        "schema_version": 1, "study_version": "N2_V2", "stage_id": "N2_V2_INPUT_IDENTIFIABILITY", "attempt_id": "N2_V2_I0_20260922",
        "status": "PASS_I0_FAIL_CLOSED_VALIDATION", "checks": checks,
        "summary": {"check_count": len(checks), "legacy_excluded_source_count": len(source_ids), "artifact_regression_case_count": len(regression),
                    "accepted_scene_asset_count": 0, "sealed_reviewer_bundle_count": 0, "model_access_count": 0, "validated_i0_status": "BLOCKED_INPUT_ASSETS_AND_REVIEWERS"},
    }
    jsonschema.validate(result, schemas["n2_v2_i0_validation.schema.json"])
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if args.write:
        path = OUT / "validation.json"
        if path.exists(): raise RuntimeError("I0_VALIDATION_ALREADY_EXISTS")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
    print(text, end=""); return 0


if __name__ == "__main__":
    raise SystemExit(main())
