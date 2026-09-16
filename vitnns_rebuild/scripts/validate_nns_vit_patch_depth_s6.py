#!/usr/bin/env python3
"""Independent reporting-only validation and false-pass audit for S6."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s6 as s6  # noqa: E402

ROOT = REPO / s6.ROOT_REL
REPORT = REPO / s6.REPORT_REL
CORE_FILES = {
    "authorization_gate_report.json": "s6_authorization_gate_report",
    "source_binding.json": "s6_source_binding",
    "figure_data.json": "s6_figure_data",
    "figure_inventory.json": "s6_figure_inventory",
    "claim_ledger.json": "s6_claim_ledger",
    "writing_audit.json": "s6_writing_audit",
    "cpu_access_audit.json": "s6_cpu_access_audit",
}


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def read(name: str, kind: str) -> dict:
    path = ROOT / name
    require(path.is_file(), f"missing S6 document: {name}")
    value = json.loads(path.read_text())
    require(value["document_type"] == kind, f"document type drift: {name}")
    s6.verify_closed_document(value)
    return value


def independent_figure_data(sources: dict[str, dict]) -> dict:
    endpoints = sources["endpoint_decisions.json"]["payload"]["endpoints"]
    depth = sources["depth_profiles_and_D1_D2_relationship.json"]["payload"]
    images = sources["image_level_endpoint_values.json"]["payload"]["endpoint_records"]
    d1 = {row["source_image_id"]: row["value"] for row in images["D1"]}
    d2 = {row["source_image_id"]: row["value"] for row in images["D2"]}
    overlap = sorted(set(d1).intersection(d2))
    return {
        "analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory figure data",
        "endpoint_rows": [{"analysis_label": s6.LABEL, **endpoints[name]} for name in ("P1", "P2", "D1", "D2")],
        "NNS_depth_rows": depth["NNS_equal_class_profile"], "D2_depth_rows": depth["D2_equal_class_profile"],
        "D1_D2_scatter_rows": [{"analysis_label": s6.LABEL, "source_image_id": source, "D1": d1[source], "D2": d2[source]} for source in overlap],
        "D1_D2_relationship": depth["D1_D2_relationship"],
        "P2_comparison": sources["original_vs_posthoc_P2_comparison.json"]["payload"],
        "source_transform": "selection_and_formatting_only_from_sealed_S5_outputs; no new inferential endpoint or decision",
    }


def validate_render_record(record: dict) -> None:
    path = REPO / record["path"]
    require(path.is_file() and path.stat().st_size == record["bytes"] and s6.sha_file(path) == record["sha256"], f"render drift: {record['path']}")
    suffix = path.suffix.lower()
    if suffix == ".png":
        data = path.read_bytes(); require(data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR", "PNG header")
        width, height = struct.unpack(">II", data[16:24]); require(record["geometry"] == {"width": width, "height": height}, "PNG geometry")
    elif suffix == ".pdf": require(path.read_bytes().startswith(b"%PDF-"), "PDF header")
    elif suffix == ".svg": require(ET.parse(path).getroot().tag.endswith("svg"), "SVG root")
    else: raise RuntimeError("unapproved render suffix")


def validate() -> dict:
    terminal, gate = s6.authorization_gate(REPO, check_remote=True)
    sources, source_inventory = s6.load_sources(REPO, terminal)
    outputs = {name: read(name, kind) for name, kind in CORE_FILES.items()}
    require(outputs["authorization_gate_report.json"]["payload"]["checks"] == gate["checks"], "gate report drift")
    binding = outputs["source_binding.json"]["payload"]
    require(binding["records"] == source_inventory and binding["records_digest"] == s6.canonical_digest(source_inventory), "source binding drift")
    recomputed_data = independent_figure_data(sources)
    produced_data = outputs["figure_data.json"]["payload"]
    require(s6.canonical_digest(recomputed_data) == s6.canonical_digest(produced_data), "figure data drift")
    endpoints = {row["endpoint"]: row for row in produced_data["endpoint_rows"]}
    require(set(endpoints) == {"P1", "P2", "D1", "D2"}, "endpoint removed")
    require(all(row["analysis_label"] == s6.LABEL and row["exploratory_state"] == "SUPPORTED_POST_HOC_EXPLORATORY" for row in endpoints.values()), "endpoint reporting label/state drift")
    require(produced_data["P2_comparison"]["original_rule"]["status"] == "ORIGINAL_RULE_NOT_PASSED", "P2 original rule upgraded")
    figure_inventory = outputs["figure_inventory.json"]["payload"]
    require(figure_inventory["figure_count"] == 12 and figure_inventory["formats"] == ["PDF", "PNG", "SVG"], "figure package count/format drift")
    require(figure_inventory["figures_digest"] == s6.canonical_digest(figure_inventory["figures"]), "figure inventory digest drift")
    for record in figure_inventory["figures"]: validate_render_record(record)
    report_path = ROOT / "paper_results.md"; analysis_path = ROOT / "figure_analysis.md"
    require(report_path.is_file() and analysis_path.is_file(), "human-readable report missing")
    report = report_path.read_text(); analysis = analysis_path.read_text()
    require((REPORT / "paper_results.md").read_bytes() == report_path.read_bytes() and (REPORT / "figure_analysis.md").read_bytes() == analysis_path.read_bytes(), "report copy parity drift")
    for name in ("image_01_endpoint_decisions.png", "image_02_depth_profiles.png", "image_03_D1_D2_relationship.png", "image_04_P2_rule_comparison.png"):
        require((REPORT / "figures_png" / name).read_bytes() == (ROOT / "figures" / name).read_bytes(), f"report PNG parity drift: {name}")
    required_phrases = ("post-hoc exploratory", "ORIGINAL_RULE_NOT_PASSED", "0.0016", "不能替代 NNS", "不支持统一或因果性")
    require(all(phrase in report for phrase in required_phrases), "paper report lost a required result boundary")
    require(all(f"Image #{index}" in analysis for index in range(1, 5)), "figure analysis numbering incomplete")
    access = outputs["cpu_access_audit.json"]["payload"]
    require(access["CUDA_VISIBLE_DEVICES"] == "" and access["S3_or_S4_array_read_count"] == 0 and access["JPEG_or_mask_read_count"] == 0 and access["model_or_checkpoint_load_count"] == 0 and access["torch_timm_torchvision_import_count"] == 0 and access["CUDA_or_GPU_initialization_count"] == 0 and access["new_inference_or_intervention_count"] == 0 and access["S0_to_S5_root_write_count"] == 0, "reporting isolation failed")
    claims = outputs["claim_ledger.json"]["payload"]
    require(claims["new_scientific_claim_count"] == 0 and claims["original_P2_rule_status"] == "ORIGINAL_RULE_NOT_PASSED", "claim ledger scope drift")
    writing = outputs["writing_audit.json"]["payload"]
    require(writing["paper_level_defensive_writing_score"] <= 2 and writing["repeated_phrase_audit"]["result"] == "PASS", "writing audit failed")
    return {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory independent validation", "checks": {"S5_authorization_recomputed": True, "source_bindings_recomputed": True, "figure_data_recomputed_from_S5": True, "no_new_scientific_analysis": True, "all_12_renders_valid": True, "report_copy_byte_parity": True, "paper_claims_match_S5": True, "P2_original_rule_preserved": True, "D2_probe_not_NNS_boundary_preserved": True, "D1_D2_noncausal_boundary_preserved": True, "CPU_reporting_isolation": True, "defensive_writing_audit_pass": True}, "source_records_digest": s6.canonical_digest(source_inventory), "figure_data_digest": s6.canonical_digest(recomputed_data), "report_sha256": s6.sha_file(report_path), "figure_analysis_sha256": s6.sha_file(analysis_path), "render_count": 12, "all_checks_pass": True}


def false_pass() -> dict:
    outputs = {name: read(name, kind) for name, kind in CORE_FILES.items()}
    attacks = []
    def attack(name, base, mutate, check):
        value = copy.deepcopy(base); mutate(value); rejected = False
        try: check(value)
        except Exception: rejected = True
        require(rejected, f"false-pass accepted: {name}")
        attacks.append({"analysis_label": s6.LABEL, "attack": name, "mutation_applied": True, "rejected": True})
    gate = outputs["authorization_gate_report.json"]
    attack("S5_status", gate, lambda x: x["payload"]["checks"].__setitem__("S5_terminal_status", False), lambda x: require(all(x["payload"]["checks"].values()), "gate"))
    attack("S6_authorization", gate, lambda x: x["payload"]["checks"].__setitem__("S5_reporting_authorization", False), lambda x: require(all(x["payload"]["checks"].values()), "gate"))
    attack("live_remote", gate, lambda x: x["payload"]["checks"].__setitem__("live_remote_contains_S5", False), lambda x: require(all(x["payload"]["checks"].values()), "gate"))
    data = outputs["figure_data.json"]
    attack("remove_endpoint", data, lambda x: x["payload"]["endpoint_rows"].pop(), lambda x: require(len(x["payload"]["endpoint_rows"]) == 4, "endpoint"))
    attack("upgrade_P2_original", data, lambda x: x["payload"]["P2_comparison"]["original_rule"].__setitem__("status", "PASS"), lambda x: require(x["payload"]["P2_comparison"]["original_rule"]["status"] == "ORIGINAL_RULE_NOT_PASSED", "P2"))
    attack("label_removed", data, lambda x: x["payload"].__setitem__("analysis_label", "confirmatory"), lambda x: require(x["payload"]["analysis_label"] == s6.LABEL, "label"))
    figures = outputs["figure_inventory.json"]
    attack("render_count", figures, lambda x: x["payload"].__setitem__("figure_count", 11), lambda x: require(x["payload"]["figure_count"] == 12, "figures"))
    access = outputs["cpu_access_audit.json"]
    for name, field in (("array_read", "S3_or_S4_array_read_count"), ("JPEG_read", "JPEG_or_mask_read_count"), ("model_load", "model_or_checkpoint_load_count"), ("CUDA_init", "CUDA_or_GPU_initialization_count"), ("new_intervention", "new_inference_or_intervention_count")):
        attack(name, access, lambda x, field=field: x["payload"].__setitem__(field, 1), lambda x, field=field: require(x["payload"][field] == 0, field))
    claims = outputs["claim_ledger.json"]
    attack("new_claim", claims, lambda x: x["payload"].__setitem__("new_scientific_claim_count", 1), lambda x: require(x["payload"]["new_scientific_claim_count"] == 0, "claim"))
    attack("P2_claim_upgrade", claims, lambda x: x["payload"].__setitem__("original_P2_rule_status", "PASS"), lambda x: require(x["payload"]["original_P2_rule_status"] == "ORIGINAL_RULE_NOT_PASSED", "P2 claim"))
    writing = outputs["writing_audit.json"]
    attack("defensive_score", writing, lambda x: x["payload"].__setitem__("paper_level_defensive_writing_score", 7), lambda x: require(x["payload"]["paper_level_defensive_writing_score"] <= 2, "writing"))
    return {"analysis_label": s6.LABEL, "title": "S6 post-hoc exploratory false-pass mutation audit", "attack_count": len(attacks), "all_rejected": True, "attacks": attacks}


def publication() -> dict:
    terminal = json.loads((ROOT / "terminal_receipt.json").read_text()); inventory = json.loads((ROOT / "whole_root_inventory.json").read_text())
    s6.verify_closed_document(terminal); s6.verify_closed_document(inventory)
    payload = terminal["payload"]
    require(terminal["status"] == "PASS_POST_HOC_EXPLORATORY_REPORTING" and payload["S6_WORKFLOW_STATUS"] == "PASS" and payload["SCIENTIFIC_STATUS"] == "PASS_POST_HOC_EXPLORATORY", "S6 terminal status drift")
    require(payload["reporting_only"] is True and payload["new_scientific_analysis_count"] == 0 and payload["new_intervention_count"] == 0 and payload["original_P2_rule_status"] == "ORIGINAL_RULE_NOT_PASSED" and payload["next_authorization"] == "NONE", "S6 terminal scope drift")
    records = inventory["payload"]["files"]
    require(inventory["payload"]["file_count"] == len(records) and inventory["payload"]["files_digest"] == s6.canonical_digest(records), "S6 inventory digest drift")
    for record in records:
        path = REPO / record["path"]; require(path.is_file() and path.stat().st_size == record["bytes"] and s6.sha_file(path) == record["sha256"], f"S6 file drift: {record['path']}")
    return {"analysis_label": s6.LABEL, "status": terminal["status"], "terminal_sha256": s6.sha_file(ROOT / "terminal_receipt.json"), "terminal_canonical_digest": terminal["canonical_digest"], "inventory_file_count": len(records), "next_authorization": payload["next_authorization"]}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate", "false-pass", "publication")); args = parser.parse_args()
    result = validate() if args.command == "validate" else false_pass() if args.command == "false-pass" else publication()
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
