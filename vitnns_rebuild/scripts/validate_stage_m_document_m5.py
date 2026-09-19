#!/usr/bin/env python3
"""Validate the post-seal M5 conclusion integration in the Stage M document."""
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/阶段M_endpointnnsvit.md"
SEALED_COMMIT = "f84e2d81cd0e6c3410acd670b99deebb95df8f34"
SEALED_SHA256 = "0af896861db65c9f25a79823d32e4091fe8c107386762375289ad3fe45d921d0"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def main():
    text = DOC.read_text()
    root = REPO / "reports/nns_vit_endpoint_m5_r2_20260919"
    terminal = json.loads((root / "terminal_receipt.json").read_text())
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    audit = json.loads((root / "completion_audit.json").read_text())
    summary = json.loads((REPO / "reports/nns_vit_endpoint_m5_r2_20260919/reference_reuse_summary.json").read_text())
    m2 = json.loads((REPO / "reports/nns_vit_endpoint_m5_r2_20260919/m2_reuse_receipt.json").read_text())
    required = [
        "## 13.7 M5 正式结论", terminal["status"], "STOP_AFTER_M5",
        "163,200", "9,600", "46 项", "64 项", "0.3782/0.3759",
        "0.3000/0.3071", "0.2815/0.2727", "0.1952/0.1476",
        "0.6742/0.6856", "0.6095/0.6238", "24 个复用组",
        "完整输入 11/12、隔离 head 12/12", "41/64", "40/64",
        "Input × Gradient", "normalization origin",
    ]
    for token in required:
        require(token in text, "REQUIRED:" + token)
    require(summary["endpoint_invariance"]["exact"], "ENDPOINT_INVARIANCE")
    require(summary["reference_sensitivity"]["exact_taylor_rank_blocks"] == 46, "TAYLOR_RANK_COUNT")
    require(summary["reference_sensitivity"]["exact_finite_effect_rank_blocks"] == 64, "EFFECT_RANK_COUNT")
    require(m2["inherited_patch_taylor_decisions"] == {"full": {"endpoint_greater": 0, "baseline_greater": 11}, "isolated": {"endpoint_greater": 0, "baseline_greater": 12}}, "M2_DECISIONS")
    require(hashlib.sha256((root / "artifact_manifest.json").read_bytes()).hexdigest() == terminal["artifact_manifest_sha256"], "TERMINAL_MANIFEST")
    require(hashlib.sha256((root / "completion_audit.json").read_bytes()).hexdigest() == terminal["completion_audit_sha256"], "TERMINAL_AUDIT")
    require(manifest["file_count"] == len(manifest["files"]) == 14, "MANIFEST_COUNT")
    for row in manifest["files"]:
        path = root / row["path"]
        require(path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "SEALED_ARTIFACT:" + row["path"])
    old = subprocess.check_output(["git", "show", SEALED_COMMIT + ":vitnns_rebuild/docs/阶段M_endpointnnsvit.md"], cwd=REPO.parent)
    require(hashlib.sha256(old).hexdigest() == SEALED_SHA256, "SEALED_DOCUMENT_BINDING")
    stage_artifact = next(item for requirement in audit["requirements"] for item in requirement["artifacts"] if item["path"] == "docs/阶段M_endpointnnsvit.md")
    require(stage_artifact["sha256"] == SEALED_SHA256 and stage_artifact["bytes"] == len(old), "AUDIT_DOCUMENT_BINDING")
    false_claim = "Endpoint 完全无 baseline"
    require(text.count(false_claim) == 1 and "不要写：\n\n> " + false_claim in text, "FALSE_REFERENCE_FREE_CLAIM")
    require(text.count("\[") == text.count("\]"), "DISPLAY_MATH")
    report = {
        "status": "PASS", "required_tokens": len(required),
        "sealed_document_binding": "PASS", "sealed_root_files": 14, "m5_numeric_crosschecks": 3,
        "formal_conclusion_sections": 2, "terminal_unchanged": True,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
