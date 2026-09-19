#!/usr/bin/env python3
"""Validate post-seal M6 conclusion integration in the Stage M document."""
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/阶段M_endpointnnsvit.md"


def require(condition, code):
    if not condition: raise ValueError(code)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"; text = DOC.read_text(); receipt = json.loads((root / "terminal_receipt.json").read_text())
    for token in ("M6 已完成", "ENDPOINT_M6_PASS_EXECUTION_AND_AUDIT", "12 类案例", "5 张主图", "STOP_AFTER_M6", "local input dependence"):
        require(token in text, "REQUIRED:" + token)
    require(receipt["status"] == "ENDPOINT_M6_PASS_EXECUTION_AND_AUDIT" and receipt["case_count"] == 12 and receipt["figure_count"] == 5, "TERMINAL")
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    require(sha(root / "artifact_manifest.json") == receipt["artifact_manifest_sha256"], "MANIFEST_HASH")
    require(sha(root / "completion_audit.json") == receipt["completion_audit_sha256"], "AUDIT_HASH")
    for row in manifest["files"]:
        path = root / row["path"]; require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "ARTIFACT:" + row["path"])
    pre = subprocess.check_output(["git", "show", "00efba30e:vitnns_rebuild/docs/阶段M_endpointnnsvit.md"], cwd=REPO.parent)
    require(hashlib.sha256(pre).hexdigest() == "aab722cfa658f585be6c70a321a5c2bedf12d8f72f2d37d4f21a3eb54947b3f3", "PRE_M6_DOC")
    print(json.dumps({"status": "PASS", "sealed_root_files": manifest["file_count"], "cases": 12, "figures": 5, "terminal_unchanged": True}))


if __name__ == "__main__": main()
