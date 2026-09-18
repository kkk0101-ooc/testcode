#!/usr/bin/env python3
"""Validate the bilingual Endpoint Scanner R1 draft against sealed evidence."""

import hashlib
import json
import re
from pathlib import Path


MANUSCRIPT = Path(__file__).resolve().parents[1]
REPO = MANUSCRIPT.parent
ZH = MANUSCRIPT / "zh/endpoint_scanner_full_draft_r1.md"
EN = MANUSCRIPT / "en/endpoint_scanner_full_draft_r1.md"
OUT = MANUSCRIPT / "qc/endpoint_scanner_draft_validation_report_r1.json"


def load_json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checks = []


def check(name: str, condition: bool, detail: str = ""):
    checks.append({"check": name, "pass": bool(condition), "detail": detail})


zh = ZH.read_text(encoding="utf-8")
en = EN.read_text(encoding="utf-8")

required_sections = [
    ("摘要", "Abstract"),
    ("1. 引言", "1. Introduction"),
    ("2. 相关工作", "2. Related Work"),
    ("3. 方法", "3. Method"),
    ("4. 实验设计", "4. Experimental Design"),
    ("5. 结果", "5. Results"),
    ("6. 讨论", "6. Discussion"),
    ("7. 局限", "7. Limitations"),
    ("8. 结论", "8. Conclusion"),
]
for zh_name, en_name in required_sections:
    check(f"section:{zh_name}|{en_name}", f"## {zh_name}" in zh and f"## {en_name}" in en)

check("balanced_display_math:zh", zh.count("$$") % 2 == 0, str(zh.count("$$")))
check("balanced_display_math:en", en.count("$$") % 2 == 0, str(en.count("$$")))
check("no_literal_patch_escape:zh", "\\n+" not in zh)
check("no_literal_patch_escape:en", "\\n+" not in en)

for table in range(1, 6):
    check(f"table:{table}:zh", f"**Table {table}." in zh)
    check(f"table:{table}:en", f"**Table {table}." in en)

e2 = load_json("reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json")
check("e2:task_count", e2["image_model_task_count"] == 800)
check("e2:target_count", e2["target_evaluation_count"] == 336800)

e4 = load_json("reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json")
check("e4:jvp_count", e4["local_component_evaluation_count"] == 6144)

families = load_json("reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json")
c4a = [item for item in families["families"] if item["claim"] == "C4a"]
c4b = [item for item in families["families"] if item["claim"] == "C4b"]
check("e6:c4a_family_count", len(c4a) == 12, str(len(c4a)))
check("e6:c4b_family_count", len(c4b) == 4, str(len(c4b)))
check("e6:c4a_supported_count", sum(x["status"] == "SUPPORTED" for x in c4a) == 11)
check("e6:c4b_supported_count", sum(x["status"] == "SUPPORTED" for x in c4b) == 0)

boundary = next(x for x in c4a if x["family_id"] == "C4a.vit_b16.normalized_zero.routing_value")
boundary_test = boundary["test_results"][0]
check("e6:boundary_status", boundary["status"] == "NOT_SUPPORTED")
check("e6:boundary_estimate", boundary_test["estimate"] == 0.09956521739130435)
check("e6:boundary_holm", boundary_test["holm_adjusted_p_value"] == 3.0036026037128944e-14)

model_labels = {"vit_b16": "ViT-B/16", "deit_s16": "DeiT-S/16"}
component_labels = {
    "heads": "heads",
    "routing_value": "routing/value",
    "residual_branches": "residual branches",
}
reference_labels = {"normalized_zero": ("normalized-zero", "normalized zero"), "blur_rgb": ("blur", "blur")}

for family in c4a:
    result = family["test_results"][0]
    common = (
        f"| {model_labels[family['model_id']]} | {{reference}} | "
        f"{component_labels[family['component_family']]} | {result['estimate']:.4f} | "
        f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] | "
        f"{result['holm_adjusted_p_value']:.2e} | {family['status']} |"
    )
    zh_row = common.format(reference=reference_labels[family["reference_id"]][0])
    en_row = common.format(reference=reference_labels[family["reference_id"]][1])
    check(f"table4:zh:{family['family_id']}", zh_row in zh, zh_row)
    check(f"table4:en:{family['family_id']}", en_row in en, en_row)

for family in c4b:
    estimates = {item["baseline"]: item["estimate"] for item in family["test_results"]}

    def display(value: float) -> str:
        return f"{value:.4f}".replace("-", "−")

    common = (
        f"| {model_labels[family['model_id']]} | {{reference}} | "
        f"{display(estimates['total_gxi'])} | {display(estimates['raw_gradient'])} | "
        f"{display(estimates['regional_taylor'])} | {display(estimates['random_regions'])} | "
        f"{family['status']} |"
    )
    zh_row = common.format(reference=reference_labels[family["reference_id"]][0])
    en_row = common.format(reference=reference_labels[family["reference_id"]][1])
    check(f"table5:zh:{family['family_id']}", zh_row in zh, zh_row)
    check(f"table5:en:{family['family_id']}", en_row in en, en_row)

for language, text in (("zh", zh), ("en", en)):
    for value in (
        "336,800", "76,800", "28,800", "6,144", "0.09956521739130435",
        "0.000435", "1,752.71", "1.9475", "192.16", "9.290",
    ):
        check(f"headline:{language}:{value}", value in text)
    check(f"rq3_omnibus:{language}", "C4a" in text and "NOT_SUPPORTED" in text)
    check(f"rq4:{language}", "RQ4" in text and "NOT_SUPPORTED" in text)

check("rq3_synthesis:zh", "整体支持，并存在一个注册边界例外" in zh)
check("rq3_synthesis:en", "supported overall with one registered boundary exception" in en)

manifest = load_json("manuscript/endpoint_figure_manifest.json")
check("figures:count", manifest.get("figure_count") == 28)
for figure in manifest.get("figures", []):
    for kind in ("png", "pdf", "svg"):
        record = figure[kind]
        path = REPO / record["path"]
        check(f"figure:{figure['figure']}:{kind}:exists", path.is_file(), record["path"])
        if path.is_file():
            check(
                f"figure:{figure['figure']}:{kind}:sha256",
                sha256(path) == record["sha256"],
                record["path"],
            )

for path in (ZH, EN):
    data = path.read_bytes()
    check(
        f"control_characters:{path.name}",
        not any(byte < 32 and byte not in (10, 13) for byte in data),
    )
    figure_refs = [int(value) for value in re.findall(r"Figure(?:s)?\s+(\d+)", path.read_text(encoding="utf-8"))]
    check(
        f"figure_reference_range:{path.name}",
        all(1 <= value <= 28 for value in figure_refs),
        str(sorted(set(figure_refs))),
    )

errors = [item for item in checks if not item["pass"]]
report = {
    "status": "PASS" if not errors else "FAIL",
    "check_count": len(checks),
    "error_count": len(errors),
    "errors": errors,
    "source_revision": manifest.get("source_revision"),
    "figure_count": manifest.get("figure_count"),
    "c4a_supported_families": sum(x["status"] == "SUPPORTED" for x in c4a),
    "c4b_supported_families": sum(x["status"] == "SUPPORTED" for x in c4b),
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if not errors else 1)
