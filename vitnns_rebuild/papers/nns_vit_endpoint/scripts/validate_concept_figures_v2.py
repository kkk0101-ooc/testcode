#!/usr/bin/env python3
"""Validate the concept-figure v2 redesign for package Figures 1--7."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "papers/nns_vit_endpoint/publication_r2_20260920"
ACTIVE = ROOT.parent / "figure_manifest.json"
SCRIPT = REPO / "papers/nns_vit_endpoint/scripts/render_concept_figures_v2.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    manifest = load(ROOT / "figure_manifest.json")
    active = load(ACTIVE)
    display = load(ROOT / "evidence/display_receipt.json")
    sources = load(ROOT / "evidence/figure_sources.json")
    review = load(ROOT / "qc/revision_v6/visual_review.json")
    texts = {
        "zh": {
            "main": (ROOT / "zh/manuscript.md").read_text(),
            "supplement": (ROOT / "zh/supplement.md").read_text(),
        },
        "en": {
            "main": (ROOT / "en/manuscript.md").read_text(),
            "supplement": (ROOT / "en/supplement.md").read_text(),
        },
    }
    figure_analysis = (ROOT / "figure_analysis.md").read_text()

    check("package_revision", manifest.get("revision") == "PUBLICATION-R2-CONCEPT-V2")
    check("active_revision", active.get("source_revision") == "PUBLICATION-R2-CONCEPT-V2")
    check("script_bound", sources.get(str(SCRIPT.relative_to(REPO))) == sha(SCRIPT))
    check("review_status", review.get("status") == "PASS_VISUAL_REVIEW_CONCEPT_V2")
    check("review_scope", review.get("reviewed_current_package_figures") == list(range(1, 8)))
    check("display_scope", sorted({row["figure"] for row in display["panels"] if 1 <= row["figure"] <= 7}) == [1, 6, 7])

    required_titles = {
        1: "One image, one target, many auditable local maps",
        2: "Use identity and valid cuts make the decomposition auditable",
        3: "Initialization keeps both input value and location",
        4: "Local Jacobians propagate coordinate maps to the target",
        5: "Patch embedding begins with strictly local support",
        6: "Attention separates routing from value transmission",
        7: "Residual closure is visible in the path maps",
    }
    active_by_number = {row["figure"]: row for row in active["figures"]}
    for number, title in required_titles.items():
        row = next(item for item in manifest["figures"] if item["number"] == number)
        check(f"title:{number}", row["title"] == title, row["title"])
        if number in (1, 2, 6, 7):
            check(f"caption_zh:{number}", row["caption_zh"] in texts["zh"]["main"])
            check(f"caption_en:{number}", row["caption_en"] in texts["en"]["main"])
        else:
            check(f"analysis_title:{number}", row["title"] in figure_analysis)
            check(f"supplement_link_zh:{number}", row["files"]["png"]["path"] in texts["zh"]["supplement"])
            check(f"supplement_link_en:{number}", row["files"]["png"]["path"] in texts["en"]["supplement"])
        active_row = active_by_number[number]
        for extension, record in row["files"].items():
            path = ROOT / record["path"]
            digest = sha(path)
            check(f"file_exists:{number}:{extension}", path.is_file())
            check(f"manifest_hash:{number}:{extension}", digest == record["sha256"])
            check(f"active_hash:{number}:{extension}", digest == active_row[extension]["sha256"])

    phrases = {
        1: ["sloth bear", "Total Endpoint", "Closure does not imply"],
        2: ["branch partition", "global cut", "target s"],
        6: ["Routing-induced", "Value-induced", "Add residual identity only for Total Endpoint"],
    }
    for number, expected in phrases.items():
        svg_text = (ROOT / next(item for item in manifest["figures"] if item["number"] == number)["files"]["svg"]["path"]).read_text()
        for phrase in expected:
            check(f"figure{number}_phrase:{phrase}", phrase in svg_text)

    errors = [row for row in checks if not row["passed"]]
    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "check_count": len(checks),
        "errors": errors,
        "checks": checks,
    }
    out = ROOT / "qc/revision_v6/concept_v2_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
