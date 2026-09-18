#!/usr/bin/env python3
"""Validate structure, figure bindings, placeholders, and key bilingual facts."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
SECTIONS = (
    "outline.md", "abstract.md", "introduction.md", "related_work.md",
    "methods.md", "experimental.md", "results_discussion.md",
    "limitations.md", "conclusion.md", "full_draft.md",
)
EXPECTED_FACTS = (
    "500/500", "2.3082", "2.1111", "2.4452", "1.7535",
    "0.0734", "0.0389", "0.0562", "2.2955", "2.3395",
    "-0.05585", "0.03943", "0.1391", "0.2492", "0.1137",
    "0.2177", "0.3378", "0.06260", "0.07494", "0.03631", "0.04615",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


errors = []
for lang in ("zh", "en"):
    for name in SECTIONS:
        path = ROOT / lang / name
        if not path.is_file():
            errors.append(f"missing section: {path.relative_to(ROOT)}")
    full = (ROOT / lang / "full_draft.md").read_text(encoding="utf-8")
    for fact in EXPECTED_FACTS:
        if fact not in full:
            errors.append(f"{lang}/full_draft.md missing key fact {fact}")
    if full.count("```") % 2:
        errors.append(f"unbalanced code fences in {lang}/full_draft.md")
    if full.count("$$") % 2:
        errors.append(f"unbalanced display math in {lang}/full_draft.md")
    refs = sorted({int(x) for x in re.findall(r"Figure(?:s)?\s+(\d+)", full)})
    if any(x < 1 or x > 35 for x in refs):
        errors.append(f"out-of-range figure citation in {lang}/full_draft.md: {refs}")

manifest = json.loads((ROOT / "figure_manifest.json").read_text(encoding="utf-8"))
if manifest.get("figure_count") != 35:
    errors.append("figure manifest count is not 35")
for figure in manifest.get("figures", []):
    for kind in ("png", "pdf", "svg"):
        record = figure[kind]
        path = REPO / record["path"]
        if not path.is_file():
            errors.append(f"missing figure asset: {record['path']}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"figure digest mismatch: {record['path']}")
for record in manifest.get("support_files", []):
    path = REPO / record["path"]
    if not path.is_file():
        errors.append(f"missing support file: {record['path']}")
    elif sha256(path) != record["sha256"]:
        errors.append(f"support-file digest mismatch: {record['path']}")

for path in ROOT.rglob("*.md"):
    data = path.read_bytes()
    if any(byte < 32 and byte not in (10, 13) for byte in data):
        errors.append(f"control character: {path.relative_to(ROOT)}")

report = {
    "status": "PASS" if not errors else "FAIL",
    "errors": errors,
    "figure_count": manifest.get("figure_count"),
    "section_count_per_language": len(SECTIONS),
    "known_placeholders": [
        "AUTHOR REQUIRED", "AFFILIATION REQUIRED", "REFERENCE REQUIRED"
    ],
}
(ROOT / "qc/validation_report.json").parent.mkdir(parents=True, exist_ok=True)
(ROOT / "qc/validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if not errors else 1)
