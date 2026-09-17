#!/usr/bin/env python3
"""Validate standalone Markdown and bilingual PDF publication files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

deps = os.environ.get("NNS_PDF_DEPS")
if deps:
    sys.path.insert(0, deps)

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript"
OUTPUT = MANUSCRIPT / "output"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


errors = []
markdown = MANUSCRIPT / "NNS_ViT_论文初稿_中文版.md"
md_text = markdown.read_text(encoding="utf-8")
for forbidden in ("500/500", "[2.1111, 2.4452]", "block 0/head 9", "14x14 patch"):
    abstract = md_text.split("## 1. 引言", 1)[0]
    if forbidden in abstract:
        errors.append(f"forbidden abstract detail: {forbidden}")
if len(re.findall(r"^### 图 \d+\.", md_text, flags=re.MULTILINE)) != 35:
    errors.append("standalone Chinese Markdown does not contain 35 localized figure titles")
if md_text.count("![Figure " ) != 35:
    errors.append("standalone Chinese Markdown does not contain 35 figure links")

specs = {
    "NNS_ViT_论文初稿_中文版.pdf": {
        "min_pages": 45,
        "needles": ["通过学习图像理解视觉Transformer", "视觉Transformer以图像patch为基本输入单元", "结论与未来工作", "图35.正确类、竞争类"],
    },
    "NNS_ViT_manuscript_draft_en.pdf": {
        "min_pages": 45,
        "needles": ["UnderstandingVisionTransformers", "VisionTransformersprocessimagepatches", "ConclusionsandFutureWork", "Figure35.Classifieroutputs"],
    },
}
pdf_records = []
for name, spec in specs.items():
    path = OUTPUT / name
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    packed = compact(text)
    images = sum(len(page.images) for page in reader.pages)
    checks = {needle: compact(needle) in packed for needle in spec["needles"]}
    if len(reader.pages) < spec["min_pages"]:
        errors.append(f"{name}: too few pages")
    if images < 35:
        errors.append(f"{name}: fewer than 35 image objects")
    if not all(checks.values()):
        errors.append(f"{name}: missing searchable text {checks}")
    pdf_records.append({
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "pages": len(reader.pages),
        "image_objects": images,
        "text_characters": len(text),
        "checks": checks,
    })

report = {
    "status": "PASS" if not errors else "FAIL",
    "errors": errors,
    "standalone_chinese_markdown": {
        "path": str(markdown.relative_to(ROOT)),
        "sha256": sha256(markdown),
        "bytes": markdown.stat().st_size,
        "localized_figure_titles": 35,
    },
    "pdfs": pdf_records,
}
(OUTPUT / "pdf_validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if not errors else 1)
