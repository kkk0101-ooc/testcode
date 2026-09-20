#!/usr/bin/env python3
"""Audit every page of the generic anonymous A4 review PDF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[1]
PDF = PACKAGE / "NNS_ViT_学位论文_匿名内容审阅版.pdf"
OUT = PACKAGE / "qc" / "page_audit.json"
SHEET = PACKAGE / "qc" / "page_contact_sheet.png"
EXPECTED_CHAPTERS = [f"第 {i} 章" for i in range(1, 9)]
FORBIDDEN = ["AUTHOR REQUIRED", "AFFILIATION REQUIRED", "CORRESPONDING AUTHOR", "/data01", "yeziqi", "ByteDance", "字节"]


def main() -> int:
    doc = pymupdf.open(PDF)
    page_details = []
    thumbnails = []
    errors = []
    all_text = []
    image_count = 0
    for index, page in enumerate(doc):
        text = page.get_text()
        all_text.append(text)
        images = len(page.get_images(full=True))
        image_count += images
        rect = page.rect
        a4_ok = abs(rect.width - 595.276) < 1.0 and abs(rect.height - 841.89) < 1.0
        blocks = page.get_text("blocks")
        overflow = [b[:4] for b in blocks if b[0] < -1 or b[1] < -1 or b[2] > rect.width + 1 or b[3] > rect.height + 1]
        blank = len(text.strip()) < 200 and images == 0 and index != 0
        if not a4_ok:
            errors.append(f"page {index + 1}: not A4")
        if overflow:
            errors.append(f"page {index + 1}: text overflow")
        if blank:
            errors.append(f"page {index + 1}: blank or nearly blank")
        key = next((x.strip() for x in text.splitlines() if x.strip() and "匿名内容审阅版" not in x and not x.strip().startswith("第 ") or x.strip().startswith("第 1 章")), "")
        page_details.append({
            "page": index + 1,
            "text_characters": len(text.strip()),
            "embedded_images": images,
            "a4": a4_ok,
            "text_overflow_blocks": len(overflow),
            "blank": blank,
            "leading_text": key[:100],
        })
        pix = page.get_pixmap(matrix=pymupdf.Matrix(0.55, 0.55), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        canvas = Image.new("RGB", (image.width + 8, image.height + 28), "white")
        canvas.paste(image, (4, 22))
        ImageDraw.Draw(canvas).text((6, 4), f"Page {index + 1}", fill="black")
        thumbnails.append(canvas)

    text = "\n".join(all_text)
    missing_chapters = [x for x in EXPECTED_CHAPTERS if x not in text]
    if missing_chapters:
        errors.append(f"missing chapter titles: {missing_chapters}")
    if "[1] Dosovitskiy" not in text or "[37] Holm" not in text:
        errors.append("bibliography endpoints missing")
    if image_count != 25:
        errors.append(f"expected 25 embedded images (7 figures + 18 equations), got {image_count}")
    metadata_text = json.dumps(doc.metadata, ensure_ascii=False)
    identity_hits = [x for x in FORBIDDEN if x.lower() in (text + metadata_text).lower()]
    if identity_hits:
        errors.append(f"blind-review identity hits: {identity_hits}")
    if doc.metadata.get("author"):
        errors.append("PDF author metadata is not empty")

    cols = 5
    width = max(x.width for x in thumbnails)
    height = max(x.height for x in thumbnails)
    rows = (len(thumbnails) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), "#cccccc")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % cols) * width, (index // cols) * height))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET)

    result = {
        "status": "PASS" if not errors else "FAIL",
        "pdf": str(PDF.relative_to(PACKAGE)),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pdf_bytes": PDF.stat().st_size,
        "page_count": doc.page_count,
        "embedded_image_count": image_count,
        "expected_images": {"scientific_figures": 7, "numbered_equation_images": 18},
        "metadata": doc.metadata,
        "identity_hits": identity_hits,
        "missing_chapters": missing_chapters,
        "errors": errors,
        "pages": page_details,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "page_count": result["page_count"], "embedded_image_count": image_count, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
