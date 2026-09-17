#!/usr/bin/env python3
"""Assemble bilingual Markdown reading copies from authoritative section files."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = (
    "abstract.md",
    "introduction.md",
    "related_work.md",
    "methods.md",
    "experimental.md",
    "results_discussion.md",
    "limitations.md",
    "conclusion.md",
)
TITLES = {
    "zh": "# 通过学习图像理解视觉 Transformer 的内部响应与贡献分配",
    "en": "# Understanding Vision Transformers through Learning Images of Internal Responses and Contributions",
}


def demote_headings(text: str) -> str:
    """Keep one document title by moving section headings down one level."""
    return re.sub(r"^(#{1,5})(?=\s)", r"\1#", text, flags=re.MULTILINE)


def assemble(language: str) -> None:
    chunks = [TITLES[language]]
    for name in SECTIONS:
        text = (ROOT / language / name).read_text(encoding="utf-8").strip()
        chunks.append(demote_headings(text))
    output = "\n\n".join(chunks).rstrip() + "\n"
    (ROOT / language / "full_draft.md").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    for lang in ("zh", "en"):
        assemble(lang)
