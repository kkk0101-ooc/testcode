#!/usr/bin/env python3
"""Assemble bilingual Markdown reading copies from authoritative section files."""

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


def assemble(language: str) -> None:
    chunks = [TITLES[language]]
    for name in SECTIONS:
        text = (ROOT / language / name).read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines.pop(0)
        chunks.append("\n".join(lines))
    output = "\n\n".join(chunks).rstrip() + "\n"
    (ROOT / language / "full_draft.md").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    for lang in ("zh", "en"):
        assemble(lang)
