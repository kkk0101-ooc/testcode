#!/usr/bin/env python3
"""Create versioned M5 manuscripts without mutating the sealed M4-review drafts."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "papers/nns_vit_endpoint"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def integrate(language, heading, marker):
    source = PAPER / language / "full_draft_m4_review_r2.md"
    supplement = PAPER / language / "endpoint_m5_positioning.md"
    destination = PAPER / language / "full_draft_m5.md"
    text = source.read_text()
    section = supplement.read_text().splitlines()
    require(section and section[0].startswith("# "), "SUPPLEMENT_HEADER:" + language)
    body = "\n".join(section[1:]).strip()
    block = f"<!-- M5_POSITIONING_START -->\n{heading}\n\n{body}\n<!-- M5_POSITIONING_END -->\n\n"
    require(text.count(marker) == 1 and "M5_POSITIONING_START" not in text, "INSERTION_POINT:" + language)
    destination.write_text(text.replace(marker, block + marker))


def main():
    integrate("en", "### 6.7 Method positioning and reference conditions", "## 7. Limitations")
    integrate("zh", "### 6.7 方法定位与 reference 信息条件", "## 7. 局限")


if __name__ == "__main__":
    main()
