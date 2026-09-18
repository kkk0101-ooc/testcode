#!/usr/bin/env python3
"""Build the manuscript figure binding without modifying the frozen package."""

import hashlib
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = ROOT / "reports/nns_vit_unified_paper_figures_r10_r4_20260917"
INDEX = FIGURE_ROOT / "figure_index.md"
ROW = re.compile(
    r"^\| (?P<number>\d+) \| (?P<title>.*?) \| \[PNG\]\((?P<png>[^)]+)\) \| \[PDF\]\((?P<pdf>[^)]+)\) \| \[SVG\]\((?P<svg>[^)]+)\) \|$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


figures = []
for line in INDEX.read_text(encoding="utf-8").splitlines():
    match = ROW.match(line)
    if not match:
        continue
    item = {"figure": int(match.group("number")), "title": match.group("title")}
    for kind in ("png", "pdf", "svg"):
        path = FIGURE_ROOT / match.group(kind)
        item[kind] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    figures.append(item)

if [item["figure"] for item in figures] != list(range(1, 36)):
    raise SystemExit("figure index is not the complete ordered range 1..35")

receipt = FIGURE_ROOT / "terminal_receipt.json"
support_names = (
    "figure_index.md",
    "captions_en.md",
    "captions_zh.md",
    "figure_analysis.md",
    "render_manifest.json",
    "terminal_receipt.json",
)
payload = {
    "schema_version": 1,
    "source_revision": "R10-R4",
    "source_root": str(FIGURE_ROOT.relative_to(ROOT)),
    "source_terminal_receipt": {
        "path": str(receipt.relative_to(ROOT)),
        "sha256": sha256(receipt),
    },
    "support_files": [
        {
            "path": str((FIGURE_ROOT / name).relative_to(ROOT)),
            "sha256": sha256(FIGURE_ROOT / name),
            "bytes": (FIGURE_ROOT / name).stat().st_size,
        }
        for name in support_names
    ],
    "figure_count": len(figures),
    "figures": figures,
}
(PAPER / "figure_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
