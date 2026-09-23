#!/usr/bin/env python3
"""CPU-only completion validator for GxI roadmap Step 5."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
REPORT = ROOT / "qc/gxi_step5_validation.json"
CHECKS: list[dict] = []


def check(name: str, passed: bool, detail="") -> None:
    CHECKS.append({"name": name, "passed": bool(passed), "detail": str(detail)})


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror(text: str, lang: str) -> str:
    return (text.replace("](../figures/", "](../publication_r2_20260920/figures/")
                .replace("](../references.md)", "](../publication_r2_20260920/references.md)")
                .replace("](supplement.md)", f"](../publication_r2_20260920/{lang}/supplement.md)")
                .replace("](supplement_experiments_v1.md)", f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)"))


def table_numbers(lines: list[str]) -> list[list[str]]:
    number = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z])")
    return [number.findall(line) for line in lines if line.startswith("|")]


def validate_markdown(path: Path) -> None:
    text = path.read_text()
    label = str(path.relative_to(ROOT))
    check("markdown_fences:" + label, text.count("```") % 2 == 0)
    check("latex_blocks:" + label, text.count("$$") % 2 == 0)
    equations = re.findall(r"\$\$(.*?)\$\$", text, re.S)
    check("latex_braces:" + label, all(eq.count("{") == eq.count("}") for eq in equations))
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http", "#")):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        check("link:" + label + ":" + target, resolved.is_file() and (resolved == REPO.resolve() or REPO.resolve() in resolved.parents))


def main() -> int:
    from revision_v6_publication_checks import final_checks
    from validate_revision_v6 import contracts, mirror as current_mirror
    texts = {lang: (ROOT / lang / 'manuscript.md').read_text() for lang in ('zh', 'en')}
    current = contracts(texts) + final_checks(texts)
    for lang, text in texts.items():
        current.append({'name': 'entry:' + lang, 'passed': (PAPER / lang / 'full_draft.md').read_text() == current_mirror(text, lang)})
    errors = [r for r in current if not r['passed']]
    out = {'schema_version': 1, 'status': 'PASS' if not errors else 'FAIL', 'contract': 'V6-A1; active draft and new review, historical Step5 receipts retained', 'checks': current, 'errors': errors}
    (ROOT / 'qc/revision_v6/gxi_step5_validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({'status': out['status'], 'checks': len(current), 'errors': errors}, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
