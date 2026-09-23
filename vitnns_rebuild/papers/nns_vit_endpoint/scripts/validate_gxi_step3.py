#!/usr/bin/env python3
"""Validate GxI positioning plan Step 3 against the active bilingual paper package."""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
PLAN = REPO / "Endpoint_GxI创新定位与后续五步计划.md"
REPORT = ROOT / "qc/gxi_step3_validation.json"
checks = []


def check(name, value, detail=""):
    checks.append({"name": name, "passed": bool(value), "detail": detail})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror(text, lang):
    return (
        text.replace("](../figures/", "](../publication_r2_20260920/figures/")
        .replace("](../references.md)", "](../publication_r2_20260920/references.md)")
        .replace("](supplement.md)", f"](../publication_r2_20260920/{lang}/supplement.md)")
        .replace("](supplement_experiments_v1.md)", f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)")
    )


def main():
    from validate_revision_v6 import contracts, mirror as current_mirror
    texts = {lang: (ROOT / lang / 'manuscript.md').read_text() for lang in ('zh', 'en')}
    current = contracts(texts)
    for lang, value in texts.items():
        current.append({'name': 'active_mirror:' + lang, 'passed': (PAPER / lang / 'full_draft.md').read_text() == current_mirror(value, lang)})
    errors = [row for row in current if not row['passed']]
    out = {'schema_version': 1, 'status': 'PASS' if not errors else 'FAIL', 'contract': 'V6-A1 supersedes obsolete NOV-in-main and title assertions; historical receipts immutable', 'checks': current, 'errors': errors}
    (ROOT / 'qc/revision_v6/gxi_step3_validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({'status': out['status'], 'checks': len(current), 'errors': errors}, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
