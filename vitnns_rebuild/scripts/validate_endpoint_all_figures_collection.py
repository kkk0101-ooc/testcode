#!/usr/bin/env python3
"""Validate the flat Endpoint-NNSViT PNG collection and its source bindings."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_all_figures_r1_20260919"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> int:
    errors: list[str] = []
    manifest_path = ROOT / "manifest.json"
    if not manifest_path.is_file():
        print(json.dumps({"status": "FAIL", "errors": ["manifest missing"]}, indent=2))
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    observed_digest = unsigned.pop("canonical_digest", None)
    if canonical(unsigned) != observed_digest:
        errors.append("manifest canonical digest mismatch")

    records = manifest.get("records", [])
    names: list[str] = []
    digests: list[str] = []
    for record in records:
        name = record.get("name", "")
        names.append(name)
        target = ROOT / name
        source = REPO / record.get("source_path", "")
        if not target.is_file() or target.is_symlink():
            errors.append(f"invalid collection file: {name}")
            continue
        target_hash = sha256(target)
        digests.append(target_hash)
        if target_hash != record.get("sha256") or target.stat().st_size != record.get("bytes"):
            errors.append(f"collection identity mismatch: {name}")
        if not source.is_file() or source.is_symlink():
            errors.append(f"invalid source: {record.get('source_path')}")
        elif sha256(source) != record.get("source_sha256") or target_hash != sha256(source):
            errors.append(f"source binding mismatch: {name}")

    if len(names) != len(set(names)):
        errors.append("duplicate output names")
    if len(digests) != len(set(digests)):
        errors.append("duplicate image bytes")
    if manifest.get("image_count") != len(records):
        errors.append("image count mismatch")
    main_numbers = sorted(
        int(match.group(1)) for name in names
        if (match := re.fullmatch(r"fig_(\d{2})_.+\.png", name))
    )
    if main_numbers != list(range(1, 41)):
        errors.append("main Figure 1-40 coverage mismatch")
    if sum(name.startswith("signed_fig_") for name in names) != 7:
        errors.append("signed companion count mismatch")
    if sum(name.startswith("all_heads_") for name in names) != 24:
        errors.append("all-head page count mismatch")

    actual_files = {path.name for path in ROOT.iterdir() if path.is_file()}
    expected_files = set(names) | {"README.md", "gallery.html", "manifest.json"}
    if actual_files != expected_files:
        errors.append("flat directory inventory mismatch")
    if any(path.is_dir() for path in ROOT.iterdir()):
        errors.append("collection is not flat")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "image_count": len(records),
        "main_figure_count": len(main_numbers),
        "signed_companion_count": sum(name.startswith("signed_fig_") for name in names),
        "all_head_page_count": sum(name.startswith("all_heads_") for name in names),
        "supplement_count": sum(name.startswith("supp_") for name in names),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
