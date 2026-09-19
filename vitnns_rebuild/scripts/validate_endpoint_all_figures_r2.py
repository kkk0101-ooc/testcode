#!/usr/bin/env python3
"""Validate and optionally seal the organized Endpoint figure package R2."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_all_figures_r2_20260919"
R1 = REPO / "reports/nns_vit_endpoint_all_figures_r1_20260919"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "canonical_digest": canonical(value)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_signed(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing signed JSON: {path.relative_to(ROOT)}")
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value); observed = unsigned.pop("canonical_digest", None)
    if canonical(unsigned) != observed:
        errors.append(f"canonical digest mismatch: {path.relative_to(ROOT)}")
    return value


def inventory() -> list[dict[str, Any]]:
    records = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in {"whole_root_inventory.json", "terminal_receipt.json", "validation_report.json"}:
            continue
        records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--seal", action="store_true"); args = parser.parse_args()
    errors: list[str] = []
    manifest = verify_signed(ROOT / "manifest.json", errors)
    source_inventory = verify_signed(ROOT / "source_inventory.json", errors)
    display = verify_signed(ROOT / "display_receipt.json", errors)
    array_receipt = verify_signed(ROOT / "source_arrays/display_receipt.json", errors)
    qc = verify_signed(ROOT / "manual_visual_qc.json", errors)

    if manifest.get("revision") != "ENDPOINT-ALL-FIGURES-R2": errors.append("wrong revision")
    if manifest.get("black_bear_binding") != {"sample_id": "ILSVRC2012_val_00005415.JPEG", "selection_order": 14, "class_id": 297, "wnid": "n02134418"}: errors.append("black-bear binding mismatch")
    if display.get("sample_id") != "ILSVRC2012_val_00005415.JPEG" or display.get("class_id") != 297: errors.append("display receipt sample mismatch")
    if array_receipt.get("sample_id") != "ILSVRC2012_val_00005415.JPEG" or array_receipt.get("class_id") != 297: errors.append("array receipt sample mismatch")

    rows = manifest.get("main_figures", [])
    if [row.get("number") for row in rows] != list(range(1, 41)): errors.append("main figure numbering is not exactly 1-40")
    if manifest.get("main_figure_count") != 40: errors.append("main figure count mismatch")
    formats_by_number: dict[int, set[str]] = {}
    for row in rows:
        number = int(row.get("number", -1)); outputs = row.get("outputs", []); formats_by_number[number] = {item.get("format") for item in outputs}
        if formats_by_number[number] != {"PNG", "PDF", "SVG"}: errors.append(f"format coverage mismatch: figure {number}")
        for item in outputs:
            path = ROOT / item.get("path", "")
            if not path.is_file() or path.is_symlink(): errors.append(f"invalid main output: {item.get('path')}"); continue
            if path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"): errors.append(f"main output identity mismatch: {item.get('path')}")

    for extension in ("png", "pdf", "svg"):
        paths = sorted((ROOT / f"figures_{extension}").glob(f"figure_*.{extension}"))
        numbers = [int(re.match(r"figure_(\d{2})_", path.name).group(1)) for path in paths if re.match(r"figure_(\d{2})_", path.name)]
        if numbers != list(range(1, 41)): errors.append(f"figures_{extension} inventory mismatch")
    if any(path.suffix.lower() in {".png", ".pdf", ".svg"} for path in ROOT.iterdir() if path.is_file()): errors.append("image exports remain in root")

    for item in source_inventory.get("sources", []):
        path = REPO / item.get("path", "")
        if not path.is_file() or path.is_symlink(): errors.append(f"invalid source: {item.get('path')}"); continue
        if path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"): errors.append(f"source drift: {item.get('path')}")
    sources = source_inventory.get("sources", [])
    if canonical(sources) != source_inventory.get("sources_digest"): errors.append("source-list digest mismatch")
    if source_inventory.get("torch_timm_model_forward_calls") != 0 or not source_inventory.get("cpu_only"): errors.append("CPU-only boundary mismatch")

    r1_manifest = R1 / "manifest.json"
    source_by_path = {row.get("path"): row for row in sources}
    r1_key = r1_manifest.relative_to(REPO).as_posix()
    if r1_key not in source_by_path or sha256(r1_manifest) != source_by_path[r1_key].get("sha256"): errors.append("R1 predecessor drift")

    signed_numbers = {int(match.group(1)) for path in (ROOT / "supplement/signed").glob("figure_*_signed.*") if (match := re.match(r"figure_(\d{2})_", path.name))}
    if signed_numbers != {12, 13, 17, 24, 34, 35, 36, 37, 38, 39, 40}: errors.append(f"signed companion coverage mismatch: {sorted(signed_numbers)}")
    for model in ("deit_s16", "vit_b16"):
        blocks = sorted((ROOT / "supplement/all_heads" / model).glob("block_*.png"))
        if [path.stem for path in blocks] != [f"block_{i:02d}" for i in range(12)]: errors.append(f"all-head coverage mismatch: {model}")

    for number in (1, 12, 13, 17, 24):
        if not any(row.get("figure") == number and row.get("view") == "positive_blue" for row in display.get("panels", [])): errors.append(f"missing blue display receipt: {number}")
    if len(display.get("panels", [])) < 41: errors.append("display receipt unexpectedly small")

    for path in sorted((ROOT / "figures_png").glob("*.png")):
        try:
            with Image.open(path) as image:
                image.verify()
                width, height = image.size
            if width < 1100 or height < 600: errors.append(f"undersized main PNG: {path.name} {width}x{height}")
        except Exception as exc: errors.append(f"invalid PNG {path.name}: {exc}")

    expected_docs = {"README.md", "figure_index.md", "figure_analysis.md", "captions_en.md", "captions_zh.md", "gallery.html"}
    for name in expected_docs:
        if not (ROOT / name).is_file(): errors.append(f"missing document: {name}")
    if args.seal and qc.get("status") != "PASS": errors.append("manual visual QC is not PASS")

    report = {
        "status": "PASS" if not errors else "FAIL", "sealed": bool(args.seal and not errors),
        "main_figures": len(rows), "main_exports": sum(len(row.get("outputs", [])) for row in rows),
        "supplement_records": len(manifest.get("supplemental_records", [])), "display_panel_records": len(display.get("panels", [])),
        "checks": ["canonical digests", "Figure 1-40 coverage", "PNG/PDF/SVG organization", "source hashes", "R1 predecessor binding", "black-bear binding", "signed companions", "all-head coverage", "PNG decode and minimum dimensions", "manual visual QC gate"],
        "errors": errors,
    }
    write_json(ROOT / "validation_report.json", signed(report))
    if args.seal and not errors:
        sealed_manifest = dict(manifest)
        sealed_manifest.pop("canonical_digest", None)
        sealed_manifest["status"] = "PASS_PRESENTATION_RENDER"
        write_json(ROOT / "manifest.json", signed(sealed_manifest))
        records = inventory(); write_json(ROOT / "whole_root_inventory.json", signed({"records": records, "inventory_digest": canonical(records)}))
        terminal = signed({"status": "PASS_PRESENTATION_RENDER", "revision": "ENDPOINT-ALL-FIGURES-R2", "main_figures": 40, "main_exports": 120, "manual_visual_qc": "PASS", "new_scientific_inference": False, "next_authorization": "PAPER_LAYOUT_OR_USER_REVIEW"})
        write_json(ROOT / "terminal_receipt.json", terminal)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
