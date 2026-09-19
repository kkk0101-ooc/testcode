#!/usr/bin/env python3
"""Build a flat, hash-bound PNG collection for the Endpoint-NNSViT paper."""
from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = REPO / "reports/nns_vit_endpoint_all_figures_r1_20260919"
BASE_MANIFEST = PAPER / "figure_manifest.json"
BLUE_MANIFEST = PAPER / "figure_manifest_fine_grained_blue_r2.json"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_record(records: list[dict[str, Any]], *, name: str, source: str, group: str, title: str) -> None:
    source_path = REPO / source
    if not source_path.is_file():
        raise FileNotFoundError(source)
    target = ROOT / name
    if target.exists():
        raise FileExistsError(name)
    shutil.copyfile(source_path, target)
    source_hash = sha256(source_path)
    target_hash = sha256(target)
    if source_hash != target_hash:
        raise RuntimeError(f"COPY_HASH_MISMATCH:{name}")
    records.append({
        "name": name,
        "group": group,
        "title": title,
        "source_path": source,
        "source_sha256": source_hash,
        "bytes": target.stat().st_size,
        "sha256": target_hash,
    })


def unique_pngs(manifest_name: str) -> list[tuple[str, str]]:
    manifest = read_json(PAPER / manifest_name)
    seen: set[str] = set()
    outputs: list[tuple[str, str]] = []
    for row in manifest.get("figures", []):
        candidates: list[tuple[str, str]] = []
        if row.get("path", "").endswith(".png"):
            candidates.append((str(row.get("id", row.get("figure_id", "figure"))), row["path"]))
        if row.get("format") == "png":
            candidates.append((str(row.get("figure_id", "figure")), row.get("report_path", row.get("paper_path"))))
        for record in row.get("files", []):
            path = record.get("path", record.get("report_path", record.get("paper_path", "")))
            if record.get("format", "").lower() == "png" or path.endswith(".png"):
                candidates.append((str(row.get("id", row.get("figure_id", "figure"))), path))
        for identifier, path in candidates:
            if not path:
                continue
            digest = sha256(REPO / path)
            if digest in seen:
                continue
            seen.add(digest)
            outputs.append((identifier, path))
    return outputs


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"collection root already exists: {ROOT}")
    ROOT.mkdir(parents=True)
    records: list[dict[str, Any]] = []

    base = read_json(BASE_MANIFEST)
    for row in base["figures"]:
        number = int(row["figure"])
        copy_record(
            records,
            name=f"fig_{number:02d}_{Path(row['png']['path']).name.removeprefix(f'figure_{number:02d}_')}",
            source=row["png"]["path"],
            group="paper_main_1_28",
            title=row["title"],
        )

    m3_sup = read_json(PAPER / "figure_manifest_m3_sup_r2.json")
    for row in m3_sup["figures"]:
        number = int(row["paper_number"])
        source = next(record["path"] for record in row["files"] if record["path"].endswith(".png"))
        copy_record(
            records, name=f"fig_{number:02d}_{row['id']}.png", source=source,
            group="paper_main_29_33", title=row["description"],
        )

    blue = read_json(BLUE_MANIFEST)
    for row in blue["figures"]:
        number = int(row["paper_number"])
        copy_record(
            records,
            name=f"fig_{number:02d}_{Path(row['png']['path']).name.removeprefix(f'figure_{number:02d}_')}",
            source=row["png"]["path"], group="paper_main_34_40", title=row["title"],
        )
        copy_record(
            records, name=f"signed_fig_{number:02d}_{Path(row['signed_companion']['path']).name.removeprefix(f'figure_{number:02d}_')}",
            source=row["signed_companion"]["path"], group="signed_companions",
            title=row["title"] + " - signed companion",
        )

    for row in blue["supplemental_all_head_pages"]:
        copy_record(
            records, name=f"all_heads_{row['model_id']}_block_{int(row['block']):02d}.png",
            source=row["path"], group="all_head_pages",
            title=f"{row['model_id']} block {int(row['block']):02d} all heads",
        )

    supplements = (
        ("figure_manifest_m2.json", "supp_m2"),
        ("figure_manifest_m3.json", "supp_m3"),
        ("figure_manifest_m4.json", "supp_m4"),
        ("figure_manifest_m4_review_r2.json", "supp_m4_review"),
        ("figure_manifest_m6.json", "supp_m6"),
    )
    for manifest_name, prefix in supplements:
        for index, (identifier, source) in enumerate(unique_pngs(manifest_name), 1):
            copy_record(
                records, name=f"{prefix}_{index:02d}_{identifier}.png", source=source,
                group=prefix, title=identifier,
            )

    names = [record["name"] for record in records]
    digests = [record["sha256"] for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("DUPLICATE_OUTPUT_NAME")
    if len(digests) != len(set(digests)):
        raise RuntimeError("DUPLICATE_IMAGE_BYTES_AFTER_DEDUP")

    manifest_base = {
        "schema_version": 1,
        "document_type": "endpoint_nnsvit_flat_all_figure_collection",
        "status": "PASS",
        "image_count": len(records),
        "flat_directory": True,
        "formats": ["png"],
        "main_figure_numbers": list(range(1, 41)),
        "fine_grained_source_revision": "ENDPOINT-FINE-GRAINED-BLUE-R2",
        "display_sample": blue.get("display_sample"),
        "deduplication": "exact SHA-256; identical PNG bytes retained once",
        "records": records,
    }
    manifest = {**manifest_base, "canonical_digest": canonical(manifest_base)}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    group_order = [
        "paper_main_1_28", "paper_main_29_33", "paper_main_34_40",
        "signed_companions", "all_head_pages", "supp_m2", "supp_m3",
        "supp_m4", "supp_m4_review", "supp_m6",
    ]
    index = [
        "# Endpoint-NNSViT 全图片汇总", "",
        f"共 `{len(records)}` 张去重 PNG，全部平铺在当前目录。", "",
        "- `fig_01_...` 到 `fig_40_...`：现行论文 Figure 1–40；",
        "- `signed_fig_...`：Figure 34–40 的完整 signed companion；",
        "- `all_heads_...`：DeiT-S/16 与 ViT-B/16 各 12 层的全 head 页；",
        "- `supp_...`：M2、M3、M4、M4 review 与 M6 补充图。", "",
        "图片来自原权威目录的字节级复制；`manifest.json` 记录原路径和 SHA-256。", "",
    ]
    for group in group_order:
        rows = [record for record in records if record["group"] == group]
        if not rows:
            continue
        index.extend([f"## {group}", ""])
        index.extend(f"- [{row['name']}]({row['name']}) — {row['title']}" for row in rows)
        index.append("")
    (ROOT / "README.md").write_text("\n".join(index).rstrip() + "\n")

    cards = []
    for group in group_order:
        rows = [record for record in records if record["group"] == group]
        if not rows:
            continue
        cards.append(f"<h2>{html.escape(group)}</h2>")
        cards.extend(
            f'<article><a href="{html.escape(row["name"])}"><img loading="lazy" src="{html.escape(row["name"])}"></a>'
            f'<p><b>{html.escape(row["name"])}</b><br>{html.escape(row["title"])}</p></article>'
            for row in rows
        )
    gallery = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Endpoint-NNSViT all figures</title><style>'
        'body{font:15px/1.5 system-ui;color:#172033;background:#eef3f6;margin:20px}'
        'h1,h2{clear:both}article{background:white;padding:10px;margin:8px;width:30%;display:inline-block;vertical-align:top}'
        'img{width:100%;height:240px;object-fit:contain}p{overflow-wrap:anywhere}</style>'
        f'<h1>Endpoint-NNSViT 全图片汇总 · {len(records)} PNG</h1>' + "".join(cards) + '</html>'
    )
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")

    print(json.dumps({
        "status": "PASS", "root": ROOT.relative_to(REPO).as_posix(),
        "image_count": len(records), "manifest_sha256": sha256(ROOT / "manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
