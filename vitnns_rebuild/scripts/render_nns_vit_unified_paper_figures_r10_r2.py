#!/usr/bin/env python3
"""Assemble R10-R2 and add the four-target classifier comparison figure."""
from __future__ import annotations

import html
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-unified-paper-r10-r2")
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import render_nns_vit_unified_paper_figures_r7 as graphics
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r2_20260917"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r10_r1_20260917"
TITLE = "Classifier outputs: correct, competitor, margin, and low-score target"
SLUG = "classifier_output_comparison"
SOURCES: dict[str, dict] = {}


def register(path: Path, role: str = "source") -> None:
    path = path.resolve()
    relative = os.path.relpath(path, REPO)
    SOURCES[relative] = {
        "path": relative, "bytes": path.stat().st_size,
        "sha256": common.sha(path), "role": role,
    }


def checked(path: Path) -> dict:
    register(path)
    value = common.read(path)
    if not common.verify(value):
        raise ValueError(f"Bad canonical digest: {path}")
    return value


def copy_tree(source: Path, target: Path, role: str) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            register(path, role)
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def classifier_comparison(photo: np.ndarray, maps: np.ndarray, receipt: dict):
    titles = [
        "Correct · class 235\nGerman shepherd",
        "Runner-up · class 174\nNorwegian elkhound",
        "Decision margin\nlogit[235] − logit[174]",
        "Lowest score · class 171\nItalian greyhound",
    ]
    fig = plt.figure(figsize=(13.2, 6.8))
    grid = fig.add_gridspec(1, 5, left=0.025, right=0.985, bottom=0.24, top=0.76,
                           width_ratios=[0.78, 1, 1, 1, 1], wspace=0.08)
    original = fig.add_subplot(grid[0, 0])
    original.imshow(photo)
    original.axis("off")
    original.set_title("Input\nGT: German shepherd", fontsize=9.5, weight="bold", pad=9)
    limit = float(np.max(np.abs(maps)))
    shown = np.sign(maps) * np.log1p(9 * np.abs(maps) / max(limit, 1e-30)) / np.log(10)
    targets = receipt["targets"]
    for index in range(4):
        axis = fig.add_subplot(grid[0, index + 1])
        axis.imshow(shown[index], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axis.axis("off")
        axis.set_title(titles[index], fontsize=10.5, weight="bold", pad=9)
        target = targets[index]
        axis.text(0.5, -0.08, f"clean {target['clean_logit']:.4f}  |  Δ {target['delta']:.4f}",
                  transform=axis.transAxes, ha="center", va="top", fontsize=9, color="#172B3A")
        axis.text(0.5, -0.145, f"ΣL = {maps[index].sum():.4f}",
                  transform=axis.transAxes, ha="center", va="top", fontsize=8.5, color="#637486")
    fig.suptitle(TITLE, fontsize=18, weight="bold", y=0.96)
    fig.text(0.54, 0.875,
             "One frozen image · full 12-block path · one shared signed-log scale across all four maps",
             ha="center", fontsize=10.5, color="#637486")
    colorbar = fig.add_axes([0.39, 0.12, 0.31, 0.022])
    colorbar.imshow(np.linspace(-1, 1, 257)[None], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    colorbar.set_xticks([0, 128, 256], ["negative", "zero", "positive"], fontsize=9)
    colorbar.set_yticks([])
    fig.text(0.54, 0.045,
             "Margin is computed pixelwise as L235 − L174 · lowest-score class is global argmin before viewing maps",
             ha="center", fontsize=9.5, color="#637486")
    return fig, {
        "panel": "35_four_classifier_targets",
        "scale": "shared signed log1p kappa9 global maxabs; no clipping",
        "limit": limit, "shape": list(maps.shape), "clipped_count": 0,
    }


def save(fig) -> list[dict]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for axis in fig.axes:
        for item in axis.texts:
            if not all(fig.bbox.contains(*point) for point in item.get_window_extent(renderer).get_points()):
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"Figure 35 out of bounds: {outside}")
    return graphics.base.save_figure(fig, 35, SLUG)


def build_docs(figures: list[dict], receipt: dict) -> None:
    zh = common.sections(OLD / "captions_zh.md", "## Figure ")
    en = common.sections(OLD / "captions_en.md", "## Figure ")
    analysis = common.sections(OLD / "figure_analysis.md", "## Image #")
    zh[35] = (
        "同一冻结输入的四个 classifier 标量 learning image。类别 235 是官方真值 German shepherd 且为 clean argmax；"
        "类别 174 Norwegian elkhound 是最高竞争输出；第三列为逐像素 L235−L174 的判别 margin；类别 171 Italian greyhound "
        "是 1000 个 clean logits 的全局最低分。四图采用共同 max-abs signed-log 色标。低分类别按 logit 排序预先确定，不按归因图观感挑选。"
    )
    en[35] = (
        "Four classifier-scalar learning images for the same frozen input. Class 235 is both the official German-shepherd label and the clean argmax; "
        "class 174 (Norwegian elkhound) is the strongest competing output; the third map is the pixelwise decision margin L235-L174; "
        "and class 171 (Italian greyhound) is the global minimum among all 1,000 clean logits. One shared max-absolute signed-log scale is used. "
        "The low-score target was selected by logits before viewing attribution maps."
    )
    last = receipt["attempts"][-1]
    analysis[35] = (
        zh[35] + " 正确类与 runner-up 的 clean logit 分别为 8.0826 和 7.6327，margin 为 0.4499；最低分输出为 −2.9210。"
        f"4096 阶相对 2048 阶的四图 L1 变化依次为 {last['map_L1_relative_change'][0]:.3e}、"
        f"{last['map_L1_relative_change'][1]:.3e}、{last['map_L1_relative_change'][2]:.3e}、"
        f"{last['map_L1_relative_change'][3]:.3e}。该图只比较单样本输出单元，不推出总体类别机制。"
    )
    for path in [OLD / "captions_zh.md", OLD / "captions_en.md", OLD / "figure_analysis.md"]:
        register(path, "inherited_text")

    header = "[画廊](gallery.html) · [图表索引](figure_index.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R10_20260917.md)"
    for filename, values, marker in [("captions_zh.md", zh, "## Figure "),
                                      ("captions_en.md", en, "## Figure "),
                                      ("figure_analysis.md", analysis, "## Image #")]:
        lines = ["# R10-R2 · 35 张论文图", "", header, ""]
        for figure in figures:
            number = figure["number"]
            suffix = f": {figure['title']}" if marker.endswith("#") else ""
            lines.extend([marker + str(number) + suffix, "", values[number], ""] )
            if marker.endswith("#"):
                png = next(item["path"] for item in figure["outputs"] if item["format"] == "PNG")
                lines.extend([f"![Figure {number}]({Path(png).relative_to(ROOT.relative_to(REPO)).as_posix()})", ""] )
        (ROOT / filename).write_text("\n".join(lines), encoding="utf-8")

    index = ["# R10-R2 完整论文图集：Figure 1–35", "", header, "",
             "Figure 1–34 从已封存 R10-R1 逐字节继承；Figure 35 新增 classifier 输出单元对比。", "",
             "| 图号 | 内容 | PNG | PDF | SVG |", "|---:|---|---|---|---|"]
    for figure in figures:
        links = {item["format"]: Path(item["path"]).relative_to(ROOT.relative_to(REPO)).as_posix()
                 for item in figure["outputs"]}
        index.append(f"| {figure['number']} | {figure['title']} | " +
                     " | ".join(f"[{kind}]({links[kind]})" for kind in ["PNG", "PDF", "SVG"]) + " |")
    (ROOT / "figure_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    cards = []
    for figure in figures:
        number = figure["number"]
        links = {item["format"]: Path(item["path"]).relative_to(ROOT.relative_to(REPO)).as_posix()
                 for item in figure["outputs"]}
        cards.append(f'<article id="figure-{number}"><h3>Figure {number} · {html.escape(figure["title"])}</h3>'
                     f'<a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}" alt="Figure {number}"></a>'
                     f'<p>{html.escape(zh[number])}</p><p>' +
                     " · ".join(f'<a href="{links[kind]}">{kind}</a>' for kind in ["PNG", "PDF", "SVG"]) +
                     "</p></article>")
    gallery = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>NNS-ViT R10-R2</title><style>body{font:16px/1.6 system-ui;color:#172B3A;background:#eef3f6;max-width:1500px;margin:auto;padding:24px}'
               'article{background:white;padding:20px;margin:22px 0}img{width:100%}a{color:#137EAB}nav{position:sticky;top:0;background:#eef3f6;padding:12px;z-index:2}</style>'
               '<h1>NNS-ViT · R10-R2 · 35 张论文图</h1><p>R10-R1 全图集，加上正确类、runner-up、margin 与最低分输出单元的真实对比。</p>'
               '<p><a href="figure_index.md">图表索引</a> · <a href="figure_analysis.md">逐图说明</a> · '
               '<a href="../../docs/论文大纲_NNS_ViT_R10_20260917.md">论文大纲</a></p><nav>' +
               " · ".join(f'<a href="#figure-{number}">{number}</a>' for number in range(1, 36)) +
               "</nav>" + "".join(cards) + "</html>")
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# NNS-ViT 完整论文图集 R10-R2\n\n" + header +
        "\n\n35 张编号图、105 个 PNG/PDF/SVG 主导出。Figure 1–34 从 R10-R1 逐字节继承；Figure 35 比较正确类别 235、最高竞争类别 174、二者 logit margin，以及 clean logits 全局最低的类别 171。\n\n"
        "所有新 classifier 图使用同一输入、灰色参考和完整 12-block 前向路径。Figure 35 使用共同 signed 色标，仅作为单图方法展示，不构成总体类别机制结论。\n", encoding="utf-8")


def main() -> None:
    if (ROOT / "terminal_receipt.json").exists():
        raise RuntimeError("Sealed R10-R2 root")
    old_manifest = checked(OLD / "render_manifest.json")
    old_terminal = checked(OLD / "terminal_receipt.json")
    if old_terminal["status"] != "PASS_PRESENTATION_RENDER" or old_terminal["revision"] != "R10-R1":
        raise RuntimeError("R10-R1 predecessor is not passed")
    before = common.inventory(OLD)
    receipt = checked(ROOT / "classifier_comparison_receipt.json")
    if receipt["status"] != "PASS_SINGLE_IMAGE_CLASSIFIER_COMPARISON":
        raise RuntimeError("Classifier comparison is not passed")
    for row in receipt["source_files"]:
        path = REPO / row["path"]
        if common.sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Comparison source drift: {path}")
        register(path, "comparison_source")
    for path in [Path(__file__), REPO / "scripts/validate_nns_vit_unified_paper_figures_r10_r2.py",
                 Path(graphics.__file__), Path(graphics.base.__file__), Path(common.__file__),
                 REPO / "config/nns_vit_unified_paper_figures_r4.json",
                 REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"]:
        register(path, "implementation_or_document")

    graphics.base.configure(common.read(REPO / "config/nns_vit_unified_paper_figures_r4.json"))
    graphics.base.OUTPUT = ROOT
    plt.rcParams["svg.hashsalt"] = "nns-vit-r10-r2"
    figures = []
    for figure in old_manifest["figures"]:
        outputs = []
        for record in figure["outputs"]:
            source = REPO / record["path"]
            if common.sha(source) != record["sha256"]:
                raise RuntimeError(f"R10-R1 figure drift: {source}")
            register(source, "inherited_figure")
            target = ROOT / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            outputs.append({**record, "path": target.relative_to(REPO).as_posix(),
                            "inherited_from": source.relative_to(REPO).as_posix()})
        figures.append({"number": figure["number"], "title": figure["title"],
                        "kind": figure.get("kind", "inherited_paper_figure"),
                        "origin": OLD.relative_to(REPO).as_posix(), "new_render": False,
                        "outputs": outputs})

    photo = np.load(ROOT / "source_arrays/original_rgb.npy", allow_pickle=False)
    maps = np.load(ROOT / "source_arrays/classifier_four_targets_maps.npy", allow_pickle=False)
    figure, display = classifier_comparison(photo, maps, receipt)
    outputs = save(figure)
    figures.append({"number": 35, "title": TITLE, "kind": "classifier_output_comparison",
                    "new_render": True, "outputs": outputs})
    (ROOT / "real_outputs").mkdir(exist_ok=True)
    limit = float(np.max(np.abs(maps)))
    shown = np.sign(maps) * np.log1p(9 * np.abs(maps) / max(limit, 1e-30)) / np.log(10)
    for name, array in zip(receipt["target_order"], shown):
        plt.imsave(ROOT / "real_outputs" / f"classifier_{name}_shared_signed.png", array,
                   cmap="RdBu_r", vmin=-1, vmax=1)
    for folder in ["all_heads", "reference_gallery", "supplement", "real_outputs"]:
        copy_tree(OLD / folder, ROOT / folder, "inherited_companion")

    figures.sort(key=lambda item: item["number"])
    build_docs(figures, receipt)
    if common.inventory(OLD) != before:
        raise RuntimeError("R10-R1 predecessor changed during rendering")
    common.write(ROOT / "predecessor_snapshot.json", common.signed({
        "root": OLD.relative_to(REPO).as_posix(), "files": before,
        "files_digest": common.canonical(before)}))
    common.write(ROOT / "source_arrays_manifest.json", common.signed({
        "files": common.inventory(ROOT / "source_arrays")}))
    common.write(ROOT / "numeric_display_data.json", common.signed({
        "classifier_comparison_receipt_sha256": common.sha(ROOT / "classifier_comparison_receipt.json"),
        "display": [display], "new_cohort_statistics": False}))
    old_audit = checked(OLD / "paper_evidence_audit.json")
    sources = [SOURCES[key] for key in sorted(SOURCES)]
    common.write(ROOT / "render_manifest.json", common.signed({
        "revision": "R10-R2", "status": "RENDERED_PENDING_VISUAL_REVIEW",
        "figure_count": 35, "artifact_count": 105, "figures": figures,
        "source_files": sources, "source_files_digest": common.canonical(sources),
        "inherited_figures": list(range(1, 35)), "new_figures": [35],
        "cpu_only": True,
        "new_inference": "single-input classifier-output comparison only"}))
    common.write(ROOT / "layout_audit.json", common.signed({
        "status": "PASS", "records": [{"number": 35, "text_outside_canvas": []}]}))
    common.write(ROOT / "paper_evidence_audit.json", common.signed({
        "status": "PASS", "scope": "R10-R1 evidence inherited; Figure 35 is a single-input comparison",
        "predecessor_audit_sha256": common.sha(OLD / "paper_evidence_audit.json"),
        "predecessor_status": old_audit["status"],
        "outline_sha256": common.sha(REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"),
        "source_records": []}))
    print("R10-R2 rendered: 35 figures / 105 exports", flush=True)


if __name__ == "__main__":
    main()
