#!/usr/bin/env python3
"""Create R10-R4 with Figure 2 attention-channel-255 positive/negative views."""
from __future__ import annotations

import html
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-unified-paper-r10-r4")
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import render_nns_vit_unified_paper_figures_r7 as graphics
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r4_20260917"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r10_r2_20260917"
DESIGN = REPO / "docs/unified_paper_figures_r10_r4_figure2_attention_ch255_positive_design_20260917.md"
TITLE = "A learning image keeps the input origins of an internal response"
SOURCES: dict[str, dict] = {}


def register(path: Path, role: str = "source") -> None:
    path = path.resolve()
    relative = os.path.relpath(path, REPO)
    SOURCES[relative] = {"path": relative, "bytes": path.stat().st_size,
                         "sha256": common.sha(path), "role": role}


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


def figure2_positive(photo: np.ndarray, raw: np.ndarray, cmap):
    fig, axis = graphics.canvas(12, 6.1)
    text = graphics.text
    text(axis, 0.32, 5.73, TITLE, 18, weight="bold")
    text(axis, 0.32, 5.27,
         "Real input  →  selected ViT response  →  positive contributions at the original input locations",
         10.5, color=graphics.MUTED)
    graphics.image_tile(axis, photo, 0.45, 2.13, 2.45)
    text(axis, 1.675, 1.86, "Input image", 11, weight="bold", ha="center")
    graphics.arrow(axis, (3.02, 3.36), (3.68, 3.36))
    graphics.box(axis, 3.83, 2.08, 3.15, 2.5, face=graphics.LIGHT, edge=graphics.BLUE)
    text(axis, 5.405, 4.25, "Trained ViT · Block 0", 12, weight="bold", ha="center")
    text(axis, 5.405, 3.82, "Patch embedding → Attention", 11, ha="center")
    text(axis, 5.405, 3.39, "Channel 255 · 196 patch tokens", 10.5, ha="center")
    text(axis, 5.405, 2.96, r"$s(x)=\sum_{t=1}^{196} O_{t,255}(x)$", 14, ha="center")
    delta = float(raw.sum())
    text(axis, 5.405, 2.45, f"Response change Δs = {delta:.4f}", 11,
         color=graphics.BLUE, weight="bold", ha="center")
    graphics.arrow(axis, (7.10, 3.36), (7.80, 3.36), graphics.BLUE)

    positive = np.maximum(raw, 0)
    positive_limit = float(np.percentile(positive, 99)) or float(positive.max())
    positive_shown = np.clip(positive / max(positive_limit, 1e-30), 0, 1)
    axis.imshow(positive_shown, extent=(8.0, 10.8, 1.94, 4.74), cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest", aspect="auto")
    text(axis, 9.4, 1.66, "Positive learning image · 224 × 224", 11, weight="bold", ha="center")
    axis.imshow(np.linspace(0, 1, 129)[None], extent=(8.5, 10.3, 1.29, 1.40),
                cmap=cmap, vmin=0, vmax=1, aspect="auto")
    text(axis, 8.39, 1.35, "low", 9, ha="right"); text(axis, 10.41, 1.35, "high", 9)
    text(axis, 9.4, 1.08, f"Positive part · P99 = {positive_limit:.4f} · deep blue = zero/low",
         8.8, color=graphics.MUTED, ha="center")
    axis.plot([0.35, 11.65], [0.85, 0.85], color="#DCE5ED")
    text(axis, 0.4, 0.52,
         r"$L_s(i)=(x_i-x_{0i})\int_0^1\partial_i s(x_0+\alpha(x-x_0))\,\mathrm{d}\alpha$",
         13)
    text(axis, 8.0, 0.59, f"Full signed map sum = {delta:.4f}",
         10.5, color=graphics.BLUE, weight="bold")
    text(axis, 8.0, 0.32, f"Displayed positive mass = {positive.sum():.4f}",
         9.5, color=graphics.MUTED)
    text(axis, 0.4, 0.15,
         "Main panel shows max(Ls,0) · full signed map retained in supplement · no spatial smoothing",
         9, color=graphics.MUTED)
    display = {"panel": "figure2_positive",
               "scale": "positive P99 with max fallback; display-only clipping; R4 heat palette",
               "limit": positive_limit,
               "shape": list(raw.shape),
               "clipped_count": int((positive > positive_limit).sum()),
               "positive_mass": float(positive.sum()),
               "signed_sum": delta}
    return fig, display


def save(fig) -> list[dict]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for axis in fig.axes:
        for item in axis.texts:
            if not all(fig.bbox.contains(*point) for point in item.get_window_extent(renderer).get_points()):
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"Figure 2 out of bounds: {outside}")
    return graphics.base.save_figure(fig, 2, "value_path_construction")


def strip_image_links(value: str) -> str:
    return re.sub(r"\n*!\[Figure \d+\]\([^)]+\)\n*", "\n", value).strip()


def build_docs(figures: list[dict]) -> None:
    zh = common.sections(OLD / "captions_zh.md", "## Figure ")
    en = common.sections(OLD / "captions_en.md", "## Figure ")
    analysis = {key: strip_image_links(value) for key, value in
                common.sections(OLD / "figure_analysis.md", "## Image #").items()}
    zh[2] = (
        "真实输入经过 ViT 的 patch embedding 和 B0 attention 后，选定 Figure 30 的 channel 255 在 196 个 patch-token 输出上的总和作为内部响应 s。"
        "右侧只显示正贡献 max(Ls,0)，采用与 Figure 30 channel 255 上排完全一致的蓝底 heat palette、逐面板 P99 缩放和 nearest。"
        "深蓝表示零或低正贡献，亮色表示较高正贡献。底部 full signed map sum 仍由未裁剪 signed 原始数组计算。"
        "当前目标的完整 signed 对照见 supplement/figure_30_signed 的 channel 255；旧 Figure 2 的 channel-0 signed 图仅作为历史备份。输入与 channel 255 均在计算前固定；本图仍是模块自身响应的单样本展示。"
    )
    en[2] = (
        "A real response learning image for the sum of channel 255 over 196 block-0 attention patch-token outputs, using the same raw target as the channel-255 panel in Figure 30. "
        "The main panel shows only max(Ls,0), using the same blue-base heat palette, per-panel P99 scaling, and nearest display as the channel-255 top panel in Figure 30. "
        "Deep blue denotes zero or low positive contribution. The full signed map sum is still computed from the unchanged raw signed array. "
        "The complete signed view for the current target is channel 255 in supplement/figure_30_signed; the previous channel-0 Figure 2 is retained only as a historical display backup. This remains a one-input module-own-response illustration."
    )
    analysis[2] = (
        zh[2] + " 这样 Figure 2 与 Figure 30 的 channel 255 严格绑定；若要查看负贡献或检查原始 completeness，可查看 supplement 的 signed 版本及 source_arrays。"
    )
    for path in [OLD / "captions_zh.md", OLD / "captions_en.md", OLD / "figure_analysis.md"]:
        register(path, "inherited_text")
    header = "[画廊](gallery.html) · [图表索引](figure_index.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R10_20260917.md)"
    for filename, values, marker in [("captions_zh.md", zh, "## Figure "),
                                      ("captions_en.md", en, "## Figure "),
                                      ("figure_analysis.md", analysis, "## Image #")]:
        lines = ["# R10-R4 · 35 张论文图", "", header, ""]
        for figure in figures:
            number = figure["number"]
            suffix = f": {figure['title']}" if marker.endswith("#") else ""
            lines.extend([marker + str(number) + suffix, "", values[number], ""] )
            if marker.endswith("#"):
                png = next(item["path"] for item in figure["outputs"] if item["format"] == "PNG")
                lines.extend([f"![Figure {number}]({Path(png).relative_to(ROOT.relative_to(REPO)).as_posix()})", ""] )
        (ROOT / filename).write_text("\n".join(lines), encoding="utf-8")

    index = ["# R10-R4 完整论文图集：Figure 1–35", "", header, "",
             "Figure 2 使用 Figure 30 attention channel 255 的单张蓝底正贡献图；Figure 1、3–35 从封存 R10-R2 逐字节继承。", "",
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
               '<title>NNS-ViT R10-R4</title><style>body{font:16px/1.6 system-ui;color:#172B3A;background:#eef3f6;max-width:1500px;margin:auto;padding:24px}'
               'article{background:white;padding:20px;margin:22px 0}img{width:100%}a{color:#137EAB}nav{position:sticky;top:0;background:#eef3f6;padding:12px;z-index:2}</style>'
               '<h1>NNS-ViT · R10-R4 · 35 张论文图</h1><p>Figure 2 使用 Figure 30 attention channel 255 的蓝底正贡献图。</p>'
               '<p><a href="figure_index.md">图表索引</a> · <a href="figure_analysis.md">逐图说明</a> · '
               '<a href="../../docs/论文大纲_NNS_ViT_R10_20260917.md">论文大纲</a></p><nav>' +
               " · ".join(f'<a href="#figure-{number}">{number}</a>' for number in range(1, 36)) +
               "</nav>" + "".join(cards) + "</html>")
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# NNS-ViT 完整论文图集 R10-R4\n\n" + header +
        "\n\n35 张编号图、105 个 PNG/PDF/SVG 主导出。Figure 2 使用 Figure 30 attention channel 255 的同一原始数组，只显示单张蓝底正贡献图；Figure 1、3–35 从 R10-R2 逐字节继承。\n\n"
        "Figure 2 的旧 signed 版位于 supplement，原始 float64 数组位于 source_arrays。本修订不运行模型、不重算 attribution、不产生新科学推断。\n",
        encoding="utf-8")


def main() -> None:
    if (ROOT / "terminal_receipt.json").exists():
        raise RuntimeError("Sealed R10-R4 root")
    ROOT.mkdir(parents=True, exist_ok=True)
    old_manifest = checked(OLD / "render_manifest.json")
    old_terminal = checked(OLD / "terminal_receipt.json")
    if old_terminal["status"] != "PASS_PRESENTATION_RENDER" or old_terminal["revision"] != "R10-R2":
        raise RuntimeError("R10-R2 predecessor is not passed")
    before = common.inventory(OLD)
    for path in [Path(__file__), REPO / "scripts/validate_nns_vit_unified_paper_figures_r10_r4.py",
                 Path(graphics.__file__), Path(graphics.base.__file__), Path(common.__file__),
                 REPO / "config/nns_vit_unified_paper_figures_r4.json", DESIGN,
                 REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"]:
        register(path, "implementation_or_document")

    copy_tree(OLD / "source_arrays", ROOT / "source_arrays", "inherited_source_array")
    shutil.copyfile(OLD / "classifier_comparison_receipt.json", ROOT / "classifier_comparison_receipt.json")
    cmap = graphics.base.configure(common.read(REPO / "config/nns_vit_unified_paper_figures_r4.json"))
    graphics.base.OUTPUT = ROOT
    plt.rcParams["svg.hashsalt"] = "nns-vit-r10-r4"

    photo = np.load(ROOT / "source_arrays/original_rgb.npy", allow_pickle=False)
    raw = np.load(ROOT / "source_arrays/module_response_maps.npy", allow_pickle=False)[1, 1]
    clean = np.load(ROOT / "source_arrays/attention_clean_tokens.npy", allow_pickle=False)
    baseline = np.load(ROOT / "source_arrays/attention_baseline_tokens.npy", allow_pickle=False)
    delta = float((clean - baseline)[:, 255].sum())
    if abs(float(raw.sum()) - delta) >= 1e-9:
        raise RuntimeError("Figure 2 raw-map completeness drift")

    figures = []
    old_figure2 = None
    for figure in old_manifest["figures"]:
        if figure["number"] == 2:
            old_figure2 = figure
            rendered, display = figure2_positive(photo, raw, cmap)
            figures.append({"number": 2, "title": figure["title"],
                            "kind": "positive_response_definition", "new_render": True,
                            "outputs": save(rendered)})
            continue
        outputs = []
        for record in figure["outputs"]:
            source = REPO / record["path"]
            if common.sha(source) != record["sha256"] or source.stat().st_size != record["bytes"]:
                raise RuntimeError(f"R10-R2 figure drift: {source}")
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
    if old_figure2 is None:
        raise RuntimeError("Missing predecessor Figure 2")

    for folder in ["all_heads", "reference_gallery", "supplement", "real_outputs"]:
        copy_tree(OLD / folder, ROOT / folder, "inherited_companion")
    predecessor_display_companions = []
    (ROOT / "supplement").mkdir(exist_ok=True)
    for record in old_figure2["outputs"]:
        source = REPO / record["path"]
        register(source, "signed_figure2_predecessor")
        target = ROOT / "supplement" / f"figure_02_signed_original.{record['format'].lower()}"
        shutil.copyfile(source, target)
        predecessor_display_companions.append({"format": record["format"],
                                               "path": target.relative_to(REPO).as_posix(),
                                               "bytes": target.stat().st_size, "sha256": common.sha(target),
                                               "source_path": source.relative_to(REPO).as_posix(),
                                               "historical_target": "attention channel 0"})
    signed_companions = []
    for extension in ["png", "pdf", "svg"]:
        target = ROOT / "supplement" / f"figure_30_signed.{extension}"
        source = OLD / "supplement" / f"figure_30_signed.{extension}"
        signed_companions.append({"format": extension.upper(),
                                  "path": target.relative_to(REPO).as_posix(),
                                  "bytes": target.stat().st_size, "sha256": common.sha(target),
                                  "source_path": source.relative_to(REPO).as_posix(),
                                  "selected_channel": 255})
    positive = np.maximum(raw, 0)
    positive_limit = float(np.percentile(positive, 99)) or float(positive.max())
    common.write(ROOT / "figure_02_display_receipt.json", common.signed({
        "status": "PASS_DISPLAY_ONLY_POSITIVE_RESTYLE",
        "target": "sum of attention channel 255 over 196 patch-token outputs",
        "source_array": "source_arrays/module_response_maps.npy[1,1]",
        "source_figure_panel": "Figure 30, channel 255, top row",
        "raw_shape": list(raw.shape), "signed_sum": float(raw.sum()),
        "token_recomputed_delta": delta, "positive_mass": float(positive.sum()),
        "display_rule": "max(raw,0), per-panel P99 with max fallback, clip display to [0,1], R4 heat palette, nearest",
        "p99_limit": positive_limit, "raw_positive_max": float(positive.max()),
        "display_clipped_count": int((positive > positive_limit).sum()),
        "signed_companions": signed_companions,
        "predecessor_display_companions": predecessor_display_companions, "model_forward_calls": 0,
        "attribution_calls": 0, "new_scientific_inference": False}))

    figures.sort(key=lambda item: item["number"])
    build_docs(figures)
    if common.inventory(OLD) != before:
        raise RuntimeError("R10-R2 predecessor changed during rendering")
    common.write(ROOT / "predecessor_snapshot.json", common.signed({
        "root": OLD.relative_to(REPO).as_posix(), "files": before,
        "files_digest": common.canonical(before)}))
    common.write(ROOT / "source_arrays_manifest.json", common.signed({
        "files": common.inventory(ROOT / "source_arrays")}))
    old_display = checked(OLD / "numeric_display_data.json")
    common.write(ROOT / "numeric_display_data.json", common.signed({
        "display": old_display["display"] + [display], "new_cohort_statistics": False,
        "figure_02_display_receipt_sha256": common.sha(ROOT / "figure_02_display_receipt.json")}))
    old_audit = checked(OLD / "paper_evidence_audit.json")
    sources = [SOURCES[key] for key in sorted(SOURCES)]
    common.write(ROOT / "render_manifest.json", common.signed({
        "revision": "R10-R4", "status": "RENDERED_PENDING_VISUAL_REVIEW",
        "figure_count": 35, "artifact_count": 105, "figures": figures,
        "source_files": sources, "source_files_digest": common.canonical(sources),
        "redesigned_figures": [2],
        "inherited_figures": [number for number in range(1, 36) if number != 2],
        "cpu_only": True, "new_inference": False,
        "revision_scope": "Figure 2 rebound to existing Figure 30 attention channel 255 positive blue-base view"}))
    common.write(ROOT / "layout_audit.json", common.signed({
        "status": "PASS", "records": [{"number": 2, "text_outside_canvas": []}]}))
    common.write(ROOT / "paper_evidence_audit.json", common.signed({
        "status": "PASS", "scope": "R10-R2 evidence inherited; Figure 2 uses existing attention channel 255 positive display",
        "predecessor_audit_sha256": common.sha(OLD / "paper_evidence_audit.json"),
        "predecessor_status": old_audit["status"],
        "outline_sha256": common.sha(REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"),
        "source_records": []}))
    print("R10-R4 rendered: Figure 2 attention channel 255 positive view; 35 figures / 105 exports", flush=True)


if __name__ == "__main__":
    main()
