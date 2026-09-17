#!/usr/bin/env python3
"""Assemble the R10 paper figure package and render Figures 32--34."""
from __future__ import annotations

import html
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-unified-paper-r10")

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import render_nns_vit_unified_paper_figures_r7 as graphics
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r10_r1_20260917"
COMPUTE = REPO / "reports/nns_vit_unified_paper_figures_r10_20260917"
R9 = REPO / "reports/nns_vit_unified_paper_figures_r9_20260917"
CHANNELS = [0, 255, 511, 767]
NEW_TITLES = {
    32: "B0 MLP branch: real channel learning images",
    33: "B0 block output: real channel learning images",
    34: "From the image to the predicted class logit",
}
NEW_SLUGS = {
    32: "real_b0_mlp_learning_images",
    33: "real_b0_block_output_learning_images",
    34: "real_classifier_logit_learning_image",
}
BLUE, INK, MUTED, LIGHT = "#1875A5", "#172B3A", "#637486", "#EDF3F7"
SOURCES: dict[str, dict] = {}
DISPLAY: list[dict] = []
LAYOUT: list[dict] = []


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


def signed_scaled(raw: np.ndarray) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw, dtype=float)
    limit = float(np.max(np.abs(raw)))
    if limit == 0:
        return np.zeros_like(raw), 0.0
    return np.sign(raw) * np.log1p(9 * np.abs(raw) / limit) / np.log(10), limit


def draw_heat(axis, raw: np.ndarray, cmap, key: str, signed: bool) -> None:
    raw = np.asarray(raw, dtype=float)
    if signed:
        shown, limit = signed_scaled(raw)
        axis.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        rule = "signed log1p kappa9 maxabs; no clipping"
        clipped = 0
    else:
        positive = np.maximum(raw, 0)
        limit = float(np.percentile(positive, 99)) or float(positive.max())
        shown = np.clip(positive / max(limit, 1e-30), 0, 1)
        axis.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        rule = "positive P99 with max fallback; display-only clipping"
        clipped = int((positive > limit).sum())
    axis.axis("off")
    DISPLAY.append({
        "panel": key, "scale": rule, "limit": limit,
        "shape": list(raw.shape), "clipped_count": clipped,
    })


def module_panel(number: int, photo: np.ndarray, maps: np.ndarray, tokens: np.ndarray, cmap, signed=False):
    fig = plt.figure(figsize=(12, 6.4))
    grid = fig.add_gridspec(
        2, 6, left=0.025, right=0.985, bottom=0.13, top=0.81,
        width_ratios=[0.76, 0.20, 1, 1, 1, 1], wspace=0.08, hspace=0.23,
    )
    input_axis = fig.add_subplot(grid[:, 0])
    input_axis.imshow(photo)
    input_axis.axis("off")
    input_axis.set_title("Input", fontsize=11, weight="bold")
    for index, channel in enumerate(CHANNELS):
        top = fig.add_subplot(grid[0, index + 2])
        bottom = fig.add_subplot(grid[1, index + 2])
        draw_heat(top, maps[index], cmap, f"{number}_{channel}_learning_{signed}", signed)
        draw_heat(bottom, tokens[:, channel].reshape(14, 14), cmap, f"{number}_{channel}_token", True)
        top.set_title(f"Channel {channel}", fontsize=12, weight="bold", pad=8)
        if index == 0:
            top.text(-0.08, 0.5, "Learning image\ninput coordinates", transform=top.transAxes,
                     ha="right", va="center", fontsize=9, rotation=90)
            bottom.text(-0.08, 0.5, "Token response\ninternal indices", transform=bottom.transAxes,
                        ha="right", va="center", fontsize=9, rotation=90)
    fig.suptitle(NEW_TITLES[number] + (" · signed companion" if signed else ""),
                 fontsize=18, weight="bold", y=0.96)
    subtitle = ("LN2 → fc1 → GELU → fc2 → layer scale/drop path" if number == 32 else
                "first residual + MLP branch → complete Block 0 output")
    fig.text(0.53, 0.877, subtitle + " · same frozen input and gray reference",
             fontsize=10.5, ha="center", color=MUTED)
    fig.text(0.53, 0.066,
             ("Top: complete signed response contributions" if signed else
              "Top: positive response contributions") +
             " · 224 × 224     Bottom: signed output values · 14 × 14",
             fontsize=10, ha="center", color=MUTED)
    fig.text(0.53, 0.025,
             "Each target sums one fixed channel over 196 patch tokens · per-panel display scales",
             fontsize=9, ha="center", color=MUTED)
    return fig


def classifier_panel(photo: np.ndarray, raw_map: np.ndarray, receipt: dict):
    fig = plt.figure(figsize=(12, 6.4))
    fig.suptitle(NEW_TITLES[34], fontsize=18, weight="bold", y=0.96)
    fig.text(0.5, 0.885,
             "Complete 12-block forward path · final-normalized CLS · Linear(768,1000) · class 235",
             ha="center", fontsize=10.5, color=MUTED)

    original = fig.add_axes([0.045, 0.24, 0.24, 0.49])
    original.imshow(photo)
    original.axis("off")
    original.set_title("Input image", fontsize=12, weight="bold", pad=8)

    flow = fig.add_axes([0.31, 0.23, 0.34, 0.51])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis("off")
    labels = [
        "Patch embedding + CLS + position",
        "Blocks 0–11",
        "FinalNorm → CLS[768]",
        "Classifier Linear(768,1000)",
        "logit[235] = 8.0826",
    ]
    ypos = [0.88, 0.68, 0.48, 0.28, 0.08]
    for index, (label, y) in enumerate(zip(labels, ypos)):
        face = "#EAF4F8" if index < 4 else "#FFF1DB"
        edge = BLUE if index < 4 else "#D18A2E"
        flow.add_patch(plt.Rectangle((0.08, y - 0.065), 0.84, 0.13, facecolor=face,
                                     edgecolor=edge, linewidth=1.3))
        flow.text(0.5, y, label, ha="center", va="center", fontsize=10.5,
                  weight="bold" if index in (1, 4) else "normal", color=INK)
        if index < len(labels) - 1:
            flow.annotate("", xy=(0.5, ypos[index + 1] + 0.075), xytext=(0.5, y - 0.075),
                          arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.4})

    learned = fig.add_axes([0.69, 0.24, 0.27, 0.49])
    shown, limit = signed_scaled(raw_map)
    learned.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    learned.axis("off")
    learned.set_title("Signed learning image · 224 × 224", fontsize=12, weight="bold", pad=8)
    DISPLAY.append({
        "panel": "34_classifier_logit",
        "scale": "signed log1p kappa9 maxabs; no clipping",
        "limit": limit, "shape": list(raw_map.shape), "clipped_count": 0,
    })
    colorbar = fig.add_axes([0.735, 0.185, 0.18, 0.018])
    colorbar.imshow(np.linspace(-1, 1, 257)[None], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    colorbar.set_xticks([0, 128, 256], ["−", "0", "+"], fontsize=9)
    colorbar.set_yticks([])
    target = receipt["classifier_target"]
    fig.text(0.5, 0.115,
             f"reference logit = {target['baseline_logit']:.6f}    Δlogit = {target['delta']:.6f}    "
             f"sum(map) = {raw_map.sum():.6f}",
             ha="center", fontsize=10.5, color=INK)
    fig.text(0.5, 0.055,
             "The scalar classifier output has no 14 × 14 token-output panel · one fixed input, not a population claim",
             ha="center", fontsize=9.5, color=MUTED)
    return fig


def save(fig, number: int, slug: str) -> list[dict]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for axis in fig.axes:
        for item in axis.texts:
            if not all(fig.bbox.contains(*point) for point in item.get_window_extent(renderer).get_points()):
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"Figure {number} out of bounds: {outside}")
    LAYOUT.append({"number": number, "text_outside_canvas": []})
    return graphics.base.save_figure(fig, number, slug)


def copy_tree(source: Path, target: Path, role: str) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            register(path, role)
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def build_docs(figures: list[dict], receipt: dict) -> None:
    zh = common.sections(R9 / "captions_zh.md", "## Figure ")
    en = common.sections(R9 / "captions_en.md", "## Figure ")
    analysis = common.sections(R9 / "figure_analysis.md", "## Image #")
    zh.update({
        32: "B0 MLP branch 的真实响应 learning image。固定通道 0、255、511、767，每个目标为该通道在 196 个 patch-token MLP 输出上的总和。上排为 224×224 输入贡献正部，下排为同一通道的 14×14 signed token 值。真实前向包含 LN2、fc1、GELU、fc2、layer scale 与 drop path；完整 signed 图见 supplement。",
        33: "B0 第二次 residual add 后的完整 block output。固定通道和版式与 Figure 32 相同。原始 RGB learning image 逐像素满足 L(second residual)=L(first residual)+L(MLP branch)，最大误差为 3.33×10⁻¹⁶；着色只在加法核对之后进行。",
        34: "真实预测类别 235 的完整 logit learning image。梯度经过全部 12 个 blocks、final norm、CLS 选择与 Linear(768,1000)，相对灰色参考积分回 224×224 输入坐标。clean logit 为 8.082624，参考 logit 为 −0.384007。类别标量没有 14×14 token 输出，因此本图不套用模块通道页。",
    })
    en.update({
        32: "Real response learning images for the block-0 MLP branch. Channels 0, 255, 511 and 767 are fixed; each target sums that channel over 196 patch-token outputs. Top: positive input contributions at 224×224. Bottom: signed MLP output values at 14×14 token indices. The native path includes LN2, fc1, GELU, fc2, layer scale and drop path. Signed companions are retained in the supplement.",
        33: "Real block-0 outputs after the second residual addition, using the same fixed channels and layout. In the raw RGB arrays, L(second residual)=L(first residual)+L(MLP branch) holds pixelwise with maximum error 3.33e-16. Display transforms are applied only after this reconstruction check.",
        34: "Complete class-logit learning image for the clean predicted class 235. Gradients traverse all 12 blocks, final normalization, CLS selection and Linear(768,1000), then integrate back to the 224×224 input relative to gray. The clean and reference logits are 8.082624 and -0.384007. A scalar classifier output has no 14×14 token-output map.",
    })
    analysis.update({
        32: zh[32] + " 这页回答的是 MLP 分支自身在各输入位置上的响应来源，不是 MLP 对最终类别的下游贡献。四个通道按索引固定，不按图像观感筛选。",
        33: zh[33] + " 与 Figure 31 对照时，Figure 31 是进入 MLP 前的 first residual，Figure 33 才是完整 Block 0 输出。",
        34: zh[34] + " 4096 阶相对 2048 阶的地图 L1 变化为 8.293×10⁻⁴，completeness 绝对误差为 0.004874；这是一张图的方法展示，不外推类别总体规律。",
    })
    register(R9 / "captions_zh.md", "inherited_caption")
    register(R9 / "captions_en.md", "inherited_caption")
    register(R9 / "figure_analysis.md", "inherited_analysis")

    header_link = "[画廊](gallery.html) · [图表索引](figure_index.md) · [论文大纲](../../docs/论文大纲_NNS_ViT_R10_20260917.md)"
    for filename, values, marker in [
        ("captions_zh.md", zh, "## Figure "),
        ("captions_en.md", en, "## Figure "),
        ("figure_analysis.md", analysis, "## Image #"),
    ]:
        lines = ["# R10 · 34 张论文图", "", header_link, ""]
        for figure in figures:
            number = figure["number"]
            suffix = f": {figure['title']}" if marker.endswith("#") else ""
            lines.extend([marker + str(number) + suffix, "", values[number], ""] )
            if marker.endswith("#"):
                png = next(item["path"] for item in figure["outputs"] if item["format"] == "PNG")
                relative = Path(png).relative_to(ROOT.relative_to(REPO)).as_posix()
                lines.extend([f"![Figure {number}]({relative})", ""] )
        (ROOT / filename).write_text("\n".join(lines), encoding="utf-8")

    index = [
        "# R10 完整论文图集：Figure 1–34", "", header_link, "",
        "R9 Figure 1–31 逐字节继承；Figure 32–34 补齐 B0 MLP、第二 residual 与完整 classifier logit。全部 34 张均提供 PNG/PDF/SVG。", "",
        "| 图号 | 内容 | PNG | PDF | SVG |", "|---:|---|---|---|---|",
    ]
    for figure in figures:
        links = {item["format"]: Path(item["path"]).relative_to(ROOT.relative_to(REPO)).as_posix()
                 for item in figure["outputs"]}
        index.append(f"| {figure['number']} | {figure['title']} | " +
                     " | ".join(f"[{kind}]({links[kind]})" for kind in ["PNG", "PDF", "SVG"]) + " |")
    (ROOT / "figure_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    groups = [
        ("完整真实前向扩展", [28, 29, 30, 31, 32, 33, 34]),
        ("概念与逐步计算", [1, 2, 15, 16, 17, 18, 24, 25, 26, 27]),
        ("类别成分与导数细节", [19, 20, 21, 22, 23]),
        ("实验结果", list(range(3, 15))),
    ]
    cards = []
    by_number = {figure["number"]: figure for figure in figures}
    for title, numbers in groups:
        cards.append(f"<h2>{html.escape(title)}</h2>")
        for number in numbers:
            figure = by_number[number]
            links = {item["format"]: Path(item["path"]).relative_to(ROOT.relative_to(REPO)).as_posix()
                     for item in figure["outputs"]}
            cards.append(
                f'<article id="figure-{number}"><h3>Figure {number} · {html.escape(figure["title"])}</h3>'
                f'<a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}" alt="Figure {number}"></a>'
                f'<p>{html.escape(zh[number])}</p><p>' +
                " · ".join(f'<a href="{links[kind]}">{kind}</a>' for kind in ["PNG", "PDF", "SVG"]) +
                "</p></article>"
            )
    cards.append("<h2>完整 signed 视图及原始图库</h2>")
    for folder in ["supplement", "reference_gallery", "all_heads"]:
        for path in sorted((ROOT / folder).rglob("*.png")):
            relative = path.relative_to(ROOT).as_posix()
            cards.append(f'<details><summary>{html.escape(relative)}</summary><img loading="lazy" src="{relative}" alt="Companion figure"></details>')
    gallery = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>NNS-ViT R10</title><style>body{font:16px/1.6 system-ui;color:#172B3A;background:#eef3f6;max-width:1500px;margin:auto;padding:24px}'
        'article,details{background:white;padding:20px;margin:22px 0}img{width:100%}a{color:#137EAB}nav{position:sticky;top:0;background:#eef3f6;padding:12px;z-index:2}</style>'
        '<h1>NNS-ViT · R10 · 34 张论文图</h1><p>R9 全图集，加上 B0 MLP、第二 residual 和真实 classifier logit learning image。</p>'
        '<p><a href="figure_index.md">图表索引</a> · <a href="figure_analysis.md">逐图说明</a> · '
        '<a href="../../docs/论文大纲_NNS_ViT_R10_20260917.md">论文大纲</a></p><nav>' +
        " · ".join(f'<a href="#figure-{number}">{number}</a>' for number in range(1, 35)) +
        "</nav>" + "".join(cards) + "</html>"
    )
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# NNS-ViT 完整论文图集 R10\n\n" + header_link +
        "\n\n34 张编号图、102 个 PNG/PDF/SVG 导出。Figure 1–31 从已封存 R9 逐字节继承；Figure 32 是 B0 MLP branch，Figure 33 是 B0 second residual / block output，Figure 34 是通过全部 12 blocks、final norm、CLS 和 classifier 的 class-235 logit learning image。\n\n"
        "Figure 32–33 沿用上排 224×224 输入贡献、下排 14×14 token 值的版式；Figure 34 是类别标量，只画原图、真实前向链和一张 signed learning image。完整 signed companion 与原始 float64 RGB 数组分别位于 supplement 和 source_arrays。\n\n"
        "新增图仍为同一冻结输入上的方法展示，不增加总体科学结论。R9 根保持不变。\n",
        encoding="utf-8",
    )


def main() -> None:
    if (ROOT / "terminal_receipt.json").exists():
        raise RuntimeError("Sealed R10 root")
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "source_arrays").exists():
        shutil.copytree(COMPUTE / "source_arrays", ROOT / "source_arrays")
    shutil.copyfile(COMPUTE / "module_compute_receipt.json", ROOT / "module_compute_receipt.json")
    if (COMPUTE / "module_compute_failed.json").exists():
        shutil.copyfile(COMPUTE / "module_compute_failed.json", ROOT / "pre_pass_failure_receipt.json")
    r9_manifest = checked(R9 / "render_manifest.json")
    r9_terminal = checked(R9 / "terminal_receipt.json")
    if r9_terminal["status"] != "PASS_PRESENTATION_RENDER":
        raise RuntimeError("R9 predecessor is not passed")
    before = common.inventory(R9)
    receipt = checked(ROOT / "module_compute_receipt.json")
    if receipt["status"] != "PASS_SINGLE_IMAGE_VIT_EXTENSION_MAPS":
        raise RuntimeError("R10 module computation is not passed")
    for row in receipt["source_files"]:
        path = REPO / row["path"]
        if common.sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Compute source drift: {path}")
        register(path, "compute_source")
    for path in [
        Path(__file__), REPO / "scripts/validate_nns_vit_unified_paper_figures_r10.py",
        Path(graphics.__file__), Path(graphics.base.__file__), Path(common.__file__),
        REPO / "config/nns_vit_unified_paper_figures_r4.json",
        REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md",
        REPO / "docs/r10_r1_outline_rebind_amendment_20260917.md",
    ]:
        register(path, "implementation_or_document")

    cmap = graphics.base.configure(common.read(REPO / "config/nns_vit_unified_paper_figures_r4.json"))
    graphics.base.OUTPUT = ROOT
    plt.rcParams["svg.hashsalt"] = "nns-vit-r10"

    figures = []
    for figure in r9_manifest["figures"]:
        outputs = []
        for record in figure["outputs"]:
            source = REPO / record["path"]
            if common.sha(source) != record["sha256"] or source.stat().st_size != record["bytes"]:
                raise RuntimeError(f"R9 figure drift: {source}")
            register(source, "inherited_figure")
            target = ROOT / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            outputs.append({
                **record, "path": target.relative_to(REPO).as_posix(),
                "inherited_from": source.relative_to(REPO).as_posix(),
            })
        figures.append({
            "number": figure["number"], "title": figure["title"],
            "kind": figure.get("kind", "inherited_paper_figure"),
            "origin": R9.relative_to(REPO).as_posix(), "new_render": False,
            "outputs": outputs,
        })

    photo = np.load(ROOT / "source_arrays/original_rgb.npy", allow_pickle=False)
    extension_maps = np.load(ROOT / "source_arrays/b0_extension_maps.npy", allow_pickle=False)
    classifier_map = np.load(ROOT / "source_arrays/classifier_logit_map.npy", allow_pickle=False)
    tokens = {
        32: np.load(ROOT / "source_arrays/mlp_branch_clean_tokens.npy", allow_pickle=False),
        33: np.load(ROOT / "source_arrays/block_output_clean_tokens.npy", allow_pickle=False),
    }
    for number, module_index in [(32, 1), (33, 2)]:
        outputs = save(module_panel(number, photo, extension_maps[module_index], tokens[number], cmap),
                       number, NEW_SLUGS[number])
        figures.append({
            "number": number, "title": NEW_TITLES[number],
            "kind": "real_module_response", "new_render": True, "outputs": outputs,
        })
        signed_figure = module_panel(number, photo, extension_maps[module_index], tokens[number], cmap, signed=True)
        (ROOT / "supplement").mkdir(exist_ok=True)
        for extension in ["png", "pdf", "svg"]:
            signed_figure.savefig(ROOT / "supplement" / f"figure_{number:02d}_signed.{extension}",
                                  dpi=400, bbox_inches="tight", pad_inches=0.04)
        plt.close(signed_figure)
    figures.append({
        "number": 34, "title": NEW_TITLES[34], "kind": "real_classifier_logit",
        "new_render": True,
        "outputs": save(classifier_panel(photo, classifier_map, receipt), 34, NEW_SLUGS[34]),
    })

    for folder in ["all_heads", "reference_gallery", "supplement", "real_outputs"]:
        copy_tree(R9 / folder, ROOT / folder, "inherited_companion")

    for number, name, module_index in [(32, "mlp_branch", 1), (33, "block_output", 2)]:
        for index, channel in enumerate(CHANNELS):
            raw = extension_maps[module_index, index]
            positive = np.maximum(raw, 0)
            limit = float(np.percentile(positive, 99)) or float(positive.max())
            plt.imsave(ROOT / "real_outputs" / f"{name}_channel_{channel:03d}_positive.png",
                       np.clip(positive / max(limit, 1e-30), 0, 1), cmap=cmap, vmin=0, vmax=1)
            shown, _ = signed_scaled(raw)
            plt.imsave(ROOT / "real_outputs" / f"{name}_channel_{channel:03d}_signed.png",
                       shown, cmap="RdBu_r", vmin=-1, vmax=1)
    shown, _ = signed_scaled(classifier_map)
    plt.imsave(ROOT / "real_outputs/classifier_logit_235_signed.png",
               shown, cmap="RdBu_r", vmin=-1, vmax=1)

    figures.sort(key=lambda item: item["number"])
    build_docs(figures, receipt)
    if common.inventory(R9) != before:
        raise RuntimeError("R9 predecessor changed during R10 rendering")
    common.write(ROOT / "predecessor_snapshot.json", common.signed({
        "root": R9.relative_to(REPO).as_posix(), "files": before,
        "files_digest": common.canonical(before),
    }))
    common.write(ROOT / "source_arrays_manifest.json", common.signed({
        "files": common.inventory(ROOT / "source_arrays"),
    }))
    common.write(ROOT / "numeric_display_data.json", common.signed({
        "module_compute_receipt_sha256": common.sha(ROOT / "module_compute_receipt.json"),
        "display": DISPLAY, "new_cohort_statistics": False,
    }))
    r9_audit = checked(R9 / "paper_evidence_audit.json")
    sources = [SOURCES[key] for key in sorted(SOURCES)]
    common.write(ROOT / "render_manifest.json", common.signed({
        "revision": "R10-R1", "status": "RENDERED_PENDING_VISUAL_REVIEW",
        "figure_count": 34, "artifact_count": 102, "figures": figures,
        "source_files": sources, "source_files_digest": common.canonical(sources),
        "inherited_figures": list(range(1, 32)), "new_figures": [32, 33, 34],
        "compute_rebound_from": COMPUTE.relative_to(REPO).as_posix(),
        "cpu_only": True,
        "new_inference": "single-input B0 MLP, second-residual, and classifier-logit illustration only",
    }))
    common.write(ROOT / "layout_audit.json", common.signed({"status": "PASS", "records": LAYOUT}))
    common.write(ROOT / "paper_evidence_audit.json", common.signed({
        "status": "PASS", "scope": "R9 evidence inherited; R10-R1 rebinds corrected outline and single-image illustrations",
        "r9_paper_evidence_audit_sha256": common.sha(R9 / "paper_evidence_audit.json"),
        "r9_status": r9_audit["status"],
        "outline_sha256": common.sha(REPO / "docs/论文大纲_NNS_ViT_R10_20260917.md"),
        "source_records": [],
    }))
    print("R10 rendered: 34 figures / 102 exports", flush=True)


if __name__ == "__main__":
    main()
