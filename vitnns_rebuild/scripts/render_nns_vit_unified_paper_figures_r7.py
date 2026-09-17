#!/usr/bin/env python3
"""Render the scanner-oriented R7 figure package from existing CPU-readable evidence."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-unified-paper-r7")

import argparse
import hashlib
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import render_nns_vit_unified_paper_figures_r1 as base
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = base.REPO
ROOT = REPO / "reports/nns_vit_unified_paper_figures_r7_20260916"
OLD = REPO / "reports/nns_vit_unified_paper_figures_r6_20260916"
H_REPORT = REPO / "reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907"
C_REPORT = REPO / "reports/stage_f_f3c_visual_v1_20260908"
DESIGN = REPO / "docs/unified_paper_figures_r7_design_20260916.md"
CHANGED = [1, 2, 8, 9, 10]
BLOCKS = (0, 5, 11)
HEADS = (0, 5, 11)
BLUE, INK, MUTED, LIGHT = "#1875A5", "#172B3A", "#637486", "#EDF3F7"
COLORS = ("#137EAB", "#D18A2E", "#7B68AC")
COMPONENTS = ("nns_positive", "matched_positive", "nns_negative", "matched_negative")
SOURCES = base.SOURCES
TITLES = {
    1: "Scanning a Vision Transformer with NNS",
    2: "From an internal response to an input-space image",
    8: "A median-effect image viewed through all 144 heads",
    9: "Spatial support across transformer depth",
    10: "Different explanations of the same images",
}

CAPTIONS_ZH = {
    1: "NNS 扫描 ViT 的内部响应。左侧将同一张输入图像分成 patch tokens，经过已训练的 ViT；右侧按层（B0、B5、B11）和 head（H0、H5、H11）排列实测 NNS 正贡献图。每格表示输入位置对该坐标 CLS head-direction Value-path 标量的正贡献。层和 head 采用首、中、末位置作展示，不按效果排名。底色到亮色表示该坐标显示尺度下的正贡献。示例为先前冻结的 order 44 中位效应样本。网格标记输入 patch 位置；这些地图的查询目标是 CLS，不是各 patch-query 的独立归因。Figure 2 解释对应关系，Figure 8–9 展示全部 head 的汇总。",
    2: "ViT 内部响应与 NNS 输入图的对应关系。上排展示 patch tokens 经过不同 Transformer block，每个 block 内具有多个 head；本图在 B0、B5、B11 都选择 H9 的 CLS 响应。下排每对图与其正上方的目标对应，分别显示正贡献和负贡献幅度。灰色虚线表示响应到解释图的对应关系，下排地图不是沿箭头彼此传播的中间结果。计算使用固定 clean attention 和方向 u 的 Value-path 标量 f，并对参考输入到原图的路径积分；公式中的 i 是输入标量坐标，绘图时按带符号 RGB 和及 4×4 cells 聚合。这里借鉴原 NNS 为内部单元配置输入位置图的构图，ViT 的实际计算为条件路径积分。原来的详细目标/基线对照图另附为补充。",
    8: "中位效应图像的全头空间汇总。使用用户指定的 F3H Image 19 所对应的同一数据：order 44、ILSVRC2012_val_00002287.JPEG。左至右为原图、NNS 正贡献汇总、同目标 token baseline 正贡献汇总、NNS 负贡献幅度汇总及 baseline 负贡献幅度汇总。每个 B/H 坐标的两种方法先共用最大绝对值 L，采用 log1p(999·m/L)/log(1000) 并按原 F3H 规则线性放大，再对全部 12×12=144 个 head 等权平均。图像“中位效应”标签来自主实验 B0/H9 的冻结选样，不是对平均图的 fidelity 评价；平均图描述空间支持，不是类别归因。原始 Image 19 也逐字节收录在 reference_gallery。",
    9: "同一中位效应样本在四个深度段的空间支持。列为 B0–2、B3–5、B6–8、B9–11；行依次为 NNS 正、token baseline 正、NNS 负贡献幅度及 token baseline 负贡献幅度。每列使用该段全部 3×12=36 个 head，沿用 F3H Image 18 的逐坐标配对色阶和显示数组平均。该图用来观察深度与空间形态的关系，颜色不表示跨层可比的原始幅值，也不把这些阶段预先命名为边缘、部件或语义阶段。",
    10: "同一输入的多种解释视图。完整保留 F3C visual 第 1 页的五张样本，按冻结 formal positions 1–5 排列。列为原图、NNS 原生正贡献、NNS 4×4 最大池化后双线性显示、Grad-CAM、full-model IG 正贡献、full IG 的相同池化显示、rollout 和 raw CLS attention。NNS 在这里以条件 class logit 为目标，与 Figure 1、8、9 的单 head 目标不同。每个热图分别按自身 P99 缩放，池化仅用于显示；方法定量比较见 Figure 11，完整 50 图及 signed/unclipped 对照附在画廊。",
}
CAPTIONS_EN = {
    1: "NNS scans internal ViT responses. One image is embedded as patch tokens and processed by a trained ViT. Measured positive input-contribution maps are arranged by block (B0, B5, B11) and head (H0, H5, H11). Each panel targets that coordinate's CLS head-direction Value-path scalar. First, middle and last indices provide a display subset, without performance ranking. The input grid marks patch locations; the recorded targets are CLS queries, not separate patch-query attributions. The previously frozen order-44 median-effect image is used throughout. Figures 2 and 8–9 explain the mapping and the all-head summaries.",
    2: "An internal response and its NNS image. Above, patch tokens pass through transformer blocks, each containing multiple heads. H9 is selected at B0, B5 and B11. Below, measured positive and negative-magnitude input maps correspond to the response directly above them. Dashed lines denote correspondence, not propagation between the displayed maps. For each target, clean attention and direction u remain fixed while input-path integration produces signed attribution. Index i denotes an input scalar; channels and 4×4 cells are summed with sign for the displayed maps. The layout follows the original NNS unit-to-input-image concept; the ViT implementation uses conditional path integration. The previous detailed target/baseline diagram is retained as a supplement.",
    8: "All-head spatial support for the frozen median-effect image, order 44 (ILSVRC2012_val_00002287.JPEG). The original image is followed by NNS positive, target-matched token-baseline positive, NNS negative-magnitude and baseline negative-magnitude summaries. At each block/head coordinate, both methods share the maximum absolute value L; each component uses log1p(999m/L)/log(1000), the original F3H linear enlargement, and an equal average over all 144 display maps. These arrays reproduce F3H Image 19. Median-effect selection refers to the original B0/H9 primary endpoint, not to the fidelity of the consensus. The consensus describes spatial support, not class attribution. The original image export is included unchanged in reference_gallery.",
    9: "Spatial support across four depth bands for the same frozen image. Columns show B0–2, B3–5, B6–8 and B9–11; rows show NNS positive, token-baseline positive, NNS negative magnitude and baseline negative magnitude. Each band averages all 36 head display maps under the original F3H pair-shared scales, reproducing Image 18. Colors describe within-coordinate display support rather than raw cross-layer magnitude. The bands are indexed by depth without assigning untested semantic stages.",
    10: "Explanation views of the same inputs. All five images from F3C visual primary page 1 are retained in frozen formal-position order. Columns show the original, native positive NNS, 4×4 max-pooled/bilinear NNS, Grad-CAM, positive full-model IG, equally pooled full IG, rollout and raw CLS attention. NNS here targets a conditional class logit, unlike the single-head targets in Figures 1, 8 and 9. Each panel uses its own P99 scale; pooling is display-only. Figure 11 reports quantitative comparisons. All 50 inputs and signed/unclipped companion pages are included in the gallery.",
}
ANALYSIS = {
    1: "先看左边：同一张图被分成 patch，送进 ViT。再看右边：每一列换一层，每一行换一个 head，NNS 给每个内部响应配了一张能对回原图位置的图。\n\n这就是首图要讲的价值：我们不只看 ViT 最后给出的类别，还能打开它，查看不同层、不同 head 的响应与输入哪些位置有关。亮点是这些响应的正贡献位置，不是 attention 权重。这里只画了九个真实例子，完整 144-head 图在附带画廊。\n\n图里的 patch 网格说明输入位置；已有数据测的是 CLS 的 head 响应，并没有把每个 patch token 都单独算一张解释图。",
    2: "把上下两排一一对应起来看。上面是 ViT 内部选中的响应，下面是它对应的两张输入图：一张看哪些位置在推高它，一张看哪些位置在压低它。换层，下面的图也会变。\n\nNNS 在这里可以理解成“给内部响应配一张能看见输入位置的图”。图底下的公式交代本项目怎样算这张图：选好目标后，从参考图走向原图，累计每个输入位置对目标的影响。它不是重新训练 ViT，也不是把 attention 矩阵直接涂成热图。原论文 CNN 的算法和这里的 ViT 路径积分有区别。",
    8: "这张就是你喜欢的全 144-head 汇总视图。它仍然是原来的中位效应样本，只是不再只展示最浅层的一个 head，而是把全部 12 层、每层 12 个 head 的显示图汇总起来。\n\n从左到右看原图、NNS 正向、baseline 正向、NNS 负向、baseline 负向。可以对照按键、显示屏、面板边界附近的空间纹理；“正、负”说的是对各自内部响应的作用方向。\n\n这里的汇总是把显示图平均起来看空间形态，不是说 144 个 head 的类别贡献已经加总完成。“中位效应”仍然指这张图在原主实验里的位置。",
    9: "这张图把 Figure 8 的全头汇总拆成四段：浅层、两段中间层、深层。每一列都包含三层的全部 head，所以能看到整体形态怎样随深度变化。\n\n先横着看第一行，再看第三行，分别观察正向和负向结构；第二、四行是同目标 token baseline。不要仅凭亮度说哪一层更重要，每个内部坐标的显示尺度不同。",
    10: "这张沿用你喜欢的 F3C 图库风格，并完整放入第一页的五张图，没有从中只留 NNS 好看的样本。每一行是一张输入，横着看不同方法给出的解释。\n\nNNS 和 full IG 各有两列：原图尺寸的细节版，以及 4×4 最大池化后的易读版。后者更容易看到成片的结构，但只是换了显示方式，没有提高实验分数。Grad-CAM、rollout、raw CLS 也都保留。\n\n这里问的是类别输出，前面全头图问的是内部 head 响应，两组图的目标不同。到底哪个方法分数更高，接着看 Figure 11。",
}


def text(axis, xpos, ypos, value, size=10, color=INK, weight="normal", ha="left"):
    return axis.text(xpos, ypos, value, fontsize=size, color=color, weight=weight,
                     ha=ha, va="center")


def canvas(width, height):
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set(xlim=(0, width), ylim=(0, height))
    axis.axis("off")
    return figure, axis


def arrow(axis, start, end, color=MUTED, dashed=False):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                  lw=1.3, color=color, linestyle="--" if dashed else "-"))


def box(axis, xpos, ypos, width, height, face=LIGHT, edge="none"):
    axis.add_patch(FancyBboxPatch((xpos, ypos), width, height,
                                 boxstyle="round,pad=0.02,rounding_size=0.09",
                                 facecolor=face, edgecolor=edge, lw=1.2))


def image_tile(axis, array, xpos, ypos, width, cmap=None, edge=None):
    axis.imshow(array, extent=(xpos, xpos + width, ypos, ypos + width),
                cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                interpolation="nearest", aspect="auto")
    if edge:
        axis.add_patch(Rectangle((xpos, ypos), width, width, fill=False, ec=edge, lw=1.3))


def array_digest(array):
    source = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(source.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def load_head_evidence():
    manifest = SOURCES.json(base.F3 / "panel_manifest.json")
    base.verify_signed_json(manifest, "F3 panel manifest")
    case = next(row for row in manifest["panels"]
                if row.get("selection_order") == 44 and row.get("sources"))
    if case["selection_rules"] != ["median_effect"] or case["coordinate"] != [0, 9]:
        raise base.RenderError("Median-effect sample binding mismatch")
    photo_path = REPO / case["jpeg"]["path"]
    base.require_sha(photo_path, case["jpeg"]["file_sha256"], "Median-effect image")
    photo = base.crop_224(photo_path)
    if array_digest(photo.astype(np.float32) / np.float32(255.0)) != case["jpeg"]["display_array_sha256"]:
        raise base.RenderError("Frozen F3 image crop digest mismatch")
    inventory = SOURCES.json(base.F3H / "signed_map_inventory.json")
    base.verify_signed_json(inventory, "F3H maps")
    rows = {(row["block"], row["head"], row["method"]): row for row in inventory["maps"]}
    if len(rows) != 288:
        raise base.RenderError("Expected all 288 NNS/baseline maps")
    displays = {}
    for block in range(12):
        for head in range(12):
            raw = {}
            for method in (base.NNS_F3, base.MATCHED_F3):
                record = rows[(block, head, method)]
                path = REPO / record["source_path"]
                base.require_sha(path, record["source_file_sha256"], "F3H source map")
                raw[method] = SOURCES.array(path)
                if raw[method].shape != (56, 56) or raw[method].dtype != np.float32:
                    raise base.RenderError("F3H map contract")
                if array_digest(raw[method]) != record["signed_map_sha256"]:
                    raise base.RenderError("F3H array digest")
            limit = max(float(np.abs(value).max()) for value in raw.values())
            for method, prefix in ((base.NNS_F3, "nns"), (base.MATCHED_F3, "matched")):
                for component, sign in (("positive", 1), ("negative", -1)):
                    magnitude = np.maximum(sign * raw[method], np.float32(0))
                    shaped = ((np.log1p(999.0 * magnitude.astype(np.float64) / limit)
                               / math.log1p(999.0)).astype(np.float32)
                              if limit else np.zeros((56, 56), dtype=np.float32))
                    displays[(block, head, prefix + "_" + component)] = np.ascontiguousarray(
                        zoom(shaped, (4.0, 4.0), order=1, mode="nearest", prefilter=False).astype(np.float32))
    consensus = SOURCES.json(base.F3H / "equal_head_consensus_manifest.json")
    base.verify_signed_json(consensus, "F3H consensus")
    checked_groups = []
    for group in consensus["groups"]:
        prefix = "nns" if group["method"] == base.NNS_F3 else "matched"
        component = "positive" if group["component"] == "positive" else "negative"
        values = [displays[(block, head, prefix + "_" + component)]
                  for block, head in group["coordinates"]]
        averaged = np.mean(np.stack(values), axis=0, dtype=np.float64).astype(np.float32)
        if array_digest(averaged) != group["consensus_display_sha256"]:
            raise base.RenderError("Original F3H display parity: " + group["group_id"])
        checked_groups.append({key: group[key] for key in
                               ("group_id", "method", "component", "consensus_display_sha256")})
    metadata = {
        "selection_order": 44, "sample_id": case["sample_id"],
        "selection_rule": "median_effect", "selection_endpoint_coordinate": [0, 9],
        "primary_case_delta_fid": case["delta_fid"],
        "image": case["jpeg"], "map_count": len(rows),
        "consensus_groups_checked": checked_groups,
        "figure_1_blocks": list(BLOCKS), "figure_1_heads": list(HEADS),
        "figure_2_coordinates": [[block, 9] for block in BLOCKS],
        "patch_query_maps_available": False, "target_query": "CLS",
        "display": {"kappa": 999.0, "pair_scale": "union_max_abs",
                    "resize": "scipy_zoom_order1_mode_nearest", "clipping": None,
                    "consensus": "equal_mean_of_display_arrays_not_class_attribution"},
    }
    return photo, displays, metadata


def average(displays, blocks, component):
    return np.mean(np.stack([displays[(block, head, component)]
                             for block in blocks for head in range(12)]),
                   axis=0, dtype=np.float64).astype(np.float32)


def figure_01(photo, displays, cmap):
    figure, axis = canvas(12.0, 6.05)
    text(axis, .30, 5.70, TITLES[1], 20, weight="bold")
    text(axis, .30, 5.28, "Which input regions contribute to different layers and attention heads?",
         11, color=MUTED)
    image_tile(axis, photo, .35, 2.18, 2.12)
    for index in range(1, 14):
        axis.plot([.35 + 2.12 * index / 14] * 2, [2.18, 4.30],
                  color="white", lw=.4, alpha=.60)
        axis.plot([.35, 2.47], [2.18 + 2.12 * index / 14] * 2,
                  color="white", lw=.4, alpha=.60)
    text(axis, 1.41, 1.88, "One input image", 11, weight="bold", ha="center")
    text(axis, 1.41, 1.53, "14 × 14 patch tokens", 9, color=MUTED, ha="center")
    arrow(axis, (2.56, 3.20), (3.03, 3.20))
    for offset, color in zip((.24, .12, 0), reversed(COLORS)):
        box(axis, 3.14 + offset, 2.34 + offset, 1.18, 1.86, face="#F5F8FA", edge=color)
    text(axis, 3.73, 3.66, "ViT", 17, weight="bold", ha="center")
    for index in range(12):
        axis.add_patch(Circle((3.40 + (index % 3) * .33, 3.20 - (index // 3) * .18),
                              .050, color=COLORS[index % 3]))
    text(axis, 3.83, 1.89, "12 blocks", 10, ha="center")
    text(axis, 3.83, 1.55, "12 heads / block", 9, color=MUTED, ha="center")
    arrow(axis, (4.60, 3.20), (5.02, 3.20), BLUE)
    box(axis, 5.08, 2.60, 1.13, 1.18, face=BLUE)
    text(axis, 5.64, 3.38, "NNS", 18, color="white", weight="bold", ha="center")
    text(axis, 5.64, 2.96, "scan", 11, color="white", ha="center")
    arrow(axis, (6.30, 3.20), (6.95, 3.20), BLUE)
    for column, block in enumerate(BLOCKS):
        xpos = 7.35 + column * 1.48
        text(axis, xpos + .61, 4.83, f"Block {block}", 10, weight="bold", ha="center")
        for row, head in enumerate(HEADS):
            ypos = 3.38 - row * 1.37
            image_tile(axis, displays[(block, head, "nns_positive")], xpos, ypos, 1.22, cmap)
            if column == 0:
                text(axis, 7.16, ypos + .61, f"H{head}", 9, color=MUTED, ha="right")
    text(axis, .35, .80, "One input. Multiple internal views.", 13, color=BLUE, weight="bold")
    text(axis, .35, .42, "Each NNS image keeps the input's spatial coordinates.", 10, color=MUTED)
    text(axis, 9.43, .25, "Measured positive support · same frozen image", 9, color=MUTED, ha="center")
    return figure


def figure_02(photo, displays, cmap):
    figure, axis = canvas(12.0, 6.70)
    text(axis, .30, 6.35, "What is an NNS image in a ViT?", 20, weight="bold")
    text(axis, .30, 5.94, "An input-space contribution map paired with a chosen internal response.",
         11, color=MUTED)
    text(axis, .32, 5.42, "ViT computation", 11, weight="bold")
    image_tile(axis, photo, .40, 3.90, 1.18)
    for index in range(1, 14):
        axis.plot([.40 + 1.18 * index / 14] * 2, [3.90, 5.08], lw=.3, color="white", alpha=.6)
        axis.plot([.40, 1.58], [3.90 + 1.18 * index / 14] * 2, lw=.3, color="white", alpha=.6)
    text(axis, .99, 3.63, "Patch tokens + CLS", 8.8, color=MUTED, ha="center")
    arrow(axis, (1.70, 4.46), (2.58, 4.46))
    for column, block in enumerate(BLOCKS):
        xpos = 2.72 + column * 3.08
        color = COLORS[column]
        box(axis, xpos, 3.87, 2.32, 1.22, face="#F7F9FB", edge=color)
        text(axis, xpos + 1.16, 4.83, f"Block {block}", 11, weight="bold", ha="center")
        for head in range(12):
            center = (xpos + .30 + (head % 6) * .34, 4.49 - (head // 6) * .30)
            axis.add_patch(Circle(center, .105, fc=color if head == 9 else "#DCE5ED", ec="none"))
            if head == 9:
                text(axis, center[0], center[1], "9", 7.5, color="white", ha="center")
        if column < 2:
            arrow(axis, (xpos + 2.40, 4.46), (xpos + 2.93, 4.46))
            text(axis, xpos + 2.66, 4.76, "···", 12, color=MUTED, ha="center")
        arrow(axis, (xpos + 1.16, 3.76), (xpos + 1.16, 3.04), color, dashed=True)
        text(axis, xpos + 1.56, 3.42, "CLS · H9", 8.5, color=color)
        for offset, component, label in ((0, "nns_positive", "+"), (1.21, "nns_negative", "−")):
            image_tile(axis, displays[(block, 9, component)], xpos + offset, 1.84, 1.11, cmap, color)
            text(axis, xpos + offset + .56, 1.58, label, 13, color=color, weight="bold", ha="center")
    text(axis, .32, 2.65, "NNS image", 11, color=BLUE, weight="bold")
    text(axis, .32, 2.26, "Same input", 10, color=MUTED)
    text(axis, .32, 1.97, "coordinates", 10, color=MUTED)
    axis.plot([.32, 11.64], [1.26, 1.26], color="#DCE5ED", lw=1)
    text(axis, .35, .96, "Choose a response", 9, color=MUTED)
    text(axis, .35, .52, r"$f_{\ell,h}(z)=u^\top\sum_j A_{\ell,h}(x)_{\mathrm{CLS},j}V_{\ell,h}(z)_j$", 12)
    arrow(axis, (5.21, .70), (5.82, .70), BLUE)
    text(axis, 6.10, .96, "Trace each input coordinate along the reference-to-image path", 9, color=MUTED)
    text(axis, 6.10, .52, r"$R_i=(x_i-x_{0i})\int_0^1\partial_i f_{\ell,h}(x_0+\alpha(x-x_0))\,d\alpha$", 12)
    return figure


def figure_08(photo, displays, cmap):
    figure, axes = plt.subplots(1, 5, figsize=(11.6, 3.12))
    figure.subplots_adjust(left=.015, right=.99, bottom=.16, top=.78, wspace=.07)
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(photo)
    axes[0].set_title("Original", fontsize=11, weight="bold")
    for axis, component, label in zip(axes[1:], COMPONENTS,
                                      ("NNS +", "Token baseline +", "NNS −", "Token baseline −")):
        axis.imshow(average(displays, range(12), component), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(label, fontsize=11, weight="bold")
    figure.suptitle(TITLES[8], fontsize=17, weight="bold", y=.97)
    figure.text(.5, .055, "Same frozen median-effect sample · equal average of all 144 head display maps",
                fontsize=10, color=MUTED, ha="center")
    return figure


def figure_09(photo, displays, cmap):
    figure, axes = plt.subplots(4, 4, figsize=(9.4, 8.7))
    figure.subplots_adjust(left=.16, right=.99, bottom=.045, top=.90, wspace=.075, hspace=.10)
    for column, blocks in enumerate(((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))):
        for row, component in enumerate(COMPONENTS):
            axis = axes[row, column]
            axis.imshow(average(displays, blocks, component), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"Blocks {blocks[0]}–{blocks[-1]}", fontsize=12, weight="bold", pad=9)
            if column == 0:
                label = ("NNS +", "Token\nbaseline +", "NNS −", "Token\nbaseline −")[row]
                axis.text(-.10, .5, label, transform=axis.transAxes,
                          ha="right", va="center", fontsize=11, weight="bold")
    figure.suptitle(TITLES[9], fontsize=18, weight="bold", y=.974)
    figure.text(.56, .018, "36 heads in every band · same frozen input", fontsize=10, color=MUTED, ha="center")
    return figure


def figure_10(cmap):
    formal = SOURCES.json(base.F3C3 / "formal_manifest.json")["samples"]
    samples = [next(row for row in formal if row["formal_position"] == position)
               for position in range(1, 6)]
    source_manifest = SOURCES.json(C_REPORT / "render_manifest.json")
    source_page = next(row for row in source_manifest["records"]
                       if row["page"] == 1 and "/primary_pages/" in row["png"]["path"])
    source_scales = {(row["formal_position"], row["panel_id"]): row
                     for row in source_page["scale_records"]}
    panel_ids = ("nns_native_positive", "nns_pool4_bilinear_positive", "grad_cam_native",
                 "full_ig_native_positive", "full_ig_pool4_bilinear_positive",
                 "rollout_native", "raw_cls_attention_native")
    verified_scales = []
    figure, axes = plt.subplots(5, 8, figsize=(13.6, 9.1))
    figure.subplots_adjust(left=.04, right=.995, bottom=.055, top=.90, wspace=.07, hspace=.10)
    for row_index, sample in enumerate(samples):
        order = sample["selection_order"]
        path = REPO / "assets/data/ILSVRC2012_val" / sample["relative_path"]
        base.require_sha(path, sample["image_sha256"], "F3C input")
        original = base.crop_224(path)
        nns = base.load_f3c_array(order, base.NNS_C)
        full = base.load_f3c_array(order, base.FULL_IG)
        values = [base.positive_map(nns), base.pooled_positive(nns),
                  base.positive_map(base.load_f3c_array(order, base.GRADCAM)),
                  base.positive_map(full), base.pooled_positive(full),
                  base.positive_map(base.load_f3c_array(order, base.ROLLOUT)),
                  base.positive_map(base.load_f3c_array(order, base.RAW_ATTN))]
        axes[row_index, 0].imshow(original)
        axes[row_index, 0].text(-.07, .5, f"{sample['formal_position']:02d}",
                                transform=axes[row_index, 0].transAxes,
                                ha="right", va="center", fontsize=10, color=MUTED)
        for column, value in enumerate(values, 1):
            scale = source_scales[(sample["formal_position"], panel_ids[column - 1])]
            limit = float(np.percentile(value, 99.0))
            if (limit != scale["vmax_raw"] or float(np.max(value)) != scale["raw_maximum"]
                    or int(np.count_nonzero(value > limit)) != scale["clipped_pixel_count"]):
                raise base.RenderError("Original F3C scale parity mismatch")
            verified_scales.append(scale)
            axes[row_index, column].imshow(base.normalize_p99(value), cmap=cmap,
                                           vmin=0, vmax=1, interpolation="bilinear")
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    labels = ("Original", "NNS", "NNS\n4×4 pooled", "Grad-CAM",
              "Full IG", "Full IG\n4×4 pooled", "Rollout", "Raw CLS\nattention")
    for axis, label in zip(axes[0], labels):
        axis.set_title(label, fontsize=11, weight="bold", pad=9)
    figure.suptitle(TITLES[10], fontsize=19, weight="bold", y=.983)
    figure.text(.5, .021, "All five inputs from F3C page 1 · positive support · per-panel P99 display",
                fontsize=10, color=MUTED, ha="center")
    metadata = {"formal_positions": list(range(1, 6)),
                "selection_orders": [sample["selection_order"] for sample in samples],
                "selection_rule": "complete_original_primary_page_01",
                "column_labels": list(labels), "display_only_pooling": True,
                "original_scale_records_verified": verified_scales,
                "scale": "per_panel_P99", "target": "conditional_class_logit"}
    return figure, metadata


def copy_references():
    render = SOURCES.json(base.F3H / "render_inventory.json")
    base.verify_signed_json(render, "F3H render inventory")
    source_hashes = {row["outputs"]["png"]["sha256"] for row in render["renders"]}
    records = []
    paths = [(path, "f3h") for path in sorted((H_REPORT / "figures_png").glob("*.png"))]
    for folder in ("primary_pages", "signed_pages", "unclipped_pages"):
        paths.extend((path, "f3c_" + folder) for path in sorted((C_REPORT / folder).glob("*.png")))
    c_manifest = SOURCES.json(C_REPORT / "figure_sha256_manifest.json")
    c_hashes = set()

    def collect_hashes(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "sha256":
                    c_hashes.add(item)
                else:
                    collect_hashes(item)
        elif isinstance(value, list):
            for item in value:
                collect_hashes(item)

    collect_hashes(c_manifest)
    for source, group in paths:
        observed = base.sha256_file(source)
        if observed not in (source_hashes if group == "f3h" else c_hashes):
            raise base.RenderError("Reference image missing from its source inventory: " + str(source))
        SOURCES.add(source, "unchanged_reference_gallery_image")
        destination = ROOT / "reference_gallery" / group / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append({"group": group, "source_path": source.relative_to(REPO).as_posix(),
                        "path": destination.relative_to(ROOT).as_posix(), "sha256": observed,
                        "bytes": source.stat().st_size})
    old = SOURCES.json(OLD / "render_manifest.json")
    for output in old["figures"][1]["outputs"]:
        source = REPO / output["path"]
        base.require_sha(source, output["sha256"], "Detailed method supplement")
        SOURCES.add(source, "previous_method_detail")
        destination = ROOT / "supplement" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        records.append({"group": "method_detail", "source_path": output["path"],
                        "path": destination.relative_to(ROOT).as_posix(), "sha256": output["sha256"],
                        "bytes": output["bytes"]})
    return records


def write_docs(figures, references):
    for language, replacements in (("zh", CAPTIONS_ZH), ("en", CAPTIONS_EN)):
        path = OLD / f"captions_{language}.md"
        SOURCES.add(path, "inherited_captions")
        sections = common.sections(path, "## Figure ")
        sections.update(replacements)
        lines = [f"# NNS ViT R7 · {language}", ""]
        for number in range(1, 15):
            lines += [f"## Figure {number}", "", sections[number], ""]
        (ROOT / f"captions_{language}.md").write_text("\n".join(lines), encoding="utf-8")
    SOURCES.add(OLD / "figure_analysis.md", "inherited_plain_language_analysis")
    sections = common.sections(OLD / "figure_analysis.md", "## Image #")
    sections.update(ANALYSIS)
    lines = ["# R7：这 14 张图分别在说什么", "",
             "先看 Figure 1–2 理解 NNS 怎样查看 ViT，再看 8–10 的实际图像；4–7、11–14 放实验结果。", ""]
    index = ["# NNS ViT R7 · Figure 1–14", "", "[完整画廊](gallery.html) · [逐图说明](figure_analysis.md)", "",
             "| Figure | 内容 | 版本 | PNG | PDF | SVG |", "|---:|---|---|---|---|---|"]
    cards = []
    for record in figures:
        number = record["number"]
        lines += [f"## Image #{number}: {record['title']}", "", sections[number], ""]
        links = {output["format"]: Path(output["path"]).relative_to(ROOT.relative_to(REPO)).as_posix()
                 for output in record["outputs"]}
        revision = "R7 重画" if number in CHANGED else "原样保留 R6"
        index.append(f"| {number} | {record['title']} | {revision} | " +
                     " | ".join(f"[{extension}]({links[extension]})" for extension in ("PNG", "PDF", "SVG")) + " |")
        cards.append(f'<article id="figure-{number}"><h2>Figure {number} · {html.escape(record["title"])}</h2>'
                     f'<p>{revision}</p><a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}" '
                     f'alt="Figure {number}"></a><p>' +
                     " · ".join(f'<a href="{links[extension]}">{extension}</a>' for extension in ("PNG", "PDF", "SVG"))
                     + "</p></article>")
    (ROOT / "figure_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "figure_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    ref_cards = []
    for record in references:
        if record["path"].endswith(".png"):
            ref_cards.append(f'<details><summary>{html.escape(record["group"])} · {Path(record["path"]).name}</summary>'
                             f'<a href="{record["path"]}"><img loading="lazy" src="{record["path"]}" '
                             'alt="Original source figure"></a></details>')
    style = ("body{margin:0;background:#edf2f6;color:#172B3A;font:16px/1.6 system-ui}"
             "header{background:#172B3A;color:white;padding:26px 4vw}nav a{color:#b9def7;margin-right:16px}"
             "main{max-width:1600px;margin:auto;padding:24px}article,details{padding:22px;margin-bottom:22px;background:white}"
             "h1{margin:0;font-size:28px}h2{font-size:21px}img{width:100%;height:auto}a{color:#1875A5}summary{cursor:pointer}")
    gallery = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<title>NNS ViT R7</title><style>{style}</style><header><h1>NNS 扫描 ViT · R7</h1>'
               '<p>层、head 与输入位置的对应关系。完整 Figure 1–14，另附两个原始图库。</p><nav>' +
               " ".join(f'<a href="#figure-{number}">{number}</a>' for number in range(1, 15)) +
               '<a href="#references">原始图库</a></nav></header><main>' + "\n".join(cards) +
               '<h2 id="references">你选中的原始图库 · 逐字节保留</h2>' + "\n".join(ref_cards) + "</main></html>")
    (ROOT / "gallery.html").write_text(gallery, encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# NNS ViT unified paper figures R7\n\n"
        "[完整画廊](gallery.html) · [Figure 1–14 索引](figure_index.md) · [说人话逐图说明](figure_analysis.md)\n\n"
        "重画 Figure 1、2、8、9、10，另九张原样保留。每张均有 PNG/PDF/SVG，共 42 个主图文件。\n"
        "Figure 1 讲 NNS 扫描 ViT；Figure 2 讲内部响应和输入图的对应关系；"
        "Figure 8 换成同一中位效应样本的全 144-head 汇总；Figure 9、10 使用 F3H/F3C 图库的原显示规则。\n\n"
        "reference_gallery 收录 F3H 全部 20 张和 F3C 全部 30 页原 PNG。"
        "supplement 保留原详细目标/基线计算图的三种格式。历史弱效应、误分类案例仍在 "
        "[R6 Figure 8](../nns_vit_unified_paper_figures_r6_20260916/figures_png/figure_08_frozen_case_atlas.png)。\n\n"
        "本包仅 CPU 制图，不运行模型；全头显示平均与 class-logit 归因分别标注。"
        "中位效应样本身份不变，全部统计图和数值不变。\n\n"
        "入口：scripts/render_nns_vit_unified_paper_figures_r7.py；"
        "验证：scripts/validate_nns_vit_unified_paper_figures_r7.py。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    if ROOT.exists() and ((ROOT / "terminal_receipt.json").exists() or not args.draft):
        raise base.RenderError("Only unsealed R7 drafts may be rendered again")
    predecessor = common.inventory(OLD)
    old_manifest = SOURCES.json(OLD / "render_manifest.json")
    base.verify_signed_json(old_manifest, "R6 render manifest")
    for name in ("terminal_receipt.json", "manual_visual_qc.json", "validation_report.json"):
        value = SOURCES.json(OLD / name)
        base.verify_signed_json(value, "R6 " + name)
        if not value["status"].startswith("PASS"):
            raise base.RenderError("R6 predecessor has not passed")
    config = SOURCES.json(REPO / "config/nns_vit_unified_paper_figures_r4.json")
    cmap = base.configure(config)
    plt.rcParams["svg.hashsalt"] = "nns-vit-unified-paper-r7"
    base.OUTPUT = ROOT
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in (Path(__file__), Path(base.__file__), Path(common.__file__), DESIGN,
                 Path(__file__).with_name("validate_nns_vit_unified_paper_figures_r7.py")):
        SOURCES.add(path, "render_implementation_or_design")
    for name in ("fig-01.jpg", "fig-02.jpg"):
        path = REPO.parent / "相关文章/Understanding neural network through neuron level visualization/assets/figures" / name
        SOURCES.add(path, "composition_reference_not_embedded_as_experimental_data")
    SOURCES.add(C_REPORT / "display_parameters.json", "F3C_display_contract")
    SOURCES.add(REPO / "src/vitnns/stage_f_f3h.py", "original_F3H_display_definition")
    photo, displays, metadata = load_head_evidence()
    print("Verified 288 source maps and 68 original consensus digests", flush=True)
    figures = []
    layout = []
    for spec in old_manifest["figures"]:
        number = spec["number"]
        if number in CHANGED:
            if number == 10:
                figure, f3c_metadata = figure_10(cmap)
            else:
                figure = globals()[f"figure_{number:02d}"](photo, displays, cmap)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            outside = []
            for axis in figure.axes:
                for artist in axis.texts:
                    if not all(figure.bbox.contains(*point)
                               for point in artist.get_window_extent(renderer).get_points()):
                        outside.append(artist.get_text())
            if outside:
                raise base.RenderError("Text outside figure canvas: " + repr(outside))
            layout.append({"number": number, "text_outside_canvas": outside})
            outputs = base.save_figure(figure, number, spec["slug"])
        else:
            outputs = []
            for output in spec["outputs"]:
                source = REPO / output["path"]
                base.require_sha(source, output["sha256"], "Inherited R6 export")
                SOURCES.add(source, "unchanged_R6_figure")
                destination = ROOT / source.parent.name / source.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, destination)
                outputs.append({**output, "path": destination.relative_to(REPO).as_posix()})
        figures.append({**spec, "title": TITLES.get(number, spec["title"]), "revision": "R7" if number in CHANGED else "R6",
                        "redesigned": number in CHANGED, "outputs": outputs})
        print(f"Figure {number:02d}: {'redrawn' if number in CHANGED else 'copied'}", flush=True)
    references = copy_references()
    write_docs(figures, references)
    inherited = SOURCES.json(OLD / "numeric_display_data.json")
    base.verify_signed_json(inherited, "R6 numeric data")
    common.write(ROOT / "numeric_display_data.json", common.signed({
        "revision": "R7", "head_display": metadata, "f3c_display": f3c_metadata,
        "inherited_R6_numeric": inherited, "new_inference": False}))
    if common.inventory(OLD) != predecessor:
        raise base.RenderError("R6 predecessor changed during rendering")
    common.write(ROOT / "predecessor_snapshot.json", common.signed({
        "root": OLD.relative_to(REPO).as_posix(), "files": predecessor,
        "files_digest": common.canonical(predecessor)}))
    common.write(ROOT / "reference_gallery_manifest.json", common.signed({
        "records": references, "source_png_count": 50, "method_supplement_files": 3}))
    sources = SOURCES.inventory()
    common.write(ROOT / "input_inventory.json", common.signed({"source_files": sources}))
    common.write(ROOT / "render_manifest.json", common.signed({
        "revision": "R7", "status": "RENDERED_PENDING_VISUAL_REVIEW", "figure_count": 14,
        "artifact_count": 42, "redesigned_figures": CHANGED, "figures": figures,
        "source_files": sources, "source_files_digest": common.canonical(sources),
        "cpu_only": True, "new_inference": False}))
    common.write(ROOT / "layout_audit.json", common.signed({"status": "PASS", "records": layout}))
    common.write(ROOT / "cpu_access_audit.json", common.signed({
        "status": "PASS_CPU_ONLY_PRESENTATION", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_imported": "torch" in sys.modules, "timm_imported": "timm" in sys.modules,
        "sys_executable": str(Path(sys.executable).resolve()), "new_inference": False,
        "decoded_samples": ["F3 order 44", "F3C formal positions 1–5"],
        "figure_1_patch_query_maps_claimed": False}))
    print(f"R7 rendered: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
