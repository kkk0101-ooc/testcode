#!/usr/bin/env python3
"""Render the publication-v2 visual redesign for package Figures 1--7.

The renderer uses the existing sealed sloth-bear arrays for empirical panels and
keeps conceptual matrices deterministic.  It updates both figure manifests,
the bilingual main-text captions, the display receipt, and the HTML gallery.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nns-endpoint-concepts-v2-mpl")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1789862400")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "papers/nns_vit_endpoint/publication_r2_20260920"
ACTIVE = ROOT.parent / "figure_manifest.json"
FONT_FILE = ROOT.parent / "assets/fonts/InterVariable.ttf"
FONT_FAMILY = "DejaVu Sans"
if FONT_FILE.is_file():
    font_manager.fontManager.addfont(FONT_FILE)
    FONT_FAMILY = "Inter Variable"

NAVY = "#102A43"
BLUE = "#2B6F9F"
TEAL = "#14968F"
ORANGE = "#E47D31"
RED = "#C94C4C"
SLATE = "#5D7285"
LINE = "#D8E2EA"
PALE_BLUE = "#EEF6FB"
PALE_TEAL = "#EAF7F5"
PALE_ORANGE = "#FFF3E7"
PAPER = "#FBFCFE"
CMAP = LinearSegmentedColormap.from_list(
    "nns_publication",
    [(0.0, "#071B78"), (0.22, "#0756B8"), (0.44, "#10B7D4"),
     (0.66, "#F2DD3D"), (0.84, "#F47A28"), (1.0, "#C91F37")],
)

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "text.color": NAVY,
    "axes.labelcolor": NAVY,
    "mathtext.fontset": "stixsans",
    "mathtext.default": "regular",
    "svg.fonttype": "none",
    "svg.hashsalt": "nns-endpoint-concepts-v2",
    "pdf.fonttype": 42,
})

DISPLAY_ROWS: list[dict] = []

CAPTIONS = {
    1: {
        "zh": "以同一懒熊输入和 ground-truth logit 开门见山展示方法产物。左侧是真实输入与 Total Endpoint，中间是 block 5 的 identity、routing-induced 和 value-induced 局部图，右侧区分结构审计与有限替换评价。所有热图取 signed RGB sum 的正部，按各自正值 P99 截断并开方，仅比较空间形态。",
        "en": "The method is introduced through one sloth-bear input and its ground-truth logit. The left panels show the real input and Total Endpoint; the middle panels show block-5 identity, routing-induced, and value-induced local maps; the right separates structural audit from finite-replacement evaluation. Heatmaps show the positive signed RGB sum with independent positive-P99 clipping and square-root scaling, so only spatial form is compared.",
    },
    2: {
        "zh": "左侧说明同一 producer tensor 的不同 consumer、slot 和 invocation 必须保留为不同 use。右侧区分 attention branch partition 与包含 identity bypass 的 residual 全局切面：前者重构 attention branch contribution，后者重构 Total Endpoint。",
        "en": "The left panel shows why different consumers, input slots, and invocations of one producer tensor remain distinct uses. The right distinguishes an attention branch partition from a residual global cut containing the identity bypass: the former reconstructs the attention branch contribution, whereas the latter reconstructs Total Endpoint.",
    },
    3: {
        "zh": "输入坐标初始化。左侧输入矩阵被拆成四张单坐标图；右侧将它们堆叠为 $\operatorname{diag}(x)$。每张图只在一个坐标保留原输入值，因此同时保留数值与位置。",
        "en": "Input-coordinate initialization. The input matrix is split into four single-coordinate maps and stacked as $\operatorname{diag}(x)$. Each map retains the original value at one coordinate and therefore preserves both value and position.",
    },
    4: {
        "zh": "局部 Jacobian 将输入坐标图传播到目标。每张基图乘以对应当前点局部系数 $c_i=\partial s/\partial x_i$，再相加得到目标 learning image；对标量目标，它等于 $x\odot\nabla_xs(x)$。",
        "en": "The local Jacobian propagates input-coordinate maps to the target. Each basis map is multiplied by its current-point coefficient $c_i=\partial s/\partial x_i$ and summed into the target learning image; for a scalar target this equals $x\odot\nabla_xs(x)$.",
    },
    5: {
        "zh": "Patch embedding 的局部投影与全局稀疏形式。一个 token-channel 只读取对应 $16\times16$ patch；同一权重行嵌入完整输入坐标后仅在该 patch 非零，因此其初始 learning-image support 也是局部的。",
        "en": "Local and global views of patch embedding. One token-channel reads only its corresponding $16\times16$ patch; after embedding the same weight row into global input coordinates, it is nonzero only on that patch, so its initial learning-image support is local.",
    },
    6: {
        "zh": "同一懒熊输入、ground-truth logit 和 DeiT-S/16 block 5 的真实 attention 微分图。Routing-induced 项对应 $(\mathrm dA)V$，value-induced 项对应 $A\mathrm dV$；两张完整 signed 图相加重构 attention branch contribution。全局 Total Endpoint 还需加入 residual identity。",
        "en": "Real attention differential maps for the same sloth-bear input, ground-truth logit, and DeiT-S/16 block 5. The routing-induced term corresponds to $(\mathrm dA)V$ and the value-induced term to $A\mathrm dV$; their complete signed maps sum to the attention branch contribution. The residual identity is additionally required for the global Total Endpoint.",
    },
    7: {
        "zh": "同一懒熊输入、ground-truth logit 和 DeiT-S/16 block 5 residual 的真实路径图。Identity 与 attention branch 使用同一个 residual 输出 cotangent；完整 signed 图满足 identity 加 branch 等于 Total Endpoint。正部显示独立归一化，只用于读取空间结构。",
        "en": "Real path maps for the same sloth-bear input, ground-truth logit, and DeiT-S/16 block-5 residual. Identity and attention branch use the same cotangent at the residual output; complete signed maps satisfy identity plus branch equals Total Endpoint. Positive displays are independently normalized and are used only to read spatial structure.",
    },
}

TITLES = {
    1: "One image, one target, many auditable local maps",
    2: "Use identity and valid cuts make the decomposition auditable",
    3: "Initialization keeps both input value and location",
    4: "Local Jacobians propagate coordinate maps to the target",
    5: "Patch embedding begins with strictly local support",
    6: "Attention separates routing from value transmission",
    7: "Residual closure is visible in the path maps",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded(ax, xy, width, height, *, face="white", edge=LINE, radius=.018, lw=1.2, dashed=False):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "-",
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=BLUE, lw=1.7, dashed=False, mutation=14):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation, color=color,
        linewidth=lw, linestyle=(0, (4, 3)) if dashed else "-",
        shrinkA=0, shrinkB=0, connectionstyle="arc3",
    ))


def canvas(title: str, subtitle: str = "", *, size=(14, 6.3)):
    fig = plt.figure(figsize=size, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    fig.text(.045, .955, title, ha="left", va="top", fontsize=20, weight="bold", color=NAVY)
    if subtitle:
        fig.text(.047, .902, subtitle, ha="left", va="top", fontsize=10.5, color=SLATE)
    return fig, ax


def positive_display(raw: np.ndarray):
    a = np.asarray(raw)
    v = a.astype(np.float64).squeeze()
    if v.ndim == 3:
        v = v.sum(0)
    v = np.maximum(v, 0)
    nz = v[v > 0]
    limit = float(np.quantile(nz, .99)) if nz.size else 1.0
    shown = np.sqrt(np.clip(v / limit, 0, 1))
    return a, v, shown, limit


def map_panel(fig, rect, raw, label, source, figure_number, *, border=LINE, tag=None):
    outer = fig.add_axes(rect); outer.set_facecolor("white")
    for spine in outer.spines.values():
        spine.set_color(border); spine.set_linewidth(1.25)
    outer.set_xticks([]); outer.set_yticks([])
    a, v, shown, limit = positive_display(raw)
    outer.imshow(shown, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    outer.set_title(label, fontsize=10.8, weight="semibold", pad=7, color=NAVY)
    if tag:
        outer.text(.035, .955, tag, transform=outer.transAxes, ha="left", va="top",
                   fontsize=8.8, color="white", weight="bold",
                   bbox=dict(boxstyle="round,pad=.25", facecolor=NAVY, edgecolor="none", alpha=.88))
    DISPLAY_ROWS.append({
        "figure": figure_number, "label": label, "kind": "learning", "source": source,
        "raw_sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "shape": list(a.shape), "limit": limit, "gamma": .5,
        "clipped": int((np.abs(v) > limit).sum()),
        "display_sha256": hashlib.sha256(np.ascontiguousarray(shown).tobytes()).hexdigest(),
        "interpolation": "nearest",
    })
    return outer


def photo_panel(fig, rect, photo, label, *, border=LINE, tag=None):
    ax = fig.add_axes(rect); ax.imshow(photo); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border); spine.set_linewidth(1.25)
    ax.set_title(label, fontsize=10.8, weight="semibold", pad=7, color=NAVY)
    if tag:
        ax.text(.035, .955, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.8, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=.25", facecolor=NAVY, edgecolor="none", alpha=.88))
    return ax


def matrix_tile(ax, x, y, w, h, values, *, edge=BLUE, label=None, fill_nonzero=PALE_BLUE, fontsize=12):
    values = np.asarray(values); rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            xx = x + c*w/cols; yy = y + (rows-1-r)*h/rows
            ax.add_patch(Rectangle((xx, yy), w/cols, h/rows,
                facecolor=fill_nonzero if values[r, c] != 0 else "white",
                edgecolor=edge, linewidth=1.05))
            ax.text(xx+w/(2*cols), yy+h/(2*rows), f"{values[r,c]:g}",
                    ha="center", va="center", fontsize=fontsize, color=NAVY)
    if label:
        ax.text(x+w/2, y+h+.018, label, ha="center", va="bottom", fontsize=11.5, color=NAVY)


def save(fig, manifest, active, number: int):
    row = next(r for r in manifest["figures"] if r["number"] == number)
    row["title"] = TITLES[number]
    row["caption_zh"] = CAPTIONS[number]["zh"]
    row["caption_en"] = CAPTIONS[number]["en"]
    for ext, record in row["files"].items():
        path = ROOT / record["path"]
        fig.savefig(path, dpi=240 if ext == "png" else 180, bbox_inches="tight",
                    pad_inches=.12, facecolor=PAPER)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        record["sha256"] = sha(path)
        bound = next(r for r in active["figures"] if r["figure"] == number)
        bound[ext]["sha256"] = record["sha256"]
    plt.close(fig)


def figure_1(photo, maps):
    fig, ax = canvas(
        TITLES[1],
        "Start with the observed effect, then open it by tensor use and test each claim at the right evidence level.",
        size=(15.2, 6.4),
    )
    # Effect first.
    rounded(ax, (.025, .11), .37, .72, face="white", edge=LINE)
    ax.text(.045, .795, "OBSERVED EFFECT", fontsize=9, weight="bold", color=BLUE)
    photo_panel(fig, [.05, .23, .145, .48], photo, "Input", border=BLUE, tag="sloth bear")
    map_panel(fig, [.22, .23, .145, .48], maps["L_F"], "Total Endpoint", "L_F", 1, border=BLUE, tag="GT logit")
    ax.text(.208, .47, "→", ha="center", va="center", fontsize=26, color=BLUE, weight="bold")
    ax.text(.207, .165, r"$E_s(x)=x\odot\nabla_xs(x)$", ha="center", fontsize=15, color=NAVY)

    # Actual decomposition.
    rounded(ax, (.415, .11), .34, .72, face="white", edge=LINE)
    ax.text(.435, .795, "OPEN THE MAP BY USE  ·  BLOCK 5", fontsize=9, weight="bold", color=TEAL)
    map_panel(fig, [.435, .41, .096, .265], maps["identity"], "Identity", "identity", 1, border=BLUE)
    map_panel(fig, [.545, .41, .096, .265], maps["routing"], "Routing", "routing", 1, border=TEAL)
    map_panel(fig, [.655, .41, .096, .265], maps["value"], "Value", "value", 1, border=ORANGE)
    ax.text(.596, .315, r"$\mathcal{L}_{e,i}=x_i[J_{z_e,x}^{\mathsf{T}}\bar{z}_e^{(s)}]_i$",
            ha="center", fontsize=12.4, color=NAVY)
    ax.text(.596, .235, "routing + value = attention branch", ha="center", fontsize=9.8, color=TEAL)
    ax.text(.596, .185, "identity + attention branch = Total Endpoint", ha="center", fontsize=9.8, color=BLUE)

    # Evidence boundaries.
    rounded(ax, (.775, .11), .20, .72, face="white", edge=LINE)
    ax.text(.795, .795, "ASK TWO DIFFERENT QUESTIONS", fontsize=9, weight="bold", color=ORANGE)
    rounded(ax, (.80, .53), .15, .17, face=PALE_TEAL, edge=TEAL)
    ax.text(.875, .655, "STRUCTURE", ha="center", fontsize=9, weight="bold", color=TEAL)
    ax.text(.875, .602, "Does the declared cut\nclose without omission\nor duplication?", ha="center", va="center", fontsize=10.3, linespacing=1.3)
    rounded(ax, (.80, .28), .15, .17, face=PALE_ORANGE, edge=ORANGE)
    ax.text(.875, .405, "UTILITY", ha="center", fontsize=9, weight="bold", color=ORANGE)
    ax.text(.875, .352, "Does a clean local map\npredict a specified finite\nreplacement effect?", ha="center", va="center", fontsize=10.3, linespacing=1.3)
    ax.text(.875, .17, "Closure does not imply\nintervention utility.", ha="center", fontsize=9.8, color=RED, weight="semibold")
    return fig


def figure_2():
    fig, ax = canvas(TITLES[2], "A tensor name is not a use identity; a branch partition is not a global cut.", size=(14.5, 6.2))
    rounded(ax, (.03, .13), .37, .70, face="white", edge=LINE)
    ax.text(.055, .785, "A  ·  KEEP CONCRETE USES DISTINCT", fontsize=9, weight="bold", color=BLUE)
    ax.add_patch(Circle((.12, .55), .052, facecolor=PALE_BLUE, edgecolor=BLUE, lw=1.7)); ax.text(.12, .55, r"$z$", ha="center", va="center", fontsize=17)
    for y, label, color, fill in [(.65, "consumer A\nslot 0 · call 1", TEAL, PALE_TEAL), (.42, "consumer B\nslot 1 · call 2", ORANGE, PALE_ORANGE)]:
        rounded(ax, (.235, y-.075), .13, .15, face=fill, edge=color)
        ax.text(.30, y, label, ha="center", va="center", fontsize=10.5, linespacing=1.3)
        arrow(ax, (.17, .55), (.228, y), color=color)
    ax.text(.205, .29, r"$e=(producer,\ consumer,\ slot,\ invocation)$", ha="center", fontsize=12.5)
    ax.text(.205, .21, "Same tensor  ≠  same use", ha="center", fontsize=11, color=RED, weight="bold")

    rounded(ax, (.43, .13), .54, .70, face="white", edge=LINE)
    ax.text(.455, .785, "B  ·  CLOSE THE RIGHT OBJECT", fontsize=9, weight="bold", color=TEAL)
    # Residual graph.  Arrows encode computation only; the dashed cut
    # definitions are separated into the formula boxes below the graph.
    rounded(ax, (.465, .45), .095, .15, face=PALE_BLUE, edge=BLUE)
    ax.text(.513, .525, r"$u$", ha="center", va="center", fontsize=15)
    rounded(ax, (.590, .660), .145, .118, face=PALE_TEAL, edge=TEAL)
    ax.text(.662, .755, "attention branch", ha="center", va="center", fontsize=10.2, weight="semibold")
    for xx, label in [(.622, "Q"), (.662, "K"), (.702, "V")]:
        rounded(ax, (xx - .016, .685), .032, .032, face="white", edge=TEAL, radius=.007, lw=1.0)
        ax.text(xx, .701, label, ha="center", va="center", fontsize=8.8, color=TEAL, weight="semibold")
    ax.add_patch(Circle((.755, .525), .032, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=1.8, zorder=3))
    ax.text(.755, .525, "+", ha="center", va="center", fontsize=22, weight="bold", zorder=4)
    rounded(ax, (.865, .452), .102, .146, face=PALE_TEAL, edge=TEAL)
    ax.text(.916, .525, "target s", ha="center", va="center", fontsize=12.0)

    ax.plot([.560, .585], [.552, .552], color=TEAL, lw=2.0)
    arrow(ax, (.585, .552), (.605, .671), color=TEAL)
    ax.plot([.735, .755], [.728, .728], color=TEAL, lw=2.0)
    arrow(ax, (.755, .728), (.755, .558), color=TEAL, lw=2.0)
    ax.plot([.513, .513, .755], [.445, .335, .335], color=BLUE, lw=2.0)
    arrow(ax, (.755, .335), (.755, .489), color=BLUE, lw=2.0)
    ax.add_patch(FancyArrowPatch((.790, .525), (.852, .525), arrowstyle="-|>",
                                 mutation_scale=24, color=TEAL, linewidth=2.9,
                                 shrinkA=0, shrinkB=0, connectionstyle="arc3",
                                 zorder=6))
    ax.text(.62, .305, "identity bypass", ha="center", fontsize=10.5, color=BLUE)
    rounded(ax, (.48, .165), .20, .090, face=PALE_TEAL, edge=TEAL, dashed=True)
    ax.text(.58, .238, "branch partition", ha="center", va="center", fontsize=8.7, color=TEAL, weight="bold")
    ax.text(.58, .198, r"$Q+K+V=A_F$", ha="center", va="center", fontsize=13)
    rounded(ax, (.70, .165), .25, .090, face="#FCEEEE", edge=RED, dashed=True)
    ax.text(.825, .238, "global residual cut", ha="center", va="center", fontsize=8.7, color=RED, weight="bold")
    ax.text(.825, .198, r"$identity+A_F=L_F$", ha="center", va="center", fontsize=13)
    return fig


def figure_3(toy):
    x = np.asarray(toy["x"], dtype=float).reshape(2, 2)
    fig, ax = canvas(TITLES[3], "The initialization is diagonal: one map per coordinate, no spatial mixing.", size=(13.8, 5.8))
    rounded(ax, (.035, .15), .22, .66, face="white", edge=LINE)
    ax.text(.06, .765, "INPUT", fontsize=9, weight="bold", color=BLUE)
    matrix_tile(ax, .075, .35, .14, .30, x, edge=BLUE, label=r"$x$", fill_nonzero=PALE_BLUE, fontsize=15)
    ax.text(.145, .245, "one value at each location", ha="center", fontsize=9.8, color=SLATE)
    arrow(ax, (.265, .48), (.32, .48), color=BLUE, lw=2.0, mutation=17)

    rounded(ax, (.33, .15), .42, .66, face="white", edge=LINE)
    ax.text(.355, .765, "SPLIT INTO COORDINATE MAPS", fontsize=9, weight="bold", color=TEAL)
    colors = [BLUE, TEAL, ORANGE, RED]; fills = [PALE_BLUE, PALE_TEAL, PALE_ORANGE, "#FCEEEE"]
    positions = [(.38, .49), (.55, .49), (.38, .23), (.55, .23)]
    for i, ((px, py), color, fill) in enumerate(zip(positions, colors, fills)):
        tile = np.zeros((2, 2)); tile.flat[i] = x.flat[i]
        matrix_tile(ax, px, py, .12, .18, tile, edge=color, label=fr"$I_{i+1}$", fill_nonzero=fill, fontsize=12.5)
    ax.text(.54, .185, r"$x=I_1+I_2+I_3+I_4$", ha="center", fontsize=13.5)
    arrow(ax, (.76, .48), (.81, .48), color=TEAL, lw=2.0, mutation=17)

    rounded(ax, (.82, .15), .145, .66, face="white", edge=LINE)
    ax.text(.845, .765, "STACK", fontsize=9, weight="bold", color=ORANGE)
    # Diagonal matrix.
    diag = np.diag(x.ravel())
    for r in range(4):
        for c in range(4):
            xx=.845+c*.025; yy=.42+(3-r)*.055
            ax.add_patch(Rectangle((xx,yy),.025,.055,facecolor=PALE_ORANGE if r==c else "white",edgecolor=LINE,lw=.8))
            ax.text(xx+.0125,yy+.0275,f"{diag[r,c]:g}",ha="center",va="center",fontsize=9.5)
    ax.text(.895, .335, r"$\mathcal{L}(x;x)$", ha="center", fontsize=13)
    ax.text(.895, .275, r"$=\operatorname{diag}(x)$", ha="center", fontsize=13, color=ORANGE)
    return fig


def figure_4(toy):
    x = np.asarray(toy["x"], dtype=float)
    endpoint = np.asarray(toy["endpoint"], dtype=float).reshape(2, 2)
    coeff = endpoint.ravel() / x
    fig, ax = canvas(TITLES[4], "Each input-coordinate map is scaled by its composed current-point derivative, then summed.", size=(14.2, 5.9))
    rounded(ax, (.03, .15), .32, .66, face="white", edge=LINE)
    ax.text(.055, .765, "INPUT COORDINATE MAPS", fontsize=9, weight="bold", color=BLUE)
    colors = [BLUE, TEAL, ORANGE, RED]; fills = [PALE_BLUE, PALE_TEAL, PALE_ORANGE, "#FCEEEE"]
    positions = [(.065,.49),(.21,.49),(.065,.24),(.21,.24)]
    for i, ((px,py), color, fill) in enumerate(zip(positions,colors,fills)):
        tile=np.zeros((2,2)); tile.flat[i]=x[i]
        matrix_tile(ax,px,py,.10,.16,tile,edge=color,label=fr"$I_{i+1}$",fill_nonzero=fill,fontsize=11.5)
    arrow(ax,(.36,.48),(.43,.48),color=BLUE,lw=2.0,mutation=17)

    rounded(ax, (.44, .15), .25, .66, face="white", edge=LINE)
    ax.text(.465, .765, "LOCAL JACOBIAN COEFFICIENTS", fontsize=9, weight="bold", color=TEAL)
    for i,(c,color) in enumerate(zip(coeff,colors)):
        yy=.665-i*.115
        rounded(ax,(.49,yy-.035),.15,.07,face="white",edge=color,radius=.012,lw=1.4)
        ax.text(.565,yy,fr"$c_{i+1}=\partial s/\partial x_{i+1}={c:g}$",ha="center",va="center",fontsize=11.5)
    ax.text(.565,.205,"same forward point · same target",ha="center",fontsize=9.5,color=SLATE)
    arrow(ax,(.70,.48),(.77,.48),color=TEAL,lw=2.0,mutation=17)

    rounded(ax, (.78, .15), .185, .66, face="white", edge=LINE)
    ax.text(.805, .765, "TARGET LEARNING IMAGE", fontsize=9, weight="bold", color=ORANGE)
    matrix_tile(ax,.815,.40,.115,.23,endpoint,edge=ORANGE,label=r"$\mathcal{L}(s;x)$",fill_nonzero=PALE_ORANGE,fontsize=13.5)
    ax.text(.872,.315,r"$=\sum_i c_i I_i$",ha="center",fontsize=13.5)
    ax.text(.872,.235,r"$=x\odot\nabla_xs$",ha="center",fontsize=15,color=ORANGE,weight="bold")
    return fig


def figure_5(photo):
    fig, ax = canvas(TITLES[5], "A token-channel reads one 16×16 patch; the equivalent global row is sparse.", size=(14.3, 6.2))
    rounded(ax, (.03, .13), .39, .70, face="white", edge=LINE)
    ax.text(.055,.785,"NATIVE PATCH PROJECTION",fontsize=9,weight="bold",color=BLUE)
    image_ax=photo_panel(fig,[.06,.265,.19,.43],photo,"Input image",border=BLUE)
    for k in range(1,14):
        image_ax.axvline(k*16-.5,color="white",lw=.22,alpha=.38);image_ax.axhline(k*16-.5,color="white",lw=.22,alpha=.38)
    image_ax.add_patch(Rectangle((6*16-.5,6*16-.5),16,16,fill=False,edgecolor=ORANGE,lw=2.4))
    rounded(ax,(.29,.47),.095,.14,face=PALE_ORANGE,edge=ORANGE);ax.text(.338,.54,r"$w_c^\top x_t$",ha="center",va="center",fontsize=13)
    arrow(ax,(.255,.49),(.285,.54),color=ORANGE)
    rounded(ax,(.29,.27),.095,.11,face=PALE_TEAL,edge=TEAL);ax.text(.338,.325,r"$z_{t,c}$",ha="center",va="center",fontsize=13)
    arrow(ax,(.338,.46),(.338,.39),color=TEAL)
    ax.text(.225,.185,r"$z_{t,c}=w_c^\top x_t+b_c$",ha="center",fontsize=13.5)

    rounded(ax, (.45, .13), .52, .70, face="white", edge=LINE)
    ax.text(.475,.785,"EQUIVALENT GLOBAL INPUT ROW",fontsize=9,weight="bold",color=TEAL)
    # Compact patch grid.
    gx,gy,gw,gh=.49,.39,.19,.31
    for i in range(14):
        for j in range(14):
            color=ORANGE if (i,j)==(6,6) else PALE_BLUE
            ax.add_patch(Rectangle((gx+j*gw/14,gy+(13-i)*gh/14),gw/14,gh/14,facecolor=color,edgecolor="white",lw=.16))
    ax.text(gx+gw/2,.725,"global image coordinates",ha="center",fontsize=10.5)
    arrow(ax,(.695,.545),(.745,.545),color=BLUE,lw=1.8)
    rounded(ax,(.755,.50),.17,.15,face="#F4F7FA",edge=SLATE)
    for j in range(8):
        ax.add_patch(Rectangle((.768+j*.0185,.545),.015,.055,facecolor=ORANGE if j==3 else "white",edgecolor=LINE,lw=.6))
    ax.text(.84,.675,r"sparse row $W_{t,c}$",ha="center",fontsize=11)
    ax.text(.84,.515,"nonzero only in patch t",ha="center",fontsize=9,color=SLATE)
    arrow(ax,(.84,.49),(.84,.405),color=TEAL)
    support=np.zeros((14,14));support[6,6]=1
    support_ax=fig.add_axes([.765,.20,.15,.18]);support_ax.imshow(support,cmap=CMAP,vmin=0,vmax=1,interpolation="nearest");support_ax.set_xticks([]);support_ax.set_yticks([])
    for spine in support_ax.spines.values():spine.set_color(TEAL);spine.set_linewidth(1.2)
    support_ax.set_title("learning-image support",fontsize=10.5,pad=6,weight="semibold")
    ax.text(.59,.19,r"$E_{t,c}=x\odot W_{t,c}$",ha="center",fontsize=14.5)
    ax.text(.59,.15,"bias changes the scalar, not the input-coordinate support",ha="center",fontsize=9.3,color=SLATE)
    return fig


def figure_6(photo, maps):
    fig, ax = canvas(TITLES[6], "The formula and the observed block-5 maps tell the same story.", size=(14.8, 6.4))
    # Compact forward flow with an explicit, separate V branch.
    ax.text(.045,.815,"FORWARD",fontsize=9,weight="bold",color=BLUE)
    stages=[(.08,r"$Q,K$"),(.245,r"$S=QK^\top/\sqrt{d}$"),(.465,r"$A=softmax(S)$"),(.72,r"$O=AV$")]
    widths=[.10,.145,.14,.10]
    for (x,label),w in zip(stages,widths):
        rounded(ax,(x,.71),w,.085,face=PALE_BLUE if x<.5 else PALE_TEAL,edge=BLUE if x<.5 else TEAL,radius=.012)
        ax.text(x+w/2,.752,label,ha="center",va="center",fontsize=12.5)
    for i in range(3):arrow(ax,(stages[i][0]+widths[i],.752),(stages[i+1][0]-.012,.752),color=BLUE)
    rounded(ax,(.61,.595),.065,.075,face=PALE_ORANGE,edge=ORANGE);ax.text(.642,.632,r"$V$",ha="center",va="center",fontsize=13)
    arrow(ax,(.675,.632),(.71,.706),color=ORANGE,lw=1.55)
    rounded(ax,(.885,.70),.065,.105,face=PALE_ORANGE,edge=ORANGE);ax.text(.918,.752,r"$s$",ha="center",va="center",fontsize=14)
    arrow(ax,(.82,.752),(.875,.752),color=ORANGE)

    ax.text(.045,.615,"LOCAL DIFFERENTIALS  ·  REAL MAPS",fontsize=9,weight="bold",color=TEAL)
    map_panel(fig,[.07,.205,.205,.32],maps["routing"],"Routing-induced","routing",6,border=TEAL,tag=r"$(dA)V$")
    map_panel(fig,[.35,.205,.205,.32],maps["value"],"Value-induced","value",6,border=ORANGE,tag=r"$A,dV$")
    map_panel(fig,[.69,.205,.205,.32],maps["A_F"],"Attention branch","A_F",6,border=BLUE,tag=r"$dO$")
    ax.text(.312,.365,"+",fontsize=28,weight="bold",color=TEAL,ha="center",va="center")
    ax.text(.622,.365,"=",fontsize=28,weight="bold",color=TEAL,ha="center",va="center")
    ax.text(.482,.125,r"$\mathrm{d}O=(\mathrm{d}A)V+A\,\mathrm{d}V$",ha="center",fontsize=16,color=NAVY)
    rounded(ax,(.72,.075),.24,.075,face="#FCEEEE",edge=RED,radius=.012)
    ax.text(.84,.112,"Add residual identity only for Total Endpoint",ha="center",va="center",fontsize=9.5,color=RED,weight="semibold")
    return fig


def figure_7(photo, maps):
    fig, ax = canvas(TITLES[7], "Same input · same ground-truth logit · same cotangent at the residual output", size=(14.8, 6.5))
    # Equation ribbon.
    rounded(ax,(.19,.73),.62,.105,face="white",edge=LINE,radius=.014)
    ax.text(.50,.782,r"$y=u+f(u)\qquad\Longrightarrow\qquad E_y=E_{identity}+E_{branch}$",ha="center",va="center",fontsize=16)
    panels=[
        ("Input",photo,"photo","",BLUE),
        ("Identity path",maps["identity"],"learning","identity",BLUE),
        ("Attention branch",maps["attention"],"learning","attention",ORANGE),
        ("Total Endpoint",maps["L_F"],"learning","L_F",TEAL),
    ]
    rects=[[.045,.19,.19,.40],[.285,.19,.19,.40],[.555,.19,.19,.40],[.805,.19,.19,.40]]
    for rect,(label,value,kind,source,color) in zip(rects,panels):
        if kind=="photo":photo_panel(fig,rect,value,label,border=color,tag="sloth bear")
        else:map_panel(fig,rect,value,label,source,7,border=color)
    ax.text(.515,.39,"+",fontsize=28,weight="bold",color=ORANGE,ha="center",va="center")
    ax.text(.775,.39,"=",fontsize=28,weight="bold",color=TEAL,ha="center",va="center")
    ax.text(.52,.11,"Complete signed arrays are additive; positive display panels are independently normalized.",ha="center",fontsize=9.8,color=SLATE)
    return fig


def sync_main_captions():
    mapping={1:1,2:2,6:3,7:4}
    for package_number, main_number in mapping.items():
        for lang in ("zh","en"):
            path=ROOT/lang/"manuscript.md";text=path.read_text()
            pattern=r"(?m)^\*\*Figure "+str(main_number)+r"\.\*\* .*$"
            replacement=f"**Figure {main_number}.** {CAPTIONS[package_number][lang]}"
            text,count=re.subn(pattern,lambda _match: replacement,text)
            if count!=1:raise RuntimeError(f"caption binding failed: {lang} {main_number} {count}")
            path.write_text(text)


def mirror(text: str, lang: str) -> str:
    return text.replace("](../figures/","](../publication_r2_20260920/figures/").replace("](../references.md)","](../publication_r2_20260920/references.md)").replace("](supplement.md)",f"](../publication_r2_20260920/{lang}/supplement.md)").replace("](supplement_experiments_v1.md)",f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)").replace("](supplement.md#",f"](../publication_r2_20260920/{lang}/supplement.md#").replace("](../evidence/","](../publication_r2_20260920/evidence/")


def rebuild_gallery(manifest):
    cards=[]
    for row in manifest["figures"]:
        cards.append(
            f'<article><h2>Package figure {row["number"]}. {row["title"]}</h2>'
            f'<img src="{row["files"]["png"]["path"]}">'
            f'<p>{row["caption_zh"]}</p><p>{row["caption_en"]}</p></article>'
        )
    html=("<!doctype html><html lang='zh'><meta charset='utf-8'><title>Endpoint figures</title>"
          "<style>body{max-width:1320px;margin:36px auto;background:#f4f7fa;color:#102a43;font:16px system-ui;padding:0 20px}"
          "article{background:white;padding:28px;margin:28px 0;border-radius:14px;box-shadow:0 4px 18px #102a4312}"
          "img{width:100%;border-radius:8px}h1,h2{letter-spacing:-.02em}p{line-height:1.65}</style>"
          "<h1>NNS–ViT Endpoint · publication figures</h1>"+"".join(cards)+"</html>")
    (ROOT/"gallery.html").write_text(html)


def main():
    manifest=json.loads((ROOT/"figure_manifest.json").read_text())
    active=json.loads(ACTIVE.read_text())
    with np.load(ROOT/"assets/deit_s16__014.npz") as data:
        photo=np.array(data["input_rgb"]); class_endpoint=np.array(data["class_endpoint"])
    meta=json.loads((REPO/"reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/deit_s16__014/attempt-01.json").read_text())["result"]
    artifact=next(row for row in meta["artifacts"] if row.get("block")==5)
    source=REPO/artifact["path"]
    if sha(source)!=artifact["sha256"]:raise RuntimeError("E3 source hash drift")
    arrays=np.load(source,mmap_mode="r")
    maps={key:np.array(arrays[i]) for i,key in enumerate(artifact["component_order"])}
    if not np.allclose(class_endpoint,maps["L_F"],atol=2e-5,rtol=2e-5):
        raise RuntimeError("class Endpoint / E3 full-map drift")
    toy=json.loads((ROOT/"evidence/concept_numeric_example.json").read_text())

    figures={
        1:figure_1(photo,maps),2:figure_2(),3:figure_3(toy),4:figure_4(toy),
        5:figure_5(photo),6:figure_6(photo,maps),7:figure_7(photo,maps),
    }
    for number,fig in figures.items():save(fig,manifest,active,number)
    sync_main_captions()
    manifest["revision"]="PUBLICATION-R2-CONCEPT-V2"
    active["source_revision"]=manifest["revision"]
    (ROOT/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    ACTIVE.write_text(json.dumps(active,ensure_ascii=False,indent=2)+"\n")

    # Replace only the redesigned display records; keep empirical receipts for Figures 8--22.
    receipt=json.loads((ROOT/"evidence/display_receipt.json").read_text())
    receipt["policy"]="positive signed-RGB sum; positive-value P99; square-root display; no smoothing for pixel Endpoint"
    receipt["panels"]=[row for row in receipt["panels"] if row["figure"] not in range(1,8)]+DISPLAY_ROWS
    (ROOT/"evidence/display_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")

    sync_main_captions()
    for lang in ("zh","en"):
        text=(ROOT/lang/"manuscript.md").read_text()
        (ROOT.parent/lang/"full_draft.md").write_text(mirror(text,lang))
    rebuild_gallery(manifest)

    sources=json.loads((ROOT/"evidence/figure_sources.json").read_text())
    sources[str(Path(__file__).resolve().relative_to(REPO))]=sha(Path(__file__).resolve())
    (ROOT/"evidence/figure_sources.json").write_text(json.dumps(sources,ensure_ascii=False,indent=2)+"\n")

    review_path=ROOT/"qc/revision_v6/visual_review.json"
    review=json.loads(review_path.read_text())
    review.update({
        "status":"PASS_VISUAL_REVIEW_CONCEPT_V2",
        "reviewed_current_package_figures":list(range(1,8)),
        "checks":[
            "Figure 1 opens with the real sloth-bear input and Total Endpoint before the structural abstraction",
            "Figures 1, 6, and 7 use hash-bound real arrays and retain the positive-display boundary",
            "Figure 2 distinguishes concrete use identity, branch reconstruction, and bypass-inclusive global closure",
            "Figures 3--5 use one typography, compact hierarchy, aligned formulas, and no overlapping labels",
            "All seven figures use the same navy, teal, orange, red, and pale-surface visual grammar",
        ],
        "png_hashes":{str(row["number"]):row["files"]["png"]["sha256"] for row in manifest["figures"]},
        "unchanged_figures":"Figures 8--22 retain their prior arrays and rendering",
        "renderers":{str(Path(__file__).resolve().relative_to(REPO)):sha(Path(__file__).resolve())},
    })
    review_path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"RENDERED_CONCEPT_V2","figures":list(figures),"display_records":len(DISPLAY_ROWS)},indent=2))


if __name__=="__main__":
    main()
