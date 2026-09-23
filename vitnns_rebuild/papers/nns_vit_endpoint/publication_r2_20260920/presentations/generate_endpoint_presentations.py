#!/usr/bin/env python3
"""Generate the 20-slide and 40-slide Chinese Endpoint NNS-ViT decks.

The decks are deliberately presentation-oriented: the manuscript remains the
scientific source of truth, while this script turns its claims, figures, and
boundaries into plain-language slides with editable text.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


HERE = Path(__file__).resolve().parent
PUBLICATION = HERE.parent
FIGURES = PUBLICATION / "figures" / "png"
MANUSCRIPT = PUBLICATION / "zh" / "manuscript.md"
QC_DIR = HERE / "qc"

W, H = 13.333, 7.5
FONT_CN = "Microsoft YaHei"
FONT_MATH = "Cambria Math"

NAVY = "0B1F33"
BLUE = "176B87"
TEAL = "00A6A6"
CYAN = "DFF7F6"
ORANGE = "F59E0B"
RED = "D9485F"
GREEN = "2F855A"
INK = "183044"
MUTED = "60758A"
LIGHT = "F4F8FB"
PALE = "E8F0F5"
WHITE = "FFFFFF"
DARK_BG = "071A2B"


def rgb(value: str) -> RGBColor:
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_fill(shape, color: str, transparency: int = 0) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    shape.fill.transparency = transparency


def set_line(shape, color: str, width: float = 1.0, transparency: int = 0) -> None:
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    shape.line.transparency = transparency


@dataclass(frozen=True)
class DeckInfo:
    key: str
    filename: str
    total: int
    label: str


class DeckBuilder:
    def __init__(self, info: DeckInfo):
        self.info = info
        self.prs = Presentation()
        self.prs.slide_width = Inches(W)
        self.prs.slide_height = Inches(H)
        self.outline: list[dict[str, object]] = []

    def _blank(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _background(self, slide, color: str = LIGHT) -> None:
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(color)

    def _rect(
        self, slide, x: float, y: float, w: float, h: float,
        fill: str = WHITE, line: str | None = None, radius: bool = False,
    ):
        kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
        shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        set_fill(shape, fill)
        if line:
            set_line(shape, line, 1.0)
        else:
            shape.line.fill.background()
        if radius:
            try:
                shape.adjustments[0] = 0.08
            except Exception:
                pass
        return shape

    def _text(
        self, slide, text: str, x: float, y: float, w: float, h: float,
        size: float = 20, color: str = INK, bold: bool = False,
        font: str = FONT_CN, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
        margin: float = 0.02, line_spacing: float = 1.05,
    ):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(margin)
        tf.margin_top = tf.margin_bottom = Inches(margin)
        tf.vertical_anchor = valign
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = line
            p.alignment = align
            p.line_spacing = line_spacing
            p.space_after = Pt(max(0, size * 0.12))
            for run in p.runs:
                run.font.name = font
                run.font.size = Pt(size)
                run.font.bold = bold
                run.font.color.rgb = rgb(color)
        return shape

    def _rich_lines(
        self, slide, lines: Sequence[tuple[str, str, bool]],
        x: float, y: float, w: float, h: float, size: float = 18, gap: float = 7,
    ):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.04)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        for idx, (text, color, bold) in enumerate(lines):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = text
            p.space_after = Pt(gap)
            p.line_spacing = 1.05
            for run in p.runs:
                run.font.name = FONT_CN
                run.font.size = Pt(size)
                run.font.bold = bold
                run.font.color.rgb = rgb(color)
        return shape

    def _header(self, slide, title: str, kicker: str = "") -> None:
        if kicker:
            self._text(slide, kicker.upper(), 0.55, 0.22, 3.2, 0.24, 9.5, TEAL, True)
        self._text(slide, title, 0.55, 0.48, 12.15, 0.62, 28, NAVY, True)
        self._rect(slide, 0.55, 1.12, 0.78, 0.05, TEAL)

    def _footer(self, slide, page: int) -> None:
        self._text(slide, "NNS–ViT Endpoint Scanner", 0.55, 7.18, 4.2, 0.16, 8.5, MUTED)
        self._text(slide, f"{self.info.label}  ·  {page:02d}/{self.info.total:02d}", 9.5, 7.16, 3.25, 0.18, 8.5, MUTED, align=PP_ALIGN.RIGHT)

    def _notes(self, slide, notes: str) -> None:
        slide.notes_slide.notes_text_frame.text = "讲解建议：" + notes.strip()

    def _register(self, title: str, figures: Sequence[str] = ()) -> int:
        page = len(self.prs.slides)
        self.outline.append({"page": page, "title": title, "figures": list(figures)})
        return page

    def _new(self, title: str, kicker: str = "", background: str = LIGHT):
        slide = self._blank()
        self._background(slide, background)
        self._header(slide, title, kicker)
        page = self._register(title)
        self._footer(slide, page)
        return slide

    def _add_figure(
        self, slide, filename: str, x: float, y: float, w: float, h: float,
        caption: str | None = None, border: str = PALE,
    ) -> None:
        path = FIGURES / filename
        if not path.exists():
            raise FileNotFoundError(path)
        self._rect(slide, x, y, w, h, WHITE, border, radius=True)
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min((w - 0.16) / iw, (h - 0.16) / ih)
        pw, ph = iw * scale, ih * scale
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
        if caption:
            self._text(slide, caption, x + 0.10, y + h - 0.29, w - 0.20, 0.19, 8.5, MUTED, align=PP_ALIGN.CENTER)

    def _takeaway(self, slide, text: str, color: str = NAVY) -> None:
        self._rect(slide, 0.55, 6.53, 12.15, 0.48, color, radius=True)
        self._text(slide, "一句话：" + text, 0.78, 6.63, 11.7, 0.23, 13.5, WHITE, True, valign=MSO_ANCHOR.MIDDLE)

    def _card(
        self, slide, x: float, y: float, w: float, h: float,
        label: str, body: str, accent: str = TEAL, body_size: float = 16,
        value: str | None = None,
    ) -> None:
        self._rect(slide, x, y, w, h, WHITE, PALE, radius=True)
        self._rect(slide, x, y, 0.08, h, accent, radius=True)
        self._text(slide, label, x + 0.22, y + 0.18, w - 0.40, 0.30, 12, accent, True)
        if value is not None:
            self._text(slide, value, x + 0.22, y + 0.55, w - 0.40, 0.52, 27, NAVY, True)
            self._text(slide, body, x + 0.22, y + 1.10, w - 0.40, h - 1.25, body_size, INK)
        else:
            self._text(slide, body, x + 0.22, y + 0.56, w - 0.40, h - 0.72, body_size, INK)

    def title_slide(self, subtitle: str, figure: str, notes: str) -> None:
        slide = self._blank()
        self._background(slide, DARK_BG)
        self._rect(slide, 0.0, 0.0, 0.12, H, TEAL)
        self._text(slide, "NNS–ViT ENDPOINT SCANNER", 0.70, 0.62, 5.5, 0.25, 10, TEAL, True)
        self._text(slide, "看懂 ViT 内部目标\n到底依赖了图像哪里", 0.70, 1.12, 5.55, 1.45, 28, WHITE, True)
        self._text(slide, subtitle, 0.72, 2.82, 5.30, 0.75, 16, "B9CFDC")
        self._rect(slide, 0.72, 4.05, 4.95, 1.30, "102A40", "24465E", radius=True)
        self._text(slide, "论文题目", 0.96, 4.28, 1.1, 0.22, 10, TEAL, True)
        self._text(slide, "面向视觉 Transformer 内部目标的\n双索引局部 Jacobian 扫描与结构审计", 0.96, 4.63, 4.40, 0.52, 14.5, WHITE, True)
        self._add_figure(slide, figure, 6.25, 0.62, 6.48, 5.95, border="24465E")
        self._text(slide, "来源：publication_r2_20260920 中文稿  ·  2026-09-23", 0.72, 6.90, 6.5, 0.20, 9, "8EA6B8")
        self._text(slide, f"{self.info.label}  ·  01/{self.info.total:02d}", 9.5, 6.90, 3.20, 0.20, 9, "8EA6B8", align=PP_ALIGN.RIGHT)
        self._register("看懂 ViT 内部目标到底依赖了图像哪里", [figure])
        self._notes(slide, notes)

    def statement(
        self, title: str, statement: str, support: str, notes: str,
        figure: str | None = None, kicker: str = "核心判断", takeaway: str | None = None,
    ) -> None:
        slide = self._new(title, kicker)
        if figure:
            self._rect(slide, 0.55, 1.45, 4.55, 4.72, NAVY, radius=True)
            self._text(slide, statement, 0.88, 1.88, 3.90, 1.65, 25, WHITE, True, valign=MSO_ANCHOR.MIDDLE)
            self._text(slide, support, 0.90, 3.85, 3.84, 1.42, 16, "CFE0EB")
            self._add_figure(slide, figure, 5.35, 1.45, 7.35, 4.72)
            self.outline[-1]["figures"] = [figure]
        else:
            self._rect(slide, 0.82, 1.55, 11.70, 2.12, NAVY, radius=True)
            self._text(slide, statement, 1.23, 1.98, 10.85, 1.18, 27, WHITE, True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
            self._text(slide, support, 1.55, 4.12, 10.20, 1.20, 18, INK, align=PP_ALIGN.CENTER)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def bullets_figure(
        self, title: str, bullets: Sequence[str], figure: str, notes: str,
        kicker: str = "方法", takeaway: str | None = None, image_right: bool = True,
        figure_caption: str | None = None,
    ) -> None:
        slide = self._new(title, kicker)
        bx, ix = (0.55, 5.05) if image_right else (7.78, 0.55)
        bw, iw = 4.20, 6.95
        lines = [(f"{i + 1:02d}  {item}", TEAL if i == 0 else INK, i == 0) for i, item in enumerate(bullets)]
        self._rect(slide, bx, 1.38, bw, 4.83, WHITE, PALE, radius=True)
        self._rich_lines(slide, lines, bx + 0.22, 1.67, bw - 0.44, 4.30, 17.3, 10)
        self._add_figure(slide, figure, ix, 1.38, iw, 4.83, figure_caption)
        self.outline[-1]["figures"] = [figure]
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def figure_focus(
        self, title: str, figure: str, reading: str, notes: str,
        kicker: str = "结果", takeaway: str | None = None,
    ) -> None:
        slide = self._new(title, kicker)
        self._add_figure(slide, figure, 0.55, 1.32, 9.45, 5.05)
        self._rect(slide, 10.22, 1.32, 2.48, 5.05, NAVY, radius=True)
        self._text(slide, "怎么看", 10.50, 1.68, 1.90, 0.28, 12, TEAL, True)
        self._text(slide, reading, 10.50, 2.16, 1.92, 3.66, 16, WHITE, True)
        self.outline[-1]["figures"] = [figure]
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def equation(
        self, title: str, formula: str, explanation: Sequence[str], notes: str,
        kicker: str = "理论", takeaway: str | None = None, warning: str | None = None,
    ) -> None:
        slide = self._new(title, kicker)
        self._rect(slide, 0.72, 1.48, 11.90, 1.48, NAVY, radius=True)
        self._text(slide, formula, 0.98, 1.82, 11.38, 0.72, 25, WHITE, True, font=FONT_MATH, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        n = len(explanation)
        gap = 0.22
        cw = (11.90 - gap * (n - 1)) / n
        for i, item in enumerate(explanation):
            self._card(slide, 0.72 + i * (cw + gap), 3.30, cw, 2.25, f"0{i + 1}", item, [TEAL, BLUE, ORANGE, RED][i % 4], 16)
        if warning:
            self._text(slide, warning, 0.95, 5.86, 11.4, 0.34, 13.2, RED, True, align=PP_ALIGN.CENTER)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def cards(
        self, title: str, cards: Sequence[tuple[str, str, str]], notes: str,
        kicker: str = "框架", takeaway: str | None = None, columns: int = 2,
    ) -> None:
        slide = self._new(title, kicker)
        rows = math.ceil(len(cards) / columns)
        gap_x, gap_y = 0.25, 0.25
        total_w, total_h = 12.15, 4.90
        cw = (total_w - gap_x * (columns - 1)) / columns
        ch = (total_h - gap_y * (rows - 1)) / rows
        for i, (label, body, accent) in enumerate(cards):
            col, row = i % columns, i // columns
            self._card(slide, 0.55 + col * (cw + gap_x), 1.38 + row * (ch + gap_y), cw, ch, label, body, accent, 16)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def metric_cards(
        self, title: str, metrics: Sequence[tuple[str, str, str, str]], notes: str,
        kicker: str = "证据", takeaway: str | None = None, columns: int = 3,
    ) -> None:
        slide = self._new(title, kicker)
        rows = math.ceil(len(metrics) / columns)
        gap_x, gap_y = 0.22, 0.24
        total_w, total_h = 12.15, 4.92
        cw = (total_w - gap_x * (columns - 1)) / columns
        ch = (total_h - gap_y * (rows - 1)) / rows
        for i, (label, value, body, accent) in enumerate(metrics):
            col, row = i % columns, i // columns
            self._card(slide, 0.55 + col * (cw + gap_x), 1.38 + row * (ch + gap_y), cw, ch, label, body, accent, 14.5, value)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def comparison(
        self, title: str, left: tuple[str, Sequence[str], str],
        right: tuple[str, Sequence[str], str], notes: str,
        kicker: str = "边界", takeaway: str | None = None,
    ) -> None:
        slide = self._new(title, kicker)
        for x, item in [(0.55, left), (6.75, right)]:
            label, bullets, accent = item
            self._rect(slide, x, 1.42, 5.95, 4.75, WHITE, PALE, radius=True)
            self._rect(slide, x, 1.42, 5.95, 0.62, accent, radius=True)
            self._text(slide, label, x + 0.25, 1.61, 5.45, 0.25, 18, WHITE, True, align=PP_ALIGN.CENTER)
            lines = [("✓  " + b, INK, False) for b in bullets]
            self._rich_lines(slide, lines, x + 0.34, 2.34, 5.25, 3.40, 16.2, 11)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def table(
        self, title: str, headers: Sequence[str], rows: Sequence[Sequence[str]],
        notes: str, kicker: str = "数据", takeaway: str | None = None,
        col_widths: Sequence[float] | None = None, font_size: float = 13.5,
    ) -> None:
        slide = self._new(title, kicker)
        x, y, w, h = 0.55, 1.40, 12.15, 4.82
        shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
        table = shape.table
        if col_widths:
            scale = w / sum(col_widths)
            for col, width in zip(table.columns, col_widths):
                col.width = Inches(width * scale)
        for c, value in enumerate(headers):
            cell = table.cell(0, c)
            cell.text = value
            set_fill(cell, NAVY)
        for r, row in enumerate(rows, 1):
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.text = value
                set_fill(cell, WHITE if r % 2 else "EDF5F7")
        for r in range(len(rows) + 1):
            for c in range(len(headers)):
                cell = table.cell(r, c)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                for p in cell.text_frame.paragraphs:
                    p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                    for run in p.runs:
                        run.font.name = FONT_CN
                        run.font.size = Pt(font_size)
                        run.font.bold = r == 0
                        run.font.color.rgb = rgb(WHITE if r == 0 else INK)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def bar_comparison(
        self, title: str, groups: Sequence[tuple[str, float, float]],
        labels: tuple[str, str], notes: str, kicker: str = "数据分析",
        takeaway: str | None = None, max_value: float | None = None,
    ) -> None:
        slide = self._new(title, kicker)
        self._rect(slide, 0.75, 1.45, 11.70, 4.70, WHITE, PALE, radius=True)
        maxv = max_value or max(max(a, b) for _, a, b in groups) * 1.08
        chart_x, chart_y, chart_w, chart_h = 2.55, 1.95, 9.10, 3.48
        for tick in range(5):
            xx = chart_x + chart_w * tick / 4
            self._rect(slide, xx, chart_y, 0.008, chart_h, PALE)
            self._text(slide, f"{maxv * tick / 4:.2f}", xx - 0.20, chart_y + chart_h + 0.10, 0.40, 0.18, 8.5, MUTED, align=PP_ALIGN.CENTER)
        row_h = chart_h / len(groups)
        for i, (name, a, b) in enumerate(groups):
            yy = chart_y + i * row_h
            self._text(slide, name, 0.95, yy + 0.11, 1.40, 0.24, 12.5, INK, True, align=PP_ALIGN.RIGHT)
            ah = row_h * 0.28
            self._rect(slide, chart_x, yy + 0.06, chart_w * a / maxv, ah, BLUE, radius=True)
            self._rect(slide, chart_x, yy + 0.08 + ah, chart_w * b / maxv, ah, ORANGE, radius=True)
            self._text(slide, f"{a:.3f}", chart_x + chart_w * a / maxv + 0.07, yy + 0.04, 0.55, 0.19, 9, BLUE, True)
            self._text(slide, f"{b:.3f}", chart_x + chart_w * b / maxv + 0.07, yy + 0.06 + ah, 0.55, 0.19, 9, ORANGE, True)
        self._rect(slide, 8.15, 5.63, 0.20, 0.12, BLUE, radius=True)
        self._text(slide, labels[0], 8.42, 5.55, 1.35, 0.22, 10.5, INK)
        self._rect(slide, 10.08, 5.63, 0.20, 0.12, ORANGE, radius=True)
        self._text(slide, labels[1], 10.35, 5.55, 1.45, 0.22, 10.5, INK)
        if takeaway:
            self._takeaway(slide, takeaway)
        self._notes(slide, notes)

    def save(self) -> Path:
        if len(self.prs.slides) != self.info.total:
            raise AssertionError(f"{self.info.key}: expected {self.info.total} slides, got {len(self.prs.slides)}")
        path = HERE / self.info.filename
        self.prs.core_properties.title = "NNS–ViT Endpoint Scanner 中文汇报"
        self.prs.core_properties.subject = self.info.label
        self.prs.core_properties.author = "TraeCode，依据 publication_r2_20260920/zh/manuscript.md 生成"
        self.prs.core_properties.comments = "图件来自论文发布包；文字与讲者备注为说人话汇报版。"
        self.prs.save(path)
        return path


def build_short() -> tuple[Path, list[dict[str, object]]]:
    d = DeckBuilder(DeckInfo("short", "Endpoint_NNS_ViT_20页重点汇报.pptx", 20, "20页重点版"))
    d.title_slide(
        "20页重点汇报｜简单、直观、只讲最重要的事", "fig_01_concept_overview.png",
        "先告诉听众，这不是又一种新热力图，而是一套把 ViT 内部依赖拆开、对账、再比较的办法。",
    )
    d.statement(
        "先用一句话说清楚",
        "把 ViT 里某个目标对输入图像的依赖，\n拆成“经过哪里”与“落在图像哪里”两条线索。",
        "最终得到的不是一张孤立热图，而是一组能够相加还原、能够检查漏项、也能够跨模块比较的局部依赖图。",
        "这一页只讲核心价值。可以把它类比成财务对账：总账有了，还要知道每一笔从哪个科目来。",
        takeaway="既看空间位置，也保留结构来源。",
    )
    d.bullets_figure(
        "为什么普通可视化还不够？",
        ["特征图告诉我们“激活在哪里”，不等于目标依赖哪里。",
         "Attention 权重告诉我们“怎么分配”，不等于对最终目标的局部影响。",
         "一张总梯度图把所有路径混在一起，看不出 Q/K/V 和残差分工。"],
        "fig_14_cross_model_examples.png",
        "不要说现有方法没用。要说它们回答的是不同问题，而本文补的是结构来源这一维。",
        kicker="研究动机", takeaway="问题不是“有没有热图”，而是“这张图来自哪条结构分项”。", image_right=True,
    )
    d.cards(
        "本文实际回答四个问题",
        [("1｜数学对象", "NNS learning image 到底是什么？", TEAL),
         ("2｜结构推广", "怎么把它放进 ViT 的 attention 与 residual？", BLUE),
         ("3｜空间规律", "这些局部依赖如何随层深变化？", ORANGE),
         ("4｜干预边界", "看起来重要，是否就能预测有限替换效果？", RED)],
        "四个问题按“定义—实现—发现—边界”展开，听众会更容易跟上。",
        kicker="研究问题", takeaway="先证明对象算对了，再讨论它告诉了我们什么。",
    )
    d.equation(
        "核心公式其实很熟悉",
        "Endpoint = x ⊙ ∇ₓ s(x)",
        ["x：当前输入坐标", "s(x)：选定的内部标量目标", "∇ₓs：当前点的精确局部梯度"],
        "明确说出：Endpoint 在数值上就是同目标 Gradient × Input。创新不在这个基础公式。",
        takeaway="基础量不是新算法；新意在 ViT 结构分解与可审计切面。",
        warning="它描述当前点的一阶依赖，不保证各像素相加等于目标函数值。",
    )
    d.statement(
        "真正新增的是“双索引”",
        "i 记“图像上的位置”\ne 记“计算图里的具体使用位置”",
        "保留 i，才能看空间形状；保留 e，才能分清 Q/K/V、head、identity 与 branch。两者一起保留，才是本文的结构化局部图。",
        "这里用坐标和道路的比喻：i 像目的地坐标，e 像走的是哪条路。只记一个都会丢信息。",
        figure="fig_02_neuron_learning_image.png", takeaway="总图回答“在哪”，双索引图还回答“经过谁”。", kicker="核心方法",
    )
    d.figure_focus(
        "合法切面：必须不漏，也不能重复", "fig_02_neuron_learning_image.png",
        "分支切面只重构该分支。\n全局切面必须把 residual bypass 也算上。\n同一张量被多次使用时，要按具体 use 分开记。",
        "强调“切一刀后，每条输入到目标的路径恰好穿过一次”。这是结构审计的直观定义。",
        kicker="结构审计", takeaway="数值相等不一定证明结构完整，所以还要检查计算图。",
    )
    d.figure_focus(
        "Attention 被拆成 routing 与 value", "fig_06_attention_differential.png",
        "Q、K 决定注意力分配怎么变。\nV 决定内容怎么传。\n两部分相加，才得到完整 attention branch。",
        "说人话：Q/K 更像“往哪里送”，V 更像“送什么”。图中两部分是微分分项，不是可以直接替换的前向张量。",
        takeaway="深层 attention 不能只看 Value，routing 的作用会变强。",
    )
    d.figure_focus(
        "Residual 被拆成 carry 与 branch", "fig_07_residual_principle.png",
        "identity 保留原信息。\nbranch 写入新变化。\n完整 signed 图满足：identity + branch = Total Endpoint。",
        "提醒听众，图上展示的是正部视图，不能拿显示图直接做相加；闭合计算使用完整 signed 数组。",
        takeaway="残差不是一条路，而是“带着旧信息走”加“分支写入”。",
    )
    d.metric_cards(
        "实验设计：对象固定，统计单位清楚",
        [("模型", "2 个", "ViT-B/16 与 DeiT-S/16；均为 12 blocks", TEAL),
         ("主评估图像", "400×2", "两模型使用同一组 ImageNet 图像 ID", BLUE),
         ("注册目标/图", "457 / 385", "固定结构锚点，不声称穷举全部神经元", ORANGE),
         ("单元研究", "100 图", "每模型 6,144 个固定抽样单元", GREEN),
         ("区间", "10,000 次", "以图像为统计单位做 bootstrap", BLUE),
         ("独立队列", "400 图", "另加四个公开 checkpoint 稳健性检查", TEAL)],
        "不要把 target 当统计单位。这里的推断单位是图像，目标集合与 checkpoint 是固定条件。",
        kicker="实验设置", takeaway="展示图只是例子，主要结论来自完整队列。",
    )
    d.metric_cards(
        "先证明“算对了”",
        [("数值身份", "5,000", "兼容网络比较；float64 最大相对误差 4.769×10⁻¹⁶", TEAL),
         ("正向闭合", "76,800", "全部通过", GREEN),
         ("遗漏/重复负控", "28,800", "全部被拒绝", RED),
         ("解析 JVP", "6,144", "全部通过", BLUE),
         ("算子方向", "88", "全部通过", ORANGE),
         ("float32 超差复核", "36→0", "同坐标 float64 复核全部通过", TEAL)],
        "这一页建立的是实现正确性与结构记账，不是干预有效性。二者必须分开。",
        kicker="正确性证据", takeaway="闭合证明对账正确，不等于能预测大幅度干预。",
    )
    d.figure_focus(
        "同一张图，不同内部目标看到不同区域", "fig_08_internal_targets.png",
        "输入没变。\n变的是我们问网络里的哪个标量。\n不同目标对应不同的局部依赖图。",
        "指出 energy target、attention projection、MLP output 与 class logit 是不同目标，不要把所有热图理解成类别解释。",
        takeaway="解释结果取决于“你问了哪个目标”。",
    )
    d.figure_focus(
        "发现一：依赖范围总体随深度扩展", "fig_12_spatial_broadening.png",
        "熵上升：分布更广。\n局部性下降：不再只盯附近 patch。\n平均距离上升：依赖扩到更远位置。",
        "这是固定 checkpoint 上的层深分布轨迹，不把它说成深度的因果效应。",
        takeaway="越到深层，内部单元通常越依赖更广的图像区域。",
    )
    d.figure_focus(
        "一张图看完 12 层的变化", "fig_09_deit_s16_layerwise.png",
        "上排：目标对原图的局部依赖。\n下排：该层真实前向激活。\n两者不是同一个对象。",
        "让听众先看空间从局部向全局展开，再指出前向 feature 与 learning image 不能互换。",
        takeaway="激活在哪里，不等于这个目标依赖输入哪里。",
    )
    d.figure_focus(
        "发现二：深层 attention 更不能忽略 routing", "fig_16_attention_depth.png",
        "左：routing 相对质量随层深上升。\n中：Value-only 与完整分支的差距增大。\n右：方向导数也显示相同问题。",
        "可用“只看 V 就像只看运送的货物，却忽略路线怎么变”来解释。",
        takeaway="Value-only 是近似，而且在深层会漏掉越来越多。",
    )
    d.figure_focus(
        "发现三：两类 residual 的分工不同", "fig_18_residual_depth.png",
        "Attention residual 的 identity 比例总体下降。\nMLP residual 的 identity 比例总体上升。\n两者呈现不同层深轨迹。",
        "不要把 identity 比例叫作因果贡献率。它是当前点局部微分的 absolute-mass 比例。",
        takeaway="Attention 与 MLP residual 不是同一种“跳连行为”。",
    )
    d.metric_cards(
        "独立队列仍复现主要趋势",
        [("H1｜单元扩展", "4/4 正", "四个 checkpoint 的 block11−block0 均为正", TEAL),
         ("H2｜routing 增强", "4/4 正", "深层 routing 相对份额上升", BLUE),
         ("H3｜residual 分化", "4/4 正", "MLP 与 attention identity 轨迹存在稳定差异", ORANGE),
         ("H4｜排序增益", "4/4 负", "局部图没有优于 activation×gradient 的 head 排序", RED)],
        "前面三项复现结构趋势；H4 是重要的负结果，说明更细的空间图不自动带来更好的干预排序。",
        kicker="稳健性", takeaway="主要结构规律可复现，但干预排序优势没有出现。", columns=2,
    )
    d.figure_focus(
        "有限替换：Taylor 明显更适合回答“改了会怎样”", "fig_21_intervention_boundary.png",
        "Endpoint：当前状态依赖哪里。\nTaylor：再加入 clean−reference 的实际位移。\n候选包络：已测试区域中的经验上界。",
        "四个模型×reference 组合里，Taylor AUC 都高于 Endpoint 区域分数；差值为 0.023–0.036。",
        kicker="干预边界", takeaway="想预测特定替换效果，必须把 reference 位移带进来。",
    )
    d.comparison(
        "这套方法能说什么，不能说什么",
        ("可以说", ["能把当前点局部依赖按 ViT 结构拆开", "能在合法切面上检查漏项与重复", "能比较 Q/K/V、head 与 residual 的空间分布", "能在固定 checkpoint 上描述层深轨迹"], GREEN),
        ("不能直接说", ["不是新的基础 Gradient × Input 算法", "热图相加不保证等于函数值", "结构闭合不等于干预预测能力", "抽样目标趋势不等于全模型普遍定律"], RED),
        "这页用于主动收紧结论边界，不需要防御式口吻。",
        takeaway="把“表示、正确性、效用”分开，结论就不会混。",
    )
    d.statement(
        "最后带走三句话",
        "看哪里：Endpoint 统一到输入坐标。\n看谁贡献：双索引保留具体 use。\n看改了会怎样：再加入 reference 位移。",
        "这三层分别对应表示、结构审计和有限变化。它们可以衔接，但不能互相替代。",
        "结束时不再重复所有数字，只收束到三个问题。若有提问，再回到正确性、层深结果或干预边界页。",
        figure="fig_01_concept_overview.png", kicker="结论", takeaway="从“当前依赖在哪里”走向“经过哪个结构，以及沿指定方向会怎样变化”。",
    )
    return d.save(), d.outline


def build_long() -> tuple[Path, list[dict[str, object]]]:
    d = DeckBuilder(DeckInfo("long", "Endpoint_NNS_ViT_40页详细汇报.pptx", 40, "40页详细版"))
    d.title_slide(
        "40页详细汇报｜理论、验证、图像与数据分析", "fig_01_concept_overview.png",
        "开场先给出整套框架：同一目标、同一输入，既有总 Endpoint，也有结构分项和有限替换评价。",
    )
    d.statement(
        "研究主线",
        "把 ViT 内部目标的局部输入依赖，\n从“一张总图”展开成“可分解、可闭合、可比较”的结构图组。",
        "论文按表示 → 贡献 → 干预推进：先定义数学对象，再验证结构记账，最后检查它与有限替换的关系。",
        "详细版的主线始终是 representation—contribution—intervention。",
        takeaway="先把对象定义清楚，后面的数据分析才不会混淆。",
    )
    d.bullets_figure(
        "ViT 内部解释为什么难？",
        ["Token-channel、attention head 和 residual 分支是不同内部对象。",
         "Attention 同时包含 Q/K 路由变化与 V 内容传输。",
         "Residual 让信息可以走 identity，也可以走 branch。",
         "所有对象最终还要落回同一输入坐标，才能比较空间分布。"],
        "fig_11_module_learning_images.png",
        "用图中的 LN、attention、MLP 模块说明：原生表示空间不同，直接比较很困难。",
        kicker="问题背景", takeaway="难点不只是空间分辨率，更是结构身份会在总图里被混掉。",
    )
    d.comparison(
        "三类常见视图回答的是不同问题",
        ("已有视图", ["Feature map：哪里激活", "Attention weight：token 如何分配", "Total G×I：目标整体依赖哪里", "Activation patching：指定替换造成多大变化"], BLUE),
        ("本文补充", ["同一目标下按具体 consumer use 拆分", "Q/K/V 与 routing/value 有独立空间图", "Residual identity/branch 可在合法切面闭合", "再与 reference-aware finite effect 分开比较"], TEAL),
        "强调是对象级差异，而不是简单宣称“我们的图更好”。",
        kicker="相关方法定位", takeaway="不同图不是互相替代，而是回答不同层次的问题。",
    )
    d.cards(
        "四个研究问题",
        [("RQ1｜数学对象", "NNS learning image 与 Jacobian、Input × Gradient 的关系是什么？", TEAL),
         ("RQ2｜结构分解", "怎样保留具体 tensor use，并让分项在合法切面上闭合？", BLUE),
         ("RQ3｜层深轨迹", "单元、attention 与 residual 的空间依赖怎样随深度变化？", ORANGE),
         ("RQ4｜有限替换", "当前状态局部图能否排序 reference-specific 的替换效应？", RED)],
        "这一页是整场汇报的目录。", kicker="研究问题", takeaway="定义—验证—经验规律—效用边界。",
    )
    d.comparison(
        "贡献在哪里？先把创新边界讲清楚",
        ("不是本文的新意", ["Endpoint = Input × Gradient", "组件分解本身", "Activation patching", "Attention 或 token 可视化"], RED),
        ("本文的技术重点", ["输入位置 i 与具体 use e 的联合索引", "分支切面与全局切面的闭合合同", "对遗漏、重复、错误 slot 的结构审计", "同一坐标下的跨模块空间比较"], GREEN),
        "这页要直接，不用用大量文献防御。", kicker="贡献定位", takeaway="新意在结构化 ViT 扫描与审计，不在基础 G×I 公式。",
    )
    d.equation(
        "NNS 的一般计算图表示",
        "L(x;x)=diag(x)    ·    L(v;x)=Σₐ J(f,uₐ) L(uₐ;x)",
        ["输入处：每个坐标单独保留原值", "中间层：用当前点局部 Jacobian 传播", "多输入算子：各输入槽分别求和"],
        "先从一般有向无环图讲，不急着进入 ViT。", takeaway="学习图像就是把每个输入坐标的局部影响一路传播到目标。",
    )
    d.figure_focus(
        "初始化：既保留数值，也保留位置", "fig_03_coordinate_initialization.png",
        "每个输入坐标生成一张只有一个非零位置的图。\n堆叠后就是 diag(x)。\n位置不会在初始化时被抹掉。",
        "用 2×2 的例子解释 diag(x)，听众不需要先接受复杂符号。", kicker="理论基础", takeaway="输入不是一个总标量，而是一组带坐标的基底。",
    )
    d.figure_focus(
        "传播：局部导数负责把坐标图送到目标", "fig_04_local_derivative_generation.png",
        "每张坐标图乘当前点导数。\n再把所有坐标贡献组合起来。\n标量目标最终得到 x ⊙ ∇ₓs。",
        "强调“当前点”三个字。这里没有沿 reference 到输入做路径积分。", kicker="理论基础", takeaway="精确局部传播最终等价于同目标 Gradient × Input。",
    )
    d.equation(
        "命题 1：Endpoint 的数值身份",
        "L(s;x)ᵀ = diag(x) ∇ₓs(x) = x ⊙ ∇ₓs(x)",
        ["所有 Jacobian 在同一前向状态求值", "传播中不做归一化、截断或重分配", "标量目标由 registry 明确固定"],
        "把结论说清：这是恒等式，不是经验近似。", takeaway="本文沿用 NNS 的组织方式，但基础标量图就是 G×I。",
        warning="一般 ViT 含 bias、LayerNorm、softmax、GELU，因此 ΣᵢEᵢ 通常不等于 s(x)。",
    )
    d.equation(
        "输入坐标原点会影响 Endpoint",
        "x=(r−μ)/σ   ⇒   xᵢ ∂s/∂xᵢ = (rᵢ−μᵢ) ∂s/∂rᵢ",
        ["模型输入坐标 x", "原始像素 r", "归一化均值 μ 是 raw-RGB 空间中的坐标原点"],
        "这不是外部干预 reference，但确实是坐标选择。后面会做 raw gradient 与 zero-origin weighted gradient 敏感性检查。",
        takeaway="Endpoint 不需要干预参考图，但依赖模型采用的输入坐标。",
    )
    d.equation(
        "双索引对象：位置 i × 使用边 e",
        "Lₑ,ᵢ(s;x) = xᵢ [ J(zₑ,x)ᵀ · z̄ₑ⁽ˢ⁾ ]ᵢ",
        ["i：输入像素与通道坐标", "e：producer、consumer、slot、invocation 四元组", "s：registry 固定的标量目标"],
        "z 上横线表示目标在该具体 input slot 的 cotangent。不要把同一 producer 的多个 fan-out use 提前合并。",
        takeaway="固定 e 得一张空间图；对合法切面中的 e 求和回到 Total Endpoint。",
    )
    d.figure_focus(
        "为什么必须记录具体 use？", "fig_02_neuron_learning_image.png",
        "同一 tensor 可被多个 consumer 使用。\n不同 slot 或不同调用不能只靠 tensor name 区分。\n切面必须与目标祖先图匹配。",
        "用“一份数据被多个部门引用”的类比说明 fan-out。", kicker="结构身份", takeaway="不保留 use，就可能把不同路径提前混在一起。",
    )
    d.equation(
        "合法切面：每条路径恰好穿过一次",
        "Σₑ∈C(global) Lₑ,ᵢ(s;x) = xᵢ ∂s(x)/∂xᵢ",
        ["不漏：所有输入到目标的结构路径都经过切面", "不重：每条路径只穿过切面一次", "同态：所有分项使用同一输入、参数与导数约定"],
        "结构合法性是图上的条件，点态数值闭合是某个输入上的结果。必须先有结构条件，再用数值检查实现。",
        takeaway="单点数值刚好相等，不能反推结构上一定完整。",
    )
    d.figure_focus(
        "Attention 微分：routing 与 value 分开", "fig_06_attention_differential.png",
        "dO = (dA)V + A(dV)。\n(dA)V 来自 Q/K 引起的 routing 变化。\nA(dV) 是 value 内容变化。",
        "先讲前向 O=AV，再讲微分的乘积法则。这样听众能自然理解两项。", kicker="Attention 理论", takeaway="Q/K 改“怎么分配”，V 改“传什么内容”。",
    )
    d.equation(
        "Q/K/V 分项怎样闭合？",
        "B(attn) = B(Q)+B(K)+B(V) = B(routing)+B(value)",
        ["Q-induced 与 K-induced 是原生 projection slot 的导数分项", "routing-induced = Q-induced + K-induced", "各 head cotangent 包含 Wₒ 与所有后续导数"],
        "提醒：routing-induced 与 value-induced 是解析微分分组，不是可以直接替换的 forward tensor。",
        takeaway="这些分项重构 attention branch，而不是单独等于完整类别目标。",
    )
    d.figure_focus(
        "Residual：全局切面必须把 bypass 算进去", "fig_07_residual_principle.png",
        "y = u + branch(u)。\nidentity 与 branch 共用 addition 输出处的 cotangent。\n两者相加才是 Total Endpoint。",
        "若直接用已经包含 branch 的 ∇ᵤs 当 identity，再加 branch，就会重复记账。", kicker="Residual 理论", takeaway="只切 branch 会漏 identity；把完整上游梯度当 identity 又会重复。",
    )
    d.cards(
        "四类对象不要混用",
        [("真实计算边", "residual 输入边、head-to-concat、Q/K/V projection slot。", TEAL),
         ("解析微分分项", "Q/K/routing/value；用于重构 branch，不能直接替换。", BLUE),
         ("可干预前向变量", "attention probability、V、head output、residual branch activation。", ORANGE),
         ("统计分数", "absolute mass、regional score、reference-aware Taylor score。", RED)],
        "这页能解决很多术语混乱：能画图、能相加、能替换、能排序是四件不同的事。",
        kicker="对象合同", takeaway="局部图的分项名称，不自动等于一个可替换的前向变量。",
    )
    d.metric_cards(
        "Target registry：我们到底扫描哪些目标？",
        [("Post-block token-channel", "192", "固定 token 与 channel 读出", TEAL),
         ("Patch sum / energy", "48+48", "聚合标量，不是单个神经元", BLUE),
         ("Attention-head coordinate", "144 / 72", "ViT-B / DeiT-S", ORANGE),
         ("Module aggregate", "18", "attention 与 MLP 内部锚点", GREEN),
         ("Patch embedding", "4", "输入到 token 的局部结构", BLUE),
         ("Final / logits", "1+2", "最终 CLS coordinate 与类别目标", RED)],
        "合计每图 457/385 个标量目标。这是冻结的结构锚点，不是全部神经元的代表性抽样。",
        kicker="实验对象", takeaway="registry 固定语义，但不会让每个目标自动拥有所有切面。",
    )
    d.table(
        "模型、图像与固定分析范围",
        ["设置", "ViT-B/16", "DeiT-S/16"],
        [["Blocks / heads / hidden dim", "12 / 12 / 768", "12 / 6 / 384"],
         ["注册标量目标 / 图", "457", "385"],
         ["Evaluation images", "400", "400（相同 ID）"],
         ["Resize / crop", "248 / 224", "248 / 224"],
         ["Normalization mean", "[0.5,0.5,0.5]", "[0.485,0.456,0.406]"],
         ["Unit-study subset", "100 图 / 6,144 单元", "100 图 / 6,144 单元"]],
        "图像按文件身份和固定 SHA-256 排序选择，不按正确分类或置信度筛选。",
        kicker="实验设置", takeaway="两模型共享图像 ID，但 checkpoint 与目标集合固定，结论是条件性的。",
        col_widths=[4.0, 4.1, 4.1], font_size=14.5,
    )
    d.cards(
        "统计设计：把图像当作统计单位",
        [("图内先汇总", "固定 targets / heads 在每张图内先按规则汇总，避免把大量组件当独立样本。", TEAL),
         ("图像级区间", "10,000 次 percentile bootstrap，报告中位数与逐项 95% 区间。", BLUE),
         ("精确符号检验", "检验非零差的正负号概率；ties 保留在样本与区间中。", ORANGE),
         ("多重比较", "原成分与区域研究共 28 项联合 Holm；其他 family 单独校正。", RED)],
        "bootstrap 区间不是 p 值；一类一图也不覆盖类内变异。", kicker="统计口径", takeaway="统计单位若错用成组件，样本量会被虚增。",
    )
    d.figure_focus(
        "可视化只负责“看”，统计仍用原始数组", "fig_22_signed_and_blue.png",
        "蓝底正部：看空间细节。\nSigned：看方向。\nAbsolute mass：看抵消前总幅值。\n所有统计使用完整 signed/absolute 数组。",
        "说明各 panel 独立 P99 截断并开方，所以颜色强弱不能跨 panel 当成数值比较。", kicker="显示协议", takeaway="好看的热图是展示层，不是数值证据层。",
    )
    d.metric_cards(
        "正确性验证覆盖了什么？",
        [("NNS 数值身份", "5,000", "float64 最大相对误差 4.769×10⁻¹⁶", TEAL),
         ("Positive closures", "76,800", "全部通过", GREEN),
         ("遗漏/重复 controls", "28,800", "全部拒绝", RED),
         ("Analytic JVP", "6,144", "全部通过", BLUE),
         ("Operator directions", "88", "全部通过", ORANGE),
         ("独立解析 reference", "32+9", "逐分量与中心差分方向全部通过", TEAL)],
        "36 个 float32 有限差分超差点，在相同坐标 float64 复核中全部通过。",
        kicker="实现验证", takeaway="这些证据支持局部微分记账正确，不代替后面的有限干预测试。",
    )
    d.figure_focus(
        "同一输入，不同目标得到不同局部图", "fig_08_internal_targets.png",
        "Patch embedding、block energy、attention、MLP 与 class logit 各问不同问题。\nEnergy 是聚合标量。\n展示图不参与目标筛选。",
        "让听众观察同一懒熊图像在不同目标下的空间差异。", kicker="结果 1", takeaway="局部图永远是“相对于某个明确标量目标”的。",
    )
    d.figure_focus(
        "DeiT-S/16：12 层 learning image", "fig_09_deit_s16_layerwise.png",
        "上排是输入依赖图。\n下排是 14×14 前向 activation。\n每层选择高空间方差通道定义 energy target。",
        "先看整体扩展趋势，再强调跨层可能是不同通道，不能解释成同一语义单元的成长。", kicker="层深可视化", takeaway="深层依赖更广，但前向激活与输入依赖仍是两类对象。",
    )
    d.figure_focus(
        "ViT-B/16：相同趋势，细节不同", "fig_10_vit_b16_layerwise.png",
        "同一 12-block 结构。\n模型宽度与 head 数不同。\n空间扩展方向复现，具体图形不要求一致。",
        "把这页与前一页配对展示，不做颜色强度比较。", kicker="层深可视化", takeaway="共同趋势来自总体统计，不来自挑选某张最好看的图。",
    )
    d.figure_focus(
        "核心空间结果：越深，依赖通常越广", "fig_12_spatial_broadening.png",
        "Entropy：+0.220 / +0.225。\nLocality：−0.356 / −0.342。\nDistance：+0.131 / +0.122。\n顺序均为 ViT-B / DeiT-S。",
        "这些是每图 late blocks 8–11 减 early blocks 0–3 的中位差，再跨 100 图汇总。", kicker="数据分析", takeaway="三种空间指标从不同角度支持“局部依赖扩展”。",
    )
    d.comparison(
        "为什么空间结果需要双索引？",
        ("只保留一个索引", ["Total Endpoint 只保留 i：能算熵和距离，但看不出结构来源", "Component scalar 只保留 e：能看质量比例，但没有空间分布"], RED),
        ("保留 [e,i]", ["能比较 Q/K/V 的空间熵", "能计算 output–identity 的 spatial TV", "能看到不同结构在相同输入坐标上的重组"], GREEN),
        "routing share 与 identity share 只需 component scalar，不应拿它们单独证明双索引必要性。", kicker="表示消融", takeaway="联合索引的独特价值，是“结构标签 × 空间位置”的统计量。",
    )
    d.figure_focus(
        "同一层的 heads 也会分配到不同空间", "fig_13_all_heads.png",
        "同一图像、同一类别目标。\nBlocks 0、5、11 的全部 heads 原样展示。\n没有按地图观感挑选 head。",
        "这页主要给直觉：多头不是重复画同一张图。", kicker="Attention 空间", takeaway="结构分辨率让 head 间空间差异直接可见。",
    )
    d.figure_focus(
        "Routing 与 value 的空间形状不同", "fig_15_routing_value_examples.png",
        "浅层与深层都展示 Q、K、routing、value。\nRouting = Q + K。\n完整 signed 分项才满足闭合。",
        "单例只用于读图，不能承担总体推断。", kicker="Attention 分项", takeaway="“往哪里送”与“送什么”可以在输入空间中呈现不同结构。",
    )
    d.figure_focus(
        "Attention 随深度：routing 上升，Value-only 误差变大", "fig_16_attention_depth.png",
        "Routing share 差：+0.052 / +0.135。\nValue-only L2 error 差：+0.144 / +0.290。\nK/V entropy 在两模型均下降。",
        "DeiT-S 的 Q entropy 区间跨零，要如实保留模型依赖。", kicker="数据分析", takeaway="深层 attention 的局部微分越来越不能只用 Value 解释。",
    )
    d.figure_focus(
        "Residual 分项：attention 与 MLP 是两种轨迹", "fig_17_residual_examples.png",
        "两行使用同一 block 5 与同一类别目标。\nidentity 指向不同 residual 输入边。\n正部显示图不能直接相加。",
        "用这页衔接到下一页的总体曲线。", kicker="Residual 分项", takeaway="两次 residual addition 的 identity 语义并不相同。",
    )
    d.figure_focus(
        "Residual 随深度：一个减，一个增", "fig_18_residual_depth.png",
        "Attention identity：−0.124 / −0.045。\nMLP identity：+0.180 / +0.178。\nOutput–identity spatial TV 也呈相反方向。",
        "这些比例是 absolute local-differential mass，不叫功能贡献率。", kicker="数据分析", takeaway="Attention 与 MLP residual 在深层承担了不同的局部依赖重组。",
    )
    d.table(
        "独立队列与四 checkpoint 稳健性检查",
        ["Checkpoint", "H1 单元扩展", "H2 routing", "H3 residual", "H4 排序增益"],
        [["vit_b16", "0.413", "0.239", "0.851", "−0.114"],
         ["deit_s16", "0.378", "0.322", "0.522", "−0.186"],
         ["vit_b16_in1k", "0.336", "0.241", "0.632", "−0.186"],
         ["vit_s16_augreg", "0.390", "0.276", "0.867", "−0.136"]],
        "每项为 400 张独立队列图像的中位数；H1–H3 为正，H4 为负。",
        kicker="稳健性", takeaway="结构趋势复现，但局部图没有改善 head 有限效应排序。",
        col_widths=[2.6, 2.4, 2.4, 2.4, 2.4], font_size=14.5,
    )
    d.cards(
        "坐标与质量敏感性：哪些结论稳，哪些会变？",
        [("多数主趋势保留", "三种输入表示下，单元扩展、K/V 熵与 residual TV 的主要方向保持。", GREEN),
         ("Q 更敏感", "DeiT-S 的 Q B11−B0 对坐标与区间较敏感，不能写成普遍下降。", RED),
         ("低质量不是唯一原因", "剔除最低 1%/5%/10% mass 后，两模型 entropy 扩展仍为正。", TEAL),
         ("但熵依赖总质量", "log1p(total mass) 与 entropy 的图内相关约 −0.729 / −0.760。", ORANGE)],
        "敏感性分析不是为了消除所有边界，而是辨认哪些结果依赖坐标或幅值。",
        kicker="稳健性诊断", takeaway="主要方向稳健，不代表每个分项都坐标不变。",
    )
    d.table(
        "成分质量与有限替换效应：关系是“广泛但非普遍”",
        ["模型 / reference", "A-only", "V-only", "Attn branch", "MLP branch"],
        [["ViT-B / zero", "0.049", "0.224", "−0.028", "0.287"],
         ["ViT-B / blur", "0.073", "0.311", "0.000", "0.378"],
         ["DeiT-S / zero", "0.091", "0.035", "−0.157", "0.378"],
         ["DeiT-S / blur", "0.101", "0.147", "−0.024", "0.371"]],
        "数值为每图跨 12 blocks 的 Spearman 中位数。原 12 个 family 中 11 个达到预设复合条件，但类型差异明显。",
        kicker="有限替换", takeaway="MLP 关联稳定为正；attention branch 并不呈统一正相关。",
        col_widths=[3.2, 2.1, 2.1, 2.4, 2.4], font_size=14.2,
    )
    d.bar_comparison(
        "同层 head 排序：加入位移后明显更好",
        [("ViT-B / zero", 0.281, 0.674), ("ViT-B / blur", 0.273, 0.686),
         ("DeiT-S / zero", 0.195, 0.610), ("DeiT-S / blur", 0.148, 0.624)],
        ("Endpoint local-map rho", "Reference-aware Taylor rho"),
        "两种量使用相同当前点局部导数，Taylor 多用了 clean-reference activation displacement。",
        takeaway="要排序特定 reference 下的 head 替换效应，位移信息很关键。", max_value=0.75,
    )
    d.figure_focus(
        "区域干预：Endpoint 接近随机，Taylor 更接近候选上界", "fig_21_intervention_boundary.png",
        "四个 family 中 Taylor−Endpoint AUC：0.023–0.036。\nEndpoint 区域排序与修正随机基线接近。\n候选包络只是已评价区域的经验上界。",
        "这页是效用边界的核心证据。强调任务固定为 block 5 三个 heads 与 4×4 候选网格。", kicker="有限替换", takeaway="结构图适合描述当前依赖；reference-aware Taylor 更适合预测指定替换。",
    )
    d.comparison(
        "方法局限：把外推边界说清楚",
        ("局部与实现边界", ["一阶 Jacobian 不覆盖大位移非线性交互", "分项依赖指定计算图、内部坐标与切面", "等价重参数化可改变分项而保持总 Endpoint", "归一化 TV 不是功能重要性"], ORANGE),
        ("实证边界", ["只覆盖 ImageNet 与 patch-16 ViT", "目标是固定抽样锚点，覆盖率不是全模型普查", "公开 checkpoint 不是独立训练种子", "不同训练、规模与数据因素尚未分离"], RED),
        "这不是道歉页，而是告诉听众结论在哪个范围内可靠。", kicker="局限", takeaway="可靠的结论必须与实际扫描对象和干预合同绑定。",
    )
    d.statement(
        "结论：三层问题，三种答案",
        "当前依赖在哪里 → Endpoint\n经过哪个结构 → 双索引图与合法切面\n沿指定替换方向会怎样 → Reference-aware Taylor",
        "本文把这三层放进同一 ViT 分析框架，并用结构认证、数值闭合、独立解析实现和真实替换实验分别验证。",
        "最后强调互补性：Endpoint 与 Taylor 共享局部导数，但回答不同问题。",
        figure="fig_01_concept_overview.png", kicker="总结", takeaway="结构分辨率提升了可解释性，但不会自动变成干预预测能力。",
    )
    return d.save(), d.outline


def _font(size: int, bold: bool = False):
    candidates = [
        Path("/data01/home/yeziqi.831/.trae-cn-server/bin/stable-c39cf6689a6e21ad0619269b3b8c05cc79de4cb7-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _shape_rgb(shape, default: str = WHITE) -> tuple[int, int, int]:
    try:
        value = str(shape.fill.fore_color.rgb)
        if len(value) == 6:
            return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return tuple(int(default[i:i+2], 16) for i in (0, 2, 4))


def _slide_bg_rgb(slide) -> tuple[int, int, int]:
    try:
        value = str(slide.background.fill.fore_color.rgb)
        if len(value) == 6:
            return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return tuple(int(LIGHT[i:i+2], 16) for i in (0, 2, 4))


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    result: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            result.append("")
            continue
        line = ""
        for char in raw:
            candidate = line + char
            if line and draw.textlength(candidate, font=font) > width:
                result.append(line)
                line = char
            else:
                line = candidate
        if line:
            result.append(line)
    return result


def render_preview(pptx_path: Path, out_path: Path, columns: int) -> None:
    prs = Presentation(pptx_path)
    sw, sh = prs.slide_width, prs.slide_height
    thumb_w, thumb_h = 384, 216
    previews: list[Image.Image] = []
    for slide in prs.slides:
        canvas = Image.new("RGB", (1280, 720), _slide_bg_rgb(slide))
        draw = ImageDraw.Draw(canvas)
        for shape in slide.shapes:
            x = int(shape.left / sw * 1280)
            y = int(shape.top / sh * 720)
            w = max(1, int(shape.width / sw * 1280))
            h = max(1, int(shape.height / sh * 720))
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    with Image.open(io.BytesIO(shape.image.blob)) as im:
                        im = im.convert("RGB")
                        im.thumbnail((w, h), Image.Resampling.LANCZOS)
                        px = x + (w - im.width) // 2
                        py = y + (h - im.height) // 2
                        canvas.paste(im, (px, py))
                except Exception:
                    draw.rectangle((x, y, x + w, y + h), outline=(210, 220, 228), width=2)
                continue
            if getattr(shape, "has_table", False):
                table = shape.table
                rows, cols = len(table.rows), len(table.columns)
                rh, cw = h / rows, w / cols
                for rr in range(rows):
                    for cc in range(cols):
                        xx0, yy0 = int(x + cc * cw), int(y + rr * rh)
                        xx1, yy1 = int(x + (cc + 1) * cw), int(y + (rr + 1) * rh)
                        fill = (11, 31, 51) if rr == 0 else ((255, 255, 255) if rr % 2 else (237, 245, 247))
                        draw.rectangle((xx0, yy0, xx1, yy1), fill=fill, outline=(220, 228, 234))
                        f = _font(13 if rr == 0 else 12, rr == 0)
                        color = (255, 255, 255) if rr == 0 else (24, 48, 68)
                        lines = _wrap(draw, table.cell(rr, cc).text, f, max(20, xx1 - xx0 - 8))
                        for li, line in enumerate(lines[:3]):
                            draw.text((xx0 + 5, yy0 + 4 + li * 15), line, font=f, fill=color)
                continue
            try:
                if shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                    draw.rounded_rectangle((x, y, x + w, y + h), radius=min(12, h // 5), fill=_shape_rgb(shape), outline=None)
            except Exception:
                pass
            if getattr(shape, "has_text_frame", False) and shape.text.strip():
                first_size = 18
                bold = False
                color = INK
                try:
                    run = next(r for p in shape.text_frame.paragraphs for r in p.runs if r.text)
                    if run.font.size:
                        first_size = max(8, int(run.font.size.pt * 1.26))
                    bold = bool(run.font.bold)
                    if run.font.color and run.font.color.rgb:
                        color = str(run.font.color.rgb)
                except Exception:
                    pass
                f = _font(first_size, bold)
                fill = tuple(int(color[i:i+2], 16) for i in (0, 2, 4)) if len(color) == 6 else (24, 48, 68)
                lines = _wrap(draw, shape.text, f, max(16, w - 12))
                line_h = max(10, int(first_size * 1.18))
                max_lines = max(1, h // line_h)
                for li, line in enumerate(lines[:max_lines]):
                    draw.text((x + 6, y + 4 + li * line_h), line, font=f, fill=fill)
        previews.append(canvas.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS))
    rows = math.ceil(len(previews) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 24)), (226, 234, 240))
    label_font = _font(14, True)
    draw = ImageDraw.Draw(sheet)
    for i, preview in enumerate(previews):
        col, row = i % columns, i // columns
        x, y = col * thumb_w, row * (thumb_h + 24)
        sheet.paste(preview, (x, y))
        draw.text((x + 7, y + thumb_h + 3), f"{i + 1:02d}", font=label_font, fill=(11, 31, 51))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def validate_deck(path: Path, expected_slides: int) -> dict[str, object]:
    report: dict[str, object] = {"path": str(path.relative_to(PUBLICATION)), "sha256": sha256(path)}
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        report["zip_integrity"] = "PASS" if bad is None else f"FAIL:{bad}"
        report["media_files"] = len([n for n in zf.namelist() if n.startswith("ppt/media/")])
    prs = Presentation(path)
    report["slide_count"] = len(prs.slides)
    report["expected_slide_count"] = expected_slides
    report["slide_count_pass"] = len(prs.slides) == expected_slides
    titles: list[str] = []
    notes_present = 0
    out_of_bounds: list[dict[str, object]] = []
    min_font = 999.0
    for page, slide in enumerate(prs.slides, 1):
        texts = [shape.text.strip() for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
        titles.append(texts[0] if texts else "")
        if slide.notes_slide.notes_text_frame.text.strip():
            notes_present += 1
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > prs.slide_width + 2 or shape.top + shape.height > prs.slide_height + 2:
                out_of_bounds.append({"page": page, "shape": shape.name})
            if getattr(shape, "has_text_frame", False):
                for p in shape.text_frame.paragraphs:
                    for run in p.runs:
                        if run.font.size:
                            min_font = min(min_font, run.font.size.pt)
    report["all_slides_have_text"] = all(titles)
    report["speaker_notes_count"] = notes_present
    report["speaker_notes_pass"] = notes_present == expected_slides
    report["out_of_bounds"] = out_of_bounds
    report["minimum_explicit_font_pt"] = min_font if min_font < 999 else None
    report["file_size_bytes"] = path.stat().st_size
    report["status"] = "PASS" if all([report["zip_integrity"] == "PASS", report["slide_count_pass"], report["all_slides_have_text"], report["speaker_notes_pass"], not out_of_bounds]) else "FAIL"
    return report


def write_outline(short_outline: list[dict[str, object]], long_outline: list[dict[str, object]]) -> None:
    lines = [
        "# Endpoint NNS-ViT 汇报 PPT 页级提纲",
        "",
        f"- 内容来源：`zh/manuscript.md`",
        f"- 源稿 SHA-256：`{sha256(MANUSCRIPT)}`",
        "- 论文正式图：`figures/png/fig_01` 至 `fig_22`",
        "- 讲者备注：两套 PPT 每页均已写入",
        "",
    ]
    for heading, outline in [("20页重点版", short_outline), ("40页详细版", long_outline)]:
        lines += [f"## {heading}", ""]
        for item in outline:
            figures = "，".join(item["figures"]) if item["figures"] else "—"
            lines.append(f"{item['page']:02d}. {item['title']}  （图：{figures}）")
        lines.append("")
    (HERE / "slide_outline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    short_path, short_outline = build_short()
    long_path, long_outline = build_long()
    render_preview(short_path, QC_DIR / "contact_sheet_20.png", 4)
    render_preview(long_path, QC_DIR / "contact_sheet_40.png", 5)
    reports = {
        "schema_version": 1,
        "source_manuscript": str(MANUSCRIPT.relative_to(PUBLICATION)),
        "source_manuscript_sha256": sha256(MANUSCRIPT),
        "decks": [validate_deck(short_path, 20), validate_deck(long_path, 40)],
    }
    reports["status"] = "PASS" if all(d["status"] == "PASS" for d in reports["decks"]) else "FAIL"
    (QC_DIR / "pptx_validation.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_outline(short_outline, long_outline)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
