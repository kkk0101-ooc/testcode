#!/usr/bin/env python3
"""Render the Chinese Markdown thesis as a generic A4 anonymous review PDF."""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "zh" / "manuscript.md"
OUTPUT = PACKAGE / "NNS_ViT_学位论文_匿名内容审阅版.pdf"
FORMULA_CACHE = PACKAGE / "qc" / "formula_cache"
def clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace(chr(96), "")
    return html.escape(text, quote=False).replace("\n", "<br/>")


def formula_image(math_lines, width):
    formula = " ".join(math_lines)
    digest = hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16]
    FORMULA_CACHE.mkdir(parents=True, exist_ok=True)
    path = FORMULA_CACHE / (digest + ".png")
    if not path.is_file():
        fig = plt.figure(figsize=(8.0, 0.75), dpi=180)
        fig.patch.set_alpha(0)
        fig.text(0.5, 0.5, "$" + formula + "$", ha="center", va="center", fontsize=13, color="black")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, transparent=True)
        plt.close(fig)
    img = Image(str(path))
    ratio = min(width * 0.92 / img.imageWidth, 16 * mm / img.imageHeight)
    img.drawWidth = img.imageWidth * ratio
    img.drawHeight = img.imageHeight * ratio
    img.hAlign = "CENTER"
    return img


def styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN", parent=base["Title"], fontName="STSong-Light", fontSize=23, leading=34, alignment=TA_CENTER, spaceAfter=18 * mm, wordWrap="CJK"),
        "subtitle": ParagraphStyle("SubtitleCN", parent=base["Normal"], fontName="STSong-Light", fontSize=12, leading=20, alignment=TA_CENTER, wordWrap="CJK"),
        "h1": ParagraphStyle("H1CN", parent=base["Heading1"], fontName="STSong-Light", fontSize=18, leading=27, spaceAfter=9 * mm, alignment=TA_CENTER, wordWrap="CJK"),
        "h2": ParagraphStyle("H2CN", parent=base["Heading2"], fontName="STSong-Light", fontSize=13, leading=20, spaceBefore=2.5 * mm, spaceAfter=1.0 * mm, wordWrap="CJK"),
        "body": ParagraphStyle("BodyCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=10.2, leading=14, firstLineIndent=20.4, alignment=TA_JUSTIFY, spaceAfter=0.7 * mm, splitLongWords=True, wordWrap="CJK"),
        "caption": ParagraphStyle("CaptionCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=9, leading=14, alignment=TA_CENTER, spaceAfter=3 * mm, wordWrap="CJK"),
        "formula": ParagraphStyle("FormulaCN", parent=base["Code"], fontName="STSong-Light", fontSize=8.1, leading=12, alignment=TA_CENTER, leftIndent=4 * mm, rightIndent=4 * mm, spaceAfter=2 * mm, wordWrap="CJK"),
        "label": ParagraphStyle("LabelCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=9.5, leading=15, alignment=TA_CENTER, spaceBefore=2 * mm, spaceAfter=1 * mm, wordWrap="CJK"),
        "list": ParagraphStyle("ListCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=10, leading=16, leftIndent=7 * mm, firstLineIndent=-4 * mm, spaceAfter=1.2 * mm, wordWrap="CJK"),
        "ref": ParagraphStyle("RefCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=8.6, leading=13, leftIndent=7 * mm, firstLineIndent=-7 * mm, spaceAfter=1.5 * mm, wordWrap="CJK"),
        "cell": ParagraphStyle("CellCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=6.2, leading=8, wordWrap="CJK"),
    }


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"匿名内容审阅版 · 第 {doc.page} 页")
    canvas.restoreState()


def parse_table(lines, style, width):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append([Paragraph(clean_inline(c), style) for c in cells])
    if not rows:
        return Spacer(1, 1)
    cols = len(rows[0])
    weights = [1.0] * cols
    if cols >= 6:
        weights[0] = 1.15
        weights[1] = 1.05
    total = sum(weights)
    table = Table(rows, colWidths=[width * w / total for w in weights], repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def build_story(text: str, st, width):
    story = []
    lines = text.splitlines()
    i = 0
    title_seen = False
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            heading = line[2:]
            if not title_seen:
                title_seen = True
                story.extend([Spacer(1, 48 * mm), Paragraph(clean_inline(heading), st["title"]), Paragraph("硕士学位论文 · 匿名内容审阅版", st["subtitle"]), Spacer(1, 95 * mm), Paragraph("本审阅版不含作者、单位、学号或导师信息", st["subtitle"]), PageBreak()])
            else:
                story.extend([PageBreak(), Paragraph(clean_inline(heading), st["h1"])])
            i += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            path = (SOURCE.parent / image_match.group(2)).resolve()
            img = Image(str(path))
            ratio = min(width / img.imageWidth, 95 * mm / img.imageHeight)
            img.drawWidth = img.imageWidth * ratio
            img.drawHeight = img.imageHeight * ratio
            img.hAlign = "CENTER"
            parts = [img]
            if i + 1 < len(lines) and lines[i + 1].startswith("**图"):
                parts.extend([Spacer(1, 1.5 * mm), Paragraph(clean_inline(lines[i + 1]), st["caption"])])
                i += 1
            story.append(KeepTogether(parts))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), st["h2"]))
            i += 1
            continue
        if line == "$$":
            math_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i].strip())
                i += 1
            try:
                story.append(formula_image(math_lines, width))
            except Exception:
                story.append(Paragraph(clean_inline(" ".join(math_lines)), st["formula"]))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.extend([parse_table(table_lines, st["cell"], width), Spacer(1, 3 * mm)])
            continue
        if line.startswith("**式（") or line.startswith("**表 ") or line.startswith("**命题"):
            story.append(Paragraph(clean_inline(line), st["label"]))
            i += 1
            continue
        if re.match(r"^\d+\. \*\*", line):
            story.append(Paragraph(clean_inline(line), st["list"]))
            i += 1
            continue
        if re.match(r"^\[\d+\] ", line):
            para = [line]
            i += 1
            while i < len(lines) and lines[i] and not re.match(r"^\[\d+\] ", lines[i]) and not lines[i].startswith("#"):
                para.append(lines[i])
                i += 1
            story.append(Paragraph(clean_inline(" ".join(para)), st["ref"]))
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i] and not lines[i].startswith(("#", "|", "![", "$$", "**式（", "**表 ", "**命题")) and not re.match(r"^\d+\. \*\*", lines[i]) and not re.match(r"^\[\d+\] ", lines[i]):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(clean_inline(" ".join(para)), st["body"]))
    return story


def main():
    st = styles()
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=24 * mm, leftMargin=26 * mm, topMargin=24 * mm, bottomMargin=22 * mm, title="视觉 Transformer 计算图上的 NNS 结构路径分解与系统验证", author="")
    width = A4[0] - doc.leftMargin - doc.rightMargin
    doc.build(build_story(SOURCE.read_text(encoding="utf-8"), st, width), onFirstPage=page_number, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
