#!/usr/bin/env python3
"""Build a standalone Chinese Markdown manuscript and bilingual PDF drafts."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path

deps = os.environ.get("NNS_PDF_DEPS")
if deps:
    sys.path.insert(0, deps)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, ListFlowable
from reportlab.platypus import ListItem, PageBreak, PageTemplate, Paragraph
from reportlab.platypus import Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript"
FIGURE_ROOT = ROOT / "reports/nns_vit_unified_paper_figures_r10_r4_20260917"
OUTPUT = MANUSCRIPT / "output"
SECTIONS = ("abstract.md", "introduction.md", "related_work.md", "methods.md",
            "experimental.md", "results_discussion.md", "limitations.md", "conclusion.md")
TITLES = {
    "zh": "通过学习图像理解视觉 Transformer 的内部响应与贡献分配",
    "en": "Understanding Vision Transformers through Learning Images of Internal Responses and Contributions",
}
SUBTITLES = {
    "zh": "中文论文初稿 · R10-R4 图件绑定版",
    "en": "English manuscript draft · R10-R4 figure-bound edition",
}
ZH_FIGURE_TITLES = {
    1: "使用 NNS 扫描视觉 Transformer",
    2: "学习图像保留内部响应的输入来源",
    3: "在统一空间尺度上评价区域排序",
    4: "细粒度主效应分布",
    5: "忠实度轨迹",
    6: "尺度与算子敏感性",
    7: "层头坐标与预测分层",
    8: "中位效应图像的 144-head 图谱",
    9: "Transformer 深度方向的空间支持",
    10: "同一图像的不同解释",
    11: "解释方法的统计比较",
    12: "事后探索性多尺度机制",
    13: "ViT 各 block 的分支贡献分配",
    14: "事后探索性深度扩展",
    15: "一个内部成分对应一张输入空间图像",
    16: "保留每个输入值的原始位置",
    17: "构建并组合学习图像",
    18: "从 patch projection 到 Transformer 组件",
    19: "类别成分学习图像与 token 响应",
    20: "同一图像和类别目标下的不同 head",
    21: "内部成分重构总输入归因",
    22: "Attention 中的 Value 与 routing 导数",
    23: "Residual 路径对下游目标的贡献",
    24: "正常前向与学习图像的逐步总览",
    25: "Patch embedding 的数值与学习图像",
    26: "Attention 的数值与学习图像",
    27: "Residual 的数值与学习图像",
    28: "真实 ViT 三模块中的学习图像演化",
    29: "真实 patch-embedding 通道学习图像",
    30: "真实 attention 通道学习图像",
    31: "真实第一次 residual 学习图像",
    32: "真实 B0 MLP 分支学习图像",
    33: "真实 B0 block-output 学习图像",
    34: "从输入图像到预测类别 logit",
    35: "正确类、竞争类、margin 与低分输出的学习图像",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def demote_headings(text: str) -> str:
    return re.sub(r"^(#{1,5})(?=\s)", r"\1#", text, flags=re.MULTILINE)


def parse_captions(language: str) -> dict[int, str]:
    captions = {}
    current = None
    for line in (FIGURE_ROOT / f"captions_{language}.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^## Figure (\d+)\s*$", line)
        if match:
            current = int(match.group(1))
            captions[current] = []
        elif current is not None:
            captions[current].append(line)
    return {number: "\n".join(lines).strip() for number, lines in captions.items()}


def build_chinese_markdown() -> Path:
    target = MANUSCRIPT / "NNS_ViT_论文初稿_中文版.md"
    parts = [f"# {TITLES['zh']}", "", f"> {SUBTITLES['zh']} · 2026-09-17"]
    for name in SECTIONS:
        parts.extend(("", demote_headings((MANUSCRIPT / "zh" / name).read_text(encoding="utf-8").strip())))
    refs = (MANUSCRIPT / "references/references.md").read_text(encoding="utf-8").splitlines()
    parts.extend(("", "## 参考文献", "", "\n".join(refs[1:]).strip(), "", "## 图版与图注"))
    captions = parse_captions("zh")
    manifest = json.loads((MANUSCRIPT / "figure_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["figures"]:
        number = item["figure"]
        rel = Path("..") / item["png"]["path"]
        parts.extend(("", f"### 图 {number}. {ZH_FIGURE_TITLES[number]}", "",
                      f"![Figure {number}]({rel.as_posix()})", "", captions[number]))
    target.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return target


def register_fonts() -> tuple[str, str]:
    font_path = os.environ.get("NNS_PDF_FONT")
    if font_path and Path(font_path).is_file():
        pdfmetrics.registerFont(TTFont("NNSBody", font_path))
        pdfmetrics.registerFontFamily(
            "NNSBody", normal="NNSBody", bold="NNSBody",
            italic="NNSBody", boldItalic="NNSBody",
        )
        body = "NNSBody"
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        body = "STSong-Light"
    mono_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if mono_path.is_file():
        pdfmetrics.registerFont(TTFont("NNSMono", str(mono_path)))
        mono = "NNSMono"
    else:
        mono = body
    return body, mono


def plain_latex(text: str) -> str:
    replacements = (
        (r"\operatorname{", ""), (r"\mathrm{", ""), (r"\text{", ""),
        (r"\Delta", "Δ"), (r"\sigma", "σ"), (r"\alpha", "α"),
        (r"\ell", "ℓ"), (r"\approx", "≈"), (r"\times", "×"),
        (r"\rightarrow", "→"), (r"\top", "ᵀ"), (r"\sum", "Σ"),
        (r"\int", "∫"), (r"\partial", "∂"), (r"\qquad", "   "),
        (r"\left", ""), (r"\right", ""), (r"\lVert", "‖"),
        (r"\rVert", "‖"), (r"\,", " "),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.replace("{", "").replace("}", "").strip()


def inline_markup(text: str, mono: str) -> str:
    placeholders = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"@@PLACEHOLDER{len(placeholders)-1}@@"

    tick = chr(96)
    code_pattern = re.escape(tick) + r"([^" + re.escape(tick) + r"]+)" + re.escape(tick)
    text = re.sub(code_pattern, lambda m: stash(f'<font name="{mono}">{html.escape(m.group(1))}</font>'), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: m.group(1), text)
    text = re.sub(r"\$([^$]+)\$", lambda m: stash(html.escape(plain_latex(m.group(1)))), text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"@@PLACEHOLDER{index}@@", value)
    return text


def make_styles(body: str, mono: str):
    base = getSampleStyleSheet()
    common = dict(fontName=body, textColor=colors.HexColor("#172033"))
    return {
        "title": ParagraphStyle("DocTitle", parent=base["Title"], fontSize=22,
                                leading=29, alignment=TA_CENTER, spaceAfter=14, **common),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=10,
                                   leading=15, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#536078"), fontName=body),
        "h1": ParagraphStyle("Heading1", parent=base["Heading1"], fontSize=16,
                             leading=21, spaceBefore=14, spaceAfter=8,
                             keepWithNext=True, **common),
        "h2": ParagraphStyle("Heading2", parent=base["Heading2"], fontSize=13,
                             leading=18, spaceBefore=11, spaceAfter=6,
                             keepWithNext=True, **common),
        "h3": ParagraphStyle("Heading3", parent=base["Heading3"], fontSize=11,
                             leading=16, spaceBefore=9, spaceAfter=5,
                             keepWithNext=True, **common),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=9.2,
                               leading=14.2, alignment=TA_JUSTIFY,
                               firstLineIndent=14.7, spaceAfter=6, **common),
        "body_noindent": ParagraphStyle("BodyNoIndent", parent=base["BodyText"],
                                        fontSize=9.2, leading=14.2,
                                        alignment=TA_JUSTIFY, spaceAfter=6, **common),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontSize=9.2,
                                 leading=13.8, leftIndent=15, spaceAfter=3, **common),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontSize=7.4,
                                  leading=10.3, alignment=TA_LEFT,
                                  textColor=colors.HexColor("#3d4657"),
                                  fontName=body, spaceBefore=5),
        "equation": ParagraphStyle("Equation", parent=base["BodyText"],
                                   fontName=mono, fontSize=8.6, leading=12,
                                   alignment=TA_CENTER, leftIndent=10,
                                   rightIndent=10, spaceBefore=5, spaceAfter=7,
                                   textColor=colors.HexColor("#15233b")),
        "table": ParagraphStyle("TableText", parent=base["BodyText"], fontSize=7.2,
                                leading=9.2, fontName=body,
                                textColor=colors.HexColor("#172033")),
        "toc": ParagraphStyle("TOCHeading", parent=base["Heading1"], fontSize=18,
                              leading=22, spaceAfter=10, **common),
    }


class ManuscriptDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, language: str, body_font: str, **kwargs):
        super().__init__(filename, **kwargs)
        self.language = language
        self.body_font = body_font
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=self._decorate))
        self._bookmark_index = 0

    def beforeDocument(self):
        # multiBuild lays the document out repeatedly until the TOC stabilizes.
        # Reuse deterministic bookmark identifiers on every pass.
        self._bookmark_index = 0

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(self.body_font, 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        if doc.page > 1:
            canvas.drawString(self.leftMargin, A4[1] - 1.15 * cm, TITLES[self.language][:70])
        canvas.drawCentredString(A4[0] / 2, 0.72 * cm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Heading1", "Heading2"}:
            level = 0 if flowable.style.name == "Heading1" else 1
            text = flowable.getPlainText()
            key = f"heading-{self._bookmark_index}"
            self._bookmark_index += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def parse_markdown(text: str, style, mono: str):
    lines = text.splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.strip() == "$$":
            i += 1
            equation = []
            while i < len(lines) and lines[i].strip() != "$$":
                equation.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline_markup(plain_latex(" ".join(equation)), mono), style["equation"]))
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            story.append(Paragraph(inline_markup(heading.group(2), mono), style[f"h{level}"]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append([Paragraph(inline_markup(cell, mono), style["table"]) for cell in cells])
                i += 1
            if rows:
                widths = [17 * cm / len(rows[0])] * len(rows[0])
                table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab4c4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.extend((table, Spacer(1, 6)))
            continue
        if re.match(r"^[-*]\s+", line) or re.match(r"^\d+\.\s+", line):
            ordered = bool(re.match(r"^\d+\.\s+", line))
            items = []
            pattern = r"^\d+\.\s+" if ordered else r"^[-*]\s+"
            while i < len(lines) and re.match(pattern, lines[i]):
                value = re.sub(pattern, "", lines[i]).strip()
                items.append(ListItem(Paragraph(inline_markup(value, mono), style["bullet"])))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet",
                                      leftIndent=18, bulletFontName=style["body"].fontName,
                                      bulletFontSize=8))
            story.append(Spacer(1, 4))
            continue
        if line.startswith(">"):
            line = line.lstrip("> ")
        paragraph = [line]
        i += 1
        stop_pattern = r"^(#{1,6})\s+|^\$\$$|^[-*]\s+|^\d+\.\s+|^\|"
        while i < len(lines) and lines[i].strip() and not re.match(stop_pattern, lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph), mono), style["body"]))
    return story


def figure_flowables(language: str, style, mono: str):
    from PIL import Image as PILImage

    manifest = json.loads((MANUSCRIPT / "figure_manifest.json").read_text(encoding="utf-8"))
    captions = parse_captions(language)
    section_title = "图版与图注" if language == "zh" else "Figures and Captions"
    result = [PageBreak(), Paragraph(section_title, style["h1"])]
    for item in manifest["figures"]:
        number = item["figure"]
        image_path = ROOT / item["png"]["path"]
        with PILImage.open(image_path) as raster:
            width, height = raster.size
        scale = min(17.0 * cm / width, 11.7 * cm / height)
        figure = Image(str(image_path), width=width * scale, height=height * scale)
        if language == "zh":
            figure_title = f"图 {number}. {ZH_FIGURE_TITLES[number]}"
        else:
            figure_title = f"Figure {number}. {item['title']}"
        title = Paragraph(html.escape(figure_title), style["h2"])
        caption = Paragraph(inline_markup(captions[number].replace("\n", " "), mono), style["caption"])
        result.extend((PageBreak(), title, Spacer(1, 4), figure, caption))
    return result


def build_pdf(language: str, body_font: str, mono: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    filename = "NNS_ViT_论文初稿_中文版.pdf" if language == "zh" else "NNS_ViT_manuscript_draft_en.pdf"
    target = OUTPUT / filename
    style = make_styles(body_font, mono)
    story = [
        Spacer(1, 4.5 * cm), Paragraph(TITLES[language], style["title"]),
        Paragraph(SUBTITLES[language], style["subtitle"]), Spacer(1, 0.7 * cm),
        Paragraph("[AUTHOR REQUIRED] · [AFFILIATION REQUIRED]", style["subtitle"]),
        Spacer(1, 0.3 * cm), Paragraph("2026-09-17", style["subtitle"]), PageBreak(),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName=body_font, fontSize=9.5, leading=14,
                       leftIndent=0, firstLineIndent=0, spaceBefore=2),
        ParagraphStyle("TOC2", fontName=body_font, fontSize=8.5, leading=12,
                       leftIndent=14, firstLineIndent=0, spaceBefore=1),
    ]
    story.extend((Paragraph("目录" if language == "zh" else "Contents", style["toc"]),
                  toc, PageBreak()))
    for name in SECTIONS:
        text = (MANUSCRIPT / language / name).read_text(encoding="utf-8").strip()
        story.extend(parse_markdown(text, style, mono))
    story.extend((PageBreak(), Paragraph("参考文献" if language == "zh" else "References", style["h1"])))
    refs = (MANUSCRIPT / "references/references.md").read_text(encoding="utf-8").splitlines()
    for line in refs:
        if not line.strip() or line.startswith("#") or line.startswith(">"):
            continue
        story.append(Paragraph(inline_markup(line, mono), style["body_noindent"]))
    story.extend(figure_flowables(language, style, mono))
    doc = ManuscriptDocTemplate(
        str(target), language, body_font, pagesize=A4, leftMargin=2.0 * cm,
        rightMargin=2.0 * cm, topMargin=1.7 * cm, bottomMargin=1.5 * cm,
        title=TITLES[language], author="[AUTHOR REQUIRED]", subject=SUBTITLES[language],
    )
    doc.multiBuild(story)
    return target


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    markdown = build_chinese_markdown()
    body, mono = register_fonts()
    pdfs = [build_pdf(language, body, mono) for language in ("zh", "en")]
    receipt = {
        "status": "PASS_BUILD",
        "source_revision": "R10-R4",
        "standalone_chinese_markdown": {
            "path": str(markdown.relative_to(ROOT)),
            "sha256": sha256(markdown),
            "bytes": markdown.stat().st_size,
        },
        "pdfs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
             "bytes": path.stat().st_size}
            for path in pdfs
        ],
        "font": os.environ.get("NNS_PDF_FONT", "ReportLab STSong-Light fallback"),
        "figure_count": 35,
    }
    (OUTPUT / "build_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
