"""
pdf_report.py
-------------
Builds the downloadable PDF from the analysis result, the narrative and the
chart PNGs. Uses reportlab (pure Python, no system dependencies).

Fonts: matplotlib ships DejaVu Sans, which covers symbols such as the rupee
sign, so we reuse it instead of reportlab's built-in Helvetica.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .analysis import fmt_num
from .charts import RATIO

INK = colors.HexColor("#12263A")
MUTED = colors.HexColor("#64748B")
TEAL = colors.HexColor("#0F8B8D")
RULE = colors.HexColor("#D5DCE3")
TINT = colors.HexColor("#F1F5F7")


def _register_fonts() -> tuple[str, str]:
    try:
        folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVu", str(folder / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(folder / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                      italic="DejaVu", boldItalic="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _styles(font: str, bold: str) -> dict:
    return {
        "title": ParagraphStyle("title", fontName=bold, fontSize=22, leading=26, textColor=INK, spaceAfter=4),
        "meta": ParagraphStyle("meta", fontName=font, fontSize=9, leading=13, textColor=MUTED),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13, leading=16, textColor=INK, spaceBefore=16, spaceAfter=6),
        "body": ParagraphStyle("body", fontName=font, fontSize=10, leading=15, textColor=INK),
        "bullet": ParagraphStyle("bullet", fontName=font, fontSize=10, leading=14.5, textColor=INK,
                                 leftIndent=14, bulletIndent=2, spaceAfter=3),
        "caption": ParagraphStyle("caption", fontName=font, fontSize=8, leading=11, textColor=MUTED, spaceAfter=10),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=9, leading=12, textColor=INK),
    }


def _p(text, style) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _bullets(items, style) -> list:
    return [Paragraph(escape(str(t)), style, bulletText="•") for t in items]


def _footer(font: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.1 * cm, "Data Analyst Agent")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def build_pdf(result: dict, narrative: dict, charts: list[dict], out_path, question: str = "") -> str:
    font, bold = _register_fonts()
    st = _styles(font, bold)
    out_path = Path(out_path)
    filename = result["dataset"].get("filename", "data.csv")

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
        title=f"Data analysis report - {filename}", author="Data Analyst Agent",
    )
    width = A4[0] - 4 * cm
    story: list = []

    # Header
    story.append(_p("Data Analysis Report", st["title"]))
    story.append(_p(f"{filename}   |   {datetime.now():%d %b %Y, %H:%M}", st["meta"]))
    if question:
        story.append(_p(f"Question: {question}", st["meta"]))
    story.append(Spacer(1, 12))

    # KPI strip: 3 columns x up to 2 rows
    cards = result["kpi_cards"]
    cells = [
        Paragraph(
            f'<font size="8" color="#64748B">{escape(c["label"])}</font><br/>'
            f'<font size="{14 if len(c["value"]) <= 16 else 11}" name="{bold}">{escape(c["value"])}</font>', st["cell"])
        for c in cards
    ]
    while len(cells) % 3:
        cells.append("")
    rows = [cells[i:i + 3] for i in range(0, len(cells), 3)]
    kpi = Table(rows, colWidths=[width / 3] * 3)
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("LINEBEFORE", (1, 0), (-1, -1), 0.6, RULE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.6, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(kpi)

    # Narrative
    story.append(_p("Executive summary", st["h2"]))
    story.append(_p(narrative["executive_summary"], st["body"]))
    story.append(_p("Key findings", st["h2"]))
    story += _bullets(narrative["key_findings"], st["bullet"])

    # Charts
    if charts:
        story.append(_p("Charts", st["h2"]))
        for c in charts:
            img = Image(c["path"], width=width, height=width * RATIO)
            story.append(KeepTogether([img, Spacer(1, 10)]))

    # Data quality
    q = result["quality"]
    rows = [[_p("Duplicate rows", st["cell"]), _p(f'{q["duplicate_rows"]:,}', st["cell"])]]
    for m in q["missing"][:5]:
        rows.append([_p(f'Missing values: {m["column"]}', st["cell"]),
                     _p(f'{m["missing"]:,} ({m["missing_pct"]}%)', st["cell"])])
    for o in result["outliers"][:5]:
        rows.append([_p(f'Unusual values: {o["column"]}', st["cell"]),
                     _p(f'{o["count"]:,} ({o["pct"]}%), normal range {fmt_num(o["lower"])} to {fmt_num(o["upper"])}', st["cell"])])
    qt = Table(rows, colWidths=[width * 0.42, width * 0.58])
    qt.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(KeepTogether([_p("Data quality", st["h2"]), qt]))

    if narrative.get("recommendations"):
        story.append(_p("Recommendations", st["h2"]))
        story += _bullets(narrative["recommendations"], st["bullet"])
    if narrative.get("caveats"):
        story.append(_p("Notes and limits", st["h2"]))
        story += _bullets(narrative["caveats"], st["bullet"])

    origin = f"AI summary by {narrative['model']}" if narrative.get("source") == "ai" else "Automatic summary (no AI)"
    story.append(Spacer(1, 14))
    story.append(_p(origin, st["meta"]))

    footer = _footer(font)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(out_path)
