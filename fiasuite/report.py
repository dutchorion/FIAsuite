"""Case report: self-contained HTML, a paginated PDF with fitted, unbroken images, and a JSON record."""
from __future__ import annotations

import base64
import html
import re
from datetime import datetime

import cv2
import numpy as np

from . import APP_NAME, __version__
from .core import Evidence, Result
from .tools import BY_ID, TOOLS
from .verdict import AxisVerdict, headline


def build_json(ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict], params: dict[str, dict],
               examiner: str = "", notes: str = "") -> dict:
    """Machine-readable record of the examination: parameters, measurements and findings of every tool run."""
    return {
        "application": APP_NAME, "version": __version__, "generated": datetime.now().isoformat(timespec="seconds"),
        "examiner": examiner, "notes": notes,
        "evidence": {"path": str(ev.path), "bytes": len(ev.data), "width": ev.width, "height": ev.height, **ev.hashes},
        "verdicts": {axis: {"label": v.label, "level": v.level, "combined_score": round(v.score, 4), "tools_run": v.tools_run}
                     for axis, v in verdicts.items()},
        "calibrated": False,       # scores are screening aids until measured with the calibration tool
        "tools": {tool_id: {"parameters": params.get(tool_id, {}), "score": res.score, "metrics": res.metrics,
                            "findings": [{"severity": f.severity, "title": f.title, "detail": f.detail} for f in res.findings],
                            "table": [[str(k), str(v)] for k, v in res.table]}
                  for tool_id, res in results.items()},
    }


_SEV = {"info": "#6b7280", "note": "#b58900", "warn": "#d9480f", "alert": "#c92a2a"}
_DPI = 96.0                                    # CSS pixels per inch: the unit the PDF page is laid out in
_SOFT = re.compile(r"(\S{28})(?=\S)")         # long hashes and paths need break opportunities to stay inside a table


def _img_tag(bgr: np.ndarray, max_w: int, max_h: int | None = None, density: float = 1.0) -> str:
    """<img> scaled to fit max_w x max_h (never enlarged), aspect ratio kept. `density` stores more
    pixels than the display size so that print stays sharp."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_w / w, (max_h / h) if max_h else 1.0)
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    store = min(1.0, scale * density)
    if store < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(w * store)), max(1, round(h * store))), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(enc).decode()}" width="{dw}" height="{dh}">'


def build_html(ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict], params: dict[str, dict],
               examiner: str = "", notes: str = "", page: tuple[float, float] | None = None) -> str:
    """page=None: a web page. page=(width, height) in CSS pixels: laid out for that PDF page, every image
    fits the column and is short enough to share a page with its heading and caption."""
    pdf = page is not None
    img_w = int(page[0]) - 2 if pdf else 860
    img_h = int(page[1] * 0.72) if pdf else None
    density = 2.0 if pdf else 1.0

    def e(text) -> str:
        text = html.escape(str(text))
        return _SOFT.sub("\\1&#8203;", text) if pdf else text

    body = "margin:0" if pdf else "margin:32px;max-width:920px"
    out = [f"""<html><head><meta charset="utf-8"><title>{e(ev.path.name)} - forensic image report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;color:#1c1c1c;font-size:9.5pt;{body}}}
h1{{font-size:18pt;margin-bottom:0}} h2{{font-size:13pt;margin-top:22px;margin-bottom:6px}}
h3{{font-size:11pt;margin-top:16px;margin-bottom:3px}} table{{border-collapse:collapse;width:100%;font-size:8.5pt}}
td{{border:1px solid #bbb;padding:3px 6px;vertical-align:top;word-break:break-all}} td.k{{background:#f0f0f0}}
img{{max-width:100%;height:auto}} .small{{color:#666;font-size:8.5pt}} .f{{margin:3px 0}}</style></head><body>
<h1>Forensic image analysis report</h1>
<p class="small">{APP_NAME} {__version__} &middot; generated {datetime.now():%Y-%m-%d %H:%M:%S}{' &middot; examiner: ' + e(examiner) if examiner else ''}</p>
<h2>Evidence</h2>"""]

    def table(rows) -> str:
        cells = "".join(f'<tr><td class="k" width="30%">{k}</td><td width="70%">{v}</td></tr>' for k, v in rows)
        return f'<table width="100%" cellspacing="0" cellpadding="3">{cells}</table>'

    out.append(table([("File", e(ev.path)), ("Size", f"{len(ev.data):,} bytes &middot; {ev.width} x {ev.height} px"),
                      *[(k.upper(), e(v)) for k, v in ev.hashes.items()]]))
    out.append(f"<p>{_img_tag(ev.bgr, min(img_w, 520), img_h and min(img_h, 420), density)}</p>")
    if notes:
        out.append(f"<h2>Examiner notes</h2><p>{e(notes).replace(chr(10), '<br>')}</p>")

    out.append(f"<h2>Summary</h2><p>{e(headline(verdicts))}</p>")
    rows = []
    for v in verdicts.values():
        reasons = "<br>".join(f'<span style="color:{_SEV[s]}">&#9632;</span> {e(tool)}: {e(title)}' for s, tool, title in v.reasons[:6])
        rows.append((f'<b>{e(v.label)}</b><br><span style="color:{v.color}"><b>{e(v.level)}</b></span>'
                     f'<br><span class="small">combined score {v.score:.2f} from {v.tools_run} tools</span>', reasons or "-"))
    out.append(table(rows))
    out.append("<p class='small'>Scores are screening aids, not probabilities of forgery. Every finding must be "
               "reviewed by the examiner; absence of findings does not establish authenticity.</p>")

    out.append("<h2>Analyses</h2>")
    order = [t.id for t in TOOLS] + [k for k in results if k not in BY_ID]
    for tool_id in order:
        res = results.get(tool_id)
        if res is None:
            continue
        tool = BY_ID.get(tool_id)
        out.append(f"<h3>{e(tool.title if tool else tool_id)}</h3>")
        if tool:
            used = ", ".join(f"{k}={v}" for k, v in params.get(tool_id, {}).items())
            out.append(f'<p class="small">{e(tool.help)}{" Parameters: " + e(used) if used else ""}</p>')
        if res.score is not None:
            out.append(f'<p class="small">Screening score: {res.score:.2f}</p>')
        for f in res.findings:
            out.append(f'<p class="f"><span style="color:{_SEV[f.severity]}"><b>[{f.severity.upper()}]</b></span> '
                       f'<b>{e(f.title)}</b><br>{e(f.detail).replace(chr(10), "<br>")}</p>')
        for name, view in list(res.views.items())[:2]:
            out.append(f'<p>{_img_tag(view, img_w, img_h, density)}<br><span class="small">{e(name)}</span></p>')
        shown = res.table[:40] if tool_id not in ("metadata",) else res.table[:400]
        if shown:
            out.append(table([(e(k), e(str(v)[:1500])) for k, v in shown]))
    out.append("</body></html>")
    return "\n".join(out)


# ------------------------------------------------------------------ PDF

def layout_pdf(html_for_page, page_mm: tuple[float, float] = (210, 297), margin_mm: float = 15, footer_px: float = 26):
    """Lay the report out on real pages. Returns (document, page width, page height) in CSS pixels.
    html_for_page(width, height) -> HTML. Qt's rich-text engine has no 'keep together', so after the layout
    every block that holds an image (image + caption) and straddles a page boundary is moved to the next page,
    and a heading is never left alone at the bottom of a page."""
    from PySide6.QtCore import QSizeF
    from PySide6.QtGui import QTextCursor, QTextDocument, QTextFormat

    width = (page_mm[0] - 2 * margin_mm) / 25.4 * _DPI
    height = (page_mm[1] - 2 * margin_mm) / 25.4 * _DPI - footer_px
    doc = QTextDocument()
    doc.setDocumentMargin(0)
    doc.setPageSize(QSizeF(width, height))
    doc.setHtml(html_for_page(width, height))
    layout = doc.documentLayout()

    def has_image(block) -> bool:
        it = block.begin()
        while not it.atEnd():
            if it.fragment().charFormat().isImageFormat():
                return True
            it += 1
        return False

    def page_of(y: float) -> int:
        return int(y // height)

    for _ in range(400):                                        # each pass fixes the first offender, then re-measures
        block, moved = doc.begin(), False
        while block.isValid() and not moved:
            rect = layout.blockBoundingRect(block)
            fmt = block.blockFormat()
            already = fmt.pageBreakPolicy() & QTextFormat.PageBreak_AlwaysBefore
            straddles = has_image(block) and rect.height() < height and page_of(rect.top()) != page_of(rect.bottom() - 1)
            nxt = block.next()
            orphan = (fmt.headingLevel() > 0 and nxt.isValid()
                      and page_of(layout.blockBoundingRect(nxt).top()) != page_of(rect.top()))
            if (straddles or orphan) and not already:
                fmt.setPageBreakPolicy(QTextFormat.PageBreak_AlwaysBefore)
                QTextCursor(block).setBlockFormat(fmt)
                moved = True
            block = nxt
        if not moved:
            break
    return doc, width, height


def write_pdf(target: str, ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict],
              params: dict[str, dict], examiner: str = "", notes: str = "") -> int:
    """A4 report with fitted, unbroken images and a footer on every page. Returns the page count."""
    from PySide6.QtCore import QMarginsF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPageLayout, QPageSize, QPainter
    from PySide6.QtPrintSupport import QPrinter

    margin_mm, footer_px = 15, 26
    doc, width, height = layout_pdf(lambda w, h: build_html(ev, results, verdicts, params, examiner, notes, page=(w, h)),
                                    margin_mm=margin_mm, footer_px=footer_px)
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(target)
    printer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                                      QMarginsF(margin_mm, margin_mm, margin_mm, margin_mm), QPageLayout.Millimeter))
    painter = QPainter(printer)
    scale = printer.width() / width                             # device pixels per CSS pixel
    pages = doc.pageCount()
    stamp = f"{ev.path.name}   SHA-256 {ev.hashes['sha256'][:16]}..."
    for i in range(pages):
        if i:
            printer.newPage()
        painter.save()
        painter.scale(scale, scale)
        painter.translate(0, -i * height)
        doc.drawContents(painter, QRectF(0, i * height, width, height))
        painter.restore()
        painter.save()
        painter.scale(scale, scale)
        font = QFont("Segoe UI")
        font.setPixelSize(10)                                   # CSS pixels: the painter is already scaled to the page
        painter.setFont(font)
        painter.setPen(QColor("#777777"))
        line = QRectF(0, height + 8, width, footer_px - 8)
        painter.drawText(line, Qt.AlignLeft | Qt.AlignVCenter, stamp)
        painter.drawText(line, Qt.AlignRight | Qt.AlignVCenter, f"page {i + 1} of {pages}")
        painter.restore()
    painter.end()
    return pages
