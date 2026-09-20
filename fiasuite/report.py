"""Case report: self-contained HTML, a paginated PDF, and a JSON record.

PDF rules: every analysis starts on a new page; a page that continues a section carries
"<section> (continued)" at the top; images are fitted to the column and never split across pages.
"""
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


def ordered(results: dict[str, Result], include=None) -> list[str]:
    """Tool ids in report order, limited to `include` when given."""
    order = [t.id for t in TOOLS] + [k for k in results if k not in BY_ID]
    return [k for k in order if k in results and (include is None or k in include)]


def title_of(tool_id: str) -> str:
    if tool_id in BY_ID:
        return BY_ID[tool_id].title
    return tool_id.split(":", 1)[1] + " (online)" if tool_id.startswith("online:") else tool_id


def build_json(ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict], params: dict[str, dict],
               examiner: str = "", notes: str = "", include=None) -> dict:
    """Machine-readable record of the examination: parameters, measurements and findings of every tool run."""
    return {
        "application": APP_NAME, "version": __version__, "generated": datetime.now().isoformat(timespec="seconds"),
        "examiner": examiner, "notes": notes,
        "evidence": {"path": str(ev.path), "bytes": len(ev.data), "width": ev.width, "height": ev.height, **ev.hashes},
        "verdicts": {axis: {"label": v.label, "level": v.level, "combined_score": round(v.score, 4), "tools_run": v.tools_run}
                     for axis, v in verdicts.items()},
        "calibrated": False,       # scores are screening aids until measured with the calibration tool
        "tools": {tool_id: {"parameters": params.get(tool_id, {}), "score": results[tool_id].score,
                            "metrics": results[tool_id].metrics,
                            "findings": [{"severity": f.severity, "title": f.title, "detail": f.detail}
                                         for f in results[tool_id].findings],
                            "table": [[str(k), str(v)] for k, v in results[tool_id].table]}
                  for tool_id in ordered(results, include)},
    }


_SEV = {"info": "#6b7280", "note": "#b58900", "warn": "#d9480f", "alert": "#c92a2a"}
_DPI = 96.0                                    # CSS pixels per inch: the unit the PDF page is laid out in
_SOFT = re.compile(r"(\S{28})(?=\S)")         # long hashes and paths need break opportunities to stay inside a table
_BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
_BREAK = ' style="page-break-before:always"'


def _img_tag(bgr: np.ndarray, max_w: int, max_h: int | None = None, density: float = 1.0, placeholder: bool = False) -> str:
    """<img> scaled to fit max_w x max_h (never enlarged), aspect ratio kept. `density` stores more pixels than
    the display size so that print stays sharp. placeholder: same box, no pixel data (fast layout measuring)."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_w / w, (max_h / h) if max_h else 1.0)
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    if placeholder:
        return f'<img src="{_BLANK}" width="{dw}" height="{dh}">'
    store = min(1.0, scale * density)
    if store < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(w * store)), max(1, round(h * store))), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(enc).decode()}" width="{dw}" height="{dh}">'


def build_html(ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict], params: dict[str, dict],
               examiner: str = "", notes: str = "", include=None, page: tuple[float, float] | None = None,
               breaks: dict | None = None, placeholder: bool = False) -> str:
    """page=None: a web page. page=(width, height) in CSS pixels: the PDF layout, where every analysis starts a new
    page. breaks (found by layout_pdf) numbers the blocks that need a different treatment: "image" / "heading" get a
    page break before them, "flow" headings lose their forced break because they already flow onto a new page."""
    breaks = breaks or {}
    image_breaks, heading_breaks, flow = (breaks.get(k, ()) for k in ("image", "heading", "flow"))
    pdf = page is not None
    img_w = int(page[0]) - 2 if pdf else 860
    img_h = int(page[1] * 0.72) if pdf else None
    density = 2.0 if pdf else 1.0
    counters = {"image": 0, "heading": 0}

    def e(text) -> str:
        text = html.escape(str(text))
        return _SOFT.sub("\\1&#8203;", text) if pdf else text

    def heading(level: int, text: str, new_page: bool = False) -> str:
        index = counters["heading"]
        counters["heading"] += 1
        forced = pdf and (new_page or index in heading_breaks) and index not in flow
        style = ' style="page-break-before:always;margin-top:0"' if forced else ""
        return f"<h{level}{style}>{e(text)}</h{level}>"

    def image(bgr: np.ndarray, caption: str = "", max_w: int | None = None, max_h: int | None = None) -> str:
        index = counters["image"]
        counters["image"] += 1
        style = _BREAK if index in image_breaks else ""
        tag = _img_tag(bgr, max_w or img_w, max_h or img_h, density, placeholder)
        return f'<p{style}>{tag}' + (f'<br><span class="small">{e(caption)}</span>' if caption else "") + "</p>"

    def table(rows) -> str:
        cells = "".join(f'<tr><td class="k" width="30%">{k}</td><td width="70%">{v}</td></tr>' for k, v in rows)
        return f'<table width="100%" cellspacing="0" cellpadding="3">{cells}</table>'

    body = "margin:0" if pdf else "margin:32px;max-width:920px"
    out = [f"""<html><head><meta charset="utf-8"><title>{e(ev.path.name)} - forensic image report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;color:#1c1c1c;font-size:9.5pt;{body}}}
h1{{font-size:18pt;margin-bottom:0}} h2{{font-size:13pt;margin-top:20px;margin-bottom:6px}}
table{{border-collapse:collapse;width:100%;font-size:8.5pt}}
td{{border:1px solid #bbb;padding:3px 6px;vertical-align:top;word-break:break-all}} td.k{{background:#f0f0f0}}
img{{max-width:100%;height:auto}} .small{{color:#666;font-size:8.5pt}} .f{{margin:3px 0}}</style></head><body>""",
           heading(1, "Forensic image analysis report"),
           f'<p class="small">{APP_NAME} {__version__} &middot; generated {datetime.now():%Y-%m-%d %H:%M:%S}'
           f'{" &middot; examiner: " + e(examiner) if examiner else ""}</p>',
           heading(2, "Evidence"),
           table([("File", e(ev.path)), ("Size", f"{len(ev.data):,} bytes &middot; {ev.width} x {ev.height} px"),
                  *[(k.upper(), e(v)) for k, v in ev.hashes.items()]]),
           image(ev.bgr, "", min(img_w, 520), 420)]
    if notes:
        out += [heading(2, "Examiner notes"), f"<p>{e(notes).replace(chr(10), '<br>')}</p>"]

    out += [heading(2, "Summary"), f"<p>{e(headline(verdicts))}</p>"]
    rows = []
    for v in verdicts.values():
        reasons = "<br>".join(f'<span style="color:{_SEV[s]}">&#9632;</span> {e(tool)}: {e(title)}' for s, tool, title in v.reasons[:6])
        rows.append((f'<b>{e(v.label)}</b><br><span style="color:{v.color}"><b>{e(v.level)}</b></span>'
                     f'<br><span class="small">combined score {v.score:.2f} from {v.tools_run} tools</span>', reasons or "-"))
    selected = ordered(results, include)
    out += [table(rows),
            "<p class='small'>Scores are screening aids, not probabilities of forgery. Every finding must be reviewed by "
            "the examiner; absence of findings does not establish authenticity.</p>",
            f"<p class='small'>Analyses in this report ({len(selected)} of {len(results)} run): "
            + e(", ".join(title_of(k) for k in selected)) + "</p>"]

    for tool_id in selected:
        res, tool = results[tool_id], BY_ID.get(tool_id)
        out.append(heading(2, title_of(tool_id), new_page=True))
        if tool:
            used = ", ".join(f"{k}={v}" for k, v in params.get(tool_id, {}).items())
            out.append(f'<p class="small">{e(tool.help)}{" Parameters: " + e(used) if used else ""}</p>')
        if res.score is not None:
            out.append(f'<p class="small">Screening score: {res.score:.2f}</p>')
        for f in res.findings:
            out.append(f'<p class="f"><span style="color:{_SEV[f.severity]}"><b>[{f.severity.upper()}]</b></span> '
                       f'<b>{e(f.title)}</b><br>{e(f.detail).replace(chr(10), "<br>")}</p>')
        for name, view in list(res.views.items())[:2]:
            out.append(image(view, name))
        shown = res.table[:40] if tool_id not in ("metadata",) else res.table[:400]
        if shown:
            out.append(table([(e(k), e(str(v)[:1500])) for k, v in shown]))
    out.append("</body></html>")
    return "\n".join(out)


# ------------------------------------------------------------------ PDF

def _document(markup: str, width: float, height: float):
    from PySide6.QtCore import QSizeF
    from PySide6.QtGui import QTextDocument
    doc = QTextDocument()
    doc.setDocumentMargin(0)
    doc.setPageSize(QSizeF(width, height))
    doc.setHtml(markup)
    doc.documentLayout().documentSize()                          # forces the complete layout
    return doc


def _blocks(doc):
    """(kind, index, block, rect) for every heading and every block that holds an image, in document order."""
    layout, block, counts = doc.documentLayout(), doc.begin(), {"image": 0, "heading": 0}
    while block.isValid():
        kind = "heading" if block.blockFormat().headingLevel() > 0 else None
        it = block.begin()
        while kind is None and not it.atEnd():
            if it.fragment().charFormat().isImageFormat():
                kind = "image"
            it += 1
        if kind:
            yield kind, counts[kind], block, layout.blockBoundingRect(block)
            counts[kind] += 1
        block = block.next()


def layout_pdf(make_html, page_mm: tuple[float, float] = (210, 297), margin_mm: float = 15,
               header_px: float = 30, footer_px: float = 26):
    """Returns (document, page width, page height) in CSS pixels. make_html(page, breaks, placeholder) -> HTML.

    Qt's rich text has no 'keep together', and editing a laid-out document makes it drop content when painted. So
    the breaks are found in measuring rounds on a light copy (images replaced by empty boxes of identical size):
    the first image block that straddles a page boundary, or heading left alone at a page bottom, is recorded, the
    HTML is rebuilt with a page break before it, and the round repeats. The final document is built once, with the
    real images, and never modified.

    A third correction: when the previous section ends within a few pixels of the page bottom, the next heading
    flows to a new page by itself and Qt applies the forced break on top, leaving a blank page. Such headings are
    recorded under "flow" and lose their forced break."""
    # whole pixels: Qt rounds block positions, a fractional page height puts page-top blocks a hair before the boundary
    width = float(int((page_mm[0] - 2 * margin_mm) / 25.4 * _DPI))
    height = float(int((page_mm[1] - 2 * margin_mm) / 25.4 * _DPI - header_px - footer_px))
    breaks = {"image": set(), "heading": set(), "flow": set()}

    def page_of(y: float) -> int:
        return int((y + 0.5) // height)

    for _ in range(300):
        doc = _document(make_html((width, height), breaks, True), width, height)
        found = None
        items = list(_blocks(doc))
        for kind, index, block, rect in items:
            before = block.previous()
            if (kind == "heading" and index not in breaks["flow"] and before.isValid()
                    and block.blockFormat().pageBreakPolicy().value
                    and page_of(doc.documentLayout().blockBoundingRect(before).bottom() - 1) < page_of(rect.top()) - 1):
                found = ("flow", index)                         # a blank page lies before this heading
                break
            if index in breaks[kind]:
                continue
            if kind == "image" and rect.height() < height and page_of(rect.top()) != page_of(rect.bottom() - 1):
                found = (kind, index)
            elif kind == "heading" and block.next().isValid():
                after = doc.documentLayout().blockBoundingRect(block.next())
                if page_of(after.top()) != page_of(rect.top()):
                    found = (kind, index)
            if found:
                break
        if not found:
            break
        breaks[found[0]].add(found[1])
    final = _document(make_html((width, height), breaks, False), width, height)
    return final, width, height


def continued_titles(doc, height: float) -> dict[int, str]:
    """page index -> title of the section that runs on from the previous page (pages that open a section are absent)."""
    sections = [(rect.top(), block.text()) for kind, _, block, rect in _blocks(doc)
                if kind == "heading" and block.blockFormat().headingLevel() >= 2]
    out = {}
    for page in range(1, doc.pageCount()):
        top = page * height
        if any(-1 <= y - top < 40 for y, _ in sections):        # a section opens this page
            continue
        running = [title for y, title in sections if y < top - 1]
        if running:
            out[page] = running[-1]
    return out


def write_pdf(target: str, ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict],
              params: dict[str, dict], examiner: str = "", notes: str = "", include=None) -> int:
    """A4 report. Returns the page count."""
    from PySide6.QtCore import QMarginsF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPageLayout, QPageSize, QPainter, QPen
    from PySide6.QtPrintSupport import QPrinter

    margin_mm, header_px, footer_px = 15, 30, 26

    def make_html(page, breaks, placeholder):
        return build_html(ev, results, verdicts, params, examiner, notes, include, page, breaks, placeholder)

    doc, width, height = layout_pdf(make_html, margin_mm=margin_mm, header_px=header_px, footer_px=footer_px)
    continued = continued_titles(doc, height)
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(target)
    printer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                                      QMarginsF(margin_mm, margin_mm, margin_mm, margin_mm), QPageLayout.Millimeter))
    painter = QPainter(printer)
    scale = printer.width() / width                             # device pixels per CSS pixel
    pages = doc.pageCount()
    stamp = f"{ev.path.name}   SHA-256 {ev.hashes['sha256'][:16]}..."

    def font(px: int, bold: bool = False) -> QFont:
        f = QFont("Segoe UI")
        f.setPixelSize(px)                                      # CSS pixels: the painter is already scaled to the page
        f.setBold(bold)
        return f

    for i in range(pages):
        if i:
            printer.newPage()
        painter.save()
        painter.scale(scale, scale)
        if i in continued:                                      # running title of a section that continues here
            painter.setFont(font(15, bold=True))
            painter.setPen(QColor("#1c1c1c"))
            painter.drawText(QRectF(0, 0, width, header_px - 8), Qt.AlignLeft | Qt.AlignVCenter, f"{continued[i]} (continued)")
            painter.setPen(QPen(QColor("#bbbbbb"), 1))
            painter.drawLine(0, int(header_px - 5), int(width), int(header_px - 5))
        painter.setFont(font(10))
        painter.setPen(QColor("#777777"))
        line = QRectF(0, header_px + height + 8, width, footer_px - 8)
        painter.drawText(line, Qt.AlignLeft | Qt.AlignVCenter, stamp)
        painter.drawText(line, Qt.AlignRight | Qt.AlignVCenter, f"page {i + 1} of {pages}")
        painter.translate(0, header_px - i * height)
        doc.drawContents(painter, QRectF(0, i * height, width, height))
        painter.restore()
    painter.end()
    return pages
