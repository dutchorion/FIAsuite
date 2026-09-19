"""Self-contained HTML case report (also the source for the PDF export)."""
from __future__ import annotations

import base64
import html
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


def _img_tag(bgr: np.ndarray, width: int = 860) -> str:
    h, w = bgr.shape[:2]
    if w > width:
        bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(enc).decode()}" width="{bgr.shape[1]}">'


def build_html(ev: Evidence, results: dict[str, Result], verdicts: dict[str, AxisVerdict], params: dict[str, dict],
               examiner: str = "", notes: str = "") -> str:
    e = html.escape
    out = [f"""<html><head><meta charset="utf-8"><title>{e(ev.path.name)} - forensic image report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;color:#1c1c1c;margin:32px;max-width:920px}}
h1{{font-size:20pt;margin-bottom:0}} h2{{font-size:14pt;border-bottom:2px solid #333;padding-bottom:3px;margin-top:28px}}
h3{{font-size:11.5pt;margin-bottom:4px}} table{{border-collapse:collapse;width:100%;font-size:9pt}}
td{{border:1px solid #ccc;padding:3px 6px;vertical-align:top;word-break:break-all}} td.k{{width:30%;background:#f3f3f3}}
.small{{color:#666;font-size:9pt}} .f{{margin:3px 0;font-size:10pt}}</style></head><body>
<h1>Forensic image analysis report</h1>
<p class="small">{APP_NAME} {__version__} &middot; generated {datetime.now():%Y-%m-%d %H:%M:%S}{' &middot; examiner: ' + e(examiner) if examiner else ''}</p>
<h2>Evidence</h2><table>
<tr><td class="k">File</td><td>{e(str(ev.path))}</td></tr>
<tr><td class="k">Size</td><td>{len(ev.data):,} bytes &middot; {ev.width} x {ev.height} px</td></tr>"""]
    out += [f'<tr><td class="k">{k.upper()}</td><td>{v}</td></tr>' for k, v in ev.hashes.items()]
    out.append(f"</table><p>{_img_tag(ev.bgr, 520)}</p>")
    if notes:
        out.append(f"<h2>Examiner notes</h2><p>{e(notes).replace(chr(10), '<br>')}</p>")

    out.append(f"<h2>Summary</h2><p>{e(headline(verdicts))}</p><table>")
    for v in verdicts.values():
        reasons = "<br>".join(f'<span style="color:{_SEV[s]}">&#9632;</span> {e(tool)}: {e(title)}' for s, tool, title in v.reasons[:6])
        out.append(f'<tr><td class="k"><b>{e(v.label)}</b><br><span style="color:{v.color}"><b>{e(v.level)}</b></span>'
                   f'<br><span class="small">combined score {v.score:.2f} from {v.tools_run} tools</span></td><td>{reasons or "-"}</td></tr>')
    out.append("</table><p class='small'>Scores are screening aids, not probabilities of forgery. Every finding must be "
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
            out.append(f'<p>{_img_tag(view)}<br><span class="small">{e(name)}</span></p>')
        rows = res.table[:40] if tool_id not in ("metadata",) else res.table[:400]
        if rows:
            out.append("<table>" + "".join(f'<tr><td class="k">{e(str(k))}</td><td>{e(str(v)[:1500])}</td></tr>' for k, v in rows) + "</table>")
    out.append("</body></html>")
    return "\n".join(out)
