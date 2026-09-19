"""Combine tool results into three independent questions:

provenance  Is this file something other than a camera original?
edit        Were parts of the picture manipulated?
ai          Was the picture (or part of it) produced by generative AI?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .core import ALERT, NOTE, SEVERITY_ORDER, WARN, Result
from .tools import BY_ID

AXES = {"provenance": "File history", "edit": "Local manipulation", "ai": "AI generation"}
LEVELS = [(0.8, "Strong indication", "#e5484d"), (0.55, "Suspicious", "#f76b15"),
          (0.3, "Weak indication", "#f5d90a"), (0.0, "No indication found", "#46a758")]

_AI_WORDS = re.compile(r"\bAI\b|generat", re.I)


@dataclass
class AxisVerdict:
    label: str
    score: float = 0.0
    level: str = LEVELS[-1][1]
    color: str = LEVELS[-1][2]
    reasons: list[tuple[str, str, str]] = field(default_factory=list)   # (severity, tool title, finding title)
    tools_run: int = 0


def summarize(results: dict[str, Result]) -> dict[str, AxisVerdict]:
    out = {axis: AxisVerdict(label) for axis, label in AXES.items()}
    clean = {axis: 1.0 for axis in AXES}
    for tool_id, res in results.items():
        tool = BY_ID.get(tool_id)
        axis = tool.axis if tool else ("ai" if tool_id.startswith("online:") else None)
        if axis is None or res.score is None:
            continue
        title = tool.title if tool else tool_id.split(":", 1)[1]
        weight = tool.weight if tool else 1.2
        out[axis].tools_run += 1
        flagged = [f for f in res.findings if SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER[NOTE]]
        # declared AI generation in metadata / C2PA answers the AI question, not only the provenance one
        if tool_id in ("metadata", "c2pa"):
            ai_hits = [f for f in flagged if f.severity in (WARN, ALERT) and _AI_WORDS.search(f.title)]
            if ai_hits:
                clean["ai"] *= 1 - 0.97
                out["ai"].reasons += [(f.severity, title, f.title) for f in ai_hits]
        if res.score >= 0.3:                       # baseline scores must not accumulate
            clean[axis] *= 1 - min(res.score * weight, 0.99)
        out[axis].reasons += [(f.severity, title, f.title) for f in flagged]
    for axis, verdict in out.items():
        verdict.score = 1 - clean[axis]
        verdict.level, verdict.color = next((name, color) for t, name, color in LEVELS if verdict.score >= t)
        verdict.reasons.sort(key=lambda r: -SEVERITY_ORDER[r[0]])
        if verdict.tools_run == 0:
            verdict.level, verdict.color = "Not analysed", "#8b8d98"
    return out


def headline(verdicts: dict[str, AxisVerdict]) -> str:
    ai, edit, prov = verdicts["ai"], verdicts["edit"], verdicts["provenance"]
    parts = []
    if ai.score >= 0.55:
        parts.append("signs of AI generation")
    if edit.score >= 0.55:
        parts.append("signs of local manipulation")
    if parts:
        return "This image shows " + " and ".join(parts) + ". Review the flagged tools before concluding."
    if prov.score >= 0.55:
        return ("No manipulation was detected, but the file is not a camera original (re-saved, edited or stripped). "
                "Absence of findings is not proof of authenticity.")
    return ("No tool raised a significant finding. This does not prove authenticity: well-made forgeries, recompressed "
            "copies and modern AI generators can pass every test.")
