"""Calibration harness: how well does a detector separate YOUR authentic images
from YOUR manipulated / synthetic ones, per laundering condition, at the
false-positive rates a review queue can live with. Scores and metrics are
uncalibrated until measured this way. (Method after imtriage's evaluate, MIT.)"""
from __future__ import annotations

import numpy as np

from . import launder
from .core import Evidence
from .tools import BY_ID, defaults

# tools that depend on the pixels only, so they stay meaningful after laundering
PIXEL_TOOLS = ["spectrum", "ai", "trufor", "ghosts", "grid", "dct", "noisemap", "cfa", "resampling", "clone_kp", "clone_blocks"]


def measure(path: str, tool_id: str, chains: list[str]) -> dict[str, dict[str, float]]:
    """{chain: {metric: value}} for one file; the screening score is reported as metric 'score'."""
    tool, ev, out = BY_ID[tool_id], Evidence(path), {}
    for chain in chains:
        res = tool.fn(launder.apply(ev, chain), **defaults(tool))
        values = dict(res.metrics)
        if res.score is not None:
            values["score"] = res.score
        out[chain] = values
    return out


def roc_summary(real: np.ndarray, fake: np.ndarray, fprs=(0.01, 0.05)) -> dict:
    """AUC by rank statistic, and TPR at fixed FPR with the threshold taken from the authentic set."""
    if not len(real) or not len(fake):
        return {"auc": None}
    allv = np.concatenate([real, fake])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for val in np.unique(allv):                    # average ranks over ties
        m = allv == val
        ranks[m] = ranks[m].mean()
    auc = (ranks[len(real):].sum() - len(fake) * (len(fake) + 1) / 2) / (len(real) * len(fake))
    out = {"auc": float(auc)}
    for f in fprs:
        thr = float(np.quantile(real, 1 - f, method="higher"))
        out[f"tpr@{f}"] = float(np.mean(fake > thr))
        out[f"thr@{f}"] = thr
    return out


def summarize(real: list[dict], fake: list[dict], chains: list[str]) -> list[dict]:
    """One row per laundering chain and metric."""
    rows = []
    metrics = sorted({m for rec in real + fake for values in rec.values() for m in values})
    for chain in chains:
        for metric in metrics:
            r = np.array([rec[chain][metric] for rec in real if metric in rec.get(chain, {})], float)
            f = np.array([rec[chain][metric] for rec in fake if metric in rec.get(chain, {})], float)
            rows.append({"laundering": chain, "metric": metric, "n_real": len(r), "n_fake": len(f), **roc_summary(r, f)})
    return rows
