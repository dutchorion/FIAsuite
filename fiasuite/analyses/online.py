"""Optional online detectors. Nothing here runs unless the user explicitly asks
for it in the UI: the image is uploaded to a third party."""
from __future__ import annotations

import json

import requests

from ..core import ALERT, INFO, NOTE, WARN, Evidence, Finding, Result, app_dir


def config_path():
    return app_dir() / "config.json"


def load_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config: dict) -> None:
    config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")


def sightengine(ev: Evidence) -> Result:
    """https://sightengine.com/docs/ai-generated-image-detection"""
    res = Result()
    cfg = load_config().get("sightengine", {})
    if not cfg.get("api_user") or not cfg.get("api_secret"):
        res.findings.append(Finding(NOTE, "Sightengine is not configured",
                                    "Enter your API user and secret under Settings. A free tier is available."))
        return res
    try:
        reply = requests.post("https://api.sightengine.com/1.0/check.json",
                              data={"models": "genai", "api_user": cfg["api_user"], "api_secret": cfg["api_secret"]},
                              files={"media": (ev.path.name, ev.data)}, timeout=60).json()
    except (requests.RequestException, ValueError) as exc:
        res.findings.append(Finding(NOTE, "Sightengine request failed", str(exc)[:300]))
        return res
    if reply.get("status") != "success":
        res.findings.append(Finding(NOTE, "Sightengine returned an error", json.dumps(reply.get("error", reply))[:300]))
        return res
    kind = reply.get("type", {})
    score = float(kind.get("ai_generated", 0))
    res.score = score
    res.table = [("AI-generated probability", f"{score:.1%}")]
    generators = sorted((kind.get("ai_generators") or {}).items(), key=lambda kv: -kv[1])
    res.table += [(f"  {name}", f"{value:.1%}") for name, value in generators]
    if score > 0.85:
        likely = f" Most likely generator: {generators[0][0]}." if generators and generators[0][1] > 0.3 else ""
        res.findings.append(Finding(ALERT, "Sightengine: AI-generated", f"Probability {score:.0%}.{likely}"))
    elif score > 0.5:
        res.findings.append(Finding(WARN, "Sightengine: probably AI-generated", f"Probability {score:.0%}."))
    else:
        res.findings.append(Finding(INFO, "Sightengine: not flagged", f"Probability {score:.0%}."))
    return res


PROVIDERS = {"Sightengine": sightengine}
