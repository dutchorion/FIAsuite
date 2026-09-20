"""Optional online detectors. Nothing here runs unless the user explicitly asks
for it in the UI: the image is uploaded to a third party."""
from __future__ import annotations

import json
import os

import requests

from ..core import ALERT, INFO, NOTE, WARN, Evidence, Finding, Result, app_dir


def config_path():
    return app_dir() / "config.json"


def load_config() -> dict:
    try:
        config = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def save_config(config: dict) -> bool:
    """Written to a temporary file and swapped in, so a crash cannot truncate the stored API keys.
    False when the folder is read-only (write-protected stick): settings then last for the session only."""
    temp = config_path().with_suffix(".json.tmp")
    try:
        temp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temp, config_path())
        return True
    except OSError:
        return False


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


def _upload_type(ev: Evidence) -> str | None:
    if ev.is_jpeg:
        return "image/jpeg"
    if ev.is_png:
        return "image/png"
    if ev.data[:4] == b"RIFF" and ev.data[8:12] == b"WEBP":
        return "image/webp"
    return None


def openai_provenance(ev: Evidence) -> Result:
    """OpenAI content provenance check: C2PA Content Credentials and the SynthID watermark in content made with
    OpenAI tools. https://developers.openai.com/api/docs/guides/content-provenance"""
    res = Result()
    key = load_config().get("openai", {}).get("api_key")
    if not key:
        res.findings.append(Finding(NOTE, "OpenAI is not configured", "Enter your OpenAI API key under Settings."))
        return res
    mime = _upload_type(ev)
    if mime is None or len(ev.data) > 50 * 2 ** 20:
        res.findings.append(Finding(NOTE, "File not accepted by OpenAI",
                                    "The service takes PNG, JPEG and WebP files up to 50 MiB. Converting the file would "
                                    "destroy its Content Credentials, so it was not sent."))
        return res
    try:
        reply = requests.post("https://api.openai.com/v1/content_provenance_checks",
                              headers={"Authorization": f"Bearer {key}"},
                              files={"file": (ev.path.name, ev.data, mime)}, timeout=120)
    except requests.RequestException as exc:
        res.findings.append(Finding(NOTE, "OpenAI request failed", str(exc)[:300]))
        return res
    if reply.status_code == 401:
        res.findings.append(Finding(NOTE, "OpenAI rejected the API key", "Check the key under Settings."))
        return res
    if reply.status_code == 429:
        wait = reply.headers.get("Retry-After", "a while")
        res.findings.append(Finding(NOTE, "OpenAI rate limit reached",
                                    f"This service is strictly rate-limited. Try again in {wait} seconds."))
        return res
    try:
        body = reply.json()
    except ValueError:
        body = {}
    if reply.status_code != 200 or "results" not in body:
        message = (body.get("error") or {}).get("message", reply.text[:300]) if isinstance(body, dict) else reply.text[:300]
        res.findings.append(Finding(NOTE, f"OpenAI returned an error (HTTP {reply.status_code})", str(message)[:300]))
        return res

    detected = []
    for item in body["results"]:
        kind = str(item.get("type", "?"))
        label = {"c2pa": "Content Credentials (C2PA)", "synthid": "SynthID watermark"}.get(kind, kind)
        outcome = str(item.get("outcome", "?"))
        extra = [f"{name}: {item[field]}" for name, field in (("validation", "validation_state"), ("issuer", "issuer"),
                 ("model", "model"), ("generated", "generated_at")) if item.get(field)]
        res.table.append((label, outcome.replace("_", " ") + ("   " + ", ".join(extra) if extra else "")))
        if outcome == "detected":
            detected.append((kind, label, item))
    for kind, label, item in detected:
        origin = " by " + str(item["model"]) if item.get("model") else ""
        when = " on " + str(item["generated_at"]) if item.get("generated_at") else ""
        if kind == "c2pa" and item.get("validation_state") == "invalid":
            res.findings.append(Finding(WARN, "OpenAI Content Credentials found but invalid",
                                        "The manifest names OpenAI but fails validation: the file was changed after signing."))
        else:
            res.findings.append(Finding(ALERT, f"OpenAI: {label} detected",
                f"Generated with OpenAI tools{origin}{when}. "
                + ("The watermark is part of the pixels and survives screenshots, resizing and recompression."
                   if kind == "synthid" else "Signed provenance metadata embedded in the file.")))
    if detected:
        res.score = 0.97
    else:
        res.score = 0.05
        res.findings.append(Finding(INFO, "OpenAI: no provenance signal detected",
            "This does NOT show the image is authentic or human-made: the check only recognises content from OpenAI's "
            "own tools, older OpenAI images carry no watermark, and metadata is lost when a file is re-saved."))
    return res


# name -> (function, what the tool does; shown in the tool panel)
PROVIDERS = {
    "Sightengine": (sightengine, "Commercial AI-image detector with per-generator scores (Midjourney, DALL-E, Flux, "
                                 "Stable Diffusion, Firefly). Covers newer generators than the local models."),
    "OpenAI provenance check": (openai_provenance, "OpenAI's official check for content made with its tools (ChatGPT, API): "
                                "looks for OpenAI's C2PA Content Credentials and for the invisible SynthID watermark, which "
                                "survives screenshots and recompression. Says nothing about other generators. Uploaded files "
                                "are not covered by OpenAI's zero-data-retention option."),
}
