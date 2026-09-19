"""OpenAI provenance connector against a mocked API (no key, no network)."""
import cv2
import numpy as np
import pytest

from fiasuite.analyses import online
from fiasuite.core import Evidence


class Reply:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture()
def png(tmp_path):
    path = tmp_path / "e.png"
    cv2.imwrite(str(path), np.full((64, 64, 3), 127, np.uint8))
    return Evidence(path)


@pytest.fixture()
def api(monkeypatch):
    calls = []
    monkeypatch.setattr(online, "load_config", lambda: {"openai": {"api_key": "sk-test"}})

    def install(reply):
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return reply
        monkeypatch.setattr(online.requests, "post", post)
        return calls
    return install


def test_detected_watermark_and_credentials(api, png):
    calls = api(Reply(body={"object": "content_provenance_check", "results": [
        {"type": "c2pa", "outcome": "detected", "validation_state": "trusted", "issuer": "OpenAI OpCo, LLC",
         "model": "gpt-image-1", "generated_at": "2026-08-01T10:00:00Z"},
        {"type": "synthid", "outcome": "detected", "model": None, "generated_at": None}]}))
    res = online.openai_provenance(png)
    url, kwargs = calls[0]
    assert url == "https://api.openai.com/v1/content_provenance_checks"
    assert kwargs["headers"] == {"Authorization": "Bearer sk-test"}
    assert kwargs["files"]["file"][0] == "e.png" and kwargs["files"]["file"][2] == "image/png"
    assert res.score == 0.97 and [f.severity for f in res.findings] == ["alert", "alert"]
    assert "gpt-image-1" in res.findings[0].detail and "SynthID" in res.findings[1].title


def test_not_detected_is_not_a_clean_bill(api, png):
    api(Reply(body={"results": [{"type": "c2pa", "outcome": "not_detected", "validation_state": "not_present"},
                                {"type": "synthid", "outcome": "not_detected"}]}))
    res = online.openai_provenance(png)
    assert res.score == 0.05 and res.findings[0].severity == "info" and "does NOT show" in res.findings[0].detail


def test_tampered_credentials(api, png):
    api(Reply(body={"results": [{"type": "c2pa", "outcome": "detected", "validation_state": "invalid"}]}))
    assert online.openai_provenance(png).findings[0].severity == "warn"


@pytest.mark.parametrize("reply, expected", [
    (Reply(401, {"error": {"message": "bad key"}}), "rejected the API key"),
    (Reply(429, {}, {"Retry-After": "30"}), "rate limit"),
    (Reply(500, {"error": {"message": "boom"}}), "HTTP 500"),
    (Reply(502, None, text="<html>bad gateway</html>"), "HTTP 502"),
])
def test_errors_are_explained(api, png, reply, expected):
    api(reply)
    res = online.openai_provenance(png)
    assert res.score is None and expected in res.findings[0].title


def test_unsupported_file_is_never_uploaded(api, tmp_path):
    calls = api(Reply(body={"results": []}))
    path = tmp_path / "e.bmp"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
    res = online.openai_provenance(Evidence(path))
    assert not calls and "not accepted" in res.findings[0].title


def test_without_a_key_nothing_is_sent(monkeypatch, png):
    monkeypatch.setattr(online, "load_config", lambda: {})
    monkeypatch.setattr(online.requests, "post", lambda *a, **k: pytest.fail("must not upload"))
    assert "not configured" in online.openai_provenance(png).findings[0].title
