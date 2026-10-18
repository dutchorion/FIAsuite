"""Setup-wizard logic, exercised against a local web server (no internet)."""
import hashlib
import io
import shutil
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fiasuite import components, core


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)      # no system ExifTool (Linux) in a fresh folder
    return tmp_path


@pytest.fixture()
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("www")
    payload = bytes(range(256)) * 4096                        # 1 MB
    (root / "model.onnx").write_bytes(payload)
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", hashlib.sha256(payload).hexdigest()
    httpd.shutdown()


def test_missing_and_installed(app):
    assert [c.id for c in components.missing()] == ["exiftool", "commfor", "swinv2", "trustmark_q", "trustmark_p", "trufor"]
    target = components.target(components.BY_ID["commfor"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert "commfor" not in [c.id for c in components.missing()]


def test_download_verifies_checksum(app, server):
    url, sha = server
    dest, seen = app / "models" / "m.onnx", []
    components.download(f"{url}/model.onnx", dest, sha, lambda done, total: seen.append((done, total)), lambda: False)
    assert dest.stat().st_size == 1 << 20 and seen[-1] == (1 << 20, 1 << 20)
    assert not list(dest.parent.glob("*.part"))


def test_corrupt_download_installs_nothing(app, server):
    url, _ = server
    dest = app / "models" / "m.onnx"
    with pytest.raises(components.SetupError, match="checksum"):
        components.download(f"{url}/model.onnx", dest, "0" * 64, lambda *a: None, lambda: False)
    assert not dest.exists() and not list(dest.parent.glob("*.part"))


def test_cancel_leaves_no_partial_file(app, server):
    url, sha = server
    dest = app / "models" / "m.onnx"
    with pytest.raises(components.Cancelled):
        components.download(f"{url}/model.onnx", dest, sha, lambda *a: None, lambda: True)
    assert not dest.exists() and not list(dest.parent.glob("*.part"))


def test_http_error_is_explained(app, server):
    url, _ = server
    with pytest.raises(components.SetupError, match="HTTP 404"):
        components.download(f"{url}/nope.onnx", app / "x.onnx", "", lambda *a: None, lambda: False)


def test_private_models_need_github_access(app, monkeypatch):
    monkeypatch.setattr(components, "github_cli_token", lambda: None)
    with pytest.raises(components.SetupError, match="private GitHub repository"):
        components.install(components.BY_ID["trufor"], lambda *a: None, lambda: False, token=None)


def test_exiftool_zip_layout(app):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("exiftool-13.59_64/exiftool(-k).exe", b"MZ")
        z.writestr("exiftool-13.59_64/exiftool_files/perl.exe", b"MZ")
        z.writestr("exiftool-13.59_64/exiftool_files/lib/Image/ExifTool.pm", b"1;")
        z.writestr("exiftool-13.59_64/../evil.txt", b"zip-slip attempt")
    folder = app / "vendor" / "exiftool"
    components.install_exiftool_zip(buf.getvalue(), folder)
    assert (folder / "exiftool.exe").read_bytes() == b"MZ"
    assert (folder / "exiftool_files" / "lib" / "Image" / "ExifTool.pm").exists()
    assert not (app / "vendor" / "evil.txt").exists() and components.installed(components.BY_ID["exiftool"])
