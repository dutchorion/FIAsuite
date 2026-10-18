"""Optional components that are not part of the source tree or the slim build
(ExifTool, the neural models): what they are, whether they are installed, and
how to fetch them. Pure logic, no GUI; the setup wizard drives it.

Every download is written to a .part file, verified against a SHA-256 checksum
and only then moved into place.
"""
from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from . import core
from .analyses.metadata import exiftool_path

RELEASE_REPO = "dutchorion/FIAsuite"          # private: the converted models live in its release assets
RELEASE_TAG = "models-v1"
TOKEN_HELP = ("github.com > Settings > Developer settings > Fine-grained tokens > Generate new token. "
              f"Repository access: only {RELEASE_REPO}. Permissions: Contents = Read-only.")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

Progress = Callable[[int, int], None]          # bytes done, bytes total (0 = unknown)


class SetupError(Exception):
    """A problem the user can act on; the message is shown as is."""


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class Component:
    id: str
    title: str
    enables: str                    # what the user gains
    source: str                     # where it comes from, shown in the wizard
    licence: str
    size_mb: int
    dest: str                       # path relative to the application folder that proves it is installed
    kind: str                       # "exiftool" | "github" | "url"
    url: str = ""
    sha256: str = ""
    default: bool = True            # pre-ticked in the wizard


COMPONENTS = [
    Component("exiftool", "ExifTool", "Metadata analysis, embedded-thumbnail check, metadata diff in Compare",
              "exiftool.org (latest Windows 64-bit release, checksum verified)", "Perl Artistic / GPL", 11,
              "vendor/exiftool/exiftool.exe", "exiftool"),
    Component("commfor", "Community Forensics detector", "AI-generation detection (best open detector in the 2026 benchmark)",
              f"private release of {RELEASE_REPO} (ONNX conversion of OwensLab/commfor-model-384)", "MIT", 83,
              "models/commfor_vits16_384.onnx", "github",
              sha256="b03ff65cf535386c5c23a4a87e2f06350220ed65f92f4a9d3bbf86807258a536"),
    Component("swinv2", "SwinV2 detector", "Second opinion for AI-generation detection (half weight in the ensemble)",
              "huggingface.co/LPX55/detection-model-1-ONNX", "Apache-2.0", 432,
              "models/swinv2_detector_fp16.onnx", "url",
              url="https://huggingface.co/LPX55/detection-model-1-ONNX/resolve/"
                  "04b323d0b2f4300fd57b9059b1bb8b93782e2854/onnx/model_fp16.onnx",
              sha256="40809e0144cf3eb28bcc062710245266a689fb6d6c52f94c3876c04514394fef"),
    Component("trustmark_q", "TrustMark decoder Q", "Content Credentials watermark check (Adobe TrustMark, default variant)",
              "cai-watermark.adobe.net (Adobe's ONNX export, checksum verified)", "MIT", 46,
              "models/trustmark_decoder_Q.onnx", "url",
              url="https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.onnx",
              sha256="ee3268f057c9dabef680e169302f5973d0589feea86189ed229a896cc3aa88df"),
    Component("trustmark_p", "TrustMark decoder P", "Content Credentials watermark check (Adobe TrustMark, low-visibility variant)",
              "cai-watermark.adobe.net (Adobe's ONNX export, checksum verified)", "MIT", 46,
              "models/trustmark_decoder_P.onnx", "url",
              url="https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_P.onnx",
              sha256="be6d7c33f8a7b376f179e75f3f7c58ff816a9ac7bb6d37fd0a729a635f624c35"),
    Component("trufor", "TruFor forgery localiser", "Heat map of spliced, cloned, inpainted and locally AI-edited regions",
              f"private release of {RELEASE_REPO} (ONNX conversion of grip-unina/TruFor)",
              "GRIP-UNINA research licence: NON-COMMERCIAL use only", 263,
              "models/trufor_1024.onnx", "github",
              sha256="80329fb9c7cc104e1d0f28a0d2806231b0602d51aa9f01eea67800fa89236286"),
]
BY_ID = {c.id: c for c in COMPONENTS}


def target(c: Component) -> Path:
    return core.app_dir() / c.dest


def installed(c: Component) -> bool:
    # ExifTool also counts when it is found outside vendor/ (the system exiftool on Linux)
    return target(c).exists() or (c.kind == "exiftool" and exiftool_path().exists())


def missing() -> list[Component]:
    return [c for c in COMPONENTS if not installed(c)]


# ------------------------------------------------------------------ GitHub access

def github_cli_token() -> str | None:
    """Token of a signed-in GitHub CLI, if there is one. Never stored, never shown."""
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    token = out.stdout.strip()
    return token if out.returncode == 0 and token else None


def _release_asset_url(name: str, token: str | None) -> str:
    if not token:
        raise SetupError("This model is stored in a private GitHub repository. Sign in with GitHub CLI "
                         "('gh auth login') or paste an access token in the wizard.")
    reply = _get(f"https://api.github.com/repos/{RELEASE_REPO}/releases/tags/{RELEASE_TAG}",
                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    if reply.status_code in (401, 403, 404):
        raise SetupError(f"GitHub refused access to {RELEASE_REPO} (HTTP {reply.status_code}). The token needs read "
                         "access to that repository's contents.")
    reply.raise_for_status()
    for asset in reply.json().get("assets", []):
        if asset["name"] == name:
            return asset["url"]
    raise SetupError(f"Release {RELEASE_TAG} of {RELEASE_REPO} has no file named {name}.")


# ------------------------------------------------------------------ downloading

def _get(url: str, **kwargs) -> requests.Response:
    try:
        return requests.get(url, timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise SetupError(f"No connection to {url.split('/')[2]}: check the internet connection.\n({exc})") from exc


def download(url: str, dest: Path, sha256: str, progress: Progress, cancelled: Callable[[], bool],
             headers: dict | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    reply = _get(url, headers=headers, stream=True)
    if reply.status_code != 200:
        raise SetupError(f"Download failed: HTTP {reply.status_code} from {url.split('/')[2]}.")
    total, done = int(reply.headers.get("Content-Length", 0)), 0
    try:
        with open(part, "wb") as fh:
            for chunk in reply.iter_content(1 << 20):
                if cancelled():
                    raise Cancelled()
                fh.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, total)
        if sha256 and digest.hexdigest() != sha256.lower():
            raise SetupError(f"{dest.name}: checksum mismatch. The file was damaged in transit or changed at the "
                             "source; nothing was installed. Try again.")
        part.replace(dest)
    except requests.RequestException as exc:
        raise SetupError(f"The download of {dest.name} was interrupted: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)


def _install_exiftool(c: Component, progress: Progress, cancelled: Callable[[], bool]) -> None:
    version = _get("https://exiftool.org/ver.txt").text.strip()
    if not re.fullmatch(r"\d+\.\d+", version):
        raise SetupError("exiftool.org did not report a version number.")
    name = f"exiftool-{version}_64.zip"
    sums = _get("https://exiftool.org/checksums.txt").text
    match = re.search(rf"SHA2-256\({re.escape(name)}\)\s*=\s*([0-9a-f]{{64}})", sums, re.I)
    if not match:
        raise SetupError(f"exiftool.org publishes no checksum for {name}.")
    folder = target(c).parent
    archive = folder / name
    # exiftool.org hosts the Windows package on SourceForge; the checksum above comes from exiftool.org itself
    download(f"https://downloads.sourceforge.net/project/exiftool/{name}", archive, match.group(1), progress, cancelled)
    try:
        install_exiftool_zip(archive.read_bytes(), folder)
    finally:
        archive.unlink(missing_ok=True)


def install_exiftool_zip(data: bytes, folder: Path) -> None:
    """The official zip holds <top>/exiftool(-k).exe and <top>/exiftool_files/."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        exe = next((n for n in z.namelist() if re.search(r"exiftool(\(-k\))?\.exe$", n, re.I)), None)
        if exe is None:
            raise SetupError("The ExifTool archive does not contain exiftool.exe.")
        top = exe.rsplit("/", 1)[0] + "/" if "/" in exe else ""
        shutil.rmtree(folder / "exiftool_files", ignore_errors=True)
        (folder / "exiftool.exe").unlink(missing_ok=True)
        base = folder.resolve()
        for info in z.infolist():
            rel = info.filename[len(top):]
            out = (folder / rel).resolve()
            if info.is_dir() or info.filename == exe or not info.filename.startswith(top) or not out.is_relative_to(base):
                continue                               # never write outside the target folder (zip-slip)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
        # the exe marks the component as installed: write it last, so an interrupted extraction is retried
        (folder / "exiftool.exe").write_bytes(z.read(exe))


def install(c: Component, progress: Progress, cancelled: Callable[[], bool], token: str | None = None) -> None:
    if c.kind == "exiftool":
        _install_exiftool(c, progress, cancelled)
    elif c.kind == "github":
        token = token or github_cli_token()
        url = _release_asset_url(Path(c.dest).name, token)
        download(url, target(c), c.sha256, progress, cancelled,
                 headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"})
    else:
        download(c.url, target(c), c.sha256, progress, cancelled)
