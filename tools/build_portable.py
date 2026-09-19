"""Build the portable folders:

    dist/FIASuite-slim/   program only; the setup wizard fetches ExifTool and the models on first start
    dist/FIASuite/        the same plus vendor/ and models/: works offline straight away

    .venv/Scripts/python tools/build_portable.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "dist" / "FIASuite"
SLIM = ROOT / "dist" / "FIASuite-slim"


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2 ** 20


subprocess.check_call([
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", "FIASuite",
    "--icon", str(ROOT / "assets" / "fiasuite.ico"),
    "--collect-all", "c2pa", "--collect-binaries", "onnxruntime", "--collect-all", "pillow_heif",
    "--exclude-module", "sympy", "--exclude-module", "pywt", "--exclude-module", "pytest",
    "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
    "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build"),
    str(ROOT / "run_fiasuite.py")], cwd=ROOT)

# development-only files that some wheels ship (static libs, debug symbols) and OpenCV's unused video backend
for pattern in ("c2pa/libs/*.lib", "c2pa/libs/*.pdb", "cv2/opencv_videoio_ffmpeg*.dll"):
    for f in (FULL / "_internal").glob(pattern):
        f.unlink()
for name in ("README.md", "LICENSES.md"):
    shutil.copy2(ROOT / name, FULL / name)
(FULL / "models").mkdir(exist_ok=True)
shutil.copy2(ROOT / "models" / "models.json", FULL / "models" / "models.json")    # describes the models the wizard installs

shutil.rmtree(SLIM, ignore_errors=True)
shutil.copytree(FULL, SLIM)

# ExifTool and the models stay plain files beside the exe, so they can be updated or replaced by hand
for folder in ("vendor", "models"):
    if (ROOT / folder).exists():
        shutil.copytree(ROOT / folder, FULL / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("_src"))     # model sources / raw weights stay out of the release

print(f"\nSlim build: {SLIM}  ({folder_mb(SLIM):.0f} MB)\nFull build: {FULL}  ({folder_mb(FULL):.0f} MB)")
