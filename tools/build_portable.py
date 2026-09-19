"""Build the portable folder:  dist/FIASuite/FIASuite.exe  (+ vendor/, models/, README)

    .venv/Scripts/python tools/build_portable.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "FIASuite"

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
    for f in (DIST / "_internal").glob(pattern):
        f.unlink()

# ExifTool and the models stay plain files beside the exe, so they can be updated or replaced by hand
for folder in ("vendor", "models"):
    if (ROOT / folder).exists():
        shutil.copytree(ROOT / folder, DIST / folder, dirs_exist_ok=True)
for name in ("README.md", "LICENSES.md"):
    if (ROOT / name).exists():
        shutil.copy2(ROOT / name, DIST / name)
size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 2 ** 20
print(f"\nPortable build ready: {DIST}  ({size:.0f} MB)")
