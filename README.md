# FIASuite - Forensic Image Analysis Suite

A portable Windows dashboard for finding image manipulation and AI generation, whether it affects the whole
picture or only part of it. No installation, no command line: copy the `FIASuite` folder anywhere (USB stick
included) and double-click **FIASuite.exe**. Everything runs locally; nothing leaves the PC unless you press an
upload button.

## Using it

1. **Open image** (or drop a file on the window). The full scan starts by itself.
2. The **Dashboard** answers three separate questions, each with the findings behind it:
   - **AI generation** - was the picture, or part of it, made by a generator?
   - **Local manipulation** - was something spliced in, cloned, removed or retouched?
   - **File history** - is this file something other than a camera original?
3. Click a card, or a tool on the left, to inspect its map. *View* switches between the maps a tool produces,
   *Overlay* blends the map over the photo, *Hold: original* flips back while pressed, *Side by side* shows the
   untouched photo next to it with zoom and pan linked. Wheel = zoom, drag = pan, double-click = fit / 1:1.
4. Change a parameter on the right and the tool re-runs.

Dots in the tool list: green = nothing found, yellow = note, orange = suspicious, red = strong finding.

| Toolbar | |
|---|---|
| Compare | Align a second version of the picture onto this one: difference map, changed regions, metadata diff, noise spectra side by side |
| Batch triage | Screen a whole folder, sort by suspicion, double-click to open, export CSV |
| Calibrate | Measure a detector on **your own** authentic and manipulated images (AUC, detection rate at 1 % and 5 % false alarms), also after "laundering" such as `resize0.5+jpeg75` |
| Export report | HTML or PDF case report (hashes, parameters, findings, maps) and a JSON record for machines |
| Reverse image search | Copies the image to the clipboard and opens Google Lens / TinEye / Bing / Yandex: press Ctrl+V there |
| Settings | API key for the optional online detector |

## The tools

**File & provenance** - ExifTool metadata with consistency rules (AI-generator tags, editing software, XMP
history, Photoshop document ancestors, date conflicts, missing MakerNotes), C2PA Content Credentials with
signature validation, JPEG structure (quantization tables, quality, trailing data), embedded-thumbnail
comparison, hashes, string extraction.

**Compression** - Error level analysis; JPEG ghosts (Farid) including misaligned grids; DCT histograms for
double compression and JPEG history inside PNG files; 8x8 block-grid analysis (cropping, pasted JPEG content).

**Noise & sensor** - Noise residual; noise-level consistency; CFA / demosaicing traces (Popescu & Farid,
Ferrara); resampling detection (Popescu & Farid, Gallagher, Mahdian & Saic).

**Copy-move** - Keypoint clone detection (scaled / rotated copies) and block clone detection (smooth areas).

**AI & deep learning** - Ensemble of neural AI-image detectors with per-region maps; TruFor forgery
localisation with reliability map; spectral fingerprint of the noise residual with classified peaks.

**Visual inspection** - Magnifier enhancement, luminance gradient, principal components, level sweep, bit
planes, colour channels (the Forensically tool set).

**Online (optional)** - Sightengine AI detection with per-generator scores. Uploads only on your click.

## Reading the results honestly

- A finding is a reason to look, not a verdict. Every tool explains what else can cause the same picture.
- **No finding is not proof of authenticity.** Recompression and resizing (every social platform) erase most
  traces. Independent 2026 benchmarks put the best open AI detectors near 78 % accuracy overall and far lower
  on the newest commercial generators. Use *Calibrate* to learn what the detectors achieve on your material.
- Scores are screening aids (0 = nothing found, 1 = strong indication), not probabilities of forgery.
- The evidence file is opened read-only and is never modified. Pixels are analysed exactly as stored (no EXIF
  rotation, no colour management) so that maps line up with the JPEG grid and the sensor pattern.

## Folder layout

    FIASuite.exe        the program            _internal/     its runtime (do not edit)
    vendor/exiftool/    ExifTool (replace with a newer release at will)
    models/             ONNX detectors + models.json (add or remove detectors here)
    config.json         created when you save settings (holds the API key in plain text)

## Building from source

    python -m venv .venv                            # Python 3.12, Windows
    .venv\Scripts\pip install -r requirements.txt
    .venv\Scripts\python -m pytest tests            # ground-truth tests on synthetic forgeries
    .venv\Scripts\python -m fiasuite                # run from source
    .venv\Scripts\python tools\build_portable.py    # -> dist\FIASuite

The tests that need the neural models skip themselves while `models/` is empty.

**GPU.** Neural models run on the CPU unless *Settings > Run neural models on the GPU* is ticked. The GPU path
uses DirectML, which works on NVIDIA, AMD and Intel. It is opt-in because on an AMD integrated GPU it froze
the desktop and returned wrong TruFor maps; verify GPU results against the CPU before relying on them.

Two things are not kept in the repository and must be restored after cloning:

- **ExifTool**: download the Windows package from https://exiftool.org, rename `exiftool(-k).exe` to
  `exiftool.exe` and place it together with its `exiftool_files` folder in `vendor/exiftool/`.
- **Models**: `tools/convert_models.py` downloads the detectors and converts them to ONNX into `models/`.
  Run it in a separate environment that has PyTorch:

      python -m venv .venv-convert
      .venv-convert\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      .venv-convert\Scripts\pip install timm onnx onnxruntime huggingface_hub safetensors pyyaml yacs numpy
      git clone https://github.com/grip-unina/TruFor models/_src/TruFor
      .venv-convert\Scripts\python tools\convert_models.py commfor swinv2 trufor

PyTorch is not needed to run the program, and the program works without models (the AI tools then report
"not installed").
