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
| Export report | HTML or PDF case report (hashes, parameters, findings, maps) and a JSON record for machines. The PDF is laid out on A4: every image is fitted to the column, none is split across pages, each page carries the file name, hash and page number |
| Reverse image search | Copies the image to the clipboard and opens Google Lens / TinEye / Bing / Yandex: press Ctrl+V there |
| Watermark check | Same clipboard hand-over for provenance checks that only exist as web pages: Google Gemini and the SynthID Detector portal (Google's SynthID watermark), OpenAI's verify page |
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

**Online (optional)** - Sightengine AI detection with per-generator scores; OpenAI provenance check (OpenAI's Content Credentials and its invisible SynthID watermark, which survives screenshots and recompression, for content made with OpenAI tools only). Both need your own API key under Settings and upload only on your click. Google offers no public API for its SynthID image watermark, hence the *Watermark check* menu.

## Reading the results honestly

- A finding is a reason to look, not a verdict. Every tool explains what else can cause the same picture.
- **No finding is not proof of authenticity.** Recompression and resizing (every social platform) erase most
  traces. Independent 2026 benchmarks put the best open AI detectors near 78 % accuracy overall and far lower
  on the newest commercial generators. Use *Calibrate* to learn what the detectors achieve on your material.
- Scores are screening aids (0 = nothing found, 1 = strong indication), not probabilities of forgery.
- The evidence file is opened read-only and is never modified. Pixels are analysed exactly as stored (no EXIF
  rotation, no colour management) so that maps line up with the JPEG grid and the sensor pattern.

## Getting it onto another PC

| You have | Do this | First start |
|---|---|---|
| `FIASuite` folder (full build, 1.2 GB) | copy it, double-click **FIASuite.exe** | works offline straight away |
| `FIASuite-slim` folder (0.4 GB) | copy it, double-click **FIASuite.exe** | the setup wizard offers to download ExifTool and the models |
| a git clone / source zip | double-click **FIASuite.bat** | finds Python 3.12 (offers to install it with winget if it is missing), creates a private environment in the folder, installs the packages, starts the program; then the setup wizard as above |

**Setup wizard.** At start-up the program checks for ExifTool and the three models. If something is missing
it lists each item with its size, source and licence, and installs what you tick: files are downloaded,
verified against SHA-256 checksums and placed next to the program. Nothing else on the PC is changed. It can
be reopened any time with **Setup ...** and silenced with "Do not show this at start-up again".

Two of the models are ONNX conversions kept in the release assets of the private repository
`dutchorion/FIAsuite`. The wizard reaches them through a signed-in GitHub CLI (`gh auth login`), or with a
read-only access token that you paste once (used for that download only, never stored). ExifTool comes from
exiftool.org and the SwinV2 detector from Hugging Face; neither needs an account.

## Folder layout

    FIASuite.exe        the program            _internal/     its runtime (do not edit)
    vendor/exiftool/    ExifTool (replace with a newer release at will)
    models/             ONNX detectors + models.json (add or remove detectors here)
    config.json         created when you save settings (holds the API key in plain text)

## Developing

    FIASuite.bat                                          # or by hand: python -m venv .venv (Python 3.12)
    .venv\Scripts\pip install -r requirements-dev.txt     # adds pytest and PyInstaller
    .venv\Scripts\python -m pytest tests                  # ground-truth tests; model tests skip without models
    .venv\Scripts\python -m fiasuite                      # run from source
    .venv\Scripts\python tools\build_portable.py         # -> dist\FIASuite (full) and dist\FIASuite-slim

**GPU.** Neural models run on the CPU unless *Settings > Run neural models on the GPU* is ticked. The GPU path
uses DirectML, which works on NVIDIA, AMD and Intel; on an AMD Radeon 780M it gives the same scores as the CPU
(differences below 1e-3) about twice as fast. It is opt-in because the same GPU, with a driver in a bad state
after weeks of uptime, froze the desktop and returned wrong TruFor maps without raising an error. Therefore each
model is run once on both devices when it is first used on the GPU; on a mismatch, non-finite output or an
abnormally slow GPU it stays on the CPU for that session and its tool reports why.

**Re-creating the models.** `tools/convert_models.py` downloads the original weights and converts them to
ONNX; it needs a separate environment with PyTorch (the program itself does not):

    python -m venv .venv-convert
    .venv-convert\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    .venv-convert\Scripts\pip install timm onnx onnxruntime huggingface_hub safetensors pyyaml yacs numpy
    git clone https://github.com/grip-unina/TruFor models/_src/TruFor
    .venv-convert\Scripts\python tools\convert_models.py commfor swinv2 trufor

After a new conversion, upload the files to a new release tag and update the tag and SHA-256 values in
`fiasuite/components.py`.
