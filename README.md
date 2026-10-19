# FIASuite - Forensic Image Analysis Suite

A portable Windows dashboard for finding image manipulation and AI generation, whether it affects the whole
picture or only part of it. No installation, no command line: copy the `FIASuite` folder anywhere (USB stick
included) and double-click **FIASuite.bat**. Everything runs locally; nothing leaves the PC unless you press an
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
5. **How to read this ...** (top of the right-hand panel) opens a guide for the selected tool: what the output
   shows, how to interpret it, typical cases with example pictures (a clean original, a splice, a re-saved copy
   ...), what every option does, what causes false alarms, and links to the papers and projects the technique
   comes from.
6. Drag the borders of the two side panels to make them wider or narrower; the widths are remembered.

Dots in the tool list: green = nothing found, yellow = note, orange = suspicious, red = strong finding.

| Toolbar | |
|---|---|
| Compare | Align a second version of the picture onto this one: difference map, changed regions, metadata diff, noise spectra side by side |
| Batch triage | Screen a whole folder, sort by suspicion, double-click to open, export CSV |
| Camera fingerprint | Build a sensor-noise (PRNU) fingerprint from 20 or more photos of one camera; the *Camera fingerprint (PRNU)* tool then tests whether an image came from that camera and which regions lack its pattern |
| Calibrate | Measure a detector on **your own** authentic and manipulated images (AUC, detection rate at 1 % and 5 % false alarms), also after "laundering" such as `resize0.5+jpeg75` |
| Export report | HTML or PDF case report (hashes, parameters, findings, maps) and a JSON record for machines. Tick which analyses to include (check all / none). The PDF is laid out on A4: every analysis starts on a new page, a page that continues a section is headed "<section> (continued)", every image is fitted to the column and none is split across pages, each page carries the file name, hash and page number |
| Reverse image search | Copies the image to the clipboard and opens Google Lens / TinEye / Bing / Yandex: press Ctrl+V there |
| Watermark check | Opens Google's SynthID Detector (synthid.com) in the browser. Detects the invisible SynthID watermark of Google, OpenAI, NVIDIA and Kakao tools; sign in there with a Google, OpenAI or Apple account and upload the file yourself |
| Settings | API key for the optional online detector |

## The tools

**File & provenance** - ExifTool metadata with consistency rules (AI-generator tags, editing software, XMP
history, Photoshop document ancestors, date conflicts, missing MakerNotes), C2PA Content Credentials with
signature validation, computational photography (smartphones that merge frames, zoom digitally or blur
backgrounds by default, and the traces of it in the file), Adobe TrustMark watermark (the C2PA soft binding
behind durable Content Credentials),
JPEG structure (quantization tables, quality, trailing data), embedded-thumbnail comparison, hashes, string
extraction.

**Compression** - Error level analysis (automatic error scale); JPEG ghosts (Farid) including misaligned grids;
DCT histograms for double compression and JPEG history inside PNG files, with a per-block double-quantization
map (Lin et al.) that localises content added between two compressions; 8x8 block-grid analysis (cropping,
pasted JPEG content).

**Noise & sensor** - Noise residual (choice of denoiser); noise-level consistency; CFA / demosaicing traces
(Popescu & Farid, Ferrara); resampling detection (Popescu & Farid, Gallagher, Mahdian & Saic); camera
fingerprint (PRNU: Lukas, Fridrich, Goljan) for camera identification and a fingerprint-based tamper map.

**Copy-move** - Keypoint clone detection (scaled / rotated copies, optionally mirrored ones) and block clone
detection (low-contrast areas).

**AI & deep learning** - Ensemble of neural AI-image detectors with per-region maps (Community Forensics and
SwinV2, combined as log-odds; B-Free as an optional third opinion that is shown but not counted); TruFor forgery
localisation with reliability map; spectral fingerprint of the noise residual with classified peaks and the
full set of controls (tile size and limit, denoiser and strength, channel, overlap, peak thresholds); decoder
for the invisible watermark of Stable Diffusion's reference software (SD 1.x, 2.x, SDXL).

**Visual inspection** - Geometry (draw lines to test shadows, vanishing points and reflections, after Farid);
magnifier enhancement, histogram with contrast-adjustment traces, hidden pixels under transparency, luminance
gradient, principal components, level sweep, bit planes, colour channels (the Forensically tool set).

**Online (optional)** - Sightengine AI detection with per-generator scores; OpenAI provenance check (OpenAI's Content Credentials and its invisible SynthID watermark, which survives screenshots and recompression, for content made with OpenAI tools only). Both need your own API key under Settings and upload only on your click. Google offers no public API for its SynthID Detector, hence the *Watermark check* button.

## Reading the results honestly

- A finding is a reason to look, not a verdict. Every tool explains what else can cause the same picture.
- **No finding is not proof of authenticity.** Recompression and resizing (every social platform) erase most
  traces. Independent 2026 benchmarks put the best open AI detectors near 78 % accuracy overall and far lower
  on the newest commercial generators. Use *Calibrate* to learn what the detectors achieve on your material.
- Scores are screening aids (0 = nothing found, 1 = strong indication), not probabilities of forgery.
- The evidence file is opened read-only and is never modified. Pixels are analysed exactly as stored (no EXIF
  rotation, no colour management) so that maps line up with the JPEG grid and the sensor pattern.

## Measured accuracy

Numbers from this project's own runs (September 2026) on real photographs, not on synthetic test scenes. They
are small samples: read them as orders of magnitude, and use *Calibrate* on your own material.

**Test sets.** *Korus*: the Realistic Tampering Dataset (Korus & Huang), 220 never-compressed 1920 x 1080 photos
from four cameras and 220 hand-made, carefully blended forgeries covering 1-8 % of the picture, with masks; 100 +
100 were used. *JPEG chains*: 40 Korus originals put through controlled histories (saved once at Q90; Q70 then
Q90; an insert from a Q60 JPEG pasted off-grid and saved at Q92; never-compressed content inserted between Q70
and Q90). *OpenFake*: 245 images from 16 current generators (Flux.2, GPT-image-1.5, Midjourney 7, Z-Image,
Veo 3 frames, Nano Banana Pro ...) and 180 photographs (DOCCI phone photos, ImageNet web JPEGs).
Neither set is redistributed; `bench/` is git-ignored.

| Tool | Detected | False alarms | Set |
|---|---|---|---|
| Camera fingerprint (PRNU), identification | 100 / 100 photos matched their camera | 0 / 300 foreign-camera comparisons | Korus, fingerprints from 30 photos |
| Camera fingerprint (PRNU), tamper map | 3 / 12 forgeries | 0 / 12 | Korus |
| Double-quantization map (DCT tool) | 37 / 40 inserts | 1 / 40 | JPEG chains |
| JPEG ghosts | 27 / 40 inserts | 1 / 80 | JPEG chains |
| JPEG block grid (foreign grid) | 14 / 40 inserts | 0 / 80 | JPEG chains |
| TruFor forgery localisation | 7 / 20 forgeries (12 / 20 score above 0.4) | 1 / 20 (3 / 20 above 0.4) | Korus |
| Clone detection, keypoints | 35 / 100 forgeries | 6 / 100 (identical manufactured objects) | Korus |
| Clone detection, blocks | 15 / 100 | 6 / 100 | Korus |
| CFA traces | 7 / 100 | 2 / 100 | Korus |
| Resampling | 2 / 100 (few forgeries were resized) | 1 / 100 | Korus |
| Noise-level consistency | 0 / 100 | 0 / 100 | Korus |
| Spectral fingerprint | 13 / 245 generated images | 1 / 280 photographs | OpenFake + Korus |
| AI detectors, ensemble above 50 % | 166 / 245 generated images (68 %) | 0 / 180 photographs | OpenFake |
| - Community Forensics alone | 91 / 245 | 1 / 180 | OpenFake |
| - SwinV2 alone | 192 / 245 | 10 / 180 | OpenFake |
| - B-Free alone (optional model) | 135 / 245 | 10 / 90 | OpenFake (every second photograph) |
| Metadata / C2PA declaration of AI | 73 / 245 | 0 / 180 | OpenFake (files as published) |

By generator, share the ensemble rates above 50 %: Midjourney 7 28/30, Illustrious 29/30, GPT-image-1.5 23/30,
Wan 2.5 14/16, Recraft v3 10/14, Veo 3 frames 17/28, Z-Image Turbo 14/30, Nano Banana Pro 2/4, **Flux.2 2/30**.
B-Free is offered as an optional third detector but is *not* counted: on this material it reached AUC 0.81,
rated 10 of 90 photographs as generated, and adding it did not improve the ensemble (AUC 0.952 to 0.944). The SwinV2 detector, which earlier
spot checks had marked as unreliable, turned out to be the strongest single model here and has full weight.

**OmniAID (October 2026).** OmniAID v2 (ICML 2026) was measured on a fresh sample of the same OpenFake test
data: 275 images from 20 generators, 100 DOCCI and 100 ImageNet photographs. Alone it reached AUC 0.918 (161 of
275 generated images and 4 of 200 photographs above 50 %), against 0.932 for Community Forensics and 0.953 for
SwinV2. Adding it to the ensemble moved the AUC from 0.973 to 0.976 and found 196 instead of 201 generated
images, with 1 photograph above 50 % either way: no gain worth a 1 GB model at 1.7 s per image on the CPU, so it
was not added to the program. Flux.2 stays the blind spot: 3 of 25 for the ensemble.

What the measurements changed. Several detectors that passed every synthetic test failed on real content and
were reworked: JPEG ghosts found 0 of 40 real inserts (now 27) because the valley in the error curve is shallower
on real texture and the insert's grid alignment was not being proposed; on re-saved images harmonics of the
first quality produced 15 false ghosts in 40 (now 1). The block clone detector flagged 34 of 100 untouched
photographs (sky, water, foliage that merely look alike); a copy is now required to carry the *same pixel noise*
as its source (6 of 100). The keypoint detector reported every clone twice, as two half-sized groups. The
spectral tool flagged fine scene texture as generator peaks. The AI ensemble averaged probabilities, which let a
confident miss by one network cancel a confident hit by the other.

What stays weak. Small, well-blended edits in never-compressed or heavily recompressed images: on the Korus
forgeries no tool, TruFor included, finds more than about one in three. Taken together the signal tools flag
49 of 100 forgeries and 15 of 100 untouched photographs (before this round: 70 and 43), so a single orange
finding is a reason to look, never a result. Where TruFor does respond, its map is the most precise
(typically under 0.1 % of the picture wrongly marked). Current generators leave
no spectral peaks. AI detectors are strong on some generators and blind to others (see the guide of the AI
tool); a low score never establishes authenticity.

## Getting it onto another PC

| You have | Do this | First start |
|---|---|---|
| `FIASuite` folder (full build, 1.2 GB) | copy it, double-click **FIASuite.exe** | works offline straight away |
| `FIASuite-slim` folder (0.4 GB) | copy it, double-click **FIASuite.exe** | the setup wizard offers to download ExifTool and the models |
| a git clone / source zip | double-click **FIASuite.bat** | finds Python 3.12 (offers to install it with winget if it is missing), creates a private environment in the folder, installs the packages, starts the program; then the setup wizard as above |

**Setup wizard.** At start-up the program checks for ExifTool, the three standard models and the two TrustMark
decoders (the optional B-Free detector is not offered by the wizard; it is created with
`tools/convert_models.py bfree`). If something is missing
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
    fingerprints/       camera fingerprints you built (one .npz per camera)
    fiasuite-error.log  written only if the program hits an unexpected error

## Developing

    FIASuite.bat                                          # or by hand: python -m venv .venv (Python 3.12)
    .venv\Scripts\pip install -r requirements-dev.txt     # adds pytest and PyInstaller
    .venv\Scripts\python -m pytest tests                  # ground-truth tests; model tests skip without models
    .venv\Scripts\python -m fiasuite                      # run from source
    .venv\Scripts\python tools\build_portable.py         # -> dist\FIASuite (full) and dist\FIASuite-slim

**Linux.** The engine (not the desktop window) also runs headless on Linux: install `requirements-linux.txt`
(the same pins with `onnxruntime` instead of `onnxruntime-directml`) or build the `Dockerfile`, and point
`FIASUITE_HOME` at the folder holding `models/`, `config.json` and `fingerprints/`; ExifTool is taken from the system.

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
    .venv-convert\Scripts\python tools\convert_models.py commfor swinv2 trufor bfree

`tools/bench_ai_detectors.py` measures the installed detectors and every ensemble of them on a labelled folder
(the OpenFake figures above); `tools/make_trustmark_fixtures.py` re-creates the TrustMark test images.

After a new conversion, upload the files to a new release tag and update the tag and SHA-256 values in
`fiasuite/components.py`.
