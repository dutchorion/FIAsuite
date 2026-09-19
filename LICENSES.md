# Third-party components

FIASuite is assembled for personal and research use. Some bundled models do not allow commercial use.

| Component | Licence | Notes |
|---|---|---|
| ExifTool (Phil Harvey) | Perl Artistic / GPL | `vendor/exiftool`, unmodified |
| Qt for Python (PySide6) | LGPL-3.0 | dynamically linked, replaceable in `_internal` |
| OpenCV, NumPy, SciPy, Pillow, pillow-heif, ONNX Runtime, requests | Apache-2.0 / BSD / MIT / HPND / LGPL (libheif) | |
| c2pa-python (Content Authenticity Initiative) | MIT / Apache-2.0 | |
| Community Forensics detector (Park & Owens) | MIT | `models/commfor_vits16_384.onnx`, converted from huggingface.co/OwensLab/commfor-model-384 |
| SwinV2 AI-image detector (haywoodsloan, ONNX by LPX55) | Apache-2.0 | `models/swinv2_detector_fp16.onnx`; training data undocumented, weighted 0.5 in the ensemble |
| **TruFor** (Guillaro, Cozzolino, Sud, Dufour, Verdoliva - GRIP-UNINA, CVPR 2023) | **Research licence, non-commercial only** | `models/trufor_1024.onnx`; delete this file and its entry in `models.json` for any commercial use |
| imtriage 0.2.0 | MIT | source of the spectral-residual refinements (Wiener residual, radial flattening, peak classes, angular profile), the laundering chains and the calibration method |

Methods implemented from the literature: Krawetz (ELA); Farid (JPEG ghosts); Popescu & Farid (resampling, CFA,
double quantization); Lukas & Fridrich; Li et al. (block-artifact grid); Gallagher, Mahdian & Saic, Kirchner
(resampling); Ferrara et al. (CFA localisation); Fridrich et al., Amerini et al. (copy-move); Immerkaer (noise
estimation); Bammey (Synthbuster); Wagner (Forensically tool set).
