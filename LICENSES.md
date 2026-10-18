# Third-party components

FIASuite is assembled for personal and research use. Some bundled models do not allow commercial use.

| Component | Licence | Notes |
|---|---|---|
| ExifTool (Phil Harvey) | Perl Artistic / GPL | `vendor/exiftool`, unmodified |
| Qt for Python (PySide6) | LGPL-3.0 | dynamically linked, replaceable in `_internal` |
| OpenCV, NumPy, SciPy, Pillow, pillow-heif, ONNX Runtime, requests | Apache-2.0 / BSD / MIT / HPND / LGPL (libheif) | |
| c2pa-python (Content Authenticity Initiative) | MIT / Apache-2.0 | |
| Community Forensics detector (Park & Owens) | MIT | `models/commfor_vits16_384.onnx`, converted from huggingface.co/OwensLab/commfor-model-384 |
| SwinV2 AI-image detector (haywoodsloan, ONNX by LPX55) | Apache-2.0 | `models/swinv2_detector_fp16.onnx`; training data undocumented |
| OmniAID v2 detector (Guo et al., ICML 2026) | MIT (Hugging Face model card; the GitHub repository shows an MIT badge but has no licence file) | optional `models/omniaid_v2_clip336.onnx`, converted from huggingface.co/Yunncheng/OmniAID with weights stored as fp16; built on OpenAI CLIP ViT-L/14-336 (MIT) |
| **TruFor** (Guillaro, Cozzolino, Sud, Dufour, Verdoliva - GRIP-UNINA, CVPR 2023) | **Research licence, non-commercial only** | `models/trufor_1024.onnx`; delete this file and its entry in `models.json` for any commercial use |
| **B-Free** (Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva - GRIP-UNINA, CVPR 2025) | **GRIP-UNINA licence: informational and nonprofit use only** | optional `models/bfree_dinov2_504.onnx`, converted from the authors' weights (github.com/grip-unina/B-Free); delete the file and its `models.json` entry for any commercial use |
| Adobe TrustMark decoders Q and P | MIT | `models/trustmark_decoder_Q.onnx`, `models/trustmark_decoder_P.onnx`, Adobe's ONNX exports, unmodified |
| Adobe TrustMark BCH error correction | MIT | `fiasuite/analyses/_trustmark_bch.py`, copied unchanged from github.com/adobe/trustmark |
| C2PA soft-binding algorithm list (Coalition for Content Provenance and Authenticity) | CC BY 4.0 | identifiers and algorithm names from github.com/c2pa-org/softbinding-algorithm-list, snapshot of 18 October 2026 in `fiasuite/analyses/trustmark.py` |
| invisible-watermark (ShieldMnt) | MIT | the dwtDct decoding rule is re-implemented in `fiasuite/analyses/watermark.py`; no code or files of the library are bundled |
| Guide example pictures | CC0 / public domain | made from Wikimedia Commons photographs, credits in `fiasuite/guide_examples/SOURCES.md` |
| imtriage 0.2.0 | MIT | source of the spectral-residual refinements (Wiener residual, radial flattening, peak classes, angular profile), the laundering chains and the calibration method |

Methods implemented from the literature: Krawetz (ELA); Farid (JPEG ghosts); Popescu & Farid (resampling, CFA,
double quantization); Lukas & Fridrich; Li et al. (block-artifact grid); Gallagher, Mahdian & Saic, Kirchner
(resampling); Ferrara et al. (CFA localisation); Fridrich et al., Amerini et al. (copy-move); Immerkaer (noise
estimation); Bammey (Synthbuster); Wagner (Forensically tool set); Lin et al. (double-quantization map); Lukas,
Fridrich, Goljan, Chen, Filler (PRNU); Kee, O'Brien, Farid (shadow, reflection and perspective geometry); Stamm & Liu
(histogram fingerprints). Benchmarks used for the measured accuracy (not redistributed): Korus & Huang, Realistic
Tampering Dataset; ComplexDataLab OpenFake.
