"""Computational photography. Smartphones do not record one exposure: by default they merge bursts of frames
(HDR, night modes), crop and enlarge (digital zoom, multi-frame super-resolution, on some models generative AI
zoom), render depth-based background blur and apply AI scene processing. Such pixels were computed, and they
can mislead the signal tools and the AI detectors. This check names the default processing of the device that
took the picture and every trace of computational capture recorded in the file.

Tag names and values were checked against ExifTool's sample files of 66 Apple, 26 Google, 702 Samsung, 29
Huawei, 19 Xiaomi and 7 OnePlus models (exiftool.org/sample_images.html). Samsung's capture records sit in a
trailer after the image (named blocks such as 'Photo_HDR_Info'); their names are read from the file itself.
"""
from __future__ import annotations

import re

from ..core import INFO, WARN, Evidence, Finding, Result
from . import metadata

SCORE = 0.6                                       # orange: a reason to review, not a finding of manipulation

# Samsung trailer block names (ExifTool Samsung.pm, Samsung::Trailer) -> what they record
SAMSUNG_BLOCKS = {
    b"Photo_HDR_Info": "HDR processing record (Samsung)",
    b"Camera_Scene_Info": "Scene Optimizer: AI scene recognition and enhancement (Samsung)",
    b"Livefocus_JDM_Info": "Live focus / portrait: background blur computed from depth (Samsung)",
    b"DualShot_Core_Info": "Live focus / portrait: dual-camera depth capture (Samsung)",
    b"DualShot_Extra_Info": "Live focus / portrait: dual-camera depth capture (Samsung)",
    b"Single_Relighting_Bokeh_Info": "Portrait relighting and background blur (Samsung)",
    b"Dual_Relighting_Bokeh_Info": "Portrait relighting and background blur (Samsung)",
    b"Portrait_Effect_Info": "Portrait effect applied (Samsung)",
    b"SingeShot_DepthMap_1": "Depth map for single-camera portrait blur (Samsung)",
    b"Food_Blur_Effect_Info": "Food mode: background blur (Samsung)",
    b"ZoomInOut_Info": "Zoom processing record (Samsung)",
    b"Remaster_Info": "AI Remaster applied in the Gallery (Samsung)",
    b"Intelligent_PhotoEditor_Data": "Edited in Samsung's photo editor",
    b"PhotoEditor_Re_Edit_Data": "Edited in Samsung's photo editor (re-editable)",
    b"UltraWide_PhotoEditor_Data": "Ultra-wide distortion corrected in Samsung's photo editor",
}
_SAMSUNG = re.compile(b"|".join(re.escape(name) for name in SAMSUNG_BLOCKS))

# makers whose current phones merge frames and run AI scene processing by default; value: (brand, phone model?)
OTHER_MAKERS = {
    "huawei": "Huawei", "honor": "Honor", "xiaomi": "Xiaomi", "redmi": "Xiaomi (Redmi)", "poco": "Xiaomi (POCO)",
    "oneplus": "OnePlus", "oppo": "OPPO", "vivo": "vivo", "realme": "realme", "motorola": "Motorola",
    "nothing": "Nothing", "hmd global": "HMD", "zte": "ZTE", "nubia": "nubia", "tecno": "TECNO", "infinix": "Infinix",
    "fairphone": "Fairphone", "sony": "Sony", "asus": "ASUS", "lge": "LG", "lg electronics": "LG", "nokia": "Nokia",
}
# these makers also build cameras or other devices: only their phone model names count
PHONE_MODELS = {"sony": ("XQ-", "SO-", "SOG", "Xperia"), "asus": ("ASUS_", "ROG"), "lge": ("LM-", "LG-"),
                "lg electronics": ("LM-", "LG-"), "nokia": ("Nokia", "TA-")}


def _get(meta: dict, *keys):
    for key in keys:
        value = meta.get(key)
        if value not in (None, "") and str(value).strip():
            return value
    return None


def _number(value) -> float | None:
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _year(meta: dict) -> int | None:
    when = metadata._parse_date(_get(meta, "ExifIFD:DateTimeOriginal", "ExifIFD:CreateDate", "IFD0:ModifyDate"))
    return when.year if when else None


def _iphone_generation(model: str) -> float:
    """8 = iPhone 8, 10 = iPhone X, 10.5 = XS / XR / SE 2nd and 3rd generation, 11 = iPhone 11 ..."""
    if m := re.match(r"iPhone (\d+)", model):
        return float(m.group(1))
    if re.match(r"iPhone X[SR]", model):
        return 10.5
    if model == "iPhone X":
        return 10
    if m := re.match(r"iPhone SE \((\d)", model):
        return 10.5 if int(m.group(1)) >= 2 else 6.5
    if model in ("iPhone", "iPhone SE"):
        return 1 if model == "iPhone" else 6.5
    return 99                                      # a newer name without a number (iPhone Air)


def device_defaults(meta: dict) -> tuple[str, list[str]] | None:
    """(device, processing it applies by default) for phones known to merge frames by default, else None."""
    make = str(_get(meta, "IFD0:Make") or "").strip()
    model = str(_get(meta, "IFD0:Model") or "").strip()
    low, year = make.lower(), _year(meta)
    device = f"{make} {model}".strip()
    if low.startswith("apple") and model.startswith("iPhone"):
        gen = _iphone_generation(model)
        if gen < 8:
            return None
        lines = ["Auto HDR merges several exposures when the scene needs it (on by default since the iPhone 8)"]
        if gen >= 10.5:
            lines = ["Smart HDR merges a burst of frames into every photo (on by default since the iPhone XS and XR)"]
        if gen >= 11:
            lines.append("Deep Fusion and Night mode stack frames automatically in medium and low light")
        if gen >= 13:
            lines.append("Smart HDR can no longer be switched off")
        if gen >= 14:
            lines.append("Photonic Engine: frames are merged early in the image pipeline")
        return device, lines
    if low.startswith("google") and model.startswith("Pixel"):
        m = re.match(r"Pixel (\d+)", model)
        number = int(m.group(1)) if m else (7 if "Fold" in model else 1)
        lines = ["HDR+: every photo is merged from a burst of frames (default on all Pixel phones)"]
        if number >= 3:
            lines.append("Night Sight stacks frames in low light; Super Res Zoom merges frames when zooming")
        if number >= 10:
            lines.append("every photo carries Content Credentials declaring a multi-frame 'computational capture'")
            if "Pro" in model:
                lines.append("Pro Res Zoom: beyond 30x zoom, detail is regenerated by a generative AI model")
        return device, lines
    if low.startswith("samsung") and (model.upper().startswith("SM-") or model.startswith("Galaxy")):
        if year and year < 2018:
            return None
        return device, ["Scene Optimizer (AI scene recognition and detail enhancement) and multi-frame processing "
                        "are on by default",
                        "since the Galaxy S21 the moon is recognised and its detail enhanced by AI; Space Zoom on "
                        "Ultra models merges 10 or more frames from 25x and enhances them with AI"]
    for key, brand in OTHER_MAKERS.items():
        if low.startswith(key) and (key not in PHONE_MODELS or model.startswith(PHONE_MODELS[key])):
            if year and year < 2018:
                return None
            return device, [f"current {brand} phones merge several frames (automatic HDR, night stacking) and apply "
                            "AI scene processing by default"]
    return None


def recorded_processing(meta: dict, data: bytes = b"") -> list[tuple[str, str]]:
    """(source tag, what it shows) for every trace of computational capture in the metadata and Samsung trailer."""
    found = []
    composite = _get(meta, "ExifIFD:CompositeImage")
    if composite in ("General Composite Image", "Composite Image Captured While Shooting"):
        count = _get(meta, "ExifIFD:CompositeImageCount")
        frames = f"; {int(_number(count))} source frames" if _number(count) else ""
        found.append(("EXIF CompositeImage", f"{composite}: the picture is a composite of several frames{frames}"))
    rendered = _get(meta, "ExifIFD:CustomRendered", "IFD0:CustomRendered")
    meaning = {"HDR (no original saved)": "HDR merge of several exposures", "HDR (original saved)": "HDR merge of several "
               "exposures (an unmerged original was saved too)", "Portrait HDR": "portrait mode with HDR: background "
               "blurred by software", "Portrait": "portrait mode: background blurred by software",
               "Panorama": "panorama stitched from many frames"}.get(str(rendered))
    if meaning:
        found.append(("EXIF CustomRendered", meaning))
    if _get(meta, "Apple:HDRImageType") == "HDR Image":
        found.append(("Apple HDRImageType", "HDR image merged from several exposures"))
    capture = str(_get(meta, "Apple:ImageCaptureType") or "")
    if capture == "Portrait":
        found.append(("Apple ImageCaptureType", "portrait mode: background blurred by software"))
    elif capture == "ProRAW":
        found.append(("Apple ImageCaptureType", "Apple ProRAW: a raw file computed from several frames"))
    if (_number(_get(meta, "Apple:HDRHeadroom")) or 0) > 0 or _get(meta, "Apple:HDRGain", "AROT:HDRGainCurve"):
        found.append(("Apple HDRHeadroom / HDRGain", "Smart HDR gain map recorded"))
    if _get(meta, "Apple:SemanticStyle"):
        found.append(("Apple SemanticStyle", "Photographic Style: tone and warmth rendered by the camera"))
    hdrp = _get(meta, "XMP-GCamera:HDRPMakerNote", "XMP-GCamera:HDRPlusMakerNote", "Google:HDRPSoftware")
    if hdrp:
        found.append(("Google HDR+ maker note", "HDR+ burst merge" + (f" ({hdrp})" if str(hdrp).startswith("HDR+") else "")))
    special = " ".join(map(str, _get(meta, "XMP-GCamera:SpecialTypeID") or []) if isinstance(
        _get(meta, "XMP-GCamera:SpecialTypeID"), list) else [str(_get(meta, "XMP-GCamera:SpecialTypeID") or "")])
    if "PORTRAIT" in special.upper() or _get(meta, "XMP-GCamera:PortraitVersion", "XMP-GCamera:PortraitRequest"):
        found.append(("Google camera XMP", "portrait mode: background blurred by software"))
    if "NIGHT" in special.upper():
        found.append(("Google SpecialTypeID", "Night Sight: frames stacked in low light"))
    if any(k.startswith(("XMP-GDepth:", "XMP-GFocus:")) or (k.startswith("XMP-Device:") and "DepthMap" in k)
           or k in ("Samsung:DepthMapData", "Samsung:SingleShotDepthMap", "XMP-WT_Refocus:BokehLevel") for k in meta):
        found.append(("depth map", "depth map embedded: background blur (bokeh) computed by software"))
    if _get(meta, "XMP-hdrgm:Version"):
        found.append(("XMP hdrgm", "Ultra HDR gain map recorded"))
    if any("computationalCapture" in str(meta.get(k, "")) for k in
           ("CBOR:ActionsDigitalSourceType", "XMP-iptcExt:DigitalSourceType")):
        found.append(("IPTC digital source type", "declared 'multi-frame computational capture'"))
    if str(_get(meta, "XMP-OPMedia:IsHDRActive")) == "True":
        found.append(("OnePlus IsHDRActive", "HDR merge active"))
    if str(_get(meta, "XMP-OPMedia:IsNightModeActive")) == "True":
        found.append(("OnePlus IsNightModeActive", "night mode: frames stacked"))
    if str(_get(meta, "XMP-OPMedia:IsBokehActive")) == "True" or _get(meta, "XMP-OPMedia:CaptureMode") == "Bokeh":
        found.append(("OnePlus bokeh", "portrait mode: background blurred by software"))
    if str(_get(meta, "JSON:Hdr")).lower() in ("on", "auto"):
        found.append(("Xiaomi Hdr", f"HDR set to '{_get(meta, 'JSON:Hdr')}': frames merged when the scene needs it"))
    if str(_get(meta, "JSON:AIScene") or "0") not in ("0", "None"):
        found.append(("Xiaomi AIScene", "AI scene detection applied"))
    zoom = _number(_get(meta, "ExifIFD:DigitalZoomRatio", "IFD0:DigitalZoomRatio")) or 0
    if zoom > 1.05:
        model, make = str(_get(meta, "IFD0:Model") or ""), str(_get(meta, "IFD0:Make") or "").lower()
        text = f"digital zoom x{zoom:g}: the picture was cropped and enlarged in the camera"
        if make.startswith("google") and "Pro" in model and zoom >= 30 and (_number(model.split()[1]) or 0) >= 10:
            text += "; in Pro Res Zoom range, where detail is regenerated by generative AI"
        elif make.startswith("samsung") and zoom >= 25:
            text += "; in Space Zoom range, where frames are merged and enhanced by AI"
        found.append(("EXIF DigitalZoomRatio", text))
    for name in dict.fromkeys(_SAMSUNG.findall(data)):
        found.append((f"Samsung trailer '{name.decode()}'", SAMSUNG_BLOCKS[name]))
    return found


def detect(ev: Evidence) -> Result:
    res = Result()
    meta = metadata.read_metadata(ev)
    device = device_defaults(meta)
    recorded = recorded_processing(meta, ev.data)
    if device:
        res.table.append(("Device", device[0]))
        res.table += [("Default processing", line) for line in device[1]]
    res.table += [(source, meaning) for source, meaning in recorded]
    if not device and not recorded:
        res.score = 0.05
        if not meta:
            res.findings.append(Finding(INFO, "No camera metadata",
                "The device cannot be identified: the metadata was stripped or ExifTool is missing. Computational "
                "capture can neither be confirmed nor excluded."))
        else:
            res.findings.append(Finding(INFO, "No sign of computational photography",
                "The camera is not a smartphone known to merge frames by default, and the file records no multi-frame, "
                "depth or zoom processing."))
        return res
    res.score = SCORE
    caveat = ("\n\nNot a finding of manipulation: the camera computed these pixels from several frames or with "
              "software effects. Merged, sharpened, depth-blurred or enlarged pixels can make the noise, CFA, "
              "resampling and ELA tools and the AI detectors respond; weigh their results with that in mind, and "
              "prefer the camera original for important cases.")
    if device:
        res.findings.append(Finding(WARN, "Smartphone with computational photography by default",
            f"{device[0]}: " + "; ".join(device[1]) + "." + ("" if recorded else caveat)))
    if recorded:
        res.findings.append(Finding(WARN, "Computational capture recorded in the file",
            "\n".join(f"{meaning} ({source})" for source, meaning in recorded) + caveat))
    return res
