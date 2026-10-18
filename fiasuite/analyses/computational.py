"""Computational photography. Smartphones do not record one exposure: by default they merge bursts of frames
(HDR, night modes), crop and enlarge (digital zoom, multi-frame super-resolution, on some models generative AI
zoom), render depth-based background blur and apply AI scene processing. Such pixels were computed, and they
can mislead the signal tools and the AI detectors.

The check reports, section by section and tag by tag, what the file says about the device, the camera module,
the exposure, the camera mode used and the processing applied, plus the default processing of the phone model.
Rows marked as evidence of computational capture make the result orange.

Tag names and values were checked against ExifTool's sample files of 66 Apple, 26 Google, 702 Samsung, 29
Huawei, 19 Xiaomi and 7 OnePlus models (exiftool.org/sample_images.html). Samsung's capture records sit in a
trailer after the image (named blocks such as 'Photo_HDR_Info'); their names are read from the file itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core import INFO, WARN, Evidence, Finding, Result
from . import metadata

SCORE = 0.6                                       # orange: a reason to review, not a finding of manipulation

SECTIONS = ["Device", "Camera module", "Exposure", "Camera mode", "Multi-frame and HDR", "Depth and portrait", "Zoom",
            "Scene and style processing", "Embedded media", "Default processing of this model"]


@dataclass
class Row:
    section: str
    tag: str                                      # ExifTool group:tag, or a description where no single tag applies
    value: str
    meaning: str = ""
    evidence: bool = False                        # proves computational capture or processing


# Samsung trailer block names (ExifTool Samsung.pm, Samsung::Trailer) -> (section, what they record, evidence?)
SAMSUNG_BLOCKS = {
    b"Photo_HDR_Info": ("Multi-frame and HDR", "HDR processing record", True),
    b"Camera_Scene_Info": ("Scene and style processing", "Scene Optimizer: AI scene recognition and enhancement", True),
    b"Livefocus_JDM_Info": ("Depth and portrait", "Live focus / portrait: background blur computed from depth", True),
    b"DualShot_Core_Info": ("Depth and portrait", "Live focus: dual-camera depth capture", True),
    b"DualShot_Extra_Info": ("Depth and portrait", "Live focus: dual-camera depth capture", True),
    b"Single_Relighting_Bokeh_Info": ("Depth and portrait", "portrait relighting and background blur", True),
    b"Dual_Relighting_Bokeh_Info": ("Depth and portrait", "portrait relighting and background blur", True),
    b"Portrait_Effect_Info": ("Depth and portrait", "portrait effect applied", True),
    b"SingeShot_DepthMap_1": ("Depth and portrait", "depth map for single-camera portrait blur", True),
    b"Food_Blur_Effect_Info": ("Depth and portrait", "food mode: background blur", True),
    b"ZoomInOut_Info": ("Zoom", "zoom processing record", True),
    b"Remaster_Info": ("Scene and style processing", "AI Remaster applied in the Gallery", True),
    b"Intelligent_PhotoEditor_Data": ("Scene and style processing", "edited in Samsung's photo editor", True),
    b"PhotoEditor_Re_Edit_Data": ("Scene and style processing", "edited in Samsung's photo editor (re-editable)", True),
    b"UltraWide_PhotoEditor_Data": ("Scene and style processing", "ultra-wide distortion corrected in the editor", True),
    b"Camera_Capture_Mode_Info": ("Camera mode", "camera mode record (Samsung)", False),
    b"MotionPhoto_Data": ("Embedded media", "Motion photo: a short video clip is embedded", False),
    b"Interactive_Panorama": ("Camera mode", "panorama mode", True),
    b"Document_Scan_Info": ("Camera mode", "document scan mode", True),
}
_SAMSUNG = re.compile(b"|".join(re.escape(name) for name in SAMSUNG_BLOCKS))

# makers whose current phones merge frames and run AI scene processing by default
OTHER_MAKERS = {
    "huawei": "Huawei", "honor": "Honor", "xiaomi": "Xiaomi", "redmi": "Xiaomi (Redmi)", "poco": "Xiaomi (POCO)",
    "oneplus": "OnePlus", "oppo": "OPPO", "vivo": "vivo", "realme": "realme", "motorola": "Motorola",
    "nothing": "Nothing", "hmd global": "HMD", "zte": "ZTE", "nubia": "nubia", "tecno": "TECNO", "infinix": "Infinix",
    "fairphone": "Fairphone", "sony": "Sony", "asus": "ASUS", "lge": "LG", "lg electronics": "LG", "nokia": "Nokia",
}
# these makers also build cameras or other devices: only their phone model names count
PHONE_MODELS = {"sony": ("XQ-", "SO-", "SOG", "Xperia"), "asus": ("ASUS_", "ROG"), "lge": ("LM-", "LG-"),
                "lg electronics": ("LM-", "LG-"), "nokia": ("Nokia", "TA-")}

APPLE_STYLE_PRESETS = {1: "Standard", 2: "Vibrant", 3: "Rich Contrast", 4: "Warm", 5: "Cool"}    # ExifTool Apple.pm
APPLE_CAPTURE = {"Photo": ("Photo mode", False), "Portrait": ("Portrait mode: background blurred by software", True),
                 "ProRAW": ("Apple ProRAW: a raw file computed from several frames", True),
                 "Manual Focus": ("focus set by tapping or locking", False),
                 "Scene": ("scene-analysed capture (meaning not documented by Apple)", False)}


def _get(meta: dict, *keys):
    for key in keys:
        value = meta.get(key)
        if value not in (None, "") and str(value).strip():
            return value
    return None


def _number(value) -> float | None:
    try:
        text = str(value).split()[0]
        if "/" in text:                            # exposure times such as 1/91
            a, b = text.split("/")
            return float(a) / float(b)
        return float(text)
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
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


def _module(meta: dict, phone: bool) -> list[Row]:
    rows = []
    lens = _get(meta, "ExifIFD:LensModel")
    if lens:
        text = str(lens).lower()
        which = ("front (selfie) camera" if "front" in text else "back camera" if "back" in text else "")
        rows.append(Row("Camera module", "ExifIFD:LensModel", str(lens), which))
    if (camera := _get(meta, "Apple:CameraType")) is not None:
        rows.append(Row("Camera module", "Apple:CameraType", str(camera), {"Back Wide Angle": "ultra-wide camera",
                    "Back Normal": "main (wide) or telephoto camera", "Front": "front (selfie) camera"}.get(str(camera), "")))
    if (sensor := _get(meta, "JSON:SensorType")) is not None:
        rows.append(Row("Camera module", "JSON:SensorType", str(sensor), f"{sensor} camera (Xiaomi)"))
    eq = _number(_get(meta, "ExifIFD:FocalLengthIn35mmFormat"))
    focal = _get(meta, "ExifIFD:FocalLength")
    if eq:
        kind = ("ultra-wide" if eq < 20 else "main (wide)" if eq <= 35 else "2x to 3x: telephoto or a crop of the main "
                "sensor" if eq <= 80 else "telephoto (periscope) or digital zoom")
        rows.append(Row("Camera module", "ExifIFD:FocalLengthIn35mmFormat", f"{eq:g} mm",
                        f"field of view of a {eq:g} mm lens on full frame: {kind}" if phone else ""))
    if focal:
        rows.append(Row("Camera module", "ExifIFD:FocalLength", str(focal), "physical focal length of the lens"))
    return rows


def _exposure(meta: dict, phone: bool) -> list[Row]:
    rows = []
    t = _number(_get(meta, "ExifIFD:ExposureTime"))
    if t:
        note = ""
        if phone and t >= 0.25:
            note = ("long for a handheld phone: typical of a night mode that stacks frames, where the recorded time can "
                    "be the total of the stack")
        rows.append(Row("Exposure", "ExifIFD:ExposureTime", str(_get(meta, "ExifIFD:ExposureTime")) + " s", note))
    for tag, note in (("ExifIFD:ISO", ""), ("ExifIFD:FNumber", "phone apertures are fixed per camera module"),
                      ("ExifIFD:ExposureProgram", ""), ("ExifIFD:ExposureMode", ""),
                      ("ExifIFD:ExposureCompensation", ""), ("ExifIFD:Flash", ""), ("ExifIFD:WhiteBalance", ""),
                      ("ExifIFD:MeteringMode", "")):
        value = _get(meta, tag)
        if value is not None:
            rows.append(Row("Exposure", tag, str(value), note if phone else ""))
    bv = _number(_get(meta, "ExifIFD:BrightnessValue"))
    if bv is not None:
        rows.append(Row("Exposure", "ExifIFD:BrightnessValue", f"{bv:g}",
                        "low light: phones switch to multi-frame night processing automatically in such light"
                        if phone and bv < 0 else "scene brightness (APEX)"))
    focus = _get(meta, "Apple:FocusDistanceRange", "ExifIFD:SubjectDistanceRange")
    if focus is not None:
        rows.append(Row("Exposure", "Apple:FocusDistanceRange" if meta.get("Apple:FocusDistanceRange") else
                        "ExifIFD:SubjectDistanceRange", str(focus), "focus distance"))
    if (depth := _get(meta, "Apple:AFMeasuredDepth")) is not None:
        rows.append(Row("Exposure", "Apple:AFMeasuredDepth", str(depth),
                        "distance measured by the time-of-flight (LiDAR) sensor for focusing"))
    return rows


def _modes(meta: dict, data: bytes) -> list[Row]:
    rows = []
    capture = _get(meta, "Apple:ImageCaptureType")
    if capture is not None:
        meaning, evidence = APPLE_CAPTURE.get(str(capture), ("mode code not documented", False))
        rows.append(Row("Camera mode", "Apple:ImageCaptureType", str(capture), meaning, evidence))
    rendered = str(_get(meta, "ExifIFD:CustomRendered", "IFD0:CustomRendered") or "")
    if rendered == "Panorama":
        rows.append(Row("Camera mode", "ExifIFD:CustomRendered", rendered, "panorama stitched from many frames", True))
    elif rendered == "Original (for HDR)":
        rows.append(Row("Camera mode", "ExifIFD:CustomRendered", rendered,
                        "the unmerged original iOS kept next to an HDR version: this file is the single exposure"))
    special = _get(meta, "XMP-GCamera:SpecialTypeID")
    if special:
        text = " ".join(map(str, special)) if isinstance(special, list) else str(special)
        kind = text.rsplit("-", 1)[-1].rsplit(".", 1)[-1].upper()
        meaning = {"PORTRAIT": "Portrait mode: background blurred by software", "NIGHT": "Night Sight: frames stacked in "
                   "low light", "PANORAMA": "panorama", "PHOTOSPHERE": "Photo Sphere"}.get(kind, f"special mode {kind}")
        rows.append(Row("Camera mode", "XMP-GCamera:SpecialTypeID", text, meaning, kind != "MOTION"))
    if _get(meta, "XMP-GCamera:BurstID", "Apple:BurstUUID"):
        tag = "XMP-GCamera:BurstID" if meta.get("XMP-GCamera:BurstID") else "Apple:BurstUUID"
        rows.append(Row("Camera mode", tag, str(meta[tag])[:40], "part of a burst or group of shots taken together"))
    if (mode := _get(meta, "XMP-OPMedia:CaptureMode")) is not None:
        rows.append(Row("Camera mode", "XMP-OPMedia:CaptureMode", str(mode), "camera mode (OnePlus / OPPO)",
                        str(mode).lower() in ("bokeh", "portrait", "night", "nightscape")))
    if (scene := _get(meta, "XMP-OPMedia:Scene")) is not None:
        rows.append(Row("Camera mode", "XMP-OPMedia:Scene", str(scene), "scene the camera detected (OnePlus / OPPO)"))
    if (opmode := _get(meta, "JSON:OpMode")) is not None:
        rows.append(Row("Camera mode", "JSON:OpMode", str(opmode), "camera mode code (Xiaomi; not documented publicly)"))
    if (comment := _get(meta, "ExifIFD:UserComment")) is not None and str(comment).startswith("oplus_"):
        rows.append(Row("Camera mode", "ExifIFD:UserComment", str(comment), "camera mode code (OPPO / OnePlus; "
                        "not documented publicly)"))
    if (scene := _get(meta, "ExifIFD:SceneCaptureType")) not in (None, "Standard"):
        rows.append(Row("Camera mode", "ExifIFD:SceneCaptureType", str(scene), "scene program recorded by the camera"))
    if (app := _get(meta, "Google:Application")) is not None:
        rows.append(Row("Camera mode", "Google:Application", str(app),
                        f"camera app {_get(meta, 'Google:AppVersion') or ''}".strip()))
    return rows


def _multiframe(meta: dict) -> list[Row]:
    rows = []
    composite = _get(meta, "ExifIFD:CompositeImage")
    if composite in ("General Composite Image", "Composite Image Captured While Shooting"):
        rows.append(Row("Multi-frame and HDR", "ExifIFD:CompositeImage", str(composite),
                        "the picture is a composite of several frames", True))
        if (count := _number(_get(meta, "ExifIFD:CompositeImageCount"))):
            rows.append(Row("Multi-frame and HDR", "ExifIFD:CompositeImageCount", str(meta["ExifIFD:CompositeImageCount"]),
                            f"{int(count)} source frames", True))
        if (times := _get(meta, "ExifIFD:CompositeImageExposureTimes")) is not None:
            rows.append(Row("Multi-frame and HDR", "ExifIFD:CompositeImageExposureTimes", str(times),
                            "exposure times of the source frames (EXIF 2.32 layout)"))
    rendered = str(_get(meta, "ExifIFD:CustomRendered", "IFD0:CustomRendered") or "")
    if rendered.startswith("HDR"):
        rows.append(Row("Multi-frame and HDR", "ExifIFD:CustomRendered", rendered, "HDR merge of several exposures", True))
    if _get(meta, "Apple:HDRImageType") == "HDR Image":
        rows.append(Row("Multi-frame and HDR", "Apple:HDRImageType", "HDR Image", "merged from several exposures", True))
    headroom, gain = _get(meta, "Apple:HDRHeadroom"), _get(meta, "Apple:HDRGain")
    if (_number(headroom) or 0) > 0 or gain is not None:
        rows.append(Row("Multi-frame and HDR", "Apple:HDRHeadroom / Apple:HDRGain", f"{headroom} / {gain}",
                        "Smart HDR gain map: how far the merged highlights exceed standard white", True))
    hdrp = _get(meta, "XMP-GCamera:HDRPlusMakerNote", "XMP-GCamera:HDRPMakerNote")
    software = _get(meta, "Google:HDRPSoftware")
    if hdrp or software:
        tag = "Google:HDRPSoftware" if software else ("XMP-GCamera:HDRPlusMakerNote" if meta.get(
            "XMP-GCamera:HDRPlusMakerNote") else "XMP-GCamera:HDRPMakerNote")
        rows.append(Row("Multi-frame and HDR", tag, str(software or "present"), "HDR+ burst merge record", True))
    if (frames := _number(_get(meta, "Google:FrameCount"))):
        rows.append(Row("Multi-frame and HDR", "Google:FrameCount", f"{int(frames)}",
                        f"{int(frames)} frames merged into this photo", True))
    if str(_get(meta, "XMP-OPMedia:IsHDRActive")) == "True":
        rows.append(Row("Multi-frame and HDR", "XMP-OPMedia:IsHDRActive", "True", "HDR merge active", True))
    if str(_get(meta, "XMP-OPMedia:IsNightModeActive")) == "True":
        rows.append(Row("Multi-frame and HDR", "XMP-OPMedia:IsNightModeActive", "True", "night mode: frames stacked", True))
    if str(_get(meta, "JSON:Hdr")).lower() in ("on", "auto"):
        rows.append(Row("Multi-frame and HDR", "JSON:Hdr", str(meta["JSON:Hdr"]),
                        "HDR switched on or automatic: frames merged when the scene needs it", True))
    elif _get(meta, "JSON:Hdr") is not None:
        rows.append(Row("Multi-frame and HDR", "JSON:Hdr", str(meta["JSON:Hdr"]), "HDR switched off"))
    if _get(meta, "XMP-hdrgm:Version"):
        rows.append(Row("Multi-frame and HDR", "XMP-hdrgm:Version", str(meta["XMP-hdrgm:Version"]),
                        "Ultra HDR gain map: brighter highlights for HDR screens", True))
    for tag in ("CBOR:ActionsDigitalSourceType", "XMP-iptcExt:DigitalSourceType"):
        if "computationalCapture" in str(meta.get(tag, "")):
            rows.append(Row("Multi-frame and HDR", tag, str(meta[tag]).rsplit("/", 1)[-1],
                            "declared 'multi-frame computational capture' (IPTC)", True))
    return rows


def _depth(meta: dict) -> list[Row]:
    rows = []
    rendered = str(_get(meta, "ExifIFD:CustomRendered", "IFD0:CustomRendered") or "")
    if rendered in ("Portrait", "Portrait HDR"):
        rows.append(Row("Depth and portrait", "ExifIFD:CustomRendered", rendered,
                        "portrait mode: background blurred by software" + (", with HDR" if "HDR" in rendered else ""), True))
    if _get(meta, "XMP-GCamera:PortraitVersion", "XMP-GCamera:PortraitRequest"):
        tag = "XMP-GCamera:PortraitVersion" if meta.get("XMP-GCamera:PortraitVersion") is not None else "XMP-GCamera:PortraitRequest"
        rows.append(Row("Depth and portrait", tag, str(meta[tag])[:40], "portrait processing record (Google)", True))
    depth_keys = [k for k in meta if k.startswith(("XMP-GDepth:", "XMP-GFocus:")) or (k.startswith("XMP-Device:") and
                  "DepthMap" in k) or k in ("Samsung:DepthMapData", "Samsung:SingleShotDepthMap")]
    if depth_keys:
        rows.append(Row("Depth and portrait", ", ".join(sorted({k.split(":")[0] for k in depth_keys})),
                        f"{len(depth_keys)} depth tags", "depth map embedded: background blur (bokeh) computed by "
                        "software, and refocusable afterwards", True))
    if (level := _get(meta, "XMP-WT_Refocus:BokehLevel")) is not None:
        rows.append(Row("Depth and portrait", "XMP-WT_Refocus:BokehLevel", str(level), "software background blur level", True))
    if str(_get(meta, "XMP-OPMedia:IsBokehActive")) == "True":
        rows.append(Row("Depth and portrait", "XMP-OPMedia:IsBokehActive", "True",
                        "portrait mode: background blurred by software", True))
    return rows


def _zoom(meta: dict) -> list[Row]:
    rows = []
    tag = "ExifIFD:DigitalZoomRatio" if meta.get("ExifIFD:DigitalZoomRatio") is not None else "IFD0:DigitalZoomRatio"
    zoom = _number(meta.get(tag))
    if zoom and zoom > 1.05:
        model, make = str(_get(meta, "IFD0:Model") or ""), str(_get(meta, "IFD0:Make") or "").lower()
        text = "the picture was cropped and enlarged in the camera"
        if make.startswith("google") and "Pro" in model and zoom >= 30 and (_number(model.split()[1]) or 0) >= 10:
            text += "; Pro Res Zoom range, where detail is regenerated by generative AI"
        elif make.startswith("samsung") and zoom >= 25:
            text += "; Space Zoom range, where frames are merged and enhanced by AI"
        elif make.startswith("google"):
            text += "; on a Pixel by Super Res Zoom, which merges a burst"
        rows.append(Row("Zoom", tag, f"x{zoom:g}", text, True))
    elif zoom is not None and meta.get(tag) is not None:
        rows.append(Row("Zoom", tag, str(meta[tag]), "no digital zoom" if zoom <= 1.05 else ""))
    if (multiple := _number(_get(meta, "JSON:ZoomMultiple"))):
        rows.append(Row("Zoom", "JSON:ZoomMultiple", f"x{multiple:g}", "zoom chosen in the camera app (Xiaomi)",
                        multiple > 1.05))
    return rows


def _style(meta: dict) -> list[Row]:
    rows = []
    style = _get(meta, "Apple:SemanticStyle")
    if style:
        parts = dict(re.findall(r"_(\d)=([-\d.]+)", str(style)))
        preset = APPLE_STYLE_PRESETS.get(int(float(parts.get("3", 0))), "custom")
        rows.append(Row("Scene and style processing", "Apple:SemanticStyle", str(style),
                        f"Photographic Style '{preset}', tone {parts.get('1', '?')}, warmth {parts.get('2', '?')}: "
                        "tone and colour rendered by the camera", True))
    if str(_get(meta, "JSON:AIScene") or "0") not in ("0", "None"):
        rows.append(Row("Scene and style processing", "JSON:AIScene", str(meta["JSON:AIScene"]),
                        "AI scene detection applied (Xiaomi)", True))
    elif _get(meta, "JSON:AIScene") is not None:
        rows.append(Row("Scene and style processing", "JSON:AIScene", "0", "AI scene detection off or no scene found"))
    if (ids := _get(meta, "XMP-OPMedia:SceneDetectResultIds")) is not None:
        rows.append(Row("Scene and style processing", "XMP-OPMedia:SceneDetectResultIds", str(ids),
                        "AI scene detection results (OnePlus / OPPO)", True))
    if (fid := _get(meta, "JSON:FilterId")) is not None:
        rows.append(Row("Scene and style processing", "JSON:FilterId", str(fid), "colour filter code (Xiaomi)"))
    return rows


def _media(meta: dict) -> list[Row]:
    rows = []
    items = _get(meta, "XMP-GContainer:DirectoryItemSemantic")
    if items:
        names = items if isinstance(items, list) else [items]
        meanings = {"GainMap": "HDR gain map", "MotionPhoto": "video clip", "Depth": "depth map", "Primary": "the photo"}
        rows.append(Row("Embedded media", "XMP-GContainer:DirectoryItemSemantic", ", ".join(map(str, names)),
                        "images and video stored in the file: " + ", ".join(meanings.get(str(n), str(n)) for n in names)))
    if _get(meta, "XMP-GCamera:MotionPhoto", "XMP-GCamera:MicroVideo"):
        rows.append(Row("Embedded media", "XMP-GCamera:MotionPhoto" if meta.get("XMP-GCamera:MotionPhoto") else
                        "XMP-GCamera:MicroVideo", "1", "Motion photo: a short video around the shot is embedded; it shows "
                        "the moments before and after"))
    if _get(meta, "Apple:LivePhotoVideoIndex", "Apple:ContentIdentifier"):
        tag = "Apple:LivePhotoVideoIndex" if meta.get("Apple:LivePhotoVideoIndex") else "Apple:ContentIdentifier"
        rows.append(Row("Embedded media", tag, str(meta[tag])[:40], "Live Photo: a 3-second video was recorded with "
                        "the shot; on the phone it is a separate file with the same identifier"))
    if (video := _get(meta, "XMP-OpCamera:OLivePhotoVersion")) is not None:
        rows.append(Row("Embedded media", "XMP-OpCamera:OLivePhotoVersion", str(video),
                        "live photo: a video around the shot is embedded (OPPO / OnePlus)"))
    return rows


def describe(meta: dict, data: bytes = b"") -> list[Row]:
    """Every row of the report, in section order."""
    device = device_defaults(meta)
    phone = device is not None
    rows = []
    make, model = _get(meta, "IFD0:Make"), _get(meta, "IFD0:Model")
    if make or model:
        rows.append(Row("Device", "IFD0:Make / IFD0:Model", f"{make or '?'} / {model or '?'}",
                        "smartphone" if phone else ""))
    for tag, meaning in (("ExifIFD:XiaomiModel", "marketing name"), ("Google:DeviceCodename", "device code name"),
                         ("IFD0:Software", "firmware or camera software"), ("Google:AndroidRelease", "Android build")):
        if (value := _get(meta, tag)) is not None:
            if tag == "IFD0:Software" and str(value).startswith("HDR+"):
                meaning = "Google's HDR+ pipeline wrote this file"
            rows.append(Row("Device", tag, str(value), meaning))
    rows += _module(meta, phone) + _exposure(meta, phone) + _modes(meta, data) + _multiframe(meta) + _depth(meta)
    rows += _zoom(meta) + _style(meta) + _media(meta)
    for name in dict.fromkeys(_SAMSUNG.findall(data)):
        section, meaning, evidence = SAMSUNG_BLOCKS[name]
        rows.append(Row(section, f"Samsung trailer '{name.decode()}'", "present", meaning, evidence))
    if device:
        rows += [Row("Default processing of this model", device[0], "", line) for line in device[1]]
    return sorted(rows, key=lambda r: SECTIONS.index(r.section))


def recorded_processing(meta: dict, data: bytes = b"") -> list[tuple[str, str]]:
    """(tag, what it shows) for every row that is evidence of computational capture."""
    return [(r.tag, r.meaning) for r in describe(meta, data) if r.evidence]


def detect(ev: Evidence) -> Result:
    res = Result()
    meta = metadata.read_metadata(ev)
    rows = describe(meta, ev.data)
    device = device_defaults(meta)
    evidence = [r for r in rows if r.evidence]
    section = None
    for r in rows:
        if r.section != section:
            res.table.append((f"--- {r.section} ---", ""))
            section = r.section
        value = f"{r.value}  ->  {r.meaning}" if r.value and r.meaning else (r.value or r.meaning)
        res.table.append((r.tag, value + ("   [evidence]" if r.evidence else "")))
    if not device and not evidence:
        res.score = 0.05
        if not meta:
            res.findings.append(Finding(INFO, "No camera metadata",
                "The device cannot be identified: the metadata was stripped or ExifTool is missing. Computational "
                "capture can neither be confirmed nor excluded."))
        else:
            res.findings.append(Finding(INFO, "No sign of computational photography",
                "The camera is not a smartphone known to merge frames by default, and the file records no multi-frame, "
                "depth or zoom processing. The table lists what the metadata says about the capture."))
        return res
    res.score = SCORE
    caveat = ("\n\nNot a finding of manipulation: the camera computed these pixels from several frames or with "
              "software effects. Merged, sharpened, depth-blurred or enlarged pixels can make the noise, CFA, "
              "resampling and ELA tools and the AI detectors respond; weigh their results with that in mind, and "
              "prefer the camera original for important cases. The table shows each tag and what it means.")
    if device:
        res.findings.append(Finding(WARN, "Smartphone with computational photography by default",
            f"{device[0]}: " + "; ".join(device[1]) + "." + ("" if evidence else caveat)))
    if evidence:
        res.findings.append(Finding(WARN, "Computational capture recorded in the file",
            "\n".join(f"{r.meaning} ({r.tag} = {r.value})" if r.value and r.value != "present" else f"{r.meaning} ({r.tag})"
                      for r in evidence) + caveat))
    return res
