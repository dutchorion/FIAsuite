"""Computational photography check. The metadata below is copied from ExifTool's output for its sample files of
real phones and cameras (exiftool.org/sample_images.html), cut to the tags that matter."""
import pytest

from fiasuite.analyses import computational
from fiasuite.core import WARN, Evidence
from fiasuite.verdict import summarize
from tests import synth

IPHONE_15_PLUS = {"IFD0:Make": "Apple", "IFD0:Model": "iPhone 15 Plus", "ExifIFD:DateTimeOriginal": "2023:10:01 12:00:00",
                  "ExifIFD:CompositeImage": "Composite Image Captured While Shooting", "ExifIFD:CompositeImageCount": "8 0",
                  "Apple:HDRHeadroom": 1.00999999, "Apple:HDRGain": 0.9525256166,
                  "Apple:SemanticStyle": "{_0=1,_1=0,_2=0,_3=0}", "Apple:ImageCaptureType": "Manual Focus"}
IPHONE_11_PORTRAIT = {"IFD0:Make": "Apple", "IFD0:Model": "iPhone 11", "ExifIFD:CustomRendered": "Portrait HDR",
                      "Apple:HDRHeadroom": 1}
PIXEL_10_PRO_XL = {"IFD0:Make": "Google", "IFD0:Model": "Pixel 10 Pro XL", "ExifIFD:DigitalZoomRatio": 2.21,
                   "ExifIFD:CompositeImage": "Composite Image Captured While Shooting", "XMP-hdrgm:Version": "1.0",
                   "XMP-GCamera:HDRPlusMakerNote": "(Binary data 49872 bytes, use -b option to extract)",
                   "Google:HDRPSoftware": "HDR+ 1.0.796157346",
                   "CBOR:ActionsDigitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/computationalCapture"}
PIXEL_4A_PORTRAIT = {"IFD0:Make": "Google", "IFD0:Model": "Pixel 4a", "ExifIFD:DigitalZoomRatio": 1.5,
                     "XMP-GCamera:SpecialTypeID": "com.google.android.apps.camera.gallery.specialtype.SpecialType-PORTRAIT",
                     "XMP-Device:DepthMapFormat": "RangeInverse"}
ONEPLUS_BOKEH = {"IFD0:Make": "OnePlus", "IFD0:Model": "CPH2449", "ExifIFD:DateTimeOriginal": "2023:05:01 10:00:00",
                 "XMP-OPMedia:CaptureMode": "Bokeh", "XMP-OPMedia:IsHDRActive": "True", "XMP-OPMedia:IsBokehActive": "True"}
IPHONE_7 = {"IFD0:Make": "Apple", "IFD0:Model": "iPhone 7", "ExifIFD:CustomRendered": "Custom"}
CANON = {"IFD0:Make": "Canon", "IFD0:Model": "Canon EOS R5", "ExifIFD:CustomRendered": "Normal"}
OLD_GALAXY = {"IFD0:Make": "SAMSUNG", "IFD0:Model": "SM-G900F", "ExifIFD:DateTimeOriginal": "2014:06:01 10:00:00"}


def sources(meta, data=b""):
    return [source for source, _ in computational.recorded_processing(meta, data)]


def test_iphone_composite_frames_gain_map_and_style():
    recorded = dict(computational.recorded_processing(IPHONE_15_PLUS))
    assert "8 source frames" in recorded["EXIF CompositeImage"]
    assert {"Apple HDRHeadroom / HDRGain", "Apple SemanticStyle"} <= set(recorded)
    device, lines = computational.device_defaults(IPHONE_15_PLUS)
    assert any("Smart HDR" in line for line in lines) and any("Photonic Engine" in line for line in lines)


def test_portrait_hdr_on_iphone_and_depth_map_on_pixel():
    assert "portrait" in dict(computational.recorded_processing(IPHONE_11_PORTRAIT))["EXIF CustomRendered"]
    assert {"Google camera XMP", "depth map", "EXIF DigitalZoomRatio"} <= set(sources(PIXEL_4A_PORTRAIT))


def test_pixel_10_hdr_plus_and_computational_capture_declaration():
    assert {"Google HDR+ maker note", "IPTC digital source type", "XMP hdrgm"} <= set(sources(PIXEL_10_PRO_XL))
    _, lines = computational.device_defaults(PIXEL_10_PRO_XL)
    assert any("Pro Res Zoom" in line for line in lines)


def test_generative_zoom_range_is_named():
    zoomed = dict(PIXEL_10_PRO_XL, **{"ExifIFD:DigitalZoomRatio": 50})
    assert "generative AI" in dict(computational.recorded_processing(zoomed))["EXIF DigitalZoomRatio"]
    tiny = dict(IPHONE_15_PLUS, **{"ExifIFD:DigitalZoomRatio": 1.001988072})          # rounding, not zoom
    assert "EXIF DigitalZoomRatio" not in sources(tiny)


def test_samsung_trailer_records_and_galaxy_names():
    trailer = b"\xff\xd9" + b"\x00" * 16 + b"Photo_HDR_Info" + b"\x01" * 8 + b"Livefocus_JDM_Info" + b"SEFT"
    assert {"Samsung trailer 'Photo_HDR_Info'", "Samsung trailer 'Livefocus_JDM_Info'"} <= set(sources({}, trailer))
    assert computational.device_defaults({"IFD0:Make": "samsung", "IFD0:Model": "Galaxy S25 Ultra"})
    assert computational.device_defaults(OLD_GALAXY) is None                       # 2014: no such default yet


def test_other_makers_and_cameras():
    assert {"OnePlus IsHDRActive", "OnePlus bokeh"} <= set(sources(ONEPLUS_BOKEH))
    assert computational.device_defaults(ONEPLUS_BOKEH)
    for meta in (IPHONE_7, CANON):
        assert computational.device_defaults(meta) is None and sources(meta) == []
    assert computational.device_defaults({"IFD0:Make": "SONY", "IFD0:Model": "ILCE-7M4"}) is None     # camera
    assert computational.device_defaults({"IFD0:Make": "Sony", "IFD0:Model": "XQ-DQ54"})               # Xperia


@pytest.fixture()
def evidence(tmp_path):
    def make(meta):
        ev = Evidence(synth.save(tmp_path / "photo.jpg", synth.camera(synth.scene(240, 320)), 90))
        ev.cache["meta"] = meta                    # as read by ExifTool
        return ev
    return make


def test_phone_photo_is_orange_and_makes_file_history_suspicious(evidence):
    res = computational.detect(evidence(IPHONE_15_PLUS))
    assert [f.severity for f in res.findings] == [WARN, WARN] and res.score == computational.SCORE
    assert "Not a finding of manipulation" in res.findings[-1].detail
    verdict = summarize({"computational": res})["provenance"]
    assert verdict.level == "Suspicious" and verdict.color == "#f76b15"


def test_camera_and_stripped_files_stay_green(evidence):
    assert computational.detect(evidence(CANON)).findings[0].title == "No sign of computational photography"
    assert computational.detect(evidence({})).findings[0].title == "No camera metadata"
