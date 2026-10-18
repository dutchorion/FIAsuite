"""Second half of the guide: entries for the newer tools, explanations of options added later, and worked example
cases for every tool. A case is (title, what you see and what it means, picture or None); pictures live in
fiasuite/guide_examples and are produced by tools/make_guide_examples.py from public-domain photographs."""
from __future__ import annotations

FARID_BOOK = ("H. Farid, Photo Forensics, MIT Press 2016 (book)", "https://mitpress.mit.edu/9780262537001/photo-forensics/")

NEW_TOOLS: dict[str, dict] = {
    "prnu": {
        "what": "Sensor pattern noise (PRNU, photo-response non-uniformity): every image sensor has a fixed, unique "
                "pattern of pixel sensitivities that is stamped into each photo it takes. With a fingerprint built "
                "from reference photos of a camera, this tool tests whether the image came from that camera and "
                "which regions lack its pattern.",
        "read": ["Build the fingerprint first: 'Camera fingerprint ...' in the toolbar, 20 or more photos from the one "
                 "camera, full resolution, never edited. Bright, evenly lit, low-detail shots (sky, a wall) work best.",
                 "PCE (peak-to-correlation energy) is the match statistic. Above 60 the photo carries the pattern; "
                 "unrelated cameras stay around 20-40. In this program's own test on four real cameras (fingerprints from "
                 "30 photos each) 100 of 100 photos matched their camera, the weakest with PCE 152, and none of 300 "
                 "comparisons with a foreign camera matched (highest PCE 44).",
                 "'Local correlation' shows where the pattern is found; 'Fingerprint missing' marks connected areas "
                 "where it should be present but is not: content that this sensor did not record.",
                 "A cropped photo is located inside the sensor frame automatically; rotations by 90 degrees are tried."],
        "options": {"Camera fingerprint": "Which stored fingerprint to test against.",
                    "Analysis window": "Size of the area each local correlation is measured on. A correlation measured "
                                       "on b x b pixels scatters by about 1/b, so weak fingerprints need large windows, "
                                       "and forged areas smaller than the window are missed. Auto picks the smallest "
                                       "window in which the pattern stands clear of the noise."},
        "pitfalls": ["No match does not prove another camera: resizing, heavy compression, digital stabilisation, HDR / "
                     "night modes, panorama and lens-distortion correction all break the pixel alignment.",
                     "The tamper map needs a strong fingerprint. With 30 reference photos it found about a quarter of "
                     "small hand-made forgeries (1-8% of the picture) and raised no false alarm on 12 untouched photos.",
                     "Dark, clipped and heavily textured areas carry little pattern and are left untested."],
        "sources": [("J. Lukas, J. Fridrich, M. Goljan, Digital Camera Identification From Sensor Pattern Noise, IEEE TIFS 2006",
                     "https://doi.org/10.1109/TIFS.2006.873602"),
                    ("M. Chen, J. Fridrich, M. Goljan, J. Lukas, Determining Image Origin and Integrity Using Sensor Noise, IEEE TIFS 2008",
                     "https://doi.org/10.1109/TIFS.2007.916285"),
                    ("M. Goljan, J. Fridrich, T. Filler, Large scale test of sensor fingerprint camera identification, SPIE 2009 (PCE threshold)",
                     "https://doi.org/10.1117/12.805701"),
                    ("P. Korus, J. Huang, Multi-Scale Analysis Strategies in PRNU-Based Tampering Localization, IEEE TIFS 2017",
                     "https://doi.org/10.1109/TIFS.2016.2636089"),
                    ("prnu-python (Politecnico di Milano) - reference implementation", "https://github.com/polimi-ispl/prnu-python")],
    },
    "geometry": {
        "what": "Physical consistency checks that you draw yourself. Light and perspective obey simple geometry that "
                "forgers and image generators rarely get right, and that survives any amount of recompression, "
                "resizing or screenshotting.",
        "read": ["Drag with the left mouse button to draw a line; it is extended across the picture. Ctrl + drag pans, "
                 "the wheel zooms, 'Fit' shows the whole construction including points outside the frame.",
                 "Shadows: draw a line from a point on a cast shadow through the point of the object that casts it (tip of "
                 "a shadow to the tip of the post). All such lines in one picture must meet in ONE point, the image of "
                 "the light source (it may lie outside the frame, or behind the camera).",
                 "Perspective: draw edges that are parallel in the real scene (one side of a building, lane markings, "
                 "tile joints). They must meet in ONE vanishing point. Use a second set for a second direction; the "
                 "vanishing points of horizontal directions lie on one straight horizon (reported when three sets have a point).",
                 "Reflections: lines joining a point and its mirror image in a flat mirror or still water also meet in one point.",
                 "The table lists, per line, by how many degrees it misses the common point of the others. With four or "
                 "more lines the common point is taken from the most consistent subset, so a single stray line is named."],
        "pitfalls": ["It can only show inconsistency. Consistent geometry proves nothing.",
                     "The result is as good as the annotation: soft shadow tips, short lines and lens distortion (wide-"
                     "angle phones) easily cost a few degrees. Misses below about 3 degrees are not evidence.",
                     "Several light sources (street lamps, flash plus sun) legitimately give several shadow directions."],
        "sources": [("E. Kee, J. O'Brien, H. Farid, Exposing photo manipulation with inconsistent shadows, ACM TOG 2013",
                     "https://doi.org/10.1145/2487228.2487236"),
                    ("J. O'Brien, H. Farid, Exposing photo manipulation with inconsistent reflections, ACM TOG 2012",
                     "https://doi.org/10.1145/2077341.2077345"),
                    ("H. Farid, Perspective (In)consistency of Paint by Text (2022) - the same tests on AI images",
                     "https://arxiv.org/abs/2206.14617"),
                    ("H. Farid, Lighting (In)consistency of Paint by Text (2022)", "https://arxiv.org/abs/2207.13744"), FARID_BOOK],
    },
    "watermark": {
        "what": "Decodes the invisible watermark that Stability AI's reference software writes into generated images "
                "(the 'invisible-watermark' library, method dwtDct): Stable Diffusion 1.x, 2.x and SDXL.",
        "read": ["For each known message the table gives how many bits were read back correctly and the probability of "
                 "that happening by chance. A hit (chance below one in a million) is conclusive: the pixels were "
                 "written by that software.",
                 "Google's SynthID and OpenAI's watermark cannot be decoded locally; use 'Watermark check' in the "
                 "toolbar and the OpenAI provenance tool."],
        "pitfalls": ["A miss means nothing. AUTOMATIC1111, ComfyUI and most hosted services do not embed this mark.",
                     "The mark is fragile: it survives lossless copies only. In this program's test JPEG compression at "
                     "quality 90 and a 10% resize each erased it completely.",
                     "None of 187 real photographs tested gave a false hit."],
        "sources": [("invisible-watermark library (ShieldMnt), MIT licence", "https://github.com/ShieldMnt/invisible-watermark"),
                    ("Stable Diffusion reference script that embeds 'StableDiffusionV1'",
                     "https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py"),
                    ("Stability AI generative-models: the SDXL watermark message",
                     "https://github.com/Stability-AI/generative-models/blob/main/sgm/inference/helpers.py")],
    },
    "histogram": {
        "what": "Histograms of the pixel values per channel, a map of clipped pixels, and two measurements of how "
                "comb-like the histogram is.",
        "read": ["A photograph straight from a camera has a smooth histogram. Remapping 8-bit values (levels, curves, "
                 "contrast, brightness, gamma, many filters) stretches some tonal ranges, leaving empty bins, and "
                 "compresses others, leaving spikes: a comb.",
                 "'Empty bins inside the used range' and 'Bin-to-bin ripple' quantify this. On 35 untouched photographs "
                 "the ripple stayed below 0.13; a Levels adjustment gave 2.5, a gamma change 0.25.",
                 "The clipped-pixel view shows burnt-out highlights (red) and blocked shadows (blue): large clipped "
                 "areas carry no information for any other tool."],
        "options": {"Channel": "All channels or one.", "Logarithmic scale": "Makes thinly populated bins visible."},
        "pitfalls": ["A global tonal adjustment is ordinary photo editing, not content manipulation.",
                     "JPEG compression after the adjustment smooths the comb away (at quality 90 it was gone).",
                     "Graphics, screenshots and images with few colours have comb histograms by nature."],
        "sources": [("M. Stamm, K. J. R. Liu, Forensic detection of image manipulation using statistical intrinsic fingerprints, IEEE TIFS 2010",
                     "https://doi.org/10.1109/TIFS.2010.2053202"),
                    ("Sherloq (G. Bartoli) - histogram and contrast-enhancement tools", "https://github.com/GuidoBartoli/sherloq")],
    },
    "hidden": {
        "what": "For files with transparency (PNG, WebP, TIFF): the alpha channel itself and the colour values stored "
                "underneath fully transparent pixels.",
        "read": ["Erasing or masking in an editor usually only sets the transparency to zero. The colour of the erased "
                 "pixels stays in the file and is shown in 'Hidden pixels only'.",
                 "All other tools in this program ignore transparency and always analyse the stored colours."],
        "pitfalls": ["Many exporters fill transparent areas with black or white: then nothing is hidden.",
                     "Hidden content shows how a file was made; it is not by itself a sign of deception."],
        "sources": [("FotoForensics tutorial: Hidden Pixels", "https://fotoforensics.com/tutorial.php?tt=hidden")],
    },
}

# explanations of options that were added to existing tools
EXTRA_OPTIONS: dict[str, dict[str, str]] = {
    "ela": {"Automatic error scale": "Chooses the amplification so that the strongest errors are just short of white. "
                                     "High-quality files leave errors of one or two levels that a fixed scale shows as "
                                     "black. Switch it off to compare two images at the same fixed scale.",
            "Display": "Colour shows the error of each channel (colour fringes are normal with chroma subsampling); "
                       "'Brightest channel' shows one grey value per pixel, which is easier to compare across surfaces."},
    "noise": {"Denoiser": "What is subtracted to isolate the noise. Median 3x3 is the classic view; Median 5x5 and "
                          "Gaussian keep coarser grain; Non-local means keeps edges out of the residual but is slow.",
              "Grey (strength only)": "Shows the strength of the grain without its colour."},
    "noisemap": {"Outlier threshold": "How many robust standard deviations (pooled over 3x3 blocks) a block must depart "
                                      "from the trend to count. Lower = more sensitive and more false alarms."},
    "resampling": {"Peak threshold": "How far a spectral peak must rise above its surroundings for a window to count as "
                                     "resampled. Lower it for inserts that were compressed afterwards; regular textures "
                                     "then trigger more often."},
    "clone_kp": {"Also search mirrored copies": "SIFT does not recognise a flipped copy. This also matches against the "
                                                "mirrored image. Truly symmetric objects (faces, facades, cars seen "
                                                "head-on) then match themselves, so it is off by default."},
    "spectrum": {"Reduce tile size": "Halves the tile until at least 8 tiles fit. Switch off to force the chosen size.",
                 "Denoiser strength": "Gaussian sigma in pixels for the Wiener and Gaussian residuals; the median "
                                      "filter uses a 3x3 window below 1.5 and 5x5 above. Larger values let lower "
                                      "frequencies into the residual. Not used by Cross-difference.",
                 "Tile limit": "At most this many tiles, spread evenly over the image, are averaged; 0 uses all.",
                 "Peak threshold": "Robust z-score a peak must reach. With few tiles a peak also counts at 60% of this "
                                   "value if it is ten times the background.",
                 "Minimal peak / background": "Smallest ratio between a peak and its surroundings.",
                 "Colour map": "Display only."},
}

# what the tools achieved on real photographs (see "Measured accuracy" in the README for the test sets)
EXTRA_PITFALLS: dict[str, list[str]] = {
    "ghosts": ["Measured on 40 real photographs with an insert from a quality-60 JPEG, saved at 92: 27 found; 1 false "
               "alarm in 80 untouched single- or double-compressed photographs.",
               "When the WHOLE image was compressed twice, its own grid is not searched for local ghosts: harmonics of "
               "the first quality imitate them (15 of 40 clean re-saved photos were flagged before this rule)."],
    "dct": ["Double-quantization map, measured on real photographs first saved at quality 70, then at 90: 37 of 40 "
            "inserts found, 1 false region in 40 untouched re-saved photos. It needs the second quality to be higher "
            "than the first."],
    "clone_kp": ["A group related by a plain shift is kept only if the smooth parts of both areas carry the SAME pixel "
                 "noise; look-alike content has independent noise there and is discarded and counted in the table. "
                 "Scaled or rotated copies cannot be checked this way.",
                 "Measured on 100 untouched and 100 hand-forged camera photographs: 35 forgeries flagged, 6 false "
                 "alarms - identical manufactured objects such as two ornamental window frames of one house, which "
                 "only an examiner can tell from a copy."],
    "clone_blocks": ["The same-noise check removed most false alarms on real photographs: 34 of 100 untouched photos "
                     "were flagged before it, 6 after; 15 of 100 forgeries are flagged."],
    "cfa": ["Measured on 100 untouched never-compressed camera photographs: 2 false alarms, 7 of 100 small forgeries found."],
    "resampling": ["Measured on 100 untouched camera photographs: 1 false alarm. Hand-made forgeries that were not "
                   "resized are, correctly, not found."],
    "spectrum": ["Peaks weaker than 4x the background come from fine scene texture and are ignored: 1 of 280 photographs "
                 "is flagged with this rule. Only 13 of 245 images from current generators showed peaks at all."],
    "trufor": ["Measured on 20 untouched and 20 hand-forged never-compressed camera photographs (forgeries of 1-8% of "
               "the picture): 7 forgeries reached the alarm level of 0.7 with 1 false alarm; at 0.4 it is 12 against 3. "
               "Where it responds, the map is precise: typically under 0.1% of the picture wrongly marked."],
    "noisemap": ["Conservative: no false alarm on 100 untouched photographs, but none of 100 small, carefully blended "
                 "forgeries was flagged either. Use the noise residual view and TruFor for subtle edits."],
}

EXTRA_READ: dict[str, list[str]] = {
    "dct": ["'Double-quantization map' (shown when the image was compressed twice): per 8x8 block, the probability "
            "that its coefficients do NOT follow the comb of the rest of the image. A connected bright region is "
            "content that entered the picture after the first compression."],
}
EXTRA_SOURCES: dict[str, list[tuple[str, str]]] = {
    "dct": [("Z. Lin, J. He, X. Tang, C.-K. Tang, Fast, automatic and fine-grained tampered JPEG image detection via DCT "
             "coefficient analysis, Pattern Recognition 2009", "https://doi.org/10.1016/j.patcog.2009.03.019"),
            ("T. Bianchi, A. Piva, Image Forgery Localization via Block-Grained Analysis of JPEG Artifacts, IEEE TIFS 2012",
             "https://doi.org/10.1109/TIFS.2012.2187516")],
}

CASES: dict[str, list[tuple[str, str, str | None]]] = {
    "metadata": [
        ("Camera original", "Make, Model, lens and exposure data, MakerNotes of the brand, identical DateTimeOriginal / "
         "CreateDate / ModifyDate, an embedded thumbnail. No finding above 'info'.", None),
        ("Edited in Photoshop or similar", "Software or CreatorTool names the editor, ModifyDate is later than the capture "
         "date, XMP History lists the saves. 'DocumentAncestors' is the strongest entry: it lists other files whose "
         "content was pasted in.", None),
        ("Copy from social media or a messenger", "No EXIF at all, only JFIF and perhaps an ICC profile: 'No camera "
         "metadata'. Normal for such copies and no sign of manipulation; ask for the original file.", None),
        ("Straight from an AI generator", "PNG text chunks with prompt, seed and sampler (Stable Diffusion front-ends), "
         "IPTC DigitalSourceType 'trainedAlgorithmicMedia', or a credit such as 'Made with Google AI'. Red finding.", None),
        ("Screenshot", "PNG without camera data, often with the operating system as software and screen-sized dimensions. "
         "The content may be anything; pixel-level tools then examine the screenshot, not the original.", None)],
    "c2pa": [
        ("Signed by a camera", "Claim generator is the camera firmware (Leica, Sony, Nikon, Google Pixel), action "
         "'c2pa.created' with a digital source type of digitalCapture, signature valid: strong evidence of a camera original.", None),
        ("Signed by a generator", "OpenAI, Adobe Firefly, Google and Microsoft sign their output with "
         "'trainedAlgorithmicMedia'. Red finding: a signed declaration that the image is AI-generated.", None),
        ("Edited with credentials kept", "A chain of actions (opened, cropped, 'compositeWithTrainedAlgorithmicMedia' for "
         "generative fill) and ingredients: the recorded editing history.", None),
        ("Validation failure", "The manifest is present but the hash no longer matches: the file was changed after signing.", None),
        ("No manifest", "By far the most common case. It says nothing: almost every platform strips the manifest.", None)],
    "trustmark": [
        ("Watermarked, metadata intact", "A TrustMark identifier and a valid C2PA manifest: the file is as its "
         "issuer published it, and the identifier can recover the credentials if the metadata is lost later.", None),
        ("Watermarked, metadata stripped", "A TrustMark identifier but no manifest: the file was re-saved or uploaded "
         "after watermarking. Ask the issuer to look the identifier up.", None),
        ("No watermark", "By far the most common case. It says nothing.", None)],
    "jpeg": [
        ("Camera file", "Baseline JPEG, the manufacturer's own quantization tables, 4:2:2 or 4:2:0 subsampling, an Exif "
         "segment, often a Multi-Picture segment with a preview after the main image.", None),
        ("Saved by Adobe software", "APP13 Photoshop and APP14 Adobe segments, Adobe's own tables, 4:4:4 at high quality; "
         "'Ducky' marks Save for Web.", None),
        ("Re-encoded by a platform", "JFIF header, standard IJG tables at quality 70-85, 4:2:0, frequently progressive, "
         "no Exif. The normal state of anything downloaded from social media.", None),
        ("Appended data", "Bytes after the end-of-image marker: harmless for phones (gain map, depth map, motion photo), "
         "otherwise worth a look with String extraction - another file may be attached.", None)],
    "thumbnail": [
        ("Editor left the camera thumbnail in place", "The thumbnail still shows the scene before the edit; 'Difference' "
         "lights up exactly where content was added or removed.", "thumbnail_edited"),
        ("Cropped picture", "The thumbnail has another aspect ratio and shows more of the scene than the main image.", None),
        ("Consistent", "Faint overall differences only (the thumbnail is tiny and heavily compressed).", None)],
    "hashes": [
        ("Same file", "Identical SHA-256: bit-for-bit the same file as the one documented earlier.", None),
        ("Same picture, other file", "Different SHA-256 but a perceptual hash within a few bits: a re-saved, resized or "
         "re-compressed copy of the same picture. Use Compare to see what changed.", None)],
    "strings": [
        ("Leftovers of the workflow", "Software names, file paths of the editing computer, XMP packets that ExifTool "
         "did not map to tags, generator prompts.", None),
        ("Attached data", "Readable text after the image data (see JPEG structure: trailing bytes).", None)],
    "ela": [
        ("Saved once", "Edges and texture are bright, flat areas dark, and similar surfaces look alike everywhere.", "ela_original"),
        ("Object pasted from another JPEG", "The insert responds differently from comparable surfaces around it: here "
         "clearly brighter, with a sharp rectangular border.", "ela_splice"),
        ("Same forgery after several re-saves", "Every re-save brings all content to the same error level. After five "
         "saves at quality 75 the insert is invisible: a clean ELA never clears a recompressed image.", "ela_resaved"),
        ("Typical false alarms", "Sharpened edges, text and logos, high-contrast patterns and images scaled after "
         "compression light up although nothing was pasted.", None)],
    "ghosts": [
        ("Insert from a lower-quality JPEG", "Content once saved at quality 60, pasted into a picture saved at 92: bright "
         "rectangle in 'Ghost strength', a dark tile at Q60 in 'All qualities'.", "ghosts_splice"),
        ("Whole picture re-saved", "No local ghost, but 'Image-wide error minima' lists the earlier quality besides the "
         "current one: double compression of the entire image.", None),
        ("What it cannot see", "Inserts that were never JPEG, or came from a higher quality than the final save, leave "
         "no ghost. Use the double-quantization map (DCT tool) for those.", None)],
    "dct": [
        ("Single compression", "Histograms decay smoothly from the centre; periodicity below 0.35.", "dct_histograms"),
        ("Double compression", "Regular gaps and peaks (a comb): the image was decoded and saved again at another "
         "quality. Normal for anything edited or shared; it shows the file is not a camera original.", "dct_histograms"),
        ("Object added between two compressions", "The background carries the comb of both compressions, the insert "
         "only that of the last one: bright region in the double-quantization map.", "dct_map"),
        ("JPEG history inside a PNG", "Coefficients sit on a comb although the file is lossless: it was a JPEG before. "
         "A rough earlier quality is estimated.", None)],
    "grid": [
        ("Pasted from another JPEG", "The insert brings its own 8x8 grid, which almost never lines up with the grid of "
         "the host: bright region in 'Foreign grid'. The insert here came from a quality-50 JPEG and covers several "
         "64 px windows; smaller or better-quality inserts are often missed (14 of 40 found in this program's test on "
         "real photographs, with no false alarm in 80 untouched ones). JPEG ghosts and the double-quantization map "
         "are the stronger tests.", "grid_foreign"),
        ("Cropped after compression", "The dominant grid does not start at the top-left corner: 'Block grid is shifted'. "
         "Shows a crop, not a content change.", None),
        ("No grid at all", "Never JPEG-compressed, saved at very high quality, or resized afterwards (resizing and AI "
         "upscaling destroy the grid).", None)],
    "noise": [
        ("Retouched area", "Soft brushes, healing tools and inpainting remove the sensor grain: a dark, smooth patch in "
         "the residual where comparable surfaces are grainy.", "noise_smoothed"),
        ("Spliced content", "Grain that is coarser, finer or differently coloured than on similar surfaces nearby.", None),
        ("Generated images", "Often unnaturally uniform or patterned residual; grain that does not get stronger in "
         "brighter areas as sensor noise does.", None)],
    "noisemap": [
        ("Locally smoothed area", "Blocks inside the retouched region fall far below the noise level expected for their "
         "brightness (blue in 'Deviation from trend').", "noisemap_smoothed"),
        ("False alarms", "Phone portrait mode (synthetic background blur), HDR stacking and strong noise reduction in the "
         "shadows produce genuine regional differences.", None)],
    "cfa": [
        ("Insert in a never-compressed camera image", "The camera's demosaicing trace covers the picture except where "
         "content was pasted, resized or retouched (RAW conversions, TIFF, some PNG workflows).", None),
        ("Camera JPEG", "In-camera JPEG compression already weakens the trace; on the 18-megapixel camera JPEG used for "
         "this guide's examples the map was too noisy to show an insert. Treat a missing trace as no information.", None),
        ("No trace anywhere", "The usual result: resizing, strong compression, phone processing pipelines, screenshots "
         "and AI generation leave none. Then the tool says nothing about manipulation.", None)],
    "resampling": [
        ("Insert that was scaled or rotated to fit", "Windows over the insert show spectral peaks that the rest lacks.", "resampling_insert"),
        ("Whole image resized", "Peaks in most windows: 'Whole image has been resampled'. Usual for web copies.", None),
        ("False alarms", "Regular fine texture: fabric, fences, brickwork, halftone print, screens.", None)],
    "prnu": [
        ("Photo from the camera", "PCE in the hundreds or thousands, 'Sensor pattern matches', uniform local correlation.", None),
        ("Photo from another camera of the same model", "PCE stays low (20-40): the pattern belongs to the individual "
         "sensor, not to the model.", None),
        ("Matching photo with an insert", "High PCE overall, and a connected region in 'Fingerprint missing'.", None),
        ("Crop", "The position of the crop inside the sensor frame is reported; a crop that does not start where the "
         "file claims is itself informative.", None)],
    "clone_kp": [
        ("Duplicated textured area", "A stretch of shoreline copied to the right: matching points joined by lines of one "
         "colour, both areas outlined. Twenty or more pairs with one common transform is a strong result.", "clone_keypoints"),
        ("Genuinely repeating content", "Windows of a facade, tiles, printed text and product shelves match themselves. "
         "Look for identical noise and identical flaws, which real repetitions never share.", None),
        ("Flipped copy", "Invisible to the normal search; enable 'Also search mirrored copies'.", None)],
    "clone_blocks": [
        ("Duplicated low-contrast area", "A patch of water surface copied to cover something: coloured blocks at both "
         "positions, linked by one offset.", "clone_blocks"),
        ("Very smooth areas", "Hazy hills or an even sky carry too little detail and are skipped by default; lower "
         "'Minimal detail' to include them, at the price of more false matches.", None),
        ("False alarms", "Perfectly uniform or evenly graded areas (studio backdrops, clipped sky) and synthetic graphics. "
         "Raise 'Minimal detail' and see whether the match survives.", None)],
    "ai": [
        ("Generated image", "High probabilities from the detectors, usually over the whole picture.", "ai_generated"),
        ("Camera photograph", "Low probabilities everywhere.", "ai_photo"),
        ("Local AI edit", "Low overall score but one region above 90%: generative fill, object removal, face swap. The "
         "finding 'a region looks generated' points to it.", None),
        ("Where detectors fail", "Generators the networks never saw (Flux.2 in this program's test: 2 of 30 found) and "
         "anything resized, recompressed or screenshotted are often missed; heavily processed phone photos and "
         "digital art are sometimes wrongly flagged. When the detectors disagree, say so in the report.", None)],
    "trufor": [
        ("Spliced object", "High tampering probability over the insert, and the reliability map is bright there.", "trufor_splice"),
        ("Low reliability", "Red areas where the reliability map is dark (sky, blur, saturated regions) are not findings.", None),
        ("Fully generated image", "Often scores low: nothing is inconsistent inside a uniformly synthetic picture.", None)],
    "spectrum": [
        ("Camera photo, up-sampling artifact, current diffusion image", "Left: a camera photo gives a smooth cloud with at "
         "most JPEG-lattice peaks (blue). Middle: the same photo with a simulated up-sampling artifact of the kind GANs and "
         "older diffusion decoders leave - a lattice of isolated amber peaks. Right: a real SDXL image, which shows NO "
         "such peaks: for current generators a clean spectrum is the rule, so only a positive result counts.", "spectrum_compare"),
        ("Camera original at native resolution", "Green peaks at the Nyquist corners: the Bayer trace, a sign of a camera original.", None),
        ("Peaks without AI", "Resized images, halftone prints, photographed screens and fine regular textures.", None)],
    "watermark": [
        ("File straight from Stable Diffusion's reference software", "All or nearly all bits of one message match; red finding.", None),
        ("Anything else", "About half of the bits match by chance: no information.", None)],
    "geometry": [
        ("Consistent perspective", "Three edges of one facade that are parallel in reality meet in one vanishing point "
         "to within a fraction of a degree.", "geometry_facade"),
        ("Inserted object", "Edges of a pasted vehicle, sign or building converge elsewhere than the matching edges of "
         "the scene; its shadow line misses the point where the other shadow lines meet.", None),
        ("AI-generated scenes", "Generators produce plausible but not exact geometry: parallel edges of one room or "
         "street that fail to meet, shadows of neighbouring objects pointing to different light positions.", None)],
    "enhance": [("Inspecting an edge", "Zoom in on the outline of a suspicious object: halos, abrupt changes of sharpness "
                 "and leftover background pixels become visible with CLAHE or equalization.", None)],
    "gradient": [("Pasted object", "The insert shows a different texture of gradient directions and a crisp rectangular "
                  "outline against its surroundings.", "gradient_splice")],
    "pca": [("Pasted object", "In the minor components the scene fades and differences in colour statistics and "
             "compression remain: the insert stands out as a block.", "pca_splice")],
    "sweep": [("Cut-out in a bright area", "Sweeping through the highlights shows a hard outline around an object pasted "
               "onto a near-white background that looked seamless.", None)],
    "bitplane": [("Synthetic fill", "Areas painted, filled or heavily denoised are flat or patterned in the lowest plane "
                  "where a photograph shows random noise.", None)],
    "channel": [("Colour retouching", "Chroma channels (Cb, Cr, a, b) reveal splice borders and recolouring that the "
                 "luminance hides, because editors and JPEG treat chroma more coarsely.", None)],
    "histogram": [("Untouched versus Levels-adjusted", "Left a smooth histogram, right the comb left by stretching the tones.", "histogram_levels")],
    "hidden": [("Content 'erased' with transparency", "A viewer shows a hole; the file still holds the pixels.", "hidden_pixels")],
    "online:Sightengine": [("Second opinion", "Useful when the local detectors disagree or the generator is very new. "
                            "Record the date: commercial models change without notice.", None)],
    "online:OpenAI provenance check": [("Image from ChatGPT / DALL-E / gpt-image", "'detected' even after a screenshot or "
                                        "moderate recompression, thanks to the pixel watermark.", None),
                                       ("Anything else", "'not detected' - which says nothing about other generators.", None)],
}
