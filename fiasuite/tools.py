"""Tool registry: what the dashboard lists, which parameters each tool exposes,
and which tools take part in the automatic scan."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .analyses import ai, clone, compression, interpolation, jpegstruct, metadata, noise, spectral, visual


@dataclass
class Param:
    key: str
    label: str
    default: object
    lo: float = 0
    hi: float = 100
    step: float = 1
    choices: list[str] | None = None        # set -> combo box; bool default -> check box


@dataclass
class Tool:
    id: str
    title: str
    group: str
    fn: Callable
    help: str
    params: list[Param] = field(default_factory=list)
    auto: bool = False                       # part of the automatic scan / batch triage
    axis: str | None = None                  # verdict axis fed by this tool: "provenance" | "edit" | "ai"
    weight: float = 1.0                      # trust in this tool's score when forming the verdict
    manual: bool = False                     # runs only on an explicit click (uploads the image)


ENH = Param("enhancement", "Enhancement", "Auto contrast", choices=visual.ENHANCEMENTS)

TOOLS: list[Tool] = [
    # ---------------------------------------------------------------- file & provenance
    Tool("metadata", "Metadata (ExifTool)", "File & provenance", metadata.run_metadata,
         "Full ExifTool dump with consistency rules: AI-generator tags, editing software, XMP history and document "
         "ancestors, date conflicts, missing MakerNotes, size mismatches.", auto=True, axis="provenance"),
    Tool("c2pa", "Content Credentials (C2PA)", "File & provenance", metadata.run_c2pa,
         "Reads and cryptographically validates the C2PA manifest: who signed the file, with which tool, and whether "
         "generative AI was declared.", auto=True, axis="provenance"),
    Tool("jpeg", "JPEG structure", "File & provenance", jpegstruct.run,
         "Marker map, quantization tables with quality estimate, subsampling, application segments and trailing data.",
         auto=True, axis="provenance", weight=0.6),
    Tool("thumbnail", "Embedded thumbnail", "File & provenance", metadata.run_thumbnail,
         "Compares the embedded preview with the main image. Editors often forget to update it, so it can show the "
         "picture as it was before editing.", auto=True, axis="edit"),
    Tool("hashes", "Hashes", "File & provenance", metadata.run_hashes,
         "MD5 / SHA-1 / SHA-256 of the file for chain of custody, and a perceptual hash for similarity search."),
    Tool("strings", "String extraction", "File & provenance", metadata.run_strings,
         "Printable strings hidden in the binary data: software names, paths, prompts, leftovers of earlier saves.",
         [Param("min_length", "Minimum length", 6, 4, 32)]),

    # ---------------------------------------------------------------- compression
    Tool("ela", "Error level analysis", "Compression", compression.ela,
         "Re-saves the image as JPEG and shows what changed. Regions edited after the last save respond differently. "
         "Interpret by comparing similar surfaces; edges and textures are always brighter.",
         [Param("quality", "JPEG quality", 90, 50, 100), Param("scale", "Error scale", 20, 1, 100)]),
    Tool("ghosts", "JPEG ghosts", "Compression", compression.ghosts,
         "Farid's JPEG-ghost method: content that was once saved at a lower JPEG quality reveals itself as a minimum "
         "of the recompression error at that quality. Misaligned grids found by the block-grid tool are tested too.",
         [Param("qmin", "Lowest quality", 40, 10, 90), Param("qmax", "Highest quality", 96, 50, 100),
          Param("step", "Quality step", 2, 1, 8), Param("block", "Averaging block", 16, 8, 64, 8),
          Param("all_shifts", "Test all 64 grid alignments (slow)", False)], auto=True, axis="edit"),
    Tool("dct", "DCT histograms / double compression", "Compression", compression.dct_analysis,
         "Histograms of the first DCT coefficients. Periodic gaps or peaks mean double quantization: the image was "
         "decoded and re-saved. Also reveals earlier JPEG compression inside PNG or other lossless files.",
         auto=True, axis="provenance", weight=0.5),
    Tool("grid", "JPEG block grid", "Compression", compression.block_grid,
         "Locates the 8x8 JPEG block grid. A shifted grid means cropping after compression; a region with its own "
         "grid alignment was pasted from another JPEG.",
         [Param("window", "Window size", 64, 32, 256, 8)], auto=True, axis="edit"),

    # ---------------------------------------------------------------- noise & interpolation
    Tool("noise", "Noise residual", "Noise & sensor", noise.residual,
         "Isolates the image noise with a median filter. Missing, coarser or differently coloured grain betrays "
         "airbrushing, inpainting and pasted content.",
         [Param("amplitude", "Noise amplitude", 25, 1, 100), Param("equalize", "Equalize histogram", False)]),
    Tool("noisemap", "Noise-level consistency", "Noise & sensor", noise.noise_map,
         "Estimates the noise level per block and compares blocks of equal brightness. A photo from one sensor has "
         "one noise characteristic; splices and local smoothing do not.",
         [Param("block", "Block size", 32, 16, 128, 16)], auto=True, axis="edit", weight=0.7),
    Tool("cfa", "CFA / demosaicing traces", "Noise & sensor", interpolation.cfa,
         "A camera sensor records one colour per pixel and interpolates the rest, leaving a periodic trace. Regions "
         "without that trace did not come through the camera pipeline.",
         [Param("block", "Block size", 32, 16, 128, 16)], auto=True, axis="edit", weight=0.7),
    Tool("resampling", "Resampling detection", "Noise & sensor", interpolation.resampling,
         "Scaling or rotating image content leaves periodic correlations between pixels (Popescu & Farid). Finds "
         "inserted objects that were resized to fit.",
         [Param("window", "Window size", 128, 64, 512, 64)], auto=True, axis="edit", weight=0.8),

    # ---------------------------------------------------------------- clone
    Tool("clone_kp", "Clone detection (keypoints)", "Copy-move", clone.keypoints,
         "Matches distinctive points against the rest of the same image. Finds duplicated objects even when scaled "
         "or rotated. Needs texture.",
         [Param("min_distance", "Minimal distance", 40, 8, 400), Param("ratio", "Match strictness", 0.55, 0.3, 0.8, 0.05),
          Param("min_matches", "Minimal matches", 8, 4, 50)], auto=True, axis="edit"),
    Tool("clone_blocks", "Clone detection (blocks)", "Copy-move", clone.blocks,
         "Compares every small block with every other. Finds copied smooth areas (sky, skin, walls) that carry no "
         "keypoints, for copies that were moved but not scaled.",
         [Param("block", "Block size", 16, 8, 32, 4), Param("similarity", "Minimal similarity", 1.5, 0.2, 5, 0.1),
          Param("min_detail", "Minimal detail", 2.0, 0, 20, 0.5), Param("min_cluster", "Minimal cluster size", 300, 20, 2000, 10),
          Param("max_size", "Maximal image size", 1600, 600, 3000, 100)], auto=True, axis="edit", weight=0.8),

    # ---------------------------------------------------------------- AI
    Tool("ai", "AI-generation detectors", "AI & deep learning", ai.detect,
         "Ensemble of neural detectors trained to separate camera images from generator output. Each map shows which "
         "regions look generated, to expose local AI edits.",
         [Param("max_tiles", "Regions analysed", 24, 1, 96)], auto=True, axis="ai"),
    Tool("trufor", "Forgery localisation (TruFor)", "AI & deep learning", ai.localize,
         "TruFor (Guillaro et al., CVPR 2023) fuses the image with a learned camera-noise fingerprint and marks "
         "manipulated regions, with a reliability map. Non-commercial research licence.",
         [Param("max_side", "Maximal image size", 1600, 512, 4096, 64)], auto=True, axis="edit", weight=1.2),
    Tool("spectrum", "Spectral fingerprint", "AI & deep learning", spectral.spectrum,
         "Averaged Fourier spectrum of the noise residual. Generator up-sampling leaves a regular lattice of peaks; "
         "also reports how the spectrum falls off towards fine detail.",
         [Param("tile", "Tile size", 512, 128, 1024, 128),
          Param("residual", "Noise residual", "Wiener (adaptive)", choices=spectral.RESIDUALS),
          Param("channel", "Channel", "Luminance", choices=spectral.CHANNELS),
          Param("overlap", "Overlapping tiles", False)], auto=True, axis="ai", weight=0.7),

    # ---------------------------------------------------------------- visual inspection
    Tool("enhance", "Magnifier enhancement", "Visual inspection", visual.enhance,
         "Contrast enhancement for zoomed inspection of edges, halos and texture.",
         [Param("mode", "Enhancement", "CLAHE", choices=visual.ENHANCEMENTS[1:])]),
    Tool("gradient", "Luminance gradient", "Visual inspection", visual.luminance_gradient,
         "Colour-codes the direction of brightness change. Inconsistent lighting and the soft edges of composited "
         "objects stand out."),
    Tool("pca", "Principal components", "Visual inspection", visual.pca,
         "Projects the colours onto their principal axes. The minor components hide the scene and expose seams, "
         "cloning and compression differences.",
         [Param("component", "Component", 2, 1, 3), Param("invert", "Invert", False), ENH]),
    Tool("sweep", "Level sweep", "Visual inspection", visual.level_sweep,
         "Stretches a narrow brightness band to full contrast. Sweep it to reveal cut-out outlines in highlights "
         "and shadows.",
         [Param("sweep", "Sweep", 128, 0, 255), Param("width", "Width", 32, 2, 128)]),
    Tool("bitplane", "Bit planes", "Visual inspection", visual.bit_plane,
         "Shows a single bit of every pixel. Low planes should be uniform noise.",
         [Param("plane", "Bit plane", 0, 0, 7), Param("channel", "Channel", "All", choices=["All", "Red", "Green", "Blue"])]),
    Tool("channel", "Colour channels", "Visual inspection", visual.channel,
         "Any single channel of RGB, HSV, Lab or YCbCr.",
         [Param("which", "Channel", "YCbCr: Cb", choices=visual.CHANNELS), ENH]),
]

BY_ID = {t.id: t for t in TOOLS}


def defaults(tool: Tool) -> dict:
    return {p.key: p.default for p in tool.params}
