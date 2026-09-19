"""Fetch the neural detectors and convert them to ONNX for the app (run in .venv-convert).

    .venv-convert/Scripts/python tools/convert_models.py commfor swinv2 trufor

Writes models/*.onnx and merges the matching entries into models/models.json.
PyTorch is needed only here; the app itself runs the ONNX files with ONNX Runtime.
"""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SRC = MODELS / "_src"
IMAGENET = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def register(entry: dict) -> None:
    path = MODELS / "models.json"
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entries = [e for e in entries if e["id"] != entry["id"]] + [entry]
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print("registered", entry["id"])


def parity(onnx_path: Path, model: torch.nn.Module, sample: torch.Tensor, pick=lambda out: out) -> None:
    """The exported graph must reproduce PyTorch."""
    with torch.no_grad():
        expected = pick(model(sample))
    expected = expected if isinstance(expected, (tuple, list)) else [expected]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: sample.numpy()})
    for name, e, g in zip([o.name for o in sess.get_outputs()], expected, got):
        diff = float(np.abs(e.numpy() - g).max())
        print(f"  parity {name}: max |torch - onnx| = {diff:.2e}")
        assert diff < 1e-2, "ONNX export does not match PyTorch"


def commfor() -> None:
    """Community Forensics (Park & Owens, CVPR 2025), ViT-S/16 at 384 px, MIT licence."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights = load_file(hf_hub_download("OwensLab/commfor-model-384", "model.safetensors"))
    model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    model.load_state_dict({k.removeprefix("vit."): v for k, v in weights.items()}, strict=True)
    model.eval()
    out = MODELS / "commfor_vits16_384.onnx"
    sample = torch.randn(1, 3, 384, 384)
    torch.onnx.export(model, sample, str(out), input_names=["image"], output_names=["logit"], opset_version=17, dynamo=False)
    parity(out, model, sample)
    register({"id": "commfor", "name": "Community Forensics ViT-S/384", "file": out.name, "kind": "classifier",
              "size": 384, "resize": 440, "policy": "resize_crop", "output": "sigmoid", **IMAGENET,
              "license": "MIT", "source": "https://huggingface.co/OwensLab/commfor-model-384"})


def swinv2() -> None:
    """Community SwinV2 detector (haywoodsloan/ai-image-detector-deploy, Apache-2.0), published as ONNX."""
    from huggingface_hub import hf_hub_download

    repo = "LPX55/detection-model-1-ONNX"
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    pre = json.loads(Path(hf_hub_download(repo, "preprocessor_config.json")).read_text())
    print("  labels:", cfg.get("id2label"), "| preprocessor:", {k: pre.get(k) for k in ("size", "image_mean", "image_std", "resample", "do_rescale")})
    out = MODELS / "swinv2_detector_fp16.onnx"
    shutil.copy2(hf_hub_download(repo, "onnx/model_fp16.onnx"), out)
    labels = {int(k): v.lower() for k, v in cfg["id2label"].items()}
    fake = next(i for i, name in labels.items() if any(w in name for w in ("artificial", "fake", "ai", "generated")))
    size = pre["size"]["height"] if isinstance(pre["size"], dict) else int(pre["size"])
    register({"id": "swinv2", "name": "SwinV2 AI-image detector", "file": out.name, "kind": "classifier",
              "size": size, "policy": "squash", "output": "softmax", "fake_index": fake, "weight": 0.5,
              "cpu_optimization": "basic",       # ORT's extended fusions fail on this fp16 graph on the CPU
              "mean": pre["image_mean"], "std": pre["image_std"], "license": "Apache-2.0",
              "source": "https://huggingface.co/haywoodsloan/ai-image-detector-deploy (ONNX: LPX55/detection-model-1-ONNX)"})


def trufor() -> None:
    """TruFor (Guillaro et al., CVPR 2023). Research / non-commercial licence, see models/_src/TruFor."""
    import urllib.request

    code = SRC / "TruFor" / "TruFor_train_test"
    weights = code / "pretrained_models" / "trufor.pth.tar"
    if not weights.exists():
        archive = SRC / "TruFor_weights.zip"
        if not archive.exists():
            print("  downloading TruFor_weights.zip ...")
            urllib.request.urlretrieve("https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip", archive)
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith("trufor.pth.tar"))
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_bytes(z.read(member))
    sys.path.insert(0, str(code))
    os.chdir(code)                                 # TruFor resolves its yaml configs relative to its own folder
    from lib.config import config, update_config
    from lib.models.cmx.builder_np_conf import EncoderDecoder      # lib.utils.get_model drags in matplotlib

    class Args:
        experiment, opts = "trufor_ph3", ["TEST.MODEL_FILE", str(weights)]
    update_config(config, Args)
    model = EncoderDecoder(cfg=config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()

    class Wrapper(torch.nn.Module):               # fixed output order, no Noiseprint++ dump
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, rgb):
            pred, conf, det, _ = self.net(rgb)
            return pred, conf, det

    size = 1024
    wrapped = Wrapper(model).eval()
    sample = torch.rand(1, 3, size, size)
    out = MODELS / "trufor_1024.onnx"
    torch.onnx.export(wrapped, sample, str(out), input_names=["rgb"], output_names=["pred", "conf", "det"],
                      opset_version=17, dynamo=False)
    parity(out, wrapped, sample)
    register({"id": "trufor", "name": "TruFor", "file": out.name, "kind": "localizer", "size": size,
              "license": "GRIP-UNINA research licence (non-commercial)", "source": "https://github.com/grip-unina/TruFor"})


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    for name in sys.argv[1:] or ["commfor", "swinv2", "trufor"]:
        print(f"== {name}")
        {"commfor": commfor, "swinv2": swinv2, "trufor": trufor}[name]()
