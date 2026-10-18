"""Fetch the neural detectors and convert them to ONNX for the app (run in .venv-convert).

    .venv-convert/Scripts/python tools/convert_models.py commfor swinv2 trufor bfree omniaid

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


def fp16_weights(src: Path, dst: Path, min_size: int = 2 ** 16) -> None:
    """Store the large float32 weights as float16, each followed by a Cast back to float32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))                        # pulls in external weight files
    casts = []
    for init in list(m.graph.initializer):
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_size:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            m.graph.initializer.remove(init)
            m.graph.initializer.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
    m.graph.node.extend(casts)
    m.graph.node.sort(key=lambda node: node.op_type != "Cast" or not node.input[0].endswith("__fp16"))   # casts first
    onnx.save(m, str(dst))
    print(f"  {len(casts)} weights stored as fp16: {dst.stat().st_size / 2**20:.0f} MB")


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
              "size": size, "policy": "squash", "output": "softmax", "fake_index": fake,
              "cpu_optimization": "basic",       # ORT's extended fusions fail on this fp16 graph on the CPU
              "mean": pre["image_mean"], "std": pre["image_std"], "license": "Apache-2.0",
              "source": "https://huggingface.co/haywoodsloan/ai-image-detector-deploy (ONNX: LPX55/detection-model-1-ONNX)"})


def bfree() -> None:
    """B-Free (Guillaro et al., CVPR 2025): DINOv2 ViT with registers, trained bias-free on 504 px crops.
    GRIP-UNINA licence: informational and nonprofit use only."""
    import hashlib
    import urllib.request

    import timm
    import yaml

    folder = SRC / "bfree"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "BFREE_dino2reg4.zip"
    if not archive.exists():
        print("  downloading BFREE_dino2reg4.zip (307 MB) ...")
        urllib.request.urlretrieve("https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip", archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    assert digest == "f3f53fa647848b16cf81c913f148a198", f"unexpected MD5 {digest}"      # published in the B-Free README
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)
    cfg = yaml.safe_load((folder / "BFREE_dino2reg4" / "config.yaml").read_text())
    print("  config:", cfg)
    assert cfg["arch"].startswith("timm_c5i504_"), cfg["arch"]
    model = timm.create_model(cfg["arch"][12:], num_classes=1, pretrained=False)
    model.set_input_size(img_size=504)
    state = torch.load(folder / "BFREE_dino2reg4" / cfg["weights_file"], map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)      # the authors' five-crop wrapper only re-orders patch embedding and cropping
    model.eval()
    norm = {"resnet": IMAGENET, "clip": {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]},
            "none": {"mean": [0, 0, 0], "std": [1, 1, 1]}, "xception": {"mean": [0.5] * 3, "std": [0.5] * 3}}[cfg["norm_type"]]
    out = MODELS / "bfree_dinov2_504.onnx"
    sample = torch.randn(1, 3, 504, 504)
    torch.onnx.export(model, sample, str(out), input_names=["image"], output_names=["logit"], opset_version=17, dynamo=False)
    parity(out, model, sample)
    register({"id": "bfree", "name": "B-Free (DINOv2, 504 px crops)", "file": out.name, "kind": "classifier", "size": 504,
              "policy": "five_crop", "output": "sigmoid", **norm,
              # measured on 245 images from current generators and 90 photographs: AUC 0.81, 10 photographs rated
              # generated, and adding it did not improve the ensemble (AUC 0.952 -> 0.944). Reported, not counted.
              "weight": 0,
              "license": "GRIP-UNINA licence (informational and nonprofit use only)", "source": "https://github.com/grip-unina/B-Free"})


def omniaid() -> None:
    """OmniAID v2 (Guo et al., ICML 2026): hybrid mixture of experts on CLIP ViT-L/14 at 336 px, trained on Mirage.
    MIT licence (Hugging Face model card). The authors' simplified inference model (reward/omniaid.py) is fetched at
    a fixed commit; the export returns the log-odds instead of the softmax so confident scores do not saturate."""
    import importlib.util
    import types
    import urllib.request

    from huggingface_hub import hf_hub_download

    commit = "40749406fbcd8893c11a160edf4a72a2d4dc7056"
    folder = SRC / "omniaid"
    folder.mkdir(parents=True, exist_ok=True)
    code = folder / "omniaid_model.py"
    if not code.exists():
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/yunncheng/OmniAID/{commit}/reward/omniaid.py", code)
    spec = importlib.util.spec_from_file_location("omniaid_model", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.loads(Path(hf_hub_download("Yunncheng/OmniAID", "config/config_omniaid_v2.json", local_dir=folder)).read_text())
    model = mod.OmniAID(types.SimpleNamespace(CLIP_path=cfg["CLIP_path"], num_experts=cfg["num_experts"],
                                              rank_per_expert=cfg["rank_per_expert"], moe_top_k=cfg["moe_top_k"],
                                              moe_router_hidden_dim=cfg["moe_router_hidden_dim"], is_hybrid=True))
    state = torch.load(hf_hub_download("Yunncheng/OmniAID", "ckpt/checkpoint_omniaid_v2.pth", local_dir=folder),
                       map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in state.get("model", state).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)     # the authors load with strict=False too
    print(f"  {len(missing)} missing, {len(unexpected)} unexpected keys:", missing[:5], unexpected[:5])
    assert not missing, "checkpoint does not cover the model"
    model.eval()

    class LogOdds(torch.nn.Module):               # model.forward up to the head, then logit(fake) - logit(real)
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            n = self.net
            routing = n.feature_extractor(images, output_hidden_states=False).pooler_output
            hidden = n.ln_pre(n.embeddings(images))
            semantic = n.gating_network(routing)
            batch = images.shape[0]
            gating = {"top_k_indices": torch.cat([semantic["top_k_indices"],
                                                  torch.full((batch, 1), n.artifact_expert_idx, dtype=torch.long)], 1),
                      "top_k_gates": torch.cat([semantic["top_k_gates"], torch.ones((batch, 1))], 1)}
            for layer in n.encoder_layers:
                hidden = layer(hidden, gating_outputs=gating)[0]
            pred = n.head(n.ln_post(hidden[:, 0, :]))
            return pred[:, 1:2] - pred[:, 0:1]

    wrapped = LogOdds(model).eval()
    sample = torch.randn(1, 3, 336, 336)
    with torch.no_grad():                         # the wrapper must reproduce the authors' probability exactly
        assert torch.allclose(torch.sigmoid(wrapped(sample))[:, 0], model(sample), atol=1e-5)
    # ~600 M parameters exceed the 2 GB limit of a single ONNX file: the exact fp32 graph goes to _src (with its
    # weights in separate files) as the reference; the app gets one file whose weights are stored as fp16 and cast
    # back to fp32 when the session loads, so the arithmetic stays fp32
    exact = folder / "fp32" / "omniaid_v2_clip336.onnx"
    exact.parent.mkdir(exist_ok=True)
    torch.onnx.export(wrapped, sample, str(exact), input_names=["image"], output_names=["logit"], opset_version=17,
                      dynamo=False)
    parity(exact, wrapped, sample)
    out = MODELS / "omniaid_v2_clip336.onnx"
    fp16_weights(exact, out)
    parity(out, wrapped, sample)
    register({"id": "omniaid", "name": "OmniAID v2 (CLIP ViT-L/336)", "file": out.name, "kind": "classifier",
              "size": 336, "policy": "whole", "output": "sigmoid",
              "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
              "weight": 0, "license": "MIT", "source": "https://github.com/yunncheng/OmniAID (weights: huggingface.co/Yunncheng/OmniAID)"})


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
        {"commfor": commfor, "swinv2": swinv2, "trufor": trufor, "bfree": bfree, "omniaid": omniaid}[name]()
