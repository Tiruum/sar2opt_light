"""Export the WaveNeXt generator to ONNX (fp32 + int8 dynamic) for Hugging Face.

Loads a Lightning ``.ckpt`` via :func:`inference.load_generator`, traces the
generator to ONNX (dynamic batch axis), verifies the graph, checks numerical
parity against PyTorch via onnxruntime, then writes an int8 dynamically
quantized variant and re-checks parity.

Run from repo root::

    python -m src.models.wavenext.export_onnx \
        --ckpt checkpoints/llwt_v45_base/llwt-v0.4.6-base/epoch=199-psnr=18.5361.ckpt \
        --out  output/hf_export
"""
import argparse
import os

# Mirror inference.py offline-HF guards (backbone is cached from training).
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if os.environ.get('HF_HUB_OFFLINE') == '1':
    import transformers.models.auto.auto_factory as _af
    _af.repo_exists = lambda *a, **k: True

import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.wavenext.inference import load_generator

CONFIG = "./src/models/wavenext/config.yaml"


def _mb(path):
    return os.path.getsize(path) / 1e6


def _ort_run(path, sar_np):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    return sess.run(None, {iname: sar_np})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="output/hf_export")
    ap.add_argument("--opset", type=int, default=17)
    # fp32 is the shippable default. fp16 (onnxconverter-common) and int8-dynamic
    # are opt-in: fp16 tooling is broken on current onnx, and int8-dynamic destroys
    # this conv model's output (parity ~1.56 over a [-1,1] range). Left as flags for
    # future static-quant / GPU-native experiments.
    ap.add_argument("--fp16", action="store_true", help="also emit fp16 (experimental, may fail)")
    ap.add_argument("--int8", action="store_true", help="also emit int8-dynamic (NOT usable for this CNN)")
    args = ap.parse_args()

    cfg = OmegaConf.load(CONFIG)
    # ONNX export on CPU in fp32 (bf16 is a runtime detail, not the graph).
    g = load_generator(args.ckpt, cfg, "cpu").float().eval()
    os.makedirs(args.out, exist_ok=True)

    fp32_path = os.path.join(args.out, "model.onnx")
    dummy = torch.randn(1, 1, cfg.data.image_size, cfg.data.image_size)

    torch.onnx.export(
        g, dummy, fp32_path,
        input_names=["sar"], output_names=["optical"],
        dynamic_axes={"sar": {0: "batch"}, "optical": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: robust for the HF backbone, no onnxscript dep
    )
    import onnx
    onnx.checker.check_model(onnx.load(fp32_path))
    print(f"[onnx] exported fp32 -> {fp32_path} ({_mb(fp32_path):.1f} MB)")

    # ---- numerical parity (fp32) ----
    with torch.no_grad():
        torch_out = g(dummy).numpy()
    onnx_out = _ort_run(fp32_path, dummy.numpy())
    err = float(np.abs(torch_out - onnx_out).max())
    print(f"[parity] fp32 max|torch - onnx| = {err:.3e}  ({'OK' if err < 1e-3 else 'CHECK'})")

    # ---- fp16 (half-size, near-lossless, GPU-friendly) ----
    if args.fp16:
        try:
            from onnxconverter_common import float16
            fp16_path = os.path.join(args.out, "model_fp16.onnx")
            # disable_shape_infer=True works around an onnxconverter-common bug on
            # newer onnx ("'list' object has no attribute 'input'" during shape infer).
            m16 = float16.convert_float_to_float16(
                onnx.load(fp32_path), keep_io_types=True, disable_shape_infer=True)
            onnx.save(m16, fp16_path)
            f16_out = _ort_run(fp16_path, dummy.numpy())
            f16_err = float(np.abs(torch_out - f16_out).max())
            print(f"[onnx] exported fp16 -> {fp16_path} ({_mb(fp16_path):.1f} MB)")
            print(f"[parity] fp16 max|torch - onnx_fp16| = {f16_err:.3e}  ({'OK' if f16_err < 1e-2 else 'CHECK'})")
        except Exception as e:
            print(f"[onnx] fp16 export FAILED ({type(e).__name__}: {e}) — skipping fp16")

    # ---- int8 dynamic quantization ----
    if args.int8:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8_path = os.path.join(args.out, "model_int8.onnx")
            quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
            q_out = _ort_run(int8_path, dummy.numpy())
            q_err = float(np.abs(torch_out - q_out).max())
            print(f"[onnx] exported int8 -> {int8_path} ({_mb(int8_path):.1f} MB)")
            print(f"[parity] int8 max|torch - onnx_int8| = {q_err:.3e}  (quantization error, expected larger)")
        except Exception as e:
            print(f"[onnx] int8 export FAILED ({type(e).__name__}: {e}) — skipping int8")

    print("[onnx] done")


if __name__ == "__main__":
    main()
