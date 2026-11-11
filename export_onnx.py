"""Export to ONNX and create two INT8 variants (dynamic + static/QDQ)."""
import argparse
import json
import os
import random

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from PIL import Image

from common import eval_transform, split_samples
from train import build_model


class CalibReader(CalibrationDataReader):
    def __init__(self, paths, input_name):
        tf = eval_transform()
        self.data = iter([{input_name: tf(Image.open(p).convert("RGB")).unsqueeze(0).numpy()} for p in paths])

    def get_next(self):
        return next(self.data, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train/train")
    p.add_argument("--out", default="models")
    p.add_argument("--calib-images", type=int, default=200)
    args = p.parse_args()

    classes = json.load(open(os.path.join(args.out, "classes.json")))
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pth"), map_location="cpu"))
    model.eval()

    fp32 = os.path.join(args.out, "model_fp32.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), fp32, export_params=True, opset_version=13,
                      do_constant_folding=True, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(fp32))

    # sanity check: PyTorch vs ONNX Runtime outputs
    x = torch.randn(2, 3, 224, 224)
    ref = model(x).detach().numpy()
    out = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    print("max |torch - onnx| =", float(np.abs(ref - out).max()))

    # recommended pre-processing before quantization (shape inference + optimisation)
    pre = os.path.join(args.out, "model_fp32_pre.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(fp32, pre)
    except Exception as e:
        print("quant_pre_process skipped:", e)
        pre = fp32

    dyn = os.path.join(args.out, "model_int8_dynamic.onnx")
    quantize_dynamic(pre, dyn, weight_type=QuantType.QUInt8)

    train, _, _ = split_samples(args.data)
    random.seed(0)
    calib = [p for p, _ in random.sample(train, min(args.calib_images, len(train)))]
    stat = os.path.join(args.out, "model_int8_static.onnx")
    quantize_static(pre, stat, CalibReader(calib, "input"), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)

    for path in (fp32, dyn, stat):
        print(f"{os.path.basename(path)}: {os.path.getsize(path)/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
