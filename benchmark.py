"""Compare quality (accuracy, macro-F1), size and latency of all model variants on the same val set."""
import argparse
import copy
import json
import os
import tempfile
import time

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score

from common import make_loaders
from train import build_model


def latency(fn, sync=None, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - s) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def quality(predict, loader):
    preds, labels = [], []
    for x, y in loader:
        preds.append(predict(x))
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return float((preds == labels).mean()), float(f1_score(labels, preds, average="macro"))


def torch_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(model.state_dict(), f.name)
    size = os.path.getsize(f.name) / 1024**2
    os.remove(f.name)
    return size


def ort_session(path, device):
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return None
        s = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return s if s.get_providers()[0] == "CUDAExecutionProvider" else None
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train/train")
    p.add_argument("--models", default="models")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    _, val_loader, classes, _ = make_loaders(args.data, workers=2)
    sample = torch.randn(1, 3, 224, 224)
    rows = []

    def add(name, device, size, acc, f1, lat):
        rows.append(dict(model=name, device=device, size_mb=size, accuracy=acc, macro_f1=f1,
                         mean_ms=lat[0], p95_ms=lat[1]))
        print(rows[-1])

    state = torch.load(os.path.join(args.models, "best_model.pth"), map_location="cpu")
    fp32 = build_model(len(classes), pretrained=False)
    fp32.load_state_dict(state)
    fp32.eval()

    # --- PyTorch FP32 (CPU, GPU) and PyTorch dynamic INT8 (CPU only)
    with torch.no_grad():
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            m = copy.deepcopy(fp32).to(device)
            sync = torch.cuda.synchronize if device == "cuda" else None
            acc, f1 = quality(lambda x: m(x.to(device)).argmax(1).cpu().numpy(), val_loader)
            lat = latency(lambda: m(sample.to(device)), sync, iters=args.iters)
            add("PyTorch FP32", device, torch_size_mb(fp32), acc, f1, lat)

        # note: dynamic quantization in PyTorch only covers nn.Linear (here: the final fc layer)
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8)
        acc, f1 = quality(lambda x: q(x).argmax(1).numpy(), val_loader)
        add("PyTorch INT8 (dynamic, Linear only)", "cpu", torch_size_mb(q), acc, f1,
            latency(lambda: q(sample), iters=args.iters))

    # --- ONNX Runtime variants
    variants = [("ONNX Runtime FP32", "model_fp32.onnx"),
                ("ONNX Runtime INT8 dynamic", "model_int8_dynamic.onnx"),
                ("ONNX Runtime INT8 static (QDQ)", "model_int8_static.onnx")]
    x_np = sample.numpy()
    for name, fname in variants:
        path = os.path.join(args.models, fname)
        for device in ["cpu", "cuda"]:
            s = ort_session(path, device)
            if s is None:
                print(f"[skip] {name} on {device}: CUDAExecutionProvider not active")
                continue
            inp = s.get_inputs()[0].name
            acc, f1 = quality(lambda x: s.run(None, {inp: x.numpy()})[0].argmax(1), val_loader)
            lat = latency(lambda: s.run(None, {inp: x_np}), iters=args.iters)
            add(name, device, os.path.getsize(path) / 1024**2, acc, f1, lat)

    json.dump(rows, open("results.json", "w"), indent=2)
    with open("results.md", "w") as f:
        f.write("| Model | Device | Size (MB) | Accuracy | Macro-F1 | Latency mean (ms) | Latency p95 (ms) |\n")
        f.write("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['device'].upper()} | {r['size_mb']:.2f} | {r['accuracy']:.4f} | "
                    f"{r['macro_f1']:.4f} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} |\n")
    print("\nSaved results.md and results.json")


if __name__ == "__main__":
    main()
