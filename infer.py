"""Single-image inference with ONNX Runtime (used inside Docker)."""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path, size=224):
    # bilinear resize == torchvision.transforms.Resize on PIL images (used in training)
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/model_int8_dynamic.onnx")
    p.add_argument("--classes", default="models/classes.json")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    classes = json.load(open(args.classes)) if os.path.exists(args.classes) else None

    logits = session.run(None, {session.get_inputs()[0].name: preprocess(args.image)})[0][0]
    probs = softmax(logits)
    print(f"Model: {args.model} | provider: {session.get_providers()[0]}")
    for i in np.argsort(probs)[::-1][: args.top]:
        print(f"{classes[i] if classes else i}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
