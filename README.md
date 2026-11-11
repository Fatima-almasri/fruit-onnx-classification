# Fruit Classification with ResNet18, ONNX Runtime and Quantization

Image classification of 33 fruit classes ([Fruit Recognition](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition))
with PyTorch. The trained model is exported to ONNX, quantized to INT8 (dynamic and static), and compared
with the original PyTorch model in terms of quality, size and latency on CPU and GPU.
Inference runs inside Docker.

## Project structure
```
common.py               data split, transforms, loaders
train.py                training (ResNet18 fine-tuning)
export_onnx.py          ONNX export + dynamic / static INT8 quantization
benchmark.py            accuracy / macro-F1 / size / latency for every variant (CPU + GPU)
infer.py                single-image inference with ONNX Runtime
Dockerfile              inference image (contains the model, no volume needed)
requirements.txt        training / export / benchmark dependencies
requirements-infer.txt  minimal inference dependencies (used in Docker)
models/                 classes.json, best_model.pth, model_fp32.onnx,
                        model_int8_dynamic.onnx, model_int8_static.onnx
samples/sample.jpg      example image used by the Docker default command
results.md / .json      benchmark output
```

## 1. Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
For the GPU benchmark replace the CPU package:
`pip uninstall -y onnxruntime && pip install onnxruntime-gpu`.

## 2. Download the dataset
Create a Kaggle API token (Kaggle → Settings → API) and place it in `~/.kaggle/kaggle.json`:
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d sshikamaru/fruit-recognition
unzip -q fruit-recognition.zip -d data/
```
Expected layout: `data/train/train/<class_name>/*.jpg`. The `test` folder has no labels, so all
metrics are computed on a validation split of `train`.

## 3. Train
```bash
python train.py --data data/train/train --epochs 10 --batch-size 32 --lr 1e-4
```
Produces `models/best_model.pth`, `models/classes.json` and `models/confusion_matrix.csv`.

## 4. Export to ONNX and quantize
```bash
python export_onnx.py
```
Produces `models/model_fp32.onnx`, `models/model_int8_dynamic.onnx` and `models/model_int8_static.onnx`.

## 5. Benchmark
```bash
python benchmark.py        # writes results.md and results.json
```

## 6. Inference in Docker
The image already contains the quantized model and a sample image, so no volume mount is required:
```bash
docker build -t fruit-classifier .
docker run --rm fruit-classifier
```
**Actual output** (Docker on Windows, CPU):
```text
Model: models/model_int8_dynamic.onnx | provider: CPUExecutionProvider
Apple Braeburn: 0.9972
Pomegranate: 0.0007
Strawberry: 0.0003
```
The sample image is classified as **Apple Braeburn** (confidence 99.72%) using the dynamically
quantized ONNX model on ONNX Runtime.

To classify your own image, mount its folder and pass the path
(`--model` selects another ONNX file, `--top` the number of classes shown):
```bash
# Linux / macOS
docker run --rm -v "$PWD/my_images:/images" fruit-classifier --image /images/apple.jpg
# Windows cmd
docker run --rm -v "%cd%\my_images:/images" fruit-classifier --image /images/apple.jpg
```

## Results
All variants are evaluated on the same validation split. CPU and GPU numbers were measured on Google Colab (T4 GPU).

| Model | Device | Size (MB) | Accuracy | Macro-F1 | Latency mean (ms) | Latency p95 (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | CPU | 42.77 | 1.0000 | 1.0000 | 61.92 | 81.74 |
| PyTorch FP32 | GPU | 42.77 | 1.0000 | 1.0000 | 3.27 | 3.58 |
| PyTorch INT8 (dynamic, Linear only) | CPU | 42.73 | 1.0000 | 1.0000 | 68.15 | 85.90 |
| ONNX Runtime FP32 | CPU | 42.69 | 1.0000 | 1.0000 | 38.14 | 51.80 |
| ONNX Runtime FP32 | GPU | 42.69 | 1.0000 | 1.0000 | 3.43 | 3.53 |
| ONNX Runtime INT8 dynamic | CPU | 10.73 | 1.0000 | 1.0000 | 54.67 | 71.27 |
| ONNX Runtime INT8 dynamic | GPU | 10.73 | 1.0000 | 1.0000 | 54.77 | 62.62 |
| ONNX Runtime INT8 static (QDQ) | CPU | 10.79 | 0.9708 | 0.9596 | 23.91 | 26.01 |
| ONNX Runtime INT8 static (QDQ) | GPU | 10.79 | 1.0000 | 1.0000 | 3.95 | 4.03 |

**Measurement protocol:** batch size 1, input 224×224 (random data), 10 warm-up runs and 100 timed runs
with `time.perf_counter()` (plus `torch.cuda.synchronize()` for PyTorch on GPU). A GPU row is only
reported if `CUDAExecutionProvider` is the active provider, because ONNX Runtime otherwise falls back
to CPU silently. Colab CPUs are shared, so CPU latencies are noisy; compare them as trends, not exact values.

### Discussion
- **Size:** both INT8 variants are about 75% smaller than FP32 (≈42.7 MB → ≈10.7 MB). The dynamic model is marginally
  smaller than the static one (10.73 vs 10.79 MB) because static quantization adds Q/DQ nodes and scale parameters.
  PyTorch dynamic quantization only quantizes `nn.Linear` layers (here just the final `fc`), so the size is almost unchanged.
- **Accuracy:** every variant reaches 100% except static INT8 on CPU (97.08% accuracy, 0.9596 macro-F1), the only
  variant where activations are really quantized to INT8 with calibration. This is still far above the 0.85 requirement.
- **CPU latency:** ONNX Runtime FP32 (38.1 ms) is faster than PyTorch FP32 (61.9 ms). Static INT8 is the fastest
  (23.9 ms, about 1.6× faster than ONNX FP32 and 2.6× faster than PyTorch). Dynamic INT8 is slower than FP32
  (54.7 ms): dynamic quantization of convolutions adds runtime quantization overhead and does not always speed up CNNs on CPU.
- **GPU latency:** FP32 PyTorch and ONNX Runtime are both about 3.3–3.4 ms (≈10–18× faster than CPU). Dynamic INT8 on
  "GPU" takes 54.8 ms, almost identical to its CPU time, which suggests its INT8 operators have no CUDA kernels and
  run on the CPU (with extra memory copies). Static INT8 on GPU matches FP32 speed and accuracy (100%) – most likely because the
  CUDA provider executes the Q/DQ pattern in floating point, so INT8 brings no GPU speed-up. This also explains why the same
  static model scores 97.08% on CPU (real INT8 kernels) but 100% on GPU. We did not verify this with profiling.
- **Conclusion:** static INT8 is the best choice on CPU (≈75% smaller and fastest, for a small accuracy drop); FP32 is the best on GPU;
  dynamic INT8 is only useful when model size matters and CPU latency does not.

## Design decisions
- **ResNet18, pretrained on ImageNet:** small (≈45 MB) and fast; residual connections make fine-tuning stable, and transfer
  learning reaches high accuracy in few epochs for 33 classes. It consists of Conv/BN/ReLU/Linear operators, which ONNX and INT8
  quantization support well.
- **CrossEntropyLoss:** standard loss for single-label multi-class classification (softmax + negative log-likelihood).
- **Adam (lr 1e-4) + cosine schedule:** adaptive steps converge fast when fine-tuning. In a preliminary run with lr 1e-3 and a
  random split, validation accuracy fluctuated strongly (0.9597 at epoch 4, 0.6894 at epoch 7), so the learning rate was lowered
  and cosine decay added to stabilise the last epochs.
- **Augmentation:** random resized crop, horizontal flip and colour jitter improve robustness.
- **Metrics:** *accuracy* (task requirement ≥ 0.85) and *macro-F1*, which weights all 33 classes equally and exposes weak classes.
  A confusion matrix is saved for error analysis. Latency and size are reported for the performance comparison.
- **Validation split:** a random split would likely put near-duplicate images into both train and validation, because images of the same
  fruit look very alike (probably consecutive frames of one object; not verified for every class). Each class is therefore split into
  contiguous blocks (last 20% by filename), dropping 10 images between train and validation.
- **Quantization:** dynamic (INT8 weights, activations quantized at runtime, no data needed) and static QDQ (calibrated on 200 training
  images, per-channel weights) are both evaluated, because their speed/accuracy trade-offs differ for CNNs.

## Limitations
- Validation accuracy is 100% for most variants, which suggests that the validation split is still very similar to the training data
  (the contiguous split reduces, but cannot rule out, near-duplicates). The Kaggle `test` folder is unlabeled, so no fully independent
  evaluation was possible. Treat the accuracy as an upper bound for real-world performance.
- Latency was measured with random input at batch size 1 on shared Colab hardware.
- The GPU explanations for INT8 behaviour are plausible hypotheses, not profiled results.

## Author
Fatima Almasri