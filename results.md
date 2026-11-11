| Model | Device | Size (MB) | Accuracy | Macro-F1 | Latency mean (ms) | Latency p95 (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | CPU | 42.77 | 1.0000 | 1.0000 | 61.92 | 81.74 |
| PyTorch FP32 | CUDA | 42.77 | 1.0000 | 1.0000 | 3.27 | 3.58 |
| PyTorch INT8 (dynamic, Linear only) | CPU | 42.73 | 1.0000 | 1.0000 | 68.15 | 85.90 |
| ONNX Runtime FP32 | CPU | 42.69 | 1.0000 | 1.0000 | 38.14 | 51.80 |
| ONNX Runtime FP32 | CUDA | 42.69 | 1.0000 | 1.0000 | 3.43 | 3.53 |
| ONNX Runtime INT8 dynamic | CPU | 10.73 | 1.0000 | 1.0000 | 54.67 | 71.27 |
| ONNX Runtime INT8 dynamic | CUDA | 10.73 | 1.0000 | 1.0000 | 54.77 | 62.62 |
| ONNX Runtime INT8 static (QDQ) | CPU | 10.79 | 0.9708 | 0.9596 | 23.91 | 26.01 |
| ONNX Runtime INT8 static (QDQ) | CUDA | 10.79 | 1.0000 | 1.0000 | 3.95 | 4.03 |
