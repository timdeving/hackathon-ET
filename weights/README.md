# Weights

The submission loads the exported detector from this folder with plain PyTorch. Nothing is
downloaded at run time.

| File | Model | Input (h × w) | Precision | Size | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `yolo26m_736x1280_fp16.torchscript` | YOLO26m (COCO), exported with Ultralytics 8.4.163; NMS-free output (300 rows) | 736 × 1280 | FP16, CUDA | 41.5 MB | `24f6b6983115f1b89e78fe723450a44f35d91c307b0112838736edf33a7e6714` |

## How a file is made

On the GPU PC, in the export environment (`requirements-export.txt`):

```bash
python -m tools.export_detector --model yolo26m --size 736 1280
```

This downloads Ultralytics' official COCO-pretrained `yolo26m.pt` into this folder (it is not
committed), exports it to TorchScript for CUDA in FP16 with a fixed 736 × 1280 input, checks
that plain PyTorch can load and run it, and prints the size and SHA-256 for the table above.
736 is 720 rounded up to a multiple of 32, the model's stride, so a 1280 × 720 working image
needs only 16 rows of padding.

## Licence

Ultralytics YOLO models and weights are licensed under AGPL-3.0
(<https://github.com/ultralytics/ultralytics>). They were trained by Ultralytics on COCO; we
use them as released, without fine-tuning.
