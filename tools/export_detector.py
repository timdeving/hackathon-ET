"""Export a COCO-pretrained Ultralytics YOLO model to TorchScript for the submission.

Runs on the GPU PC in the export environment (requirements-export.txt), the only place
Ultralytics is installed. Downloads the official .pt weights into weights/ (gitignored),
exports them for CUDA in FP16 at a fixed input size, checks that plain PyTorch can load and run
the result, and prints the file's size and SHA-256 for weights/README.md.

    python -m tools.export_detector --model yolo26m --size 736 1280
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import REPO_ROOT

WEIGHTS_DIR = REPO_ROOT / "weights"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="yolo26m", help="Ultralytics model (default yolo26m)")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[736, 1280],
        metavar=("HEIGHT", "WIDTH"),
        help="fixed model input size in pixels (default 736 1280)",
    )
    parser.add_argument("--device", default="0", help="CUDA device to export on (default 0)")
    args = parser.parse_args()
    height, width = args.size

    model = YOLO(str(WEIGHTS_DIR / f"{args.model}.pt"))  # downloads the official weights once
    # nms=False keeps YOLO26's NMS-free head; Ultralytics' default (None) exports the raw head,
    # which needs NMS. Models without that head (YOLO11) export their raw output either way.
    exported = model.export(
        format="torchscript", imgsz=[height, width], half=True, device=args.device, nms=False
    )
    target = WEIGHTS_DIR / f"{args.model}_{height}x{width}_fp16.torchscript"
    shutil.move(Path(exported), target)

    # The submission loads the file with plain PyTorch, so check exactly that.
    loaded = torch.jit.load(str(target), map_location="cuda")
    blank = torch.zeros(1, 3, height, width, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        output = loaded(blank)
    if isinstance(output, (list, tuple)):
        output = output[0]
    kind = "NMS-free rows" if output.shape[-1] == 6 else "raw, needs NMS"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"wrote {target.relative_to(REPO_ROOT)}: {target.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")
    print(f"output for one image: {tuple(output.shape)} ({kind})")


if __name__ == "__main__":
    main()
