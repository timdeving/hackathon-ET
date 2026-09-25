"""Object detector: an exported YOLO model, run with plain PyTorch.

Ultralytics only runs on the GPU PC, to export the model to TorchScript
(tools/export_detector.py). Here we do the small parts it would otherwise do: fit the image to
the model's fixed input size, run the network, and keep our classes above a confidence
threshold. NMS-free models (YOLO26) output final boxes; for models with raw output (YOLO11),
duplicate boxes are removed with NMS on the GPU.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

from src.config import WEIGHTS_DIR
from src.perception.boxes import PAD_VALUE, Detections, detections_from_rows, letterbox


def load_detector(params: Mapping[str, Any], device: str = "cuda") -> Detector:
    """The detector configured in params (the whole of configs/params.yaml)."""
    settings = params["detector"]
    return Detector(
        WEIGHTS_DIR / settings["weights"],
        sorted(settings["classes"]),
        settings["conf"],
        settings["iou"],
        settings["max_det"],
        device=device,
    )


def _configure_torch() -> None:
    """Deterministic GPU kernels, so runs repeat exactly; and no TF32 (the judges' T4 has none),
    so an RTX card computes like the T4."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class Detector:
    """Runs one exported YOLO TorchScript file on BGR images.

    Args:
        weights: TorchScript file made by tools/export_detector.py.
        class_ids: COCO class ids to keep.
        conf: minimum score for a detection to be kept.
        iou: overlap above which NMS drops the weaker box (raw-output models only).
        max_det: at most this many detections per image.
        device: "cuda" or "cpu"; the file must have been exported for that device.
    """

    def __init__(
        self,
        weights: str | Path,
        class_ids: Sequence[int],
        conf: float,
        iou: float,
        max_det: int,
        device: str = "cuda",
    ) -> None:
        _configure_torch()
        self.device = torch.device(device)
        extra_files = {"config.txt": ""}  # Ultralytics stores the export settings here
        self.model = torch.jit.load(
            str(weights), map_location=self.device, _extra_files=extra_files
        ).eval()
        metadata = json.loads(extra_files["config.txt"] or "{}")
        if "imgsz" not in metadata:
            raise ValueError(f"{weights} has no input size in its metadata; export it with "
                             "tools/export_detector.py")
        size = metadata["imgsz"]
        self.input_size = (size, size) if isinstance(size, int) else tuple(size)  # (h, w)
        self.dtype = next(self.model.parameters()).dtype
        self.class_ids = np.asarray(class_ids, dtype=np.int64)
        self._class_ids = torch.as_tensor(self.class_ids, device=self.device)
        self.conf, self.iou, self.max_det = conf, iou, max_det

        # Warm-up run, so GPU initialisation doesn't land on the first real frame. The output's
        # shape tells the two formats apart: NMS-free rows (N, 6) or raw (4 + classes, anchors).
        blank = np.full((*self.input_size, 3), PAD_VALUE, dtype=np.uint8)
        self.end2end = self._forward(blank).shape[-1] == 6

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> Detections:
        """Detect objects in one BGR uint8 image; boxes come back in that image's pixels."""
        boxed = letterbox(image, self.input_size)
        output = self._forward(boxed.image)
        rows = output if self.end2end else self._nms(output)
        return detections_from_rows(
            rows.float().cpu().numpy(),
            self.class_ids,
            self.conf,
            self.max_det,
            boxed.scale_x,
            boxed.scale_y,
        )

    @torch.inference_mode()
    def _forward(self, image: np.ndarray) -> torch.Tensor:
        """The network's output for one letterboxed image, without the batch dimension."""
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.flip(-1).permute(2, 0, 1)[None]  # BGR (H, W, C) -> RGB (1, C, H, W)
        output = self.model(tensor.to(self.dtype).div(255).contiguous())
        if isinstance(output, (list, tuple)):  # some exports also return intermediate outputs
            output = output[0]
        return output[0]

    def _nms(self, output: torch.Tensor) -> torch.Tensor:
        """Raw output (4 + classes, anchors) -> rows [x0, y0, x1, y1, score, class] after NMS."""
        scores, best = output[4:][self._class_ids].max(0)  # best of our classes per anchor
        keep = scores >= self.conf
        cx, cy, w, h = output[:4, keep].float()
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        scores = scores[keep].float()
        classes = self._class_ids[best[keep]]
        kept = torchvision.ops.batched_nms(boxes, scores, classes, self.iou)[: self.max_det]
        return torch.cat([boxes[kept], scores[kept, None], classes[kept, None].float()], dim=1)
