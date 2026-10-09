"""Trained fixed-class YOLO masks adapted to the existing grasp geometry."""
from dataclasses import dataclass
from pathlib import Path
import time
import hashlib
import cv2
import numpy as np


def canonical_name(name):
    name = " ".join(str(name).lower().replace("_", " ").split())
    for prefix in ("an ", "a "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return {"adjustable wrench": "wrench", "adjustable spanner": "wrench",
            "crescent wrench": "wrench", "tape dispenser": "tape cutter"}.get(name, name)


@dataclass(frozen=True)
class Detection:
    box: tuple
    point: tuple
    score: float
    area: float


@dataclass(frozen=True)
class Segmentation:
    detection: Detection
    mask: np.ndarray
    sam_score: float  # Compatibility field; contains YOLO confidence, not SAM quality.


@dataclass(frozen=True)
class EngineResult:
    segmentations: list
    detection_seconds: float
    segmentation_seconds: float = 0.0
    cuda_peak_allocated_mb: float = 0.0
    cuda_peak_reserved_mb: float = 0.0


class YoloEngine:
    def __init__(self, prompt, checkpoint, device="cpu"):
        from ultralytics import YOLO
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing local weights: {path}")
        print("Model SHA256:", hashlib.sha256(path.read_bytes()).hexdigest())
        self.model = YOLO(str(path))
        if self.model.task != "segment":
            raise ValueError("Grasping requires instance segmentation weights")
        self.device = device
        self.class_ids = [int(i) for i, name in self.model.names.items()
                          if canonical_name(name) == canonical_name(prompt)]
        if not self.class_ids:
            raise ValueError(f"Target {prompt!r} is absent from weights. Classes: {self.model.names}")
        print(f"YOLO weights: {path}; target class IDs: {self.class_ids}; device: {device}")

    def infer(self, image_bgr):
        started = time.perf_counter()
        result = self.model.predict(image_bgr, device=self.device, classes=self.class_ids,
                                    conf=0.35, imgsz=640, retina_masks=True, verbose=False)[0]
        segments = []
        shape = image_bgr.shape[:2]
        if result.boxes is not None and result.masks is not None:
            for box, data in zip(result.boxes, result.masks.data):
                if int(box.cls.item()) not in self.class_ids:
                    continue
                mask = data.cpu().numpy() > 0.5
                # Retina masks must already be in original image coordinates.
                # Do not stretch a letterboxed mask and corrupt grasp geometry.
                if mask.shape != shape:
                    raise ValueError(f"Mask coordinates mismatch: {mask.shape} != {shape}")
                if not mask.any():
                    continue
                distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
                point = cv2.minMaxLoc(distance)[3]
                coords = tuple(int(round(v)) for v in box.xyxy[0].tolist())
                score = float(box.conf.item())
                segments.append(Segmentation(Detection(coords, point, score,
                                                       float(mask.sum())), mask, score))
        return EngineResult(segments, time.perf_counter() - started)
