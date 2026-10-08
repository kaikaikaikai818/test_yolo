"""Save full-frame segmentation evidence without a camera dependency."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def instance_masks(result) -> list[np.ndarray]:
    """Return full-resolution binary masks in the prediction image coordinates."""
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy()
    if tuple(masks.shape[1:]) != tuple(result.orig_shape):
        raise ValueError("Full-resolution masks are required; use retina_masks=True.")
    return [(mask > 0.5).astype(np.uint8) * 255 for mask in masks]


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def save_snapshot(
    output_root: Path,
    frame: np.ndarray,
    display: np.ndarray,
    result,
    roi_xyxy: tuple[int, int, int, int],
    context: dict,
) -> Path:
    """Save the same frame, prediction and masks, including zero-detection cases."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi_xyxy
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Snapshot ROI must be inside the full frame.")
    if tuple(result.orig_shape) != (y2 - y1, x2 - x1):
        raise ValueError("Prediction shape does not match the snapshot ROI.")
    if display.shape != frame.shape:
        raise ValueError("Display and raw frame must have the same shape.")
    masks = instance_masks(result)
    boxes = result.boxes if result.boxes is not None else []
    if masks and len(masks) != len(boxes):
        raise ValueError("Mask count does not match the detected instance count.")

    captured_at = datetime.now().astimezone()
    role = context["camera"]["role"]
    stem = f"{role}_{captured_at.strftime('%Y%m%d_%H%M%S_%f')}"
    camera_root = output_root / role
    paths = {name: camera_root / name for name in ("images", "predictions", "masks", "metadata")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    metadata_path = paths["metadata"] / f"{stem}.json"
    raw_path = paths["images"] / f"{stem}.jpg"
    overlay_path = paths["predictions"] / f"{stem}.jpg"
    if any(path.exists() for path in (metadata_path, raw_path, overlay_path)):
        raise FileExistsError(f"Snapshot already exists: {stem}")

    write_image(raw_path, frame)
    write_image(overlay_path, display)
    instances = []
    for index, box in enumerate(boxes):
        roi_box = box.xyxy[0].cpu().tolist()
        full_box = [roi_box[0] + x1, roi_box[1] + y1, roi_box[2] + x1, roi_box[3] + y1]
        item = {
            "instance_id": index,
            "class_id": int(box.cls.item()),
            "class": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "box_xyxy": full_box,
            "box_xyxy_roi": roi_box,
            "mask_file": None,
        }
        if masks:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = masks[index]
            mask_path = paths["masks"] / f"{stem}_mask_{index}.png"
            write_image(mask_path, mask)
            item["mask_file"] = mask_path.relative_to(camera_root).as_posix()
            item["mask_area_pixels"] = int(np.count_nonzero(mask))
        instances.append(item)

    report = {
        **context,
        "captured_at": captured_at.isoformat(),
        "raw_image": raw_path.relative_to(camera_root).as_posix(),
        "prediction_image": overlay_path.relative_to(camera_root).as_posix(),
        "frame_shape": list(frame.shape),
        "roi_xyxy": list(roi_xyxy),
        "coordinate_system": "full_color_frame_pixels",
        "annotation_source": "model_prediction",
        "mask_values": {"background": 0, "foreground": 255},
        "speed_ms": result.speed,
        "instances": instances,
    }
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path
