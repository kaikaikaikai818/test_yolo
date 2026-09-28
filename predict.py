"""Run YOLO11 instance segmentation and export masks for OpenCV."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / "config" / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "config"))

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ASSETS / "bus.jpg"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = YOLO(str(ROOT / "yolo11s-seg.pt"))
    results = model.predict(source=args.source, device=args.device, imgsz=640,
                            retina_masks=True, verbose=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report = {"torch": torch.__version__, "device": args.device, "images": []}
    for i, result in enumerate(results):
        result.save(filename=str(out / f"image_{i}_seg.jpg"))
        instances = []
        if result.masks is not None:
            for j, (box, mask) in enumerate(zip(result.boxes, result.masks.data)):
                binary = (mask.cpu().numpy() > 0.5).astype("uint8") * 255
                cv2.imwrite(str(out / f"image_{i}_mask_{j}.png"), binary)
                instances.append({"class": result.names[int(box.cls.item())],
                                  "confidence": float(box.conf.item()),
                                  "box_xyxy": box.xyxy[0].cpu().tolist(),
                                  "mask_shape": list(binary.shape)})
        report["images"].append({"source": str(result.path), "speed_ms": result.speed,
                                 "instances": instances})
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved results to: {out}")


if __name__ == "__main__":
    main()
