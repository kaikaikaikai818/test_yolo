"""Run the trained YOLO segmentation model on a RealSense color stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is missing. Install it with: "
        r".\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt"
    ) from exc

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def camera_role(name: str) -> str:
    normalized = name.lower().replace(" ", "")
    if "d435i" in normalized:
        return "d435i"
    if "d455" in normalized:
        return "d455"
    return "unknown"


def choose_device(camera: str, serial: str | None) -> dict[str, str]:
    devices = [
        {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
        }
        for device in rs.context().query_devices()
    ]
    if serial:
        matches = [item for item in devices if item["serial"] == serial]
    else:
        matches = [item for item in devices if camera_role(item["name"]) == camera]
    if not matches:
        found = ", ".join(f'{d["name"]} ({d["serial"]})' for d in devices) or "none"
        raise SystemExit(f"No matching RealSense camera. Detected: {found}")
    if len(matches) > 1:
        raise SystemExit("More than one camera matched; choose one with --serial.")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live YOLO instance segmentation from a RealSense camera."
    )
    parser.add_argument("--camera", choices=("d455", "d435i"), default="d455")
    parser.add_argument("--serial", help="Select an exact RealSense serial number.")
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--device", default=None, help="Use cpu or 0 (default: choose automatically).")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not model_path.is_file():
        raise SystemExit(
            f"Model weights not found: {model_path}\n"
            "Copy the trained best.pt into the project folder or pass --model."
        )

    selected = choose_device(args.camera, args.serial)
    model = YOLO(str(model_path))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected["serial"])
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    window = f"YOLO tools on {selected['name']} - Q/Esc: quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    profile = pipeline.start(config)
    actual_device = args.device or (0 if torch.cuda.is_available() else "cpu")
    print(f"Camera: {selected['name']} | serial={selected['serial']}")
    print(f"Model: {model_path}")
    print(f"Inference device: {actual_device}")
    print("Press Q or Esc in the video window to quit. Robot control is not enabled.")

    previous_time = time.perf_counter()
    fps = 0.0
    try:
        for _ in range(20):
            pipeline.wait_for_frames(5000)

        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            result = model.predict(
                source=frame,
                imgsz=640,
                conf=args.conf,
                device=actual_device,
                retina_masks=True,
                verbose=False,
            )[0]
            display = result.plot()

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instantaneous_fps if fps == 0 else 0.85 * fps + 0.15 * instantaneous_fps
            previous_time = now
            cv2.putText(
                display,
                f"{selected['name']} | {actual_device} | {fps:.1f} FPS",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
