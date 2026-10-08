"""Run the trained YOLO segmentation model on a RealSense color stream."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from ultralytics import YOLO
from segmentation_outputs import save_snapshot


ROOT = Path(__file__).resolve().parent


def plot_result_with_contours(result) -> np.ndarray:
    """Show segmentation boundaries clearly on top of YOLO's usual overlay."""
    annotated = result.plot()
    if result.masks is None:
        return annotated

    for polygon in result.masks.xy:
        if len(polygon) < 3:
            continue
        points = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [points], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(annotated, [points], True, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "live_checks",
        help="Save raw images, predictions, masks and reports here with Space or S.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(0.28, 0.10, 0.70, 0.70),
        help="Detection area as frame fractions from 0 to 1 (default covers the center table).",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Detect on the full image instead of the default center-table area.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if rs is None:
        raise SystemExit(
            "pyrealsense2 is missing. Install it with: "
            r".\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt"
        )
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not model_path.is_file():
        raise SystemExit(
            f"Model weights not found: {model_path}\n"
            "Copy the trained best.pt into the project folder or pass --model."
        )

    selected = choose_device(args.camera, args.serial)
    model = YOLO(str(model_path))
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    actual_device = args.device or (0 if torch.cuda.is_available() else "cpu")
    left, top, right, bottom = args.roi
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise SystemExit("--roi values must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1.")

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

    window = f"YOLO tools on {selected['name']} - Space/S: save, Q/Esc: quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    profile = pipeline.start(config)
    print(f"Camera: {selected['name']} | serial={selected['serial']}")
    print(f"Model: {model_path}")
    print(f"Inference device: {actual_device}")
    print(f"Model SHA256: {model_hash}")
    print(f"Snapshot output: {output_root}")
    print("Green rectangle is the detection area. Space/S: save current frame. Q/Esc: quit.")

    previous_time = time.perf_counter()
    fps = 0.0
    saved_count = 0
    save_status = ""
    status_until = 0.0
    try:
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        snapshot_context = {
            "camera": {**selected, "role": camera_role(selected["name"])},
            "model": {"path": str(model_path.resolve()), "sha256": model_hash},
            "settings": {"device": str(actual_device), "conf": args.conf, "imgsz": 640,
                         "full_frame": args.full_frame},
            "color_intrinsics": {
                "width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            },
        }
        for _ in range(20):
            pipeline.wait_for_frames(5000)

        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            height, width = frame.shape[:2]
            if args.full_frame:
                x1, y1, x2, y2 = 0, 0, width, height
            else:
                x1 = max(0, min(width - 1, round(left * width)))
                y1 = max(0, min(height - 1, round(top * height)))
                x2 = max(x1 + 1, min(width, round(right * width)))
                y2 = max(y1 + 1, min(height, round(bottom * height)))
            roi_frame = frame[y1:y2, x1:x2]
            result = model.predict(
                source=roi_frame,
                imgsz=640,
                conf=args.conf,
                device=actual_device,
                retina_masks=True,
                verbose=False,
            )[0]
            display = frame.copy()
            display[y1:y2, x1:x2] = plot_result_with_contours(result)
            cv2.rectangle(display, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
            cv2.putText(
                display,
                "Detection area" if not args.full_frame else "Full frame",
                (x1 + 8, max(24, y1 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

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
            cv2.putText(
                display,
                f"Objects: {len(result.boxes)} | Saved: {saved_count} | Space/S: save",
                (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
            if now < status_until:
                cv2.putText(
                    display, save_status, (18, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S"), ord(" ")):
                try:
                    report_path = save_snapshot(
                        output_root, frame, display, result, (x1, y1, x2, y2),
                        {**snapshot_context,
                         "frame_number": color_frame.get_frame_number(),
                         "frame_timestamp_ms": color_frame.get_timestamp(),
                         "frame_timestamp_domain": str(color_frame.get_frame_timestamp_domain())},
                    )
                    saved_count += 1
                    save_status = f"Saved snapshot {saved_count}"
                    print(f"Saved: {report_path}")
                except (OSError, ValueError, cv2.error) as exc:
                    save_status = "Save failed - check terminal"
                    print(f"Save failed: {exc}")
                status_until = time.perf_counter() + 3.0
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
