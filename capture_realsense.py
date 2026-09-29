"""Capture labeled-dataset images from a RealSense D455 or D435i."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a RealSense camera and save images with Space."
    )
    parser.add_argument(
        "--camera",
        choices=("auto", "d455", "d435i"),
        default="auto",
        help="Camera model to open. Use auto only when one RealSense is connected.",
    )
    parser.add_argument("--serial", help="Select a device by serial number.")
    parser.add_argument("--list", action="store_true", help="List devices and exit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Dataset root (default: data/raw).",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Also save aligned 16-bit depth PNG and camera metadata.",
    )
    return parser.parse_args()


def require_realsense() -> None:
    if rs is None:
        raise SystemExit(
            "pyrealsense2 is not installed. Run: "
            r".\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt"
        )


def device_records() -> list[dict[str, str]]:
    require_realsense()
    records = []
    for device in rs.context().query_devices():
        records.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
        )
    return records


def camera_role(name: str) -> str:
    normalized = name.lower().replace(" ", "")
    if "d435i" in normalized:
        return "d435i"
    if "d455" in normalized:
        return "d455"
    return "unknown"


def select_device(
    records: list[dict[str, str]], requested_role: str, requested_serial: Optional[str]
) -> dict[str, str]:
    if not records:
        raise SystemExit("No RealSense camera detected. Check USB 3 cable and driver.")

    if requested_serial:
        matches = [item for item in records if item["serial"] == requested_serial]
    elif requested_role != "auto":
        matches = [item for item in records if camera_role(item["name"]) == requested_role]
    else:
        matches = records

    if not matches:
        raise SystemExit(
            f"Requested camera was not found. Connected: "
            + ", ".join(f'{item["name"]} ({item["serial"]})' for item in records)
        )
    if len(matches) > 1:
        raise SystemExit("Multiple cameras match. Select one with --serial SERIAL_NUMBER.")
    return matches[0]


def intrinsics_dict(profile) -> dict[str, object]:
    intr = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def prepare_directories(root: Path, role: str, save_depth: bool) -> dict[str, Path]:
    paths = {
        "images": root / role / "images",
        "metadata": root / role / "metadata",
    }
    if save_depth:
        paths["depth"] = root / role / "depth"
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    records = device_records()
    if args.list:
        if not records:
            print("No RealSense camera detected.")
            return
        for item in records:
            print(
                f'{item["name"]} | serial={item["serial"]} | '
                f'firmware={item["firmware"]}'
            )
        return

    selected = select_device(records, args.camera, args.serial)
    role = camera_role(selected["name"])
    if role == "unknown":
        role = args.camera if args.camera != "auto" else "realsense"
    paths = prepare_directories(args.output.resolve(), role, args.save_depth)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected["serial"])
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    if args.save_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color) if args.save_depth else None
    depth_scale = (
        profile.get_device().first_depth_sensor().get_depth_scale()
        if args.save_depth
        else None
    )
    saved = len(list(paths["images"].glob("*.jpg")))
    window_name = f"Capture {role.upper()} - Space: save, Q/Esc: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f'Opened {selected["name"]}, serial {selected["serial"]}')
    print(f'Saving color images to {paths["images"]}')
    try:
        # Discard initial auto-exposure frames.
        for _ in range(30):
            pipeline.wait_for_frames(5000)

        while True:
            frames = pipeline.wait_for_frames(5000)
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if args.save_depth else None
            if not color_frame or (args.save_depth and not depth_frame):
                continue

            color = np.asanyarray(color_frame.get_data())
            preview = color.copy()
            cv2.putText(
                preview,
                f"{role.upper()}  saved: {saved}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key not in (ord("s"), 32):
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = f"{role}_{timestamp}"
            image_path = paths["images"] / f"{stem}.jpg"
            if not cv2.imwrite(str(image_path), color, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                print(f"Failed to save {image_path}", file=sys.stderr)
                continue

            metadata = {
                "camera_role": role,
                "camera_name": selected["name"],
                "serial": selected["serial"],
                "firmware": selected["firmware"],
                "captured_at": datetime.now().astimezone().isoformat(),
                "color_file": image_path.name,
                "color_intrinsics": intrinsics_dict(color_frame.profile),
            }
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data())
                depth_path = paths["depth"] / f"{stem}.png"
                cv2.imwrite(str(depth_path), depth)
                metadata["depth_file"] = depth_path.name
                metadata["depth_scale_meters"] = depth_scale
                metadata["depth_aligned_to_color"] = True

            metadata_path = paths["metadata"] / f"{stem}.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved += 1
            print(f"Saved {image_path.name} (total {saved})")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
