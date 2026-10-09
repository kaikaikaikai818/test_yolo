"""Offline checks. Never opens cameras, serial ports or robot connections."""
import importlib.util
import json
from pathlib import Path
import numpy as np


def preflight(config, weights, stage="vision"):
    config, weights = Path(config), Path(weights)
    errors, warnings = [], []
    hardware = {}
    try:
        hardware = json.loads((config / "hardware.json").read_text(encoding="utf-8"))
        if set(hardware) != {"ROBOT_IP", "HI_SERIAL", "HO_SERIAL", "GRIP_PORT"}:
            raise ValueError("requires ROBOT_IP, HI_SERIAL, HO_SERIAL, GRIP_PORT")
        if not all(isinstance(v, str) and v.strip() for v in hardware.values()):
            raise ValueError("connection identifiers must be nonempty strings")
        if hardware["HI_SERIAL"] == hardware["HO_SERIAL"]:
            raise ValueError("camera serials must differ")
    except (OSError, ValueError) as exc:
        errors.append(f"hardware.json: {exc}")
    for filename in ("camera_pose.txt", "cam2end_20260906.txt"):
        try:
            matrix = np.loadtxt(config / filename)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1])
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=0.002)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=0.002)):
                raise ValueError("invalid rigid transform")
        except (OSError, ValueError) as exc:
            errors.append(f"{filename}: {exc}")
    try:
        scale = float(np.loadtxt(config / "camera_depth_scale.txt"))
        if not np.isfinite(scale) or not 0 < scale < 0.01:
            raise ValueError("expected positive metres per raw depth unit")
    except (OSError, ValueError) as exc:
        errors.append(f"camera_depth_scale.txt: {exc}")
    if not (config / "camera_20260906.ini").is_file():
        errors.append("missing camera_20260906.ini")
    if not weights.is_file():
        errors.append(f"missing model: {weights}")
    for name in ("camera_alignment.json", "grasp_surface_calibration.json"):
        if not (config / name).is_file():
            required = stage != "vision" if name == "camera_alignment.json" else stage in ("descent", "grasp", "place")
            (errors if required else warnings).append(f"missing {name}; recover from the calibrated workstation")
    missing = [name for name in ("pyrealsense2", "rtde_receive", "rtde_control", "minimalmodbus")
               if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("missing dependencies: " + ", ".join(missing) + "; install requirements-grasp.txt")
    return {"stage": stage, "config": str(config.resolve()), "weights": str(weights.resolve()),
            "hardware": hardware, "errors": errors, "warnings": warnings,
            "hardware_connected": False,
            "note": "Offline checks do not validate live calibration or authorize motion."}
