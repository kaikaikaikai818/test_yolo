"""Create local grasp configuration; optionally import an existing calibrated project."""
import argparse
import ast
import json
import shutil
from pathlib import Path

FILES = ("camera_pose.txt", "camera_depth_scale.txt", "cam2end_20260906.txt",
         "camera_20260906.ini", "camera_alignment.json", "grasp_surface_calibration.json")
KEYS = ("ROBOT_IP", "HI_SERIAL", "HO_SERIAL", "GRIP_PORT")


def initialize(destination, source=None):
    destination.mkdir(parents=True, exist_ok=True)
    if source is not None:
        source = source.resolve()
        if not source.is_dir():
            raise ValueError(f"Source folder does not exist: {source}")
        for name in FILES + ("hardware.json",):
            target = destination / name
            if not target.exists() and (source / name).is_file():
                shutil.copy2(source / name, target)
                print(f"Copied: {name}")
        hardware_path = destination / "hardware.json"
        old_entry = source / "grasp_tool.py"
        if not hardware_path.exists() and old_entry.is_file():
            # Read constant values only. Never execute the old robot program.
            tree = ast.parse(old_entry.read_text(encoding="utf-8-sig"))
            values = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in KEYS:
                            value = node.value.value
                            if isinstance(value, str) and value.strip():
                                values[target.id] = value
            if set(values) == set(KEYS):
                with hardware_path.open("x", encoding="utf-8") as handle:
                    json.dump(values, handle, indent=2)
                print("Created hardware.json from old connection constants")
    missing = [name for name in FILES + ("hardware.json",) if not (destination / name).is_file()]
    print(f"Local config: {destination}")
    print("Missing: " + (", ".join(missing) if missing else "none"))
    print("Existing files were preserved. No hardware was connected.")
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="old calibrated ur5_grasp-main folder")
    args = parser.parse_args()
    initialize(Path(__file__).resolve().parent / "config" / "grasp", args.source)
