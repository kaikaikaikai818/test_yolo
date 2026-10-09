# Project instructions

Before starting work, read PROJECT_PROGRESS.md if present. It records the latest progress, unresolved issues and next tasks; update it when the user requests a handoff or end-of-day record.

Use the repository root as the project directory. The primary workstation uses E:\test_yolo; other computers may clone anywhere.
Run Python with .\.venv\Scripts\python.exe and install dependencies with python -m pip.
Keep generated results, camera data, model weights, local configuration and virtual environments out of Git.
The project supports image segmentation through predict.py and run.ps1, dataset capture through capture_realsense.py, and staged dual-camera robot tests through grasp_test.py (implementation in grasp_test_workflow.py). Read the latest field validation status in PROJECT_PROGRESS.md before proposing the next stage.
Keep the current staged test program separate from the future production main.py. The future main program should accept tool text and automate grasping and delivery through reusable modules, without test menus or test hotkeys. Do not treat the existing long test workflow as the finished production program.
