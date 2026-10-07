# Project instructions

Before starting work, read PROJECT_PROGRESS.md if present. It records the latest progress, unresolved issues and next tasks; update it when the user requests a handoff or end-of-day record.

Use the repository root as the project directory. The primary workstation uses E:\test_yolo; other computers may clone anywhere.
Run Python with .\.venv\Scripts\python.exe and install dependencies with python -m pip.
Keep generated results, camera data, model weights, local configuration and virtual environments out of Git.
The project supports image segmentation through predict.py and run.ps1, and dataset capture from a D455 or D435i through capture_realsense.py. Robot control is not implemented yet.
