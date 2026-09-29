# YOLO11-Seg 工具分割项目

使用 Ultralytics YOLO11s-Seg，计划识别：螺丝刀、扳手、剪刀、胶带切割器、卷尺。
系统使用固定安装的 RealSense D455 进行远景类别识别与粗定位，使用 UR5 末端的 D435i 进行近景精定位。目前已完成单张图片推理、实例 mask 导出和相机数据采集脚本，尚未进行自定义模型训练、相机标定或机械臂控制。

## 在另一台 Windows 电脑安装

安装 Git 和 Python 3.12，然后克隆本仓库并进入仓库目录。执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1
```

首次运行会自动下载官方 `yolo11s-seg.pt` 权重，需要网络。默认使用包内公交车图片并在 CPU 上测试。
若 PowerShell 阻止脚本运行，可直接执行：

```powershell
.\.venv\Scripts\python.exe predict.py
```

自己的图片：

```powershell
.\.venv\Scripts\python.exe predict.py --source 'D:\images\test.jpg'
```

输出在 `results/`：叠加分割图、各目标的二值 mask PNG、包含类别/置信度/检测框/耗时的 `report.json`。重复运行会覆盖同名结果。

## GPU 环境

GPU 电脑应先按照 https://pytorch.org/get-started/locally/ 安装适合驱动的 CUDA 版 PyTorch，再安装 requirements.txt。
验证 GPU 后运行：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe predict.py --device 0
```

`requirements-lock.txt` 记录初次 CPU 环境的依赖版本，供复现参考；GPU 环境不要直接照搬其中的 PyTorch 安装方式。

## 两台电脑同步

开始工作前执行 `git pull --ff-only`。完成代码修改后执行 `git add <文件>`、`git commit -m "说明修改"` 和 `git push`，再到另一台电脑拉取。

虚拟环境、权重、原始图片、结果和本机配置不进入 Git。另一台电脑需要重建环境，权重自动下载；后续采集图片需要单独备份或传输。

五类工具需自行采集与标注后训练；当前预训练模型只用于验证流程，尚不能代表五类工具识别能力。

## RealSense 数据采集

先在安装好 RealSense 驱动的电脑上，为虚拟环境安装 Python 相机接口：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt
```

连接相机后列出设备及序列号：

```powershell
.\.venv\Scripts\python.exe capture_realsense.py --list
```

分别采集 D455 远景和 D435i 近景：

```powershell
.\.venv\Scripts\python.exe capture_realsense.py --camera d455
.\.venv\Scripts\python.exe capture_realsense.py --camera d435i
```

预览窗口中按空格或 `S` 保存，按 `Q` 或 `Esc` 退出。彩色图片分别写入：

```text
data/raw/d455/images/
data/raw/d435i/images/
```

每张图片还会生成一个同名 JSON，记录相机型号、序列号、时间和内参。数据目录已加入 `.gitignore`，不会被误传到 GitHub。

训练 YOLO11-Seg 只需彩色图片。需要同时保存对齐后的 16 位深度 PNG 时添加 `--save-depth`：

```powershell
.\.venv\Scripts\python.exe capture_realsense.py --camera d455 --save-depth
```

如果连接了两台相同型号的相机，使用 `--serial` 指定 `--list` 显示的序列号。数据采集阶段建议一次开启一台相机，便于区分视角并减少 USB 带宽占用。

官方项目：https://github.com/ultralytics/ultralytics
官方安装文档：https://docs.ultralytics.com/quickstart/
