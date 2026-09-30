# YOLO11-Seg 工具分割项目

使用 Ultralytics YOLO11s-Seg 进行工具实例分割。第一版模型包含：电工胶带、锤子、手、手锯、钳子、螺丝刀、卷尺和扳手。后续将补充剪刀与胶带切割器。

系统计划使用固定安装的 RealSense D455 进行远景类别识别与粗定位，使用 UR5 末端的 D435i 进行近景精定位。目前已完成第一版自定义模型训练、单张图片推理、实例 mask 导出和相机数据采集脚本；实时双相机识别、相机标定和机械臂控制尚未实现。

## 当前程序入口

- `predict.py`：加载项目根目录的 `best.pt`，识别图片、视频或摄像头输入并导出 mask。
- `run.ps1`：调用 `predict.py` 的快捷启动脚本。
- `capture_realsense.py`：从 D455 或 D435i 预览并保存彩色图、深度图和相机参数，仅负责数据采集。

训练得到的 `best.pt` 不进入 Git。使用本项目前需将权重放在仓库根目录：

```text
test_yolo/best.pt
```

## 在另一台 Windows 电脑安装

安装 Git 和 Python 3.12，然后克隆本仓库并进入仓库目录。执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

复制 `best.pt` 到项目根目录后，对自己的图片运行：

```powershell
.\run.ps1 -Source 'D:\images\test.jpg'
```

若 PowerShell 阻止脚本运行，可直接执行：

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

虚拟环境、权重、原始图片、结果和本机配置不进入 Git。另一台电脑需要重建环境，并单独复制 `best.pt`；采集图片也需要单独备份或传输。

## RealSense 数据采集

先在安装好 RealSense 驱动的电脑上，为虚拟环境安装 Python 相机接口：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-camera.txt
```

连接相机后列出设备及序列号：

```powershell
.\.venv\Scripts\python.exe capture_realsense.py --list
```

可以分别采集两台相机，也可以同时打开 D455 和 D435i；双相机模式下按一次空格，两台相机都会保存当前画面：

```powershell
.\.venv\Scripts\python.exe capture_realsense.py --camera d455
.\.venv\Scripts\python.exe capture_realsense.py --camera d435i
.\.venv\Scripts\python.exe capture_realsense.py --camera both
```

双相机模式会弹出两个预览窗口。两台相机会同时持续取流，按空格时分别保存最新画面，并使用同一时间标记便于配对；这适合工具静止摆放时采集成对图片，不是硬件级帧同步。需要同时保存深度时，在命令末尾添加 `--save-depth`。

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

## RealSense 实时识别

将训练得到的 `best.pt` 单独复制到项目根目录，连接相机后运行：

```powershell
.\.venv\Scripts\python.exe detect_realsense.py --camera d455
```

默认在画面中央显示绿色工作区框，并只识别框内区域；该范围适用于工具放在画面中央工作台的初始测试。可通过 `--roi LEFT TOP RIGHT BOTTOM` 调整范围，数值是画面宽高的比例（0 到 1）；添加 `--full-frame` 可改为全画面识别。程序会自动使用可用的 NVIDIA GPU，否则使用 CPU。按 `Q` 或 `Esc` 退出。指定 D435i 时将 `d455` 改成 `d435i`；也可通过 `--serial` 选择相机。此程序只做视觉识别，不发送机械臂运动指令。

官方项目：https://github.com/ultralytics/ultralytics
官方安装文档：https://docs.ultralytics.com/quickstart/
