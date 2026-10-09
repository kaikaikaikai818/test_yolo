# YOLO 双相机抓取接入（2026-10-09）

## 当前结果

从 `E:\test_sam3\small\MobileSAM\ur5_grasp-main` 迁移了抓取核心，保留旧项目的双相机定位、连续帧判断、目标关联、手柄候选点、方向对齐、分阶段下降、夹持与放置逻辑。只接入固定类别 YOLO 实例分割，不再依赖 MobileSAM、CLIPSeg 或文本编码器。

`detect_realsense.py` 仍用于单相机识别和补拍。当前入口是 `grasp_test.py`，完整测试流程留在 `grasp_test_workflow.py`；`grasp_tool.py` 仅作为旧命令兼容入口。今后的文本指令自动抓取和递送程序将使用独立的 `main.py`，不放进测试流程。
用户已在实测电脑完成螺丝刀的双相机视觉、观察点移动、高位方向对齐和40mm无接触下降；实际夹取和递送尚未验证。旧项目的成功记录不等于新入口已实机通过。

## 本地文件

已原样复制到 `config/grasp/`：
- `camera_pose.txt`：D455 到基座，平移单位米。
- `cam2end_20260906.txt`：D435i 到末端，平移单位毫米；原有转换代码保留。
- `camera_20260906.ini`：D435i 内参。
- `camera_depth_scale.txt`：D455 原始深度到米的标定系数。
- `hardware.json`：从旧入口提取的机器人 IP、两台相机序列号和夹爪串口。

以上本地配置不进入 Git。跨电脑需单独复制整个 `config/grasp`，同时保留实测电脑现有的新 `best.pt`。开发电脑的旧权重不能覆盖实测电脑的新权重。

以下文件最初在开发电脑的旧目录中未找到，用户现已从实测电脑找回并导入 `config/grasp/`：
- `camera_alignment.json`：双相机一致性验收文件，旧记录表示已完成。
- `grasp_surface_calibration.json`：支撑面与夹爪高度配置。

不依据旧记录中的数值手工伪造这两个文件。`vision` 可以在缺少它们时查看原始坐标；运动阶段会拒绝缺失的必需文件。运行中仍核对相机序列号、640×480 流尺寸和标定哈希。

## 实测电脑操作

先退出占用相机的旧识别窗口，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-grasp.txt
.\.venv\Scripts\python.exe grasp_test.py --preflight
.\.venv\Scripts\python.exe grasp_test.py --prompt screwdriver --stage vision --check-calib
```

默认读取项目根目录 `best.pt`，也可使用 `--checkpoint` 指定路径。启动显示实际模型路径、SHA256 和目标类别编号；按名称匹配权重中的类别，不硬编码8类或11类编号。

第一步只放一把螺丝刀，同时确认 D455 和腕部 D435i 都能识别。当前 YOLO 主要在 D455 场景训练，D435i 近景能力尚未验证。如果腕部漏检，先保存近景证据解决视觉问题，再推进动作。

抓取相机流沿用原标定的640×480；D455检测扩展到全图，运动仍受原工作空间范围限制。全图能看到的区域不等于机器人可抓取区域。CPU 双相机速度需现场测量。

`vision` 不创建机器人控制接口、不启用夹爪；会尝试只读 TCP，以便显示腕部目标的基座坐标。按Q退出。

## 后续阶段

找回配置并通过静止定位核对后，依次使用 `--stage observe`、`rotate`、`descent`、`grasp`。每一级通过后才推进；默认无自动执行。
- observe：按p到观察点，腕部初始看不到目标时原流程支持a粗定位。
- rotate：到观察点稳定后按y，高位对齐方向。
- descent：重新稳定后按d，停在夹持点上方40毫米；u垂直回升。
- grasp：按r执行夹持和抬升。自动推进需显式 `--auto`。
- place：额外需要现场审核后的 `config/grasp/fixed_placement.json`；根目录示例不能直接用于动作。

首次使用螺丝刀；卷尺挂绳问题与重新训练按用户决定暂缓。扳手、钳子等沿用分别审核的抓取策略，不把一种工具的夹持参数直接套到所有工具。

## 螺丝刀实际夹取测试

`descent` 阶段禁用夹爪；这与工具配置缺失不同。螺丝刀使用迁移的内置夹取路径，无需向 `tool_profiles.json` 新增螺丝刀配置，也无需修改源码中的开关。其他类别的配置要求不变。

若仍停在40mm检查点，先按U垂直回升，确认完成后按Q退出。Q/Esc只退出程序，不会自动回升。

在实测电脑项目根目录执行：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe grasp_test.py --stage grasp --preflight
.\.venv\Scripts\python.exe grasp_test.py --prompt screwdriver --stage grasp --check-calib
```

预检出现errors时先处理，不继续启动。夹取阶段会连接夹爪串口，启动不发送开合指令。保留单把螺丝刀和本轮已验证的摆放区域；不添加 `--auto`。

1. 等双相机稳定，通过门控后按P，到观察点。
2. 等 `BASE AXIS ... STABLE` 后按Y，高位对齐；再等腕部相机重新稳定。
3. 等 `SAFE DESCENT READY 3/3` 后按D，停在夹持点上方40mm。新的运行必须重新锁定计划，不能复用上一次已退出的下降记录。
4. 确认夹爪对着手柄、工具未移动，再按一次R。R会张开夹爪、下降到锁定高度、闭合，接触反馈通过后抬升50mm并停住；反馈失败则张开并退回40mm检查点。
5. 检查螺丝刀是否实际离开桌面、是否滑落。控制器的接触反馈不能单独证明工具稳定持握。首次测试停在这里，不按O空中释放，也不执行T或递送。

沿用当前参数：张开3500、闭合目标11000、夹持力度35、最后下降速度0.03m/s、试抬升50mm。闭合位置的回显不是实际接触证据；程序要求接触标志及实时力矩同时通过。夹持高度依当前手柄厚度计算，并保留至少5mm的TCP到固定支撑面净空检查。上述数值来自旧工作流，仍需在当前YOLO路径上做现场夹取验证。

## 离线验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_grasp_*.py"
```

包含原项目坐标/目标关联/运动门控/夹持几何测试，以及YOLO类别筛选、mask孔洞、ROI偏移、深度单位和无效深度测试。测试使用模拟接口，不连接硬件。`--preflight` 仅检查本地文件和依赖，不能替代现场校验。

## 从仓库拉取后的目录初始化

`config` 不进入Git，拉取后先运行：

```powershell
.\.venv\Scripts\python.exe init_grasp_config.py
```

它创建本项目的 `config/grasp` 并列出缺少的文件，不覆盖已有文件。
如实测电脑保留完整旧项目，可一次导入：

```powershell
.\.venv\Scripts\python.exe init_grasp_config.py --source "旧项目ur5_grasp-main的完整路径"
```

导入六份标定/验收文件；hardware.json不存在时从旧版ur5_grasp-main/grasp_tool.py读取连接常量生成，不执行旧代码。若两个新增找回的JSON在别处，再手工复制进去。只有这两个JSON不足以启动，目录还需四份核心标定与hardware.json。运行 `--preflight` 查看剩余缺项。

## 简易视觉测试启动

在项目根目录运行 `./run_grasp_test.ps1`，默认使用螺丝刀进入只读视觉阶段。可用 `./run_grasp_test.ps1 -Prompt pliers` 指定其他类别。该快捷入口固定使用 `vision` 阶段，不启动机械臂运动；后续逐阶段测试仍通过 `grasp_test.py --stage ...` 明确启动。
