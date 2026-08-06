# LinkerHand 手部重定向与遥操作

本项目将摄像头或录像中的人手关键点转换为 LinkerHand L20/G20 的关节目标，并提供运动学仿真、安全过滤、ROS 2 硬件桥接、动作库与评测工具。

核心数据流：

```text
MediaPipe / RealSense / 录像
             ↓
       手部关键点预处理
             ↓
      单手指姿态重定向
             ↓
       碰撞与限位过滤
             ↓
  仿真可视化 / ROS 2 硬件接口
```

> 安全说明：默认流程不会驱动真实硬件。真实运动必须同时提供 `--enable-motion` 和由操作人员设置的 `HW_ENABLE_TOKEN`。在完成硬件限位、急停、夹持力和看护检查前，请只使用仿真或 dry-run。

## 仓库布局

上游最新提交（`30216d1`）中的原始文件已完整移入 `src/`。根目录的这份 README 是重新整理后的入口文档。

| 路径 | 用途 |
| --- | --- |
| `src/perception/` | MediaPipe、RealSense、录像与回放输入 |
| `src/finger_retarget/` | 人手关键点到 L20 关节角的重定向求解器 |
| `src/kinematics/` | 关节约定与正向运动学 |
| `src/safety/` | 关节限位、速率限制与碰撞过滤 |
| `src/sim/` | PyBullet 运动学、动力学与接触仿真 |
| `src/viz/` | 实时及离线可视化 |
| `src/comms/` | ROS 2、硬件映射与动作库工具 |
| `src/humanego_linkerhand/` | HumanEgo 双指控制流程 |
| `src/scripts/` | 数据处理、训练和评测脚本 |
| `src/tests/` | G0–G2、通信与可视化测试 |
| `src/docs/` | 架构、运行指南、设计决策和任务记录 |

## 环境准备

建议使用 Python 3.10 或更高版本，并在仓库根目录执行以下命令：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

按用途安装依赖：

```bash
# 求解器、评测和测试
python -m pip install -r src/eval/requirements.txt

# PyBullet 仿真与可视化
python -m pip install -r src/sim/requirements.txt

# Webcam / 录像输入
python -m pip install -r src/perception/requirements.txt
```

ROS 2 硬件桥接还需要 ROS 2 Jazzy、已构建的 LinkerHand ROS 2 SDK，以及：

```bash
python -m pip install -r src/comms/requirements.txt
```

## 快速开始

### 1. 无摄像头安全冒烟测试

使用仓库自带的动作序列运行 30 帧，无窗口、无硬件：

```bash
python -m src.viz.app --camera-free --headless --max-frames 30
```

查看仿真窗口：

```bash
python -m src.viz.app --camera-free
```

### 2. Webcam 实时镜像

```bash
python -m src.viz.app \
  --source webcam \
  --camera-index 0 \
  --side right \
  --show-camera
```

按 `Q` 或 `Esc` 退出。若输入是自拍镜像画面，添加 `--image-mirrored`。

### 3. 处理录像

```bash
python -m src.viz.app \
  --source video \
  --video-path /path/to/hand_video.mp4 \
  --side right
```

查看所有参数：

```bash
python -m src.viz.app --help
```

## 运行测试

运行完整测试集：

```bash
python -m pytest src/tests
```

按安全阶段分别运行：

```bash
python -m pytest src/tests/g0_unit
python -m pytest src/tests/g1_kinematic
python -m pytest src/tests/g2_safety src/tests/g2_dynamic
```

测试可能依赖 PyBullet、URDF、录像样本或本机性能基线；请先安装对应依赖并阅读失败信息中的环境要求。

## 真实硬件

硬件控制是人工审核阶段，不属于快速开始流程。开始前至少确认：

- LinkerHand ROS 2 SDK 已构建，CAN 与对应手型配置正确；
- `src/hardware/LIMITS.md` 中的限位已经人工复核；
- 急停、看护、低速和低电流限制均已就绪；
- G0–G2 测试通过，并明确了解 `src/CLAUDE.md` 中的硬件安全约束。

`src.comms.camera_to_linkerhand` 默认是 dry-run。真实发布还需要操作人员显式添加 `--enable-motion` 并设置 `HW_ENABLE_TOKEN`；不要把该变量写入脚本或提交到仓库。

## 推荐阅读

- 总体架构：[`src/docs/ARCHITECTURE.md`](src/docs/ARCHITECTURE.md)
- 当前开发状态：[`src/STATE.md`](src/STATE.md)
- 硬件限位：[`src/hardware/LIMITS.md`](src/hardware/LIMITS.md)
- Isaac Gym 配置：[`src/docs/isaac_gym_setup.md`](src/docs/isaac_gym_setup.md)
- Isaac Lab 配置：[`src/docs/isaac_lab_setup.md`](src/docs/isaac_lab_setup.md)
- HumanEgo 双指流程：[`src/docs/humanego_linkerhand_two_finger.md`](src/docs/humanego_linkerhand_two_finger.md)
- Residual ACT 流程：[`src/docs/residual_act_pipeline.md`](src/docs/residual_act_pipeline.md)
- 原始 README（历史命令与实验参数）：[`src/README.md`](src/README.md)

## License

许可证文件随原项目保存在 [`src/LICENSE`](src/LICENSE)。
