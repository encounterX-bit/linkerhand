# LinkerHand G20 rotation workflow

当前任务：

- 新任务：录制 cube 水平逆时针 `90°` 的完整轨迹。
- 已有模型：保留 vertical clockwise rotation 的最佳 ACT 真机命令作为回退。
- 当前 cube yaw 只用于测量、标注和筛选数据，不参与控制；本流程尚未加入 IK/FK。

录制时，相机 0 负责真机和 cube 图像，相机 2 负责 MediaPipe。不要同时运行 GUI、
ACT runner、动作回放或 cube pose preview，以免抢占相机或 ROS 控制 topic。

## 1. CAN

每次开机或 CAN 未连接时运行：

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up

ip -details link show can0
```

## 2. 可选：录制前检查 cube yaw

该程序只显示 pose，不会控制真机。检查完成后按 `Q`/`ESC` 退出，再启动 recorder。

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source .venv/bin/activate

python -m src.comms.track_aruco_cube_pose \
  --camera-calibration data/calibration/camera0_640x480.json \
  --layout-profile data/calibration/aruco_cube_top_face.json \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG \
  --face-id 5 \
  --minimum-markers 2 \
  --max-reprojection-error-px 2 \
  --yaw-sign 1
```

把 cube 放在轨迹起点并按 `Z` 清零。水平逆时针旋转后，filtered yaw 应接近 `+90°`，
重投影误差应低于 `2 px`。当前标定面是 face 5，marker ID 为 20–23。

## 3. 录制水平逆时针 90° 轨迹

### Terminal 1：G20 driver

```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right \
  --can can0 \
  --is_touch true \
  --touch-fingers with-palm
```

### Terminal 2：轨迹 recorder

先启动 recorder，再启动 Terminal 3。它会等待 teleop 的 SPACE 触发信号。

```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

python3 scripts/linkerhand_g20_touch_recorder.py \
  --hand-type right \
  --task-id cube_yaw_ccw90_v1 \
  --trial-name cube_yaw_ccw90_v1 \
  --rate 30 \
  --duration 0 \
  --camera 0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG \
  --jpeg-quality 92 \
  --contact-threshold 20 \
  --ros-trigger \
  --require-state \
  --output-dir ~/Desktop/Jacky/sims/linker-hand-teleopt/data/act_demos
```

### Terminal 3：MediaPipe + 键盘拇指 teleop

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate

export HW_ENABLE_TOKEN=1

python -m src.comms.camera_to_linkerhand \
  --source webcam \
  --camera-index 2 \
  --min-hand-detection-confidence 0.75 \
  --min-hand-tracking-confidence 0.75 \
  --hand-confirm-frames 5 \
  --open-on-start-seconds 1 \
  --side right \
  --sdk-hand-joint g20 \
  --hardware-map g20-sim \
  --show-camera \
  --absolute \
  --motion-key-toggle \
  --motion-stop-mode open \
  --q0-key-step 10 \
  --thumb-abd-key-step 10 \
  --thumb-roll-key-step 10 \
  --thumb-tip-key-step 10 \
  --no-filter \
  --one-euro-min-cutoff 0.8 \
  --one-euro-beta 0.04 \
  --one-euro-d-cutoff 1.0 \
  --fingertip-extend "0,0.12,0.09,0.12,0.05" \
  --fingertip-lateral "0,-0.015,0.015,0.015,0.04" \
  --fingertip-straighten "0,0.32,0.42,0.20,0.15" \
  --thumb-gain 1 \
  --thumb-cross-gain 0.10 \
  --thumb-assist-smooth 0.84 \
  --thumb-orient-gain 0.58 \
  --thumb-grasp-gain 0.38 \
  --thumb-base-assist-gain 0.72 \
  --thumb-tip-gain 0.94 \
  --hardware-landmark-thumb \
  --landmark-thumb-gain 0.72 \
  --landmark-thumb-reach-gain 0.66 \
  --hardware-base-gain 1.80 \
  --hardware-base-gains "0.72,1.05,1.05,0.63" \
  --hardware-spread-gain 0.64 \
  --hardware-spread-signs "0.35,1.00,-0.15,-1.00" \
  --hardware-tip-gain 0.80 \
  --hardware-tip-gains "1.34,0.85,1.20,1.27" \
  --hardware-thumb-tip-gain 1.35 \
  --hardware-thumb-tip-offset -27 \
  --hardware-thumb-roll-gain 0.96 \
  --hardware-thumb-roll-offset -24 \
  --hardware-thumb-base-gain 0.69 \
  --hardware-thumb-base-offset 20 \
  --hardware-thumb-abd-gain 0.72 \
  --hardware-thumb-abd-offset -28 \
  --thumb-safe-mode limited \
  --max-thumb-delta 235 \
  --max-thumb-abd-delta 240 \
  --max-thumb-base-delta 165 \
  --max-spread-delta 90 \
  --spread-close-threshold 0.82 \
  --spread-recenter-gain 0.10 \
  --thumb-index-guard \
  --thumb-index-threshold 0.44 \
  --thumb-index-release 0 \
  --current-limit 35 \
  --speed-limit 35 \
  --max-range-step 5 \
  --log-period 0.25 \
  --log-sim-position \
  --enable-motion
```

### 每条 episode

1. 等 Terminal 3 显示 `STOPPED`。
2. 把 cube 放在统一起点，保持 face 5 朝向相机 0。
3. 点击 MediaPipe 窗口，保持起始姿态约 1 秒，然后按一次 `SPACE`。
4. 完成水平逆时针 `90°`，在终点稳定保持至少 2 秒。
5. 再按一次 `SPACE`，保存 episode；真机随后回到 open pose。
6. 重新摆好 cube，重复录制。先录 10 条检查，再补到 30–50 条。

键盘拇指控制：

- `W/S`：q0。
- `L/J`：q5。
- `I/K`：q10。
- `D/A`：q15。
- `Q`：切换拇指是否跟随 MediaPipe，不是退出。
- `ESC`：退出 teleop。

录坏的 episode 可以暂时保留，离线标注会将其标为 `INVALID`，不会改写原始图像、
关节、action 或触觉数据。

## 4. 离线计算 yaw 并筛选

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source .venv/bin/activate

CCW_SESSION=$(ls -dt data/act_demos/*cube_yaw_ccw90_v1 | head -1)
echo "$CCW_SESSION"

python -m src.comms.annotate_cube_yaw_episodes \
  --session "$CCW_SESSION" \
  --camera-calibration data/calibration/camera0_640x480.json \
  --layout-profile data/calibration/aruco_cube_top_face.json \
  --target-yaw-deg 90 \
  --target-tolerance-deg 5 \
  --yaw-sign 1 \
  --zero-frames 15 \
  --confirm-frames 15 \
  --max-reprojection-error-px 2 \
  --minimum-markers 2 \
  --minimum-valid-ratio 0.70
```

输出：

- `episode_*/cube_pose.jsonl`：逐帧 6D pose 和相对 yaw。
- `episode_*/cube_pose_summary.json`：单条轨迹的 `VALID/INVALID` 结果。
- `cube_yaw_session_summary.json`：整个 session 的汇总。

后续训练只使用 `VALID` episode。

## 5. 已有 vertical-CW ACT 最佳版本

这是旧 vertical clockwise rotation 的已验证回退模型，不用于新 horizontal-CCW 数据录制。
运行前只保留 Terminal 1 driver，关闭 recorder、MediaPipe 和 pose preview。

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_fingertip_vertical_cw_all_replay_long_history_act/training/checkpoints/003000/pretrained_model \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG \
  --side right \
  --device cuda \
  --rate 30 \
  --n-action-steps 30 \
  --max-range-step 30 \
  --ema-alpha 0.35 \
  --current-limit 160 \
  --speed-limit 160 \
  --max-target-delta 210 \
  --max-raw-overshoot 50 \
  --max-active-seconds 20 \
  --state-stale-seconds 0.5 \
  --log-period 0.25 \
  --reset-range-step 50 \
  --ignore-touch \
  --enable-motion
```

摄像机窗口获得焦点后，按 `SPACE` 开始/停止发布，按 `R` 复位，按 `Q`/`ESC` 退出。



cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_flipping_act_v3_posttrain_20260729_more_v2/training/checkpoints/020000/pretrained_model \
  --camera-index 0 \
  --rate 30 \
  --n-action-steps 30 \
  --max-range-step 20 \
  --ema-alpha 0.10 \
  --thumb-final-push-offset 10 \
  --current-limit 180 \
  --speed-limit 150 \
  --max-target-delta 210 \
  --max-raw-overshoot 50 \
  --state-stale-seconds 2.0 \
  --reset-tolerance 12 \
  --max-active-seconds 0 \
  --auto-stop-endpoint-profile artifacts/g20_flipping_act_full_context_momentum_v3/dataset/g20_endpoint_profiles.json \
  --auto-stop-endpoint-tolerance 12 \
  --auto-stop-endpoint-confirm-frames 10 \
  --auto-stop-min-active-seconds 8 \
  --auto-stop-departure-delta 20 \
  --hold-on-disarm \
  --ignore-touch \
  --enable-motion