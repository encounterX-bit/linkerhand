# G20：cube 在四指后方 → 向右 flip 数据采集

目标：把已有向后 flip policy 只用作前置动作，将 cube 推进四根手指后方；然后切换到
独立的 MediaPipe teleop，冻结当下 DexHand 大拇指，只控制其余四根手指完成向右
flip。数据使用独立 task ID，不和现有向后 flip/水平旋转数据混合。

同一时间只能运行一个真机 command publisher。前置 policy 完成后先停止并退出它，
再启动下面的 Terminal 3；不要让 ACT runner 和 teleop 同时运行。

## 0：准备每条 episode 的起点

1. 用现有向后 flip policy 将 cube 推到四根手指后方。
2. 按 `SPACE` 停止旧 policy，再按 `Q` 退出，保持真机当前姿态。
3. 确认旧 policy 已退出后，再启动下面的 recorder 和 teleop。

第一次先只录 5 条检查起点、方向和大拇指是否稳定，确认后再补到 30–50 条。

## Terminal 1：G20 driver

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

## Terminal 2：独立的 ACT recorder

先启动 Terminal 2，看到 `waiting for teleop trigger` 后保持运行。

```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

python3 scripts/linkerhand_g20_touch_recorder.py \
  --hand-type right \
  --task-id cube_right_flip_from_behind_fingers_v1 \
  --trial-name cube_right_flip_from_behind_fingers_v1 \
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

## Terminal 3：MediaPipe 四指 + `F` 冻结大拇指

这里不使用 `--reset-on-start` 或 `--reset-before-arm`，避免破坏旧 policy 已经准备
好的 cube/手指起点。`--reset-after-disarm` 只在一条 episode 保存后复位四指；
被冻结的大拇指保持启动位置。

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate

export HW_ENABLE_TOKEN=1

python -m src.comms.action_library_phase_teleop \
  --library data/action_library/g20_right/core_actions_v1 \
  --camera-index 2 \
  --side right \
  --tracking-mode live-pose \
  --control-mode hybrid-fingers \
  --hybrid-unlocked-fingers \
  --freeze-thumb-on-start \
  --thumb-roundtrip-key 6 \
  --thumb-roundtrip-source-action 2 \
  --rate 30 \
  --max-range-step 10 \
  --max-state-lead 30 \
  --reset-after-disarm \
  --reset-tolerance 12 \
  --reset-timeout 5 \
  --reset-confirm-frames 3 \
  --current-limit 20 \
  --speed-limit 50 \
  --enable-motion
```

## 每条 episode 的按键顺序

按键焦点必须在 Terminal 3 的 MediaPipe 摄像机窗口。

1. 保持 `DISARMED`，确认 cube 已位于四根手指后方。
2. 启动时程序已经读取真机状态，确认画面/终端显示
   `THUMB FROZEN ON START at q0/q5/q10/q15=[...]`。
3. 按 `T`，画面显示 `FULL MEDIAPIPE TELEOP`；大拇指仍保持冻结。
4. 保持人手起始姿态，按 `SPACE`。Terminal 2 开始录制。
5. 用四根手指完成向右 flip；大拇指四个通道保持在按 `F` 时的位置。
6. 终点稳定保持约 2 秒，再按 `SPACE` 停止并保存 episode；随后四指自动复位。
7. 等待 `RESET COMPLETE ... with thumb frozen`。大拇指继续冻结，重新摆好 cube，
   下一条直接按 `SPACE`；
   只有想换一个拇指保持位置时才需要再按 `F` 解除、调整后重新按 `F`。

`--freeze-thumb-on-start` 会在程序连接到真机后立刻锁住当下位置，不需要第一次手动
按 `F`。`F` 仍可在 ARMED 或 DISARMED 时解除/重新冻结。录制中不要反复切换 `F`；
如果锁错位置，先用 `SPACE` 停止，删除失败 episode，再重新准备起点。

其他按键：

- `F`：冻结/解除当下 DexHand 大拇指 `q0/q5/q10/q15`。
- `6`：读取按键瞬间的真机位置，只用 `q0/q5/q10/q15` 在动作 2 中找最近帧；
  从该帧执行到动作 2 终点，再倒放回动作 2 第 1 帧。其余四指的屈伸和侧摆全部固定在
  按键瞬间的 measured state。即使 `F` 已冻结，数字 6 在播放期间也会临时接管大拇指；
  `F` 锁不会丢失，按 `0`、`SPACE`、`D` 或 `R` 退出当前数字动作后恢复。
- `T`：无数字动作时切换 FULL MEDIAPIPE/AUTO。
- `SPACE`：开始 episode；再次按下会停止保存，并自动复位非拇指关节。
- `R`：仅在 DISARMED 时手动复位非拇指关节，不启动 recorder。
- `D`：仅在 DISARMED 时删除当前 session 最新一条已保存 episode。
- `Q` / `ESC`：退出并停止 recorder trigger。

`SPACE`、`D` 和 episode reset 都不会解除冻结；带 reset 的命令只复位其他关节。
MediaPipe 丢手或 ROS joint state 过期仍会解除冻结并 DISARM，这是故障安全行为。
