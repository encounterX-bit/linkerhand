# G20 水平逆时针 90°：键盘动作库 + MediaPipe 数据采集

目标：使用数字键选择动作库轨迹，同时用 MediaPipe 控制四根手指，录制 cube
水平逆时针旋转 `90°` 的 ACT demonstration。相机 0 录制机器人和 cube，相机 2
追踪人手。

当前流程会在录制结束后离线计算 cube 6D pose/yaw。yaw 暂时用于筛选数据，还没有
作为 ACT observation 输入，也没有使用 IK。

同一时间只能运行一个真机 command publisher。开始前关闭官方 GUI、固定动作回放、
其他 teleop、ACT runner 和 cube pose preview。

## Terminal 1：启动 G20 driver

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

## Terminal 2：启动 ACT recorder

先启动 Terminal 2，看到 `waiting for teleop trigger` 后保持运行，再启动 Terminal 3。

```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

python3 scripts/linkerhand_g20_touch_recorder.py \
  --hand-type right \
  --task-id cube_yaw_ccw90_keyboard_hybrid_v1 \
  --trial-name cube_yaw_ccw90_keyboard_hybrid_v1 \
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

## Terminal 3：数字键动作库 + MediaPipe teleop

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
  --startup-thumb-pose "116,253,254,118" \
  --rate 30 \
  --lock-margin 0.015 \
  --no-a23-spread-routing \
  --a3-contact-assist \
  --a3-contact-activate-phase 0.08 \
  --a3-contact-release-phase 0.02 \
  --a3-contact-confirm-frames 2 \
  --a3-contact-threshold-scale 1.60 \
  --a3-contact-competition-slack 0.015 \
  --finger-base-blend 0.15 \
  --finger-tip-blend 0.20 \
  --finger-base-residual-limit 20 \
  --finger-tip-residual-limit 25 \
  --max-range-step 10 \
  --max-state-lead 30 \
  --manual-blend-frames 8 \
  --a4-thumb-tip-gate \
  --a4-left-align-tolerance 5 \
  --a4-left-align-confirm-frames 3 \
  --a4-thumb-tip-tolerance 5 \
  --a4-thumb-tip-confirm-frames 3 \
  --reset-on-start \
  --reset-after-disarm \
  --reset-tolerance 12 \
  --reset-timeout 5 \
  --reset-confirm-frames 3 \
  --current-limit 20 \
  --speed-limit 50 \
  --enable-motion
```

### MediaPipe + 动作库 teleop 流程图

```mermaid
flowchart TD
    CAM["相机 2：人手画面"] --> MP["MediaPipe<br/>21 个手部 landmarks"]
    MP --> FEATURE["归一化手型与动作 phase 特征"]
    MP --> DIRECT["MediaPipe → G20<br/>直接关节映射"]

    LIB["动作库<br/>人手模板 + 20 维 DexHand 轨迹"] --> MATCH["动作 matcher<br/>识别动作 ID 与当前 phase"]
    FEATURE --> MATCH
    MATCH --> LIBTARGET["按 phase 插值<br/>得到动作库 20 维 target"]

    LIBTARGET --> MODE{"当前控制模式"}
    DIRECT --> MODE

    MODE -->|"AUTO hybrid"| AUTO["动作库控制拇指和动作 phase<br/>MediaPipe 修正四指屈伸<br/>未锁定时四指可直接跟随"]
    MODE -->|"数字动作键"| MANUAL["完整播放动作库轨迹<br/>暂时忽略 MediaPipe"]
    MODE -->|"手动动作中按 T"| MIX["动作库只控制拇指<br/>MediaPipe 控制四指屈伸和侧摆"]
    MODE -->|"没有手动动作时按 T"| FULL["FULL MEDIAPIPE<br/>全部活动关节直接映射<br/>暂停 matcher、A3 assist 和 A4 gate"]
    MODE -->|"按 0"| AUTO

    AUTO --> THUMB
    MANUAL --> THUMB
    MIX --> THUMB
    FULL --> THUMB

    THUMB{"F 拇指冻结是否开启？"}
    THUMB -->|"是"| HOLD["用锁定值覆盖<br/>q0 / q5 / q10 / q15"]
    THUMB -->|"否"| FULLCHECK{"是否为 FULL MEDIAPIPE？"}
    HOLD --> FULLCHECK

    FULLCHECK -->|"是：旁路 A4 gate"| SAFE
    FULLCHECK -->|"否"| GATE{"当前是否为动作 4？"}
    GATE -->|"是且 A4 gate 有效"| A4["严格分阶段<br/>向左对准 → 收拇指尖 → 向右转"]
    GATE -->|"否"| SAFE
    A4 --> SAFE["安全层<br/>单帧步长 ≤ 10<br/>实测状态领先 ≤ 30<br/>关节与碰撞范围保护"]
    SAFE --> HAND["LinkerHand G20 真机"]

    SPACE["SPACE"] --> REC{"当前 episode 状态"}
    REC -->|"DISARMED"| START["通知 Terminal 2<br/>开始录制并允许 teleop 发布"]
    REC -->|"ARMED"| STOP["通知 Terminal 2 停止并保存<br/>随后自动 reset"]
    START -. "同步采集图像、关节和触觉" .-> DATA["ACT 数据集"]
    STOP -. "保存 episode" .-> DATA
```

当前命令使用 `--no-a23-spread-routing`，所以“四指并拢时只能动作 2”的 A2/A3
硬路由没有参与这条数据采集流程；A3 contact assist 仍只在 AUTO matcher 路径中工作。

## 每条 episode 的操作顺序

按键焦点必须在 Terminal 3 的 MediaPipe 摄像机窗口。

1. 等待画面显示 `RESET COMPLETE`，此时为 `DISARMED`。
2. 把 cube 放在统一起点，保持 face 5 朝向相机 0。
3. 按需要的数字动作，例如 `2`、`3` 或 `4`，先将动作排队。
4. 按 `T`，确认显示：

   ```text
   MANUAL THUMB + MEDIAPIPE FOUR FINGERS
   ```

5. 人手放在相机 2 中并保持起始姿态，然后按 `SPACE`。Terminal 2 开始录制。
6. 录制期间，大拇指使用当前数字动作的轨迹；四根手指实时跟随 MediaPipe。
7. 需要切换大拇指动作时，按新的数字键，然后重新按 `T`。
8. 将 cube 水平逆时针旋转到约 `+90°`，在终点稳定保持至少 2 秒。
9. 按 `SPACE`：Terminal 2 停止并保存 episode，Terminal 3 自动复位。
10. 重新摆好 cube，重复录制。先录 10 条检查，再补到 30–50 条。

补充按键：

- `F`：冻结/解除按键瞬间的 DexHand 大拇指 `q0/q5/q10/q15`。冻结会跨越
  `SPACE` 停止、`D` 删除和 `--reset-after-disarm` 保持；reset 只移动其余关节。
  需要恢复大拇指控制时必须再按一次 `F`。
- `--freeze-thumb-on-start`：Terminal 3 启动并读到真机 state 后，自动把当时的大拇指
  位置作为基准，因此第一条 episode 不必先按 `F`。
- `--startup-thumb-pose "116,253,254,118"`：使用之前调好并录进
  `20260724_165206_cube_right_flip_from_behind_fingers_v1` 的固定拇指姿态。
  四个绝对 SDK 值依次是 `q0/q5/q10/q15`，不是动作 2 的姿态，也不依赖程序启动时
  拇指当前在 0 还是其他位置。startup reset 会先移动到这组位置，再保持冻结。
  等画面显示 `RESET COMPLETE` 后再开始录制。
- `R`：仅在 DISARMED 时手动开始一次 reset，不启动 ACT recorder；若 `F` 已锁定，
  只复位非拇指关节。
- `0`：返回 AUTO hybrid teleop。
- `D`：仅在 `DISARMED` 且复位完成后，删除当前 session 最新一条 episode。
- `Q` / `ESC`：退出并停止录制。

注意：按 `T` 后，四指屈伸和侧摆 `q1..q4/q6..q9/q16..q19` 都由 MediaPipe
控制。数字 6 是例外：它固定四指并忽略 `T`，只执行动作 2 的拇指往返。

## 录完后：离线计算 cube 6D pose/yaw

先关闭 Terminal 2 和 Terminal 3，再运行：

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source .venv/bin/activate

CCW_SESSION=$(ls -dt \
  data/act_demos/*cube_yaw_ccw90_keyboard_hybrid_v1 \
  | head -1)

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

检查输出：

- `episode_*/cube_pose.jsonl`：逐帧 cube 6D pose 和相对 yaw。
- `episode_*/cube_pose_summary.json`：单条 episode 的 `VALID/INVALID` 结果。
- `cube_yaw_session_summary.json`：整个 session 的统计结果。

当前训练器尚未读取 `cube_pose.jsonl` 作为 observation，也不会自动排除
`INVALID` episode；若这里发现失败轨迹，需要像下面排除 `episode_000` 一样增加
`--exclude-source-episode`。

## 训练 21:29 数据：history + momentum ACT

这条命令只读取
`20260724_212905_cube_yaw_ccw90_keyboard_hybrid_v1`，不会混入之前其他方向的
policy 数据。`episode_000` 长 35.8 秒且静止帧比例明显高于其他轨迹，因此从训练
split 排除；其余 14 条保留，其中固定 3 条作为 validation。

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  scripts/train_g20_visual_act.py \
  --stage all \
  --data-root data/act_demos/20260724_212905_cube_yaw_ccw90_keyboard_hybrid_v1 \
  --include-task-id cube_yaw_ccw90_keyboard_hybrid_v1 \
  --artifact-root artifacts/g20_cube_yaw_ccw90_history_momentum_v1 \
  --repo-id linkerhand_g20_cube_yaw_ccw90_history_momentum_v1 \
  --task "rotate the tagged cube 90 degrees counter-clockwise using visual and motion history" \
  --history-frame-offsets "90,72,54,36,18,0" \
  --state-history-offsets "90,60,30,0" \
  --val-episodes 3 \
  --chunk-size 30 \
  --n-action-steps 5 \
  --momentum-weight 0.2 \
  --momentum-deadband 0.01 \
  --momentum-margin 0.005 \
  --exclude-source-episode "20260724_212905_cube_yaw_ccw90_keyboard_hybrid_v1/episode_000" \
  --steps 20000 \
  --batch-size 8 \
  --num-workers 4 \
  --save-freq 2000 \
  --device cuda \
  --amp
```

这里解决“同一个单帧 observation 有多个 action”的主要输入是：

- 六张历史图像覆盖最近 3 秒：`90,72,54,36,18,0`。
- 四组关节历史明确提供运动方向：`90,60,30,0`。
- momentum loss 使用权重 `0.2`，惩罚预测动作相对示范方向发生反转。

### History + momentum ACT 训练与真机推理流程图

```mermaid
flowchart TD
    RAW["原始 demonstrations<br/>相机 0 图像 + 20 维真机 state/action"] --> CLEAN["数据筛选与切分<br/>排除 episode_000<br/>其余 14 条：11 train + 3 validation"]
    CLEAN --> SAMPLE["在 episode 中采样当前时刻 t"]

    SAMPLE --> IH["取 6 张历史图像<br/>t-90, t-72, t-54, t-36, t-18, t"]
    IH --> MOSAIC["按时间从旧到新拼成 2×3 mosaic<br/>覆盖最近约 3 秒"]
    MOSAIC --> VISION["视觉 backbone<br/>把历史 mosaic 编码为 image tokens"]

    SAMPLE --> SH["取 4 组关节历史<br/>q(t-90), q(t-60), q(t-30), q(t)"]
    SH --> STATE["拼接为 80 维 observation.state<br/>显式提供过去运动方向"]
    STATE --> PROPRIO["线性投影为 proprioceptive token"]

    SAMPLE --> DEMO["监督目标<br/>未来 30 帧 × 20 维示范动作 chunk"]
    DEMO --> POST["CVAE posterior encoder<br/>训练时由 state + 示范 chunk 采样 latent z"]
    VISION --> ACT["ACT Transformer encoder / decoder<br/>30 个 action queries"]
    PROPRIO --> ACT
    POST --> ACT
    ACT --> PRED["预测 30 帧 × 20 维<br/>绝对关节动作 chunk"]

    PRED --> BC["原始 ACT behavior loss<br/>action L1 + 10 × CVAE KL"]
    DEMO --> BC

    PRED --> VP["预测相邻速度<br/>v_pred(t) = a_pred(t+1) - a_pred(t)"]
    DEMO --> VD["示范相邻速度<br/>v_demo(t) = a_demo(t+1) - a_demo(t)"]
    VP --> RULE["归一化空间中仅 abs(v_demo) > 0.01 时生效<br/>逐步读取示范方向，允许 demonstration 中真实反向<br/>作用于除 q11–q14 外的活动关节"]
    VD --> RULE
    RULE --> MOM["Momentum direction hinge loss<br/>要求沿示范方向的预测速度至少达到<br/>min(abs(v_demo), 0.005)"]

    BC --> TOTAL["总损失<br/>L_total = L1 + 10 × KL + 0.2 × L_direction"]
    MOM --> TOTAL
    TOTAL --> OPT["反向传播与优化<br/>batch = 8，训练 20K steps"]
    OPT --> CKPT["20K checkpoint<br/>包含学到的 history + momentum 行为"]

    subgraph DEPLOY["真机部署：receding-horizon ACT"]
        BUFFER["持续缓存相机图像与实测关节"] --> OBS["使用相同 offsets<br/>构造 2×3 图像 mosaic + 80D state"]
        OBS --> INFER["20K ACT 前向推理<br/>没有示范 chunk，latent z 固定为 0"]
        INFER --> CHUNK["得到新的 30×20 动作 chunk"]
        CHUNK --> FIRST["只执行前 5 步<br/>n-action-steps = 5"]
        FIRST --> SAFE["安全限制<br/>单帧步长、state lead、关节范围"]
        SAFE --> G20["LinkerHand G20"]
        G20 --> BUFFER
    end

    CKPT --> INFER
```

这两个机制解决的是不同问题：

- `history` 是模型的实际输入，训练和真机推理都要使用；它让相同的当前画面可以根据
  前 3 秒图像与关节运动方向得到不同动作。
- `momentum` 只在训练时修改 loss，要求预测 chunk 的局部运动方向跟随 demonstration；
  它的效果已经写入 checkpoint，真机运行时不需要也没有 `--momentum-*` 参数。
- 真机端每次预测 30 帧，但只执行前 5 帧就重新观察和规划，因此仍能根据 cube 的实际
  变化闭环修正。这条配置没有使用 temporal ensemble，也没有加入 IK/FK。

训练完成后的最终 checkpoint：

```text
artifacts/g20_cube_yaw_ccw90_history_momentum_v1/training/checkpoints/020000/pretrained_model
```

## 真机查看 20K checkpoint 效果

先关闭 recorder、MediaPipe teleop 和官方 GUI，只保留 G20 driver。这个模型的示范
起点使用固定拇指 `q0/q5/q10/q15=[116,253,254,118]`；如果灵巧手还保持录制结束后的
reset 姿态，可以直接测试。不要先按 `R`，因为当前 ACT runner 的 `R` 会复位到通用
open pose，而不是这次数据的固定拇指起点。

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_cube_yaw_ccw90_history_momentum_v1/training/checkpoints/020000/pretrained_model \
  --camera-index 0 \
  --rate 30 \
  --n-action-steps 5 \
  --max-range-step 10 \
  --ema-alpha 0.10 \
  --current-limit 20 \
  --speed-limit 50 \
  --max-target-delta 180 \
  --max-raw-overshoot 40 \
  --state-stale-seconds 2.0 \
  --max-active-seconds 20 \
  --ignore-touch \
  --enable-motion
```

启动后保持 `DISARMED` 至少 3 秒，让 `[90,72,54,36,18,0]` 图像历史和
`[90,60,30,0]` 关节历史填满。摆好 cube 后让摄像机窗口获得焦点，再按 `SPACE`
开始；动作不对时立即再按 `SPACE` 停止。首次测试保留示范使用的
`current=20/speed=50/max-step=10`，先判断轨迹方向和阶段切换，不要同时提高力度和速度。

## 30K 推入手内，然后按 P 自动切换到 20K 逆时针旋转

这个模式会同时预加载两个 checkpoint。开始时运行旧的 30K flipping policy；cube
被推到手指后按 `P`，程序按真机反馈限速移动到 20K 数据的起点，稳定后原地收集 3 秒
历史，再自动 arm 并执行 20K policy。切换完成后不会再回到 30K。

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_flipping_act_full_context_momentum_v3/training/checkpoints/030000/pretrained_model \
  --policy-handoff-checkpoint-dir artifacts/g20_cube_yaw_ccw90_history_momentum_v1/training/checkpoints/020000/pretrained_model \
  --policy-handoff-start-pose "116,251,253,246,254,253,183,139,103,34,253,255,255,255,255,118,247,247,246,248" \
  --policy-handoff-n-action-steps 5 \
  --policy-handoff-range-step 10 \
  --policy-handoff-tolerance 12 \
  --policy-handoff-confirm-frames 3 \
  --policy-handoff-warmup-seconds 3 \
  --policy-handoff-timeout 12 \
  --policy-handoff-current-limit 20 \
  --policy-handoff-speed-limit 50 \
  --camera-index 0 \
  --rate 30 \
  --n-action-steps 30 \
  --max-range-step 20 \
  --ema-alpha 0.10 \
  --thumb-final-push-offset 10 \
  --current-limit 180 \
  --speed-limit 100 \
  --max-target-delta 210 \
  --max-raw-overshoot 40 \
  --state-stale-seconds 2.0 \
  --max-active-seconds 0 \
  --ignore-touch \
  --enable-motion
```

### 30K → P → 20K policy handoff 流程图

```mermaid
flowchart TD
    BOOT["启动 runner<br/>同时加载 30K 和 20K checkpoint"] --> IDLE["DISARMED<br/>保持至少 4 秒，填满 30K 长历史"]
    IDLE --> PLACE["摆好 cube，按 SPACE"]
    PLACE --> P30["30K ARMED / PUBLISHING<br/>每次执行 30 步后重规划"]
    P30 --> PUSH["30K 将 cube 推入手指<br/>仅此阶段使用 thumb-final-push-offset = 10"]
    PUSH --> READY{"cube 已正确进入手指？"}
    READY -->|"否"| P30
    READY -->|"是：按 P"| STOP30["立即停止 30K 发布<br/>切换 torque / speed 到 20 / 50"]

    STOP30 --> MOVE["根据 20 维真机反馈<br/>限速移动到 20K 数据起点<br/>每帧最多 10 ticks"]
    MOVE --> ALIGNED{"最大误差 ≤ 12<br/>并连续稳定 3 帧？"}
    ALIGNED -->|"尚未对齐且未超时"| MOVE
    ALIGNED -->|"12 秒超时"| ABORT["中止 handoff<br/>DISARMED 并保持当前实测位置"]
    ALIGNED -->|"是"| WARM["保持 20K 起点<br/>清空旧 history<br/>采集 3 秒新图像与关节历史"]

    WARM --> STABLE{"起点在 warmup 中保持稳定？"}
    STABLE -->|"否"| WARM
    STABLE -->|"是"| P20["自动切换并 arm 20K<br/>使用自身历史输入<br/>每次执行 5 步后重规划"]
    P20 --> ROTATE["20K policy<br/>将 cube 逆时针旋转约 90°"]
    ROTATE --> END["完成后按 SPACE 停止<br/>或由安全条件中止"]

    MOVE -->|"切换中按 SPACE"| ABORT
    WARM -->|"warmup 中按 SPACE"| ABORT
```

按键顺序：

1. 只保留 G20 driver 和这个 runner；摄像机窗口获得焦点。
2. 启动后保持 `DISARMED` 至少 4 秒，让 30K 的长历史先填满。
3. 摆好 cube，按 `SPACE` 执行 30K policy。
4. 等 cube 已经被推到手指后，按一次 `P`。此后不要再按 `SPACE`。
5. 画面依次显示 `MOVING TO 20K START`、`WARMING 20K HISTORY` 和
   `ARMED / PUBLISHING`；最后一个状态表示 20K 已自动开始。
6. 切换移动或历史 warmup 期间如位置不对，按 `SPACE` 立即中止并保持当前真机位置。

20K 起点来自本次训练数据其余 14 条 episode 起始帧的中位数，不是通用 open pose。
`P` 只根据 20 维关节反馈对齐；它无法把掉落或位置错误的 cube 自动恢复到示范中的
视觉状态，因此只在 cube 确实已经进入手指后使用。切换阶段和 20K 阶段会把
torque/speed 降到 `20/50`；30K 的 `thumb-final-push-offset` 不会继续影响 20K。



