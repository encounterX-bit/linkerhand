```bash

sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```


# T1
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right --can can0 --is_touch true --touch-fingers with-palm
```

# T2 （press s to start/terminate）
```bash

cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash && source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate && export HW_ENABLE_TOKEN=1
.venv/bin/python -m src.comms.camera_to_linkerhand \
  --source webcam --camera-index 2 --side right \
  --sdk-hand-joint g20 --hardware-map g20-sim \
  --show-camera --absolute --motion-key-toggle \
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
  --hardware-base-gain 1.95 \
  --hardware-base-gains "0.72,1.05,1.05,0.63" \
  --hardware-spread-gain 0.64 \
  --hardware-spread-signs "0.35,1.00,-0.15,-1.00" \
  --hardware-tip-gain 0.80 \
  --hardware-tip-gains "1.34,0.85,1.20,1.27" \
  --hardware-thumb-tip-gain 1.12 \
  --hardware-thumb-tip-offset -27 \
  --hardware-thumb-roll-gain 0.96 \
  --hardware-thumb-roll-offset -24 \
  --hardware-thumb-base-gain 0.69 \
  --hardware-thumb-base-offset 20 \
  --hardware-thumb-abd-gain 0.72 \
  --hardware-thumb-abd-offset 0 \
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
  --enable-motion \
  --max-range-step 5 \
  --log-period 0.25 \
  --log-sim-position
```

# T3 (press space to record/terminate)
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

python3 scripts/linkerhand_g20_touch_recorder.py \
  --hand-type right \
  --task-id orientation_grasp_test \
  --rate 30 \
  --camera 2 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG \
  --ros-trigger \
  --require-state \
  --output-dir /home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt/data

```

# Train visual ACT (camera episodes only; grasp_cube is skipped automatically)
```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt

.venv/bin/python scripts/train_g20_visual_act.py \
  --stage all \
  --steps 10000 \
  --batch-size 8 \
  --chunk-size 30 \
  --n-action-steps 10
```

The script uses camera 2 images + 20-D G20 state to predict 20-D absolute
command chunks. It automatically switches to the installed LeRobot training
environment, holds out three complete episodes for validation, and never sends
commands to hardware.

# Check visual ACT on held-out episodes (offline; does not move the hand)
```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt

./scripts/evaluate_g20_visual_act.py --samples-per-episode 20
```

Open these two files after it finishes:

- `artifacts/g20_visual_act/evaluation/010000/validation_preview.png`
- `artifacts/g20_visual_act/evaluation/010000/validation_metrics.png`

# T4a: preview visual ACT on live camera (dry-run, hand does not move)

Keep T1 running. Stop T2 and T3 first so they do not compete for the command
topic or camera 2.

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --camera-index 2
```

Check that the window shows the same camera angle as the training data. The
terminal should repeatedly print `preview`, with no `PUBLISH` lines.

# T4b: first visual ACT hardware trial

Place the object in a familiar training position, start from the demonstrated
open-hand pose, and keep the camera view unchanged. The process starts
**DISARMED**. Focus the camera window and press SPACE to move; press SPACE again
to stop immediately. Q/ESC exits. It automatically disarms after 10 seconds.

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --camera-index 2 \
  --rate 5 \
  --max-range-step 2 \
  --current-limit 20 \
  --speed-limit 20 \
  --max-active-seconds 10 \
  --enable-motion
```

The runner clamps commands to 0..255, fixes reserved indices 11..14 to 255,
disarms on stale ROS state, excessive raw model output, or a target more than 80
ticks from the observed pose. It does not automatically open the hand on exit.

# T4c: visual ACT chunk execution after the first slow safety trial

This restores the 10-action chunks used during training instead of repeatedly
executing only the first predicted action. SPACE still starts/stops publishing.

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --camera-index 2 \
  --rate 10 \
  --n-action-steps 10 \
  --max-range-step 10 \
  --ema-alpha 0.4 \
  --current-limit 20 \
  --speed-limit 60 \
  --max-target-delta 100 \
  --max-active-seconds 60 \
  --enable-motionjhghg\

```

# T5: ACT autonomous attempts + human 0/0.5/1 rating

Run T1 only. Stop T2 and T3 because this runner owns camera 2 and records its
own images, states, and commands.

Controls in the camera window:

- SPACE: start one autonomous attempt
- SPACE again: stop the attempt
- `0`: failed attempt
- `5`: partial attempt (`0.5`)
- `1`: successful attempt
- Q/ESC: exit

After an attempt stops (manually, automatically, or through a safety guard), a
rating is required before the next SPACE can start another attempt. After the
rating, the runner automatically ramps the hand back to the standard G20 open
pose. Wait for `RESET complete`, reposition the object, then press SPACE again.

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_visual_act_finetune_01/training/checkpoints/002000/pretrained_model \
  --camera-index 2 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG \
  --side right \
  --device cuda \
  --rate 10 \
  --n-action-steps 10 \
  --max-range-step 25 \
  --ema-alpha 0.3 \
  --current-limit 20 \
  --speed-limit 80 \
  --max-target-delta 100 \
  --max-raw-overshoot 20 \
  --max-active-seconds 0 \
  --state-timeout 5 \
  --state-stale-seconds 0.5 \
  --log-period 0.5 \
  --record-rated-attempts \
  --rated-output-dir ~/Desktop/Jacky/sims/linker-hand-teleopt/data/self_imitation \
  --record-rate 30 \
  --jpeg-quality 92 \
  --contact-on-threshold 20 \
  --contact-off-threshold 10 \
  --min-contact-fingers 3 \
  --require-thumb-contact \
  --contact-hold-seconds 0.5 \
  --touch-stale-seconds 0.5 \
  --require-touch-for-score-one \
  --stop-on-contact-success \
  --reset-after-rating \
  --reset-range-step 25 \
  --reset-tolerance 3 \
  --enable-motion
```

Rated episodes are saved under `data/self_imitation/<timestamp>_act_self_imitation/`.
The touch gate requires the thumb plus at least two of index/middle/ring/little
to remain above the hysteretic contact thresholds for 0.5 seconds. The attempt
then stops automatically so the policy does not keep squeezing. If the human
presses `1` without satisfying that gate, `episode.json` preserves
`human_quality_score: 1` but sets `quality_score: 0.5`; it will not be copied into
ACT behavior cloning as a good action. Every sample now includes `mass_values`
and `contact_6`. Score 0 and 0.5 episodes remain on disk as negative examples for
a later quality/reward model. The behavior-cloning fine-tune below still uses
only touch-confirmed score 1 attempts plus the original human demonstrations;
directly cloning bad actions would teach ACT to repeat them.

# Fine-tune ACT from checkpoint 010000

Collect at least 10 score-1 attempts before running this. Use a new artifact
directory for every fine-tune; do not overwrite the original model.

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt

.venv/bin/python scripts/train_g20_visual_act.py \
  --stage all \
  --artifact-root artifacts/g20_visual_act_finetune_01 \
  --repo-id linkerhand_g20_orientation_grasp_visual_finetune_01 \
  --finetune-from artifacts/g20_visual_act/training/checkpoints/010000/pretrained_model \
  --finetune-learning-rate 3e-6 \
  --min-rated-score 1.0 \
  --steps 3000 \
  --save-freq 500 \
  --batch-size 8 \
  --chunk-size 30 \
  --n-action-steps 10
```

Offline-check the fine-tuned checkpoint:

```bash
./scripts/evaluate_g20_visual_act.py \
  --artifact-root artifacts/g20_visual_act_finetune_01 \
  --samples-per-episode 20
```

If validation improves, test it by adding this to the T5 command:

```bash
--checkpoint-dir artifacts/g20_visual_act_finetune_01/training/checkpoints/002000/pretrained_model
```

First fine-tune result (5 score-1 attempts, 2908 frames): checkpoint `002000`
was best on the original held-out episodes. Active-joint MAE improved from
15.09 to 14.79 SDK ticks; this is a small improvement, so continue collecting
human-corrected or score-1 attempts rather than increasing hardware limits.

 # replay
 ```bash
 python3 scripts/linkerhand_g20_replay.py \
  ~/Desktop/Jacky/sims/linker-hand-teleopt/data/20260710_112549_teleop_demo \
  --episode 1 --enable-motion

```

cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source .venv/bin/activate

python -m src.viz.app \
  --source webcam \
  --camera-index 2 \
  --side right \
  --show-camera


# 

cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
export HW_ENABLE_TOKEN=1

~/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_visual_act_finetune_01/training/checkpoints/002000/pretrained_model \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-fourcc MJPG \
  --side right \
  --device cuda \
  --rate 20 \
  --n-action-steps 15 \
  --max-range-step 35 \
  --ema-alpha 0.3 \
  --current-limit 120 \
  --speed-limit 80 \
  --max-target-delta 100 \
  --max-raw-overshoot 50 \
  --max-active-seconds 5 \
  --state-timeout 5 \
  --state-stale-seconds 0.5 \
  --log-period 0.5 \
  --record-rated-attempts \
  --rated-output-dir ~/Desktop/Jacky/sims/linker-hand-teleopt/data/self_imitation \
  --record-rate 30 \
  --jpeg-quality 92 \
  --contact-on-threshold 20 \
  --contact-off-threshold 10 \
  --min-contact-fingers 3 \
  --require-thumb-contact \
  --contact-hold-seconds 0.5 \
  --touch-stale-seconds 0.5 \
  --require-touch-for-score-one \
  --stop-on-contact-success \
  --reset-after-rating \
  --reset-range-step 25 \
  --reset-tolerance 3 \
  --enable-motion










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
  --startup-thumb-pose "116,253,234,118" \
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
  --reset-on-start \
  --reset-after-disarm \
  --reset-tolerance 12 \
  --reset-timeout 5 \
  --reset-confirm-frames 3 \
  --current-limit 20 \
  --speed-limit 50 \
  --enable-motion
