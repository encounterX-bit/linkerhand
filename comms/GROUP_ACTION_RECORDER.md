# Grouped MediaPipe + official-GUI action recording

This recorder is read-only with respect to ROS. It subscribes to the official
GUI command and hardware state topics but never creates a command publisher.
Camera 2 records the human hand/MediaPipe stream and camera 0 photographs the
physical G20 at saved waypoints.

The left preview draws the 21 MediaPipe landmarks and finger skeleton for live
quality control. Green means a fresh detection; yellow means a held/stale
detection. Saved human JPEGs remain raw and unannotated, while landmark arrays
are stored separately in `samples.jsonl`.

## Terminal 1: G20 driver

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right \
  --can can0 \
  --is_touch true \
  --touch-fingers with-palm
```

## Terminal 2: official hardware GUI

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 launch gui_control gui_control.launch.py \
  hand_type:=right \
  hand_joint:=G20 \
  is_touch:=false \
  show_pressure_diagram:=false
```

Set conservative GUI speed/torque before moving the physical hand.

## Terminal 3: grouped recorder

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source /home/zhaoyan-qian/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate

python -m src.comms.group_action_recorder \
  --output-dir data/action_groups \
  --session-name g20_primitive_collection \
  --mediapipe-camera 2 \
  --robot-camera 0 \
  --side right
```

## Keyboard cycle

1. Press SPACE to create group 000 and enter HUMAN READY.
2. Press M to start human take 0, perform the gesture once, then press M again
   to stop and save take 0.
3. Repeat M/start, gesture, M/stop 3--5 times in the same group. Frames are saved
   only while a take is active, and each stopped take gets an exact boundary.
4. With no take recording, press SPACE to finish the human phase and enter robot
   capture. H is no longer used.
5. Pose the G20 using the official GUI.
6. Press S to save the current GUI command, measured hardware state, and robot
   camera photo. Save start, optional intermediate poses, and end pose this way.
7. Press SPACE to finalize this group and create the next group in HUMAN READY.
8. Press X or ESC to exit. A partially recorded last group is marked incomplete.

Retry keys operate only on the current group:

- Q archives all current human takes/images and returns to HUMAN READY. Existing
  robot waypoints are kept.
- E archives all current robot waypoints/photos and enters ROBOT capture with an
  empty waypoint list. Existing human data is kept. If HUMAN recording was
  active, E stops it first.
- Archived attempts remain recoverable under
  `group_NNN/revisions/human_retry_NNN` or `robot_retry_NNN`.

SPACE cannot finalize a group until at least one S waypoint exists. The S key
also refuses stale/missing GUI commands or hardware state.

## Output layout

```text
data/action_groups/<timestamp>_g20_primitive_collection/
  session.json
  group_000/
    group.json
    human/
      samples.jsonl
      images/000000.jpg ...
    robot/
      waypoints.json
      images/waypoint_000.jpg ...
  group_001/
  ...
```

`human/samples.jsonl` retains all camera frames, fresh/held flags, raw world
landmarks, and normalized hand-base landmarks. `robot/waypoints.json` retains
both GUI command and measured state. Its `trajectory_waypoints` field is ready
for interpolation after durations have been reviewed.

## Unified offline analysis

After the session, run:

```bash
python -m src.comms.analyze_action_groups \
  --session data/action_groups/REPLACE_WITH_SESSION_DIRECTORY
```

If M markers exist, they define repetition boundaries. Otherwise the analyzer
generates candidate takes from motion energy and neutral pauses. Each group gets
an `analysis/` directory containing `human_take_NNN.npy`, an extracted
`trajectory_waypoints.json`, and `analysis.json`. Review all automatically
segmented takes before importing them into the final 50-class library.

## Replay a recorded robot trajectory

First run the numeric preview. It creates no ROS publisher and cannot move the
hand:

```bash
python -m src.comms.replay_action_group \
  --group data/action_groups/current_actions/02_thumb_fold_inward
```

The command prints `REPLAYABLE` only when the group is complete and every
recorded active-joint command/state error is at most 10 SDK ticks. A larger
error is printed as `BLOCKED` with its waypoint and joint number.

For physical replay, leave the SDK driver running but close the official GUI;
otherwise the GUI and replay tool would both publish commands. Then run:

```bash
export HW_ENABLE_TOKEN=1  # the human operator must set this in this terminal

python -m src.comms.replay_action_group \
  --group data/action_groups/current_actions/02_thumb_fold_inward \
  --robot-camera 0 \
  --side right \
  --enable-motion
```

The replay window starts disarmed. SPACE performs the trajectory once, R makes
a step-limited return to the open pose, and Q/ESC exits without another motion.
Each command changes an active joint by at most five SDK ticks per frame. Replay
stops if state messages become stale or the measured hand falls too far behind
the previous command.
