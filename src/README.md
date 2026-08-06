# l20-retarget

# simulation
```bash

python -m src.viz.app   --source webcam --camera-index 0 --side right   --show-camera   --fingertip-extend "0,0.12,0.12,0.12,0.05"   --fingertip-lateral "0,-0.03,-0.02,0,0.03"   --fingertip-straighten "0,0.20,0.15,0.15,0.12"   --thumb-gain 1   --thumb-cross-gain 0   --thumb-orient-gain 0.6   --thumb-grasp-gain 0.25   --thumb-tip-gain 1.15   --debug-match --debug-match-period 10
```

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source .venv/bin/activate

python -m src.viz.app   --source webcam --camera-index 0 --side right   --show-camera   --no-filter   --fingertip-extend "0,0.12,0.12,0.12,0.05"   --fingertip-lateral "0,-0.03,-0.02,0.02,0.05"   --fingertip-straighten "0,0.25,0.25,0.20,0.15"   --thumb-gain 1   --thumb-cross-gain 0.28   --thumb-assist-smooth 0.72   --thumb-orient-gain 0.65   --thumb-grasp-gain 0.38   --thumb-tip-gain 1.12
```


# start sdk
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right --can can0 --is_touch true
```

## without sensor 

```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right --can can0 --is_touch false

```


## control gui
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 launch gui_control gui_control.launch.py hand_type:=right hand_joint:=G20 is_touch:=false show_pressure_diagram:=false
```


## sensor gui
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 launch matrix_touch_gui matrix_touch_gui.launch.py \
  hand_type:=right hand_joint:=G20 is_touch:=true
```




# Linkerhand



## Working version





# 防抖动
```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate
export HW_ENABLE_TOKEN=1

.venv/bin/python -m src.comms.camera_to_linkerhand \
  --source webcam --camera-index 0 --side right \
  --sdk-hand-joint g20 --hardware-map g20-sim \
  --show-camera --absolute \
  --no-filter \
  --one-euro-min-cutoff 0.8 \
  --one-euro-beta 0.04 \
  --one-euro-d-cutoff 1.0 \
  --fingertip-extend "0,0.12,0.125,0.12,0.05" \
  --fingertip-lateral "0,-0.015,0.020,0.015,0.04" \
  --fingertip-straighten "0,0.32,0.60,0.20,0.15" \
  --thumb-gain 1 \
  --thumb-cross-gain 0.31 \
  --thumb-assist-smooth 0.72 \
  --thumb-orient-gain 0.72 \
  --thumb-grasp-gain 0.44 \
  --thumb-base-assist-gain 0.68 \
  --thumb-tip-gain 1.02 \
  --hardware-base-gain 1.80 \
  --hardware-base-gains "0.84,0.88,0.84,1.00" \
  --hardware-tip-gain 0.80 \
  --hardware-tip-gains "1.04,1.12,1.00,1.00" \
  --hardware-spread-gain 0.64 \
  --hardware-thumb-tip-gain 0.82 \
  --hardware-thumb-tip-offset -3 \
  --hardware-thumb-roll-gain 0.58 \
  --hardware-thumb-base-gain 0.95 \
  --hardware-thumb-base-offset -3 \
  --hardware-thumb-abd-gain 0.94 \
  --thumb-safe-mode limited \
  --max-thumb-delta 155 \
  --max-thumb-abd-delta 180 \
  --max-thumb-base-delta 125 \
  --max-spread-delta 90 \
  --spread-close-threshold 0.82 \
  --spread-recenter-gain 0.10 \
  --thumb-index-guard \
  --thumb-index-threshold 0.32 \
  --thumb-index-release 42 \
  --current-limit 40 \
  --speed-limit 60 \
  --enable-motion \
  --max-range-step 8
  ```




# 拇指对无名指
[216, 255, 255, 85, 232, 83, 95, 100, 114, 127, 60, 255, 255, 255, 255, 80, 255, 255, 97, 255]

# 拇指对中指
[247, 255, 56, 255, 255, 96, 95, 84, 114, 127, 109, 255, 255, 255, 255, 54, 255, 122, 255, 255]

# 拇指对小拇指
[255, 255, 255, 255, 87, 27, 95, 100, 114, 121, 70, 255, 255, 255, 255, 55, 255, 255, 255, 90]

# 拇指对食指（ok）
[236, 100, 255, 255, 255, 150, 168, 100, 114, 127, 84, 255, 255, 255, 255, 51, 87, 255, 255, 255]


# Without sensor
```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate
export HW_ENABLE_TOKEN=1

mkdir -p logs
LOG="logs/hw_gtfit_$(date +%Y%m%d_%H%M%S).log"

.venv/bin/python -m src.comms.camera_to_linkerhand \
  --source webcam --camera-index 0 --side right \
  --sdk-hand-joint g20 --hardware-map g20-sim \
  --show-camera --absolute \
  --no-filter \
  --one-euro-min-cutoff 0.8 \
  --one-euro-beta 0.04 \
  --one-euro-d-cutoff 1.0 \
  --fingertip-extend "0,0.12,0.125,0.12,0.05" \
  --fingertip-lateral "0,-0.015,0.020,0.015,0.04" \
  --fingertip-straighten "0,0.32,0.60,0.20,0.15" \
  --thumb-gain 1 \
  --thumb-cross-gain 0.25 \
  --thumb-assist-smooth 0.80 \
  --thumb-orient-gain 0.59 \
  --thumb-grasp-gain 0.56 \
  --thumb-base-assist-gain 0.40 \
  --thumb-tip-gain 0.93 \
  --hardware-landmark-thumb \
  --landmark-thumb-gain 0.85 \
  --landmark-thumb-reach-gain 0.81 \
  --hardware-base-gain 1.80 \
  --hardware-base-gains "0.46,1.10,1.10,0.63" \
  --hardware-tip-gain 0.80 \
  --hardware-tip-gains "1.27,0.87,1.25,1.27" \
  --hardware-spread-gain 0.64 \
  --hardware-thumb-tip-gain 1.07 \
  --hardware-thumb-tip-offset -23 \
  --hardware-thumb-roll-gain 0.95 \
  --hardware-thumb-roll-offset -27 \
  --hardware-thumb-base-gain 0.80 \
  --hardware-thumb-base-offset 7 \
  --hardware-thumb-abd-gain 0.84 \
  --hardware-thumb-abd-offset -41 \
  --thumb-safe-mode limited \
  --max-thumb-delta 180 \
  --max-thumb-abd-delta 238 \
  --max-thumb-base-delta 149 \
  --max-spread-delta 90 \
  --spread-close-threshold 0.82 \
  --spread-recenter-gain 0.10 \
  --thumb-index-guard \
  --thumb-index-threshold 0.44 \
  --thumb-index-release 0 \
  --current-limit 35 \
  --speed-limit 45 \
  --enable-motion \
  --max-range-step 6 \
  --log-period 0.25 \
  --log-sim-position \
  2>&1 | tee "$LOG"

echo "saved log: $LOG"
```


# with sensor

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate
export HW_ENABLE_TOKEN=1

mkdir -p logs
LOG="logs/hw_gtfit_$(date +%Y%m%d_%H%M%S).log"

.venv/bin/python -m src.comms.camera_to_linkerhand \
  --source webcam --camera-index 0 --side right \
  --sdk-hand-joint g20 --hardware-map g20-sim \
  --show-camera --absolute \
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
  --landmark-thumb-gain 0.52 \
  --landmark-thumb-reach-gain 0.45 \
  --hardware-base-gain 1.80 \
  --hardware-base-gains "0.50,1.10,1.10,0.63" \
  --hardware-tip-gain 0.80 \
  --hardware-tip-gains "1.12,0.87,1.25,1.27" \
  --hardware-spread-gain 0.64 \
  --hardware-thumb-tip-gain 1.04 \
  --hardware-thumb-tip-offset -16 \
  --hardware-thumb-roll-gain 0.68 \
  --hardware-thumb-roll-offset -8 \
  --hardware-thumb-base-gain 0.98 \
  --hardware-thumb-base-offset 7 \
  --hardware-thumb-abd-gain 0.72 \
  --hardware-thumb-abd-offset -28 \
  --thumb-safe-mode limited \
  --max-thumb-delta 165 \
  --max-thumb-abd-delta 190 \
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
  --max-range-step 4 \
  --log-period 0.25 \
  --log-sim-position \
  2>&1 | tee "$LOG"

echo "saved log: $LOG"
```


# start sdk
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right --can can0 --is_touch true
```


# final version with sensor gui on
```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate
export HW_ENABLE_TOKEN=1

mkdir -p logs
LOG="logs/hw_gtfit_$(date +%Y%m%d_%H%M%S).log"

.venv/bin/python -m src.comms.camera_to_linkerhand \
  --source webcam --camera-index 0 --side right \
  --sdk-hand-joint g20 --hardware-map g20-sim \
  --show-camera --absolute \
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
  --hardware-thumb-tip-gain 1.12 \
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
  --enable-motion \
  --max-range-step 5 \
  --log-period 0.25 \
  --log-sim-position \
  2>&1 | tee "$LOG"

echo "saved log: $LOG"

```




# start sdk
```bash
cd ~/Desktop/Jacky/linker_hand_ros2_sdk
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch \
  --hand_type right --can can0 --is_touch true
```


# runnning version

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate
export HW_ENABLE_TOKEN=1

.venv/bin/python -m src.comms.camera_to_linkerhand \
  --source webcam --camera-index 2 --side right \
  --sdk-hand-joint g20 --hardware-map g20-sim \
  --show-camera --absolute \
  --q0-key-step 10 \
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
  --enable-motion \
  --max-range-step 5 \
  --log-period 0.25 \
  --log-sim-position
```
