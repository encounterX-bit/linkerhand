# Isaac Gym Setup for L20 Power-Grasp Residual ACT

This is the active setup path for the current project. Isaac Lab is not required
for V1. Use the existing Python 3.8 conda environment:

```bash
conda activate isaacgym38
```

## Environment fix

Isaac Gym Preview is built for Python 3.8. In this machine's conda environment,
`isaacgym` imports only when the conda shared library directory is visible:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/zhaoyan-qian/Desktop/Jacky/isaacgym/python:$PWD:$PYTHONPATH"
```

If `import isaacgym` fails with `libpython3.8.so.1.0`, the `LD_LIBRARY_PATH`
line above is the required fix.

Install or repair the runtime packages used by the current demo:

```bash
pip install yourdfpy==0.0.60
pip install --force-reinstall mediapipe==0.10.11 opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86
```

Quick import check:

```bash
python - <<'PY'
from isaacgym import gymapi
import torch
import mediapipe as mp
import yourdfpy
print("isaacgym ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mediapipe", mp.__version__)
PY
```

## Existing smoke test

The repo already has a live MediaPipe -> Isaac Gym cube demo:

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt
conda activate isaacgym38
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/zhaoyan-qian/Desktop/Jacky/isaacgym/python:$PWD:$PYTHONPATH"

python -m src.isaac_gym.grasp_cube_teleop \
  --source webcam \
  --side right \
  --show-camera \
  --cube-size 0.065 \
  --cube-x 0.065 \
  --cube-y 0.0 \
  --cube-z 0.115 \
  --gravity-x -4.0 \
  --thumb-grasp-gain 0.25 \
  --thumb-orient-gain 0.6
```

Headless check, useful before opening the viewer:

```bash
python -m src.isaac_gym.grasp_cube_teleop \
  --source webcam \
  --side right \
  --headless \
  --max-frames 60
```

## Current asset status

The existing demo loads the vendored regular L20 URDF:

```text
src/sim/urdf/l20/right/linkerhand_l20_right.urdf
src/sim/urdf/l20/left/linkerhand_l20_left.urdf
```

The local L20 lite URDF exists but is not wired into this repo's Isaac Gym demo:

```text
/home/zhaoyan-qian/Desktop/Jacky/linkerhand-urdf/l20lite/right/linkerhand_l20lite_right.urdf
/home/zhaoyan-qian/Desktop/Jacky/linkerhand-urdf/l20lite/left/linkerhand_l20lite_left.urdf
```

L20 lite has a different mimic structure from the regular L20, so do not reuse
the regular L20 action mapping blindly. First milestone should stay on the
regular L20 demo unless the real hardware is confirmed to be L20 lite.

## V1 task path

1. Repair `isaacgym38` imports.
2. Run the existing cube teleop smoke test.
3. Add an Isaac Gym vectorized environment for power objects only:
   sphere, box, cylinder, mug body.
4. Train a privileged RL teacher in Isaac Gym.
5. Run human MediaPipe teleop inside the same task and record `a_coarse`.
6. Query frozen teacher for `a_RL`.
7. Train ACT on `delta_a = a_RL - a_coarse`.
8. Deploy only as a bounded residual behind the existing safety filter.

## Real hardware note

No Isaac Gym command should actuate the real hand. The Isaac Gym path is sim-only
until the normal G3 hardware gate is reviewed by a human.
