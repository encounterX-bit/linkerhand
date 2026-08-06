# Isaac Lab Setup for L20 Residual-ACT Grasping

This setup is for a new Isaac Lab path. The existing `src/isaac_gym/` code can
stay as a legacy demo, but new training work should target Isaac Lab.

## Why Isaac Lab

Isaac Gym is now legacy software. NVIDIA recommends Isaac Lab for new robot
learning projects because it is the current GPU-accelerated framework built on
Isaac Sim and supports RL and imitation-learning workflows.

Useful upstream docs:

- Isaac Lab overview: https://developer.nvidia.com/isaac/lab
- Isaac Lab local install: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- Isaac Sim pip install path: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- Isaac Gym legacy notice: https://developer.nvidia.com/isaac-gym

## Machine requirements

For the current Isaac Sim 5.x path:

- Ubuntu 22.04 is the safest Linux target.
- Python 3.11 is required for Isaac Sim 5.x.
- 32 GB RAM minimum is recommended.
- 16 GB GPU VRAM or more is recommended for rendering/robot-learning workflows.
- Use a recent NVIDIA production driver.

Check before installing:

```bash
nvidia-smi
ldd --version
python3.11 --version
```

## Recommended install

Use a separate environment. Do not install Isaac Lab into this repo's current
Python 3.12 `.venv`.

```bash
cd ~/Desktop/Jacky
python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate
pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

isaacsim
```

The first `isaacsim` run may take a long time while extensions are cached, and
it will ask for NVIDIA Omniverse EULA acceptance.

Then install Isaac Lab from source:

```bash
cd ~/Desktop/Jacky
git clone https://github.com/isaac-sim/IsaacLab.git --branch main
cd IsaacLab
sudo apt install cmake build-essential
./isaaclab.sh --install
```

Verify:

```bash
cd ~/Desktop/Jacky/IsaacLab
source ~/Desktop/Jacky/env_isaaclab/bin/activate
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

## L20 project layout

Create the LinkerHand project as an external Isaac Lab project, not inside
Isaac Lab core:

```text
~/Desktop/Jacky/l20_isaac_lab/
  source/l20_isaac_lab/
    l20_isaac_lab/
      __init__.py
      assets/
        linkerhand_l20.py
      tasks/
        direct/
          l20_grasp/
            __init__.py
            l20_grasp_env.py
            l20_grasp_env_cfg.py
            agents/
              rsl_rl_ppo_cfg.py
      teleop/
        mediapipe_coarse_source.py
      dataset/
        record_teleop_teacher.py
      imitation/
        train_residual_act.py
```

The external project should import this repo for retargeting:

```bash
export L20_TELEOP_REPO=/home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt
export PYTHONPATH="$L20_TELEOP_REPO:$PYTHONPATH"
```

## Environment definition

Task name:

```text
L20-Hand-PostReach-Grasp-Direct-v0
```

V1 task scope is power grasp only. The object is assumed to be large enough for
palm-backed enveloping contact. Small-object pinch and thin-object lateral
grasps are intentionally out of scope until the power-grasp teacher/student
pipeline is stable.

Episode assumption:

- the wrist/palm base is fixed or scripted at the pre-grasp pose,
- the object starts inside the hand workspace,
- the policy controls only the 16 active L20 joints,
- joints 11-14 remain reserved and are never actuated.

Recommended action space:

```text
action[16] = active L20 joint position targets in radians
```

Map active indices back to the 20-entry L20 command:

```text
active = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,19]
reserved = [11,12,13,14] -> 0.0
```

Recommended observations for the RL teacher:

```text
q, dq
object pose and velocity
fingertip poses
contact/contact force proxy
previous action
power-grasp phase or coarse closure command
```

Recommended student observations:

```text
z_intent
q, dq
a_prev
a_coarse
object pose or depth-derived object features
contact_obs if available on the real hand
```

## Asset conversion notes

Start from the official LinkerHand L20 URDF already used by this repo. For
Isaac Lab, convert or import it into USD and inspect:

- joint names and limits,
- drive mode and stiffness/damping,
- mimic/coupled distal joints,
- collision geometry scale,
- palm and fingertip frames,
- left/right coordinate conventions.

The existing repo's semantic command convention stays authoritative:

```text
idx 0-4   base flexion
idx 5-9   abduction/spread
idx 10    thumb opposition
idx 15-19 coupled distal tip flexion
idx 11-14 reserved zeros
```

## RL teacher recipe

Start with a power-grasp object distribution:

- sphere: 35-75 mm diameter,
- box: 35-75 mm side lengths,
- cylinder: 30-70 mm diameter, 40-100 mm height.
- mug-like body: approximate as a cylinder/box in V1; ignore handle grasping.

Initial object pose:

- centered near palm/fingertip workspace,
- random translation noise up to 10-25 mm,
- random rotation,
- gravity direction chosen to match the real fixture.

Reward sketch:

```text
+ object held near palm or target hold pose
+ lift/hold success if the fixture supports that test
+ palm-backed multi-finger contact with thumb participation
+ low slip velocity
- action jerk
- joint-limit pressure
- excessive contact force proxy
- self-collision / palm penetration
- object ejection
```

Train teacher:

```bash
cd ~/Desktop/Jacky/IsaacLab
source ~/Desktop/Jacky/env_isaaclab/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=L20-Hand-PostReach-Grasp-Direct-v0 \
  --headless \
  --num_envs=4096
```

## Teleop + teacher dataset

Run a collection script inside the external project:

```bash
cd ~/Desktop/Jacky/l20_isaac_lab
source ~/Desktop/Jacky/env_isaaclab/bin/activate
python -m l20_isaac_lab.dataset.record_teleop_teacher \
  --task L20-Hand-PostReach-Grasp-Direct-v0 \
  --teacher-checkpoint /path/to/teacher.pt \
  --source webcam \
  --side right \
  --episodes 200
```

For every timestep, store:

```text
obs_student
obs_privileged
a_coarse
a_rl
delta_a = a_rl - a_coarse
success labels
domain randomization params
```

Use Zarr, HDF5, or Parquet plus compressed arrays. Avoid ad hoc text logs for
trajectory data.

## Residual ACT training

Train ACT outside Isaac Lab if that is more convenient. The only hard contract
is the dataset schema:

```text
inputs:
  z_intent window
  q/dq window
  a_coarse window
  a_prev window
  object_obs window
  optional contact_obs window

targets:
  bounded delta_a chunk
```

Deployment formula:

```text
a_final = safety_filter(a_coarse + lambda * clip(pi_ACT(obs), delta_limit))
```

Real hardware default:

```text
lambda = 0.0
```

Increase to `0.1`, then `0.2`, only after sim replay and low-force hardware
checks pass.

## First milestone checklist

- Isaac Lab opens `create_empty.py`.
- A simple built-in RL task trains headless.
- L20 asset loads with correct joint order and limits.
- Fixed-palm object scene resets thousands of envs.
- Hand closes on scripted power-grasp `a_coarse`.
- PPO teacher beats scripted baseline on held-out power objects.
- Teleop collection records `a_coarse` and teacher `a_rl`.
- ACT replay improves grasp stability in sim without exceeding residual limits.
- Real hand test remains disabled until G3 human gate.
