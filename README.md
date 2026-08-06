# LinkerHand Hand Retargeting and Teleoperation

This project converts human hand landmarks from a camera or recorded video into joint targets for the LinkerHand L20/G20. It includes kinematic simulation, safety filtering, a ROS 2 hardware bridge, action-library tools, and evaluation utilities.

The core data flow is:

```text
MediaPipe / RealSense / recorded video
                    ↓
        Hand landmark preprocessing
                    ↓
          Per-finger retargeting
                    ↓
       Collision and limit filtering
                    ↓
    Simulation / ROS 2 hardware bridge
```

> **Safety:** The default workflow does not actuate real hardware. Real motion requires both `--enable-motion` and a human-set `HW_ENABLE_TOKEN`. Use simulation or dry-run mode until the hardware limits, emergency stop, force limits, and supervision procedures have been verified.

## Repository Layout

The 100 files introduced by commit `30216d1` are located under `src/`. Files from earlier commits remain in their original top-level locations. This root README is the project entry point.

| Path | Purpose |
| --- | --- |
| `src/perception/` | MediaPipe, RealSense, recorded-video, and replay inputs |
| `src/finger_retarget/` | Retargeting solver from human landmarks to L20 joint angles |
| `src/kinematics/` | Joint conventions and forward kinematics |
| `src/safety/` | Joint-limit, rate-limit, and collision filtering |
| `src/sim/` | PyBullet kinematic, dynamic, and contact simulation |
| `src/viz/` | Live and offline visualization |
| `src/comms/` | ROS 2 integration, hardware mapping, and action-library tools |
| `src/humanego_linkerhand/` | HumanEgo two-finger control workflow |
| `src/isaac_gym/` | Isaac Gym teleoperation entry point |
| `scripts/` | Data processing, training, and evaluation scripts |
| `tests/` | G0–G2, communication, and visualization tests |
| `eval/` | Reference solver and evaluation dependencies |
| `docs/` | Architecture, setup guides, design decisions, and task records |
| `contracts/` | JSON schemas shared by the pipeline components |
| `hardware/` | Hardware limits and safety notes |

## Environment Setup

Python 3.10 or later is recommended. Run the following commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies for the features you need:

```bash
# Solver, evaluation, and tests
python -m pip install -r eval/requirements.txt

# PyBullet simulation and visualization
python -m pip install -r src/sim/requirements.txt

# Webcam and recorded-video input
python -m pip install -r src/perception/requirements.txt
```

The ROS 2 hardware bridge also requires ROS 2 Jazzy, a built LinkerHand ROS 2 SDK workspace, and:

```bash
python -m pip install -r src/comms/requirements.txt
```

## Quick Start

### 1. Safe Camera-Free Smoke Test

Run 30 frames from the bundled motion sequence without opening a window or accessing hardware:

```bash
python -m src.viz.app --camera-free --headless --max-frames 30
```

To open the simulation window:

```bash
python -m src.viz.app --camera-free
```

### 2. Live Webcam Mirroring

```bash
python -m src.viz.app \
  --source webcam \
  --camera-index 0 \
  --side right \
  --show-camera
```

Press `Q` or `Esc` to quit. Add `--image-mirrored` when using mirrored or selfie-style input.

### 3. Recorded-Video Input

```bash
python -m src.viz.app \
  --source video \
  --video-path /path/to/hand_video.mp4 \
  --side right
```

List all available options with:

```bash
python -m src.viz.app --help
```

## Tests

Run the complete test suite:

```bash
python -m pytest tests
```

Run individual safety stages:

```bash
python -m pytest tests/g0_unit
python -m pytest tests/g1_kinematic
python -m pytest tests/g2_safety tests/g2_dynamic
```

Some tests depend on PyBullet, URDF assets, recorded samples, or machine-specific performance baselines. Install the relevant dependencies and check the test output for environment-specific requirements.

## Real Hardware

Hardware control is a human-reviewed stage and is not part of the quick-start workflow. Before continuing, verify that:

- The LinkerHand ROS 2 SDK is built and the CAN interface and hand model are configured correctly.
- The limits in `hardware/LIMITS.md` have been reviewed by a human.
- An emergency stop, active supervision, low speed, and low current limits are in place.
- The G0–G2 tests pass and the hardware safety constraints in `CLAUDE.md` are understood.

`src.comms.camera_to_linkerhand` runs in dry-run mode by default. Publishing real commands additionally requires an operator to pass `--enable-motion` and set `HW_ENABLE_TOKEN`. Do not store this token in a script or commit it to the repository.

## Documentation

- Project architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Current development state: [`STATE.md`](STATE.md)
- Hardware limits: [`hardware/LIMITS.md`](hardware/LIMITS.md)
- Isaac Gym setup: [`docs/isaac_gym_setup.md`](docs/isaac_gym_setup.md)
- Isaac Lab setup: [`docs/isaac_lab_setup.md`](docs/isaac_lab_setup.md)
- HumanEgo two-finger workflow: [`docs/humanego_linkerhand_two_finger.md`](docs/humanego_linkerhand_two_finger.md)
- Residual ACT pipeline: [`docs/residual_act_pipeline.md`](docs/residual_act_pipeline.md)

## License

The original project license is available at [`LICENSE`](LICENSE).
