# HumanEgo -> LinkerHand Two-Finger MVP

This repo now has a small adapter for HumanEgo-style policies that control only
the LinkerHand thumb and index finger.  It does not vendor HumanEgo or bypass the
existing hardware gate.  It converts compact policy actions into the existing
canonical 20-joint LinkerHand radian command.

## Action Modes

`pinch3` is the first MVP action head:

```text
[close, thumb_cross, index_spread]
```

- `close`: normalized `[0, 1]`
- `thumb_cross`: normalized `[0, 1]`
- `index_spread`: normalized `[-1, 1]`

`joint7` is the direct two-finger head:

```text
[thumb_base, thumb_abd, thumb_opp, thumb_tip, index_base, index_abd, index_tip]
```

By default flexion/opposition channels are normalized `[0, 1]`; `index_abd` is
normalized `[-1, 1]`.  Use `--input-range radians` if a policy already outputs
LinkerHand radians for these seven channels.

In both modes, middle/ring/little and reserved joints are forced open/zero.

## Smoke Test

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
.venv/bin/python -m src.humanego_linkerhand.replay_two_finger \
  --demo-frames 90 \
  --show-sim
```

## Camera Direct Control

Before training a HumanEgo policy, use the camera path to tune the physical
thumb-index mapping.  This still uses the existing hand-landmark retargeter, but
locks the output to the two-finger subset.

```bash
cd ~/Desktop/Jacky/sims/linker-hand-teleopt
.venv/bin/python -m src.humanego_linkerhand.camera_two_finger \
  --source webcam \
  --camera-index 0 \
  --side right \
  --show-camera
```

For a no-camera smoke test:

```bash
.venv/bin/python -m src.humanego_linkerhand.camera_two_finger \
  --source replay \
  --headless \
  --max-frames 5
```

The camera command is sim-only.  It does not publish to ROS or move hardware.
Use the existing hardware bridge only after the sim pinch pose looks correct.

To export a policy trajectory into 20-joint command records:

```bash
.venv/bin/python -m src.humanego_linkerhand.replay_two_finger \
  --trajectory actions.jsonl \
  --mode pinch3 \
  --out-jsonl out/l20_two_finger.jsonl
```

The action file can be JSONL records such as:

```json
{"t": 0.0, "action": [0.0, 0.0, 0.0]}
{"t": 0.033, "action": [0.5, 0.7, 0.0]}
```

or JSON/NPY/NPZ arrays.  The output records follow `contracts/l20_targets` and
include `two_finger_idx` for the active thumb/index subset.

## Next Integration Point

For HumanEgo proper, keep its preprocessing/training pipeline external and make
its inference action head emit either `pinch3` or `joint7`.  Then feed each
predicted action through:

```python
from src.humanego_linkerhand import TwoFingerConfig, action_to_l20

cfg = TwoFingerConfig(side="right", mode="pinch3")
target = action_to_l20(policy_action, cfg)
joint_rad = target["joint_rad"]
```

Use `src.viz.render.L20VizModel` or the existing ROS bridge as the downstream
consumer.  Real hardware still requires the original `HW_ENABLE_TOKEN` path.
