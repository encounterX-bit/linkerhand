# L20 Residual-ACT Grasp Pipeline

Goal: after the hand has already reached a pre-grasp position, use human
MediaPipe teleoperation as the coarse command and a learned policy as a bounded
residual correction for stable Linker Hand L20 power-grasp closure.

This is not an arm-reaching policy. The object is assumed to be inside the L20
workspace, near the palm or fingertips.

V1 scope is power grasp only: sphere, box, cylinder, and mug-like objects that
can be stabilized by palm-backed enveloping contact. Pinch grasps for small
objects and lateral grasps for thin objects are future work.

## Core idea

```text
MediaPipe landmark window
        |
        v
intention encoder z_intent
        |
        +------> existing retargeter ------> a_coarse
        |
object obs + hand state + previous action + a_coarse
        |
        v
ACT residual student pi_ACT
        |
        v
delta_a = bounded(pi_ACT(...))
        |
        v
a_final = safety_filter(a_coarse + lambda * delta_a)
        |
        v
L20 joint targets
```

The residual policy is deliberately not allowed to replace the human command.
It only corrects thumb opposition, distal closure, timing, and light stabilizing
motions when the coarse MediaPipe mapping is noisy or under-specified.

## Signals

### Student inputs available on real hardware

- `landmarks_window`: MediaPipe landmarks over the last 0.3-1.0 s.
- `landmark_confidence`: per-frame hand confidence when available.
- `z_intent`: encoder output from the landmark window.
- `q`: 16 active L20 joint positions, radians.
- `dq`: optional joint velocities, filtered.
- `a_prev`: previous target command.
- `a_coarse`: existing MediaPipe -> L20 retarget output.
- `object_obs`: one of:
  - object class + width/height/depth from a fixed setup,
  - object pose from an external tracker,
  - depth crop or point cloud if available.
- `contact_obs`: tactile, pressure, current, or sim contact flags. If real
  tactile is missing at first, train an ablated student without this input.

### Teacher-only privileged inputs

The Isaac Gym RL teacher may see privileged state that the student never sees:

- object pose and velocity,
- exact contact forces,
- object center of mass,
- friction and mass randomization values,
- fingertip/object distances.

Never feed these privileged values into ACT unless the real deployment path can
measure the same signal.

## Training stages

### Stage A: baseline coarse power grasp

Build a non-learning baseline first. It should produce stable `a_coarse`
trajectories for palm-backed power grasps:

- sphere: palm-backed enveloping closure,
- box: four-finger wrap with thumb opposition,
- cylinder: wrap around the long axis,
- mug-like object: handle ignored in V1; treat body as a cylinder/box.

The baseline is useful even if it is imperfect, because ACT will learn residual
corrections around this distribution.

### Stage B: Isaac Gym RL teacher

Train `pi_RL` in simulation as a full-action teacher:

```text
pi_RL(s_privileged) -> a_RL
```

Recommended action:

- 16 active L20 joint targets in radians, or
- 16 residual targets around a neutral grasp trajectory.

Recommended reward terms:

- object remains within palm/fingertip workspace,
- object height/hold success after closure,
- stable contact without large slip,
- low action jerk,
- low contact force/current proxy,
- no self-collision or joint-limit violation,
- match the power-grasp closure template when a coarse command is given.

Use broad domain randomization:

- object size, pose, mass, center of mass, friction,
- joint friction/damping, motor strength, position-target delay,
- sensor noise, point-cloud dropout, object pose noise,
- physics timestep and contact stiffness within reasonable ranges.

### Stage C: human-in-the-sim data collection

Run MediaPipe teleoperation inside the same Isaac Gym task:

```text
human MediaPipe -> retargeter -> a_coarse -> sim L20
```

Record every timestep:

```text
episode_id, t
landmarks_window
z_intent
object_obs_student
q, dq
a_prev
a_coarse
s_privileged_for_teacher
success/failure labels
```

For each recorded state, query the frozen RL teacher:

```text
a_RL = pi_RL(s_privileged_for_teacher)
delta_a = a_RL - a_coarse
```

Keep failure and near-failure trajectories. They are the most useful data for
learning stabilizing corrections.

### Stage D: ACT residual student

Train ACT on chunks:

```text
input chunk:
  z_intent[t-k:t], q[t-k:t], object_obs[t-k:t], a_coarse[t-k:t], a_prev[t-k:t]

target chunk:
  delta_a[t:t+h]
```

Use per-joint target scaling. Thumb opposition and distal joints should have
their own residual limits.

At deployment:

```text
delta_a_raw = pi_ACT(obs)
delta_a = clip(delta_a_raw, per_joint_delta_limit)
a_final = a_coarse + lambda * delta_a
a_final = safety_filter(a_final)
```

Start with `lambda = 0.1` or `0.2` on real hardware. Increase only after slow,
low-current trials.

## Safety contract

Real hardware execution remains G3+ and human-gated:

- no code path may actuate without `HW_ENABLE_TOKEN`,
- current/force/speed limits must stay active,
- residual correction must be clipped per joint,
- watchdog opens or relaxes the hand on stale policy output,
- ACT residual must be disabled by default until sim replay passes.

## Evaluation gates

### Sim gates

- teacher success rate on held-out objects,
- student residual replay vs teacher,
- student rollout success under randomized delay/noise,
- no residual command exceeds configured bounds,
- no increase in collision or excessive force metrics vs baseline.

### Real gates

- fixed object, palm-supported, low speed,
- one or two residual joints enabled first,
- thumb/index/middle before all fingers,
- force/current cap verified with a human on e-stop,
- compare baseline-only vs residual-assisted grasp on the same object set.
