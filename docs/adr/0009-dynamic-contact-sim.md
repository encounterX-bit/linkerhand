# ADR-0009: G2 dynamic / contact closed-loop sim — PD model, mimic-under-dynamics, virtual force cap, grasp tuning findings

Status: proposed (sim-agent G2, 2026-06-09)
Related: ticket docs/tickets/sim-agent-G2-v2.md; [ADR-0003 segments],
[ADR-0005 FK authority], [ADR-0007 timing], [ADR-0008 safety filter];
src/sim/{dynamics,closed_loop,grasp}.py, tests/g2_dynamic/

## Context

G2 promotes the G1 kinematic harness (which *placed* joints with `resetJointState`)
to a **dynamic, closed-loop contact sim**: masses/inertias + gravity + contact,
position-PD motors, the safety filter inline, CAN latency, and grasps on a cylinder
and a sphere. PyBullet provides **dynamics/contact only**; the *metric* (segment
directions) is still the `src/kinematics` FK authority (ADR-0005). Decisions that
needed recording:

## Decision

### 1. PD control + mimic enforcement under dynamics
- The 16 active DoF are driven by PyBullet `POSITION_CONTROL` (Kp=0.3, Kd=0.6).
  **PD gains are SIM-ONLY** — they tune the contact simulation, are not hardware
  control gains, and never leave the sim. The default velocity motors are zeroed so
  position-PD is the only actuator.
- PyBullet ignores URDF `<mimic>`. We **re-issue each mimic setpoint at
  `ratio*driver + offset` every physics step** (the ticket's sanctioned per-step
  alternative to a gear constraint). Verified the coupling holds while stepping:
  settled abs error ≤ 0.009 rad, realized ratio within ~1.6 % of nominal
  (0.8917 non-thumb / 1.1619 thumb) — `test_mimic_under_dynamics`.

### 2. Virtual force cap = per-joint torque, tuned to the worst grasp
- The "virtual force cap" is implemented as the per-joint motor torque ceiling
  `PDGains.max_force_nm`. **Finding: a per-joint torque limit does NOT linearly
  bound TOTAL grip force** — several fingers (+ thumb) sum their contributions onto
  one object and a point/line contact concentrates them. At 0.35 Nm a sphere
  enveloping grasp reached ~24–28 N (> the 15 N `ForceClampSpec` cap). **0.12 Nm**
  holds both the cylinder (peak ~12.5 N) and the sphere (peak ~9 N) UNDER the 15 N
  cap while still tracking free motion to ~0.01 rad. So 0.12 Nm is the empirically
  chosen cap for the worst observed grasp. `test_grasp` asserts measured contact
  force ≤ 15 N **and** that raising the torque breaks the cap (the cap is
  load-bearing, not an artifact of light objects). The REAL clamp is comms/G3.

### 3. Closed loop + latency
- One control tick (30 Hz): `landmarks → retarget() → safety.filter() → [latency
  buffer] → PD motors → step → read`. `retarget()` and `safety.filter()` are
  imported **read-only**. **Latency** is an integer number of control frames of
  pure transport delay (a deque): a command computed at tick k is applied at k+N.
  It is *modeled delay, not compute*, so it does not count against the loop period.
- **Loop-rate gate (two-part):** `compute = retarget + filter + sim-step`.
  (a) absolute: p99 compute < 33,333 µs (one 30 Hz frame) — measured ≈ 11.1 ms,
  ~3× headroom; (b) regression: p99 < committed baseline (11,000 µs) × (1+0.50).
  The filter dominates the tail (~2.3 ms p50, the deep-collision frames longer),
  consistent with ADR-0008's filter baseline.

### 4. Grasp tuning findings (reported, NOT forced green — per the ticket)
- **A free fingertip PINCH is fragile on this hand.** The thumb's opposition motion
  is lateral and bats a small free sphere out of the gap before the finger closes;
  there is no wrist to pre-load a 2-point pinch. The robust, deterministic sphere
  scenario is therefore a **palm-backed enveloping grasp**, not a fingertip pinch.
- **No free object stays put without a reaction surface.** Grasps use a mild
  palm-ward gravity (−x) so the object rests on the palm's +x face and the fingers
  cage it against that face (there is no arm to hold against world gravity). The
  closing command is **ramped slowly** — slamming the fingers shut launches the
  object (an ejection artifact, not a grasp).
- Object poses, masses, and seeds are fixed → reproducible. The grasp test asserts
  contact established, bounded displacement/speed, finite state, and force ≤ cap.

### 5. Tracking penalty (monitored)
- Dynamics lag + the safety projection + transport delay add error over the G1
  kinematic numbers: dynamic no-latency overall p95 ≈ 0.232 rad vs G1 ≈ 0.178
  (+0.054); at 2-frame latency ≈ 0.324. Reported, not gated. Latency stability:
  error grows with delay (p95 0.37→0.52→1.33 rad at 0/67/200 ms) but stays
  **bounded — no divergence** (`test_latency`).

## Consequences
- PyBullet is now used for dynamics/contact in `src/sim`; it is still **not an FK
  authority** (the metric is `src/kinematics`). No hardware path; no actuation; the
  force clamp and watchdog remain comms/G3 specs gated by `HW_ENABLE_TOKEN`.
- The virtual cap (0.12 Nm) and the loop/grasp/latency baselines are
  **machine-specific where timing is involved**; re-measure on the target box.
- Do NOT advance past G2 (human gate).
