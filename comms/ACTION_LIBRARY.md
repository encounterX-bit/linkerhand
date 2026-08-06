# MediaPipe action library teleop

The library stores short gestures as raw hand-base MediaPipe landmark templates and pairs each
gesture with one hardcoded 20-D G20 SDK-range trajectory. Recording templates
never connects to ROS or actuates hardware.

## 1. Prepare a robot trajectory

Launch the offline waypoint GUI:

```bash
python -m src.comms.g20_waypoint_editor \
  --output data/action_library/waypoints/open_to_half_close.json \
  --show-sim
```

The PyBullet hand is an approximate L20-URDF visualization of the G20 range
values; the exported JSON retains the exact slider values. The editor does not
import ROS and cannot move the real hand. Save the open start pose, move the
sliders to the half-close end pose, set its duration, save it, then export.

Alternatively, copy `action_library_waypoints_example.json` and edit its 20-D poses. The first
pose is the start pose. Each later pose has a travel `duration` in seconds.
Values are G20 SDK range `0..255`; indices 11--14 are rewritten to 255.

## 2. Record the first MediaPipe take

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt
source .venv/bin/activate

python -m src.comms.record_action_primitive \
  --library data/action_library/g20_right/v1 \
  --primitive-id 0 \
  --name half_close \
  --waypoints data/action_library/waypoints/open_to_half_close.json \
  --camera-index 2
```

Focus the camera window, press SPACE, perform the short gesture, and press SPACE
again. Repeat the same command without `--waypoints` four more times so the
primitive has five human templates:

```bash
python -m src.comms.record_action_primitive \
  --library data/action_library/g20_right/v1 \
  --primitive-id 0 \
  --name half_close \
  --camera-index 2
```

Create other IDs in the same way. Keep each primitive around 0.2--1.5 seconds.
The example trajectory is only a format example; inspect and tune every pose in
dry-run/simulation before using it on hardware.

Validate with held-out takes before running online matching:

```bash
python -m src.comms.validate_action_library \
  --library data/action_library/g20_right/v1 \
  --minimum-accuracy 0.90
```

An audited grouped recording can be imported without recording the templates a
second time:

```bash
python -m src.comms.import_action_group \
  --group data/action_groups/current_actions/02_thumb_fold_inward \
  --library data/action_library/g20_right/example_library_v1 \
  --primitive-id 2 \
  --name thumb_fold_inward
```

Import refuses incomplete groups and groups whose recorded GUI command differs
from measured state by more than 10 SDK ticks on an active joint. It also
densifies the trajectory so adjacent active-joint frames differ by at most five
ticks.

To intentionally pair human repetitions and robot waypoints recorded in two
different groups, replace `--group` with both source paths:

```bash
python -m src.comms.import_action_group \
  --human-group data/action_groups/HUMAN_SESSION/group_000 \
  --robot-group data/action_groups/ROBOT_SESSION/group_000 \
  --library data/action_library/g20_right/example_library_v1 \
  --primitive-id 2 \
  --name thumb_fold_inward \
  --replace
```

Both resolved source paths are saved in the library manifest.

## 3. Recognition and command dry-run

```bash
python -m src.comms.action_library_teleop \
  --library data/action_library/g20_right/v1 \
  --camera-index 2
```

Press SPACE to arm recognition. Terminal lines labeled `WOULD_CMD` are not
published. The camera preview overlays the MediaPipe finger skeleton. Adjust
each primitive's `threshold` in `manifest.json` after checking held-out takes. A
smaller threshold is stricter.

## 4. Human-gated hardware execution

Start the G20 SDK exactly as in `rotation.md`, then in a separate terminal:

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt
source /opt/ros/jazzy/setup.bash
source /home/zhaoyan-qian/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
source .venv/bin/activate
export HW_ENABLE_TOKEN=1

python -m src.comms.action_library_teleop \
  --library data/action_library/g20_right/v1 \
  --camera-index 2 \
  --side right \
  --rate 30 \
  --max-range-step 5 \
  --current-limit 20 \
  --speed-limit 35 \
  --enable-motion
```

The process still starts DISARMED. SPACE is the final human motion gate and also
publishes `/cb_right_recording_active`, so the existing recorder using
`--ros-trigger` remains synchronized. Hand loss disarms and clears the queue.

The current audited grouped library `data/action_library/g20_right/core_actions_v1`
contains:

| ID | Name | Human templates |
|---:|---|---:|
| 1 | `four_fingers_full_close` | 6 |
| 2 | `thumb_fold_inward` | 6 |
| 3 | `thumb_right_to_left_then_return_right` | 7 |
| 4 | `coordinated_finger_transition` | 7 |
| 5 | `four_fingers_index_leads_close` | 5 |
| 6 | `four_fingers_full_open_reverse_one` | 6 reversed from action 1 |

Strict completed-gesture leave-one-take-out validation at margin 0.015 gives
37/37 correct. IDs 1 and 5 keep thumb command channels `[0,5,10,15]` exactly
static. ID 5 differs temporally from ID 1: its index base and tip lead the other
three fingers early in the close. Old thumb-only predecessors are under
`core_actions_v1/archive/20260720`.

The stored ID 6 primitive still uses six strict time-reversed human templates
from ID 1. Its robot
trajectory first reaches the exact ID-1 endpoint, holds flexion for 24 frames
while q6--q9 converge from `[193,148,105,42]` to the official G20 GUI
finger-together preset with an operator-requested outer-finger tightening:
`[115,129,125,140]`. This moves index q6 and little-finger q9 another 10 ticks
inward while leaving q7/q8 unchanged, then holds that side pinch while playing
the reversed flexion. Near the endpoint, index/little base and tip channels
q1/q4/q16/q19 are capped at 235, retaining 20 ticks of flexion for cube
contact; middle/ring still open fully to 255. The resulting trajectory has
175 frames. Its
manifest sets `manual_from_start=true`, so number key 6 always preserves this
whole prefix. It also sets `best_effort_spread_feedback=true`: q6--q9 commands
continue to be published and step-limited, but coupled spread feedback cannot
pause the primary flexion trajectory or cause its catch-up timeout. Base, tip,
and thumb feedback remain under the following-error gate. Because cube contact
made q2/q3/q17 lag by 20--22 ticks, ID 6 alone sets `max_command_lead=28`;
its trajectory waits above 28 ticks and still hard-stops at the global
35-tick following-error limit. IDs 1--5 keep the CLI's 18-tick soft limit.

At runtime, number key 6 has a deliberate operator override in
`action_library_phase_teleop` and in the fixed replay window's single-action
key path. It reads the measured robot pose, selects the nearest ID-2 frame
using only q0/q5/q10/q15, plays the ID-2 thumb to its endpoint, and then
reverses to ID-2 frame 0. Every non-thumb active channel is copied from the
measured pose and held for the complete roundtrip. The stored ID-6 trajectory
is retained for provenance and for explicit fixed `--order` sequence playback.
Use `--thumb-roundtrip-key 0` to disable the runtime override.

## 5. Continuous library-pose following

`action_library_teleop` waits for a completed gesture and then plays the fixed
trajectory. `action_library_phase_teleop` now defaults to `live-pose`: each
camera frame is matched to the nearest recorded class/phase, so motion can run
forward, backward, and change class without replaying an entire trajectory.
The current `core_actions_v1` manifest uses
`finger_flexion_thumb_little_contact_v2`: index-through-little-finger lateral
spread is projected out of recognition features, while finger flexion, full
3-D thumb geometry, and thumb-tip to little-finger-tip distance remain. This
affects recognition only; q6--q9 in the recorded robot
trajectories are unchanged.

For bounded natural variation, `--control-mode hybrid-fingers` keeps the action
library as the main target and blends a small direct-MediaPipe residual into
only q1--q4 and q16--q19. Thumb q0/q5/q10/q15 and non-thumb spread q6--q9 remain
exactly library-controlled. The residual is independently weighted and capped
before the normal command-step and measured-state-lead guards. No random noise
is injected. If the current hand pose is outside every recorded class and the
matcher is not locked, the default `--hybrid-unlocked-fingers` fallback copies
direct MediaPipe flexion into q1--q4/q16--q19. Thumb and spread hold their most
recent locked-library target (or their measured arm-time pose before the first
lock), so an unknown gesture no longer freezes the four fingers. Use
`--no-hybrid-unlocked-fingers` to restore the old hold-while-searching behavior.

```bash
python -m src.comms.action_library_phase_teleop \
  --library data/action_library/g20_right/core_actions_v1 \
  --camera-index 2 \
  --side right \
  --tracking-mode live-pose \
  --control-mode hybrid-fingers \
  --hybrid-unlocked-fingers \
  --rate 30 \
  --lock-margin 0.015 \
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
  --reset-confirm-frames 3
```

This is dry-run unless `HW_ENABLE_TOKEN=1` and `--enable-motion` are both
present. SPACE arms/holds. Number keys 1--9 enter manual playback when that ID
is present: the selected
complete robot trajectory is played from its beginning without using
MediaPipe, after a bounded transition from the latest measured pose. Its final
pose is held; press the same key to replay, another number to change action, or
0 to return to automatic hybrid tracking. `--manual-blend-frames` controls the
minimum transition length. The command advances from the preceding command by
`--max-range-step`, but cannot lead measured state by more than
`--max-state-lead`. Hand loss disarms automatic tracking, while an
out-of-library pose uses the four-finger MediaPipe fallback described above;
stale ROS state always disarms and holds.

Action 4 also has an ordered feedback interlock enabled by default. It first
drives q0/q5/q10 to the action's left-aligned first pose while holding q15 at
its entry value. After the orientation is within `--a4-left-align-tolerance`
for `--a4-left-align-confirm-frames` consecutive frames, it holds that pose and
closes q15. Right rotation is released only after measured q15 is within
`--a4-thumb-tip-tolerance` for `--a4-thumb-tip-confirm-frames` consecutive
frames. This applies to automatic matching and manual key `4`; disable it only
explicitly with `--no-a4-thumb-tip-gate`.

With the command above, `--reset-on-start` performs one open-pose reset after
hardware and camera setup and then stays DISARMED. After it reports
`RESET COMPLETE`, DISARMED SPACE starts the ACT episode immediately.
`--reset-after-disarm` makes ARMED SPACE first stop and save the episode, then
return the hand to `G20_OPEN_POSE` while recording remains inactive. Once that
reset finishes, D publishes the delete-last request and also resets
manual/matcher state. The recorder refuses deletion during an active episode
and deletes only the latest completed episode in its current session.

`--reset-before-arm` remains available as a separate opt-in mode, but is not
enabled in the current recording command.

Primitive 3 (`thumb_right_to_left_then_return_right`) opts into
`thumb_little_roundtrip_v1` phase mapping through its manifest. The library uses
`finger_flexion_thumb_little_contact_v2`: four-finger lateral directions remain
projected out, while feature 77 retains the normalized full-3D thumb-tip to
little-finger-tip distance. Human inward motion maps phase 0--70%, placing the
robot at its left pivot on frame 105/150. Human motion away from the little
finger maps phase 70--100% and drives the restored 45-frame robot return. The
other four primitives retain linear frame phase.

In live-pose AUTO mode, A3 contact assist addresses direction-dependent class
acquisition at the shared open-hand endpoint. It still requires the full pose
to lie within `1.60 * A3 threshold` and no farther than `0.015` beyond the best
competing class, then acquires after contact phase reaches 8% for two frames.
A locked non-A3 action owns control, so A3 assist cannot steal action 2 while
the human hand reverses to its start pose. The round-trip controller remains
active through its own return and holds phase 100% until 0 returns to AUTO.
This lets A3 start before the pose becomes fully distinctive while rejecting a
matching contact scalar attached to an action 2 or action 4 pose. Disable it with
`--no-a3-contact-assist` for matcher-only comparisons.

A2/A3 ambiguity is additionally routed by adjacent four-fingertip spacing on
the hand-base lateral axis only, normalized by palm width. This avoids counting
finger depth or length differences as splay. Current live-camera examples put
together fingers at 0.240--0.254 and clearly spread fingers near 0.506, so the
default threshold is 0.350 with 0.030 hysteresis. Together fingers exclude every
primitive except A2 and disable A3 contact assist; if A2 is not accepted the
matcher remains searching instead of falling back to another primitive. Open
fingers remove all of these exclusions and restore normal all-action matching.
The main no-splay feature profile and all
robot trajectories remain unchanged. Disable it with `--no-a23-spread-routing`.

While ARMED, C latches the currently locked primitive (or the visible candidate
before lock) and its exact live phase, builds only the remaining trajectory
suffix, blends safely from measured robot state, and completes to phase 100%
without further MediaPipe input. It then holds the endpoint until 0 returns to
AUTO. C is rejected when there is no primitive ID at all, during reset, or while
a manual trajectory is already active. A4 completion continues to use its
ordered thumb gate.

During number-key or C manual playback, T keeps only the action's thumb
q0/q5/q10/q15 trajectory and maps every four-finger flexion/spread channel from
live MediaPipe; pressing T again restores the complete manual trajectory. With
no manual trajectory, T toggles FULL MEDIAPIPE teleop: all active G20 joints are
mapped directly and the action-library matcher, A3 assist, and A4 gate are
bypassed. The calibrated range map, collision corridor, command step, and
measured-state lead guards remain active. Press T again or 0 to return to AUTO.

F independently freezes the measured DexHand thumb channels q0/q5/q10/q15 at
their values when the key is pressed. Non-thumb targets continue to come from
the current manual, AUTO, or FULL MEDIAPIPE mode and retain the normal command
step/state-lead guards. Press F again to release. SPACE stop and D deletion do
not release the hold. If an episode reset is enabled, the reset moves only the
non-thumb joints and its completion check uses the held thumb values, so it
cannot time out merely because the thumb differs from the normal reset pose.
F cannot be changed during an active reset. Hand loss or stale joint feedback
still clears the hold as a fault-safety action.

Number key 6 is the explicit exception to an active F hold: while its
thumb-only ID-2 roundtrip is active, it temporarily owns q0/q5/q10/q15 and
keeps all four-finger channels fixed. The F latch itself remains set and
resumes when action 6 is exited.

`--freeze-thumb-on-start` performs the same latch automatically from the first
fresh measured DexHand state during startup. This happens before an optional
startup reset, so `--reset-on-start` resets only the non-thumb joints. F can
still release or capture a new thumb hold afterward.

`--startup-thumb-offsets DQ0,DQ5,DQ10,DQ15` optionally changes that startup
latch. It adds four SDK-tick offsets to the measured thumb, clips the resulting
target to 0--255, and lets the DISARMED startup reset reach and hold it. The
option therefore requires both `--freeze-thumb-on-start` and
`--reset-on-start`. Non-thumb channels still follow the configured reset pose.

`--startup-thumb-pose Q0,Q5,Q10,Q15` instead supplies an absolute four-channel
thumb target. It is mutually exclusive with `--startup-thumb-offsets` and has
the same freeze/reset requirements. This is the reproducible choice when a
previously tuned thumb pose must be restored regardless of the measured startup
pose.

R starts the configured episode-start-pose reset only while DISARMED and keeps
the ACT recorder trigger false. It cancels queued/manual playback but preserves
an F thumb hold, so only non-thumb joints move. R is refused during an active
episode; stop and save that episode with SPACE first.

These settings remain in the manifest if human templates are rerecorded again.

The original causal recognizer is retained as
`--tracking-mode one-way-sequence`. It locks a class from its gesture prefix and
only advances phase. Use it for token recognition, not continuous teleoperation.

Leave-one-take-out online-prefix evaluation of one-way mode on the current five
classes locks 29/29 takes to the correct class with zero wrong locks. Mean lock
point is 30% of the gesture and the worst is 48%, so this method has a
class-dependent recognition delay and is not zero-latency direct mapping.

Start with 8--12 primitives. Record 3--5 human takes per primitive, validate
recognition in dry-run, then grow the same manifest toward 50 primitives.

### Re-record all human templates without changing robot trajectories

Use the human-only recorder when the five G20 trajectories are already
calibrated. It opens only the MediaPipe camera, stages all five actions, runs
strict leave-one-take-out validation, and only then replaces the library's
human NPY files and thresholds. It never imports ROS or writes robot trajectory
files. Former human templates and the manifest are archived automatically.

```bash
python -m src.comms.rerecord_action_library_human \
  --library data/action_library/g20_right/core_actions_v1 \
  --camera-index 2 \
  --side right \
  --takes-per-action 5 \
  --min-take-frames 8 \
  --validation-margin 0.015 \
  --minimum-accuracy 1.0 \
  --install
```

Add `--action-id 3` to re-record and replace only action 3. Repeat
`--action-id` to select more than one action; omitting it records every action.

For each displayed action, M starts one take and M stops it. Repeat five times,
then SPACE accepts that action and advances. Q clears only the current staged
action; X/ESC aborts and leaves the live library unchanged. After action 5,
templates install only if validation is 100%. Restart any teleop process to
load the replacements.

## 6. Fixed-order library replay

To validate all five robot trajectories without MediaPipe recognition, replay
them once in explicit order:

```bash
python -m src.comms.replay_action_library_sequence \
  --library data/action_library/g20_right/core_actions_v1 \
  --order 1,2,3,4,5 \
  --reset-before-sequence \
  --robot-camera 0 \
  --rate 30 \
  --max-range-step 10 \
  --max-command-lead 18 \
  --catchup-timeout 5 \
  --retry-command-period 0.2 \
  --current-limit 100 \
  --speed-limit 100 \
  --clear-faults-before-reset \
  --pause-between 0.5
```

This is dry-run by default. Hardware additionally requires the exact human
token, `--enable-motion`, a passing ROS/state preflight, no competing GUI
publisher, and SPACE in the preview window. Every action transitions from the
latest measured state to its recorded start pose under the step limit, then
uses closed-loop pacing: if the preceding command leads measured state by more
than `--max-command-lead`, trajectory-frame advancement pauses until the hand
catches up. Each action must also settle before the next begins. Q/ESC aborts
and holds; R returns to the open pose.
By default, the SPACE-triggered run first returns to the open pose and waits for
measured state to settle; action 1 starts only after that reset succeeds.
SPACE and R also clear latched finger faults immediately before this reset.
While feedback is behind, the held command is re-published every
`--retry-command-period` seconds and the overlay identifies the worst channel,
for example `q15/thumb_tip`. `--thumb-current-limit` can override the thumb
torque independently while the other four fingers retain `--current-limit`.
Use `--no-clear-faults-before-reset` only when automatic
fault clearing is inappropriate for the hardware session.
The hardware window also accepts number keys 1--5. A number runs only that
primitive from the latest measured pose using the same bounded transition and
feedback checks. A timeout returns to the key menu instead of continuing the
sequence automatically, so the operator can retry the same number or skip to a
different action.
