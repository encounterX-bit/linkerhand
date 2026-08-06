# STATE

Active gate: G2 (dynamics + contact — last sim gate before hardware). The G2 DYNAMIC
closed-loop harness is BUILT (sim-agent G2-v2; see top of Handoff log): dynamics +
contact + the inline safety filter + CAN latency + grasps. G0 26/26, G1 57/57,
g2_safety 47/47, g2_dynamic 17/17 — all green (147 total). Do NOT advance past G2
(HUMAN gate).

Step 0 made G1 HONESTLY GREEN: the old fail-closed test_real_residual was split into
test_proximal_residual (HARD GATE, pooled prox p95 ≤ PROXIMAL_TOL) and
test_distal_residual_monitored (report + regression guard). G1_RESIDUAL_THRESHOLD is
retired; the new env knobs are PROXIMAL_TOL, REGRESSION_MARGIN, and (optional, real-
camera) G1_DISTAL_RESIDUAL_THRESHOLD. Three PROPOSED values await human sign-off
(see Next 1/3/4).

ONE human decision was made during the refactor: the closed-form-era 3 kHz timing
budget was reconciled to the iterative fingertip-distal reality (ADR-0007). All
correctness/safety/hardware invariants are unchanged. See the handoff entry.

The G2 safety FILTER has been BUILT + validated in sim-free isolation (src/safety,
tests/g2_safety 47/47 green; see top of Handoff log). It is **built**, not gated
past: G2 itself remains a HUMAN gate and this module **needs human review before
merge** (root CLAUDE.md: src/safety changes require explicit human review). Two
items await human sign-off — see Next.

## Next
1. (G1, HUMAN — proximal sign-off) PROXIMAL_TOL = **0.15 rad** is PROPOSED, committed
   in tests/g1_kinematic/residual_baseline.json, env-overridable. **IMPORTANT — the
   proximal is NOT near-zero at the tail** as the old p50=0.000 framing implied:
   pooled prox p95=0.1236, p99=0.167, max=0.219 rad (870 samples over 174 frames).
   The tail is POSE-CORRELATED (mid-curl transition frames 40-44/88-89/106-108/
   132-137, all fingers together), consistent with 2-DoF base under-actuation on
   transient curled SYNTHETIC poses — NOT a solver regression (proximal is the exact
   base-solve segment). 0.15 gives ~21% headroom over observed p95, sits below the
   distal p95 (0.20), and trips on any real base-solve break. Decision is the same
   solver-tuning-vs-threshold axis as distal. test_proximal_residual is a LIVE hard
   gate (passes today); set PROXIMAL_TOL to bless/retighten. (per-finger prox p95:
   thumb 0.123 / index 0.103 / middle 0.095 / ring 0.130 / little 0.134.)
1b. (G1, distal — monitored) test_distal_residual_monitored passes by default and
   REGRESSION-guards vs the committed baseline (overall p95 0.178, thumb dist 0.143)
   at REGRESSION_MARGIN=0.20 (PROPOSED). Current: overall p95=0.169, distal pooled
   p95=0.200, thumb dist p95=0.143 — no regression. Set G1_DISTAL_RESIDUAL_THRESHOLD
   from real-camera data to add an absolute distal gate. Distal residual is the
   under-actuated curled distal vs the human TIP-PIP aggregate (ADR-0002/0006).
2. (comms/G3) Cross-check the thumb base↔abduction assignment against the actual
   linkerhand-ros-sdk command-array order (see Blocked) — the KINEMATIC role is now
   confirmed (test 4); the SDK index label is a separate, hardware-facing check.
3. (G2, HUMAN — safety review) src/safety needs explicit human review before merge.
   Two sign-offs requested:
   (a) FILTER_LATENCY_REGRESSION_MARGIN — proposed 0.50 (src/safety/config.py).
       Committed p99 baseline = 11,500 µs (best-of-3; measured right ~11.4 ms /
       left ~11.9 ms, p50 ~2.3 ms, collision-free fast path ~0.85 ms). The hard
       real-time ceiling is the 33,333 µs frame (worst observed ~11.9 ms, ~3×
       headroom). Baseline is machine-specific — re-measure on the target box and
       adjust if it moves materially. Test: tests/g2_safety/test_timing.py.
   (b) The collision PROXY + margin (ADR-0008): capsule radii = ½ smallest mesh
       extent, separation margin 2 mm, palmar half-plane x0=-0.005. Non-penetration
       is guaranteed only vs this proxy, not the full meshes; confirm the proxy is
       conservative enough (or tighten radii/margin) before it ever gates hardware.
   The force-clamp (15 N « 100 N) + watchdog (0.20 s → open-hand pose) are SPECS
   only; comms enforces at G3 and a HUMAN sets HW_ENABLE_TOKEN. Do NOT advance
   past G2 (human gate).
4. (G2 dynamic, HUMAN — baselines) committed in tests/g2_dynamic/baseline.json,
   all machine-specific where timing is involved:
   (a) LOOP_REGRESSION_MARGIN = 0.50 (PROPOSED) on p99-compute baseline 11,000 µs
       (live ~11.1 ms; absolute ceiling 33,333 µs, ~3× headroom).
   (b) The VIRTUAL FORCE CAP = per-joint motor torque 0.12 Nm (ADR-0009). A torque
       limit does NOT linearly bound TOTAL grip force (fingers sum); 0.12 Nm keeps
       the worst grasp (sphere) ≤ 15 N. The 15 N safety cap itself is comms/G3.
   (c) PD gains (Kp0.3/Kd0.6) + grasp poses + latency ceiling (2.0 rad) — sim-only.
   Grasp TUNING FINDINGS (not forced green): a free fingertip PINCH is fragile (the
   thumb's lateral opposition bats a small free sphere; no wrist to pre-load) — the
   robust sphere scenario is a palm-backed enveloping grasp; objects need a reaction
   surface (palm-ward gravity) + a slow ramped close or they eject.

## Blocked
- Thumb base↔abduction KINEMATIC role CONFIRMED at G1 (test_thumb_axis): driving
  idx 0 = thumb_cmc_pitch flexes the thumb IN the palm plane toward the fingers,
  while idx 5 = thumb_cmc_roll lifts it OUT of the plane (abduction). So the
  LIMITS.md label fix (base<-cmc_pitch, abduction<-cmc_roll) is geometrically
  right. STILL OPEN: whether the L20 SDK's command-array index order calls these
  "base" vs "abduction" the same way — verify against linkerhand-ros-sdk in
  comms/G3. Ranges for both recorded, so clamping is safe regardless.

## Interface note (cross-module, do not patch here)
- The ticket says import finger_retarget's `solve()`; the solver's actual public
  entry point is `retarget(landmarks, side=...)` (src/finger_retarget/__init__.py).
  The G1 harness imports `retarget` read-only. If a `solve` alias is wanted,
  solver-agent should add it; sim-agent did not modify finger_retarget.

## Handoff log

- (2026-07-20, A4 ordering correction: left then close then right) Fixed the first feedback-gate implementation reversing the operator's intended A4 prelude: it froze q0/q5/q10 and closed q15 first, so manual key 4 visibly curled before moving left. Reworked the gate into a measured three-stage state machine shared by auto A4 and key 4. Stage 1 drives q0/q5/q10 to the primitive's left-aligned first pose while pinning q15 to its entry value; after max orientation error <=5 for 3 frames, stage 2 holds that left pose and closes q15; after q15 error <=5 for 3 frames, stage 3 releases the right turn, keeping q15 closed on the release frame. Manual frame advancement remains frozen through both waiting stages and playback is rebased from measured state on release. Added `--a4-left-align-tolerance`/`--a4-left-align-confirm-frames`, updated the hardware runbook and docs, and retained the existing tip tolerance controls. Closed-loop offline simulation held q15 exactly at 100 throughout left alignment, began closure only at orientation error 0, held orientation error 0 during closure, and released right q5 only with q15=0. Whole-library validation remains 30/30 and 40 focused tests pass. No camera, ROS publisher, or hardware motion was started.

- (2026-07-20, A4 measured thumb-tip-before-rotation gate) Fixed phase teleop allowing A4 thumb rotation to begin while the physical q15 motor was still closing. The action's first target was already q15=0, but live-pose matching can initialize at a later phase and manual playback previously advanced frames while feedback lagged; the global per-joint state-lead cap did not express a cross-joint ordering constraint. Added a default-on feedback interlock shared by automatic A4 matching and manual key 4: on A4 entry, measured q0/q5/q10 orientation is held while q15 is driven to the primitive's closed first-pose target; rotation releases only after measured q15 error <=5 for 3 consecutive frames. The release frame itself keeps q15 closed. Manual key 4 freezes its frame counter while waiting, then rebases/rebuilds playback from the latest measured state so no A4 frames are skipped. The gate resets on action change, key replay/restart, AUTO, delete, SPACE, and hand loss. Added `--[no-]a4-thumb-tip-gate`, tolerance, and confirm-frame options; enabled them explicitly in `hardcode_position.md` and documented them. Feedback simulation held q5 exactly at entry until q15=0 and released rotation afterward. Whole-library validation remains 30/30 and 40 focused tests pass. No camera, ROS publisher, or hardware motion was started.

- (2026-07-20, action 2 endpoint q5=60/q15=25 and digit-key audit) Refined shared-library primitive 2 `thumb_fold_inward` endpoint only: q5 thumb side-swing 55->60 and q15 thumb tip 30->25 (lower q15 means 5 ticks more closure). All earlier trajectory waypoints, other endpoint joints, six human templates, and factual recorded command/state waypoints remain exact. Regenerated the installed trajectory at 91x20 with max active step 5.0. Audited `action_library_phase_teleop`: digit keys resolve `library.primitives[requested]` from the startup-loaded `core_actions_v1` and pass that exact trajectory to `playback_trajectory`; an offline digit-2 build ended at q5=60/q15=25. Therefore no separate key mapping needed a code change; an already-running process must be restarted, and with the current reset-on-start runbook the operator must SPACE-arm before digit 2 moves (or queue 2 while disarmed, then arm). Archived the q5=55/q15=30 version under `archive/20260720_action2_before_endpoint_q5_60_q15_25`. Whole-library validation remains 30/30 and 38 focused tests pass. Physical verification remains pending; no ROS publisher or hardware motion was started.

- (2026-07-20, action 2 final thumb side-swing +10) Updated shared-library primitive 2 `thumb_fold_inward` so only its final curated q5 (G20 thumb side-swing) changes 45->55; the preceding q5=109 waypoint, q15=30 endpoint, all other trajectory waypoint values, six human templates, and factual recorded command/state waypoints remain unchanged. Regenerated the installed trajectory at 91x20 with max active step 5.0. The `core_actions_v1` update is consumed by phase/live teleop, digit-2 manual replay, and fixed sequence replay after process restart. Archived the preceding manifest/trajectory/source under `archive/20260720_action2_before_endpoint_q5_plus10`. Whole-library validation remains 30/30 and 38 focused tests pass. Physical verification remains pending; no ROS publisher or hardware motion was started.

- (2026-07-20, action 4 two live-pose prelude) Replaced curated Action-4 GUI/trajectory poses A4-01 through A4-04 with two operator-positioned dexhand states read once from `/cb_right_hand_state`: pose 1 `[254,178,202,203,179,0,190,147,104,87,51,0,200,200,158,209]` at stamp 1784549847.470271842 and pose 2 `[254,178,202,203,179,126,190,147,104,88,51,0,200,200,158,209]` at stamp 1784549882.175204018 (16 active dimensions). For command poses, reserved q11-q14 are pinned to 255. The prior post-prelude hand-transition, forward-push, and endpoint poses remain exact and were renumbered A4-03 through A4-05 in the SDK GUI. Curated waypoints are now 5 and installed action 4 is 91x20 with max active step 4.80. Shared `core_actions_v1` means phase/live teleop, manual digit replay, and fixed sequence replay all use the update after process restart. Archived the preceding manifest/trajectory/source/GUI constants under `archive/20260720_action4_before_two_live_pose_prelude`. Whole-library validation remains 30/30 and 38 focused tests pass. Physical replay remains pending; both ROS captures were read-only and no hardware command was sent by Codex.

- (2026-07-20, action 4 direct three-step prelude) Inspected the 11.49 s `22-04-41` physical replay and compared the action-3/4 boundary. The main redundant motion was structural: action 3 ended at q15=140 but action 4 started at q15=30, so sequence replay first closed the thumb tip before the requested left turn; action 4 also contained a duplicate settle plus q0=204/q5=55 and q0=255/q5=53 right-turn intermediates. Simplified the curated action-4 waypoints from 10 to 7: inherit action 3 at q15=140, one left move to q0=95/q5=15, close q15 to 0, move directly to the right-aligned q0=255/q5=134 pose, then preserve the three post-alignment transition/push/end poses. Action-3->4 active-joint boundary max delta is now 1 and q15 delta is 0. Regenerated installed action 4 from 175 to 154 frames at max active step 4.67; endpoint is unchanged. Synced the SDK GUI from 9 to 7 matching A4 presets. Archived the preceding manifest/trajectory/source/GUI constants under `archive/20260720_action4_before_three_step_prelude`. Whole-library validation remains 30/30 and 38 focused tests pass. Physical cube verification of the simplified path remains pending; no ROS publisher or hardware motion was started.

- (2026-07-20, action 4 thumb side-swing synchronized) Propagated the operator-tested GUI adjustment into the shared `core_actions_v1` primitive used by hybrid/live teleop and sequence/manual replay. Action-4 trajectory waypoints 1-3 now move/hold q5 (G20 thumb side-swing) at 15 instead of 70; the raw recorded waypoints, initial q5=70, later q5 path, all other semantic waypoint values, timing, and final pose remain unchanged. The duplicate close/settle pose was updated as well. Regenerated the installed action-4 trajectory at 30 Hz with max-step 5: 175x20, max active step 4.07. Archived the preceding manifest, installed trajectory, and source metadata under `archive/20260720_action4_before_thumb_side_q5_15_sync`. Whole-library validation remains 30/30 and 38 focused library/replay/teleop tests pass. Matching GUI presets `A4-02`/`A4-03` live in the SDK constants. Physical verification remains pending; no ROS publisher or hardware motion was started.

- (2026-07-20, action 4 left-close-right thumb staging) Inspected the 22.03 s follow-up screencast: the prior close-then-right version still contacted the cube poorly and eventually hit the existing q0/thumb-base catch-up timeout. Updated only the action-4 prelude to the operator-requested order: q0 moves left 114->95 over 0.3 s, q15 then closes 30->0 over 0.5 s while q0 holds 95, both settle for 0.2 s, and only then q0 begins the original rightward path toward 204/255. Installed trajectory is 166 frames: left completes at frame 9, q15 motion starts at 10 and reaches 0 at 24, right motion starts at 31. All joints except q0/q15 retain the prior path exactly after the prefix; raw recorded waypoints and final pose remain unchanged; max active step is 4.07. Archived the preceding version under `archive/20260720_action4_before_left_close_right_stage`. Whole-library validation remains 30/30; action-4 dry-run and 38 focused tests pass. Catch-up limits were intentionally not relaxed; physical cube verification remains pending and no hardware motion was started.

- (2026-07-20, action 4 pre-rotation thumb-tip staging) Inspected the 15.14 s operator screencast and current primitive-4 trajectory. Diagnosed that q0 rotation began immediately while q15 stayed at 30 for the first 31 installed frames, making side contact precede the desired frontal push. Updated the curated action-4 trajectory waypoints to stage q15 30->0 over 0.5 s, hold q15=0 for 0.2 s, and only then begin the original q0/other-joint motion. Regenerated the installed trajectory from 136 to 157 frames: q15 reaches 0 at frame 15, q0 begins at frame 22, endpoint remains q15=50, all non-q15 original frames are bit-exact after the 21-frame prefix, max active step is 4.07, and raw recorded waypoints remain unchanged. Archived the prior manifest/trajectory/source metadata under `archive/20260720_action4_before_preturn_q15_stage`. Whole-library validation remains 30/30; action-4 sequence dry-run and 38 focused tests pass. Physical cube verification remains pending; no ROS publisher or hardware motion was started.

- (2026-07-20, action 2 paired from separate latest captures) Replaced primitive 2 `thumb_fold_inward` by intentionally pairing six human takes (40/43/34/33/35/34 frames) from penultimate complete human capture `20260720_210828.../group_000` with four robot waypoints from latest complete dexhand capture `20260720_211140.../group_000`; the latest group's unrelated 5-frame human take was excluded. Extended `import_action_group` with explicit `--human-group`/`--robot-group` inputs and persisted both resolved provenance paths. The installed trajectory has 91 frames, max adjacent active step 5, recorded command/state error 3, open start, and endpoint q5=45/q15=30. Old action 2 manifest and complete primitive folder are archived under `archive/20260720_action2_before_210828human_211140robot_pair`. Whole-library validation remains 30/30 (100%); sequence dry-run passes. 18 importer/library tests pass. No camera, ROS publisher, or hardware motion was started.

- (2026-07-20, one-time startup open reset) Added opt-in `--reset-on-start` to hybrid phase teleop and enabled it in the current `hardcode_position.md` command. Once hardware and camera setup finish, the runner drives `G20_OPEN_POSE` under the existing step/state-lead/tolerance/timeout guards while recording-active stays false; reset completion remains DISARMED, so the operator's later SPACE starts the episode directly. The existing post-episode `--reset-after-disarm` behavior remains enabled. Updated runbook and action-library docs. No driver, camera, ROS publisher, recorder, or hardware motion was started.

- (2026-07-20, post-episode-only open reset and D delete hotkey) Added opt-in `--reset-after-disarm` to hybrid phase teleop and enabled it in the current `hardcode_position.md` command; the current command intentionally does not enable `--reset-before-arm`. DISARMED SPACE starts recording immediately. ARMED SPACE publishes recording-inactive first, then returns to `G20_OPEN_POSE`; completion remains DISARMED and never starts another ACT episode. The delete-last hotkey moved from R to D/d and is rejected while armed or resetting. Updated overlay/runbook/action-library docs. 24 focused tests plus byte-compilation and CLI checks pass; the wider comms collection remains blocked during collection by the environment's missing optional `yourdfpy`. No driver, camera, ROS publisher, recorder, or hardware motion was started.

- (2026-07-20, restored three-terminal action-library recording runbook) Added workspace-level `record_action_library.md` containing the current three-terminal raw primitive collection workflow: G20 palm-touch driver, official GUI as the sole command publisher, and read-only grouped MediaPipe/robot waypoint recorder on cameras 2/0. Documented the current M-toggle repetition flow, SPACE phase/group transitions, S waypoint snapshots, Q/E scoped retry archives, obsolete H key, output layout, conflict warnings, and offline analysis command. Recorder/analyzer CLI and byte-compilation checks pass. No driver, GUI, camera, ROS publisher, or hardware motion was started.

- (2026-07-20, pre-episode open reset and delete-last recording hotkey) Added opt-in `--reset-before-arm` to hybrid phase teleop. A DISARMED SPACE now drives `G20_OPEN_POSE` under existing command-step/state-lead guards, requires three measured frames within 12 ticks, and only then arms teleop and publishes recording-active; timeout/cancel/stale state remain disarmed and never start an ACT episode. Added `/cb_<side>_recording_delete_last`: DISARMED R publishes an Empty request while also clearing manual/matcher state. The SDK touch recorder now accepts that request, refuses it during recording, deletes only the latest completed episode in its current session after exact path validation, decrements/reuses its index, and reports the result. Updated the current hardcode command/docs. Offline deletion behavior passed against temporary episode directories; 33 focused tests plus Python/CLI checks pass. No hardware was actuated.

- (2026-07-20, hardcode runbook cleanup) Rewrote the workspace-level `hardcode_position.md` as a concise current-workflow runbook (229 -> 140 lines). It now contains only the G20 palm-touch driver, camera-0 ACT recorder, current camera-2 hybrid phase teleop with key behavior/action IDs, ACT episode inspection, offline human-template rerecording, and library validation. Removed the competing GUI, obsolete token-only teleop, duplicate/non-hybrid phase command, early group recorder, and standalone sequence replay now superseded by number-key manual playback. Checked all retained Python entrypoints via `--help` or byte compilation. No process was started and no hardware was actuated.

- (2026-07-20, ACT demonstration recording handoff) Confirmed the existing SDK `linkerhand_g20_touch_recorder.py` records synchronized camera images, 20-D measured state, 20-D published command, and tactile mass/matrices, and that `action_library_phase_teleop` publishes `/cb_right_recording_active` from its SPACE arm state. For the current camera layout, hybrid teleop owns camera 2, so ACT scene recording must use camera 0 to avoid a camera conflict. Runtime inspection found the G20 palm-touch driver and state/command/touch topics live, with no teleop or recorder process running. No hardware command was published by Codex.

- (2026-07-20, action-3 motion-progress phase mapping) Diagnosed long/variable low-motion tails in the six rerecorded action-3 human templates. Added persisted per-primitive phase mapping settings and `motion_progress_v1` phase axes to both live-pose and causal matchers. Only action 3 opts in: cumulative feature motion replaces linear frame time, and a stable suffix within RMS distance 0.012 of the final five-frame median snaps to phase 1.0. Existing data now reaches >=98% 3-7 frames earlier; leave-one-take-out cross-take phase MAE fell from about 0.076 to 0.053, while library recognition remains 30/30 and 47 relevant tests pass. Other actions remain frame-linear; robot trajectories and human NPY files were not modified. Full comms collection still requires optional environment packages `yourdfpy` and `torch`. No hardware was actuated.

- (2026-07-20, manual number-key playback in hybrid teleop) Changed live-pose number keys 1-9 from forced MediaPipe phase matching to camera-independent full action-library playback. A key builds a max-step-bounded transition from the latest measured/commanded pose into the recorded primitive, plays the complete trajectory while ignoring MediaPipe, then holds the endpoint; the same key replays, another key switches actions, and 0 restores AUTO hybrid tracking. Manual playback remains usable with no detected hand and retains joint-state-stale disarm plus normal command/state-lead guards. Added `--manual-blend-frames` (default 8), overlay/log status, and runbook/docs updates. Offline expansion of all five primitives stayed within the 10-tick step bound; 37 focused tests passed. No hardware was actuated.

- (2026-07-20, out-of-library four-finger fallback) Fixed `hybrid-fingers` freezing all joints whenever the action matcher was unlocked. Fresh MediaPipe frames now directly drive q1-q4/q16-q19 while unmatched; q0/q5/q10/q15 and q6-q9 hold the most recent locked-library target, seeded from the measured pose when SPACE arms. Locked poses retain the library-plus-bounded-residual behavior. Added the default-on `--hybrid-unlocked-fingers` / opt-out `--no-hybrid-unlocked-fingers`, runbook/docs, and a channel-isolation unit test. Existing command-step, measured-state-lead, hand-loss disarm, and stale-state disarm guards remain active. No hardware was actuated.

- (2026-07-20, looser non-thumb MediaPipe clipping) Increased the opt-in `hybrid-fingers` residual caps from 12 to 20 ticks for q1-q4 and from 15 to 25 ticks for q16-q19, while leaving blend weights at 0.15/0.20. Thumb q0/q5/q10/q15 and spread q6-q9 remain exactly library-controlled. Updated both dry-run/hardware runbook commands and action-library docs. No hardware was actuated.

- (2026-07-20, library-thumb/finger-residual hybrid teleop) Added opt-in `hybrid-fingers` control to online phase teleop. `q0/q5/q10/q15` (thumb) and `q6-q9` (finger spread) are copied exactly from the action-library trajectory; MediaPipe retargeting affects only `q1-q4` (15%, max 12 ticks) and `q16-q19` (20%, max 15 ticks) before the existing step/state guards. The residual is applied only while recognition is fresh and locked. Updated the runbook/docs and added coverage for mapping, residual limits, and exact thumb/spread preservation. Offline action-4 inspection showed identical thumb/spread values before and after blending; 31 focused tests passed and the library still validates at 30/30. No hardware was actuated.
- (2026-07-20, human-only five-action rerecorder) Added
  `src.comms.rerecord_action_library_human` for replacing all MediaPipe takes
  while preserving the calibrated G20 motion library. It imports no ROS,
  creates no publisher, stages five takes per displayed primitive with M/M and
  SPACE progression, supports Q redo and abort-without-install, rejects short
  takes, and requires configurable leave-one-take-out accuracy (runbook uses
  100%) before installation. Installation archives the old manifest and human
  NPYs and updates only human templates/thresholds; a binary-preservation test
  confirms robot trajectory NPYs are unchanged. Added the complete command to
  `hardcode_position.md` and ACTION_LIBRARY.md. Twenty-nine focused tests pass;
  the current library remains 29/29. No capture was started and no hardware was
  actuated by Codex.
- (2026-07-20, action recognition ignores non-thumb finger splay) Added the
  manifest-pinned `finger_flexion_no_splay_v1` MediaPipe feature profile. It
  projects index-through-little-finger geometry onto the hand-base x/z flexion
  plane while retaining the full 3-D thumb; this changes recognition only and
  leaves robot q6--q9 trajectories untouched. Both completed-token and online
  phase teleop now extract live features through the library profile, and
  future imports preserve/recalibrate that profile. Recalibrated the five
  thresholds and reduced live AUTO's class margin from 0.030 to 0.015: stored
  replay locks 98.0% of frames with zero wrong locked frames. Synthetic lateral
  offsets produce exactly zero feature change. Leave-one-take-out validation
  remains 29/29 and 27 focused tests pass. The full comms test collection is
  unavailable in system Python because optional `yourdfpy` is absent. No
  hardware was actuated by Codex.
- (2026-07-20, calibrated primitives synced to real-time teleop) Confirmed that
  `action_library_phase_teleop` loads the same
  `data/action_library/g20_right/core_actions_v1` trajectories used by fixed
  replay, so no duplicate export is needed and a process restart picks up each
  calibration. Current IDs 1--5 load successfully; action 4 loads 136 frames
  with q15 start/min/end 30/0/50. Updated the operator runbook to make the
  shared-library/restart behavior explicit. Library validation remains 29/29;
  no hardware was actuated by Codex.
- (2026-07-20, action-4 thumb-tip timing override) Updated only primitive 4
  `coordinated_finger_transition`'s replay q15 (thumb tip): the first two
  waypoints use 30 instead of the recorded 140 for more initial curl, the
  middle three remain fully curled at 0, and the last two use 50 for more final
  extension than the preceding q15=20 trial. The other 19 channels are
  unchanged. Preserved the factual recorded command/state waypoints and
  archived each former compiled variant with its JSON and manifest under the
  action/library revision folders. The rebuilt trajectory is 136 frames with
  max active step 4.07 ticks and endpoint q15=50. Library
  validation is 29/29 and the 20 focused replay/library tests pass. Physical
  verification remains pending; no command was published and no hardware was
  actuated by Codex.
- (2026-07-20, deeper action-2 thumb endpoint) Updated primitive 2
  `thumb_fold_inward` to end at the operator-provided GUI command
  `[79,255,255,255,255,95,193,148,105,42,222,255,255,255,255,157,255,255,255,255]`.
  Relative to the former endpoint, q5 moves 125->95 and q15 moves 255->157.
  Preserved the factual original recorded command/state waypoints, added an
  explicit pending-verification trajectory override, and archived the former
  waypoint JSON and compiled trajectory. The rebuilt trajectory is 76 frames
  with max active step 3.63 ticks. Live state at edit time still reported the
  old pose (q5=125, q15=254), so physical verification remains pending. Library
  validation is 29/29; no command was published and no hardware was actuated by
  Codex.
- (2026-07-20, torque/speed 100 and manual primitive keys) Set G20 driver
  startup torque and speed to `[100]*5`; fixed action-library replay now also
  defaults to and accepts `--current-limit 100 --speed-limit 100`. Added
  hardware-window number keys 1--5: each runs only the corresponding primitive
  from latest measured state with the existing bounded transition, fault clear,
  stale-state/following-error checks, and returns to the menu on completion or
  timeout. SPACE still runs the full order; R returns open; Q/ESC aborts. A
  timeout never automatically skips a safety check, but the operator can retry
  or choose another primitive. Tests pass and no hardware was actuated by Codex.
- (2026-07-20, fixed-sequence torque 70) Set the G20 palm-touch driver's
  startup `DEFAULT_TORQUE` to `[70]*5` and the fixed action-library sequence's
  default/command `--current-limit` to 70. Removed the old thumb-only 30
  override so the sequence publishes `[70,70,70,70,70]`. The fixed runner
  accepts 1--70 and still requires the hardware token, enable flag, ROS
  preflight, and SPACE. Direct MediaPipe/action-token teleop commands remain at
  their prior lower torque settings. Tests pass and no hardware was actuated by
  Codex.
- (2026-07-20, restore index-leading four-finger close) Corrected the previous
  endpoint-only duplicate classification after inspecting trajectory timing.
  The former action has a distinct early phase: index base/tip reach 108/207
  while the other base channels are 170/162/141 and middle/ring tips remain
  255. Restored it as ID 5 `four_fingers_index_leads_close`, with six matching
  MediaPipe templates. To avoid renaming directories whenever the library
  grows, active paths are now the count-independent
  `data/action_groups/current_actions` and
  `data/action_library/g20_right/core_actions_v1`. Fixed replay defaults to
  `1,2,3,4,5`; five-class leave-one-out validation is 29/29 and dry-run is 715
  frames / 25.83 s. No hardware was actuated by Codex.
- (2026-07-20, temporary duplicate classification; superseded by the entry
  above) Compared all active
  G20 trajectory endpoints and archived former ID 3
  `four_fingers_partial_tip_close`: it was the nearest pair with ID 1
  `four_fingers_full_close` (13-tick mean endpoint difference), while its source
  recording had the worse maximum following error (43 versus 18 ticks). The
  active set is now four distinct consecutively numbered actions under
  `data/action_groups/current_four_actions` and
  `data/action_library/g20_right/core_four_actions_v1`: 1 four-finger full
  close, 2 thumb fold inward, 3 all-finger partial close, and 4 coordinated
  transition. The removed source and compiled primitive remain recoverable in
  their respective `archive/duplicates` directories. Commands, docs, and the
  fixed-order default are now `1,2,3,4`; validation is 23/23. No hardware was
  actuated by Codex.
- (2026-07-20, action-data cleanup and semantic naming) Reorganized only the
  G20 action-library data; ACT/rotation/self-imitation datasets were left
  untouched. The five active source recordings now live under
  `data/action_groups/current_five_actions/01_...` through `05_...`; unused
  complete groups, incomplete captures, and original session metadata are under
  `data/action_groups/archive/{retired_complete,incomplete,session_metadata}`.
  Renamed the loadable library to
  `data/action_library/g20_right/core_five_actions_v1`, action 1 to
  `four_fingers_full_close`, and action 3 to
  `four_fingers_partial_tip_close`. Updated manifest relative paths, absolute
  source provenance, docs, and run commands. Added `data/README.md`. Archived
  data remains recoverable; nothing outside action_groups/action_library was
  moved or deleted. No hardware was actuated by Codex.
- (2026-07-20, pre-sequence reset) Fixed-order library replay now defaults to
  `--reset-before-sequence`: the operator's SPACE press first performs a
  step-limited, closed-loop return to `G20_OPEN_POSE`, waits for measured state
  to settle, and only then starts primitive 1. A failed/stale/timed-out reset
  blocks the rest of the sequence. `--no-reset-before-sequence` exists only as
  an explicit override. No hardware was actuated by Codex.
- (2026-07-20, replace primitives 1 and 3 with thumb-static four-finger actions)
  Replaced library ID 1 `thumb_across_palm` with `four_fingers_full_close` from
  `20260720_100232.../group_000` (5 takes, 151 frames), and ID 3
  `thumb_full_opposition` with `four_fingers_partial_tip_close` from that session's
  `group_001` (6 takes, 106 frames). Both source trajectories have exactly zero
  command span on thumb channels `[0,5,10,15]`. Primary joint command/state
  error is at most 4/10 ticks; the larger 18/43 values are isolated to coupled
  four-finger spread feedback. The importer now supports an explicit separate
  spread-coupling audit plus a required-static-thumb check and records both in
  the manifest. Fixed-sequence feedback/settling ignores only static q6..q9
  channels, while any spread channel intentionally moved by a trajectory remains
  checked. Old ID 1/3 directories were moved to recoverable
  `archive/20260720` and are absent from the manifest. Completed-gesture
  leave-one-out is 29/29; causal prefix mode at the new 0.03 default margin is
  also 29/29, mean lock 30%, worst 48%. No hardware was actuated by Codex.
- (2026-07-20, fixed 1-to-5 library replay) Added
  `src/comms/replay_action_library_sequence.py` to replay an explicit primitive
  order (default `1,2,3,4,5`) without MediaPipe recognition. It is dry-run by
  default; hardware requires the exact human token, `--enable-motion`, ROS/state
  readiness, no competing GUI publisher, and operator SPACE. Each primitive is
  safely blended from latest measured state, step limited, following-error and
  stale-state checked, and required to settle before the next primitive. After
  the first hardware trial stopped at `following error 43`, playback was changed
  to closed-loop pacing: frame advancement pauses whenever command lead exceeds
  18 ticks and resumes only after measured state catches up, with a 5 s timeout;
  the 35-tick following error remains a separate hard stop. ESC/Q aborts and
  holds; R returns open. Current-library dry-run produces 716 frames,
  max active step under 10 ticks, and 25.87 s nominal duration including pauses.
  Unit/regression tests pass; Codex did not actuate hardware.
- (2026-07-20, continuous action-library pose teleop) Changed
  `action_library_phase_teleop.py` default tracking to reversible `live-pose`.
  A new `LivePoseMatcher` performs per-frame nearest recorded pose matching,
  two-frame class hysteresis, bounded bidirectional phase updates, and
  out-of-library hold. Number keys 1--5 force a known primitive immediately and
  0 restores automatic selection; `--primitive-id` provides the same selection
  at startup. The previous causal forward-only behavior remains available as
  `--tracking-mode one-way-sequence`. Hardware commands now slew from the last
  command rather than repeatedly from measured state, with a separate
  `--max-state-lead` guard (default 30 ticks) to prevent command backlog. This
  directly addresses the run where phase reached 100% in about one second but
  the state-based five-tick command then spent many seconds chasing the end
  pose. Recommended first hardware validation remains current 20, speed 50,
  step 10, manual ID selection, operator SPACE, and no GUI publisher. Relevant
  matcher/command tests pass; the full comms test collection still requires the
  optional `torch` dependency used by unrelated visual-ACT tests. No hardware
  was actuated by Codex.
- (2026-07-20, online action-token phase teleop) Added a causal
  `OnlinePhaseMatcher` and `src/comms/action_library_phase_teleop.py` so action
  library motion can begin after a distinctive prefix and then follow live
  MediaPipe progress, rather than waiting for the full gesture and replaying the
  complete trajectory. Prefix alignment allows bounded template-rate changes;
  class locking requires phase, distance, margin, and consecutive-frame gates.
  After lock, identity cannot switch and phase is monotonic with an 8%-per-frame
  cap; cumulative prefix alignment is augmented by current-pose phase to handle
  held-out timing variation. A full leave-one-take-out run over all 5 classes / 29
  takes produced 29 correct locks, zero wrong locks, mean lock point 30%, and
  worst lock point 73%. The new runner publishes no position while SEARCHING,
  starts DISARMED, requires human token + `--enable-motion` + SPACE, refuses a
  competing GUI command publisher, caps measured-state motion at five ticks per
  frame, and disarms/holds on hand loss or stale state. Dry-run and hardware
  commands are documented in `../hardcode_position.md`. No camera, ROS command,
  enable token, or hardware motion was started by Codex.
- (2026-07-20, four new grouped actions audited and imported) Audited the two
  new sessions 20260720_112330 and 20260720_114259. Their four complete groups
  contain 24 exact takes / 1086 fresh MediaPipe frames and 22 robot waypoints;
  frame/image/take/JSON counts all agree and every image decodes. Maximum
  recorded active-joint command/state errors are 4, 4, 2, and 2 SDK ticks, so
  all four pass the 10-tick hardware-library import gate. Visual and trajectory
  review assigned provisional names `thumb_fold_inward`,
  `thumb_full_opposition`, `all_fingers_partial_close`, and
  `coordinated_finger_transition`; imported them as IDs 2--5 beside existing ID
  1 in `data/action_library/g20_right/core_five_actions_v1`. Changed automatic import
  thresholds from largest pairwise intra-class distance to worst leave-one-out
  nearest-same-class distance plus 0.025, reducing overlap caused by outlier
  pairs. The five-class library has 29 templates; strict margin-0.015 validation
  is 28/29 (96.55%), with one conservative unknown and zero wrong-class
  executions. All five trajectories are densified below five ticks/frame.
  Generated offline analysis artifacts for both sessions and updated the token
  teleop action list. Twenty-two focused tests pass. No camera, ROS command,
  enable token, or hardware motion was started by Codex.
- (2026-07-20, camera-2 direct real-time MediaPipe teleop) User requested
  frame-by-frame hand following instead of waiting for a complete token before
  playback. Documented the existing calibrated G20 direct retarget command in
  `../hardcode_position.md` with camera 2, 30 Hz, max five SDK ticks/frame,
  current 20, speed 35, and the existing thumb/collision guards. Added
  `--motion-stop-mode hold` to `camera_to_linkerhand`: together with
  `--motion-key-toggle`, `--open-on-start-seconds 0`, and
  `--no-open-on-exit`, the process starts camera preview without publishing a
  position; human SPACE starts live following and the next SPACE stops new
  commands while holding position. The original `open` stop mode remains the
  default for existing recording workflows. A focused test confirms stopped
  hold mode emits no position command; 35 related tests pass when the project
  virtualenv dependencies are exposed to system pytest. No camera, ROS command
  publisher, enable token, or hardware motion was started by Codex.
- (2026-07-20, camera-2 grouped-token teleop bring-up) Added
  `src/comms/import_action_group.py` to convert an audited grouped capture into
  the existing MediaPipe DTW action-library format. Import rejects incomplete
  groups and >10-tick recorded command/state mismatch, preserves exact human
  take boundaries, derives a conservative intra-class threshold, and densifies
  the G20 trajectory to max five SDK ticks between frames. Imported the only
  currently replay-safe capture (20260719/group_001) as primitive
  `1:thumb_across_palm` under the library now named
  `data/action_library/g20_right/core_five_actions_v1` (the primitive is now archived):
  5 templates, 136 trajectory frames, threshold 0.0840; leave-one-take-out
  validation is 5/5. `action_library_teleop` now overlays the MediaPipe skeleton
  on camera 2, clamps accepted current/speed CLI settings to conservative
  ranges, and refuses hardware when another command publisher such as the GUI
  is present. Dry-run and human-gated hardware commands were added to
  `../hardcode_position.md`. Twenty-two focused tests pass. No camera session,
  ROS command publisher, token, or hardware motion was started by Codex.
- (2026-07-20, action-group audit and gated G20 replay) Audited both grouped
  sessions end to end: 4 complete groups, 2 intentionally empty/incomplete
  trailing groups, 22 exact human takes, 966/966 fresh MediaPipe frames, and 21
  robot waypoints. Human/robot JPEG counts, take coverage, landmark shapes, and
  JSON metadata all agree; every JPEG decodes. Leave-one-take-out DTW nearest
  neighbour classification is 22/22, although the two 20260720 four-finger curl
  groups have the smallest cross-group margin. Recorded GUI command/state
  checks found only 20260719 group_001 below the default 10-tick replay limit;
  the other complete groups have q9 maxima of 41, 18, and 43 ticks. Added
  `src/comms/replay_action_group.py`: default numeric dry-run, metadata and
  command/state preflight, interpolated/densified max-5-tick playback, live
  stale-state/following-error stops, conflicting-GUI-publisher refusal, and
  conservative current/speed bounds. Hardware still requires a human-set
  `HW_ENABLE_TOKEN=1`, `--enable-motion`, and a human SPACE press; R is the
  explicit step-limited open return. Generated analysis artifacts for both
  sessions and documented the first replay command. Twenty focused comms tests
  pass. Codex did not set the token, publish ROS commands, or actuate hardware.
- (visual ACT training entrypoint, 2026-07-13) Added
  `scripts/train_g20_visual_act.py` for the new G20 recorder layout. It scans
  nested `data/**/samples.jsonl`, keeps only rows with real camera JPEGs plus
  20-D state/action, and therefore skips all 13 camera-free `grasp_cube`
  episodes automatically. Current scan: 30 camera episodes / 10,445 frames,
  split by whole episode into 27 train + 3 held out. Conversion writes a
  LeRobot v3 image dataset at 320x240; ACT observes scene RGB + 20-D SDK-range
  state and predicts 30x20 absolute command chunks. The wrapper auto-selects the
  installed LeRobot environment and CUDA, uses a 19M-parameter ResNet18 ACT,
  and creates standard LeRobot checkpoints. The local torchcodec and legacy
  torchvision VideoReader backends are incompatible, so frames intentionally
  remain LeRobot `image` features rather than being re-encoded as video. A real
  2-episode/1-step CUDA smoke test completed forward/backward and saved
  `model.safetensors` plus pre/postprocessor artifacts. Training only; no ROS or
  hardware command path was added or executed. `artifacts/` is gitignored.
- (comms open-thumb raised correction, 2026-07-13) User reported that an open
  human thumb made the hardware thumb stand out of the palm plane. The G1 axis
  audit identifies q5 as that lift/CMC-roll channel, and the collection command
  still applied `--hardware-thumb-abd-offset -28`, so even q5=0 mapped the G20
  open range from 255 to 227. Changed the active `collectdata.md` command offset
  to 0; visual q5 motion remains enabled through gain 0.72. Synthetic mapping
  check: q5=0 now maps 255 (previously 227), while q5=0.2 maps 229 (previously
  201). No hardware was actuated by Codex.
- (comms motion-toggle reset without detection, 2026-07-13) User reported that
  some SPACE-toggle episodes did not return to the expected starting pose, with
  the thumb left raised. Root cause: the inactive MotionGate reset ran only from
  `LinkerHandHardwareSink.set_joints`, but `drive()` does not call the sink when
  MediaPipe has no processed hand frame. The toggle callback now ramps directly
  to the G20 open pose on STOP, using the existing `--max-range-step`; dry-run
  mode still publishes no hardware command. Verified with `py_compile`. No
  hardware was actuated by Codex.
- (comms thumb contradiction reduction, 2026-06-30) User suspected the current
  combined command sometimes gives the thumb contradicting commands. Inspection
  found `--hardware-landmark-thumb` was adding positive deltas to both q5 thumb
  side swing and q10 thumb opposition/roll, while q10 was also still produced by
  the base camera retargeter and later limited by `--max-thumb-delta`. Removed
  the q10 landmark delta so q0 thumb base and q10 opposition now come only from
  the normal camera retarget mapping; distal thumb landmarks only enhance q5
  side swing and q15 fingertip response. Synthetic check: q0 raw/adjusted stayed
  0.373/0.373, q10 stayed 1.400/1.400, q5 increased 0.000 -> 0.220, and q15 was
  unchanged when already at its upper limit. No hardware was actuated by Codex.
- (comms non-thumb accidental-closure deadzone, 2026-06-30) User reported the
  four non-thumb fingers sometimes bend unexpectedly while tuning thumb control.
  Added hardware-output `--nonthumb-close-deadzone` in
  `src/comms/camera_to_linkerhand.py`: for non-thumb base/tip channels
  1-4/16-19, small closure deltas from the G20 open range are treated as open,
  while larger closures are reduced by the deadzone instead of blocked. This is
  applied after absolute/relative mapping and before the existing collision
  guard, and it does not touch thumb channels or non-thumb spread channels.
  README recommended command now uses `--nonthumb-close-deadzone 45`. Synthetic
  check: base/tip small closures 230/225 returned to 255/255, larger closures
  180/150 became 225/195, and thumb side swing index 5 stayed 35. No hardware
  was actuated by Codex.
- (comms thumb tip sensitivity, 2026-06-30) User reported the latest thumb
  behavior is much better but the thumb fingertip is not sensitive enough.
  Updated `--hardware-landmark-thumb` so q15 receives relative response from the
  actual distal thumb bend angle formed by landmarks 2-3-4, instead of only from
  cross-palm opposition. This keeps the no-preset behavior: q15 is incremented
  relative to the normal camera retarget output. Also updated the README command
  to set `--hardware-thumb-tip-gain 1.4`, because the global
  `--hardware-tip-gain` is overridden by the thumb-specific gain for q15.
  Mapping check: q15=0.4 maps range index 15 from 158 at gain 1.0 to 119 at gain
  1.4; q15=0.6 maps from 109 to 51. No hardware was actuated by Codex.
- (comms thumb preset removal, 2026-06-30) User suspected the thumb pose was
  preset and asked to cancel it. Removed the fixed absolute targets inside
  `--hardware-landmark-thumb`: q5/q10/q15 are no longer blended toward preset
  postures; distal thumb landmarks now add only relative response on top of the
  normal camera retarget output, with q0 still untouched. Also removed the
  recommended README thumb range offsets
  `--hardware-thumb-base/abd/roll/tip-offset`, leaving only
  `--hardware-thumb-abd-gain 1.35` and the separated thumb side-swing guard.
  Synthetic check: q0 raw/adjusted stayed 0.373/0.373, q5 changed only
  relatively 0.000 -> 0.364, q10 stayed at its raw upper-limit 1.400, and q15
  changed mildly 0.935 -> 0.973. No hardware was actuated by Codex.
- (comms thumb base returned to camera map, 2026-06-30) User reported the latest
  thumb side-swing version is much better, but the thumb no longer opens well.
  They asked to keep this version while making the thumb root control come from
  the normal camera retarget mapping, not the distal three thumb landmarks.
  Updated `--hardware-landmark-thumb` so distal landmarks no longer overwrite
  q0 `thumb_base`; q0 stays as produced by the base retargeter. The distal thumb
  chain still drives q5 side swing and contributes to q10/q15. Synthetic check:
  with the landmark adjuster enabled, q0 raw/adjusted stayed 0.373/0.373, while
  q5 still increased from 0.000 to 0.663. No hardware was actuated by Codex.
- (comms separate thumb side-swing guard, 2026-06-30) User pasted hardware logs
  showing only-thumb motion where raw index 5 thumb side swing varied strongly
  (`raw[5]` often 0/16/38/60), but the published command always had index 5 =
  155. Root cause: `--thumb-safe-mode limited --max-thumb-delta 100` was clamping
  both thumb side swing q5 and thumb roll q10 to open-100, and G20 open q5 is
  255, so q5 could never go below 155. Added separate
  `--max-thumb-abd-delta` for index 5 while leaving `--max-thumb-delta` for q10.
  README recommended command now uses `--max-thumb-delta 100
  --max-thumb-abd-delta 220`, allowing side swing down to 35 while keeping q10
  limited to 145. Verification with user's raw example: old guard q5/q10 =
  155/145; new guard q5/q10 = 35/145. No hardware was actuated by Codex.
- (comms distal-thumb-direction side swing, 2026-06-30) User asked whether the
  three distal thumb landmarks can indirectly control the sixth SDK range value,
  index 5 thumb side swing. Updated the opt-in `--hardware-landmark-thumb` map so
  q5 `thumb_abduction` is driven mostly by the visible distal thumb-chain
  direction from landmarks 2 -> 4, while q10 remains mostly cross-palm
  opposition from thumb position. This lets distal thumb rotation change side
  swing even when the camera root/CMC point barely moves. Synthetic check with
  thumb tip at the same cross-palm position but distal chain rotated: q5 increased
  from 0.594 to 0.665, mapping range index 5 from 59 to 41 with the current
  `--hardware-thumb-abd-gain 1.35` and offsets. No hardware was actuated by
  Codex.
- (comms thumb abduction sensitivity, 2026-06-30) User clarified the desired
  stronger thumb motion is the sixth SDK range value, index 5
  `thumb_abduction` / thumb side swing, e.g. output `[... 84, ...]`. The code
  already exposes this as `--hardware-thumb-abd-gain`, so the README recommended
  hardware command now sets `--hardware-thumb-abd-gain 1.35` while preserving the
  current thumb landmark map and offsets. Mapping check for q5=0.66 rad with the
  current offsets: gain 1.0 -> range index 5 = 85, gain 1.2 -> 61, gain 1.35 ->
  43, gain 1.5 -> 25. No hardware was actuated by Codex.
- (comms front-landmark thumb hardware map, 2026-06-30) User reported the real
  thumb now arches up, but its rotation/opposition angle is poor because the
  camera sees only the front thumb landmarks moving reliably; the rear/CMC point
  is hard to move. Added opt-in `--hardware-landmark-thumb` with
  `--landmark-thumb-gain` to `src/comms/camera_to_linkerhand.py`. When enabled,
  the hardware candidate adjuster estimates thumb opposition from landmarks
  2/3/4 projected onto the index-to-little MCP root line, then overwrites only
  thumb q0/q5/q10/q15 before the existing G20 sim-to-range map. This leaves the
  current four-finger mapping unchanged. README recommended hardware command now
  enables this front-landmark thumb map plus the current thumb arch offsets.
  Verification: compileall, CLI help, and synthetic thumb sweep showing q10
  increases from 0.856 to 1.291 as the thumb tip moves from index side to little
  side. No hardware was actuated by Codex.
- (comms non-thumb base flexion gain, 2026-06-30) User wanted to keep the current
  sim-spread sign correction but make the four non-thumb root joints bend more
  visibly. Increased default `--hardware-base-gain` in
  `src/comms/camera_to_linkerhand.py` from 1.0 to 1.8 and updated README commands
  from explicit 1.25 to 1.8. This affects BASE_IDX 1-4; thumb base is overridden
  separately by `--hardware-thumb-base-gain` and remains unchanged by the global
  base gain. Mapping check: non-thumb qbase 0.45 rad maps from range 153 with
  gain 1.25 to range 107 with gain 1.8, giving visibly stronger closure.
  Verification: compileall, parser default check, CLI help. No hardware was
  actuated by Codex.
- (comms sim-spread sign correction, 2026-06-30) User asked to tune hardware
  finger orientation according to the simulation command
  `python -m src.viz.app --source webcam --camera-index 0 --side right --show-camera --thumb-gain 1`.
  Kept hardware spread driven by the sim/solver q6-q9 path and added
  `--hardware-spread-signs` (default `-1,-1,-1,-1`) to reverse the G20 side-swing
  direction relative to sim. Reason: spread-test-only showed the previous
  positive-sign extreme [255,255,0,0] physically behaves like finger closing/
  gathering, while sim visual spread-apart q pattern previously mapped toward
  that side. Now the same sim spread q6-q9=[0.17,0.17,-0.17,-0.17] maps from old
  range [255,248,5,0] to new [93,48,205,142]. README commands include the signs.
  Verification: compileall, CLI help, mapping comparison. No hardware was
  actuated by Codex.
- (comms calibrated landmark spread, 2026-06-30) User reported spread-test-only
  clearly moves G20 fingers together/apart, but camera teleop does not. Previous
  landmark spread override was worse because it had no open-hand baseline and
  treated naturally straight fingers as spread, causing two real fingers to stay
  half bent. Added calibrated landmark spread: when `--hardware-landmark-spread`
  is enabled, the candidate adjuster collects
  `--landmark-spread-calibration-frames` open-hand frames (default 30), stores
  centered MCP->PIP lateral angles as baseline, commands q6-q9 = 0 during
  calibration, then maps only relative angle changes to q6-q9. Synthetic check:
  baseline open produces q6-q9 [0,0,0,0]; after baseline, a spread pose produces
  [0.17,0.17,-0.17,-0.17]. README now includes this as an optional calibrated
  command, not the default stable command. Verification: compileall, CLI help,
  calibrated synthetic test. No hardware was actuated by Codex.
- (comms landmark-spread rollback + spread test, 2026-06-30) User reported the
  landmark spread override made hardware worse: camera fingers were straight but
  two real fingers stayed half bent. Root cause likely G20 side-swing channels
  mechanically couple into root posture when driven to extreme 6-9 values.
  Changed `--hardware-landmark-spread` default to OFF, restored gentler default
  `--hardware-spread-gain 1.0` and `--roll-range-ticks 100`, and removed
  landmark-spread flags from README recommended teleop commands. Added
  `--spread-test-only --spread-test-seconds` to publish open, [255,255,0,0],
  [0,0,255,255], open on channels 6-9 without camera, so the user can verify
  whether G20 side-swing physically produces visible finger close/open before we
  trust camera-driven spread. Verification: compileall, CLI help, default parser
  check. No hardware was actuated by Codex.
- (comms landmark-derived spread override, 2026-06-30) User showed real G20 still
  does not visibly imitate finger close/open orientation even with larger
  spread_gain/ticks. Added hardware-only landmark spread override in
  `src/comms/camera_to_linkerhand.py`: `--hardware-landmark-spread` (default on),
  `--no-hardware-landmark-spread`, `--landmark-spread-gain` (default 2.5), and
  `--landmark-spread-limit` (default 0.17 rad). The candidate adjuster now reads
  current hand_base landmarks, computes each non-thumb MCP->PIP lateral angle,
  subtracts the four-finger mean to remove whole-hand tilt, and overwrites q6-q9
  with amplified relative spread before safety/hardware mapping. Synthetic
  spread test maps q6-q9 to [0.17,0.17,-0.17,-0.17], which maps through G20 to
  spread range [255,255,0,0] from open [193,148,105,42]. README hardware commands
  now include the landmark spread flags. Verification: compileall, CLI help,
  landmark spread test, G20 range mapping check. No hardware was actuated.
- (comms spread mapping sensitivity, 2026-06-30) User reported two-finger
  close/open imitation remains weak and suspected the finger orientation joints
  are not mapped well. Increased G20 hardware spread mapping sensitivity in
  `src/comms/camera_to_linkerhand.py`: default `--hardware-spread-gain` 1.0 ->
  2.0 and default `--roll-range-ticks` 80 -> 140. Updated README hardware
  commands to remove `--thumb-gain 1.5` and use spread gain 2.0 / ticks 140.
  Synthetic small spread rad input [-0.05,-0.025,0.025,0.05] now maps spread
  deltas from old [-29,-15,15,29] ticks to new [-82,-41,41,82] ticks.
  Verification: compileall, CLI help, mapping check. No hardware was actuated.
- (comms finger-orientation responsiveness, 2026-06-30) User reported real-hand
  finger orientation/spread was not sensitive enough, likely because hardware
  safety values were too restrictive. Relaxed default G20 range-space spread
  guard in `src/comms/camera_to_linkerhand.py`: `--max-spread-delta` 18 -> 70,
  `--spread-close-threshold` 0.20 -> 0.75, `--spread-recenter-gain` 0.85 -> 0.20,
  and `--min-spread-gap` 8 -> 2. Startup log now prints these spread guard
  values. Extreme synthetic spread check that old defaults clamped
  `[255,215,55,0]` to `[211,166,87,24]` now preserves `[255,215,55,0]`.
  Verification: compileall, CLI help, synthetic guard check. No hardware was
  actuated.
- (viz thumb-cross sim tuning, 2026-06-30) User no longer wanted the temporary
  sim tuning GUI, so removed `--tune-sim` and all PyBullet slider code from
  `src/viz/app.py`. Added landmark-driven sim thumb crossing instead:
  `--thumb-cross-gain` defaults to 1.0 and projects the human thumb tip onto the
  index-to-little MCP root line, pushing only thumb joints 0/5/10/15 farther
  across the palm near the root line. Synthetic checks map index/middle/ring/
  little roots to increasing opposition q10 = 0.12/0.489/0.905/1.32, so little
  can pass the middle-finger region. `--thumb-cross-gain 0` disables it for raw
  solver comparison. Verification: compileall, CLI help shows no `--tune-sim`,
  synthetic thumb-root check, and headless smoke pass. No hardware was actuated.
- (comms thumb-only hardware tuning, 2026-06-30) User reported the real thumb map
  is still inaccurate. Added thumb-only G20 hardware tuning in
  `src/comms/camera_to_linkerhand.py`: `--hardware-thumb-base-gain`,
  `--hardware-thumb-abd-gain`, `--hardware-thumb-roll-gain`,
  `--hardware-thumb-tip-gain`, plus per-channel range offsets
  `--hardware-thumb-*-offset`. These affect only command channels 0, 5, 10, and
  15 after the sim-radian to G20 range map, leaving four-finger mapping and the
  PyBullet simulation path unchanged. Recommended pairing is
  `--thumb-safe-mode limited --max-thumb-delta ...` so the thumb can oppose
  without fully freeing collision-prone rotation. Verification: compileall,
  thumb-only assertion showing only channels {0,5,10,15} change, CLI help, and
  webcam dry-run startup log with thumb gains. No hardware was actuated by Codex.
- (comms hardware anti-fighting guard, 2026-06-30) User reported thumb random
  motion and four-finger angle jumps causing real-finger fighting. Added a
  hardware-output range-space guard in `src/comms/camera_to_linkerhand.py`
  without modifying `src/safety`: default guard locks thumb collision rotation
  channels 5 and 10 to the G20 open pose (`--thumb-safe-mode open`), clamps
  non-thumb spread channels 6-9 around open (`--max-spread-delta`), recenters
  spread as finger flexion increases (`--spread-close-threshold`,
  `--spread-recenter-gain`), preserves ordered spread gaps (`--min-spread-gap`),
  and then applies the existing per-frame step limit. Added
  `--no-collision-guard`, `--thumb-safe-mode {open,limited,free}`,
  `--max-thumb-delta`, `--max-spread-delta`, `--spread-close-threshold`,
  `--spread-recenter-gain`, and `--min-spread-gap`. Verification: compileall,
  guard assertions, and webcam dry-run showing extreme raw thumb/spread values
  clamped before publish. No hardware was actuated by Codex.
- (comms sim-to-G20 hardware map, 2026-06-30) User reported that the PyBullet
  webcam simulation pose is satisfactory, but the real hand mapping is wrong and
  finger posture amplitude is too small. Root cause: the hardware bridge was
  still using L20 SDK arc/range constants for the G20 palm-touch SDK path.
  Added a `g20-sim` radian->range map to `src/comms/camera_to_linkerhand.py`,
  mirroring the SDK HORA `sim_rad_to_active_range` convention while preserving
  the G20 palm-touch open pose. `--hardware-map auto` now selects `g20-sim` when
  `--sdk-hand-joint g20`; old L20 mapping remains available as `--hardware-map
  l20-sdk`. Added hardware-only gains `--hardware-base-gain`,
  `--hardware-spread-gain`, and `--hardware-tip-gain`, plus
  `--roll-range-ticks`, so real-hand amplitude can be tuned without changing the
  simulation path. Verification: compileall, open/closed G20 mapping assertions,
  CLI help, and webcam dry-run with `--absolute --hardware-map g20-sim`. No
  hardware was actuated by Codex.
- (comms G20 palm-touch open pose, 2026-06-30) User identified the SDK entrypoint
  that opens the physical hand:
  `ros2 run linker_hand_ros2_sdk linker_hand_g20_palm_touch --hand_type right --can can0 --is_touch true`.
  That node's `DEFAULT_POSITION` is `[255,255,255,255,255,255,193,148,105,42,245,255,255,255,255,255,255,255,255,255]`,
  not the earlier L20 open range. Added `--sdk-hand-joint {g20,l20}` to the
  camera bridge and defaulted it to `g20` so `--open-only`, startup open,
  relative calibration, and reserved/default channels use the G20 palm-touch
  open pose. Verification: compileall, CLI help, and a small assertion covering
  G20 open pose selection. No hardware was actuated by Codex.
- (comms SDK-start open pose, 2026-06-30) Added token-gated startup open support
  to `src/comms/camera_to_linkerhand.py`: `--open-only --open-seconds N` publishes
  the L20 `OPEN_RANGE` pose after the SDK is up, without opening the camera, and
  normal hardware teleop now sends open pose for `--open-on-start-seconds` before
  camera calibration. Verification: compileall, CLI help, no-token
  `--open-only --enable-motion` refusal, and dry-run `--open-only` exits without
  constructing a camera source. No hardware was actuated by Codex.
- (comms hardware closure fix, 2026-06-30) User reported that real L20 closed all
  fingers immediately while the camera was tracking normally. Root cause:
  hardware bridge was sending absolute retargeted SDK ranges; the user's open
  camera pose retargeted to non-open L20 angles on the monocular webcam path.
  Changed `src/comms/camera_to_linkerhand.py` default to calibrated relative
  mode: first N detected open-hand frames publish `OPEN_RANGE` and build a raw
  baseline, then command `OPEN_RANGE + (raw - baseline)` with clamp and step
  limits. Added `--absolute` to opt back into old behavior, plus
  `--calibration-frames`, `--relative-scale`, `--max-relative-delta`, and
  open-on-exit release. Also guarded `publish_pose` with `rclpy.ok()` for clean
  external termination. Verification: compileall, CLI help, relative calibration
  assertion, and webcam dry-run showing `calibrating ... OPEN_RANGE` before
  relative output. Human should retry with open hand held still through
  calibration and very low `--relative-scale`/`--max-range-step`.
- (comms webcam bring-up follow-up, 2026-06-30) User hit Ctrl-C while
  `MediaPipeHandSource` was opening `cv2.VideoCapture(0)`, which exposed a noisy
  double ROS shutdown traceback. Updated webcam capture to use OpenCV V4L2 on
  Linux camera indices, and made `src/comms/camera_to_linkerhand.py` handle
  `KeyboardInterrupt` with an idempotent `rclpy.ok()` shutdown. Verified
  `/dev/video0` opens and reads 640x480, `/dev/video1` is not a capture stream,
  compileall passes, CLI help works, and an 8 s dry-run produced live
  `/cb_right_hand_control_cmd` range poses. No hardware was actuated.
- (comms camera->hardware bridge, 2026-06-30) Added a G3 bring-up bridge in
  `src/comms/camera_to_linkerhand.py`: webcam/RealSense -> existing perception
  pipeline -> retarget -> safety.filter -> LinkerHand ROS2 SDK `JointState`
  range command on `/cb_<side>_hand_control_cmd`. The bridge is dry-run by
  default and refuses real motion unless BOTH `--enable-motion` and a human-set
  `HW_ENABLE_TOKEN` are present. For hardware mode it waits for command +
  `/cb_hand_setting_cmd` subscribers, requires state by default, publishes
  conservative current/speed settings, maps 20 radian commands to SDK L20
  0..255 range units, preserves reserved indices 11-14 at open range, and
  per-frame limits range-unit steps. Added `src/comms/requirements.txt` with
  PyYAML so `.venv` can import ROS2 `rclpy` after sourcing `/opt/ros/jazzy`.
  Verification: `python -m compileall -q src/comms src/viz src/perception`,
  `python -m src.comms.camera_to_linkerhand --help`, L20 mapping/step assertions,
  and the no-token `--enable-motion` refusal. No hardware was actuated; G3
  remains human-gated.
- (solver-agent closed-form-restore, 2026-06-16) WHOLE-SOLVER CLOSED-FORM AUDIT +
  non-thumb distal converted to FIXED-COST ANALYTIC. Wrote ONLY
  src/finger_retarget/solver.py (+ docs/adr/0010, STATE.md). G0 26/26 GREEN; no
  should-not-change number moved. ADR-0010 records the full derivation.
  AUDIT MATRIX (before -> after):
    | subproblem        | DoF | before                  | after                          |
    | non-thumb proximal | 2  | closed PK-2 + nearest   | UNCHANGED (exact; 2.1e-8)      |
    | non-thumb distal   | 1  | grid+Brent (fixed search)| FIXED-COST ANALYTIC (PK-1 seed+Newton) |
    | thumb proximal     | 3  | closed-form per cd      | UNCHANGED (closed-form)        |
    | thumb distal(+base)| 1+1| cd fixed point + grid   | KEPT (inner tip-align analytic)|
  PROXIMAL CONFIRMED GENUINELY CLOSED-FORM (no silent fallback): prox worst-case
  = 2.1e-8 rad (machine-eps) over 3000 random reachable configs/side, every
  finger -> exact subproblem-2 on reachable; the nearest path is a bounded 4-step
  closed-form coordinate descent, only when the 2R cone misses. Thumb proximal is
  closed-form per distal latitude cd (two-plane ktip + PK-2/PK-1).
  WHY NO CLOSED-FORM DISTAL (headline): the fingertip v(θ)=R(k,θ)·mvec0 +
  R(k,(1+ratio)θ)·dvec0 sums TWO bones at rates 1 and 1+ratio about the common
  flexion axis (DIP/IP ∥ PIP, verified dot=1.0). The ticket's "parallel axes ->
  circle -> PK-1" holds only for a SINGLE downstream vector; the fingertip is a
  SUM at different rates, so unit(v)·k (latitude) is NOT constant (verified spread
  up to ~8e-4 in cos, offset up to ~0.016) -> a transcendental EPITROCHOID, not a
  circle -> no exact PK-1. t=tan(θ/2) needs INTEGER rate; ratio=0.8917/1.1619 is
  non-integer (nearest 33/37, 43/37, err ~2e-4) -> forcing a polynomial gives
  degree ~O(100) AND only approximate (would change the mimic) -> INTRACTABLE,
  not degree≤4. So the correct method is fixed-cost analytic, not closed-form.
  CONVERSION (non-thumb distal): subproblem-1 ANALYTIC SEED (rotate v(0) about k
  toward u_dist; the ticket's circle, exact to 1st order) + hard-capped Newton (4
  steps, exact v/v'/v'') correcting the wobble to machine precision + endpoint
  guard for the under-actuated optimum. Deterministic, bounded, NO grid -> the
  ticket's sanctioned "Newton from an analytic seed", NOT a search. The thumb's
  inner tip-align calls the same solve; its redundant-base cd fixed point + grid
  are unchanged. Removed the dead _brent_min/_TIP_GRID.
  ORACLE-MATCH (3000 reachable configs/side, worst segment residual, BEFORE==AFTER):
    prox 2.1e-8 (all) | non-thumb dist 2.1e-8 | thumb dist 8.3e-7 (R)/1.08e-6 (L).
  New analytic non-thumb distal == prior grid+Brent to ~4e-11 over 20k random
  dirs/finger (incl. unreachable), NEVER worse. TIMING improved (corr. is the
  gate, not speed): representative p50 220->200us, worst-case p99 884->800us.
  SHOULD-NOT-CHANGE — ALL HOLD: oracle unchanged (eval/ untouched); G0 26/26;
  proximal numbers untouched; reachable round-trip ~1e-7 (distal 2.1e-8/thumb
  1e-6); reserved idx 11-14 = 0; ADR-0003 target + mimic ratios + ALL tolerances
  unchanged (no test/threshold loosened); hardware untouched (no src/comms, no
  HW_ENABLE_TOKEN). Only src/finger_retarget/solver.py changed.
  ADR-0007 timing reconciliation still valid (the thumb redundant-base tail, not
  the distal align, is the cost driver). NO subproblem could reach exact closed
  form for the distal; the correct fixed-cost analytic method is kept + the reason
  (non-integer mimic ratio -> transcendental) is recorded (ADR-0010).
- (process-mov-videos FOLLOW-UP, 2026-06-16) CHIRALITY BUG FOUND + FIXED in the
  runner scripts (NOT a pipeline/retarget/filter fault). User watching the live
  side-by-side viewer saw the VIDEO show a left hand but the SIM render a right
  hand. ROOT CAUSE: `L20VizModel` loads a left- OR right-hand URDF at construction
  and never swaps it (setting `.side` later is informational; render.py does not
  reload) — and both runner scripts hardcoded `L20VizModel("right")`, then only set
  `model.side = pf.side`. So left-hand clips (all 10 resolve to LEFT) were rendered
  on the RIGHT-hand URDF = mirrored chirality. The detection/retarget/filter were
  always correct (retarget ran side=left); ONLY the displayed model hand was wrong.
  This CORRECTS the prior entry's claim that the captures "render a natural, non-
  mirrored left hand" — they were right-URDF until this fix.
  FIX (runner-level only — render.py/loop core/retarget/filter/kinematics UNTOUCHED,
  per ticket guardrails): both scripts now resolve each clip's physical side up
  front (first detected frame -> to_l20_side, same mapping validation uses; or
  --side), BUILD the model for that side, and force the pipeline to the same side so
  cmd + render agree. scripts/loop_video_with_view.py also now takes MULTIPLE clips
  (nargs="+"), plays each once and AUTO-ADVANCES, cycling the whole list until `q`
  (was: one clip, looped forever) — and rebuilds the model only on a side change.
  Regenerated all 10 tests/viz/out/sim_HAND-*.{gif,_last.png} with side=left;
  spot-checked PNGs now show the thumb on the LEFT (left-hand chirality), mirror of
  the earlier wrong renders. Numbers unchanged (detection 100% all 10, motion,
  filter-modified counts identical). The latent viz limitation (model can't hot-swap
  sides) is a src/viz concern, left to viz-agent; runners work around it.
  NOTE on "which hand": default (image_mirrored=False) maps these clips to LEFT and
  now both windows agree on left. If the footage was selfie/front-camera or the real
  hand was the RIGHT, pass --image-mirrored (or --side right) — both windows flip
  together. Wrote ONLY scripts/loop_video_with_view.py, scripts/stream_hand_videos.py,
  STATE.md (+ regenerated gitignored tests/viz/out artifacts).
- (process-mov-videos task, 2026-06-16) USER'S OWN PHONE MOVs NORMALIZED, VALIDATED
  + STREAMED through the `--source video` loop. Sim-only, hardware-free, NO gate, NO
  module logic touched (additive: one new normalize utility + reuse of the existing
  validate/stream runners + assets/work dir + .gitignore + STATE.md). Built the repo
  `.venv` fresh (absent in this checkout): mediapipe 0.10.21, opencv 4.11.0.86,
  pybullet, pillow, yourdfpy 0.0.60, trimesh 4.12.2, scipy 1.17.1, numpy pinned
  1.26.4 (mediapipe wants numpy<2). All import + the loop runs clean.
  INPUT: 10 `.MOV` files in `assets/video/` (HAND-C, -FINGERS-TGT, -FINGERSCROSSED,
  -FIST, -OK, -OK-BACK, -PEACE, -PINCH-CLICK, -SIX, -SPIDERMAN). ffprobe showed ALL
  10 are **HEVC/H.265, 1920x1080@30, rotation=-90** — i.e. BOTH ticket gotchas
  (HEVC codec + ignored rotation flag) present on every clip. Gitignored the raw
  MOVs + the normalized work dir (`assets/video/*.MOV`, `assets/video/normalized/`).
  STEP 2 — NORMALIZE (new `scripts/normalize_mov_videos.py`, wraps the ticket's
  ffmpeg cmd: `-vf scale=-2:720 -r 30 -c:v libx264 -pix_fmt yuv420p -an`; ffmpeg
  auto-applies rotation). 10/10 OK -> upright H.264 **406x720** (portrait, rotation
  BAKED IN, rot=None on output), in `assets/video/normalized/` (gitignored; report
  `normalize_report.json`).
  STEP 3 — MediaPipe DETECTION RATE (scripts/validate_hand_videos.py, the bad-clip-
  vs-broken-pipeline gate, >=70% to keep; report tests/viz/out/mov_detection_report
  .json). **ALL 10 PASS at 100%** (no held frames), high scores:
    * HAND-C            : 100% (109/109)  cam=Right->l20 left  score 0.966
    * HAND-FINGERS-TGT  : 100% (173/173)  cam=Right->l20 left  score 0.958
    * HAND-FINGERSCROSSED:100% (577/577)  cam=Right->l20 left  score 0.960
    * HAND-FIST         : 100% (110/110)  cam=Right->l20 left  score 0.968
    * HAND-OK           : 100% (152/152)  cam=Right->l20 left  score 0.970
    * HAND-OK-BACK      : 100% (135/135)  cam=Right->l20 left  score 0.927
    * HAND-PEACE        : 100% (275/275)  cam=Right->l20 left  score 0.986
    * HAND-PINCH-CLICK  : 100% (174/174)  cam=Right->l20 left  score 0.976
    * HAND-SIX          : 100% (135/135)  cam=Right->l20 left  score 0.965
    * HAND-SPIDERMAN    : 100% (368/368)  cam=Right->l20 left  score 0.960
  (Pre-normalization these would have detected ~0 — sideways HEVC. The 100% rate IS
  the evidence the rotation+codec fix worked.)
  STEP 4 — STREAMED all 10 through the EXISTING viz loop (VideoHandSource->
  HandPipeline->retarget->safety.filter->L20VizModel) into the live PyBullet GUI
  (DISPLAY=:0, NVIDIA GL 3.3). Ran one clip PER PROCESS — a single process opening/
  closing 10 GUI windows hit the same transient X11/GL flake STATE.md noted before;
  one-window-per-clip renders all 10 cleanly. Captures (GIF every-3rd-frame + last-
  frame PNG) in tests/viz/out/ (gitignored): sim_HAND-*.{gif,_last.png}, plus
  mov_stream_summary.txt. side_majority=**left** every clip (matches validation
  l20=left). Filter was LOAD-BEARING on this real input (frames modified /
  total): FINGERSCROSSED 45/577, PEACE 43/275, FINGERS-TGT 25/173, SPIDERMAN
  20/368, PINCH-CLICK 10/174, SIX 8/135, FIST 6/110, OK 5/152, C 1/109,
  OK-BACK 0/135. Verified the sim is genuinely MOVING (not a frozen pose): every
  inter-frame GIF delta > 0.1 px on the spot-checks (FIST 36/36, PEACE 91/91,
  SPIDERMAN 122/122, OK 50/50; mean |dpix| ~1.3-2.0, max ~7-10).
  STEP 5 — HANDEDNESS / MIRROR: ROTATION was the only transform needed and it was
  handled UPSTREAM by ffmpeg (baked into the normalized MP4s); no in-reader fix.
  No MIRROR flip was applied — ran the non-selfie default (image_mirrored=False),
  under which MediaPipe's camera-label majority (Right on all 10) maps via
  to_l20_side to physical **left**, and the sim renders a natural, non-mirrored
  left hand. CAVEAT for the user's eyeball: all 10 came out cam=Right->l20 left
  consistently; IF these were filmed selfie/front-camera (front cam mirrors) OR you
  actually gestured with your RIGHT hand, the true side is right — re-run with
  `--image-mirrored` (validate) / `--source video --image-mirrored` (app) to flip.
  The mechanical pipeline is side-agnostic either way (clamped both sides); only the
  rendered chirality changes. Definitive "sim mirrors my hand correctly" call is the
  user's GUI review.
  OBSERVED (expected monocular RGB limits, NOT failures per ticket): thumb / depth-
  axis (z) motion is soft — weak monocular z is the known RGB limit (metric depth is
  the RealSense path, not this). OK-BACK (back-of-hand view) needed 0 filter
  projections; the high-occlusion / fast clips (crossed fingers, peace) drove the
  most filter activity, consistent with bigger candidate jumps under self-occlusion.
  DONE per ticket: all 10 usable clips stream into the GUI with reported per-clip
  detection rates + saved capture artifacts. Wrote ONLY scripts/normalize_mov_
  videos.py, .gitignore, STATE.md (+ gitignored work dir & tests/viz/out artifacts).
- (video-fetch task, 2026-06-09) REAL HAND CLIPS FETCHED + STREAMED through the
  `--source video` loop. NO module logic touched (additive runner + assets only):
  wrote scripts/validate_hand_videos.py, scripts/stream_hand_videos.py,
  assets/video/{4 clips + NOTICE.md}, tests/viz/out/* (gitignored), .gitignore,
  STATE.md. Both runners import VideoHandSource / HandPipeline / core.drive /
  L20VizModel READ-ONLY (the stream runner only hangs a capture hook on drive()'s
  existing on_record callback). Installed mediapipe 0.10.21 + opencv 4.11.0.86
  into .venv per src/perception/requirements.txt (mediapipe pins numpy<2 → numpy
  downgraded 2.4.6→1.26.4; pybullet/yourdfpy/perception all still import + the
  stream runs clean).
  SOURCE + LICENSE (all Wikimedia Commons, free-to-use, direct upload URLs — no
  scraped/YouTube; full provenance in assets/video/NOTICE.md): woman_counting CC0,
  finger_counting_dutch CC BY-SA 4.0, hand_wave_example CC BY-SA 4.0,
  hand_gesture_67 CC0. Only the tiny CC0 hand_gesture_67.webm (0.57 MB) is
  COMMITTED as a reproducibility fixture; the 3 larger clips are gitignored
  (re-fetch via NOTICE URLs).
  STEP 2 — MediaPipe DETECTION RATE (the bad-clip-vs-broken-pipeline gate; ≥70%
  to keep; report tests/viz/out/detection_report.json). ALL FOUR PASS:
    * woman_counting_on_fingers : 94.2% (355/377) cam=Left→l20 right  score 0.977
    * finger_counting_dutch     : 100%  (397/397) cam=Left→l20 right  score 0.821
    * hand_wave_example         : 100%  (156/156) cam=Right→l20 left  score 0.986
    * hand_gesture_67           : 95.7% (89/93)   cam=Right→l20 left  score 0.921
  (l20_side via to_l20_side with image_mirrored=False — the correct default for
  third-person, non-selfie footage; it swaps MediaPipe's camera-view label.)
  STEP 3 — STREAMED all four survivors through the EXISTING viz loop
  (VideoHandSource→HandPipeline→retarget→safety.filter→L20VizModel) into PyBullet.
  Captures (GIF + last-frame PNG per clip) in tests/viz/out/ (gitignored):
  sim_{woman_counting_on_fingers,finger_counting_dutch,hand_wave_example,
  hand_gesture_67}.{gif,_last.png}. 3 clips rendered into the live p.GUI window
  (DISPLAY=:1, GL 3.3); the 1920×1080 dutch clip hit a transient X11 IO-error 11
  (X resource flake, NOT a pipeline fault) so it was captured via the SAME
  CPU tiny-renderer headlessly — identical artifact. Verified the sim is genuinely
  MOVING, not a frozen pose: woman_counting GIF 139/181 inter-frame deltas show
  motion (mean |Δpix| 1.25, max 13.65). Filter was load-bearing on real input:
  modified frames dutch 292/397, hand_gesture_67 23/93, hand_wave 2/156.
  STEP 4 — HANDEDNESS: no clip needed a mirror flip — the default non-selfie
  assumption (image_mirrored=False) held; the streamed side_majority matched the
  validation l20_side every clip (dutch→right, gesture/wave→left, woman→right) and
  all four sim poses render as natural, non-mirrored hands. The definitive
  "sim mirrors the real hand correctly" eyeball is still the user's GUI review.
  OBSERVED (expected monocular limits, NOT failures per ticket): finger_counting_
  dutch flips its MediaPipe label across frames (104 Right / 293 Left over 397 —
  Dutch counting briefly shows the other hand / both), which correlates with its
  high filter-modified count (fast counting + label churn → bigger candidate jumps
  → more safety projections); thumb/depth-axis motion is soft (weak monocular z,
  the known RGB limit — depth accuracy is the RealSense path, not this).
  DONE per ticket: ≥1 clip streams cleanly into the GUI with reported per-clip
  detection rates + saved capture artifacts (here: 4/4). Sim-only, hardware-free,
  no gate touched.
- (viz-agent video-amendment, 2026-06-09) VIDEO-FILE SOURCE ADDED — a third viz
  input alongside live RealSense + camera-free replay. ADDITIVE ONLY (no gate, no
  hardware, no comms): the src/viz loop core and the RealSense path are UNCHANGED;
  no existing perception file was modified. Monocular RGB (no depth) -> validates
  the end-to-end PLUMBING (real hand motion -> MediaPipe -> palm-plane frame ->
  retarget() -> safety.filter() -> sim joints), NOT depth/retarget accuracy (still
  the RealSense path); depth_confidence is LOW (0.3), so the pipeline DOES raise
  low_depth_confidence (opposite of the metric RealSense backend).
  NEW src/perception/video_source.py: VideoHandSource SUBCLASSES MediaPipeHandSource
  and reuses its RGB detection read() VERBATIM — the only differences are frame
  acquisition (cv2.VideoCapture(path), via the parent's video= arg) and timing
  (timestamps honour the clip's NATIVE FPS via resolve_fps(), optional fps override;
  playback_rate scales wall-clock display pacing only, not the emitted t). No-
  detection/low-conf -> hold-last-good in the pipeline (never NaN); EOF ->
  StopIteration -> clean stop; missing file -> FileNotFoundError. resolve_fps()
  rejects 0/NaN/inf container FPS -> _DEFAULT_FPS=30. mediapipe/opencv stay lazy in
  the parent ctor (src/perception/requirements.txt); no new CI deps.
  WIRED into src/viz/app.run_video + the CLI: --source {replay,realsense,video}
  (default = realsense; --camera-free still => replay, back-compat), --video-path,
  --playback-rate, --image-mirrored (threaded to live + video for selfie clips).
  Everything downstream (retarget -> filter -> resetJointState + mimic enforcement
  from src/kinematics -> GUI) is unchanged.
  TESTS tests/viz/test_video_source.py — 11 GREEN (headless, camera-free): fakes
  drive the REAL inherited MediaPipe read() (proving reuse, not a reimpl): valid
  hand_landmarks via the EXISTING pipeline (schema-valid, side swapped Left->right,
  depth LOW, low_depth_confidence WARNED), no-detection -> ok=False/landmarks None,
  EOF -> StopIteration, native-FPS timestamps, frame_period vs playback_rate,
  missing-file raise, + pure resolve_fps cases. Stage-1 EQUIVALENCE TEST UNCHANGED +
  still GREEN (12/12) — the new source did not perturb the loop. Full tests/viz now
  36/36; tests/g1_kinematic/test_perception.py still 39/39 (perception unmodified).
  MANUAL: the live "sim follows the video" eyeball is the user's. Handedness note
  (per ticket): selfie clips mirror — use --image-mirrored or the opposite --side if
  the sim hand mirrors wrong, before suspecting the retargeter. Did NOT commit a
  sample clip (kept repo light / CI camera-free); --video-path takes a user file.
  Wrote ONLY src/perception/video_source.py, src/viz/app.py, src/viz/__init__.py,
  tests/viz/test_video_source.py, STATE.md.
- (viz-agent, 2026-06-09) STAGE 1 BUILT — live RealSense -> sim L20 mirror
  (sim-only, NO gate, NO hardware, NO comms, NO HW_ENABLE_TOKEN). Additive only:
  imported perception/retarget/safety/kinematics/sim READ-ONLY; the ONE new
  perception file is the authorised backend. AWAITING the manual visual win
  condition (user will watch it); Stage 2 NOT started (gated behind Stage 1 review).
  NEW BACKEND src/perception/realsense_source.py (RealSenseHandSource, a new
  HandSource — existing perception code UNMODIFIED): aligned RGB-D via
  pyrealsense2 (rs.align to colour), MediaPipe Hands for the 21 2D landmarks, each
  landmark's z taken from the ALIGNED DEPTH MAP (not MediaPipe world-z) and
  deprojected with the colour intrinsics -> metric 3D in the camera frame. Depth
  holes: small-neighbourhood median -> per-landmark hold-last -> resolved-frame
  median -> finite fallback (NEVER NaN); >50% holes -> ok=False so the existing
  pipeline holds last good (reused robustness path). depth_confidence = HIGH (1.0)
  vs the RGB backend's 0.3, so the pipeline does NOT raise low_depth_confidence.
  Deproject/sample/assembly are pure module-level helpers -> fully camera-free
  testable. pyrealsense2/mediapipe/opencv lazy-imported in the ctor only.
  NEW MODULE src/viz/: core.teleop_command + drive = the ONE per-frame seam
  (landmarks -> retarget() -> safety.filter() -> 20-vec command, prev_safe threaded
  + reset on side change, dt=1/30); render.L20VizModel = PyBullet GUI(/DIRECT for
  tests) kinematic mirror (resetJointState on the 16 active DoF + the 5 mimics
  enforced as mult*driver+off using L20FK.mimics ratios from src/kinematics);
  app.run_camera_free (replays committed synthetic_openclose through the IDENTICAL
  loop, camera-free) + app.run_live (RealSenseHandSource -> HandPipeline -> loop,
  smoothing on, INPUT-side only) + a CLI (--camera-free/--side/--no-filter/--loop/
  --headless/--show-camera). Runtime deps in src/viz/requirements.txt (NOT in CI).
  TESTS tests/viz/ — 25/25 GREEN (headless): deproject math (principal-point,
  known offset, project/deproject round-trip), depth-hole handling (neighbourhood
  median, total-hole unresolved finite, OOB clamp, hold-last, frame-median borrow,
  fallback — never NaN), contract conformance (assembly -> existing HandPipeline ->
  schema-valid hand_landmarks, depth_confidence HIGH, no low-depth warning,
  camera-view label swap), excessive-holes -> ok=False; PIPELINE EQUIVALENCE both
  sides: viz loop filter-OFF == retarget() per frame (the G1 track_frame candidate)
  and filter-ON == retarget()->safety.filter() with prev_safe threaded (the command
  the G2 closed-loop computes) — BIT-EXACT, proving the loop is orchestration over
  the real components, not a drifting reimplementation; reserved 11-14 == 0;
  DIRECT-render applies the commanded active joints (resetJointState wiring);
  side-change prev_safe reset. Headless camera-free smoke: 90 frames, filter
  modified 61/90 (in line with G2 closed-loop's ~58/66).
  DEPTH-CONFIDENCE BEHAVIOR: by construction the RGB-D backend reports HIGH (1.0)
  metric confidence so weak-monocular-z warnings are suppressed; the LIVE-observed
  value + the thumb-tracking tell (the depth-wiring win condition) are the user's
  manual review and not yet recorded — to be filled in after the visual check.
  NOTE (do NOT act on here): the thumb-palm collision blind spot (Stage-2 risk per
  the ticket) is a separate safety ticket; the viz layer does not touch src/safety.
  Wrote ONLY src/perception/realsense_source.py, src/viz/*, tests/viz/*, STATE.md.
- (sim-agent G2-v2, 2026-06-09) STEP 0 + G2 DYNAMIC CLOSED-LOOP SIM BUILT.
  STEP 0 (G1 honest green): split the fail-closed test_real_residual into
  test_proximal_residual (HARD GATE: pooled prox p95 ≤ PROXIMAL_TOL=0.15 PROPOSED)
  and test_distal_residual_monitored (report + regression guard vs committed
  baseline overall p95 0.178 / thumb dist 0.143, REGRESSION_MARGIN 0.20 PROPOSED).
  Committed tests/g1_kinematic/residual_baseline.json. G1 is now 57/57 honest green
  (was 56 collected: 55 pass + 1 fail-closed; the 1 split into 2 passing tests → 57,
  no skips, no fail-closed). KEY FINDING flagged for sign-off: proximal is NOT
  near-zero at the tail (pooled p95 0.1236, max 0.219); tail is pose-correlated
  (mid-curl frames), consistent with 2-DoF base under-actuation on synthetic poses,
  not a regression (see Next 1). G1_RESIDUAL_THRESHOLD retired.
  ENTRY GATE confirmed before the dynamic work: Finding 1 resolved ✓ (r_dist
  fingertip-inclusive; test_fk_authority green) AND G1 green with proximal hard-
  gated ✓. Should-not-change set UNMOVED: G0 26/26, round-trip green, proximal exact
  at p50.
  G2 DYNAMIC HARNESS (src/sim, PyBullet dynamics/contact ONLY; metric FK stays the
  src/kinematics authority, ADR-0005): dynamics.py (masses/inertias, gravity, PD
  motors, mimic enforced PER STEP under stepping); closed_loop.py (landmarks →
  retarget() → safety.filter() → [CAN-latency delay buffer] → PD motors → step →
  read; retarget + filter imported READ-ONLY); grasp.py (cylinder power grasp +
  sphere enveloping grasp, palm-backed, ramped close, virtual force cap). ADR-0009.
  TESTS tests/g2_dynamic/ — 17/17 GREEN (per-gate run; project runs gate dirs
  separately): loop-rate two-part (p99 compute ~11.1 ms < 33,333 µs ceiling, ~3×
  headroom + regression vs 11,000 µs baseline), mimic-under-dynamics (abs err ≤
  0.009 settled / ratio within ~1.6% of 0.8917·1.1619 while stepping, both sides),
  grasp+force-cap (cylinder peak 12.5 N / sphere 9.1 N ≤ 15 N; cap proven load-
  bearing: uncapped torque → 25.5 N > cap), filter-ablation (adversarial fist:
  filter ON cmd+achieved overlap 0.00 mm, OFF overlaps ~9.6/10.7 mm > 2 mm margin —
  filter load-bearing), latency-stability (bounded, no divergence at 0/67/200 ms),
  limits/reserved/no-NaN under dynamics, tracking-penalty (monitored: dyn no-latency
  overall p95 0.232 vs G1 0.178, +0.054), closed-loop+filter both sides (every
  applied cmd non-penetrating, filter modifies 58/66 frames). Demo artifacts written
  to tests/g2_dynamic/out/ (gitignored): grasp_demo.json, closed_loop_{l,r}.csv,
  tracking_penalty.json.
  FORCE-CAP FINDING (ADR-0009): a per-joint torque cap does NOT linearly bound TOTAL
  grip force (fingers sum on one object); 0.12 Nm tuned to keep the worst grasp ≤
  15 N. GRASP TUNING FINDINGS (reported, NOT forced green per ticket): free
  fingertip pinch fragile → palm-backed enveloping grasp for the sphere; objects
  need a reaction surface + slow ramped close or they eject.
  PD GAINS ARE SIM-ONLY (not hardware gains). HARDWARE UNTOUCHED: no src/comms, no
  HW_ENABLE_TOKEN, no actuation; force clamp/watchdog remain comms/G3 specs. Do NOT
  advance past G2 (HUMAN gate). Wrote ONLY src/sim, tests/g1_kinematic,
  tests/g2_dynamic, docs/adr/0009, .gitignore, STATE.md.
- (safety-agent G2-v2, 2026-06-09) SELF-COLLISION FILTER + COMMAND GUARDS BUILT
  in src/safety (sim-free, hardware-free). NEEDS HUMAN REVIEW BEFORE MERGE.
  SEAM (locked first, per ticket): filter(candidate, prev_safe, dt) ->
  {joint_rad[20], clamped: True, modified: bool, reason: str|None}. Accepts a
  20-vector or an l20_targets dict (infers side); module-level filter() + a cached
  SafetyFilter per side. The G2 harness inserts this callable.
  FK SOURCE: imports the ONE authority src.kinematics (L20FK + conventions). NO
  third FK, NO PyBullet, NO sim import, NO runtime mesh load, NO actuation.
  WHAT IT DOES (a PROJECTION, not a checker): (1) static clamp — limits incl.
  mimic-tightened tip ranges, reserved idx 11-14 = 0, NaN/inf -> safe lower limit;
  (2) XPBD-style fixed-iteration (10) self-collision projection over adjacent
  fingers + thumb-vs-finger + fingertip-vs-palm, using an ANALYTIC rigid-body
  Jacobian read off the FK link transforms (axis x lever; mimic DIP/IP folded in
  by walking parent joints) — one FK eval per iteration; (3) rate limiting vs
  prev_safe (8 rad/s). Chatter-free composition: project candidate in the FULL-
  limits box to a rate-INDEPENDENT stable target, then rate-limit toward it, then
  re-project in the band ONLY if the rate clipped. ADR-0008 records the model +
  projection + timing.
  COLLISION PROXY (baked offline from the URDF collision meshes by
  _gen_collision_model.py, mesh-free at runtime): capsule/phalanx, radius = ½
  smallest mesh extent; PALM = palmar half-plane + y/z footprint (NOT a box — a
  box false-positives the natural fist; fist tips clear the real palm mesh by
  4-10 mm). Rest + full-fist validated collision-free.
  TESTS tests/g2_safety/ — 47/47 GREEN (both sides): interface/seam, idempotence
  (+ fixed-point on own output), projection-correctness (colliding -> non-
  penetrating, minimally changed), adversarial batch (50+ colliding/side ->
  no penetration; out-of-limits + dirty-reserved + NaN sanitised), rate limiting
  (teleport bounded by vmax·dt), continuity/no-chatter (sweep + 60-frame hold
  settles, spread <=1e-4), limits/reserved (+ mimic dependents in range),
  determinism, two-part timing, config sanity. Verified non-flaky (3× full runs).
  TIMING (two-part gate): (a) ABSOLUTE — every call < 33,333 µs frame (worst
  ~11.9 ms, ~3× headroom). (b) REGRESSION — committed p99 baseline 11,500 µs,
  margin 0.50 PROPOSED for human sign-off (see Next 3). p50 ~2.3 ms; collision-
  free fast path ~0.85 ms. The 3 kHz budget is retired (ADR-0007); the loop is
  camera-rate (30 Hz), so the filter has ~the whole frame.
  SPECS ONLY (config.py, comms enforces at G3): force clamp 15 N (« 100 N) + 0.6 A
  current + require HW_ENABLE_TOKEN; watchdog 0.20 s stale -> open-hand safe pose.
  HARDWARE UNTOUCHED. Do NOT advance past G2 (human gate). ADR-0008 added.
- (kinematics-agent-refactor, 2026-06-09) FK-authority + Finding-1 refactor DONE.
  WHAT BECAME CANONICAL: src/kinematics is the ONE sim-independent FK authority
  (pure yourdfpy; ADR-0005). conventions.py is the single convention source
  (DoF/joint/segment/landmark maps + mesh-derived TIP_LOCAL); oracle, sim and the
  solver codegen all import it. PyBullet stays in src/sim for dynamics + the mimic
  check only (no longer an FK authority). r_dist now runs to the PHYSICAL FINGERTIP
  incl. the DIP/IP mimic curl (ADR-0003 amended, ADR-0006).
  DISTAL SOLVE CHOICE (ticket §4 "report which you used"): the fingertip is a 1-DoF
  CURVE (two parallel-axis rotations at rates 1 and 1+ratio), so there is NO
  closed form. Non-thumb = bounded 1-D minimisation over the tip DoF (sanctioned
  fallback). Thumb = the old two-plane is invalid (u_dist.ktip not constant) ->
  joint solve: a fixed point on the distal latitude cd=vk/|v| (closed-form base
  per cd) + a robust base grid for near-parallel/under-actuated, with any high-
  residual result routed to the grid (robust on ALL reachable).
  BEFORE/AFTER DISTAL RESIDUAL (real set, 174 frames): overall p95 0.197 -> 0.178;
  thumb dist p95 0.275 -> 0.143; per-finger distal shifted (now fingertip-scored).
  ACCEPTANCE INVARIANTS — must-NOT-change (all hold):
    * G0 26/26 GREEN. (The 24 CORRECTNESS tests are unchanged + green. The 2 TIMING
      tests' BUDGET was reconciled to the iterative-distal reality (ADR-0007,
      human-approved) — see the one decision below; not a silent weakening.)
    * Reachable round-trip ~1e-7: G0 cache worst 6.4e-7; G1 round-trip green both
      sides; worst over 3000 independent random configs/side < 5e-6.
    * New-FK <-> oracle-FK ~1e-8: legacy segment_dirs == historical oracle FK to
      0.0 over 600 configs/side; L20FK link transforms == PyBullet to <1e-7 pos /
      <1e-6 rot (tests/g1_kinematic/test_fk_authority.py).
    * Proximal residual ~exact: real-set prox p50=0.000 every finger.
    * Hardware UNTOUCHED: no src/comms, no HW_ENABLE_TOKEN, no actuation.
  INTENDED-to-change (only these): r_dist fingertip-inclusive; oracle distal cache
  regenerated (plausible J_oracle rose); solver distal re-derived + its validation
  numbers; G1 distal residual re-measured.
  STRUCTURAL: no FK logic duplicated outside src/kinematics; conventions read from
  one source; oracle + sim import swaps complete.
  THE ONE HUMAN DECISION (ADR-0007): the closed-form-era 3 kHz timing budget no
  longer holds at the tail because Finding-1 removed the thumb closed form (it is
  now a 2-DoF iterative solve, ~7x the old thumb). Non-thumb alone meets the hard
  budget (p50 72/ p99 88us); full hand is p50 ~175us (median clears the 3 kHz
  period) but p99 ~780us (reachable) / ~710us (worst-case). Reconciled gate:
  representative p50 < 333us AND p99 < 1200us; worst-case p99 < 1200us.
  PATH BACK TO 3 kHz (tracked, gate before G2): port the thumb distal to
  C/Cython/vectorised; correctness is complete, this is purely a perf item.
  ADRs added: 0005 (FK authority), 0006 (fingertip distal solve), 0007 (timing).
- (Step 3, sim-agent G1) Built src/sim/ kinematic harness on PyBullet (DIRECT,
  useFixedBase, KINEMATIC ONLY: resetJointState/getLinkState — no motors, gravity,
  or contact). conventions.py = self-contained ADR-0003 segment + L20 joint maps
  (limits/mimic ratios read from URDF, not hardcoded). kinematics.L20Kinematics:
  loads URDF, sets the 16 active DoF, then ENFORCES the 5 mimics manually (PyBullet
  ignores <mimic>) = ratio*driver; FK link-frame origins give r_prox/r_dist.
  Verified PyBullet FK == yourdfpy FK (solver/oracle's basis) to ~1e-8.
  KEY GEOMETRY FINDING: the ADR-0003 segment *origins* are mimic-INDEPENDENT (a
  dip/ip mimic rotates the leaf distal link about its own origin, which doesn't
  move) — so segment_dirs don't see the mimic, but the physical fingertip /distal
  link ORIENTATION does; the mimic test asserts on link orientation, not origins.
  pipeline.track_frame = landmarks->retarget->set_config(+mimics)->FK->per-segment
  geodesic error. synth.py round-trips real configs; viz.py writes CSV + PNG/GIF
  (PIL only). Tests tests/g1_kinematic/: reachable round-trip (worst ~7e-7 rad,
  both sides, <=0.01 hard pass), mimic-enforcement (ratios vs LIMITS.md, enforced
  angles, FK sensitivity; thumb mimic name resolved per-side: thumb_dip right /
  thumb_ip left), limits/reserved (G0 fixtures + 150 jittered clouds/side: in
  range, idx11-14==0, mimics in range, no NaN), thumb-axis confirmation (renders +
  asserts cmc_pitch=in-plane flexion vs cmc_roll=out-of-plane abduction),
  visualization. ALL GREEN except test_real_residual, which FAILS CLOSED pending a
  human G1_RESIDUAL_THRESHOLD (distribution measured + reported above; artifacts in
  out/, gitignored). Imports finger_retarget.retarget read-only (see Interface
  note). Deps appended to src/sim/requirements.txt (pybullet, pillow).
- (perception-agent) Built src/perception/ (vision -> hand_landmarks). HandSource
  ABC + RawDetection; backends: MediaPipeHandSource (default, RGB world
  landmarks, lazy mediapipe/opencv import — see src/perception/requirements.txt),
  plus camera-free ReplayHandSource/SyntheticHandSource for CI. Frame transform
  (frame.py) builds the ADR-0003 hand_base frame from the PALM PLANE (wrist + 4
  finger MCPs): those 5 points are pose-invariant and coplanar (x==0) in the
  oracle frame, so to_hand_base is EXACT identity on every G0 fixture (silent-
  rotation guard) and a pure rigid recovery under any camera pose. Handedness-
  aware (radial axis mirrored for left -> matches mirrored fixtures). One-euro
  smoothing on by default, INPUT-side only (finger_retarget stays pure; no
  smoothing in solver hot path). Handedness mapping to_l20_side swaps MediaPipe's
  camera-view label unless image_mirrored. Robustness: no-detection/low-conf/
  non-finite -> hold last good + flag, never emits NaN; depth_confidence carried
  + flagged for weak monocular z (RGB-D can raise it). Recorder writes contract-
  valid sequences to tests/g1_kinematic/fixtures/real/ (CONVERGENCE POINT with
  sim-agent's G1 replay) — committed synthetic_openclose_{left,right}.json (90f
  each, camera-free, NOT real-camera; real-camera depth validation is a manual
  step). Tests: tests/g1_kinematic/test_perception.py — 39 GREEN (schema, frame-
  convention incl. exact match to G0 fixtures, rigid recovery, smoothing
  variance-drop + lag bound, handedness both sides x mirrored, hold-last-good,
  no-NaN, <60Hz latency, recorder round-trip). NOTE: 2 failures in the same dir's
  test_mimic_enforcement.py (KeyError 'thumb_dip') are sim-agent's (src/sim),
  outside this module — not introduced here.
- (bootstrap) repo scaffold created.
- (Step 0, sim/kinematics) Cloned linkerhand-urdf; vendored L20 only into
  src/sim/urdf/ (stripped nested .git + 8 unused hand models + archive, 581M->32M;
  see src/sim/urdf/PROVENANCE.md). L20 is already plain URDF (no xacro).
  Parsed all 21 revolute joints. KEY FINDING: 5 `mimic` joints reduce 21->16
  independent DoF, and the 16 independent joints == the 16 actuated L20 DoF.
  Distal "tip" = driver + fixed-ratio mimic (NOT one joint): non-thumb
  dip=0.8917*pip (tip cmd=pip); thumb ip/dip=1.1619*thumb_mcp (tip cmd=thumb_mcp).
  Wrote real lower/upper radians + URDF->semantic-index map into hardware/LIMITS.md.
  Confident map for all non-thumb fingers + thumb opposition(cmc_yaw) + thumb
  tip(mcp). Flagged: thumb base vs abduction (cmc_pitch vs cmc_roll) ambiguous.
  ADR-0002 distal-collapse assumption is confirmed well-founded by the mimic tags.
- (Step 1, eval-agent) Built eval/reference_solver/ (yourdfpy URDF FK + scipy
  multi-start oracle). Canonical segment convention recorded as ADR-0003
  (r_prox=unit(P_b-P_a), r_dist=unit(P_c-P_b); positional landmark groups;
  geodesic J). Oracle GRID-VALIDATED: per-finger min matches dense brute force.
  Scale-invariant + deterministic + both hands OK. Wrote 10 synthetic fixtures
  (flat/fist/pinch/point/thumbs_up x left/right) to tests/g0_unit/fixtures/.
  Note: synthetic thumb in some flexed poses is partly outside the L20 thumb
  reachable cone (genuine under-actuation: u_prox vs u_dist ~64 deg); oracle
  correctly returns nearest-reachable. Reachable-bound tests (G0 test 2) will use
  round-trip robot-derived targets, not these fixtures. Deps: eval/requirements.txt
  (venv at .venv/, gitignored). REFINED in Step 2: thumb segment anchor P_a moved
  to thumb_metacarpals so r_prox is a body-fixed vector (ADR-0003); cached oracle
  results (tests/g0_unit/fixtures/oracle_cache_*.json, 1000 reachable + 80
  plausible/side) for fast offline-graded G0 tests.
- (Step 2, solver-agent) Implemented src/finger_retarget/ closed-form solver.
  Non-thumb: subproblem-2 (base) + subproblem-1 (distal), EXACT on reachable.
  Thumb: exact 4-DoF/two-direction solve via a two-plane tip-axis construction
  (ktip = G.s satisfies u_prox.ktip=p.s, u_dist.ktip=rdist0.s) -> subproblem-2 +
  subproblem-1; reachable J~3e-6 over 2000 round-trips/side. Under-actuated
  (unreachable) thumb -> bounded base-DoF grid (real-time capped). Hot path is
  SCALAR (3-tuple) math + baked PoE constants (constants.py via gen_constants.py)
  -- numpy was too slow for size-3 vectors. ALL tests/g0_unit GREEN (26):
  matches-oracle (1000 reachable J<=1e-3; plausible p95<=8e-3, bounded),
  per-segment, limits+reserved, scale+translation invariance, degenerate,
  determinism, handedness, timing (representative p99=73us; worst-case p99=265us;
  budget 333us). ADRs 0001/0002/0003/0004 recorded. NOTE: src/finger_retarget
  is pure/no-eval-import; tests + oracle live in eval/ + tests/ (architecture).
- (2026-07-13, visual ACT offline evaluation) Added
  `scripts/evaluate_g20_visual_act.py`, which automatically uses the installed
  LeRobot environment, selects evenly spaced frames from the three held-out
  camera episodes, runs independent one-step ACT predictions, and writes a JSON
  report plus preview/metric PNGs. It is strictly offline: no ROS import, live
  camera access, or hardware command path. Checkpoint 010000 on 60 held-out
  frames: active-joint MAE 15.09 SDK ticks versus 16.87 for repeating the current
  state (clamped ACT MAE 14.70). Largest errors are output indices 1, 2, and 4
  (~28-29 ticks); 84/960 active output values were outside 0..255, affecting
  34/60 samples, so this is evidence of modest learning rather than
  hardware-readiness. Outputs live under
  ignored `artifacts/g20_visual_act/evaluation/010000/`; the repeatable command is
  documented in `../collectdata.md`.
- (2026-07-13, G20 visual ACT first hardware runner) Added
  `src/comms/visual_act_to_linkerhand.py` for the trained camera-2 + 20-D state
  ACT checkpoint. Default is prediction-only dry-run and the process always
  starts DISARMED. Actual SDK-range publishing requires human-set
  `HW_ENABLE_TOKEN=1`, `--enable-motion`, and a SPACE press in the OpenCV window.
  First-bring-up defaults: 5 Hz, max 2 SDK ticks from freshly observed state per
  cycle, current/speed 20, automatic disarm after 10 s; also disarms on stale
  state, raw-output overshoot, or >80-tick target delta. Commands clamp 0..255
  and force G20 reserved indices 11..14 to 255. The environment/import path,
  CLI help, syntax, and pure clamp/step gates were checked; no SDK node/topics
  were live during this turn, so no hardware command was published. T4a dry-run
  and T4b gated hardware commands are documented in `../collectdata.md`.
- (2026-07-13, visual ACT chunk execution correction) Hardware logs confirmed
  commands were publishing and moving substantially without a safety trip, but
  the first-bring-up runner had overridden the trained ACT configuration from
  `n_action_steps=10` to 1. Added `--n-action-steps` (default 10) and now load the
  checkpoint with that value, so one inferred trajectory chunk supplies 10
  sequential actions before re-planning. The safety gates, observed-state step
  limiter, SPACE disarm, and low current are unchanged. Added the gated T4c
  command to `../collectdata.md`; hardware was not actuated by Codex.
- (2026-07-13, human-rated ACT self-imitation + fine-tune) Extended
  `src/comms/visual_act_to_linkerhand.py` with opt-in integrated recording so
  camera 2 is opened only once. With `--record-rated-attempts`, each SPACE-armed
  hardware attempt records camera JPEGs, observed 20-D SDK state, and the exact
  safety-limited 20-D command actually published. When it stops, the next
  attempt is blocked until the human presses 0, 5, or 1 in the OpenCV window;
  `episode.json` stores quality 0/0.5/1. Output matches the existing T3
  `samples.jsonl` fields and lives under `data/self_imitation/` by default.
  Extended `scripts/train_g20_visual_act.py`: unscored expert episodes remain;
  rated episodes below `--min-rated-score` (default 1) are retained on disk but
  excluded from BC; accepted rated episodes are forced into train rather than
  held-out validation. `--finetune-from` loads an existing pretrained_model via
  LeRobot's path config and defaults to a 3e-6 fine-tune LR. Verified: syntax +
  no-token hardware gate; synthetic 40-frame rated episode round-trip; quality
  0/0.5 exclusion and quality-1/expert inclusion; validation split isolation;
  and a real one-step CUDA fine-tune loaded the 19M-param 010000 checkpoint and
  updated at lr=3e-6. No hardware was actuated by Codex. Full T5 collection,
  scoring, fine-tuning, evaluation, and deployment commands are in
  `../collectdata.md`.
- (2026-07-13, rated-attempt stop diagnosis + reset) Inspected actual
  `data/self_imitation` episode metadata: the latest attempts stopped because
  raw ACT output crossed the configured safety envelope, not because the model
  completed or the duration elapsed. Fixed the UI to preserve the triggering
  raw min/max and envelope while waiting for a rating (preview inference no
  longer overwrites it). Added a post-rating reset state machine: after 0/5/1,
  the runner safety-ramps from freshly observed state to the canonical G20 open
  SDK pose `[255x6,193,148,105,42,245,255x9]`, default max 15 ticks/cycle and
  3-tick completion tolerance. It blocks SPACE until `RESET complete`; reset
  pauses on stale ROS state and uses the existing low current/speed settings.
  `--no-reset-after-rating` opts out. Updated T5 docs; no hardware command was
  sent by Codex.
- (2026-07-13, first human-rated ACT fine-tune completed) User requested an
  immediate batch fine-tune. Found 5 score-1 attempts / 2908 frames, alongside
  3 score-0, 4 score-0.5, and 4 pending episodes. Built independent ignored
  artifact `artifacts/g20_visual_act_finetune_01` from all 30 original camera
  demonstrations plus only the 5 accepted self-imitation attempts: 35 episodes
  / 13353 total frames, with 32 train / the same original held-out IDs
  [10,14,19]. Fine-tuned from original 010000 for 3000 steps at lr=3e-6,
  batch=8, saving every 500; training completed normally (loss ~0.192 -> ~0.14).
  Evaluated all six checkpoints on the same 60 held-out frames. Best is 002000:
  active-joint raw/clamped MAE 14.79/14.38 vs original 15.09/14.70 (1.94% raw
  improvement); out-of-range active values 80/960 vs original 84/960. Later
  003000 regressed to 15.07, so deployment docs explicitly select 002000.
  Artifact size 2.2G. No hardware was actuated during conversion, training, or
  evaluation.
- (2026-07-14, tactile-qualified ACT self-imitation) Extended
  `src/comms/visual_act_to_linkerhand.py` to subscribe to
  `/cb_<side>_hand_matrix_touch_mass`, flatten the six SDK mass regions, and
  record real `mass_values`, hysteretic `contact_6`, touch age, and an
  episode-level contact summary in every new rated attempt. Added a configurable
  grasp-success gate (defaults: contact on/off 20/10 g, thumb plus at least two
  other fingers, continuously 0.5 s, touch no older than 0.5 s). A rated attempt
  stops publishing when this stable condition is reached. Human score 1 without
  the touch gate is preserved as `human_quality_score: 1` but stored as
  `quality_score: 0.5`, keeping it out of ACT behavior cloning; scores 0/0.5 stay
  available as negatives for a future reward/quality model. The camera overlay
  shows live contact count, thumb state, hold time, and PASS/WAIT. Updated the T5
  command in `../collectdata.md`. Verified pure nested mass parsing, contact
  hysteresis/hold/staleness, tactile recorder JSON round-trip and touch-adjusted
  rating, Python syntax, CLI flags, and the no-token motion refusal. No hardware
  command was published by Codex.
- (2026-07-14, 90-degree rotation recording parameters) Updated the T2 block in
  `rotation.md` to use the user's current calibrated teleop values:
  `hardware-base-gain=1.80`, `hardware-thumb-abd-offset=-28`, and
  `--log-sim-position`, while keeping camera 2 for the human-hand view and the
  required `--motion-key-toggle` so T3's ROS-triggered camera-0 recorder starts
  and stops with SPACE. Documentation-only change; no hardware was actuated.
- (2026-07-14, CW rotation ACT training filter) Added repeatable
  `--include-task-id` filtering to `scripts/train_g20_visual_act.py`, allowing a
  policy dataset to include only matching recorder sessions instead of mixing
  rotation with earlier orientation-grasp/self-imitation data. Verified the two
  `rotate_object_90deg_cw` sessions: 23 usable episodes / 17,581 valid image,
  20-D state, and 20-D command frames; the deterministic split is 20 train / 3
  held out. Added the complete offline 5,000-step fine-tune and evaluation
  commands to `rotation.md`, initialized from the best prior grasp checkpoint
  `g20_visual_act_finetune_01/002000`. Python syntax and a full dry-run training
  command were checked. No training was started and no hardware was actuated.
- (2026-07-14, CW rotation continuation setup) The first CW run completed 5,000
  steps (~2.6 epochs, loss ~0.29-0.31). Evaluated all ten checkpoints on the same
  60 held-out frames: active-joint MAE ranged 26.65 to 24.50 ticks; 003500 was
  best at 24.50 and 005000 measured 24.73, while the repeat-current-state
  baseline was 21.11. This does not yet establish useful generalization, but the
  low epoch count supports a controlled longer run. Added
  `--reuse-dataset-from` to reuse a completed artifact dataset only with
  `--stage train`, allowing continuation output to live in a separate artifact
  without duplicating the 1.0 GB dataset or deleting source checkpoints. Added
  a safe 10,000-additional-step command (effective 15,000 total) and held-out
  evaluation command to `rotation.md`. No continuation training or hardware
  actuation was started by Codex.
- (2026-07-14, CW rotation continuation evaluated) User completed the additional
  10,000-step run from the original rotation 005000 checkpoint (effective
  15,000 total). Evaluated every continuation checkpoint on the unchanged 3
  held-out episodes / 60 frames. Final continuation 010000 is the best overall:
  active/clamped MAE 23.92/23.38, improving on original-run best 003500 at 24.50
  and continuation 006000 at 24.16. It produced 61/960 out-of-range active
  values in 35/60 samples. The repeat-current-state baseline remains better at
  21.11, so this is evidence that longer training helped, not evidence of
  hardware readiness. Recorded the selected candidate and caveat in
  `rotation.md`; all evaluation was offline and no hardware was actuated.
- (2026-07-14, CW rotation gated hardware handoff) Added live-camera dry-run and
  first human-gated hardware commands to `rotation.md` using the selected
  continuation 010000 checkpoint and camera 0. The hardware command retains the
  triple gate (`HW_ENABLE_TOKEN`, `--enable-motion`, human SPACE), starts
  disarmed, uses current/speed 20/35, max 5 observed-state ticks per cycle,
  target/raw/stale guards, and a 30-second active limit. Rated attempts are
  recorded separately under `data/rotation_self_imitation`; stable three-finger
  touch remains a score-1 qualification but `--no-stop-on-contact-success`
  prevents the grasp phase from ending before the 90-degree rotation. Current
  read-only checks found no T1 process, ROS hand topics, camera owner, or runner,
  so the human must start T1 before dry-run/test. Codex did not set the hardware
  token, press SPACE, publish a command, or actuate hardware.
- (2026-07-14, CW rotation first-trial speed adjustment) User's first gated
  rotation trial published normally but moved too slowly with max 5 ticks at
  10 Hz, then disarmed at target delta 100.6 against a 100-tick guard. Updated
  the T7 command to max 12 ticks/cycle, speed limit 60, target-delta guard 140,
  active limit 60 seconds, and reset step 15; current remains 20 and raw/stale
  safety guards remain unchanged. Read-only ROS inspection also found the
  running palm-touch node publishes state but has no
  `/cb_right_hand_matrix_touch_mass` publisher (the ACT subscriber is present),
  matching the UI's `TOUCH: no fresh data`; this does not slow motion but means
  score 1 will be downgraded until T1 detects matrix touch again. Codex did not
  set the token, press SPACE, or actuate hardware.
- (2026-07-14, CW rotation second-trial timing adjustment) The second human-run
  trial stayed within raw/target guards and continued publishing, but hit the
  configured 60-second duration before task completion (931 recorded samples).
  Since demonstrations are 30 FPS while the runner was executing actions at
  10 Hz, the learned trajectory was being played on an approximately 3x slower
  time base. Updated T7 moderately to 15 Hz, max 15 ticks/cycle, speed limit 75,
  and a 90-second active limit. Current remains 20; target/raw/stale guards and
  human SPACE disarm remain unchanged. No hardware was actuated by Codex.
- (2026-07-14, CW rotation per-step adjustment) At the user's request to make
  each movement increment faster, updated only T7's observed-state command step
  from 15 to 25 SDK ticks and reset step from 15 to 25. Rate stays 15 Hz,
  current/speed stay 20/75, and target/raw/stale/duration/SPACE gates are
  unchanged, isolating step limiting as the variable. No hardware was actuated
  by Codex.
- (2026-07-14, CW rotation teleop-timebase configuration) Replaced the temporary
  large-step setup with a timing-matched T7 configuration: rotation demos were
  recorded at 30 FPS, so ACT now executes at 30 Hz instead of 15 Hz, with the
  original teleop-like max step 5 and speed 35. EMA alpha was reduced from 0.3
  to 0.1 to reduce temporal lag, while current stays 20 and all target/raw/stale,
  duration, human SPACE, and reset/rating gates remain. Reset step is 10. This
  should approach the demonstrated teleop time scale through continuous action
  playback rather than larger jumps; actual rate remains bounded by camera and
  inference throughput. No hardware was actuated by Codex.
- (2026-07-14, CW rotation two-minute limit) User requested a 120-second attempt
  limit. Inspection of the immediately preceding pending episode showed it had
  actually stopped after 101 samples because raw max 275.4 exceeded the
  configured upper envelope 275 by 0.4, not because of the 90-second limit.
  Updated T7 to `max-active-seconds=120` and modestly widened raw overshoot from
  20 to 25 (raw envelope -25..280); final commands still clamp to 0..255 and
  remain subject to target-delta, per-observed-state step, stale-state, current,
  and human SPACE gates. No hardware was actuated by Codex.
- (2026-07-14, ACT preview/safety logging) Debugged a run where position-like
  commands appeared before the operator pressed SPACE. This was the intended
  disarmed inference preview: the command publisher is gated by `armed`, and
  `policy.reset()` clears the preview action queue when SPACE arms the runner.
  Updated the terminal label to `PREVIEW ONLY (not published)` with
  `would_cmd`, and now print the exact safety-disarm reason before the recorder
  reports `STOP`. The latest attempt stopped because raw max 280.3 exceeded the
  configured 280 envelope by 0.3, so T7 now uses a modest raw overshoot of 30
  (raw envelope -30..285). Final commands remain clamped to 0..255 and limited
  to 5 observed-state ticks per cycle. No hardware was actuated by Codex.
- (2026-07-14, CW rotation action-chunk stall diagnosis) Inspected the user's
  1,400-sample attempt (~93 seconds). The runner remained alive and continuously
  published; recorded state stayed within 5 ticks of the command, so low motor
  current was not the cause of the apparent stall. The policy instead entered
  a local/repeating posture. Its ACT checkpoint predicts 30-frame chunks but
  the runner was executing only the first 10 frames before replanning, which
  can repeatedly select the early stabilization/closing portion. Updated the
  T6/T7 rotation tests to run at the demonstrated 30 Hz and execute all 30
  actions per chunk. Per-state 5-tick limiting and raw/target/stale/human gates
  remain active. The prior attempt eventually stopped at raw -44.7 outside the
  configured [-40,295] envelope. No hardware was actuated by Codex.
- (2026-07-14, T7 camera ownership preflight) The 30-action dry-run could not
  open camera 0 because the previous 10-action hardware runner (PID 51948) was
  still holding `/dev/video0`. Updated the T7 command to run in a subshell,
  interrupt only older `visual_act_to_linkerhand` processes, wait up to three
  seconds for cleanup, and refuse to start while any other camera owner remains.
  It does not stop T1 or indiscriminately kill camera users. No process was
  stopped and no hardware was actuated by Codex.
- (2026-07-14, T7 moderate speed increase) User requested faster execution and
  proposed max step 15, current 80, and speed 60; the working file had current
  set even higher at 180 and had lost most of the camera preflight. Restored the
  old-runner/camera ownership checks, kept the 30 Hz full 30-action chunk,
  increased observed-state step from 5 to 10 and SDK speed from 35 to 60, and
  used the already teleop-tested current limit 35 because current controls
  force/thermal load rather than policy playback rate. Reset step is now 15.
  All target/raw/stale, human SPACE, duration, and rating gates remain. No
  hardware was actuated by Codex.
- (2026-07-14, T8 lower-protection configuration) Added a separate aggressive
  rotation test rather than weakening T7 in place. T8 uses 30 Hz, the full
  30-action chunk, step 15, speed 75, current 35, target delta 220, raw envelope
  [-60,315], and reset step 20. It intentionally keeps the human token/SPACE,
  stale-state, 120-second duration, current, final 0..255 clamp, per-state step,
  old-runner, and camera-ownership protections. This accommodates the observed
  -44.7/286.7 model outputs without making current/thermal limits unbounded. No
  hardware was actuated by Codex.
- (2026-07-19, MediaPipe action-library teleop) Added a G20 SDK-range motion
  primitive library with raw hand-base MediaPipe takes, scale-invariant feature
  extraction, DTW sequence matching, confirmation/margin/cooldown gates, and a
  step-limited trajectory queue. Added hardware-free recording and validation
  CLIs, an offline G20 waypoint slider/JSON editor with optional approximate
  PyBullet preview, plus a teleop CLI that remains dry-run by default and requires the human
  token, `--enable-motion`, and SPACE before ROS publishing. Seven focused tests
  pass; the wider comms suite could not collect in the system Python because
  optional `yourdfpy` is absent. No hardware was actuated by Codex.
- (2026-07-19, grouped action-library capture) Added a read-only unified
  recorder for collecting repeated MediaPipe gestures and official-GUI G20
  waypoints in one session. SPACE advances HUMAN/ROBOT/group phases, optional M
  marks repetition boundaries, and S snapshots the fresh GUI command, measured
  state, and a second-camera robot photo. Each group is self-contained and an
  offline analyzer extracts candidate raw-landmark takes plus trajectory JSON;
  it falls back to motion-energy segmentation when markers are absent. The live
  human preview now overlays the 21-point MediaPipe skeleton in green for fresh
  detections and yellow for held detections while saved JPEGs stay raw. M now
  toggles exact per-repetition human takes on/off, allowing many takes in one
  group; SPACE advances from HUMAN READY to robot capture and H is unused. Q
  re-records only the current group's human side and E re-records
  only its robot waypoints/photos; replaced attempts are archived under the
  group's `revisions/` directory instead of deleted. X/ESC exits. Fifteen
  related tests pass. The
  recorder creates no ROS publisher and no hardware was
  actuated by Codex.
- (2026-07-23, calibrated ArUco cube pose preview) Verified from 1,490 sampled
  manipulation frames that the physical cube uses OpenCV `DICT_4X4_50` IDs
  0..23 (six consecutive four-ID faces), not AprilTag 36h11. Added reusable
  camera/layout JSON models, learning of a face's 2x2 ID placement and each
  printed marker's rotation, planar IPPE+LM 6D pose estimation with
  reprojection error, and relative-yaw calculation. Added a printable ChArUco
  generator/manual camera-calibration CLI plus a preview-only top-face tracker
  configured for the measured 55 mm cube, 18 mm markers, and approximately
  35 mm marker-center spacing. Face learning uses a center-grid homography to
  recover each physical black-border corner, including arbitrary sticker
  rotation and effective marker size; this fixed the first axis-aligned model's
  approximately 10.8 px error. Across 303 real full-face samples the corrected
  model has median 0.58 px and maximum 1.11 px reprojection error, with inferred
  black-border size 17.43 mm. The tracker has no ROS publisher and cannot
  command hardware. Eight focused/new compatibility tests pass with pytest
  plugin autoload disabled; real recorded frames also learn a complete face
  layout. No hardware was actuated by Codex.
- (2026-07-23, calibrated CCW-yaw trajectory collection) Added an offline
  episode annotator for the SDK recorder's saved camera/state/action/touch
  trajectories. It estimates cube pose from each saved frame, defines yaw zero
  from the first stable 15 valid frames, unwraps/filters relative yaw, and marks
  an episode valid only after a consecutive terminal hold near +90 degrees.
  Original recordings are not modified; per-frame pose and per-episode/session
  summaries are written beside them. Documented a three-terminal recording
  workflow using camera 0 for the calibrated robot view and camera 2 for
  MediaPipe teleoperation, with yaw sign +1 and the learned face-5 layout.
  Eleven focused pose/annotation/readiness tests pass with pytest plugin
  autoload disabled. No hardware was actuated by Codex.
- (2026-07-23, rotation workflow cleanup) Reduced `rotation.md` from 930 to
  260 lines. It now contains only CAN/driver setup, the calibrated horizontal
  CCW-90 pose check, the three-terminal recorder/MediaPipe workflow, offline
  yaw validation, and the previously verified vertical-CW ACT checkpoint as a
  clearly separated rollback command. Removed duplicated driver/recorder
  commands, obsolete training experiments, and superseded hardware-test
  variants. Retained the full tuned MediaPipe-to-G20 mapping because those
  values materially affect demonstrations. Command-line flags were checked
  against the current CLIs. No hardware was actuated by Codex.
- (2026-07-23, action-library IDs 6--9 handoff) Confirmed the current
  `action_library_phase_teleop` number-key path already accepts primitive IDs
  1--9 and resolves them directly from the loaded library, so adding ID 6 does
  not require a key-map code change. Updated the grouped-recording runbook to
  start new captures at `group_006`, analyze them, import them into
  `core_actions_v1` with `src.comms.import_action_group`, validate the complete
  library, and optionally force full frame-1 playback with repeated
  `--manual-action-from-start` arguments. New IDs explicitly omit `--replace`;
  command/state audit failures should be corrected rather than bypassed. No
  hardware was actuated by Codex.
- (2026-07-23, action 6 reverse of action 1) Added primitive 6
  `four_fingers_full_open_reverse_one` to the shared `core_actions_v1`. Its
  151-frame G20 trajectory and all six human templates are exact time reversals
  of primitive 1, while primitive 1 remains byte-for-byte unchanged. Added a
  reusable `src.comms.clone_reversed_action` CLI and manifest-backed
  `manual_from_start` primitive property. Number-key playback now respects that
  property, so key 6 always uses a bounded transition to action 1's endpoint
  and then plays the complete reverse trajectory instead of selecting a
  nearest middle frame. Full-library leave-one-out validation is 37/37 at
  margin 0.015; 59 focused action-library/phase/replay tests pass. Offline
  playback finishes exactly at action 1's first pose with max commanded step
  below 10 ticks. Updated runbooks and moved future newly recorded IDs to
  7--9. No hardware was actuated by Codex.
- (2026-07-23, action 6 side-pinch prelude) Refined primitive 6 so it can retain
  the cube while opening. Frame 0 remains the exact action-1 endpoint. The next
  24 frames hold all flexion channels fixed while q6--q9 move gradually from
  `[193,148,105,42]` to `[125,129,125,130]`, the official G20 GUI `动作2`
  finger-together preset; the remaining reversed action-1 flexion then plays
  with that spread target held. The trajectory is now 175 frames and its
  maximum frame-to-frame command delta remains 3.69 ticks. Added a reusable,
  archiving `src.comms.tune_action_spread_pinch` CLI; the former 151-frame
  primitive and manifest are recoverable under
  `archive/20260723_163417_before_action_006_spread_pinch`. Full-library
  validation remains 37/37 and 60 focused tests pass. No hardware was actuated
  by Codex.
- (2026-07-23, action 6 coupled-spread catch-up fix) Diagnosed the operator's
  `catch-up timeout` during action 6: because its new pinch moves q6--q9, the
  fixed-sequence runner added all four mechanically coupled spread channels to
  the strict 18-tick feedback gate. One lagging spread stopped frame
  advancement during the pinch prefix, so later middle/ring flexion commands
  were never published. Added manifest-backed
  `best_effort_spread_feedback`; action 6 still publishes and step-limits all
  q6--q9 targets but excludes them from command-lead, following-error, and
  terminal-settle blocking. All thumb/base/tip channels retain the original
  strict feedback gates. Dry-run reaches the expected open-flexion endpoint
  with spread `[125,129,125,130]`; full-library validation is 37/37 and 61
  focused tests pass. No hardware was actuated by Codex.
- (2026-07-23, action 6 contact command-lead fix) A second hardware log showed
  the spread exclusion worked, but cube-loaded q2/middle-base, q3/ring-base,
  and q17/middle-tip stayed 20--22 ticks behind the last command, so the
  global 18-tick soft catch-up gate repeatedly timed out before the fingers
  could reopen. Added optional manifest-backed `max_command_lead` and set it
  to 28 only for primitive 6. Action 6 now continues through normal contact
  lag up to 28 ticks, waits above 28, and retains the unchanged 35-tick hard
  following-error stop; actions 1--5 still use the CLI default of 18. The
  coupled q6--q9 feedback remains best-effort while all base/tip/thumb
  channels remain strictly monitored. Full-library validation is 37/37 and
  62 focused tests pass. No hardware was actuated by Codex.
- (2026-07-23, action 6 tighter outer-finger pinch) Tightened only action 6's
  index/little spread pair after the operator reported weak cube retention:
  its held q6--q9 target changed from `[125,129,125,130]` to
  `[115,129,125,140]`, moving q6 and q9 another 10 ticks inward while q7/q8
  stay unchanged. The installed prelude was retargeted in place rather than
  stacked, so the trajectory remains 175 frames; all 16 non-spread channels
  are exactly unchanged and max frame step is 4.09 ticks. The preceding
  trajectory and manifest are archived at
  `archive/20260723_165101_before_action_006_spread_pinch`. Extended the tuning
  utility with idempotent retuning and a regression test. Full-library
  validation remains 37/37 and 63 focused tests pass. No hardware was actuated
  by Codex.
- (2026-07-23, action 6 outer-finger flexion hold) Live read-only ROS feedback
  showed q6--q9 actually reached `[116,129,125,138]` for the commanded
  `[115,129,125,140]`, ruling out a software safety clamp as the cause of weak
  retention. Updated only q1/q4/q16/q19 in action 6: during the reverse/open
  tail they are capped at 235, retaining 20 ticks of index/little base/tip
  flexion, while middle/ring q2/q3/q17/q18 still reach 255. The prelude, spread
  targets, thumb, middle/ring, length (175 frames), and 4.09-tick maximum step
  are unchanged. The prior version is recoverable under
  `archive/20260723_165837_before_action_006_spread_pinch`. Added an optional
  `--outer-flexion-max` tuning control and regression coverage. Full-library
  validation remains 37/37 and 64 focused tests pass. No hardware command was
  published by Codex.
- (2026-07-24, right-flip collection thumb hold) Added an independent `F` key
  override to `action_library_phase_teleop`. It captures measured q0/q5/q10/q15
  when pressed and pins only those thumb channels after AUTO/manual/FULL
  MediaPipe target generation but before the existing command-step and
  measured-state-lead guards. F again releases; episode stop and D preserve the
  latch, while reset moves only non-thumb joints and checks completion against
  the held thumb values. Hand loss and stale state clear it automatically.
  Added `--freeze-thumb-on-start` to latch the first measured hardware thumb
  pose before an optional startup reset, eliminating the first manual F press.
  Added a real DISARMED-only R reset handler that keeps the recorder trigger
  false, cancels queued manual playback, and preserves the frozen thumb. The
  right-flip collection command now enables `--reset-after-disarm`, so ARMED
  SPACE saves the episode and automatically resets only non-thumb joints.
  Added a dedicated
  `cube_right_flip_from_behind_fingers_v1` three-terminal recording runbook
  which preserves the preceding policy's cube-in-fingers start pose and uses
  FULL MediaPipe for the four fingers with the thumb held. The focused phase
  teleop suite passes 32/32; CLI compilation/help and recorder flags were
  checked. No camera, ROS publisher, or hardware motion was started.
- (2026-07-24, dynamic key-6 thumb roundtrip) Number key 6 in phase teleop and
  in the fixed replay window's single-action path now reads measured hardware
  state, finds the nearest action-2 frame using q0/q5/q10/q15 only, plays the
  action-2 thumb to its endpoint, and reverses to action-2 frame 0. All
  non-thumb channels are held at their key-press values. An F thumb latch is
  temporarily bypassed during this explicit thumb action but remains latched
  afterward; T cannot hand the fixed four fingers to MediaPipe during key 6.
  The stored primitive-6 data remains unchanged for provenance and fixed-order
  sequence playback. Fixed a bring-up regression where the key-6 handler reused
  the camera variable name `source`, replacing `MediaPipeHandSource` with the
  action-2 `Primitive`; camera ownership now uses `hand_source` and the action
  uses `source_primitive` consistently across pipeline construction, read,
  overlay, and cleanup. No hardware was actuated by Codex.
- (2026-07-24, startup thumb offset) Added
  `--startup-thumb-offsets DQ0,DQ5,DQ10,DQ15` to phase teleop. With
  `--freeze-thumb-on-start --reset-on-start`, it derives a clipped 0--255
  thumb target from the first measured state, reaches it through the existing
  DISARMED step/state-lead guarded startup reset, and then retains it as the F
  hold. Non-thumb channels are unchanged by the offset helper. The CCW-yaw
  collection example uses `30,0,0,0`, taking q0 from 0 to 30 when its measured
  start is zero. No hardware was actuated by Codex.
- (2026-07-24, restored fixed startup thumb pose) Added
  `--startup-thumb-pose Q0,Q5,Q10,Q15` for an absolute, reproducible startup
  thumb hold rather than a measured-state offset or action-library pose. The
  current CCW-yaw collection command restores q0/q5/q10/q15 to
  `[116,253,254,118]`, recovered exactly from all 43 measured frames in
  `20260724_165206_cube_right_flip_from_behind_fingers_v1/episode_000`.
  Startup reset moves there while DISARMED and then retains the F hold. The
  mistaken temporary action-derived startup branch was removed. No hardware
  was actuated by Codex.
- (2026-07-24, CCW collection A2/A3 lock disabled) Changed the current
  `collect_cube_yaw_ccw90_keyboard_hybrid.md` phase-teleop command to
  `--no-a23-spread-routing` and removed its unused spread threshold/hysteresis
  overrides. Four-finger closure no longer restricts the matcher to A2 or
  disables A3 contact assist in this workflow. The general CLI default remains
  unchanged for other runbooks. No hardware was actuated by Codex.
- (2026-07-24, CCW-90 history+momentum training command) Audited the latest
  `20260724_212905_cube_yaw_ccw90_keyboard_hybrid_v1` session: all 15 episodes
  contain complete camera images plus 20-D measured state and command. Added a
  from-scratch ACT command with six-frame/three-second visual history, four
  state-history samples, 30-frame chunks, and momentum hinge weight 0.2.
  Episode 000 is explicitly excluded from the training split because its
  35.8-second duration and 28.1% active-command ratio are strong outliers
  against the other 6.5--13.9-second/approximately 37--38% episodes. A dry run
  selected 11 training and 3 held-out episodes and confirmed the momentum
  trainer command. No dataset conversion or training was started by Codex.
- (2026-07-24, CCW-90 20K checkpoint bring-up) Confirmed the completed
  checkpoint contains the expected 30-frame ACT chunk, 5 action steps,
  six-frame visual mosaic, and 80-D four-state history input. Final training
  logged BC L1 0.05298, direction loss 0.028714, weighted momentum 0.005743,
  total loss about 0.070, and finite gradient norm about 4.9 after 45.6 epochs.
  Added a conservative first hardware test using the demonstrated
  current/speed 20/50 and max step 10, with a 20-second human-gated attempt.
  The runbook warns not to use the runner's generic-open R reset because this
  dataset starts from the fixed thumb `[116,253,254,118]`, and requires three
  DISARMED seconds to populate visual/state history before SPACE. No camera,
  ROS publisher, or hardware motion was started by Codex.
- (2026-07-24, 30K-to-20K P-key policy handoff) Extended
  `visual_act_to_linkerhand` with an optional one-way second-policy handoff.
  Both checkpoints are preloaded despite their different history layouts
  (30K: six visual frames over 120 ticks and five state frames; 20K: six
  visual frames over 90 ticks and four state frames). Pressing P disarms the
  30K policy, lowers the configured handoff torque/speed, and moves all active
  joints through measured-state step limiting to the 20K demonstrations'
  median start pose. The target must remain within tolerance for three frames;
  the runner then fills three seconds of 20K history while holding the pose
  and automatically arms the 20K policy. SPACE aborts the transition, timeout
  holds the measured position, and both paths remain disarmed. Added a full
  two-checkpoint command and operating sequence to the CCW collection runbook.
  The primary-only thumb push/gate logic and primary endpoint stopper are
  disabled after switching. Eighteen focused ACT tests pass, including pose
  parsing, step limits, confirmation, warmup drift reset, timeout, overlay,
  history, thumb gate, and endpoint stopping. No camera, ROS publisher, or
  hardware motion was started by Codex.
- (2026-07-25, 30K v3 chunk/batch audit and v4 comparison command) Reviewed
  the 66.5-second hardware screencast and the completed v3 checkpoint. The
  recording shows the policy continuing its learned joint cycle while the
  cube is absent from the hand workspace, which larger batch size alone
  cannot solve. A 60-sample held-out check of checkpoints 10K/20K/30K found
  30K was the best of the existing run but still produced clamped active-joint
  MAE 4.36 ticks versus a 3.27-tick repeat-state baseline, with out-of-range
  active outputs in 54/60 samples. Added a separate, from-scratch v4 comparison
  command to `hardcode_position.md`: 60-frame chunks, batch 16, 15K steps
  (same 240K sample budget as v3's batch 8/30K steps), unchanged four-second
  history and momentum weight 0.2, plus exclusion of demonstrations above six
  four-finger or eight thumb-route reversals. The command writes a new
  artifact and does not replace v3. Changing ACT chunk length cannot be done
  by fine-tuning the 30-frame checkpoint because the action-query/output
  horizon changes. No training, ROS process, camera, or hardware motion was
  started by Codex.
