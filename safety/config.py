"""src/safety — tunable configuration, force/watchdog SPECS, and the committed
filter-latency baseline.

Everything the filter needs to be deterministic lives here as plain data: the
self-collision projection parameters, the rate limit, the force-clamp spec, the
watchdog spec, and the safe pose. Per the root CLAUDE.md this module *specifies*
the force clamp and watchdog; it never actuates and never enforces force in a
hot loop — ``comms`` consumes these specs at G3.

Nothing here imports a sim or a hardware path. ``SafetyConfig`` is frozen so a
filter built from it is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- L20 layout (mirrors src/kinematics / contracts; kept local so config is a
#     leaf module). idx 11-14 are RESERVED -> always 0.0. ---------------------- #
RESERVED_IDX = (11, 12, 13, 14)
ACTIVE_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19)
N_JOINTS = 20

# Grip-force absolute ceiling from the root CLAUDE.md ("far below 100 N").
HW_FORCE_HARD_MAX_N = 100.0

# 30 Hz camera -> one frame = 33,333 µs is the WHOLE retarget->filter->command
# loop budget (ticket "Timing note"; the retired 3 kHz budget is ADR-0007).
LOOP_PERIOD_US = 33_333.0


@dataclass(frozen=True)
class ForceClampSpec:
    """Spec consumed by ``comms`` at G3 (this module never actuates).

    Start FAR below the 100 N grip max during bring-up. ``max_grip_force_N`` is
    an absolute cap; ``max_fraction_of_hw_max`` is the policy ceiling the config
    sanity test asserts against. ``per_joint_current_a`` caps motor current.
    """

    max_grip_force_N: float = 15.0          # 15 N << 100 N (bring-up)
    max_fraction_of_hw_max: float = 0.30    # policy: never spec above 30% of HW max
    per_joint_current_a: float = 0.6        # conservative motor-current cap
    # If True, comms must refuse to actuate at all unless a HUMAN set the token.
    require_hw_enable_token: bool = True

    def is_sane(self) -> bool:
        return (
            0.0 < self.max_grip_force_N <= self.max_fraction_of_hw_max * HW_FORCE_HARD_MAX_N
            and self.per_joint_current_a > 0.0
            and self.require_hw_enable_token is True
        )


@dataclass(frozen=True)
class WatchdogSpec:
    """Stale-input watchdog spec. On timeout, command the safe pose (spec only).

    ``safe_pose`` is the flat OPEN hand (all zeros): every active joint's lower
    limit is 0 (abductions include 0), reserved idx are 0, so it is trivially
    in-limits and is the natural fail-safe (hand opens, releases any grip).
    """

    stale_timeout_s: float = 0.20           # ~6 missed 30 Hz frames
    safe_pose: tuple = field(default_factory=lambda: tuple([0.0] * N_JOINTS))

    def safe_pose_list(self) -> list:
        return list(self.safe_pose)


@dataclass(frozen=True)
class SafetyConfig:
    """All tunables for one filter instance. Frozen -> deterministic."""

    # -- self-collision projection ------------------------------------------ #
    separation_margin_m: float = 0.0020     # extra clearance added to r_a+r_b
    pbd_iterations: int = 10                # FIXED -> real-time + deterministic
    pbd_step_scale: float = 1.0             # per-constraint correction gain
    # capsule pairs to test (the ticket's three categories):
    check_adjacent_fingers: bool = True
    check_thumb_vs_fingers: bool = True
    check_fingertip_vs_palm: bool = True

    # -- rate limiting ------------------------------------------------------- #
    max_joint_vel_rad_s: float = 8.0        # per-joint velocity cap vs prev_safe

    # -- idempotence / "modified" epsilon ----------------------------------- #
    eps_rad: float = 1e-6

    # -- specs (passed through to comms; not enforced here) ----------------- #
    force: ForceClampSpec = field(default_factory=ForceClampSpec)
    watchdog: WatchdogSpec = field(default_factory=WatchdogSpec)


DEFAULT_CONFIG = SafetyConfig()

# --- committed filter-latency baseline (test 7b regression guard) ----------- #
# p99 of filter() over the representative + adversarial mix, measured on this
# module's reference hardware (see tests/g2_safety/test_timing.py and the STATE
# handoff). Re-measure and bump deliberately when the projection changes; the
# regression test fails if live p99 exceeds BASELINE * (1 + MARGIN).
#
# HUMAN SIGN-OFF REQUESTED on FILTER_LATENCY_REGRESSION_MARGIN (see STATE.md):
# proposed 0.50 (50%) — wide enough to absorb a shared-CI machine's jitter,
# tight enough that a real algorithmic slowdown (e.g. doubling iterations)
# trips it. The absolute LOOP_PERIOD_US ceiling is the hard real-time guarantee
# independent of this margin.
# Measured best-of-3 p99 over timing_workload() on this module's dev machine:
# ~11.4 ms (p50 ~2.3 ms; absolute max ~11.9 ms << 33.3 ms ceiling, ~3x headroom).
# The tail is the deep-collision third of the workload running the full PBD
# iteration budget; the common collision-free path is ~0.85 ms.
FILTER_P99_BASELINE_US = 11_500.0
FILTER_LATENCY_REGRESSION_MARGIN = 0.50
