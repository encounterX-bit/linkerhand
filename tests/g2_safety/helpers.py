"""Shared, deterministic helpers for the G2 safety suite.

Importable as a top-level module because tests/g2_safety/conftest.py puts this
directory on sys.path (matching the repo's no-test-package convention). The
collision model used here is the SAME one the filter projects with, so
"collision-free output" is asserted under the model the filter owns.
"""
import numpy as np

from src.safety.collision_model import CollisionModel

SIDES = ("right", "left")


def active_limits(model: CollisionModel):
    return model.fk.active_limits()


def rand_in_limits(model: CollisionModel, rng) -> list:
    """A uniformly random in-limits 20-vector (reserved idx 0)."""
    jr = [0.0] * 20
    for idx, (lo, hi) in active_limits(model).items():
        jr[idx] = float(rng.uniform(lo, hi))
    return jr


def sample_colliding(model: CollisionModel, rng, n, min_depth=0.002, tries=8000):
    """``n`` in-limits configs whose max penetration (margin 0) exceeds
    ``min_depth`` — genuinely interpenetrating self-collision scenarios."""
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        c = rand_in_limits(model, rng)
        if model.max_penetration(c, 0.0) >= min_depth:
            out.append(c)
    return out


def sample_safe(model: CollisionModel, rng, n, margin, tries=8000):
    """``n`` in-limits configs already collision-free at ``margin``."""
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        c = rand_in_limits(model, rng)
        if model.max_penetration(c, margin) <= 0.0:
            out.append(c)
    return out


def timing_workload(model: CollisionModel, n=150):
    """FIXED (seed 1234) representative+adversarial latency workload: ~⅓ deep
    collisions (projection runs its full iteration budget) + ~⅔ random in-limits
    configs (mostly the cheap collision-free fast path). The SAME list commits
    the baseline and re-measures, so it is reproducible."""
    rng = np.random.default_rng(1234)
    n_col = n // 3
    cols = sample_colliding(model, rng, n_col, min_depth=0.003)
    rest = [rand_in_limits(model, rng) for _ in range(n - len(cols))]
    return cols + rest
