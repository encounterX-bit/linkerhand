"""Closed-form rotation geometry for the per-finger solver (gate G0).

Scalar implementation: 3-vectors are plain ``(x, y, z)`` tuples and arithmetic
uses the ``math`` module, NOT numpy. For size-3 vectors this is ~5-10x faster
than numpy (whose per-call dispatch overhead dominates at this size), which is
what lets the full-hand solve meet the 3 kHz budget. Pure, deterministic, no I/O.

Direction-only (vectors through the origin), so the Paden-Kahan subproblems
operate on unit vectors.
"""
from __future__ import annotations

import math

_EPS = 1e-9


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add3(a, b, c):
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def unit(v):
    v = (float(v[0]), float(v[1]), float(v[2]))
    n = norm(v)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def rotv(k, theta, v):
    """Rotate vector v about unit axis k by theta (Rodrigues, vector form)."""
    c = math.cos(theta)
    s = math.sin(theta)
    kv = (k[0] * v[0] + k[1] * v[1] + k[2] * v[2]) * (1.0 - c)
    cx = k[1] * v[2] - k[2] * v[1]
    cy = k[2] * v[0] - k[0] * v[2]
    cz = k[0] * v[1] - k[1] * v[0]
    return (v[0] * c + cx * s + k[0] * kv,
            v[1] * c + cy * s + k[1] * kv,
            v[2] * c + cz * s + k[2] * kv)


def angle_between(a, b):
    na = norm(a)
    nb = norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = dot(a, b) / (na * nb)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return math.acos(c)


def subproblem1(axis, p, q):
    """Angle theta best rotating p about unit `axis` toward q (1-DoF minimiser).

    Aligns the components of p and q perpendicular to the axis (projection onto a
    circle). Returns theta in (-pi, pi]; 0 if either projection vanishes.
    """
    k = axis
    pk = dot(p, k)
    qk = dot(q, k)
    pp = (p[0] - pk * k[0], p[1] - pk * k[1], p[2] - pk * k[2])
    qp = (q[0] - qk * k[0], q[1] - qk * k[1], q[2] - qk * k[2])
    if norm(pp) < _EPS or norm(qp) < _EPS:
        return 0.0
    cr = cross(pp, qp)
    s = k[0] * cr[0] + k[1] * cr[1] + k[2] * cr[2]   # sin * |pp||qp|
    c = pp[0] * qp[0] + pp[1] * qp[1] + pp[2] * qp[2]  # cos * |pp||qp|
    return math.atan2(s, c)


def subproblem2(axis1, axis2, p, q):
    """Solve rot(axis1,t1) rot(axis2,t2) p = q for unit p, q (Paden-Kahan 2).

    axes through the origin, acting on directions. Returns a list of (t1, t2)
    solutions (0, 1 or 2). Empty when q is not exactly reachable (cones miss).
    """
    k1 = axis1
    k2 = axis2
    k1k2 = dot(k1, k2)
    denom = k1k2 * k1k2 - 1.0
    if abs(denom) < _EPS:
        return []
    k1q = dot(k1, q)
    k2p = dot(k2, p)
    alpha = (k1k2 * k2p - k1q) / denom
    beta = (k1k2 * k1q - k2p) / denom
    cr = cross(k1, k2)
    cn2 = cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]
    if cn2 < _EPS:
        return []
    gamma_sq = (1.0 - alpha * alpha - beta * beta - 2.0 * alpha * beta * k1k2) / cn2
    if gamma_sq < -1e-9:
        return []
    gamma = math.sqrt(gamma_sq) if gamma_sq > 0.0 else 0.0
    gammas = (gamma, -gamma) if gamma > _EPS else (0.0,)
    sols = []
    for g in gammas:
        z = (alpha * k1[0] + beta * k2[0] + g * cr[0],
             alpha * k1[1] + beta * k2[1] + g * cr[1],
             alpha * k1[2] + beta * k2[2] + g * cr[2])
        t2 = subproblem1(k2, p, z)
        t1 = subproblem1(k1, z, q)
        sols.append((t1, t2))
    return sols
