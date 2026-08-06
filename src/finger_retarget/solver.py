"""Per-finger SEW retargeting solver for the Linker Hand L20 (G0).

Maps 21 MediaPipe hand landmarks -> 16 actuated L20 joint radians (reserved idx
11-14 = 0), via per-finger orientation alignment (ADR-0001), using the segment
convention of ADR-0003 (Finding-1 fingertip endpoint) and the distal-collapse of
ADR-0002.

Per finger:
  (A) BASE / PROXIMAL: align r_prox to u_prox using the base DoF (non-thumb: 2
      axes via Paden-Kahan subproblem-2; thumb: 3 CMC axes). UNCHANGED, exact.
  (B) DISTAL: align the fingertip r_dist to u_dist (ADR-0006 / ADR-0010). The
      Finding-1 fingertip is a 1-DoF CURVE in the tip command (two parallel-axis
      rotations at rates 1 and 1+ratio); because the mimic ratio is NON-INTEGER it
      traces a transcendental EPITROCHOID, NOT a circle, so there is no exact
      Paden-Kahan subproblem-1 and no finite-degree polynomial (ADR-0010 audit).
      The correct method is a FIXED-COST ANALYTIC solve: a subproblem-1 analytic
      seed + a hard-capped Newton that corrects the epitrochoid wobble to machine
      precision, with an endpoint guard. Non-thumb uses this directly. Thumb: the
      old two-plane closed form is invalid (the second plane u_dist.ktip is no
      longer constant), so the redundant base + tip are solved JOINTLY -- a fixed
      point on the distal latitude cd = (v.ktip)/|v| (base closed-form per cd),
      with a robust base grid for the near-parallel / under-actuated cases; each
      inner tip-align uses the analytic distal solve above.
All outputs are clamped to the real URDF joint ranges (baked in constants.py /
hardware/LIMITS.md). Pure function: no I/O, no hardware import, deterministic.

Performance: scalar (3-tuple) math (geometry.py) with baked constants. Non-thumb
meets the 3 kHz budget comfortably; the iterative thumb redundant-base solve does
NOT at the tail (see ADR-0007) -- the closed-form-era budget is superseded for the
fingertip distal (the distal align itself is now fixed-cost analytic, ADR-0010).
"""
from __future__ import annotations

import math

from .constants import CONSTANTS, ACTIVE_IDX, RESERVED_IDX, N_JOINTS, FINGER_ORDER
from .geometry import (
    unit, rotv, dot, cross, norm, angle_between, subproblem1, subproblem2,
)

def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# --------------------------------------------------------------------------- #
# Finding-1 distal (ADR-0006 / ADR-0010): the fingertip vector is a 1-DoF CURVE
# in the tip command -- two parallel-axis rotations of the middle bone (rate 1)
# and the distal bone (rate 1+ratio) about the common flexion axis k:
#     v(th) = R(k, th).mvec0 + R(k, (1+ratio)th).dvec0.
# Because the mimic ratio is NON-INTEGER (0.8917 / 1.1619; verified, nearest
# small fraction 33/37 / 43/37 with err ~2e-4), the two summed bones rotate at
# incommensurate rates, so v(th) traces a transcendental EPITROCHOID, not a
# circle: unit(v).k is NOT constant (verified spread up to ~8e-4 in cos, offset
# up to ~0.016). There is therefore NO exact Paden-Kahan subproblem-1 and NO
# finite-degree t=tan(th/2) polynomial for the distal align (ADR-0010, audit).
#
# The correct fixed-cost method is a FIXED-COST ANALYTIC solve (the ticket's
# sanctioned form, NOT a search): a Paden-Kahan subproblem-1 ANALYTIC SEED (the
# circle approximation the ticket suggested -- exact to first order) plus a
# hard-capped Newton on -cos(angle) that corrects the epitrochoid wobble to
# machine precision, with an endpoint guard for the under-actuated optimum. This
# matches the prior grid+Brent minimiser to ~4e-11 over 20k random dirs/finger,
# never worse, while reaching the same ~2e-8 worst-case on reachable targets.
# Thumb: the redundant cmc_pitch + tip are solved JOINTLY -- a closed-form base
# per distal latitude cd, a scalar fixed point on cd, robust grid for the
# near-parallel / under-actuated tail (each tip-align uses the analytic solve
# below). The proximal (base) solve stays an exact subproblem-2.
# --------------------------------------------------------------------------- #
_GR_B = 0.6180339887498949   # (sqrt(5)-1)/2
_GR_A = 0.3819660112501051   # 1 - _GR_B
_TIP_NEWTON = 4              # hard-capped Newton steps from the subproblem-1 seed
_THUMB_REACH_TOL = 5e-3     # distal residual above which a thumb target is treated
                            # as under-actuated -> robust nearest grid (plausible set)
_THUMB_CD_ITERS = 5         # fixed-point iterations on the distal latitude cd
_THUMB_SEED_BASE = 9        # base samples for the r_prox-unreachable nearest fallback
_THUMB_GRID_NB = 13         # base samples for the near-parallel (degenerate) grid
_THUMB_GRID_REFINE = 18     # golden steps refining the near-parallel grid winner
# u_prox nearly parallel to u_dist (thumb ~straight) makes the two-plane
# ill-conditioned (|u_prox x u_dist|^2 -> 0); route those to the robust grid.
_PARALLEL_EPS = 2e-3


def _fingertip_vec(k, mvecb, dvecb, ratio, th):
    """Fingertip vector v(th) for a base-rotated finger (ADR-0006)."""
    a = rotv(k, th, mvecb)
    b = rotv(k, (1.0 + ratio) * th, dvecb)
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _tip_solve(k, mvecb, dvecb, ratio, u_dist, lim):
    """Fixed-cost ANALYTIC distal align (ADR-0010): returns (theta, angle) that
    aligns the fingertip curve v(th) with u_dist over the tip limit.

    v(th) = R(k,th).mvecb + R(k,(1+ratio)th).dvecb is a transcendental epitrochoid
    (incommensurate rates 1 and 1+ratio), so there is no exact subproblem-1. We
    instead use subproblem-1 as the ANALYTIC SEED (rotate v(0) about k toward
    u_dist -- exact to first order, the circle the ticket suggested) and correct
    the wobble with a hard-capped Newton on g(th) = -dot(unit(v), u_dist), then
    take the best of {Newton point, lo, hi} so a clamped/under-actuated optimum at
    a bound is never missed. Deterministic, bounded -- NOT a search.

    Hot path: k, mvecb, dvecb are fixed while th varies, so the per-bone Rodrigues
    basis (m = Pm.cos + km.sin + Cm, with Cm=(k.m)k, Pm=m-Cm, km=k x m) is
    precomputed once; v, v', v'' are then 4 trig + a vector combine each (no cross
    products in the loop). The derivatives are exact, so Newton converges
    quadratically from the seed (~2e-8 residual on reachable, ~4e-11 vs the prior
    grid+Brent minimiser over 20k random dirs/finger).
    """
    lo, hi = lim

    # precompute the Rodrigues basis for m (rate 1) and d (rate r1 = 1+ratio)
    km = (k[1]*mvecb[2]-k[2]*mvecb[1], k[2]*mvecb[0]-k[0]*mvecb[2],
          k[0]*mvecb[1]-k[1]*mvecb[0])
    kdm = k[0]*mvecb[0] + k[1]*mvecb[1] + k[2]*mvecb[2]
    Cm = (kdm*k[0], kdm*k[1], kdm*k[2])
    Pm = (mvecb[0]-Cm[0], mvecb[1]-Cm[1], mvecb[2]-Cm[2])
    kd = (k[1]*dvecb[2]-k[2]*dvecb[1], k[2]*dvecb[0]-k[0]*dvecb[2],
          k[0]*dvecb[1]-k[1]*dvecb[0])
    kdd = k[0]*dvecb[0] + k[1]*dvecb[1] + k[2]*dvecb[2]
    Cd = (kdd*k[0], kdd*k[1], kdd*k[2])
    Pd = (dvecb[0]-Cd[0], dvecb[1]-Cd[1], dvecb[2]-Cd[2])
    r1 = 1.0 + ratio
    r1sq = r1 * r1
    ux, uy, uz = u_dist

    def vec(th):
        ct, st = math.cos(th), math.sin(th)
        cp, sp = math.cos(r1*th), math.sin(r1*th)
        return (Pm[0]*ct + km[0]*st + Cm[0] + Pd[0]*cp + kd[0]*sp + Cd[0],
                Pm[1]*ct + km[1]*st + Cm[1] + Pd[1]*cp + kd[1]*sp + Cd[1],
                Pm[2]*ct + km[2]*st + Cm[2] + Pd[2]*cp + kd[2]*sp + Cd[2])

    def g(th):  # -cos(angle) = -dot(unit(v), u_dist)
        v = vec(th)
        n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) or 1.0
        return -(v[0]*ux + v[1]*uy + v[2]*uz) / n

    if hi <= lo:
        return lo, angle_between(vec(lo), u_dist)

    # analytic seed: subproblem-1 (rate-1 circle through v(0)); exact to 1st order
    th = subproblem1(k, vec(0.0), u_dist)
    th = lo if th < lo else (hi if th > hi else th)

    # hard-capped Newton on g(th) using exact v, v', v''
    for _ in range(_TIP_NEWTON):
        ct, st = math.cos(th), math.sin(th)
        cp, sp = math.cos(r1*th), math.sin(r1*th)
        v = (Pm[0]*ct + km[0]*st + Cm[0] + Pd[0]*cp + kd[0]*sp + Cd[0],
             Pm[1]*ct + km[1]*st + Cm[1] + Pd[1]*cp + kd[1]*sp + Cd[1],
             Pm[2]*ct + km[2]*st + Cm[2] + Pd[2]*cp + kd[2]*sp + Cd[2])
        vp = (-Pm[0]*st + km[0]*ct - r1*Pd[0]*sp + r1*kd[0]*cp,
              -Pm[1]*st + km[1]*ct - r1*Pd[1]*sp + r1*kd[1]*cp,
              -Pm[2]*st + km[2]*ct - r1*Pd[2]*sp + r1*kd[2]*cp)
        vpp = (-Pm[0]*ct - km[0]*st - r1sq*Pd[0]*cp - r1sq*kd[0]*sp,
               -Pm[1]*ct - km[1]*st - r1sq*Pd[1]*cp - r1sq*kd[1]*sp,
               -Pm[2]*ct - km[2]*st - r1sq*Pd[2]*cp - r1sq*kd[2]*sp)
        n = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
        f = v[0]*ux + v[1]*uy + v[2]*uz            # u.v
        fp = vp[0]*ux + vp[1]*uy + vp[2]*uz        # u.v'
        fpp = vpp[0]*ux + vpp[1]*uy + vpp[2]*uz    # u.v''
        np_ = (v[0]*vp[0]+v[1]*vp[1]+v[2]*vp[2]) / n           # |v|'
        npp = ((vp[0]*vp[0]+vp[1]*vp[1]+vp[2]*vp[2]
                + v[0]*vpp[0]+v[1]*vpp[1]+v[2]*vpp[2]) - np_*np_) / n  # |v|''
        nn = n*n
        gp = -(fp*n - f*np_) / nn                  # g'  (g = -f/n)
        gpp = -(fpp*n - f*npp)/nn + 2.0*np_*(fp*n - f*np_)/(nn*n)     # g''
        if abs(gpp) < 1e-12:
            break
        thn = th - gp/gpp
        if thn < lo:
            thn = lo
        elif thn > hi:
            thn = hi
        if abs(thn - th) < 1e-13:
            th = thn
            break
        th = thn

    # endpoint guard: the constrained optimum may sit at a bound (under-actuated)
    best = th
    gb = g(th)
    for e in (lo, hi):
        ge = g(e)
        if ge < gb:
            gb, best = ge, e
    return best, angle_between(vec(best), u_dist)


# --------------------------------------------------------------------------- #
# Pre-converted constants: {side: [finger_record, ...] in FINGER_ORDER}
# --------------------------------------------------------------------------- #
def _prep():
    prep = {}
    for side, fingers in CONSTANTS.items():
        recs = []
        for name in FINGER_ORDER:
            c = fingers[name]
            a, b, _c, d = c["landmarks"]
            rec = dict(
                is_thumb=c["is_thumb"], a=a, b=b, d=d, tip_idx=c["tip_idx"],
                rprox0=unit(tuple(c["rprox0"])), rdist0=unit(tuple(c["rdist0"])),
                ktip0=unit(tuple(c["ktip0"])), tip_lim=tuple(c["tip_limit"]),
                # Finding-1 distal fingertip curve (ADR-0006): v(t) = rotv(k,t,mvec0)
                # + rotv(k,(1+ratio)t,dvec0). mvec0/dvec0 are RAW (metres), NOT unit.
                mvec0=tuple(c["mvec0"]), dvec0=tuple(c["dvec0"]),
                ratio=float(c["tip_ratio"]),
            )
            ax = c["base_axes"]
            if c["is_thumb"]:
                rec["i_opp"], rec["s"], rec["lim_opp"] = ax[0][0], unit(tuple(ax[0][1])), tuple(ax[0][2])
                rec["i_abd"], rec["k_roll"], rec["lim_abd"] = ax[1][0], unit(tuple(ax[1][1])), tuple(ax[1][2])
                # actual cmc_pitch axis (≈ but not == the cmc_yaw axis s): used for
                # the high-precision distal base rotation (the proximal solve keeps
                # the historical `s` convention; see _solve_thumb).
                rec["i_base"], rec["s_base"], rec["lim_base"] = ax[2][0], unit(tuple(ax[2][1])), tuple(ax[2][2])
            else:
                rec["i_abd"], rec["k_abd"], rec["lim_abd"] = ax[0][0], unit(tuple(ax[0][1])), tuple(ax[0][2])
                rec["i_base"], rec["k_base"], rec["lim_base"] = ax[1][0], unit(tuple(ax[1][1])), tuple(ax[1][2])
            recs.append(rec)
        prep[side] = recs
    return prep


_PREP = _prep()


# --------------------------------------------------------------------------- #
# non-thumb base: 2-axis subproblem-2 (+ nearest fallback)
# --------------------------------------------------------------------------- #
def _two_axis_base(k_out, k_in, p, q, lim_out, lim_in):
    best = None
    for (t_out, t_in) in subproblem2(k_out, k_in, p, q):
        co = _clamp(t_out, *lim_out)
        ci = _clamp(t_in, *lim_in)
        err = angle_between(rotv(k_out, co, rotv(k_in, ci, p)), q)
        if best is None or err < best[0]:
            best = (err, co, ci)
    if best is not None and best[0] < 1e-9:
        return best[1], best[2]
    t_out = best[1] if best else 0.0
    t_in = best[2] if best else 0.0
    for _ in range(4):  # bounded closed-form coordinate steps -> nearest reachable
        t_in = _clamp(subproblem1(k_in, p, rotv(k_out, -t_out, q)), *lim_in)
        t_out = _clamp(subproblem1(k_out, rotv(k_in, t_in, p), q), *lim_out)
    return t_out, t_in


# --------------------------------------------------------------------------- #
# thumb: 4-DoF solve. r_prox is aligned by the 3 CMC DoF, leaving the redundant
# cmc_pitch (base); the fingertip r_dist is aligned by the tip DoF. The two
# directions couple through the redundant base, so we solve them JOINTLY: the tip
# axis ktip is fixed in CLOSED FORM by two planes (u_prox.ktip known, u_dist.ktip
# = cd) -> the 3 CMC DoF follow by subproblem-2 + subproblem-1, leaving ONE scalar
# unknown, the distal latitude cd = vk/|v(theta*)|, closed by a short fixed point
# (base is closed-form per cd, no grid in the common case); the inner tip-align is
# the fixed-cost analytic distal solve. A robust base grid handles the near-
# parallel / under-actuated tail. The old two-plane closed form (ADR-0004) assumed
# r_dist was a pure rotation of one vector about the tip axis; the Finding-1
# fingertip curve breaks that (vk const but |v| varies), hence the cd fixed point.
# t=tan(theta/2) on the distal align would be intractable (non-integer ratio ->
# degree ~O(100), and only approximate) -- see ADR-0010.
# --------------------------------------------------------------------------- #
def _solve_thumb(rec, u_prox, u_dist, w_prox, w_dist):
    s = rec["s"]               # cmc_yaw (opposition) axis, zero pose
    k_roll = rec["k_roll"]     # cmc_roll (abduction) axis, zero pose
    s_base = rec["s_base"]     # cmc_pitch (base) axis, zero pose
    p = rec["rprox0"]
    mvec0, dvec0, ratio = rec["mvec0"], rec["dvec0"], rec["ratio"]
    ktip0 = rec["ktip0"]
    lim_opp, lim_abd, lim_base, lim_tip = (rec["lim_opp"], rec["lim_abd"],
                                           rec["lim_base"], rec["tip_lim"])
    i_opp, i_abd, i_base, i_tip = (rec["i_opp"], rec["i_abd"],
                                   rec["i_base"], rec["tip_idx"])

    # Two invariants under the (unknown) thumb base transform R: dot is preserved,
    #   r_prox . ktip == rprox0 . ktip0 =: cp           (r_prox aligned to u_prox)
    #   v(theta) . ktip == (mvec0+dvec0) . ktip0 =: vk  (fingertip k-component, const)
    # At alignment u_prox==r_prox and u_dist==unit(v), so the tip axis ktip obeys
    #   u_prox . ktip = cp        (known)
    #   u_dist . ktip = vk / |v(theta*)| =: cd  (depends on the tip solution)
    # Two planes fix ktip (closed form) -> the 3 CMC DoF follow in closed form; the
    # ONLY unknown is |v(theta*)|, which we close by a fixed point on cd. This makes
    # the redundant base CLOSED-FORM at each cd (no grid -> fast, no narrow basin).
    cp = dot(p, ktip0)
    vk = dot(mvec0, ktip0) + dot(dvec0, ktip0)
    n12 = dot(u_prox, u_dist)
    cr = cross(u_prox, u_dist)
    cn2 = cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]
    det2 = 1.0 - n12 * n12

    if cn2 < _PARALLEL_EPS or abs(det2) < _PARALLEL_EPS:
        # u_prox ~parallel u_dist (thumb ~straight): two-plane is ill-conditioned,
        # use the robust base grid instead. Rare -> negligible timing impact.
        return _thumb_grid(rec, u_prox, u_dist, w_prox, w_dist)

    def cfgs_for_cd(cd):
        """Closed-form CMC configs (opp, abd, base) whose tip axis satisfies the two
        planes for this cd; each tagged (sign, branch) for locking across iters."""
        out = []
        if cn2 < 1e-12 or abs(det2) < 1e-12:
            return out
        a = (cp - cd * n12) / det2
        b = (cd - cp * n12) / det2
        gsq = (1.0 - a*a - b*b - 2.0*a*b*n12) / cn2
        if gsq < -1e-9:
            return out
        g = math.sqrt(gsq) if gsq > 0.0 else 0.0
        for si, gg in enumerate((g, -g) if g > 1e-9 else (0.0,)):
            ktip = (a*u_prox[0] + b*u_dist[0] + gg*cr[0],
                    a*u_prox[1] + b*u_dist[1] + gg*cr[1],
                    a*u_prox[2] + b*u_dist[2] + gg*cr[2])
            for bi, (opp, abd) in enumerate(subproblem2(s, k_roll, ktip0, ktip)):
                oc = _clamp(opp, *lim_opp)
                ac = _clamp(abd, *lim_abd)
                gtup = rotv(k_roll, -ac, rotv(s, -oc, u_prox))
                base = _clamp(subproblem1(s_base, p, gtup), *lim_base)
                out.append((si, bi, oc, ac, base))
        return out

    def evaluate(oc, ac, base):
        """(J, |v|, cfg, ed) for a CMC config: 1-D fingertip tip solve + residuals."""
        Rb = lambda v: rotv(s, oc, rotv(k_roll, ac, rotv(s_base, base, v)))
        kb, mb, db = Rb(ktip0), Rb(mvec0), Rb(dvec0)
        tip, ed = _tip_solve(kb, mb, db, ratio, u_dist, lim_tip)
        nv = norm(_fingertip_vec(kb, mb, db, ratio, tip))
        ep = angle_between(Rb(p), u_prox)
        return (w_prox * ep + w_dist * ed, nv,
                {i_opp: oc, i_abd: ac, i_base: base, i_tip: tip}, ed)

    best = None  # (J, cfg, ed)
    lock = None  # (sign, branch) chosen after the first iteration
    v0 = norm((mvec0[0]+dvec0[0], mvec0[1]+dvec0[1], mvec0[2]+dvec0[2]))
    cd = _clamp(vk / v0, -1.0, 1.0) if v0 > 1e-9 else 0.0
    for _ in range(_THUMB_CD_ITERS):
        cands = cfgs_for_cd(cd)
        if not cands:
            break
        if lock is not None:
            locked = [c for c in cands if (c[0], c[1]) == lock]
            if locked:
                cands = locked
        evs = [(evaluate(c[2], c[3], c[4]), c) for c in cands]
        (J, nv, cfg, ed), c = min(evs, key=lambda e: e[0][0])
        if best is None or J < best[0]:
            best = (J, cfg, ed)
        lock = (c[0], c[1])
        if nv < 1e-9:
            break
        cd_new = _clamp(vk / nv, -1.0, 1.0)
        if abs(cd_new - cd) < 1e-12:
            break
        cd = cd_new

    # Route to the robust grid whenever the fixed point did not reach a small TOTAL
    # residual (prox + dist). This catches both under-actuated targets and the
    # cases where the cd fixed point converged to a wrong basin / clamped base
    # (good distal but misaligned prox). Reachable, well-conditioned targets
    # converge here, so the grid only runs on the residual tail.
    if best is None or best[0] > _THUMB_REACH_TOL:
        grid = _thumb_grid(rec, u_prox, u_dist, w_prox, w_dist)
        if best is None:
            return grid
        if _thumb_J(rec, u_prox, u_dist, w_prox, w_dist, grid) < best[0]:
            return grid
    return best[1]


def _thumb_J(rec, u_prox, u_dist, w_prox, w_dist, cfg):
    """Objective J = w_prox*angle(r_prox,u_prox) + w_dist*angle(r_dist,u_dist) for a
    full thumb cfg dict (to compare the fixed-point vs grid solutions)."""
    s, k_roll, s_base = rec["s"], rec["k_roll"], rec["s_base"]
    p, mvec0, dvec0, ratio, ktip0 = (rec["rprox0"], rec["mvec0"], rec["dvec0"],
                                     rec["ratio"], rec["ktip0"])
    oc, ac = cfg[rec["i_opp"]], cfg[rec["i_abd"]]
    bc, tip = cfg[rec["i_base"]], cfg[rec["tip_idx"]]
    Rb = lambda v: rotv(s, oc, rotv(k_roll, ac, rotv(s_base, bc, v)))
    ep = angle_between(Rb(p), u_prox)
    ed = angle_between(_fingertip_vec(Rb(ktip0), Rb(mvec0), Rb(dvec0), ratio, tip),
                       u_dist)
    return w_prox * ep + w_dist * ed


def _thumb_grid(rec, u_prox, u_dist, w_prox, w_dist):
    """Robust thumb solve for the near-parallel / ill-conditioned case: grid the
    redundant base (cmc_pitch), align r_prox by EXACT subproblem-2 per branch, tip
    by the 1-D fingertip minimiser; locate the global best over both branches then
    golden-refine the winner within +/-1 cell."""
    s, k_roll, s_base = rec["s"], rec["k_roll"], rec["s_base"]
    p, mvec0, dvec0, ratio, ktip0 = (rec["rprox0"], rec["mvec0"], rec["dvec0"],
                                     rec["ratio"], rec["ktip0"])
    lim_opp, lim_abd, lim_base, lim_tip = (rec["lim_opp"], rec["lim_abd"],
                                           rec["lim_base"], rec["tip_lim"])
    i_opp, i_abd, i_base, i_tip = (rec["i_opp"], rec["i_abd"],
                                   rec["i_base"], rec["tip_idx"])
    lo, hi = lim_base

    def cfg_at(bc, br):
        w = rotv(s_base, bc, p)
        sols = subproblem2(s, k_roll, w, u_prox)
        if br >= len(sols):
            return None
        oc = _clamp(sols[br][0], *lim_opp)
        ac = _clamp(sols[br][1], *lim_abd)
        Rb = lambda v: rotv(s, oc, rotv(k_roll, ac, rotv(s_base, bc, v)))
        tip, ed = _tip_solve(Rb(ktip0), Rb(mvec0), Rb(dvec0), ratio, u_dist, lim_tip)
        ep = angle_between(rotv(s, oc, rotv(k_roll, ac, w)), u_prox)
        return w_prox * ep + w_dist * ed, {i_opp: oc, i_abd: ac,
                                           i_base: bc, i_tip: tip}

    n = _THUMB_GRID_NB
    best = None
    base_star = br = None
    for b in (0, 1):
        for i in range(n):
            bc = lo + (hi - lo) * i / (n - 1)
            r = cfg_at(bc, b)
            if r is not None and (best is None or r[0] < best[0]):
                best, base_star, br = r, bc, b
    if best is None:
        return _thumb_nearest(rec, u_prox, u_dist, w_prox, w_dist)
    cell = (hi - lo) / (n - 1)
    a = max(lo, base_star - cell)
    c = min(hi, base_star + cell)
    x1 = a + _GR_A * (c - a)
    x2 = a + _GR_B * (c - a)
    r1, r2 = cfg_at(x1, br), cfg_at(x2, br)
    for _ in range(_THUMB_GRID_REFINE):
        j1 = r1[0] if r1 else float("inf")
        j2 = r2[0] if r2 else float("inf")
        if j1 < j2:
            c, x2, r2 = x2, x1, r1
            x1 = a + _GR_A * (c - a)
            r1 = cfg_at(x1, br)
        else:
            a, x1, r1 = x1, x2, r2
            x2 = a + _GR_B * (c - a)
            r2 = cfg_at(x2, br)
    return min((r for r in (best, r1, r2) if r is not None), key=lambda r: r[0])[1]


def _thumb_nearest(rec, u_prox, u_dist, w_prox, w_dist):
    """Fallback when no subproblem-2 branch aligns r_prox at any seeded base
    (r_prox unreachable): nearest 2-axis proximal + 1-D tip, gridded over base."""
    s, k_roll, s_base = rec["s"], rec["k_roll"], rec["s_base"]
    p, mvec0, dvec0, ratio, ktip0 = (rec["rprox0"], rec["mvec0"], rec["dvec0"],
                                     rec["ratio"], rec["ktip0"])
    lim_opp, lim_abd, lim_base, lim_tip = (rec["lim_opp"], rec["lim_abd"],
                                           rec["lim_base"], rec["tip_lim"])
    i_opp, i_abd, i_base, i_tip = (rec["i_opp"], rec["i_abd"],
                                   rec["i_base"], rec["tip_idx"])
    lo, hi = lim_base
    best = None
    for i in range(_THUMB_SEED_BASE):
        bc = lo + (hi - lo) * i / (_THUMB_SEED_BASE - 1)
        w = rotv(s_base, bc, p)
        ac = _clamp(subproblem1(k_roll, w, u_prox), *lim_abd)
        oc = _clamp(subproblem1(s, rotv(k_roll, ac, w), u_prox), *lim_opp)
        ep = angle_between(rotv(s, oc, rotv(k_roll, ac, w)), u_prox)
        Rb = lambda v: rotv(s, oc, rotv(k_roll, ac, rotv(s_base, bc, v)))
        tip, ed = _tip_solve(Rb(ktip0), Rb(mvec0), Rb(dvec0), ratio, u_dist, lim_tip)
        J = w_prox * ep + w_dist * ed
        if best is None or J < best[0]:
            best = (J, {i_opp: oc, i_abd: ac, i_base: bc, i_tip: tip})
    return best[1]


# Direction-canonicalisation grid (radians). The distal solve is iterative, so a
# 1-ULP change in u_prox/u_dist (e.g. landmarks scaled by a non-power-of-2 factor)
# could flip a discrete branch/argmin choice and shift the output well above the
# 1e-9 scale-invariance bound. Snapping the unit target directions to this grid
# makes the solver INPUT identical across scales -> bit-exact scale invariance.
# The snap (~1e-12 rad) is ~5 orders below the solve precision (~1e-7), so it does
# not affect accuracy. (The old closed-form thumb was scale-exact algebraically;
# the fingertip distal has no closed form, hence this input canonicalisation.)
_SNAP = 1e12


def _snap(v):
    return (round(v[0] * _SNAP) / _SNAP,
            round(v[1] * _SNAP) / _SNAP,
            round(v[2] * _SNAP) / _SNAP)


def _solve_finger(rec, lm, w_prox, w_dist):
    a, b, d = rec["a"], rec["b"], rec["d"]
    u_prox = _snap(unit(_sub3(lm[b], lm[a])))
    u_dist = _snap(unit(_sub3(lm[d], lm[b])))

    if rec["is_thumb"]:
        return _solve_thumb(rec, u_prox, u_dist, w_prox, w_dist)

    k_abd, k_base = rec["k_abd"], rec["k_base"]
    # PROXIMAL (untouched): align r_prox with the 2 base axes (subproblem-2).
    t_abd, t_base = _two_axis_base(k_abd, k_base, rec["rprox0"], u_prox,
                                   rec["lim_abd"], rec["lim_base"])
    # DISTAL (Finding-1): base-rotate the tip axis + fingertip-curve vectors, then
    # the fixed-cost analytic distal solve aligns the fingertip r_dist with u_dist.
    kb = rotv(k_abd, t_abd, rotv(k_base, t_base, rec["ktip0"]))
    mb = rotv(k_abd, t_abd, rotv(k_base, t_base, rec["mvec0"]))
    db = rotv(k_abd, t_abd, rotv(k_base, t_base, rec["dvec0"]))
    tip, _ = _tip_solve(kb, mb, db, rec["ratio"], u_dist, rec["tip_lim"])
    return {rec["i_abd"]: t_abd, rec["i_base"]: t_base, rec["tip_idx"]: tip}


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def retarget(landmarks, side="right", t=0.0, weights=(1.0, 1.0)):
    """Map hand_landmarks -> l20_targets (closed-form, clamped).

    Parameters
    ----------
    landmarks : (21, 3) sequence in the hand_base frame (MediaPipe order).
    side      : 'left' or 'right'.
    weights   : (w_prox, w_dist) per-finger objective weights (default 1.0/1.0).

    Returns an l20_targets dict: {side, joint_rad[20] (idx 11-14 = 0.0),
    active_idx, clamped=True, t}.
    """
    if side not in _PREP:
        raise ValueError(f"side must be 'left'/'right', got {side!r}")
    lm = [(float(r[0]), float(r[1]), float(r[2])) for r in landmarks]
    if len(lm) != 21:
        raise ValueError(f"landmarks must have 21 points, got {len(lm)}")
    for r in lm:
        if not (math.isfinite(r[0]) and math.isfinite(r[1]) and math.isfinite(r[2])):
            raise ValueError("landmarks contain non-finite values")
    w_prox, w_dist = weights

    joint_rad = [0.0] * N_JOINTS
    for rec in _PREP[side]:
        for idx, val in _solve_finger(rec, lm, w_prox, w_dist).items():
            joint_rad[idx] = float(val)
    for idx in RESERVED_IDX:
        joint_rad[idx] = 0.0  # explicit: reserved always zero

    return {
        "side": side,
        "joint_rad": joint_rad,
        "active_idx": list(ACTIVE_IDX),
        "clamped": True,
        "t": float(t),
    }
