# Review: G2 safety filter — conservativeness & per-call allocation

**Date:** 2026-06-09 · **Scope:** review-only, `src/safety/` unmodified · **Side:** right
**Reviewed:** `src/safety/filter.py`, `src/safety/collision_model.py`, `src/safety/config.py`

Two checks on the merged safety filter: (1) is the baked capsule/palm proxy
*conservative* w.r.t. the real collision meshes, and (2) does the
collision-resolution path allocate heap per call. **No source was changed.**

---

## 1. Conservativeness — capsule/palm proxy vs. full collision mesh

### Method
- Sampled **5000** uniformly random in-limits L20 configs (reserved idx 11–14 = 0),
  RNG seed `20260609`, using the suite's own `helpers.rand_in_limits`.
- **Proxy verdict:** `CollisionModel.penetrations(q, margin)` — *collision-free* when
  the list is empty. Evaluated at `margin = 0` (pure capsule geometry) and at
  `margin = 0.002 m` (the shipped `separation_margin_m`, i.e. the filter's actual
  "leave it alone" condition).
- **Ground truth:** the 21 URDF collision STLs placed at the **same `src/kinematics`
  FK** link transforms (all `<collision>` origins are identity), full triangle-mesh
  collision via FCL (`python-fcl` through `trimesh.collision.CollisionManager`,
  penetration depth from contact data). Pairs joined by a URDF joint are excluded
  (they touch by design); at the neutral pose **0** non-jointed pairs collide, so the
  joint-adjacency set is the only exclusion.
- A **false negative** = proxy says collision-free, meshes interpenetrate.

### Results — false-negative counts (of 5000)

| proxy verdict | mesh >0.0 mm | >0.2 mm | >0.5 mm | >1.0 mm |
|---|---|---|---|---|
| free @ `margin 0`        | 86 | 80 | 67 | 39 |
| free @ `margin 0.002 m` (shipped) | 33 | 31 | **27** | 19 |

For reference: proxy flags collision in 2221 (`m0`) / 2932 (`m2mm`) configs; meshes
collide in 1873 (any) / 1757 (>0.5 mm). So at the shipped 2 mm margin the proxy
misses **~0.5–0.7 %** of configs (27–33 / 5000), with ~19 showing >1 mm
interpenetration.

### Where the misses are (composition of the 30 offending pairs across the 27
shipped-margin / >0.5 mm false-negative configs)

| count | pair | proxy scope |
|---|---|---|
| **21** | `base_link` ↔ `thumb_metacarpals` | OUT — palm check is fingertip-only |
| 3 | `thumb_metacarpals` ↔ `thumb_metacarpals_base1` | OUT — CMC link stack not modeled |
| 2 | `*_middle` ↔ `thumb_distal` | thumb-finger (in scope) |
| 2 | `*_proximal` ↔ `thumb_distal` | thumb-finger (in scope) |
| 1 | `*_distal` ↔ `thumb_distal` | thumb-finger (in scope) |
| 1 | `*_distal` ↔ `thumb_proximal` | thumb-finger (in scope) |

Scope tally: **thumb-finger (in scope) = 6**, **OUT-OF-SCOPE = 24**.
**Adjacent-finger (index/middle/ring/little) false negatives: 0** — the inter-finger
capsules are conservative.

Deepest misses (config idx · pair · depth):

```
cfg#4388  base_link <-> thumb_metacarpals          5.30 mm
cfg#3438  base_link <-> thumb_metacarpals          4.85 mm
cfg#4122  base_link <-> thumb_metacarpals          4.85 mm
cfg#3572  base_link <-> thumb_metacarpals          4.82 mm
cfg#1460  base_link <-> thumb_metacarpals          4.69 mm
cfg# 335  base_link <-> thumb_metacarpals          3.44 mm
cfg#4134  thumb_metacarpals <-> thumb_metacarpals_base1  2.24 mm
cfg#1231  pinky_proximal <-> thumb_distal          1.89 mm
```

### Interpretation
- The false negatives are **not** broad random capsule under-enclosure. They cluster
  on the **thumb base against the palm body**: 21/30 pairs (and every miss deeper than
  ~2 mm, up to **5.3 mm**) are `thumb_metacarpals` driven into `base_link`. This is a
  **structural coverage gap, not a tuning error**: ADR-0008 deliberately models the
  palm as a *fingertip-vs-palmar-half-plane*. The thumb's metacarpal/proximal links
  pressing into the palm body are not in any checked category, so the proxy cannot see
  them regardless of margin. This is the one finding worth a human decision — the
  thumb-into-palm event ADR-0008 names as "essentially a thumb-into-palm event" is
  caught only at the *fingertip*, not for the thumb base.
- 3 pairs are the thumb CMC multi-link stack interpenetrating itself
  (`thumb_metacarpals` vs `thumb_metacarpals_base1`); the proxy collapses the CMC to a
  single capsule, so this is invisible to it. Low teleop-risk (internal mechanism), but
  a real mesh overlap.
- The 6 **in-scope** misses (thumb_distal vs finger phalanges) are genuine
  capsule-approximation false negatives, but all shallow (≤1.9 mm) — consistent with
  the radius being *half the smallest* bounding-box extent, which under-encloses the
  bone in its two wider dimensions near the knuckles. The shipped 2 mm margin absorbs
  most of these (27 remain at >0.5 mm vs 67 at margin 0).

---

## 2. Per-call heap allocation — collision-resolution path

### Method
`tracemalloc`, `gc` **disabled**, 200–500 warm-up calls. Per-call transient is the
peak live-bytes growth *within a single call* (`reset_peak` per call); "net" is bytes
still live after thousands of calls (leak/cache-growth check). Deep-collision
candidate (`min_depth = 0.004`) with `prev_safe` present and `dt = 1/30 s`, so the
full PBD projection (`_project_collision`) and the rate-limit re-projection branch run.

### Results

| path | mean | median | min | max | net |
|---|---|---|---|---|---|
| `filter()` deep-collision (resolution path) | 27.9 KB | 27.9 KB | 27.9 KB | 39.4 KB | ≈0 (+2.8 B/call) |
| `filter()` collision-free (fast path) | 10.8 KB | 10.8 KB | 10.8 KB | 11.0 KB | ≈0 (+0.2 B/call) |
| `model.penetrations()` deep-collision | 15.7 KB | 15.7 KB | 15.7 KB | 16.2 KB | ≈0 (+0.1 B/call) |

Top allocation sites (one deep-collision `filter()` call):

```
22128 B  461 blk  numpy/_core/numeric.py   (np.cross / np.zeros in the Jacobian loop)
 4704 B   42 blk  trimesh/scene/transforms.py  (yourdfpy-backed FK link transforms)
 ~3300 B          numpy/linalg/_linalg.py  (np.linalg.norm in seg-seg / palm dist)
```

### Interpretation
- **Yes, it allocates on every call**, and the amount scales with collision depth:
  ~**11 KB** on the common collision-free path, ~**28 KB** (up to ~39 KB) on the
  deep-collision resolution path. The dominant source is `CollisionModel.penetrations`
  rebuilding per-call NumPy temporaries — `frames` dict over all 22 links, `J =
  np.zeros((20,3))` and an `np.cross` per Jacobian contributor, `np.linalg.norm` per
  capsule pair — multiplied across up to `pbd_iterations = 10` re-evaluations. The
  FK itself (`yourdfpy`/`trimesh` scene transforms) adds the rest.
- **No leak / no per-call cache growth:** net retained ≈ 0 (a few bytes per call,
  tracemalloc noise) over thousands of calls. Allocation is purely transient.
- This is not a correctness concern and the committed p99 (~11.4 ms, see
  `config.py`) sits well under the 33.3 ms loop budget. It is flagged only as the
  obvious lever if the path is ever moved to a hard-real-time / GC-sensitive context:
  the per-call temporaries (Jacobian buffers, `frames`, `dq`) are pre-allocatable.

---

## Caveats
- `python-fcl` was `pip install`ed into the venv for the mesh ground truth
  (dev/measurement dependency only — **no repo source changed**; the filter remains
  mesh-free and sim-free as designed).
- FCL penetration depth is the minimum-translation distance; "collide" thresholds of
  0.0/0.2/0.5/1.0 mm are reported so the conclusion does not hinge on grazing contacts.
- Right hand only; geometry is mirror-symmetric, so left-hand behavior is expected to
  match. Findings are over uniform-random in-limits sampling, which over-weights
  extreme thumb poses relative to teleop traffic.
- Scratch scripts: `/tmp/conservativeness.py`, `/tmp/cons_detail.py`, `/tmp/alloc.py`,
  `/tmp/mesh_common.py` (not committed).
