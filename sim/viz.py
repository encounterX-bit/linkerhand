"""Human-eyeball artifacts for G1: per-frame CSV + direction overlay PNG/GIF.

Pure PIL (no matplotlib/imageio dependency). For each frame we draw, per finger,
the human target bones (u_prox->u_dist) and the robot achieved bones
(r_prox->r_dist) chained from a shared origin, in two orthographic views:
  - back-of-hand view: Y (across fingers) horizontal, Z (finger length) up,
  - side view:         X (palm normal) horizontal, Z up.
Target = green, achieved = magenta; the thumb is highlighted. This is for human
review, not an automated assertion.
"""
from __future__ import annotations

import csv
import os

from PIL import Image, ImageDraw

from .conventions import FINGER_ORDER

_HUMAN = (40, 170, 70)     # green
_ROBOT = (200, 40, 160)    # magenta
_AXIS = (120, 120, 120)
_TXT = (30, 30, 30)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["frame", "t", "finger", "segment", "err_rad",
              "ux", "uy", "uz", "rx", "ry", "rz"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    return path


def _panel(draw, ox, oy, scale, h_axis, v_axis, record, title):
    """Draw one orthographic panel. h_axis/v_axis are component indices (0=X,1=Y,2=Z)."""
    draw.text((ox, oy - 110), title, fill=_TXT)
    for fi, finger in enumerate(FINGER_ORDER):
        cx = ox + fi * 70 + 35
        cy = oy
        draw.text((cx - 12, oy + 12), finger[:3], fill=_TXT)
        for dirs, color in ((record["human"][finger], _HUMAN),
                            (record["robot"][finger], _ROBOT)):
            u_prox, u_dist = dirs
            # chain: origin -> +prox -> +dist  (screen Y is inverted)
            p0 = (cx, cy)
            p1 = (cx + scale * u_prox[h_axis], cy - scale * u_prox[v_axis])
            p2 = (p1[0] + scale * u_dist[h_axis], p1[1] - scale * u_dist[v_axis])
            wkey = 3 if finger == "thumb" else 2
            draw.line([p0, p1], fill=color, width=wkey)
            draw.line([p1, p2], fill=color, width=wkey)
            draw.ellipse([p2[0] - 2, p2[1] - 2, p2[0] + 2, p2[1] + 2], fill=color)


def render_frame(record, frame=0, t=0.0, label=""):
    img = Image.new("RGB", (760, 300), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.text((10, 6), f"frame {frame}  t={t:.3f}  {label}", fill=_TXT)
    d.text((10, 280), "green = human target   magenta = robot achieved   (thumb bold)",
           fill=_TXT)
    _panel(d, 30, 150, 26.0, 1, 2, record, "back-of-hand  (Y across, Z up)")
    _panel(d, 410, 150, 26.0, 0, 2, record, "side  (X palm-normal, Z up)")
    return img


def render_thumb_sweep(tips, basis, out_dir, name, axis_labels):
    """Render a thumb-tip trajectory in the palm frame for human review.

    tips  : list of (forward, width, normal) tip components along the sweep.
    basis : (fwd, width, normal) unit vectors, for the legend only.
    Two panels: forward-vs-normal and width-vs-normal. A flexion sweep should hug
    the normal=0 axis (in-palm-plane curl); an abduction sweep should pull off it.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = Image.new("RGB", (640, 340), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.text((10, 6), f"thumb sweep: {name}   ({axis_labels})", fill=_TXT)
    sc = 700.0
    for px, (hx, vy, ttl) in ((40, (0, 2, "forward (toward fingers) vs palm-normal")),
                              (340, (1, 2, "width (across palm) vs palm-normal"))):
        ox, oy = px + 110, 200
        d.line([(ox - 110, oy), (ox + 110, oy)], fill=_AXIS)  # horizontal axis
        d.line([(ox, oy - 120), (ox, oy + 60)], fill=_AXIS)   # normal axis (vertical)
        d.text((px, 250), ttl, fill=_TXT)
        d.text((ox + 90, oy + 4), "h", fill=_AXIS)
        d.text((ox + 4, oy - 118), "normal", fill=_AXIS)
        pts = [(ox + sc * t[hx], oy - sc * t[vy]) for t in tips]
        if len(pts) > 1:
            d.line(pts, fill=_ROBOT, width=2)
        for q in pts:
            d.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=_HUMAN)
    img.save(os.path.join(out_dir, f"thumb_sweep_{name}.png"))
    return os.path.join(out_dir, f"thumb_sweep_{name}.png")


def render_sequence(records, out_dir, labels=None):
    """Write per-frame PNGs + an animated GIF. records: list of track_frame dicts."""
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i, rec in enumerate(records):
        lbl = labels[i] if labels else ""
        t = rec.get("targets", {}).get("t", 0.0)
        img = render_frame(rec, frame=i, t=t, label=lbl)
        img.save(os.path.join(out_dir, f"frame_{i:04d}.png"))
        frames.append(img)
    if frames:
        frames[0].save(os.path.join(out_dir, "sequence.gif"), save_all=True,
                       append_images=frames[1:], duration=120, loop=0)
    return out_dir
