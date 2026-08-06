#!/usr/bin/env python3
"""Offline G20 SDK-range waypoint editor with optional PyBullet preview.

The editor never imports ROS and cannot actuate hardware. Save two or more
waypoints, assign the travel duration for each waypoint after the first, then
export the JSON consumed by ``record_action_primitive``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.comms.action_library import (
    ACTIVE_IDX,
    G20_OPEN_POSE,
    RESERVED_IDX,
    g20_range_to_sim_radians,
    interpolate_waypoints,
)


JOINT_NAMES = {
    0: "thumb_base", 1: "index_base", 2: "middle_base", 3: "ring_base", 4: "little_base",
    5: "thumb_side", 6: "index_spread", 7: "middle_spread", 8: "ring_spread", 9: "little_spread",
    10: "thumb_roll", 15: "thumb_tip", 16: "index_tip", 17: "middle_tip", 18: "ring_tip", 19: "little_tip",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("g20_waypoints.json"))
    parser.add_argument("--load", type=Path, help="load an existing waypoint JSON")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--show-sim", action="store_true", help="show approximate L20 URDF preview")
    parser.add_argument("--roll-range-ticks", type=float, default=100.0)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args(argv)


class WaypointEditor:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.args = args
        self.root = tk.Tk()
        self.root.title("G20 waypoint editor (offline)")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.variables: dict[int, tk.IntVar] = {}
        self.waypoints: list[dict] = []
        self.sim = None

        controls = tk.Frame(self.root)
        controls.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        for row, index in enumerate(ACTIVE_IDX):
            tk.Label(controls, text=f"q{index:02d} {JOINT_NAMES[index]}", width=19, anchor="w").grid(row=row, column=0)
            variable = tk.IntVar(value=int(G20_OPEN_POSE[index]))
            self.variables[index] = variable
            tk.Scale(
                controls, from_=0, to=255, orient=tk.HORIZONTAL, length=330,
                variable=variable, command=lambda _value: self._preview(),
            ).grid(row=row, column=1)

        actions = tk.Frame(self.root)
        actions.grid(row=0, column=1, padx=8, pady=8, sticky="ns")
        tk.Label(actions, text="Travel duration to this waypoint (s)").pack(anchor="w")
        self.duration = tk.StringVar(value="0.50")
        tk.Entry(actions, textvariable=self.duration, width=12).pack(anchor="w", pady=(0, 8))
        tk.Button(actions, text="Save waypoint", command=self.save_waypoint, width=22).pack(pady=2)
        tk.Button(actions, text="Replace selected", command=self.replace_selected, width=22).pack(pady=2)
        tk.Button(actions, text="Load selected pose", command=self.load_selected_pose, width=22).pack(pady=2)
        tk.Button(actions, text="Delete selected", command=self.delete_selected, width=22).pack(pady=2)
        tk.Button(actions, text="Export JSON", command=self.export, width=22).pack(pady=(12, 2))
        tk.Button(actions, text="Reset sliders to OPEN", command=self.reset_open, width=22).pack(pady=2)
        self.listbox = tk.Listbox(actions, width=42, height=18)
        self.listbox.pack(pady=(12, 0), fill="both", expand=True)
        self.status = tk.StringVar(value="Offline only: no ROS, no hardware")
        tk.Label(actions, textvariable=self.status, fg="#9a4f00", wraplength=300, justify="left").pack(pady=8)

        if args.load:
            self.waypoints = json.loads(args.load.read_text(encoding="utf-8"))
            interpolate_waypoints(self.waypoints, fps=args.fps)  # validate before showing
            self._refresh_list()
            if self.waypoints:
                self._set_pose(self.waypoints[0]["pose"])
        if args.show_sim:
            from src.viz.render import L20VizModel
            self.sim = L20VizModel(args.side, gui=True)
            self._preview()

    def current_pose(self) -> list[int]:
        pose = [int(value) for value in G20_OPEN_POSE]
        for index, variable in self.variables.items():
            pose[index] = int(variable.get())
        for index in RESERVED_IDX:
            pose[index] = 255
        return pose

    def _duration_value(self) -> float:
        value = float(self.duration.get())
        if value <= 0:
            raise ValueError("duration must be positive")
        return value

    def _record(self) -> dict:
        item = {"pose": self.current_pose()}
        if self.waypoints:
            item["duration"] = self._duration_value()
        return item

    def save_waypoint(self) -> None:
        try:
            self.waypoints.append(self._record())
            self._refresh_list(select=len(self.waypoints) - 1)
        except ValueError as exc:
            self.messagebox.showerror("Invalid waypoint", str(exc))

    def replace_selected(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        try:
            item = {"pose": self.current_pose()}
            if index > 0:
                item["duration"] = self._duration_value()
            self.waypoints[index] = item
            self._refresh_list(select=index)
        except ValueError as exc:
            self.messagebox.showerror("Invalid waypoint", str(exc))

    def load_selected_pose(self) -> None:
        selection = self.listbox.curselection()
        if selection:
            item = self.waypoints[int(selection[0])]
            self._set_pose(item["pose"])
            self.duration.set(str(item.get("duration", 0.5)))

    def delete_selected(self) -> None:
        selection = self.listbox.curselection()
        if selection:
            index = int(selection[0])
            del self.waypoints[index]
            if self.waypoints:
                self.waypoints[0].pop("duration", None)
            self._refresh_list(select=min(index, len(self.waypoints) - 1))

    def reset_open(self) -> None:
        self._set_pose(G20_OPEN_POSE)

    def export(self) -> None:
        if len(self.waypoints) < 2:
            self.messagebox.showerror("Cannot export", "save at least a start and end waypoint")
            return
        try:
            trajectory = interpolate_waypoints(self.waypoints, fps=self.args.fps)
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.args.output.write_text(
                json.dumps(self.waypoints, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            self.status.set(f"Exported {len(self.waypoints)} waypoints / {len(trajectory)} frames to {self.args.output}")
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Export failed", str(exc))

    def _set_pose(self, pose) -> None:
        for index in ACTIVE_IDX:
            self.variables[index].set(int(round(float(pose[index]))))
        self._preview()

    def _preview(self) -> None:
        if self.sim is not None:
            self.sim.set_joints(g20_range_to_sim_radians(
                self.current_pose(), roll_range_ticks=self.args.roll_range_ticks
            ))

    def _refresh_list(self, select: Optional[int] = None) -> None:
        self.listbox.delete(0, self.tk.END)
        for index, item in enumerate(self.waypoints):
            duration = "start" if index == 0 else f"{float(item.get('duration', 0.5)):.2f}s"
            active = [item["pose"][joint] for joint in ACTIVE_IDX]
            self.listbox.insert(self.tk.END, f"{index:02d} {duration}  {active}")
        if select is not None and select >= 0:
            self.listbox.selection_set(select)
            self.listbox.see(select)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.root.destroy()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    editor = WaypointEditor(args)
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
