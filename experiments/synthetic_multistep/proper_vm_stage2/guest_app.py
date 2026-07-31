#!/usr/bin/env python3
"""Tiny guest-side full-screen event probe used only by the live KVM smoke."""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from pathlib import Path


CONFIG_PATH = Path(sys.argv[1])
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
state_path = Path(config["state_path"])
bbox = tuple(int(value) for value in config["bbox"])
start = tuple(int(value) for value in config["cursor"])
operation = config["operation"]
revision = config["revision"]

state = {
    "schema_version": 1,
    "revision": revision,
    "operation": operation,
    "ready": False,
    "down": False,
    "down_position": None,
    "release_position": None,
    "click_success": False,
    "drag_success": False,
    "button_presses": 0,
    "button_releases": 0,
    "motions_while_down": 0,
}


def write_state() -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(tmp, state_path)


def inside(point: tuple[int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def on_down(event: tk.Event) -> None:
    point = (int(event.x_root), int(event.y_root))
    state["down"] = True
    state["down_position"] = list(point)
    state["button_presses"] += 1
    write_state()


def on_motion(event: tk.Event) -> None:
    if state["down"]:
        state["motions_while_down"] += 1
        write_state()


def on_up(event: tk.Event) -> None:
    point = (int(event.x_root), int(event.y_root))
    down_position = tuple(state["down_position"] or point)
    moved = down_position != point
    state["release_position"] = list(point)
    state["button_releases"] += 1
    state["click_success"] = bool(operation == "click" and inside(point))
    state["drag_success"] = bool(
        operation == "drag"
        and abs(down_position[0] - start[0]) <= 2
        and abs(down_position[1] - start[1]) <= 2
        and moved
        and inside(point)
    )
    state["down"] = False
    write_state()


root = tk.Tk()
root.overrideredirect(True)
root.geometry(f"{config['screen'][0]}x{config['screen'][1]}+0+0")
root.configure(cursor="none", background="black")
root.attributes("-topmost", True)
canvas = tk.Canvas(
    root,
    width=int(config["screen"][0]),
    height=int(config["screen"][1]),
    borderwidth=0,
    highlightthickness=0,
    cursor="none",
)
canvas.pack(fill="both", expand=True)
photo = tk.PhotoImage(file=config["image_path"])
canvas.create_image(0, 0, anchor="nw", image=photo)
canvas.bind("<ButtonPress-1>", on_down)
canvas.bind("<B1-Motion>", on_motion)
canvas.bind("<ButtonRelease-1>", on_up)


def mark_ready() -> None:
    root.focus_force()
    state["ready"] = True
    write_state()


root.after(250, mark_ready)
root.mainloop()
