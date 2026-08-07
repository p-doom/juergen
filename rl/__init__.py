"""RL environments as verifiers Tasksets.

Three envs, one episode driver. `movebox` and `grounding` are container-free:
their "desktop" is an in-process canvas (`rl/desktop.py`) that implements the same
session surface a real VM does, so they run under `evals.harness.DesktopHarness`
rather than carrying rollout loops of their own. `target_box` uses the real VM
pool and differs only in its preparer and rewards.

Rendering and geometry are general; the coordinate convention is the codec's. No
env implements `round(delta/1000 * screen_dim)` — `codec.compile(text, geometry,
cursor)` returns `Operation`s already in absolute screen pixels, and an env only
applies pixel operations.
"""

from rl.geometry import (
    distance_to_box,
    draw_box,
    in_bbox,
    png_bytes,
    render_cursor,
    render_step,
)

__all__ = [
    "distance_to_box",
    "draw_box",
    "in_bbox",
    "png_bytes",
    "render_cursor",
    "render_step",
]
