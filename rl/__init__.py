"""RL environments as verifiers Tasksets.

Three envs, one episode driver. `movebox` and `grounding` are container-free:
their "desktop" is an in-process canvas (`rl/desktop.py`) that implements the same
session surface a real VM does, so they run under `evals.harness.DesktopHarness`
rather than carrying rollout loops of their own. `target_box` uses the real VM
pool and differs only in its preparer and rewards.

The seam the refactor draws: **rendering and geometry are general, the coordinate
convention is the codec's.** `movebox/dataset.py:159` used to document *"Apply a
NORMALIZED 0-999 relative delta to the cursor (px), clamp to screen"* and
implement `round(delta/1000 * screen_dim)` inline — three times, in three files,
with a fourth copy in an offline inspector that got it wrong (raw pixels) and
therefore disagreed with the reward it claimed to reproduce. That arithmetic is
now inside the codec: `codec.compile(text, geometry, cursor)` returns `Operation`s
already in absolute screen pixels, and an env only ever applies pixel operations.
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
