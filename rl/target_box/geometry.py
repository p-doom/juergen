"""target_box scene derivation: a synthetic box over a real OSWorld desktop.

Nothing about the box is in the task row. The box and cursor start are derived
deterministically from `instance_key = f"{task_id}:{path}"`, so one OSWorld task
always gets the same scene while the background is a genuine post-`reset()`
desktop. The real task's instruction is not used — `TARGET_BOX_INSTRUCTION` is.

`TargetBoxConfig.validate` rejects a box wider than `screen - 2*margin` or a
cursor margin covering half the screen: both produce episodes that are impossible
rather than hard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rl.geometry import distance_to_box, in_bbox

__all__ = [
    "TARGET_BOX_INSTRUCTION",
    "TargetBoxConfig",
    "annotate",
    "sample_box",
    "sample_cursor_start",
]

TARGET_BOX_INSTRUCTION = (
    "Move the cursor into the green box. If the cursor is already inside the box, "
    "terminate with success. If the cursor is outside the box, move it towards the "
    "box center."
)


@dataclass(frozen=True)
class TargetBoxConfig:
    box_width: int = 150
    box_height: int = 150
    margin: int = 40
    cursor_margin: int = 20
    seed: int = 0

    def validate(self, *, screen_width: int, screen_height: int) -> None:
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        if self.margin < 0 or self.cursor_margin < 0:
            raise ValueError("margins must be non-negative")
        if self.box_width + 2 * self.margin > screen_width:
            raise ValueError("box width plus margins exceeds the screen width")
        if self.box_height + 2 * self.margin > screen_height:
            raise ValueError("box height plus margins exceeds the screen height")
        if 2 * self.cursor_margin >= min(screen_width, screen_height):
            raise ValueError("cursor margin leaves no admissible cursor region")


def sample_box(
    config: TargetBoxConfig, *, screen_width: int, screen_height: int, instance_key: str
) -> tuple[int, int, int, int]:
    config.validate(screen_width=screen_width, screen_height=screen_height)
    rng = random.Random(f"{config.seed}:{instance_key}:box")
    x1 = rng.randint(config.margin, screen_width - config.margin - config.box_width)
    y1 = rng.randint(config.margin, screen_height - config.margin - config.box_height)
    return (x1, y1, x1 + config.box_width - 1, y1 + config.box_height - 1)


def sample_cursor_start(
    config: TargetBoxConfig,
    box: tuple[int, int, int, int],
    *,
    screen_width: int,
    screen_height: int,
    instance_key: str,
) -> tuple[int, int]:
    rng = random.Random(f"{config.seed}:{instance_key}:cursor")
    low_x, high_x = config.cursor_margin, screen_width - config.cursor_margin - 1
    low_y, high_y = config.cursor_margin, screen_height - config.cursor_margin - 1
    for _ in range(100):
        candidate = (rng.randint(low_x, high_x), rng.randint(low_y, high_y))
        if not in_bbox(candidate, box):
            return candidate
    corners = [(low_x, low_y), (high_x, low_y), (low_x, high_y), (high_x, high_y)]
    outside = [c for c in corners if not in_bbox(c, box)]
    if not outside:
        raise ValueError("every admissible cursor corner is inside the target box")
    return max(outside, key=lambda c: distance_to_box(c, box))


def annotate(screenshot: bytes, box: tuple[int, int, int, int]) -> bytes:
    """Draw the target box on a live screenshot.

    No cursor marker: the genuine desktop cursor is already in the frame, and a
    synthetic one is not present at inference.
    """
    import io

    from PIL import Image

    from rl.geometry import draw_box, jpeg_bytes

    with Image.open(io.BytesIO(screenshot)) as handle:
        base = handle.convert("RGB")
        return jpeg_bytes(draw_box(base, box, width=3))
