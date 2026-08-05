"""Geometry and rendering shared by every box-target env.

Pre-refactor these were duplicated with quiet inconsistencies that changed scores:

  * two `distance_to_box` implementations, one taking an xyxy tuple and one a
    `TargetBox` dataclass, with identical bodies;
  * a **half-open** point test (`in_bbox`, grounding) and a **closed** one
    (`point_in_box`, target_box), while both envs constructed boxes as
    `(x1, y1, x1+w-1, y1+h-1)` — so a "150 px" box was a 149 px hit region in one
    env and 150 px in the other;
  * three box outlines at widths 5, 3 and 3, and a cursor marker defined once but
    reached for through two wrappers.

One definition each. `in_bbox` keeps the half-open convention because that is what
the published grounding numbers were computed under; `BOX_EDGE_INCLUSIVE` names the
alternative rather than leaving it implicit.
"""

from __future__ import annotations

import io
import math

__all__ = [
    "BOX_EDGE_INCLUSIVE",
    "CURSOR_COLOR",
    "BOX_COLOR",
    "box_center",
    "distance_to_box",
    "draw_box",
    "in_bbox",
    "png_bytes",
    "render_cursor",
    "render_step",
]

BOX_EDGE_INCLUSIVE = False
"""Whether the max edge counts as inside. False = half-open, the convention every
published grounding/movebox reach number was computed under. Flipping it widens a
150 px box's hit region by one pixel row and column, which is small but not zero:
do not flip it to "fix" an off-by-one without re-baselining."""

CURSOR_COLOR = (255, 0, 0)
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 5
CURSOR_RADIUS = 10
CURSOR_WIDTH = 3


def in_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    if BOX_EDGE_INCLUSIVE:
        return bbox[0] <= pos[0] <= bbox[2] and bbox[1] <= pos[1] <= bbox[3]
    return bbox[0] <= pos[0] < bbox[2] and bbox[1] <= pos[1] < bbox[3]


def distance_to_box(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> float:
    """Euclidean distance to the box's nearest point; 0 inside."""
    dx = max(bbox[0] - pos[0], 0, pos[0] - bbox[2])
    dy = max(bbox[1] - pos[1], 0, pos[1] - bbox[3])
    return math.hypot(dx, dy)


def box_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def png_bytes(image: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # type: ignore[attr-defined]
    return buffer.getvalue()


def draw_box(base: object, bbox: tuple[int, int, int, int], *, width: int = BOX_WIDTH):
    """A bright-green outline on a copy of `base`."""
    from PIL import ImageDraw

    image = base.copy()  # type: ignore[attr-defined]
    ImageDraw.Draw(image).rectangle(
        [bbox[0], bbox[1], bbox[2], bbox[3]], outline=BOX_COLOR, width=width
    )
    return image


def render_cursor(base: object, cursor: tuple[int, int]):
    """A red crosshair-plus-ring marker at `cursor`, on a copy of `base`.

    Synthetic because these envs have no real desktop cursor. The real-VM env
    (`target_box`) deliberately does NOT draw one — the genuine GNOME cursor is
    already in the screenshot, and compositing a second marker on top would train
    the model to look for a marker that inference will not have.
    """
    from PIL import ImageDraw

    image = base.copy()  # type: ignore[attr-defined]
    draw = ImageDraw.Draw(image)
    x, y = cursor
    r = CURSOR_RADIUS
    draw.line([(x - r, y), (x + r, y)], fill=CURSOR_COLOR, width=CURSOR_WIDTH)
    draw.line([(x, y - r), (x, y + r)], fill=CURSOR_COLOR, width=CURSOR_WIDTH)
    draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=CURSOR_COLOR, width=CURSOR_WIDTH)
    return image


def render_step(base_with_box: object, cursor: tuple[int, int]) -> bytes:
    """Composite the cursor marker on a (background + box) image -> PNG bytes."""
    return png_bytes(render_cursor(base_with_box, cursor))
