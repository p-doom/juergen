"""grounding scene loading: a labelled screenshot, a target bbox, a cursor start.

Container-free: the observation is the *cached* labelled screenshot with a
synthetic cursor marker composited on it, so the env needs no VM. That is what
makes this the cheap single-step probe; the VM-backed variant is
`evals.tasks.GroundingTaskset` + the `grounding` preparer.

`to_norm` / `from_norm` are gone. They implemented the normalized 0-999 round trip
(`from_norm(to_norm(cursor) + delta)`) that decided whether a move landed in the
bbox, and the offline inspector re-implemented the same step in **raw pixels** —
so its rendered vectors and reported hit/miss disagreed with the reward by ~1.9x in
x and ~1.08x in y at 1920x1080. One conversion, inside the codec, removes the
possibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["REGIMES", "GroundingTarget", "cursor_start", "load_canvas", "load_targets"]

REGIMES: tuple[str, ...] = ("near", "medium", "far")


@dataclass(frozen=True)
class GroundingTarget:
    idx: int
    app: str
    task_id: str
    instruction: str
    bbox: tuple[int, int, int, int]
    image_path: Path
    screen: tuple[int, int]


def load_targets(jsonl_path: Path) -> list[GroundingTarget]:
    out: list[GroundingTarget] = []
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        image_path = Path(label["image_path"])
        parts = image_path.parts
        if parts[-2] != "steps":
            raise ValueError(f"unexpected image_path shape: {label['image_path']!r}")
        out.append(
            GroundingTarget(
                idx=int(label["idx"]),
                app=str(label["app"]),
                task_id=parts[-3],
                instruction=str(label["instruction"]),
                bbox=tuple(int(v) for v in label["bbox_xyxy"]),
                image_path=image_path,
                screen=_screenshot_size(image_path),
            )
        )
    return out


def _screenshot_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as handle:
        return handle.size


def cursor_start(
    target: GroundingTarget, screen_w: int, screen_h: int, regime: str
) -> tuple[int, int]:
    """Deterministic start by regime — see `evals.tasks.cursor_start` for the rule.

    Delegates so the container-free and VM-backed grounding evals cannot drift
    apart on the one variable the whole eval controls.
    """
    from evals.tasks import cursor_start as shared

    return shared(target.bbox, screen_w, screen_h, regime, target.task_id)


def load_canvas(target: GroundingTarget):
    from PIL import Image

    with Image.open(target.image_path) as handle:
        return handle.convert("RGB")
