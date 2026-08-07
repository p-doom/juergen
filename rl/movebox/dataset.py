"""movebox scene sampling: a background screenshot, a green box, a cursor start.

No relative-delta arithmetic lives here: the codec hands the env `Operation`s in
absolute pixels.

`band_weights` carries the curriculum shape. The band list is not in
dict-insertion order: that would correlate task index with difficulty, making any
prefix or shard of the taskset a biased sample.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CURRICULUM_BANDS",
    "DEFAULT_BACKGROUNDS_DIR",
    "MoveBoxScene",
    "SCREEN_H",
    "SCREEN_W",
    "band_sequence",
    "list_backgrounds",
    "load_canvas",
    "sample_scene",
]

DEFAULT_BACKGROUNDS_DIR = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/"
    "converted/_osworld_images"
)
SCREEN_W = 1920
SCREEN_H = 1080

CURRICULUM_BANDS: dict[str, int] = {
    "near": 250,
    "medium": 600,
    "far": 1200,
    "uniform": 0,
}
"""Max cursor-to-box-centre distance in px per band; `uniform` = no curriculum."""


@dataclass(frozen=True)
class MoveBoxScene:
    idx: int
    background_path: str
    box: tuple[int, int, int, int]
    cursor_start: tuple[int, int]
    screen_w: int
    screen_h: int
    band: str
    start_distance: float
    """Distance to the box centre, the sampler's own control variable. Not the
    reward's distance, which is to the nearest box edge — the two names stay
    distinct so the units cannot be confused."""


def list_backgrounds(backgrounds_dir: str = DEFAULT_BACKGROUNDS_DIR) -> list[str]:
    paths = sorted(str(p) for p in Path(backgrounds_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no *.png backgrounds under {backgrounds_dir}")
    return paths


def sample_scene(
    idx: int,
    backgrounds: list[str],
    *,
    band: str,
    box_w: int = 150,
    box_h: int = 150,
    margin: int = 40,
    screen_w: int = SCREEN_W,
    screen_h: int = SCREEN_H,
    seed: int = 0,
) -> MoveBoxScene:
    """Deterministic scene for `(seed, idx, band)`.

    Every pool worker runs `Taskset.load()` independently, so generation must be a
    pure function of the key — a `random` module call seeded once at import would
    give two workers different task 7.
    """
    rng = random.Random(f"movebox:v1:{seed}:{idx}:{band}")
    background = backgrounds[rng.randrange(len(backgrounds))]
    x1 = rng.randint(margin, max(margin, screen_w - margin - box_w))
    y1 = rng.randint(margin, max(margin, screen_h - margin - box_h))
    box = (x1, y1, x1 + box_w - 1, y1 + box_h - 1)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    min_dist = max(int(math.hypot(box_w, box_h) / 2) + 20, 60)
    limit = CURRICULUM_BANDS[band]
    cursor = _sample_cursor(
        rng, box, cx, cy, min_dist, limit, screen_w=screen_w, screen_h=screen_h
    )
    return MoveBoxScene(
        idx=idx,
        background_path=background,
        box=box,
        cursor_start=cursor,
        screen_w=screen_w,
        screen_h=screen_h,
        band=band,
        start_distance=math.hypot(cursor[0] - cx, cursor[1] - cy),
    )


def _sample_cursor(
    rng: random.Random,
    box: tuple[int, int, int, int],
    cx: int,
    cy: int,
    min_dist: int,
    limit: int,
    *,
    screen_w: int,
    screen_h: int,
) -> tuple[int, int]:
    from rl.geometry import in_bbox

    for _ in range(200):
        if limit <= 0:
            candidate = (rng.randrange(screen_w), rng.randrange(screen_h))
        else:
            dist = rng.uniform(min_dist, max(min_dist + 1, limit))
            angle = rng.uniform(0.0, 2.0 * math.pi)
            candidate = (
                max(0, min(screen_w - 1, cx + int(round(dist * math.cos(angle))))),
                max(0, min(screen_h - 1, cy + int(round(dist * math.sin(angle))))),
            )
        if not in_bbox(candidate, box):
            return candidate
    # A box that swallows every sample is a misconfigured scene, not a hard task.
    return (0 if cx > screen_w // 2 else screen_w - 1, 0 if cy > screen_h // 2 else screen_h - 1)


def load_canvas(scene: MoveBoxScene):
    """Background resized to the scene's screen, with the target box drawn on it."""
    from PIL import Image

    from rl.geometry import draw_box

    with Image.open(scene.background_path) as handle:
        base = handle.convert("RGB")
        if base.size != (scene.screen_w, scene.screen_h):
            base = base.resize((scene.screen_w, scene.screen_h))
        return draw_box(base, scene.box)


def band_sequence(band_weights: dict[str, float], n_tasks: int, seed: int) -> list[str]:
    """Bands for `n_tasks`, shuffled so index is uncorrelated with difficulty.

    Unshuffled, any prefix, shard or `max_tasks` cut would be a biased sample of
    the curriculum.
    """
    bands: list[str] = []
    for name, weight in band_weights.items():
        bands.extend([name] * int(round(weight * n_tasks)))
    while len(bands) < n_tasks:
        bands.append(next(iter(band_weights)))
    bands = bands[:n_tasks]
    random.Random(f"movebox:bands:{seed}:{n_tasks}").shuffle(bands)
    return bands
