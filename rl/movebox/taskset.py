"""movebox: multi-step move-the-cursor-into-the-box, container-free.

Reward is **pure sparse task success** — 1.0 iff the cursor entered the box. The
pre-refactor module kept `STEP_PENALTY`, `REPEAT_PENALTY`, `NO_OP_PENALTY` and a
dense shaping term all present and all set to 0/off, which is a trap: the constants
read as tunable while `efficiency` was identically `-0.0`. They are gone, and the
one remaining knob is named.

Shaping parameters stay **module constants, never config fields.** Inside a
`vf.Task`, `self.config` is the per-task `TaskConfig`, which has no custom fields —
reading one raises `AttributeError`, and one throwing reward inside `Task.score`'s
`asyncio.gather` drops the *whole group's* rewards, silently deleting the gradient.
That bug killed prior runs; the constant form makes it unreachable.
"""

from __future__ import annotations

from typing import Any, Iterable

import verifiers.v1 as vf

from evals.indicators import MouseIndicators, SamplingProvenance
from evals.tasks import RESULT_KEY, DesktopState, DesktopTask, DesktopTaskData
from rl.movebox.dataset import (
    DEFAULT_BACKGROUNDS_DIR,
    SCREEN_H,
    SCREEN_W,
    band_sequence,
    list_backgrounds,
    sample_scene,
)

__all__ = ["MoveBoxTask", "MoveBoxTaskset", "MoveBoxTasksetConfig"]

REACH_REWARD = 1.0
"""The only reward term. Sparse task success, per the explicit directive that this
env report exactly what it measures and nothing shaped on top."""


class MoveBoxTask(MouseIndicators, SamplingProvenance, DesktopTask):
    @vf.reward
    async def reach(self, trace: vf.Trace) -> float:
        """1.0 iff the cursor entered the box at any step.

        Raises when the harness published no result: an absent rollout is
        infrastructure-invalid and must not be trained as a zero.
        """
        result = trace.info.get(RESULT_KEY)
        if not isinstance(result, dict):
            raise RuntimeError("movebox rollout published no result")
        return REACH_REWARD if int(result.get("reach_frame", -1)) >= 0 else 0.0

    @vf.metric
    async def movebox(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        steps = result.get("steps_detail") or []
        keys = [
            (
                s.get("control"),
                tuple((s.get("parsed_action") or {}).get("elements") or ()),
                str(s.get("raw_model_output", ""))[-64:],
            )
            for s in steps
            if isinstance(s, dict)
        ]
        repeats = sum(1 for a, b in zip(keys, keys[1:]) if a == b)
        return {
            "reached": 1.0 if int(result.get("reach_frame", -1)) >= 0 else 0.0,
            "reach_step": float(result.get("reach_frame", -1)),
            "best_distance_px": float(result.get("best_distance", -1.0)),
            "repeat_actions": float(repeats),
            # From the preparer's evidence, which is where the band actually lands.
            # This read `result["band_index"]`, a key nothing ever publishes, so the
            # curriculum metric was the constant -1.0 — the one number this env needs
            # to slice its results by.
            "band": _band_index(result),
        }


class MoveBoxTasksetConfig(vf.TasksetConfig):
    backgrounds_dir: str = DEFAULT_BACKGROUNDS_DIR
    n_tasks: int = 512
    seed: int = 0
    box_w: int = 150
    box_h: int = 150
    margin: int = 40
    screen_w: int = SCREEN_W
    screen_h: int = SCREEN_H
    band_weights: dict[str, float] = {"near": 0.6, "medium": 0.3, "far": 0.1}
    max_steps: int = 8
    """Read here and passed onto the row, unlike the pre-refactor field of the same
    name which was silently ignored in favour of the harness's."""


class MoveBoxTaskset(vf.Taskset[MoveBoxTask, MoveBoxTasksetConfig]):
    def load(self) -> Iterable[MoveBoxTask]:
        backgrounds = list_backgrounds(self.config.backgrounds_dir)
        bands = band_sequence(self.config.band_weights, self.config.n_tasks, self.config.seed)
        for idx in range(self.config.n_tasks):
            scene = sample_scene(
                idx,
                backgrounds,
                band=bands[idx],
                box_w=self.config.box_w,
                box_h=self.config.box_h,
                margin=self.config.margin,
                screen_w=self.config.screen_w,
                screen_h=self.config.screen_h,
                seed=self.config.seed,
            )
            yield MoveBoxTask(
                DesktopTaskData(
                    idx=idx,
                    name=f"movebox_{idx:05d}_{scene.band}",
                    prompt=_INSTRUCTION,
                    instruction=_INSTRUCTION,
                    kind="movebox",
                    max_steps=self.config.max_steps,
                    bbox=scene.box,
                    cursor_start=scene.cursor_start,
                    expected={"box": list(scene.box)},
                    setup={
                        "background_path": scene.background_path,
                        "band": scene.band,
                        "start_distance": scene.start_distance,
                        "screen": [scene.screen_w, scene.screen_h],
                    },
                ),
                self.config.task,
            )


BAND_ORDER: tuple[str, ...] = ("near", "medium", "far", "uniform")
"""Band -> metric index. Explicit rather than `CURRICULUM_BANDS` insertion order, so
a reordering of that dict cannot silently renumber a recorded metric."""


def _band_index(result: dict[str, Any]) -> float:
    band = str((result.get("setup") or {}).get("band", ""))
    return float(BAND_ORDER.index(band)) if band in BAND_ORDER else -1.0


_INSTRUCTION = (
    "Move the mouse cursor INTO the green highlighted box. The cursor is at the red "
    "crosshair marker. When the cursor is inside the box, you are done."
)


def state_type() -> type[DesktopState]:
    return DesktopState
