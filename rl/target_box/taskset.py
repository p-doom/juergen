"""target_box: move a real VM's cursor into a synthetic box on a real desktop.

Reward is dense: reach plus closest-approach shaping. Sparsifying to in-box-only
gives ~0 GRPO signal at the ~1% reach this env starts from.

The shaping term reads best (closest-ever) distance, not final — the
anti-limit-cycle term. A policy that oscillates through the neighbourhood of the
box and ends far away is closer to solving the task than one that never
approaches, and a final-distance reward would score them the other way round.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import verifiers.v1 as vf

from evals.indicators import MouseIndicators, SamplingProvenance
from evals.tasks import RESULT_KEY, DesktopTask, DesktopTaskData, valid_result
from rl.target_box.geometry import TARGET_BOX_INSTRUCTION

__all__ = ["TargetBoxTask", "TargetBoxTaskset", "TargetBoxTasksetConfig"]

SHAPING_WEIGHT = 0.3
SHAPING_SCALE = 400.0
NO_SIGNAL_PENALTY = 0.15


class TargetBoxTask(MouseIndicators, SamplingProvenance, DesktopTask):
    @vf.reward
    async def target_box(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Success requires reaching the box AND declaring it.

        Declares `runtime` because the verdict is a live-VM cursor read.
        """
        del runtime
        result = valid_result(trace, "target_box")
        return 1.0 if result.get("outcome") == "postcondition_reached" else 0.0

    @vf.reward
    async def shaped_progress(self, trace: vf.Trace) -> float:
        # The verdict above is runtime-gated, so without a runtime this is the only
        # reward left; unguarded it floors at -NO_SIGNAL_PENALTY, below any real miss.
        result = valid_result(trace, "target_box")
        if result.get("outcome") == "postcondition_reached":
            return 0.0
        best = float(result["best_distance"])
        if best < 0:
            return -NO_SIGNAL_PENALTY
        return SHAPING_WEIGHT * math.exp(-best / max(SHAPING_SCALE, 1.0))

    @vf.metric
    async def target_box_stats(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        return {
            "best_distance_px": float(result.get("best_distance", -1.0)),
            "entered_box": 1.0 if int(result.get("reach_frame", -1)) >= 0 else 0.0,
            "declared_success": 1.0 if result.get("control_terminate") == "terminate" else 0.0,
        }


class TargetBoxTasksetConfig(vf.TasksetConfig):
    base_path: str = ""
    """Directory of OSWorld task JSONs used only as realistic backgrounds.

    NO-LEAK: point this at the 259-task train set. The 110 held-out tasks are
    eval-only; the synthetic overlay does not make the underlying desktop safe to
    train on."""
    tasks_file: str = ""
    max_tasks: int = 0
    max_steps: int = 10
    box_width: int = 150
    box_height: int = 150
    margin: int = 40
    cursor_margin: int = 20
    seed: int = 0
    screen_width: int = 1920
    screen_height: int = 1080


class TargetBoxTaskset(vf.Taskset[TargetBoxTask, TargetBoxTasksetConfig]):
    def load(self) -> Iterable[TargetBoxTask]:
        for idx, path in enumerate(self._paths()):
            if self.config.max_tasks and idx >= self.config.max_tasks:
                return
            payload = json.loads(path.read_text())
            task_id = payload.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"OSWorld task has no string id: {path}")
            yield TargetBoxTask(
                DesktopTaskData(
                    idx=idx,
                    name=f"target_box/{task_id}",
                    prompt=TARGET_BOX_INSTRUCTION,
                    instruction=TARGET_BOX_INSTRUCTION,
                    kind="target_box",
                    max_steps=self.config.max_steps,
                    task_path=str(path),
                    setup={
                        "config": list(payload.get("config", [])),
                        "instance_key": f"{task_id}:{path}",
                        "box": {
                            "box_width": self.config.box_width,
                            "box_height": self.config.box_height,
                            "margin": self.config.margin,
                            "cursor_margin": self.config.cursor_margin,
                            "seed": self.config.seed,
                        },
                        "screen": [self.config.screen_width, self.config.screen_height],
                    },
                ),
                self.config.task,
            )

    def _paths(self) -> list[Path]:
        if self.config.tasks_file:
            names = [
                line.strip()
                for line in Path(self.config.tasks_file).read_text().splitlines()
                if line.strip()
            ]
            root = Path(self.config.base_path)
            return [root / f"{name}.json" if not name.endswith(".json") else root / name for name in names]
        return sorted(Path(self.config.base_path).rglob("*.json"))
