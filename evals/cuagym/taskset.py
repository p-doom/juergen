"""The suite's 28 rows, verified against their pins before a VM ever boots."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

import verifiers.v1 as vf

# Importing the preparer registers kind="cuagym"; without it the first episode
# fails inside a booted VM instead of at load.
import evals.cuagym.guest  # noqa: F401
from evals.cuagym.bundles import load_suite, verify_bundle
from evals.tasks import DesktopTask, DesktopTaskData

__all__ = ["CuaGymTaskset", "CuaGymTasksetConfig"]

_LOGGER = logging.getLogger(__name__)


class CuaGymTasksetConfig(vf.TasksetConfig):
    """Where the extracted bundles live, and the usual row-count knobs.

    `bundles_root` (or `$CUAGYM_BUNDLES`) holds one directory per task id, as
    written by `python -m evals.cuagym.bundles`. `max_steps=0` means the
    suite's own default.
    """

    bundles_root: str = ""
    suite_path: str = ""
    max_steps: int = 0
    max_tasks: int = 0


class CuaGymTaskset(vf.Taskset[DesktopTask, CuaGymTasksetConfig]):
    def load(self) -> Iterable[DesktopTask]:
        suite = load_suite(self.config.suite_path or None)
        root = Path(self.config.bundles_root or os.environ.get("CUAGYM_BUNDLES", ""))
        if not root.is_dir():
            raise FileNotFoundError(
                "cuagym bundles_root is not a directory; extract the suite's "
                "bundles first (python -m evals.cuagym.bundles --dataset-root "
                f"... --out ...) and point bundles_root/$CUAGYM_BUNDLES at it: {root}"
            )
        max_steps = self.config.max_steps or int(
            (suite.get("defaults") or {}).get("max_steps", 25)
        )
        for idx, row in enumerate(suite["tasks"]):
            if self.config.max_tasks and idx >= self.config.max_tasks:
                return
            task_id = str(row["id"])
            bundle_dir = root / task_id
            verify_bundle(bundle_dir, dict(row["sha256"]), task_id)
            payload = json.loads((bundle_dir / "task.json").read_text(encoding="utf-8"))
            if str(payload["instruction"]) != str(row["instruction"]):
                raise ValueError(
                    f"bundle {task_id} instruction differs from the suite's — "
                    "the pins passed, so the suite file itself was edited"
                )
            yield DesktopTask(
                DesktopTaskData(
                    idx=idx,
                    name=task_id,
                    prompt=str(payload["instruction"]),
                    instruction=str(payload["instruction"]),
                    kind="cuagym",
                    max_steps=max_steps,
                    app=str(row.get("app") or payload.get("app_type") or "") or None,
                    task_path=str(bundle_dir / "task.json"),
                    setup={
                        "task_config": payload,
                        "config": list(payload.get("config") or []),
                        "bundle_dir": str(bundle_dir),
                        "reward_path": str(bundle_dir / "reward.py"),
                    },
                ),
                self.config.task,
            )
