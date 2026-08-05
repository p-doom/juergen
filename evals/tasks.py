"""Task rows, rollout state, and the tasksets that enumerate them.

The six collapsed drivers differed less in their loop than in *what they did to
the VM before the loop* and *what they read out of it afterwards*. Those two
things are the `Preparer` seam below; everything else is one harness.

  * freeroll               -> `none` / `terminal`
  * grounding              -> `grounding` (OSWorld setup + cached-trajectory
                              replay + stratified cursor placement)
  * fullbench / one_task   -> `osworld` (OSWorld `SetupController`, scored by
                              `DesktopEnv.evaluate()`)
  * sign-of-life           -> four kinds registered by `evals/signoflife/guest.py`

A `Preparer` is the only place a family-specific in-VM side effect may happen.
The harness never branches on task kind.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

import verifiers.v1 as vf

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "PREPARERS",
    "DesktopState",
    "DesktopTask",
    "DesktopTaskData",
    "FreerollTaskset",
    "FreerollTasksetConfig",
    "GroundingTaskset",
    "GroundingTasksetConfig",
    "OSWorldTaskset",
    "OSWorldTasksetConfig",
    "Preparer",
    "REGIMES",
    "cursor_start",
    "in_bbox",
    "register_preparer",
]

REGIMES: tuple[str, ...] = ("near", "medium", "far")
RESULT_KEY = "episode"
RESULT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# rows and state
# --------------------------------------------------------------------------- #


class DesktopTaskData(vf.TaskData):
    """One desktop episode's row.

    `kind` selects the `Preparer`; `expected` is whatever that family's oracle
    needs; `setup` is static per-task setup input (an OSWorld task config path, a
    replay trajectory, a bbox). Nothing here is grammar-specific — the codec is a
    harness config field, so the same row runs under every grammar, which is what
    makes a grammar A/B a one-field change.
    """

    kind: str = "none"
    instruction: str = ""
    max_steps: int = 15
    expected: dict[str, Any] = {}
    setup: dict[str, Any] = {}
    app: str | None = None
    snapshot: str | None = None
    task_path: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    regime: str | None = None
    cursor_start: tuple[int, int] | None = None
    no_submit: bool = False
    """True for a cell whose success *requires* not pressing Return. Read by the
    over-submission indicator (D), which is otherwise unable to tell a correct
    submit from an over-generalised one."""


class DesktopState(vf.State):
    """Live rollout state. Mirrored into `trace.info[RESULT_KEY]` for offline
    re-scoring, because `state` is `exclude=True` and never reaches traces.jsonl."""

    success: bool | None = None
    outcome: str | None = None
    steps: int = 0
    parse_errors: int = 0
    action_errors: int = 0
    executor_errors: int = 0
    control_terminate: str | None = None
    terminate_step: int | None = None
    infra_valid: bool = True
    infra_error: dict[str, str] | None = None
    codec: str | None = None
    history_policy: str | None = None
    temperature: float | None = None
    temperature_source: str | None = None
    reach_frame: int = -1
    best_distance: float = -1.0
    final_probe: dict[str, Any] | None = None
    initial_probe: dict[str, Any] | None = None
    task_reward: float | None = None
    scripted: bool = False
    negative_control: bool = False


class DesktopTask(vf.Task[DesktopTaskData, DesktopState]):
    """Base task: no rewards of its own.

    Rewards come from the oracle mixins in `evals/oracles.py` and metrics from
    `evals/indicators.py`; a concrete taskset composes exactly the ones its family
    can actually evaluate. Keeping the base empty means a taskset never inherits a
    reward that raises for lack of evidence — one throwing reward inside
    `Task.score`'s `asyncio.gather` drops the whole group's rewards.
    """


# --------------------------------------------------------------------------- #
# preparers
# --------------------------------------------------------------------------- #


@runtime_checkable
class Preparer(Protocol):
    """Family-specific VM preparation and read-only state extraction.

    `prepare` may drive the guest (launch a terminal, run OSWorld setup commands,
    replay a cached trajectory, move the cursor). `probe` must not: it is the
    read-only path the oracle depends on, and the pre-refactor runner asserted
    exactly that (`read_only is True`, `input_events == []`).
    """

    kind: str

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]: ...
    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]: ...


PREPARERS: dict[str, Preparer] = {}


def register_preparer(preparer: Preparer) -> Preparer:
    PREPARERS[preparer.kind] = preparer
    return preparer


def preparer_for(kind: str) -> Preparer:
    try:
        return PREPARERS[kind]
    except KeyError as exc:
        raise LookupError(
            f"no preparer registered for task kind {kind!r}; known: {sorted(PREPARERS)}"
        ) from exc


class _NoPreparation:
    """freeroll `--desktop_setup none`: boot, screenshot, go."""

    kind = "none"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del session, task
        return {"prepared": "none"}

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {"cursor": list(session.cursor_position()), "screen": list(session.screen_size())}


class _TerminalPreparation:
    """freeroll `--desktop_setup terminal`.

    Only freeroll had this, and only typing evals need it: start with a focused
    terminal so the first model turn is not spent finding one. Preserved verbatim,
    including the `ctrl-l` clear.
    """

    kind = "terminal"

    _SCRIPT = (
        "import subprocess; "
        "subprocess.Popen(['bash', '-lc', "
        "\"(command -v gnome-terminal >/dev/null && gnome-terminal) || "
        "(command -v xfce4-terminal >/dev/null && xfce4-terminal) || "
        "(command -v xterm >/dev/null && xterm)\"]); "
        "time.sleep(2.0); "
        "pyautogui.hotkey('ctrl', 'l'); "
        "time.sleep(0.2)"
    )

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        session.execute_pyautogui(self._SCRIPT)
        return {"prepared": "terminal"}

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {"cursor": list(session.cursor_position())}


class _OSWorldPreparation:
    """OSWorld benchmark tasks: run the task JSON's `config` setup commands.

    Both `osworld_fullbench_runner.py` and `osworld_one_task_runner.py` got this
    for free from `DesktopEnv.reset(task_config=...)`; `osworld_grounding_runner.py`
    reached for `SetupController` directly so the VM lifecycle stayed under our
    qemu+KVM control rather than the apptainer provider that strips KVM ioctls on
    hai-* nodes. The session owns the lifecycle now, so setup goes through it.
    """

    kind = "osworld"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        config = _osworld_config(task)
        if not config:
            return {"prepared": "osworld", "steps": 0}
        session.setup(config)
        return {"prepared": "osworld", "steps": len(config)}

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {"cursor": list(session.cursor_position())}


class _GroundingPreparation(_OSWorldPreparation):
    """Grounding targets: OSWorld setup, optional cached-trajectory replay, then a
    deterministic stratified cursor placement.

    Two behaviours here existed in `osworld_grounding_runner.py` alone.

    *Replay.* Our labelled bboxes were sampled from `step_001.png` frames sitting
    next to the originating rollout's `traj.jsonl`, so the desktop must be advanced
    from post-setup to post-replay before the bbox means anything. Upstream
    OSWorld's `_replay_setup` is a `NotImplementedError` stub; the replacement
    dispatches each cached `action`, which may be either a raw pyautogui
    expression or one of our grammar's action lines — the codec is tried first and
    a raw `execute` is the fallback, so one replay step works against either cache
    origin.

    *Stratified starts.* `near`/`medium`/`far` are not jitter: they make the
    distance-to-target a controlled variable, and the minimum radius is raised by
    the bbox half-diagonal so a full-window target cannot collect a degenerate
    reach-at-step-0.
    """

    kind = "grounding"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        evidence = super().prepare(session, task)
        replay = task.setup.get("replay_trajectory")
        n_steps = int(task.setup.get("replay_n_steps", 1))
        if replay:
            evidence["replayed"] = _replay(session, Path(replay), n_steps)
        width, height = session.screen_size()
        target = task.cursor_start or cursor_start(
            task.bbox or (0, 0, 1, 1), width, height, task.regime or "far", task.name or ""
        )
        session.execute_pyautogui(f"pyautogui.moveTo({target[0]}, {target[1]})")
        observed = tuple(session.cursor_position())
        if observed != tuple(target):
            _LOGGER.warning(
                "cursor_start: requested %s, got %s — using actual position",
                target,
                observed,
            )
        evidence.update(
            {
                "requested_cursor_start": list(target),
                "observed_cursor_start": list(observed),
                "regime": task.regime,
            }
        )
        return evidence

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        cursor = tuple(session.cursor_position())
        bbox = task.bbox
        return {
            "cursor": list(cursor),
            "in_bbox": bool(bbox and in_bbox(cursor, bbox)),
            "distance": None if not bbox else distance_to_box(cursor, bbox),
        }


def _osworld_config(task: DesktopTaskData) -> list[dict[str, Any]]:
    inline = task.setup.get("config")
    if isinstance(inline, list):
        return list(inline)
    if task.task_path:
        payload = json.loads(Path(task.task_path).read_text())
        return list(payload.get("config", []))
    return []


def _replay(session: Any, trajectory: Path, n_steps: int, sleep_s: float = 0.5) -> int:
    """Dispatch the first `n_steps` non-reset cached actions."""
    import time

    replayed = 0
    for raw in trajectory.read_text().splitlines():
        if not raw.strip() or replayed >= n_steps:
            if replayed >= n_steps:
                break
            continue
        entry = json.loads(raw)
        if entry.get("step_num", 0) == 0:
            continue
        action = (entry.get("action") or "").strip()
        if not action or action == "<reset>":
            continue
        try:
            session.execute_pyautogui(action)
        except Exception as exc:  # noqa: BLE001 - a bad cached row must not kill the run
            _LOGGER.warning("replay: skipping unexecutable action %r: %s", action[:120], exc)
            continue
        replayed += 1
        time.sleep(sleep_s)
    _LOGGER.info("replay: dispatched %d action(s) from %s", replayed, trajectory)
    return replayed


for _preparer in (
    _NoPreparation(),
    _TerminalPreparation(),
    _OSWorldPreparation(),
    _GroundingPreparation(),
):
    register_preparer(_preparer)


# --------------------------------------------------------------------------- #
# geometry (shared with rl/)
# --------------------------------------------------------------------------- #


def in_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    """Half-open on the max edge, verbatim from both pre-refactor definitions."""
    return bbox[0] <= pos[0] < bbox[2] and bbox[1] <= pos[1] < bbox[3]


def distance_to_box(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> float:
    dx = max(bbox[0] - pos[0], 0, pos[0] - bbox[2])
    dy = max(bbox[1] - pos[1], 0, pos[1] - bbox[3])
    return math.hypot(dx, dy)


def cursor_start(
    bbox: tuple[int, int, int, int],
    screen_w: int,
    screen_h: int,
    regime: str,
    key: str,
) -> tuple[int, int]:
    """Deterministic start position by distance regime, guaranteed OUTSIDE the bbox.

      near:   >= 200 px from bbox centre at a seeded angle
      medium: >= 500 px
      far:    the screen mirror (sw-cx, sh-cy), when that lands outside the bbox

    Seeded from an md5 of `(key, regime)` rather than `hash()`, which is
    PYTHONHASHSEED-randomised and silently made the old runs unreproducible across
    processes. The minimum radius rises with the bbox half-diagonal (+30 px) so the
    unclipped sample is outside the box at any angle; eight deterministic angles
    are tried before falling back to the far screen corner, which handles targets
    against a screen edge whose clipped samples land back inside.

    Every regime — `far` included — goes through that containment ladder, and a bbox
    that admits no on-screen point outside itself raises rather than returning a
    start inside the target. The runtime caller is `_GroundingPreparation.prepare`,
    so the raise fails the episode loudly instead of scoring a degenerate
    reach-at-step-0: `in_bbox` true at step 0, `reach_frame` 1, reward 1.0 before the
    model has acted. Same ladder-then-raise shape as `rl.target_box.geometry.
    sample_cursor_start`.

    ⚠️ RE-BASELINES EVERY PUBLISHED FAR-REGIME REACH NUMBER. `far` used to return the
    bare mirror with no containment check at all, so for a target whose centre sits
    near the screen centre the start landed inside the target and the cell was already
    solved at step 0. Grounding runs with `require_unsolved_start=False`
    (`rl/grounding/harness.py:77`), so those episodes WERE scored. Measured incidence
    at 1920x1080 — 20-60 px elements 0.02%, 60-200 px widgets 0.11%, 200-600 px panels
    2.10%, 600-1400 px windows 42.90% (the defect is analytic; the rates are
    Monte-Carlo over synthetic target distributions). Correcting it moves the `far`
    start for exactly those targets. Far starts whose mirror was already outside the
    box are byte-unchanged, and `near`/`medium` are byte-unchanged except where the old
    corner fallback itself returned an in-box start. This is the same caution
    `rl/geometry.py:37-41` records for `BOX_EDGE_INCLUSIVE`: do not compare a
    far-regime reach number across this commit.
    """
    import hashlib

    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2

    def _on_screen(x: int, y: int) -> tuple[int, int]:
        return max(0, min(screen_w - 1, x)), max(0, min(screen_h - 1, y))

    if regime == "far":
        mirror = _on_screen(screen_w - cx, screen_h - cy)
        if not in_bbox(mirror, bbox):
            return mirror
    seed = int.from_bytes(
        hashlib.md5(f"{key}:{regime}:v0".encode()).digest()[:4], "big"
    )
    rng = random.Random(seed)
    # `far` overshoots both screen dimensions so the clamp parks the sample on the
    # furthest edge; near/medium keep their published radii.
    base = {"near": 200, "medium": 500, "far": max(screen_w, screen_h)}.get(regime, 200)
    span = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    dist = max(base, int(span / 2) + 30)
    for _ in range(8):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        sx, sy = _on_screen(
            cx + int(round(dist * math.cos(angle))), cy + int(round(dist * math.sin(angle)))
        )
        if not in_bbox((sx, sy), bbox):
            return sx, sy
    corners = [(0, 0), (screen_w - 1, 0), (0, screen_h - 1), (screen_w - 1, screen_h - 1)]
    outside = [c for c in corners if not in_bbox(c, bbox)]
    if not outside:
        raise ValueError(
            f"no on-screen cursor start lies outside bbox {bbox} on a {screen_w}x{screen_h} "
            f"screen (regime {regime!r}, key {key!r}): every screen corner is inside the "
            "target, so any start would score a reach at step 0"
        )
    # The per-axis furthest screen extreme maximises both legs of the distance at
    # once, so this is the furthest corner; it is preferred verbatim to keep the
    # pre-fix near/medium fallback byte-identical wherever it was already admissible.
    furthest = (
        0 if cx > screen_w // 2 else screen_w - 1,
        0 if cy > screen_h // 2 else screen_h - 1,
    )
    if furthest in outside:
        return furthest
    return max(outside, key=lambda c: distance_to_box(c, bbox))


# --------------------------------------------------------------------------- #
# tasksets
# --------------------------------------------------------------------------- #


class OSWorldTasksetConfig(vf.TasksetConfig):
    """A split file (`{app: [task_id, ...]}`) or an explicit list of task JSONs.

    NO-LEAK: point `split_path` only at the training split when this feeds RL. The
    held-out split is eval-only, and a benchmark used for eval should ideally never
    appear in training at all.
    """

    osworld_root: str = ""
    split_path: str = ""
    task_paths: list[str] = []
    max_steps: int = 15
    max_tasks: int = 0
    resume_dir: str = ""
    """If set, skip tasks that already have `<resume_dir>/<app>/<id>/result.json`.
    `osworld_fullbench_runner.py` alone had this, and an interrupted 369-task array
    run is exactly when you need it."""


class OSWorldTaskset(vf.Taskset[DesktopTask, OSWorldTasksetConfig]):
    def load(self) -> Iterable[DesktopTask]:
        root = Path(self.config.osworld_root or os.environ.get("OSWORLD_ROOT", ""))
        paths: list[tuple[str | None, Path]] = []
        if self.config.split_path:
            split = json.loads(Path(self.config.split_path).read_text())
            for app, ids in sorted(split.items()):
                for task_id in ids:
                    paths.append(
                        (app, root / "evaluation_examples" / "examples" / app / f"{task_id}.json")
                    )
        paths.extend((None, Path(p)) for p in self.config.task_paths)
        emitted = 0
        for idx, (app, path) in enumerate(paths):
            payload = json.loads(path.read_text())
            task_id = str(payload.get("id") or path.stem)
            app_name = app or path.parent.name
            if self.config.resume_dir and (
                Path(self.config.resume_dir) / app_name / task_id / "result.json"
            ).exists():
                _LOGGER.info("resume: skipping %s/%s", app_name, task_id)
                continue
            if self.config.max_tasks and emitted >= self.config.max_tasks:
                return
            emitted += 1
            yield DesktopTask(
                DesktopTaskData(
                    idx=idx,
                    name=task_id,
                    prompt=str(payload["instruction"]),
                    instruction=str(payload["instruction"]),
                    kind="osworld",
                    max_steps=self.config.max_steps,
                    app=app_name,
                    snapshot=str(payload.get("snapshot") or "") or None,
                    task_path=str(path),
                    setup={"config": list(payload.get("config", []))},
                ),
                self.config.task,
            )


class GroundingTasksetConfig(vf.TasksetConfig):
    bboxes_jsonl: str = ""
    osworld_root: str = ""
    regimes: list[str] = list(REGIMES)
    max_steps: int = 100
    """K — rollout length per (target, regime). 100 = 10 s at 10 fps."""
    target_idxs: list[int] = []
    max_targets: int = 0
    replay_n_steps: int = 1


class GroundingTaskset(vf.Taskset[DesktopTask, GroundingTasksetConfig]):
    """bboxes.jsonl x {near, medium, far}. One task per (target, regime).

    The old runner nested two loops inside one process and rebooted a VM per
    rollout; here the cross-product *is* the taskset, so verifiers shards it and
    the pool supplies VMs.
    """

    def load(self) -> Iterable[DesktopTask]:
        root = Path(self.config.osworld_root or os.environ.get("OSWORLD_ROOT", ""))
        keep = set(self.config.target_idxs)
        idx = 0
        seen_targets = 0
        for line in Path(self.config.bboxes_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            label = json.loads(line)
            if keep and int(label["idx"]) not in keep:
                continue
            if self.config.max_targets and seen_targets >= self.config.max_targets:
                return
            seen_targets += 1
            image_path = Path(label["image_path"])
            parts = image_path.parts
            if parts[-2] != "steps":
                raise ValueError(f"unexpected image_path shape: {label['image_path']!r}")
            task_id = parts[-3]
            app = str(label["app"])
            bbox = tuple(int(v) for v in label["bbox_xyxy"])
            trajectory = image_path.parent.parent / "traj.jsonl"
            config_path = root / "evaluation_examples" / "examples" / app / f"{task_id}.json"
            for regime in self.config.regimes:
                yield DesktopTask(
                    DesktopTaskData(
                        idx=idx,
                        name=f"{app}/{task_id}/{regime}",
                        prompt=str(label["instruction"]),
                        instruction=str(label["instruction"]),
                        kind="grounding",
                        max_steps=self.config.max_steps,
                        app=app,
                        task_path=str(config_path) if config_path.is_file() else None,
                        bbox=bbox,
                        regime=regime,
                        expected={"bbox": list(bbox)},
                        setup={
                            "replay_trajectory": (
                                str(trajectory) if trajectory.is_file() else None
                            ),
                            "replay_n_steps": self.config.replay_n_steps,
                            "image_path": str(image_path),
                        },
                    ),
                    self.config.task,
                )
                idx += 1


class FreerollTasksetConfig(vf.TasksetConfig):
    instructions: list[str] = []
    instructions_file: str = ""
    desktop_setup: str = "none"
    max_steps: int = 60


class FreerollTaskset(vf.Taskset[DesktopTask, FreerollTasksetConfig]):
    """Goal-only episodes with no state oracle — the qualitative probe.

    `freeroll.py` took several newline-separated instructions and looped them in
    one process, rebooting the VM between each. One instruction is one task now;
    blank and `#`-comment lines are still dropped, and an empty list still yields a
    single no-goal rollout.
    """

    def load(self) -> Iterable[DesktopTask]:
        raw = list(self.config.instructions)
        if self.config.instructions_file:
            raw.extend(Path(self.config.instructions_file).read_text().splitlines())
        cleaned = [
            line.strip()
            for line in raw
            if line.strip() and not line.strip().startswith("#")
        ]
        for idx, instruction in enumerate(cleaned or [""]):
            yield DesktopTask(
                DesktopTaskData(
                    idx=idx,
                    name=f"task_{idx:02d}_{_slug(instruction)}",
                    prompt=instruction or None,
                    instruction=instruction,
                    kind=self.config.desktop_setup,
                    max_steps=self.config.max_steps,
                ),
                self.config.task,
            )


def _slug(text: str) -> str:
    import re

    base = re.sub(r"[^a-z0-9]+", "-", (text or "no-instruction").lower()).strip("-")
    return base[:40] or "task"
