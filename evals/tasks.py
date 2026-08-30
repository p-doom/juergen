"""Task rows, rollout state, and the tasksets that enumerate them.

A family differs from another in what it does to the VM before the loop and what
it reads out of it afterwards. Those two things are the `Preparer` seam below;
everything else is one harness.

  * freeroll               -> `none` / `terminal`
  * fullbench / one_task   -> `osworld` (OSWorld `SetupController`, scored by
                              `DesktopEnv.evaluate()`)
  * CUA-Gym desktop        -> `cua_gym_desktop`
  * sign-of-life           -> four kinds registered by `evals/signoflife/guest.py`

A `Preparer` is the only place a family-specific in-VM side effect and trusted
final score may happen. The harness never branches on task kind.
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

from evals.osworld_assets import stage_offline_task

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "FULL_SUCCESS_THRESHOLD",
    "PREPARERS",
    "DesktopState",
    "DesktopTask",
    "DesktopTaskData",
    "FreerollTaskset",
    "FreerollTasksetConfig",
    "OSWorldTaskset",
    "OSWorldTasksetConfig",
    "Preparer",
    "cursor_start",
    "in_bbox",
    "osworld_task_config",
    "register_preparer",
    "unscored_evaluate",
    "valid_result",
]

RESULT_KEY = "episode"
RESULT_SCHEMA_VERSION = 1

FULL_SUCCESS_THRESHOLD = 1.0
"""The bar an OSWorld `evaluate()` score must clear to count the task as solved.

`DesktopEnv.evaluate()` returns 1 only when every conjoined metric passed; a
partly-failed list returns `sum(results) / len(results)`
(`desktop_env/desktop_env.py:509`).
"""


def valid_result(trace: vf.Trace, family: str) -> dict[str, Any]:
    """The episode result the rollout published, or raise.

    Every family's sparse reward opens with this, and raising is the point: an
    infrastructure failure publishes the initial `reach_frame` of -1 and no
    `steps_detail`, so returning 0.0 trains a booted-VM failure as an ordinary miss
    — or, wherever a no-move penalty applies, as the worst thing the policy can do.
    One throwing reward drops the whole group (`Task.score`'s `asyncio.gather`), so
    the sparse reward covers its family's shaping term too.
    """
    result = trace.info.get(RESULT_KEY)
    if not isinstance(result, dict):
        raise RuntimeError(f"{family} rollout published no result")
    if result.get("validity") != "valid":
        raise RuntimeError(
            f"{family} rollout is infrastructure-invalid: {result.get('infra_error')}"
        )
    return result


class DesktopTaskData(vf.TaskData):
    """One desktop episode's row.

    `kind` selects the `Preparer`; `expected` is whatever that family's oracle
    needs; `setup` is static per-task setup input (an OSWorld task config path, a
    replay trajectory, a bbox). Nothing here is grammar-specific: the codec is a
    harness config field, so the same row runs under every grammar.
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
    """True for a cell whose success requires not pressing Return. Read by the
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
    ignored_after_terminate: int = 0
    """Calls the rollout emitted after its own terminate; see
    `agent.agent._calls_after_terminate`."""
    released_holds: list[dict[str, Any]] = []
    """The presses the rollout left down and the harness lifted at teardown, before
    `evaluate()` ran. Empty for a rollout that released its own."""
    screen_size: list[int] | None = None
    """The guest display the codec resolved its coordinates against."""
    infra_valid: bool = True
    infra_error: dict[str, str] | None = None
    codec: str | None = None
    render_spec_id: str | None = None
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
    `evals/indicators.py`; a concrete taskset composes the ones its family can
    actually evaluate. The base stays empty so a taskset never inherits a reward
    that raises for lack of evidence: one throwing reward inside `Task.score`'s
    `asyncio.gather` drops the whole group's rewards.
    """


@runtime_checkable
class Preparer(Protocol):
    """Family-specific VM preparation, state extraction, and final scoring.

    `prepare` may drive the guest (launch a terminal, run OSWorld setup commands,
    replay a cached trajectory, move the cursor). `probe` must not: it is the
    read-only path the oracle depends on. `evaluate` is the one trusted scorer.
    """

    kind: str

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]: ...
    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]: ...
    def evaluate(
        self, session: Any, task: DesktopTaskData, *, declared: str | None
    ) -> float: ...


def unscored_evaluate(
    session: Any, task: DesktopTaskData, *, declared: str | None
) -> float:
    """Refuse an evaluator flag where a family has no trusted guest scorer."""

    del session, declared
    raise LookupError(
        f"evaluate_on_finish is unsupported for task kind {task.kind!r}; "
        "its preparer has no trusted guest evaluator"
    )


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
    evaluate = staticmethod(unscored_evaluate)

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del session, task
        return {"prepared": "none"}

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {"cursor": list(session.cursor_position()), "screen": list(session.screen_size())}


class _TerminalPreparation:
    """freeroll `--desktop_setup terminal`.

    Only typing evals need it: start with a focused terminal, cleared with
    `ctrl-l`, so the first model turn is not spent finding one.
    """

    kind = "terminal"
    evaluate = staticmethod(unscored_evaluate)

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

    Setup goes through the session rather than `DesktopEnv.reset(task_config=...)`
    so the VM lifecycle stays under our qemu+KVM control: OSWorld's apptainer
    provider strips KVM ioctls on the hai-* nodes.
    """

    kind = "osworld"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        task_config = osworld_task_config(task)
        if not task_config.get("config") and not task_config.get("evaluator"):
            return {"prepared": "osworld", "steps": 0}
        steps = session.setup(task_config)
        return {
            "prepared": "osworld",
            "steps": len(task_config.get("config") or []) if steps is None else int(steps),
            "scorable": bool(task_config.get("evaluator")),
        }

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {"cursor": list(session.cursor_position())}

    def evaluate(
        self, session: Any, task: DesktopTaskData, *, declared: str | None
    ) -> float:
        del task
        session.declare_terminal(declared)
        return session.evaluate()


def osworld_task_config(task: DesktopTaskData) -> dict[str, Any]:
    """The whole OSWorld task JSON a row stands for.

    Whole, not just its `config` list, because the `evaluator` block travels with
    it: `DesktopFacade.setup()` binds both at once so `evaluate()` can stay
    argument-free (`evals/vm.py`). Precedence is inline row -> file on disk, so a
    synthetic row still works and a benchmark row does not need its 369 JSONs
    copied into the taskset.
    """
    inline = task.setup.get("task_config")
    if isinstance(inline, dict):
        return dict(inline)
    if task.task_path:
        payload = json.loads(Path(task.task_path).read_text())
        if isinstance(payload, dict):
            return payload
    config = task.setup.get("config")
    return {
        "id": task.name or f"task_{task.idx:04d}",
        "instruction": task.instruction,
        "config": list(config) if isinstance(config, list) else [],
    }


for _preparer in (
    _NoPreparation(),
    _TerminalPreparation(),
    _OSWorldPreparation(),
):
    register_preparer(_preparer)


def in_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    """Half-open on the max edge."""
    return bbox[0] <= pos[0] < bbox[2] and bbox[1] <= pos[1] < bbox[3]


def distance_to_box(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> float:
    dx = max(bbox[0] - pos[0], 0, pos[0] - bbox[2])
    dy = max(bbox[1] - pos[1], 0, pos[1] - bbox[3])
    return math.hypot(dx, dy)


# Reached only through `rl.grounding.dataset.cursor_start`, which delegates here so the
# container-free plugin cannot drift from this rule. There is no direct caller left in
# `evals/`, so a dead-code sweep will read this as unused: it is not.
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
    PYTHONHASHSEED-randomised and so not reproducible across processes. The
    minimum radius rises with the bbox half-diagonal (+30 px) so the
    unclipped sample is outside the box at any angle; eight deterministic angles
    are tried before falling back to the far screen corner, which handles targets
    against a screen edge whose clipped samples land back inside.

    Every regime — `far` included — goes through that containment ladder, and a bbox
    that admits no on-screen point outside itself raises rather than returning a
    start inside the target. The caller is `rl.grounding.taskset.GroundingTaskset.
    load`, so the raise fails enumeration instead of scoring a degenerate
    reach-at-step-0: `in_bbox` true at step 0, `reach_frame` 1, reward 1.0 before the
    model has acted. Same ladder-then-raise shape as `rl.target_box.geometry.
    sample_cursor_start`.

    ⚠️ A bare mirror degenerates for window-sized targets, which is why `far` goes
    through the ladder too. Without a containment check, a target whose centre sits
    near the screen centre gets a start inside itself, and grounding runs with
    `require_unsolved_start=False` (`rl/grounding/harness.py:77`), so such an
    episode scores rather than raising. Incidence at 1920x1080 — 20-60 px elements
    0.02%, 60-200 px widgets 0.11%, 200-600 px panels 2.10%, 600-1400 px windows
    42.90% (analytic defect; Monte-Carlo rates over synthetic target distributions).
    Re-check whenever the target set grows toward window-sized targets.
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


class OSWorldTasksetConfig(vf.TasksetConfig):
    """A split file (`{app: [task_id, ...]}`) or an explicit list of task JSONs.

    Leakage: point `split_path` only at the training split when this feeds RL. The
    held-out split is eval-only, and a benchmark used for eval should ideally never
    appear in training at all.
    """

    osworld_root: str = ""
    split_path: str = ""
    task_paths: list[str] = []
    asset_bundle: str = ""
    max_steps: int = 15
    max_tasks: int = 0
    resume_dir: str = ""
    """If set, skip tasks that already have `<resume_dir>/<app>/<id>/result.json`,
    for resuming an interrupted 369-task array run."""


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
        rows: list[DesktopTask] = []
        for idx, (app, path) in enumerate(paths):
            payload = json.loads(path.read_text())
            task_id = str(payload.get("id") or path.stem)
            app_name = app or path.parent.name
            if self.config.resume_dir and (
                Path(self.config.resume_dir) / app_name / task_id / "result.json"
            ).exists():
                _LOGGER.info("resume: skipping %s/%s", app_name, task_id)
                continue
            if self.config.max_tasks and len(rows) >= self.config.max_tasks:
                break
            payload = stage_offline_task(payload, self.config.asset_bundle)
            rows.append(DesktopTask(
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
                    # The whole JSON, not just `config`: the `evaluator` block is
                    # what `DesktopFacade.evaluate()` scores, and re-reading the
                    # file inside the preparer would make the row and the score
                    # disagree the moment the checkout moves under a running array.
                    setup={
                        "task_config": payload,
                        "config": list(payload.get("config", [])),
                    },
                ),
                self.config.task,
            ))
        yield from rows


class FreerollTasksetConfig(vf.TasksetConfig):
    instructions: list[str] = []
    instructions_file: str = ""
    desktop_setup: str = "none"
    max_steps: int = 60


class FreerollTaskset(vf.Taskset[DesktopTask, FreerollTasksetConfig]):
    """Goal-only episodes with no state oracle — the qualitative probe.

    One instruction is one task. Blank and `#`-comment lines are dropped, and an
    empty list yields a single no-goal rollout.
    """

    def load(self) -> Iterable[DesktopTask]:
        # `desktop_setup` is an unvalidated string copied straight onto `kind`, so an
        # unregistered one (`grounding`, until its preparer was deleted) otherwise
        # reaches `preparer_for` only after a VM has booted.
        preparer_for(self.config.desktop_setup)
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
