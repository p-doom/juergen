"""The one episode driver.

The loop:

    screenshot -> shared render contract -> sample
    -> codec.parse -> codec.compile -> desktop executes -> oracle -> repeat

A task family is a taskset plus a `Preparer`; an arm is a `DesktopHarnessConfig`.

What this file does not own, and who does: the sglang lifecycle and the endpoint
(verifiers), qemu boot and port allocation (the desktop pool), the CLI and config
(verifiers), and `AsyncOpenAI` construction and sampling (`agent.Agent`).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import socket
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Protocol

import verifiers.v1 as vf
from desktop.execute.guest_program import HeldStateError
from pydantic import Field, SecretStr, model_validator

import grammars
import stream_cuagym_qwen35 as stream_render
from agent.agent import (
    Agent,
    Decision,
    EffectiveSampling,
    ModelCallError,
    build_transport,
    dump_prompt,
    load_codec,
)
from agent.desktop import DEFAULT_SLOT_DIR, PoolSpec, default_pool_factory, pool_for
from evals.cua_gym.web.runtime import CuaGymWebPreparer, CuaGymWebTaskData
from evals.tasks import (
    FULL_SUCCESS_THRESHOLD,
    RESULT_KEY,
    RESULT_SCHEMA_VERSION,
    DesktopState,
    DesktopTaskData,
    distance_to_box,
    in_bbox,
    preparer_for,
)
from harness_render import HarnessRenderer

_LOGGER = logging.getLogger(__name__)

__all__ = ["CuaGymWebConfig", "DesktopHarness", "DesktopHarnessConfig"]


class Desktop(Protocol):
    """The session surface an episode needs from `desktop.vm.pool`.

    Everything optional is probed with `getattr`, so a pool that cannot settle a
    frame or evaluate an OSWorld task still runs the families that do not need it.
    """

    def screen_size(self) -> tuple[int, int]: ...
    def cursor_position(self) -> tuple[int, int]: ...
    def screenshot(self) -> bytes: ...
    def execute_atomic(self, operations: Any) -> Receipt: ...


class Receipt(Protocol):
    """The guest's own account of one dispatched action.

    `desktop.execute.guest_program.AtomicExecutionResult` is the real one, reached
    through `evals/vm.py`'s adapter; the in-process canvas (`rl/desktop.py`)
    implements the same four members. Written down rather than left implied because
    the return value used to be assigned and dropped, so any shape satisfied it —
    and a session that reports nothing publishes a turn in which a failed action
    looks exactly like a successful one.
    """

    ok: bool
    failure_kind: str | None
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]


class SettleConfig(vf.BaseConfig):
    """Post-action wait before the next screenshot.

    `stability_timeout_s > 0` polls the framebuffer until two consecutive frames
    are identical, adapting the wait to how long the UI actually takes. `per_kind`
    exists because one family needed 2.0 s (launching Chrome) where the rest need
    0.75 s, and a global 2.0 s tripled a 100-step grounding rollout.
    """

    min_delay_s: float = Field(default=0.75, ge=0.0)
    stability_timeout_s: float = Field(default=0.0, ge=0.0)
    poll_s: float = Field(default=0.1, gt=0.0)
    per_kind: dict[str, float] = Field(default_factory=lambda: {"open_chrome": 2.0})


class ScriptedConfig(vf.BaseConfig):
    """Oracle / negative-control cells: no model, the same codec and executor.

    The oracle arm must read 4/4 and the negative arm 0/4 through the same parse
    and compile path the model arm uses.
    """

    enabled: bool = False
    negative: bool = False


class BudgetConfig(vf.BaseConfig):
    """Hard per-episode ceilings: model turns, dispatched operations, output
    tokens, wall time, consecutive unparseable turns. `0` is off, on all five.
    """

    model_turns: int = Field(default=0, ge=0)
    operations: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    wall_time_s: float = Field(default=0.0, ge=0.0)
    consecutive_parse_errors: int = Field(default=0, ge=0)
    """Unparseable turns in a row before the episode ends; the streak resets on any
    turn that parsed.

    Off by default, unlike the other four, and deliberately: `parse_error_rate` is
    measured over the whole episode, so an arm studying a parse collapse has to be
    able to run through one. It is the arms that pay for a VM per cell — an OSWorld
    array, a sign-of-life gate — that set it, because `model_turns` cannot separate
    a checkpoint that acted for 60 turns from one that emitted 60 turns nothing
    could read."""


class ArtifactConfig(vf.BaseConfig):
    """Where the run's browsable artifact lands.

    There is no registration knob. `labctl register-external` accepts only
    `--alias/--path/--kind/--cluster`, so no invocation of it can populate
    `metadata.result` — and the rollout viewer is gated on
    `metadata.result.traj_path`. The runner copies the whole marker file into
    `metadata.result` for an `eval_result` output (`runner.rs:1885-1891`), so
    registration goes through the recipe's `[outputs]` block with
    `marker = "result.json"` pointing at `output_dir`.
    """

    output_dir: str = ""
    save_frames: bool = True
    save_prompts: bool = True
    write_gif: bool = True
    write_result_json: bool = True

    @model_validator(mode="after")
    def _the_artifact_needs_its_frames(self) -> ArtifactConfig:
        if self.write_result_json and not self.save_frames:
            raise ValueError(
                "write_result_json emits the trajectory labctl reads, and every row "
                "of it is fetched by frame filename; save_frames=False would publish "
                "an artifact whose every frame is a 404"
            )
        return self


class DesktopPoolConfig(vf.BaseConfig):
    key: str = "default"
    max_node_slots: int = Field(default=14, ge=1)
    slot_dir: str = ""
    pool_idle_ttl_s: float = Field(default=900.0, ge=0.0)
    acquire_timeout_s: float = Field(default=1800.0, gt=0.0)
    reap_interval_s: float = Field(default=15.0, gt=0.0)
    """How often the reaper looks for an idle pool."""
    session_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Passed verbatim to the session-pool constructor named by `pool_target`."""
    pool_target: str = "evals.vm:kvm_desktop_pool"
    """The session-pool constructor, as `module:attribute`.

    A constructor, not a provider name. Override it to inject a fake pool, not to
    select a VM backend."""
    hide_gpu_during_boot: bool = True
    """Blank `CUDA_VISIBLE_DEVICES` while the VM boots: the process that forks
    qemu may also hold a GPU, and a child that inherits the visible device can
    wedge the allocation."""


class CuaGymWebConfig(vf.BaseConfig):
    hub_image: str = ""
    apptainer_binary: str = "/usr/bin/apptainer"
    port_lock_dir: str = ""
    guest_password: SecretStr = SecretStr("")

    def runtime_values(self) -> tuple[Path, Path, Path, str]:
        values: list[Path] = []
        for name in ("hub_image", "apptainer_binary", "port_lock_dir"):
            raw = getattr(self, name)
            if not raw:
                raise ValueError(f"web.{name} is required for a CUA-Gym web task")
            path = Path(raw)
            if not path.is_absolute():
                raise ValueError(f"web.{name} must be an absolute path")
            values.append(path)
        hub_image, apptainer_binary, port_lock_dir = values
        if not hub_image.is_file():
            raise FileNotFoundError(f"CUA-Gym Hub image is missing: {hub_image}")
        if not apptainer_binary.is_file() or not os.access(apptainer_binary, os.X_OK):
            raise FileNotFoundError(
                f"CUA-Gym apptainer executable is missing: {apptainer_binary}"
            )
        password = self.guest_password.get_secret_value()
        if not password or any(character in password for character in "\0\r\n"):
            raise ValueError("web.guest_password must be non-empty and single-line")
        return hub_image, apptainer_binary, port_lock_dir, password


class DesktopHarnessConfig(vf.HarnessConfig):
    settle: SettleConfig = SettleConfig()
    scripted: ScriptedConfig = ScriptedConfig()
    budget: BudgetConfig = BudgetConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    pool: DesktopPoolConfig = DesktopPoolConfig()
    web: CuaGymWebConfig = CuaGymWebConfig()
    max_steps: int = Field(default=0, ge=0)
    """Overrides the task's own `max_steps` when > 0."""
    max_tokens: int = Field(default=256, ge=1, strict=True)
    """Fallback only — used for a knob `ctx.sampling` leaves unset."""
    model_request_timeout_s: float = Field(default=180.0, gt=0.0, allow_inf_nan=False)
    """One model request's inactivity timeout.

    The sign-of-life supervisor also uses this declared value when deriving the
    outer attempt deadline. A server that keeps yielding bytes can outlive an
    inactivity timeout, so this is not itself the attempt deadline.
    """
    temperature: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    """The arm's sampling temperature, which `evals/signoflife/__main__.py` promotes
    into the eval's `sampling` block; `ctx.sampling` still wins at the wire
    (`agent.agent.resolve_sampling`). `None` names no temperature, which is what a
    scripted arm must do and what the validator below enforces: it renders its own
    action and calls no model, so a number here would be recorded in the run and
    never sent.

    There is deliberately no default for a model arm, because greedy is not a
    neutral choice. Measured on the eov3 relative family: temperature 0 confines
    100% of mouse deltas to {0, ±1, ±10, ±100} and lands no click within 1000 px of
    its target, which scores the decoder rather than the checkpoint; at 0.7 the
    on-lattice share is 3.9%, clicks land 36-49 px out, and the target application
    opens. So `evals/signoflife/__main__.py` refuses a model arm that names a
    temperature neither here nor on the command line, rather than supplying one of
    its own."""
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, allow_inf_nan=False)
    """`temperature`'s sibling, same contract. 1.0 is the no-op value, not the
    absence: an arm names it so the wire body carries the arm's nucleus setting and
    not the server's."""
    stop_on_click: bool = False
    """End the episode at the first left-button press, turning a free rollout into
    a single-decision probe."""
    require_unsolved_start: bool = True
    """Refuse to score a cell whose postcondition already holds before the first
    action."""
    evaluate_on_finish: bool = False
    """Call the task preparer's trusted evaluator and publish `task_reward`."""
    prefer_context_transport: bool = False
    """Sample through `ctx.client` instead of posting to `endpoint`."""

    @model_validator(mode="after")
    def _a_scripted_arm_names_no_sampling(self) -> DesktopHarnessConfig:
        if self.scripted.enabled and (
            self.temperature is not None or self.top_p is not None
        ):
            raise ValueError(
                "a scripted arm renders its own action and never calls a model, so a "
                "temperature or top_p here would be published as the run's sampling "
                "and never sent to anything"
            )
        return self


@dataclass
class _Budget:
    config: BudgetConfig
    started: float = 0.0
    model_turns: int = 0
    operations: int = 0
    output_tokens: int = 0
    consecutive_parse_errors: int = 0
    failure: str | None = None

    def __post_init__(self) -> None:
        self.started = time.monotonic()

    def turn(self) -> None:
        self.model_turns += 1
        self._check("model_turns", self.model_turns, self.config.model_turns)

    def dispatched(self, count: int) -> None:
        self.operations += count
        self._check("operations", self.operations, self.config.operations)

    def tokens(self, count: int) -> None:
        self.output_tokens += count
        self._check("output_tokens", self.output_tokens, self.config.output_tokens)

    def parsed(self, ok: bool) -> None:
        self.consecutive_parse_errors = 0 if ok else self.consecutive_parse_errors + 1
        self._check(
            "consecutive_parse_errors",
            self.consecutive_parse_errors,
            self.config.consecutive_parse_errors,
        )

    def _check(self, name: str, used: int, limit: int) -> None:
        if limit and used > limit and self.failure is None:
            self.failure = f"{name}_exceeded"
        self.clock()

    def clock(self) -> None:
        if (
            self.config.wall_time_s
            and time.monotonic() - self.started > self.config.wall_time_s
            and self.failure is None
        ):
            self.failure = "wall_time_exceeded"

    def snapshot(self) -> dict[str, Any]:
        return {
            "model_turns": self.model_turns,
            "operations": self.operations,
            "output_tokens": self.output_tokens,
            "consecutive_parse_errors": self.consecutive_parse_errors,
            "wall_time_s": round(time.monotonic() - self.started, 3),
            "failure": self.failure,
        }


async def _await_owned(awaitable: Awaitable[Any]) -> Any:
    """Finish owned work before propagating cancellation."""
    task = asyncio.ensure_future(awaitable)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except BaseException:  # noqa: BLE001,S110 - owned by the task, re-raised below
            pass
    if cancelled is not None:
        raise cancelled from task.exception()
    return task.result()


async def _to_thread(function: Any, *args: Any) -> Any:
    """Run a blocking guest call without abandoning its thread on cancellation."""
    return await _await_owned(asyncio.to_thread(function, *args))


@contextlib.contextmanager
def _hidden_gpu(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _geometry(session: Any) -> Any:
    """The `desktop.geometry.DisplayGeometry` a codec compiles against.

    Not a `(w, h)` pair: `codec.compile` clamps against the full display
    description, and handing it a bare size puts the clamp back on the caller,
    which is where the coordinate bugs lived.
    """
    from desktop.geometry import DisplayGeometry  # type: ignore[import-not-found]

    # Field names are `desktop_width` / `desktop_height`, verbatim from Harbor.
    width, height = session.screen_size()
    return DisplayGeometry(desktop_width=int(width), desktop_height=int(height))


def _screenshot(session: Any, settle: SettleConfig, kind: str) -> bytes:
    delay = settle.per_kind.get(kind, settle.min_delay_s)
    if settle.stability_timeout_s > 0:
        settled = getattr(session, "screenshot_settled", None)
        if not callable(settled):
            raise LookupError(
                "settle.stability_timeout_s asks for framebuffer stability polling, "
                f"which {type(session).__name__} does not implement"
            )
        return settled(
            min_delay_s=delay,
            stability_timeout_s=settle.stability_timeout_s,
            poll_s=settle.poll_s,
        )
    if delay > 0:
        time.sleep(delay)
    return session.screenshot()


_RELEASE_OF = {"mouse_down": "mouse_up", "key_down": "key_up"}


def _update_held(held: list[tuple[str, tuple]], operations: Any) -> None:
    """Track the presses a dispatched turn left down, in press order.

    Our own prompt advertises `mouse_down` as a hold that survives the turn
    (`grammars/move_rel/codec.py:199`) and the guest honours it, so a rollout that
    never emits the matching release runs `evaluate()` — and the whole scoring
    window — with the button down.

    `click`, `drag` and the typing kinds press and release inside one operation, so
    only the explicit `*_down` / `*_up` pair is tracked.
    """
    for operation in operations:
        kind, args = operation.kind, tuple(operation.args)
        release = _RELEASE_OF.get(kind)
        if release is not None:
            held.append((release, args))
        elif (kind, args) in held:
            held.remove((kind, args))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str
            )
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def _write_gif(frames: list[bytes], path: Path) -> None:
    if len(frames) < 2:
        return
    import io

    from PIL import Image

    images = []
    for payload in frames:
        with Image.open(io.BytesIO(payload)) as handle:
            frame = handle.convert("RGB")
            if frame.width > 960:
                frame = frame.resize(
                    (960, int(frame.height * 960 / frame.width)), Image.LANCZOS
                )
            images.append(frame)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=300,
        loop=0,
        optimize=True,
    )


_TRAJECTORY = "trajectory.jsonl"
_FRAME = "step_{index:03d}.jpg"
_RESET = "<reset>"
_PROMOTED_STEP_KEYS = frozenset({"step", "raw_model_output", "sampling"})


def _receipt_record(receipt: Any) -> dict[str, Any] | None:
    """The guest's own account of one action. `None` = nothing was dispatched.

    A subset of `AtomicExecutionResult`, not its `as_dict()`: the full receipt
    carries per-primitive X injection evidence and timestamp tables, and this lands
    in every row of every trajectory. These four are the ones nothing else can
    supply. `ok` and `failure_kind` are the guest's verdict on whether the action
    happened — the record otherwise shows a plausible cursor pair for a click the
    guest reported as failed — and the cursor pair is observed inside the VM around
    the action, where the published `cursor_before`/`cursor_after` are two separate
    host round-trips taken before and well after it.

    Both pairs are published rather than one being preferred, because they are
    measured differently and a disagreement between them is itself a finding.
    """
    if receipt is None:
        return None
    if not hasattr(receipt, "ok"):
        raise TypeError(
            f"{type(receipt).__name__} is not an execution receipt: "
            "`session.execute_atomic` must return desktop's `AtomicExecutionResult` "
            "(`ok`, `failure_kind`, `cursor_before`, `cursor_after`). Publishing the "
            "guest's verdict is not optional — without it a failed action is "
            "indistinguishable from a successful one in the trajectory."
        )
    return {
        "ok": bool(receipt.ok),
        "failure_kind": receipt.failure_kind,
        "cursor_before": list(receipt.cursor_before),
        "cursor_after": list(receipt.cursor_after),
    }


def _trajectory_rows(
    steps_detail: list[dict[str, Any]], n_frames: int
) -> list[dict[str, Any]]:
    """The rollout rows labctl's viewer and `datasets/convert.py` both read.

    One row per frame: row 0 is the pre-action observation, row n the turn whose
    post-action screenshot is `step_{n:03d}.jpg`. Row ordinal, `step_num` and frame
    index are all the same number, because the viewer indexes `steps[frame]` and
    `convert.py:458` reads the frame a step SAW as `step_{step_num - 1:03d}.jpg`.
    Hence `n_steps + 1` frames, not `n_steps`.

    A turn whose post-action screenshot never happened — the executor died mid-turn,
    the token budget cut the turn off — has no frame, so it has no row; `result.json`
    still carries it in `steps_detail`.

    `action` is the raw model output, not a rendered summary: `convert.py:462` reads
    prose out of `action or response`, so condensing it there empties the prose
    channel for every record built from this rollout.

    An episode refused before its first observation has no frames and therefore no
    rows, not even the reset: `result.json` carries why it was refused.
    """
    if n_frames == 0:
        return []
    rows: list[dict[str, Any]] = [
        {
            "step_num": 0,
            "action": _RESET,
            "response": _RESET,
            "reward": 0.0,
            "done": False,
            "info": {"kind": "reset", "frame": _FRAME.format(index=0)},
        }
    ]
    for step in steps_detail[: max(0, n_frames - 1)]:
        info = {
            key: value for key, value in step.items() if key not in _PROMOTED_STEP_KEYS
        }
        info["parsed"] = info.pop("parsed_action")
        info["frame"] = _FRAME.format(index=step["step"])
        rows.append(
            {
                "step_num": step["step"],
                "action": step["raw_model_output"],
                "response": step["raw_model_output"],
                "reward": 1.0
                if (step.get("probe") or {}).get("postcondition_success")
                else 0.0,
                "done": False,
                "info": info,
            }
        )
    rows[-1]["done"] = True
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write, never append: a re-run into the same directory must replace."""
    path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8"
    )


def _assert_frame_set(steps_dir: Path, n_rows: int) -> None:
    """Exactly `step_000..step_{n-1}.jpg` files, with no other `step_*` entry.

    labctl never lists the directory. One endpoint formats `step_{n:03}.jpg` from the
    index it wants (`server.rs:1655`) while a sibling reports `frame_count` by
    counting every `*.jpg` (`server.rs:1606`), so a gap is a frame the viewer offers
    and cannot fetch, and a stray image — a longer earlier run into the same
    directory — offers frames that do not exist.
    """
    expected = {_FRAME.format(index=index) for index in range(n_rows)}
    present = {path.name for path in steps_dir.glob("step_*")}
    if present == expected:
        return
    raise RuntimeError(
        f"{steps_dir}: {len(present)} frame(s) for {n_rows} trajectory row(s) — "
        f"missing {sorted(expected - present)}, "
        f"unexpected {sorted(present - expected)}"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tri_state(value: bool | None) -> float | None:
    return None if value is None else (1.0 if value else 0.0)


def _succeeded(outcome: str, task_reward: float | None) -> bool:
    """The single place an episode's verdict is decided.

    A scored arm's verdict is its OSWorld score and nothing else: `outcome` carries
    the sign-of-life families' postcondition probe, which an OSWorld task never
    populates, so reading it on a benchmark arm reports every episode as a failure.
    """
    if task_reward is None:
        return outcome == "postcondition_reached"
    return task_reward >= FULL_SUCCESS_THRESHOLD


_CODEC: Any | None = None


def _codec() -> Any:
    """Process-level codec cache. Not on the harness instance: one `Harness` serves
    every rollout, so instance state is shared state."""
    global _CODEC
    if _CODEC is None:
        _CODEC = load_codec(grammars.GRAMMAR_NAME)
    return _CODEC


class DesktopHarness(vf.Harness[DesktopHarnessConfig]):
    SUPPORTS_MESSAGE_PROMPT = True

    def __init__(self, config: DesktopHarnessConfig) -> None:
        super().__init__(config)
        if not config.artifacts.output_dir:
            raise ValueError(
                "artifacts.output_dir is unset. It used to fall back to the system temp "
                "dir, which publishes the `<root>/result.json` a recipe registers as its "
                "`eval_result` into a directory that gets reaped -- /tmp on this cluster "
                "was wiped by an environment restart -- and makes two concurrent runs "
                "overwrite each other's index, since the root is shared. Arms declare "
                "the rest of the artifact contract; the dispatcher supplies this."
            )
        self._runs: dict[str, dict[str, Any]] = {}
        """Every episode this process published, keyed by artifact subdir.

        Instance state on purpose, unlike `_codec`: one harness instance serves one
        run, and the artifact-root `runs[]` index is a property of the run rather
        than of a rollout. `_persist` runs synchronously inside `_publish`, so
        concurrent rollouts in one event loop cannot interleave here.
        """

    def pool_factory(self) -> Any:
        """Build the underlying session pool.

        The one override point for an environment family. Real desktops come from
        `desktop.vm.pool.DesktopSessionPool`; the container-free RL envs return an
        in-process virtual desktop with the same session surface.
        """
        return default_pool_factory(
            dict(self.config.pool.session_kwargs), self.config.pool.pool_target
        )

    @vf.metric
    async def harness_provenance(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        metrics = {
            "scripted": 1.0 if self.config.scripted.enabled else 0.0,
            "negative_control": 1.0 if self.config.scripted.negative else 0.0,
            "infra_valid": 1.0 if result.get("validity") == "valid" else 0.0,
        }
        # Only a control arm has a value to conform to; see `_control_ok`.
        if self.config.scripted.enabled:
            metrics["control_conformant"] = float(result.get("control_ok") or 0.0)
        return metrics

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        if not runtime.is_local:
            raise ValueError(
                "DesktopHarness requires a host-local verifiers runtime; the "
                "prewarmed desktop pool is node-local"
            )
        del mcp_urls
        task = trace.task.data
        if not isinstance(task, DesktopTaskData):
            raise TypeError("DesktopHarness requires DesktopTaskData")
        if not isinstance(trace.state, DesktopState):
            raise TypeError("DesktopHarness requires DesktopState")

        # Everything resolvable from config, resolved before a VM is booted:
        # an unknown grammar, an unregistered task kind and a scripted arm on a
        # family with no gold plan are config errors, and discovering one at step 1
        # has already cost a boot and the cell's whole guest setup.
        codec = _codec()
        renderer = stream_render.renderer()
        preparer = preparer_for(task.kind)
        web_values: tuple[Path, Path, Path, str] | None = None
        if isinstance(preparer, CuaGymWebPreparer) and not isinstance(
            task, CuaGymWebTaskData
        ):
            raise TypeError("CUA-Gym web preparer requires CuaGymWebTaskData")
        if isinstance(task, CuaGymWebTaskData):
            if not isinstance(preparer, CuaGymWebPreparer):
                raise TypeError("CUA-Gym web task resolved to another preparer")
            if not self.config.evaluate_on_finish:
                raise ValueError("CUA-Gym web tasks require evaluate_on_finish=True")
            web_values = self.config.web.runtime_values()
        if self.config.scripted.enabled and not callable(
            getattr(preparer, "script_plan", None)
        ):
            raise LookupError(
                f"task kind {task.kind!r} has no scripted arm; scripted.enabled "
                "requires a preparer implementing script_plan() + render_step()"
            )
        render_metadata = stream_render.metadata()
        trace.info["render"] = render_metadata
        trace.info["images"] = self._image_report(render_metadata)

        spec = PoolSpec(
            key=self.config.pool.key,
            max_node_slots=self.config.pool.max_node_slots,
            slot_dir=self.config.pool.slot_dir or str(DEFAULT_SLOT_DIR),
            pool_idle_ttl_s=self.config.pool.pool_idle_ttl_s,
            reap_interval_s=self.config.pool.reap_interval_s,
            acquire_timeout_s=self.config.pool.acquire_timeout_s,
        )
        pool = pool_for(spec, self.pool_factory())

        lease = None
        failed = True
        error: str | None = None

        def acquire() -> None:
            # Stored, not returned: `_to_thread` re-raises a cancellation once the
            # thread has finished, which discards whatever the await would have
            # produced. A lease that never reaches `lease` is never released, so
            # its node slot stays checked out — a handful of cancellations wedge a
            # node at `max_node_slots`.
            nonlocal lease
            lease = pool.acquire(trace.id)

        try:
            with _hidden_gpu(self.config.pool.hide_gpu_during_boot):
                await _to_thread(acquire)
            trace.info["desktop_session"] = getattr(lease.session, "session_id", None)
            artifacts = self._artifact_dir(task)
            cleanup = None
            if web_values is not None:
                hub_image, apptainer_binary, port_lock_dir, guest_password = web_values
                preparer = preparer.episode(
                    session=lease.session,
                    task=task,
                    episode_id=hashlib.sha256(trace.id.encode("utf-8")).hexdigest(),
                    artifacts=artifacts,
                    hub_image=hub_image,
                    apptainer_binary=apptainer_binary,
                    port_lock_dir=port_lock_dir,
                    guest_password=guest_password,
                )
                cleanup = preparer.close
            await self._run(
                ctx,
                trace,
                task,
                lease.session,
                endpoint,
                secret,
                codec,
                renderer,
                preparer,
                artifacts,
                cleanup,
            )
            published = trace.info.get(RESULT_KEY) or {}
            if published.get("validity") == "infra_invalid":
                error = json.dumps(published.get("infra_error"), default=str)
            else:
                failed = False
            return vf.ProgramResult(0, "", "")
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if lease is not None:
                lease.release(failed=failed, error=error)

    async def _run(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        task: DesktopTaskData,
        session: Any,
        endpoint: str,
        secret: str,
        codec: Any,
        renderer: HarnessRenderer[bytes],
        preparer: Any,
        artifacts: Path,
        cleanup: Any,
    ) -> None:
        state = trace.state
        assert isinstance(state, DesktopState)
        budget = _Budget(self.config.budget)
        max_steps = self.config.max_steps or task.max_steps
        agent = Agent(
            codec=codec,
            renderer=renderer,
            transport=build_transport(
                endpoint=endpoint,
                secret=secret,
                prefer_context=self.config.prefer_context_transport,
                timeout_s=self.config.model_request_timeout_s,
            ),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        state.codec = codec.name
        state.render_spec_id = renderer.spec.spec_id
        state.scripted = self.config.scripted.enabled
        state.negative_control = self.config.scripted.negative

        steps_detail: list[dict[str, Any]] = []
        frames: list[bytes] = []
        held: list[tuple[str, tuple]] = []
        outcome = "max_steps"
        infra_error: dict[str, str] | None = None
        fatal_error: BaseException | None = None
        sampling_record: dict[str, Any] = {}
        setup_evidence: dict[str, Any] = {}

        try:
            setup_evidence = await _to_thread(preparer.prepare, session, task)
            geometry = await _to_thread(_geometry, session)
            # Published with the result because `datasets/convert.py:440-447` refuses
            # a rollout without it: every grammar resolves against the screen.
            state.screen_size = [geometry.desktop_width, geometry.desktop_height]
            initial = await _to_thread(preparer.probe, session, task)
            state.initial_probe = initial
            self._assert_unsolved(task, initial)

            frame = await self._observe(preparer, session, task)
            frames.append(frame)
            renderer.start(frame)
            if self.config.artifacts.save_frames:
                (artifacts / "steps").mkdir(parents=True, exist_ok=True)
                (artifacts / "steps" / "step_000.jpg").write_bytes(frame)

            script = (
                self._script_plan(preparer, task)
                if self.config.scripted.enabled
                else None
            )
            reach_frame = -1
            best_distance = -1.0
            probe = initial

            for step in range(1, max_steps + 1):
                budget.turn()
                if budget.failure:
                    outcome = f"budget_{budget.failure}"
                    break
                if trace.stop_condition is not None:
                    outcome = f"framework_stop_{trace.stop_condition}"
                    break
                cursor = tuple(await _to_thread(session.cursor_position))
                _LOGGER.info(
                    "turn start: trace=%s cell=%s turn=%d/%d",
                    trace.id,
                    task.name,
                    step,
                    max_steps,
                )
                decision, step_sampling = await self._decide(
                    agent,
                    ctx,
                    trace,
                    task,
                    step=step,
                    geometry=geometry,
                    cursor=cursor,
                    script=script,
                    preparer=preparer,
                    session=session,
                    codec=codec,
                    artifacts=artifacts,
                )
                _LOGGER.info(
                    "turn done: trace=%s cell=%s turn=%d/%d response=%s",
                    trace.id,
                    task.name,
                    step,
                    max_steps,
                    decision is not None,
                )
                # Only a real turn updates the provenance. Assigning unconditionally
                # erased it on the terminal non-turn: a scripted arm exhausting its
                # script is the normal end of every negative control, and it left
                # `sampling` empty in the published result, so `SamplingProvenance`
                # reported temperature -1.0 and no source for the whole arm.
                if step_sampling:
                    sampling_record = step_sampling
                if decision is None:
                    outcome = "script_exhausted"
                    break
                if decision.truncated:
                    # `max_tokens` is our knob, so this is a measurement that did
                    # not happen: dispatching the fragment, or scoring the rest of
                    # the rollout, would publish our own token cap as model
                    # behaviour. Distinct from `parse_errors`, which counts the
                    # system under test.
                    outcome = "truncated_action"
                    steps_detail.append(
                        self._record(decision, step, cursor, cursor, None, None, None)
                    )
                    break

                receipt: Any = None
                action_error: dict[str, Any] | None = None
                if decision.operations:
                    try:
                        receipt = await _to_thread(
                            session.execute_atomic, decision.operations
                        )
                        budget.dispatched(len(decision.operations))
                        _update_held(held, decision.operations)
                    except (TypeError, ValueError, HeldStateError) as exc:
                        action_error = {"type": type(exc).__name__, "message": str(exc)}
                        state.action_errors += 1
                    except Exception as exc:  # noqa: BLE001 - transport, fails closed
                        state.executor_errors += 1
                        infra_error = {
                            "stage": "execute",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        outcome = "executor_error"
                        steps_detail.append(
                            self._record(
                                decision, step, cursor, cursor, None, None, action_error
                            )
                        )
                        break
                if decision.parse_error:
                    state.parse_errors += 1
                budget.parsed(decision.parse_error is None)
                state.ignored_after_terminate += decision.ignored_after_terminate

                frame = await self._observe(preparer, session, task)
                frames.append(frame)
                if self.config.artifacts.save_frames:
                    (artifacts / "steps" / f"step_{step:03d}.jpg").write_bytes(frame)
                action_line = (
                    codec.format(decision.action)
                    if decision.parse_error is None and decision.action is not None
                    else None
                )
                renderer.complete(
                    assistant=decision.text.strip() or "NO_OP",
                    action=action_line,
                    next_image=frame,
                )
                cursor_after = tuple(await _to_thread(session.cursor_position))
                probe = await _to_thread(preparer.probe, session, task)
                steps_detail.append(
                    self._record(
                        decision,
                        step,
                        cursor,
                        cursor_after,
                        frame,
                        probe,
                        action_error,
                        receipt,
                    )
                )
                state.steps = step

                if probe.get("in_bbox") and reach_frame < 0:
                    reach_frame = step
                if task.bbox is not None:
                    distance = distance_to_box(cursor_after, task.bbox)
                    best_distance = (
                        distance if best_distance < 0 else min(best_distance, distance)
                    )

                verdict = self._postcondition_reached(task, probe)
                if verdict:
                    outcome = "postcondition_reached"
                    break
                if decision.terminated:
                    state.control_terminate = decision.control
                    state.terminate_step = step
                    outcome = f"model_{decision.control}_without_postcondition"
                    break
                if self.config.stop_on_click and _is_left_click(decision):
                    outcome = "click"
                    break
                if budget.failure:
                    outcome = f"budget_{budget.failure}"
                    break

            state.reach_frame = reach_frame
            state.best_distance = best_distance
        except ModelCallError as exc:
            # The orchestrator halts a rollout by stamping `stop_condition` and
            # refusing the call with a 400 (`interception/server.py:400-406`,
            # `:445-449`), which arrives here as a transport failure. Publishing
            # that as `infra_invalid` scores a healthy rollout the framework ended
            # as a booted-VM failure and drops it from N.
            if trace.stop_condition is None:
                infra_error = {
                    "stage": "model",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                outcome = "model_error"
            else:
                outcome = f"framework_stop_{trace.stop_condition}"
        except Exception as exc:
            infra_error = {
                "stage": "episode",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            outcome = "infrastructure_error"
            _LOGGER.exception("episode %s failed", task.name)
        except BaseException as exc:
            fatal_error = exc

        if fatal_error is None:
            try:
                state.released_holds = await self._release_held(session, held)
                await _to_thread(_screenshot, session, self.config.settle, task.kind)
                state.final_probe = await _to_thread(preparer.probe, session, task)
                if infra_error is None and not self.config.evaluate_on_finish:
                    terminal_success = state.final_probe.get("postcondition_success")
                    if terminal_success is True:
                        outcome = "postcondition_reached"
                    elif terminal_success is False and outcome == "postcondition_reached":
                        outcome = "postcondition_lost"
                if infra_error is None and self.config.evaluate_on_finish:
                    state.task_reward = await self._evaluate(
                        preparer,
                        session,
                        task,
                        declared=state.control_terminate,
                    )
                    if state.task_reward is None:
                        infra_error = {
                            "stage": "evaluate",
                            "type": "EvaluateFailed",
                            "message": "evaluate_on_finish returned no score",
                        }
            except Exception as exc:  # noqa: BLE001 - published as infra-invalid
                if infra_error is None:
                    infra_error = {
                        "stage": "episode",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    outcome = "infrastructure_error"
                _LOGGER.exception("episode %s cleanup failed", task.name)
            except BaseException as exc:
                fatal_error = exc

        if cleanup is not None:
            try:
                await _to_thread(cleanup)
            except Exception as exc:
                if fatal_error is None:
                    if infra_error is None:
                        infra_error = {
                            "stage": "cleanup",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    else:
                        infra_error["message"] += f"; cleanup also failed: {exc}"
                    outcome = "infrastructure_error"
                _LOGGER.exception("episode %s cleanup failed", task.name)
            except BaseException as exc:
                if fatal_error is None:
                    fatal_error = exc

        close_error: BaseException | None = None
        try:
            await _await_owned(agent.close())
        except BaseException as exc:
            close_error = exc
        if fatal_error is not None:
            raise fatal_error from close_error
        if close_error is not None:
            raise close_error

        self._publish(
            trace,
            state,
            outcome=outcome,
            infra_error=infra_error,
            sampling=sampling_record,
            setup=setup_evidence,
            budget=budget,
            steps_detail=steps_detail,
            frames=frames,
            artifacts=artifacts,
        )

    def _publish(
        self,
        trace: vf.Trace,
        state: DesktopState,
        *,
        outcome: str,
        infra_error: dict[str, str] | None,
        sampling: dict[str, Any],
        setup: dict[str, Any],
        budget: _Budget,
        steps_detail: list[dict[str, Any]],
        frames: list[bytes],
        artifacts: Path,
    ) -> None:
        """Publish one result shape for every family and arm.

        `trace.state` is `exclude=True` and never reaches `traces.jsonl`, so the same
        fields are mirrored into `trace.info[RESULT_KEY]`; that mirror is what every
        reward, metric and offline re-score reads.

        `success` is `None`, not `False`, on infrastructure failure: rewards raise on
        `None` so prime-rl drops the rollout instead of training a booted-VM failure
        as a task failure. Otherwise it is derived here from the `task_reward`
        published in the same dict, so the two cannot disagree.
        """
        state.outcome = outcome
        state.infra_error = infra_error
        state.infra_valid = infra_error is None
        state.success = (
            _succeeded(outcome, state.task_reward) if infra_error is None else None
        )
        state.temperature = sampling.get("temperature")
        state.temperature_source = sampling.get("temperature_source")

        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        trace.info[RESULT_KEY] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "validity": "valid" if infra_error is None else "infra_invalid",
            "codec": grammars.GRAMMAR_NAME,
            "instruction": task.instruction,
            "screen_size": state.screen_size,
            "render": trace.info["render"],
            "sampling": sampling,
            "images": trace.info["images"],
            "success": state.success,
            "outcome": outcome,
            "steps": state.steps,
            "parse_errors": state.parse_errors,
            "action_errors": state.action_errors,
            "executor_errors": state.executor_errors,
            "control_terminate": state.control_terminate,
            "terminate_step": state.terminate_step,
            "ignored_after_terminate": state.ignored_after_terminate,
            "released_holds": state.released_holds,
            "reach_frame": state.reach_frame,
            "best_distance": state.best_distance,
            "task_reward": state.task_reward,
            "initial_probe": state.initial_probe,
            "final_probe": state.final_probe,
            "setup": setup,
            "budget": budget.snapshot(),
            "infra_error": infra_error,
            "control_ok": _tri_state(self._control_ok(state)),
            "scripted": state.scripted,
            "negative_control": state.negative_control,
            "steps_detail": steps_detail,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "labctl_run_id": os.environ.get("LABCTL_RUN_ID"),
        }
        self._persist(artifacts, trace, frames)
        trace.stop(outcome)

    async def _decide(
        self,
        agent: Agent,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        task: DesktopTaskData,
        *,
        step: int,
        geometry: Any,
        cursor: tuple[int, int],
        script: list[Any] | None,
        preparer: Any,
        session: Any,
        codec: Any,
        artifacts: Path,
    ) -> tuple[Decision | None, dict[str, Any]]:
        """One decision: a scripted intent or a sampled turn, then parse + compile."""
        if script is not None:
            if step > len(script):
                return None, {}
            sampling = EffectiveSampling(
                model="scripted",
                temperature=None,
                max_tokens=None,
                top_p=None,
                stop=(),
                temperature_source="scripted",
                wire_body_keys=(),
            )
            # Rendered here, not up front: the relative arms resolve their delta
            # against a cursor read taken now.
            text = await _to_thread(
                self._render_step, preparer, session, task, codec, script[step - 1]
            )
            decision = agent.decide(
                text, step=step, geometry=geometry, cursor=cursor, sampling=sampling
            )
            return decision, sampling.as_dict()

        if self.config.artifacts.save_prompts:
            body = agent.build_body(instruction=task.instruction)
            (artifacts / "steps").mkdir(parents=True, exist_ok=True)
            (artifacts / "steps" / f"prompt_{step:03d}.json").write_text(
                dump_prompt(body)
            )
        decision = await agent.step(
            ctx,
            instruction=task.instruction,
            step=step,
            geometry=geometry,
            cursor=cursor,
            session_id=trace.id,
        )
        _LOGGER.info("step %d | response=%r", step, decision.text)
        return decision, decision.sampling.as_dict()

    def _record(
        self,
        decision: Decision,
        step: int,
        cursor_before: tuple[int, int],
        cursor_after: tuple[int, int],
        frame: bytes | None,
        probe: dict[str, Any] | None,
        action_error: dict[str, Any] | None,
        receipt: Any = None,
    ) -> dict[str, Any]:
        return {
            **decision.as_record(),
            "step": step,
            "cursor_before": list(cursor_before),
            "cursor_after": list(cursor_after),
            "frame_sha256": _sha256(frame) if frame else None,
            "probe": probe,
            "parse_ok": decision.parse_error is None,
            "action_error": action_error,
            "guest_receipt": _receipt_record(receipt),
        }

    async def _observe(
        self, preparer: Any, session: Any, task: DesktopTaskData
    ) -> bytes:
        """Settled screenshot, optionally post-processed by the family.

        `Preparer.observe` is how `target_box` draws its synthetic box onto every
        real screenshot. It is a harness-side hook rather than a session method
        because the annotation is a task property (which box) and the pool has no
        task; threading it through the session would mean mutating a shared,
        concurrently-checked-out object.
        """
        frame = await _to_thread(_screenshot, session, self.config.settle, task.kind)
        observe = getattr(preparer, "observe", None)
        if callable(observe):
            frame = await _to_thread(observe, frame, task)
        return frame

    def _script_plan(self, preparer: Any, task: DesktopTaskData) -> list[Any]:
        """The scripted arm's plan: intents, not yet rendered to text.

        Not rendered up front. `compact_raw.from_target` (and the relative renderers
        generally) need one fresh cursor read and are wrong if that read is stale,
        while `compact_absolute.from_target` needs only element geometry.
        Rendering the whole script before the first action would make every click
        after the first resolve against a stale cursor.
        """
        return list(preparer.script_plan(task, negative=self.config.scripted.negative))

    def _render_step(
        self,
        preparer: Any,
        session: Any,
        task: DesktopTaskData,
        codec: Any,
        intent: Any,
    ) -> str:
        """Render one intent into codec text, reading the cursor now.

        Codec text rather than operations directly, so the control arms exercise the
        same `parse` and `compile` the model arm does.
        """
        return preparer.render_step(session, task, codec=codec, intent=intent)

    def _postcondition_reached(
        self, task: DesktopTaskData, probe: dict[str, Any]
    ) -> bool:
        """Early stop when the family's postcondition is observable in-loop.

        The authoritative verdict is still the oracle reward; this only decides
        whether to keep spending turns. Grounding never stops early — it wants the
        full K frames so trajectory length is comparable across regimes.
        """
        del task
        return probe.get("postcondition_success") is True

    def _assert_unsolved(self, task: DesktopTaskData, probe: dict[str, Any]) -> None:
        if not self.config.require_unsolved_start:
            return
        if probe.get("postcondition_status") not in (None, "ok"):
            raise RuntimeError("task reset/setup produced unreadable initial state")
        if probe.get("postcondition_success") is True:
            raise RuntimeError(
                "task reset/setup did not begin in a valid unsolved state"
            )
        if task.bbox is not None and in_bbox(
            tuple(probe.get("cursor") or (-1, -1)), task.bbox
        ):
            # Grounding records this instead of refusing: the cursor-start sampler
            # already guarantees an outside-bbox start except against a screen edge,
            # and a refusal there would silently drop the hardest targets.
            _LOGGER.warning("grounding cell %s starts inside its bbox", task.name)

    def _image_report(self, render_metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_domain": stream_render.OBSERVATION_CONTRACT,
            **{
                key: render_metadata[key]
                for key in (
                    "media_type",
                    "jpeg_quality",
                    "color_mode",
                    "chroma_subsampling",
                    "width",
                    "height",
                )
            },
        }

    def _control_ok(self, state: DesktopState) -> bool | None:
        """Calibration conformance for a control arm; `None` for a model arm.

        Oracle arm: must pass. Negative arm: must fail. A model arm has no expected
        value, so there is nothing to conform to.
        """
        if not state.scripted:
            return None
        if state.infra_error is not None:
            return False
        return bool(state.success) is not state.negative_control

    async def _release_held(
        self, session: Any, held: list[tuple[str, tuple]]
    ) -> list[dict[str, Any]]:
        """Undo the presses the rollout left down, newest first.

        A failure here is an episode failure, not a swallowed warning: a guest whose
        buttons cannot be lifted is wedged, and `_run`'s handler retires the VM
        instead of recycling it into the next rollout with the button still down.
        """
        if not held:
            return []
        from desktop.ir import Operation  # type: ignore[import-not-found]

        operations = tuple(Operation(kind, args) for kind, args in reversed(held))
        await _to_thread(session.execute_atomic, operations)
        return [operation.as_dict() for operation in operations]

    async def _evaluate(
        self,
        preparer: Any,
        session: Any,
        task: DesktopTaskData,
        *,
        declared: str | None = None,
    ) -> float | None:
        try:
            raw = await _to_thread(
                lambda: preparer.evaluate(session, task, declared=declared)
            )
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"evaluate() returned a non-numeric score: {raw!r}")
            score = float(raw)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"evaluate() returned an invalid score: {score!r}")
        except Exception as exc:  # noqa: BLE001 - recorded as missing, never as 0.0
            _LOGGER.warning("evaluate() failed: %r", exc)
            return None
        return score

    @property
    def _artifact_root(self) -> Path:
        return Path(self.config.artifacts.output_dir)

    def _artifact_dir(self, task: DesktopTaskData) -> Path:
        directory = self._artifact_root / (task.name or f"task_{task.idx:04d}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _persist(self, artifacts: Path, trace: vf.Trace, frames: list[bytes]) -> None:
        config = self.config.artifacts
        if config.write_result_json:
            result = trace.info[RESULT_KEY]
            _atomic_json(artifacts / "result.json", result)
            rows = _trajectory_rows(result["steps_detail"], len(frames))
            _write_jsonl(artifacts / _TRAJECTORY, rows)
            _assert_frame_set(artifacts / "steps", len(rows))
            subdir = artifacts.relative_to(self._artifact_root).as_posix()
            self._runs[subdir] = {
                "subdir": subdir,
                "instruction": result["instruction"],
                "success": result["success"],
                "validity": result["validity"],
                "stop_reason": result["outcome"],
                "n_steps": len(rows) - 1,
                "traj_path": str((artifacts / _TRAJECTORY).resolve()),
            }
            _atomic_json(self._artifact_root / "result.json", self._artifact_index())
        if config.write_gif and len(frames) > 1:
            try:
                _write_gif(frames, artifacts / "rollout.gif")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("GIF write failed: %s", exc)

    def _artifact_index(self) -> dict[str, Any]:
        """The artifact-root `result.json` a recipe registers as its `eval_result`.

        The runner copies this whole file into `metadata.result` for such an output
        (`runner.rs:1885-1891`), which is the only way `metadata.result.traj_path`
        gets set — and that key is what gates the rollout viewer
        (`server.rs:1577-1588`). `traj_path` names ONE episode because the endpoint
        takes one path and the viewer has no per-episode selector.

        `tasks` must be a flat metric -> number dict with no nulls: anything else
        falls through to a raw JSON tree (`ui/src/lib/metrics.ts`), and `primary`
        is only honoured alongside it. `success_rate` is over the valid episodes
        only, so an infra failure cannot look like a task failure — and with no
        valid episode there is no rate to report, so the key and `primary` are
        both absent rather than 0.0, which reads as "every episode failed".
        """
        runs = [
            {"index": index, **run} for index, run in enumerate(self._runs.values())
        ]
        valid = [run for run in runs if run["validity"] == "valid"]
        name = self.config.id
        tasks = {
            f"{name}/n_episodes": float(len(runs)),
            f"{name}/n_valid": float(len(valid)),
            f"{name}/mean_steps": (
                sum(run["n_steps"] for run in runs) / len(runs) if runs else 0.0
            ),
        }
        if valid:
            tasks[f"{name}/success_rate"] = sum(
                1 for run in valid if run["success"]
            ) / len(valid)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task": name,
            "primary": f"{name}/success_rate" if valid else None,
            "tasks": tasks,
            "n_episodes": len(runs),
            "traj_path": runs[0]["traj_path"],
            "runs": runs,
        }


def _is_left_click(decision: Decision) -> bool:
    """True iff this action presses the left mouse button.

    Reads the compiled operations rather than the grammar's own action shape, so it
    works for every codec.
    """
    for operation in decision.operations:
        kind = getattr(operation, "kind", None) or (
            operation.get("kind") if isinstance(operation, dict) else None
        )
        args = getattr(operation, "args", None) or (
            operation.get("args") if isinstance(operation, dict) else ()
        )
        if kind == "mouse_down" and tuple(args or ())[:1] in {("left",), (1,)}:
            return True
    return False
