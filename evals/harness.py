"""The one episode driver.

The loop:

    screenshot -> prompt from codec.describe() + history policy -> sample
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
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

import verifiers.v1 as vf
from pydantic import Field, model_validator

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
from agent.history import History, ImageBudget, history_policy
from evals.tasks import (
    RESULT_KEY,
    RESULT_SCHEMA_VERSION,
    DesktopState,
    DesktopTaskData,
    distance_to_box,
    in_bbox,
    preparer_for,
)

_LOGGER = logging.getLogger(__name__)

__all__ = ["DesktopHarness", "DesktopHarnessConfig"]


class Desktop(Protocol):
    """The session surface an episode needs from `desktop.vm.pool`.

    Everything optional is probed with `getattr`, so a pool that cannot settle a
    frame or evaluate an OSWorld task still runs the families that do not need it.
    """

    def screen_size(self) -> tuple[int, int]: ...
    def cursor_position(self) -> tuple[int, int]: ...
    def screenshot(self) -> bytes: ...
    def execute_atomic(self, operations: Any) -> Any: ...


class HistoryConfig(vf.BaseConfig):
    """The injected history policy."""

    name: str = "interleaved_frames"
    n_history_frames: int = Field(default=16, ge=1)
    persist_instruction: bool = True
    """`InterleavedFrames` is the only policy that implements this."""

    @model_validator(mode="after")
    def _persist_instruction_is_implemented(self) -> "HistoryConfig":
        if self.name != "interleaved_frames" and not self.persist_instruction:
            raise ValueError(
                f"persist_instruction=False is implemented by interleaved_frames only; "
                f"policy {self.name!r} would ignore it"
            )
        return self


class ImageBudgetConfig(vf.BaseConfig):
    max_images: int = Field(default=16, ge=1)
    media: Literal["jpeg", "png"] = "jpeg"
    quality: int = Field(default=85, ge=1, le=100)
    max_pixels: int = Field(default=0, ge=0)


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
    tokens, wall time.
    """

    model_turns: int = Field(default=0, ge=0)
    operations: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    wall_time_s: float = Field(default=0.0, ge=0.0)


class ArtifactConfig(vf.BaseConfig):
    output_dir: str = ""
    save_frames: bool = True
    save_prompts: bool = True
    write_gif: bool = True
    write_result_json: bool = True
    register_labctl: bool = False
    """`labctl register-external --kind eval_result`, so the run shows up in the
    RolloutViewer."""
    labctl_alias: str = ""


class DesktopPoolConfig(vf.BaseConfig):
    key: str = "default"
    max_node_slots: int = Field(default=14, ge=1)
    slot_dir: str = ""
    episode_ttl_s: float = Field(default=1800.0, gt=0.0)
    scoring_grace_s: float = Field(default=120.0, ge=0.0)
    """How long after `launch` the desktop stays leased so a runtime-declaring
    `@vf.reward` can still probe live guest state."""
    pool_idle_ttl_s: float = Field(default=900.0, ge=0.0)
    acquire_timeout_s: float = Field(default=1800.0, gt=0.0)
    reap_interval_s: float = Field(default=15.0, gt=0.0)
    """How often the reaper looks for expired leases and an idle pool."""
    session_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Passed verbatim to the session-pool constructor named by `pool_target`."""
    pool_target: str = "desktop.vm.pool:DesktopSessionPool"
    """The session-pool constructor, as `module:attribute`.

    A constructor, not a provider name. Override it to inject a fake pool, not to
    select a VM backend."""
    hide_gpu_during_boot: bool = True
    """Blank `CUDA_VISIBLE_DEVICES` while the VM boots: the process that forks
    qemu may also hold a GPU, and a child that inherits the visible device can
    wedge the allocation."""


class DesktopHarnessConfig(vf.HarnessConfig):
    codec: str = "deltatype_v2"
    """Grammar entry-point name. The one field a grammar A/B changes."""
    system_prompt_override: str | None = None
    system_prompt_sha256: str | None = None
    """A checkpoint's sealed training-prompt digest. Recorded, never enforced.

    `codec.describe()` is docstring-derived and not byte-identical to a
    checkpoint's sealed training prompt, so a hash gate here would fail every run.
    The digests ride `trace.info["prompt"]` as data alongside the codec's own
    `report()`, so a run that must not be compared across the prompt boundary is
    identifiable from the record."""
    history: HistoryConfig = HistoryConfig()
    images: ImageBudgetConfig = ImageBudgetConfig()
    settle: SettleConfig = SettleConfig()
    scripted: ScriptedConfig = ScriptedConfig()
    budget: BudgetConfig = BudgetConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    pool: DesktopPoolConfig = DesktopPoolConfig()
    max_steps: int = Field(default=0, ge=0)
    """Overrides the task's own `max_steps` when > 0."""
    max_tokens: int = Field(default=256, ge=1)
    """Fallback only — used for a knob `ctx.sampling` leaves unset."""
    temperature: float | None = None
    """Fallback only. `ctx.sampling.temperature` wins at the wire; see
    `agent.agent.resolve_sampling`."""
    top_p: float | None = None
    stop_on_click: bool = False
    """End the episode at the first left-button press, turning a free rollout into
    a single-decision probe."""
    require_unsolved_start: bool = True
    """Refuse to score a cell whose postcondition already holds before the first
    action."""
    evaluate_on_finish: bool = False
    """Call the session's OSWorld `evaluate()` and publish it as `task_reward`."""
    prefer_context_transport: bool = False
    """Sample through `ctx.client` instead of posting to `endpoint`."""


@dataclass
class _Budget:
    config: BudgetConfig
    started: float = 0.0
    model_turns: int = 0
    operations: int = 0
    output_tokens: int = 0
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
            "wall_time_s": round(time.monotonic() - self.started, 3),
            "failure": self.failure,
        }


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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
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
        path, save_all=True, append_images=images[1:], duration=300, loop=0, optimize=True
    )


def _register_labctl(alias: str, path: Path) -> bool:
    """Best-effort artifact registration; a registry hiccup must not tank a run."""
    try:
        proc = subprocess.run(
            [
                "labctl",
                "register-external",
                "--alias",
                alias,
                "--kind",
                "eval_result",
                "--path",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        _LOGGER.warning("register-external invocation failed for %s: %s", alias, exc)
        return False
    if proc.returncode != 0:
        _LOGGER.warning(
            "register-external failed (rc=%d) for %s: %s",
            proc.returncode,
            alias,
            proc.stderr.strip()[:500],
        )
        return False
    return True


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tri_state(value: bool | None) -> float | None:
    return None if value is None else (1.0 if value else 0.0)


_CODECS: dict[str, Any] = {}


def _codec(name: str) -> Any:
    """Process-level codec cache. Not on the harness instance: one `Harness` serves
    every rollout, so instance state is shared state."""
    codec = _CODECS.get(name)
    if codec is None:
        codec = load_codec(name)
        _CODECS[name] = codec
    return codec


class DesktopHarness(vf.Harness[DesktopHarnessConfig]):
    SUPPORTS_MESSAGE_PROMPT = True

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
        del runtime, mcp_urls
        task = trace.task.data
        if not isinstance(task, DesktopTaskData):
            raise TypeError("DesktopHarness requires DesktopTaskData")
        if not isinstance(trace.state, DesktopState):
            raise TypeError("DesktopHarness requires DesktopState")

        # Everything resolvable from config, resolved before a VM is booted:
        # an unknown grammar, an unregistered task kind and a scripted arm on a
        # family with no gold plan are config errors, and discovering one at step 1
        # has already cost a boot and the cell's whole guest setup.
        codec = _codec(self.config.codec)
        preparer = preparer_for(task.kind)
        if self.config.scripted.enabled and not callable(
            getattr(preparer, "script_plan", None)
        ):
            raise LookupError(
                f"task kind {task.kind!r} has no scripted arm; scripted.enabled "
                "requires a preparer implementing script_plan() + render_step()"
            )

        spec = PoolSpec(
            key=self.config.pool.key,
            max_node_slots=self.config.pool.max_node_slots,
            slot_dir=self.config.pool.slot_dir or str(DEFAULT_SLOT_DIR),
            episode_ttl_s=self.config.pool.episode_ttl_s,
            scoring_grace_s=self.config.pool.scoring_grace_s,
            pool_idle_ttl_s=self.config.pool.pool_idle_ttl_s,
            reap_interval_s=self.config.pool.reap_interval_s,
            acquire_timeout_s=self.config.pool.acquire_timeout_s,
        )
        pool = pool_for(spec, self.pool_factory())

        lease = None
        failed = True
        error: str | None = None
        try:
            with _hidden_gpu(self.config.pool.hide_gpu_during_boot):
                lease = await asyncio.to_thread(pool.acquire, trace.id)
            trace.info["desktop_session"] = getattr(lease.session, "session_id", None)
            await self._run(
                ctx, trace, task, lease.session, endpoint, secret, codec, preparer
            )
            failed = False
            return vf.ProgramResult(0, "", "")
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if lease is not None:
                # `_run` publishes an episode failure as `infra_invalid` instead of
                # letting it escape, so this flag would otherwise only ever catch an
                # `acquire` failure. `failed` is what makes desktop retire the VM
                # rather than hand it to the next rollout (`vm/pool.py:509-519`):
                # a wedged guest — dead executor transport, unreadable state — was
                # being recycled as healthy.
                published = trace.info.get(RESULT_KEY) or {}
                if published.get("validity") == "infra_invalid":
                    failed = True
                    error = error or json.dumps(published.get("infra_error"), default=str)
                # Hand the VM to the scoring phase, then let the reaper release it.
                lease.finish(
                    failed=failed, error=error, grace_s=self.config.pool.scoring_grace_s
                )

    async def _run(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        task: DesktopTaskData,
        session: Any,
        endpoint: str,
        secret: str,
        codec: Any,
        preparer: Any,
    ) -> None:
        state = trace.state
        assert isinstance(state, DesktopState)
        trace.info["prompt"] = self._prompt_report(codec)
        budget = _Budget(self.config.budget)
        max_steps = self.config.max_steps or task.max_steps
        artifacts = self._artifact_dir(task)

        agent = Agent(
            codec=codec,
            policy=history_policy(
                self.config.history.name,
                **(
                    {"persist_instruction": self.config.history.persist_instruction}
                    if self.config.history.name == "interleaved_frames"
                    else {}
                ),
            ),
            budget=ImageBudget(
                max_images=self.config.images.max_images,
                media="png" if self.config.images.media == "png" else "jpeg",
                quality=self.config.images.quality,
                max_pixels=self.config.images.max_pixels,
            ),
            transport=build_transport(
                endpoint=endpoint,
                secret=secret,
                prefer_context=self.config.prefer_context_transport,
            ),
            system_prompt=self.config.system_prompt_override,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        state.codec = self.config.codec
        state.history_policy = agent.policy.name
        state.scripted = self.config.scripted.enabled
        state.negative_control = self.config.scripted.negative

        steps_detail: list[dict[str, Any]] = []
        frames: list[bytes] = []
        outcome = "max_steps"
        infra_error: dict[str, str] | None = None
        sampling_record: dict[str, Any] = {}
        setup_evidence: dict[str, Any] = {}

        try:
            setup_evidence = await asyncio.to_thread(preparer.prepare, session, task)
            geometry = await asyncio.to_thread(_geometry, session)
            initial = await asyncio.to_thread(preparer.probe, session, task)
            state.initial_probe = initial
            self._assert_unsolved(task, initial)

            frame = await self._observe(preparer, session, task)
            frames.append(frame)
            history = History(n_history_frames=self.config.history.n_history_frames)
            history.start(frame)
            if self.config.artifacts.save_frames:
                (artifacts / "steps").mkdir(parents=True, exist_ok=True)
                (artifacts / "steps" / "step_000.png").write_bytes(frame)

            script = self._script_plan(preparer, task) if self.config.scripted.enabled else None
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
                cursor = tuple(await asyncio.to_thread(session.cursor_position))
                decision, step_sampling = await self._decide(
                    agent,
                    ctx,
                    trace,
                    task,
                    history=history,
                    step=step,
                    geometry=geometry,
                    cursor=cursor,
                    script=script,
                    preparer=preparer,
                    session=session,
                    codec=codec,
                    artifacts=artifacts,
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
                        receipt = await asyncio.to_thread(
                            session.execute_atomic, decision.operations
                        )
                        budget.dispatched(len(decision.operations))
                    except (TypeError, ValueError) as exc:
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
                            self._record(decision, step, cursor, cursor, None, None, action_error)
                        )
                        break
                if decision.parse_error:
                    state.parse_errors += 1
                state.ignored_after_terminate += decision.ignored_after_terminate

                frame = await self._observe(preparer, session, task)
                frames.append(frame)
                if self.config.artifacts.save_frames:
                    (artifacts / "steps" / f"step_{step:03d}.png").write_bytes(frame)
                history.append(decision.text, frame)
                cursor_after = tuple(await asyncio.to_thread(session.cursor_position))
                probe = await asyncio.to_thread(preparer.probe, session, task)
                steps_detail.append(
                    self._record(
                        decision, step, cursor, cursor_after, frame, probe, action_error
                    )
                )
                state.steps = step

                if probe.get("in_bbox") and reach_frame < 0:
                    reach_frame = step
                if task.bbox is not None:
                    distance = distance_to_box(cursor_after, task.bbox)
                    best_distance = distance if best_distance < 0 else min(best_distance, distance)

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

            state.final_probe = probe
            state.reach_frame = reach_frame
            state.best_distance = best_distance
            if self.config.evaluate_on_finish and infra_error is None:
                state.task_reward = await self._evaluate(
                    session, declared=state.control_terminate
                )
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
        except Exception as exc:  # noqa: BLE001 - published as infra-invalid
            infra_error = {
                "stage": "episode",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            outcome = "infrastructure_error"
            _LOGGER.exception("episode %s failed", task.name)
        finally:
            await agent.close()

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
        as a task failure.
        """
        state.outcome = outcome
        state.infra_error = infra_error
        state.infra_valid = infra_error is None
        state.success = bool(outcome == "postcondition_reached") if infra_error is None else None
        state.temperature = sampling.get("temperature")
        state.temperature_source = sampling.get("temperature_source")

        trace.info[RESULT_KEY] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "validity": "valid" if infra_error is None else "infra_invalid",
            "codec": self.config.codec,
            "history_policy": state.history_policy,
            "sampling": sampling,
            "success": state.success,
            "outcome": outcome,
            "steps": state.steps,
            "parse_errors": state.parse_errors,
            "action_errors": state.action_errors,
            "executor_errors": state.executor_errors,
            "control_terminate": state.control_terminate,
            "terminate_step": state.terminate_step,
            "ignored_after_terminate": state.ignored_after_terminate,
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
        history: History,
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
            text = await asyncio.to_thread(
                self._render_step, preparer, session, task, codec, script[step - 1]
            )
            decision = agent.decide(
                text, step=step, geometry=geometry, cursor=cursor, sampling=sampling
            )
            return decision, sampling.as_dict()

        if self.config.artifacts.save_prompts:
            body = agent.build_body(history=history, instruction=task.instruction, step=step)
            (artifacts / "steps").mkdir(parents=True, exist_ok=True)
            (artifacts / "steps" / f"prompt_{step:03d}.json").write_text(dump_prompt(body))
        decision = await agent.step(
            ctx,
            history=history,
            instruction=task.instruction or None,
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
        frame = await asyncio.to_thread(
            _screenshot, session, self.config.settle, task.kind
        )
        observe = getattr(preparer, "observe", None)
        if callable(observe):
            frame = await asyncio.to_thread(observe, frame, task)
        return frame

    def _script_plan(self, preparer: Any, task: DesktopTaskData) -> list[Any]:
        """The scripted arm's plan: intents, not yet rendered to text.

        Not rendered up front. `compact_raw.from_target` (and the relative renderers
        generally) need one fresh cursor read and are wrong if that read is stale,
        while `native_absolute_control.from_target` needs only element geometry.
        Rendering the whole script before the first action would make every click
        after the first resolve against a stale cursor.
        """
        return list(preparer.script_plan(task, negative=self.config.scripted.negative))

    def _render_step(
        self, preparer: Any, session: Any, task: DesktopTaskData, codec: Any, intent: Any
    ) -> str:
        """Render one intent into codec text, reading the cursor now.

        Codec text rather than operations directly, so the control arms exercise the
        same `parse` and `compile` the model arm does.
        """
        return preparer.render_step(session, task, codec=codec, intent=intent)

    def _postcondition_reached(self, task: DesktopTaskData, probe: dict[str, Any]) -> bool:
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
            raise RuntimeError("task reset/setup did not begin in a valid unsolved state")
        if task.bbox is not None and in_bbox(
            tuple(probe.get("cursor") or (-1, -1)), task.bbox
        ):
            # Grounding records this instead of refusing: the cursor-start sampler
            # already guarantees an outside-bbox start except against a screen edge,
            # and a refusal there would silently drop the hardest targets.
            _LOGGER.warning("grounding cell %s starts inside its bbox", task.name)

    def _prompt_report(self, codec: Any) -> dict[str, Any]:
        """Prompt provenance as data. Never raises; a digest mismatch is recorded,
        never enforced.

        Baseline warning, recorded on every run: the off-the-shelf Qwen3-VL-8B =
        33.9% OSWorld-Verified figure is our only calibrated reference and was
        measured through the old sealed prompts. `native_absolute`, `move_rel` and
        `native_absolute_control` now describe themselves from docstrings and are
        not byte-identical to those prompts, so numbers must not be compared across
        that boundary and the baseline needs re-measuring through the new prompt.
        """
        prompt = self.config.system_prompt_override
        if prompt is None:
            prompt = codec.describe()
        observed = hashlib.sha256(prompt.encode()).hexdigest()
        expected = self.config.system_prompt_sha256
        report = getattr(codec, "report", None)
        return {
            "codec": self.config.codec,
            "prompt_sha256": observed,
            "expected_prompt_sha256": expected,
            "matches_expected": None if expected is None else observed == expected,
            "codec_report": report() if callable(report) else None,
            "comparable_to_sealed_baseline": False,
            "baseline_note": (
                "Qwen3-VL-8B=33.9% OSWorld-Verified was measured through the sealed "
                "prompts; describe() is not byte-identical. Re-measure before comparing."
            ),
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

    async def _evaluate(self, session: Any, *, declared: str | None = None) -> float | None:
        evaluate = getattr(session, "evaluate", None)
        if not callable(evaluate):
            raise LookupError(
                "evaluate_on_finish asks for the OSWorld scorer, which "
                f"{type(session).__name__} does not implement"
            )
        # OSWorld inverts the reward on its `infeasible` tasks: declaring FAIL is
        # the success condition there and forfeits everywhere else. The scorer
        # cannot see our control tokens, so the verdict is handed over
        # explicitly. Optional — a session that does not offer it simply never
        # claims a FAIL, which is what a model that never declared one produces.
        declare = getattr(session, "declare_terminal", None)
        if callable(declare):
            declare(declared)
        try:
            score = float(await asyncio.to_thread(evaluate))
        except Exception as exc:  # noqa: BLE001 - recorded as missing, never as 0.0
            _LOGGER.warning("evaluate() failed: %r", exc)
            return None
        return score

    def _artifact_dir(self, task: DesktopTaskData) -> Path:
        root = Path(self.config.artifacts.output_dir or tempfile.gettempdir())
        directory = root / (task.name or f"task_{task.idx:04d}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _persist(self, artifacts: Path, trace: vf.Trace, frames: list[bytes]) -> None:
        config = self.config.artifacts
        if config.write_result_json:
            _atomic_json(artifacts / "result.json", trace.info.get(RESULT_KEY))
        if config.write_gif and len(frames) > 1:
            try:
                _write_gif(frames, artifacts / "rollout.gif")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("GIF write failed: %s", exc)
        if config.register_labctl:
            alias = config.labctl_alias or artifacts.name
            trace.info.setdefault("artifacts", {})["labctl_registered"] = _register_labctl(
                alias, artifacts
            )


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
