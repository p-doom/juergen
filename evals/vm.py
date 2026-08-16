"""The one adapter between desktop's session pool and the harness's session.

`DesktopPoolConfig.pool_target` names a constructor and `session_kwargs` is handed
to it verbatim (`agent.desktop.default_pool_factory`). Its nominal default,
`desktop.vm.pool:DesktopSessionPool`, cannot be used that way, for two reasons
that only show up against a real VM:

1. The constructor is not callable from config. `DesktopSessionPool.__init__`
requires a `DesktopPoolConfig` dataclass and a `session_factory` callable
(`pool.py:373-382`). Neither survives a recipe, a TOML file or a JSON
`session_kwargs`. `desktop.vm.factory:build_desktop_pool` is the
plain-arguments entry point, but it still takes the dataclass for `config`.

2. `checkout()` does not return a session the harness can drive. It returns a
`CheckedOutDesktopSession`, whose whole surface is `env`, `session_id`,
`tracked_env()`, `touch()` and `release()` (`pool.py:320-367`). The harness calls
`screen_size()`, `cursor_position()`, `screenshot()` and `execute_atomic()` on
whatever it gets (`evals/harness.py:76-83`), and the preparers additionally call
`execute_argv()`. Those live on two different objects — `HttpGuiTransport` has
the input/geometry half, `OSWorldClient` has the pixels-and-guest-commands half —
and `DesktopSession` holds both without merging them.

So `kvm_desktop_pool` is the callable entry point and `DesktopFacade` is the
union: no policy, no retries, no provider selection by name.

Reset isolation is explicit here, because the pool does not do it.
`DesktopSessionPool.release` returns a session to the `ready` list untouched until
`rollouts_completed >= max_rollouts_per_session` (`pool.py:495-519`) — no snapshot
restore, no `reset()`. A second rollout on a reused VM would therefore start with
the first one's Chrome still open, which for a four-cell gate scored on realized
guest state is a wrong measurement rather than a degraded one. Two settings are
supported:

  * `max_rollouts_per_session=1` (the default here) retires the VM after every
    rollout, so each one boots its own. Strictest, and what a multi-trial gate run
    wants: the trials are then independent draws in the VM too, not just the model.
  * a higher value plus `reset_on_reuse=True` calls `DesktopSession.reset()` — an
    attested restore of the clean checkpoint, the direct descendant of the old
    runner's `reset_to_ready()` — on every checkout *after* the first for a given
    session. Never on the first: a just-booted session is already at the
    checkpoint, and `reset_with_receipt` refuses a restore that does not change the
    observed runtime state (`session.py:526-528`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

__all__ = ["DesktopFacade", "kvm_desktop_pool"]

_LOGGER = logging.getLogger(__name__)


class DesktopFacade:
    """`HttpGuiTransport` + `OSWorldClient` + the lease, as one object.

    Explicit delegation rather than `__getattr__`: the harness probes optional
    capabilities with `getattr(session, "screenshot_settled", None)` and
    `getattr(session, "evaluate", None)`, and a catch-all proxy would answer yes to
    every probe and then fail at call time.
    """

    def __init__(
        self, checkout: Any, session: Any, *, osworld_cache_dir: str | Path | None = None
    ) -> None:
        self._checkout = checkout
        self._session = session
        self._osworld_cache_dir = osworld_cache_dir
        self._osworld_bridge: Any = None

    @property
    def session_id(self) -> str:
        return self._checkout.session_id

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        self._checkout.release(failed=failed, error=error)

    def _touch(self) -> None:
        """Feed the pool's lease watchdog.

        Done here rather than through `CheckedOutDesktopSession.tracked_env()`,
        which cannot help: its proxy only wraps *callables* it finds on the env
        (`pool.py:1000-1012`), and everything the harness calls lives one level
        down on `env.transport` / `env.client` — non-callable attributes, returned
        raw and untracked. A tracked env would look right and silently let the
        watchdog reclaim a VM mid-episode.
        """
        self._checkout.touch()

    @property
    def _transport(self) -> Any:
        """Read live: `DesktopSession.reset()` replaces the transport object
        (`session.py:535-536`), so a cached reference would keep driving a stale
        input audit after a restore."""
        return self._session.transport

    def execute_atomic(self, operations: Any) -> Any:
        self._touch()
        try:
            return self._transport.execute_atomic(tuple(operations))
        finally:
            self._touch()

    def cursor_position(self) -> tuple[int, int]:
        self._touch()
        return self._transport.cursor_position()

    def screen_size(self) -> tuple[int, int]:
        self._touch()
        return self._transport.screen_size()

    def execute_argv(self, argv: list[str]) -> dict[str, Any]:
        """One guest command. `check=False` on purpose.

        The preparers and the oracle read `result["output"]` and decide for
        themselves; a non-zero exit is frequently the *answer* (`pgrep chrome`
        returning 1 is "no Chrome"), so raising on it would turn a legitimate
        negative observation into an infrastructure error and the episode would be
        dropped instead of scored.
        """
        self._touch()
        try:
            return self._transport.execute_argv(list(argv), check=False)
        finally:
            self._touch()

    def execute_pyautogui(self, code: str) -> None:
        """Unused by the sign-of-life cells (they go through the codec and
        `execute_atomic`); the grounding and freeroll preparers do use it."""
        self._touch()
        try:
            self._transport.execute_pyautogui(code)
        finally:
            self._touch()

    def screenshot(self) -> bytes:
        self._touch()
        try:
            return self._session.client.screenshot()
        finally:
            self._touch()

    def screenshot_settled(
        self,
        *,
        min_delay_s: float = 0.0,
        stability_timeout_s: float = 0.0,
        poll_s: float = 0.1,
    ) -> bytes:
        self._touch()
        try:
            return self._session.client.screenshot_settled(
                min_delay_s=min_delay_s,
                stability_timeout_s=stability_timeout_s,
                poll_s=poll_s,
            )
        finally:
            self._touch()

    # The OSWorld half (`evals/osworld.py`) is named, not proxied:
    # `harness._evaluate` probes `getattr(session, "evaluate")` and the OSWorld
    # preparer calls `session.setup(...)`, so under a `__getattr__` catch-all both
    # would answer yes on a session with no OSWorld tree behind it and fail
    # mid-episode, after the boot and the guest setup.

    @property
    def _osworld(self) -> Any:
        """The bridge, built on first use and kept for the lease's lifetime.

        Lazy for the same reason `kvm_desktop_pool` imports lazily: a freeroll or
        an RL episode holds this same facade and must not pay for — or fail on —
        an OSWorld checkout it never touches. Built once because `bind()` is what
        makes `evaluate()` answerable with no arguments, and a fresh bridge per
        call would forget the task between setup and scoring.
        """
        if self._osworld_bridge is None:
            from evals.osworld import OSWorldBridge

            transport = self._transport
            self._osworld_bridge = OSWorldBridge(
                base_url=transport.base_url,
                cache_dir=self._osworld_cache_dir,
                screen_size=transport.screen_size(),
                **self._guest_ports(),
            )
        return self._osworld_bridge

    def _guest_ports(self) -> dict[str, int]:
        """`chromium_port` / `vlc_port` from the runtime, when it will say.

        `DesktopSession` does not keep the `RuntimeState` it started with — it
        forwards the ports into the metadata file and drops them — so they are
        read back off the runtime. `state()` is on `QemuRuntime` and not on the
        `Runtime` protocol, so a backing that does not offer it falls through to
        OSWorld's own defaults rather than failing: a task whose evaluator never
        touches Chrome or VLC does not need them.
        """
        runtime = getattr(self._session, "runtime", None)
        state = getattr(runtime, "state", None)
        if not callable(state):
            return {}
        try:
            ports = state().ports
        except Exception as exc:  # noqa: BLE001 - defaults are a valid answer
            _LOGGER.info("runtime will not report guest ports (%r); using defaults", exc)
            return {}
        found = {
            key: int(getattr(ports, key, 0) or 0) for key in ("chromium", "vlc")
        }
        return {f"{key}_port": value for key, value in found.items() if value}

    def setup(self, task_config: dict[str, Any]) -> int:
        """Run an OSWorld task JSON's `config` steps, and bind it for scoring.

        Takes the whole task config, not just its `config` list, which is the shape
        `DesktopEnv.reset(task_config=...)` uses and the only shape that lets
        `evaluate()` stay argument-free: the evaluator block arrives with the setup,
        so the desktop is never asked to score a task it was not prepared for.
        """
        self._touch()
        try:
            return self._osworld.setup(dict(task_config))
        finally:
            self._touch()

    def declare_terminal(self, control: str | None) -> None:
        """Tell the scorer how the episode ended.

        OSWorld inverts the reward on `infeasible` tasks — declaring FAIL is
        success there and forfeits everywhere else — and reads that off its action
        history, which we do not keep. Optional: the harness probes for it, and a
        session that cannot answer simply never claims a FAIL, which is the same
        verdict as a model that never declared one.
        """
        self._osworld.declare_terminal(control)

    def evaluate(self) -> float:
        """The OSWorld benchmark score for the task this desktop was set up for.

        No arguments, because `DesktopHarness._evaluate` has none to give (see
        `setup()`). Raises rather than returning 0.0 when there is nothing to score;
        the harness records a raise as a missing reward, and
        `OSWorldEvaluateOracle` refuses a missing reward outright, so infrastructure
        failure is never trained as task failure.
        """
        self._touch()
        try:
            return float(self._osworld.evaluate())
        finally:
            self._touch()


class _AdaptedPool:
    """`DesktopSessionPool` with the two gaps above closed."""

    def __init__(
        self,
        pool: Any,
        *,
        reset_on_reuse: bool,
        osworld_cache_dir: str | Path | None = None,
    ) -> None:
        self._pool = pool
        self._reset_on_reuse = reset_on_reuse
        self._osworld_cache_dir = osworld_cache_dir
        self._seen: set[str] = set()

    def start(self) -> None:
        self._pool.start()

    def checkout(self) -> DesktopFacade:
        checkout = self._pool.checkout()
        session = checkout.env
        session_id = checkout.session_id
        if self._reset_on_reuse and session_id in self._seen:
            _LOGGER.info("desktop %s: reused, restoring clean checkpoint", session_id)
            session.reset()
        self._seen.add(session_id)
        return DesktopFacade(
            checkout, session, osworld_cache_dir=self._osworld_cache_dir
        )

    def close(self) -> None:
        self._pool.close()


def kvm_desktop_pool(
    *,
    image: str | Path,
    root_dir: str | Path,
    qemu_binary: str | Path | None = None,
    qemu_img_binary: str | Path | None = None,
    smp: int | None = None,
    memory: str | None = None,
    accelerator: str | None = None,
    transport_timeout_s: float = 60.0,
    min_ready_sessions: int = 1,
    max_sessions: int = 1,
    max_rollouts_per_session: int = 1,
    checkout_timeout_s: float = 1800.0,
    lease_timeout_s: float = 1800.0,
    startup_timeout_s: float = 900.0,
    reset_on_reuse: bool = True,
    osworld_cache_dir: str | Path | None = None,
) -> _AdaptedPool:
    """A QEMU-backed pool the harness can drive, from JSON-able arguments only.

    Every parameter is a scalar or a path so this is reachable through
    `DesktopPoolConfig.session_kwargs` from a labctl recipe. Imported lazily inside
    the function for the same reason `default_pool_factory` is: a text-only eval
    must not pull the VM stack in.

    `lease_timeout_s` defaults to 1800 s rather than desktop's 300 s. The pool's
    watchdog reclaims a leased session that has been quiet for that long, and a gate
    cell that waits on a Chrome launch or a model call legitimately is: at 300 s the
    watchdog, not the episode, decides when the VM goes away.

    `osworld_cache_dir` is where OSWorld's getters put the files they pull out of
    the guest to score. Left unset it is a per-process temp directory, which is
    right for a gate run and wrong for a 369-task benchmark array on a node whose
    /tmp is small: name it then, on a filesystem with room.
    """
    from desktop.vm.factory import build_desktop_pool
    from desktop.vm.pool import DesktopPoolConfig

    config = DesktopPoolConfig(
        min_ready_sessions=min_ready_sessions,
        max_sessions=max_sessions,
        max_rollouts_per_session=max_rollouts_per_session,
        checkout_timeout_s=checkout_timeout_s,
        lease_timeout_s=lease_timeout_s,
        startup_timeout_s=startup_timeout_s,
    )
    runtime_options: dict[str, Any] = {"transport_timeout_s": transport_timeout_s}
    if qemu_binary is not None:
        runtime_options["qemu_binary"] = qemu_binary
    if qemu_img_binary is not None:
        runtime_options["qemu_img_binary"] = qemu_img_binary
    if smp is not None:
        runtime_options["smp"] = int(smp)
    if memory is not None:
        runtime_options["memory"] = memory
    if accelerator is not None:
        runtime_options["accelerator"] = accelerator
    pool = build_desktop_pool(
        root_dir=Path(root_dir),
        image=Path(image),
        config=config,
        **runtime_options,
    )
    return _AdaptedPool(
        pool, reset_on_reuse=reset_on_reuse, osworld_cache_dir=osworld_cache_dir
    )
