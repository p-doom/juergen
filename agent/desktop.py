"""Prewarmed desktop sessions with process-global pooling and node admission.

Two facts about verifiers force this module to exist.

1. The worker pool is upscale-only. `verifiers/v1/serve/pool.py:77` —
   "Upscale-only for now — workers are never reclaimed." `_maybe_scale_up`
   (pool.py:148-153) spawns a worker when in-flight rollouts hit 90% of
   `workers * multiplex`; there is no counterpart. A worker that holds a VM holds
   it until the broker's `_shutdown` (pool.py:243-261) at the end of the whole
   run. Scaling down the load leaves the VMs pinned: a 14-VM pool per worker
   times four scaled-up workers is 56 VMs on a node sized for 14, and the extras
   are idle.

2. One `Harness` instance is shared by every rollout (`env.py:257,352`), and
   `Harness.__init__` must not be overridden. So the pool cannot live on the
   harness instance, and per-rollout state must not either.

Three parts:

  * Process-global pools keyed by `PoolSpec.key`, created lazily on first
    `acquire` and torn down by `atexit`/SIGTERM. Not per-Harness, not per-rollout.
    Reusing one key with two different specs is refused, not resolved to whichever
    spec arrived first.
  * A node-wide slot lease (`NodeSlots`): one `flock`ed file per admissible VM
    under a shared directory, so the sum over spawn-workers cannot exceed the node
    budget however many workers the broker starts. A per-worker `max_sessions`
    cannot bound it.
  * An idle reaper: a pool with no live leases for `pool_idle_ttl_s` is closed
    and its slots returned. Each rollout releases its own lease synchronously.
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DesktopLease",
    "LeasedDesktopPool",
    "NodeSlots",
    "PoolSpec",
    "SlotExhausted",
    "close_all_pools",
    "pool_for",
]

DEFAULT_SLOT_DIR = Path(
    os.environ.get("JUERGEN_VM_SLOT_DIR", "/tmp/juergen-vm-slots")
)


class SlotExhausted(RuntimeError):
    """The node's VM budget is fully leased. A caller must wait, not oversubscribe."""


@dataclass
class _Slot:
    index: int
    path: Path
    handle: Any

    def release(self) -> None:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


class NodeSlots:
    """`max_slots` admission tickets shared by every process on the node.

    A slot is an advisory `flock` on `slot_{i:04d}.lock`. The lock dies with the
    process that holds it, so a SIGKILLed worker returns its slots to the node
    without a reaper; a per-worker in-memory counter would not.
    """

    def __init__(self, *, directory: Path = DEFAULT_SLOT_DIR, max_slots: int = 14) -> None:
        self.directory = Path(directory)
        self.max_slots = int(max_slots)
        self.directory.mkdir(parents=True, exist_ok=True)

    def acquire(self, *, timeout_s: float = 0.0, poll_s: float = 1.0) -> _Slot:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            for index in range(self.max_slots):
                path = self.directory / f"slot_{index:04d}.lock"
                handle = path.open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    handle.close()
                    continue
                slot = _Slot(index=index, path=path, handle=handle)
                # The lock is already held here, so anything that can raise has to
                # give it back: an unreleased flock with no `_Slot` to release it
                # burns one of the node's admission tickets until the process exits.
                try:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(f"pid={os.getpid()} acquired={time.time():.3f}\n")
                    handle.flush()
                except BaseException:
                    slot.release()
                    raise
                return slot
            if time.monotonic() >= deadline:
                raise SlotExhausted(
                    f"all {self.max_slots} VM slots under {self.directory} are held"
                )
            time.sleep(poll_s)


@dataclass(eq=False)
class DesktopLease:
    """One desktop session owned by one rollout.

    `eq=False` is load-bearing: `LeasedDesktopPool._leases` is a set, and a plain
    `@dataclass` generates `__eq__`, which sets `__hash__ = None`. With the
    generated `__eq__` every `acquire` died on `TypeError: unhashable type` after
    the node slot and the VM had already been taken but before either was tracked,
    leaking both. Identity is also the semantics the pool wants: two leases are
    never interchangeable, and `forget` discards the object it was handed.
    """

    trace_id: str
    session: Any
    slot: _Slot
    pool: "LeasedDesktopPool"
    failed: bool | None = None
    error: str | None = None
    released: bool = False
    _release_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def release(self, *, failed: bool, error: str | None) -> None:
        with self._release_lock:
            if self.released:
                return
            self.released = True
            self.failed = failed
            self.error = error
            try:
                release = getattr(self.session, "release", None)
                if callable(release):
                    release(failed=failed, error=error)
            except Exception:  # noqa: BLE001 - release must not mask the rollout error
                _LOGGER.exception("desktop lease %s: release failed", self.trace_id)
            finally:
                try:
                    self.slot.release()
                finally:
                    self.pool.forget(self)


@dataclass(frozen=True)
class PoolSpec:
    """Everything that identifies a pool, so two harness configs that want the
    same VMs share one pool instead of each starting its own.

    `max_node_slots` is separate from whatever the underlying pool calls its own
    maximum: the pool's limit is per process, this one is per node.
    """

    key: str
    max_node_slots: int = 14
    slot_dir: str = str(DEFAULT_SLOT_DIR)
    pool_idle_ttl_s: float = 900.0
    reap_interval_s: float = 15.0
    acquire_timeout_s: float = 1800.0


class LeasedDesktopPool:
    """A process-global `DesktopSessionPool` with a node-wide cap."""

    def __init__(self, spec: PoolSpec, factory: Callable[[], Any]) -> None:
        self.spec = spec
        self._factory = factory
        self._pool: Any = None
        self._leases: set[DesktopLease] = set()
        self._lock = threading.RLock()
        self._idle_since = time.monotonic()
        self._closed = False
        self._reaper = threading.Thread(
            target=self._reap_forever, name=f"vm-reaper[{spec.key}]", daemon=True
        )
        self._reaper.start()

    def _ensure_pool(self) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"desktop pool {self.spec.key!r} is closed")
            if self._pool is None:
                # Assigned only once `start()` has returned: caching a pool whose
                # start raised would make every later `acquire` check out of a pool
                # that was never brought up, and report no error while doing it.
                pool = self._factory()
                pool.start()
                self._pool = pool
                _LOGGER.info("desktop pool %s: started", self.spec.key)
            return self._pool

    def acquire(self, trace_id: str) -> DesktopLease:
        slot = NodeSlots(
            directory=Path(self.spec.slot_dir), max_slots=self.spec.max_node_slots
        ).acquire(timeout_s=self.spec.acquire_timeout_s)
        try:
            pool = self._ensure_pool()
            session = pool.checkout()
        except BaseException:
            slot.release()
            raise
        lease = DesktopLease(trace_id=trace_id, session=session, slot=slot, pool=self)
        with self._lock:
            self._leases.add(lease)
        return lease

    def forget(self, lease: DesktopLease) -> None:
        with self._lock:
            self._leases.discard(lease)
            if not self._leases:
                self._idle_since = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            leases = list(self._leases)
        for lease in leases:
            lease.release(failed=True, error="desktop pool closed")
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.close()
            except Exception:  # noqa: BLE001
                _LOGGER.exception("desktop pool %s: close failed", self.spec.key)
            _LOGGER.info("desktop pool %s: closed", self.spec.key)

    def _reap_once(self) -> None:
        with self._lock:
            idle = not self._leases
            pool_live = self._pool is not None
            idle_since = self._idle_since
        if (
            idle
            and pool_live
            and self.spec.pool_idle_ttl_s > 0
            and time.monotonic() - idle_since > self.spec.pool_idle_ttl_s
        ):
            # A worker that has finished its share of the rollouts stops holding
            # VMs, and its node slots go back so a busier worker can take them.
            _LOGGER.info(
                "desktop pool %s: idle %.0fs, releasing VMs",
                self.spec.key,
                time.monotonic() - idle_since,
            )
            with self._lock:
                pool, self._pool = self._pool, None
                self._idle_since = time.monotonic()
            if pool is not None:
                try:
                    pool.close()
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("desktop pool %s: idle close failed", self.spec.key)

    def _reap_forever(self) -> None:
        while True:
            time.sleep(max(1.0, self.spec.reap_interval_s))
            if self._closed:
                return
            try:
                self._reap_once()
            except Exception:  # noqa: BLE001 - a reaper must not die
                _LOGGER.exception("desktop pool %s: reap failed", self.spec.key)


_POOLS: dict[str, LeasedDesktopPool] = {}
_POOLS_LOCK = threading.Lock()


def pool_for(spec: PoolSpec, factory: Callable[[], Any]) -> LeasedDesktopPool:
    """The process's pool for `spec`, created on first use.

    Process-global rather than per-`Harness` because one harness instance serves
    every rollout and must not carry per-rollout state, and because two harness
    configs pointed at the same VM image should share VMs rather than double the
    node's load.
    """
    with _POOLS_LOCK:
        pool = _POOLS.get(spec.key)
        if pool is None:
            _install_teardown()
            pool = LeasedDesktopPool(spec, factory)
            _POOLS[spec.key] = pool
        elif pool.spec != spec:
            raise ValueError(
                f"desktop pool {spec.key!r} already exists with a different spec; "
                f"live={pool.spec!r} requested={spec!r}. Returning the live one would "
                "silently run the episode under someone else's pool settings."
            )
        return pool


def close_all_pools() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


_TEARDOWN_INSTALLED = False


def _install_teardown() -> None:
    """Register the process-wide teardown, once, when the first pool is created.

    Not at import: `import agent.desktop` would then replace the process's SIGINT
    and SIGTERM handlers in a process that may own no VM at all. There is nothing
    to tear down until a pool exists.
    """
    global _TEARDOWN_INSTALLED
    if _TEARDOWN_INSTALLED:
        return
    _TEARDOWN_INSTALLED = True
    atexit.register(close_all_pools)

    def _on_signal(signum: int, frame: Any) -> None:
        close_all_pools()
        previous = _PREVIOUS.get(signum)
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    _PREVIOUS: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            _PREVIOUS[signum] = signal.getsignal(signum)
            signal.signal(signum, _on_signal)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass


DEFAULT_POOL_TARGET = "desktop.vm.pool:DesktopSessionPool"


def default_pool_factory(
    session_kwargs: dict[str, Any], target: str = DEFAULT_POOL_TARGET
) -> Callable[[], Any]:
    """Call desktop's constructor directly with explicit config.

    Not a provider-by-name lookup: no name registry, no plugin resolution, and
    nothing patched into the OSWorld tree, which is re-clonable and would lose the
    patch. `target` names a constructor (`module:attribute`) and `session_kwargs`
    is passed to it verbatim; overriding it is how a test injects a fake, not how
    a VM backend is selected.

    Imported lazily so a text-only eval never pulls the VM stack in.
    """
    module_path, _, attribute = target.partition(":")
    if not module_path or not attribute:
        raise ValueError(f"pool target must be 'module:attribute', got {target!r}")

    def factory() -> Any:
        from importlib import import_module

        constructor = getattr(import_module(module_path), attribute)
        return constructor(**session_kwargs)

    return factory
