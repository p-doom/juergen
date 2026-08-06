"""Item 1 — `NodeSlots` / `LeasedDesktopPool` / `LeaseRegistry`.

The load-bearing claim is in the module docstring: *"The lock dies with the process
that holds it, so a SIGKILLed worker returns its slots to the node without a reaper
— which is the property a per-worker in-memory counter can never have."* That is
asserted here against a **real** SIGKILLed child process, because it is the only
property that justifies the file existing: verifiers' worker pool is upscale-only,
so a worker that dies holding VMs must not hold their admission tickets either.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agent import desktop as dsk
from agent.desktop import (
    DesktopLease,
    LeaseRegistry,
    LeasedDesktopPool,
    NodeSlots,
    PoolSpec,
    SlotExhausted,
    close_all_pools,
    default_pool_factory,
    lease_for_trace,
    pool_for,
)
from juergen_doubles import FakePool, FakeSession

_REPO = Path(__file__).resolve().parents[1]
_PATH = os.pathsep.join([str(_REPO), str(_REPO.parent / "pixeldesk")])


@pytest.fixture(autouse=True)
def _no_pool_leak():
    yield
    close_all_pools()


# --------------------------------------------------------------------------- #
# NodeSlots
# --------------------------------------------------------------------------- #


def test_node_slots_admits_exactly_max_slots(tmp_path: Path) -> None:
    slots = NodeSlots(directory=tmp_path, max_slots=3)
    held = [slots.acquire() for _ in range(3)]
    assert sorted(s.index for s in held) == [0, 1, 2]
    with pytest.raises(SlotExhausted):
        slots.acquire()
    held[1].release()
    reused = slots.acquire()
    assert reused.index == 1, "a released slot is the lowest free index, so reuse is dense"
    for slot in (*held[::2], reused):
        slot.release()


def test_node_slots_records_the_holding_pid(tmp_path: Path) -> None:
    slot = NodeSlots(directory=tmp_path, max_slots=1).acquire()
    try:
        assert f"pid={os.getpid()}" in slot.path.read_text()
    finally:
        slot.release()


def test_a_slot_that_cannot_be_stamped_gives_its_lock_back(tmp_path: Path, monkeypatch) -> None:
    """The flock is taken before the pid is written. A raise in between used to leave
    the lock held by a `_Slot` nobody has, burning one of the node's tickets until the
    process exits."""
    slots = NodeSlots(directory=tmp_path, max_slots=1)
    real = Path.open

    def exploding(self, *args, **kwargs):
        handle = real(self, *args, **kwargs)
        handle.truncate = lambda *a: (_ for _ in ()).throw(OSError("no space left"))
        return handle

    monkeypatch.setattr(Path, "open", exploding)
    with pytest.raises(OSError, match="no space"):
        slots.acquire()
    monkeypatch.undo()
    recovered = slots.acquire()
    assert recovered.index == 0, "the ticket came back"
    recovered.release()


def test_node_slots_timeout_waits_then_raises(tmp_path: Path) -> None:
    slots = NodeSlots(directory=tmp_path, max_slots=1)
    held = slots.acquire()
    started = time.monotonic()
    with pytest.raises(SlotExhausted):
        slots.acquire(timeout_s=0.4, poll_s=0.05)
    assert time.monotonic() - started >= 0.35, "a timeout must actually wait"
    held.release()


def test_a_second_process_sees_the_same_node_budget(tmp_path: Path) -> None:
    """The point of a file lock: the cap is per *node*, not per process."""
    slots = NodeSlots(directory=tmp_path, max_slots=2)
    mine = [slots.acquire(), slots.acquire()]
    probe = subprocess.run(
        [sys.executable, "-c", _CHILD_PROBE, str(tmp_path), "2"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PATH},
        timeout=60,
    )
    assert probe.stdout.strip().endswith("acquired=0"), probe.stderr[-2000:]
    for slot in mine:
        slot.release()


_CHILD_PROBE = textwrap.dedent(
    """
    import sys
    from agent.desktop import NodeSlots, SlotExhausted
    slots = NodeSlots(directory=sys.argv[1], max_slots=int(sys.argv[2]))
    got = []
    while True:
        try:
            got.append(slots.acquire())
        except SlotExhausted:
            break
    print("acquired=%d" % len(got))
    """
)


_CHILD_HOLDER = textwrap.dedent(
    """
    import sys, time
    from agent.desktop import NodeSlots
    slots = NodeSlots(directory=sys.argv[1], max_slots=int(sys.argv[2]))
    held = [slots.acquire() for _ in range(int(sys.argv[3]))]
    print("HELD %d" % len(held), flush=True)
    time.sleep(600)
    """
)


def _spawn_holder(directory: Path, max_slots: int, count: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLDER, str(directory), str(max_slots), str(count)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": _PATH},
    )
    line = proc.stdout.readline() if proc.stdout else ""
    assert line.startswith("HELD"), (line, proc.stderr.read() if proc.stderr else "")
    return proc


def test_sigkilled_worker_returns_its_slots_with_no_reaper(tmp_path: Path) -> None:
    """★ The claim the whole module rests on.

    No reaper runs here and nothing in this process knows the child existed. The
    slots come back because the kernel drops the `flock` when the process dies —
    exactly what an in-memory counter in a worker that just got SIGKILLed cannot do.
    """
    slots = NodeSlots(directory=tmp_path, max_slots=4)
    holder = _spawn_holder(tmp_path, 4, 4)
    try:
        with pytest.raises(SlotExhausted):
            slots.acquire()  # the child holds every ticket
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=30)
        recovered = []
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and len(recovered) < 4:
            try:
                recovered.append(slots.acquire())
            except SlotExhausted:
                time.sleep(0.05)
        assert len(recovered) == 4, "SIGKILL must return every slot the child held"
        for slot in recovered:
            slot.release()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=30)


def test_the_sum_over_spawned_workers_cannot_exceed_the_node_budget(tmp_path: Path) -> None:
    """However many workers the broker starts, the node total is `max_node_slots`.

    Each child asks for the *whole* budget, which is what a per-worker `max_sessions`
    would let them all get. Three of the four must come back with zero.
    """
    budget = 3
    holders = []
    try:
        first = _spawn_holder(tmp_path, budget, budget)
        holders.append(first)
        results = []
        for _ in range(3):
            probe = subprocess.run(
                [sys.executable, "-c", _CHILD_PROBE, str(tmp_path), str(budget)],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": _PATH},
                timeout=60,
            )
            results.append(probe.stdout.strip())
        assert results == ["acquired=0"] * 3, results
    finally:
        for holder in holders:
            holder.kill()
            holder.wait(timeout=30)


# --------------------------------------------------------------------------- #
# LeaseRegistry
# --------------------------------------------------------------------------- #


def _lease(trace_id: str, *, pool) -> DesktopLease:
    class _NullSlot:
        def release(self) -> None:
            return None

    return DesktopLease(
        trace_id=trace_id, session=FakeSession(), slot=_NullSlot(), pool=pool
    )


def test_registry_hides_a_released_lease() -> None:
    registry = LeaseRegistry()
    pool = LeasedDesktopPool(PoolSpec(key="registry-test"), FakePool)
    lease = _lease("trace-a", pool=pool)
    registry.put(lease)
    assert registry.get("trace-a") is lease
    assert registry.live() == [lease]
    lease.released = True
    assert registry.get("trace-a") is None, "a released lease must not be handed out"
    assert registry.live() == []
    registry.drop("trace-a")
    assert registry.get("trace-a") is None
    pool.close()


def test_lease_release_is_idempotent_and_reports_failure_to_the_session() -> None:
    pool = LeasedDesktopPool(PoolSpec(key="idempotent"), FakePool)
    lease = _lease("trace-b", pool=pool)
    lease.finish(failed=True, error="boom", grace_s=0.0)
    lease.release()
    lease.release()
    assert lease.session.released == [(True, "boom")], "release must run exactly once"
    pool.close()


def test_a_raising_session_release_does_not_mask_the_rollout_error() -> None:
    class Angry(FakeSession):
        def release(self, **kwargs: object) -> None:
            raise RuntimeError("release exploded")

    pool = LeasedDesktopPool(PoolSpec(key="angry"), FakePool)

    class _NullSlot:
        released = False

        def release(self) -> None:
            type(self).released = True

    slot = _NullSlot()
    lease = DesktopLease(trace_id="t", session=Angry(), slot=slot, pool=pool)
    lease.release()  # must not raise
    assert _NullSlot.released, "the node slot must come back even if release() throws"
    pool.close()


# --------------------------------------------------------------------------- #
# LeasedDesktopPool
# --------------------------------------------------------------------------- #


def test_acquire_publishes_the_lease_and_finish_starts_the_grace_window(tmp_path: Path) -> None:
    spec = PoolSpec(key="grace", slot_dir=str(tmp_path), max_node_slots=2, reap_interval_s=1.0)
    backing = FakePool()
    pool = LeasedDesktopPool(spec, lambda: backing)
    lease = pool.acquire("trace-grace")
    assert backing.started == 1, "the pool starts lazily, on first acquire"
    assert lease_for_trace("trace-grace") is lease
    lease.finish(failed=False, error=None, grace_s=5.0)
    assert not lease.expired(), "the grace window keeps the VM readable for scoring"
    assert lease_for_trace("trace-grace") is lease
    lease.release()
    assert lease_for_trace("trace-grace") is None
    pool.close()


def test_a_lease_past_its_deadline_is_reclaimed_by_the_reaper(tmp_path: Path) -> None:
    spec = PoolSpec(
        key="reaper", slot_dir=str(tmp_path), max_node_slots=1, reap_interval_s=1.0
    )
    backing = FakePool()
    pool = LeasedDesktopPool(spec, lambda: backing)
    lease = pool.acquire("trace-reap")
    lease.finish(failed=False, error=None, grace_s=0.0)  # deadline is now
    assert lease.expired()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and not lease.released:
        time.sleep(0.1)
    assert lease.released, "the idle reaper must release a lease past its deadline"
    assert lease.session.released == [(False, None)]
    # And the slot came back, so the next rollout can have it.
    again = pool.acquire("trace-reap-2")
    again.release()
    pool.close()


def test_an_idle_pool_releases_its_vms(tmp_path: Path) -> None:
    """The scale-down verifiers does not do."""
    spec = PoolSpec(
        key="idle",
        slot_dir=str(tmp_path),
        max_node_slots=1,
        reap_interval_s=1.0,
        pool_idle_ttl_s=0.5,
    )
    backing = FakePool()
    pool = LeasedDesktopPool(spec, lambda: backing)
    pool.acquire("trace-idle").release()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and backing.closed == 0:
        time.sleep(0.1)
    assert backing.closed == 1, "an idle pool must give its VMs back"
    pool.close()


def test_a_failed_checkout_does_not_leak_the_node_slot(tmp_path: Path) -> None:
    class Broken:
        def start(self) -> None:
            return None

        def checkout(self):
            raise RuntimeError("no VM for you")

        def close(self) -> None:
            return None

    spec = PoolSpec(key="broken", slot_dir=str(tmp_path), max_node_slots=1)
    pool = LeasedDesktopPool(spec, Broken)
    with pytest.raises(RuntimeError):
        pool.acquire("trace-broken")
    # If the slot leaked, this second attempt would block for acquire_timeout_s.
    slot = NodeSlots(directory=tmp_path, max_slots=1).acquire()
    slot.release()
    pool.close()


def test_close_releases_live_leases_and_the_backing_pool(tmp_path: Path) -> None:
    spec = PoolSpec(key="closing", slot_dir=str(tmp_path), max_node_slots=2)
    backing = FakePool()
    pool = LeasedDesktopPool(spec, lambda: backing)
    a, b = pool.acquire("close-a"), pool.acquire("close-b")
    pool.close()
    assert a.released and b.released
    assert backing.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire("close-c")


def test_pool_for_is_process_global_and_keyed_by_spec(tmp_path: Path) -> None:
    """Not per-`Harness`: one harness instance serves every rollout (`env.py:257`)."""
    spec = PoolSpec(key="shared", slot_dir=str(tmp_path))
    other = PoolSpec(key="other", slot_dir=str(tmp_path))
    first = pool_for(spec, FakePool)
    assert pool_for(spec, FakePool) is first, "two harness configs must share one pool"
    assert pool_for(other, FakePool) is not first
    close_all_pools()
    assert pool_for(spec, FakePool) is not first, "close_all_pools must forget the pool"
    close_all_pools()


def test_pool_for_refuses_one_key_with_two_specs(tmp_path: Path) -> None:
    """The cache key is `spec.key` alone. Handing back the pool that got there first
    would run the episode under someone else's slot budget and TTLs, silently."""
    spec = PoolSpec(key="collide", slot_dir=str(tmp_path), max_node_slots=2)
    pool_for(spec, FakePool)
    try:
        with pytest.raises(ValueError, match="different spec"):
            pool_for(PoolSpec(key="collide", slot_dir=str(tmp_path), max_node_slots=9), FakePool)
    finally:
        close_all_pools()


# --------------------------------------------------------------------------- #
# teardown on both signals
# --------------------------------------------------------------------------- #


_CHILD_TEARDOWN = textwrap.dedent(
    """
    import os, signal, sys, threading, time
    import agent.desktop as dsk

    closed = threading.Event()
    real = dsk.close_all_pools
    def spy():
        real()
        print("CLOSED", flush=True)
        closed.set()
    dsk.close_all_pools = spy
    # atexit/signal handlers captured the module attribute at import time, so
    # re-register the spy the same way _install_teardown does.
    import atexit
    atexit.register(spy)

    class Backing:
        def start(self): return None
        def checkout(self): return object()
        def close(self): print("POOL CLOSED", flush=True)

    pool = dsk.pool_for(dsk.PoolSpec(key="child", slot_dir=sys.argv[1]), Backing)
    pool.acquire("child-trace")
    print("READY", flush=True)
    signum = int(sys.argv[2])
    os.kill(os.getpid(), signum)
    time.sleep(5)
    """
)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_teardown_releases_on_both_signals(tmp_path: Path, signum: int) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_TEARDOWN, str(tmp_path), str(int(signum))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": _PATH},
    )
    out, err = proc.communicate(timeout=60)
    assert "READY" in out, (out, err[-2000:])
    assert "POOL CLOSED" in out, f"{signum!r} must tear the pool down: {out!r} {err[-2000:]!r}"
    assert proc.returncode != 0, "the signal must still take the process down"


def test_atexit_teardown_closes_the_pool(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        import agent.desktop as dsk

        class Backing:
            def start(self): return None
            def checkout(self): return object()
            def close(self): print("POOL CLOSED", flush=True)

        dsk.pool_for(dsk.PoolSpec(key="atexit", slot_dir=sys.argv[1]), Backing).acquire("t")
        print("READY", flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PATH},
        timeout=60,
    )
    assert "POOL CLOSED" in proc.stdout, (proc.stdout, proc.stderr[-2000:])


# --------------------------------------------------------------------------- #
# runtime construction
# --------------------------------------------------------------------------- #


def test_default_pool_factory_imports_a_constructor_with_explicit_config() -> None:
    factory = default_pool_factory({"a": 1, "b": "two"}, "juergen_fake_pool:Recorder")
    built = factory()
    assert built.kwargs == {"a": 1, "b": "two"}, "session_kwargs pass verbatim"


def test_default_pool_factory_rejects_a_non_constructor_target() -> None:
    for bad in ("no_colon", ":attribute", "module:"):
        with pytest.raises(ValueError, match="module:attribute"):
            default_pool_factory({}, bad)


def test_default_pool_factory_defers_the_import_so_a_text_only_eval_stays_light() -> None:
    factory = default_pool_factory({}, "juergen_fake_pool:Missing")
    with pytest.raises(AttributeError):
        factory()  # the failure happens on call, not on construction


def test_the_default_target_is_the_desktop_env_constructor_not_a_provider_name() -> None:
    assert dsk.DEFAULT_POOL_TARGET == "pixeldesk.vm.pool:DesktopSessionPool"
    assert ":" in dsk.DEFAULT_POOL_TARGET
    module, _, attribute = dsk.DEFAULT_POOL_TARGET.partition(":")
    from importlib import import_module

    assert isinstance(getattr(import_module(module), attribute), type)
