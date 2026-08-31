from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from desktop import ir
from desktop.geometry import DisplayGeometry
from juergen_doubles import FakeSession

from evals.harness import DesktopHarness, _update_held
from grammars.ordered_events_v3.codec import CODEC, OrderedEventsV3Error

GEOMETRY = DisplayGeometry(desktop_width=1920, desktop_height=1080)


def test_typed_controls_lift_to_the_ordered_protocols_existing_events() -> None:
    source = (ir.coalesced_type("a\nb\tc"),)
    action = CODEC.action_from_operations(source, geometry=GEOMETRY, cursor=(100, 100))
    line = (
        'type("a"); down(Return); up(Return); type("b"); down(Tab); up(Tab); type("c")'
    )
    expected = (
        ir.coalesced_type("a"),
        ir.key_down("Return"),
        ir.key_up("Return"),
        ir.coalesced_type("b"),
        ir.key_down("Tab"),
        ir.key_up("Tab"),
        ir.coalesced_type("c"),
    )

    assert CODEC.format(action) == line
    assert CODEC.compile(line, GEOMETRY, (100, 100)) == expected
    with pytest.raises(OrderedEventsV3Error, match="Press Return"):
        CODEC.parse(r'type("a\nb")')
    with pytest.raises(OrderedEventsV3Error, match="press Tab"):
        CODEC.parse('type("a\tb")')
    prompt = CODEC.describe()
    assert "down(Return); up(Return)" in prompt
    assert "down(Tab); up(Tab)" in prompt


@pytest.mark.parametrize(
    ("button", "token"),
    [("left", "LMB"), ("middle", "MMB"), ("right", "RMB")],
)
def test_a_click_lifts_and_compiles_with_its_own_button(
    button: str, token: str
) -> None:
    action = CODEC.action_from_operations(
        (ir.click(button),), geometry=GEOMETRY, cursor=(100, 100)
    )
    line = f"down({token}); up({token})"

    assert CODEC.format(action) == line
    assert CODEC.compile(line, GEOMETRY, (100, 100)) == (
        ir.mouse_down(button),
        ir.mouse_up(button),
    )


def test_episode_cleanup_releases_held_inputs_newest_first() -> None:
    held: list[tuple[str, tuple]] = []
    operations = CODEC.compile("down(ControlLeft); down(RMB)", GEOMETRY, (100, 100))
    _update_held(held, operations)
    session = FakeSession()
    harness = object.__new__(DesktopHarness)

    released = asyncio.run(harness._release_held(session, held))
    expected = (ir.mouse_up("right"), ir.key_up("ControlLeft"))

    assert released == [operation.as_dict() for operation in expected]
    assert session.operations_log == [list(expected)]


def test_leased_pool_drives_the_tracked_session_and_releases_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.desktop as desktop_pools

    events: list[object] = []

    class Slot:
        def release(self) -> None:
            events.append("slot_released")

    class Slots:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def acquire(self, **_kwargs: object) -> Slot:
            return Slot()

    class TrackedSession:
        session_id = "tracked-session"

    tracked_session = TrackedSession()

    class Checkout:
        def tracked_env(self) -> TrackedSession:
            events.append("tracked_env")
            return tracked_session

        def release(self, *, failed: bool, error: str | None) -> None:
            events.append(("checkout_released", failed, error))

    class Pool:
        def start(self) -> None:
            events.append("pool_started")

        def checkout(self) -> Checkout:
            events.append("checked_out")
            return Checkout()

        def close(self) -> None:
            events.append("pool_closed")

    monkeypatch.setattr(desktop_pools, "NodeSlots", Slots)
    pool = desktop_pools.LeasedDesktopPool(
        desktop_pools.PoolSpec(
            key="tracked-session-test",
            max_node_slots=1,
            slot_dir=str(tmp_path),
            reap_interval_s=60.0,
        ),
        Pool,
    )
    lease = pool.acquire("trace")
    assert lease.session is tracked_session

    lease.release(failed=True, error="episode failed")
    pool.close()

    assert events == [
        "pool_started",
        "checked_out",
        "tracked_env",
        ("checkout_released", True, "episode failed"),
        "slot_released",
        "pool_closed",
    ]
