import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    CompactRawExecutor,
    NativeAbsoluteExecutor,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import RecordingTransport
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.trajectory import (
    ARMS,
    build_trajectory,
    execute_trajectory,
)


@pytest.mark.parametrize("fixture", load_manifest().fixtures, ids=lambda f: f.id)
@pytest.mark.parametrize("arm", ARMS)
def test_gold_trajectory_dispatches_cleanly(fixture, arm):
    transport = RecordingTransport(cursor=(71, 83))
    trajectory = build_trajectory(fixture, arm=arm, cursor=transport.cursor_position())
    results = execute_trajectory(
        trajectory, NativeAbsoluteExecutor(transport), CompactRawExecutor(transport)
    )
    assert len(results) == fixture.horizon
    assert not transport.audit.held_buttons
    assert not transport.audit.held_keys
    if fixture.template == "vscode_focus_type":
        assert transport.audit.typed_texts == [fixture.expected["text"]]
    if fixture.template == "local_document_scroll":
        sign = -1 if fixture.params["direction"] == "down" else 1
        assert transport.audit.scroll_total * sign > 0


@pytest.mark.parametrize("fixture", load_manifest().fixtures, ids=lambda f: f.id)
@pytest.mark.parametrize("arm", ARMS)
def test_near_miss_trajectory_is_distinct(fixture, arm):
    gold = build_trajectory(fixture, arm=arm, cursor=(71, 83))
    near = build_trajectory(fixture, arm=arm, cursor=(71, 83), near_miss=True)
    assert gold.actions != near.actions


def test_drag_uses_explicit_hold_move_release():
    fixture = load_manifest().by_id("r1b-files-drag-dev-3301")
    for arm in ARMS:
        trajectory = build_trajectory(fixture, arm=arm, cursor=(71, 83))
        assert trajectory.action_classes == ("button_hold", "mouse_move", "button_release")
