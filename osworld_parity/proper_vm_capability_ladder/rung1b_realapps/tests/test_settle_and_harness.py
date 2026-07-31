from __future__ import annotations

import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.transport import InputAudit
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps import selfcheck, vm
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.fixtures import (
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.states import (
    gold_state,
    near_miss_state,
    reset_state,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.trajectory import (
    ARMS,
    UiGeometry,
    build_trajectory,
)


def test_action_settle_requires_ack_then_three_fresh_identical_probes(monkeypatch):
    fixture = load_manifest().by_id("r1b-vscode-type-dev-3101")
    initial = reset_state(fixture)
    changed = near_miss_state(fixture)
    sequence = iter([changed, changed, changed, changed])
    monkeypatch.setattr(vm, "probe_fixture", lambda transport, item: next(sequence))

    settled = vm.wait_for_action_settle(
        object(), fixture, initial, phase="near_miss", poll_interval_s=0
    )

    assert settled.state == changed
    assert settled.acknowledgement["poll_index"] == 1
    assert settled.stable_probe_count == 3
    assert len(settled.polls) == 4
    assert "identical_post_ack_probe_count" not in settled.polls[0]
    assert settled.polls[-1]["identical_post_ack_probe_count"] == 3


def test_action_settle_timeout_raises_and_never_returns_stale_state(monkeypatch):
    fixture = load_manifest().by_id("r1b-vscode-type-dev-3101")
    initial = reset_state(fixture)
    clock = iter([0.0, 0.0, 0.0, 0.2])
    monkeypatch.setattr(vm.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(vm.time, "sleep", lambda _: None)
    monkeypatch.setattr(vm, "probe_fixture", lambda transport, item: initial)

    with pytest.raises(vm.AppSettleTimeout) as raised:
        vm.wait_for_action_settle(
            object(), fixture, initial, phase="gold", timeout_s=0.1
        )

    assert raised.value.evidence["acknowledged"] is False
    assert raised.value.evidence["last_state"] == initial


def test_setup_readiness_requires_three_identical_geometry_probes(monkeypatch):
    fixture = load_manifest().by_id("r1b-vscode-type-dev-3101")
    geometries = iter(
        [
            UiGeometry(editor=(10, 10)),
            UiGeometry(editor=(11, 10)),
            UiGeometry(editor=(11, 10)),
            UiGeometry(editor=(11, 10)),
        ]
    )

    class Transport:
        def execute_argv(self, argv):
            return {"status": "success", "returncode": 0, "output": ""}

    monkeypatch.setattr(
        vm, "_guest_dir", lambda transport, item: PurePosixPath("/tmp/r1b-test")
    )
    monkeypatch.setattr(vm, "probe_fixture", lambda transport, item: reset_state(item))
    monkeypatch.setattr(vm, "probe_geometry", lambda transport, item: next(geometries))

    guest = vm.setup_fixture(Transport(), fixture, poll_interval_s=0)

    assert guest.geometry.editor == (11, 10)
    assert guest.readiness["stable_geometry_probe_count"] == 3
    assert len(guest.readiness["polls"]) == 4
    assert guest.readiness["phases"][-1]["phase"] == "stable_geometry"


def test_readiness_failure_names_guest_root_phase(monkeypatch):
    fixture = load_manifest().by_id("r1b-vscode-type-dev-3101")
    monkeypatch.setattr(
        vm,
        "_guest_dir",
        lambda transport, item: (_ for _ in ()).throw(vm.TransportError("no root")),
    )
    with pytest.raises(vm.AppReadinessError) as raised:
        vm.setup_fixture(object(), fixture)
    assert raised.value.failed_phase == "guest_root_resolution"
    assert "no root" in raised.value.evidence["last_error"]


def test_arm_order_is_seed_parity_counterbalanced():
    fixtures = load_manifest().fixtures
    for fixture in fixtures:
        expected = ARMS if fixture.parameter_seed % 2 == 0 else tuple(reversed(ARMS))
        assert selfcheck._arm_order(fixture) == expected


@pytest.mark.parametrize("fixture_id", ["r1b-scroll-dev-3201", "r1b-scroll-dev-3202"])
@pytest.mark.parametrize("arm", ARMS)
def test_scroll_near_miss_is_correct_direction_undershoot(fixture_id, arm):
    fixture = load_manifest().by_id(fixture_id)
    initial = reset_state(fixture)["scroll_y"]
    near = near_miss_state(fixture)["scroll_y"]
    gold = gold_state(fixture)["scroll_y"]
    requested_sign = 1 if fixture.params["direction"] == "down" else -1
    assert (near - initial) * requested_sign > 0
    assert abs(near - initial) < int(fixture.expected["min_delta"])
    assert near not in {initial, gold}
    trajectory = build_trajectory(
        fixture, arm=arm, cursor=(71, 83), near_miss=True
    )
    action = trajectory.actions[-1]
    clicks = action["clicks"] if isinstance(action, dict) else int(action.split()[-1])
    assert clicks * (-requested_sign) > 0
    assert abs(clicks) == 1


def test_vm_harness_persists_attempt_before_assertion_and_attempts_every_cell(
    monkeypatch, tmp_path
):
    fixtures = tuple(
        item for item in load_manifest().fixtures if item.template == "files_drag"
    )
    manifest = SimpleNamespace(fixtures=fixtures, manifest_payload_sha256="a" * 64)

    class Transport:
        def __init__(self):
            self.audit = InputAudit()
            self.base_url = "http://unused"

        def cursor_position(self):
            return (50, 60)

    class Session:
        def __init__(self, **kwargs):
            self.resets = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def reset_to_ready(self):
            self.resets += 1
            return Transport()

    monkeypatch.setattr(selfcheck, "load_manifest", lambda: manifest)
    monkeypatch.setattr(selfcheck, "sha256_file", lambda path: "provider")
    monkeypatch.setattr(selfcheck, "KvmFixtureSession", Session)
    monkeypatch.setattr(
        selfcheck,
        "setup_fixture",
        lambda transport, fixture: vm.GuestFixture(
            reset_state(fixture), UiGeometry(), {"phases": [{"phase": "stable_geometry"}]}
        ),
    )
    monkeypatch.setattr(
        selfcheck,
        "_capture_screenshot",
        lambda transport, path: {"path": str(path), "sha256": "b" * 64},
    )
    monkeypatch.setattr(
        selfcheck,
        "_execute_with_journal",
        lambda trajectory, native, compact, journal: journal,
    )
    monkeypatch.setattr(
        selfcheck,
        "wait_for_action_settle",
        lambda transport, fixture, initial, phase: vm.SettledFixture(
            near_miss_state(fixture) if phase == "near_miss" else gold_state(fixture),
            {"kind": "hidden_state_changed"},
            ({"identical_post_ack_probe_count": 3},),
            3,
        ),
    )
    monkeypatch.setattr(selfcheck, "probe_fixture", lambda transport, fixture: reset_state(fixture))
    monkeypatch.setattr(selfcheck, "probe_geometry", lambda transport, fixture: UiGeometry())
    monkeypatch.setattr(selfcheck, "collect_fixture_diagnostics", lambda *args: {"logs": {}})
    calls = 0

    def oracle(fixture, state, solved, label):
        nonlocal calls
        calls += 1
        progress = json.loads((tmp_path / "progress.json").read_text())
        assert progress["attempted_cells"] >= 1
        if calls == 1:
            raise selfcheck.SelfcheckError("injected assertion failure")
        return {"oracle_status": "ok", "MOUSE_SOLVED": solved}

    monkeypatch.setattr(selfcheck, "_assert_oracle", oracle)

    with pytest.raises(selfcheck.SelfcheckError, match="1 failed cells"):
        selfcheck.run_vm_selfcheck(
            output=tmp_path,
            qcow=tmp_path / "vm.qcow2",
            qemu=tmp_path / "qemu",
            provider=tmp_path / "provider.py",
            expected_provider_sha256="provider",
        )

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["attempted_cells"] == len(fixtures) * len(ARMS)
    assert progress["failed_cells"] == 1
    failed = next(cell for cell in progress["cells"] if cell["status"] == "failed")
    assert failed["post_failure_clean_reset"]["status"] == "ok"
    context = json.loads(
        (tmp_path / "cells" / failed["fixture_id"] / failed["arm"] / "failure_context.json").read_text()
    )
    assert context["cell"]["failure"]["phase"] == "reset_a_recorded_before_oracle"


def test_main_never_leaves_a_pass_marker_on_failure(monkeypatch, tmp_path):
    marker = tmp_path / "selfcheck.json"
    marker.write_text('{"status":"stale-pass"}')
    monkeypatch.setattr(
        selfcheck,
        "run_build_selfcheck",
        lambda: (_ for _ in ()).throw(selfcheck.SelfcheckError("injected failure")),
    )

    assert selfcheck.main(["--mode=build", f"--output={tmp_path}"]) == 2
    assert not marker.exists()
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["status"] == "failed"
