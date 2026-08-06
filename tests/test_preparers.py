"""The `Preparer` seam and its 8 preparers.

The seam's contract: `prepare` may drive the guest, `probe` must not. A `probe`
that dispatched input would break the read-only property the oracle depends on.

Eight preparers are registered: `none`, `terminal`, `osworld`, `grounding`, the four
sign-of-life kinds, plus the fixture and RL ones that register on import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evals.fixtures.preparers  # noqa: F401  registers web_fixture / app_fixture
import evals.signoflife.guest  # noqa: F401  registers the four sign-of-life kinds
import rl.grounding.harness  # noqa: F401
import rl.movebox.harness  # noqa: F401
import rl.target_box.harness  # noqa: F401
from evals.signoflife.suite import ALLOWED_KINDS
from evals.tasks import PREPARERS, Preparer, preparer_for, register_preparer
from juergen_doubles import FakeSession, make_task_data

SIGN_OF_LIFE_KINDS = sorted(ALLOWED_KINDS)
CORE_KINDS = ["grounding", "none", "osworld", "terminal"]


def test_the_eight_preparers_are_registered() -> None:
    assert set(CORE_KINDS) <= set(PREPARERS)
    assert set(SIGN_OF_LIFE_KINDS) <= set(PREPARERS)
    assert len(set(CORE_KINDS) | set(SIGN_OF_LIFE_KINDS)) == 8


def test_the_fixture_and_rl_preparers_register_on_import() -> None:
    assert {"web_fixture", "app_fixture"} <= set(PREPARERS)
    assert {"movebox", "grounding_canvas", "target_box"} <= set(PREPARERS)


def test_every_registered_preparer_satisfies_the_protocol() -> None:
    for kind, preparer in PREPARERS.items():
        assert preparer.kind == kind, f"{kind} is registered under the wrong key"
        assert isinstance(preparer, Preparer), kind
        assert callable(preparer.prepare) and callable(preparer.probe)


def test_an_unregistered_kind_is_a_loud_lookup_error() -> None:
    with pytest.raises(LookupError, match="no preparer registered"):
        preparer_for("kind_that_does_not_exist")
    with pytest.raises(LookupError, match="known:"):
        preparer_for("")


def test_register_preparer_returns_its_argument_so_it_composes() -> None:
    class Custom:
        kind = "test_only_custom"

        def prepare(self, session, task):
            return {}

        def probe(self, session, task):
            return {}

    instance = Custom()
    try:
        assert register_preparer(instance) is instance
        assert preparer_for("test_only_custom") is instance
    finally:
        PREPARERS.pop("test_only_custom", None)


class _WitnessSession(FakeSession):
    """Records every call that could change guest state."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_events: list[str] = []

    def execute_atomic(self, operations):
        self.input_events.append("execute_atomic")
        return super().execute_atomic(operations)

    def execute_pyautogui(self, code):
        self.input_events.append(f"pyautogui:{code}")
        return super().execute_pyautogui(code)

    def setup(self, config):
        self.input_events.append("setup")
        return super().setup(config)


_SOL_STATE = json.dumps(
    {
        "schema_version": 1,
        "task_id": "cell",
        "active_window": "xterm",
        "windows": "",
        "chrome_process": False,
        "history": None,
        "transcript": None,
        "prompt_count": 0,
        "capture_file_exists": False,
        "captured_text": None,
        "proof_file_exists": False,
        "proof_file_content": None,
    }
)


@pytest.mark.parametrize("kind", CORE_KINDS)
def test_probe_dispatches_no_input_events(kind: str) -> None:
    """The seam's load-bearing property, asserted per preparer."""
    session = _WitnessSession()
    task = make_task_data(kind=kind, bbox=(10, 10, 50, 50))
    preparer_for(kind).probe(session, task)
    assert session.input_events == [], f"{kind}.probe drove the guest"


@pytest.mark.parametrize("kind", SIGN_OF_LIFE_KINDS)
def test_sign_of_life_probe_dispatches_no_input_events(kind: str) -> None:
    session = _WitnessSession(argv_responses={"python3": f"SOLV2_STATE={_SOL_STATE}"})
    task = make_task_data(kind=kind, expected=_expected_for(kind))
    preparer_for(kind).probe(session, task)
    assert session.input_events == []
    assert all(argv[0] == "python3" for argv in session.argv_log), session.argv_log


def _expected_for(kind: str) -> dict:
    return {
        "terminal_command": {"command": "ls", "listing_marker": "m.txt"},
        "terminal_exact_text": {"text": "hello"},
        "open_chrome": {"active_window_class_any": ["chrome"]},
        "focus_terminal_and_type": {
            "command": "printf x > /tmp/p",
            "file": "/tmp/p",
            "content": "x",
        },
    }[kind]


@pytest.mark.parametrize("kind", ["movebox", "grounding_canvas", "target_box"])
def test_the_rl_probes_dispatch_no_input_events(kind: str) -> None:
    from rl.desktop import VirtualDesktop

    task = make_task_data(kind=kind, bbox=(10, 10, 50, 50), setup={"screen": [200, 200]})
    if kind == "target_box":
        session = _WitnessSession(screen=(1920, 1080))
        task = make_task_data(
            kind=kind,
            setup={"screen": [1920, 1080], "instance_key": "k", "box": {}},
        )
    else:
        session = VirtualDesktop(screen=(200, 200))
    preparer_for(kind).probe(session, task)
    if isinstance(session, _WitnessSession):
        assert session.input_events == []


def test_the_none_preparer_boots_screenshots_and_goes() -> None:
    session = FakeSession(cursor=(7, 9), screen=(800, 600))
    task = make_task_data(kind="none")
    assert preparer_for("none").prepare(session, task) == {"prepared": "none"}
    assert session.pyautogui_log == [] and session.argv_log == []
    assert preparer_for("none").probe(session, task) == {
        "cursor": [7, 9],
        "screen": [800, 600],
    }


def test_the_terminal_preparer_launches_a_terminal_and_clears_it() -> None:
    """Preserved verbatim, including the `ctrl-l`."""
    session = FakeSession()
    task = make_task_data(kind="terminal")
    assert preparer_for("terminal").prepare(session, task) == {"prepared": "terminal"}
    script = session.pyautogui_log[0]
    assert "gnome-terminal" in script and "xfce4-terminal" in script and "xterm" in script
    assert "hotkey('ctrl', 'l')" in script, "the clear is part of the contract"


def test_the_osworld_preparer_runs_the_task_configs_setup_commands() -> None:
    session = FakeSession()
    config = [{"type": "launch", "parameters": {"command": ["true"]}}]
    task = make_task_data(kind="osworld", setup={"config": config})
    evidence = preparer_for("osworld").prepare(session, task)
    assert evidence == {"prepared": "osworld", "steps": 1, "scorable": False}
    assert session.argv_log[0][0] == "<osworld-setup>"
    assert json.loads(session.argv_log[0][1]) == config


def test_the_osworld_preparer_hands_the_session_the_whole_task_config() -> None:
    """Not just the `config` list. The `evaluator` block has to travel with the
    setup, because that is the only thing that makes `DesktopFacade.evaluate()`
    answerable with no arguments — and it not travelling is why the OSWorld family
    could not be scored through the production adapter at all."""
    session = FakeSession()
    evaluator = {"func": "is_expected_active_tab", "result": {"type": "active_tab_info"}}
    task = make_task_data(
        kind="osworld",
        setup={"task_config": {"id": "t", "config": [{"type": "a"}], "evaluator": evaluator}},
    )
    evidence = preparer_for("osworld").prepare(session, task)
    assert evidence == {"prepared": "osworld", "steps": 1, "scorable": True}
    assert session.task_config is not None
    assert session.task_config["evaluator"] == evaluator


def test_an_osworld_task_that_is_pure_evaluator_still_binds() -> None:
    """A task with an evaluator and no setup steps has nothing to run and
    everything to score; returning early on an empty `config` would leave the
    session unbound and `evaluate()` unanswerable."""
    session = FakeSession()
    task = make_task_data(
        kind="osworld", setup={"task_config": {"id": "t", "evaluator": {"func": "check_include_exclude"}}}
    )
    assert preparer_for("osworld").prepare(session, task)["scorable"] is True
    assert session.task_config is not None


def test_the_osworld_preparer_reads_config_from_the_task_path(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"id": "t", "instruction": "i", "config": [{"type": "a"}, {"type": "b"}]}))
    session = FakeSession()
    task = make_task_data(kind="osworld", task_path=str(path))
    assert preparer_for("osworld").prepare(session, task)["steps"] == 2


def test_an_osworld_task_with_no_config_does_not_call_setup() -> None:
    session = FakeSession()
    task = make_task_data(kind="osworld")
    assert preparer_for("osworld").prepare(session, task) == {"prepared": "osworld", "steps": 0}
    assert session.argv_log == []
    assert session.task_config is None


def test_the_grounding_preparer_places_the_stratified_cursor() -> None:
    session = FakeSession(screen=(1920, 1080))
    task = make_task_data(
        kind="grounding", bbox=(900, 500, 1000, 600), regime="near", name="app/task/near"
    )
    evidence = preparer_for("grounding").prepare(session, task)
    assert evidence["regime"] == "near"
    assert evidence["requested_cursor_start"] == evidence["observed_cursor_start"]
    assert session.cursor == tuple(evidence["observed_cursor_start"])
    assert "moveTo" in session.pyautogui_log[-1]


def test_an_explicit_cursor_start_wins_over_the_sampler() -> None:
    session = FakeSession(screen=(1920, 1080))
    task = make_task_data(
        kind="grounding", bbox=(10, 10, 50, 50), regime="near", cursor_start=(444, 333)
    )
    evidence = preparer_for("grounding").prepare(session, task)
    assert evidence["requested_cursor_start"] == [444, 333]


def test_the_grounding_preparer_records_a_refused_cursor_move() -> None:
    """A VM that clamps the cursor must be recorded, not asserted away."""

    class Stubborn(FakeSession):
        def execute_pyautogui(self, code):
            self.pyautogui_log.append(code)  # accept the call, ignore it

    session = Stubborn(screen=(1920, 1080), cursor=(1, 1))
    task = make_task_data(kind="grounding", bbox=(900, 500, 1000, 600), regime="near")
    evidence = preparer_for("grounding").prepare(session, task)
    assert evidence["observed_cursor_start"] == [1, 1]
    assert evidence["requested_cursor_start"] != evidence["observed_cursor_start"]


def test_the_grounding_probe_reports_containment_and_distance() -> None:
    session = FakeSession(cursor=(20, 20))
    task = make_task_data(kind="grounding", bbox=(10, 10, 50, 50))
    probe = preparer_for("grounding").probe(session, task)
    assert probe == {"cursor": [20, 20], "in_bbox": True, "distance": 0.0}
    outside = FakeSession(cursor=(60, 20))
    probe = preparer_for("grounding").probe(outside, task)
    assert probe["in_bbox"] is False and probe["distance"] == 10.0


def test_the_grounding_probe_reports_no_distance_without_a_bbox() -> None:
    probe = preparer_for("grounding").probe(FakeSession(), make_task_data(kind="grounding"))
    assert probe["distance"] is None and probe["in_bbox"] is False


def _traj(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "traj.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_replay_dispatches_the_first_n_non_reset_actions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    path = _traj(
        tmp_path,
        [
            {"step_num": 0, "action": "<reset>"},
            {"step_num": 1, "action": "pyautogui.click(10, 10)"},
            {"step_num": 2, "action": "pyautogui.click(20, 20)"},
        ],
    )
    session = FakeSession(screen=(1920, 1080))
    task = make_task_data(
        kind="grounding",
        bbox=(10, 10, 50, 50),
        regime="near",
        setup={"replay_trajectory": str(path), "replay_n_steps": 1},
    )
    evidence = preparer_for("grounding").prepare(session, task)
    assert evidence["replayed"] == 1
    replays = [c for c in session.pyautogui_log if "click" in c]
    assert replays == ["pyautogui.click(10, 10)"], "n_steps is honoured"


def test_replay_skips_a_step_zero_a_blank_line_and_a_reset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    from evals.tasks import _replay

    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step_num": 0, "action": "pyautogui.click(0,0)"}),
                "",
                json.dumps({"step_num": 1, "action": "<reset>"}),
                json.dumps({"step_num": 2, "action": ""}),
                json.dumps({"step_num": 3, "action": "pyautogui.click(9,9)"}),
            ]
        )
    )
    session = FakeSession()
    assert _replay(session, path, n_steps=5) == 1
    assert session.pyautogui_log == ["pyautogui.click(9,9)"]


def test_a_bad_cached_row_is_skipped_not_fatal(tmp_path: Path, monkeypatch) -> None:
    """A bad cached row must not kill a 369-task array run."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    from evals.tasks import _replay

    class Picky(FakeSession):
        def execute_pyautogui(self, code):
            if "boom" in code:
                raise ValueError("unexecutable")
            super().execute_pyautogui(code)

    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step_num": 1, "action": "boom()"}),
                json.dumps({"step_num": 2, "action": "pyautogui.click(1,1)"}),
            ]
        )
    )
    session = Picky()
    assert _replay(session, path, n_steps=2) == 1
    assert session.pyautogui_log == ["pyautogui.click(1,1)"]


def test_the_sign_of_life_probe_carries_the_verdict_alongside_the_state() -> None:
    session = FakeSession(argv_responses={"python3": f"SOLV2_STATE={_SOL_STATE}"})
    task = make_task_data(
        kind="terminal_exact_text", name="cell", expected={"text": "hello"}
    )
    probe = preparer_for("terminal_exact_text").probe(session, task)
    assert probe["postcondition_status"] == "ok"
    assert probe["postcondition_success"] is False
    assert "postcondition_reason" in probe and "postcondition_evidence" in probe
    assert probe["schema_version"] == 1, "the raw state is still there"


def test_an_ambiguous_guest_probe_fails_closed() -> None:
    doubled = f"SOLV2_STATE={_SOL_STATE}\nSOLV2_STATE={_SOL_STATE}"
    session = FakeSession(argv_responses={"python3": doubled})
    task = make_task_data(kind="terminal_exact_text", expected={"text": "x"})
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        preparer_for("terminal_exact_text").probe(session, task)


def test_a_missing_guest_marker_fails_closed() -> None:
    session = FakeSession(argv_responses={"python3": "nothing useful here"})
    task = make_task_data(kind="terminal_exact_text", expected={"text": "x"})
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        preparer_for("terminal_exact_text").probe(session, task)


def test_a_guest_command_with_no_stdout_fails_closed() -> None:
    class Silent(FakeSession):
        def execute_argv(self, argv):
            return {}

    task = make_task_data(kind="terminal_exact_text", expected={"text": "x"})
    with pytest.raises(RuntimeError, match="no stdout"):
        preparer_for("terminal_exact_text").probe(Silent(), task)


def test_sign_of_life_setup_is_hermetic_per_cell() -> None:
    """Two cells that shared a shell would share history."""
    import evals.signoflife.guest as guest

    scripts = []
    for name in ("cell_a", "cell_b"):
        session = FakeSession(
            argv_responses={
                "wmctrl": "SOLV2_GEOMETRY=" + json.dumps(
                    {"window_id": "0x1", "x": 80, "y": 120, "width": 1120, "height": 720, "window_line": "x"}
                ),
                "python3": "SOLV2_GEOMETRY=" + json.dumps(
                    {"window_id": "0x1", "x": 80, "y": 120, "width": 1120, "height": 720, "window_line": "x"}
                ),
            }
        )
        task = make_task_data(kind="terminal_exact_text", name=name, expected={"text": "t"})
        guest._setup_terminal_exact_text(session, task)
        scripts.append(" ".join(session.argv_log[0]))
    assert str(guest.ROOT / "cell_a") in scripts[0]
    assert str(guest.ROOT / "cell_b") in scripts[1]
    assert str(guest.ROOT / "cell_b") not in scripts[0], "no shared root between cells"
    for script in scripts:
        assert "rm -rf" in script, "each cell wipes and rebuilds its own root"
        assert "captured.txt" in script


def test_the_per_kind_settle_is_2s_for_chrome_and_0_75s_elsewhere() -> None:
    """One source of truth: the arms' `SettleConfig`, which is what the harness reads.

    A per-preparer `settle_s()` would be a second copy the harness never reads.
    """
    from evals.signoflife.cells import CONTROL_ARMS

    for config in CONTROL_ARMS.values():
        assert config.settle.per_kind == {"open_chrome": 2.0}
        assert config.settle.min_delay_s == 0.75


def test_an_unsupported_sign_of_life_kind_is_refused_at_construction() -> None:
    from evals.signoflife.guest import SignOfLifePreparer

    with pytest.raises(ValueError, match="unsupported sign-of-life kind"):
        SignOfLifePreparer("terminal_maybe_text")
