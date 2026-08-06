"""Items 14 + 17 — the correctness of the gate itself.

`terminal_exact_text` is a **submit** cell
by its own instruction, so listing it in `NO_SUBMIT_CELLS` was the bug and indicator D
must not fire there.

The evidence that decided it, each asserted separately below:

  * the cell's instruction ends *"and press Enter"* (14b);
  * its guest fixture completes an `IFS= read -r` only on a newline (14c);
  * its oracle requires the capture file that only a completed `read` writes, so
    NOT submitting cannot pass and submitting is the only way to pass (14d);
  * its **oracle control arm — defined by the calibration as 4/4 — presses Return** (14e).

An indicator that fires on the behaviour four independent parts of the cell require,
including the gold plan, measures nothing about the model.

Two consequences, both asserted:

  * `terminal_exact_text` was the **only** entry, so `NO_SUBMIT_CELLS` is now empty and
    indicator D never had a valid cell to fire on in this suite (14a, 14i);
  * D is a `@vf.metric` and `no_submit` is read in exactly one place, so no published
    pass count — including the Phase-B-compact 2/4 — was ever a function of it (14g).

The contradiction is now structurally unreachable: `load_suite` refuses a cell that is
both listed as no-submit and phrased as a submission (14j), and refuses a
`NO_SUBMIT_CELLS` entry naming no real cell (14k).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import evals.indicators as indicators
import evals.signoflife.guest as guest
from evals.signoflife.cells import (
    ARMS,
    CONTROL_ARMS,
    COMPACT_CODEC,
    MODEL_ARMS,
    NATIVE_CODEC,
    verify_phaseb_provenance,
)
from evals.signoflife.oracle import evaluate_postcondition, history_has_exact
from evals.signoflife.suite import (
    ALLOWED_KINDS,
    NO_SUBMIT_CELLS,
    SUBMIT_VERBS,
    SUITE_PATH,
    canonical_json,
    load_suite,
)
from evals.signoflife.taskset import SignOfLifeTask, SignOfLifeTaskset, SignOfLifeTasksetConfig
from juergen_doubles import make_trace

MANIFEST_SHA256 = "1bf13a84808fd144cf6565c61a303d97e37716d7129f22f9ae46a4dcc3bfbaac"
EXACT_TEXT_CELL = "terminal_exact_text"


def _cell(cell_id: str = EXACT_TEXT_CELL):
    return load_suite().by_id(cell_id)


def test_14a_the_cell_is_no_longer_classified_no_submit() -> None:
    assert EXACT_TEXT_CELL not in NO_SUBMIT_CELLS
    rows = list(SignOfLifeTaskset(SignOfLifeTasksetConfig()).load())
    assert [t.data.name for t in rows if t.data.no_submit] == [], (
        "no cell in this suite is a no-submit cell"
    )


def test_14b_the_cells_own_instruction_demands_submission() -> None:
    assert "press Enter" in _cell().instruction, _cell().instruction


def test_14c_the_guest_fixture_completes_only_on_a_newline() -> None:
    """`IFS= read -r` returns on newline (or EOF). No Return, no capture file."""
    source = inspect.getsource(guest._setup_terminal_exact_text)
    assert "read -r SOLV2_LINE" in source
    capture_index = source.index("captured.txt")
    read_index = source.index("read -r SOLV2_LINE")
    assert read_index < capture_index, (
        "the capture file is written by the line AFTER read returns, so its very "
        "existence is evidence that Enter was pressed"
    )


def test_14d_the_oracle_requires_the_file_that_only_enter_creates() -> None:
    cell = _cell()
    wanted = cell.expected["text"]
    # Typed but never submitted: `read` has not returned, so no capture file.
    unsubmitted = evaluate_postcondition(
        cell.id,
        cell.kind,
        dict(cell.expected),
        {
            "schema_version": 1,
            "task_id": cell.id,
            "capture_file_exists": False,
            "captured_text": None,
        },
    )
    assert unsubmitted.status == "ok"
    assert unsubmitted.success is False, "not submitting cannot pass this cell"
    submitted = evaluate_postcondition(
        cell.id,
        cell.kind,
        dict(cell.expected),
        {
            "schema_version": 1,
            "task_id": cell.id,
            "capture_file_exists": True,
            "captured_text": wanted,
        },
    )
    assert submitted.success is True, "submitting is the only way to pass this cell"


def test_14e_the_oracle_control_arm_for_this_cell_presses_return() -> None:
    """The arm the calibration *defines* as 4/4 submits."""
    plan = guest.script_plan(_cell_data(), negative=False)
    assert [intent.kind for intent in plan] == ["type", "submit"]
    rendered = [
        guest._render_relative(intent, session=None, task=_cell_data()) for intent in plan
    ]
    assert rendered[1] == "0 0 0 ; +Return -Return"


def _cell_data():
    from evals.tasks import DesktopTaskData

    cell = _cell()
    return DesktopTaskData(
        idx=1,
        name=cell.id,
        prompt=cell.instruction,
        instruction=cell.instruction,
        kind=cell.kind,
        max_steps=cell.max_steps,
        expected=dict(cell.expected),
        no_submit=cell.id in NO_SUBMIT_CELLS,
    )


def test_indicator_D_does_not_fire_on_the_gold_plan() -> None:
    """The Return that the cell's instruction demands, that its fixture requires,
    that its oracle scores as the only pass, and that its own oracle control arm
    emits, must not be counted as over-generalisation.
    """
    task_data = _cell_data()
    steps = []
    for intent in guest.script_plan(task_data, negative=False):
        text = guest._render_relative(intent, session=None, task=task_data)
        steps.append(
            {
                "raw_model_output": text,
                "parsed_action": _parse(COMPACT_CODEC, text),
            }
        )
    trace = make_trace(
        task_data,
        episode={"success": True, "outcome": "postcondition_reached", "steps_detail": steps},
    )
    task = SignOfLifeTask(task_data)
    asyncio.run(task.score(trace))
    assert trace.metrics["D_submitted_in_no_submit_cell"] == 0.0, (
        "indicator D must not flag the gold, calibrated, oracle-arm behaviour"
    )
    assert task_data.no_submit is False
    assert trace.metrics["B_same_action_submit_actions"] == 0.0, (
        "B is clean: type and submit are separate steps here, so B is not the "
        "indicator in conflict — D specifically is"
    )


def _parse(codec_name: str, text: str):
    from agent.agent import _action_record, load_codec

    return _action_record(load_codec(codec_name).parse(text))


def test_14g_the_misclassification_cannot_move_a_pass_count() -> None:
    """`no_submit` feeds indicator D and nothing else.

    So resolving item 14 either way changes a *metric*, never `success`. A published
    2/4 is a count of `postcondition` rewards, which never read this flag.
    """
    repo = Path(__file__).resolve().parents[1]
    readers = []
    for path in sorted(repo.glob("**/*.py")):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "no_submit" in text:
            readers.append(path.relative_to(repo).as_posix())
    assert sorted(readers) == [
        "evals/indicators.py",  # the only *consumer*: indicator D
        "evals/signoflife/suite.py",  # documents the resolution (prose only)
        "evals/signoflife/taskset.py",  # sets it from NO_SUBMIT_CELLS
        "evals/tasks.py",  # declares the field
    ], readers
    # And the modules that decide `success` never mention it at all.
    for module in ("evals/oracles.py", "evals/harness.py", "evals/signoflife/oracle.py"):
        assert "no_submit" not in (repo / module).read_text(), module
    oracle_source = inspect.getsource(evaluate_postcondition)
    assert "no_submit" not in oracle_source, "the oracle must not read the flag"


def test_14i_indicator_D_has_no_valid_cell_to_fire_on_in_this_suite() -> None:
    """The larger finding: `terminal_exact_text` was the ONLY entry.

    So every non-zero D reading ever published against the sign-of-life gate came
    from the misclassified cell. D is structurally zero on this suite now, and stays
    that way until a genuine no-submit cell is added.
    """
    assert NO_SUBMIT_CELLS == frozenset()
    submit_everywhere = [
        task.id
        for task in load_suite().tasks
        if any(verb in task.instruction.casefold() for verb in SUBMIT_VERBS)
    ]
    assert sorted(submit_everywhere) == [
        "focus_terminal_and_type",
        "terminal_exact_text",
        "terminal_ls",
    ], submit_everywhere
    rows = list(SignOfLifeTaskset(SignOfLifeTasksetConfig()).load())
    for row in rows:
        steps = [
            {
                "raw_model_output": "0 0 0 ; +Return -Return",
                "parsed_action": _parse(COMPACT_CODEC, "0 0 0 ; +Return -Return"),
            }
        ]
        trace = make_trace(row.data, episode={"success": True, "steps_detail": steps})
        asyncio.run(SignOfLifeTask(row.data).score(trace))
        assert trace.metrics["D_submitted_in_no_submit_cell"] == 0.0, row.data.name


def test_14j_the_loader_refuses_a_no_submit_cell_that_demands_submission(monkeypatch) -> None:
    """The structural invariant: the item-14 defect class cannot be reintroduced."""
    import evals.signoflife.suite as suite_module

    monkeypatch.setattr(suite_module, "NO_SUBMIT_CELLS", frozenset({EXACT_TEXT_CELL}))
    with pytest.raises(ValueError, match="its own instruction requires submission"):
        suite_module.load_suite()


def test_14k_the_loader_refuses_a_no_submit_entry_naming_no_real_cell(monkeypatch) -> None:
    """So renaming a cell can never silently drop the flag."""
    import evals.signoflife.suite as suite_module

    monkeypatch.setattr(suite_module, "NO_SUBMIT_CELLS", frozenset({"renamed_away"}))
    with pytest.raises(ValueError, match="not in the suite"):
        suite_module.load_suite()


def test_14l_a_genuine_no_submit_cell_is_still_supported(monkeypatch) -> None:
    """The fix must not make the flag unusable — only inconsistent uses are refused."""
    import evals.signoflife.suite as suite_module

    monkeypatch.setattr(suite_module, "NO_SUBMIT_CELLS", frozenset({"desktop_open_chrome"}))
    suite = suite_module.load_suite()  # the Chrome cell demands no submission
    assert "desktop_open_chrome" in suite_module.NO_SUBMIT_CELLS
    assert not any(
        verb in suite.by_id("desktop_open_chrome").instruction.casefold()
        for verb in SUBMIT_VERBS
    )


def test_14m_the_cell_id_and_kind_coincidence_that_hid_the_coupling() -> None:
    """Recorded: the removed entry was both a valid cell id and a valid cell kind, so
    `cell.id in NO_SUBMIT_CELLS` read correctly by coincidence. 14k now closes that."""
    cell = _cell()
    assert cell.id == cell.kind == EXACT_TEXT_CELL


def test_the_gate_is_four_controls_plus_two_model_arms() -> None:
    assert set(CONTROL_ARMS) == {
        "native_oracle",
        "native_negative",
        "compact_oracle",
        "compact_negative",
    }
    assert set(MODEL_ARMS) == {"offshelf_native", "phaseb_compact"}
    assert set(ARMS) == set(CONTROL_ARMS) | set(MODEL_ARMS)
    assert len(ARMS) == 6


def test_both_controls_exist_per_grammar() -> None:
    """A control that certified only one grammar leaves the other's parse path unmeasured."""
    by_codec: dict[str, set[bool]] = {}
    for config in CONTROL_ARMS.values():
        by_codec.setdefault(config.codec, set()).add(config.scripted.negative)
    assert by_codec == {NATIVE_CODEC: {False, True}, COMPACT_CODEC: {False, True}}


def test_control_arms_are_scripted_and_model_arms_are_not() -> None:
    assert all(c.scripted.enabled for c in CONTROL_ARMS.values())
    assert not any(c.scripted.enabled for c in MODEL_ARMS.values())


@pytest.mark.parametrize("name", sorted(CONTROL_ARMS))
def test_control_ok_encodes_oracle_4_of_4_and_negative_0_of_4(name: str) -> None:
    """`_control_ok` is the calibration assertion; this pins its truth table."""
    from evals.harness import DesktopHarness
    from evals.tasks import DesktopState

    config = CONTROL_ARMS[name]
    harness = DesktopHarness(config)
    negative = config.scripted.negative
    for success in (True, False):
        state = DesktopState(
            scripted=True, negative_control=negative, success=success, infra_error=None
        )
        expected = success is not negative
        assert harness._control_ok(state) is expected, (name, success)
    # Infra failure is never conformant, for either polarity.
    assert not harness._control_ok(
        DesktopState(
            scripted=True,
            negative_control=negative,
            success=None,
            infra_error={"stage": "episode"},
        )
    )


def test_a_model_arm_has_no_conformance_verdict_at_all() -> None:
    """Not `True`: a model arm has no expected value, so any verdict computed from its
    own rows restates the measurement. `None`, and the field disappears from the
    provenance metric."""
    from evals.harness import DesktopHarness
    from evals.tasks import DesktopState

    harness = DesktopHarness(MODEL_ARMS["phaseb_compact"])
    for success in (True, False, None):
        assert harness._control_ok(DesktopState(scripted=False, success=success)) is None
    trace = make_trace(episode={"control_ok": None, "validity": "valid"})
    assert "control_conformant" not in asyncio.run(harness.harness_provenance(trace))


def test_negative_plans_are_plausible_wrong_actions_not_no_ops() -> None:
    """A no-op negative would prove only that doing nothing fails."""
    from evals.tasks import DesktopTaskData

    for cell in load_suite().tasks:
        data = DesktopTaskData(
            idx=cell.idx if hasattr(cell, "idx") else 0,
            name=cell.id,
            prompt=cell.instruction,
            instruction=cell.instruction,
            kind=cell.kind,
            max_steps=cell.max_steps,
            expected=dict(cell.expected),
        )
        gold = guest.script_plan(data, negative=False)
        bad = guest.script_plan(data, negative=True)
        assert bad, f"{cell.id}: the negative arm must still act"
        assert gold != bad, f"{cell.id}: the negative arm must differ from the gold"


def test_negative_plans_cannot_satisfy_their_own_oracle() -> None:
    """0/4 must be a property of the plan, not luck."""
    suite = load_suite()
    text_cell = suite.by_id(EXACT_TEXT_CELL)
    outcome = evaluate_postcondition(
        text_cell.id,
        text_cell.kind,
        dict(text_cell.expected),
        {
            "schema_version": 1,
            "task_id": text_cell.id,
            "capture_file_exists": True,
            "captured_text": "wrong text",
        },
    )
    assert outcome.success is False
    ls_cell = suite.by_id("terminal_ls")
    outcome = evaluate_postcondition(
        ls_cell.id,
        ls_cell.kind,
        dict(ls_cell.expected),
        {
            "schema_version": 1,
            "task_id": ls_cell.id,
            "history": "pwd\n",
            "transcript": "SOLV2-LS$ pwd\n/tmp\nSOLV2-LS$ ",
            "prompt_count": 2,
        },
    )
    assert outcome.success is False, "`pwd` must not satisfy the `ls` cell"


def test_the_manifest_sha256_still_holds() -> None:
    suite = load_suite()
    assert suite.manifest_sha256 == MANIFEST_SHA256
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(canonical_json(raw)).hexdigest() == MANIFEST_SHA256


def test_the_suite_json_copy_is_byte_identical_to_its_source() -> None:
    """The in-package copy is the source; any second copy in the tree must match it."""
    repo = Path(__file__).resolve().parents[1]
    canonical = SUITE_PATH.read_bytes()
    copies = [
        p
        for p in repo.glob("**/suite.json")
        if ".venv" not in p.parts and p != SUITE_PATH
    ]
    for copy in copies:
        assert copy.read_bytes() == canonical, f"{copy} has drifted from {SUITE_PATH}"


def test_the_loader_refuses_drift() -> None:
    raw = json.loads(SUITE_PATH.read_text())
    assert {t["kind"] for t in raw["tasks"]} == ALLOWED_KINDS

    import tempfile

    def load_mutated(mutate) -> None:
        value = json.loads(SUITE_PATH.read_text())
        mutate(value)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(value, handle)
            path = Path(handle.name)
        try:
            load_suite(path)
        finally:
            path.unlink(missing_ok=True)

    cases = {
        "schema": lambda v: v.update(schema_version=2),
        "role": lambda v: v.update(role="benchmark"),
        "final_benchmark": lambda v: v.update(final_benchmark=True),
        "three_cells": lambda v: v["tasks"].pop(),
        "duplicate_id": lambda v: v["tasks"][1].update(id=v["tasks"][0]["id"]),
        "bad_kind": lambda v: v["tasks"][0].update(kind="ssh_in_and_hope"),
        "empty_instruction": lambda v: v["tasks"][0].update(instruction=""),
        "empty_expected": lambda v: v["tasks"][0].update(expected={}),
        "max_steps_zero": lambda v: v["tasks"][0].update(max_steps=0),
        "max_steps_thirteen": lambda v: v["tasks"][0].update(max_steps=13),
        "max_steps_bool": lambda v: v["tasks"][0].update(max_steps=True),
    }
    for name, mutate in cases.items():
        with pytest.raises(ValueError):
            load_mutated(mutate)
    # A duplicated kind is coverage drift even with four unique ids.
    with pytest.raises(ValueError, match="coverage drift"):
        load_mutated(lambda v: v["tasks"][1].update(kind=v["tasks"][0]["kind"]))


def test_history_has_exact_is_exact() -> None:
    assert history_has_exact("  12  ls\n", "ls")
    assert history_has_exact("ls", "ls")
    assert not history_has_exact("ls -la", "ls"), "a superstring must not pass"
    assert not history_has_exact(None, "ls")
    assert not history_has_exact(b"ls", "ls")


def test_an_unreadable_probe_is_an_error_not_a_failure() -> None:
    """Collapsing status='error' into success=False is how a broken probe becomes 0/4."""
    outcome = evaluate_postcondition("terminal_ls", "terminal_command", {}, {})
    assert outcome.status == "error"
    assert outcome.success is False
    assert "schema" in outcome.reason or "identity" in outcome.reason
    mismatched = evaluate_postcondition(
        "terminal_ls",
        "terminal_command",
        {"command": "ls", "listing_marker": "m"},
        {"schema_version": 1, "task_id": "someone_elses_cell"},
    )
    assert mismatched.status == "error"


def test_verify_phaseb_provenance_fails_closed_on_missing_registration(tmp_path: Path) -> None:
    model = tmp_path / "step_900"
    model.mkdir()
    with pytest.raises(RuntimeError, match="registration metadata is missing"):
        verify_phaseb_provenance(model)


def test_verify_phaseb_provenance_rejects_a_substituted_export(tmp_path: Path) -> None:
    root = tmp_path
    model = root / "step_900"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    (root / ".meta.json").write_text(json.dumps({"id": "artifact_wrong", "producer_run_id": "x"}))
    (root / "export_manifest.json").write_text(
        json.dumps(
            {
                "arm": "raw_v2",
                "step": 900,
                "lora_rank": 256,
                "lora_alpha": 256,
                "model_id": "Qwen/Qwen3-VL-8B-Instruct",
                "status": "complete",
                "weights": [{"name": "model.safetensors", "size": 7, "sha256": "deadbeef"}],
            }
        )
    )
    with pytest.raises(RuntimeError, match="provenance mismatch") as excinfo:
        verify_phaseb_provenance(model)
    message = str(excinfo.value)
    assert "artifact_id" in message and "export_manifest_sha256" in message


def test_the_paired_group_reward_is_not_mixed_into_the_gate_task() -> None:
    """A `@vf.group_reward` makes the episode require n >= 2, breaking a 1-sample gate."""
    from evals.oracles import PairedArmDivergence

    assert not issubclass(SignOfLifeTask, PairedArmDivergence)
    from verifiers.v1.decorators import discover_decorated

    task = SignOfLifeTask(_cell_data())
    assert discover_decorated(task, "group_reward") == []


def test_the_gate_reports_oracle_metrics_in_a_stable_order() -> None:
    """`discover_decorated` sorts by (priority, name), so no mixin shadows another."""
    from verifiers.v1.decorators import discover_decorated

    task = SignOfLifeTask(_cell_data())
    names = [fn.__name__ for fn in discover_decorated(task, "metric")]
    assert names == sorted(names)
    assert {"failure_modes", "mouse", "sampling", "postcondition_recorded"} <= set(names)
    rewards = [fn.__name__ for fn in discover_decorated(task, "reward")]
    assert rewards == ["postcondition"], rewards
