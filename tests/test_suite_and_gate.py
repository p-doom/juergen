"""The correctness of the gate itself.

`terminal_exact_text` is a submit cell by its own instruction, so listing it in
`NO_SUBMIT_CELLS` was the bug and indicator D must not fire there. Four independent
parts of the cell require the Return, each asserted separately below: the
instruction ends "and press Enter"; the guest fixture completes an `IFS= read -r`
only on a newline; the oracle requires the capture file that only a completed `read`
writes, so not submitting cannot pass and submitting is the only way to pass; and
the oracle control arm the calibration defines as 4/4 presses Return.

`terminal_exact_text` was the only entry, so indicator D never had a valid cell to
fire on in the scored tier; the first genuine no-submit cell is the candidate
`panel_no_submit_entry`. D is a `@vf.metric` and the flag is read in exactly one
place, so no published pass count — including the Phase-B-compact 2/4 — was ever a
function of it.

`load_suite` refuses a cell that is both listed as no-submit and phrased as a
submission, and refuses a `NO_SUBMIT_CELLS` entry naming no real cell.

The candidate tier's own tests are at the bottom of this file. They are the
non-VM half of rule 1 — the oracle accepts the gold realized state and rejects
the negative's, per cell — and they are not a substitute for measuring the oracle
arm on a real VM, which is what promotion to the scored tier requires.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest

import evals.indicators as indicators
import evals.signoflife.guest as guest
from evals.signoflife.cells import (
    ARMS,
    PHASEB_SYSTEM_PROMPT_SHA256,
    CONTROL_ARMS,
    COMPACT_CODEC,
    MODEL_ARMS,
    NATIVE_CODEC,
    ORDERED_CODEC,
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

MANIFEST_SHA256 = "014d334e3dc4dd93f4af9b3b2ba762423f5e1be0fba78b87a41b81f18b6b18c4"
SCORED_SHA256 = "f95e03c263c1c9befdd508f6a9b34a00f601137572b0345005cb0f2a382d0aa9"
EXACT_TEXT_CELL = "terminal_exact_text"


def _cell(cell_id: str = EXACT_TEXT_CELL):
    return load_suite().by_id(cell_id)


def test_14a_the_cell_is_no_longer_classified_no_submit() -> None:
    assert EXACT_TEXT_CELL not in NO_SUBMIT_CELLS
    rows = list(SignOfLifeTaskset(SignOfLifeTasksetConfig()).load())
    assert [t.data.name for t in rows if t.data.no_submit] == [], (
        "no cell in the scored tier is a no-submit cell"
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
    """The arm the calibration defines as 4/4 submits."""
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
    """The Return this cell requires must not be counted as over-generalisation."""
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

    Resolving it either way changes a metric, never `success`. A published 2/4 is a
    count of `postcondition` rewards, which never read this flag.
    """
    repo = Path(__file__).resolve().parents[1]
    # The flag, not the substring: `tk_no_submit_entry` and `panel_no_submit_entry`
    # are a cell kind and a cell id that merely contain the word, and counting them
    # as readers of the flag would make this test claim a coupling that is not there.
    flag = re.compile(r"(?<!\w)no_submit")
    readers = []
    for path in sorted(repo.glob("**/*.py")):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if flag.search(text):
            readers.append(path.relative_to(repo).as_posix())
    assert sorted(readers) == [
        "evals/indicators.py",  # the only *consumer*: indicator D
        "evals/signoflife/suite.py",  # documents the resolution (prose only)
        "evals/signoflife/taskset.py",  # sets it from NO_SUBMIT_CELLS
        "evals/tasks.py",  # declares the field
    ], readers
    # And the modules that decide `success` never mention it at all.
    for module in ("evals/oracles.py", "evals/harness.py", "evals/signoflife/oracle.py"):
        assert not flag.search((repo / module).read_text()), module
    oracle_source = inspect.getsource(evaluate_postcondition)
    assert not flag.search(oracle_source), "the oracle must not read the flag"


def test_14i_indicator_D_has_no_valid_cell_to_fire_on_in_the_scored_tier() -> None:
    """`terminal_exact_text` was the only entry.

    So every non-zero D reading ever published against the scored gate came from
    the misclassified cell, and it stays structurally zero there: the first genuine
    no-submit cell is `panel_no_submit_entry`, which is a candidate.
    """
    suite = load_suite()
    assert NO_SUBMIT_CELLS == frozenset({"panel_no_submit_entry"})
    assert {task.tier for task in suite.tasks if task.id in NO_SUBMIT_CELLS} == {
        "candidate"
    }, "a no-submit cell in the scored tier would change what the scored gate means"
    submit_everywhere = [
        task.id
        for task in suite.for_tier("scored")
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
    """So the misclassification cannot be reintroduced."""
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


def test_the_gate_is_six_controls_plus_three_model_arms() -> None:
    assert set(CONTROL_ARMS) == {
        "native_oracle",
        "native_negative",
        "compact_oracle",
        "compact_negative",
        "ordered_oracle",
        "ordered_negative",
    }
    assert set(MODEL_ARMS) == {"offshelf_native", "phaseb_compact", "ordered"}
    assert set(ARMS) == set(CONTROL_ARMS) | set(MODEL_ARMS)
    assert len(ARMS) == 9


def test_both_controls_exist_for_every_grammar_with_a_model_arm() -> None:
    """A model arm without its pair is an uncalibrated number.

    `ordered_events_v3` — the production format — had a training job running on it
    and no arm here at all, which is the same defect one step earlier.
    """
    by_codec: dict[str, set[bool]] = {}
    for config in CONTROL_ARMS.values():
        by_codec.setdefault(config.codec, set()).add(config.scripted.negative)
    assert by_codec == {
        NATIVE_CODEC: {False, True},
        COMPACT_CODEC: {False, True},
        ORDERED_CODEC: {False, True},
    }
    assert {config.codec for config in MODEL_ARMS.values()} <= set(by_codec)


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


def test_the_scored_digest_does_not_move_when_a_candidate_is_added() -> None:
    """Otherwise every candidate cell silently re-identifies the calibrated gate.

    `result.json` records both. A reader comparing two scored runs has to be able
    to tell "the scored cells changed" from "someone added a candidate", and the
    whole-manifest hash cannot say that.
    """
    suite = load_suite()
    assert suite.scored_sha256 == SCORED_SHA256
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    scored = [task for task in raw["tasks"] if task["tier"] == "scored"]
    assert hashlib.sha256(canonical_json(scored)).hexdigest() == SCORED_SHA256
    raw["tasks"].append(
        {
            "id": "another_candidate",
            "kind": "submit_only",
            "tier": "candidate",
            "instruction": "x",
            "expected": {"keystroke_prefix": ""},
            "max_steps": 3,
        }
    )
    grown = [task for task in raw["tasks"] if task["tier"] == "scored"]
    assert hashlib.sha256(canonical_json(grown)).hexdigest() == SCORED_SHA256
    assert hashlib.sha256(canonical_json(raw)).hexdigest() != MANIFEST_SHA256


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
        "schema": lambda v: v.update(schema_version=1),  # the pre-tier manifest
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
    # A duplicated kind is coverage drift even with unique ids.
    with pytest.raises(ValueError, match="one per kind"):
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


def test_every_shipped_arm_passes_its_own_prompt_digest_check(tmp_path) -> None:
    """The digest check is enforced, so a shipped arm that cannot pass it is a
    calibration cell that no longer runs.

    `phaseb_compact` is the case that matters: its expected digest is a prompt
    sealed before `describe()` existed, so it cannot match and cannot be
    recomputed. It carries the reason in `expect_prompt_mismatch`, which is what
    keeps the 4-cell gate whole while an UNjustified mismatch still fails loudly.
    """
    from agent.agent import load_codec
    from evals.harness import DesktopHarness

    for name, arm in ARMS.items():
        harness = DesktopHarness(arm.model_copy(update={"artifacts": arm.artifacts}))
        report = harness._prompt_report(load_codec(arm.codec))
        assert report["matches_expected"] in (None, True) or arm.expect_prompt_mismatch, (
            f"{name} would refuse to run"
        )

    phaseb = ARMS["phaseb_compact"]
    assert phaseb.system_prompt_sha256 == PHASEB_SYSTEM_PROMPT_SHA256
    assert phaseb.expect_prompt_mismatch, "the sealed-prompt arm must state why"
    report = DesktopHarness(phaseb)._prompt_report(load_codec(phaseb.codec))
    assert report["matches_expected"] is False
    assert "sealed" in report["expect_prompt_mismatch"]


# --- the candidate tier -------------------------------------------------------
#
# Rule 1 has two halves. This file holds the half that needs no VM: for each
# candidate cell, the oracle accepts the realized state its gold plan produces and
# rejects the state its negative plan produces. The other half — the oracle arm
# actually reading full marks on hardware — is what promotion to the scored tier
# waits for, and nothing here stands in for it.

CANDIDATE_STATES = {
    "terminal_submit_only": (
        {"keystroke_state": {"prefix": "", "prefix_len": 0, "completed": True}},
        # `type("\n")`: two literal characters, and the reader never completed.
        {"keystroke_state": {"prefix": "\\n", "prefix_len": 2, "completed": False}},
    ),
    "terminal_staged_confirm": (
        {"stage_one_text": "SOLV2-4718", "commit_text": "SOLV2-4718"},
        # Stage one answered, then stopped: exactly the premature-terminate shape.
        {"stage_one_text": "SOLV2-4718", "commit_text": None},
    ),
    "panel_offset_button": (
        {"panel_state": {"schema_version": 1, "clicked": ["Commit B3"]}},
        {"panel_state": {"schema_version": 1, "clicked": ["Commit B1"]}},
    ),
    "panel_no_submit_entry": (
        {
            "panel_state": {
                "schema_version": 1,
                "clicked": ["Save draft"],
                "entry_text": "Ada Lovelace",
                "submitted": False,
            }
        },
        # The reflexive Return: the text is right and the cell is still failed.
        {
            "panel_state": {
                "schema_version": 1,
                "clicked": [],
                "entry_text": "Ada Lovelace",
                "submitted": True,
            }
        },
    ),
}


@pytest.mark.parametrize("cell_id", sorted(CANDIDATE_STATES))
def test_a_candidate_oracle_accepts_its_gold_state_and_rejects_its_negative(
    cell_id: str,
) -> None:
    cell = load_suite().by_id(cell_id)
    assert cell.tier == "candidate"
    gold, bad = CANDIDATE_STATES[cell_id]
    identity = {"schema_version": 1, "task_id": cell.id}
    passed = evaluate_postcondition(
        cell.id, cell.kind, dict(cell.expected), {**identity, **gold}
    )
    failed = evaluate_postcondition(
        cell.id, cell.kind, dict(cell.expected), {**identity, **bad}
    )
    assert passed.status == "ok" and passed.success is True, passed.evidence
    assert failed.status == "ok" and failed.success is False, failed.evidence


@pytest.mark.parametrize("cell_id", sorted(CANDIDATE_STATES))
def test_a_candidate_cell_starts_unsolved(cell_id: str) -> None:
    """`require_unsolved_start` refuses an episode that begins already passing.

    The initial probe of each candidate is the fixture's own startup state: no
    keystrokes, no stage-one file, no clicks, an empty entry.
    """
    cell = load_suite().by_id(cell_id)
    initial = {
        "terminal_submit_only": {
            "keystroke_state": {"prefix": "", "prefix_len": 0, "completed": False}
        },
        "terminal_staged_confirm": {"stage_one_text": None, "commit_text": None},
        "panel_offset_button": {"panel_state": {"schema_version": 1, "clicked": []}},
        "panel_no_submit_entry": {
            "panel_state": {
                "schema_version": 1,
                "clicked": [],
                "entry_text": "",
                "submitted": False,
            }
        },
    }[cell_id]
    outcome = evaluate_postcondition(
        cell.id,
        cell.kind,
        dict(cell.expected),
        {"schema_version": 1, "task_id": cell.id, **initial},
    )
    assert outcome.status == "ok" and outcome.success is False, outcome.evidence


@pytest.mark.parametrize("cell_id", sorted(CANDIDATE_STATES))
def test_an_unreadable_candidate_fixture_is_an_error_not_a_failure(cell_id: str) -> None:
    """A panel that never published must not read as a model that never clicked."""
    cell = load_suite().by_id(cell_id)
    outcome = evaluate_postcondition(
        cell.id, cell.kind, dict(cell.expected), {"schema_version": 1, "task_id": cell.id}
    )
    if cell.kind == "staged_confirm":
        # This one is decided from file contents, and their absence IS the failure.
        assert outcome.status == "ok" and outcome.success is False
    else:
        assert outcome.status == "error", outcome.reason


def test_the_candidate_tier_is_not_in_the_scored_run() -> None:
    """The dilution guard: one run is one tier, and `scored` is the default."""
    suite = load_suite()
    scored = {task.id for task in suite.for_tier("scored")}
    candidates = {task.id for task in suite.for_tier("candidate")}
    assert scored.isdisjoint(candidates)
    assert candidates == set(CANDIDATE_STATES)
    default = {t.data.name for t in SignOfLifeTaskset(SignOfLifeTasksetConfig()).load()}
    assert default == scored
    asked = {
        t.data.name
        for t in SignOfLifeTaskset(SignOfLifeTasksetConfig(tier="candidate")).load()
    }
    assert asked == candidates
    for row in SignOfLifeTaskset(SignOfLifeTasksetConfig(tier="candidate")).load():
        assert row.data.setup["suite_tier"] == "candidate", (
            "every episode records its own tier, or a result.json cannot say which "
            "cells the number is over"
        )


def test_an_unknown_tier_is_refused_rather_than_silently_empty() -> None:
    with pytest.raises(ValueError, match="unknown suite tier"):
        load_suite().for_tier("candidates")
    with pytest.raises(ValueError, match="unknown suite tier"):
        list(SignOfLifeTaskset(SignOfLifeTasksetConfig(tier="")).load())


def test_the_loader_refuses_a_cell_with_no_tier_or_a_bad_one() -> None:
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

    for name, mutate in {
        "missing_tier": lambda v: v["tasks"][0].pop("tier"),
        "unknown_tier": lambda v: v["tasks"][0].update(tier="provisional"),
        "candidate_promoted_without_review": lambda v: v["tasks"][4].update(
            tier="scored"
        ),
        "scored_demoted": lambda v: v["tasks"][0].update(tier="candidate"),
        "candidate_kind_dropped": lambda v: v["tasks"].pop(),
        "duplicate_candidate_kind": lambda v: v["tasks"][5].update(
            kind=v["tasks"][4]["kind"]
        ),
    }.items():
        with pytest.raises(ValueError):
            load_mutated(mutate)
            raise AssertionError(name)


def test_every_candidate_cell_has_both_control_plans_and_they_differ() -> None:
    from evals.tasks import DesktopTaskData

    for cell in load_suite().for_tier("candidate"):
        data = DesktopTaskData(
            idx=0,
            name=cell.id,
            prompt=cell.instruction,
            instruction=cell.instruction,
            kind=cell.kind,
            max_steps=cell.max_steps,
            expected=dict(cell.expected),
        )
        gold = guest.script_plan(data, negative=False)
        bad = guest.script_plan(data, negative=True)
        assert gold and bad and gold != bad, cell.id
        assert len(gold) <= cell.max_steps, (
            f"{cell.id}: the gold plan does not fit in max_steps, so the oracle arm "
            "would run out of turns before the postcondition"
        )
        assert len(bad) <= cell.max_steps, cell.id


def test_the_submit_only_negative_is_the_literal_newline_defect() -> None:
    """The measured defect, verbatim: `0 0 0 ; type("ls\\n")` typed a backslash and
    an `n` in all three Phase-B draws instead of pressing Return.

    So the negative control for this cell types exactly that and nothing else, and
    it must survive the grammar round trip as two literal characters — if the
    payload were a real newline, `lower_typing` would refuse it and the negative
    would be a parse error rather than a dispatched wrong action.
    """
    from agent.agent import _action_record, load_codec

    from evals.indicators import typed_texts

    cell = load_suite().by_id("terminal_submit_only")
    data = _candidate_data(cell)
    plan = guest.script_plan(data, negative=True)
    assert [intent.kind for intent in plan] == ["type"]
    for codec_name in (COMPACT_CODEC, NATIVE_CODEC, ORDERED_CODEC):
        codec = load_codec(codec_name)
        text = guest.render_step(None, data, codec=codec, intent=plan[0])
        typed = typed_texts(_action_record(codec.parse(text)))
        assert typed == ["\\n"], (codec_name, text, typed)
        assert "\n" not in typed[0], "a real newline would be refused, not dispatched"


def _candidate_data(cell):
    from evals.tasks import DesktopTaskData

    return DesktopTaskData(
        idx=0,
        name=cell.id,
        prompt=cell.instruction,
        instruction=cell.instruction,
        kind=cell.kind,
        max_steps=cell.max_steps,
        expected=dict(cell.expected),
        no_submit=cell.id in NO_SUBMIT_CELLS,
    )


def test_indicator_D_fires_on_the_no_submit_cells_negative_and_not_on_its_gold() -> None:
    """The first cell that gives indicator D something real to measure.

    D has been structurally zero on this gate since its only entry was removed as a
    misclassification. `panel_no_submit_entry` is a cell whose success genuinely
    requires not submitting, so D must read 1.0 for the reflexive-Return negative
    and 0.0 for the gold plan.
    """
    cell = load_suite().by_id("panel_no_submit_entry")
    data = _candidate_data(cell)
    assert data.no_submit is True
    readings = {}
    for negative in (False, True):
        steps = []
        for intent in guest.script_plan(data, negative=negative):
            if intent.kind == "click":
                continue  # the click needs the guest's measured geometry
            text = guest._render_relative(intent, session=None, task=data)
            steps.append(
                {"raw_model_output": text, "parsed_action": _parse(COMPACT_CODEC, text)}
            )
        trace = make_trace(data, episode={"success": not negative, "steps_detail": steps})
        asyncio.run(SignOfLifeTask(data).score(trace))
        readings[negative] = trace.metrics["D_submitted_in_no_submit_cell"]
    assert readings == {False: 0.0, True: 1.0}, readings
