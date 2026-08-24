"""`python -m evals.signoflife`, end to end, with no VM and no GPU.

The path every real gate run takes — CLI -> argument validation -> taskset load ->
harness construction -> pool adapter -> episode -> `result.json` — and where the
missing `DesktopFacade.evaluate()` once hid.

This runs the real `main()`. The only substitution is
`desktop.vm.factory.build_desktop_pool` (see `juergen_fake_desktop`), so
`evals.vm.kvm_desktop_pool` and `DesktopFacade` are the production objects under
test rather than doubles.

Scoring semantics are untouched. The guest reports a desktop where nothing happened,
which is the negative control's calibrated reading — 0/4 with `control_ok` true on
every cell. Nothing here asserts, changes or fabricates the oracle arm's 4/4: that
is a calibration against a real VM, and a fixture that produced it would be fiction.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import juergen_fake_desktop
import pytest
from test_model_attestation import _register_model

from evals.signoflife.__main__ import main
from evals.signoflife.suite import load_suite

# The scored tier: what an unqualified run of this dispatcher is. `CANDIDATE_IDS`
# is the other tier, and it is covered too -- `juergen_fake_desktop` serves the Tk
# fixture's published measurement, so the panel cells' dispatch path is exercised
# here without a VM.
CELL_IDS = [task.id for task in load_suite().for_tier("scored")]
CANDIDATE_IDS = [task.id for task in load_suite().for_tier("candidate")]


@pytest.fixture(autouse=True)
def _fake_desktop(monkeypatch):
    """Substitute the one function between the dispatcher and a booted VM.

    `agent.desktop._POOLS` is process-global and keyed by arm name, so without
    the teardown the second test in a session would silently reuse the first
    one's pool, its slot directory and its already-checked-out sessions.
    """
    import agent.desktop as desktop
    import desktop.vm.factory as factory
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setenv("SIGN_OF_LIFE_API_KEY", "test-only-secret")
    juergen_fake_desktop.FakeDesktopPool.instances.clear()
    monkeypatch.setattr(
        factory, "build_desktop_pool", juergen_fake_desktop.build_desktop_pool
    )
    # Production uses spawn. The fake factory is an in-process test substitution,
    # so fork is the exact way for a worker to inherit it without adding a
    # production CLI escape hatch for pool ownership.
    monkeypatch.setattr(dispatcher, "_WORKER_START_METHOD", "fork")
    yield
    for pool in list(desktop._POOLS.values()):
        pool.close()
    desktop._POOLS.clear()
    juergen_fake_desktop.FakeDesktopPool.instances.clear()


def _argv(output: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--arm",
        "native_negative",
        "--output",
        str(output),
        "--qcow",
        str(tmp_path / "desktop.qcow2"),
        # Both are real flags and both are here for wall clock: the default
        # 120 s scoring grace plus a single node slot serialises the cells
        # behind a 120 s lease each, and the reaper only looks every 15 s.
        "--scoring-grace-s",
        "0",
        "--vm-slots",
        "4",
        *extra,
    ]


def _fresh_process() -> None:
    """Model what a real dispatch is: one `python -m evals.signoflife`, one pool.

    `agent.desktop.pool_for` deliberately refuses to hand back a pool registered
    under the same key with a different `PoolSpec` — returning the live one would
    run the episode under someone else's slot budget and TTLs. The dispatcher's
    key is `signoflife-<arm>` while its `slot_dir` is derived from `--output`, so
    two runs of one arm in one process collide by construction. In production
    they never share a process; in a test they do, so the process-global registry
    is reset instead of the guard being weakened.
    """
    import agent.desktop as desktop

    for pool in list(desktop._POOLS.values()):
        pool.close()
    desktop._POOLS.clear()


def _run(output: Path, tmp_path: Path, *extra: str) -> tuple[int, dict]:
    _fresh_process()
    code = main(_argv(output, tmp_path, *extra))
    return code, json.loads((output / "result.json").read_text())


@pytest.mark.slow
def test_the_dispatcher_runs_the_whole_scored_tier_and_writes_the_readers_shape(
    tmp_path, capsys,
) -> None:
    output = tmp_path / "run"
    code, result = _run(output, tmp_path)

    assert code == 0, result["infrastructure_errors"]
    assert not list(tmp_path.glob(".run.staging-*"))
    assert result["status"] == "complete"
    assert result["schema_version"] == 3
    assert result["arm"] == "native_negative"
    assert result["arm_kind"] == "scripted_negative"

    aggregate = result["aggregate"]
    assert sorted(aggregate["per_cell"]) == sorted(CELL_IDS), "the gate is the whole scored tier"
    for cell, row in aggregate["per_cell"].items():
        assert row["trials"] == 1, cell
        assert row["valid_trials"] == 1, f"{cell}: {result['infrastructure_errors']}"
        assert row["pass_rate"] == 0.0, f"{cell} is a negative control"
        assert len(row["outcomes"]) == 1
    assert aggregate["expected_per_cell_pass_rate"] == 0.0
    assert aggregate["controls_ok"] is True
    assert result["suite_manifest_sha256"] == load_suite().manifest_sha256
    assert len(result["episodes"]) == len(CELL_IDS)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["episodes"] == len(CELL_IDS)
    assert printed["valid_episodes"] == len(CELL_IDS)


@pytest.mark.slow
def test_the_pool_adapter_is_the_production_one(tmp_path) -> None:
    """`pool_target` names `evals.vm:kvm_desktop_pool`, and that is what runs:
    the JSON-able arguments become a `DesktopPoolConfig`, and every checkout comes
    back as a `DesktopFacade`, the merged surface an episode needs."""
    output = tmp_path / "run"
    code, result = _run(output, tmp_path, "--vm-smp", "4", "--vm-mem", "8G")
    assert code == 0
    assert result["vm"]["smp"] == 4 and result["vm"]["memory"] == "8G"
    assert result["vm"]["sessions_per_worker"] == 1
    assert result["vm"]["slots"] == 4
    assert all(row["validity"] == "valid" for row in result["episodes"])


@pytest.mark.slow
def test_the_pool_flags_reach_the_pool(tmp_path) -> None:
    """`--scoring-grace-s` was pinned in code with no way to name it, and
    `reap_interval_s` is what decides how long after that grace the VM actually
    goes away. Both are now on the config the harness builds its `PoolSpec` from."""
    from evals.harness import DesktopPoolConfig
    from evals.signoflife.__main__ import _harness_payload

    output = tmp_path / "run"
    code, result = _run(output, tmp_path)
    assert code == 0
    assert result["aggregate"]["controls_ok"] is True

    payload = _harness_payload(
        "native_negative",
        artifacts=output,
        pool={"scoring_grace_s": 7.5, "pool_target": "evals.vm:kvm_desktop_pool"},
    )
    pool = DesktopPoolConfig(**payload["pool"])
    assert pool.scoring_grace_s == 7.5
    assert pool.pool_target == "evals.vm:kvm_desktop_pool"
    # The reaper interval is what decides how long *after* the grace the VM
    # actually goes away, and it is only reachable because the same config object
    # carries both. Both are bounded, so neither can be turned off by accident.
    assert pool.reap_interval_s == 15.0
    with pytest.raises(ValueError):
        DesktopPoolConfig(reap_interval_s=0.0)
    with pytest.raises(ValueError):
        DesktopPoolConfig(scoring_grace_s=-1.0)


@pytest.mark.slow
def test_a_single_cell_can_be_reproduced_by_id_and_by_index(tmp_path) -> None:
    """`--cell` and `--task-index` are mutually exclusive and both narrow the
    taskset; the gate is only a gate when all four run, which is why the result
    records the full suite size next to the selection."""
    by_id = tmp_path / "by-id"
    code, result = _run(by_id, tmp_path, "--cell", CELL_IDS[0])
    assert code == 0
    assert result["selection"] == {
        "task_ids": [CELL_IDS[0]],
        "full_tier_task_count": len(CELL_IDS),
    }
    assert list(result["aggregate"]["per_cell"]) == [CELL_IDS[0]]

    by_index = tmp_path / "by-index"
    code, result = _run(by_index, tmp_path, "--task-index", "2")
    assert code == 0
    assert result["selection"]["task_ids"] == [CELL_IDS[2]]


@pytest.mark.slow
def test_trials_are_separate_passes_that_keep_their_own_artifacts(tmp_path) -> None:
    """N rollouts of one task overwrite each other's frames and `result.json`
    (`_artifact_dir` keys on the task name alone), so a trial is a whole pass."""
    output = tmp_path / "run"
    code, result = _run(output, tmp_path, "--cell", CELL_IDS[0], "--trials", "2")
    assert code == 0
    assert result["trials"] == 2
    row = result["aggregate"]["per_cell"][CELL_IDS[0]]
    assert row["trials"] == 2 and row["valid_trials"] == 2
    assert row["pass_rate"] == 0.0
    assert (output / "trial_01").is_dir() and (output / "trial_02").is_dir()
    assert {episode["trial"] for episode in result["episodes"]} == {1, 2}


@pytest.mark.slow
def test_the_result_is_written_atomically_and_leaves_no_partial(tmp_path) -> None:
    output = tmp_path / "run"
    code, _ = _run(output, tmp_path, "--cell", CELL_IDS[0])
    assert code == 0
    assert (output / "result.json").is_file()
    assert not (output / "result.json.partial").exists()


@pytest.mark.slow
def test_the_ordered_events_v3_arm_runs_the_whole_dispatch(tmp_path) -> None:
    """The production format had no arm at all — no model leg and no renderer.

    Only the negative half is reachable without a VM: this guest reports a desktop
    where nothing happened, which is exactly the negative control's calibrated
    reading. The oracle's 4/4 is a real-VM calibration and a fixture that produced
    it would be fiction, so it is not asserted here.
    """
    output = tmp_path / "run"
    code, result = _run(output, tmp_path, "--arm", "ordered_negative")

    assert code == 0, result["infrastructure_errors"]
    assert result["codec"] == "ordered_events_v3"
    assert result["arm_kind"] == "scripted_negative"
    aggregate = result["aggregate"]
    assert sorted(aggregate["per_cell"]) == sorted(CELL_IDS), "the gate is the whole scored tier"
    for cell, row in aggregate["per_cell"].items():
        assert row["valid_trials"] == 1, f"{cell}: {result['infrastructure_errors']}"
        assert row["pass_rate"] == 0.0, f"{cell} is a negative control"
    assert aggregate["controls_ok"] is True
    # Every scripted step went through this grammar's own parse and compile.
    assert all(episode["parse_errors"] == 0 for episode in result["episodes"])


def test_a_scripted_arm_refuses_a_model_it_would_silently_ignore(tmp_path) -> None:
    """Recorded and ignored, the run record would name a checkpoint that never
    answered a single token."""
    with pytest.raises(SystemExit) as excinfo:
        main(_argv(tmp_path / "run", tmp_path, "--model-path", str(tmp_path)))
    assert "scripted" in str(excinfo.value)

    with pytest.raises(SystemExit):
        main(_argv(tmp_path / "run", tmp_path, "--base-url", "http://127.0.0.1:9/v1"))


def test_a_model_arm_without_a_model_is_refused_before_a_vm_is_booted(tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--arm",
                "offshelf_native",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
            ]
        )
    assert "--model-path" in str(excinfo.value)
    assert not juergen_fake_desktop.FakeDesktopPool.instances, "no VM may be spent"


def test_phaseb_always_fails_closed_on_a_checkpoint_that_is_not_the_one(
    tmp_path, monkeypatch
) -> None:
    """`verify_phaseb_provenance` is the strict check for the step-900 export and
    must not be satisfiable by any directory that happens to exist."""
    model, _ = _register_model(tmp_path)
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_attest_external_server",
        lambda base_url, artifact: pytest.fail(
            "Phase-B identity validation ran after server attestation"
        ),
    )
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        main(
            [
                "--arm",
                "phaseb_compact",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--model-path",
                str(model),
                "--base-url",
                "http://127.0.0.1:9/v1",
            ]
        )


def test_an_unknown_arm_is_rejected_by_argparse(tmp_path) -> None:
    """An arm is a whole `DesktopHarnessConfig` in `cells.py`; `--arm` names one
    and cannot build one, so a name that is not registered is a hard stop."""
    argv = _argv(tmp_path / "run", tmp_path)
    argv[argv.index("native_negative")] = "no_such_arm"
    with pytest.raises(SystemExit):
        main(argv)


def test_cell_and_task_index_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(_argv(tmp_path / "run", tmp_path, "--cell", CELL_IDS[0], "--task-index", "0"))


def test_a_model_arm_with_too_few_trials_warns_but_runs(tmp_path, caplog) -> None:
    """One draw cannot separate a model difference from the open_chrome
    window-mapping race — a warning, not a refusal, because a single-cell debug
    run is legitimate."""
    import logging

    with caplog.at_level(logging.WARNING, logger="signoflife"):
        code, _ = _run(
            tmp_path / "run", tmp_path, "--cell", CELL_IDS[0]
        )
    assert code == 0
    assert not [r for r in caplog.records if "trials=" in r.getMessage()], (
        "a scripted arm has no model to draw from, so the warning must not fire"
    )


def test_controls_ok_is_null_for_a_model_arm_by_construction() -> None:
    """The old runner computed `expected_passed = 0 if negative else len(rows)`
    for every mode, so a model arm's `controls_ok` restated its own pass count."""
    from evals.signoflife.__main__ import _aggregate

    rows = [
        {
            "trial": 1,
            "cell": "c",
            "validity": "valid",
            "success": True,
            "outcome": "ok",
            "control_ok": 1.0,
        }
    ]
    model = _aggregate(
        rows, cell_ids=["c"], expected_trials=1, scripted=False, negative=False
    )
    assert model["controls_ok"] is None
    assert model["expected_per_cell_pass_rate"] is None
    assert model["per_cell"]["c"]["pass_rate"] == 1.0
    assert "calibration comes from the separate scripted" in model["controls_ok_note"]


def test_a_cell_with_no_valid_draw_has_a_null_pass_rate_not_a_zero() -> None:
    """A rate over zero valid trials is not 0.0; publishing one would read as a
    measured failure."""
    from evals.signoflife.__main__ import _aggregate

    rows = [
        {
            "cell": "c",
            "validity": "infra_invalid",
            "success": None,
            "outcome": "infrastructure_error",
            "control_ok": None,
        }
    ]
    aggregate = _aggregate(
        rows, cell_ids=["c"], expected_trials=1, scripted=True, negative=True
    )
    assert aggregate["per_cell"]["c"]["pass_rate"] is None
    assert aggregate["per_cell"]["c"]["valid_trials"] == 0
    assert aggregate["controls_ok"] is False, "an invalid draw cannot certify a control"


def test_missing_valid_trial_is_a_hard_run_error(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    row = {
        "index": 0,
        "trial": 1,
        "cell": CELL_IDS[0],
        "validity": "valid",
        "success": False,
        "outcome": "model_failure",
        "control_ok": 1.0,
    }
    monkeypatch.setattr(dispatcher, "_run_attempts", lambda runtime, specs: [row])
    output = tmp_path / "incomplete"

    code, result = _run(
        output,
        tmp_path,
        "--cell",
        CELL_IDS[0],
        "--trials",
        "2",
    )

    assert code == 3
    assert result["status"] == "infrastructure_failure"
    assert result["aggregate"]["valid_trial_contract_complete"] is False
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["pass_rate"] is None
    assert any(
        error["type"] == "ValidTrialCountError"
        for error in result["infrastructure_errors"]
    )


def test_output_must_be_absent_before_any_desktop_resource(tmp_path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(RuntimeError, match="must not already exist"):
        main(_argv(output, tmp_path, "--cell", CELL_IDS[0]))

    assert not juergen_fake_desktop.FakeDesktopPool.instances


def test_attempt_exception_removes_unpublished_staging_output(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    output = tmp_path / "crashed"

    def crash(runtime, specs):
        raise RuntimeError("synthetic dispatcher crash")

    monkeypatch.setattr(dispatcher, "_run_attempts", crash)
    with pytest.raises(RuntimeError, match="synthetic dispatcher crash"):
        main(_argv(output, tmp_path, "--cell", CELL_IDS[0]))

    assert not output.exists()
    assert not list(tmp_path.glob(".crashed.staging-*"))


def test_atomic_publication_never_replaces_a_competing_run(tmp_path) -> None:
    from evals.signoflife.__main__ import _stage_output

    final = tmp_path / "one-run-id"
    first = _stage_output(final)
    second = _stage_output(final)
    try:
        (first.staging / "result.json").write_text("first\n")
        (second.staging / "result.json").write_text("second\n")
        first.publish(forbidden_values=())
        with pytest.raises(FileExistsError, match="appeared before atomic publication"):
            second.publish(forbidden_values=())
    finally:
        first.cleanup()
        second.cleanup()

    assert (final / "result.json").read_text() == "first\n"
    assert not list(tmp_path.glob(".one-run-id.staging-*"))


def test_credential_in_staging_refuses_publication_and_is_cleaned(tmp_path) -> None:
    from evals.signoflife.__main__ import _stage_output

    final = tmp_path / "redacted-run"
    publication = _stage_output(final)
    (publication.staging / "trace.log").write_text("prefix evaluation-secret suffix")
    try:
        with pytest.raises(RuntimeError, match="credential value found"):
            publication.publish(forbidden_values=("evaluation-secret",))
    finally:
        publication.cleanup()

    assert not final.exists()
    assert not list(tmp_path.glob(".redacted-run.staging-*"))


def test_an_episode_that_publishes_nothing_still_records_why(tmp_path, monkeypatch) -> None:
    """A raise before `DesktopHarness._run` — a bad pool spec, an unknown grammar,
    an unregistered kind — never reaches the code that publishes `infra_error`, so
    the row must not read `validity: null, infra_error: null`.

    Provoked at the production pool-constructor seam: launch fails before
    `DesktopHarness._run`, so the worker boundary must explain the row.
    """
    import evals.signoflife.__main__ as dispatcher

    output = tmp_path / "run"
    monkeypatch.setattr(dispatcher, "POOL_TARGET", "evals.vm:no_such_constructor")
    assert main(_argv(output, tmp_path, "--cell", CELL_IDS[0])) == 3
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "infrastructure_failure"
    error = result["episodes"][0]["infra_error"]
    assert error["stage"] == "harness"
    assert error["type"] == "HarnessError"
    assert "no_such_constructor" in error["message"]
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["valid_trials"] == 0
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["pass_rate"] is None


@pytest.mark.slow
def test_a_model_arm_records_which_bytes_answered_and_refuses_to_score_a_dead_server(
    tmp_path, monkeypatch
) -> None:
    """Two properties in one run, because they are the same run.

    Every byte is registered and verified before dispatch, and the external server
    must attest that exact artifact identity. The same identity is retained on the
    result and its attempt row.

    A model arm that cannot reach its server is an infrastructure failure: exit 3
    and `pass_rate: null`, not 0/4, which would read as a model that tried and
    failed.
    """
    model, expected = _register_model(tmp_path)
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_attest_external_server",
        lambda base_url, artifact: {
            "source": "external_endpoint",
            "url": "http://127.0.0.1:9/model-attestation",
            "artifact_sha256": artifact.artifact_sha256,
            "config_sha256": artifact.config_sha256,
            "served_model": artifact.served_model,
        },
    )

    output = tmp_path / "run"
    _fresh_process()
    code = main(
        [
            "--arm",
            "offshelf_native",
            "--output",
            str(output),
            "--qcow",
            str(tmp_path / "desktop.qcow2"),
            "--scoring-grace-s",
            "0",
            "--vm-slots",
            "4",
            "--cell",
            CELL_IDS[0],
            "--model-path",
            str(model),
            # Port 9 (discard) never answers, so no model is ever contacted.
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]
    )
    result = json.loads((output / "result.json").read_text())
    assert code == 3, result["aggregate"]
    assert result["arm_kind"] == "model"
    assert result["status"] == "infrastructure_failure"
    assert result["model"]["path"] == str(model)
    assert result["model"]["artifact_id"] == expected["id"]
    assert result["model"]["artifact_sha256"] == expected["artifact_sha256"]
    assert result["model"]["served_model"].endswith(expected["artifact_sha256"])
    assert result["episodes"][0]["model"] == result["model"]
    assert "test-only-secret" not in (output / "result.json").read_text()
    assert not any(
        "test-only-secret" in path.read_text(errors="replace")
        for path in output.rglob("*")
        if path.is_file() and path.suffix != ".png"
    )
    assert result["aggregate"]["controls_ok"] is None, "a model arm calibrates nothing"
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["pass_rate"] is None


def test_the_tier_reaches_the_taskset_and_the_record(tmp_path) -> None:
    """`--tier` is the one knob that decides which cells a number is over."""
    from evals.signoflife.__main__ import _eval_config

    config = _eval_config(
        arm="ordered_oracle",
        tier="candidate",
        task_ids=["panel_offset_button"],
        artifacts=tmp_path / "a",
        traces_dir=tmp_path / "t",
        pool={},
        base_url="http://127.0.0.1:1/v1",
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        served_model="scripted-no-model",
        seed=None,
    )
    assert config.taskset.tier == "candidate"
    assert config.taskset.task_ids == ["panel_offset_button"]


@pytest.mark.slow
def test_an_unattended_model_arm_samples_at_its_own_knobs(tmp_path, monkeypatch) -> None:
    """The arm decides, the flag overrides, and the record says which ran.

    `--temperature` used to default to 0.0, so every arm was scored greedy unless
    an operator remembered the flag — and greedy is a measurement of the decoder on
    the eov3 family, which confines 100% of its mouse deltas to
    {0, ±1, ±10, ±100} at temperature 0. The dispatcher supplying its own default
    is the defect; `sampling` in `result.json` is where a reader would have caught
    it, so it has to carry what actually ran rather than what was typed.
    """
    from evals.signoflife.cells import ARMS

    monkeypatch.setitem(
        ARMS,
        "ordered",
        ARMS["ordered"].model_copy(update={"top_p": 0.8, "max_tokens": 1024}),
    )
    model, _ = _register_model(tmp_path)
    monkeypatch.setattr(
        "evals.signoflife.__main__._attest_external_server",
        lambda base_url, artifact: {
            "source": "external_endpoint",
            "url": "http://127.0.0.1:9/model-attestation",
            "artifact_sha256": artifact.artifact_sha256,
            "config_sha256": artifact.config_sha256,
            "served_model": artifact.served_model,
        },
    )

    def _dispatch(*extra: str) -> dict:
        _fresh_process()
        output = tmp_path / f"run{len(extra)}"
        main(
            [
                "--arm",
                "ordered",
                "--output",
                str(output),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--scoring-grace-s",
                "0",
                "--cell",
                CELL_IDS[0],
                # Port 9 (discard) never answers: the sampling record is written
                # before any model is contacted.
                "--base-url",
                "http://127.0.0.1:9/v1",
                "--model-path",
                str(model),
                *extra,
            ]
        )
        return json.loads((output / "result.json").read_text())

    first = _dispatch()["sampling"]
    assert {key: first[key] for key in ("temperature", "top_p", "max_tokens")} == {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 1024,
    }
    assert first["seed_contract"]["arm_independent"] is True
    overridden = _dispatch(
        "--temperature",
        "0.0",
        "--top-p",
        "0.9",
        "--max-tokens",
        "64",
    )["sampling"]
    assert {
        key: overridden[key] for key in ("temperature", "top_p", "max_tokens")
    } == {"temperature": 0.0, "top_p": 0.9, "max_tokens": 64}


@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        ({"temperature": None, "top_p": 1.0}, "names no temperature"),
        ({"temperature": 0.7, "top_p": None}, "names no top_p"),
    ],
)
def test_a_model_arm_that_names_incomplete_sampling_is_refused(
    tmp_path, monkeypatch, sampling, message
) -> None:
    import evals.signoflife.__main__ as dispatcher
    from evals.harness import DesktopHarnessConfig
    from evals.signoflife.cells import ARMS, ORDERED_CODEC

    monkeypatch.setitem(
        ARMS,
        "incomplete",
        DesktopHarnessConfig(id="sol_incomplete", codec=ORDERED_CODEC, **sampling),
    )
    monkeypatch.setattr(
        dispatcher,
        "_eval_config",
        lambda **kwargs: pytest.fail("sampling validation ran after resource setup"),
    )
    with pytest.raises(SystemExit, match=message):
        main(
            [
                "--arm",
                "incomplete",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--base-url",
                "http://127.0.0.1:9/v1",
            ]
        )


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--temperature", "nan", "invalid temperature"),
        ("--temperature", "inf", "invalid temperature"),
        ("--temperature", "-0.1", "invalid temperature"),
        ("--top-p", "nan", "invalid top_p"),
        ("--top-p", "0", "invalid top_p"),
        ("--top-p", "1.1", "invalid top_p"),
        ("--max-tokens", "0", "invalid max_tokens"),
        ("--max-tokens", "-1", "invalid max_tokens"),
    ],
)
def test_a_model_arm_refuses_invalid_sampling_before_resource_setup(
    tmp_path, monkeypatch, flag, value, message
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_eval_config",
        lambda **kwargs: pytest.fail("sampling validation ran after resource setup"),
    )
    with pytest.raises(SystemExit, match=message):
        main(
            [
                "--arm",
                "ordered",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--base-url",
                "http://127.0.0.1:9/v1",
                flag,
                value,
            ]
        )


def test_a_scripted_arm_refuses_a_sampling_knob_it_would_never_send(tmp_path) -> None:
    """A scripted arm renders its own action, so a temperature would be published
    as the run's sampling and sent to nothing."""
    for flag, value in (("--temperature", "0.7"), ("--top-p", "0.8")):
        with pytest.raises(SystemExit, match="never calls a model"):
            main(_argv(tmp_path / "run", tmp_path, flag, value))


def test_a_cell_from_the_other_tier_is_refused_rather_than_quietly_mixed(tmp_path) -> None:
    """Averaging a calibrated cell with an unmeasured one is the uncalibrated
    number the controls exist to prevent, so the runner refuses the mix."""
    with pytest.raises(SystemExit, match="is not in the 'scored' tier"):
        main(_argv(tmp_path / "run", tmp_path, "--cell", "panel_offset_button"))
    with pytest.raises(SystemExit, match="is not in the 'candidate' tier"):
        main(
            _argv(
                tmp_path / "run",
                tmp_path,
                "--tier",
                "candidate",
                "--cell",
                "terminal_ls",
            )
        )


@pytest.mark.slow
def test_the_candidate_tier_dispatches_too_and_reads_its_negative(tmp_path) -> None:
    """The other tier's cells reach a desktop and score, with no VM.

    Worth covering because two of them are panel cells: their setup only works if
    the guest publishes a widget measurement, and the double serves the one a real
    run produced. The reading is the negative arm's calibrated 0/N -- the guest
    reports a desktop where nothing happened.
    """
    output = tmp_path / "candidate"
    _fresh_process()
    code = main(_argv(output, tmp_path, "--tier", "candidate"))
    result = json.loads((output / "result.json").read_text())
    assert code == 0, result["infrastructure_errors"]
    assert result["tier"] == "candidate"
    assert sorted(result["aggregate"]["per_cell"]) == sorted(CANDIDATE_IDS)
    assert result["aggregate"]["controls_ok"] is True
    assert {row["cell"]: row["success"] for row in result["episodes"]} == {
        cell: False for cell in CANDIDATE_IDS
    }, "a negative control must fail every cell of the tier it runs"


def test_the_attempt_deadline_is_derived_from_the_selected_arm_and_cell() -> None:
    from evals.signoflife.__main__ import _attempt_wall_bound_s, _suite_wall_bound_s
    from evals.signoflife.cells import ARMS

    task = load_suite().by_id("terminal_submit_only")
    arm = ARMS["ordered"]

    assert _attempt_wall_bound_s(task, arm) == 5498.0
    assert _suite_wall_bound_s(
        [task, task, task],
        arm=arm,
        trials=1,
        vm_slots=2,
        local_sglang=False,
        sglang_ready_timeout_s=1500.0,
    ) == 10996.0
    assert _suite_wall_bound_s(
        [task],
        arm=arm,
        trials=1,
        vm_slots=1,
        local_sglang=True,
        sglang_ready_timeout_s=1500.0,
    ) == 7898.0


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("UNLIMITED", math.inf),
        ("2-03:04:05", 183845.0),
        ("03:04:05", 11045.0),
        ("04:05", 245.0),
        ("5", 300.0),
    ],
)
def test_slurm_duration_parsing_matches_scontrol(value, seconds) -> None:
    from evals.signoflife.__main__ import _parse_slurm_duration_s

    assert _parse_slurm_duration_s(value) == seconds


def test_slurm_preflight_rejects_before_creating_the_output_or_starting_resources(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    output = tmp_path / "run"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(dispatcher, "_slurm_remaining_wall_s", lambda job_id: 300.0)
    monkeypatch.setattr(
        dispatcher,
        "_sglang",
        lambda **kwargs: pytest.fail("sglang started before the allocation preflight"),
    )
    with pytest.raises(RuntimeError, match="cannot cover declared suite bound"):
        main(_argv(output, tmp_path, "--tier", "candidate", "--cell", "terminal_submit_only"))
    assert not output.exists()
    assert not juergen_fake_desktop.FakeDesktopPool.instances


def _idle_in_own_process_group() -> None:
    os.setsid()
    time.sleep(60)


def test_attempt_process_group_termination_reaps_the_owned_worker() -> None:
    import multiprocessing

    from evals.signoflife.__main__ import _terminate_attempt_process_group

    process = multiprocessing.get_context("fork").Process(target=_idle_in_own_process_group)
    process.start()
    assert process.pid is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and os.getpgid(process.pid) != process.pid:
        time.sleep(0.01)
    try:
        assert os.getpgid(process.pid) == process.pid
        _terminate_attempt_process_group(process)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def _scheduler_runtime(tmp_path):
    from evals.signoflife.__main__ import _WorkerRuntime

    return _WorkerRuntime(
        arm="ordered",
        tier="candidate",
        output=tmp_path,
        base_url="http://127.0.0.1:9/v1",
        temperature=0.7,
        top_p=1.0,
        max_tokens=256,
        served_model="sign-of-life-sha256-test",
        model={"artifact_sha256": "test"},
        qcow=tmp_path / "desktop.qcow2",
        qemu=None,
        qemu_img=None,
        vm_smp=None,
        vm_mem=None,
        vm_slots=2,
        scoring_grace_s=0.0,
        pool_target="evals.vm:kvm_desktop_pool",
    )


def test_the_dynamic_scheduler_refills_a_free_slot_and_returns_canonical_order(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    runtime = _scheduler_runtime(tmp_path)
    task = load_suite().by_id("terminal_submit_only")
    specs = [
        dispatcher._AttemptSpec(
            index=index, cell_ordinal=index, trial=1, task=task, wall_bound_s=10.0
        )
        for index in range(3)
    ]
    events = []

    class Process:
        pid = 1000
        exitcode = 0

        def __init__(self, index):
            self.index = index
            self.polls = 0
            self.finished = False

        def is_alive(self):
            self.polls += 1
            alive = self.index == 0 and self.polls < 3
            if not alive and not self.finished:
                events.append(("finish", self.index))
                self.finished = True
            return alive

        def join(self, timeout=None):
            del timeout

    def spawn(worker_runtime, spec):
        events.append(("start", spec.index))
        row = {
            **dispatcher._attempt_row_identity(worker_runtime, spec),
            "validity": "valid",
            "success": False,
        }
        dispatcher._atomic_json(
            dispatcher._attempt_result_path(worker_runtime, spec), row
        )
        return Process(spec.index)

    monkeypatch.setattr(dispatcher, "_spawn_attempt_process", spawn)
    monkeypatch.setattr(dispatcher.time, "sleep", lambda seconds: None)

    rows = dispatcher._run_attempts(runtime, specs)
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert events.index(("start", 2)) < events.index(("finish", 0))


def test_a_spawn_worker_runs_the_owned_pool_and_publishes_its_attempt(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    output = tmp_path / "run"
    monkeypatch.setattr(dispatcher, "_WORKER_START_METHOD", "spawn")
    monkeypatch.setattr(
        dispatcher, "POOL_TARGET", "juergen_fake_desktop:kvm_desktop_pool"
    )

    assert (
        main(
            _argv(
                output,
                tmp_path,
                "--tier",
                "candidate",
                "--cell",
                "terminal_submit_only",
            )
        )
        == 0
    )
    result = json.loads((output / "result.json").read_text())
    assert result["episodes"][0]["index"] == 0
    assert result["episodes"][0]["validity"] == "valid"
    attempt = output / result["episodes"][0]["artifact_subdir"]
    assert (attempt / "attempt.json").is_file()
    assert (
        attempt / "artifacts" / "terminal_submit_only" / "steps" / "step_000.png"
    ).is_file()


def test_the_scheduler_types_a_wall_timeout_and_preserves_null_success(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    runtime = _scheduler_runtime(tmp_path)
    runtime = dispatcher._WorkerRuntime(**{**runtime.__dict__, "vm_slots": 1})
    task = load_suite().by_id("terminal_submit_only")
    spec = dispatcher._AttemptSpec(
        index=0, cell_ordinal=0, trial=1, task=task, wall_bound_s=0.0
    )

    class Process:
        pid = 1000
        exitcode = None

        def is_alive(self):
            return True

    monkeypatch.setattr(
        dispatcher, "_spawn_attempt_process", lambda runtime, spec: Process()
    )
    monkeypatch.setattr(
        dispatcher, "_terminate_attempt_process_group", lambda process: None
    )

    rows = dispatcher._run_attempts(runtime, [spec])
    assert rows[0]["index"] == 0 and rows[0]["trial"] == 1
    assert rows[0]["validity"] == "infra_invalid"
    assert rows[0]["success"] is None
    assert rows[0]["infra_error"]["type"] == "AttemptWallTimeout"
    assert rows[0]["attempt_wall_bound_s"] == 0.0
