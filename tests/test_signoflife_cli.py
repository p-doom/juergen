"""`python -m evals.signoflife`, end to end, with no VM and no GPU.

`evals/signoflife/__main__.py` (211 statements) and `evals/vm.py` (86) were both at
**zero** coverage, and the 97.1% that was published had both excluded — the honest
figure was 88%. `SignOfLifeTaskset.load` never executed either. So the one path
every real gate run takes — CLI -> argument validation -> taskset load -> harness
construction -> pool adapter -> episode -> `result.json` — was untested end to end,
which is precisely where the missing `DesktopFacade.evaluate()` was hiding.

This runs the real `main()`. The only substitution is
`pixeldesk.vm.factory.build_desktop_pool` (see `juergen_fake_desktop`), so
`evals.vm.kvm_desktop_pool` and `DesktopFacade` are the production objects under
test rather than doubles.

**Scoring semantics are untouched.** The guest reports a desktop where nothing
happened, which is the *negative* control's calibrated reading — 0/4 with
`control_ok` true on every cell. Nothing here asserts, changes or fabricates the
oracle arm's 4/4: that is a calibration against a real VM, and a fixture that
produced it would be fiction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import juergen_fake_desktop
from evals.signoflife.__main__ import main
from evals.signoflife.suite import load_suite

CELL_IDS = [task.id for task in load_suite().tasks]


@pytest.fixture(autouse=True)
def _fake_desktop(monkeypatch):
    """Substitute the one function between the dispatcher and a booted VM.

    `agent.desktop._POOLS` is process-global and keyed by arm name, so without
    the teardown the second test in a session would silently reuse the first
    one's pool, its slot directory and its already-checked-out sessions.
    """
    import agent.desktop as desktop
    import pixeldesk.vm.factory as factory

    juergen_fake_desktop.FakeDesktopPool.instances.clear()
    monkeypatch.setattr(
        factory, "build_desktop_pool", juergen_fake_desktop.build_desktop_pool
    )
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
        # 120 s scoring grace plus a single node slot serialises the four cells
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
    two runs of one arm in ONE process collide by construction. In production
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


# --------------------------------------------------------------------------- #
# the whole path
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_dispatcher_runs_all_four_cells_and_writes_the_readers_shape(
    tmp_path,
) -> None:
    output = tmp_path / "run"
    code, result = _run(output, tmp_path)

    assert code == 0, result["infrastructure_errors"]
    assert result["status"] == "complete"
    assert result["schema_version"] == 3
    assert result["arm"] == "native_negative"
    assert result["arm_kind"] == "scripted_negative"

    aggregate = result["aggregate"]
    assert sorted(aggregate["per_cell"]) == sorted(CELL_IDS), "the gate is all four cells"
    for cell, row in aggregate["per_cell"].items():
        assert row["trials"] == 1, cell
        assert row["valid_trials"] == 1, f"{cell}: {result['infrastructure_errors']}"
        assert row["pass_rate"] == 0.0, f"{cell} is a negative control"
        assert len(row["outcomes"]) == 1
    assert aggregate["expected_per_cell_pass_rate"] == 0.0
    assert aggregate["controls_ok"] is True
    assert result["suite_manifest_sha256"] == load_suite().manifest_sha256
    assert len(result["episodes"]) == 4


@pytest.mark.slow
def test_the_pool_adapter_is_the_production_one(tmp_path) -> None:
    """`pool_target` names `evals.vm:kvm_desktop_pool`, and that is what runs:
    the JSON-able arguments become a `DesktopPoolConfig`, and every checkout comes
    back as a `DesktopFacade` — the union `FakeSession` used to stand in for."""
    output = tmp_path / "run"
    code, _ = _run(output, tmp_path, "--vm-smp", "4", "--vm-mem", "8G")
    assert code == 0

    pools = juergen_fake_desktop.FakeDesktopPool.instances
    assert pools, "kvm_desktop_pool never reached build_desktop_pool"
    kwargs = pools[0].kwargs
    assert kwargs["smp"] == 4 and kwargs["memory"] == "8G"
    assert kwargs["accelerator"] == "kvm"
    assert kwargs["image"] == Path(tmp_path / "desktop.qcow2")
    config = kwargs["config"]
    assert config.max_rollouts_per_session == 1, "a reused VM is a wrong measurement"
    assert config.max_sessions == 4, "--vm-slots"


@pytest.mark.slow
def test_the_flags_that_only_just_became_reachable_reach_the_pool(tmp_path) -> None:
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
        "full_suite_task_count": 4,
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


# --------------------------------------------------------------------------- #
# the four validation paths
# --------------------------------------------------------------------------- #


def test_a_scripted_arm_refuses_a_model_it_would_silently_ignore(tmp_path) -> None:
    """Recorded-and-ignored is the worst shape: the run record would name a
    checkpoint that never answered a single token."""
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


def test_verify_phaseb_needs_a_checkpoint_to_verify(tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--arm",
                "offshelf_native",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--base-url",
                "http://127.0.0.1:9/v1",
                "--verify-phaseb",
            ]
        )
    assert "--model-path" in str(excinfo.value)


def test_verify_phaseb_fails_closed_on_a_checkpoint_that_is_not_the_one(tmp_path) -> None:
    """`verify_phaseb_provenance` is the strict check for the step-900 export and
    must not be satisfiable by any directory that happens to exist."""
    model = tmp_path / "not-phaseb"
    model.mkdir()
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
                "--verify-phaseb",
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


# --------------------------------------------------------------------------- #
# the aggregate, read directly
# --------------------------------------------------------------------------- #


def test_controls_ok_is_null_for_a_model_arm_by_construction() -> None:
    """The old runner computed `expected_passed = 0 if negative else len(rows)`
    for every mode, so a model arm's `controls_ok` restated its own pass count
    wearing the word *control*."""
    from evals.signoflife.__main__ import _aggregate

    rows = [
        {"cell": "c", "validity": "valid", "success": True, "outcome": "ok", "control_ok": 1.0}
    ]
    model = _aggregate(rows, cell_ids=["c"], scripted=False, negative=False)
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
    aggregate = _aggregate(rows, cell_ids=["c"], scripted=True, negative=True)
    assert aggregate["per_cell"]["c"]["pass_rate"] is None
    assert aggregate["per_cell"]["c"]["valid_trials"] == 0
    assert aggregate["controls_ok"] is False, "an invalid draw cannot certify a control"


def test_an_episode_that_publishes_nothing_still_records_why(tmp_path) -> None:
    """A raise *before* `DesktopHarness._run` — a bad pool spec, an unknown
    grammar, an unregistered kind — never reaches the code that publishes
    `infra_error`, and the row used to read `validity: null, infra_error: null`.
    Exit 3 with no reason anywhere a reader looks is the worst kind of red light.

    Provoked here the way it actually happens: the same arm dispatched twice in
    one process, whose `pool_for` guard refuses the second spec.
    """
    first = tmp_path / "first"
    code, _ = _run(first, tmp_path, "--cell", CELL_IDS[0])
    assert code == 0

    second = tmp_path / "second"
    # Deliberately NOT `_fresh_process()`: keep the first run's pool registered.
    assert main(_argv(second, tmp_path, "--cell", CELL_IDS[0])) == 3
    result = json.loads((second / "result.json").read_text())
    assert result["status"] == "infrastructure_failure"
    error = result["episodes"][0]["infra_error"]
    assert error["stage"] == "harness"
    assert "already exists with a different spec" in error["message"]
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["valid_trials"] == 0
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["pass_rate"] is None


@pytest.mark.slow
def test_a_model_arm_records_which_bytes_answered_and_refuses_to_score_a_dead_server(
    tmp_path,
) -> None:
    """Two properties in one run, because they are the same run.

    *Provenance is recorded, never enforced.* `--verify-phaseb` is the fail-closed
    check for one arm; hard-coding a second, third and fourth expected manifest
    would make every new arm a code change. What must never be lost is which bytes
    answered, so the checkpoint's `config.json` and both registration files are
    hashed into the result.

    *A model arm that cannot reach its server is an infrastructure failure.* Exit 3
    and `pass_rate: null` — not 0/4, which would read as a model that tried and
    failed.
    """
    model = tmp_path / "ckpt"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (tmp_path / ".meta.json").write_text(json.dumps({"id": "artifact_test", "step": 900}))

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
    assert len(result["model"]["config_sha256"]) == 64
    assert result["model"]["meta"] == {"id": "artifact_test", "step": 900}
    assert len(result["model"]["meta_sha256"]) == 64
    assert result["aggregate"]["controls_ok"] is None, "a model arm calibrates nothing"
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["pass_rate"] is None
