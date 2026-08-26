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

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import juergen_fake_desktop
import pytest
from test_model_attestation import _register_model

import evals.signoflife.__main__ as dispatcher
from evals.signoflife.__main__ import main, read_committed_result
from evals.signoflife.suite import load_suite

# The scored tier: what an unqualified run of this dispatcher is. `CANDIDATE_IDS`
# is the other tier, and it is covered too -- `juergen_fake_desktop` serves the Tk
# fixture's published measurement, so the panel cells' dispatch path is exercised
# here without a VM.
CELL_IDS = [task.id for task in load_suite().for_tier("scored")]
CANDIDATE_IDS = [task.id for task in load_suite().for_tier("candidate")]


@pytest.mark.parametrize("value", ["", "1"])
def test_main_refuses_disabled_cudnn_validation_before_argument_parsing(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("SGLANG_DISABLE_CUDNN_CHECK", value)
    monkeypatch.setattr(
        dispatcher,
        "_parse_args",
        lambda *_args, **_kwargs: pytest.fail("arguments parsed before cuDNN rejection"),
    )

    with pytest.raises(RuntimeError, match="cuDNN compatibility validation"):
        dispatcher.main([])


@pytest.mark.parametrize("value", ["", "1"])
def test_sglang_refuses_disabled_cudnn_validation_before_acquisition(
    monkeypatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv("SGLANG_DISABLE_CUDNN_CHECK", value)
    monkeypatch.setattr(
        dispatcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("SGLang process was acquired"),
    )
    log_path = tmp_path / "output" / "sglang.log"

    with pytest.raises(RuntimeError, match="cuDNN compatibility validation"):
        with dispatcher._sglang(
            python="/runtime/bin/python",
            model_path=tmp_path / "model",
            log_path=log_path,
            mem_fraction_static=0.65,
            ready_timeout_s=1.0,
        ):
            pytest.fail("SGLang became ready")

    assert not log_path.parent.exists()


def test_sglang_child_does_not_receive_cudnn_check_bypass(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SGLANG_DISABLE_CUDNN_CHECK", raising=False)
    child_environment = None

    class ExpectedPopen(Exception):
        pass

    def capture_environment(*_args, **kwargs):
        nonlocal child_environment
        child_environment = kwargs["env"]
        raise ExpectedPopen

    monkeypatch.setattr(dispatcher.subprocess, "Popen", capture_environment)
    with pytest.raises(ExpectedPopen):
        with dispatcher._sglang(
            python="/runtime/bin/python",
            model_path=tmp_path / "model",
            log_path=tmp_path / "output" / "sglang.log",
            mem_fraction_static=0.65,
            ready_timeout_s=1.0,
        ):
            pytest.fail("SGLang became ready")

    assert child_environment is not None
    assert "SGLANG_DISABLE_CUDNN_CHECK" not in child_environment


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

    monkeypatch.delenv(dispatcher.API_KEY_VAR, raising=False)
    monkeypatch.delenv(dispatcher._LOCAL_NO_AUTH_API_KEY_VAR, raising=False)
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


def _fake_owned_sglang(monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    @contextlib.contextmanager
    def launch(**kwargs):
        yield dispatcher._LocalServer(
            base_url="http://127.0.0.1:9/v1",
            launch={"test_only_owned_process": True},
        )

    monkeypatch.setattr(dispatcher, "_sglang", launch)
    monkeypatch.setattr(
        dispatcher,
        "_attest_local_server",
        lambda base_url, artifact: {
            "source": "local_verified_launch",
            "served_model": artifact.served_model,
            "server": {"test_only": True},
        },
    )
    monkeypatch.setattr(
        dispatcher,
        "_probe_seeded_sampling",
        lambda *args, **kwargs: {"test_only": True},
    )


def _run(output: Path, tmp_path: Path, *extra: str) -> tuple[int, dict]:
    _fresh_process()
    code = main(_argv(output, tmp_path, *extra))
    return code, read_committed_result(output)


@pytest.mark.slow
def test_the_dispatcher_runs_the_whole_scored_tier_and_writes_the_readers_shape(
    tmp_path, capsys,
) -> None:
    output = tmp_path / "run"
    code, result = _run(output, tmp_path)

    assert code == 0, result["infrastructure_errors"]
    assert (output / "RESULT_COMMITTED.json").is_file()
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


def test_attempt_exception_leaves_an_uncommitted_nonreusable_run_id(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    output = tmp_path / "crashed"

    def crash(runtime, specs):
        raise RuntimeError("synthetic dispatcher crash")

    monkeypatch.setattr(dispatcher, "_run_attempts", crash)
    with pytest.raises(RuntimeError, match="synthetic dispatcher crash"):
        main(_argv(output, tmp_path, "--cell", CELL_IDS[0]))

    assert output.is_dir()
    assert not (output / "RESULT_COMMITTED.json").exists()
    with pytest.raises(RuntimeError, match="must not already exist"):
        main(_argv(output, tmp_path, "--cell", CELL_IDS[0]))


def test_atomic_run_id_creation_never_reuses_a_competing_run(tmp_path) -> None:
    from evals.signoflife.__main__ import _create_uncommitted_output

    final = tmp_path / "one-run-id"
    first = _create_uncommitted_output(final)
    try:
        with pytest.raises(RuntimeError, match="must not already exist"):
            _create_uncommitted_output(final)
    finally:
        first.cleanup()

    assert final.is_dir()
    assert not (final / "RESULT_COMMITTED.json").exists()


def test_commit_marker_is_linked_last_and_the_strict_reader_accepts_it(tmp_path) -> None:
    from evals.signoflife.__main__ import _create_uncommitted_output, read_committed_result

    final = tmp_path / "complete"
    publication = _create_uncommitted_output(final)
    (publication.path / "result.json").write_text('{"status":"complete"}\n')
    publication.publish(forbidden_values=())

    assert publication.published is True
    assert publication.durable is True
    assert oct((final / "RESULT_COMMITTED.json").stat().st_mode)[-3:] == "400"
    assert (final / ".RESULT_COMMITTED.source").stat().st_ino == (
        final / "RESULT_COMMITTED.json"
    ).stat().st_ino
    assert read_committed_result(final) == {"status": "complete"}


def test_strict_reader_rejects_missing_marker_mutation_and_extra_files(tmp_path) -> None:
    from evals.signoflife.__main__ import _create_uncommitted_output, read_committed_result

    orphan = _create_uncommitted_output(tmp_path / "orphan")
    (orphan.path / "result.json").write_text("{}\n")
    orphan.cleanup()
    with pytest.raises(FileNotFoundError):
        read_committed_result(orphan.final)

    mutated = _create_uncommitted_output(tmp_path / "mutated-after-commit")
    (mutated.path / "result.json").write_text("{}\n")
    mutated.publish(forbidden_values=())
    (mutated.final / "result.json").write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="does not match generation bytes"):
        read_committed_result(mutated.final)

    extra = _create_uncommitted_output(tmp_path / "extra-after-commit")
    (extra.path / "result.json").write_text("{}\n")
    extra.publish(forbidden_values=())
    (extra.final / "unexpected").write_text("late")
    with pytest.raises(RuntimeError, match="does not match generation bytes"):
        read_committed_result(extra.final)


def test_output_verifier_bounds_marker_result_count_depth_names_paths_and_bytes(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    cases = [
        ("result", "_OUTPUT_MAX_RESULT_BYTES", 1, "result.json exceeds"),
        ("files", "_OUTPUT_MAX_FILES", 1, "file count exceeds"),
        ("depth", "_OUTPUT_MAX_DEPTH", 0, "directory depth exceeds"),
        ("path", "_OUTPUT_MAX_PATH_BYTES", 8, "path exceeds"),
        ("bytes", "_OUTPUT_MAX_TOTAL_BYTES", 1, "output bytes exceed"),
    ]
    for name, constant, limit, message in cases:
        publication = dispatcher._create_uncommitted_output(tmp_path / f"bounded-{name}")
        (publication.path / "result.json").write_text("{}\n")
        if name == "files":
            (publication.path / "payload").write_text("x")
        if name == "depth":
            (publication.path / "child").mkdir()
        if name == "path":
            (publication.path / "longname").write_text("x")
        with monkeypatch.context() as patch:
            patch.setattr(dispatcher, constant, limit)
            try:
                with pytest.raises(RuntimeError, match=message):
                    publication.publish(forbidden_values=())
            finally:
                publication.cleanup()

    noncanonical = dispatcher._create_uncommitted_output(tmp_path / "noncanonical")
    (noncanonical.path / "result.json").write_text("{}\n")
    (noncanonical.path / "space in name").write_text("x")
    try:
        with pytest.raises(RuntimeError, match="noncanonical component"):
            noncanonical.publish(forbidden_values=())
    finally:
        noncanonical.cleanup()

    committed = dispatcher._create_uncommitted_output(tmp_path / "bounded-marker")
    (committed.path / "result.json").write_text("{}\n")
    committed.publish(forbidden_values=())
    monkeypatch.setattr(dispatcher, "_OUTPUT_MAX_MARKER_BYTES", 1)
    with pytest.raises(RuntimeError, match="commit marker exceeds"):
        dispatcher.read_committed_result(committed.final)


def test_output_verifier_rejects_a_short_fd_read(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    publication = dispatcher._create_uncommitted_output(tmp_path / "short-read")
    (publication.path / "result.json").write_text("{}\n")
    original_read = dispatcher.os.read

    def short_read(descriptor, size):
        if os.readlink(f"/proc/self/fd/{descriptor}").endswith("/result.json"):
            return b""
        return original_read(descriptor, size)

    monkeypatch.setattr(dispatcher.os, "read", short_read)
    try:
        with pytest.raises(RuntimeError, match="byte count changed"):
            publication.publish(forbidden_values=())
    finally:
        publication.cleanup()


def test_group_write_on_generation_refuses_commit(tmp_path) -> None:
    from evals.signoflife.__main__ import _create_uncommitted_output

    publication = _create_uncommitted_output(tmp_path / "group-writable")
    (publication.path / "result.json").write_text("{}\n")
    publication.path.chmod(0o770)
    try:
        with pytest.raises(RuntimeError, match="private 0700"):
            publication.publish(forbidden_values=())
    finally:
        publication.cleanup()

    assert not (publication.final / "RESULT_COMMITTED.json").exists()


def test_concurrent_content_mutation_refuses_commit(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    publication = dispatcher._create_uncommitted_output(tmp_path / "mutated")
    (publication.path / "payload.bin").write_bytes(b"a" * (1024 * 1024 + 1))
    (publication.path / "result.json").write_text("{}\n")
    mutate = threading.Event()
    mutated = threading.Event()

    def writer() -> None:
        assert mutate.wait(timeout=10)
        with (publication.path / "payload.bin").open("r+b") as handle:
            handle.write(b"b")
            handle.flush()
            os.fsync(handle.fileno())
        mutated.set()

    thread = threading.Thread(target=writer)
    thread.start()
    original_read = dispatcher.os.read

    def racing_read(descriptor, size):
        chunk = original_read(descriptor, size)
        if (
            not chunk
            and os.readlink(f"/proc/self/fd/{descriptor}").endswith("/payload.bin")
        ):
            mutate.set()
            assert mutated.wait(timeout=10)
        return chunk

    monkeypatch.setattr(dispatcher.os, "read", racing_read)
    try:
        with pytest.raises(RuntimeError, match="changed while sealing"):
            publication.publish(forbidden_values=())
    finally:
        thread.join(timeout=10)
        publication.cleanup()

    assert not (publication.final / "RESULT_COMMITTED.json").exists()


def test_concurrent_symlink_swap_refuses_commit(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    publication = dispatcher._create_uncommitted_output(tmp_path / "symlink-swap")
    payload = publication.path / "payload.bin"
    payload.write_text("original")
    (publication.path / "result.json").write_text("{}\n")
    replacement = tmp_path / "replacement"
    replacement.write_text("replacement")
    swap = threading.Event()
    swapped = threading.Event()

    def attacker() -> None:
        assert swap.wait(timeout=10)
        payload.unlink()
        payload.symlink_to(replacement)
        swapped.set()

    thread = threading.Thread(target=attacker)
    thread.start()
    original_stat = dispatcher.os.stat
    triggered = False

    def racing_stat(path, *args, **kwargs):
        nonlocal triggered
        observed = original_stat(path, *args, **kwargs)
        if path == "payload.bin" and kwargs.get("follow_symlinks") is False and not triggered:
            triggered = True
            swap.set()
            assert swapped.wait(timeout=10)
        return observed

    monkeypatch.setattr(dispatcher.os, "stat", racing_stat)
    try:
        with pytest.raises(OSError):
            publication.publish(forbidden_values=())
    finally:
        thread.join(timeout=10)
        publication.cleanup()

    assert not (publication.final / "RESULT_COMMITTED.json").exists()


def test_concurrent_directory_swap_refuses_commit(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    publication = dispatcher._create_uncommitted_output(tmp_path / "directory-swap")
    child = publication.path / "child"
    child.mkdir()
    (child / "payload.bin").write_text("original")
    (publication.path / "result.json").write_text("{}\n")
    swap = threading.Event()
    swapped = threading.Event()

    def attacker() -> None:
        assert swap.wait(timeout=10)
        child.rename(publication.path / "moved-child")
        child.mkdir()
        (child / "payload.bin").write_text("replacement")
        swapped.set()

    thread = threading.Thread(target=attacker)
    thread.start()
    original_open = dispatcher.os.open
    triggered = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal triggered
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "child" and flags & os.O_DIRECTORY and not triggered:
            triggered = True
            swap.set()
            assert swapped.wait(timeout=10)
        return descriptor

    monkeypatch.setattr(dispatcher.os, "open", racing_open)
    try:
        with pytest.raises(RuntimeError, match="changed while sealing"):
            publication.publish(forbidden_values=())
    finally:
        thread.join(timeout=10)
        publication.cleanup()

    assert not (publication.final / "RESULT_COMMITTED.json").exists()


def test_post_link_directory_fsync_failure_is_visible_quarantine(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    publication = dispatcher._create_uncommitted_output(tmp_path / "fsync-failed")
    (publication.path / "result.json").write_text("{}\n")
    original_fsync = dispatcher.os.fsync
    directory_fsyncs = 0

    def failing_fsync(descriptor):
        nonlocal directory_fsyncs
        if descriptor == publication.staging_fd:
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("synthetic directory fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(dispatcher.os, "fsync", failing_fsync)
    try:
        with pytest.raises(RuntimeError, match="marker is visible"):
            publication.publish(forbidden_values=())
    finally:
        publication.cleanup()

    assert publication.published is True
    assert publication.durable is False
    assert (publication.final / "RESULT_COMMITTED.json").exists()
    assert dispatcher.read_committed_result(publication.final) == {}


def test_credential_in_staging_refuses_publication_and_is_cleaned(tmp_path) -> None:
    from evals.signoflife.__main__ import _create_uncommitted_output

    final = tmp_path / "redacted-run"
    publication = _create_uncommitted_output(final)
    (publication.path / "trace.log").write_text("prefix evaluation-secret suffix")
    try:
        with pytest.raises(RuntimeError, match="credential value found"):
            publication.publish(forbidden_values=("evaluation-secret",))
    finally:
        publication.cleanup()

    assert final.is_dir()
    assert not (final / "RESULT_COMMITTED.json").exists()


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
    result = read_committed_result(output)
    assert result["status"] == "infrastructure_failure"
    error = result["episodes"][0]["infra_error"]
    assert error["stage"] == "harness"
    assert error["type"] == "HarnessError"
    assert "no_such_constructor" in error["message"]
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["valid_trials"] == 0
    assert result["aggregate"]["per_cell"][CELL_IDS[0]]["pass_rate"] is None


@pytest.mark.slow
def test_a_model_arm_records_the_local_model_and_refuses_to_score_a_dead_server(
    tmp_path, monkeypatch
) -> None:
    model, expected = _register_model(tmp_path)
    _fake_owned_sglang(monkeypatch)

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
            "--sglang-python",
            sys.executable,
        ]
    )
    result = read_committed_result(output)
    assert code == 3, result["aggregate"]
    assert result["arm_kind"] == "model"
    assert result["status"] == "infrastructure_failure"
    assert result["model"]["path"] == expected["path"]
    assert result["model"]["config_identity"] == expected["config_identity"]
    assert result["model"]["served_model"] == expected["served_model"]
    assert result["episodes"][0]["model"] == result["model"]
    assert result["claim_scope"] == "local_causal_seeded"
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
    _fake_owned_sglang(monkeypatch)

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
                "--model-path",
                str(model),
                "--sglang-python",
                sys.executable,
                *extra,
            ]
        )
        return read_committed_result(output)

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
    result = read_committed_result(output)
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

    assert _attempt_wall_bound_s(task, arm) == 5538.0
    assert _suite_wall_bound_s(
        [task, task, task],
        arm=arm,
        trials=1,
        vm_slots=2,
        local_sglang=False,
        sglang_ready_timeout_s=1500.0,
    ) == 11226.0
    assert _suite_wall_bound_s(
        [task],
        arm=arm,
        trials=1,
        vm_slots=1,
        local_sglang=True,
        sglang_ready_timeout_s=1500.0,
    ) == 8068.0


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


def _scontrol_command(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o700)
    return path


def test_scontrol_query_uses_the_absolute_owned_command(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    command = _scontrol_command(
        tmp_path / "scontrol",
        "import sys\n"
        "if sys.argv[1:] != ['show', 'job', '123', '-o']:\n"
        "    raise SystemExit(2)\n"
        "print('JobId=123 TimeLimit=02:00:00 RunTime=00:01:00')\n",
    )
    monkeypatch.setattr(dispatcher, "_SCONTROL_PATH", command)

    assert dispatcher._slurm_remaining_wall_s("123") == 7140.0
    with pytest.raises(RuntimeError, match="invalid SLURM_JOB_ID"):
        dispatcher._slurm_remaining_wall_s("--bad")


@pytest.mark.parametrize("descriptor", [1, 2])
def test_scontrol_stdout_and_stderr_are_online_bounded(
    tmp_path, monkeypatch, descriptor
) -> None:
    import evals.signoflife.__main__ as dispatcher

    command = _scontrol_command(
        tmp_path / "scontrol",
        f"import os\nos.write({descriptor}, b'x' * 70000)\n",
    )
    monkeypatch.setattr(dispatcher, "_SCONTROL_PATH", command)

    label = "stdout" if descriptor == 1 else "stderr"
    with pytest.raises(RuntimeError, match=rf"scontrol {label} exceeds"):
        dispatcher._slurm_remaining_wall_s("123")


def test_scontrol_timeout_kills_and_reaps_its_process_session(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    pid_path = tmp_path / "pid"
    command = _scontrol_command(
        tmp_path / "scontrol",
        "import os,pathlib,time\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
    )
    monkeypatch.setattr(dispatcher, "_SCONTROL_PATH", command)
    monkeypatch.setattr(dispatcher, "_SCONTROL_TIMEOUT_S", 0.5)

    with pytest.raises(RuntimeError, match="scontrol query timed out"):
        dispatcher._slurm_remaining_wall_s("123")

    pid = int(pid_path.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{pid}").exists()


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


def test_attempt_session_termination_reaps_the_owned_worker() -> None:
    import multiprocessing

    from evals.signoflife.__main__ import _terminate_attempt_process_session

    process = multiprocessing.get_context("fork").Process(target=_idle_in_own_process_group)
    process.start()
    assert process.pid is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and os.getpgid(process.pid) != process.pid:
        time.sleep(0.01)
    try:
        assert os.getpgid(process.pid) == process.pid
        _terminate_attempt_process_session(process)
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
        model={"path": "/model"},
        qcow=tmp_path / "desktop.qcow2",
        qemu=None,
        qemu_img=None,
        vm_smp=None,
        vm_mem=None,
        vm_slots=2,
        scoring_grace_s=0.0,
        pool_target="evals.vm:kvm_desktop_pool",
    )


def _crash_with_stubborn_descendant(pid_path: str) -> None:
    os.setsid()
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,pathlib,signal,sys,time; os.setpgid(0,0); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(300)",
            pid_path,
        ]
    )
    while not Path(pid_path).exists():
        time.sleep(0.01)
    os._exit(7)


def _wait_with_stubborn_descendant(pid_path: str) -> None:
    os.setsid()
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,pathlib,signal,sys,time; os.setpgid(0,0); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(300)",
            pid_path,
        ]
    )
    while not Path(pid_path).exists():
        time.sleep(0.01)
    time.sleep(300)


def test_natural_worker_crash_reaps_an_escaped_stubborn_descendant(
    tmp_path, monkeypatch
) -> None:
    import multiprocessing

    import evals.signoflife.__main__ as dispatcher

    runtime = _scheduler_runtime(tmp_path)
    spec = dispatcher._AttemptSpec(
        index=0,
        cell_ordinal=0,
        trial=1,
        task=load_suite().by_id("terminal_submit_only"),
        wall_bound_s=10.0,
    )
    descendant_path = tmp_path / "descendant.pid"
    launched = []

    def spawn(_runtime, _spec):
        process = multiprocessing.get_context("fork").Process(
            target=_crash_with_stubborn_descendant,
            args=(str(descendant_path),),
        )
        process.start()
        launched.append(process)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if os.getpgid(process.pid) == process.pid and descendant_path.exists():
                    break
            except ProcessLookupError:
                pass
            time.sleep(0.01)
        assert descendant_path.exists()
        assert os.getpgid(process.pid) == process.pid
        descendant = int(descendant_path.read_text(encoding="utf-8"))
        assert os.getpgid(descendant) == descendant
        assert os.getsid(descendant) == process.pid
        return process

    monkeypatch.setattr(dispatcher, "_spawn_attempt_process", spawn)
    monkeypatch.setattr(dispatcher, "_SCHEDULER_POLL_S", 0.01)
    monkeypatch.setattr(dispatcher, "_SUPERVISOR_REAP_TIMEOUT_S", 0.1)
    monkeypatch.setattr(dispatcher, "_SUPERVISOR_KILL_TIMEOUT_S", 3.0)
    monkeypatch.setattr(dispatcher, "_SGLANG_GROUP_POLL_S", 0.01)
    try:
        rows = dispatcher._run_attempts(runtime, [spec])
        descendant = int(descendant_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.killpg(launched[0].pid, 0)
        assert not Path(f"/proc/{descendant}").exists()
        assert rows[0]["infra_error"]["type"] == "AttemptDescendantLeak"
    finally:
        if descendant_path.exists() and launched:
            descendant = int(descendant_path.read_text(encoding="utf-8"))
            try:
                if os.getsid(descendant) == launched[0].pid:
                    os.killpg(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if launched and launched[0].pid is not None:
            try:
                os.killpg(launched[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            launched[0].join(timeout=5)


def test_supervisor_baseexception_reaps_an_escaped_stubborn_descendant(
    tmp_path, monkeypatch
) -> None:
    import multiprocessing

    import evals.signoflife.__main__ as dispatcher

    runtime = _scheduler_runtime(tmp_path)
    spec = dispatcher._AttemptSpec(
        index=0,
        cell_ordinal=0,
        trial=1,
        task=load_suite().by_id("terminal_submit_only"),
        wall_bound_s=10.0,
    )
    descendant_path = tmp_path / "descendant.pid"
    launched = []

    def spawn(_runtime, _spec):
        process = multiprocessing.get_context("fork").Process(
            target=_wait_with_stubborn_descendant,
            args=(str(descendant_path),),
        )
        process.start()
        launched.append(process)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if os.getpgid(process.pid) == process.pid and descendant_path.exists():
                break
            time.sleep(0.01)
        assert descendant_path.exists()
        assert os.getpgid(process.pid) == process.pid
        descendant = int(descendant_path.read_text(encoding="utf-8"))
        assert os.getpgid(descendant) == descendant
        assert os.getsid(descendant) == process.pid
        return process

    original_exited = dispatcher._attempt_process_exited
    interrupted = False

    def interrupt(process):
        nonlocal interrupted
        if interrupted:
            return original_exited(process)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not descendant_path.exists():
            time.sleep(0.01)
        assert descendant_path.exists()
        interrupted = True
        raise KeyboardInterrupt

    monkeypatch.setattr(dispatcher, "_spawn_attempt_process", spawn)
    monkeypatch.setattr(dispatcher, "_attempt_process_exited", interrupt)
    monkeypatch.setattr(dispatcher, "_SUPERVISOR_REAP_TIMEOUT_S", 0.1)
    monkeypatch.setattr(dispatcher, "_SUPERVISOR_KILL_TIMEOUT_S", 3.0)
    monkeypatch.setattr(dispatcher, "_SGLANG_GROUP_POLL_S", 0.01)
    try:
        with pytest.raises(KeyboardInterrupt):
            dispatcher._run_attempts(runtime, [spec])
        descendant = int(descendant_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.killpg(launched[0].pid, 0)
        assert not Path(f"/proc/{descendant}").exists()
    finally:
        if descendant_path.exists() and launched:
            descendant = int(descendant_path.read_text(encoding="utf-8"))
            try:
                if os.getsid(descendant) == launched[0].pid:
                    os.killpg(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if launched and launched[0].pid is not None:
            try:
                os.killpg(launched[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            launched[0].join(timeout=5)


def test_supervisor_cleanup_visits_every_active_session_after_failures(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    runtime = _scheduler_runtime(tmp_path)
    task = load_suite().by_id("terminal_submit_only")
    specs = [
        dispatcher._AttemptSpec(
            index=index,
            cell_ordinal=index,
            trial=1,
            task=task,
            wall_bound_s=10.0,
        )
        for index in range(2)
    ]

    class Process:
        exitcode = None

        def __init__(self, pid):
            self.pid = pid

    cleaned = []
    processes = iter((Process(1000), Process(1001)))
    monkeypatch.setattr(
        dispatcher, "_spawn_attempt_process", lambda _runtime, _spec: next(processes)
    )
    monkeypatch.setattr(
        dispatcher,
        "_attempt_process_exited",
        lambda _process: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    def fail_cleanup(process, *, terminate=True):
        del terminate
        cleaned.append(process.pid)
        raise RuntimeError(f"cleanup failed for {process.pid}")

    monkeypatch.setattr(dispatcher, "_terminate_attempt_process_session", fail_cleanup)

    with pytest.raises(BaseExceptionGroup) as raised:
        dispatcher._run_attempts(runtime, specs)

    assert cleaned == [1000, 1001]
    assert [type(error) for error in raised.value.exceptions] == [
        KeyboardInterrupt,
        RuntimeError,
        RuntimeError,
    ]


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
    monkeypatch.setattr(
        dispatcher, "_attempt_process_exited", lambda process: not process.is_alive()
    )
    monkeypatch.setattr(
        dispatcher,
        "_terminate_attempt_process_session",
        lambda process, *, terminate=False: False,
    )
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
    result = read_committed_result(output)
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
        dispatcher, "_attempt_process_exited", lambda process: False
    )
    monkeypatch.setattr(
        dispatcher,
        "_terminate_attempt_process_session",
        lambda process, *, terminate=True: False,
    )

    rows = dispatcher._run_attempts(runtime, [spec])
    assert rows[0]["index"] == 0 and rows[0]["trial"] == 1
    assert rows[0]["validity"] == "infra_invalid"
    assert rows[0]["success"] is None
    assert rows[0]["infra_error"]["type"] == "AttemptWallTimeout"
    assert rows[0]["attempt_wall_bound_s"] == 0.0
