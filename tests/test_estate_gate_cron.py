from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


CRON = Path(__file__).resolve().parents[1] / "tooling" / "estate_gate_cron.sh"
REMOTE_REF = "refs/remotes/origin/release"
_COMMITTER = (
    "-c",
    "user.email=gate@estate.test",
    "-c",
    "user.name=estate gate",
    "-c",
    "commit.gpgsign=false",
)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *_COMMITTER, *args),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _checkout(tmp_path: Path) -> tuple[Path, str, Path]:
    root = (tmp_path / "checkout").resolve()
    tooling = root / "tooling"
    tooling.mkdir(parents=True)
    shutil.copy(CRON, tooling / CRON.name)
    (tooling / "estate_gate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$PWD\" >> \"$FAKE_GATE_RECORD\"\n"
        "case \"${FAKE_GATE_MODE:-green}\" in\n"
        "  green) printf 'ESTATE GATE: GREEN\\n'; exit 0 ;;\n"
        "  red) printf 'ESTATE GATE: RED\\n'; exit 1 ;;\n"
        "  broken) printf 'gate crashed before its verdict\\n'; exit 1 ;;\n"
        "  moved) git commit --allow-empty -qm moved; printf 'ESTATE GATE: GREEN\\n'; exit 0 ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n"
    )
    (root / "README").write_text("scheduled estate gate fixture\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "fixture")
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", REMOTE_REF, head)

    spool = (tmp_path / "spool" / "slurm_script").resolve()
    spool.parent.mkdir()
    shutil.copy(tooling / CRON.name, spool)
    return root, head, spool


def _scheduler(tmp_path: Path) -> dict[str, Path]:
    fake_bin = (tmp_path / "bin").resolve()
    fake_bin.mkdir()
    state = (tmp_path / "pending").resolve()
    calls = (tmp_path / "sbatch.calls").resolve()

    squeue = fake_bin / "squeue"
    squeue.write_text(
        "#!/usr/bin/env bash\n"
        "if [ -f \"$FAKE_SCHEDULER_STATE\" ]; then cat \"$FAKE_SCHEDULER_STATE\"; fi\n"
    )
    squeue.chmod(0o770)

    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CALL\\n' >> \"$FAKE_SBATCH_CALLS\"\n"
        "printf '<%s>\\n' \"$@\" >> \"$FAKE_SBATCH_CALLS\"\n"
        "printf '12345\\n' > \"$FAKE_SCHEDULER_STATE\"\n"
        "if [ \"${FAKE_SBATCH_OUTPUT:-numeric}\" = malformed ]; then\n"
        "  printf 'Submitted batch job 12345\\n'\n"
        "else\n"
        "  printf '12345\\n'\n"
        "fi\n"
    )
    sbatch.chmod(0o770)
    return {"bin": fake_bin, "state": state, "calls": calls}


def _env(tmp_path: Path, scheduler: dict[str, Path]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        ESTATE_GATE_LOG_DIR=str((tmp_path / "logs").resolve()),
        FAKE_GATE_RECORD=str((tmp_path / "gate.calls").resolve()),
        FAKE_SCHEDULER_STATE=str(scheduler["state"]),
        FAKE_SBATCH_CALLS=str(scheduler["calls"]),
        PATH=f"{scheduler['bin']}:{env['PATH']}",
    )
    return env


def _run(
    spool: Path,
    root: Path,
    head: str,
    env: dict[str, str],
    remote_ref: str = REMOTE_REF,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", str(spool), str(root), head, remote_ref),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_a_spool_copy_runs_the_tracked_gate_from_the_bound_root_and_rearms_the_tracked_script(
    tmp_path: Path,
) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    done = _run(spool, root, head, _env(tmp_path, scheduler))

    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "gate.calls").read_text().splitlines() == [str(root)]
    call = scheduler["calls"].read_text()
    assert call.count("CALL\n") == 1
    assert "<--parsable>" in call
    assert f"<{root / 'tooling' / CRON.name}>" in call
    assert f"<{root}>" in call
    assert f"<{head}>" in call
    assert f"<{REMOTE_REF}>" in call
    assert f"<{spool}>" not in call
    assert scheduler["state"].read_text() == "12345\n"


def test_an_authoritative_red_gate_alerts_and_rearms(tmp_path: Path) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    env = _env(tmp_path, scheduler)
    env["FAKE_GATE_MODE"] = "red"

    done = _run(spool, root, head, env)

    assert done.returncode == 1, done.stdout + done.stderr
    assert scheduler["calls"].read_text().count("CALL\n") == 1
    assert scheduler["state"].read_text() == "12345\n"
    verdicts = (tmp_path / "logs" / "verdicts.log").read_text()
    assert "ESTATE GATE: RED" in verdicts
    assert "ALERT" in verdicts


def test_a_failed_wrapper_without_an_authoritative_verdict_does_not_rearm(
    tmp_path: Path,
) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    env = _env(tmp_path, scheduler)
    env["FAKE_GATE_MODE"] = "broken"

    done = _run(spool, root, head, env)

    assert done.returncode == 2, done.stdout + done.stderr
    assert "authoritative verdict" in done.stderr
    assert not scheduler["calls"].exists()
    assert not scheduler["state"].exists()


def test_two_successful_runs_leave_exactly_one_pending_successor(tmp_path: Path) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    env = _env(tmp_path, scheduler)
    argv = ("bash", str(spool), str(root), head, REMOTE_REF)

    first = subprocess.Popen(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    second = subprocess.Popen(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    first_out, first_err = first.communicate(timeout=30)
    second_out, second_err = second.communicate(timeout=30)

    assert first.returncode == 0, first_out + first_err
    assert second.returncode == 0, second_out + second_err
    assert scheduler["calls"].read_text().count("CALL\n") == 1
    assert scheduler["state"].read_text().splitlines() == ["12345"]


def test_malformed_sbatch_output_is_not_accepted_as_a_job_id(tmp_path: Path) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    env = _env(tmp_path, scheduler)
    env["FAKE_SBATCH_OUTPUT"] = "malformed"

    done = _run(spool, root, head, env)

    assert done.returncode == 2, done.stdout + done.stderr
    assert "sbatch returned a non-numeric job ID" in done.stderr
    assert scheduler["calls"].read_text().count("CALL\n") == 1


@pytest.mark.parametrize(
    ("pending", "message"),
    (
        ("12345\n67890\n", "multiple pending successors"),
        ("Submitted batch job 12345\n", "squeue returned a non-numeric job ID"),
    ),
)
def test_ambiguous_pending_state_is_refused(
    tmp_path: Path,
    pending: str,
    message: str,
) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    scheduler["state"].write_text(pending)

    done = _run(spool, root, head, _env(tmp_path, scheduler))

    assert done.returncode == 2, done.stdout + done.stderr
    assert message in done.stderr
    assert not scheduler["calls"].exists()


def test_a_dirty_or_moved_checkout_does_not_rearm(tmp_path: Path) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    env = _env(tmp_path, scheduler)
    (root / "README").write_text("dirty\n")

    dirty = _run(spool, root, head, env)

    assert dirty.returncode == 2, dirty.stdout + dirty.stderr
    assert "checkout is dirty" in dirty.stderr
    assert not (tmp_path / "gate.calls").exists()
    assert not scheduler["calls"].exists()

    _git(root, "restore", "README")
    env["FAKE_GATE_MODE"] = "moved"
    moved = _run(spool, root, head, env)

    assert moved.returncode == 2, moved.stdout + moved.stderr
    assert "checkout HEAD moved" in moved.stderr
    assert not scheduler["calls"].exists()


def test_a_remote_ref_that_does_not_publish_expected_head_is_refused(tmp_path: Path) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    _git(root, "commit", "--allow-empty", "-qm", "unpublished-local-head")
    other = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", REMOTE_REF, other)
    _git(root, "reset", "--hard", head)

    done = _run(spool, root, head, _env(tmp_path, scheduler))

    assert done.returncode == 2, done.stdout + done.stderr
    assert "remote ref does not publish expected HEAD" in done.stderr
    assert not (tmp_path / "gate.calls").exists()
    assert not scheduler["calls"].exists()


@pytest.mark.parametrize(
    ("mutate_spool", "remote_ref", "message"),
    (
        (True, REMOTE_REF, "invoked cron script does not match"),
        (False, "refs/heads/main", "remote ref must be under refs/remotes"),
    ),
)
def test_a_nonremote_cron_script_is_refused(
    tmp_path: Path,
    mutate_spool: bool,
    remote_ref: str,
    message: str,
) -> None:
    root, head, spool = _checkout(tmp_path)
    scheduler = _scheduler(tmp_path)
    if mutate_spool:
        spool.write_text(spool.read_text() + "\n")

    done = _run(spool, root, head, _env(tmp_path, scheduler), remote_ref)

    assert done.returncode == 2, done.stdout + done.stderr
    assert message in done.stderr
    assert not (tmp_path / "gate.calls").exists()
    assert not scheduler["calls"].exists()
