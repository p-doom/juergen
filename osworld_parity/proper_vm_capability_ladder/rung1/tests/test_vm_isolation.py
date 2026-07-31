from __future__ import annotations

import json
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.vm import (
    KvmFixtureSession,
    VmHarnessError,
    task_unique_vm_id,
)


def test_vm_ids_are_unique_and_auditable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("LABCTL_RUN_ID", "run_abc")
    first = task_unique_vm_id()
    second = task_unique_vm_id()
    assert first != second
    assert first.startswith("4242-3-run_abc-")


def test_one_live_vm_per_task_and_scratch_under_slurm_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    first = KvmFixtureSession(vm_id="first", scratch_root=tmp_path)
    second = KvmFixtureSession(vm_id="second", scratch_root=tmp_path)
    first._prepare_isolation()
    assert first.vm_scratch_dir is not None
    assert first.vm_scratch_dir.is_relative_to(tmp_path)
    with pytest.raises(VmHarnessError, match="another VM"):
        second._prepare_isolation()
    second.close()
    first.close()
    assert not (tmp_path / "proper-vm-first").exists()


def test_job_unique_tmp_fallback_when_slurm_tmpdir_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "vm-fallback-test")
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    session = KvmFixtureSession(vm_id="fallback")
    session._prepare_isolation()
    scratch = session.vm_scratch_dir
    assert scratch is not None
    assert session.scratch_source == "job_unique_tmp_fallback"
    assert scratch.is_relative_to(Path("/tmp/proper-vm-job-0-vm-fallback-test-7")) or (
        "proper-vm-job-" in str(scratch)
    )
    session.close()
    assert not scratch.exists()


def test_vm_metadata_rejects_port_collisions_and_long_qmp(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "result" / "vm_logs"
    log_dir.mkdir(parents=True)
    log = log_dir / "qemu.log"
    log.write_text("", encoding="utf-8")
    session = KvmFixtureSession(vm_log_dir=log_dir, vm_id="metadata")
    session.vm_scratch_dir = tmp_path / "scratch"
    base = {
        "ports": {"server": 10001, "chromium": 10002, "vnc": 10003, "vlc": 10004},
        "qmp_path": "/tmp/oswqmp_short.sock",
        "log": log,
    }
    metadata = session._metadata(state=base, overlay_fds=[str(session.vm_scratch_dir / "overlay (deleted)")])
    assert len(set(metadata["ports"].values())) == 4
    collided = dict(base)
    collided["ports"] = {**base["ports"], "vlc": 10001}
    with pytest.raises(VmHarnessError, match="colliding"):
        session._metadata(state=collided, overlay_fds=[])
    long_qmp = dict(base)
    long_qmp["qmp_path"] = "/tmp/" + "x" * 90
    with pytest.raises(VmHarnessError, match="QMP"):
        session._metadata(state=long_qmp, overlay_fds=[])
