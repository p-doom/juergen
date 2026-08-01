from __future__ import annotations

from dataclasses import replace

import pytest

from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.qualify import (
    qualify_vm,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.schema import (
    CorpusError,
    load_corpus,
    sha256_value,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.smoke_schema import (
    load_smoke,
)


def test_shard_rejects_zero_task_limit_before_vm_start(tmp_path) -> None:
    with pytest.raises(ValueError, match="per-app"):
        qualify_vm(
            load_corpus(),
            per_app=0,
            plumbing_smoke=False,
            shard_index=0,
            task_id=None,
            qcow=tmp_path / "unused.qcow2",
            qemu=tmp_path / "unused-qemu",
            provider=tmp_path / "unused-provider.py",
            work_dir=tmp_path / "unused-work",
        )


def test_vscode_smoke_rejects_resealed_weak_verifier() -> None:
    task = next(task for task in load_smoke().tasks if task.app == "vscode")
    source = dict(task.source_task)
    source["verifier"] = {"fresh_process": False}
    unsigned_source = dict(source)
    unsigned_source.pop("task_sha256")
    source["task_sha256"] = sha256_value(unsigned_source)
    mutated = replace(
        task,
        source_task=source,
        source_task_payload_sha256=sha256_value(source),
    )
    mutated = replace(mutated, record_sha256=sha256_value(mutated.unsigned_record()))
    with pytest.raises(CorpusError, match="verifier contract"):
        mutated.verify()


def test_legacy_forty_task_corpus_is_permanently_auxiliary() -> None:
    assert load_corpus().eligibility == {
        "purpose": "auxiliary_development_only",
        "stage0": False,
        "final": False,
    }
