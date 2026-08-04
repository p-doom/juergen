"""Full-source rebuild of the sealed Phase-B dataset.

This is the heavyweight reproduction test: it re-runs ``build.py`` over the
complete teacher corpus and asserts every count, order and byte invariant.

It needs cluster-local inputs that are far too large to commit (the normalized
teacher source split and the OSWorld rollout trees, plus the out-of-repo
``build_osworld_format_records.py`` contract module). Paths default to their
values on hai; override with the environment variables below, and the test skips
when they are absent so the rest of the suite still runs on a clean checkout.

  PHASEB_AUDIT_OPERAND    default <shared>/audit_operand
  PHASEB_ROLLOUTS         default <shared>/onpolicy_distill/rollouts/teacher_8b_osworld_train_v1
  PHASEB_ONPOLICY_SCRIPTS default <shared>/onpolicy_distill/scripts
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build import build, read_jsonl
from conftest import external_root, repo_relative

_SHARED = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz"

AUDIT_OPERAND = external_root("PHASEB_AUDIT_OPERAND", f"{_SHARED}/audit_operand")
ROLLOUTS = external_root(
    "PHASEB_ROLLOUTS",
    f"{_SHARED}/onpolicy_distill/rollouts/teacher_8b_osworld_train_v1",
)
ONPOLICY_SCRIPTS = external_root(
    "PHASEB_ONPOLICY_SCRIPTS", f"{_SHARED}/onpolicy_distill/scripts"
)
SOURCE = AUDIT_OPERAND / "phaseb"

_MISSING = [
    str(path)
    for path in (
        SOURCE / "prose_keep/_normalized/train/chat.jsonl",
        ROLLOUTS,
        ONPOLICY_SCRIPTS / "build_osworld_format_records.py",
    )
    if not path.exists()
]


@pytest.mark.skipif(
    bool(_MISSING), reason=f"cluster-local build inputs absent: {_MISSING}"
)
def test_full_source_exactness_and_record_order(tmp_path: Path):
    output = tmp_path / "dataset"
    manifest = build(
        source_root=SOURCE,
        collected_root=ROLLOUTS,
        # build.py loads action_span_conversion.py from --audit-dir; the vendored
        # copy is byte-identical, so build.py's own hash pin still matches.
        audit_dir=repo_relative("osworld_parity/phaseb_deltatype_raw_v2/vendor"),
        onpolicy_scripts=ONPOLICY_SCRIPTS,
        production_parser=repo_relative("eval/action_parser.py"),
        train_split=repo_relative("osworld_parity/split/osworld_train.json"),
        heldout_split=repo_relative("osworld_parity/split/osworld_eval_heldout.json"),
        output=output,
    )
    assert manifest["record_counts"] == {"train": 2383, "val": 233}
    assert manifest["assistant_spans"] == 10721
    assert manifest["tool_calls"] == 11471
    assert manifest["multi_call_spans"] == 750
    assert manifest["legacy_spans"] == 10277
    assert manifest["drag_spans"] == 444
    assert manifest["exact_command_plans"] == 10721
    assert manifest["drag_split_counts"] == {"train": 437, "val": 7}
    assert manifest["train_val_task_intersection_count"] == 0
    assert manifest["heldout_intersection_count"] == 0
    assert manifest["all_source_calls_consumed"] is True
    assert manifest["legacy_label_byte_invariance"] is True
    assert manifest["legacy_transition_invariance"] is True
    assert manifest["all_drag_command_sequences_exact"] is True
    assert manifest["production_gpu_training_authorized"] is False

    # The rebuilt splits must be byte-identical to the sealed artifact.
    assert manifest["output_file_sha256"] == {
        "train/chat.jsonl": (
            "5f449f3d57b368e55cfe2ba486bcdd9953aa6f9bad343948e0b8653b2ab4de99"
        ),
        "val/chat.jsonl": (
            "a819011d5f8524cad1980d720fcdbc98a838a37b33de499c46eb4c13c94acadd"
        ),
    }

    for split in ("train", "val"):
        source_rows = read_jsonl(
            SOURCE / "prose_keep/_normalized" / split / "chat.jsonl"
        )
        output_rows = read_jsonl(output / split / "chat.jsonl")
        identity = ("sample_id", "recording_id", "app", "task_id", "step")
        assert [tuple(row.get(key) for key in identity) for row in output_rows] == [
            tuple(row.get(key) for key in identity) for row in source_rows
        ]
        for source_row, output_row in zip(source_rows, output_rows, strict=True):
            source_users = [m for m in source_row["messages"] if m["role"] == "user"]
            output_users = [m for m in output_row["messages"] if m["role"] == "user"]
            assert output_users == source_users

    on_disk = json.loads((output / "dataset_manifest.json").read_text())
    payload = on_disk.pop("payload_sha256")
    canonical = json.dumps(on_disk, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == payload
