from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from evals.cua_gym import load_default_manifest
from evals.cua_gym.errors import SnapshotValidationError
from evals.cua_gym.models import (
    DatasetSnapshotConfig,
    FileFingerprint,
    TaskId,
)
from evals.cua_gym.task_health import (
    EMPTY_REWARD_SCRIPT,
    GRADER_PAYS_BEFORE_AGENT_ACTS,
    REWARD_SCRIPT_EMITS_NO_SCORE,
    UNPARSEABLE_REWARD_SCRIPT,
    ResetProbeRecord,
    SnapshotIdentity,
    analyze_reward_script,
    analyze_task,
    append_journal_line,
    build_blocklist,
    classify_setup_delivery,
    documented_state_keys,
    journal_header,
    parse_grader_score,
    read_journal,
    read_journal_identity,
    read_snapshot_identity,
    score_emitting_statements,
    seeded_state_keys,
    shard_task_ids,
    snapshot_file,
    undocumented_state_keys,
)

IDENTITY = SnapshotIdentity(
    dataset="xlangai/CUA-Gym",
    revision="0123456789abcdef0123456789abcdef01234567",
    file_digests={"artifact": "a" * 64, "metadata": "b" * 64},
)
KNOWN_ENDPOINTS = frozenset({"slack", "shopify_admin", "gitlab"})
SCHEMA_MARKDOWN = """# slack_mock Schema

## Routes

| Path | Component |
|------|-----------|
| `/` | Home |

## State Schema

| Key | Type | Description |
|-----|------|-------------|
| `channels` | array | Channels |
| `messages` | array | Messages |

### Channel Object

| Field | Type |
|-------|------|
| `id` | string |
"""


def _reward_findings(source: str) -> Any:
    return analyze_reward_script(source.encode("utf-8"))


def test_empty_reward_script_is_a_defect() -> None:
    findings = analyze_reward_script(b"")

    assert findings.defects == (EMPTY_REWARD_SCRIPT,)
    assert findings.byte_count == 0


def test_whitespace_only_reward_script_is_empty() -> None:
    assert analyze_reward_script(b"\n  \n\t\n").defects == (EMPTY_REWARD_SCRIPT,)


def test_unparseable_reward_script_is_a_defect() -> None:
    findings = _reward_findings("def broken(:\n    pass\n")

    assert findings.defects == (UNPARSEABLE_REWARD_SCRIPT,)
    assert findings.parse_error is not None
    assert findings.score_statements == ()


def test_reward_script_that_is_not_utf8_is_unparseable() -> None:
    assert analyze_reward_script(b"\xff\xfe\x00print(1)").defects == (
        UNPARSEABLE_REWARD_SCRIPT,
    )


def test_reward_script_with_print_has_no_static_defect() -> None:
    findings = _reward_findings('score = 0.5\nprint(f"REWARD: {score}")\n')

    assert findings.defects == ()
    assert findings.score_statements == ("print",)


def test_reward_script_writing_to_stdout_has_no_static_defect() -> None:
    findings = _reward_findings('import sys\nsys.stdout.write("REWARD: 1.0\\n")\n')

    assert findings.defects == ()
    assert findings.score_statements == ("stdout_write",)


def test_reward_script_whose_body_is_a_string_literal_emits_no_score() -> None:
    source = 'final_script = """\ndef verify():\n    print("REWARD: 1.0")\n"""\n'

    findings = _reward_findings(source)

    assert findings.defects == (REWARD_SCRIPT_EMITS_NO_SCORE,)
    assert findings.score_statements == ()


def test_score_statements_ignore_unrelated_write_calls() -> None:
    tree_source = 'open("/tmp/x", "w").write("REWARD: 1.0")\n'

    assert score_emitting_statements(__import__("ast").parse(tree_source)) == ()


@pytest.mark.parametrize(
    ("setup_kind", "expected"),
    [
        ("py", "script"),
        ("sh", "script"),
        ("docx", "shipped_document"),
        ("pptx", "shipped_document"),
        ("xlsx", "shipped_document"),
        ("odt", "unknown"),
    ],
)
def test_setup_delivery_classification(setup_kind: str, expected: str) -> None:
    assert classify_setup_delivery(setup_kind) == expected


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("REWARD: 0.75\n", 0.75),
        ("noise\nREWARD: 0.0\n", 0.0),
        ("Total score: 0.50\n", 0.5),
        ("Score: 3 / 5\n", 3.0),
        ("0.25\n", 0.25),
        ("REWARD: 0.1\nREWARD: 0.9\n", 0.9),
        ("checked file\nno score here\n", None),
        ("", None),
    ],
)
def test_parse_grader_score(stdout: str, expected: float | None) -> None:
    assert parse_grader_score(stdout) == expected


def test_documented_state_keys_reads_only_the_state_schema_table() -> None:
    assert documented_state_keys(SCHEMA_MARKDOWN) == frozenset({"channels", "messages"})


def test_documented_state_keys_accepts_a_qualified_heading() -> None:
    markdown = SCHEMA_MARKDOWN.replace(
        "## State Schema", "## State Schema (Top-Level Keys)"
    )

    assert documented_state_keys(markdown) == frozenset({"channels", "messages"})


def test_documented_state_keys_is_empty_without_a_state_section() -> None:
    assert documented_state_keys("# slack_mock Schema\n\n## Routes\n") == frozenset()


def test_seeded_state_keys_are_attributed_to_the_posted_endpoint() -> None:
    source = (
        "import requests\n"
        "BASE = '__CUA_GYM_SLACK_URL__'\n"
        "state = {'channels': [], 'messages_dm': []}\n"
        "requests.post(f'{BASE}/post?sid=abc', json={'action': 'set', 'state': state})\n"
    )

    assert seeded_state_keys(source, KNOWN_ENDPOINTS) == {
        "slack": frozenset({"channels", "messages_dm"})
    }


def test_seeded_state_keys_ignore_set_current_and_unknown_endpoints() -> None:
    source = (
        "import requests\n"
        "requests.post('__CUA_GYM_SLACK_URL__/post', "
        "json={'action': 'set_current', 'state': {'ui': {}}})\n"
        "requests.post('__CUA_GYM_UNLISTED_URL__/post', "
        "json={'action': 'set', 'state': {'ui': {}}})\n"
    )

    assert seeded_state_keys(source, KNOWN_ENDPOINTS) == {}


def test_seeded_state_keys_cover_several_mocks_in_one_setup() -> None:
    source = (
        "import requests\n"
        "requests.post('__CUA_GYM_SLACK_URL__/post', "
        "json={'action': 'set', 'state': {'channels': []}})\n"
        "requests.post('__CUA_GYM_GITLAB_URL__/post', "
        "json={'action': 'set', 'state': {'issues': []}})\n"
    )

    assert seeded_state_keys(source, KNOWN_ENDPOINTS) == {
        "slack": frozenset({"channels"}),
        "gitlab": frozenset({"issues"}),
    }


def test_undocumented_state_keys_reports_only_keys_outside_the_schema() -> None:
    source = (
        "import requests\n"
        "requests.post('__CUA_GYM_SLACK_URL__/post', "
        "json={'action': 'set', 'state': {'channels': [], 'messages_dm': []}})\n"
    )
    documented = {"slack": documented_state_keys(SCHEMA_MARKDOWN)}

    seeded = seeded_state_keys(source, KNOWN_ENDPOINTS)

    assert undocumented_state_keys(seeded, documented) == {"slack": ("messages_dm",)}


def test_undocumented_state_keys_is_silent_for_conforming_tasks() -> None:
    source = (
        "import requests\n"
        "requests.post('__CUA_GYM_SLACK_URL__/post', "
        "json={'action': 'set', 'state': {'channels': []}})\n"
    )
    documented = {"slack": documented_state_keys(SCHEMA_MARKDOWN)}

    seeded = seeded_state_keys(source, KNOWN_ENDPOINTS)

    assert undocumented_state_keys(seeded, documented) == {}


def test_undocumented_state_keys_skips_mocks_without_a_documented_schema() -> None:
    source = (
        "import requests\n"
        "requests.post('__CUA_GYM_SLACK_URL__/post', "
        "json={'action': 'set', 'state': {'anything': []}})\n"
    )

    seeded = seeded_state_keys(source, KNOWN_ENDPOINTS)

    assert undocumented_state_keys(seeded, {}) == {}


def test_analyze_task_combines_metadata_and_bundle_findings() -> None:
    row = {
        "id": "task-1",
        "app_type": "slack_mock",
        "platform": "web",
        "setup_kind": "py",
    }
    files = {
        "reward.py": b'print("REWARD: 0.0")\n',
        "initial_setup.py": (
            b"import requests\n"
            b"requests.post('__CUA_GYM_SLACK_URL__/post', "
            b"json={'action': 'set', 'state': {'channels': [], 'extra': 1}})\n"
        ),
    }

    findings = analyze_task(
        row, files, KNOWN_ENDPOINTS, {"slack": documented_state_keys(SCHEMA_MARKDOWN)}
    )

    assert findings.defects == ()
    assert findings.setup_delivery == "script"
    assert findings.seeded_state_endpoints == ("slack",)
    assert findings.undocumented_state_keys == {"slack": ("extra",)}


def test_analyze_task_survives_a_broken_setup_script() -> None:
    row = {
        "id": "task-2",
        "app_type": "libreoffice_writer",
        "platform": "desktop",
        "setup_kind": "docx",
    }
    files = {"reward.py": b"print(1)\n", "initial_setup.py": b"def broken(:\n"}

    findings = analyze_task(row, files, KNOWN_ENDPOINTS, {})

    assert findings.setup_delivery == "shipped_document"
    assert findings.seeded_state_endpoints == ()


def test_analyze_task_round_trips_through_json() -> None:
    row = {
        "id": "task-3",
        "app_type": "vscode",
        "platform": None,
        "setup_kind": "sh",
    }
    findings = analyze_task(row, {"reward.py": b""}, KNOWN_ENDPOINTS, {})

    restored = type(findings).from_json(json.loads(json.dumps(findings.to_json())))

    assert restored == findings


def test_shard_task_ids_partitions_deterministically() -> None:
    task_ids = [f"task-{index:02d}" for index in range(10)]

    shards = [shard_task_ids(task_ids, index, 3) for index in range(3)]

    assert sorted(sum(shards, ())) == sorted(task_ids)
    assert all(len(set(shard)) == len(shard) for shard in shards)
    assert shards[0] == shard_task_ids(list(reversed(task_ids)), 0, 3)


def test_shard_task_ids_rejects_an_out_of_range_shard() -> None:
    with pytest.raises(ValueError):
        shard_task_ids(["a"], 3, 3)


def test_journal_round_trip_preserves_records(tmp_path: Path) -> None:
    path = tmp_path / "reset_probe.0000-of-0001.jsonl"
    append_journal_line(path, journal_header(IDENTITY, "reset_probe"))
    append_journal_line(path, {"task_id": "task-1"})

    assert read_journal(path, IDENTITY) == [{"task_id": "task-1"}]
    assert read_journal_identity(path) == IDENTITY


def test_journal_from_another_snapshot_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "static.jsonl"
    append_journal_line(path, journal_header(IDENTITY, "static"))
    other = replace(IDENTITY, revision="f" * 40)

    with pytest.raises(SnapshotValidationError):
        read_journal(path, other)


def test_journal_without_a_header_has_no_identity(tmp_path: Path) -> None:
    path = tmp_path / "static.jsonl"
    append_journal_line(path, {"task_id": "task-1"})

    with pytest.raises(SnapshotValidationError):
        read_journal_identity(path)


def _findings(task_id: str, source: bytes) -> Any:
    row = {
        "id": task_id,
        "app_type": "libreoffice_writer",
        "platform": "desktop",
        "setup_kind": "docx",
    }
    return analyze_task(row, {"reward.py": source}, KNOWN_ENDPOINTS, {})


def _probe(task_id: str, score: float | None) -> ResetProbeRecord:
    return ResetProbeRecord(
        task_id=TaskId(task_id),
        setup_completed=True,
        setup_failure=None,
        pre_grade_completed=True,
        pre_grade_failure=None,
        grader_exit_code=0,
        grader_emitted_score=score is not None,
        grader_score_at_reset=score,
        grader_stdout_tail="" if score is None else f"REWARD: {score}",
        seconds=21.5,
    )


def test_blocklist_blocks_static_defects_and_free_reward() -> None:
    static = {
        TaskId("dead"): _findings("dead", b""),
        TaskId("free"): _findings("free", b'print("REWARD: 1.0")'),
        TaskId("zero"): _findings("zero", b'print("REWARD: 0.0")'),
    }
    probes = {
        TaskId("free"): _probe("free", 1.0),
        TaskId("zero"): _probe("zero", 0.0),
    }

    document = build_blocklist(IDENTITY, static, probes)

    assert document["blocked_tasks"] == {
        "dead": {"reasons": [EMPTY_REWARD_SCRIPT]},
        "free": {
            "reasons": [GRADER_PAYS_BEFORE_AGENT_ACTS],
            "grader_score_at_reset": 1.0,
        },
    }
    assert document["revision"] == IDENTITY.revision
    assert document["measured"]["reset_probe_task_ids"] == ["free", "zero"]


def test_blocklist_never_calls_an_unblocked_task_solvable() -> None:
    static = {TaskId("zero"): _findings("zero", b'print("REWARD: 0.0")')}

    document = build_blocklist(IDENTITY, static, {TaskId("zero"): _probe("zero", 0.0)})

    rendered = json.dumps(document).lower()
    assert "zero" not in document["blocked_tasks"]
    assert document["measured"]["reset_probe_task_ids"] == ["zero"]
    assert "solvable" in document["meaning"]["not_blocked"].lower()
    assert "usable" not in rendered
    assert "healthy" not in rendered


def test_blocklist_records_graders_that_emitted_no_score_as_an_advisory() -> None:
    static = {TaskId("mute"): _findings("mute", b'print("checked")')}
    probes = {TaskId("mute"): _probe("mute", None)}

    document = build_blocklist(IDENTITY, static, probes)

    assert document["blocked_tasks"] == {}
    assert document["advisories"]["reset_probe_emitted_no_score"] == ["mute"]
    assert document["advisories"]["reset_preparation_did_not_complete"] == []


def test_blocklist_merges_static_and_probe_reasons_for_one_task() -> None:
    static = {TaskId("both"): _findings("both", b"def broken(:\n")}
    probes = {TaskId("both"): _probe("both", 0.4)}

    document = build_blocklist(IDENTITY, static, probes)

    assert document["blocked_tasks"]["both"]["reasons"] == [
        GRADER_PAYS_BEFORE_AGENT_ACTS,
        UNPARSEABLE_REWARD_SCRIPT,
    ]


def test_snapshot_identity_rejects_a_revision_mismatch(tmp_path: Path) -> None:
    manifest = load_default_manifest()
    config = DatasetSnapshotConfig(dataset_root=tmp_path, revision="deadbeef")

    with pytest.raises(SnapshotValidationError):
        read_snapshot_identity(config, manifest)


def test_snapshot_identity_rejects_content_drift(tmp_path: Path) -> None:
    manifest = load_default_manifest()
    config = DatasetSnapshotConfig(dataset_root=tmp_path, revision=manifest.revision)
    for fingerprint in manifest.files.values():
        path = tmp_path / fingerprint.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the pinned bytes")

    with pytest.raises(SnapshotValidationError, match="SHA-256 mismatch"):
        read_snapshot_identity(config, manifest)


def test_snapshot_file_rejects_paths_outside_the_snapshot(tmp_path: Path) -> None:
    manifest = load_default_manifest()
    escaping = replace(
        manifest,
        files=MappingProxyType(
            {"metadata": FileFingerprint(path="../escape.json", sha256="0" * 64)}
        ),
    )
    config = DatasetSnapshotConfig(dataset_root=tmp_path, revision=manifest.revision)

    with pytest.raises(SnapshotValidationError):
        snapshot_file(config, escaping, "metadata")
