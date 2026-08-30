from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evals.cua_gym import (
    PINNED_REVISION,
    BundleFile,
    CompatibilityManifest,
    CuaGymDatasetSnapshot,
    DatasetSnapshotConfig,
    EndpointName,
    FrozenJsonObject,
    MaterializationError,
    RewardOutputFormat,
    RewardParseError,
    SnapshotValidationError,
    TaskBundle,
    TaskId,
    TaskPlatform,
    load_default_manifest,
    materialize_task_bundle,
    parse_reward_stdout,
)

ELIGIBLE_ID = "00000000-0000-0000-0000-000000000001"
EXCLUDED_ID = "00000000-0000-0000-0000-000000000002"
DESKTOP_ID = "00000000-0000-0000-0000-000000000003"
DESKTOP_EXCLUDED_ID = "00000000-0000-0000-0000-000000000004"
DESKTOP_SETUP_FILE = "initial_setup.docx"
DESKTOP_TARGET_PATH = "/home/user/report.docx"
POSTCONFIG = [
    {
        "type": "execute",
        "parameters": {
            "command": [
                "python",
                "-c",
                'import pyautogui; pyautogui.hotkey("ctrl", "s");',
            ]
        },
    },
    {"type": "sleep", "parameters": {"seconds": 0.5}},
]
POSTCONFIG_SHA256 = hashlib.sha256(
    json.dumps(POSTCONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
PLACEHOLDER = "__CUA_GYM_GMAIL_URL__"
HOST_PLACEHOLDER = "__CUA_GYM_GMAIL_HOST__"
ZERO_REWARD_DIAGNOSTIC_IDS = {
    "25c5f25c-185c-5142-95cd-5a2e3cdfffc1",
    "9601da01-11fa-5959-9971-a9a836cc1777",
}


def test_snapshot_config_resolves_hf_cache_repo_root(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots" / PINNED_REVISION
    snapshot_root.mkdir(parents=True)

    assert DatasetSnapshotConfig(tmp_path).snapshot_root == snapshot_root


def test_snapshot_config_accepts_hash_named_exact_root(tmp_path: Path) -> None:
    snapshot_root = tmp_path / PINNED_REVISION
    snapshot_root.mkdir()

    assert DatasetSnapshotConfig(snapshot_root).snapshot_root == snapshot_root


def test_snapshot_loads_arbitrary_local_dir(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "cua-gym-local-download"
    snapshot, root = _synthetic_snapshot(tmp_path, snapshot_root=snapshot_root)

    assert root == snapshot_root
    assert snapshot.root == snapshot_root
    assert len(snapshot.load_catalog()) == 1


@pytest.mark.parametrize("create_root", [False, True])
def test_snapshot_rejects_missing_or_invalid_exact_root(
    tmp_path: Path, create_root: bool
) -> None:
    snapshot_root = tmp_path / "invalid-local-download"
    if create_root:
        snapshot_root.mkdir()
    snapshot = CuaGymDatasetSnapshot(DatasetSnapshotConfig(snapshot_root))

    message = "Missing CUA-Gym snapshot file" if create_root else "does not exist"
    with pytest.raises(SnapshotValidationError, match=message):
        snapshot.load_catalog()


def test_small_snapshot_catalog_bundle_and_in_memory_materialization(
    tmp_path: Path,
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path)

    catalog = snapshot.load_catalog()
    assert len(catalog) == 1
    assert catalog.tasks[0].id == ELIGIBLE_ID
    assert catalog.tasks[0].compatibility.required_endpoints == ("gmail",)

    bundle = snapshot.load_task_bundle(ELIGIBLE_ID)
    materialized = snapshot.materialize_task_bundle(
        bundle, {"gmail": "https://gateway.example.test/gmail/"}
    )
    ip_materialized = snapshot.materialize_task_bundle(
        bundle, {"gmail": "http://127.0.0.1:8123/api/v1~x%20y"}
    )

    assert "https://gateway.example.test/gmail" in materialized.setup_files[0].text()
    assert "gateway.example.test" in materialized.reward_source
    assert PLACEHOLDER not in materialized.setup_files[0].text()
    assert HOST_PLACEHOLDER not in materialized.reward_source
    assert "http://172.17.46.46:8000" not in materialized.setup_files[0].text()
    assert materialized.gateway_urls == {"gmail": "https://gateway.example.test/gmail"}
    assert "http://127.0.0.1:8123/api/v1~x%20y" in ip_materialized.setup_files[0].text()
    assert PLACEHOLDER in bundle.setup_files[0].text()
    assert PLACEHOLDER in _archive_member_text(
        root / "artifacts/cua_gym_tasks_v1.tar.zst",
        f"{ELIGIBLE_ID}/initial_setup.py",
    )

    assert isinstance(bundle.task_config, FrozenJsonObject)
    assert materialized.task_config is not bundle.task_config
    assert materialized.task_config["evaluator"] is not bundle.task_config["evaluator"]
    assert isinstance(bundle.task_config["config"], tuple)
    with pytest.raises(TypeError):
        bundle.task_config["id"] = "mutated"  # type: ignore[index]

    mutable_config = bundle.mutable_task_config()
    materialized_mutable_config = materialized.mutable_task_config()
    assert json.loads(json.dumps(mutable_config)) == mutable_config
    mutable_config["id"] = "changed"
    config_steps = mutable_config["config"]
    assert isinstance(config_steps, list)
    first_step = config_steps[0]
    assert isinstance(first_step, dict)
    first_step["type"] = "changed"
    assert bundle.task_config["id"] == ELIGIBLE_ID
    frozen_steps = bundle.task_config["config"]
    assert isinstance(frozen_steps, tuple)
    frozen_first_step = frozen_steps[0]
    assert isinstance(frozen_first_step, FrozenJsonObject)
    assert frozen_first_step["type"] == "download"
    assert materialized_mutable_config["id"] == ELIGIBLE_ID


def test_bundle_catalog_is_memoized_indexed_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    original_load = snapshot.load_task_bundles
    archive_scans = 0

    def counted_load(
        task_ids: Iterable[str | TaskId],
    ) -> tuple[TaskBundle, ...]:
        nonlocal archive_scans
        archive_scans += 1
        return original_load(task_ids)

    monkeypatch.setattr(snapshot, "load_task_bundles", counted_load)

    first = snapshot.load_bundle_catalog()
    second = snapshot.load_bundle_catalog()

    assert first is second
    assert archive_scans == 1
    assert tuple(first) == (ELIGIBLE_ID,)
    assert first[TaskId(ELIGIBLE_ID)].task_id == ELIGIBLE_ID
    with pytest.raises(TypeError):
        first[TaskId(ELIGIBLE_ID)] = first[TaskId(ELIGIBLE_ID)]  # type: ignore[index]


def test_snapshot_rejects_revision_and_fingerprint_mismatches(tmp_path: Path) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path)
    wrong_revision = CuaGymDatasetSnapshot(
        DatasetSnapshotConfig(root, revision="not-the-pinned-revision"),
        snapshot.manifest,
    )
    with pytest.raises(SnapshotValidationError, match="does not match"):
        wrong_revision.load_catalog()

    (root / "stats.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(SnapshotValidationError, match="SHA-256 mismatch"):
        snapshot.load_catalog()


def test_snapshot_rejects_schema_manifest_mismatch(tmp_path: Path) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    payload = _manifest_payload(snapshot.manifest)
    payload["metadata_schema"][0]["arrow_type"] = "int64"
    incompatible = CompatibilityManifest.from_dict(payload)

    with pytest.raises(SnapshotValidationError, match="schema"):
        CuaGymDatasetSnapshot(snapshot.config, incompatible).load_catalog()


@pytest.mark.parametrize("reward_format", ["bare_number", "total_score"])
def test_manifest_rejects_diagnostics_for_non_prefixed_reward_tasks(
    tmp_path: Path, reward_format: str
) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    payload = _manifest_payload(snapshot.manifest)
    payload["zero_reward_diagnostic_task_ids"] = [ELIGIBLE_ID]
    if reward_format == "bare_number":
        payload["bare_reward_task_ids"] = [ELIGIBLE_ID]
    else:
        payload["reward_output_overrides"] = {ELIGIBLE_ID: reward_format}

    with pytest.raises(SnapshotValidationError, match="only contain reward_prefix"):
        CompatibilityManifest.from_dict(payload)


def test_bundle_read_revalidates_fingerprint_after_catalog_cache(
    tmp_path: Path,
) -> None:
    snapshot, root = _synthetic_snapshot(tmp_path)
    snapshot.load_catalog()
    artifact_path = root / "artifacts/cua_gym_tasks_v1.tar.zst"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(SnapshotValidationError, match="SHA-256 mismatch"):
        snapshot.load_task_bundle(ELIGIBLE_ID)


def test_materialization_requires_every_gateway_and_rejects_unknown_placeholder(
    tmp_path: Path,
) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    bundle = snapshot.load_task_bundle(ELIGIBLE_ID)

    with pytest.raises(MaterializationError, match="Missing gateway"):
        snapshot.materialize_task_bundle(bundle, {})

    modified = type(bundle)(
        metadata=bundle.metadata,
        task_config=bundle.task_config,
        reward_source=bundle.reward_source + "\nURL = '__CUA_GYM_UNKNOWN_URL__'\n",
        setup_files=bundle.setup_files,
    )
    with pytest.raises(MaterializationError, match="placeholders remain"):
        snapshot.materialize_task_bundle(modified, {"gmail": "http://127.0.0.1:8000"})


@pytest.mark.parametrize(
    "gateway_url",
    [
        "https://gateway.test/a' + attack + '",
        'https://gateway.test/a" + attack + "',
        "https://gateway.test/a\\escape",
        "https://gateway.test/a\nattack",
        "https://gateway.test/a\tattack",
        "https://gateway.test/a\x00attack",
        "https://gateway.test/{attack}",
        "https://gateway.test/`attack`",
    ],
)
def test_materialization_rejects_python_literal_breaking_gateway_urls(
    tmp_path: Path, gateway_url: str
) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    bundle = snapshot.load_task_bundle(ELIGIBLE_ID)

    with pytest.raises(MaterializationError, match="unsafe for Python source"):
        snapshot.materialize_task_bundle(bundle, {"gmail": gateway_url})


def test_materialization_rejects_malformed_ipv6_gateway_url(tmp_path: Path) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    bundle = snapshot.load_task_bundle(ELIGIBLE_ID)

    with pytest.raises(MaterializationError, match="invalid host or port"):
        snapshot.materialize_task_bundle(bundle, {"gmail": "https://["})


def test_materialization_does_not_rescan_gateway_replacement_text(
    tmp_path: Path,
) -> None:
    snapshot, _ = _synthetic_snapshot(tmp_path)
    base_bundle = snapshot.load_task_bundle(ELIGIBLE_ID)
    manifest = load_default_manifest()
    task_id = TaskId("b627c1f8-b914-5340-88a9-3da938e17d49")
    compatibility = manifest.task(task_id)
    gateway_urls: dict[str | EndpointName, str] = {
        endpoint: f"https://{endpoint}.gateway.invalid"
        for endpoint in compatibility.required_endpoints
    }
    salesforce_gateway = "https://salesforce.gateway.invalid/__CUA_GYM_SLACK_URL__"
    gateway_urls[EndpointName("salesforce")] = salesforce_gateway
    setup_source = "\n".join(
        f"{str(endpoint).upper()} = {manifest.endpoint_specs[endpoint].url_tokens[0]!r}"
        for endpoint in compatibility.required_endpoints
    )
    bundle = TaskBundle(
        metadata=replace(
            base_bundle.metadata,
            id=task_id,
            compatibility=compatibility,
        ),
        task_config=base_bundle.task_config,
        reward_source="print('REWARD: 0.5')\n",
        setup_files=(BundleFile("initial_setup.py", setup_source.encode("utf-8")),),
    )

    materialized = materialize_task_bundle(bundle, gateway_urls, manifest)
    materialized_source = materialized.setup_files[0].text()

    assert f"SALESFORCE = {salesforce_gateway!r}" in materialized_source
    assert "SLACK = 'https://slack.gateway.invalid'" in materialized_source
    assert materialized.gateway_urls[EndpointName("salesforce")] == salesforce_gateway


def test_checked_in_manifest_covers_pinned_compatibility_edges() -> None:
    manifest = load_default_manifest()

    assert manifest.revision == PINNED_REVISION
    assert manifest.metadata_row_count == 10_910
    assert manifest.eligible_task_count == len(manifest.tasks) == 1_074
    assert set(manifest.excluded_tasks) == {"9bbdfe1c-8098-5771-9cf0-a526a705c266"}
    assert "LibreOffice" in next(iter(manifest.excluded_tasks.values()))
    assert sum(task.allow_bare_reward for task in manifest.tasks.values()) == 14
    assert {
        str(task.task_id)
        for task in manifest.tasks.values()
        if task.allow_zero_reward_diagnostic
    } == ZERO_REWARD_DIAGNOSTIC_IDS
    total_score_tasks = {
        str(task.task_id)
        for task in manifest.tasks.values()
        if task.reward_output_format is RewardOutputFormat.TOTAL_SCORE
    }
    assert total_score_tasks == {"d9a20d0e-ccb7-5aea-bfa1-fbf83eaf68c6"}
    assert (
        sum(
            task.setup_target_path == "/root/initial_setup.py"
            for task in manifest.tasks.values()
        )
        == 2
    )
    assert sum(task.hard_coded_sid is not None for task in manifest.tasks.values()) == 4

    endpoint_outliers = {
        str(task.task_id): dict(task.hard_coded_endpoint_urls)
        for task in manifest.tasks.values()
        if task.hard_coded_endpoint_urls
    }
    assert endpoint_outliers == {
        "3abc6b98-3958-5fb7-8f77-bf5fd82c4459": {"wechat": "http://172.17.46.46:8057"},
        "d29a004a-9adb-5979-9f80-83154bc3e10e": {
            "google_calendar": "http://172.17.46.46:8017"
        },
        "d9a01f5e-3d4d-51b1-8a61-0421f6e92853": {
            "microsoft_teams": "http://172.17.46.46:8028"
        },
        "ec624fc3-a97c-539c-99c5-a05521e40844": {"postman": "http://172.17.46.46:8037"},
        "549ded14-dbef-550d-8a0f-d6ee11c808ba": {
            "microsoft_teams": "http://172.17.46.46:8028"
        },
        "e6db80f1-d6a4-55bd-8016-c9a7c1beeb5a": {
            "outlook_web": "http://172.17.46.46:8033"
        },
    }
    assert manifest.task("b627c1f8-b914-5340-88a9-3da938e17d49").required_endpoints == (
        "github",
        "gmail",
        "jira",
        "notion",
        "salesforce",
        "slack",
    )


def test_strict_reward_parser_preserves_fractional_rewards() -> None:
    manifest = load_default_manifest()
    regular_id = next(
        task_id
        for task_id, compatibility in manifest.tasks.items()
        if not compatibility.allow_bare_reward
    )
    bare_id = next(
        task_id
        for task_id, compatibility in manifest.tasks.items()
        if compatibility.allow_bare_reward
    )

    assert (
        parse_reward_stdout(
            regular_id,
            "component 1: 0.25 points\nREWARD: 0.375\n",
            manifest,
        )
        == 0.375
    )
    assert parse_reward_stdout(bare_id, "0.625\ndiagnostic 2\n", manifest) == 0.625
    with pytest.raises(RewardParseError, match="not approved"):
        parse_reward_stdout(regular_id, "0.5\n", manifest)


def test_total_score_reward_policy_is_task_scoped() -> None:
    manifest = load_default_manifest()
    total_score_id = "d9a20d0e-ccb7-5aea-bfa1-fbf83eaf68c6"
    regular_id = next(
        task_id
        for task_id, compatibility in manifest.tasks.items()
        if compatibility.reward_output_format is RewardOutputFormat.REWARD_PREFIX
    )

    assert (
        parse_reward_stdout(
            total_score_id,
            "[PASS] Check 1 (0.15)\nTotal score: 0.35\n",
            manifest,
        )
        == 0.35
    )
    with pytest.raises(RewardParseError, match="no numeric"):
        parse_reward_stdout(regular_id, "Total score: 0.35\n", manifest)
    with pytest.raises(RewardParseError, match="expects total_score"):
        parse_reward_stdout(total_score_id, "REWARD: 0.35\n", manifest)
    with pytest.raises(RewardParseError, match="Malformed Total score"):
        parse_reward_stdout(total_score_id, "Total score: nope\n", manifest)


@pytest.mark.parametrize("approved_id", sorted(ZERO_REWARD_DIAGNOSTIC_IDS))
def test_zero_reward_diagnostics_are_task_scoped(approved_id: str) -> None:
    manifest = load_default_manifest()
    regular_id = next(
        task_id
        for task_id, compatibility in manifest.tasks.items()
        if compatibility.reward_output_format is RewardOutputFormat.REWARD_PREFIX
        and not compatibility.allow_zero_reward_diagnostic
    )

    assert (
        parse_reward_stdout(
            approved_id,
            "REWARD: 0.0 — initial_state or current_state is missing\n",
            manifest,
        )
        == 0.0
    )
    with pytest.raises(RewardParseError, match="not approved"):
        parse_reward_stdout(regular_id, "REWARD: 0.0 — state is missing\n", manifest)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("REWARD: 0.5 — state is missing\n", "must report zero"),
        ("REWARD: 0.0 —\n", "Malformed"),
        (
            "REWARD: 0.0 — state is missing\nREWARD: 0.0\n",
            "multiple candidates",
        ),
    ],
)
@pytest.mark.parametrize("approved_id", sorted(ZERO_REWARD_DIAGNOSTIC_IDS))
def test_zero_reward_diagnostics_reject_invalid_stdout(
    approved_id: str, stdout: str, message: str
) -> None:
    manifest = load_default_manifest()

    with pytest.raises(RewardParseError, match=message):
        parse_reward_stdout(approved_id, stdout, manifest)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("", "no numeric"),
        ("REWARD: nan", "finite"),
        ("REWARD: inf", "finite"),
        ("REWARD: -0.1", "range"),
        ("REWARD: 1.1", "range"),
        ("REWARD: 0.2\nREWARD: 0.3", "multiple candidates"),
        ("REWARD: 0.2\n0.3", "multiple candidates"),
        ("0.3\nREWARD: 0.2", "multiple candidates"),
        ("0.2\ndiagnostic\n0.3", "multiple candidates"),
        ("component 1: 0.2 points", "no numeric"),
        ("debug REWARD: 0.2", "Malformed"),
        ("REWARD: nope", "Malformed"),
    ],
)
def test_strict_reward_parser_rejects_invalid_stdout(stdout: str, message: str) -> None:
    manifest = load_default_manifest()
    regular_id = next(
        task_id
        for task_id, compatibility in manifest.tasks.items()
        if not compatibility.allow_bare_reward
    )
    with pytest.raises(RewardParseError, match=message):
        parse_reward_stdout(regular_id, stdout, manifest)


def _synthetic_snapshot(
    tmp_path: Path,
    *,
    snapshot_root: Path | None = None,
    platform: TaskPlatform = TaskPlatform.WEB,
) -> tuple[CuaGymDatasetSnapshot, Path]:
    root = snapshot_root or tmp_path / "snapshots" / PINNED_REVISION
    (root / "data").mkdir(parents=True)
    (root / "artifacts").mkdir()
    rows = [
        _metadata_row(ELIGIBLE_ID, "web"),
        _metadata_row(EXCLUDED_ID, "web"),
        _desktop_metadata_row(DESKTOP_ID, DESKTOP_SETUP_FILE, 3),
        _desktop_metadata_row(DESKTOP_EXCLUDED_ID, "initial_setup.py", 2),
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "data/tasks.parquet")
    stored_schema = pq.read_schema(root / "data/tasks.parquet")
    (root / "stats.json").write_text('{"num_tasks": 4}\n', encoding="utf-8")
    (root / "url_variables.json").write_text('{"variables": {}}\n', encoding="utf-8")
    task_config = {
        "evaluator": {"type": "python", "url": "./reward.py"},
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": "./initial_setup.py",
                            "path": "/home/user/initial_setup.py",
                        }
                    ]
                },
            },
            {
                "type": "execute",
                "parameters": {"command": "python3 /home/user/initial_setup.py"},
            },
        ],
        "id": ELIGIBLE_ID,
        "instruction": "Send a test email.",
        "app_type": "gmail_mock",
    }
    archive_path = root / "artifacts/cua_gym_tasks_v1.tar.zst"
    _write_zstd_tar(
        archive_path,
        {
            f"{ELIGIBLE_ID}/task.json": json.dumps(task_config).encode(),
            f"{ELIGIBLE_ID}/initial_setup.py": (
                f"BASE_URL = '{PLACEHOLDER}'\nLEGACY_URL = 'http://172.17.46.46:8000'\n"
            ).encode(),
            f"{ELIGIBLE_ID}/reward.py": (
                f"HOST = '{HOST_PLACEHOLDER}'\nprint('REWARD: 0.5')\n"
            ).encode(),
            f"{DESKTOP_ID}/task.json": json.dumps(_desktop_task_config()).encode(),
            f"{DESKTOP_ID}/{DESKTOP_SETUP_FILE}": b"PK\x03\x04 binary docx",
            f"{DESKTOP_ID}/reward.py": b"print('REWARD: 0.25')\n",
            f"{DESKTOP_EXCLUDED_ID}/task.json": json.dumps(
                _desktop_task_config(task_id=DESKTOP_EXCLUDED_ID)
            ).encode(),
            f"{DESKTOP_EXCLUDED_ID}/initial_setup.py": b"pass\n",
            f"{DESKTOP_EXCLUDED_ID}/reward.py": b"",
        },
    )
    manifest_payload = {
        "manifest_version": 4,
        "dataset": "synthetic/CUA-Gym",
        "revision": PINNED_REVISION,
        "files": {
            name: {"path": relative, "sha256": _sha256(root / relative)}
            for name, relative in {
                "artifact": "artifacts/cua_gym_tasks_v1.tar.zst",
                "metadata": "data/tasks.parquet",
                "stats": "stats.json",
                "url_variables": "url_variables.json",
            }.items()
        },
        "metadata_schema": [
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "nullable": field.nullable,
            }
            for field in stored_schema
        ],
        "metadata_row_count": 4,
        "eligible_task_count": 1,
        "excluded_tasks": {EXCLUDED_ID: "synthetic exclusion"},
        "endpoint_specs": {
            "gmail": {
                "url_tokens": [
                    PLACEHOLDER,
                    "https://cua-gym-gmail.xlang.ai",
                ],
                "host_tokens": [
                    HOST_PLACEHOLDER,
                    "cua-gym-gmail.xlang.ai",
                ],
            }
        },
        "eligible_task_ids": [ELIGIBLE_ID],
        "endpoint_task_ids": {"gmail": [ELIGIBLE_ID]},
        "bare_reward_task_ids": [],
        "zero_reward_diagnostic_task_ids": [],
        "reward_output_overrides": {},
        "setup_target_overrides": {},
        "hard_coded_endpoint_overrides": {
            ELIGIBLE_ID: {"gmail": "http://172.17.46.46:8000"}
        },
        "hard_coded_sid_overrides": {},
        "desktop": synthetic_desktop_manifest(),
    }
    manifest = CompatibilityManifest.from_dict(manifest_payload)
    return (
        CuaGymDatasetSnapshot(
            DatasetSnapshotConfig(root, revision=PINNED_REVISION),
            manifest,
            platform=platform,
        ),
        root,
    )


def synthetic_desktop_manifest() -> dict[str, object]:
    return {
        "task_count": 2,
        "eligible_task_count": 1,
        "excluded_tasks": {DESKTOP_EXCLUDED_ID: "reward.py is empty."},
        "setup_step_types": ["download", "execute", "launch", "open", "sleep"],
        "reward_output_overrides": {},
        "evaluator_postconfig_sha256": POSTCONFIG_SHA256,
        "evaluator_postconfig_task_count": 1,
    }


def _desktop_task_config(task_id: str = DESKTOP_ID) -> dict[str, object]:
    return {
        "evaluator": {
            "type": "python",
            "url": "./reward.py",
            "postconfig": POSTCONFIG,
        },
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": f"./{DESKTOP_SETUP_FILE}",
                            "path": DESKTOP_TARGET_PATH,
                        }
                    ]
                },
            },
            {"type": "open", "parameters": {"path": DESKTOP_TARGET_PATH}},
            {"type": "sleep", "parameters": {"seconds": 2}},
        ],
        "id": task_id,
        "instruction": "Add a two-column table.",
        "app_type": "libreoffice_writer",
    }


def _desktop_metadata_row(
    task_id: str, setup_file: str, num_setup_steps: int
) -> dict[str, object]:
    return {
        "id": task_id,
        "instruction": "Add a two-column table.",
        "app_type": "libreoffice_writer",
        "app_family": "desktop_office",
        "platform": "desktop",
        "difficulty": "medium",
        "setup_kind": "docx",
        "num_setup_steps": num_setup_steps,
        "num_setup_files": 1,
        "has_ground_truth": False,
        "setup_files": [setup_file],
        "archive_path": "artifacts/cua_gym_tasks_v1.tar.zst",
        "archive_member": task_id,
        "task_json_member": f"{task_id}/task.json",
        "reward_member": f"{task_id}/reward.py",
        "setup_file_members": [f"{task_id}/{setup_file}"],
    }


def _metadata_row(task_id: str, platform: str) -> dict[str, object]:
    return {
        "id": task_id,
        "instruction": "Send a test email.",
        "app_type": "gmail_mock",
        "app_family": "mock_web",
        "platform": platform,
        "difficulty": "easy",
        "setup_kind": "py",
        "num_setup_steps": 2,
        "num_setup_files": 1,
        "has_ground_truth": False,
        "setup_files": ["initial_setup.py"],
        "archive_path": "artifacts/cua_gym_tasks_v1.tar.zst",
        "archive_member": task_id,
        "task_json_member": f"{task_id}/task.json",
        "reward_member": f"{task_id}/reward.py",
        "setup_file_members": [f"{task_id}/initial_setup.py"],
    }


def _write_zstd_tar(path: Path, members: dict[str, bytes]) -> None:
    with pa.output_stream(str(path), compression="zstd") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, content in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def _archive_member_text(path: Path, expected_name: str) -> str:
    with pa.input_stream(str(path), compression="zstd") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                if member.name == expected_name:
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    return extracted.read().decode()
    raise AssertionError(f"Archive member not found: {expected_name}")


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _manifest_payload(manifest: CompatibilityManifest) -> dict[str, object]:
    endpoint_task_ids = {
        str(endpoint): [
            str(task_id)
            for task_id, task in manifest.tasks.items()
            if endpoint in task.required_endpoints
        ]
        for endpoint in manifest.endpoint_specs
    }
    return {
        "manifest_version": 4,
        "dataset": manifest.dataset,
        "revision": manifest.revision,
        "files": {
            name: {"path": value.path, "sha256": value.sha256}
            for name, value in manifest.files.items()
        },
        "metadata_schema": [
            {
                "name": field.name,
                "arrow_type": field.arrow_type,
                "nullable": field.nullable,
            }
            for field in manifest.metadata_schema
        ],
        "metadata_row_count": manifest.metadata_row_count,
        "eligible_task_count": manifest.eligible_task_count,
        "excluded_tasks": {
            str(task_id): reason for task_id, reason in manifest.excluded_tasks.items()
        },
        "endpoint_specs": {
            str(name): {
                "url_tokens": list(spec.url_tokens),
                "host_tokens": list(spec.host_tokens),
            }
            for name, spec in manifest.endpoint_specs.items()
        },
        "eligible_task_ids": [str(task_id) for task_id in manifest.tasks],
        "endpoint_task_ids": endpoint_task_ids,
        "bare_reward_task_ids": [
            str(task_id)
            for task_id, task in manifest.tasks.items()
            if task.allow_bare_reward
        ],
        "zero_reward_diagnostic_task_ids": [
            str(task_id)
            for task_id, task in manifest.tasks.items()
            if task.allow_zero_reward_diagnostic
        ],
        "reward_output_overrides": {
            str(task_id): task.reward_output_format.value
            for task_id, task in manifest.tasks.items()
            if task.reward_output_format
            not in {
                RewardOutputFormat.REWARD_PREFIX,
                RewardOutputFormat.BARE_NUMBER,
            }
        },
        "setup_target_overrides": {
            str(task_id): task.setup_target_path
            for task_id, task in manifest.tasks.items()
            if task.setup_target_path != "/home/user/initial_setup.py"
        },
        "hard_coded_endpoint_overrides": {
            str(task_id): dict(task.hard_coded_endpoint_urls)
            for task_id, task in manifest.tasks.items()
            if task.hard_coded_endpoint_urls
        },
        "hard_coded_sid_overrides": {
            str(task_id): task.hard_coded_sid
            for task_id, task in manifest.tasks.items()
            if task.hard_coded_sid is not None
        },
        "desktop": {
            "task_count": manifest.desktop.task_count,
            "eligible_task_count": manifest.desktop.eligible_task_count,
            "excluded_tasks": {
                str(task_id): reason
                for task_id, reason in manifest.desktop.excluded_tasks.items()
            },
            "setup_step_types": [
                str(step_type) for step_type in manifest.desktop.setup_step_types
            ],
            "reward_output_overrides": {
                str(task_id): output_format.value
                for task_id, output_format in (
                    manifest.desktop.reward_output_overrides.items()
                )
            },
            "evaluator_postconfig_sha256": manifest.desktop.evaluator_postconfig_sha256,
            "evaluator_postconfig_task_count": (
                manifest.desktop.evaluator_postconfig_task_count
            ),
        },
    }
