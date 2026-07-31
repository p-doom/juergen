"""OSWorld and CUA-Gym task indexes -> one train-only canonical manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION
from experiments.teacher_sft.contracts import (
    ContractError,
    artifact_ref,
    assert_not_heldout,
    ensure_empty_output,
    file_sha256,
    iter_jsonl,
    load_heldout_denylist,
    object_sha256,
    read_json,
    require_train_split,
    verify_declared_hash,
    write_json,
    write_jsonl,
)

TRAIN_SPLITS = ("train", "train_validation")
_FORBIDDEN_INDEX_PARTS = ("test_all", "heldout", "official_eval", "evaluation_split")


def _reject_suspicious_index(path: Path) -> None:
    lowered = str(path).lower()
    if any(part in lowered for part in _FORBIDDEN_INDEX_PARTS):
        raise ContractError(
            f"refusing an index that appears heldout/eval-scoped: {path}"
        )
    if "train" not in path.name.lower():
        raise ContractError(
            f"task index filename must explicitly contain 'train': {path}"
        )


def _instruction(config: dict[str, Any], *, path: Path) -> str:
    value = (
        config.get("instruction")
        or config.get("task_instruction")
        or config.get("task")
    )
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"task has no non-empty instruction: {path}")
    return value.strip()


def _osworld_rows(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    require_train_split(source.get("source_split"), context="OSWorld source")
    index_path = Path(str(source.get("task_index", ""))).resolve()
    task_root = Path(str(source.get("task_root", ""))).resolve()
    _reject_suspicious_index(index_path)
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ContractError("OSWorld train index must map app names to task-id arrays")
    revision = source.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ContractError("OSWorld source_revision is required")
    for app in sorted(index):
        task_ids = index[app]
        if not isinstance(app, str) or not isinstance(task_ids, list):
            raise ContractError("malformed OSWorld train index")
        for task_id in sorted(task_ids):
            if not isinstance(task_id, str):
                raise ContractError("OSWorld task ids must be strings")
            task_path = task_root / app / f"{task_id}.json"
            config = read_json(task_path)
            if not isinstance(config, dict):
                raise ContractError(
                    f"OSWorld task config is not an object: {task_path}"
                )
            instruction = _instruction(config, path=task_path)
            yield {
                "schema_version": SCHEMA_VERSION,
                "task_key": f"osworld:{task_id}",
                "source": "osworld",
                "source_task_id": task_id,
                "source_split": "train",
                "source_revision": revision,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                "app": app,
                "platform": str(config.get("platform", "linux")),
                "artifacts": [artifact_ref(task_path, "task_config")],
                "source_metadata": {
                    "task_index_path": str(index_path),
                    "task_index_sha256": file_sha256(index_path),
                    "evaluator_type": config.get("evaluator", {}).get("func")
                    if isinstance(config.get("evaluator"), dict)
                    else None,
                },
            }


def _load_cuagym_index(path: Path) -> list[dict[str, Any]]:
    _reject_suspicious_index(path)
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        payload = [
            item.get("row", item) if isinstance(item, dict) else item
            for item in payload["rows"]
        ]
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ContractError("CUA-Gym index must be JSONL or a JSON row array")
    return payload


def _cuagym_rows(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    require_train_split(source.get("source_split"), context="CUA-Gym source")
    index_path = Path(str(source.get("task_index", ""))).resolve()
    bundle_root = Path(str(source.get("bundle_root", ""))).resolve()
    revision = source.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ContractError("CUA-Gym source_revision is required")
    for index_row in sorted(
        _load_cuagym_index(index_path), key=lambda row: str(row.get("id", ""))
    ):
        task_id = index_row.get("id") or index_row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ContractError("CUA-Gym index row lacks id")
        row_split = index_row.get("split", source.get("source_split"))
        require_train_split(row_split, context=f"CUA-Gym task {task_id}")
        bundle = bundle_root / task_id
        task_path = bundle / "task.json"
        config = read_json(task_path)
        if not isinstance(config, dict):
            raise ContractError(f"CUA-Gym task config is not an object: {task_path}")
        index_instruction = index_row.get("instruction")
        instruction = _instruction(config, path=task_path)
        if index_instruction is not None and index_instruction != instruction:
            raise ContractError(
                f"CUA-Gym index/bundle instruction mismatch for {task_id}"
            )
        reward_path = bundle / "reward.py"
        setup_names = index_row.get("setup_files") or config.get("setup_files")
        if not isinstance(setup_names, list) or not setup_names:
            setup_names = sorted(path.name for path in bundle.glob("initial_setup.*"))
        artifacts = [
            artifact_ref(task_path, "task_config"),
            artifact_ref(reward_path, "reward"),
        ]
        for name in setup_names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ContractError(
                    f"unsafe CUA-Gym setup filename for {task_id}: {name!r}"
                )
            artifacts.append(artifact_ref(bundle / name, "setup"))
        yield {
            "schema_version": SCHEMA_VERSION,
            "task_key": f"cua_gym:{task_id}",
            "source": "cua_gym",
            "source_task_id": task_id,
            "source_split": "train",
            "source_revision": revision,
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            "app": str(index_row.get("app_type") or config.get("domain") or "unknown"),
            "platform": str(
                index_row.get("platform") or config.get("platform") or "unknown"
            ),
            "artifacts": artifacts,
            "source_metadata": {
                "task_index_path": str(index_path),
                "task_index_sha256": file_sha256(index_path),
                "app_family": index_row.get("app_family"),
                "setup_kind": index_row.get("setup_kind"),
            },
        }


def _split_score(task_key: str, *, seed: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{task_key}".encode()).digest()


def build_task_manifest(
    source_spec_path: Path, denylist_path: Path, output_dir: Path
) -> dict[str, Any]:
    ensure_empty_output(output_dir)
    spec = read_json(source_spec_path)
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("source spec must be a schema_version=1 object")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ContractError("source spec must contain a non-empty sources array")
    denylist = load_heldout_denylist(denylist_path)
    seed = str(spec.get("split_seed", "teacher-sft-v1"))
    validation_fraction = float(spec.get("validation_fraction", 0.1))
    if not 0.0 <= validation_fraction < 1.0:
        raise ContractError("validation_fraction must be in [0,1)")
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ContractError("source entries must be objects")
        source = dict(source)
        for path_key in ("task_index", "task_root", "bundle_root"):
            raw_path = source.get(path_key)
            if isinstance(raw_path, str) and not Path(raw_path).is_absolute():
                source[path_key] = str((source_spec_path.parent / raw_path).resolve())
        kind = source.get("kind")
        if kind == "osworld":
            source_rows = _osworld_rows(source)
        elif kind == "cua_gym":
            source_rows = _cuagym_rows(source)
        else:
            raise ContractError(f"unsupported task source: {kind!r}")
        for row in source_rows:
            assert_not_heldout(
                denylist=denylist,
                task_key=row["task_key"],
                source_task_id=row["source_task_id"],
                instruction=row["instruction"],
                asset_hashes=(artifact["sha256"] for artifact in row["artifacts"]),
            )
            rows.append(row)
    keys = [row["task_key"] for row in rows]
    if not rows:
        raise ContractError("source spec selected no train tasks")
    if len(keys) != len(set(keys)):
        raise ContractError("duplicate task_key across task sources")
    rows.sort(key=lambda row: row["task_key"])
    n_validation = round(len(rows) * validation_fraction)
    if validation_fraction > 0 and len(rows) > 1:
        n_validation = max(1, min(len(rows) - 1, n_validation))
    else:
        n_validation = 0
    validation_keys = {
        row["task_key"]
        for row in sorted(
            rows,
            key=lambda row: (_split_score(row["task_key"], seed=seed), row["task_key"]),
        )[:n_validation]
    }
    for row in rows:
        row["split"] = (
            "train_validation" if row["task_key"] in validation_keys else "train"
        )
        row["task_row_sha256"] = object_sha256(row)
    split_rows = {
        split: [row["task_key"] for row in rows if row["split"] == split]
        for split in TRAIN_SPLITS
    }
    if set(split_rows["train"]) & set(split_rows["train_validation"]):
        raise ContractError("task split overlap")
    if not split_rows["train"]:
        raise ContractError("deterministic split assigned no tasks to train")
    tasks_path = output_dir / "tasks.jsonl"
    splits_path = output_dir / "splits.json"
    write_jsonl(tasks_path, rows)
    write_json(splits_path, {"schema_version": SCHEMA_VERSION, "splits": split_rows})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_task_manifest",
        "construction_scope": "train_only",
        "source_spec_sha256": file_sha256(source_spec_path),
        "heldout_denylist_sha256": denylist["denylist_sha256"],
        "tasks_sha256": file_sha256(tasks_path),
        "splits_sha256": file_sha256(splits_path),
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "counts": {split: len(values) for split, values in split_rows.items()},
        "sources": sorted({row["source"] for row in rows}),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_task_rows(task_manifest_dir: Path) -> list[dict[str, Any]]:
    manifest_path = task_manifest_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("construction_scope") != "train_only"
    ):
        raise ContractError("task manifest is not train-only")
    tasks_path = task_manifest_dir / "tasks.jsonl"
    splits_path = task_manifest_dir / "splits.json"
    if file_sha256(tasks_path) != manifest.get("tasks_sha256"):
        raise ContractError("task manifest tasks.jsonl hash mismatch")
    if file_sha256(splits_path) != manifest.get("splits_sha256"):
        raise ContractError("task manifest splits.json hash mismatch")
    split_payload = read_json(splits_path)
    if not isinstance(split_payload, dict) or not isinstance(
        split_payload.get("splits"), dict
    ):
        raise ContractError("task split manifest is malformed")
    declared_splits = split_payload["splits"]
    if set(declared_splits) != set(TRAIN_SPLITS) or any(
        not isinstance(declared_splits[split], list)
        or any(not isinstance(key, str) for key in declared_splits[split])
        for split in TRAIN_SPLITS
    ):
        raise ContractError("task split manifest must contain exact train splits")
    rows = list(iter_jsonl(tasks_path))
    seen_keys: set[str] = set()
    for row in rows:
        require_train_split(row.get("source_split"), context=str(row.get("task_key")))
        if row.get("split") not in TRAIN_SPLITS:
            raise ContractError(f"invalid train-derived split: {row.get('split')!r}")
        task_key = row.get("task_key")
        if not isinstance(task_key, str) or not task_key or task_key in seen_keys:
            raise ContractError(f"missing or duplicate task key: {task_key!r}")
        seen_keys.add(task_key)
        declared = row.get("task_row_sha256")
        body = dict(row)
        body.pop("task_row_sha256", None)
        if declared != object_sha256(body):
            raise ContractError(f"task row hash mismatch: {row.get('task_key')}")
        expected_keys = declared_splits[row["split"]]
        if row["task_key"] not in expected_keys:
            raise ContractError(
                f"task missing from declared split: {row.get('task_key')}"
            )
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ContractError(
                f"task has no artifact provenance: {row.get('task_key')}"
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ContractError(f"malformed task artifact: {row.get('task_key')}")
            verify_declared_hash(
                Path(str(artifact.get("path", ""))),
                artifact.get("sha256"),
                context=f"task artifact {row.get('task_key')}/{artifact.get('role')}",
            )
    actual_splits = {
        split: [row["task_key"] for row in rows if row["split"] == split]
        for split in TRAIN_SPLITS
    }
    if any(declared_splits[split] != actual_splits[split] for split in TRAIN_SPLITS):
        raise ContractError("task split manifest is not an exact task-row partition")
    expected_counts = {split: len(actual_splits[split]) for split in TRAIN_SPLITS}
    if manifest.get("counts") != expected_counts:
        raise ContractError("task manifest counts differ from task rows")
    return rows
