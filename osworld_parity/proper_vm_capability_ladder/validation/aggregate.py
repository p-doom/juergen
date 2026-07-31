from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_index import (
    BUILD_SCHEMA_VERSION,
    PINNED_SUBSTRATE_SHA256,
    ArtifactIndexError,
    _context_provenance,
    _load_object,
    canonical_bytes,
    sha256_file,
    validate_index,
)
from .failure_artifact_probe import SCHEMA_VERSION as FAILURE_SCHEMA


SCHEMA_VERSION = "proper_vm_executor_cert_v1"
ARMS = ("native_absolute_control", "compact_raw_phaseb")
SHARD_COUNT = 4
PAIRS_PER_SHARD = 25
CLICK_SPEC_SHA256 = "3059bd4c8057f0922652a7320fc0e6362ab3a850aeecc0ac6027855dd4f943b6"


class CertificationError(RuntimeError):
    pass


def capability_report_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("capability_report_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificationError(message)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def _common_qualification(value: dict[str, Any], label: str) -> None:
    expected = {
        "status": "passed",
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"{label}: {key} must be exactly {wanted!r}")


def _oracle(value: Any, *, solved: bool, label: str) -> None:
    _require(isinstance(value, dict), f"{label}: oracle result is missing")
    _require(value.get("oracle_status") == "ok", f"{label}: oracle did not run cleanly")
    _require(value.get("MOUSE_SOLVED") is solved, f"{label}: oracle solved state mismatch")


def _validate_atomic(state: dict[str, Any], *, compact: bool, label: str) -> None:
    expected = {
        "ok": True,
        "cleanup_attempted": False,
        "error": None,
        "failure_kind": None,
        "guest_process_count": 1,
    }
    for key, wanted in expected.items():
        _require(state.get(key) == wanted, f"{label}: atomic {key} mismatch")
    for key in ("semantic_operations", "lowered_operations", "operations"):
        _require(isinstance(state.get(key), list), f"{label}: missing atomic {key}")
    for key in ("cursor_before", "cursor_after", "cursor"):
        value = state.get(key)
        _require(
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, int) for item in value),
            f"{label}: invalid atomic {key}",
        )
    _require(state["cursor"] == state["cursor_after"], f"{label}: stale cursor readback")
    final_mask = state.get("pointer_button_mask")
    observed_mask = state.get("observed_pointer_button_mask")
    expected_mask = state.get("expected_pointer_button_mask")
    _require(
        isinstance(final_mask, int)
        and observed_mask == expected_mask == final_mask,
        f"{label}: successful atomic button masks differ",
    )
    primitives = state.get("backend_primitives")
    sync = state.get("x_event_sync_evidence")
    _require(isinstance(primitives, list), f"{label}: backend primitives missing")
    _require(isinstance(sync, list), f"{label}: X sync evidence missing")
    for item in sync:
        _require(
            isinstance(item, dict)
            and item.get("flush") is True
            and item.get("sync") is True,
            f"{label}: X event was not flushed and synchronized",
        )
    for primitive in primitives:
        if isinstance(primitive, dict) and primitive.get("kind") == "click":
            _require(primitive.get("dwell_ms") == 50, f"{label}: click dwell drifted")
            _require(
                primitive.get("ordering")
                == ["mouse_down", "flush", "sync", "dwell", "mouse_up", "flush", "sync"],
                f"{label}: click backend ordering drifted",
            )
    if compact:
        for key in ("compact_cursor_before_read", "compact_cursor_after_read"):
            value = state.get(key)
            _require(
                isinstance(value, list) and len(value) == 2,
                f"{label}: compact cursor evidence {key} is missing",
            )
        _require(
            state.get("cursor_readback_verified") is True,
            f"{label}: compact cursor readback was not verified",
        )


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _validate_dispatches(value: Any, *, arm: str, label: str) -> None:
    atomic_count = 0
    for item in _walk(value):
        for key in ("retry_count", "infrastructure_error_count", "verifier_failure_count"):
            if key in item:
                _require(item.get(key) == 0, f"{label}: nonzero {key}")
        if "executor_dispatch_status" in item:
            _require(
                item.get("parse_status") == "ok"
                and item.get("executor_dispatch_status") == "ok",
                f"{label}: executor dispatch failed",
            )
        if "dispatch_status" in item:
            _require(item.get("dispatch_status") == "dispatched", f"{label}: dispatch blocked")
        if "baseline_matches" in item:
            _require(item.get("baseline_matches") is True, f"{label}: stale cursor baseline")
        if "endpoint_matches" in item:
            _require(item.get("endpoint_matches") is True, f"{label}: cursor endpoint mismatch")
        if "planned_action_count" in item:
            _require(
                item.get("completed_action_count") == item.get("planned_action_count"),
                f"{label}: action stream was incomplete",
            )
        if "atomic_errors" in item:
            _require(item.get("atomic_errors") == [], f"{label}: atomic errors were recorded")
        state = item.get("atomic_state")
        if isinstance(state, dict):
            _validate_atomic(state, compact=arm == "compact_raw_phaseb", label=label)
            atomic_count += 1
    _require(atomic_count > 0, f"{label}: no atomic execution evidence")


def _manifest_rows(path: Path, *, split: str | None = None) -> list[dict[str, Any]]:
    value = _load_object(path)
    rows = value.get("fixtures")
    _require(isinstance(rows, list), f"manifest {path} has no fixtures")
    parsed = [row for row in rows if isinstance(row, dict)]
    _require(len(parsed) == len(rows), f"manifest {path} contains invalid fixture rows")
    if split is not None:
        parsed = [row for row in parsed if row.get("split") == split]
    return parsed


def validate_build(value: dict[str, Any]) -> None:
    _require(value.get("schema_version") == BUILD_SCHEMA_VERSION, "build schema mismatch")
    _common_qualification(value, "build")
    _require(value.get("baseline_test_count") == 109, "build baseline test count drifted")
    _require(
        isinstance(value.get("current_test_count"), int)
        and value["current_test_count"] >= 109,
        "build ran fewer than the 109-test baseline",
    )
    _require(value.get("failure_count") == 0, "build contains test failures")
    _require(value.get("error_count") == 0, "build contains test errors")


def _validate_click_trial(trial: dict[str, Any], *, label: str) -> None:
    _require(trial.get("status") == "passed", f"{label}: trial did not pass")
    _require(trial.get("retry_count") == 0, f"{label}: retry observed")
    _require(trial.get("dispatch_count") == 1, f"{label}: dispatch count drifted")
    _require(trial.get("reset_before_trial") is True, f"{label}: no clean reset")
    _require(trial.get("oracle_invocation_count") == 0, f"{label}: oracle-conditioned click")
    _require(trial.get("oracle_conditioned_dispatch") is False, f"{label}: oracle-conditioned dispatch")
    _require(trial.get("final_pointer_button_mask") == 0, f"{label}: held pointer button")
    _require(
        trial.get("final_state") == {"checked": True, "decoy_checked": False},
        f"{label}: click did not reach exact state",
    )
    _require(
        trial.get("lowered_operations") == ["click"],
        f"{label}: click did not use the shared primitive",
    )
    primitives = trial.get("backend_primitives")
    _require(isinstance(primitives, list) and len(primitives) == 1, f"{label}: click primitive count drifted")
    primitive = primitives[0]
    _require(
        primitive.get("kind") == "click"
        and primitive.get("button") == "left"
        and primitive.get("call") == "pyautogui.click(clicks=1, interval=0.05)"
        and primitive.get("x11_per_event_sync_hooked") is True,
        f"{label}: shared click backend contract drifted",
    )
    sync = trial.get("x_event_sync_evidence")
    _require(
        isinstance(sync, list)
        and [item.get("event") for item in sync] == ["mouse_down", "mouse_up"],
        f"{label}: click X-event sequence drifted",
    )
    for item in sync:
        _require(item.get("flush") is True and item.get("sync") is True, f"{label}: unsynced X event")


def validate_preflight(value: dict[str, Any]) -> None:
    _common_qualification(value, "click preflight")
    expected = {"mode": "vm", "pair_count": 5, "trial_count": 10, "passed_trial_count": 10}
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"click preflight {key} mismatch")
    trials = value.get("trials")
    pairs = value.get("pairs")
    _require(isinstance(trials, list) and len(trials) == 10, "click preflight trial set incomplete")
    _require(isinstance(pairs, list) and len(pairs) == 5, "click preflight pair set incomplete")
    keys = {(trial.get("pair_index"), trial.get("arm")) for trial in trials}
    _require(keys == {(pair, arm) for pair in range(1, 6) for arm in ARMS}, "click preflight cells differ")
    for trial in trials:
        _validate_click_trial(trial, label=f"preflight/{trial.get('trial_index')}")


def validate_failure_probe(value: dict[str, Any], index: dict[str, Any]) -> None:
    _require(value.get("schema_version") == FAILURE_SCHEMA, "failure probe schema mismatch")
    _common_qualification(value, "failure probe")
    _require(value.get("expected_outcome") == "injected_executor_failure", "failure probe outcome drifted")
    checks = value.get("checks")
    _require(isinstance(checks, dict) and checks and all(item is True for item in checks.values()), "failure probe checks incomplete")
    terminal = index.get("terminal_results")
    _require(isinstance(terminal, dict) and "raw_failure" in terminal, "raw failure artifact was not indexed")
    _require(
        terminal["raw_failure"].get("sha256") == value.get("failure_artifact_sha256"),
        "raw failure artifact hash differs from the probe marker",
    )
    _require("failure_screenshot" in terminal, "failure screenshot artifact was not indexed")
    screenshot = value.get("failure_screenshot")
    _require(isinstance(screenshot, dict), "failure probe marker has no screenshot metadata")
    _require(
        terminal["failure_screenshot"].get("sha256") == screenshot.get("sha256")
        and terminal["failure_screenshot"].get("size") == screenshot.get("bytes"),
        "failure screenshot artifact hash/size differs from the probe marker",
    )


def validate_click_shards(values: Mapping[int, dict[str, Any]]) -> None:
    _require(set(values) == set(range(SHARD_COUNT)), "full click shard IDs are incomplete")
    global_cells: set[tuple[int, str]] = set()
    for shard_id, value in values.items():
        _require(value.get("schema_version") == 1, f"click shard {shard_id} schema mismatch")
        _common_qualification(value, f"click shard {shard_id}")
        expected = {
            "shard_index": shard_id,
            "shard_count": SHARD_COUNT,
            "pair_count": PAIRS_PER_SHARD,
            "trial_count": 2 * PAIRS_PER_SHARD,
            "arm_trial_counts": {arm: PAIRS_PER_SHARD for arm in ARMS},
        }
        for key, wanted in expected.items():
            _require(value.get(key) == wanted, f"click shard {shard_id} {key} mismatch")
        _require(value.get("spec_sha256") == CLICK_SPEC_SHA256, f"click shard {shard_id} spec hash mismatch")
        _require(value.get("verifier_failure_count") == 0, f"click shard {shard_id} verifier failure")
        trials = value.get("trials")
        _require(isinstance(trials, list) and len(trials) == 50, f"click shard {shard_id} incomplete")
        for trial in trials:
            global_pair = trial.get("global_pair_index")
            cell = (global_pair, trial.get("arm"))
            _require(cell not in global_cells, f"duplicate full click cell {cell}")
            arm_slug = "native" if trial.get("arm") == ARMS[0] else "compact"
            expected_id = f"cert-{CLICK_SPEC_SHA256[:12]}-s{shard_id}-pair-{int(global_pair):03d}-{arm_slug}"
            _require(trial.get("trial_id") == expected_id, f"click shard {shard_id} trial id mismatch")
            global_cells.add(cell)
            _validate_click_trial(trial, label=f"click-shard-{shard_id}/{trial.get('trial_index')}")
    expected_cells = {(pair, arm) for pair in range(1, 101) for arm in ARMS}
    _require(global_cells == expected_cells, "full click certification is not exactly 100 trials per arm")


def validate_rung1a(value: dict[str, Any], manifest: Path) -> list[str]:
    _common_qualification(value, "rung1a")
    rows = _manifest_rows(manifest, split="development")
    expected = {(row["id"], arm) for row in rows for arm in ARMS}
    cells = value.get("cells")
    _require(isinstance(cells, list) and len(cells) == 16, "rung1a must contain 16 cells")
    _require(value.get("selfcheck_cell_count") == 16, "rung1a completed count mismatch")
    _require(value.get("expected_selfcheck_cell_count") == 16, "rung1a expected count mismatch")
    observed = {(cell.get("fixture_id"), cell.get("arm")) for cell in cells}
    _require(len(observed) == len(cells) and observed == expected, "rung1a missing or duplicate cells")
    for cell in cells:
        label = f"rung1a/{cell.get('fixture_id')}/{cell.get('arm')}"
        _require(cell.get("status") == "passed" and cell.get("journal_stage") == "cell_passed", f"{label}: unattempted cell")
        _oracle(cell.get("reset_oracle"), solved=False, label=label + "/reset-a")
        _oracle(cell.get("near_miss_oracle"), solved=False, label=label + "/near")
        _oracle(cell.get("second_reset_oracle"), solved=False, label=label + "/reset-b")
        _oracle(cell.get("gold_oracle"), solved=True, label=label + "/gold")
        _require(cell.get("reset_component_diff", {}).get("all_equal") is True, f"{label}: reset mismatch")
        _require(cell.get("reset_component_diff", {}).get("differing_components") == [], f"{label}: reset components differ")
        _require(cell.get("reset_a_snapshot", {}).get("pointer_buttons") == 0, f"{label}: held reset-a input")
        _require(cell.get("reset_b_snapshot", {}).get("pointer_buttons") == 0, f"{label}: held reset-b input")
        _validate_dispatches(cell, arm=str(cell["arm"]), label=label)
    return sorted(f"{fixture}:{arm}" for fixture, arm in observed)


def validate_rung1b(value: dict[str, Any], manifest: Path) -> list[str]:
    _common_qualification(value, "rung1b")
    _require(value.get("arm_order_policy") == "fixture_seed_parity_v1", "rung1b arm-order policy drifted")
    rows = _manifest_rows(manifest, split="development")
    seed_by_id = {str(row["id"]): int(row["parameter_seed"]) for row in rows}
    expected = {(fixture_id, arm) for fixture_id in seed_by_id for arm in ARMS}
    cells = value.get("cells")
    _require(isinstance(cells, list) and len(cells) == 12, "rung1b must contain 12 cells")
    observed = {(cell.get("fixture_id"), cell.get("arm")) for cell in cells}
    _require(len(observed) == len(cells) and observed == expected, "rung1b missing or duplicate cells")
    for cell in cells:
        fixture_id = str(cell.get("fixture_id"))
        arm = str(cell.get("arm"))
        seed = seed_by_id[fixture_id]
        order = list(ARMS if seed % 2 == 0 else reversed(ARMS))
        label = f"rung1b/{fixture_id}/{arm}"
        _require(cell.get("arm_order_seed") == seed, f"{label}: swap seed mismatch")
        _require(cell.get("arm_order") == order, f"{label}: swapped arm order mismatch")
        _require(cell.get("arm_order_index") == order.index(arm), f"{label}: arm order index mismatch")
        _require(cell.get("status") == "passed", f"{label}: unattempted cell")
        _oracle(cell.get("reset_a_oracle"), solved=False, label=label + "/reset-a")
        _oracle(cell.get("near_miss_oracle"), solved=False, label=label + "/near")
        _oracle(cell.get("reset_b_oracle"), solved=False, label=label + "/reset-b")
        _oracle(cell.get("gold_oracle"), solved=True, label=label + "/gold")
        _require(cell.get("reset_equal") is True, f"{label}: reset mismatch")
        _require(cell.get("reset_a_sha256") == cell.get("reset_b_sha256"), f"{label}: reset hashes differ")
        _validate_dispatches(cell, arm=arm, label=label)
    return sorted(f"{fixture}:{arm}" for fixture, arm in observed)


def validate_sameapp(value: dict[str, Any], manifest: Path) -> list[str]:
    _common_qualification(value, "same-app")
    _require(value.get("mode") == "vm" and value.get("split") == "development", "same-app split/mode mismatch")
    _require(value.get("sealed_eval_executed") is False, "same-app opened sealed evaluation")
    rows = _manifest_rows(manifest, split="development")
    expected = {(row["id"], arm) for row in rows for arm in ARMS}
    cells = value.get("rows")
    _require(isinstance(cells, list) and len(cells) == 8, "same-app must contain 8 cells")
    observed = {(cell.get("fixture_id"), cell.get("arm")) for cell in cells}
    _require(len(observed) == len(cells) and observed == expected, "same-app missing or duplicate cells")
    for cell in cells:
        label = f"same-app/{cell.get('fixture_id')}/{cell.get('arm')}"
        _oracle(cell.get("initial_oracle"), solved=False, label=label + "/reset")
        _oracle(cell.get("near_miss_oracle"), solved=False, label=label + "/near")
        _oracle(cell.get("gold_oracle"), solved=True, label=label + "/gold")
        _require(isinstance(cell.get("reset_signature"), str), f"{label}: reset signature missing")
        _validate_dispatches(cell, arm=str(cell["arm"]), label=label)
    return sorted(f"{fixture}:{arm}" for fixture, arm in observed)


def _load_index(path: Path) -> dict[str, Any]:
    value = _load_object(path)
    try:
        validate_index(value)
    except ArtifactIndexError as exc:
        raise CertificationError(str(exc)) from exc
    return value


def _bind_result(index: dict[str, Any], result_path: Path, label: str) -> None:
    terminal = index.get("terminal_results")
    _require(isinstance(terminal, dict), f"{label}: index has no terminal results")
    resolved = str(result_path.resolve())
    matches = [record for record in terminal.values() if record.get("path") == resolved]
    _require(len(matches) == 1, f"{label}: result is not bound exactly once in its index")
    _require(matches[0].get("sha256") == sha256_file(result_path), f"{label}: result hash mismatch")


def _parse_shards(values: list[str], option: str) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for raw in values:
        key, separator, path = raw.partition("=")
        _require(bool(separator and path), f"{option} must be SHARD=/absolute/path")
        shard = int(key)
        _require(shard not in parsed, f"duplicate {option} shard {shard}")
        candidate = Path(path)
        _require(candidate.is_absolute(), f"{option} paths must be absolute")
        parsed[shard] = candidate.resolve()
    return parsed


def aggregate(
    *,
    artifacts: Mapping[str, tuple[Path, Path]],
    click_shards: Mapping[int, tuple[Path, Path]],
    rung1a_manifest: Path,
    rung1b_manifest: Path,
    sameapp_manifest: Path,
    preregistration_evidence: Path,
    context: dict[str, Any],
    lock_file: Path,
) -> dict[str, Any]:
    required = {"build", "preflight", "failure_probe", "rung1a", "rung1b", "sameapp"}
    _require(set(artifacts) == required, "aggregate artifact inputs are incomplete")
    results = {name: _load_object(pair[0]) for name, pair in artifacts.items()}
    indexes = {name: _load_index(pair[1]) for name, pair in artifacts.items()}
    shard_results = {key: _load_object(pair[0]) for key, pair in click_shards.items()}
    shard_indexes = {key: _load_index(pair[1]) for key, pair in click_shards.items()}
    for name, (result_path, _) in artifacts.items():
        _bind_result(indexes[name], result_path, name)
    for shard, (result_path, _) in click_shards.items():
        _bind_result(shard_indexes[shard], result_path, f"click shard {shard}")

    aggregate_provenance = _context_provenance(context, lock_file)
    all_indexes = {**indexes, **{f"click_full_{key}": value for key, value in shard_indexes.items()}}
    identity = {
        "git_commit": aggregate_provenance["git_commit"],
        "git_tree": aggregate_provenance["git_tree"],
        "source_sha256": aggregate_provenance["source_sha256"],
    }
    for name, index in all_indexes.items():
        _require(index.get("source") == {**identity, "git_status_porcelain": "", "tracked_patch_sha256": aggregate_provenance["tracked_patch_sha256"], "untracked_patch_sha256": aggregate_provenance["untracked_patch_sha256"]}, f"{name}: integration source identity differs")
        _require(index.get("lock", {}).get("sha256") == aggregate_provenance["lock"]["sha256"], f"{name}: lock hash differs")
    build_sha = sha256_file(artifacts["build"][0])
    for name, index in all_indexes.items():
        if name == "build":
            continue
        _require(index.get("build_dependency", {}).get("sha256") == build_sha, f"{name}: build is not an exact dependency")
    vm_indexes = {name: index for name, index in all_indexes.items() if name != "build"}
    for name, index in vm_indexes.items():
        substrate = index.get("substrate")
        _require(isinstance(substrate, dict), f"{name}: VM substrate is missing")
        observed = {
            "provider": substrate.get("provider", {}).get("sha256"),
            "qemu_wrapper": substrate.get("qemu", {}).get("wrapper", {}).get("sha256"),
            "qemu_binary": substrate.get("qemu", {}).get("binary", {}).get("sha256"),
            "qemu_loader": substrate.get("qemu", {}).get("loader", {}).get("sha256"),
            "base_qcow": substrate.get("base_qcow", {}).get("sha256"),
        }
        _require(observed == PINNED_SUBSTRATE_SHA256, f"{name}: pinned VM substrate differs")
        vm = index.get("vm")
        _require(isinstance(vm, dict) and vm.get("closed") is True, f"{name}: VM was not cleanly closed")
        overlay = vm.get("overlay", {})
        _require(overlay.get("job_unique_scratch") is True, f"{name}: overlay lacks job-unique scratch containment")
        _require(overlay.get("removed") is True, f"{name}: overlay scratch was not removed")

    manifest_hashes = {
        "rung1a": sha256_file(rung1a_manifest),
        "rung1b": sha256_file(rung1b_manifest),
        "sameapp": sha256_file(sameapp_manifest),
    }
    fixture_expectations = {
        "preflight": manifest_hashes["rung1a"],
        "rung1a": manifest_hashes["rung1a"],
        "rung1b": manifest_hashes["rung1b"],
        "sameapp": manifest_hashes["sameapp"],
        **{f"click_full_{key}": manifest_hashes["rung1a"] for key in range(4)},
    }
    for name, expected_hash in fixture_expectations.items():
        records = all_indexes[name].get("fixtures")
        _require(
            isinstance(records, dict)
            and sum(record.get("sha256") == expected_hash for record in records.values()) == 1,
            f"{name}: exact qualification fixture manifest is not indexed",
        )

    validate_build(results["build"])
    validate_preflight(results["preflight"])
    validate_failure_probe(results["failure_probe"], indexes["failure_probe"])
    validate_click_shards(shard_results)
    rung_cells = {
        "rung1a": validate_rung1a(results["rung1a"], rung1a_manifest),
        "rung1b": validate_rung1b(results["rung1b"], rung1b_manifest),
        "sameapp": validate_sameapp(results["sameapp"], sameapp_manifest),
    }
    _require(
        preregistration_evidence.is_file() and not preregistration_evidence.is_symlink(),
        "executor preregistration evidence is missing",
    )
    marker_sha = sha256_file(preregistration_evidence)
    recipe_hashes = {name: index["submission"]["recipe_sha256"] for name, index in all_indexes.items()}
    result_hashes = {
        name: sha256_file(pair[0]) for name, pair in artifacts.items()
    } | {f"click_full_{key}": sha256_file(pair[0]) for key, pair in click_shards.items()}
    runs = {
        name: {
            "artifact_inputs": index["submission"]["input_artifacts"],
            "run_id": index["submission"]["run_id"],
            "job_id": index["submission"]["job_id"],
            "node": index["submission"]["node"],
            "content_address": index["content_address"],
            "commands": index["commands"],
            "vm": (
                {
                    "vm_id": index["vm"]["vm_id"],
                    "ports": index["vm"]["ports"],
                    "qmp_path": index["vm"]["qmp_path"],
                    "overlay": index["vm"]["overlay"],
                }
                if isinstance(index.get("vm"), dict)
                else None
            ),
        }
        for name, index in all_indexes.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "certification_schema": SCHEMA_VERSION,
        "status": "ready",
        "development_only": True,
        "scored_execution_completed": False,
        "validated_interfaces": [
            "native_absolute_control",
            "compact_raw_phaseb",
            "shared_atomic_gui_executor",
            "http_vm_transport",
        ],
        "executor_commit": identity["git_commit"],
        "vm_snapshot_id": "osworld_ready",
        "integration": identity,
        "lock_sha256": aggregate_provenance["lock"]["sha256"],
        "pinned_substrate_sha256": PINNED_SUBSTRATE_SHA256,
        "recipe_sha256": recipe_hashes,
        "terminal_result_sha256": result_hashes,
        "runs": runs,
        "aggregate_run": {
            "run_id": context.get("run_id"),
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("LABCTL_JOB_ID")),
            "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
            "outputs": context.get("outputs"),
        },
        "fixture_manifests": {
            "rung1a": {"path": str(rung1a_manifest), "sha256": manifest_hashes["rung1a"]},
            "rung1b": {"path": str(rung1b_manifest), "sha256": manifest_hashes["rung1b"]},
            "sameapp": {"path": str(sameapp_manifest), "sha256": manifest_hashes["sameapp"]},
        },
        "expected_cells": rung_cells,
        "full_click": {"shards": 4, "pairs_per_shard": 25, "trials_per_arm": 100, "total_trials": 200},
        "preregistration_evidence": {
            "path": str(preregistration_evidence),
            "sha256": marker_sha,
        },
        "checks": {
            "clean_build_at_least_109_tests": True,
            "narrow_click_preflight_10_trials": True,
            "forced_failure_artifact_probe_with_png": True,
            "full_click_100_trials_per_arm": True,
            "rung1a_16_cells": True,
            "rung1b_12_counterbalanced_cells": True,
            "sameapp_8_cells": True,
            "vm_isolation_and_provenance": True,
        },
    }
    payload["capability_report_sha256"] = capability_report_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in ("build", "preflight", "failure-probe", "rung1a", "rung1b", "sameapp"):
        parser.add_argument(f"--{name}-result", type=Path, required=True)
        parser.add_argument(f"--{name}-index", type=Path, required=True)
    parser.add_argument("--click-shard-result", action="append", default=[])
    parser.add_argument("--click-shard-index", action="append", default=[])
    parser.add_argument("--rung1a-manifest", type=Path, required=True)
    parser.add_argument("--rung1b-manifest", type=Path, required=True)
    parser.add_argument("--sameapp-manifest", type=Path, required=True)
    parser.add_argument("--preregistration-evidence", type=Path, required=True)
    parser.add_argument("--context", type=Path, default=os.environ.get("LABCTL_CONTEXT"))
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.context is None:
        raise CertificationError("--context or LABCTL_CONTEXT is required")
    artifacts = {
        "build": (args.build_result.resolve(), args.build_index.resolve()),
        "preflight": (args.preflight_result.resolve(), args.preflight_index.resolve()),
        "failure_probe": (args.failure_probe_result.resolve(), args.failure_probe_index.resolve()),
        "rung1a": (args.rung1a_result.resolve(), args.rung1a_index.resolve()),
        "rung1b": (args.rung1b_result.resolve(), args.rung1b_index.resolve()),
        "sameapp": (args.sameapp_result.resolve(), args.sameapp_index.resolve()),
    }
    shard_results = _parse_shards(args.click_shard_result, "--click-shard-result")
    shard_indexes = _parse_shards(args.click_shard_index, "--click-shard-index")
    _require(set(shard_results) == set(shard_indexes), "click shard result/index IDs differ")
    value = aggregate(
        artifacts=artifacts,
        click_shards={key: (shard_results[key], shard_indexes[key]) for key in shard_results},
        rung1a_manifest=args.rung1a_manifest.resolve(),
        rung1b_manifest=args.rung1b_manifest.resolve(),
        sameapp_manifest=args.sameapp_manifest.resolve(),
        preregistration_evidence=args.preregistration_evidence.resolve(),
        context=_load_object(args.context.resolve()),
        lock_file=args.lock_file.resolve(),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "EXECUTOR_READY.json"
    _atomic_json(marker, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
