from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ..rung1.transport import RecordingTransport
from .fixtures import load_manifest, sha256_value
from .oracle import evaluate_in_fresh_process
from .states import gold_state, near_miss_state, reset_state
from .trajectory import ARMS, build_trajectory, execute_trajectory
from .vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    AppReadinessError,
    AppSettleTimeout,
    KvmFixtureSession,
    collect_fixture_diagnostics,
    probe_fixture,
    probe_geometry,
    setup_fixture,
    sha256_file,
    wait_for_action_settle,
)


PINNED_PROVIDER_SHA256 = "76a8f44fab16c6dd38a4378a270e38758ba8d31885f244baedb95d8178f588d7"


class SelfcheckError(RuntimeError):
    pass


def _assert_oracle(fixture: Any, state: dict[str, Any], solved: bool, label: str) -> dict[str, Any]:
    result = evaluate_in_fresh_process(fixture, state)
    if result.oracle_status != "ok" or result.MOUSE_SOLVED is not solved:
        raise SelfcheckError(f"{fixture.id}: {label} oracle mismatch: {result}")
    return asdict(result)


def _atomic_json(path: Path, value: Any) -> None:
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


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def _arm_order(fixture: Any) -> tuple[str, ...]:
    """Deterministically counterbalance the two arms across fixture seeds."""
    return ARMS if int(fixture.parameter_seed) % 2 == 0 else tuple(reversed(ARMS))


def _capture_screenshot(transport: Any, path: Path) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            transport.base_url + "/screenshot", timeout=15
        ) as response:
            payload = response.read()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("VM screenshot endpoint did not return PNG")
        _atomic_bytes(path, payload)
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _audit_evidence(transport: Any) -> dict[str, Any]:
    audit = transport.audit
    return {
        "operations": [asdict(item) for item in audit.operations],
        "held_buttons": sorted(audit.held_buttons),
        "held_keys": sorted(audit.held_keys),
        "scroll_total": audit.scroll_total,
        "typed_texts": list(audit.typed_texts),
    }


def _qemu_log_evidence(vm_log_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not vm_log_dir.exists():
        return rows
    for path in sorted(item for item in vm_log_dir.rglob("*") if item.is_file()):
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                handle.seek(max(0, size - 12000))
                tail = handle.read()
            rows.append(
                {
                    "path": str(path),
                    "bytes": size,
                    "sha256": sha256_file(path),
                    "tail": tail.decode("utf-8", errors="replace"),
                }
            )
        except OSError as exc:
            rows.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _execute_with_journal(
    trajectory: Any,
    native: NativeAbsoluteExecutor,
    compact: CompactRawExecutor,
    journal: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for index, action in enumerate(trajectory.actions):
        row: dict[str, Any] = {"index": index, "action": action}
        journal.append(row)
        if trajectory.arm == "native_absolute_control":
            if not isinstance(action, dict):
                raise TypeError("native action was not an object")
            result = native.execute(action)
        else:
            if not isinstance(action, str):
                raise TypeError("compact action was not text")
            result = compact.execute(action)
        row["dispatch"] = asdict(result)
        if result.executor_dispatch_status != "ok":
            raise RuntimeError(f"scripted dispatch failed: {result}")
    return journal


def run_build_selfcheck() -> dict[str, Any]:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for fixture in manifest.fixtures:
        row: dict[str, Any] = {
            "fixture_id": fixture.id,
            "fixture_sha256": fixture.fixture_sha256,
            "template": fixture.template,
            "gate_role": fixture.gate_role,
            "coverage_label": fixture.coverage_label,
            "arm_order_seed": fixture.parameter_seed,
            "arm_order": list(_arm_order(fixture)),
            "reset_oracle": _assert_oracle(fixture, reset_state(fixture), False, "reset"),
            "near_miss_oracle": _assert_oracle(
                fixture, near_miss_state(fixture), False, "near miss"
            ),
            "gold_oracle": _assert_oracle(fixture, gold_state(fixture), True, "gold"),
            "arms": {},
        }
        for arm in _arm_order(fixture):
            transport = RecordingTransport(cursor=(73, 91))
            trajectory = build_trajectory(
                fixture, arm=arm, cursor=transport.cursor_position()
            )
            results = execute_trajectory(
                trajectory,
                NativeAbsoluteExecutor(transport),
                CompactRawExecutor(transport),
            )
            if transport.audit.held_buttons or transport.audit.held_keys:
                raise SelfcheckError(f"{fixture.id}/{arm}: input state leaked")
            if fixture.template == "vscode_focus_type":
                expected = str(fixture.expected["text"])
                if transport.audit.typed_texts != [expected]:
                    raise SelfcheckError(f"{fixture.id}/{arm}: coalesced typing drift")
            if fixture.template == "local_document_scroll":
                expected_sign = -1 if fixture.params["direction"] == "down" else 1
                if transport.audit.scroll_total * expected_sign <= 0:
                    raise SelfcheckError(f"{fixture.id}/{arm}: signed scroll drift")
            row["arms"][arm] = {
                "action_count": len(results),
                "action_classes": list(trajectory.action_classes),
                "typed_texts": transport.audit.typed_texts,
                "scroll_total": transport.audit.scroll_total,
                "final_cursor": list(transport.cursor_position()),
            }
        rows.append(row)
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": "rung1b_real_application_development",
        "fixture_count": len(rows),
        "split": "development",
        "split_counts": {"development": len(rows), "evaluation": 0},
        "evaluation_opened": 0,
        "sealed_eval_executed": False,
        "sealed_evaluation_access": False,
        "model_access": False,
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "arm_order_policy": "fixture_seed_parity_v1",
        "gate_roles": {
            "primary_gate": sum(row["gate_role"] == "primary_gate" for row in rows),
            "capability_probe": sum(
                row["gate_role"] == "capability_probe" for row in rows
            ),
        },
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "provider_expected_sha256": PINNED_PROVIDER_SHA256,
        "drag_implementation": "Files/Nautilus filesystem-state task",
        "browser_slider_fallback": False,
        "rows": rows,
    }


def run_vm_selfcheck(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider: Path,
    expected_provider_sha256: str,
) -> dict[str, Any]:
    manifest = load_manifest()
    observed_provider_sha = sha256_file(provider)
    if observed_provider_sha != expected_provider_sha256:
        raise SelfcheckError(
            f"provider hash mismatch: {observed_provider_sha} != {expected_provider_sha256}"
        )
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_cells = len(manifest.fixtures) * len(ARMS)
    vm_log_dir = output / "vm_logs"

    def checkpoint(cell: dict[str, Any], phase: str) -> None:
        cell["phase"] = phase
        _atomic_json(
            output / "progress.json",
            {
                "schema_version": 1,
                "status": "running",
                "split": "development",
                "sealed_eval_executed": False,
                "expected_cells": expected_cells,
                "attempted_cells": len(cells),
                "failed_cells": len(failures),
                "cells": cells,
            },
        )

    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        vm_log_dir=vm_log_dir,
    ) as session:
        for fixture in manifest.fixtures:
            arm_order = _arm_order(fixture)
            for arm_order_index, arm in enumerate(arm_order):
                cell: dict[str, Any] = {
                    "fixture_id": fixture.id,
                    "fixture_sha256": fixture.fixture_sha256,
                    "template": fixture.template,
                    "gate_role": fixture.gate_role,
                    "coverage_label": fixture.coverage_label,
                    "arm": arm,
                    "arm_order_seed": fixture.parameter_seed,
                    "arm_order": list(arm_order),
                    "arm_order_index": arm_order_index,
                    "status": "attempted",
                }
                cells.append(cell)
                # The attempted cell is durable before reset/setup/oracle assertions.
                checkpoint(cell, "cell_attempted")
                transport: Any | None = None
                cell_dir = output / "cells" / fixture.id / arm
                try:
                    transport = session.reset_to_ready()
                    checkpoint(cell, "reset_a_restored")
                    first = setup_fixture(transport, fixture)
                    reset_a = {
                        "state": first.state,
                        "geometry": asdict(first.geometry),
                        "cursor": list(transport.cursor_position()),
                    }
                    cell["reset_a"] = {
                        **reset_a,
                        "readiness": first.readiness,
                        "screenshot": _capture_screenshot(
                            transport, cell_dir / "reset_a.png"
                        ),
                    }
                    checkpoint(cell, "reset_a_recorded_before_oracle")
                    cell["reset_a_oracle"] = _assert_oracle(
                        fixture, first.state, False, "reset A"
                    )
                    checkpoint(cell, "reset_a_oracle_rejected")

                    near = build_trajectory(
                        fixture,
                        arm=arm,
                        cursor=transport.cursor_position(),
                        geometry=first.geometry,
                        near_miss=True,
                    )
                    cell["near_actions"] = list(near.actions)
                    cell["near_action_classes"] = list(near.action_classes)
                    cell["near_dispatch"] = []
                    checkpoint(cell, "near_miss_actions_recorded")
                    _execute_with_journal(
                        near,
                        NativeAbsoluteExecutor(transport),
                        CompactRawExecutor(transport),
                        cell["near_dispatch"],
                    )
                    cell["near_audit"] = _audit_evidence(transport)
                    checkpoint(cell, "near_miss_dispatched")
                    near_settled = wait_for_action_settle(
                        transport,
                        fixture,
                        first.state,
                        phase="near_miss",
                    )
                    cell["near_settle"] = asdict(near_settled)
                    cell["near_screenshot"] = _capture_screenshot(
                        transport, cell_dir / "near_settled.png"
                    )
                    checkpoint(cell, "near_miss_settled_before_oracle")
                    cell["near_miss_oracle"] = _assert_oracle(
                        fixture, near_settled.state, False, "near miss"
                    )
                    checkpoint(cell, "near_miss_oracle_rejected")

                    transport = session.reset_to_ready()
                    checkpoint(cell, "reset_b_restored")
                    second = setup_fixture(transport, fixture)
                    reset_b = {
                        "state": second.state,
                        "geometry": asdict(second.geometry),
                        "cursor": list(transport.cursor_position()),
                    }
                    cell["reset_b"] = {
                        **reset_b,
                        "readiness": second.readiness,
                        "screenshot": _capture_screenshot(
                            transport, cell_dir / "reset_b.png"
                        ),
                    }
                    checkpoint(cell, "reset_b_recorded_before_oracle")
                    cell["reset_b_oracle"] = _assert_oracle(
                        fixture, second.state, False, "reset B"
                    )
                    cell["reset_a_sha256"] = sha256_value(reset_a)
                    cell["reset_b_sha256"] = sha256_value(reset_b)
                    cell["reset_equal"] = reset_a == reset_b
                    checkpoint(cell, "reset_comparison_recorded")
                    if not cell["reset_equal"]:
                        raise SelfcheckError(f"{fixture.id}/{arm}: clean reset mismatch")

                    gold = build_trajectory(
                        fixture,
                        arm=arm,
                        cursor=transport.cursor_position(),
                        geometry=second.geometry,
                    )
                    cell["gold_actions"] = list(gold.actions)
                    cell["gold_action_classes"] = list(gold.action_classes)
                    cell["gold_dispatch"] = []
                    checkpoint(cell, "gold_actions_recorded")
                    _execute_with_journal(
                        gold,
                        NativeAbsoluteExecutor(transport),
                        CompactRawExecutor(transport),
                        cell["gold_dispatch"],
                    )
                    cell["gold_audit"] = _audit_evidence(transport)
                    checkpoint(cell, "gold_dispatched")
                    gold_settled = wait_for_action_settle(
                        transport,
                        fixture,
                        second.state,
                        phase="gold",
                    )
                    cell["gold_settle"] = asdict(gold_settled)
                    cell["gold_state_sha256"] = sha256_value(gold_settled.state)
                    cell["gold_screenshot"] = _capture_screenshot(
                        transport, cell_dir / "gold_settled.png"
                    )
                    checkpoint(cell, "gold_settled_before_oracle")
                    cell["gold_oracle"] = _assert_oracle(
                        fixture, gold_settled.state, True, "gold"
                    )
                    checkpoint(cell, "gold_oracle_accepted")
                    if transport.audit.held_buttons or transport.audit.held_keys:
                        raise SelfcheckError(
                            f"{fixture.id}/{arm}: gold leaked input state"
                        )
                    if (
                        fixture.template == "vscode_focus_type"
                        and transport.audit.typed_texts != [fixture.expected["text"]]
                    ):
                        raise SelfcheckError(
                            f"{fixture.id}/{arm}: exact coalesced-type audit failed"
                        )
                    if fixture.template == "local_document_scroll":
                        sign = -1 if fixture.params["direction"] == "down" else 1
                        if transport.audit.scroll_total * sign <= 0:
                            raise SelfcheckError(
                                f"{fixture.id}/{arm}: signed scroll audit failed"
                            )
                    cell["status"] = "passed"
                    checkpoint(cell, "cell_passed")
                except Exception as exc:
                    failed_phase = str(cell.get("phase", "unknown"))
                    failure: dict[str, Any] = {
                        "phase": failed_phase,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    if isinstance(exc, AppReadinessError):
                        failure["readiness"] = {
                            "failed_phase": exc.failed_phase,
                            "evidence": exc.evidence,
                        }
                    if isinstance(exc, AppSettleTimeout):
                        failure["settle"] = exc.evidence
                    if transport is not None:
                        failure["audit"] = _audit_evidence(transport)
                        failure["screenshot"] = _capture_screenshot(
                            transport, cell_dir / "failure.png"
                        )
                        live: dict[str, Any] = {}
                        for name, probe in (
                            ("state", lambda: probe_fixture(transport, fixture)),
                            ("geometry", lambda: asdict(probe_geometry(transport, fixture))),
                            ("cursor", lambda: list(transport.cursor_position())),
                        ):
                            try:
                                live[name] = probe()
                            except Exception as probe_exc:
                                live[name + "_error"] = (
                                    f"{type(probe_exc).__name__}: {probe_exc}"
                                )
                        failure["live"] = live
                        try:
                            failure["guest_logs"] = collect_fixture_diagnostics(
                                transport, fixture
                            )
                        except Exception as diagnostic_exc:
                            failure["guest_logs"] = {
                                "diagnostic_error": (
                                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                                )
                            }
                    failure["qemu_logs"] = _qemu_log_evidence(vm_log_dir)
                    cell["status"] = "failed"
                    cell["failure"] = failure
                    context_path = cell_dir / "failure_context.json"
                    cell["failure_context_path"] = str(context_path)
                    failures.append(
                        {
                            "fixture_id": fixture.id,
                            "arm": arm,
                            "phase": failed_phase,
                            "path": str(context_path),
                        }
                    )
                    _atomic_json(context_path, {"schema_version": 1, "cell": cell})
                    checkpoint(cell, "failure_context_persisted")
                    try:
                        session.reset_to_ready()
                        cell["post_failure_clean_reset"] = {"status": "ok"}
                    except Exception as reset_exc:
                        cell["post_failure_clean_reset"] = {
                            "status": "error",
                            "error": f"{type(reset_exc).__name__}: {reset_exc}",
                        }
                    _atomic_json(context_path, {"schema_version": 1, "cell": cell})
                    checkpoint(cell, "post_failure_reset_attempted")
                    # Every following cell gets its own fresh reset attempt.
                    continue
    final_progress = {
        "schema_version": 1,
        "status": "failed" if failures else "complete",
        "split": "development",
        "sealed_eval_executed": False,
        "expected_cells": expected_cells,
        "attempted_cells": len(cells),
        "failed_cells": len(failures),
        "failures": failures,
        "cells": cells,
    }
    _atomic_json(output / "progress.json", final_progress)
    if len(cells) != expected_cells:
        raise SelfcheckError(
            f"VM selfcheck attempted {len(cells)} of {expected_cells} cells"
        )
    if failures:
        raise SelfcheckError(
            f"VM selfcheck had {len(failures)} failed cells; see {output / 'progress.json'}"
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": "rung1b_real_application_development",
        "split": "development",
        "split_counts": {"development": len(manifest.fixtures), "evaluation": 0},
        "snapshot": READY_SNAPSHOT,
        "provider": {"path": str(provider), "sha256": observed_provider_sha},
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "cells": cells,
        "evaluation_opened": 0,
        "sealed_eval_executed": False,
        "sealed_evaluation_access": False,
        "model_access": False,
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "arm_order_policy": "fixture_seed_parity_v1",
        "attempted_cells": len(cells),
        "expected_cells": expected_cells,
        "gpu_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "vm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--expected-provider-sha256",
        "--expected_provider_sha256",
        default=PINNED_PROVIDER_SHA256,
        dest="expected_provider_sha256",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "selfcheck.json"
    marker.unlink(missing_ok=True)
    (args.output / "failure.json").unlink(missing_ok=True)
    try:
        payload = (
            run_build_selfcheck()
            if args.mode == "build"
            else run_vm_selfcheck(
                output=args.output,
                qcow=args.qcow,
                qemu=args.qemu,
                provider=args.provider,
                expected_provider_sha256=args.expected_provider_sha256,
            )
        )
        _atomic_json(marker, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "mode": args.mode,
            "split": "development",
            "evaluation_opened": 0,
            "sealed_eval_executed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "progress": str(args.output / "progress.json"),
        }
        if isinstance(exc, AppReadinessError):
            failure["readiness"] = {
                "fixture_id": exc.fixture_id,
                "failed_phase": exc.failed_phase,
                "evidence": exc.evidence,
            }
        if isinstance(exc, AppSettleTimeout):
            failure["settle"] = exc.evidence
        _atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
