from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ..rung1.transport import RecordingTransport
from .fixtures import canonical_bytes, load_manifest, sha256_value
from .oracle import evaluate_in_fresh_process
from .states import gold_state, near_miss_state, reset_state
from .trajectory import ARMS, build_trajectory, execute_trajectory
from .vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    KvmFixtureSession,
    probe_fixture,
    setup_fixture,
    sha256_file,
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
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def run_build_selfcheck() -> dict[str, Any]:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for fixture in manifest.fixtures:
        row: dict[str, Any] = {
            "fixture_id": fixture.id,
            "fixture_sha256": fixture.fixture_sha256,
            "template": fixture.template,
            "reset_oracle": _assert_oracle(fixture, reset_state(fixture), False, "reset"),
            "near_miss_oracle": _assert_oracle(
                fixture, near_miss_state(fixture), False, "near miss"
            ),
            "gold_oracle": _assert_oracle(fixture, gold_state(fixture), True, "gold"),
            "arms": {},
        }
        for arm in ARMS:
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
                    raise SelfcheckError(f"{fixture.id}/{arm}: Unicode typing drift")
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
        "split_counts": {"development": len(rows), "evaluation_opened": 0},
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "provider_expected_sha256": PINNED_PROVIDER_SHA256,
        "drag_implementation": "Files/Nautilus filesystem-state task",
        "browser_slider_fallback": False,
        "rows": rows,
    }


def _wait_changed(
    transport: Any, fixture: Any, initial: dict[str, Any], timeout_s: float = 15.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    current = initial
    while time.monotonic() < deadline:
        current = probe_fixture(transport, fixture)
        if current != initial:
            return current
        time.sleep(0.25)
    return current


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
    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        vm_log_dir=output / "vm_logs",
    ) as session:
        for fixture in manifest.fixtures:
            for arm in ARMS:
                cell: dict[str, Any] = {
                    "fixture_id": fixture.id,
                    "fixture_sha256": fixture.fixture_sha256,
                    "template": fixture.template,
                    "arm": arm,
                }
                transport = session.reset_to_ready()
                first = setup_fixture(transport, fixture)
                cell["reset_a_oracle"] = _assert_oracle(
                    fixture, first.state, False, "reset A"
                )
                reset_a = {
                    "state": first.state,
                    "geometry": asdict(first.geometry),
                    "cursor": list(transport.cursor_position()),
                }
                near = build_trajectory(
                    fixture,
                    arm=arm,
                    cursor=transport.cursor_position(),
                    geometry=first.geometry,
                    near_miss=True,
                )
                cell["near_dispatch"] = [
                    asdict(result)
                    for result in execute_trajectory(
                        near,
                        NativeAbsoluteExecutor(transport),
                        CompactRawExecutor(transport),
                    )
                ]
                near_state = _wait_changed(transport, fixture, first.state)
                cell["near_miss_oracle"] = _assert_oracle(
                    fixture, near_state, False, "near miss"
                )

                transport = session.reset_to_ready()
                second = setup_fixture(transport, fixture)
                cell["reset_b_oracle"] = _assert_oracle(
                    fixture, second.state, False, "reset B"
                )
                reset_b = {
                    "state": second.state,
                    "geometry": asdict(second.geometry),
                    "cursor": list(transport.cursor_position()),
                }
                cell["reset_a_sha256"] = sha256_value(reset_a)
                cell["reset_b_sha256"] = sha256_value(reset_b)
                cell["reset_equal"] = reset_a == reset_b
                if not cell["reset_equal"]:
                    cell["reset_a"] = reset_a
                    cell["reset_b"] = reset_b
                    raise SelfcheckError(f"{fixture.id}/{arm}: clean reset mismatch")

                gold = build_trajectory(
                    fixture,
                    arm=arm,
                    cursor=transport.cursor_position(),
                    geometry=second.geometry,
                )
                cell["gold_dispatch"] = [
                    asdict(result)
                    for result in execute_trajectory(
                        gold,
                        NativeAbsoluteExecutor(transport),
                        CompactRawExecutor(transport),
                    )
                ]
                final = _wait_changed(transport, fixture, second.state)
                cell["gold_state_sha256"] = sha256_value(final)
                cell["gold_oracle"] = _assert_oracle(fixture, final, True, "gold")
                if transport.audit.held_buttons or transport.audit.held_keys:
                    raise SelfcheckError(f"{fixture.id}/{arm}: gold leaked input state")
                if fixture.template == "vscode_focus_type" and transport.audit.typed_texts != [fixture.expected["text"]]:
                    raise SelfcheckError(f"{fixture.id}/{arm}: exact Unicode audit failed")
                if fixture.template == "local_document_scroll":
                    sign = -1 if fixture.params["direction"] == "down" else 1
                    if transport.audit.scroll_total * sign <= 0:
                        raise SelfcheckError(f"{fixture.id}/{arm}: signed scroll audit failed")
                cell["status"] = "passed"
                cells.append(cell)
                _atomic_json(
                    output / "progress.json",
                    {"status": "running", "expected_cells": 12, "cells": cells},
                )
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": "rung1b_real_application_development",
        "snapshot": READY_SNAPSHOT,
        "provider": {"path": str(provider), "sha256": observed_provider_sha},
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "cells": cells,
        "evaluation_opened": 0,
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
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
