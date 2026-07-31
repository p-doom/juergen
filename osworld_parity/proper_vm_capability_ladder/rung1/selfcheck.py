from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .executor import CompactRawExecutor, NativeAbsoluteExecutor
from .fixtures import Fixture, load_manifest
from .oracle import evaluate_in_fresh_process
from .server import FixtureHttpServer, render_fixture_html
from .trajectory import Arm, build_trajectory
from .transport import Operation
from .vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    KvmFixtureSession,
    sha256_file,
)


ARMS: tuple[Arm, ...] = ("native_absolute_control", "compact_raw_phaseb")


class SelfcheckError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _operations(values: tuple[Operation, ...]) -> list[dict[str, Any]]:
    return [{"kind": item.kind, "args": list(item.args)} for item in values]


def _initial_signature(state: dict[str, Any]) -> str:
    payload = {
        "fixture_id": state["fixture_id"],
        "fixture_sha256": state["fixture_sha256"],
        "ready": state["ready"],
        "geometry": state["geometry"],
        "current": state["current"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execute(
    arm: Arm, transport: Any, trajectory: tuple[dict[str, Any] | str, ...]
) -> list[dict[str, Any]]:
    executor = (
        NativeAbsoluteExecutor(transport)
        if arm == "native_absolute_control"
        else CompactRawExecutor(transport)
    )
    records: list[dict[str, Any]] = []
    for action in trajectory:
        result = executor.execute(action)  # type: ignore[arg-type]
        records.append(
            {
                "parse_status": result.parse_status,
                "executor_dispatch_status": result.executor_dispatch_status,
                "action_class": result.action_class,
                "operations": _operations(result.operations),
            }
        )
        time.sleep(0.2)
    return records


def _assert_negative(fixture: Fixture, state: dict[str, Any], stage: str) -> dict[str, Any]:
    result = evaluate_in_fresh_process(fixture, state)
    if result.oracle_status != "ok" or result.MOUSE_SOLVED:
        raise SelfcheckError(f"{fixture.id}: {stage} oracle was not a clean negative")
    return asdict(result)


def _assert_positive(fixture: Fixture, state: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_in_fresh_process(fixture, state)
    if result.oracle_status != "ok" or not result.MOUSE_SOLVED:
        raise SelfcheckError(
            f"{fixture.id}: gold oracle rejected state: {result.reason}; state={state['current']}"
        )
    return asdict(result)


def _held_button_action(arm: Arm) -> dict[str, Any] | str:
    if arm == "native_absolute_control":
        return {"action": "mouse_down", "button": "left"}
    return "0 0 0 ; +LMB"


def run_vm_selfcheck(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider_path: Path,
    expected_provider_sha256: str | None,
) -> dict[str, Any]:
    manifest = load_manifest()
    fixtures = manifest.select(split="development")
    provider_sha256 = sha256_file(provider_path)
    if expected_provider_sha256 and provider_sha256 != expected_provider_sha256:
        raise SelfcheckError(
            f"KVM provider hash mismatch: {provider_sha256} != {expected_provider_sha256}"
        )
    cells: list[dict[str, Any]] = []
    vm_log_dir = output / "vm_logs"
    with FixtureHttpServer(manifest) as server, KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider_path,
        vm_log_dir=vm_log_dir,
    ) as session:
        for fixture in fixtures:
            for arm in ARMS:
                cell: dict[str, Any] = {
                    "fixture_id": fixture.id,
                    "fixture_sha256": fixture.fixture_sha256,
                    "template": fixture.template,
                    "arm": arm,
                    "horizon": fixture.horizon,
                }

                # Reset A: deterministic setup and clean negative oracle.
                transport = session.reset_to_ready()
                first = session.launch_fixture(server, fixture)
                first_cursor = transport.cursor_position()
                cell["reset_oracle"] = _assert_negative(fixture, first, "reset")
                first_signature = _initial_signature(first)
                if session.probe_pointer_buttons(server, fixture) != 0:
                    raise SelfcheckError(f"{fixture.id}: button held after first reset")

                # Scripted near miss must be rejected.
                near = build_trajectory(
                    fixture,
                    first,
                    arm=arm,
                    cursor=transport.cursor_position(),
                    near_miss=True,
                )
                cell["near_miss_dispatch"] = _execute(arm, transport, near.actions)
                time.sleep(0.5)
                cell["near_miss_oracle"] = _assert_negative(
                    fixture, server.store.snapshot(fixture.id), "near miss"
                )

                # Deliberately leave LMB held, then prove the VM snapshot—not a
                # host-side cleanup helper—removes it on the second reset.
                cell["leak_injection"] = _execute(
                    arm, transport, (_held_button_action(arm),)
                )
                time.sleep(0.3)
                if server.store.snapshot(fixture.id)["last_pointer_buttons"] != 1:
                    raise SelfcheckError(f"{fixture.id}: held-button injection was not observed")

                # Reset B: exact fixture state, cursor, scroll/type state, and
                # button state must match Reset A.
                transport = session.reset_to_ready()
                second = session.launch_fixture(server, fixture)
                second_cursor = transport.cursor_position()
                cell["second_reset_oracle"] = _assert_negative(
                    fixture, second, "second reset"
                )
                second_signature = _initial_signature(second)
                if first_signature != second_signature:
                    raise SelfcheckError(
                        f"{fixture.id}: setup/reset signature changed "
                        f"{first_signature} != {second_signature}"
                    )
                if first_cursor != second_cursor:
                    raise SelfcheckError(
                        f"{fixture.id}: cursor leaked across reset: "
                        f"{first_cursor} != {second_cursor}"
                    )
                if session.probe_pointer_buttons(server, fixture) != 0:
                    raise SelfcheckError(f"{fixture.id}: button leaked across reset")
                cell["reset_leakage"] = {
                    "initial_state_sha256": second_signature,
                    "cursor": list(second_cursor),
                    "pointer_buttons": 0,
                    "current": second["current"],
                }

                # Scripted gold must pass a fresh host oracle process.
                gold = build_trajectory(
                    fixture,
                    second,
                    arm=arm,
                    cursor=transport.cursor_position(),
                )
                cell["gold_dispatch"] = _execute(arm, transport, gold.actions)
                time.sleep(0.8)
                final_state = server.store.snapshot(fixture.id)
                cell["gold_oracle"] = _assert_positive(fixture, final_state)
                if transport.audit.held_buttons:
                    raise SelfcheckError(
                        f"{fixture.id}: gold left held buttons: {transport.audit.held_buttons}"
                    )
                if fixture.template == "focus_type":
                    expected = fixture.expected["text"]
                    if transport.audit.typed_texts != [expected]:
                        raise SelfcheckError(
                            f"{fixture.id}: typing audit mismatch: {transport.audit.typed_texts!r}"
                        )
                if fixture.template == "scroll":
                    expected_sign = -1 if fixture.params["direction"] == "down" else 1
                    if transport.audit.scroll_total * expected_sign <= 0:
                        raise SelfcheckError(f"{fixture.id}: wrong signed scroll dispatch")
                cell["status"] = "passed"
                cells.append(cell)
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": "rung1a_instrumented_browser_microbench",
        "scientific_interpretation": "action mechanics and stateful closed loop only",
        "snapshot_name": READY_SNAPSHOT,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "development_fixture_count": len(fixtures),
        "evaluation_fixture_count": len(manifest.select(split="evaluation")),
        "selfcheck_cell_count": len(cells),
        "expected_selfcheck_cell_count": len(fixtures) * len(ARMS),
        "provider": {
            "path": str(provider_path.resolve()),
            "sha256": provider_sha256,
        },
        "qcow": {"path": str(qcow.resolve()), "size": qcow.stat().st_size},
        "qemu": str(qemu.resolve()),
        "cells": cells,
    }


def run_build_selfcheck() -> dict[str, Any]:
    manifest = load_manifest()
    html_hashes = {}
    for fixture in manifest.fixtures:
        rendered = render_fixture_html(fixture, 1).encode("utf-8")
        html_hashes[fixture.id] = hashlib.sha256(rendered).hexdigest()
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": "rung1a_instrumented_browser_microbench",
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "fixture_count": len(manifest.fixtures),
        "development_fixture_count": len(manifest.select(split="development")),
        "evaluation_fixture_count": len(manifest.select(split="evaluation")),
        "rendered_fixture_sha256": html_hashes,
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
                provider_path=args.provider,
                expected_provider_sha256=args.expected_provider_sha256,
            )
        )
        _atomic_json(marker, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # fail loud and with no trusted marker
        failure = {
            "schema_version": 1,
            "status": "failed",
            "mode": args.mode,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
