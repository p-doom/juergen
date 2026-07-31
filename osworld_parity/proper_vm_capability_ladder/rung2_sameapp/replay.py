from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ..rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU, KvmFixtureSession
from .actions import compile_compact, compile_native
from .fixtures import assert_collectable_split, load_manifest
from .oracle import (
    evaluate_in_fresh_process,
    initial_state,
    reset_signature,
    scripted_state,
)
from .trajectory import build_trajectory
from .vm import AppReadinessError, probe_geometry, probe_state, reset_and_setup


ARMS = ("native_absolute_control", "compact_raw_phaseb")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reset_with_readiness_evidence(
    session: KvmFixtureSession, fixture: Any, output: Path
) -> tuple[Any, Any]:
    """Keep failure evidence before the VM context tears the guest down."""
    try:
        return reset_and_setup(session, fixture)
    except AppReadinessError as exc:
        evidence = dict(exc.evidence)
        evidence.update(
            {
                "error_type": type(exc).__name__,
                "failed_phase": exc.failed_phase,
                "message": str(exc),
            }
        )
        transport = session.transport
        if transport is not None:
            try:
                with urllib.request.urlopen(
                    transport.base_url + "/screenshot", timeout=15
                ) as response:
                    screenshot = response.read()
                if not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise RuntimeError("VM screenshot endpoint did not return PNG")
                path = output / "readiness_failure.png"
                path.write_bytes(screenshot)
                evidence["screenshot"] = {
                    "bytes": len(screenshot),
                    "path": str(path),
                    "sha256": hashlib.sha256(screenshot).hexdigest(),
                }
            except Exception as screenshot_exc:
                evidence["screenshot_error"] = (
                    f"{type(screenshot_exc).__name__}: {screenshot_exc}"
                )
        _atomic_json(output / "readiness_failure.json", evidence)
        raise


def _dummy_geometry(app: str) -> dict[str, tuple[int, int]]:
    common = {
        "editor": (820, 520),
        "cell": (640, 420),
        "source": (500, 300),
        "destination": (500, 420),
        "decoy": (500, 360),
        "moved": (500, 300),
        "nav": (260, 140),
        "decoy_nav": (460, 140),
        "toggle": (340, 760),
        "decoy_toggle": (340, 820),
        "scroll_surface": (960, 540),
    }
    return common


def _compiled_actions(fixture: Any, *, near_miss: bool) -> dict[str, list[Any]]:
    trajectory = build_trajectory(fixture, near_miss=near_miss)
    geometry = _dummy_geometry(fixture.app)
    native = [compile_native(turn, geometry) for turn in trajectory.turns]
    cursor = (960, 540)
    compact: list[str] = []
    for turn in trajectory.turns:
        action, cursor = compile_compact(turn, geometry, cursor)
        compact.append(action)
    return {ARMS[0]: native, ARMS[1]: compact}


def run_build_replay(split: str) -> dict[str, Any]:
    assert_collectable_split(split)
    manifest = load_manifest(split)
    rows: list[dict[str, Any]] = []
    for fixture in manifest.fixtures:
        first = initial_state(fixture)
        first_signature = reset_signature(fixture, first)
        initial_oracle = evaluate_in_fresh_process(fixture, first)
        if initial_oracle.MOUSE_SOLVED or initial_oracle.oracle_status != "ok":
            raise RuntimeError(f"{fixture.id}: reset state was not a valid negative")
        near_state = scripted_state(fixture, near_miss=True)
        near_oracle = evaluate_in_fresh_process(fixture, near_state)
        if near_oracle.MOUSE_SOLVED or near_oracle.oracle_status != "ok":
            raise RuntimeError(f"{fixture.id}: scripted near miss was accepted")
        second = initial_state(fixture)
        second_signature = reset_signature(fixture, second)
        if first_signature != second_signature:
            raise RuntimeError(f"{fixture.id}: deterministic reset signature changed")
        gold_state = scripted_state(fixture, near_miss=False)
        gold_oracle = evaluate_in_fresh_process(fixture, gold_state)
        if not gold_oracle.MOUSE_SOLVED or gold_oracle.oracle_status != "ok":
            raise RuntimeError(f"{fixture.id}: scripted gold was rejected")
        rows.append(
            {
                "fixture_id": fixture.id,
                "app": fixture.app,
                "fixture_sha256": fixture.fixture_sha256,
                "horizon": fixture.horizon,
                "semantic_steps": fixture.semantic_steps,
                "reset_signature": first_signature,
                "initial_oracle": asdict(initial_oracle),
                "near_miss_oracle": asdict(near_oracle),
                "gold_oracle": asdict(gold_oracle),
                "gold_actions": _compiled_actions(fixture, near_miss=False),
                "near_miss_actions": _compiled_actions(fixture, near_miss=True),
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "mode": "build",
        "split": split,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "sealed_eval_executed": False,
        "rows": rows,
    }


def _execute_native(transport: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    executor = NativeAbsoluteExecutor(transport)
    results = []
    for operation in payload["operations"]:
        action = dict(operation)
        action["action"] = {
            "click": "left_click",
            "key_chord": "key",
        }.get(action["action"], action["action"])
        results.append(asdict(executor.execute(action)))
    return results


def _execute_vm_trajectory(
    transport: Any,
    fixture: Any,
    geometry: dict[str, tuple[int, int]],
    *,
    arm: str,
    near_miss: bool,
    frame_dir: Path | None = None,
) -> list[dict[str, Any]]:
    trajectory = build_trajectory(fixture, near_miss=near_miss)
    journal: list[dict[str, Any]] = []
    cursor = transport.cursor_position()
    for index, turn in enumerate(trajectory.turns):
        frame: dict[str, Any] | None = None
        if frame_dir is not None:
            frame_dir.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(
                transport.base_url + "/screenshot", timeout=15
            ) as response:
                payload_bytes = response.read()
            if not payload_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("VM screenshot endpoint did not return PNG")
            frame_path = frame_dir / f"turn_{index:02d}.png"
            frame_path.write_bytes(payload_bytes)
            frame = {
                "path": str(frame_path),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "bytes": len(payload_bytes),
            }
        if arm == ARMS[0]:
            payload = compile_native(turn, geometry)
            dispatched = _execute_native(transport, payload)
        else:
            payload, cursor = compile_compact(turn, geometry, cursor)
            dispatched = [asdict(CompactRawExecutor(transport).execute(payload))]
        journal.append(
            {
                "turn": index,
                "semantic_step": turn.semantic_step,
                "screenshot": frame,
                "action": payload,
                "dispatch": dispatched,
            }
        )
        time.sleep(0.35)
        if fixture.app == "chrome" and turn.semantic_step == 2:
            refreshed_state = probe_state(transport, fixture)
            geometry.update(probe_geometry(transport, fixture, refreshed_state))
    return journal


def run_vm_replay(
    split: str,
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider: Path,
    fixture_id: str | None = None,
) -> dict[str, Any]:
    assert_collectable_split(split)
    manifest = load_manifest(split)
    rows: list[dict[str, Any]] = []
    with KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider,
        vm_log_dir=output / "vm_logs",
        smp=int(os.environ.get("OSWORLD_VM_SMP", "4")),
        memory=os.environ.get("OSWORLD_VM_MEM", "8G"),
    ) as session:
        fixtures = manifest.fixtures
        if fixture_id is not None:
            fixtures = (manifest.by_id(fixture_id),)
        for fixture in fixtures:
            for arm in ARMS:
                transport, first = _reset_with_readiness_evidence(
                    session, fixture, output
                )
                initial_oracle = evaluate_in_fresh_process(fixture, first.state)
                if initial_oracle.MOUSE_SOLVED or initial_oracle.oracle_status != "ok":
                    raise RuntimeError(f"{fixture.id}/{arm}: invalid initial state")
                near_journal = _execute_vm_trajectory(
                    transport,
                    fixture,
                    first.geometry,
                    arm=arm,
                    near_miss=True,
                )
                near_state = probe_state(transport, fixture)
                near_oracle = evaluate_in_fresh_process(fixture, near_state)
                if near_oracle.MOUSE_SOLVED or near_oracle.oracle_status != "ok":
                    _atomic_json(
                        output / "failure_context.json",
                        {
                            "fixture_id": fixture.id,
                            "arm": arm,
                            "stage": "near_miss_oracle",
                            "state": near_state,
                            "oracle": asdict(near_oracle),
                            "journal": near_journal,
                        },
                    )
                    raise RuntimeError(f"{fixture.id}/{arm}: near miss accepted")
                transport, second = _reset_with_readiness_evidence(
                    session, fixture, output
                )
                if first.reset_signature != second.reset_signature:
                    raise RuntimeError(f"{fixture.id}/{arm}: reset signature changed")
                gold_journal = _execute_vm_trajectory(
                    transport,
                    fixture,
                    second.geometry,
                    arm=arm,
                    near_miss=False,
                    frame_dir=(output / "frames" / fixture.id / arm / "gold"),
                )
                final_state = probe_state(transport, fixture)
                gold_oracle = evaluate_in_fresh_process(fixture, final_state)
                if not gold_oracle.MOUSE_SOLVED or gold_oracle.oracle_status != "ok":
                    _atomic_json(
                        output / "failure_context.json",
                        {
                            "fixture_id": fixture.id,
                            "arm": arm,
                            "stage": "gold_oracle",
                            "reset_signature": second.reset_signature,
                            "state": final_state,
                            "oracle": asdict(gold_oracle),
                            "journal": gold_journal,
                        },
                    )
                    raise RuntimeError(f"{fixture.id}/{arm}: scripted gold rejected")
                rows.append(
                    {
                        "fixture_id": fixture.id,
                        "fixture_sha256": fixture.fixture_sha256,
                        "app": fixture.app,
                        "arm": arm,
                        "horizon": fixture.horizon,
                        "semantic_steps": fixture.semantic_steps,
                        "reset_signature": first.reset_signature,
                        "reset_a_readiness": first.readiness,
                        "reset_b_readiness": second.readiness,
                        "initial_oracle": asdict(initial_oracle),
                        "near_miss_journal": near_journal,
                        "near_miss_oracle": asdict(near_oracle),
                        "gold_journal": gold_journal,
                        "gold_oracle": asdict(gold_oracle),
                    }
                )
                _atomic_json(
                    output / "progress.json",
                    {
                        "schema_version": 1,
                        "status": "running",
                        "split": split,
                        "sealed_eval_executed": False,
                        "completed_cells": len(rows),
                        "rows": rows,
                    },
                )
    return {
        "schema_version": 1,
        "status": "passed",
        "mode": "vm",
        "split": split,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "sealed_eval_executed": False,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "vm"), required=True)
    parser.add_argument("--split", choices=("train", "development", "sealed_eval"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--fixture-id")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "replay.json"
    marker.unlink(missing_ok=True)
    try:
        payload = (
            run_build_replay(args.split)
            if args.mode == "build"
            else run_vm_replay(
                args.split,
                output=args.output,
                qcow=args.qcow,
                qemu=args.qemu,
                provider=args.provider,
                fixture_id=args.fixture_id,
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
            "split": args.split,
            "sealed_eval_executed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, AppReadinessError):
            failure["readiness"] = {
                "fixture_id": exc.fixture_id,
                "failed_phase": exc.failed_phase,
                "evidence": exc.evidence,
            }
        _atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
