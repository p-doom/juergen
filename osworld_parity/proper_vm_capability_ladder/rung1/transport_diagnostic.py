from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .fixtures import Fixture, FixtureManifest, load_manifest
from .selfcheck import (
    _assert_dispatch_journal,
    _atomic_json,
    _execute,
    _validate_loaded_geometry,
)
from .server import FixtureHttpServer
from .trajectory import Arm, build_trajectory
from .vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    KvmFixtureSession,
    sha256_file,
)


SPEC_PATH = Path(__file__).with_name("transport_diagnostic_spec.json")
FIXTURE_ID = "r1a-click-dev-1101"
FIXTURE_SHA256 = "0124b5dab062e69ed83c37f9b91396b152b1f27a6cd3b9de72a0f9fa18ff5c0e"
MANIFEST_PAYLOAD_SHA256 = (
    "5d4ea3ab33c084f1a5de1b716429c242a97452416f5b74efc3654b7d4b338097"
)
PAIR_COUNT = 5
ARM_ORDER: tuple[Arm, ...] = (
    "native_absolute_control",
    "compact_raw_phaseb",
)
SEMANTIC_KINDS = ("move_to", "mouse_down", "mouse_up")
LOWERED_KINDS = ("click",)
BROWSER_SEQUENCE = ("pointerdown", "pointerup", "click")
X_EVENT_SEQUENCE = ("mouse_down", "mouse_up")
CLICK_CALL = "pyautogui.click(clicks=1, interval=0.05)"


class TransportDiagnosticError(RuntimeError):
    pass


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_transport_diagnostic_spec(
    path: Path = SPEC_PATH,
) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    try:
        spec = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise TransportDiagnosticError("transport diagnostic spec is invalid JSON") from exc
    if not isinstance(spec, dict):
        raise TransportDiagnosticError("transport diagnostic spec must be an object")
    expected = {
        "schema_version": 1,
        "suite": "rung1_shared_click_transport_diagnostic",
        "status": "authorized_narrow_diagnostic",
        "manifest_payload_sha256": MANIFEST_PAYLOAD_SHA256,
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "pair_count": PAIR_COUNT,
        "arm_order": list(ARM_ORDER),
        "reset_before_every_trial": True,
        "stop_on_first_mismatch": True,
        "required_browser_sequence": list(BROWSER_SEQUENCE),
        "required_semantic_operations": list(SEMANTIC_KINDS),
        "required_lowered_operations": list(LOWERED_KINDS),
        "required_backend_primitive": CLICK_CALL,
        "required_x_event_sync": list(X_EVENT_SEQUENCE),
        "required_final_pointer_mask": 0,
        "dispatches_per_trial": 1,
        "retry_count": 0,
        "oracle_conditioned_dispatch": False,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }
    if spec != expected:
        differing = sorted(
            key
            for key in set(spec) | set(expected)
            if spec.get(key) != expected.get(key)
        )
        raise TransportDiagnosticError(
            f"transport diagnostic spec drifted in fields: {differing}"
        )
    return spec, hashlib.sha256(raw_bytes).hexdigest()


def _select_fixture(manifest: FixtureManifest, spec: dict[str, Any]) -> Fixture:
    if manifest.manifest_payload_sha256 != spec["manifest_payload_sha256"]:
        raise TransportDiagnosticError("frozen fixture manifest seal drifted")
    fixture = manifest.by_id(str(spec["fixture_id"]))
    if (
        fixture.id != FIXTURE_ID
        or fixture.fixture_sha256 != spec["fixture_sha256"]
        or fixture.split != "development"
        or fixture.template != "click"
    ):
        raise TransportDiagnosticError("fixed development click fixture contract drifted")
    return fixture


def _semantic_contract(
    dispatch: list[dict[str, Any]], endpoint: tuple[int, int]
) -> list[dict[str, Any]]:
    if len(dispatch) != 1:
        raise TransportDiagnosticError(
            f"trial dispatched {len(dispatch)} actions instead of exactly one"
        )
    record = dispatch[0]
    if record.get("parse_status") != "ok" or record.get("executor_dispatch_status") != "ok":
        raise TransportDiagnosticError(f"adapter dispatch failed: {record}")
    operations = record.get("operations")
    expected = [
        {"kind": "move_to", "args": [endpoint[0], endpoint[1]]},
        {"kind": "mouse_down", "args": ["left"]},
        {"kind": "mouse_up", "args": ["left"]},
    ]
    if operations != expected:
        raise TransportDiagnosticError(
            f"semantic operation contract mismatch: {operations} != {expected}"
        )
    return operations


def _atomic_contract(journal: dict[str, Any]) -> dict[str, Any]:
    states = journal.get("atomic_action_states")
    if not isinstance(states, list) or len(states) != 1:
        raise TransportDiagnosticError("trial did not produce exactly one atomic action state")
    state = states[0]
    if not isinstance(state, dict):
        raise TransportDiagnosticError("atomic action state is not an object")
    required = {
        "ok": True,
        "pointer_button_mask": 0,
        "observed_pointer_button_mask": 0,
        "expected_pointer_button_mask": 0,
        "guest_process_count": 1,
        "cleanup_attempted": False,
        "error": None,
    }
    mismatches = {
        key: {"observed": state.get(key), "expected": value}
        for key, value in required.items()
        if state.get(key) != value
    }
    if mismatches:
        raise TransportDiagnosticError(f"atomic state contract mismatch: {mismatches}")

    primitives = state.get("backend_primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise TransportDiagnosticError("click did not lower to exactly one backend primitive")
    primitive = primitives[0]
    if not isinstance(primitive, dict) or {
        "kind": primitive.get("kind"),
        "button": primitive.get("button"),
        "call": primitive.get("call"),
        "x11_per_event_sync_hooked": primitive.get("x11_per_event_sync_hooked"),
    } != {
        "kind": "click",
        "button": "left",
        "call": CLICK_CALL,
        "x11_per_event_sync_hooked": True,
    }:
        raise TransportDiagnosticError(f"lowered click primitive mismatch: {primitive}")

    sync = state.get("x_event_sync_evidence")
    if not isinstance(sync, list) or [item.get("event") for item in sync] != list(
        X_EVENT_SEQUENCE
    ):
        raise TransportDiagnosticError(f"X event sync sequence mismatch: {sync}")
    for item in sync:
        if (
            not isinstance(item, dict)
            or item.get("flush") is not True
            or item.get("sync") is not True
            or "x11" not in str(item.get("backend", "")).lower()
        ):
            raise TransportDiagnosticError(f"unsupported X event sync evidence: {item}")
    return {
        "lowered_operations": [str(primitive["kind"])],
        "backend_primitives": primitives,
        "x_event_sync_evidence": sync,
        "final_pointer_button_mask": int(state["pointer_button_mask"]),
    }


def _browser_contract(
    acknowledgement: dict[str, Any], endpoint: tuple[int, int]
) -> dict[str, Any]:
    events = acknowledgement.get("events")
    if not isinstance(events, list):
        raise TransportDiagnosticError("browser acknowledgement lacks events")
    causal: list[tuple[str, dict[str, Any]]] = []
    for event in events:
        if not isinstance(event, dict):
            raise TransportDiagnosticError("browser event is not an object")
        label = ""
        if event.get("kind") == "pointer" and event.get("event") in {
            "pointerdown",
            "pointerup",
        }:
            label = str(event["event"])
        elif event.get("kind") == "click":
            label = "click"
        if label:
            causal.append((label, event))
    labels = [label for label, _ in causal]
    if labels != list(BROWSER_SEQUENCE):
        raise TransportDiagnosticError(f"browser causal sequence mismatch: {labels}")
    sequences = [int(event.get("client_sequence", -1)) for _, event in causal]
    if any(value <= 0 for value in sequences) or sequences != sorted(set(sequences)):
        raise TransportDiagnosticError(
            f"browser client sequences are not strictly causal: {sequences}"
        )
    down = causal[0][1]
    up = causal[1][1]
    for label, event, buttons in (("pointerdown", down, 1), ("pointerup", up, 0)):
        if (
            int(event.get("button", -1)) != 0
            or int(event.get("buttons", -1)) != buttons
            or event.get("hit_id") != "target"
            or [int(event.get("screen_x", -1)), int(event.get("screen_y", -1))]
            != [endpoint[0], endpoint[1]]
        ):
            raise TransportDiagnosticError(f"{label} hit contract mismatch: {event}")
    if acknowledgement.get("pointer_buttons") != 0:
        raise TransportDiagnosticError("browser acknowledgement ended with a held button")
    return {
        "sequence": labels,
        "client_sequences": sequences,
        "pointer_hits": [
            {
                "event": label,
                "screen": [int(event["screen_x"]), int(event["screen_y"])],
                "buttons": int(event["buttons"]),
                "hit_id": str(event["hit_id"]),
            }
            for label, event in causal[:2]
        ],
        "state_event": "change/click",
        "final_pointer_buttons": 0,
    }


def _run_trial(
    *,
    session: KvmFixtureSession,
    server: FixtureHttpServer,
    fixture: Fixture,
    pair_index: int,
    trial_index: int,
    arm: Arm,
) -> dict[str, Any]:
    transport = session.reset_to_ready()
    initial = session.launch_fixture(server, fixture)
    _validate_loaded_geometry(fixture, initial, transport.screen_size())
    if initial.get("current") != {"checked": False, "decoy_checked": False}:
        raise TransportDiagnosticError(f"trial {trial_index} reset state was not clean")
    baseline = transport.cursor_position()
    trajectory = build_trajectory(
        fixture, initial, arm=arm, cursor=baseline, near_miss=False
    )
    if len(trajectory.actions) != 1 or trajectory.expected_endpoint is None:
        raise TransportDiagnosticError("fixed click trajectory contract drifted")
    after_sequence = int(server.store.snapshot(fixture.id)["last_client_sequence"])
    dispatch, journal = _execute(arm, transport, trajectory)
    _assert_dispatch_journal(fixture, arm, "transport diagnostic", journal)
    endpoint = trajectory.expected_endpoint
    semantic = _semantic_contract(dispatch, endpoint)
    atomic = _atomic_contract(journal)
    acknowledgement = server.store.wait_for_browser_quiescence(
        fixture.id,
        after_sequence=after_sequence,
        required_kinds=("click",),
        require_pointer_down=True,
        require_pointer_up=True,
        expected_pointer_buttons=0,
    )
    browser = _browser_contract(acknowledgement, endpoint)
    final_state = server.store.snapshot(fixture.id)
    if final_state.get("current") != {"checked": True, "decoy_checked": False}:
        raise TransportDiagnosticError(
            f"click state acknowledgement mismatch: {final_state.get('current')}"
        )
    return {
        "pair_index": pair_index,
        "trial_index": trial_index,
        "arm": arm,
        "status": "passed",
        "reset_snapshot": READY_SNAPSHOT,
        "reset_before_trial": True,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "dispatch_count": 1,
        "retry_count": 0,
        "oracle_invocation_count": 0,
        "oracle_conditioned_dispatch": False,
        "baseline": list(baseline),
        "endpoint": list(endpoint),
        "semantic_operations": semantic,
        **atomic,
        "browser_contract": browser,
        "final_state": final_state["current"],
    }


def _matching_contract(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": trial["endpoint"],
        "semantic_operations": trial["semantic_operations"],
        "lowered_operations": trial["lowered_operations"],
        "backend_primitives": trial["backend_primitives"],
        "x_event_sync_evidence": trial["x_event_sync_evidence"],
        "browser_sequence": trial["browser_contract"]["sequence"],
        "pointer_hits": trial["browser_contract"]["pointer_hits"],
        "final_pointer_button_mask": trial["final_pointer_button_mask"],
        "final_state": trial["final_state"],
    }


def _checkpoint(
    output: Path,
    *,
    trials: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    active_trial: dict[str, Any] | None,
    stage: str,
) -> None:
    _atomic_json(
        output / "transport_diagnostic_progress.json",
        {
            "status": "running",
            "expected_pair_count": PAIR_COUNT,
            "expected_trial_count": PAIR_COUNT * len(ARM_ORDER),
            "completed_pair_count": len(pairs),
            "completed_trial_count": len(trials),
            "stop_on_first_mismatch": True,
            "stage": stage,
            "active_trial": active_trial,
            "pairs": pairs,
            "trials": trials,
        },
    )


def validate_transport_diagnostic() -> dict[str, Any]:
    spec, spec_sha256 = load_transport_diagnostic_spec()
    manifest = load_manifest()
    fixture = _select_fixture(manifest, spec)
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": spec["suite"],
        "mode": "validate",
        "spec_sha256": spec_sha256,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "pair_count": PAIR_COUNT,
        "trial_count": PAIR_COUNT * len(ARM_ORDER),
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }


def run_vm_transport_diagnostic(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider_path: Path,
    expected_provider_sha256: str | None,
) -> dict[str, Any]:
    spec, spec_sha256 = load_transport_diagnostic_spec()
    manifest = load_manifest()
    fixture = _select_fixture(manifest, spec)
    provider_sha256 = sha256_file(provider_path)
    if expected_provider_sha256 and provider_sha256 != expected_provider_sha256:
        raise TransportDiagnosticError(
            f"KVM provider hash mismatch: {provider_sha256} != {expected_provider_sha256}"
        )
    trials: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    vm_log_dir = output / "vm_logs"
    _checkpoint(
        output, trials=trials, pairs=pairs, active_trial=None, stage="starting_vm"
    )
    with FixtureHttpServer(manifest) as server, KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider_path,
        vm_log_dir=vm_log_dir,
    ) as session:
        for pair_index in range(1, PAIR_COUNT + 1):
            pair_trials: list[dict[str, Any]] = []
            for arm in ARM_ORDER:
                trial_index = len(trials) + 1
                active = {
                    "pair_index": pair_index,
                    "trial_index": trial_index,
                    "arm": arm,
                    "fixture_id": fixture.id,
                }
                _checkpoint(
                    output,
                    trials=trials,
                    pairs=pairs,
                    active_trial=active,
                    stage="resetting_trial",
                )
                trial = _run_trial(
                    session=session,
                    server=server,
                    fixture=fixture,
                    pair_index=pair_index,
                    trial_index=trial_index,
                    arm=arm,
                )
                pair_trials.append(trial)
                trials.append(trial)
                _checkpoint(
                    output,
                    trials=trials,
                    pairs=pairs,
                    active_trial=active,
                    stage="trial_passed",
                )
            native_contract = _matching_contract(pair_trials[0])
            compact_contract = _matching_contract(pair_trials[1])
            if native_contract != compact_contract:
                raise TransportDiagnosticError(
                    f"pair {pair_index} native/compact contract mismatch: "
                    f"native={native_contract}, compact={compact_contract}"
                )
            pair = {
                "pair_index": pair_index,
                "status": "passed",
                "arm_order": list(ARM_ORDER),
                "trial_indices": [item["trial_index"] for item in pair_trials],
                "matched_contract_sha256": _payload_sha256(native_contract),
                "matched_contract": native_contract,
            }
            pairs.append(pair)
            _checkpoint(
                output,
                trials=trials,
                pairs=pairs,
                active_trial=None,
                stage="pair_passed",
            )
    if len(trials) != 10 or len(pairs) != 5:
        raise TransportDiagnosticError("diagnostic ended before its fixed 10-trial horizon")
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": spec["suite"],
        "mode": "vm",
        "spec_sha256": spec_sha256,
        "snapshot_name": READY_SNAPSHOT,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "pair_count": len(pairs),
        "trial_count": len(trials),
        "passed_trial_count": len(trials),
        "reset_count": len(trials),
        "dispatch_count": len(trials),
        "retry_count": 0,
        "oracle_invocation_count": 0,
        "oracle_conditioned_dispatch": False,
        "stop_on_first_mismatch": True,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "provider": {
            "path": str(provider_path.resolve()),
            "sha256": provider_sha256,
        },
        "qcow": {"path": str(qcow.resolve()), "size": qcow.stat().st_size},
        "qemu": str(qemu.resolve()),
        "pairs": pairs,
        "trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "vm"), required=True)
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
    marker = args.output / "transport_diagnostic.json"
    marker.unlink(missing_ok=True)
    try:
        payload = (
            validate_transport_diagnostic()
            if args.mode == "validate"
            else run_vm_transport_diagnostic(
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
    except Exception as exc:
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
