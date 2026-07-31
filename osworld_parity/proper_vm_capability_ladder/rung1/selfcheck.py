from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .executor import CompactRawExecutor, NativeAbsoluteExecutor
from .fixtures import Fixture, FixtureManifest, load_manifest
from .oracle import evaluate_in_fresh_process
from .server import FixtureHttpServer, render_fixture_html
from .trajectory import Arm, GoldTrajectory, build_trajectory
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


def _checkpoint_progress(
    output: Path,
    *,
    cells: list[dict[str, Any]],
    expected_cell_count: int,
    active_cell: dict[str, Any] | None,
    stage: str,
) -> None:
    if active_cell is not None:
        active_cell["journal_stage"] = stage
    _atomic_json(
        output / "progress.json",
        {
            "status": "running",
            "completed_cell_count": len(cells),
            "expected_cell_count": expected_cell_count,
            "active_cell": active_cell,
            "cells": cells,
        },
    )


def _operations(values: tuple[Operation, ...]) -> list[dict[str, Any]]:
    return [{"kind": item.kind, "args": list(item.args)} for item in values]


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reset_component_report(
    state: dict[str, Any],
    *,
    cursor: tuple[int, int],
    pointer_buttons: int,
) -> dict[str, Any]:
    geometry = state.get("geometry")
    if not isinstance(geometry, dict):
        raise SelfcheckError("reset state lacks geometry")
    payloads = {
        "fixture": {
            "fixture_id": state["fixture_id"],
            "fixture_sha256": state["fixture_sha256"],
        },
        "logical_state": {
            "ready": state["ready"],
            "current": state["current"],
        },
        "cursor_button": {
            "cursor": [int(cursor[0]), int(cursor[1])],
            "pointer_buttons": int(pointer_buttons),
        },
        "window_geometry": geometry.get("window"),
        "dom_geometry": {
            key: value for key, value in geometry.items() if key != "window"
        },
    }
    hashes = {name: _payload_sha256(value) for name, value in payloads.items()}
    return {
        "payloads": payloads,
        "hashes": hashes,
        "aggregate_sha256": _payload_sha256(hashes),
    }


def _reset_component_diff(
    reset_a: dict[str, Any], reset_b: dict[str, Any]
) -> dict[str, Any]:
    names = tuple(reset_a["payloads"])
    if set(names) != set(reset_b["payloads"]):
        raise SelfcheckError("reset component names changed")
    components = {}
    for name in names:
        components[name] = {
            "equal": reset_a["payloads"][name] == reset_b["payloads"][name],
            "reset_a_sha256": reset_a["hashes"][name],
            "reset_b_sha256": reset_b["hashes"][name],
            "reset_a": reset_a["payloads"][name],
            "reset_b": reset_b["payloads"][name],
        }
    return {
        "all_equal": all(item["equal"] for item in components.values()),
        "components": components,
        "differing_components": [
            name for name, item in components.items() if not item["equal"]
        ],
    }


def _execute(
    arm: Arm, transport: Any, trajectory: GoldTrajectory
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    executor = (
        NativeAbsoluteExecutor(transport)
        if arm == "native_absolute_control"
        else CompactRawExecutor(transport)
    )
    dispatch_cursor_before = transport.cursor_position()
    expected = trajectory.expected_endpoint
    baseline_matches = dispatch_cursor_before == trajectory.observed_cursor_baseline
    if arm == "compact_raw_phaseb" and not baseline_matches:
        return [], {
            "dispatch_status": "blocked_baseline_drift",
            "planned_observed_baseline": list(trajectory.observed_cursor_baseline),
            "dispatch_cursor_before": list(dispatch_cursor_before),
            "baseline_matches": False,
            "expected_endpoint": list(expected) if expected is not None else None,
            "final_cursor": list(dispatch_cursor_before),
            "endpoint_matches": expected is None or dispatch_cursor_before == expected,
            "actions": [],
        }
    records: list[dict[str, Any]] = []
    action_cursors: list[dict[str, Any]] = []
    atomic_states: list[dict[str, Any]] = []
    for index, action in enumerate(trajectory.actions):
        cursor_before = transport.cursor_position()
        result = executor.execute(action)  # type: ignore[arg-type]
        cursor_after = transport.cursor_position()
        record = {
            "parse_status": result.parse_status,
            "executor_dispatch_status": result.executor_dispatch_status,
            "action_class": result.action_class,
            "operations": _operations(result.operations),
            "atomic_state": result.atomic_state,
        }
        records.append(record)
        if result.atomic_state is not None:
            atomic_states.append(result.atomic_state)
        action_cursors.append(
            {
                "action_index": index,
                "cursor_before": list(cursor_before),
                "cursor_after": list(cursor_after),
                "guest_cursor_after": (
                    result.atomic_state["cursor"]
                    if result.atomic_state is not None
                    else None
                ),
                "guest_pointer_button_mask": (
                    result.atomic_state["pointer_button_mask"]
                    if result.atomic_state is not None
                    else None
                ),
            }
        )
        if result.executor_dispatch_status != "ok":
            break
    final_cursor = transport.cursor_position()
    atomic_errors = [state["error"] for state in atomic_states if not state["ok"]]
    journal = {
        "dispatch_status": "dispatched",
        "planned_observed_baseline": list(trajectory.observed_cursor_baseline),
        "dispatch_cursor_before": list(dispatch_cursor_before),
        "baseline_matches": baseline_matches,
        "expected_endpoint": list(expected) if expected is not None else None,
        "final_cursor": list(final_cursor),
        "endpoint_matches": expected is None or final_cursor == expected,
        "actions": action_cursors,
        "planned_action_count": len(trajectory.actions),
        "completed_action_count": len(records),
        "atomic_guest_process_count": sum(
            int(state["guest_process_count"]) for state in atomic_states
        ),
        "atomic_action_states": atomic_states,
        "final_pointer_button_mask": (
            atomic_states[-1]["pointer_button_mask"] if atomic_states else None
        ),
        "atomic_errors": atomic_errors,
    }
    return records, journal


def _assert_dispatch_journal(
    fixture: Fixture,
    arm: Arm,
    stage: str,
    journal: dict[str, Any],
    *,
    required_pointer_button_mask: int = 0,
) -> None:
    if arm == "compact_raw_phaseb" and not journal["baseline_matches"]:
        raise SelfcheckError(
            f"{fixture.id}: {stage} raw delta baseline drifted: "
            f"planned={journal['planned_observed_baseline']} "
            f"dispatch={journal['dispatch_cursor_before']}"
        )
    if arm == "compact_raw_phaseb":
        if journal["atomic_guest_process_count"] != journal["completed_action_count"]:
            raise SelfcheckError(
                f"{fixture.id}: {stage} compact actions were not one-process atomic"
            )
        if journal["atomic_errors"]:
            raise SelfcheckError(
                f"{fixture.id}: {stage} atomic guest action failed: "
                f"{journal['atomic_errors']}"
            )
        if journal["completed_action_count"] != journal["planned_action_count"]:
            raise SelfcheckError(
                f"{fixture.id}: {stage} compact trajectory stopped early"
            )
        if journal["final_pointer_button_mask"] != required_pointer_button_mask:
            raise SelfcheckError(
                f"{fixture.id}: {stage} pointer button mask was "
                f"{journal['final_pointer_button_mask']}, expected "
                f"{required_pointer_button_mask}"
            )
    if not journal["endpoint_matches"]:
        raise SelfcheckError(
            f"{fixture.id}: {stage} cursor missed expected endpoint: "
            f"expected={journal['expected_endpoint']} final={journal['final_cursor']}"
        )


def _assert_negative(fixture: Fixture, state: dict[str, Any], stage: str) -> dict[str, Any]:
    result = evaluate_in_fresh_process(fixture, state)
    if result.oracle_status != "ok" or result.MOUSE_SOLVED:
        raise SelfcheckError(f"{fixture.id}: {stage} oracle was not a clean negative")
    return asdict(result)


def _assert_positive(fixture: Fixture, state: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_in_fresh_process(fixture, state)
    if result.oracle_status != "ok" or not result.MOUSE_SOLVED:
        raise SelfcheckError(
            f"{fixture.id}: gold oracle rejected state: {result.reason}; "
            f"state={json.dumps(state, ensure_ascii=False, sort_keys=True)}"
        )
    return asdict(result)


def _held_button_action(arm: Arm) -> dict[str, Any] | str:
    if arm == "native_absolute_control":
        return {"action": "mouse_down", "button": "left"}
    return "0 0 0 ; +LMB"


def _wait_for_trajectory_ack(
    server: FixtureHttpServer,
    fixture: Fixture,
    *,
    after_sequence: int,
) -> dict[str, Any]:
    state_kind = {
        "click": "click",
        "focus_type": "text",
        "scroll": "scroll",
        "drag": "drag",
    }[fixture.template]
    return server.store.wait_for_browser_quiescence(
        fixture.id,
        after_sequence=after_sequence,
        required_kinds=(state_kind,),
        require_pointer_up=fixture.template != "scroll",
        expected_pointer_buttons=0,
    )


def _card_size(fixture: Fixture) -> tuple[int, int] | None:
    if fixture.template == "click":
        return 360, 180
    if fixture.template == "focus_type":
        return 520, 160
    if fixture.template == "drag":
        return int(fixture.params["width"]) + 70, 180
    return None


def _validate_manifest_bounds(
    manifest: FixtureManifest,
    measured: dict[str, Any],
    screen_size: tuple[int, int],
) -> dict[str, Any]:
    """Validate all 40 sealed rows against one measured Chrome viewport.

    Evaluation rows are checked arithmetically from their sealed design
    coordinates; they are never loaded in the browser or sent to an oracle.
    """
    required = (
        "screen_x",
        "screen_y",
        "screen_width",
        "screen_height",
        "inner_width",
        "inner_height",
        "outer_width",
        "outer_height",
        "chrome_top",
    )
    if any(key not in measured for key in required):
        raise SelfcheckError(f"measured Chrome geometry is incomplete: {measured}")
    values = {key: int(measured[key]) for key in required}
    sw, sh = screen_size
    if (values["screen_width"], values["screen_height"]) != (sw, sh):
        raise SelfcheckError(
            f"browser/agent screen mismatch: {values['screen_width']}x{values['screen_height']} "
            f"!= {sw}x{sh}"
        )
    iw, ih = values["inner_width"], values["inner_height"]
    content_left = values["screen_x"]
    content_top = values["screen_y"] + values["chrome_top"]
    if iw <= 0 or ih <= 0 or content_left < 0 or content_top < 0:
        raise SelfcheckError(f"invalid measured Chrome viewport: {values}")
    if content_left + iw > sw or content_top + ih > sh:
        raise SelfcheckError(f"Chrome viewport exceeds agent screen: {values}")
    rows: dict[str, Any] = {}
    transformed_by_template: dict[str, list[tuple[float, float]]] = {}
    for fixture in manifest.fixtures:
        size = _card_size(fixture)
        if size is None:
            rows[fixture.id] = {"kind": "scroll", "viewport_bounded": True}
            continue
        width, height = size
        if iw < width + 48 or ih < height + 128:
            raise SelfcheckError(
                f"viewport too small for {fixture.id}: {iw}x{ih}, card {width}x{height}"
            )
        requested_x = int(fixture.params["left"]) * iw / 1920.0
        requested_y = int(fixture.params["top"]) * ih / 1080.0
        x = max(24.0, min(requested_x, iw - width - 24.0))
        y = max(104.0, min(requested_y, ih - height - 24.0))
        bounds = (
            content_left + x,
            content_top + y,
            content_left + x + width,
            content_top + y + height,
        )
        if not (0 <= bounds[0] < bounds[2] <= sw and 0 <= bounds[1] < bounds[3] <= sh):
            raise SelfcheckError(f"computed card is off screen for {fixture.id}: {bounds}")
        rows[fixture.id] = {
            "kind": fixture.template,
            "design_origin": [int(fixture.params["left"]), int(fixture.params["top"])],
            "requested_viewport_origin": [round(requested_x, 3), round(requested_y, 3)],
            "transformed_viewport_origin": [round(x, 3), round(y, 3)],
            "clamped": [x != requested_x, y != requested_y],
            "computed_card_screen_bounds": [round(value, 3) for value in bounds],
        }
        transformed_by_template.setdefault(fixture.template, []).append(
            (round(x, 3), round(y, 3))
        )
    collision_audit: dict[str, Any] = {}
    for template, origins in transformed_by_template.items():
        counts = Counter(origins)
        unique_count = len(counts)
        max_multiplicity = max(counts.values())
        if unique_count < 8 or max_multiplicity > 2:
            raise SelfcheckError(
                f"scaled placement collapse for {template}: "
                f"unique={unique_count}/{len(origins)}, max multiplicity={max_multiplicity}"
            )
        collision_audit[template] = {
            "row_count": len(origins),
            "unique_origin_count": unique_count,
            "max_origin_multiplicity": max_multiplicity,
        }
    return {
        "screen_size": [sw, sh],
        "window": values,
        "rows": rows,
        "placement_collision_audit": collision_audit,
    }


def _validate_loaded_geometry(
    fixture: Fixture, state: dict[str, Any], screen_size: tuple[int, int]
) -> None:
    geometry = state.get("geometry")
    if not isinstance(geometry, dict):
        raise SelfcheckError(f"{fixture.id}: browser geometry missing")
    names = ("target", "decoy") if fixture.template == "click" else ("target",)
    if fixture.template == "scroll":
        names = ()
    sw, sh = screen_size
    window = geometry.get("window")
    if not isinstance(window, dict):
        raise SelfcheckError(f"{fixture.id}: measured window geometry missing")
    content_left = int(window["screen_x"])
    content_top = int(window["screen_y"]) + int(window["chrome_top"])
    content_right = content_left + int(window["inner_width"])
    content_bottom = content_top + int(window["inner_height"])
    rects: dict[str, tuple[int, int, int, int]] = {}
    for name in names:
        rect = geometry.get(name)
        if not isinstance(rect, dict):
            raise SelfcheckError(f"{fixture.id}: geometry lacks {name}")
        left, top = int(rect["left"]), int(rect["top"])
        right, bottom = int(rect["right"]), int(rect["bottom"])
        center_x, center_y = int(rect["center_x"]), int(rect["center_y"])
        if not (
            0 <= left < right <= sw
            and 0 <= top < bottom <= sh
            and content_left <= left < right <= content_right
            and content_top <= top < bottom <= content_bottom
            and right - left >= 12
            and bottom - top >= 12
            and left <= center_x < right
            and top <= center_y < bottom
        ):
            raise SelfcheckError(f"{fixture.id}: {name} off screen: {rect}")
        rects[name] = (left, top, right, bottom)
    if fixture.template == "click":
        target = rects["target"]
        decoy = rects["decoy"]
        separated = (
            target[2] <= decoy[0]
            or decoy[2] <= target[0]
            or target[3] <= decoy[1]
            or decoy[3] <= target[1]
        )
        if not separated:
            raise SelfcheckError(f"{fixture.id}: target and decoy hitboxes overlap")


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
    manifest_bounds: dict[str, Any] | None = None
    expected_cell_count = len(fixtures) * len(ARMS)

    def checkpoint(active_cell: dict[str, Any] | None, stage: str) -> None:
        _checkpoint_progress(
            output,
            cells=cells,
            expected_cell_count=expected_cell_count,
            active_cell=active_cell,
            stage=stage,
        )

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
                checkpoint(cell, "cell_started")

                # Reset A: deterministic setup and clean negative oracle.
                transport = session.reset_to_ready()
                first = session.launch_fixture(server, fixture)
                screen_size = transport.screen_size()
                _validate_loaded_geometry(fixture, first, screen_size)
                if manifest_bounds is None:
                    window = first["geometry"].get("window")
                    if not isinstance(window, dict):
                        raise SelfcheckError("first fixture did not report measured window geometry")
                    manifest_bounds = _validate_manifest_bounds(manifest, window, screen_size)
                first_cursor = transport.cursor_position()
                cell["reset_oracle"] = _assert_negative(fixture, first, "reset")
                first_buttons = session.probe_pointer_buttons(server, fixture)
                if first_buttons != 0:
                    raise SelfcheckError(f"{fixture.id}: button held after first reset")
                reset_a_components = _reset_component_report(
                    first, cursor=first_cursor, pointer_buttons=first_buttons
                )
                cell["reset_a_snapshot"] = {
                    "browser_state": first,
                    "cursor": list(first_cursor),
                    "pointer_buttons": first_buttons,
                }
                cell["reset_a_components"] = reset_a_components
                checkpoint(cell, "first_reset_verified")

                # Scripted near miss must be rejected.
                near_baseline = transport.cursor_position()
                near = build_trajectory(
                    fixture,
                    first,
                    arm=arm,
                    cursor=near_baseline,
                    near_miss=True,
                )
                near_after_sequence = int(
                    server.store.snapshot(fixture.id)["last_client_sequence"]
                )
                (
                    cell["near_miss_dispatch"],
                    cell["near_miss_cursor_journal"],
                ) = _execute(arm, transport, near)
                checkpoint(cell, "near_miss_dispatched")
                _assert_dispatch_journal(
                    fixture, arm, "near miss", cell["near_miss_cursor_journal"]
                )
                cell["near_miss_browser_ack"] = _wait_for_trajectory_ack(
                    server, fixture, after_sequence=near_after_sequence
                )
                checkpoint(cell, "near_miss_browser_acknowledged")
                cell["near_miss_oracle"] = _assert_negative(
                    fixture, server.store.snapshot(fixture.id), "near miss"
                )
                checkpoint(cell, "near_miss_rejected")

                # Deliberately leave LMB held, then prove the VM snapshot—not a
                # host-side cleanup helper—removes it on the second reset.
                leak_baseline = transport.cursor_position()
                leak_trajectory = GoldTrajectory(
                    arm=arm,
                    actions=(_held_button_action(arm),),
                    observed_cursor_baseline=leak_baseline,
                    expected_endpoint=leak_baseline,
                )
                leak_after_sequence = int(
                    server.store.snapshot(fixture.id)["last_client_sequence"]
                )
                cell["leak_injection"], cell["leak_cursor_journal"] = _execute(
                    arm, transport, leak_trajectory
                )
                checkpoint(cell, "held_button_injected")
                _assert_dispatch_journal(
                    fixture,
                    arm,
                    "held-button injection",
                    cell["leak_cursor_journal"],
                    required_pointer_button_mask=1 << 8,
                )
                cell["leak_browser_ack"] = server.store.wait_for_browser_quiescence(
                    fixture.id,
                    after_sequence=leak_after_sequence,
                    require_pointer_down=True,
                    expected_pointer_buttons=1,
                )
                checkpoint(cell, "held_button_browser_acknowledged")
                if server.store.snapshot(fixture.id)["last_pointer_buttons"] != 1:
                    raise SelfcheckError(f"{fixture.id}: held-button injection was not observed")

                # Reset B: exact fixture state, cursor, scroll/type state, and
                # button state must match Reset A.
                transport = session.reset_to_ready()
                second = session.launch_fixture(server, fixture)
                _validate_loaded_geometry(fixture, second, transport.screen_size())
                second_cursor = transport.cursor_position()
                cell["second_reset_oracle"] = _assert_negative(
                    fixture, second, "second reset"
                )
                second_buttons = session.probe_pointer_buttons(server, fixture)
                if second_buttons != 0:
                    raise SelfcheckError(f"{fixture.id}: button leaked across reset")
                reset_b_components = _reset_component_report(
                    second, cursor=second_cursor, pointer_buttons=second_buttons
                )
                reset_diff = _reset_component_diff(
                    reset_a_components, reset_b_components
                )
                cell["reset_b_snapshot"] = {
                    "browser_state": second,
                    "cursor": list(second_cursor),
                    "pointer_buttons": second_buttons,
                }
                cell["reset_b_components"] = reset_b_components
                cell["reset_component_diff"] = reset_diff
                checkpoint(cell, "reset_comparison_recorded")
                if not reset_diff["all_equal"]:
                    raise SelfcheckError(
                        f"{fixture.id}: exact reset components changed: "
                        f"{reset_diff['differing_components']}; "
                        f"diff={json.dumps(reset_diff, ensure_ascii=False, sort_keys=True)}"
                    )
                cell["reset_leakage"] = {
                    "initial_state_sha256": reset_b_components["aggregate_sha256"],
                    "component_hashes": reset_b_components["hashes"],
                    "cursor": list(second_cursor),
                    "pointer_buttons": second_buttons,
                    "current": second["current"],
                }
                checkpoint(cell, "second_reset_verified")

                # Scripted gold must pass a fresh host oracle process.
                gold_baseline = transport.cursor_position()
                gold = build_trajectory(
                    fixture,
                    second,
                    arm=arm,
                    cursor=gold_baseline,
                )
                gold_after_sequence = int(
                    server.store.snapshot(fixture.id)["last_client_sequence"]
                )
                cell["gold_dispatch"], cell["gold_cursor_journal"] = _execute(
                    arm, transport, gold
                )
                checkpoint(cell, "gold_dispatched")
                _assert_dispatch_journal(
                    fixture, arm, "gold", cell["gold_cursor_journal"]
                )
                cell["gold_browser_ack"] = _wait_for_trajectory_ack(
                    server, fixture, after_sequence=gold_after_sequence
                )
                checkpoint(cell, "gold_browser_acknowledged")
                final_state = server.store.snapshot(fixture.id)
                cell["gold_state_before_oracle"] = final_state
                checkpoint(cell, "gold_state_recorded_before_oracle")
                cell["gold_oracle"] = _assert_positive(fixture, final_state)
                checkpoint(cell, "gold_oracle_passed")
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
                cell["journal_stage"] = "cell_passed"
                cells.append(cell)
                checkpoint(None, "cell_passed")
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
        "expected_selfcheck_cell_count": expected_cell_count,
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "manifest_bounds": manifest_bounds,
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
