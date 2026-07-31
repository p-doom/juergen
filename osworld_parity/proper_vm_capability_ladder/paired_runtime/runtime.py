"""Fail-closed real-VM/model binding for short-task paired pass@k runs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..paired_eval.contracts import (
    APPROVED_CURRICULUM_COMMIT,
    APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
    ACTION_INTERFACES,
    ExecutionReceipt,
    InfrastructureFailure,
    Observation,
    RequestedAction,
    RUNTIME_CONTRACT_SCHEMA,
    SessionStart,
    StateProbe,
    infrastructure_failure_source_receipt,
    resolved_segment_budget_payload,
    sha256_json,
)
from ..rung1.transport import TransportError
from ..rung1.transport import PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND
from ..rung1.vm import KvmFixtureSession, VmHarnessError
from ..rung2_sameapp import replay
from ..rung2_sameapp.curriculum.manifests import load_manifest
from ..rung2_sameapp.curriculum.oracle import reset_signature
from ..rung2_sameapp.curriculum.program import (
    CompiledSegment,
    _resolved_event_count,
    compile_semantic_step,
    record_executed_segment,
)
from ..rung2_sameapp.curriculum.runtime import (
    RuntimeEvidenceLedger,
    RuntimeProbeError,
    bind_repeated_runtime_probes,
    probe_runtime,
    refresh_binding_after_step,
)
from ..rung2_sameapp.fixtures import canonical_json


EXECUTOR_CANDIDATE_COMMIT = "976d3d947a084ae5545d3744f92170c524f6797e"
MODEL_ENDPOINTS_ENV = "PAIRED_MODEL_ENDPOINTS_JSON"
_APP_PROBE_METHODS = {
    "writer": "uno_readonly_state_probe",
    "calc": "uno_readonly_state_probe",
    "files": "filesystem_readonly_state_probe",
    "chrome": "browser_dom_readonly_state_probe",
    "vscode": "editor_readonly_state_probe",
}


def _failure(
    failure_class: str,
    operation: str,
    message: str,
    **evidence: Any,
) -> InfrastructureFailure:
    return InfrastructureFailure(
        failure_class,
        message,
        source_receipt=infrastructure_failure_source_receipt(
            failure_class,
            operation=operation,
            raw_evidence={"event": message, **evidence},
        ),
    )


def _load_endpoints() -> dict[str, dict[str, str]]:
    raw_path = os.environ.get(MODEL_ENDPOINTS_ENV)
    if not raw_path:
        raise RuntimeError(f"{MODEL_ENDPOINTS_ENV} is required")
    path = Path(raw_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load model endpoint binding: {exc}") from exc
    if not isinstance(value, dict) or set(value) != set(ACTION_INTERFACES):
        raise RuntimeError("model endpoint arms do not match the paired evaluator")
    required = {"base_url", "api_key", "served_model", "checkpoint", "checkpoint_sha256"}
    for arm, endpoint in value.items():
        if not isinstance(endpoint, dict) or set(endpoint) != required:
            raise RuntimeError(f"{arm}: model endpoint schema drift")
        if not all(isinstance(endpoint[key], str) and endpoint[key] for key in required):
            raise RuntimeError(f"{arm}: model endpoint binding is incomplete")
    return value


def _read_prompt(arm_name: str) -> str:
    path = Path(__file__).with_name("prompts") / f"{arm_name}.txt"
    return path.read_text(encoding="utf-8")


def _screenshot(transport: Any) -> bytes:
    try:
        with urllib.request.urlopen(transport.base_url + "/screenshot", timeout=20) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise _failure(
            "observation_capture",
            "capture_observation",
            "vm_screenshot_capture_failed",
            error_type=type(exc).__name__,
        ) from exc
    if len(payload) > 32 * 1024 * 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _failure(
            "observation_capture",
            "capture_observation",
            "vm_screenshot_payload_invalid",
            payload_bytes=len(payload),
        )
    return payload


def _content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model response has no assistant content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response content is empty")
    return content.strip()


def _action_line(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 2:
        raise ValueError("raw-preamble response must contain one optional sentence and one action line")
    return lines[-1]


def _native_action(content: str, semantic_step: int) -> dict[str, Any]:
    from ..rung1.executor import parse_compact_raw

    line = _action_line(content)
    raw = parse_compact_raw(line)
    coordinate = [raw.dx, raw.dy]
    operations: list[dict[str, Any]] = []
    elements = list(raw.elements)
    if (
        raw.scroll == 0
        and len(elements) == 2
        and elements[0].kind == elements[1].kind == "event"
        and elements[0].value == elements[1].value == "LMB"
        and elements[0].pressed is True
        and elements[1].pressed is False
    ):
        operations.append({"action": "click", "coordinate": coordinate})
    else:
        if raw.dx or raw.dy or raw.scroll or any(
            element.kind == "event" and element.value in {"LMB", "RMB", "MMB"}
            for element in elements
        ):
            operations.append({"action": "mouse_move", "coordinate": coordinate})
        if raw.scroll:
            operations.append({"action": "scroll", "clicks": raw.scroll})
        for element in elements:
            if element.kind == "type":
                operations.append({"action": "type", "text": element.value})
            elif element.value in {"LMB", "RMB", "MMB"}:
                operations.append(
                    {
                        "action": "mouse_down" if element.pressed else "mouse_up",
                        "button": {"LMB": "left", "RMB": "right", "MMB": "middle"}[
                            element.value
                        ],
                    }
                )
            else:
                raise ValueError("native raw response contains an unsupported key event")
    if not operations:
        raise ValueError("native raw response resolved to no operations")
    return {
        "schema": "native_absolute_sequence_v1",
        "semantic_step": semantic_step,
        "operations": operations,
    }


def _compact_action(content: str, semantic_step: int) -> dict[str, Any]:
    from ..rung1.executor import parse_compact_raw

    line = _action_line(content)
    parse_compact_raw(line)
    return {
        "schema": "compact_raw_phaseb_v1",
        "semantic_step": semantic_step,
        "actions": [line],
    }


def _model_context(
    instruction: str,
    next_semantic_step: int,
    history: tuple[dict[str, Any], ...],
) -> str:
    return (
        f"Task instruction: {instruction}\n"
        f"Next semantic step index: {next_semantic_step}\n"
        "Completed semantic history (JSON): "
        + json.dumps(history, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _expected_cursor_after(
    action_schema: str,
    actions: tuple[Any, ...],
    before: tuple[int, int],
) -> tuple[int, int]:
    cursor = before
    if action_schema == "native_absolute_sequence_v1":
        for action in actions:
            for operation in action["operations"]:
                coordinate = operation.get("coordinate")
                if coordinate is not None and operation.get("action") in {
                    "click",
                    "mouse_down",
                    "mouse_move",
                    "mouse_up",
                }:
                    cursor = (int(round(coordinate[0])), int(round(coordinate[1])))
        return cursor
    from ..rung1.executor import parse_compact_raw

    for action in actions:
        parsed = parse_compact_raw(action)
        cursor = (cursor[0] + parsed.dx, cursor[1] + parsed.dy)
    return cursor


def _active_window(transport: Any, expected_app: str) -> dict[str, Any]:
    command = """
set -euo pipefail
wid="$(xprop -root _NET_ACTIVE_WINDOW | sed -n 's/.*# //p')"
test -n "$wid"; test "$wid" != "0x0"
printf 'WINDOW=%s\\n' "$wid"
xprop -id "$wid" WM_CLASS _NET_WM_NAME WM_NAME
""".strip()
    result = transport.execute_argv(["bash", "-lc", command])
    output = str(result.get("output", ""))
    first = next((line for line in output.splitlines() if line.startswith("WINDOW=")), "")
    window_id = first.removeprefix("WINDOW=")
    folded = output.casefold()
    markers = {
        "writer": ("libreoffice", "writer"),
        "calc": ("libreoffice", "calc"),
        "files": ("nautilus",),
        "chrome": ("chrome", "chromium"),
        "vscode": ("code", "visual studio code"),
    }[expected_app]
    verified = bool(window_id) and any(marker in folded for marker in markers)
    return {
        "verified": verified,
        "method": "x11_getactivewindow",
        "window_id": window_id,
        "expected_application": expected_app,
        "observed_application": expected_app if verified else "unrecognized",
        "raw_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


class RealVmArmSession:
    def __init__(
        self,
        runtime: "RealVmPairedRuntime",
        *,
        task: Any,
        arm: Any,
        mode: str,
        gold_prefix_length: int,
        horizon: int,
        generation_seed: int,
    ) -> None:
        self.runtime = runtime
        self.task = task
        self.semantic_task = runtime.curriculum.by_id(task.task_id)
        self.arm = arm
        self.mode = mode
        self.prefix = gold_prefix_length
        self.horizon = horizon
        self.generation_seed = generation_seed
        self._model_calls = 0
        self._executed_turns = 0
        self._closed = False
        self._next_semantic_step = gold_prefix_length + 1
        output = runtime.session_root / f"{task.task_id}-{arm.name}-{uuid.uuid4().hex}"
        self.vm = KvmFixtureSession(
            qcow=runtime.qcow,
            qemu=runtime.qemu,
            provider_path=runtime.provider,
            vm_log_dir=output / "vm_logs",
            expected_provider_sha256=runtime.provider_sha256,
        )
        stage = "vm_start"
        try:
            self.vm.start()
            self.ledger = RuntimeEvidenceLedger(
                setup_commit=runtime.setup_validation.setup_commit,
                reset_provider=str(runtime.provider),
                reset_attestor=self.vm,
            )
            stage = "first_reset_and_setup"
            _, first = replay._reset_probe(self.vm, self.semantic_task, self.ledger)
            stage = "second_reset_and_setup"
            self.transport, second = replay._reset_probe(
                self.vm, self.semantic_task, self.ledger
            )
            stage = "live_binding"
            self.binding = bind_repeated_runtime_probes(
                self.semantic_task, (first, second), ledger=self.ledger
            )
            stage = "gold_prefix_replay"
            self.prefix_replay = self._replay_prefix(gold_prefix_length)
            state = dict(second.state)
            state["held_inputs"] = []
            self.reset_signature = reset_signature(self.semantic_task, state)
            self.start = SessionStart(
                task_id=task.task_id,
                snapshot_id=task.snapshot_id,
                parameter_seed=task.parameter_seed,
                cursor_ref=task.cursor_ref_for_prefix(gold_prefix_length),
                cursor=tuple(self.transport.cursor_position()),
                reset_signature=self.reset_signature,
                cursor_source="live_probe_before_policy",
                cursor_precentered=False,
                binding_receipt=self.binding.receipt(),
                prefix_replay=tuple(self.prefix_replay),
            )
        except InfrastructureFailure:
            self.vm.close()
            raise
        except Exception as exc:
            try:
                self.vm.close()
            except Exception:
                pass
            if isinstance(exc, VmHarnessError):
                failure_class = "vm_reset"
            elif isinstance(exc, (TransportError, RuntimeProbeError)):
                failure_class = "vm_setup"
            else:
                failure_class = "harness_io"
            raise _failure(
                failure_class,
                "open_session",
                "vm_session_initialization_failed",
                task_id=task.task_id,
                stage=stage,
                error_type=type(exc).__name__,
            ) from exc

    def _replay_prefix(self, length: int) -> list[dict[str, Any]]:
        journal: list[dict[str, Any]] = []
        for semantic_step in range(1, length + 1):
            binding_receipt = self.binding.receipt()
            segment = compile_semantic_step(
                self.semantic_task,
                self.arm.action_interface,
                binding=self.binding,
                semantic_step_index=semantic_step,
            )
            started = time.monotonic_ns()
            dispatches = tuple(
                replay._dispatch_compiled_action(
                    self.transport, self.arm.action_interface, action
                )
                for action in segment.actions
            )
            completed = time.monotonic_ns()
            executed = record_executed_segment(
                segment,
                dispatches,
                execution_started_monotonic_ns=started,
                execution_completed_monotonic_ns=completed,
            )
            self.ledger.record_executed_segment(
                self.semantic_task,
                self.binding,
                segment,
                dispatches,
                executed,
                near_miss=False,
            )
            row = {
                "semantic_step": semantic_step,
                "binding_receipt": binding_receipt,
                "binding_sha256": segment.binding_sha256,
                "compiled_segment": asdict(segment),
                "executed_receipt": asdict(executed),
                "actions": [
                    {
                        "action_index": index,
                        "screenshot": None,
                        "action": action,
                        "dispatch": list(dispatch),
                    }
                    for index, (action, dispatch) in enumerate(
                        zip(segment.actions, dispatches, strict=True)
                    )
                ],
            }
            if self.semantic_task.app == "chrome" and semantic_step == 2:
                self.binding = self._refresh_after_step_two(
                    segment, dispatches, executed, started, completed
                )
                transition = self.binding.receipt()["refresh_transitions"][0]
                row.update(
                    {
                        "post_scroll_refresh": transition["refresh_evidence"],
                        "refreshed_binding_sha256": self.binding.binding_sha256,
                        "refreshed_binding_receipt": self.binding.receipt(),
                    }
                )
            journal.append(row)
        return journal

    def _refresh_after_step_two(
        self,
        segment: CompiledSegment,
        dispatches: Any,
        executed: Any,
        started: int,
        completed: int,
    ) -> Any:
        probe_started = time.monotonic_ns()
        refreshed = probe_runtime(
            self.transport, self.semantic_task, expect_initial_state=False
        )
        probe_completed = time.monotonic_ns()
        refreshed = self.ledger.issue_refresh_probe(
            self.semantic_task,
            self.binding,
            refreshed,
            completed_step=2,
            executed_segment=executed,
            action_started_monotonic_ns=started,
            action_completed_monotonic_ns=completed,
            probe_started_monotonic_ns=probe_started,
            probe_completed_monotonic_ns=probe_completed,
        )
        return refresh_binding_after_step(
            self.semantic_task,
            self.binding,
            completed_step=2,
            probe=refreshed,
            executed_segment=executed,
            ledger=self.ledger,
        )

    def observe(self) -> Observation:
        payload = _screenshot(self.transport)
        return Observation(
            {
                "instruction": self.task.instruction,
                "image_base64": base64.b64encode(payload).decode("ascii"),
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "arm": self.arm.name,
                "action_interface": self.arm.action_interface,
            },
            "application/vnd.proper-vm-observation+json",
        )

    def request_action(
        self,
        *,
        observation: Observation,
        history: tuple[dict[str, Any], ...],
        generation_seed: int,
        budget: dict[str, Any],
    ) -> RequestedAction:
        endpoint = self.runtime.endpoints[self.arm.name]
        payload = observation.payload
        assert isinstance(payload, dict)
        request = {
            "model": endpoint["served_model"],
            "seed": generation_seed,
            "temperature": self.arm.generation["temperature"],
            "max_tokens": int(budget["output_tokens_per_turn"]),
            "messages": [
                {"role": "system", "content": _read_prompt(self.arm.name)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _model_context(
                                self.task.instruction,
                                self._next_semantic_step,
                                history,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + payload["image_base64"]
                            },
                        },
                    ],
                },
            ],
        }
        raw = json.dumps(request).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint["base_url"].rstrip("/") + "/chat/completions",
            data=raw,
            method="POST",
            headers={
                "Authorization": "Bearer " + endpoint["api_key"],
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=120) as response:
                result = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise _failure(
                "model_service",
                "generate_action",
                "model_service_request_failed",
                arm=self.arm.name,
                generation_seed=generation_seed,
                error_type=type(exc).__name__,
            ) from exc
        content = ""
        try:
            content = _content(result)
            value = (
                _native_action(content, self._next_semantic_step)
                if self.arm.name == "native_absolute_control"
                else _compact_action(content, self._next_semantic_step)
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raw_content = content if content else json.dumps(result, sort_keys=True)
            value = {
                "parse_error": str(exc),
                "raw_content_sha256": hashlib.sha256(
                    raw_content.encode("utf-8")
                ).hexdigest(),
            }
        usage = result.get("usage") if isinstance(result, dict) else None
        output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        self._model_calls += 1
        return RequestedAction(
            value=value,
            model_call_id=f"{self.arm.name}-{self._model_calls}-{uuid.uuid4().hex}",
            usage={"output_tokens": int(output_tokens)},
            generation_seed=generation_seed,
        )

    def _failed_receipt(self, requested: RequestedAction, status: str) -> ExecutionReceipt:
        cursor = tuple(self.transport.cursor_position())
        return ExecutionReceipt(
            executed_action=None,
            cursor_before=cursor,
            cursor_after=cursor,
            parse_status="error" if status == "parse_error" else "ok",
            dispatch_status="error",
            executor_evidence={"failure": status},
            binding_receipt=self.binding.receipt(),
        )

    def execute(self, requested: RequestedAction) -> ExecutionReceipt:
        value = requested.value
        if not isinstance(value, dict) or "parse_error" in value:
            return self._failed_receipt(requested, "parse_error")
        semantic_step = value.get("semantic_step")
        if semantic_step != self._next_semantic_step or not 1 <= semantic_step <= self.task.semantic_step_count:
            return self._failed_receipt(requested, "semantic_step_mismatch")
        actions: tuple[Any, ...] = (
            (value,)
            if self.arm.action_interface == "native_absolute_sequence_v1"
            else tuple(value["actions"])
        )
        before = tuple(self.transport.cursor_position())
        try:
            events = _resolved_event_count(self.arm.action_interface, actions)
            if len(actions) > self.semantic_task.budget_contract["primitive_action_caps"][self.arm.action_interface]:
                raise ValueError("model action plan exceeds primitive-action cap")
            if events > self.semantic_task.budget_contract["primitive_event_caps"][self.arm.action_interface]:
                raise ValueError("model action plan exceeds primitive-event cap")
            after_expected = _expected_cursor_after(
                self.arm.action_interface, actions, before
            )
            budget_payload = resolved_segment_budget_payload(
                task_id=self.task.task_id,
                fixture_sha256=self.task.fixture_sha256,
                action_schema=self.arm.action_interface,
                semantic_step_index=semantic_step,
                actions=actions,
                resolved_primitive_actions=len(actions),
                resolved_primitive_events=events,
                binding_revision=self.binding.binding_revision,
                binding_sha256=self.binding.binding_sha256,
                expected_cursor_before=before,
                expected_cursor_after=after_expected,
            )
            segment = CompiledSegment(
                task_id=self.task.task_id,
                fixture_sha256=self.task.fixture_sha256,
                action_schema=self.arm.action_interface,
                semantic_step_index=semantic_step,
                actions=actions,
                resolved_primitive_actions=len(actions),
                resolved_primitive_events=events,
                resolved_budget_sha256=hashlib.sha256(
                    canonical_json(budget_payload)
                ).hexdigest(),
                binding_revision=self.binding.binding_revision,
                binding_sha256=self.binding.binding_sha256,
                expected_cursor_before=before,
                expected_cursor_after=after_expected,
            )
            started = time.monotonic_ns()
            dispatches = tuple(
                replay._dispatch_compiled_action(
                    self.transport, self.arm.action_interface, action
                )
                for action in actions
            )
            completed = time.monotonic_ns()
            executed = record_executed_segment(
                segment,
                dispatches,
                execution_started_monotonic_ns=started,
                execution_completed_monotonic_ns=completed,
            )
            self.ledger.record_executed_segment(
                self.semantic_task,
                self.binding,
                segment,
                dispatches,
                executed,
                near_miss=False,
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            return self._failed_receipt(requested, "dispatch_or_receipt_validation")
        binding_used = self.binding
        if self.semantic_task.app == "chrome" and semantic_step == 2:
            try:
                self.binding = self._refresh_after_step_two(
                    segment, dispatches, executed, started, completed
                )
            except Exception as exc:
                raise _failure(
                    "harness_io",
                    "execute_action",
                    "post_scroll_live_binding_refresh_failed",
                    task_id=self.task.task_id,
                    error_type=type(exc).__name__,
                ) from exc
        self._next_semantic_step += 1
        self._executed_turns += 1
        operations = tuple(
            operation
            for action_dispatch in dispatches
            for result in action_dispatch
            for operation in result["operations"]
        )
        backend = tuple(
            primitive
            for action_dispatch in dispatches
            for result in action_dispatch
            for primitive in (
                result.get("executor_v4_dispatch_evidence", result)
                .get("atomic_state", {})
                .get("backend_primitives", [])
            )
        )
        if any(
            result.get("executor_v4_dispatch_evidence", result)
            .get("atomic_state", {})
            .get("click_backend")
            != PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND
            for action_dispatch in dispatches
            for result in action_dispatch
        ):
            raise _failure(
                "harness_io",
                "execute_action",
                "shared_click_backend_binding_drift",
                task_id=self.task.task_id,
            )
        lowered: list[dict[str, Any]] = []
        action_classes: list[str] = []
        native_clicks: list[dict[str, Any]] = []
        source_index = 0
        for action_dispatch in dispatches:
            for result in action_dispatch:
                action_classes.append(str(result["action_class"]))
        if self.arm.name == "native_absolute_control":
            for operation in value["operations"]:
                lowered_row = {
                    "backend": self.arm.action_interface,
                    "source_operation_index": source_index,
                    **operation,
                }
                lowered.append(lowered_row)
                if operation.get("action") == "click":
                    coordinate = operation["coordinate"]
                    native_clicks.append(
                        {
                            "requested_operation_index": source_index,
                            "lowered_operation_index": len(lowered) - 1,
                            "requested_coordinate": coordinate,
                            "dispatched_coordinate": coordinate,
                            "post_click_cursor": coordinate,
                        }
                    )
                source_index += 1
        else:
            lowered = [
                {"backend": self.arm.action_interface, "source_action_index": index}
                for index in range(len(actions))
            ]
        after = tuple(self.transport.cursor_position())
        try:
            active_window = _active_window(self.transport, self.task.app)
        except Exception as exc:
            raise _failure(
                "harness_io",
                "execute_action",
                "active_window_capture_failed",
                task_id=self.task.task_id,
                error_type=type(exc).__name__,
            ) from exc
        evidence = {
            "cursor_readback_verified": after == after_expected,
            "interventions_between_policy_turns": [],
            "active_window": active_window,
            "post_action_cursor_verified": after == after_expected,
            "post_action_cursor": list(after),
            "native_click_dispatches": native_clicks,
            "shared_click_backend": PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
        }
        return ExecutionReceipt(
            executed_action=value,
            cursor_before=before,
            cursor_after=after,
            parse_status="ok",
            dispatch_status="ok",
            action_classes=tuple(action_classes),
            semantic_operations=tuple(value.get("operations", value.get("actions", []))),
            lowered_operations=tuple(lowered),
            operations=operations,
            backend_primitives=backend,
            executor_evidence=evidence,
            primitive_action_count=len(actions),
            resolved_actions=actions,
            semantic_step_index=semantic_step,
            resolved_primitive_actions=len(actions),
            resolved_primitive_events=events,
            resolved_budget_sha256=segment.resolved_budget_sha256,
            binding_sha256=binding_used.binding_sha256,
            binding_revision=binding_used.binding_revision,
            binding_receipt=binding_used.receipt(),
            compiled_segment=asdict(segment),
            dispatches=dispatches,
            executed_segment_receipt=asdict(executed),
        )

    def probe_state(self) -> StateProbe:
        live = probe_runtime(
            self.transport, self.semantic_task, expect_initial_state=False
        )
        initial_semantic = self._executed_turns == 0 and self.prefix > 0
        semantic = initial_semantic or self.mode == "gold_history_one_step"
        if semantic:
            step = self.prefix if initial_semantic else min(
                self.prefix + self._executed_turns, self.task.semantic_step_count
            )
            state = {
                "task_id": self.task.task_id,
                "fixture_sha256": self.task.fixture_sha256,
                "geometry": {
                    name: list(point) for name, point in live.geometry.items()
                },
                "initial_cursor": list(self.binding.resolved_initial_cursor),
                "cursor": list(self.transport.cursor_position()),
                "held_inputs": sorted(
                    self.transport.audit.held_buttons | self.transport.audit.held_keys
                ),
                "semantic_step_index": step,
            }
        else:
            state = dict(live.state)
            state["held_inputs"] = sorted(
                self.transport.audit.held_buttons | self.transport.audit.held_keys
            )
        return StateProbe(
            state=state,
            evidence={
                "read_only": True,
                "input_events": [],
                "application": self.task.app,
                "method": _APP_PROBE_METHODS[self.task.app],
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.vm.close()
        except Exception as exc:
            raise _failure(
                "harness_io",
                "close_session",
                "vm_session_close_failed",
                task_id=self.task.task_id,
                error_type=type(exc).__name__,
            ) from exc


class RealVmPairedRuntime:
    def __init__(self, manifest: Any, readiness: Any, setup_validation: Any) -> None:
        if readiness.executor_commit != EXECUTOR_CANDIDATE_COMMIT:
            raise RuntimeError("executor readiness does not pin the reviewed v4 candidate")
        self.manifest = manifest
        self.readiness = readiness
        self.setup_validation = setup_validation
        self.curriculum = load_manifest("development")
        self.endpoints = _load_endpoints()
        for arm in manifest.arms:
            endpoint = self.endpoints[arm.name]
            if (
                endpoint["checkpoint"] != arm.checkpoint
                or endpoint["checkpoint_sha256"] != arm.checkpoint_sha256
            ):
                raise RuntimeError(f"{arm.name}: endpoint/checkpoint binding mismatch")
            prompt = _read_prompt(arm.name)
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != arm.prompt_sha256:
                raise RuntimeError(f"{arm.name}: prompt hash mismatch")
        self.qcow = Path(os.environ["PAIRED_VM_QCOW"])
        self.qemu = Path(os.environ["PAIRED_VM_QEMU"])
        self.provider = Path(os.environ["PAIRED_VM_PROVIDER"])
        self.provider_sha256 = os.environ["PAIRED_VM_PROVIDER_SHA256"]
        self.session_root = Path(os.environ["PAIRED_VM_SESSION_ROOT"])
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.contract = {
            "schema": RUNTIME_CONTRACT_SCHEMA,
            "runtime_id": manifest.runtime.runtime_id,
            "executor_commit": readiness.executor_commit,
            "interfaces": dict(ACTION_INTERFACES),
            "cursor_initialization": "live_unmodified_snapshot",
            "native_coordinate_dispatch": "requested_to_lowered_to_post_cursor",
            "between_turn_interventions": "forbidden",
            "active_window_check": "true_active_window_only",
            "curriculum_commit": APPROVED_CURRICULUM_COMMIT,
            "live_binding": APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
            "resolved_budget_receipts": "executed_segment_receipt_v1",
            "ordered_execution_trace_aggregate": "paired_policy_turn_receipt_trace_v1",
            "complete_program_aggregate": "c603_compiled_program_receipt_v1",
        }

    def open_session(self, **kwargs: Any) -> RealVmArmSession:
        return RealVmArmSession(self, **kwargs)


def create_runtime(manifest: Any, readiness: Any, setup_validation: Any) -> RealVmPairedRuntime:
    return RealVmPairedRuntime(manifest, readiness, setup_validation)
