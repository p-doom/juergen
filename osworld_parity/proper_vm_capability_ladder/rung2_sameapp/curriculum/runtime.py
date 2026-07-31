"""Bind semantic tasks to live VM state, geometry, and cursor probes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from .oracle import initial_state
from .schema import SemanticTask
from ..fixtures import canonical_json


class RuntimeProbeError(RuntimeError):
    """A live VM probe does not satisfy the task's declared contract."""


@dataclass(frozen=True)
class RuntimeProbe:
    state: dict[str, Any]
    geometry: dict[str, tuple[int, int]]
    initial_cursor: tuple[int, int]
    screen_size: tuple[int, int]
    geometry_probe_version: str
    state_probe_version: str
    cursor_probe_version: str = "rung1_cursor_position_v1"


@dataclass(frozen=True)
class ValidatedRuntimeBinding:
    task_id: str
    fixture_sha256: str
    reset_probe_count: int
    reset_probes: tuple[RuntimeProbe, ...]
    initial_probe: RuntimeProbe
    initial_cursor_ref: str
    resolved_initial_cursor: tuple[int, int]
    refreshed_after_steps: dict[int, RuntimeProbe]
    binding_sha256: str

    def validate_for_task(self, task: SemanticTask) -> None:
        if self.task_id != task.task_id or self.fixture_sha256 != task.fixture_sha256:
            raise RuntimeProbeError(f"{task.task_id}: runtime binding identity mismatch")
        if self.reset_probe_count < 2 or self.reset_probe_count != len(self.reset_probes):
            raise RuntimeProbeError(f"{task.task_id}: fewer than two reset probes")
        if self.initial_probe != self.reset_probes[0]:
            raise RuntimeProbeError(f"{task.task_id}: initial/reset probe mismatch")
        for following in self.reset_probes[1:]:
            validate_repeated_runtime_probes(task, self.initial_probe, following)
        if self.initial_cursor_ref != "runtime.initial_cursor" or (
            self.resolved_initial_cursor != self.initial_probe.initial_cursor
        ):
            raise RuntimeProbeError(f"{task.task_id}: resolved initial cursor mismatch")
        for probe in self.refreshed_after_steps.values():
            _validate_probe_envelope(task, probe, expect_initial_state=False)
        observed = hashlib.sha256(
            canonical_json(
                _binding_payload(task, self.reset_probes, self.refreshed_after_steps)
            )
        ).hexdigest()
        if observed != self.binding_sha256:
            raise RuntimeProbeError(f"{task.task_id}: runtime binding seal mismatch")

    def probe_for_step(self, task: SemanticTask, step_index: int) -> RuntimeProbe:
        self.validate_for_task(task)
        if task.app == "chrome" and step_index > 2:
            try:
                return self.refreshed_after_steps[2]
            except KeyError as exc:
                raise RuntimeProbeError(
                    f"{task.task_id}: Chrome step {step_index} requires post-step-2 refresh"
                ) from exc
        return self.initial_probe


def probe_runtime(
    transport: Any, task: SemanticTask, *, expect_initial_state: bool = True
) -> RuntimeProbe:
    """Run the declared existing VM probes; never source task coordinates."""

    if task.app == "vscode":
        from ...rung1b_realapps.vm import probe_fixture, probe_geometry
        from .oracle import as_vscode_fixture

        fixture = as_vscode_fixture(task)
        state = probe_fixture(transport, fixture)
        live = probe_geometry(transport, fixture)
        geometry = {"editor": tuple(live.editor)}
    else:
        from ..vm import probe_geometry, probe_state
        from .oracle import as_sameapp_fixture

        fixture = as_sameapp_fixture(task)
        state = probe_state(transport, fixture)
        geometry = probe_geometry(transport, fixture, state)
    probe = RuntimeProbe(
        state=state,
        geometry=geometry,
        initial_cursor=tuple(transport.cursor_position()),
        screen_size=tuple(transport.screen_size()),
        geometry_probe_version=task.geometry_contract["probe_version"],
        state_probe_version=task.geometry_contract["state_probe_version"],
    )
    _validate_probe_envelope(task, probe, expect_initial_state=expect_initial_state)
    return probe


def _validate_probe_envelope(
    task: SemanticTask, probe: RuntimeProbe, *, expect_initial_state: bool
) -> None:
    contract = task.geometry_contract
    if probe.geometry_probe_version != contract["probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: geometry probe version mismatch")
    if probe.state_probe_version != contract["state_probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: state probe version mismatch")
    if probe.cursor_probe_version != task.initial_cursor["probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: cursor probe version mismatch")
    if expect_initial_state:
        expected_state = initial_state(task)
        expected_state.pop("held_inputs")
        if probe.state != expected_state:
            raise RuntimeProbeError(
                f"{task.task_id}: setup-state mismatch: {probe.state!r} != {expected_state!r}"
            )
    required = set(contract["required_targets"])
    if set(probe.geometry) != required:
        raise RuntimeProbeError(
            f"{task.task_id}: geometry targets drifted: "
            f"{sorted(probe.geometry)} != {sorted(required)}"
        )
    width, height = probe.screen_size
    if width < 1 or height < 1:
        raise RuntimeProbeError(f"{task.task_id}: invalid live viewport")
    for name, point in probe.geometry.items():
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or not all(isinstance(value, int) for value in point)
            or not 0 <= point[0] < width
            or not 0 <= point[1] < height
        ):
            raise RuntimeProbeError(f"{task.task_id}: invalid live target {name}={point!r}")
    x, y = probe.initial_cursor
    if not 0 <= x < width or not 0 <= y < height:
        raise RuntimeProbeError(f"{task.task_id}: initial cursor is outside viewport")


def validate_runtime_probe(task: SemanticTask, probe: RuntimeProbe) -> None:
    _validate_probe_envelope(task, probe, expect_initial_state=True)


def validate_repeated_runtime_probes(
    task: SemanticTask, first: RuntimeProbe, second: RuntimeProbe
) -> None:
    validate_runtime_probe(task, first)
    validate_runtime_probe(task, second)
    if first.geometry != second.geometry:
        raise RuntimeProbeError(f"{task.task_id}: geometry drift across exact resets")
    if first.initial_cursor != second.initial_cursor:
        raise RuntimeProbeError(f"{task.task_id}: cursor drift across exact resets")
    if first.screen_size != second.screen_size:
        raise RuntimeProbeError(f"{task.task_id}: viewport drift across exact resets")


def _binding_payload(
    task: SemanticTask,
    probes: tuple[RuntimeProbe, ...],
    refreshed_after_steps: dict[int, RuntimeProbe],
) -> dict[str, Any]:
    def row(probe: RuntimeProbe) -> dict[str, Any]:
        return {
            "state_sha256": hashlib.sha256(canonical_json(probe.state)).hexdigest(),
            "geometry": {name: list(point) for name, point in sorted(probe.geometry.items())},
            "initial_cursor": list(probe.initial_cursor),
            "screen_size": list(probe.screen_size),
            "geometry_probe_version": probe.geometry_probe_version,
            "state_probe_version": probe.state_probe_version,
            "cursor_probe_version": probe.cursor_probe_version,
        }

    return {
        "schema_version": 1,
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "reset_probes": [row(probe) for probe in probes],
        "refreshed_after_steps": {
            str(step): row(probe) for step, probe in sorted(refreshed_after_steps.items())
        },
    }


def bind_repeated_runtime_probes(
    task: SemanticTask, probes: tuple[RuntimeProbe, ...] | list[RuntimeProbe]
) -> ValidatedRuntimeBinding:
    values = tuple(probes)
    if len(values) < 2:
        raise RuntimeProbeError(f"{task.task_id}: at least two reset probes are required")
    first = values[0]
    for following in values[1:]:
        validate_repeated_runtime_probes(task, first, following)
    payload = _binding_payload(task, values, {})
    return ValidatedRuntimeBinding(
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        reset_probe_count=len(values),
        reset_probes=values,
        initial_probe=first,
        initial_cursor_ref="runtime.initial_cursor",
        resolved_initial_cursor=first.initial_cursor,
        refreshed_after_steps={},
        binding_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )


def refresh_binding_after_step(
    task: SemanticTask,
    binding: ValidatedRuntimeBinding,
    *,
    completed_step: int,
    probe: RuntimeProbe,
) -> ValidatedRuntimeBinding:
    binding.probe_for_step(task, min(completed_step, 2))
    _validate_probe_envelope(task, probe, expect_initial_state=False)
    refreshed = {**binding.refreshed_after_steps, completed_step: probe}
    payload = _binding_payload(task, binding.reset_probes, refreshed)
    return replace(
        binding,
        refreshed_after_steps=refreshed,
        binding_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )


def resolved_cursor_history(
    task: SemanticTask, probe: RuntimeProbe
) -> tuple[dict[str, Any], ...]:
    """Resolve semantic cursor refs only after the live probe passes."""

    validate_runtime_probe(task, probe)
    values: dict[str, tuple[int, int]] = {
        "runtime.initial_cursor": probe.initial_cursor,
        **{f"geometry.{name}": point for name, point in probe.geometry.items()},
    }
    rows: list[dict[str, Any]] = []
    for milestone in task.gold_cursor_history:
        try:
            before = values[milestone.cursor_before_ref]
            after = values[milestone.cursor_after_ref]
        except KeyError as exc:
            raise RuntimeProbeError(
                f"{task.task_id}: unresolved cursor reference {exc.args[0]!r}"
            ) from exc
        rows.append(
            {
                "prefix_length": milestone.prefix_length,
                "step_id": milestone.step_id,
                "target_ref": milestone.target_ref,
                "cursor_before_ref": milestone.cursor_before_ref,
                "cursor_before": before,
                "cursor_after_ref": milestone.cursor_after_ref,
                "cursor_after": after,
            }
        )
    return tuple(rows)


def compile_from_runtime_binding(
    task: SemanticTask,
    action_schema: str,
    binding: ValidatedRuntimeBinding,
    *,
    semantic_step_index: int,
    near_miss: bool = False,
) -> Any:
    from .program import compile_semantic_step

    return compile_semantic_step(
        task,
        action_schema,
        binding=binding,
        semantic_step_index=semantic_step_index,
        near_miss=near_miss,
    )
