"""Bind semantic tasks to live VM state, geometry, and cursor probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .oracle import initial_state
from .schema import SemanticTask


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


def probe_runtime(transport: Any, task: SemanticTask) -> RuntimeProbe:
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
    validate_runtime_probe(task, probe)
    return probe


def validate_runtime_probe(task: SemanticTask, probe: RuntimeProbe) -> None:
    contract = task.geometry_contract
    if probe.geometry_probe_version != contract["probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: geometry probe version mismatch")
    if probe.state_probe_version != contract["state_probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: state probe version mismatch")
    if probe.cursor_probe_version != task.initial_cursor["probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: cursor probe version mismatch")
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


def validate_repeated_runtime_probes(
    task: SemanticTask, first: RuntimeProbe, second: RuntimeProbe
) -> None:
    validate_runtime_probe(task, first)
    validate_runtime_probe(task, second)
    if first.geometry != second.geometry:
        raise RuntimeProbeError(f"{task.task_id}: geometry drift across exact resets")
    if first.initial_cursor != second.initial_cursor:
        raise RuntimeProbeError(f"{task.task_id}: cursor drift across exact resets")


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


def compile_from_runtime_probe(
    task: SemanticTask,
    action_schema: str,
    probe: RuntimeProbe,
    *,
    near_miss: bool = False,
) -> list[dict[str, Any] | str]:
    from .program import compile_program

    validate_runtime_probe(task, probe)
    return compile_program(
        task,
        action_schema,
        geometry=probe.geometry,
        initial_cursor=probe.initial_cursor,
        near_miss=near_miss,
    )
