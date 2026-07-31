from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .manifest import TaskDefinition, canonical_bytes
from .runtime import Episode, Observation, ResetReceipt


TeacherFormat = Literal["native_absolute_control", "compact_raw_phaseb"]


class TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObservationReference:
    frame_uri: str
    frame_sha256: str
    step_index: int


@dataclass(frozen=True)
class TeacherTrace:
    schema_version: int
    task_id: str
    split: str
    task_sha256: str
    reset_fingerprint: str
    format: TeacherFormat
    initial_cursor: tuple[int, int]
    observations: tuple[ObservationReference, ...]
    actions: tuple[dict[str, Any] | str, ...]
    source_native_trace_sha256: str | None
    trace_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "split": self.split,
            "task_sha256": self.task_sha256,
            "reset_fingerprint": self.reset_fingerprint,
            "format": self.format,
            "initial_cursor": list(self.initial_cursor),
            "observations": [asdict(item) for item in self.observations],
            "actions": list(self.actions),
            "source_native_trace_sha256": self.source_native_trace_sha256,
        }

    def verify(self) -> None:
        observed = hashlib.sha256(canonical_bytes(self.unsigned_payload())).hexdigest()
        if observed != self.trace_sha256:
            raise TeacherError(f"teacher trace seal mismatch: {self.task_id}")


def _trace(
    *,
    task: TaskDefinition,
    reset_fingerprint: str,
    format: TeacherFormat,
    observations: tuple[ObservationReference, ...],
    actions: tuple[dict[str, Any] | str, ...],
    source_native_trace_sha256: str | None = None,
) -> TeacherTrace:
    payload = {
        "schema_version": 1,
        "task_id": task.task_id,
        "split": task.split,
        "task_sha256": task.task_sha256,
        "reset_fingerprint": reset_fingerprint,
        "format": format,
        "initial_cursor": list(task.geometry.initial_cursor),
        "observations": [asdict(item) for item in observations],
        "actions": list(actions),
        "source_native_trace_sha256": source_native_trace_sha256,
    }
    return TeacherTrace(
        schema_version=1,
        task_id=task.task_id,
        split=task.split,
        task_sha256=task.task_sha256,
        reset_fingerprint=reset_fingerprint,
        format=format,
        initial_cursor=task.geometry.initial_cursor,
        observations=observations,
        actions=actions,
        source_native_trace_sha256=source_native_trace_sha256,
        trace_sha256=hashlib.sha256(canonical_bytes(payload)).hexdigest(),
    )


def _validate_native_action(action: dict[str, Any]) -> None:
    if not isinstance(action, dict):
        raise TeacherError("native teacher action must be an object")
    name = action.get("action")
    if name not in {
        "mouse_move",
        "left_click",
        "left_click_drag",
        "scroll",
        "type",
        "wait",
    }:
        raise TeacherError(f"unsupported native teacher action: {name!r}")
    if name in {"mouse_move", "left_click_drag"}:
        coordinate = action.get("coordinate")
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise TeacherError(f"{name} requires absolute coordinate [x, y]")
    if name == "left_click" and "coordinate" in action:
        coordinate = action["coordinate"]
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise TeacherError("left_click coordinate must be [x, y]")
    if name == "type" and not isinstance(action.get("text"), str):
        raise TeacherError("type requires exact text")


class NativeTeacherCollector:
    """Collect absolute native actions against policy-visible VM observations."""

    def __init__(self, task: TaskDefinition, receipt: ResetReceipt) -> None:
        if task.split not in {"train", "development"}:
            raise TeacherError("teacher collection is forbidden on sealed evaluation")
        if receipt.task_id != task.task_id or receipt.task_sha256 != task.task_sha256:
            raise TeacherError("reset receipt/task mismatch")
        self.task = task
        self.receipt = receipt
        self._observations: list[ObservationReference] = []
        self._actions: list[dict[str, Any]] = []

    def record(self, observation: Observation, action: dict[str, Any]) -> None:
        if len(self._actions) >= self.task.horizon:
            raise TeacherError("native collection exceeded frozen horizon")
        if observation.task_id != self.task.task_id:
            raise TeacherError("observation/task mismatch")
        if observation.step_index != len(self._actions):
            raise TeacherError("teacher observation/action index mismatch")
        _validate_native_action(action)
        self._observations.append(
            ObservationReference(
                frame_uri=observation.frame_uri,
                frame_sha256=observation.frame_sha256,
                step_index=observation.step_index,
            )
        )
        self._actions.append(json.loads(json.dumps(action, ensure_ascii=False)))

    def finish(self) -> TeacherTrace:
        if not self._actions:
            raise TeacherError("cannot finalize an empty teacher trace")
        trace = _trace(
            task=self.task,
            reset_fingerprint=self.receipt.reset_fingerprint,
            format="native_absolute_control",
            observations=tuple(self._observations),
            actions=tuple(self._actions),
        )
        trace.verify()
        return trace


def native_gold_actions(
    task: TaskDefinition, *, near_miss: bool = False
) -> tuple[dict[str, Any], ...]:
    actions: list[dict[str, Any]] = []
    final_step = task.steps[-1]
    for step in task.steps:
        miss = near_miss and step == final_step
        if step == "focus":
            actions.append(
                {"action": "left_click", "coordinate": list(task.geometry.field_center)}
            )
        elif step == "coalesced_type":
            text = task.target_text + ("-near-miss" if miss else "")
            actions.append({"action": "type", "text": text})
        elif step == "scroll":
            clicks = -task.scroll_clicks if miss else task.scroll_clicks
            actions.append({"action": "scroll", "clicks": clicks})
        elif step == "click":
            target = task.geometry.decoy_center if miss else task.geometry.click_center
            actions.append({"action": "left_click", "coordinate": list(target)})
        elif step == "drag":
            actions.append(
                {"action": "mouse_move", "coordinate": list(task.geometry.drag_start)}
            )
            end = task.geometry.drag_end
            if miss:
                end = (
                    (task.geometry.drag_start[0] + end[0]) // 2,
                    (task.geometry.drag_start[1] + end[1]) // 2,
                )
            actions.append({"action": "left_click_drag", "coordinate": list(end)})
        else:  # pragma: no cover - manifest validation owns this boundary
            raise TeacherError(f"unknown semantic step: {step}")
    if len(actions) > task.horizon:
        raise TeacherError("native gold exceeds common frozen horizon")
    return tuple(actions)


def _raw_move(
    cursor: tuple[int, int], target: tuple[int, int], suffix: str = ""
) -> str:
    dx, dy = target[0] - cursor[0], target[1] - cursor[1]
    value = f"{dx} {dy} 0"
    return value + (f" ; {suffix}" if suffix else "")


def convert_native_actions(
    actions: tuple[dict[str, Any], ...], initial_cursor: tuple[int, int]
) -> tuple[str, ...]:
    """Deterministically lower absolute native teacher actions to compact raw."""
    cursor = initial_cursor
    compact: list[str] = []
    index = 0
    while index < len(actions):
        action = actions[index]
        _validate_native_action(action)
        name = str(action["action"])
        if (
            name == "mouse_move"
            and index + 1 < len(actions)
            and actions[index + 1].get("action") == "left_click_drag"
        ):
            # Preserve the shared raw drag contract in three turns: move to the
            # start and hold, move while held, then release. The native teacher
            # expresses the same semantic drag as move-to-start + drag-to-end.
            start = tuple(int(value) for value in action["coordinate"])
            following = actions[index + 1]
            _validate_native_action(following)
            target = tuple(int(value) for value in following["coordinate"])
            compact.extend(
                (
                    _raw_move(cursor, start, "+LMB"),
                    _raw_move(start, target),
                    "0 0 0 ; -LMB",
                )
            )
            cursor = target
            index += 2
            continue
        if name == "mouse_move":
            target = tuple(int(value) for value in action["coordinate"])
            compact.append(_raw_move(cursor, target))
            cursor = target
        elif name == "left_click":
            coordinate = action.get("coordinate")
            target = (
                cursor
                if coordinate is None
                else tuple(int(value) for value in coordinate)
            )
            compact.append(_raw_move(cursor, target, "+LMB -LMB"))
            cursor = target
        elif name == "left_click_drag":
            target = tuple(int(value) for value in action["coordinate"])
            compact.extend(
                (
                    "0 0 0 ; +LMB",
                    _raw_move(cursor, target),
                    "0 0 0 ; -LMB",
                )
            )
            cursor = target
        elif name == "scroll":
            compact.append(f"0 0 {int(action['clicks'])}")
        elif name == "type":
            compact.append(
                "0 0 0 ; type("
                + json.dumps(str(action["text"]), ensure_ascii=False)
                + ")"
            )
        elif name == "wait":
            compact.append("0 0 0")
        index += 1
    return tuple(compact)


def convert_native_trace(
    task: TaskDefinition,
    trace: TeacherTrace,
    *,
    compact_observations: tuple[ObservationReference, ...],
    compact_reset_fingerprint: str,
) -> TeacherTrace:
    trace.verify()
    if trace.task_id != task.task_id or trace.task_sha256 != task.task_sha256:
        raise TeacherError("native trace/task mismatch")
    if trace.format != "native_absolute_control":
        raise TeacherError("only native absolute traces can be converted")
    native = tuple(action for action in trace.actions if isinstance(action, dict))
    if len(native) != len(trace.actions):
        raise TeacherError("native trace contains non-object actions")
    compact = convert_native_actions(native, trace.initial_cursor)
    if len(compact) > task.horizon:
        raise TeacherError("converted compact trace exceeds frozen common horizon")
    if compact_reset_fingerprint != trace.reset_fingerprint:
        raise TeacherError("compact derivative reset differs from native source")
    if len(compact_observations) != len(compact):
        raise TeacherError(
            "compact derivative requires one refreshed observation per action"
        )
    if tuple(item.step_index for item in compact_observations) != tuple(
        range(len(compact_observations))
    ):
        raise TeacherError("compact observation indices are not contiguous")
    converted = _trace(
        task=task,
        reset_fingerprint=compact_reset_fingerprint,
        format="compact_raw_phaseb",
        observations=compact_observations,
        actions=compact,
        source_native_trace_sha256=trace.trace_sha256,
    )
    converted.verify()
    return converted


def collect_compact_derivative(
    task: TaskDefinition,
    native_trace: TeacherTrace,
    *,
    episode: Episode | None = None,
) -> TeacherTrace:
    """Replay converted actions to refresh every compact observation.

    Production VM collection passes a VM-backed episode with the same reset and
    observation API. The default CPU episode is only the deterministic contract
    backend used by pre-gate checks.
    """
    native_trace.verify()
    if native_trace.format != "native_absolute_control":
        raise TeacherError("compact derivative requires a native source trace")
    native_actions = tuple(
        action for action in native_trace.actions if isinstance(action, dict)
    )
    if len(native_actions) != len(native_trace.actions):
        raise TeacherError("native source trace contains non-object actions")
    compact_episode = episode or Episode(task, "compact_raw_phaseb")
    receipt = compact_episode.reset()
    actions = convert_native_actions(native_actions, native_trace.initial_cursor)
    if actions != convert_native_actions(native_actions, native_trace.initial_cursor):
        raise TeacherError("compact conversion was nondeterministic")
    observations: list[ObservationReference] = []
    observation = receipt.observation
    final = None
    for index, action in enumerate(actions):
        observations.append(
            ObservationReference(
                frame_uri=observation.frame_uri,
                frame_sha256=observation.frame_sha256,
                step_index=index,
            )
        )
        final = compact_episode.step(action)
        observation = final.observation
    if final is None or not final.done or final.reward != 1:
        raise TeacherError("compact derivative replay did not pass the task oracle")
    return convert_native_trace(
        task,
        native_trace,
        compact_observations=tuple(observations),
        compact_reset_fingerprint=receipt.reset_fingerprint,
    )
