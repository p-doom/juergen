from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPEC_FIELDS = {
    "action_contract",
    "instruction_template",
    "max_completed_turns",
    "max_previous_action_chars",
    "observation_contract",
    "schema_version",
    "spec_id",
    "system_prompt_sha256",
}
_SPEC_TYPES = {
    "action_contract": str,
    "instruction_template": str,
    "max_completed_turns": int,
    "max_previous_action_chars": int,
    "observation_contract": str,
    "schema_version": int,
    "spec_id": str,
    "system_prompt_sha256": str,
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class HarnessRenderSpec:
    schema_version: int
    spec_id: str
    max_completed_turns: int
    max_previous_action_chars: int
    system_prompt_sha256: str
    action_contract: str
    observation_contract: str
    instruction_template: str

    def __post_init__(self) -> None:
        for name, expected_type in _SPEC_TYPES.items():
            if type(getattr(self, name)) is not expected_type:
                raise TypeError(
                    f"render spec {name} must be {expected_type.__name__}, "
                    f"got {type(getattr(self, name)).__name__}"
                )
        if self.schema_version != 1:
            raise ValueError(f"unsupported render spec schema {self.schema_version!r}")
        if self.max_completed_turns < 1:
            raise ValueError("max_completed_turns must be at least 1")
        if self.max_previous_action_chars < 11:
            raise ValueError("max_previous_action_chars must be at least 11")
        for name in ("spec_id", "action_contract", "observation_contract"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if _SHA256_RE.fullmatch(self.system_prompt_sha256) is None:
            raise ValueError("system_prompt_sha256 must be a lowercase SHA-256 digest")
        fields = []
        for _, field_name, format_spec, conversion in Formatter().parse(
            self.instruction_template
        ):
            if field_name is None:
                continue
            if format_spec or conversion:
                raise ValueError(
                    "instruction_template fields cannot use formatting options"
                )
            fields.append(field_name)
        if fields != ["instruction", "previous_actions"]:
            raise ValueError(
                "instruction_template must contain {instruction} then {previous_actions} exactly once"
            )

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self), ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes, *, expected_sha256: str) -> HarnessRenderSpec:
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        observed_sha256 = _sha256(raw)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"render spec digest mismatch: expected {expected_sha256}, got {observed_sha256}"
            )
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("render spec must be UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise TypeError("render spec must be a JSON object")
        if set(data) != _SPEC_FIELDS:
            missing = sorted(_SPEC_FIELDS - set(data))
            unexpected = sorted(set(data) - _SPEC_FIELDS)
            raise ValueError(
                f"render spec keys differ: missing={missing}, unexpected={unexpected}"
            )
        for name, expected_type in _SPEC_TYPES.items():
            if type(data[name]) is not expected_type:
                raise ValueError(
                    f"render spec {name} must be {expected_type.__name__}, "
                    f"got {type(data[name]).__name__}"
                )
        spec = cls(**data)
        if spec.canonical_bytes() != raw:
            raise ValueError("render spec JSON is not canonically serialized")
        return spec

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str) -> HarnessRenderSpec:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot load render spec {path}: {exc}") from exc
        return cls.from_bytes(raw, expected_sha256=expected_sha256)

    def check_runtime(
        self,
        *,
        spec_sha256: str,
        system_prompt: str,
        action_contract: str,
        observation_contract: str,
    ) -> None:
        if spec_sha256 != self.sha256:
            raise ValueError(
                f"render spec digest mismatch: expected {spec_sha256}, got {self.sha256}"
            )
        prompt_sha256 = _sha256(system_prompt.encode("utf-8"))
        if prompt_sha256 != self.system_prompt_sha256:
            raise ValueError(
                "system prompt digest mismatch: "
                f"expected {self.system_prompt_sha256}, got {prompt_sha256}"
            )
        if action_contract != self.action_contract:
            raise ValueError(
                f"action contract mismatch: expected {self.action_contract!r}, "
                f"got {action_contract!r}"
            )
        if observation_contract != self.observation_contract:
            raise ValueError(
                f"observation contract mismatch: expected {self.observation_contract!r}, "
                f"got {observation_contract!r}"
            )


@dataclass(frozen=True)
class TrainingTurn[Frame]:
    image: Frame
    assistant: str
    action: str


@dataclass(frozen=True)
class _CompletedTurn[Frame]:
    step: int
    image: Frame
    history_text: str
    action: str | None


def _prior_assistant(text: str) -> str:
    _, separator, tail = text.partition("</think>")
    if separator:
        stripped = tail.lstrip("\n")
    else:
        if "<think>" in text:
            raise ValueError(
                "prior assistant response has an unterminated <think> block"
            )
        stripped = text.strip()
    if not stripped:
        raise ValueError("prior assistant response is empty after think stripping")
    return stripped


def _history_text(text: str, action: str | None) -> str:
    history_text = _prior_assistant(text)
    if action is None:
        return history_text
    if not isinstance(action, str) or not action.strip():
        raise ValueError("completed action must be non-empty text or None")
    lines = [line.strip() for line in history_text.splitlines() if line.strip()]
    if not lines or lines[-1] != action:
        raise ValueError("completed action does not match the assistant's final line")
    return history_text


class HarnessRenderer[Frame]:
    def __init__(
        self,
        spec: HarnessRenderSpec,
        *,
        spec_sha256: str,
        system_prompt: str,
        action_contract: str,
        observation_contract: str,
    ) -> None:
        spec.check_runtime(
            spec_sha256=spec_sha256,
            system_prompt=system_prompt,
            action_contract=action_contract,
            observation_contract=observation_contract,
        )
        self.spec = spec
        self.system_prompt = system_prompt
        self._started = False
        self._current: Frame
        self._visible: list[_CompletedTurn[Frame]] = []
        self._evicted_actions: list[tuple[int, str]] = []
        self._next_step = 1

    def start(self, image: Frame) -> None:
        if image is None:
            raise ValueError("current image cannot be None")
        self._current = image
        self._visible = []
        self._evicted_actions = []
        self._next_step = 1
        self._started = True

    def complete(
        self, *, assistant: str, action: str | None, next_image: Frame
    ) -> None:
        if not self._started:
            raise RuntimeError("HarnessRenderer.complete before start")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError("assistant must be non-empty text")
        if next_image is None:
            raise ValueError("next image cannot be None")
        history_text = _history_text(assistant, action)
        self._visible.append(
            _CompletedTurn(
                step=self._next_step,
                image=self._current,
                history_text=history_text,
                action=action,
            )
        )
        self._next_step += 1
        if len(self._visible) > self.spec.max_completed_turns:
            evicted = self._visible.pop(0)
            if evicted.action is not None:
                self._evicted_actions.append((evicted.step, evicted.action))
        self._current = next_image

    def _elide_action(self, action: str) -> str:
        limit = self.spec.max_previous_action_chars
        if len(action) <= limit:
            return action
        cut = limit - 10
        return f"{action[:cut]}…[+{len(action) - cut} chars]"

    def _instruction(self, instruction: str) -> str:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty text")
        previous_actions = "\n".join(
            f"Step {step}: {self._elide_action(action)}"
            for step, action in self._evicted_actions
        )
        return self.spec.instruction_template.format(
            instruction=instruction,
            previous_actions=previous_actions or "None",
        )

    def render_prompt(self, *, instruction: str) -> list[dict[str, Any]]:
        if not self._started:
            raise RuntimeError("HarnessRenderer.render_prompt before start")
        instruction_text = self._instruction(instruction)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]
        images: list[tuple[Frame, str | None]] = [
            (turn.image, turn.history_text) for turn in self._visible
        ]
        images.append((self._current, None))
        for index, (image, assistant) in enumerate(images):
            content: list[dict[str, Any]] = [{"type": "image", "image": image}]
            if index == 0:
                content.append({"type": "text", "text": instruction_text})
            messages.append({"role": "user", "content": content})
            if assistant is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant}],
                    }
                )
        return messages


def render_sft_records[Frame](
    renderer: HarnessRenderer[Frame],
    *,
    instruction: str,
    turns: Sequence[TrainingTurn[Frame]],
) -> list[list[dict[str, Any]]]:
    if not turns:
        raise ValueError("training turns cannot be empty")
    for turn in turns:
        if turn.image is None:
            raise ValueError("training image cannot be None")
        if not isinstance(turn.assistant, str) or not turn.assistant.strip():
            raise ValueError("training assistant must be non-empty text")
        _history_text(turn.assistant, turn.action)
    renderer.start(turns[0].image)
    records: list[list[dict[str, Any]]] = []
    for index, turn in enumerate(turns):
        messages = renderer.render_prompt(instruction=instruction)
        for message in messages:
            if message["role"] == "assistant":
                message["loss"] = False
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": turn.assistant}],
            }
        )
        records.append(messages)
        if index + 1 < len(turns):
            renderer.complete(
                assistant=turn.assistant,
                action=turn.action,
                next_image=turns[index + 1].image,
            )
    return records
