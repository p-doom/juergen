"""Teacher serving contract and OpenAI-compatible absolute-action client."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION
from experiments.teacher_sft.contracts import (
    ContractError,
    file_sha256,
    object_sha256,
    read_json,
)

_TOOL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_ALLOWED_ACTIONS = {
    "mouse_move",
    "move",
    "move_absolute",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
    "mouse_down",
    "mouse_up",
    "scroll",
    "hscroll",
    "key",
    "key_down",
    "key_up",
    "type",
    "wait",
    "terminate",
}


def load_teacher_spec(path: Path) -> dict[str, Any]:
    spec = read_json(path)
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("teacher spec must be a schema_version=1 object")
    if spec.get("backend") != "openai_chat":
        raise ContractError(
            "only the backend-neutral openai_chat wire contract is supported"
        )
    required_strings = ("base_url", "model_id", "model_revision", "system_prompt")
    for key in required_strings:
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ContractError(f"teacher spec requires non-empty {key}")
    if spec.get("action_space") != "native_absolute":
        raise ContractError("teacher action_space must be native_absolute")
    coordinate_space = spec.get("coordinate_space")
    if coordinate_space not in {"absolute_px", "absolute_grid"}:
        raise ContractError(
            "teacher coordinate_space must be absolute_px or absolute_grid"
        )
    if coordinate_space == "absolute_grid":
        grid = spec.get("coordinate_grid")
        if not isinstance(grid, int) or grid <= 1:
            raise ContractError("absolute_grid teacher requires coordinate_grid > 1")
    if spec.get("temperature", 0) != 0:
        raise ContractError(
            "Stage 4 teacher collection requires deterministic temperature=0"
        )
    spec["spec_sha256"] = file_sha256(path)
    spec["system_prompt_sha256"] = object_sha256(spec["system_prompt"])
    return spec


def _validate_actions(actions: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(actions, list) or not actions:
        raise ContractError("teacher response contains no native actions")
    result = []
    for raw in actions:
        if not isinstance(raw, dict):
            raise ContractError("native action is not an object")
        action = dict(raw.get("arguments", raw))
        kind = str(action.get("action", "")).strip().lower()
        if kind not in _ALLOWED_ACTIONS:
            raise ContractError(f"teacher emitted unsupported action: {kind!r}")
        action["action"] = kind
        if "coordinate" in action:
            action["coordinate_space"] = spec["coordinate_space"]
            if spec["coordinate_space"] == "absolute_grid":
                action["coordinate_grid"] = spec["coordinate_grid"]
        result.append(action)
    if any(action["action"] == "terminate" for action in result[:-1]):
        raise ContractError("terminate must be the last native action in a turn")
    return result


def parse_native_actions(text: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for match in _TOOL_RE.finditer(text):
        try:
            candidates.append(json.loads(match.group(1)))
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid teacher tool-call JSON: {exc}") from exc
    stripped = text.strip()
    if not candidates and stripped:
        try:
            candidates.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ContractError("teacher emitted neither tool calls nor JSON") from exc
    actions: list[dict[str, Any]] = []
    for candidate in candidates:
        values = candidate if isinstance(candidate, list) else [candidate]
        for value in values:
            if not isinstance(value, dict):
                raise ContractError("teacher tool call is not an object")
            if value.get("name") not in {None, "computer_use"}:
                raise ContractError(
                    f"unexpected teacher tool name: {value.get('name')!r}"
                )
            actions.append(value)
    return _validate_actions(actions, spec)


def _image_block(image_path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


class OpenAIChatTeacher:
    def __init__(
        self, spec: dict[str, Any], *, api_key: str = "none", timeout_s: float = 180.0
    ):
        self.spec = spec
        self.api_key = api_key
        self.timeout_s = timeout_s

    def act(
        self,
        *,
        instruction: str,
        image_path: Path,
        history: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        user_content: list[dict[str, Any]] = [_image_block(image_path)]
        if not history:
            user_content.insert(0, {"type": "text", "text": instruction})
        messages = [
            {"role": "system", "content": self.spec["system_prompt"]},
            *history,
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.spec["model_id"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(self.spec.get("max_tokens", 512)),
        }
        endpoint = self.spec["base_url"].rstrip("/") + "/v1/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ContractError(f"teacher request failed: {exc}") from exc
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ContractError("teacher response lacks choices[0].message") from exc
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        if tool_calls:
            actions = []
            for call in tool_calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if function.get("name") != "computer_use":
                    raise ContractError("teacher emitted a non-computer_use tool")
                arguments = function.get("arguments")
                try:
                    actions.append(
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
                except json.JSONDecodeError as exc:
                    raise ContractError(
                        f"teacher tool arguments are invalid JSON: {exc}"
                    ) from exc
            parsed = _validate_actions(actions, self.spec)
            raw = json.dumps(actions, ensure_ascii=False, separators=(",", ":"))
        else:
            raw = str(content)
            parsed = parse_native_actions(raw, self.spec)
        return raw, parsed
