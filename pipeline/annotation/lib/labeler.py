"""Fail-fast OpenAI-compatible client for Crowd-Cast annotation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.lib.common import image_data_url


@dataclass(frozen=True)
class LabelerConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: float = 300
    transient_retries: int = 8
    transient_backoff_max_s: float = 120
    max_completion_tokens: int = 32000

    @classmethod
    def from_env(cls, *, model: str) -> LabelerConfig:
        base_url = os.environ.get("LABELER_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("LABELER_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError("LABELER_BASE_URL and LABELER_API_KEY are required")
        max_tokens = int(os.environ.get("LABELER_MAX_TOKENS", "32000"))
        if max_tokens <= 0:
            raise ValueError("LABELER_MAX_TOKENS must be positive")
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_completion_tokens=max_tokens,
        )


@dataclass(frozen=True)
class LabelResult:
    content: str
    reasoning: str
    finish_reason: str
    usage: dict[str, Any]
    model: str


def _reasoning_path(path: Path) -> Path:
    return path.with_suffix(".reasoning.txt")


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _validate_usage(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("label response usage must be an object")
    total = value.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("label response usage.total_tokens must be positive")
    return value


class Labeler:
    def __init__(self, config: LabelerConfig) -> None:
        self.config = config
        from openai import OpenAI

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=0,
        )

    def _cached(self, path: Path, request_sha256: str) -> LabelResult:
        content = path.read_text()
        meta = json.loads(_meta_path(path).read_text())
        if not content.strip():
            raise ValueError(f"cached label response is empty: {path}")
        if meta.get("model") != self.config.model:
            raise ValueError(
                f"cached label model mismatch: {meta.get('model')!r} != {self.config.model!r}"
            )
        if meta.get("base_url") != self.config.base_url:
            raise ValueError(f"cached label base_url mismatch: {path}")
        if meta.get("request_sha256") != request_sha256:
            raise ValueError(f"cached label request mismatch: {path}")
        if meta.get("finish_reason") != "stop":
            raise ValueError(f"cached label finish_reason is not stop: {path}")
        return LabelResult(
            content=content,
            reasoning=_reasoning_path(path).read_text(),
            finish_reason="stop",
            usage=_validate_usage(meta.get("usage")),
            model=self.config.model,
        )

    def _transient_wait(self, exc: Exception, attempt: int) -> float | None:
        import openai

        status = getattr(exc, "status_code", None)
        transient = isinstance(exc, openai.APIConnectionError) or (
            isinstance(status, int) and (status in (408, 429) or status >= 500)
        )
        if not transient:
            return None
        wait = min(self.config.transient_backoff_max_s, 2**attempt)
        wait *= 0.5 + random.random()
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers is not None:
            with contextlib.suppress(TypeError, ValueError):
                wait = max(wait, float(headers.get("retry-after")))
        return wait

    def call_full(
        self,
        system: str,
        user_text: str,
        *,
        images: list[Path | str],
        image_labels: list[str],
        cache_path: Path,
    ) -> LabelResult:
        if len(images) != len(image_labels):
            raise ValueError("every annotation image requires one frame label")
        parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image, label in zip(images, image_labels, strict=True):
            url = str(image)
            if not url.startswith("data:"):
                url = image_data_url(Path(image))
            parts.extend(
                (
                    {"type": "text", "text": label},
                    {"type": "image_url", "image_url": {"url": url}},
                )
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": parts},
        ]
        request_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "model": self.config.model,
                    "base_url": self.config.base_url,
                    "max_completion_tokens": self.config.max_completion_tokens,
                    "messages": messages,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if cache_path.exists():
            return self._cached(cache_path, request_sha256)
        response = None
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_completion_tokens=self.config.max_completion_tokens,
                )
                break
            except Exception as exc:
                wait = self._transient_wait(exc, attempt)
                if wait is None or attempt == self.config.transient_retries:
                    raise
                time.sleep(wait)
        if response is None:
            raise AssertionError("labeler retry loop produced no response")
        if len(response.choices) != 1:
            raise ValueError("label response must contain exactly one choice")
        response_model = str(getattr(response, "model", "") or "")
        if response_model != self.config.model:
            raise ValueError(
                f"label response model mismatch: {response_model!r} != {self.config.model!r}"
            )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        reasoning = (getattr(choice.message, "reasoning_content", None) or "").strip()
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        if finish_reason != "stop":
            raise ValueError(f"unsupported label finish_reason: {finish_reason!r}")
        if not content:
            raise ValueError("label response content is empty")
        usage = getattr(response, "usage", None)
        if usage is None or not hasattr(usage, "model_dump"):
            raise TypeError("label response has no structured usage")
        usage_dict = _validate_usage(usage.model_dump())
        result = LabelResult(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage_dict,
            model=response_model,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content)
        _reasoning_path(cache_path).write_text(reasoning)
        _meta_path(cache_path).write_text(
            json.dumps(
                {
                    "finish_reason": finish_reason,
                    "usage": usage_dict,
                    "model": self.config.model,
                    "base_url": self.config.base_url,
                    "request_sha256": request_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return result

    def call_json_full(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], LabelResult]:
        result = self.call_full(*args, **kwargs)
        parsed = json.loads(result.content)
        if not isinstance(parsed, dict):
            raise TypeError("label response JSON must be an object")
        return parsed, result
