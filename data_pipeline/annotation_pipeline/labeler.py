#!/usr/bin/env python3
"""Model-agnostic VLM labeler client (OpenAI-compatible).

One thin client used by every annotation step, selected entirely by env so we
can "iterate on frontier, distill to local later" without code changes:

    LABELER_MODEL      (default: gpt-5.5-2026-04-24)
    LABELER_BASE_URL   (default: $AZURE_OPENAI_ENDPOINT  -> Azure /openai/v1/ surface)
    LABELER_API_KEY    (default: $AZURE_OPENAI_API_KEY)

The Azure resource exposes the OpenAI-compatible ``/openai/v1/`` surface, so the
stock ``openai`` client works with ``base_url`` set to the endpoint and the
model passed by name — no Azure SDK, no api-version. To distill to the local
sglang Qwen, point the three env vars at ``http://localhost:8011/v1`` /
``Qwen/Qwen3.6-27B``.

Every call is cached to a response file (raw text), so re-running a step never
re-spends tokens. ``call_json`` validates/repairs JSON via
``common.extract_json_object``.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotation_pipeline.common import extract_json_object, image_data_url

# Azure deployment name on the mihir-4710 resource is "gpt-5.5" (the dated model
# id from /models, gpt-5.5-2026-04-24, is NOT a deployment and 404s).
DEFAULT_LABELER_MODEL = "gpt-5.5"


def labeler_model() -> str:
    return os.environ.get("LABELER_MODEL") or DEFAULT_LABELER_MODEL


def labeler_base_url() -> str:
    url = os.environ.get("LABELER_BASE_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT") or ""
    return url.rstrip("/") if url else url


def labeler_api_key() -> str:
    return os.environ.get("LABELER_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or ""


@dataclass
class LabelerConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: float = 300.0
    retries: int = 2
    # gpt-5.x reasoning models reject temperature != 1 and use
    # max_completion_tokens. Leave temperature None to omit it; set
    # reasoning_effort to e.g. "low"/"medium"/"high" when supported.
    temperature: float | None = None
    reasoning_effort: str | None = os.environ.get("LABELER_REASONING_EFFORT") or None
    max_completion_tokens: int = 8192

    @classmethod
    def from_env(cls, **overrides: Any) -> "LabelerConfig":
        cfg = cls(model=labeler_model(), base_url=labeler_base_url(), api_key=labeler_api_key())
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


def content_hash(model: str, system: str, user_text: str, image_payloads: list[str]) -> str:
    h = hashlib.sha256()
    h.update(model.encode()); h.update(b"\x00")
    h.update(system.encode()); h.update(b"\x00")
    h.update(user_text.encode()); h.update(b"\x00")
    for p in image_payloads:
        # hash only a prefix of each (large) data url for speed; collisions on a
        # 4 KB prefix of distinct JPEGs are vanishingly unlikely here.
        h.update(hashlib.sha256(p[:4096].encode()).digest())
    return h.hexdigest()[:16]


class Labeler:
    def __init__(self, config: LabelerConfig | None = None) -> None:
        self.config = config or LabelerConfig.from_env()
        if not (self.config.base_url and self.config.api_key and self.config.model):
            raise RuntimeError(
                "Labeler needs model + base_url + api_key. Set LABELER_MODEL / "
                "LABELER_BASE_URL / LABELER_API_KEY (or AZURE_OPENAI_ENDPOINT / "
                "AZURE_OPENAI_API_KEY)."
            )
        from openai import OpenAI  # local import: optional dep

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    # -- raw text -----------------------------------------------------------

    def call_text(
        self,
        system: str,
        user_text: str,
        images: list[Path | str] | None = None,
        image_labels: list[str] | None = None,
        cache_path: Path | None = None,
        no_cache: bool = False,
    ) -> str:
        image_urls = [image_data_url(Path(p)) if not str(p).startswith("data:") else str(p)
                      for p in (images or [])]

        if cache_path and cache_path.exists() and not no_cache:
            text = cache_path.read_text()
            if text.strip():
                return text

        # Interleave a text label (e.g. the frame's timestamp) before each image
        # when provided, so time is given as text — frames stay unmodified (no
        # burned-in overlay occluding the UI).
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for i, url in enumerate(image_urls):
            if image_labels and i < len(image_labels) and image_labels[i]:
                content.append({"type": "text", "text": image_labels[i]})
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_completion_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        last_err: str | None = None
        for attempt in range(self.config.retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                text = (msg.content or "").strip()
                if not text:
                    text = (getattr(msg, "reasoning_content", None) or "").strip()
                if not text:
                    raise RuntimeError("empty completion")
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(text)
                return text
            except Exception as exc:  # noqa: BLE001 - retry transient/4xx-param errors
                last_err = f"{type(exc).__name__}: {exc}"
                # On the first failure, retry once without optional params in
                # case the deployment rejects temperature/reasoning_effort.
                kwargs.pop("temperature", None)
                kwargs.pop("reasoning_effort", None)
                if attempt < self.config.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"labeler call failed after retries: {last_err}")

    # -- json ---------------------------------------------------------------

    def call_json(
        self,
        system: str,
        user_text: str,
        images: list[Path | str] | None = None,
        image_labels: list[str] | None = None,
        cache_path: Path | None = None,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        # If a cached response exists but is unparseable, re-call once.
        if cache_path and cache_path.exists() and not no_cache:
            try:
                return extract_json_object(cache_path.read_text())
            except Exception:  # noqa: BLE001
                no_cache = True
        text = self.call_text(system, user_text, images=images, image_labels=image_labels,
                              cache_path=cache_path, no_cache=no_cache)
        return extract_json_object(text)


def smoke_test() -> None:
    """Tiny connectivity + image-support probe (one paid call)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None, help="optional image to test vision")
    args = ap.parse_args()
    lab = Labeler()
    print(f"model={lab.config.model} base_url={lab.config.base_url}")
    if args.image:
        out = lab.call_text("You are concise.", "Reply with the single word OK if you can see an image.",
                            images=[args.image])
    else:
        out = lab.call_text("You are concise.", "Reply with the single word OK.")
    print("response:", out[:200])


if __name__ == "__main__":
    smoke_test()
