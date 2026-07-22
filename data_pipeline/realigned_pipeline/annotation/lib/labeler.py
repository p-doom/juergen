#!/usr/bin/env python3
"""Model-agnostic VLM labeler client (OpenAI-compatible).

One thin client used by every annotation step, selected entirely by env so we
can "iterate on frontier, distill to local later" without code changes:

    LABELER_MODEL      (default: Kimi-K2.6)
    LABELER_BASE_URL   (default: $AZURE_OPENAI_ENDPOINT  -> Azure /openai/v1/ surface)
    LABELER_API_KEY    (default: $AZURE_OPENAI_API_KEY)
    LABELER_MAX_TOKENS (default: 64000; raise if a verbose model truncates)

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

import contextlib
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realigned_pipeline.lib.common import extract_json_object, image_data_url

# Served from the same Azure mihir-4710 /openai/v1/ surface; pass the model by
# name. (Earlier iterations used the "gpt-5.5" deployment on this resource.)
DEFAULT_LABELER_MODEL = "Kimi-K2.6"


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
    # Transient failures (429 / 5xx / timeouts) get their OWN deeper ladder:
    # long sequential call chains (day-scope annotation) die mid-stream on the
    # shallow default above, and Azure Kimi throws 429s even under nominal
    # quota. Exponential with jitter, honoring Retry-After when the server
    # sends one, capped at transient_backoff_max_s per wait.
    transient_retries: int = int(os.environ.get("LABELER_TRANSIENT_RETRIES") or 8)
    transient_backoff_base_s: float = 2.0
    transient_backoff_max_s: float = 120.0
    # gpt-5.x reasoning models reject temperature != 1 and use
    # max_completion_tokens. Leave temperature None to omit it; set
    # reasoning_effort to e.g. "low"/"medium"/"high" when supported.
    temperature: float | None = None
    reasoning_effort: str | None = os.environ.get("LABELER_REASONING_EFFORT") or None
    # Reservation for the answer + in-band chain-of-thought (Kimi-K2.6 returns
    # reasoning in `reasoning_content`, which still counts against this cap).
    # RESERVED against the model's 262K context, so it competes with the ~150
    # input frames. Observed describe/extract usage maxed at ~21K, so 32K is
    # ample and leaves the most room for frames. Override via env.
    max_completion_tokens: int = int(os.environ.get("LABELER_MAX_TOKENS") or 32000)

    @classmethod
    def from_env(cls, **overrides: Any) -> LabelerConfig:
        cfg = cls(model=labeler_model(), base_url=labeler_base_url(), api_key=labeler_api_key())
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


class ContentFilteredError(RuntimeError):
    """The provider's content filter blocked this call (deterministic for
    these inputs — retrying cannot help). Methods decide how to degrade."""


@dataclass
class LabelResult:
    """One labeler call's output. ``content`` is the model's answer (the prose /
    JSON); ``reasoning`` is its chain-of-thought (Kimi returns it in a separate
    ``reasoning_content`` field — captured here so the viewer can show it).
    ``text`` is what downstream parsing should use (content, or reasoning if the
    model put everything there and content came back empty)."""

    content: str
    reasoning: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] | None = None
    model: str = ""

    @property
    def text(self) -> str:
        return self.content if self.content.strip() else self.reasoning


def _reasoning_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".reasoning.txt")


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def content_hash(model: str, system: str, user_text: str, image_payloads: list[str]) -> str:
    h = hashlib.sha256()
    for part in (model, system, user_text):
        h.update(part.encode())
        h.update(b"\x00")
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
        from openai import OpenAI  # noqa: PLC0415 - local import: optional dep

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    # -- full (content + reasoning + meta) ----------------------------------

    def call_full(
        self,
        system: str,
        user_text: str,
        images: list[Path | str] | None = None,
        image_labels: list[str] | None = None,
        cache_path: Path | None = None,
        no_cache: bool = False,
        max_completion_tokens: int | None = None,
    ) -> LabelResult:
        """One call. The answer is cached to ``cache_path`` (raw text), the
        chain-of-thought to ``<stem>.reasoning.txt``, and finish_reason/usage to
        ``<stem>.meta.json`` — so re-runs never re-spend tokens and the inspector
        can show the thinking. Returns content + reasoning + meta together.

        ``max_completion_tokens`` overrides the config reservation for THIS
        call (methods with many small calls keep a small budget so the TPM
        governor sees honest numbers); if the model spends the whole budget on
        reasoning (finish_reason=length, empty content) the call is retried
        with a doubled budget up to the config reservation — a truncated
        chain-of-thought is never cached as an answer."""
        image_urls = [image_data_url(Path(p)) if not str(p).startswith("data:") else str(p)
                      for p in (images or [])]

        if cache_path and cache_path.exists() and not no_cache:
            content = cache_path.read_text()
            if content.strip():
                meta = {}
                mp = _meta_path(cache_path)
                if mp.exists():
                    try:
                        meta = json.loads(mp.read_text())
                    except Exception:
                        meta = {}
                cached_model = str(meta.get("model", "")) if meta else ""
                # Safety net: the cache path has no model in its key, so a cached
                # response from a DIFFERENT model would otherwise be served
                # silently (e.g. K2.6 answers returned for a K2.5 run). Refuse it
                # and re-call. Keep separate run dirs per model so this never even
                # triggers; this just makes a mix-up loud instead of wrong.
                if cached_model and cached_model != self.config.model:
                    print(f"  [labeler] cache model mismatch "
                          f"({cached_model!r} != {self.config.model!r}); re-calling "
                          f"({cache_path.name}).")
                else:
                    rp = _reasoning_path(cache_path)
                    reasoning = rp.read_text() if rp.exists() else ""
                    return LabelResult(content=content, reasoning=reasoning,
                                       finish_reason=str(meta.get("finish_reason", "")),
                                       usage=meta.get("usage"),
                                       model=str(meta.get("model", self.config.model)))

        # Interleave a text label before each image when provided (frames stay
        # unmodified — no burned-in overlay occluding the UI). The v2 describe
        # pass sends no labels (frame-index anchoring was dropped).
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for i, url in enumerate(image_urls):
            if image_labels and i < len(image_labels) and image_labels[i]:
                content_parts.append({"type": "text", "text": image_labels[i]})
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content_parts},
        ]

        budget = int(max_completion_tokens or self.config.max_completion_tokens)
        budget_cap = max(budget, self.config.max_completion_tokens)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        tag = cache_path.name if cache_path else "call"
        last_err: str | None = None
        hard_left = self.config.retries
        transient_left = self.config.transient_retries
        transient_n = 0
        cap_resampled = False
        while True:
            kwargs["max_completion_tokens"] = budget
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if "content_filter" in last_err:
                    # Azure blocked the request/response for THESE inputs —
                    # deterministic; surface it as its own type immediately.
                    raise ContentFilteredError(last_err) from exc
                wait = self._transient_wait(exc, transient_n)
                if wait is not None:
                    if transient_left <= 0:
                        raise RuntimeError(
                            f"labeler call failed after {self.config.transient_retries} "
                            f"transient retries: {last_err}") from exc
                    transient_left -= 1
                    transient_n += 1
                    print(f"  [labeler] transient error ({last_err}); retrying in {wait:.0f}s ({tag}).",
                          flush=True)
                    time.sleep(wait)
                    continue
                # Non-transient: the deployment may reject optional params —
                # strip them, then give up after the (shallow) hard-retry count.
                kwargs.pop("temperature", None)
                kwargs.pop("reasoning_effort", None)
                if hard_left <= 0:
                    raise RuntimeError(f"labeler call failed after retries: {last_err}") from exc
                hard_left -= 1
                time.sleep(1.5)
                continue

            choice = resp.choices[0]
            msg = choice.message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            if finish_reason == "length":
                # The budget ran out — on reasoning (empty content) or mid-answer
                # (truncated content). Either way the response is unusable:
                # retry with a doubled budget and NEVER cache it. Only at the
                # cap does a truncated-but-present answer fall through (cached,
                # loud below) — an empty one is an error.
                if budget < budget_cap:
                    budget = min(budget_cap, budget * 2)
                    print(f"  [labeler] completion hit its budget "
                          f"({'reasoning burn' if not content else 'truncated answer'}); "
                          f"retrying with max_completion_tokens={budget} ({tag}).", flush=True)
                    continue
                if not content:
                    # Reasoning spiral all the way to the cap. It's stochastic
                    # (temperature > 0): one fresh sample usually escapes it.
                    if not cap_resampled:
                        cap_resampled = True
                        print(f"  [labeler] reasoning spiral at the {budget} cap; "
                              f"re-sampling once ({tag}).", flush=True)
                        continue
                    raise RuntimeError(
                        f"completion exhausted on reasoning at max_completion_tokens={budget} "
                        f"(finish_reason=length, empty content; raise LABELER_MAX_TOKENS)")
            if not content and not reasoning:
                last_err = "empty completion"
                if hard_left <= 0:
                    raise RuntimeError(f"labeler call failed after retries: {last_err}")
                hard_left -= 1
                time.sleep(1.5)
                continue

            usage = getattr(resp, "usage", None)
            usage_d = usage.model_dump() if hasattr(usage, "model_dump") else (dict(usage) if usage else None)
            if finish_reason == "length":
                print(f"  [labeler] WARNING: response hit max_completion_tokens "
                      f"({budget}); raise LABELER_MAX_TOKENS ({tag}).")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(content or reasoning)
                _reasoning_path(cache_path).write_text(reasoning)
                _meta_path(cache_path).write_text(json.dumps(
                    {"finish_reason": finish_reason, "usage": usage_d, "model": self.config.model}, indent=2))
            return LabelResult(content=content, reasoning=reasoning, finish_reason=finish_reason,
                               usage=usage_d, model=self.config.model)

    def _transient_wait(self, exc: Exception, attempt: int) -> float | None:
        """Backoff seconds if ``exc`` is transient (429/408/5xx, timeout,
        connection drop), else None. Full jitter; a server Retry-After floors
        the wait when present."""
        import openai  # noqa: PLC0415 - local import: optional dep

        status = getattr(exc, "status_code", None)
        transient = isinstance(exc, openai.APIConnectionError) or (
            isinstance(status, int) and (status in (408, 429) or status >= 500))
        if not transient:
            return None
        wait = min(self.config.transient_backoff_max_s,
                   self.config.transient_backoff_base_s * (2 ** attempt))
        wait *= 0.5 + random.random()  # full jitter in [0.5x, 1.5x)
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            with contextlib.suppress(TypeError, ValueError):
                wait = max(wait, float(headers.get("retry-after")))
        return wait

    # -- raw text -----------------------------------------------------------

    def call_text(
        self,
        system: str,
        user_text: str,
        images: list[Path | str] | None = None,
        image_labels: list[str] | None = None,
        cache_path: Path | None = None,
        no_cache: bool = False,
        max_completion_tokens: int | None = None,
    ) -> str:
        return self.call_full(system, user_text, images=images, image_labels=image_labels,
                              cache_path=cache_path, no_cache=no_cache,
                              max_completion_tokens=max_completion_tokens).text

    # -- json ---------------------------------------------------------------

    def call_json(
        self,
        system: str,
        user_text: str,
        images: list[Path | str] | None = None,
        image_labels: list[str] | None = None,
        cache_path: Path | None = None,
        no_cache: bool = False,
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self.call_json_full(system, user_text, images=images, image_labels=image_labels,
                                   cache_path=cache_path, no_cache=no_cache,
                                   max_completion_tokens=max_completion_tokens)[0]

    def call_json_full(
        self,
        system: str,
        user_text: str,
        images: list[Path | str] | None = None,
        image_labels: list[str] | None = None,
        cache_path: Path | None = None,
        no_cache: bool = False,
        max_completion_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LabelResult]:
        """Like ``call_json`` but also returns the full LabelResult (reasoning,
        raw content, meta). Re-calls once if a cached response is unparseable."""
        if cache_path and cache_path.exists() and not no_cache:
            try:
                res = self.call_full(system, user_text, images=images, image_labels=image_labels,
                                     cache_path=cache_path, no_cache=False,
                                     max_completion_tokens=max_completion_tokens)
                return extract_json_object(res.text), res
            except Exception:
                no_cache = True
        res = self.call_full(system, user_text, images=images, image_labels=image_labels,
                            cache_path=cache_path, no_cache=no_cache,
                            max_completion_tokens=max_completion_tokens)
        try:
            return extract_json_object(res.text), res
        except ValueError:
            # Structurally broken JSON (missing delimiter etc.) happens on a
            # small % of samples; one re-sample usually fixes it — and the
            # re-call overwrites the broken cached response.
            print(f"  [labeler] unparseable JSON response; re-sampling once "
                  f"({cache_path.name if cache_path else 'call'}).", flush=True)
        res = self.call_full(system, user_text, images=images, image_labels=image_labels,
                            cache_path=cache_path, no_cache=True,
                            max_completion_tokens=max_completion_tokens)
        return extract_json_object(res.text), res


def smoke_test() -> None:
    """Tiny connectivity + image-support probe (one paid call)."""
    import argparse  # noqa: PLC0415 - smoke-test only
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
