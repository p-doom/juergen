"""Inference-endpoint readiness, export validation, and cache placement.

Four defects, all of them "the server said READY and it was lying":

* **Defect #8** — an exported ``config.json`` missing ``architectures`` made vLLM
  auto-resolve to ``--runner pooling / --convert embed``. That is an *embedding*
  server: every ``/v1/chat/completions`` returns 404. The readiness gate polled
  ``/v1/models``, which returns 200 in pooling mode, and printed READY.
  Fixes: :func:`validate_export_config` refuses to serve such a checkpoint at
  all, and :func:`preflight_chat_completion` is a real chat completion, so a
  pooling server fails the gate instead of passing it.
* **Defect #7** — ``deployment.num_infer_gpus`` OVERRIDES ``inference.parallel.dp``
  (``rl.py:544-549``). Editing ``dp`` alone leaves ranks invalid, every request
  to them 400s, and because routing is session-hashed per group, whole *groups*
  are destroyed (20-33% loss). :func:`validate_deployment_parallelism` makes the
  inconsistency an error before a single GPU is allocated.
* **Defect #20** — vLLM's compile cache on NFS ``/fast/home`` throws
  ``Errno 121 Remote I/O error``. :func:`compile_cache_env` places every cache on
  ``/fast/project`` and refuses a ``/fast/home`` target.
* **Defect #9** — a probe that used ``return_exceptions=True`` and then filtered
  the exceptions out yielded ``success=0/0`` and wrote **0.0 as a result**.
  Nothing in this module ever returns a number derived from an empty sample.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rft.errors import (
    DeploymentConfigError,
    ExportConfigError,
    MissingFieldError,
    PreflightError,
    SchemaError,
)

# ---------------------------------------------------------------------------
# Defect #20: cache placement
# ---------------------------------------------------------------------------

#: Filesystems that must never hold a compile/inductor/triton cache. ``/fast/home``
#: is a 95G NFS mount that has thrown ``OSError: [Errno 121] Remote I/O error``
#: under vLLM's parallel compile-cache writes, and has run out of space twice.
FORBIDDEN_CACHE_PREFIXES: tuple[str, ...] = ("/fast/home", "/home")

#: Every env var that has been observed to place a cache on disk for the
#: vLLM/torch/triton stack.
CACHE_ENV_VARS: tuple[str, ...] = (
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_HOME",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "FLASHINFER_WORKSPACE_BASE",
)


def _is_forbidden(path: str | Path) -> bool:
    resolved = str(Path(path).resolve())
    return any(
        resolved == pre or resolved.startswith(pre.rstrip("/") + "/")
        for pre in FORBIDDEN_CACHE_PREFIXES
    )


def compile_cache_env(cache_root: str | Path) -> dict[str, str]:
    """Build the cache env for a served model, rooted at ``cache_root``.

    Raises:
        SchemaError: ``cache_root`` is on a forbidden filesystem (defect #20).
    """
    root = Path(cache_root)
    if _is_forbidden(root):
        raise SchemaError(
            f"refusing to place compile caches at {root}: NFS home throws "
            "[Errno 121] Remote I/O error under vLLM's cache writes (defect #20). "
            "Use a /fast/project path."
        )
    env: dict[str, str] = {}
    per_var = {
        "VLLM_CACHE_ROOT": root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
        "TRITON_CACHE_DIR": root / "triton",
        "TORCH_HOME": root / "torch",
        "XDG_CACHE_HOME": root / "xdg",
        "FLASHINFER_WORKSPACE_BASE": root / "flashinfer",
    }
    for var, target in per_var.items():
        target.mkdir(parents=True, exist_ok=True)
        env[var] = str(target)
    return env


def assert_caches_off_home(env: Mapping[str, str] | None = None) -> None:
    """Raise if any cache env var currently points at a forbidden filesystem.

    Called by the sampler before launching a server, so that a stale
    ``~/.cache`` inherited from the login shell cannot poison the run.
    """
    source = os.environ if env is None else env
    offenders = {
        var: source[var]
        for var in CACHE_ENV_VARS
        if source.get(var) and _is_forbidden(source[var])
    }
    if offenders:
        raise SchemaError(
            "cache env vars point at NFS home (defect #20): "
            + ", ".join(f"{k}={v}" for k, v in sorted(offenders.items()))
        )


# ---------------------------------------------------------------------------
# Defect #8: export validation
# ---------------------------------------------------------------------------

#: A generative HF checkpoint must declare at least one causal-LM-ish
#: architecture. vLLM's runner auto-resolution keys off this; an empty or absent
#: ``architectures`` list silently selects the pooling runner.
_POOLING_SMELL = ("Model", "Embedding", "ForSequenceClassification")


@dataclass(frozen=True)
class ExportAudit:
    path: Path
    architectures: tuple[str, ...]
    model_type: str
    has_weights: bool
    has_tokenizer: bool
    has_chat_template: bool

    def describe(self) -> str:
        return (
            f"{self.path}: architectures={list(self.architectures)} "
            f"model_type={self.model_type!r} weights={self.has_weights} "
            f"tokenizer={self.has_tokenizer} chat_template={self.has_chat_template}"
        )


def validate_export_config(export_dir: str | Path) -> ExportAudit:
    """Refuse to serve an export that vLLM would resolve to a pooling runner.

    Checks, all of them fatal:
      * ``config.json`` exists and parses;
      * ``architectures`` is present and non-empty (defect #8 — this is the one
        that produced an embedding server whose ``/v1/models`` returned 200);
      * the declared architecture is not obviously an encoder/pooling class;
      * ``model_type`` is present;
      * weights are present (``*.safetensors`` or ``*.bin``) — an export that
        wrote only a config is a *failed* export, and its evals were secretly
        scoring something else;
      * a tokenizer is present.

    Returns an :class:`ExportAudit` for the run's diagnostics.
    """
    d = Path(export_dir)
    cfg_path = d / "config.json"
    if not cfg_path.is_file():
        raise ExportConfigError(f"{d} has no config.json; it is not a servable export")
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ExportConfigError(f"{cfg_path} is not valid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ExportConfigError(f"{cfg_path} does not contain a JSON object")

    archs = cfg.get("architectures")
    if archs is None:
        raise ExportConfigError(
            f"{cfg_path} has no `architectures` key. vLLM would auto-resolve this to "
            "`--runner pooling / --convert embed` and serve an EMBEDDING model whose "
            "/v1/chat/completions returns 404, while /v1/models still returns 200 "
            "(defect #8)."
        )
    if not isinstance(archs, list) or not archs or not all(isinstance(a, str) for a in archs):
        raise ExportConfigError(
            f"{cfg_path} `architectures` must be a non-empty list of strings, got {archs!r}"
        )
    for a in archs:
        if any(a.endswith(s) for s in _POOLING_SMELL):
            raise ExportConfigError(
                f"{cfg_path} declares architecture {a!r}, which vLLM treats as a "
                "pooling/encoder model. A generative checkpoint must declare a "
                "causal-LM architecture (e.g. *ForConditionalGeneration / *ForCausalLM)."
            )
    model_type = cfg.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ExportConfigError(f"{cfg_path} has no usable `model_type`")

    has_weights = any(d.glob("*.safetensors")) or any(d.glob("*.bin"))
    if not has_weights:
        raise ExportConfigError(
            f"{d} contains a config but no weight shards (*.safetensors / *.bin). "
            "An export with no weights silently serves whatever the base path resolves to."
        )
    has_tokenizer = any((d / n).is_file() for n in ("tokenizer.json", "tokenizer_config.json"))
    if not has_tokenizer:
        raise ExportConfigError(f"{d} has no tokenizer files")
    tok_cfg_path = d / "tokenizer_config.json"
    has_chat_template = bool((d / "chat_template.jinja").is_file())
    if not has_chat_template and tok_cfg_path.is_file():
        try:
            has_chat_template = "chat_template" in json.loads(tok_cfg_path.read_text())
        except json.JSONDecodeError:
            has_chat_template = False

    return ExportAudit(
        path=d,
        architectures=tuple(archs),
        model_type=model_type,
        has_weights=has_weights,
        has_tokenizer=has_tokenizer,
        has_chat_template=has_chat_template,
    )


def assert_export_differs_from_base(
    export_dir: str | Path, base_dir: str | Path, *, n_bytes: int = 1 << 20
) -> None:
    """Raise if an export's weights are byte-identical to the base model's.

    Guards the prime-rl LoRA-export bug: ``save_adapter_separately`` defaulted to
    False, so ``weights/step_N`` exports were byte-identical to the base and every
    eval of them secretly scored the base model. Cheap to check, catastrophic to
    miss.
    """
    export = Path(export_dir)
    base = Path(base_dir)
    exp_shards = sorted(p.name for p in export.glob("*.safetensors"))
    if not exp_shards:
        raise ExportConfigError(f"{export} has no safetensors shards to compare")
    identical: list[str] = []
    for name in exp_shards:
        b = base / name
        if not b.is_file():
            continue
        e = export / name
        if e.stat().st_size != b.stat().st_size:
            continue
        with e.open("rb") as fe, b.open("rb") as fb:
            if fe.read(n_bytes) == fb.read(n_bytes) and _tail_equal(e, b, n_bytes):
                identical.append(name)
    if identical and len(identical) == len([n for n in exp_shards if (base / n).is_file()]):
        raise ExportConfigError(
            f"{export} weights appear byte-identical to base {base} "
            f"({len(identical)} shard(s) match head+tail). The fine-tuned delta was "
            "never merged into the export - evaluating this scores the BASE model."
        )


def _tail_equal(a: Path, b: Path, n: int) -> bool:
    size = a.stat().st_size
    off = max(0, size - n)
    with a.open("rb") as fa, b.open("rb") as fb:
        fa.seek(off)
        fb.seek(off)
        return fa.read(n) == fb.read(n)


# ---------------------------------------------------------------------------
# Defect #7: deployment / inference parallelism consistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelismPlan:
    """A validated inference-parallelism plan."""

    num_infer_gpus: int
    dp: int
    tp: int
    pp: int = 1

    @property
    def gpus_required(self) -> int:
        return self.dp * self.tp * self.pp

    def describe(self) -> str:
        return (
            f"num_infer_gpus={self.num_infer_gpus} = dp={self.dp} x tp={self.tp} x pp={self.pp}"
        )


def validate_deployment_parallelism(
    *, num_infer_gpus: int, dp: int, tp: int, pp: int = 1
) -> ParallelismPlan:
    """Assert the deployment GPU count and the inference parallel degrees agree.

    ``deployment.num_infer_gpus`` is authoritative in prime-rl: it OVERRIDES
    ``inference.parallel.dp`` (``rl.py:544-549``). So editing ``dp`` alone
    produces a config whose extra data-parallel ranks are never started; every
    request routed to them 400s, and because routing is session-hashed per rollout
    group, a 400 does not lose one rollout — it destroys the whole group
    (20-33% observed loss).

    Raises:
        DeploymentConfigError: if ``dp * tp * pp != num_infer_gpus``.
    """
    for name, value in (("num_infer_gpus", num_infer_gpus), ("dp", dp), ("tp", tp), ("pp", pp)):
        if not isinstance(value, int) or value < 1:
            raise DeploymentConfigError(f"{name} must be a positive int, got {value!r}")
    plan = ParallelismPlan(num_infer_gpus=num_infer_gpus, dp=dp, tp=tp, pp=pp)
    if plan.gpus_required != num_infer_gpus:
        raise DeploymentConfigError(
            f"inconsistent inference parallelism: dp*tp*pp = {plan.gpus_required} but "
            f"deployment.num_infer_gpus = {num_infer_gpus}. num_infer_gpus WINS "
            "(rl.py:544-549), so the surplus ranks would never start and every request "
            "routed to them would 400 - taking its whole session-hashed rollout group "
            "with it (defect #7). Set both, consistently."
        )
    return plan


# ---------------------------------------------------------------------------
# Real chat-completion preflight
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """Outcome of a real chat-completion preflight against a served endpoint."""

    base_url: str
    model: str
    attempts: int
    elapsed_s: float
    completion_preview: str
    served_models: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        w = ("\n  warnings: " + "; ".join(self.warnings)) if self.warnings else ""
        return (
            f"preflight OK: {self.base_url} model={self.model!r} after {self.attempts} "
            f"attempt(s) in {self.elapsed_s:.1f}s; completion starts "
            f"{self.completion_preview!r}{w}"
        )


def preflight_chat_completion(
    *,
    base_url: str,
    model: str,
    timeout_s: float = 900.0,
    poll_interval_s: float = 5.0,
    request_timeout_s: float = 120.0,
    prompt: str = "Reply with the single word: ready",
    max_tokens: int = 8,
    http_post: Any = None,
    http_get: Any = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> PreflightResult:
    """Block until the endpoint answers a REAL chat completion, or raise.

    This replaces polling ``/v1/models``. A pooling-runner vLLM (defect #8)
    answers ``/v1/models`` with 200 and ``/v1/chat/completions`` with 404, so only
    an actual completion proves the endpoint can serve rollouts.

    The HTTP callables are injected so tests can drive every failure mode without
    a GPU. Defaults use ``requests``.

    Raises:
        PreflightError: the deadline passed without a successful, non-empty
            completion. The error message carries the last status/body seen —
            never a bare timeout.
    """
    if http_post is None or http_get is None:  # pragma: no cover - exercised in prod
        import requests

        http_post = http_post or (
            lambda url, json_body, timeout: _requests_adapter(
                requests.post(url, json=json_body, timeout=timeout)
            )
        )
        http_get = http_get or (
            lambda url, timeout: _requests_adapter(requests.get(url, timeout=timeout))
        )

    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = monotonic()
    attempts = 0
    last: str = "no attempt completed"
    while True:
        attempts += 1
        try:
            status, payload = http_post(url, body, request_timeout_s)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            last = f"{type(exc).__name__}: {exc}"
        else:
            if status == 404:
                # The single most diagnostic status here: the route does not
                # exist, which is what a pooling/embed runner looks like.
                last = (
                    f"HTTP 404 on {url}. The server is up but has no chat-completions "
                    "route - this is the signature of a pooling/embedding runner "
                    "(defect #8). Check config.json `architectures`."
                )
            elif status != 200:
                last = f"HTTP {status}: {str(payload)[:300]}"
            else:
                text = _extract_completion_text(payload)
                if text is None:
                    last = f"HTTP 200 but no choices[0].message.content: {str(payload)[:300]}"
                else:
                    served: tuple[str, ...] = ()
                    warnings: list[str] = []
                    try:
                        gstatus, gpayload = http_get(
                            base_url.rstrip("/") + "/models", request_timeout_s
                        )
                        if gstatus == 200 and isinstance(gpayload, dict):
                            served = tuple(
                                str(m.get("id"))
                                for m in gpayload.get("data", [])
                                if isinstance(m, dict)
                            )
                            if served and model not in served:
                                warnings.append(
                                    f"requested model {model!r} not in /v1/models {served!r}"
                                )
                    except Exception as exc:  # noqa: BLE001 - informational only
                        warnings.append(f"/v1/models probe failed: {type(exc).__name__}: {exc}")
                    return PreflightResult(
                        base_url=base_url,
                        model=model,
                        attempts=attempts,
                        elapsed_s=monotonic() - start,
                        completion_preview=text[:60],
                        served_models=served,
                        warnings=tuple(warnings),
                    )
        if monotonic() - start >= timeout_s:
            raise PreflightError(
                f"endpoint {base_url} did not serve a chat completion for model {model!r} "
                f"within {timeout_s:.0f}s ({attempts} attempts). Last failure: {last}"
            )
        sleep(poll_interval_s)


def _requests_adapter(resp: Any) -> tuple[int, Any]:  # pragma: no cover
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


def _extract_completion_text(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        # Thinking models may put the visible answer in `reasoning_content` only
        # when max_tokens truncates. Empty content with a finish_reason of
        # `length` is a real (reportable) condition, not a readiness failure.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def read_served_model_name(base_url: str, *, http_get: Any) -> str:
    """Return the single model id a server advertises, or raise.

    Used to catch the "requested model name does not match what is served" case
    explicitly rather than letting every request 404.
    """
    status, payload = http_get(base_url.rstrip("/") + "/models", 60.0)
    if status != 200:
        raise PreflightError(f"/v1/models returned HTTP {status}")
    if not isinstance(payload, Mapping) or "data" not in payload:
        raise MissingFieldError("/v1/models.data")
    ids = [str(m.get("id")) for m in payload["data"] if isinstance(m, Mapping) and m.get("id")]
    if len(ids) != 1:
        raise PreflightError(f"expected exactly one served model, got {ids!r}")
    return ids[0]


def assert_finite(value: float, what: str) -> float:
    """Guard against a NaN/inf leaking into a config or a reported number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise SchemaError(f"{what} is not finite: {value!r}")
    return float(value)
