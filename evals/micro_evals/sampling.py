"""Single source of truth for Qwen-recommended eval sampling parameters.

Every crowd-cast OSWorld / grounding / rollout eval harness MUST decode with the
sampling parameters Qwen recommends for the checkpoint's *regime* rather than
greedy or the (partial) baked ``generation_config`` defaults. This module owns
those parameter tuples so no harness hardcodes ``temperature=0`` or silently
drops ``top_p`` / ``top_k`` / the penalties.

Two regimes, keyed on **Instruct vs Thinking** (Qwen3-VL HF model cards):

===============  ===========  =====  =====  ==================  =================
regime           temperature  top_p  top_k  repetition_penalty  presence_penalty
===============  ===========  =====  =====  ==================  =================
Instruct-VL          0.7       0.8    20           1.0                1.5
Thinking-VL          1.0       0.95   20           1.0                0.0
===============  ===========  =====  =====  ==================  =================

Instruct-VL is our current regime. Both cards ship ``greedy=false`` — sampling
is recommended and greedy is *discouraged*; ``greedy=True`` (harness ``--greedy``)
is available as an explicit opt-out (e.g. the BC imitation monitor) but is never
the enforced default.

Two footguns this module closes (see the audit, agent a398f7b):
  * the baked ``generation_config`` **omits** ``presence_penalty`` (-> 0) and is
    what sglang falls back to when a harness sends only ``temperature`` — so the
    fix is to always send the *full* tuple, not to trust the checkpoint default;
  * the Thinking tuple differs from Instruct, so a harness that hardcodes one
    silently mis-samples the other.

presence_penalty nuance
-----------------------
Qwen's Instruct card recommends ``presence_penalty=1.5``, and that is the default
here so "match Qwen exactly" holds out of the box. HOWEVER our own closed-loop
A/B found ``presence_penalty=1.5`` does NOT fix our OSWorld repetition (near
no-op: no-terminate 0.31 -> 0.34, repeat 0.53 -> 0.56) — the repetition is a
structural covariate-shift, not a decoding artifact. So ``presence_penalty`` is a
config knob: keep 1.5 to honour the card, or pass ``presence_penalty=0`` (harness
``--presence_penalty 0``) for our OSWorld runs. See the PR description.

OpenAI vs sglang wire formats
-----------------------------
sglang's OpenAI-compatible ``/chat/completions`` accepts ``top_k`` /
``repetition_penalty`` / ``presence_penalty`` at the *top level* (unlike the
stock OpenAI schema), so the in-house raw-``requests`` harnesses send them flat
via :meth:`SamplingParams.as_request_json`. The stock OpenAI python client only
accepts ``temperature`` / ``top_p`` / ``max_tokens`` at the top level, so
:meth:`SamplingParams.as_openai_kwargs` routes ``top_k`` / ``repetition_penalty``
/ ``presence_penalty`` through ``extra_body`` (which sglang forwards to the
sampler).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

INSTRUCT = "instruct"
THINKING = "thinking"

# Qwen3-VL recommended sampling tuples (from the HF model cards).
_RECOMMENDED: dict[str, dict[str, float]] = {
    INSTRUCT: dict(temperature=0.7, top_p=0.8, top_k=20,
                   repetition_penalty=1.0, presence_penalty=1.5),
    THINKING: dict(temperature=1.0, top_p=0.95, top_k=20,
                   repetition_penalty=1.0, presence_penalty=0.0),
}

# Default cap on generated tokens. NOT a Qwen "recommendation" (it is
# task-specific) but centralised so no harness truncates tool-calls with a
# too-small value. The grounding runner previously capped at 64, which truncated
# native tool-calls; OSWorld-style harnesses pass ``default_max_tokens=256``.
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class SamplingParams:
    """A resolved, ready-to-send set of decoding parameters."""

    mode: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    presence_penalty: float
    max_tokens: int
    greedy: bool = False

    def as_request_json(self) -> dict[str, Any]:
        """Flat payload merge for sglang's OpenAI-compatible /chat/completions.

        Returns only the sampling keys; the caller merges these into the request
        body alongside ``model`` / ``messages``. ``greedy`` forces temperature 0
        and drops the nucleus / top-k / penalty knobs so the server decodes
        deterministically.
        """
        if self.greedy:
            return {"max_tokens": self.max_tokens, "temperature": 0.0}
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def as_openai_kwargs(self) -> dict[str, Any]:
        """kwargs for the stock OpenAI client ``chat.completions.create``.

        ``top_k`` / ``repetition_penalty`` / ``presence_penalty`` ride
        ``extra_body`` (sglang forwards them to the sampler) because the stock
        OpenAI schema rejects them at the top level.
        """
        if self.greedy:
            return {"max_tokens": self.max_tokens, "temperature": 0.0}
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "presence_penalty": self.presence_penalty,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Full param set for logging into ``result.json`` (audit trail)."""
        return asdict(self)


def detect_mode(
    *,
    model_path: str | None = None,
    system_prompt: str | None = None,
    mode: str | None = None,
) -> str:
    """Resolve Instruct vs Thinking (priority: explicit > system prompt > name).

    * an explicit ``mode`` ("instruct"/"thinking") always wins;
    * a system prompt that contains a literal ``<think>`` tag => THINKING;
    * a checkpoint path/name containing "think" (case-insensitive) => THINKING;
    * otherwise INSTRUCT (our current regime).

    The name/tag heuristics are deliberately narrow (``<think>``, ``think`` in the
    checkpoint id) to avoid mis-flagging an Instruct run whose prompt merely says
    "think carefully"; pass ``mode=`` explicitly when in doubt.
    """
    if mode:
        m = mode.strip().lower()
        if m not in _RECOMMENDED:
            raise ValueError(
                f"unknown sampling mode {mode!r}; use 'instruct' or 'thinking'")
        return m
    if system_prompt and "<think>" in system_prompt.lower():
        return THINKING
    if model_path and "think" in model_path.lower():
        return THINKING
    return INSTRUCT


def qwen_sampling(
    mode: str,
    *,
    max_tokens: int | None = None,
    greedy: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> SamplingParams:
    """Build a :class:`SamplingParams` from the Qwen tuple for ``mode``.

    Every per-field override that is ``None`` keeps the Qwen-recommended value
    for the regime, so callers only pass the knobs the user actually set.
    """
    m = mode.strip().lower()
    if m not in _RECOMMENDED:
        raise ValueError(
            f"unknown sampling mode {mode!r}; use 'instruct' or 'thinking'")
    base = _RECOMMENDED[m]

    def pick(override: Any, key: str) -> Any:
        return base[key] if override is None else override

    return SamplingParams(
        mode=m,
        temperature=pick(temperature, "temperature"),
        top_p=pick(top_p, "top_p"),
        top_k=int(pick(top_k, "top_k")),
        repetition_penalty=pick(repetition_penalty, "repetition_penalty"),
        presence_penalty=pick(presence_penalty, "presence_penalty"),
        max_tokens=DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens),
        greedy=greedy,
    )


def add_sampling_cli(
    parser: argparse.ArgumentParser,
    *,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> argparse.ArgumentParser:
    """Register the shared sampling flags on an ``argparse`` parser.

    Each sampling flag defaults to ``None`` meaning "use the Qwen-recommended
    value for the detected regime"; pass a value to override. ``--greedy`` opts
    out of sampling entirely (discouraged; the Qwen cards ship ``greedy=false``).
    Wire the result with :func:`from_cli`.
    """
    g = parser.add_argument_group(
        "sampling", "Qwen-recommended by default; overrides win. See eval/sampling.py.")
    g.add_argument(
        "--sampling_mode", choices=("auto", INSTRUCT, THINKING), default="auto",
        help="Regime whose Qwen tuple to use. 'auto' detects from the "
             "checkpoint id / system prompt (defaults to Instruct).")
    g.add_argument("--temperature", type=float, default=None,
                   help="Override sampling temperature (default: Qwen tuple).")
    g.add_argument("--top_p", type=float, default=None,
                   help="Override nucleus top_p (default: Qwen tuple).")
    g.add_argument("--top_k", type=int, default=None,
                   help="Override top_k (default: Qwen tuple = 20).")
    g.add_argument("--repetition_penalty", type=float, default=None,
                   help="Override repetition_penalty (default: Qwen tuple = 1.0).")
    g.add_argument(
        "--presence_penalty", type=float, default=None,
        help="Override presence_penalty. Qwen recommends 1.5 (Instruct); our "
             "closed-loop A/B found it a near no-op for OSWorld repetition, so "
             "pass 0 for our OSWorld runs.")
    g.add_argument("--max_tokens", type=int, default=default_max_tokens,
                   help="Max new tokens per turn.")
    g.add_argument(
        "--greedy", action="store_true",
        help="Decode greedily (temperature 0, no sampling). DISCOURAGED — the "
             "Qwen cards ship greedy=false; use only for deterministic monitors.")
    return parser


def from_cli(
    args: argparse.Namespace,
    *,
    model_path: str | None = None,
    system_prompt: str | None = None,
) -> SamplingParams:
    """Build :class:`SamplingParams` from flags registered by
    :func:`add_sampling_cli`, auto-detecting the regime when ``--sampling_mode``
    is ``auto``."""
    mode = detect_mode(
        model_path=model_path,
        system_prompt=system_prompt,
        mode=(None if args.sampling_mode == "auto" else args.sampling_mode),
    )
    return qwen_sampling(
        mode,
        max_tokens=args.max_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )


# Sampling knobs that carry a Qwen regime default (max_tokens is excluded: its
# CLI default is a concrete int, so flag-vs-default is not recoverable from the
# namespace the way the None-defaulted knobs are).
_SOURCEABLE = ("temperature", "top_p", "top_k", "repetition_penalty",
               "presence_penalty")


def source_map(args: argparse.Namespace, params: SamplingParams) -> dict[str, str]:
    """Per-field provenance for the run record: where each resolved sampling
    value came from — ``"flag"`` when the user set it on the CLI,
    ``"qwen:<mode>"`` when it is the regime default, or ``"greedy"`` when
    ``--greedy`` zeroed sampling. Recorded alongside :meth:`SamplingParams.to_dict`
    so a result.json is self-describing: a later reader can tell an explicit
    ``temperature 0.7`` from the Instruct default of the same value."""
    if params.greedy:
        return {f: "greedy" for f in _SOURCEABLE}
    return {
        f: ("flag" if getattr(args, f, None) is not None else f"qwen:{params.mode}")
        for f in _SOURCEABLE
    }

