"""Named system prompts, each composed over exactly one grammar's own spec.

A grammar renders one prompt: `codec.describe()` is `_support.render_spec`, built
from the codec's docstrings, and its sha256 is the identity a checkpoint is
trained under. That is the right default and the wrong granularity for a dataset
knob — the same grammar is trained with a goal turn and without one, and the two
runs need different opening sentences and therefore different prompts.

So a `Prompt` is a NAMED EDIT of a grammar's spec, not a second copy of it:

    render(prompt) = preface + edited(codec spec) + epilogue + CONTROL_SPEC

Every word of the grammar still comes from the codec's docstrings. What a prompt
adds is framing the grammar does not own — whether a goal is stated, whether the
corpus contains idle turns — and it adds it by declaring the exact text it
replaces. `replace` entries are matched EXACTLY and must hit exactly once, so a
reworded codec docstring fails loudly here instead of silently dropping the edit
and producing a prompt that has quietly become the base one.

`CONTROL_SPEC` stays last, always. The codecs' own `notes` say "nothing else
except the control line below", so an epilogue placed after it would contradict
the text above it; `_body` splits it off and re-appends it.

Why not `describe(variant=...)` on the codec: seven grammars and `_support` would
carry a knob that only the crowd-cast SFT corpus uses, and the digest would stop
naming one prompt. Why not a dict of full hand-written prompts (what
`eval/osworld_system_prompts.py` was): the prompt and the grammar then drift
apart silently, which is the failure this whole layout exists to prevent.

The digest is per prompt id. `report()` carries the grammar's own digest beside
it, so a spec change is visible as a change in both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import grammars
from grammars._support import CONTROL_SPEC
from grammars import THINKING_PREAMBLE

__all__ = [
    "PROMPTS",
    "Prompt",
    "describe",
    "digest",
    "get",
    "grammar_of",
    "names",
    "register",
    "report",
]


@dataclass(frozen=True)
class Prompt:
    """One named prompt: a grammar, plus the framing that grammar does not own."""

    id: str
    grammar: str
    summary: str
    preface: str = ""
    epilogue: str = ""
    replace: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    thinking: bool = False
    """Prepend `grammars.THINKING_PREAMBLE`, exactly as
    `grammars.system_prompt(codec, thinking=True)` does.

    Declared here rather than composed as a `preface` so an unedited thinking
    prompt renders byte-identical to `THINKING_PREAMBLE + describe()` — one of
    the two forms `DesktopHarnessConfig` accepts without an
    `expect_prompt_mismatch` justification."""

    def describe(self) -> str:
        return _render(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.describe().encode()).hexdigest()

    def report(self) -> dict[str, Any]:
        """What a manifest records: this prompt's digest AND its grammar's.

        Both, because they answer different questions. The prompt digest is what
        an eval arm matches a checkpoint against; the grammar digest says whether
        the spec underneath moved.
        """
        return {
            "prompt_id": self.id,
            "grammar": self.grammar,
            "system_prompt_sha256": self.digest,
            "grammar_sha256": grammars.load(self.grammar).digest,
            "edits": len(self.replace),
            "thinking": self.thinking,
        }


def _body(grammar: str) -> str:
    """A grammar's spec with the trailing control block split off.

    The assertion is the contract with `_support.render_spec`: it ends every spec
    with `CONTROL_SPEC`. If that stops being true the split would silently return
    the whole spec and every prompt would render the control block twice.
    """
    spec = grammars.describe(grammar)
    tail = CONTROL_SPEC + "\n"
    if not spec.endswith(tail):
        raise RuntimeError(
            f"grammar {grammar!r} does not end its spec with CONTROL_SPEC; "
            "`_support.render_spec` changed shape and prompts/ must follow it"
        )
    return spec[: -len(tail)].rstrip("\n")


def _render(prompt: Prompt) -> str:
    body = _body(prompt.grammar)
    for old, new in prompt.replace:
        found = body.count(old)
        if found != 1:
            raise RuntimeError(
                f"prompt {prompt.id!r}: its replacement text occurs {found} times "
                f"in grammar {prompt.grammar!r}'s spec, expected exactly 1. The "
                "codec docstring it edits was reworded; update the `replace` "
                f"entry rather than letting the edit vanish.\n  looked for: {old!r}"
            )
        body = body.replace(old, new)
    blocks = [prompt.preface.strip(), body, prompt.epilogue.strip(), CONTROL_SPEC]
    text = "\n\n".join(b for b in blocks if b) + "\n"
    # Concatenated, not joined: THINKING_PREAMBLE carries its own trailing
    # blank line, so this is byte-identical to `THINKING_PREAMBLE + describe()`
    # for a prompt that declares no edits.
    return (THINKING_PREAMBLE + text) if prompt.thinking else text


PROMPTS: dict[str, Prompt] = {}


def register(prompt: Prompt) -> Prompt:
    if prompt.id in PROMPTS:
        raise ValueError(f"duplicate prompt id {prompt.id!r}")
    PROMPTS[prompt.id] = prompt
    return prompt


def get(prompt_id: str) -> Prompt:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        raise LookupError(
            f"no prompt {prompt_id!r}; known: {sorted(PROMPTS)}"
        ) from exc


def names() -> list[str]:
    return sorted(PROMPTS)


def describe(prompt_id: str) -> str:
    return get(prompt_id).describe()


def digest(prompt_id: str) -> str:
    return get(prompt_id).digest


def grammar_of(prompt_id: str) -> str:
    return get(prompt_id).grammar


def report(prompt_id: str) -> dict[str, Any]:
    return get(prompt_id).report()


from prompts import ordered_events_v3 as _ordered_events_v3  # noqa: E402,F401
