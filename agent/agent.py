"""One sampling path, one parse path, one compile path.

`ModelContext` is read in full:

  * `ctx.model` — the model id;
  * `ctx.sampling` — the temperature source. `Dialect.apply_overrides`
    (`dialects/base.py:197-200`, `dialects/chat.py:349-352`) silently drops a
    temperature set in the request body whenever the eval already set one, because
    the eval's sampling is authoritative at the wire. So a harness default is
    consulted only for a knob the eval left unset, and the resolved value is
    recorded with its provenance;
  * `ctx.client` — used directly by `ContextTransport`, so an episode can be
    driven without an interception proxy in front of it (offline scoring,
    single-process gates) while still producing a real `Response`.

The default transport posts to `endpoint`, because that is what commits the turn
to the trace graph. Either way the body we log is the body `apply_overrides` puts
on the wire.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import verifiers.v1 as vf
from verifiers.v1.dialects import ChatDialect, Dialect
from verifiers.v1.dialects.chat import message_to_wire

import grammars

from agent.history import History, HistoryPolicy, ImageBudget

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "Agent",
    "ContextTransport",
    "Decision",
    "EffectiveSampling",
    "EndpointTransport",
    "ModelCallError",
    "Transport",
    "load_codec",
    "resolve_sampling",
]


class ModelCallError(RuntimeError):
    """A model turn failed. Distinct from a parse or dispatch failure: those are
    scored outcomes of the system under test, this is infrastructure."""


class Codec(Protocol):
    """The half of `desktop.codec_protocol.Codec` an episode driver touches.

    `compile` hands back `desktop.ir.Operation`s already in absolute screen
    pixels — every normalisation convention (raw pixel deltas, the normalized
    0-999 grid, the drag-only MOVE form) lives inside the codec. Nothing
    downstream of `compile` may re-resolve a coordinate.

    `geometry` is a `desktop.geometry.DisplayGeometry`, not a `(w, h)` pair: the
    codec needs the full display description to clamp, and handing it a bare size
    would put the clamp back on the caller. `handlers` (each grammar's contribution
    to desktop's dispatch engine) is part of the protocol but not used here —
    the harness dispatches whole operation streams, not individual handlers.
    """

    name: str
    stop_sequences: Sequence[str]

    def parse(self, text: str) -> Any: ...
    def format(self, action: Any) -> str: ...
    def compile(
        self,
        text: str,
        geometry: Any,
        cursor: tuple[int, int],
    ) -> Sequence[Any]: ...
    def action_from_operations(
        self,
        operations: Sequence[Any],
        *,
        geometry: Any,
        cursor: tuple[int, int],
        terminate: object = None,
    ) -> Any: ...
    def describe(self) -> str: ...


_ENTRY_POINT_GROUPS = ("juergen.grammars", "desktop.codecs")


def load_codec(name: str) -> Codec:
    """Resolve a grammar by name through its entry point.

    Grammars live in `juergen/grammars/<name>/` and register themselves; this is
    the only place an episode driver resolves a grammar name.
    """
    from importlib.metadata import entry_points

    for group in _ENTRY_POINT_GROUPS:
        for entry in entry_points(group=group):
            if entry.name == name:
                loaded = entry.load()
                return loaded() if isinstance(loaded, type) else loaded
    # `grammars.load` is the tree's registry and already falls back to scanning peer
    # directories, so an uninstalled checkout resolves here. The desktop probe below
    # cannot: desktop is grammar-free and exposes neither `CODECS` nor `load_codec`,
    # so without this branch an uninstalled checkout raised LookupError for every
    # grammar.
    try:
        return grammars.load(name)
    except KeyError:
        pass
    try:
        from desktop import codec_protocol  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LookupError(
            f"codec {name!r} not found in entry-point groups {_ENTRY_POINT_GROUPS} "
            "and desktop is not importable"
        ) from exc
    registry = getattr(codec_protocol, "CODECS", None)
    if isinstance(registry, dict) and name in registry:
        candidate = registry[name]
        return candidate() if isinstance(candidate, type) else candidate
    loader = getattr(codec_protocol, "load_codec", None)
    if callable(loader):
        return loader(name)
    raise LookupError(f"codec {name!r} is not registered")


@dataclass(frozen=True)
class EffectiveSampling:
    """What actually reaches the wire, and who decided it.

    `temperature_source` exists so a run can be audited without re-deriving
    override precedence: `"ctx.sampling"` means the eval/orchestrator set it (and
    it wins), `"harness_default"` means the eval left it unset and the harness
    filled in. There is no third possibility.
    """

    model: str
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    stop: tuple[str, ...]
    temperature_source: str
    wire_body_keys: tuple[str, ...]
    seed: int | None = None
    wire_request_sha256: str | None = None
    wire_sampling_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": list(self.stop),
            "temperature_source": self.temperature_source,
            "wire_body_keys": list(self.wire_body_keys),
            "seed": self.seed,
            "wire_request_sha256": self.wire_request_sha256,
            "wire_sampling_sha256": self.wire_sampling_sha256,
        }


def _eval_set(sampling: vf.Sampling, key: str) -> bool:
    if getattr(sampling, key, None) is not None:
        return True
    extra = sampling.model_extra or {}
    return extra.get(key) is not None


def program_sampling(
    ctx: vf.ModelContext, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Only the knobs the eval left unset. Anything the eval set is dropped here:
    `apply_overrides` would overwrite it anyway, and sending it would misrepresent
    the recorded request."""
    return {
        key: value
        for key, value in defaults.items()
        if value is not None and not _eval_set(ctx.sampling, key)
    }


def resolve_sampling(
    ctx: vf.ModelContext, body: dict[str, Any], dialect: Dialect | None = None
) -> tuple[dict[str, Any], EffectiveSampling]:
    """Merge the program body with the eval's authoritative sampling.

    Returns `(wire_body, effective)`. `wire_body` is exactly what the proxy would
    send — `{**body, "model": ctx.model, **ctx.sampling.model_dump(exclude_none=True)}`
    — so recording it needs no second source of truth.
    """
    dialect = dialect or ChatDialect()
    wire = dialect.apply_overrides(dict(body), ctx.model, ctx.sampling)
    offered = [key for key in ("tools", "tool_choice") if key in wire]
    if offered:
        # `vf.Sampling` is `extra="allow"` and `apply_overrides` puts every extra key
        # on the wire (`dialects/chat.py:352`), so `--sampling.tools=...` reaches the
        # server without touching this repo. Probed on the sign-of-life serving path
        # (sglang 0.5.10.post1): `tool_choice="required"` rewrote the turn into a bare
        # JSON array that no codec parses, while `tool_calls` stayed null -- an arm
        # that reads as a model collapse and is a config change.
        raise ValueError(
            f"sampling put {offered} on the wire; this driver offers no tool schema and "
            "parses the action out of `content`, so a served tool protocol silently "
            "changes what every grammar has to parse"
        )
    stop = wire.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
    normalized = dict(wire)
    normalized["messages"] = [
        message_to_wire(message) for message in wire.get("messages", [])
    ]
    sampling = {key: value for key, value in normalized.items() if key != "messages"}

    def digest(value: dict[str, Any]) -> str:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    effective = EffectiveSampling(
        model=str(wire.get("model", ctx.model)),
        temperature=wire.get("temperature"),
        max_tokens=wire.get("max_tokens"),
        top_p=wire.get("top_p"),
        stop=tuple(stop),
        temperature_source=(
            "ctx.sampling" if _eval_set(ctx.sampling, "temperature") else "harness_default"
        ),
        wire_body_keys=tuple(sorted(k for k in wire if k != "messages")),
        seed=wire.get("seed"),
        wire_request_sha256=digest(normalized),
        wire_sampling_sha256=digest(sampling),
    )
    return wire, effective


class Transport(Protocol):
    """`(content, finish_reason)`. The finish reason is not optional to report: a
    turn cut off at `max_tokens` arrives as text that either fails to parse — a
    fake parse error attributed to the model — or parses as a fragment of the
    action it was still emitting."""

    async def complete(
        self, ctx: vf.ModelContext, body: dict[str, Any], *, session_id: str | None
    ) -> tuple[str, str | None]: ...

    async def close(self) -> None: ...


def _content(message: Any) -> str:
    """The assistant text. A native tool call is infrastructure, not a turn.

    Every grammar spells its action inside `content`, including the `<tool_call>`
    blocks the native grammars emit — those are text the codec parses. Probed
    against the serving path the evals use (sglang 0.5.10.post1 launched by
    `evals/signoflife/__main__.py`, no `--tool-call-parser`): `content` carried the
    block and `tool_calls` was null on both an off-the-shelf 4B and a LoRA-merged
    checkpoint, even when the request offered a tool schema. A server that does
    populate `tool_calls` — one with a tool-call parser, or a provider injecting its
    own server-side tools — leaves `content` null, which would read as `""` here and
    fail to parse every turn: a 0% arm indistinguishable from a model collapse.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        raise ModelCallError(
            f"the server returned {len(tool_calls)} native tool call(s); this driver "
            "parses the action out of `content`, which such a turn leaves empty"
        )
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


@dataclass
class EndpointTransport:
    """Post to verifiers' interception endpoint — the path that records the turn.

    The program body carries only the knobs the eval left unset; the proxy applies
    `ctx.model` + `ctx.sampling` on top and commits the resulting `Response` to
    the trace graph. This is the default: without the graph there are no tokens,
    logprobs or branch structure.
    """

    endpoint: str
    secret: str
    timeout_s: float = 180.0
    _client: Any = field(default=None, init=False, repr=False)

    async def complete(
        self, ctx: vf.ModelContext, body: dict[str, Any], *, session_id: str | None
    ) -> tuple[str, str | None]:
        from openai import AsyncOpenAI

        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.endpoint, api_key=self.secret, timeout=self.timeout_s
            )
        payload = {k: v for k, v in body.items() if k != "messages"}
        payload["model"] = ctx.model
        try:
            completion = await self._client.chat.completions.create(
                messages=[message_to_wire(m) for m in body["messages"]], **payload
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as infrastructure
            raise ModelCallError(f"{type(exc).__name__}: {exc}") from exc
        choice = completion.choices[0]
        return _content(choice.message), choice.finish_reason

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


@dataclass
class ContextTransport:
    """Sample through `ctx.client` directly — no proxy hop.

    `Client.get_response(dialect, body, model, sampling_args, session_id=...)` is
    the same call the interception server makes on the harness's behalf
    (`interception/server.py:415-429`), so an offline gate gets identical
    sampling semantics without standing up an HTTP endpoint. Nothing is committed
    to the trace graph on this path, so it is opt-in.
    """

    dialect: Dialect = field(default_factory=ChatDialect)

    async def complete(
        self, ctx: vf.ModelContext, body: dict[str, Any], *, session_id: str | None
    ) -> tuple[str, str | None]:
        wire = dict(body)
        wire["messages"] = [message_to_wire(m) for m in body["messages"]]
        try:
            response = await ctx.client.get_response(
                self.dialect,
                wire,
                ctx.model,
                ctx.sampling,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as infrastructure
            raise ModelCallError(f"{type(exc).__name__}: {exc}") from exc
        finish_reason = getattr(response, "finish_reason", None)
        return _content(getattr(response, "message", None)), finish_reason

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class Decision:
    """One model turn, all the way through to executable operations.

    `control` names a non-dispatching outcome, and an empty `operations` always
    carries one: `terminate` / `fail` come from the grammar-independent control
    channel (`grammars.split_control`), and `no_op` is every other way a turn
    dispatched nothing — an action that compiled to nothing, one that could not be
    read at all, or a reply cut off at `max_tokens`. WHY it dispatched nothing is
    `parse_error` and `truncated`; THAT it dispatched nothing must not depend on
    which, because only `control` is aggregated. Labelling the compiled-to-nothing
    case alone left 8.9% of archived turns dispatching nothing under `control=None`,
    where no rate over `control` could see them.
    """

    step: int
    text: str
    action: Any | None
    operations: tuple[Any, ...]
    control: str | None
    parse_error: dict[str, Any] | None
    sampling: EffectiveSampling
    truncated: bool = False
    """The turn hit `max_tokens`. Neither a parse failure nor a model decision:
    `max_tokens` is ours, so the action was never finished being emitted."""
    ignored_after_terminate: int = 0
    """Actions the turn placed after its own termination, and which were therefore
    neither parsed nor dispatched. Only the vendor tool-call spelling can have any
    — the control line has to be last — so a non-zero value means an off-the-shelf
    model kept acting after declaring it was done."""
    control_error: dict[str, Any] | None = None
    """A control line `split_control` refused, on a turn whose action still ran.

    The two channels are independent, so a defect in one costs that channel and
    not the turn: the refused line is recorded here, does NOT end the episode, and
    the action on the lines above it is parsed and dispatched as if it stood
    alone."""
    intended_cursor: Any = None
    """Where the turn asked the cursor to go, before the display clamped it
    (`grammars._support.IntendedCursor`), or None when it named no position.

    Resolved from the same action, geometry and cursor as `operations`, and
    published beside them because six of the seven grammars emit no move at all
    when the resolved target equals the current position. At an edge that makes a
    +5000 delta and a 0 delta the same empty stream and the same `no_op`: this is
    the only field in which they differ."""

    def __post_init__(self) -> None:
        if self.control is None and not self.operations:
            object.__setattr__(self, "control", "no_op")

    @property
    def terminated(self) -> bool:
        return self.control in {"terminate", "fail"}

    def as_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "raw_model_output": self.text,
            "parsed_action": _action_record(self.action),
            "operations": [_operation_record(op) for op in self.operations],
            "control": self.control,
            "parse_error": self.parse_error,
            "truncated": self.truncated,
            "ignored_after_terminate": self.ignored_after_terminate,
            "control_error": self.control_error,
            "intended_cursor": (
                None if self.intended_cursor is None else self.intended_cursor.to_dict()
            ),
            "sampling": self.sampling.as_dict(),
        }


#: The control channel's status -> the name published as `control`. Two
#: vocabularies exist by contract: `datasets/convert.py::_TERMINAL_CONTROL` maps
#: this name back to the status when it builds a training target from a rollout.
_TERMINAL = {"success": "terminate", "failure": "fail"}


def _split_refused_control(body: str) -> tuple[str, dict[str, Any] | None]:
    """A refused control line, taken off the end of the body the codec will read.

    `grammars.split_control` accepts only `CONTROL_SPEC`'s exact line and leaves a
    near-miss — a mistyped token, an unknown status — in the body on purpose, so a
    malformed termination scores as a defect instead of silently ending the
    episode. That part is deliberate and is kept.

    What was never argued is the collateral. Five of the seven codecs read the LAST
    non-empty line as the action, so the refused line BECAME the action and the
    well-formed one above it was never parsed: `0 0 0 ; +LMB -LMB` dispatches two
    operations, and the same reply with `TERMINATE` underneath it dispatches none.
    Splitting them here keeps the refusal and drops the collateral. The other two
    scan for `<tool_call>` blocks and were never affected; they gain the record.

    Only a near-miss of the CURRENT control token is separated. A retired token
    from an older vocabulary is not a misspelling of this one, and rescuing it
    would be reviving the second spelling this grammar removed on purpose.
    """
    lines = body.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if not line:
            continue
        if re.split(r"[\s:]", line, maxsplit=1)[0] != grammars.CONTROL_TOKEN:
            return body, None
        return "\n".join(lines[:index]), {
            "type": "RefusedControlLine",
            "message": (
                f"{line!r} is not {grammars.CONTROL_TOKEN}: success|failure; "
                "it does not end the episode and it does not consume the action"
            ),
        }
    return body, None


def _action_record(action: Any) -> Any:
    """`parsed_action`, straight from the grammar's own serialiser.

    Every in-tree action type defines `to_dict`; a grammar that does not is a
    contract violation, not a fallback case.
    """
    if action is None:
        return None
    to_dict = getattr(action, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            f"{type(action).__name__} has no to_dict(); a grammar's action type must "
            "serialise itself — `parsed_action` is a published trajectory field"
        )
    return to_dict()


def _operation_record(operation: Any) -> Any:
    if isinstance(operation, dict):
        return operation
    if hasattr(operation, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(operation)
    return repr(operation)


@dataclass
class Agent:
    """screenshot window -> prompt -> sample -> parse -> compile.

    The prompt is `codec.describe()` plus whatever the injected `HistoryPolicy`
    renders. An explicit `system_prompt` override exists only for replaying a
    checkpoint whose sealed training prompt differs from the codec's current
    description — pass its sha256 alongside so the drift is visible.
    """

    codec: Codec
    policy: HistoryPolicy
    budget: ImageBudget
    transport: Transport
    system_prompt: str | None = None
    max_tokens: int | None = 256
    temperature: float | None = None
    top_p: float | None = None
    include_stop_sequences: bool = True

    @property
    def system(self) -> str:
        return self.system_prompt if self.system_prompt is not None else self.codec.describe()

    def build_body(
        self,
        *,
        history: History,
        instruction: str | None,
        step: int,
    ) -> dict[str, Any]:
        messages = self.policy.render(
            history=history,
            system=self.system,
            instruction=instruction,
            step=step,
            budget=self.budget,
        )
        if history.note is not None:
            # Here, not in a policy: every policy ends on the user turn carrying the
            # newest frame (the newest window turn has no output, so no assistant
            # message follows it), so one append covers all four instead of four
            # implementations of one thing. The user channel, never the assistant one:
            # `datasets/convert.py` builds training targets out of the recorded model
            # output, and a note in there would be trained on as the model's own words.
            last = messages[-1]
            assert isinstance(last, vf.UserMessage) and isinstance(last.content, list)
            messages[-1] = vf.UserMessage(
                content=[*last.content, vf.TextContentPart(text=history.note)]
            )
        body: dict[str, Any] = {"messages": messages}
        defaults = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        body.update(defaults)
        if self.include_stop_sequences:
            stops = list(self.codec.stop_sequences or ())
            if stops:
                body["stop"] = stops
        return {k: v for k, v in body.items() if v is not None}

    async def step(
        self,
        ctx: vf.ModelContext,
        *,
        history: History,
        instruction: str | None,
        step: int,
        geometry: Any,
        cursor: tuple[int, int],
        session_id: str | None = None,
    ) -> Decision:
        body = self.build_body(history=history, instruction=instruction, step=step)
        program = {k: v for k, v in body.items() if k == "messages"}
        program.update(program_sampling(ctx, {k: v for k, v in body.items() if k != "messages"}))
        _, effective = resolve_sampling(ctx, body)
        text, finish_reason = await self.transport.complete(
            ctx, program, session_id=session_id
        )
        if finish_reason == "length":
            # Not handed to `decide`: parsing a truncated turn either invents an
            # action the model never finished emitting or, worse, succeeds on a
            # fragment and dispatches it. `pipeline/annotation/lib/labeler.py:270`
            # refuses the same response for the same reason.
            return Decision(
                step=step,
                text=text,
                action=None,
                operations=(),
                control=None,
                parse_error=None,
                sampling=effective,
                truncated=True,
            )
        return self.decide(
            text, step=step, geometry=geometry, cursor=cursor, sampling=effective
        )

    def decide(
        self,
        text: str,
        *,
        step: int,
        geometry: Any,
        cursor: tuple[int, int],
        sampling: EffectiveSampling,
    ) -> Decision:
        """Parse + compile a model turn. Pure: no VM, no network.

        Separated from `step` so the same code path scores a cached rollout, a
        scripted oracle cell, and a live episode. A parse failure is a result of
        the system under test, not an exception.

        The control channel is read FIRST and exactly once, and the codec is given
        only the body before the termination — so a turn like
        `[move_rel, left_click, TERMINATE: success]` still dispatches its work,
        while nothing on the far side of the termination can be parsed at all. A
        control line the channel REFUSED is taken off that body too
        (`_split_refused_control`): it does not end the episode, and it no longer
        consumes the action it happened to sit under.
        """
        control = grammars.split_control(text)
        body, control_error = _split_refused_control(control.body)
        terminal = _TERMINAL[control.status] if control.status else None
        try:
            action = self.codec.parse(body)
        except (TypeError, ValueError) as exc:
            if terminal is not None and isinstance(exc, grammars.NoAction):
                # A turn that only ends the episode has no action of the grammar
                # to parse — prose and a control line, or the control line alone —
                # and that is not a parse error. Only `NoAction`: a MALFORMED
                # action line alongside a termination is still one. The action the
                # turn amounts to is the grammar's own empty one, and the codec is
                # asked for it rather than left null: `parsed_action` is a
                # published field, and `datasets/convert.py:567` drops every turn
                # whose value is falsy — which would delete exactly the terminal
                # turns from any dataset built off these rollouts.
                return Decision(
                    step=step,
                    text=text,
                    action=self.codec.action_from_operations(
                        (), geometry=geometry, cursor=cursor, terminate=control.status
                    ),
                    operations=(),
                    control=terminal,
                    parse_error=None,
                    sampling=sampling,
                    ignored_after_terminate=control.ignored,
                    control_error=control_error,
                )
            return Decision(
                step=step,
                text=text,
                action=None,
                operations=(),
                control=terminal,
                parse_error={"type": type(exc).__name__, "message": str(exc)},
                sampling=sampling,
                ignored_after_terminate=control.ignored,
                control_error=control_error,
            )
        try:
            operations = tuple(self.codec.compile(body, geometry, cursor))
        except (TypeError, ValueError) as exc:
            return Decision(
                step=step,
                text=text,
                action=action,
                operations=(),
                control=terminal,
                parse_error={"type": type(exc).__name__, "message": str(exc)},
                sampling=sampling,
                ignored_after_terminate=control.ignored,
                control_error=control_error,
            )
        return Decision(
            step=step,
            text=text,
            action=action,
            operations=operations,
            control=terminal,
            parse_error=None,
            sampling=sampling,
            ignored_after_terminate=control.ignored,
            control_error=control_error,
            # Resolved from the same three inputs as `operations`, and only here:
            # a "cursor_before + parsed delta" reconstruction downstream is wrong
            # by a grid step for `move_rel`, whose deltas are thousandths of an
            # axis (~19 px per unit at 1920 wide), not pixels.
            intended_cursor=self.codec.intended_cursor(action, geometry, cursor),
        )

    async def close(self) -> None:
        await self.transport.close()


def build_transport(
    *,
    endpoint: str | None,
    secret: str | None,
    prefer_context: bool = False,
    timeout_s: float = 180.0,
) -> Transport:
    """`ContextTransport` when asked, or when there is no endpoint to post to."""
    if prefer_context or not endpoint:
        return ContextTransport()
    return EndpointTransport(
        endpoint=endpoint, secret=secret or "", timeout_s=timeout_s
    )


def dump_prompt(body: dict[str, Any]) -> str:
    """A prompt sidecar with image bytes elided — for `prompt_NNN.json`."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                return {"type": "image_url", "image_url": {"url": "<image>"}}
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    messages = [
        m.model_dump() if hasattr(m, "model_dump") else m for m in body.get("messages", [])
    ]
    payload = {**{k: v for k, v in body.items() if k != "messages"}, "messages": scrub(messages)}
    return json.dumps(payload, indent=2, default=str)
