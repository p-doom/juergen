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

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import verifiers.v1 as vf
from verifiers.v1.dialects import ChatDialect, Dialect
from verifiers.v1.dialects.chat import message_to_wire

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
    """The half of `pixeldesk.codec_protocol.Codec` an episode driver touches.

    `compile` hands back `pixeldesk.ir.Operation`s already in absolute screen
    pixels — every normalisation convention (raw pixel deltas, the normalized
    0-999 grid, the drag-only MOVE form) lives inside the codec. Nothing
    downstream of `compile` may re-resolve a coordinate.

    `geometry` is a `pixeldesk.geometry.DisplayGeometry`, not a `(w, h)` pair: the
    codec needs the full display description to clamp, and handing it a bare size
    would put the clamp back on the caller. `handlers` (each grammar's contribution
    to pixeldesk's dispatch engine) is part of the protocol but not used here —
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
    def describe(self) -> str: ...


_ENTRY_POINT_GROUPS = ("juergen.grammars", "pixeldesk.codecs")


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
    try:  # in-tree fallback while the grammar package is being installed
        import grammars  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - environment-dependent
        pass
    else:
        # `grammars.load` is the tree's registry and already falls back to scanning
        # peer directories, so an uninstalled checkout resolves here. The pixeldesk
        # probe below cannot: pixeldesk is grammar-free and exposes neither
        # `CODECS` nor `load_codec`, so without this branch an uninstalled checkout
        # raised LookupError for every grammar.
        try:
            return grammars.load(name)
        except KeyError:
            pass
    try:
        from pixeldesk import codec_protocol  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LookupError(
            f"codec {name!r} not found in entry-point groups {_ENTRY_POINT_GROUPS} "
            "and pixeldesk is not importable"
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": list(self.stop),
            "temperature_source": self.temperature_source,
            "wire_body_keys": list(self.wire_body_keys),
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
    stop = wire.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
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
    )
    return wire, effective


class Transport(Protocol):
    async def complete(
        self, ctx: vf.ModelContext, body: dict[str, Any], *, session_id: str | None
    ) -> str: ...

    async def close(self) -> None: ...


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
    ) -> str:
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
        return completion.choices[0].message.content or ""

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
    ) -> str:
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
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return content or ""

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class Decision:
    """One model turn, all the way through to executable operations.

    `control` names a non-dispatching outcome the grammar defines
    (`terminate` / `fail` / `no_op`), so `operations` can be empty without a parse
    error. `prose` is the reasoning that preceded the action line; the Phase-B
    history policy needs it.
    """

    step: int
    text: str
    prose: str
    action: Any | None
    operations: tuple[Any, ...]
    control: str | None
    parse_error: dict[str, Any] | None
    sampling: EffectiveSampling

    @property
    def terminated(self) -> bool:
        return self.control in {"terminate", "fail"}

    def as_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "raw_model_output": self.text,
            "prose": self.prose,
            "parsed_action": _action_record(self.action),
            "operations": [_operation_record(op) for op in self.operations],
            "control": self.control,
            "parse_error": self.parse_error,
            "sampling": self.sampling.as_dict(),
        }


_CONTROL_FLAGS = ("terminate", "fail", "no_op")


def _control_of(action: Any) -> str | None:
    """Read a grammar's control outcome without knowing its action type.

    Four different concepts on four names, not four spellings of one — measured
    across the seven in-tree grammars:

        terminate  deltatype_v2, diffabs, ordered_events_v3, native_absolute, move_rel
        status     native_absolute, move_rel  (tool-call families only)
        fail       deltatype_v2
        no_op      deltatype_v2, diffabs, ordered_events_v3
        (none)     compact_raw, native_absolute_control

    `compact_raw` and `native_absolute_control` declare no control tokens at all, so
    every lookup misses and their idling (`0 0 0`) surfaces as the empty-operations
    `no_op` the caller derives. A tool-call `terminate(status="failure")` is the same
    outcome as a bare-token `FAIL`, so it is normalised to `fail` — otherwise the
    premature-terminate indicator would count a self-declared failure as a claimed
    success.

    Reads attributes only: `codec.parse` returns an action object in every
    grammar.
    """
    if action is None:
        return None
    if getattr(action, "terminate", False):
        status = str(getattr(action, "status", "") or "").strip().lower()
        return "fail" if status == "failure" else "terminate"
    for flag in ("fail", "no_op"):
        if getattr(action, flag, False):
            return flag
    return None


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


def _first_prose(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines[:-1]).strip() if len(lines) > 1 else ""


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
        text = await self.transport.complete(ctx, program, session_id=session_id)
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
        """
        prose = _first_prose(text)
        try:
            action = self.codec.parse(text)
        except (TypeError, ValueError) as exc:
            return Decision(
                step=step,
                text=text,
                prose=prose,
                action=None,
                operations=(),
                control=None,
                parse_error={"type": type(exc).__name__, "message": str(exc)},
                sampling=sampling,
            )
        control = _control_of(action)
        operations: tuple[Any, ...] = ()
        if control is None:
            try:
                operations = tuple(self.codec.compile(text, geometry, cursor))
            except (TypeError, ValueError) as exc:
                return Decision(
                    step=step,
                    text=text,
                    prose=prose,
                    action=action,
                    operations=(),
                    control=None,
                    parse_error={"type": type(exc).__name__, "message": str(exc)},
                    sampling=sampling,
                )
            if not operations:
                control = "no_op"
        return Decision(
            step=step,
            text=text,
            prose=prose,
            action=action,
            operations=operations,
            control=control,
            parse_error=None,
            sampling=sampling,
        )

    async def close(self) -> None:
        await self.transport.close()


def build_transport(
    *, endpoint: str | None, secret: str | None, prefer_context: bool = False
) -> Transport:
    """`ContextTransport` when asked, or when there is no endpoint to post to."""
    if prefer_context or not endpoint:
        return ContextTransport()
    return EndpointTransport(endpoint=endpoint, secret=secret or "")


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
