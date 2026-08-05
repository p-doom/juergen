"""Items 4 + 15 — `_control_of` across all four grammars, and sampling authority.

Item 4: `terminate(status="failure")` in a tool-call grammar is the same outcome as a
bare-token `FAIL`, and must normalise to `fail` — otherwise indicator C counts a
self-declared failure as a claimed success.

Item 15: `ctx.sampling` is authoritative at the wire (`ChatDialect.apply_overrides`
is `{**body, "model": ..., **sampling.model_dump(exclude_none=True)}`), so
`program_sampling` must send only what the eval left unset. The three historical
temperatures — 1.0 train, 0.0 parity, 0.7 movebox — must not be able to reappear.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import verifiers.v1 as vf

from agent.agent import (
    Agent,
    ContextTransport,
    Decision,
    EffectiveSampling,
    EndpointTransport,
    ModelCallError,
    _control_of,
    _first_prose,
    build_transport,
    dump_prompt,
    load_codec,
    program_sampling,
    resolve_sampling,
)
from agent.history import History, ImageBudget, StatelessSingleTurn, history_policy
from juergen_doubles import FakeClient, make_ctx, png

BARE_TOKEN_GRAMMARS = ("deltatype_v2",)
TOOL_CALL_GRAMMARS = ("native_absolute", "move_rel")
ALL_FOUR = ("deltatype_v2", "compact_raw", "native_absolute", "move_rel")


def _geometry(width: int = 1920, height: int = 1080):
    from desktop_env.geometry import DisplayGeometry

    return DisplayGeometry(desktop_width=width, desktop_height=height)


def _tool_call(arguments: dict) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": arguments})
        + "\n</tool_call>"
    )


# =========================================================================== #
# ITEM 4 — _control_of
# =========================================================================== #


def test_control_of_reads_nothing_from_a_missing_action() -> None:
    assert _control_of(None) is None


@pytest.mark.parametrize("codec_name", BARE_TOKEN_GRAMMARS)
@pytest.mark.parametrize(
    "line,expected",
    [("TERMINATE", "terminate"), ("FAIL", "fail"), ("NO_OP", "no_op")],
)
def test_control_of_reads_bare_token_flags(codec_name: str, line: str, expected: str) -> None:
    action = load_codec(codec_name).parse(line)
    assert _control_of(action) == expected


@pytest.mark.parametrize("codec_name", TOOL_CALL_GRAMMARS)
def test_control_of_normalises_a_tool_call_failure_to_fail(codec_name: str) -> None:
    """★ The new normalisation. `terminate(status="failure")` is a bare-token `FAIL`."""
    codec = load_codec(codec_name)
    success = codec.parse(_tool_call({"action": "terminate", "status": "success"}))
    failure = codec.parse(_tool_call({"action": "terminate", "status": "failure"}))
    assert _control_of(success) == "terminate"
    assert _control_of(failure) == "fail", (
        "without this, indicator C reads a self-declared failure as a claimed success"
    )


@pytest.mark.parametrize("codec_name", TOOL_CALL_GRAMMARS)
def test_a_tool_call_terminate_with_no_status_is_a_success_termination(codec_name: str) -> None:
    action = load_codec(codec_name).parse(_tool_call({"action": "terminate"}))
    assert _control_of(action) == "terminate"


@pytest.mark.parametrize("status", ["FAILURE", " failure ", "Failure"])
def test_the_failure_status_match_is_case_and_whitespace_insensitive(status: str) -> None:
    assert _control_of({"terminate": True, "status": status}) == "fail"


def test_an_unknown_status_is_not_silently_a_failure() -> None:
    assert _control_of({"terminate": True, "status": "partial"}) == "terminate"


def test_control_of_reads_a_dict_action_as_well_as_an_object() -> None:
    assert _control_of({"no_op": True}) == "no_op"
    assert _control_of({"fail": True}) == "fail"
    assert _control_of({}) is None


def test_compact_raw_and_native_absolute_control_declare_no_control_tokens() -> None:
    """The paired arms are deliberately control-token free, so `_control_of` is None.

    The `_control_of` docstring claims `compact_raw` carries `no_op`/`terminate`/`fail`
    flags; it does not (`cells.py` calls it "its control-token-free sibling"). Recorded
    as a docstring defect — the behaviour is right, the comment is not.
    """
    for name in ("compact_raw", "native_absolute_control"):
        action = load_codec(name).parse("0 0 0 ;")
        assert _control_of(action) is None
        for flag in ("terminate", "fail", "no_op"):
            assert not hasattr(action, flag), (name, flag)


@pytest.mark.parametrize("codec_name", ALL_FOUR)
def test_terminated_is_true_for_both_terminate_and_fail(codec_name: str) -> None:
    sampling = EffectiveSampling("m", None, None, None, (), "harness_default", ())
    for control, terminated in (("terminate", True), ("fail", True), ("no_op", False), (None, False)):
        decision = Decision(1, "t", "", None, (), control, None, sampling)
        assert decision.terminated is terminated


# =========================================================================== #
# decide(): parse and compile are results, not exceptions
# =========================================================================== #


def _agent(codec_name: str, **kwargs) -> Agent:
    return Agent(
        codec=load_codec(codec_name),
        policy=StatelessSingleTurn(),
        budget=ImageBudget(max_images=1),
        transport=ContextTransport(),
        **kwargs,
    )


def _sampling() -> EffectiveSampling:
    return EffectiveSampling("m", None, None, None, (), "harness_default", ())


@pytest.mark.parametrize("codec_name", ALL_FOUR)
def test_a_parse_failure_is_a_recorded_result_not_an_exception(codec_name: str) -> None:
    decision = _agent(codec_name).decide(
        "this is not an action in any grammar",
        step=1,
        geometry=_geometry(),
        cursor=(5, 5),
        sampling=_sampling(),
    )
    assert decision.parse_error is not None
    assert decision.operations == () and decision.control is None
    assert decision.as_record()["parse_error"]["type"], "the error type is recorded"


def test_an_empty_compile_becomes_no_op_not_a_parse_error() -> None:
    decision = _agent("deltatype_v2").decide(
        "0 0 0 ;", step=1, geometry=_geometry(), cursor=(5, 5), sampling=_sampling()
    )
    assert decision.parse_error is None
    assert decision.control == "no_op", "an idle line dispatches nothing but parsed fine"


@pytest.mark.parametrize("codec_name", ALL_FOUR)
def test_a_control_action_compiles_to_no_operations(codec_name: str) -> None:
    text = (
        "TERMINATE"
        if codec_name == "deltatype_v2"
        else _tool_call({"action": "terminate", "status": "success"})
    )
    if codec_name == "compact_raw":
        pytest.skip("compact_raw has no control tokens by design")
    decision = _agent(codec_name).decide(
        text, step=1, geometry=_geometry(), cursor=(5, 5), sampling=_sampling()
    )
    assert decision.control == "terminate" and decision.operations == ()


def test_a_real_click_compiles_to_absolute_pixel_operations() -> None:
    decision = _agent("deltatype_v2").decide(
        "10 10 0 ; +LMB -LMB",
        step=1,
        geometry=_geometry(),
        cursor=(100, 100),
        sampling=_sampling(),
    )
    assert decision.parse_error is None and decision.control is None
    kinds = [getattr(op, "kind", None) for op in decision.operations]
    assert "mouse_down" in kinds and "mouse_up" in kinds
    moves = [op for op in decision.operations if getattr(op, "kind", "") == "move_to"]
    assert moves and tuple(moves[0].args[:2]) == (110, 110), (
        "the codec resolves the relative delta; nothing downstream re-resolves it"
    )


def test_prose_is_everything_before_the_final_line() -> None:
    assert _first_prose("thinking here\nmore\n0 0 0 ;") == "thinking here more"
    assert _first_prose("0 0 0 ;") == ""
    assert _first_prose("") == ""


def test_as_record_is_json_serialisable_for_every_grammar() -> None:
    for name in ALL_FOUR:
        text = "0 0 0 ; +LMB -LMB" if name in ("deltatype_v2", "compact_raw") else _tool_call(
            {"action": "left_click", "coordinate": [100, 100]}
        )
        record = _agent(name).decide(
            text, step=1, geometry=_geometry(), cursor=(10, 10), sampling=_sampling()
        ).as_record()
        json.dumps(record, default=str)
        assert record["step"] == 1 and record["raw_model_output"] == text


# =========================================================================== #
# ITEM 15 — sampling authority
# =========================================================================== #


def test_program_sampling_sends_only_what_the_eval_left_unset() -> None:
    ctx = make_ctx(temperature=0.2)
    sent = program_sampling(ctx, {"temperature": 1.0, "max_tokens": 256, "top_p": None})
    assert sent == {"max_tokens": 256}, (
        "sending a temperature the proxy would discard makes the recorded request a lie"
    )


def test_program_sampling_respects_a_provider_extra_the_eval_set() -> None:
    ctx = make_ctx(**{"stop": ["</x>"]})
    assert program_sampling(ctx, {"stop": ["mine"]}) == {}


def test_program_sampling_passes_a_harness_default_through_when_the_eval_is_silent() -> None:
    ctx = make_ctx()
    assert program_sampling(ctx, {"temperature": 0.0, "max_tokens": 64}) == {
        "temperature": 0.0,
        "max_tokens": 64,
    }


def test_resolve_sampling_reports_ctx_sampling_as_the_source_when_the_eval_set_it() -> None:
    ctx = make_ctx(temperature=0.3, max_tokens=99)
    wire, effective = resolve_sampling(ctx, {"messages": [], "temperature": 1.0, "max_tokens": 7})
    assert wire["temperature"] == 0.3 and wire["max_tokens"] == 99
    assert effective.temperature == 0.3
    assert effective.temperature_source == "ctx.sampling"
    assert effective.model == "test-model"


def test_resolve_sampling_reports_harness_default_when_the_eval_is_silent() -> None:
    ctx = make_ctx()
    _, effective = resolve_sampling(ctx, {"messages": [], "temperature": 0.0})
    assert effective.temperature == 0.0
    assert effective.temperature_source == "harness_default"


def test_the_effective_temperature_source_has_exactly_three_values() -> None:
    """`"ctx.sampling" | "harness_default" | "scripted"` and nothing else."""
    sources = set()
    sources.add(resolve_sampling(make_ctx(temperature=0.5), {"messages": []})[1].temperature_source)
    sources.add(resolve_sampling(make_ctx(), {"messages": []})[1].temperature_source)
    sources.add(
        EffectiveSampling("scripted", None, None, None, (), "scripted", ()).temperature_source
    )
    assert sources == {"ctx.sampling", "harness_default", "scripted"}


@pytest.mark.parametrize("historical", [1.0, 0.0, 0.7])
def test_a_historical_temperature_cannot_reappear_once_the_eval_has_spoken(historical: float) -> None:
    """1.0 (train), 0.0 (parity), 0.7 (movebox) all used to ride the request body and
    be silently dropped. Whatever the harness default is, the eval's value is what
    reaches the wire AND what gets recorded."""
    ctx = make_ctx(temperature=0.25)
    agent = _agent("deltatype_v2", temperature=historical, max_tokens=128)
    body = agent.build_body(history=_one_frame_history(), instruction="G", step=1)
    assert body["temperature"] == historical, "the harness default is in the program body"
    program = {"messages": body["messages"]}
    program.update(program_sampling(ctx, {k: v for k, v in body.items() if k != "messages"}))
    assert "temperature" not in program, "so it is never transmitted"
    _, effective = resolve_sampling(ctx, body)
    assert effective.temperature == 0.25 and effective.temperature_source == "ctx.sampling"


def _one_frame_history() -> History:
    history = History(n_history_frames=4)
    history.start(png())
    return history


def test_one_temperature_source_reaches_the_wire_through_agent_step() -> None:
    ctx = make_ctx(temperature=0.9, replies=["0 0 0 ;"])
    agent = _agent("deltatype_v2", temperature=0.0, max_tokens=32)
    decision = asyncio.run(
        agent.step(
            ctx,
            history=_one_frame_history(),
            instruction="G",
            step=1,
            geometry=_geometry(),
            cursor=(0, 0),
        )
    )
    body = ctx.client.calls[0]["body"]
    assert "temperature" not in body, "the harness default is not transmitted at all"
    assert body["max_tokens"] == 32, "an unset knob is transmitted"
    assert decision.sampling.temperature == 0.9
    assert decision.sampling.temperature_source == "ctx.sampling"
    record = decision.as_record()["sampling"]
    assert record["temperature"] == 0.9 and record["temperature_source"] == "ctx.sampling"


def test_wire_body_keys_are_recorded_without_the_messages() -> None:
    ctx = make_ctx(temperature=0.5, max_tokens=16)
    _, effective = resolve_sampling(ctx, {"messages": [], "top_p": 0.8})
    assert "messages" not in effective.wire_body_keys
    assert set(effective.wire_body_keys) >= {"model", "temperature", "max_tokens", "top_p"}


def test_a_string_stop_sequence_is_normalised_to_a_tuple() -> None:
    ctx = make_ctx(**{"stop": "</action>"})
    _, effective = resolve_sampling(ctx, {"messages": []})
    assert effective.stop == ("</action>",)


def test_build_body_drops_unset_knobs_and_keeps_codec_stop_sequences() -> None:
    agent = _agent("deltatype_v2", max_tokens=None, temperature=None, top_p=None)
    body = agent.build_body(history=_one_frame_history(), instruction="G", step=1)
    assert set(body) == {"messages"}, body
    stops = list(load_codec("deltatype_v2").stop_sequences or ())
    if stops:
        agent2 = _agent("deltatype_v2")
        assert agent2.build_body(history=_one_frame_history(), instruction="G", step=1)["stop"] == stops


def test_the_system_prompt_is_the_codecs_own_description_unless_overridden() -> None:
    agent = _agent("deltatype_v2")
    assert agent.system == load_codec("deltatype_v2").describe()
    assert _agent("deltatype_v2", system_prompt="SEALED").system == "SEALED"


# =========================================================================== #
# transports
# =========================================================================== #


def test_build_transport_prefers_the_context_when_asked_or_endpointless() -> None:
    assert isinstance(build_transport(endpoint=None, secret=None), ContextTransport)
    assert isinstance(build_transport(endpoint="", secret="s"), ContextTransport)
    assert isinstance(build_transport(endpoint="http://x", secret="s"), EndpointTransport)
    assert isinstance(
        build_transport(endpoint="http://x", secret="s", prefer_context=True), ContextTransport
    )


def test_context_transport_flattens_list_content() -> None:
    class ListClient(FakeClient):
        async def get_response(self, dialect, body, model, sampling, **kwargs):
            return type(
                "R", (), {"message": type("M", (), {"content": [{"text": "a"}, {"text": "b"}]})()}
            )()

    ctx = vf.ModelContext(model="m", client=ListClient(), sampling=vf.Sampling())
    text = asyncio.run(
        ContextTransport().complete(ctx, {"messages": []}, session_id="s")
    )
    assert text == "ab"


def test_a_transport_failure_becomes_a_model_call_error() -> None:
    class Angry(FakeClient):
        async def get_response(self, *args, **kwargs):
            raise ConnectionResetError("peer went away")

    ctx = vf.ModelContext(model="m", client=Angry(), sampling=vf.Sampling())
    with pytest.raises(ModelCallError, match="ConnectionResetError"):
        asyncio.run(ContextTransport().complete(ctx, {"messages": []}, session_id=None))


def test_the_session_id_reaches_the_client() -> None:
    ctx = make_ctx(replies=["0 0 0 ;"])
    asyncio.run(
        _agent("deltatype_v2").step(
            ctx,
            history=_one_frame_history(),
            instruction="G",
            step=1,
            geometry=_geometry(),
            cursor=(0, 0),
            session_id="trace-42",
        )
    )
    assert ctx.client.calls[0]["kwargs"]["session_id"] == "trace-42"


# =========================================================================== #
# prompt sidecar
# =========================================================================== #


def test_dump_prompt_elides_image_bytes() -> None:
    agent = _agent("deltatype_v2")
    body = agent.build_body(history=_one_frame_history(), instruction="G", step=1)
    dumped = dump_prompt(body)
    assert "base64" not in dumped and "<image>" in dumped
    payload = json.loads(dumped)
    assert payload["messages"][0]["role"] == "system"
    assert len(dumped) < 4000, "a sidecar must stay small enough to read"


# =========================================================================== #
# codec loading
# =========================================================================== #


def test_load_codec_resolves_every_registered_grammar() -> None:
    """DEFECT (fixed, `agent/agent.py:112`).

    The in-tree fallback probed `desktop_env.codec_protocol.CODECS` and
    `.load_codec`, and desktop-env has **neither** — it is deliberately grammar-free.
    So on any interpreter where juergen's entry points are not installed, every
    `load_codec` raised `LookupError` while `grammars.load` worked. The fallback now
    goes through `grammars.load`, which has its own peer-directory scan.
    """
    import grammars

    for name in grammars.available():
        assert load_codec(name).name == name


def test_load_codec_still_refuses_an_unknown_grammar() -> None:
    with pytest.raises((LookupError, KeyError)):
        load_codec("grammar_that_does_not_exist")


def test_every_grammar_satisfies_the_codec_protocol_slice_the_driver_uses() -> None:
    import grammars

    for name in grammars.available():
        codec = load_codec(name)
        for attribute in ("name", "stop_sequences", "parse", "format", "compile", "describe"):
            assert hasattr(codec, attribute), (name, attribute)
        assert isinstance(codec.describe(), str) and codec.describe()
