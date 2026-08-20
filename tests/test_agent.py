"""The control channel through `Agent.decide`, and sampling authority.

`decide` is the one place a turn's termination is read, and it reads it from
`grammars.split_control` — never from the parsed action. The channel's own
spellings, statuses and strictness are pinned in `grammars/test_vectors.py`; what
is asserted here is the driver's contract: which name is published, that the work
a terminating turn carries is still dispatched, and that nothing after the
termination is.

`ctx.sampling` is authoritative at the wire (`ChatDialect.apply_overrides` is
`{**body, "model": ..., **sampling.model_dump(exclude_none=True)}`), so
`program_sampling` must send only what the eval left unset. The three historical
temperatures are 1.0 train, 0.0 parity and 0.7 movebox.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import verifiers.v1 as vf

import grammars

from agent.agent import (
    Agent,
    ContextTransport,
    Decision,
    EffectiveSampling,
    EndpointTransport,
    ModelCallError,
    _action_record,
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

_BARE_LINE = "0 0 0 ; +LMB -LMB"
_CLICK_CALL = (
    "<tool_call>\n"
    + json.dumps({"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1, 1]}})
    + "\n</tool_call>"
)
_SMOKE_TEXT = {
    "compact_raw": _BARE_LINE,
    "deltatype_v2": _BARE_LINE,
    "diffabs": _BARE_LINE,
    "native_absolute_control": _BARE_LINE,
    "ordered_events_v3": "NO_OP",
    "native_absolute": _CLICK_CALL,
    "move_rel": _CLICK_CALL.replace("left_click", "move_rel"),
}
"""One parseable sample per in-tree grammar, so a new grammar has to declare itself
here rather than quietly skip the `to_dict` contract check below."""


def _geometry(width: int = 1920, height: int = 1080):
    from desktop.geometry import DisplayGeometry

    return DisplayGeometry(desktop_width=width, desktop_height=height)


def _tool_call(arguments: dict) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": arguments})
        + "\n</tool_call>"
    )


@pytest.mark.parametrize("codec_name", ALL_FOUR)
def test_terminated_is_true_for_both_terminate_and_fail(codec_name: str) -> None:
    sampling = EffectiveSampling("m", None, None, None, (), "harness_default", ())
    for control, terminated in (("terminate", True), ("fail", True), ("no_op", False), (None, False)):
        decision = Decision(1, "t", "", None, (), control, None, sampling)
        assert decision.terminated is terminated


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


@pytest.mark.parametrize("status,expected", [("success", "terminate"), ("failure", "fail")])
@pytest.mark.parametrize("codec_name", sorted(_SMOKE_TEXT))
def test_the_control_channel_terminates_every_grammar(
    codec_name: str, status: str, expected: str
) -> None:
    """All seven grammars, both statuses, and the action still dispatched.

    Two of the seven — `compact_raw` and `native_absolute_control` — could not
    terminate at all before the channel, and two more had TERMINATE and no FAIL,
    so this is the first assertion that reads the same in every grammar. `fail` is
    published for a failure because indicator C would otherwise read a
    self-declared failure as a claimed success.
    """
    action = _SMOKE_TEXT[codec_name]
    plain = _agent(codec_name).decide(
        action, step=1, geometry=_geometry(), cursor=(5, 5), sampling=_sampling()
    )
    decision = _agent(codec_name).decide(
        f"{action}\nTERMINATE: {status}",
        step=1,
        geometry=_geometry(),
        cursor=(5, 5),
        sampling=_sampling(),
    )
    assert decision.control == expected and decision.terminated
    assert decision.parse_error is None
    assert decision.operations == plain.operations
    assert decision.ignored_after_terminate == 0


#: One MALFORMED action line per grammar: an attempt the grammar recognises and
#: rejects, as opposed to prose it never claimed was an action. A new grammar
#: declares itself here rather than silently skipping the half of the contract
#: that keeps real parse failures counted.
_MALFORMED_ACTION = {
    "compact_raw": "0 0",
    "deltatype_v2": "0 0",
    "diffabs": "0 0",
    "native_absolute_control": "300 200",
    "ordered_events_v3": "move(1)",
    "native_absolute": _tool_call({"action": "teleport"}),
    "move_rel": _tool_call({"action": "mouse_move", "coordinate": [12, -8]}),
}


@pytest.mark.parametrize("codec_name", sorted(grammars.available()))
def test_a_bare_control_line_is_a_termination_and_not_a_parse_error(
    codec_name: str,
) -> None:
    """A turn that only ends the episode has no action to parse, in any grammar.

    An empty body is grammar-dependent, which is why this asks the codec rather
    than testing the string: the two tool-call grammars render the control line
    alone for an empty call list, the other five render an idle action first.
    """
    decision = _agent(codec_name).decide(
        "TERMINATE: success",
        step=1,
        geometry=_geometry(),
        cursor=(5, 5),
        sampling=_sampling(),
    )
    assert decision.control == "terminate" and decision.operations == ()
    assert decision.parse_error is None
    # Not falsy: `datasets/convert.py:567` drops a turn whose `parsed_action` is,
    # which would delete every terminal turn from a dataset built off a rollout.
    assert decision.as_record()["parsed_action"]
    assert decision.action.terminate == "success"


@pytest.mark.parametrize(
    "prose",
    [
        # Verbatim from job 141143, where a turn of this shape was scored a parse
        # error in 8 of 12 episodes -- including two that passed their cell.
        "Action: The task is completed successfully.",
        "The file is saved and the window is closed.",
        "Done",
    ],
)
@pytest.mark.parametrize("codec_name", sorted(grammars.available()))
def test_prose_plus_a_termination_is_not_a_parse_error_in_any_grammar(
    codec_name: str, prose: str
) -> None:
    """A turn that reasons and then stops, in every grammar.

    Prose in an assistant turn is deliberate — the loss mask supervises all of it
    — so a body carrying prose and no action line is a well-formed terminate-only
    turn. Driven through `decide` because the defect was in that seam: each codec
    rejected the residual body on its own terms, correctly, and `decide` counted
    every rejection as a parse error.
    """
    decision = _agent(codec_name).decide(
        f"{prose}\nTERMINATE: success",
        step=1,
        geometry=_geometry(),
        cursor=(5, 5),
        sampling=_sampling(),
    )
    assert decision.parse_error is None
    assert decision.control == "terminate" and decision.terminated
    assert decision.operations == ()
    assert decision.as_record()["parsed_action"]


@pytest.mark.parametrize("codec_name", sorted(_MALFORMED_ACTION))
def test_a_malformed_action_line_still_scores_when_the_turn_terminates(
    codec_name: str,
) -> None:
    """The other half of the distinction: only `NoAction` may be swallowed.

    The prose line carries no `Action:` marker on purpose — in `diffabs` that
    marker is where the action goes, so a marked line would be the text parsed
    and the malformed line below it would never be read.
    """
    decision = _agent(codec_name).decide(
        f"I am going to stop here.\n{_MALFORMED_ACTION[codec_name]}\nTERMINATE: success",
        step=1,
        geometry=_geometry(),
        cursor=(5, 5),
        sampling=_sampling(),
    )
    assert decision.parse_error is not None
    assert decision.control == "terminate", "the control is read either way"


@pytest.mark.parametrize("codec_name", TOOL_CALL_GRAMMARS)
def test_an_unreadable_tool_call_still_scores_when_the_turn_terminates(
    codec_name: str,
) -> None:
    """`iter_tool_calls` skips a block it cannot decode, so yielding nothing does
    not by itself mean the turn called nothing."""
    decision = _agent(codec_name).decide(
        '<tool_call>\n{"name": "computer_use", "arguments": {action: left_click}}\n'
        "</tool_call>\nTERMINATE: success",
        step=1,
        geometry=_geometry(),
        cursor=(5, 5),
        sampling=_sampling(),
    )
    assert decision.parse_error is not None
    assert decision.control == "terminate"


@pytest.mark.parametrize("legacy", ["TERMINATE", "FAIL"])
@pytest.mark.parametrize("codec_name", BARE_TOKEN_GRAMMARS)
def test_a_retired_control_token_is_a_scored_parse_error(
    codec_name: str, legacy: str
) -> None:
    """What a checkpoint trained before the channel emits.

    Loud rather than silent, and it does not end the episode: accepting the old
    token as a second spelling is the per-grammar surface being removed.
    """
    decision = _agent(codec_name).decide(
        legacy, step=1, geometry=_geometry(), cursor=(5, 5), sampling=_sampling()
    )
    assert decision.control is None and decision.parse_error is not None


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


@pytest.mark.parametrize("codec_name", TOOL_CALL_GRAMMARS)
def test_the_work_a_terminating_turn_carries_still_compiles(codec_name: str) -> None:
    """`[move, click]` + a termination dispatched ZERO operations once, and scored
    that turn as a bare premature terminate, which is what the TERMINATE audit was
    reading.
    """
    move = (
        {"action": "move_rel", "coordinate": [50, 50]}
        if codec_name == "move_rel"
        else {"action": "mouse_move", "coordinate": [600, 600]}
    )
    text = "\n".join(
        [
            _tool_call(move),
            _tool_call({"action": "left_click"} if codec_name == "move_rel" else
                       {"action": "left_click", "coordinate": [600, 600]}),
            "TERMINATE: success",
        ]
    )
    decision = _agent(codec_name).decide(
        text, step=1, geometry=_geometry(), cursor=(100, 100), sampling=_sampling()
    )
    assert decision.control == "terminate" and decision.terminated
    kinds = [op.kind for op in decision.operations]
    assert "move_to" in kinds and "mouse_down" in kinds and "mouse_up" in kinds
    assert decision.ignored_after_terminate == 0
    assert decision.as_record()["ignored_after_terminate"] == 0


def test_calls_placed_after_a_vendor_terminate_are_counted_and_never_parsed() -> None:
    """The only spelling that can have anything after its own termination.

    The control line has to be last, so `ignored_after_terminate` is now reachable
    only through the vendor tool-call form an off-the-shelf model emits — and the
    calls after it are cut out of the body, so they are not parsed either.
    """
    text = "\n".join(
        [
            _tool_call({"action": "terminate", "status": "success"}),
            _tool_call({"action": "left_click", "coordinate": [1, 1]}),
        ]
    )
    decision = _agent("native_absolute").decide(
        text, step=1, geometry=_geometry(), cursor=(100, 100), sampling=_sampling()
    )
    assert decision.control == "terminate" and decision.operations == ()
    assert decision.parse_error is None and decision.action.calls == ()
    assert decision.ignored_after_terminate == 1
    assert decision.as_record()["ignored_after_terminate"] == 1


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


def test_parsed_action_comes_from_the_grammars_own_to_dict() -> None:
    """`_action_record` calls `to_dict` by name, so a grammar that dropped it would
    read as covered while writing nothing."""
    import grammars

    for name in sorted(grammars.available()):
        action_type = type(load_codec(name).parse(_SMOKE_TEXT[name]))
        assert callable(getattr(action_type, "to_dict", None)), name
    action = load_codec("deltatype_v2").parse("0 0 0 ; +LMB -LMB")
    assert _action_record(action) == action.to_dict()
    assert _action_record(None) is None
    with pytest.raises(TypeError, match="to_dict"):
        _action_record(object())


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}',
        '```json\n{"name": "computer_use", "arguments": '
        '{"action": "terminate", "status": "failure"}}\n```',
        '[{"name": "computer_use", "arguments": '
        '{"action": "terminate", "status": "failure"}}]',
    ],
    ids=["bare_json", "fenced", "array"],
)
@pytest.mark.parametrize("codec_name", TOOL_CALL_GRAMMARS)
def test_untagged_tool_call_shapes_reach_the_same_control_outcome(
    codec_name: str, text: str
) -> None:
    """`_support.iter_tool_calls` accepts bare JSON, a ``` fence and a JSON array,
    because that is how the RL rollout path sees vLLM-parsed output. The control
    channel reads the vendor termination out of all three."""
    decision = _agent(codec_name).decide(
        text, step=1, geometry=_geometry(), cursor=(5, 5), sampling=_sampling()
    )
    assert decision.control == "fail" and decision.parse_error is None


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
    """A harness default riding the request body is silently dropped. Whatever that
    default is, the eval's value is what reaches the wire and what gets
    recorded."""
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
                "R",
                (),
                {
                    "message": type("M", (), {"content": [{"text": "a"}, {"text": "b"}]})(),
                    "finish_reason": "stop",
                },
            )()

    ctx = vf.ModelContext(model="m", client=ListClient(), sampling=vf.Sampling())
    assert asyncio.run(
        ContextTransport().complete(ctx, {"messages": []}, session_id="s")
    ) == ("ab", "stop")


def test_a_native_tool_call_is_infrastructure_not_an_empty_turn() -> None:
    """The invariant the recorded native-parse numbers rest on.

    Probed on the serving path the evals use (sglang 0.5.10.post1, no
    `--tool-call-parser`): `tool_calls` was null and the `<tool_call>` block came
    back inside `content`, on both an off-the-shelf 4B and a LoRA-merged
    checkpoint, with and without a tool schema in the request. A server that does
    populate `tool_calls` sends `content: null` with it, which read as `""` here
    and made every turn a parse error — 0% that looks like a model collapse.
    `vf.AssistantMessage.tool_calls` is a typed field verifiers fills from the wire
    (`dialects/chat.py:177-180`), so the shape is reachable, not hypothetical.
    """

    class NativeToolCalls(FakeClient):
        async def get_response(self, dialect, body, model, sampling, **kwargs):
            return type(
                "R",
                (),
                {
                    "message": vf.AssistantMessage(
                        content=None,
                        tool_calls=[
                            vf.ToolCall(
                                id="call_0",
                                name="computer_use",
                                arguments='{"action": "left_click"}',
                            )
                        ],
                    ),
                    "finish_reason": "tool_calls",
                },
            )()

    ctx = vf.ModelContext(model="m", client=NativeToolCalls(), sampling=vf.Sampling())
    with pytest.raises(ModelCallError, match="1 native tool call"):
        asyncio.run(ContextTransport().complete(ctx, {"messages": []}, session_id=None))


@pytest.mark.parametrize("key", ["tools", "tool_choice"])
def test_a_tool_protocol_on_the_wire_is_refused_before_the_call(key: str) -> None:
    """`vf.Sampling` is `extra="allow"`, so this needs no code change to happen.

    Probed with `tool_choice="required"`: the same server rewrote the turn into a
    bare JSON array that no codec parses, and left `tool_calls` null — so the
    response-side check above cannot see it and the arm reads as a collapse.
    """
    ctx = make_ctx(**{key: "auto" if key == "tool_choice" else []})
    with pytest.raises(ValueError, match=f"\\['{key}'\\]"):
        resolve_sampling(ctx, {"messages": []})


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


def test_dump_prompt_elides_image_bytes() -> None:
    agent = _agent("deltatype_v2")
    body = agent.build_body(history=_one_frame_history(), instruction="G", step=1)
    dumped = dump_prompt(body)
    assert "base64" not in dumped and "<image>" in dumped
    payload = json.loads(dumped)
    assert payload["messages"][0]["role"] == "system"
    assert len(dumped) < 4000, "a sidecar must stay small enough to read"


def test_load_codec_resolves_every_registered_grammar() -> None:
    """desktop exposes neither `CODECS` nor `load_codec` — it is deliberately
    grammar-free — so the in-tree fallback goes through `grammars.load`, which has
    its own peer-directory scan. Probing desktop instead would raise
    `LookupError` on any interpreter where juergen's entry points are not
    installed.
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
