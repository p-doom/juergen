"""Native-relative format: eval-side parse must recover exactly what training emits.

Guards the train/eval contract for videocua_nativerel_v1:
  1. SYSTEM_PROMPTS["native_rel_v1"] is byte-identical to the prompt baked into
     the training data (native_rel_format.SYSTEM_PROMPT).
  2. parse_computer_use_tool_calls() recovers the same ordered action list that
     the dataset builder emitted for each assistant turn -- including multi-call
     turns (drags -> mouse_down, mouse_move(s), mouse_up).

Run: python test_native_rel_prompt.py   (or via pytest)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_DS = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/franz.srambical/videocua_nativerel_v1"
sys.path.insert(0, _DS)

import native_rel_format as nrf  # noqa: E402
from action_parser import parse_computer_use_tool_calls  # noqa: E402
from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402

_VAL_CHAT = os.path.join(_DS, "_normalized", "val", "chat.jsonl")


def test_prompt_byte_identical():
    assert SYSTEM_PROMPTS["native_rel_v1"] == nrf.SYSTEM_PROMPT, (
        "native_rel_v1 eval prompt drifted from native_rel_format.SYSTEM_PROMPT"
    )


def test_parse_recovers_builder_actions():
    n_turns = 0
    n_multi = 0
    with open(_VAL_CHAT) as f:
        for line in f:
            rec = json.loads(line)
            for m in rec["messages"]:
                if m["role"] != "assistant":
                    continue
                text = m["content"][0]["text"]
                # Eval parser recovers the encoded args; re-rendering them with the
                # builder's serializer must reproduce the exact bytes -> the parser
                # sees precisely what training emitted (order + values).
                parsed = [c.arguments for c in parse_computer_use_tool_calls(text)]
                assert nrf.render_assistant_text(parsed) == text, (
                    f"parse/render mismatch:\n text={text!r}\n parsed={parsed}"
                )
                n_turns += 1
                if len(parsed) > 1:
                    n_multi += 1
    assert n_turns > 5000, f"expected many turns, got {n_turns}"
    assert n_multi > 0, "expected some multi-call (drag) turns"
    print(f"OK: {n_turns} assistant turns parsed, {n_multi} multi-call (drag) turns")


if __name__ == "__main__":
    test_prompt_byte_identical()
    print("OK: prompt byte-identical")
    test_parse_recovers_builder_actions()
