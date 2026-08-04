from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from converter import replace_action_span


from conftest import ROOT

# Vendored byte-identically from the audited operand tree; the sealed build
# hash-pinned it at 65397c1d…, asserted by test_pinned_contract.py.
AUDITED_CONVERTER = ROOT / "vendor" / "action_span_conversion.py"


def load_converter():
    name = "raw_v2_test_audited_converter"
    spec = importlib.util.spec_from_file_location(name, AUDITED_CONVERTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_converter_preserves_all_bytes_outside_multi_call_span():
    conversion = load_converter()
    source = (
        "Keep this reasoning byte exact.\n"
        '<tool_call>\n{"name":"computer_use","arguments":{"action":"mouse_move",'
        '"coordinate":[100,200]}}\n</tool_call>\n'
        '<tool_call>\n{"name":"computer_use","arguments":{"action":"left_click_drag",'
        '"coordinate":[300,400]}}\n</tool_call>\n'
    )
    label = "-768 -324 0 ; +LMB MOVE(384,216) -LMB"
    before, old_action, after = conversion.split_assistant_turn(source)
    output, returned_old = replace_action_span(conversion, source, label)
    new_before, new_action, new_after = conversion.split_assistant_turn(output)
    assert returned_old == old_action
    assert (new_before, new_after) == (before, after)
    assert new_action == label
