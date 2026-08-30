"""Stage 04b: chord-anchored windowing of stage-04 conversations.

Covers the properties the filter promises:
  * a window BEGINS with the chord -- its first assistant turn is the one whose
    program presses the trigger key,
  * a window is at most 1 + --max-frames-after turns and never runs past the end
    of its source conversation,
  * ``--combo-scope turn`` keeps the whole chord inside that first turn, while
    ``conversation`` scope accepts a modifier held from an earlier turn (v3
    programs really do carry a key across a turn boundary),
  * modifier groups are side-agnostic, and a key never satisfies a combo through
    itself.
"""

from __future__ import annotations

import unittest

from realigned_pipeline.stage_04b_filter_key_combo import (
    KeyCombo,
    aggregate_transitions,
    build_window_row,
    find_windows,
    leading_text_blocks,
    ordered_transitions,
    split_combo_specs,
    split_messages,
)

SYSTEM = {"role": "system", "content": [{"type": "text", "text": "sys"}]}


def user(i: int) -> dict:
    return {"role": "user", "content": [{"type": "image", "image": f"ar://frame#{i}"}]}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def conversation(actions: list[str], instruction: str | None = None) -> dict:
    """A stage-04-shaped row whose assistant turns are ``actions``."""
    messages: list[dict] = [SYSTEM]
    for i, action in enumerate(actions):
        turn = user(i)
        if i == 0 and instruction is not None:
            turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": f"ar://frame#{i}"},
                ],
            }
        messages.append(turn)
        messages.append(assistant(action))
    return {
        "conversation_id": "seg0_app00",
        "recording_id": "rec",
        "app": "org.mozilla.firefox",
        "instruction": instruction,
        "n_turns": len(actions),
        "n_frames": len(actions),
        "messages": messages,
    }


def windows_of(actions: list[str], combo: str = "Meta+KeyT", **kwargs) -> list[dict]:
    row = conversation(actions)
    _system, pairs = split_messages(row["messages"])
    opts = {
        "max_frames_after": 3,
        "min_frames_after": 0,
        "same_turn": True,
        "strict": False,
        "allow_overlap": False,
    }
    opts.update(kwargs)
    return find_windows(pairs, [KeyCombo(combo)], ordered_transitions, **opts)


class TestComboSpec(unittest.TestCase):
    def test_last_token_is_the_trigger(self):
        combo = KeyCombo("Meta+Shift+KeyT")
        self.assertEqual(combo.trigger, frozenset({"KeyT"}))
        self.assertEqual(len(combo.modifiers), 2)

    def test_modifier_groups_are_side_agnostic(self):
        combo = KeyCombo("Meta+KeyT")
        self.assertEqual(combo.modifiers[0], frozenset({"MetaLeft", "MetaRight"}))
        for side in ("MetaLeft", "MetaRight"):
            self.assertTrue(
                combo.matches("KeyT", {side: 0}, 0, same_turn=True, strict=False),
                side,
            )

    def test_raw_names_pass_through(self):
        # the corpus vocabulary must work verbatim, groups or not
        combo = KeyCombo("ControlRight+PageDown")
        self.assertEqual(combo.trigger, frozenset({"PageDown"}))
        self.assertTrue(
            combo.matches("PageDown", {"ControlRight": 0}, 0,
                          same_turn=True, strict=False)
        )
        self.assertFalse(
            combo.matches("PageDown", {"ControlLeft": 0}, 0,
                          same_turn=True, strict=False)
        )

    def test_bare_spec_is_an_unmodified_press(self):
        combo = KeyCombo("Return")
        self.assertEqual(combo.modifiers, [])
        self.assertTrue(combo.matches("Return", {}, 0, same_turn=True, strict=False))

    def test_strict_modifiers_rejects_extras(self):
        combo = KeyCombo("Meta+KeyT")
        held = {"MetaLeft": 0, "ShiftLeft": 0}
        self.assertTrue(combo.matches("KeyT", held, 0, same_turn=True, strict=False))
        self.assertFalse(combo.matches("KeyT", held, 0, same_turn=True, strict=True))

    def test_non_modifier_held_keys_never_count_as_extras(self):
        combo = KeyCombo("Meta+KeyT")
        held = {"MetaLeft": 0, "LMB": 0}
        self.assertTrue(combo.matches("KeyT", held, 0, same_turn=True, strict=True))

    def test_split_combo_specs(self):
        self.assertEqual(
            split_combo_specs(["Meta+KeyT,Meta+KeyL", "Return"]),
            ["Meta+KeyT", "Meta+KeyL", "Return"],
        )


class TestAnchoring(unittest.TestCase):
    def test_window_starts_at_the_chord(self):
        actions = [
            "move(1,1)",
            "scroll(0,-3)",
            "down(MetaLeft); down(KeyT)",
            "up(KeyT); up(MetaLeft)",
            'type("news")',
            "down(Return)",
        ]
        [w] = windows_of(actions)
        self.assertEqual(w["start"], 2)
        self.assertEqual(w["key_combo"], "Meta+KeyT")
        self.assertEqual(w["trigger_key"], "KeyT")

    def test_no_match_without_the_modifier(self):
        self.assertEqual(windows_of(["down(KeyT); up(KeyT)", "move(1,1)"]), [])

    def test_a_key_cannot_satisfy_a_combo_through_itself(self):
        # ShiftLeft going down must not be its own held modifier
        self.assertEqual(
            windows_of(["down(ShiftLeft)", "move(1,1)"], combo="Shift+ShiftLeft"), []
        )

    def test_held_state_survives_a_turn_boundary(self):
        actions = [
            "down(MetaLeft)",              # modifier pressed here...
            "down(KeyT); up(KeyT)",        # ...trigger a turn later
            "up(MetaLeft)",
        ]
        self.assertEqual(windows_of(actions, same_turn=True), [])
        [w] = windows_of(actions, same_turn=False)
        self.assertEqual(w["start"], 1)

    def test_release_clears_held_state(self):
        actions = [
            "down(MetaLeft); up(MetaLeft)",
            "down(KeyT); up(KeyT)",
        ]
        self.assertEqual(windows_of(actions, same_turn=False), [])


class TestWindowBounds(unittest.TestCase):
    def test_max_frames_after_bounds_the_window(self):
        actions = ["down(MetaLeft); down(KeyT)"] + ["move(1,1)"] * 10
        [w] = windows_of(actions, max_frames_after=3)
        self.assertEqual((w["start"], w["end"], w["frames_after"]), (0, 3, 3))

    def test_window_truncates_at_the_end_of_the_conversation(self):
        actions = ["move(1,1)", "down(MetaLeft); down(KeyT)", "move(1,1)"]
        [w] = windows_of(actions, max_frames_after=10)
        self.assertEqual((w["start"], w["end"], w["frames_after"]), (1, 2, 1))

    def test_min_frames_after_drops_stubs(self):
        actions = ["move(1,1)", "down(MetaLeft); down(KeyT)", "move(1,1)"]
        self.assertEqual(windows_of(actions, min_frames_after=2), [])

    def test_overlapping_triggers_are_suppressed_by_default(self):
        actions = [
            "down(MetaLeft); down(KeyT)",
            "up(KeyT); up(MetaLeft)",
            "down(MetaLeft); down(KeyT)",   # inside the first window
            "up(KeyT); up(MetaLeft)",
            "move(1,1)",
            "move(1,1)",
            "down(MetaLeft); down(KeyT)",   # clear of it
        ]
        starts = [w["start"] for w in windows_of(actions, max_frames_after=3)]
        self.assertEqual(starts, [0, 6])
        overlapped = [
            w["start"] for w in windows_of(actions, max_frames_after=3,
                                           allow_overlap=True)
        ]
        self.assertEqual(overlapped, [0, 2, 6])


class TestRowBuilding(unittest.TestCase):
    def _row(self, actions, instruction=None, **kwargs):
        row = conversation(actions, instruction=instruction)
        system, pairs = split_messages(row["messages"])
        opts = {
            "max_frames_after": 3, "min_frames_after": 0, "same_turn": True,
            "strict": False, "allow_overlap": False,
        }
        opts.update(kwargs)
        [w] = find_windows(pairs, [KeyCombo("Meta+KeyT")], ordered_transitions, **opts)
        return build_window_row(
            row, system, pairs, w, 0, ordered_transitions, carry_instruction=True
        )

    def test_messages_are_system_plus_the_window(self):
        actions = ["move(1,1)", "down(MetaLeft); down(KeyT)", "NO_OP", "move(2,2)"]
        out = self._row(actions)
        roles = [m["role"] for m in out["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"] * 1 + [
            "user", "assistant", "user", "assistant",
        ])
        firsts = [m for m in out["messages"] if m["role"] == "assistant"]
        self.assertEqual(firsts[0]["content"][0]["text"], "down(MetaLeft); down(KeyT)")
        self.assertEqual(out["n_turns"], 3)
        self.assertEqual(out["n_frames"], 3)
        self.assertEqual(out["n_non_noop"], 2)  # the NO_OP turn does not count

    def test_provenance(self):
        out = self._row(["move(1,1)", "down(MetaLeft); down(KeyT)", "move(2,2)"])
        self.assertEqual(out["conversation_id"], "seg0_app00_kc000")
        self.assertEqual(out["source_conversation_id"], "seg0_app00")
        self.assertEqual(out["combo_turn_index"], 1)
        self.assertEqual(out["source_n_turns"], 3)
        self.assertEqual(out["key_combo"], "Meta+KeyT")
        self.assertEqual(out["app"], "org.mozilla.firefox")  # source fields survive

    def test_instruction_is_carried_onto_a_mid_conversation_window(self):
        out = self._row(
            ["move(1,1)", "down(MetaLeft); down(KeyT)", "move(2,2)"],
            instruction="book a flight",
        )
        first_user = out["messages"][1]
        self.assertEqual(
            [b["type"] for b in first_user["content"]], ["text", "image"]
        )
        self.assertEqual(first_user["content"][0]["text"], "book a flight")

    def test_goal_free_windows_get_no_text_block(self):
        out = self._row(["move(1,1)", "down(MetaLeft); down(KeyT)", "move(2,2)"])
        self.assertEqual([b["type"] for b in out["messages"][1]["content"]], ["image"])


class TestParsers(unittest.TestCase):
    def test_ordered_transitions(self):
        self.assertEqual(
            ordered_transitions('move(1,2); down(MetaLeft); type("x"); up(MetaLeft)'),
            [("down", "MetaLeft"), ("up", "MetaLeft")],
        )

    def test_aggregate_transitions(self):
        self.assertEqual(
            aggregate_transitions("0 0 0 ; +MetaLeft +KeyT -KeyT -MetaLeft"),
            [("down", "MetaLeft"), ("down", "KeyT"),
             ("up", "KeyT"), ("up", "MetaLeft")],
        )

    def test_aggregate_format_anchors_too(self):
        row = conversation([
            "0 0 0",
            "0 0 0 ; +MetaLeft +KeyT",
            "0 0 0 ; -KeyT -MetaLeft",
        ])
        _system, pairs = split_messages(row["messages"])
        [w] = find_windows(
            pairs, [KeyCombo("Meta+KeyT")], aggregate_transitions,
            max_frames_after=3, min_frames_after=0, same_turn=True,
            strict=False, allow_overlap=False,
        )
        self.assertEqual(w["start"], 1)


class TestMessageSplitting(unittest.TestCase):
    def test_rejects_broken_alternation(self):
        with self.assertRaises(ValueError):
            split_messages([SYSTEM, user(0), user(1), assistant("NO_OP")])

    def test_rejects_a_trailing_unpaired_message(self):
        with self.assertRaises(ValueError):
            split_messages([SYSTEM, user(0), assistant("NO_OP"), user(1)])

    def test_leading_text_blocks_stop_at_the_image(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "goal"},
                {"type": "image", "image": "ar://x"},
                {"type": "text", "text": "trailing"},
            ],
        }
        self.assertEqual(leading_text_blocks(msg), [{"type": "text", "text": "goal"}])


if __name__ == "__main__":
    unittest.main()
