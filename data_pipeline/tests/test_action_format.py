import unittest

from annotation_pipeline.action_format import (
    HeldStateDiagnostics,
    project_ordered_action,
    update_held_state,
)


def _event(index: int, time_s: float, kind: str, **fields):
    return {
        "source_event_idx": index,
        "local_time_s": time_s,
        "kind": kind,
        **fields,
    }


class OrderedActionFormatTest(unittest.TestCase):
    def test_discrete_event_splits_movement_inside_one_motor_tick(self) -> None:
        result = project_ordered_action(
            [
                _event(0, 2.01, "move", dx=1.0, dy=0.0),
                _event(1, 2.02, "move", dx=3.0, dy=-1.0),
                _event(2, 2.03, "press", key="LMB"),
                _event(3, 2.04, "move", dx=2.0, dy=0.0),
                _event(4, 2.05, "release", key="LMB"),
            ],
            interval_start_s=2.0,
            continuous_action_hz=10.0,
        )

        self.assertEqual(result.text, "move(4,-1); down(LMB); move(2,0); up(LMB)")

    def test_motor_tick_boundary_splits_continuous_actions(self) -> None:
        result = project_ordered_action(
            [
                _event(0, 4.01, "move", dx=1.0, dy=0.0),
                _event(1, 4.11, "move", dx=2.0, dy=0.0),
            ],
            interval_start_s=4.0,
            continuous_action_hz=10.0,
        )

        self.assertEqual(result.text, "move(1,0); move(2,0)")

    def test_scroll_is_ordered_and_two_dimensional(self) -> None:
        result = project_ordered_action(
            [
                _event(0, 0.01, "scroll", dx=2.0, dy=-3.0),
                _event(1, 0.02, "scroll", dx=1.0, dy=-2.0),
                _event(2, 0.03, "press", key="KeyA"),
                _event(3, 0.04, "scroll", dx=-1.0, dy=4.0),
            ],
            interval_start_s=0.0,
            continuous_action_hz=10.0,
        )

        self.assertEqual(result.text, "scroll(3,-5); down(KeyA); scroll(-1,4)")

    def test_zero_continuous_actions_are_omitted(self) -> None:
        result = project_ordered_action(
            [
                _event(0, 0.01, "move", dx=0.4, dy=0.4),
                _event(1, 0.02, "scroll", dx=0.0, dy=0.0),
                _event(2, 0.03, "press", key="LMB"),
                _event(3, 0.04, "release", key="LMB"),
            ],
            interval_start_s=0.0,
            continuous_action_hz=10.0,
        )

        self.assertEqual(result.text, "down(LMB); up(LMB)")

    def test_empty_projection_is_no_op(self) -> None:
        result = project_ordered_action([], interval_start_s=0.0, continuous_action_hz=10.0)

        self.assertEqual(result.text, "NO_OP")
        self.assertEqual(result.primitives, ())

    def test_invalid_rate_and_event_kind_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous_action_hz"):
            project_ordered_action([], interval_start_s=0.0, continuous_action_hz=0.0)
        with self.assertRaisesRegex(ValueError, "Unsupported action event kind"):
            project_ordered_action(
                [_event(0, 0.0, "context")],
                interval_start_s=0.0,
                continuous_action_hz=10.0,
            )

    def test_state_diagnostics_do_not_mutate_primitives(self) -> None:
        result = project_ordered_action(
            [
                _event(0, 0.01, "release", key="LMB"),
                _event(1, 0.02, "press", key="KeyA"),
                _event(2, 0.03, "press", key="KeyA"),
            ],
            interval_start_s=0.0,
            continuous_action_hz=10.0,
        )
        diagnostics = HeldStateDiagnostics()
        held: set[str] = set()

        update_held_state(result.primitives, held=held, diagnostics=diagnostics)
        diagnostics.finish_trajectory(held)

        self.assertEqual(result.text, "up(LMB); down(KeyA); down(KeyA)")
        self.assertEqual(diagnostics.dangling_up, 1)
        self.assertEqual(diagnostics.duplicate_down, 1)
        self.assertEqual(diagnostics.non_neutral_trajectory, 1)
        self.assertEqual(diagnostics.held_at_trajectory_end, 1)


if __name__ == "__main__":
    unittest.main()
