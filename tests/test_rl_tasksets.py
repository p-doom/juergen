"""Item 7 (`_never_moved`) plus the `grounding` / `target_box` tasksets and geometry.

`_never_moved` lives in `rl/grounding/taskset.py:69`, and it replaces a
`distance = -1.0` sentinel that conflated three different things — no target, no
movement, and a target one pixel away. It exists because a model that emits `wait`,
`terminate` or a coordinate-less click never moves and therefore can never *miss*:
without a negative term, not trying is strictly safer than trying, and GRPO finds it.
The required total ordering is

    no-move  -0.15   <   miss  (0, 0.3)   <   hit  1.0

and that ordering is asserted directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from juergen_doubles import make_task_data, make_trace, png
from rl.geometry import in_bbox
from rl.grounding.dataset import GroundingTarget, cursor_start, load_canvas, load_targets
from rl.grounding.taskset import (
    NO_MOVE_PENALTY,
    SHAPING_SCALE,
    SHAPING_WEIGHT,
    GroundingTask,
    GroundingTaskset,
    GroundingTasksetConfig,
    _never_moved,
)
from rl.target_box.geometry import (
    TARGET_BOX_INSTRUCTION,
    TargetBoxConfig,
    annotate,
    sample_box,
    sample_cursor_start,
)
from rl.target_box.taskset import (
    NO_SIGNAL_PENALTY,
    TargetBoxTask,
    TargetBoxTaskset,
    TargetBoxTasksetConfig,
)


class _Runtime:
    id = "runtime-1"


def _step(before, after):
    return {"cursor_before": list(before), "cursor_after": list(after)}


def test_never_moved_is_true_when_no_step_changed_the_cursor() -> None:
    assert _never_moved({"steps_detail": [_step((5, 5), (5, 5))]})
    assert _never_moved({"steps_detail": [_step((5, 5), (5, 5)), _step((5, 5), (5, 5))]})


def test_never_moved_is_false_as_soon_as_one_step_moved() -> None:
    assert not _never_moved({"steps_detail": [_step((5, 5), (6, 5))]})
    assert not _never_moved(
        {"steps_detail": [_step((5, 5), (5, 5)), _step((5, 5), (900, 400))]}
    )


def test_never_moved_ignores_malformed_step_rows() -> None:
    assert _never_moved({"steps_detail": [None, "junk", _step((1, 1), (1, 1))]})


def test_never_moved_is_vacuously_true_for_an_empty_rollout() -> None:
    """Documented, not accidental: a rollout that dispatched nothing moved nothing."""
    assert _never_moved({}) and _never_moved({"steps_detail": []})


def test_never_moved_is_distinct_from_an_unparseable_reply() -> None:
    """A well-formed `wait` parses fine and still moves nothing; conflating the two
    hides which failure the policy actually has."""
    parsed_but_still = {
        "steps_detail": [{**_step((5, 5), (5, 5)), "parse_ok": True, "control": "no_op"}]
    }
    unparseable = {
        "steps_detail": [{**_step((5, 5), (5, 5)), "parse_ok": False, "parse_error": {"type": "X"}}]
    }
    assert _never_moved(parsed_but_still) and _never_moved(unparseable)
    assert parsed_but_still["steps_detail"][0]["parse_ok"] is True


def _ground(**result):
    data = make_task_data(kind="grounding_canvas", bbox=(10, 10, 50, 50))
    trace = make_trace(data, episode=result)
    asyncio.run(GroundingTask(data).score(trace, _Runtime()))
    return trace


def test_the_reward_ordering_is_no_move_below_miss_below_hit() -> None:
    """The ordering the negative term exists to create."""
    hit = _ground(reach_frame=1, best_distance=0.0, steps_detail=[_step((0, 0), (20, 20))])
    near_miss = _ground(
        reach_frame=-1, best_distance=15.0, steps_detail=[_step((0, 0), (60, 20))]
    )
    far_miss = _ground(
        reach_frame=-1, best_distance=900.0, steps_detail=[_step((0, 0), (900, 900))]
    )
    no_move = _ground(reach_frame=-1, best_distance=500.0, steps_detail=[_step((5, 5), (5, 5))])

    def total(trace):
        return sum(trace.rewards.values())

    assert total(no_move) < total(far_miss) < total(near_miss) < total(hit), {
        "no_move": total(no_move),
        "far_miss": total(far_miss),
        "near_miss": total(near_miss),
        "hit": total(hit),
    }
    assert total(no_move) == pytest.approx(-NO_MOVE_PENALTY)
    assert total(hit) == 1.0
    assert 0.0 < total(far_miss) < SHAPING_WEIGHT


def test_not_moving_is_penalised_even_when_the_start_was_close() -> None:
    """Otherwise a lucky start would make standing still profitable."""
    trace = _ground(reach_frame=-1, best_distance=1.0, steps_detail=[_step((9, 9), (9, 9))])
    assert trace.rewards["shaped_progress"] == pytest.approx(-NO_MOVE_PENALTY)


def test_an_undefined_distance_is_also_penalised() -> None:
    trace = _ground(reach_frame=-1, best_distance=-1.0, steps_detail=[_step((0, 0), (5, 5))])
    assert trace.rewards["shaped_progress"] == pytest.approx(-NO_MOVE_PENALTY)


def test_a_hit_gets_no_shaping_on_top() -> None:
    trace = _ground(reach_frame=1, best_distance=0.0, steps_detail=[_step((0, 0), (20, 20))])
    assert trace.rewards["reach"] == 1.0 and trace.rewards["shaped_progress"] == 0.0


def test_the_grounding_metric_reports_never_moved_separately_from_distance() -> None:
    trace = _ground(reach_frame=-1, best_distance=-1.0, steps_detail=[_step((5, 5), (5, 5))])
    assert trace.metrics["never_moved"] == 1.0
    assert trace.metrics["distance_px"] == -1.0
    assert trace.metrics["reached"] == 0.0
    moved = _ground(reach_frame=-1, best_distance=42.0, steps_detail=[_step((0, 0), (5, 5))])
    assert moved.metrics["never_moved"] == 0.0 and moved.metrics["distance_px"] == 42.0


def test_the_shaping_term_decays_on_the_documented_scale() -> None:
    import math

    trace = _ground(
        reach_frame=-1, best_distance=SHAPING_SCALE, steps_detail=[_step((0, 0), (1, 1))]
    )
    assert trace.rewards["shaped_progress"] == pytest.approx(SHAPING_WEIGHT * math.exp(-1.0))


def test_the_grounding_reward_raises_on_a_missing_result() -> None:
    data = make_task_data(kind="grounding_canvas", bbox=(10, 10, 50, 50))
    with pytest.raises(Exception, match="published no result"):
        asyncio.run(GroundingTask(data).score(make_trace(data), _Runtime()))


def test_the_grounding_rewards_are_all_trace_only_so_replay_scores_them() -> None:
    from verifiers.v1.task import _requires_runtime

    for name in ("reach", "shaped_progress"):
        assert not _requires_runtime(getattr(GroundingTask, name)), name


def _labels(tmp_path: Path, n: int = 2) -> Path:
    rows = []
    for i in range(n):
        steps = tmp_path / "run" / f"task_{i}" / "steps"
        steps.mkdir(parents=True, exist_ok=True)
        (steps / "step_001.png").write_bytes(png(320, 200))
        rows.append(
            {
                "idx": i,
                "app": "chrome",
                "instruction": f"click target {i}",
                "bbox_xyxy": [10 + i, 20, 60 + i, 70],
                "image_path": str(steps / "step_001.png"),
            }
        )
    path = tmp_path / "bboxes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_targets_reads_the_task_id_and_screen_from_the_image_path(tmp_path) -> None:
    targets = load_targets(_labels(tmp_path, 2))
    assert [t.task_id for t in targets] == ["task_0", "task_1"]
    assert all(t.screen == (320, 200) for t in targets), "screen comes from the PNG itself"
    assert targets[0].bbox == (10, 20, 60, 70)
    assert targets[0].app == "chrome"


def test_load_targets_skips_blank_lines(tmp_path) -> None:
    path = _labels(tmp_path, 1)
    path.write_text(path.read_text() + "\n\n")
    assert len(load_targets(path)) == 1


def test_load_targets_refuses_an_image_path_of_the_wrong_shape(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "idx": 0,
                "app": "a",
                "instruction": "i",
                "bbox_xyxy": [1, 2, 3, 4],
                "image_path": "/no/frames/here.png",
            }
        )
    )
    with pytest.raises(ValueError, match="unexpected image_path shape"):
        load_targets(path)


def test_the_container_free_grounding_start_delegates_to_the_shared_rule(tmp_path) -> None:
    """So the two grounding evals cannot drift on the one variable they control."""
    from evals.tasks import cursor_start as shared

    target = load_targets(_labels(tmp_path, 1))[0]
    for regime in ("near", "medium", "far"):
        assert cursor_start(target, 320, 200, regime) == shared(
            target.bbox, 320, 200, regime, target.task_id
        )


def test_load_canvas_returns_the_labelled_screenshot(tmp_path) -> None:
    target = load_targets(_labels(tmp_path, 1))[0]
    canvas = load_canvas(target)
    assert canvas.size == (320, 200) and canvas.mode == "RGB"


def test_the_grounding_taskset_is_the_target_by_regime_cross_product(tmp_path) -> None:
    config = GroundingTasksetConfig(bboxes_jsonl=str(_labels(tmp_path, 2)))
    rows = list(GroundingTaskset(config).load())
    assert len(rows) == 6, "2 targets x 3 regimes"
    assert len({r.data.idx for r in rows}) == 6, "indices must be unique"
    assert rows[0].data.name == "chrome/task_0/near"
    assert rows[0].data.kind == "grounding_canvas"
    assert rows[0].data.max_steps == 1, "single-step by default: the frame is a FINAL state"
    assert rows[0].data.cursor_start is not None
    assert rows[0].data.setup["screen"] == [320, 200]


def test_the_grounding_taskset_honours_target_and_regime_filters(tmp_path) -> None:
    path = _labels(tmp_path, 3)
    only_one = GroundingTasksetConfig(bboxes_jsonl=str(path), target_idxs=[1])
    rows = list(GroundingTaskset(only_one).load())
    assert {r.data.name.split("/")[1] for r in rows} == {"task_1"}
    capped = GroundingTasksetConfig(bboxes_jsonl=str(path), max_targets=2, regimes=["near"])
    rows = list(GroundingTaskset(capped).load())
    assert len(rows) == 2 and all(r.data.regime == "near" for r in rows)


def test_a_sampled_box_sits_inside_the_margins() -> None:
    config = TargetBoxConfig()
    for key in ("a:1", "b:2", "c:3"):
        box = sample_box(config, screen_width=1920, screen_height=1080, instance_key=key)
        assert box[0] >= config.margin and box[1] >= config.margin
        assert box[2] <= 1920 - config.margin and box[3] <= 1080 - config.margin
        assert box[2] - box[0] == config.box_width - 1
        assert box[3] - box[1] == config.box_height - 1


def test_the_scene_is_deterministic_in_the_instance_key() -> None:
    config = TargetBoxConfig()
    first = sample_box(config, screen_width=1920, screen_height=1080, instance_key="t:p")
    assert first == sample_box(config, screen_width=1920, screen_height=1080, instance_key="t:p")
    other = sample_box(config, screen_width=1920, screen_height=1080, instance_key="t:q")
    assert other != first, "a different task must get a different box"
    seeded = sample_box(
        TargetBoxConfig(seed=9), screen_width=1920, screen_height=1080, instance_key="t:p"
    )
    assert seeded != first, "the seed must move the scene"


def test_a_sampled_cursor_start_is_outside_the_box_and_inside_the_margins() -> None:
    config = TargetBoxConfig()
    for key in (f"k{i}:p" for i in range(30)):
        box = sample_box(config, screen_width=1920, screen_height=1080, instance_key=key)
        cursor = sample_cursor_start(
            config, box, screen_width=1920, screen_height=1080, instance_key=key
        )
        assert not in_bbox(cursor, box), (key, box, cursor)
        assert config.cursor_margin <= cursor[0] <= 1920 - config.cursor_margin - 1
        assert config.cursor_margin <= cursor[1] <= 1080 - config.cursor_margin - 1


def test_a_cursor_start_falls_back_to_the_furthest_admissible_corner() -> None:
    """A box that swallows the random samples must not loop forever."""
    config = TargetBoxConfig(cursor_margin=0)
    box = (0, 0, 1900, 1070)  # nearly the whole admissible region
    cursor = sample_cursor_start(
        config, box, screen_width=1920, screen_height=1080, instance_key="k:p"
    )
    assert not in_bbox(cursor, box)


def test_a_box_covering_every_admissible_corner_is_an_error() -> None:
    config = TargetBoxConfig(cursor_margin=0)
    with pytest.raises(ValueError, match="every admissible cursor corner"):
        sample_cursor_start(
            config, (0, 0, 1920, 1080), screen_width=1920, screen_height=1080, instance_key="k"
        )


@pytest.mark.parametrize(
    "kwargs,screen,message",
    [
        ({}, (0, 1080), "screen dimensions must be positive"),
        ({}, (1920, 0), "screen dimensions must be positive"),
        ({"margin": -1}, (1920, 1080), "margins must be non-negative"),
        ({"cursor_margin": -1}, (1920, 1080), "margins must be non-negative"),
        ({"box_width": 2000}, (1920, 1080), "box width plus margins"),
        ({"box_height": 1200}, (1920, 1080), "box height plus margins"),
        ({"cursor_margin": 600}, (1920, 1080), "no admissible cursor region"),
    ],
)
def test_a_misconfigured_scene_is_refused_rather_than_silently_impossible(
    kwargs, screen, message
) -> None:
    """A box wider than the screen minus margins produces 'hard' episodes that are
    actually impossible."""
    config = TargetBoxConfig(**kwargs)
    with pytest.raises(ValueError, match=message):
        config.validate(screen_width=screen[0], screen_height=screen[1])


def test_annotate_draws_the_box_and_no_cursor_marker() -> None:
    """The genuine desktop cursor is already in the frame; a synthetic marker would
    teach the model to look for something inference will not have."""
    import io

    from PIL import Image

    original = png(200, 150, colour=(0, 0, 0))
    annotated = annotate(original, (20, 20, 120, 100))
    assert annotated != original
    with Image.open(io.BytesIO(annotated)) as handle:
        pixels = handle.convert("RGB").load()
        assert pixels[20, 20] == (0, 255, 0), "the box outline is drawn"
        assert pixels[160, 130] == (0, 0, 0), "nothing else is"


def _osworld_tasks(tmp_path: Path, n: int = 2) -> Path:
    root = tmp_path / "examples"
    root.mkdir(parents=True)
    for i in range(n):
        (root / f"t{i}.json").write_text(
            json.dumps({"id": f"t{i}", "instruction": "real instruction", "config": [{"type": "x"}]})
        )
    return root


def test_the_target_box_taskset_replaces_the_real_instruction(tmp_path) -> None:
    """The background is a real desktop; the *task* is the synthetic box."""
    config = TargetBoxTasksetConfig(base_path=str(_osworld_tasks(tmp_path, 2)))
    rows = list(TargetBoxTaskset(config).load())
    assert len(rows) == 2
    for row in rows:
        assert row.data.instruction == TARGET_BOX_INSTRUCTION
        assert "real instruction" not in (row.data.instruction or "")
        assert row.data.kind == "target_box"
        assert row.data.setup["instance_key"].startswith(row.data.name.split("/")[1] + ":")
        assert row.data.setup["screen"] == [1920, 1080]
        assert row.data.setup["config"] == [{"type": "x"}]


def test_the_target_box_taskset_reads_an_explicit_task_list(tmp_path) -> None:
    root = _osworld_tasks(tmp_path, 3)
    listing = tmp_path / "tasks.txt"
    listing.write_text("t2\nt0.json\n\n")
    config = TargetBoxTasksetConfig(base_path=str(root), tasks_file=str(listing))
    assert [r.data.name for r in TargetBoxTaskset(config).load()] == [
        "target_box/t2",
        "target_box/t0",
    ]


def test_max_tasks_truncates_the_target_box_taskset(tmp_path) -> None:
    config = TargetBoxTasksetConfig(base_path=str(_osworld_tasks(tmp_path, 5)), max_tasks=2)
    assert len(list(TargetBoxTaskset(config).load())) == 2


def test_a_task_json_with_no_id_is_refused(tmp_path) -> None:
    root = tmp_path / "examples"
    root.mkdir()
    (root / "bad.json").write_text(json.dumps({"instruction": "x", "config": []}))
    config = TargetBoxTasksetConfig(base_path=str(root))
    with pytest.raises(ValueError, match="no string id"):
        list(TargetBoxTaskset(config).load())


def test_the_scene_derived_from_a_row_is_stable_across_two_loads(tmp_path) -> None:
    config = TargetBoxTasksetConfig(base_path=str(_osworld_tasks(tmp_path, 1)))
    first = list(TargetBoxTaskset(config).load())[0].data
    second = list(TargetBoxTaskset(config).load())[0].data
    assert first.setup["instance_key"] == second.setup["instance_key"]
    box_config = TargetBoxConfig(**first.setup["box"])
    box_a = sample_box(
        box_config, screen_width=1920, screen_height=1080, instance_key=first.setup["instance_key"]
    )
    box_b = sample_box(
        box_config, screen_width=1920, screen_height=1080, instance_key=second.setup["instance_key"]
    )
    assert box_a == box_b, "every worker must derive the same scene for the same row"


def _target_box(**result):
    data = make_task_data(kind="target_box")
    trace = make_trace(data, episode={"validity": "valid", **result})
    asyncio.run(TargetBoxTask(data).score(trace, _Runtime()))
    return trace


def test_target_box_success_requires_reaching_AND_declaring() -> None:
    declared = _target_box(
        outcome="postcondition_reached", reach_frame=2, control_terminate="terminate"
    )
    assert declared.rewards["target_box"] == 1.0
    assert declared.metrics["declared_success"] == 1.0
    entered_only = _target_box(outcome="max_steps", reach_frame=2, best_distance=0.0)
    assert entered_only.rewards["target_box"] == 0.0
    assert entered_only.metrics["entered_box"] == 1.0
    assert entered_only.metrics["declared_success"] == 0.0


def test_the_target_box_shaping_reads_best_not_final_distance() -> None:
    """The anti-limit-cycle term: a policy that passes through and overshoots is
    closer to solving than one that never approaches."""
    oscillator = _target_box(outcome="max_steps", reach_frame=-1, best_distance=5.0)
    stayer = _target_box(outcome="max_steps", reach_frame=-1, best_distance=900.0)
    assert oscillator.rewards["shaped_progress"] > stayer.rewards["shaped_progress"]


def test_a_target_box_rollout_with_no_signal_is_penalised() -> None:
    trace = _target_box(outcome="max_steps", reach_frame=-1, best_distance=-1.0)
    assert trace.rewards["shaped_progress"] == pytest.approx(-NO_SIGNAL_PENALTY)


def test_a_reached_target_box_gets_no_shaping_on_top() -> None:
    trace = _target_box(outcome="postcondition_reached", reach_frame=1, best_distance=0.0)
    assert trace.rewards["shaped_progress"] == 0.0


def test_an_infra_invalid_target_box_rollout_raises() -> None:
    data = make_task_data(kind="target_box")
    trace = make_trace(data, episode={"validity": "infra_invalid", "infra_error": {"stage": "x"}})
    with pytest.raises(Exception, match="infrastructure-invalid"):
        asyncio.run(TargetBoxTask(data).score(trace, _Runtime()))


def test_a_missing_target_box_result_raises() -> None:
    data = make_task_data(kind="target_box")
    with pytest.raises(Exception, match="published no result"):
        asyncio.run(TargetBoxTask(data).score(make_trace(data), _Runtime()))


def test_the_target_box_verdict_needs_a_runtime_but_the_shaping_does_not() -> None:
    from verifiers.v1.task import _requires_runtime

    assert _requires_runtime(TargetBoxTask.target_box)
    assert not _requires_runtime(TargetBoxTask.shaped_progress)


def test_the_shaping_constants_are_module_level_never_config_fields() -> None:
    """Reading a custom field off the per-task `TaskConfig` raises `AttributeError`,
    and one throwing reward inside `Task.score`'s gather drops the whole group."""
    for config_type in (TargetBoxTasksetConfig, GroundingTasksetConfig):
        fields = set(config_type.model_fields)
        assert not fields & {"shaping_weight", "shaping_scale", "no_move_penalty"}, config_type
