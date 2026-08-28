"""`VirtualDesktop`, the band shuffle, and md5-seeded regimes.

`VirtualDesktop` gives a canvas the same session surface a real desktop has, so
movebox/grounding run under the shared driver. It applies absolute pixel operations
only — there is deliberately no relative move, because every convention is resolved
inside the codec.

`band_sequence` decorrelates task index from difficulty: emitting band repeats in
dict-insertion order would make any prefix, shard or `max_tasks` cut a biased
curriculum sample.

Cursor regimes are md5-seeded, not `hash()`-seeded, because `hash()` is
PYTHONHASHSEED-randomised and would differ between processes. That is asserted
against two real subprocesses with different `PYTHONHASHSEED`.
"""

from __future__ import annotations

import collections
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evals.tasks import cursor_start, distance_to_box, in_bbox
from rl.grounding.dataset import REGIMES
from rl.desktop import VirtualDesktop, VirtualDesktopPool, canvas_pool
from rl.geometry import box_center, draw_box, png_bytes, render_cursor
from rl.movebox.dataset import CURRICULUM_BANDS, band_sequence, sample_scene

_REPO = Path(__file__).resolve().parents[1]
_PATH = os.pathsep.join([str(_REPO), str(_REPO.parent / "desktop")])


def _op(kind: str, *args):
    from desktop.ir import Operation

    return Operation(kind=kind, args=tuple(args))


def _canvas(width: int = 200, height: int = 150):
    from PIL import Image

    return Image.new("RGB", (width, height), (5, 5, 5))


def _desktop(**kwargs) -> VirtualDesktop:
    desktop = VirtualDesktop(**kwargs)
    desktop.configure(
        canvas=_canvas(*desktop.screen), cursor=(10, 10), screen=desktop.screen
    )
    return desktop


def test_the_virtual_desktop_has_the_session_surface_the_harness_needs() -> None:
    desktop = _desktop(screen=(200, 150))
    for attribute in (
        "screen_size",
        "cursor_position",
        "screenshot",
        "execute_atomic",
        "execute_pyautogui",
        "release",
    ):
        assert callable(getattr(desktop, attribute)), attribute
    assert desktop.screen_size() == (200, 150)
    assert desktop.cursor_position() == (10, 10)
    assert desktop.screenshot().startswith(b"\xff\xd8\xff")


def test_configure_resets_every_per_episode_accumulator() -> None:
    desktop = _desktop(screen=(200, 150))
    desktop.execute_atomic([_op("mouse_down", "left"), _op("coalesced_type", "x"), _op("scroll", 0, 3)])
    assert desktop.buttons and desktop.typed and desktop.scrolled and desktop.dispatched
    desktop.configure(canvas=_canvas(200, 150), cursor=(1, 2), screen=(200, 150))
    assert desktop.buttons == set() and desktop.typed == [] and desktop.keys == []
    assert desktop.scrolled == 0 and desktop.dispatched == 0
    assert desktop.cursor == (1, 2)


def test_a_move_operation_is_absolute_and_clamped_to_the_screen() -> None:
    desktop = _desktop(screen=(200, 150))
    desktop.execute_atomic([_op("move_to", 50, 60)])
    assert desktop.cursor == (50, 60)
    desktop.execute_atomic([_op("move_to", 10_000, -5)])
    assert desktop.cursor == (199, 0), "clamped to [0, dim-1]"


def test_glide_to_is_honoured_as_an_absolute_move() -> None:
    desktop = _desktop(screen=(200, 150))
    desktop.execute_atomic([_op("glide_to", 30, 40, 0.2)])
    assert desktop.cursor == (30, 40)


def test_there_is_no_relative_move_in_the_vocabulary() -> None:
    """A desktop that accepted a relative op would resolve a convention the codec
    already resolved."""
    desktop = _desktop(screen=(200, 150))
    before = desktop.cursor
    desktop.execute_atomic([_op("move_rel", 5, 5), _op("move_by", 5, 5)])
    assert desktop.cursor == before, "an unknown kind is skipped, never guessed at"


def test_button_press_and_release_are_tracked() -> None:
    desktop = _desktop()
    desktop.execute_atomic([_op("mouse_down", "left")])
    assert desktop.buttons == {"left"}
    desktop.execute_atomic([_op("mouse_up", "left")])
    assert desktop.buttons == set()
    desktop.execute_atomic([_op("mouse_down")])
    assert desktop.buttons == {"left"}, "a button-less press defaults to left"


def test_typing_scrolling_and_keys_are_recorded() -> None:
    desktop = _desktop()
    desktop.execute_atomic(
        [
            _op("coalesced_type", "hello"),
            _op("key_down", "Return"),
            _op("key_up", "Return"),
            _op("scroll", 0, 3),
            _op("hscroll", -2),
            _op("wait", 0.1),
        ]
    )
    assert desktop.typed == ["hello"]
    assert desktop.keys == ["Return"], "key_up is intentionally not recorded"
    assert desktop.scrolled == 1, "scroll dy=3 then hscroll dx=-2"


def test_an_unknown_operation_is_skipped_not_raised() -> None:
    """A grammar may legitimately emit something a *canvas* cannot honour."""
    desktop = _desktop()
    result = desktop.execute_atomic([_op("set_window_geometry", 1, 2, 3, 4)])
    assert result.operations == ("set_window_geometry",)
    assert desktop.dispatched == 1


def test_execute_atomic_reports_the_cursor_before_and_after() -> None:
    desktop = _desktop(screen=(200, 150))
    receipt = desktop.execute_atomic([_op("move_to", 99, 88)])
    # `evals.harness.Receipt`: the harness publishes `ok`, `failure_kind` and the
    # cursor pair beside its own round-trip read. A canvas applies what it accepts
    # and raises on the rest, so it never reports a failure.
    assert (receipt.cursor_before, receipt.cursor_after) == ((10, 10), (99, 88))
    assert receipt.operations == ("move_to",)
    assert receipt.ok is True and receipt.failure_kind is None


def test_execute_atomic_accepts_dict_operations_too() -> None:
    desktop = _desktop(screen=(200, 150))
    desktop.execute_atomic([{"kind": "move_to", "args": (7, 8)}])
    assert desktop.cursor == (7, 8)


def test_execute_pyautogui_honours_move_to_and_refuses_the_rest() -> None:
    desktop = _desktop(screen=(200, 150))
    desktop.execute_pyautogui("pyautogui.moveTo(33, 44)")
    assert desktop.cursor == (33, 44)
    with pytest.raises(ValueError, match="only honour moveTo"):
        desktop.execute_pyautogui("pyautogui.click()")
    desktop.execute_pyautogui("pyautogui.moveTo(-5, 9999)")
    assert desktop.cursor == (0, 149), "clamped like every other move"


def test_the_screenshot_moves_with_the_cursor() -> None:
    desktop = _desktop(screen=(200, 150))
    first = desktop.screenshot()
    desktop.execute_atomic([_op("move_to", 150, 100)])
    assert desktop.screenshot() != first, "the marker is composited per frame"


def test_release_is_a_no_op_so_the_lease_machinery_works_unchanged() -> None:
    _desktop().release(failed=True, error="x")


def test_the_canvas_pool_has_the_checkout_surface_the_lease_expects() -> None:
    pool = canvas_pool(VirtualDesktop)()
    assert isinstance(pool, VirtualDesktopPool)
    pool.start()
    first, second = pool.checkout(), pool.checkout()
    assert first is not second, "one desktop per rollout, never shared"
    pool.close()


def test_a_codec_compiled_action_drives_the_virtual_desktop_end_to_end() -> None:
    """The same compile path as a real desktop, with no env-side arithmetic."""
    from agent.agent import load_codec
    from desktop.geometry import DisplayGeometry

    desktop = _desktop(screen=(1920, 1080))
    desktop.configure(canvas=_canvas(1920, 1080), cursor=(100, 100), screen=(1920, 1080))
    codec = load_codec("move_rel")
    text = (
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": {"action": "move_rel", "coordinate": [100, 0]}})
        + "\n</tool_call>"
    )
    geometry = DisplayGeometry(desktop_width=1920, desktop_height=1080)
    operations = codec.compile(text, geometry, desktop.cursor_position())
    desktop.execute_atomic(operations)
    assert desktop.cursor != (100, 100), "the codec resolved 0-999 to pixels, not the env"
    assert desktop.cursor[1] == 100, "a pure-x delta must not move y"


def test_in_bbox_is_half_open_and_the_two_definitions_agree() -> None:
    from rl.geometry import in_bbox as rl_in_bbox

    box = (10, 10, 20, 20)
    for point in [(10, 10), (19, 19), (15, 15)]:
        assert in_bbox(point, box) and rl_in_bbox(point, box), point
    for point in [(20, 15), (15, 20), (9, 15), (20, 20)]:
        assert not in_bbox(point, box) and not rl_in_bbox(point, box), point


def test_distance_to_box_agrees_across_both_modules_and_is_zero_inside() -> None:
    from rl.geometry import distance_to_box as rl_distance

    box = (10, 10, 20, 20)
    for point in [(15, 15), (0, 0), (30, 30), (15, 0), (25, 15)]:
        assert distance_to_box(point, box) == rl_distance(point, box), point
    assert distance_to_box((15, 15), box) == 0.0
    assert distance_to_box((10, 0), box) == 10.0
    assert distance_to_box((30, 30), box) == pytest.approx(math.hypot(10, 10))


def test_box_center_and_the_renderers_do_not_mutate_their_input() -> None:
    assert box_center((10, 10, 20, 20)) == (15, 15)
    base = _canvas(60, 40)
    original = base.tobytes()
    boxed = draw_box(base, (5, 5, 20, 20))
    marked = render_cursor(base, (30, 20))
    assert base.tobytes() == original, "both draw on a copy"
    assert boxed.tobytes() != original and marked.tobytes() != original
    assert png_bytes(marked)[:8] == b"\x89PNG\r\n\x1a\n"


WEIGHTS = {"near": 0.6, "medium": 0.3, "far": 0.1}


def test_band_sequence_has_the_requested_length_and_the_requested_mix() -> None:
    bands = band_sequence(WEIGHTS, 100, seed=0)
    assert len(bands) == 100
    counts = collections.Counter(bands)
    assert counts["near"] == 60 and counts["medium"] == 30 and counts["far"] == 10


def test_band_sequence_decorrelates_index_from_difficulty() -> None:
    """Emitting every easy task first would make any prefix, shard or `max_tasks`
    cut a biased curriculum sample."""
    n = 600
    bands = band_sequence(WEIGHTS, n, seed=0)
    order = {"near": 0, "medium": 1, "far": 2}
    difficulty = [order[b] for b in bands]
    indices = list(range(n))
    mean_i = sum(indices) / n
    mean_d = sum(difficulty) / n
    cov = sum((i - mean_i) * (d - mean_d) for i, d in zip(indices, difficulty))
    var_i = sum((i - mean_i) ** 2 for i in indices)
    var_d = sum((d - mean_d) ** 2 for d in difficulty)
    correlation = cov / math.sqrt(var_i * var_d)
    assert abs(correlation) < 0.1, f"index/difficulty correlation is {correlation:.3f}"
    # And the unshuffled construction really would be maximally correlated.
    unshuffled = [b for name, w in WEIGHTS.items() for b in [name] * int(round(w * n))]
    d2 = [order[b] for b in unshuffled]
    mean2 = sum(d2) / len(d2)
    cov2 = sum((i - mean_i) * (d - mean2) for i, d in enumerate(d2))
    var2 = sum((d - mean2) ** 2 for d in d2)
    assert cov2 / math.sqrt(var_i * var2) > 0.8, "insertion order IS strongly correlated"


def test_every_prefix_of_the_shuffled_sequence_is_a_representative_sample() -> None:
    bands = band_sequence(WEIGHTS, 1000, seed=3)
    for cut in (100, 250, 500):
        counts = collections.Counter(bands[:cut])
        assert 0.45 <= counts["near"] / cut <= 0.75, (cut, counts)
        assert counts["far"] / cut <= 0.25, (cut, counts)


def test_band_sequence_is_seeded_and_reproducible() -> None:
    assert band_sequence(WEIGHTS, 64, seed=7) == band_sequence(WEIGHTS, 64, seed=7)
    assert band_sequence(WEIGHTS, 64, seed=7) != band_sequence(WEIGHTS, 64, seed=8)


def test_band_sequence_pads_and_truncates_to_n_tasks() -> None:
    assert len(band_sequence({"near": 0.1}, 10, seed=0)) == 10
    assert len(band_sequence({"near": 2.0, "far": 2.0}, 5, seed=0)) == 5
    assert set(band_sequence({"uniform": 1.0}, 8, seed=0)) == {"uniform"}


def test_every_band_name_is_a_declared_curriculum_band() -> None:
    for band in band_sequence(WEIGHTS, 50, seed=1):
        assert band in CURRICULUM_BANDS


def test_band_sequence_is_reproducible_across_processes() -> None:
    """`random.Random(str)` seeds from a sha512 of the bytes, so unlike `hash()` it
    does not move with PYTHONHASHSEED."""
    script = textwrap.dedent(
        """
        import json, sys
        from rl.movebox.dataset import band_sequence
        print(json.dumps(band_sequence({"near": 0.6, "medium": 0.3, "far": 0.1}, 40, seed=5)))
        """
    )
    outputs = []
    for seed_env in ("0", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": _PATH, "PYTHONHASHSEED": seed_env},
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1]
    assert outputs[0] == band_sequence(WEIGHTS, 40, seed=5)


def test_movebox_scene_sampling_is_a_pure_function_of_its_key() -> None:
    """Every pool worker runs `Taskset.load()` independently."""
    backgrounds = ["/a.png", "/b.png", "/c.png"]
    first = sample_scene(7, backgrounds, band="near", seed=2)
    again = sample_scene(7, backgrounds, band="near", seed=2)
    assert first == again
    assert sample_scene(8, backgrounds, band="near", seed=2) != first
    assert sample_scene(7, backgrounds, band="far", seed=2) != first
    assert sample_scene(7, backgrounds, band="near", seed=3) != first


def test_a_sampled_scene_starts_outside_its_box_and_inside_the_screen() -> None:
    backgrounds = ["/a.png"]
    for idx in range(60):
        for band in CURRICULUM_BANDS:
            scene = sample_scene(idx, backgrounds, band=band, seed=1)
            assert not in_bbox(scene.cursor_start, scene.box), (idx, band)
            assert 0 <= scene.cursor_start[0] < scene.screen_w
            assert 0 <= scene.cursor_start[1] < scene.screen_h
            assert scene.start_distance > 0


def test_a_band_limit_bounds_the_start_distance() -> None:
    backgrounds = ["/a.png"]
    for idx in range(40):
        near = sample_scene(idx, backgrounds, band="near", seed=4)
        far = sample_scene(idx, backgrounds, band="far", seed=4)
        assert near.start_distance <= CURRICULUM_BANDS["near"] + 2, near.start_distance
        assert far.start_distance <= CURRICULUM_BANDS["far"] + 2, far.start_distance


def test_movebox_scene_sampling_is_reproducible_across_processes() -> None:
    script = textwrap.dedent(
        """
        import json
        from rl.movebox.dataset import sample_scene
        s = sample_scene(11, ["/a.png", "/b.png"], band="medium", seed=9)
        print(json.dumps([s.background_path, list(s.box), list(s.cursor_start)]))
        """
    )
    outputs = []
    for seed_env in ("0", "999"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": _PATH, "PYTHONHASHSEED": seed_env},
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1]


def test_the_three_regimes_are_declared() -> None:
    assert REGIMES == ("near", "medium", "far")


def test_cursor_start_is_deterministic_in_process() -> None:
    box = (900, 500, 1000, 600)
    for regime in REGIMES:
        first = cursor_start(box, 1920, 1080, regime, "app/task")
        assert first == cursor_start(box, 1920, 1080, regime, "app/task")


def test_cursor_start_varies_with_the_key_and_the_regime() -> None:
    box = (900, 500, 1000, 600)
    a = cursor_start(box, 1920, 1080, "near", "app/task_a")
    b = cursor_start(box, 1920, 1080, "near", "app/task_b")
    assert a != b, "two targets must not share a start"
    assert cursor_start(box, 1920, 1080, "medium", "app/task_a") != a


def test_the_far_regime_is_the_screen_mirror_when_the_mirror_is_admissible() -> None:
    """The mirror is still `far`'s primary, and it is key-independent."""
    box = (100, 100, 200, 200)
    cx, cy = (100 + 200) // 2, (100 + 200) // 2
    mirror = (1920 - cx, 1080 - cy)
    assert not in_bbox(mirror, box), "precondition: this mirror is outside the target"
    for key in ("a", "b", "c"):
        assert cursor_start(box, 1920, 1080, "far", key) == mirror


BOXES = [
    (900, 500, 1000, 600),
    (0, 0, 1920, 1080),  # a full-window target: no on-screen start is outside it
    (0, 0, 40, 40),  # hard against the top-left edge
    (1880, 1040, 1920, 1080),  # hard against the bottom-right edge
]
UNSATISFIABLE = (0, 0, 1920, 1080)


@pytest.mark.parametrize("regime", ["near", "medium"])
def test_a_near_or_medium_start_is_outside_the_box_and_on_screen(regime: str) -> None:
    for box in BOXES:
        for key in ("a", "b", "c", "d", "e"):
            if box == UNSATISFIABLE:
                with pytest.raises(ValueError, match="no on-screen cursor start"):
                    cursor_start(box, 1920, 1080, regime, key)
                continue
            start = cursor_start(box, 1920, 1080, regime, key)
            assert 0 <= start[0] < 1920 and 0 <= start[1] < 1080, (regime, box, start)
            assert not in_bbox(start, box), (regime, box, key, start)


def test_every_regime_start_is_on_screen() -> None:
    for regime in REGIMES:
        for box in BOXES:
            for key in ("a", "b", "c"):
                if box == UNSATISFIABLE:
                    with pytest.raises(ValueError, match="no on-screen cursor start"):
                        cursor_start(box, 1920, 1080, regime, key)
                    continue
                start = cursor_start(box, 1920, 1080, regime, key)
                assert 0 <= start[0] < 1920 and 0 <= start[1] < 1080, (regime, box, start)


def test_the_far_regime_starts_OUTSIDE_the_box() -> None:
    """Every regime, `far` included, must start outside the box.

    `far` tries the screen mirror `(sw-cx, sh-cy)` first and falls through to the
    same eight-angle ladder as `near`/`medium` when the mirror is inside the box.
    Without that fallthrough, a target whose centre sits near the screen centre
    starts already solved — `in_bbox` true at step 0, `reach_frame` 1, reward 1.0
    before the model acts — and grounding runs with `require_unsolved_start=False`
    (`rl/grounding/harness.py:77`, because refusing edge cells would silently drop
    the hardest targets), so such an episode scores.

    Incidence over random targets at 1920x1080:
        20-60 px elements     0.02%
        60-200 px widgets     0.11%
        200-600 px panels     2.10%
        600-1400 px windows  42.90%
    """
    centred = (1920 // 2 - 100, 1080 // 2 - 100, 1920 // 2 + 100, 1080 // 2 + 100)
    assert in_bbox((960, 540), centred), "precondition: the bare mirror WOULD land inside"
    for regime in REGIMES:
        start = cursor_start(centred, 1920, 1080, regime, "any-key")
        assert not in_bbox(start, centred), (regime, start)
    assert cursor_start(centred, 1920, 1080, "far", "any-key") != (960, 540)


def test_no_regime_can_return_an_in_box_start_at_any_target_size() -> None:
    """Not one regime — every regime starts outside the box.

    Sweeps the four size bands whose old far-mirror degeneracy rates were 0.02% /
    0.11% / 2.10% / 42.90%. Any in-box start here is a silently-scored reward-1.0
    episode, so this asserts zero, not "negligible".
    """
    import random

    rng = random.Random(0)
    for lo, hi in ((20, 60), (60, 200), (200, 600), (600, 1400)):
        for i in range(1500):
            w = rng.randint(lo, min(hi, 1920))
            h = rng.randint(lo, min(hi, 1080))
            x = rng.randint(0, 1920 - w)
            y = rng.randint(0, 1080 - h)
            box = (x, y, x + w, y + h)
            for regime in REGIMES:
                start = cursor_start(box, 1920, 1080, regime, f"k{i}")
                assert not in_bbox(start, box), (regime, box, start)
                assert 0 <= start[0] < 1920 and 0 <= start[1] < 1080, (regime, box, start)


def test_a_bbox_with_no_admissible_start_raises_instead_of_scoring_a_free_reach() -> None:
    """A full-screen target admits no on-screen start outside itself.

    Returning anything at all would be a reward-1.0 episode the model never played,
    and `GroundingTaskset.load` is the caller, so the raise fails enumeration. Same
    ladder-then-raise contract as
    `rl.target_box.geometry.sample_cursor_start`.
    """
    for regime in REGIMES:
        with pytest.raises(ValueError, match="no on-screen cursor start"):
            cursor_start((0, 0, 1920, 1080), 1920, 1080, regime, "k")


def test_the_minimum_radius_rises_with_the_bbox_half_diagonal() -> None:
    """So a full-window target cannot collect a degenerate reach-at-step-0."""
    big = (400, 200, 1400, 900)
    for key in ("a", "b", "c"):
        start = cursor_start(big, 1920, 1080, "near", key)
        centre = box_center(big)
        span = math.hypot(big[2] - big[0], big[3] - big[1])
        assert math.dist(start, centre) >= min(200, int(span / 2) + 30) * 0.5


def test_the_near_regime_is_closer_than_the_medium_regime_on_average() -> None:
    box = (900, 500, 1000, 600)
    centre = box_center(box)
    near = [math.dist(cursor_start(box, 1920, 1080, "near", f"k{i}"), centre) for i in range(30)]
    medium = [math.dist(cursor_start(box, 1920, 1080, "medium", f"k{i}"), centre) for i in range(30)]
    assert sum(near) / 30 < sum(medium) / 30, "the regimes must actually stratify distance"


def test_cursor_regimes_are_reproducible_across_two_pythonhashseeds() -> None:
    """`hash()` is PYTHONHASHSEED-randomised and would differ between processes;
    md5 does not."""
    script = textwrap.dedent(
        """
        import json
        from evals.tasks import cursor_start
        box = (900, 500, 1000, 600)
        print(json.dumps({
            r: list(cursor_start(box, 1920, 1080, r, "app/task/one"))
            for r in ("near", "medium", "far")
        }))
        """
    )
    outputs = []
    for seed_env in ("0", "1", "424242"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": _PATH, "PYTHONHASHSEED": seed_env},
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1] == outputs[2], outputs
    in_process = {
        r: list(cursor_start((900, 500, 1000, 600), 1920, 1080, r, "app/task/one"))
        for r in REGIMES
    }
    assert outputs[0] == in_process


def test_a_builtin_hash_would_have_failed_the_same_check() -> None:
    """Demonstrates the defect being fixed, so the test above is known to bite."""
    script = "print(hash('app/task/one:near:v0'))"
    values = set()
    for seed_env in ("0", "1", "424242"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed_env},
            timeout=60,
        )
        values.add(proc.stdout.strip())
    assert len(values) > 1, "PYTHONHASHSEED must actually move hash() on this interpreter"


def test_the_band_metric_reads_the_band_the_preparer_published() -> None:
    """`band` is the one dimension this env's results must be sliced by, since the
    curriculum exists to compare near/medium/far. A metric reading a key nothing
    publishes would report it as a constant.
    """
    import asyncio

    from rl.movebox.taskset import BAND_ORDER, MoveBoxTask
    from juergen_doubles import make_task_data, make_trace

    for index, band in enumerate(BAND_ORDER):
        data = make_task_data(kind="movebox", bbox=(10, 10, 50, 50))
        trace = make_trace(
            data,
            episode={
                "validity": "valid",
                "reach_frame": 1,
                "best_distance": 0.0,
                "steps_detail": [],
                "setup": {"band": band},
            },
        )
        asyncio.run(MoveBoxTask(data).score(trace))
        assert trace.metrics["band"] == float(index), band
        assert trace.rewards["reach"] == 1.0


def test_an_unknown_or_absent_band_still_reads_as_the_unset_sentinel() -> None:
    import asyncio

    from rl.movebox.taskset import MoveBoxTask
    from juergen_doubles import make_task_data, make_trace

    for setup in ({}, {"band": "nonsense"}):
        data = make_task_data(kind="movebox", bbox=(10, 10, 50, 50))
        trace = make_trace(
            data,
            episode={
                "validity": "valid",
                "reach_frame": -1,
                "steps_detail": [],
                "setup": setup,
            },
        )
        asyncio.run(MoveBoxTask(data).score(trace))
        assert trace.metrics["band"] == -1.0


def test_the_band_order_is_pinned_so_a_dict_reorder_cannot_renumber_it() -> None:
    from rl.movebox.taskset import BAND_ORDER

    assert BAND_ORDER == ("near", "medium", "far", "uniform")
    assert set(BAND_ORDER) == set(CURRICULUM_BANDS)


def test_band_sequence_is_exported() -> None:
    import rl.movebox.dataset as dataset

    assert "band_sequence" in dataset.__all__


def test_the_movebox_reward_raises_on_a_missing_result() -> None:
    import asyncio

    from rl.movebox.taskset import MoveBoxTask
    from juergen_doubles import make_task_data, make_trace

    data = make_task_data(kind="movebox")
    with pytest.raises(Exception, match="published no result"):
        asyncio.run(MoveBoxTask(data).score(make_trace(data)))


def test_an_infra_invalid_movebox_rollout_raises_instead_of_scoring_a_zero() -> None:
    """`reach_frame` stays -1 when the VM never booted, so scoring it would train an
    infrastructure failure as a genuine miss."""
    import asyncio

    from rl.movebox.taskset import MoveBoxTask
    from juergen_doubles import make_task_data, make_trace

    data = make_task_data(kind="movebox", bbox=(10, 10, 50, 50))
    trace = make_trace(
        data,
        episode={
            "validity": "infra_invalid",
            "infra_error": {"stage": "prepare"},
            "reach_frame": -1,
            "steps_detail": [],
            "setup": {},
        },
    )
    with pytest.raises(Exception, match="infrastructure-invalid"):
        asyncio.run(MoveBoxTask(data).score(trace))
    assert trace.rewards == {}, "one throwing reward must drop the whole group"


@pytest.mark.parametrize(
    "family,module,cls,key",
    [
        ("movebox", "rl.movebox.taskset", "MoveBoxTask", "reach_frame"),
        ("grounding", "rl.grounding.taskset", "GroundingTask", "reach_frame"),
        ("target_box", "rl.target_box.taskset", "TargetBoxTask", "best_distance"),
    ],
)
def test_a_reward_raises_when_the_episode_omits_the_key_it_scores(
    family, module, cls, key
) -> None:
    """`_publish` writes both keys in one dict literal, so a rename is the only way
    to lose them -- and the sentinel default made that rename score every rollout in
    every group 0.0 instead of saying anything."""
    import asyncio
    import importlib

    from juergen_doubles import make_task_data, make_trace

    task_cls = getattr(importlib.import_module(module), cls)
    data = make_task_data(kind=family, bbox=(10, 10, 50, 50))
    episode = {
        "validity": "valid",
        "outcome": "max_steps",
        "reach_frame": -1,
        "best_distance": 12.0,
        "steps_detail": [],
        "setup": {},
    }
    del episode[key]
    trace = make_trace(data, episode=episode)

    with pytest.raises(Exception, match=f"KeyError: '{key}'"):
        asyncio.run(task_cls(data).score(trace))
    assert trace.rewards == {}, "one throwing reward must drop the whole group"


def test_the_movebox_reward_is_sparse_task_success_only() -> None:
    """No STEP_PENALTY / REPEAT_PENALTY / dense shaping: one named knob."""
    import asyncio

    from rl.movebox.taskset import REACH_REWARD, MoveBoxTask
    from juergen_doubles import make_task_data, make_trace

    assert REACH_REWARD == 1.0
    data = make_task_data(kind="movebox", bbox=(10, 10, 50, 50))
    trace = make_trace(
        data,
        episode={"validity": "valid", "reach_frame": -1, "steps_detail": [], "setup": {}},
    )
    asyncio.run(MoveBoxTask(data).score(trace))
    assert trace.rewards == {"reach": 0.0}, trace.rewards
