from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shortgoal_fixture as fixture  # noqa: E402
import shortgoal_golden as golden  # noqa: E402
import shortgoal_templates as templates  # noqa: E402

_EXPECTED_CATEGORY_COUNTS = {"terminal": 10, "editor": 4, "fixture": 8, "browser": 3}
_EXPECTED_TIER_B = (
    "gedit_select_all_delete", "fx_right_click", "fx_scroll_find_click", "web_scroll_click",
)
_EXPECTED_SPLIT_COUNTS = {"train": 105, "tier_a": 21, "tier_b": 24}
_SINGLE_ACTION_BAND = (0.35, 0.5)
_TERMINAL_LAST_KINDS = ("up", "type", "scroll", "no_op")
_START_CURSOR = (960, 540)


def _ctx() -> golden.GoldenCtx:
    return golden.GoldenCtx(cursor_xy=_START_CURSOR, screen_wh=templates.SCREEN_WH)


def _steps(task: templates.ConcreteTask) -> list[golden.GoldenStep]:
    return golden.golden_steps(task, _ctx())


def _split_of(manifest: dict, task_id: str) -> str:
    hits = [name for name in templates.SPLIT_NAMES if task_id in manifest[name]]
    if len(hits) != 1:
        raise AssertionError(f"{task_id} appears in {hits!r}")
    return hits[0]


def _replay_fixture(task: templates.ConcreteTask) -> dict:
    spec = task.params["fixture_spec"]
    state = fixture.initial_state(spec)
    cursor = list(_START_CURSOR)
    for step in _steps(task):
        held: str | None = None
        moved = False
        pairs = 0
        for prim in step:
            kind = prim["kind"]
            if kind == "move":
                cursor = list(prim["to_xy"])
                moved = held is not None
            elif kind == "scroll":
                state = fixture.apply_scroll(spec, state, prim["notches"], pointer_xy=cursor)
            elif kind == "down":
                held = prim["name"]
            elif kind == "up":
                if prim["name"] == golden.COMMIT_KEY:
                    state = fixture.apply_commit(spec, state)
                elif spec["kind"] == "slider" and moved:
                    state = fixture.apply_drag(spec, state, cursor[0])
                else:
                    pairs += 1
                    state = fixture.apply_click(
                        spec, state, cursor,
                        button="right" if prim["name"] == "RMB" else "left",
                        count=pairs,
                    )
                held, moved = None, False
    return state


class _StubWidget:
    """A Tk widget stand-in that records what the fixture configures on it."""

    def __init__(self, bbox: tuple[int, int, int, int] = (0, 0, 1, 1)) -> None:
        self.bbox = tuple(bbox)
        self.text = ""
        self.relief = "raised"

    def configure(self, **kwargs: object) -> None:
        if "text" in kwargs:
            self.text = str(kwargs["text"])
        if "relief" in kwargs:
            self.relief = str(kwargs["relief"])

    def winfo_rootx(self) -> int:
        return self.bbox[0]

    def winfo_rooty(self) -> int:
        return self.bbox[1]

    def winfo_width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    def winfo_height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class _StubCanvas:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.items = 0

    def delete(self, tag: str) -> None:
        self.deleted.append(tag)

    def _item(self, *args: object, **kwargs: object) -> int:
        self.items += 1
        return self.items

    create_rectangle = _item
    create_line = _item
    create_text = _item


class _StubRoot(_StubWidget):
    def __init__(self, screen: list[int], pointer: object) -> None:
        super().__init__((0, 0, int(screen[0]), int(screen[1])))
        self.pointer = pointer
        self.idle_flushes = 0
        self.timers: list[tuple[int, object]] = []

    def winfo_pointerxy(self) -> object:
        return self.pointer

    def update_idletasks(self) -> None:
        self.idle_flushes += 1

    def after(self, delay_ms: int, callback: object) -> None:
        self.timers.append((delay_ms, callback))


class _StubEvent:
    def __init__(self, x_root: int, y_root: int = 0) -> None:
        self.x_root = x_root
        self.y_root = y_root


class _StubFixture:
    """The real ``Fixture`` handlers over a stubbed Tk layer.

    Headless, so the invariant that broke the scroll pad — the visible label has
    to agree with the published state after EVERY notch, while the state file may
    coalesce — is unit-testable without a display."""

    def __init__(self, spec: dict, *, pointer: object = None) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self._dir.name) / "shortgoal_state.json"
        self.fixture = fixture.Fixture.__new__(fixture.Fixture)
        self.fixture.spec = fixture.validate_spec(spec)
        self.fixture.state = fixture.initial_state(spec)
        self.fixture.state_path = self.state_path
        self.fixture.widgets = {}
        self.fixture.last_click = None
        self.fixture.last_click_at = 0.0
        self.fixture.write_scheduled = False
        pane = spec.get("scroll", {}).get("pane")
        default = fixture.bbox_center(pane) if pane else [
            spec["screen"][0] // 2, spec["screen"][1] // 2,
        ]
        self.root = _StubRoot(spec["screen"], default if pointer is None else pointer)
        self.fixture.root = self.root
        self.counter = _StubWidget()
        self.fixture.counter = self.counter
        self.canvas = _StubCanvas()
        self.fixture.canvas = self.canvas
        self.fixture.canvas_origin = (0, 0)
        if spec["kind"] in ("buttons", "two_buttons", "colors"):
            for label, box in fixture.spec_widgets(spec).items():
                self.fixture.widgets[label] = _StubWidget(tuple(box))

    def close(self) -> None:
        self._dir.cleanup()

    @property
    def state(self) -> dict:
        return self.fixture.state

    @property
    def widgets(self) -> dict:
        return self.fixture.widgets

    @property
    def handles_drawn(self) -> int:
        return self.canvas.deleted.count("handle")

    def wheel(self, notches: int) -> None:
        fixture.Fixture._on_wheel(self.fixture, notches)

    def release(self, x_px: int) -> None:
        fixture.Fixture._on_slider_release(self.fixture, _StubEvent(x_px))

    def click(self, label: str, *, count: int = 1, button: str = "left") -> None:
        fixture.Fixture._on_click(self.fixture, label, count, button)

    def commit(self) -> None:
        fixture.Fixture._on_commit(self.fixture, _StubEvent(0))

    def run_timers(self) -> None:
        pending, self.root.timers = list(self.root.timers), []
        for _, callback in pending:
            callback()


class CatalogTests(unittest.TestCase):
    def test_catalog_shape(self) -> None:
        self.assertEqual(len(templates.TEMPLATES), 25)
        ids = [t.template_id for t in templates.TEMPLATES]
        self.assertEqual(len(set(ids)), 25)
        counts = dict.fromkeys(templates.CATEGORIES, 0)
        for template in templates.TEMPLATES:
            counts[template.category] += 1
            self.assertTrue(template.instruction_fmt.strip())
            self.assertTrue(template.param_space.strip())
            self.assertIn(template.policy_id, golden.POLICIES)
            self.assertTrue(template.verifier_id.strip())
        self.assertEqual(counts, _EXPECTED_CATEGORY_COUNTS)

    def test_tier_b_templates(self) -> None:
        self.assertEqual(templates.TIER_B_TEMPLATE_IDS, _EXPECTED_TIER_B)
        for template_id in _EXPECTED_TIER_B:
            self.assertTrue(templates.TEMPLATES_BY_ID[template_id].tier_b)

    def test_every_task_resolves(self) -> None:
        tasks = templates.concrete_tasks()
        self.assertEqual(len(tasks), 150)
        self.assertEqual(len({task.task_id for task in tasks}), 150)
        for task in tasks:
            self.assertTrue(task.instruction.endswith("."))
            self.assertNotIn("{", task.instruction)
            self.assertEqual(task.task_id, f"{task.template_id}__s{task.seed:02d}")


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = templates.build_split_manifest()

    def test_exact_counts(self) -> None:
        self.assertEqual(self.manifest["counts"], _EXPECTED_SPLIT_COUNTS)
        self.assertEqual(len(self.manifest["train"]), 105)
        self.assertEqual(len(self.manifest["tier_a"]), 21)
        self.assertEqual(len(self.manifest["tier_b"]), 24)
        self.assertEqual(sum(self.manifest["counts"].values()), 150)
        self.assertEqual(self.manifest["n_tasks"], 150)

    def test_disjoint_by_template_and_seed(self) -> None:
        keys: dict[tuple[str, int], str] = {}
        for name in templates.SPLIT_NAMES:
            for task_id in self.manifest[name]:
                template_id, _, seed = task_id.partition("__s")
                key = (template_id, int(seed))
                self.assertNotIn(key, keys, f"{key} in both {keys.get(key)} and {name}")
                keys[key] = name
        self.assertEqual(len(keys), 150)
        self.assertEqual(
            {key for key, name in keys.items() if name == "tier_b"},
            {(template_id, seed) for template_id in _EXPECTED_TIER_B for seed in range(6)},
        )
        for (template_id, seed), name in keys.items():
            if name == "train":
                self.assertLess(seed, 5)
                self.assertNotIn(template_id, _EXPECTED_TIER_B)
            elif name == "tier_a":
                self.assertEqual(seed, 5)
                self.assertNotIn(template_id, _EXPECTED_TIER_B)

    def test_manifest_metadata(self) -> None:
        self.assertEqual(self.manifest["n_seeds"], 6)
        self.assertEqual(self.manifest["tier_a_seed"], 5)
        self.assertEqual(self.manifest["tier_b_templates"], list(_EXPECTED_TIER_B))


class DrawParamsTests(unittest.TestCase):
    def test_determinism(self) -> None:
        for template in templates.TEMPLATES:
            for seed in range(templates.N_SEEDS):
                first = templates.draw_params(template.template_id, seed)
                second = templates.draw_params(template.template_id, seed)
                self.assertEqual(first, second, template.template_id)
                self.assertEqual(
                    json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True),
                )

    def test_seeds_differ(self) -> None:
        for template in templates.TEMPLATES:
            drawn = {
                json.dumps(templates.draw_params(template.template_id, seed), sort_keys=True)
                for seed in range(templates.N_SEEDS)
            }
            self.assertGreaterEqual(len(drawn), 4, template.template_id)

    def test_pinned_golden_values(self) -> None:
        self.assertEqual(
            templates.draw_params("term_touch", 0),
            {
                "filename": "report_89.cfg",
                "command": "touch report_89.cfg",
                "expect": {"path": "report_89.cfg", "exists": True},
            },
        )
        self.assertEqual(templates.draw_params("fx_click_button", 3)["target_xy"], [672, 187])
        self.assertEqual(
            templates.draw_params("term_special_typing", 2)["command"],
            "printf '%s\\n' 'a \"b\" and \\c' > report_23.csv",
        )

    def test_escape_stress_pool(self) -> None:
        self.assertTrue(any('"' in payload for payload in templates.ESCAPE_PAYLOADS))
        self.assertTrue(any("\\" in payload for payload in templates.ESCAPE_PAYLOADS))
        for payload in templates.ESCAPE_PAYLOADS:
            self.assertTrue('"' in payload or "\\" in payload)
            self.assertNotIn("'", payload)
        seen = {templates.draw_params("term_special_typing", seed)["payload"] for seed in range(6)}
        self.assertTrue(any('"' in payload for payload in seen))
        self.assertTrue(any("\\" in payload for payload in seen))

    def test_rejects_bad_keys(self) -> None:
        with self.assertRaises(KeyError):
            templates.draw_params("nope", 0)
        with self.assertRaises(ValueError):
            templates.draw_params("term_touch", -1)


class MixTests(unittest.TestCase):
    def test_single_action_share(self) -> None:
        tasks = templates.concrete_tasks()
        share = sum(task.single_action for task in tasks) / len(tasks)
        self.assertGreaterEqual(share, _SINGLE_ACTION_BAND[0])
        self.assertLessEqual(share, _SINGLE_ACTION_BAND[1])

    def test_single_action_matches_step_count(self) -> None:
        for task in templates.concrete_tasks():
            steps = _steps(task)
            self.assertEqual(task.single_action, len(steps) == 1, task.task_id)


class PolicyTests(unittest.TestCase):
    def test_step_bounds_and_terminal_compatibility(self) -> None:
        for task in templates.concrete_tasks():
            steps = _steps(task)
            self.assertGreaterEqual(len(steps), 1, task.task_id)
            self.assertLessEqual(len(steps), golden.MAX_STEPS, task.task_id)
            golden.validate_steps(steps)
            last = steps[-1]
            self.assertIn(last[-1]["kind"], _TERMINAL_LAST_KINDS, task.task_id)
            for step in steps:
                held: list[str] = []
                for prim in step:
                    if prim["kind"] == "down":
                        held.append(prim["name"])
                    elif prim["kind"] == "up":
                        held.remove(prim["name"])
                self.assertEqual(held, [], task.task_id)

    def test_every_policy_is_exercised(self) -> None:
        used = {template.policy_id for template in templates.TEMPLATES}
        self.assertEqual(used, set(golden.POLICIES))

    def test_policies_are_pure(self) -> None:
        for task in templates.concrete_tasks(2):
            self.assertEqual(_steps(task), _steps(task), task.task_id)

    def test_no_op_only_in_launch_editor(self) -> None:
        waits = 0
        for task in templates.concrete_tasks():
            for step in _steps(task):
                if step[0]["kind"] == "no_op":
                    self.assertEqual(len(step), 1)
                    self.assertEqual(task.template_id, "term_launch_editor")
                    waits += 1
        self.assertEqual(waits, sum(
            templates.draw_params("term_launch_editor", seed)["waits"] for seed in range(6)
        ))

    def test_targets_must_fit_the_context_screen(self) -> None:
        task = templates.concrete_task("fx_click_button", 0)
        tiny = golden.GoldenCtx(cursor_xy=_START_CURSOR, screen_wh=(320, 240))
        with self.assertRaises(ValueError):
            golden.golden_steps(task, tiny)
        with self.assertRaises(TypeError):
            golden.golden_steps(task, {"cursor_xy": _START_CURSOR})

    def test_step_builders_reject_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            golden.scroll(0)
        with self.assertRaises(ValueError):
            golden.type_text("")
        with self.assertRaises(ValueError):
            golden.click_step([10, 10], name="Return")
        with self.assertRaises(ValueError):
            golden.validate_step([golden.down("LMB")])
        with self.assertRaises(ValueError):
            golden.validate_step([golden.no_op(), golden.scroll(2)])

    def test_drag_is_one_step(self) -> None:
        task = templates.concrete_task("fx_drag_slider", 0)
        steps = _steps(task)
        self.assertEqual(
            [prim["kind"] for prim in steps[0]], ["move", "down", "move", "up"],
        )


class OverfitSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = templates.build_split_manifest()

    def test_overfit1(self) -> None:
        self.assertEqual(templates.OVERFIT1_TASK_ID, "term_touch__s00")
        self.assertEqual(_split_of(self.manifest, templates.OVERFIT1_TASK_ID), "train")

    def test_overfit32(self) -> None:
        ids = templates.OVERFIT32_TASK_IDS
        self.assertEqual(len(ids), 32)
        self.assertEqual(len(set(ids)), 32)
        categories = set()
        for task_id in ids:
            self.assertEqual(_split_of(self.manifest, task_id), "train")
            template_id, _, seed = task_id.partition("__s")
            self.assertLess(int(seed), 5)
            template = templates.TEMPLATES_BY_ID[template_id]
            self.assertFalse(template.tier_b)
            categories.add(template.category)
        self.assertEqual(categories, set(templates.CATEGORIES))
        self.assertIn(templates.OVERFIT1_TASK_ID, ids)


class HtmlPageTests(unittest.TestCase):
    def test_token_in_every_page(self) -> None:
        for template_id in ("web_click_link", "web_type_input", "web_scroll_click"):
            for seed in range(templates.N_SEEDS):
                params = templates.draw_params(template_id, seed)
                page = fixture.make_html_page(params["page_kind"], params)
                self.assertIn(params["token"], page, f"{template_id}:{seed}")
                self.assertIn("<title>shortgoal</title>", page)
                self.assertNotIn(f"<title>{params['token']}", page)
                self.assertIn("position:absolute", page)

    def test_link_grid_layout(self) -> None:
        params = templates.draw_params("web_click_link", 0)
        page = fixture.make_html_page("link_grid", params)
        for link in params["links"]:
            x0, y0, x1, y1 = fixture.widget_bbox(link["center"], link["size"])
            self.assertIn(f"left:{x0}px;top:{y0}px;width:{x1 - x0}px;height:{y1 - y0}px", page)
            self.assertIn(link["label"], page)
            self.assertIn(f'data-label="{link["label"]}"', page)
        self.assertIn(f"var TARGET={json.dumps(params['label'])};", page)
        self.assertEqual(params["expect"]["title"], params["token"])

    def test_no_page_wires_a_handler_through_an_html_attribute(self) -> None:
        for template_id in ("web_click_link", "web_type_input", "web_scroll_click"):
            for seed in range(templates.N_SEEDS):
                params = templates.draw_params(template_id, seed)
                page = fixture.make_html_page(params["page_kind"], params)
                markup = page.split("<script>")[0]
                for attribute in ("onclick", "oninput", "onkeydown", "onload"):
                    self.assertNotIn(attribute, markup, f"{template_id}:{seed}")
                self.assertIn("addEventListener", page)

    def test_input_page_mirrors_value_into_title(self) -> None:
        params = templates.draw_params("web_type_input", 1)
        page = fixture.make_html_page("input", params)
        self.assertIn("autofocus", page)
        self.assertIn('document.title=TOKEN+":"+field.value', page)
        self.assertEqual(params["expect"]["title"], f"{params['token']}:{params['text']}")

    def test_below_fold_button_page(self) -> None:
        for seed in range(templates.N_SEEDS):
            params = templates.draw_params("web_scroll_click", seed)
            page = fixture.make_html_page("below_fold_button", params)
            self.assertIn(f"#root{{height:{params['page_height']}px;}}", page)
            box = fixture.widget_bbox(params["button_page_xy"], params["button_size"])
            self.assertGreaterEqual(box[1], fixture.PAGE_WH[1])
            self.assertLess(params["notches"], 0)
            scrolled = params["button_page_xy"][1] - (params["page_height"] - fixture.PAGE_WH[1])
            self.assertEqual(params["target_xy"], [params["button_page_xy"][0], scrolled])
            self.assertIn('id="target"', page)
            self.assertIn(
                'document.getElementById("target").addEventListener("click",function(){'
                "document.title=TOKEN;});",
                page,
            )

    def test_rejects_unknown_page_kind(self) -> None:
        with self.assertRaises(ValueError):
            fixture.make_html_page("carousel", {"token": "0123456789"})

    def test_every_page_publishes_the_post_paint_ready_title(self) -> None:
        for template_id in ("web_click_link", "web_type_input", "web_scroll_click"):
            for seed in range(templates.N_SEEDS):
                params = templates.draw_params(template_id, seed)
                page = fixture.make_html_page(params["page_kind"], params)
                self.assertIn(f'document.title="{fixture.PAGE_READY_TITLE}"', page)
                self.assertIn(
                    'window.addEventListener("load",function(){requestAnimationFrame(', page,
                )
                self.assertNotIn(f"<title>{fixture.PAGE_READY_TITLE}", page)
                self.assertNotIn(
                    params["expect"]["title"], fixture.PAGE_READY_TITLE, f"{template_id}:{seed}",
                )


class FixtureSpecTests(unittest.TestCase):
    def test_spec_json_round_trip(self) -> None:
        for template_id in ("fx_click_button", "fx_drag_slider", "fx_scroll_find_click"):
            for seed in range(templates.N_SEEDS):
                spec = templates.draw_params(template_id, seed)["fixture_spec"]
                text = fixture.spec_to_json(spec)
                self.assertEqual(fixture.parse_spec(text), spec)
                self.assertEqual(fixture.spec_to_json(fixture.parse_spec(text)), text)
                self.assertEqual(spec["screen"], list(templates.SCREEN_WH))

    def test_targets_sit_inside_their_widgets(self) -> None:
        wanted = {
            "fx_click_button": (("label", "target_xy"),),
            "fx_click_color": (("color", "target_xy"),),
            "fx_double_click": (("label", "target_xy"),),
            "fx_right_click": (("label", "target_xy"),),
            "fx_scroll_find_click": (("row", "target_xy"),),
            "fx_two_buttons_order": (("first", "first_xy"), ("second", "second_xy")),
        }
        for template_id, pairs in wanted.items():
            for seed in range(templates.N_SEEDS):
                params = templates.draw_params(template_id, seed)
                boxes = fixture.spec_widgets(
                    params["fixture_spec"], offset_rows=params.get("scroll_offset", 0),
                )
                for label_key, point_key in pairs:
                    self.assertTrue(
                        fixture.bbox_contains(boxes[params[label_key]], params[point_key]),
                        f"{template_id}:{seed} {label_key}",
                    )
        for seed in range(templates.N_SEEDS):
            params = templates.draw_params("fx_drag_slider", seed)
            boxes = fixture.spec_widgets(params["fixture_spec"])
            self.assertTrue(fixture.bbox_contains(boxes["slider_handle"], params["handle_xy"]))
            dropped = fixture.spec_widgets(
                params["fixture_spec"], slider_value=params["target_value"],
            )
            self.assertTrue(fixture.bbox_contains(dropped["slider_handle"], params["target_xy"]))
            params = templates.draw_params("fx_scroll_counter", seed)
            boxes = fixture.spec_widgets(params["fixture_spec"])
            self.assertTrue(fixture.bbox_contains(boxes["scroll_pane"], params["pane_xy"]))

    def test_golden_replay_reaches_expected_state(self) -> None:
        for template in templates.TEMPLATES:
            if template.category != "fixture":
                continue
            for seed in range(templates.N_SEEDS):
                task = templates.concrete_task(template.template_id, seed)
                state = _replay_fixture(task)
                self.assertEqual(state["misses"], 0, task.task_id)
                for key, value in task.params["expect"].items():
                    self.assertEqual(state[key], value, f"{task.task_id}:{key}")

    def test_state_and_hit_test_logic(self) -> None:
        spec = templates.draw_params("fx_click_button", 0)["fixture_spec"]
        state = fixture.initial_state(spec)
        self.assertEqual(state["clicked"], [])
        self.assertFalse(state["committed"])
        self.assertEqual(sorted(state["widgets"]), sorted(t["label"] for t in spec["buttons"]))
        outside = fixture.apply_click(spec, state, [1919, 1079])
        self.assertEqual(outside["misses"], 1)
        self.assertEqual(outside["clicked"], [])
        self.assertIsNone(fixture.hit_test(spec, [1919, 1079]))

    def test_initial_state_publishes_the_window_and_keyboard_contract(self) -> None:
        for template_id in ("fx_click_button", "fx_drag_slider", "fx_scroll_find_click"):
            spec = templates.draw_params(template_id, 0)["fixture_spec"]
            state = fixture.initial_state(spec)
            self.assertEqual(state["window"], [0, 0, *templates.SCREEN_WH])
            self.assertFalse(state["keyboard"])
            self.assertEqual(state["keys_seen"], 0)
            committed = fixture.apply_commit(spec, state)
            self.assertTrue(committed["committed"])
            self.assertEqual(committed["keys_seen"], 0)
            self.assertEqual(committed["window"], state["window"])

    def _stub(self, spec: dict, **kwargs: object) -> _StubFixture:
        stub = _StubFixture(spec, **kwargs)
        self.addCleanup(stub.close)
        return stub

    def test_the_scroll_pad_label_never_trails_the_state(self) -> None:
        params = templates.draw_params("fx_scroll_counter", 0)
        spec = params["fixture_spec"]
        stub = self._stub(spec)
        for _ in range(abs(params["notches"])):
            stub.wheel(1 if params["notches"] > 0 else -1)
            self.assertEqual(stub.counter.text, fixture.counter_text(stub.state))
        self.assertEqual(stub.state["wheel_notches"], params["notches"])
        self.assertEqual(stub.counter.text, str(params["notches"]))
        self.assertEqual(stub.root.idle_flushes, abs(params["notches"]))
        self.assertEqual(len(stub.root.timers), 1)
        self.assertFalse(stub.state_path.exists())
        stub.run_timers()
        self.assertEqual(
            json.loads(stub.state_path.read_text())["wheel_notches"], params["notches"],
        )
        stub.wheel(1 if params["notches"] > 0 else -1)
        self.assertEqual(len(stub.root.timers), 1)
        stub.run_timers()
        self.assertEqual(
            json.loads(stub.state_path.read_text())["wheel_notches"],
            stub.state["wheel_notches"],
        )

    def test_a_scroll_outside_the_pad_repaints_nothing_but_still_counts_the_miss(self) -> None:
        spec = templates.draw_params("fx_scroll_counter", 0)["fixture_spec"]
        stub = self._stub(spec, pointer=(5, 5))
        stub.wheel(-3)
        self.assertEqual(stub.state["misses"], 1)
        self.assertEqual(stub.state["wheel_notches"], 0)
        self.assertEqual(stub.counter.text, "0")
        stub.run_timers()
        self.assertEqual(json.loads(stub.state_path.read_text())["misses"], 1)

    def test_the_slider_handle_and_click_feedback_repaint_before_publishing(self) -> None:
        params = templates.draw_params("fx_drag_slider", 0)
        stub = self._stub(params["fixture_spec"])
        stub.release(params["target_xy"][0])
        self.assertEqual(stub.state["slider_value"], params["target_value"])
        self.assertEqual(stub.handles_drawn, 1)
        self.assertGreaterEqual(stub.root.idle_flushes, 1)
        self.assertEqual(
            json.loads(stub.state_path.read_text())["slider_value"], params["target_value"],
        )
        buttons = templates.draw_params("fx_click_button", 0)
        tile = self._stub(buttons["fixture_spec"])
        tile.click(buttons["label"])
        self.assertEqual(tile.state["clicked"], [buttons["label"]])
        self.assertEqual(tile.widgets[buttons["label"]].relief, "sunken")
        self.assertEqual(
            json.loads(tile.state_path.read_text())["clicked"], [buttons["label"]],
        )

    def test_two_slow_left_clicks_on_one_tile_pair_into_a_double(self) -> None:
        spec = templates.draw_params("fx_double_click", 0)["fixture_spec"]
        label = templates.draw_params("fx_double_click", 0)["label"]
        other = next(t["label"] for t in spec["buttons"] if t["label"] != label)
        self.assertTrue(fixture.pairs_double_click(
            label, button="left", count=1, last_label=label, elapsed_s=0.7,
        ))
        for kwargs in (
            {"last_label": other, "elapsed_s": 0.7},
            {"last_label": None, "elapsed_s": 0.7},
            {"last_label": label, "elapsed_s": fixture.DOUBLE_CLICK_S + 0.1},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(
                    fixture.pairs_double_click(label, button="left", count=1, **kwargs),
                )
        self.assertFalse(fixture.pairs_double_click(
            label, button="right", count=1, last_label=label, elapsed_s=0.1,
        ))
        state = fixture.record_click(
            fixture.record_click(fixture.initial_state(spec), label), label, count=2,
        )
        self.assertEqual(state["clicked"], [label, label])
        self.assertEqual(state["double_clicked"], [label])
        for key, value in templates.draw_params("fx_double_click", 0)["expect"].items():
            if key != "committed":
                self.assertEqual(state[key], value)

    def test_scroll_list_clamps_to_bottom(self) -> None:
        params = templates.draw_params("fx_scroll_find_click", 0)
        spec = params["fixture_spec"]
        scroll = spec["scroll"]
        state = fixture.initial_state(spec)
        for notches in params["bursts"]:
            state = fixture.apply_scroll(spec, state, notches)
        self.assertEqual(state["scroll_offset"], fixture.max_scroll_offset(scroll))
        self.assertEqual(state["scroll_offset"], params["scroll_offset"])
        box = fixture.row_bbox(scroll, params["row_index"], state["scroll_offset"])
        self.assertIsNotNone(box)
        self.assertTrue(fixture.bbox_contains(box, params["target_xy"]))

    def test_scroll_outside_the_pane_is_a_miss(self) -> None:
        params = templates.draw_params("fx_scroll_counter", 0)
        spec = params["fixture_spec"]
        state = fixture.initial_state(spec)
        outside = fixture.apply_scroll(spec, state, params["notches"], pointer_xy=[5, 5])
        self.assertEqual(outside["wheel_notches"], 0)
        self.assertEqual(outside["misses"], 1)
        inside = fixture.apply_scroll(
            spec, state, params["notches"], pointer_xy=params["pane_xy"],
        )
        self.assertEqual(inside["wheel_notches"], params["notches"])
        self.assertEqual(inside["misses"], 0)

    def test_slider_grid_is_exact(self) -> None:
        params = templates.draw_params("fx_drag_slider", 2)
        slider = params["fixture_spec"]["slider"]
        for value in range(slider["ticks"] + 1):
            x = fixture.slider_tick_x(slider, value)
            self.assertEqual(fixture.slider_value_at(slider, x), value)
        self.assertEqual(
            fixture.slider_value_at(slider, params["target_xy"][0]), params["target_value"],
        )

    def test_state_writer_round_trip(self) -> None:
        self.assertEqual(fixture.STATE_PATH, "/tmp/shortgoal_state.json")
        spec = templates.draw_params("fx_two_buttons_order", 0)["fixture_spec"]
        state = fixture.apply_commit(spec, fixture.initial_state(spec))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "shortgoal_state.json"
            fixture.write_state(path, state)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), state)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), [path.name])

    def test_spec_validation_rejects_overlap_and_bad_kind(self) -> None:
        spec = templates.draw_params("fx_click_button", 0)["fixture_spec"]
        broken = json.loads(fixture.spec_to_json(spec))
        broken["buttons"][1]["center"] = list(broken["buttons"][0]["center"])
        with self.assertRaises(ValueError):
            fixture.validate_spec(broken)
        with self.assertRaises(ValueError):
            fixture.validate_spec({"kind": "carousel", "screen": [1920, 1080], "commit_key": "Return"})


if __name__ == "__main__":
    unittest.main()
