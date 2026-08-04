"""The 25-template short-goal catalog, seeded param draws and split manifest.

Every task in the diagnostic ladder is one ``Template`` (a task family with a
fixed instruction shape, setup, golden policy and verifier) resolved by a
seeded parameter draw. Draws are keyed by the stable string id
``f"{template_id}:{seed}"`` hashed with sha256 and fed to ``random.Random``, so
the 150 concrete tasks are reproducible offline from nothing but this module —
no wall-clock, no default random state.

Geometry lives here too: fixture widgets and page elements are placed on a
4x3 (12-cell) grid over the 1920x1080 screen with seeded per-cell jitter, and
the fixture specs are the exact JSON ``shortgoal_fixture`` renders in the guest
(shared geometry helpers are imported from it so offline targets and in-guest
widgets cannot drift).

Splits (``build_split_manifest``): the 4 Tier-B templates contribute all 6
seeds as unseen-surface holdout (24); the other 21 templates give seeds 0..4 to
train (105) and the last seed to Tier A, the unseen-params holdout (21) — 150
tasks, mutually disjoint at (template_id, seed) granularity.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import shortgoal_fixture as fixture

SCREEN_WH = (1920, 1080)
GRID_COLS, GRID_ROWS = 4, 3
N_CELLS = GRID_COLS * GRID_ROWS
CELL_WH = (SCREEN_WH[0] // GRID_COLS, SCREEN_WH[1] // GRID_ROWS)
JITTER_XY = (70, 50)

N_SEEDS = 6
CATEGORIES = ("terminal", "editor", "fixture", "browser")
SPLIT_NAMES = ("train", "tier_a", "tier_b")

SETUP_TERMINAL = "terminal"
SETUP_EDITOR = "editor_file"
SETUP_FIXTURE = "fixture"
SETUP_PAGE = "browser_page"

BUTTON_SIZE = (180, 70)
SQUARE_SIZE = (160, 160)
LINK_SIZE = (240, 80)
FIELD_SIZE = (640, 70)
PAGE_BUTTON_SIZE = (260, 84)
WEB_PAGE_HEIGHT = 2160
WEB_BOTTOM_NOTCHES = -20

FILE_STEMS = (
    "notes", "draft", "report", "ledger", "sketch", "memo", "index", "recipe",
    "roster", "budget", "journal", "outline",
)
FILE_EXTS = (".txt", ".md", ".log", ".csv", ".cfg")
DIR_WORDS = (
    "archive", "backup", "inbox", "scratch", "sandbox", "workspace", "outbox",
    "staging", "toolbox", "vault",
)
SHORT_WORDS = (
    "hello", "ready", "alpha", "bravo", "delta", "omega", "quiet", "amber",
    "cobalt", "ember",
)
SENTENCES = (
    "the quick brown fox jumps",
    "meeting moved to nine sharp",
    "remember to water the plants",
    "second draft needs a title",
    "pack the blue notebook",
    "coffee beans are running low",
    "call the workshop on friday",
    "backup finished without errors",
)
ESCAPE_PAYLOADS = (
    'say "hi" twice',
    'path C:\\temp\\log',
    'a "b" and \\c',
    'quote " and slash \\',
    '"quoted start" plain end',
    'mix "x" \\ "y" \\',
)
BUTTON_LABELS = (
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
    "India", "Juliet",
)
COLOR_NAMES = ("red", "blue", "green", "yellow", "purple", "orange")
COLOR_HEX = {
    "red": "#d33a2c",
    "blue": "#2c62d3",
    "green": "#2f8f46",
    "yellow": "#e0b400",
    "purple": "#7a3fbf",
    "orange": "#e07820",
}
ROW_WORDS = (
    "anchor", "beacon", "cedar", "dune", "elm", "fjord", "granite", "harbor",
    "ivory", "juniper", "kelp", "lagoon", "marble", "nectar", "onyx", "pebble",
)
LINK_LABELS = (
    "Overview", "Pricing", "Changelog", "Downloads", "Support", "Careers",
    "Roadmap", "Security",
)
SLIDER_TICKS = 10
SLIDER_LABEL = "level"
LIST_ROWS = 14
LIST_VISIBLE_ROWS = 5
LIST_ROW_HEIGHT = 96
COUNTER_PANE_WH = (640, 480)


@dataclass(frozen=True)
class Template:
    """One parameterised short-goal task family."""

    template_id: str
    category: str
    tier_b: bool
    single_action: bool
    instruction_fmt: str
    param_space: str
    setup_id: str
    policy_id: str
    verifier_id: str


@dataclass(frozen=True)
class ConcreteTask:
    """One seeded draw of a template: the unit that gets recorded and scored."""

    task_id: str
    template_id: str
    seed: int
    category: str
    tier_b: bool
    single_action: bool
    instruction: str
    params: dict[str, Any] = field(default_factory=dict)
    setup_id: str = SETUP_TERMINAL
    policy_id: str = ""
    verifier_id: str = ""


def _rng(template_id: str, seed: int) -> random.Random:
    key = f"{template_id}:{seed}".encode()
    return random.Random(int(hashlib.sha256(key).hexdigest(), 16))


def _token(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(10))


def _cell_point(rng: random.Random, cell: int) -> list[int]:
    if not isinstance(cell, int) or not 0 <= cell < N_CELLS:
        raise ValueError(f"grid cell must be in [0,{N_CELLS - 1}], got {cell!r}")
    col, row = cell % GRID_COLS, cell // GRID_COLS
    return [
        col * CELL_WH[0] + CELL_WH[0] // 2 + rng.randint(-JITTER_XY[0], JITTER_XY[0]),
        row * CELL_WH[1] + CELL_WH[1] // 2 + rng.randint(-JITTER_XY[1], JITTER_XY[1]),
    ]


def _tiles(rng: random.Random, labels: list[str], size: tuple[int, int]) -> list[dict[str, Any]]:
    cells = rng.sample(range(N_CELLS), len(labels))
    return [
        {"label": label, "center": _cell_point(rng, cell), "size": list(size)}
        for label, cell in zip(labels, cells, strict=True)
    ]


def _filename(rng: random.Random) -> str:
    return f"{rng.choice(FILE_STEMS)}_{rng.randrange(10, 100)}{rng.choice(FILE_EXTS)}"


def _dirname(rng: random.Random) -> str:
    return f"{rng.choice(DIR_WORDS)}_{rng.randrange(10, 100)}"


def _pane(rng: random.Random, size: tuple[int, int], jitter: tuple[int, int]) -> list[int]:
    cx = SCREEN_WH[0] // 2 + rng.randint(-jitter[0], jitter[0])
    cy = SCREEN_WH[1] // 2 + rng.randint(-jitter[1], jitter[1])
    return fixture.widget_bbox([cx, cy], list(size))


def _draw_term_touch(rng: random.Random) -> dict[str, Any]:
    name = _filename(rng)
    return {
        "filename": name,
        "command": f"touch {name}",
        "expect": {"path": name, "exists": True},
    }


def _draw_term_mkdir(rng: random.Random) -> dict[str, Any]:
    name = _dirname(rng)
    return {
        "dirname": name,
        "command": f"mkdir {name}",
        "expect": {"path": name, "is_dir": True},
    }


def _draw_term_echo_create(rng: random.Random) -> dict[str, Any]:
    name, word = _filename(rng), rng.choice(SHORT_WORDS)
    return {
        "filename": name,
        "text": word,
        "command": f"echo {word} > {name}",
        "expect": {"path": name, "content": f"{word}\n"},
    }


def _draw_term_append(rng: random.Random) -> dict[str, Any]:
    name, base, word = _filename(rng), rng.choice(SENTENCES), rng.choice(SHORT_WORDS)
    return {
        "filename": name,
        "text": word,
        "setup_files": [{"path": name, "content": f"{base}\n"}],
        "command": f"echo {word} >> {name}",
        "expect": {"path": name, "content": f"{base}\n{word}\n"},
    }


def _draw_term_rm(rng: random.Random) -> dict[str, Any]:
    name = _filename(rng)
    return {
        "filename": name,
        "setup_files": [{"path": name, "content": f"{rng.choice(SENTENCES)}\n"}],
        "command": f"rm {name}",
        "expect": {"path": name, "exists": False},
    }


def _draw_term_cp(rng: random.Random) -> dict[str, Any]:
    src, dst, body = _filename(rng), _filename(rng), rng.choice(SENTENCES)
    while dst == src:
        dst = _filename(rng)
    return {
        "src": src,
        "dst": dst,
        "setup_files": [{"path": src, "content": f"{body}\n"}],
        "command": f"cp {src} {dst}",
        "expect": {"path": dst, "content": f"{body}\n"},
    }


def _draw_term_chmod_x(rng: random.Random) -> dict[str, Any]:
    name = f"{rng.choice(FILE_STEMS)}_{rng.randrange(10, 100)}.sh"
    return {
        "filename": name,
        "setup_files": [{"path": name, "content": "#!/bin/sh\necho ok\n"}],
        "command": f"chmod +x {name}",
        "expect": {"path": name, "executable": True},
    }


def _draw_term_special_typing(rng: random.Random) -> dict[str, Any]:
    name, payload = _filename(rng), rng.choice(ESCAPE_PAYLOADS)
    return {
        "filename": name,
        "payload": payload,
        "command": f"printf '%s\\n' '{payload}' > {name}",
        "expect": {"path": name, "content": f"{payload}\n"},
    }


def _draw_term_two_commands(rng: random.Random) -> dict[str, Any]:
    folder, name = _dirname(rng), _filename(rng)
    return {
        "dirname": folder,
        "filename": name,
        "commands": [f"mkdir {folder}", f"touch {folder}/{name}"],
        "expect": {"path": f"{folder}/{name}", "exists": True},
    }


def _draw_term_launch_editor(rng: random.Random) -> dict[str, Any]:
    name = _filename(rng)
    return {
        "filename": name,
        "editor": "gedit",
        "setup_files": [{"path": name, "content": f"{rng.choice(SENTENCES)}\n"}],
        "command": f"gedit {name} &",
        "waits": rng.choice((1, 2)),
        "expect": {"window_title_contains": name},
    }


def _draw_gedit_write_save(rng: random.Random) -> dict[str, Any]:
    name, sentence = _filename(rng), rng.choice(SENTENCES)
    return {
        "filename": name,
        "sentence": sentence,
        "combos": [["ControlLeft", "KeyS"]],
        "setup_files": [{"path": name, "content": ""}],
        "expect": {"path": name, "content_stripped": sentence},
    }


def _draw_gedit_select_all_delete(rng: random.Random) -> dict[str, Any]:
    name, sentence = _filename(rng), rng.choice(SENTENCES)
    return {
        "filename": name,
        "sentence": sentence,
        "combos": [["ControlLeft", "KeyA"], ["Delete"], ["ControlLeft", "KeyS"]],
        "setup_files": [{"path": name, "content": f"{sentence}\n"}],
        "expect": {"path": name, "content_stripped": ""},
    }


def _draw_key_close_editor(rng: random.Random) -> dict[str, Any]:
    name = _filename(rng)
    return {
        "filename": name,
        "combos": [["ControlLeft", "KeyQ"]],
        "setup_files": [{"path": name, "content": f"{rng.choice(SENTENCES)}\n"}],
        "expect": {"process_absent": "gedit"},
    }


def _draw_key_terminal_newtab(rng: random.Random) -> dict[str, Any]:
    folder = _dirname(rng)
    return {
        "combos": [["ControlLeft", "ShiftLeft", "KeyT"]],
        "workdir": folder,
        "setup_dirs": [folder],
        "expect": {"min_terminal_tabs": 2},
    }


def _buttons_spec(rng: random.Random, count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = rng.sample(BUTTON_LABELS, count)
    buttons = _tiles(rng, labels, BUTTON_SIZE)
    kind = "two_buttons" if count == 2 else "buttons"
    return {
        "kind": kind,
        "screen": list(SCREEN_WH),
        "buttons": buttons,
        "commit_key": fixture.COMMIT_KEY,
    }, buttons


def _draw_fx_click_button(rng: random.Random) -> dict[str, Any]:
    spec, buttons = _buttons_spec(rng, 4)
    target = buttons[rng.randrange(len(buttons))]
    return {
        "label": target["label"],
        "target_xy": list(target["center"]),
        "fixture_spec": spec,
        "expect": {"clicked": [target["label"]], "committed": True},
    }


def _draw_fx_click_color(rng: random.Random) -> dict[str, Any]:
    names = rng.sample(COLOR_NAMES, 4)
    squares = _tiles(rng, names, SQUARE_SIZE)
    for square in squares:
        square["color"] = COLOR_HEX[square["label"]]
    target = squares[rng.randrange(len(squares))]
    return {
        "color": target["label"],
        "target_xy": list(target["center"]),
        "fixture_spec": {
            "kind": "colors",
            "screen": list(SCREEN_WH),
            "squares": squares,
            "commit_key": fixture.COMMIT_KEY,
        },
        "expect": {"clicked": [target["label"]], "committed": True},
    }


def _draw_fx_double_click(rng: random.Random) -> dict[str, Any]:
    spec, buttons = _buttons_spec(rng, 3)
    target = buttons[rng.randrange(len(buttons))]
    return {
        "label": target["label"],
        "target_xy": list(target["center"]),
        "fixture_spec": spec,
        "expect": {"double_clicked": [target["label"]], "committed": True},
    }


def _draw_fx_right_click(rng: random.Random) -> dict[str, Any]:
    spec, buttons = _buttons_spec(rng, 3)
    target = buttons[rng.randrange(len(buttons))]
    return {
        "label": target["label"],
        "target_xy": list(target["center"]),
        "fixture_spec": spec,
        "expect": {"right_clicked": [target["label"]], "committed": True},
    }


def _draw_fx_drag_slider(rng: random.Random) -> dict[str, Any]:
    track_y = SCREEN_WH[1] // 2 + rng.randint(-120, 120)
    x0 = 260 + rng.randrange(0, 160)
    slider = {
        "label": SLIDER_LABEL,
        "track": [x0, track_y - 9, x0 + 1240, track_y + 9],
        "ticks": SLIDER_TICKS,
        "value": rng.randrange(0, 4),
        "handle_size": [30, 64],
    }
    target_value = rng.randrange(6, SLIDER_TICKS + 1)
    return {
        "label": SLIDER_LABEL,
        "start_value": slider["value"],
        "target_value": target_value,
        "handle_xy": [fixture.slider_tick_x(slider, slider["value"]), track_y],
        "target_xy": [fixture.slider_tick_x(slider, target_value), track_y],
        "fixture_spec": {
            "kind": "slider",
            "screen": list(SCREEN_WH),
            "slider": slider,
            "commit_key": fixture.COMMIT_KEY,
        },
        "expect": {"slider_value": target_value, "committed": True},
    }


def _draw_fx_scroll_counter(rng: random.Random) -> dict[str, Any]:
    pane = _pane(rng, COUNTER_PANE_WH, (260, 140))
    notches = rng.choice((-5, -4, -3, 3, 4, 5))
    return {
        "notches": notches,
        "count": abs(notches),
        "direction": "up" if notches > 0 else "down",
        "pane_xy": fixture.bbox_center(pane),
        "fixture_spec": {
            "kind": "scroll_counter",
            "screen": list(SCREEN_WH),
            "scroll": {"pane": pane},
            "commit_key": fixture.COMMIT_KEY,
        },
        "expect": {"wheel_notches": notches, "committed": True},
    }


def _draw_fx_scroll_find_click(rng: random.Random) -> dict[str, Any]:
    words = rng.sample(ROW_WORDS, LIST_ROWS)
    rows = [f"{index + 1:02d} {word}" for index, word in enumerate(words)]
    pane = _pane(rng, (760, LIST_VISIBLE_ROWS * LIST_ROW_HEIGHT), (220, 120))
    scroll = {
        "pane": pane,
        "rows": rows,
        "row_height": LIST_ROW_HEIGHT,
        "visible_rows": LIST_VISIBLE_ROWS,
    }
    offset = fixture.max_scroll_offset(scroll)
    target_index = rng.randrange(offset, LIST_ROWS)
    bursts = list(rng.choice(((-6, -6), (-7, -5), (-9, -4))))
    return {
        "row": rows[target_index],
        "row_index": target_index,
        "bursts": bursts,
        "scroll_offset": offset,
        "pane_xy": fixture.bbox_center(pane),
        "target_xy": fixture.bbox_center(fixture.row_bbox(scroll, target_index, offset)),
        "fixture_spec": {
            "kind": "scroll_list",
            "screen": list(SCREEN_WH),
            "scroll": scroll,
            "commit_key": fixture.COMMIT_KEY,
        },
        "expect": {"clicked": [rows[target_index]], "scroll_offset": offset},
    }


def _draw_fx_two_buttons_order(rng: random.Random) -> dict[str, Any]:
    spec, buttons = _buttons_spec(rng, 2)
    order = rng.sample(range(2), 2)
    first, second = buttons[order[0]], buttons[order[1]]
    return {
        "first": first["label"],
        "second": second["label"],
        "first_xy": list(first["center"]),
        "second_xy": list(second["center"]),
        "fixture_spec": spec,
        "expect": {"clicked": [first["label"], second["label"]]},
    }


def _draw_web_click_link(rng: random.Random) -> dict[str, Any]:
    labels = rng.sample(LINK_LABELS, 6)
    links = _tiles(rng, labels, LINK_SIZE)
    target = links[rng.randrange(len(links))]
    token = _token(rng)
    return {
        "page_kind": "link_grid",
        "token": token,
        "links": links,
        "label": target["label"],
        "target_xy": list(target["center"]),
        "expect": {"title": token},
    }


def _draw_web_type_input(rng: random.Random) -> dict[str, Any]:
    token, text = _token(rng), rng.choice(SENTENCES)
    return {
        "page_kind": "input",
        "token": token,
        "text": text,
        "input_xy": [
            SCREEN_WH[0] // 2 + rng.randint(-240, 240),
            SCREEN_WH[1] // 2 + rng.randint(-160, 160),
        ],
        "input_size": list(FIELD_SIZE),
        "expect": {"title": f"{token}:{text}"},
    }


def _draw_web_scroll_click(rng: random.Random) -> dict[str, Any]:
    token, label = _token(rng), rng.choice(("Continue", "Accept", "Subscribe", "Download"))
    x = SCREEN_WH[0] // 2 + rng.randint(-420, 420)
    below = rng.randrange(240, 880)
    return {
        "page_kind": "below_fold_button",
        "token": token,
        "label": label,
        "page_height": WEB_PAGE_HEIGHT,
        "button_page_xy": [x, SCREEN_WH[1] + below],
        "button_size": list(PAGE_BUTTON_SIZE),
        "notches": WEB_BOTTOM_NOTCHES,
        "target_xy": [x, below],
        "expect": {"title": token},
    }


TEMPLATES: tuple[Template, ...] = (
    Template(
        template_id="term_touch", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, create an empty file named {filename} in the home directory.",
        param_space="filename = stem_NN.ext", setup_id=SETUP_TERMINAL,
        policy_id="p_term_type_enter", verifier_id="guest_path_exists",
    ),
    Template(
        template_id="term_mkdir", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, create a directory named {dirname} in the home directory.",
        param_space="dirname = word_NN", setup_id=SETUP_TERMINAL,
        policy_id="p_term_type_enter", verifier_id="guest_dir_exists",
    ),
    Template(
        template_id="term_echo_create", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, create a file named {filename} whose only line is {text}.",
        param_space="filename = stem_NN.ext, text = short word", setup_id=SETUP_TERMINAL,
        policy_id="p_term_type_enter", verifier_id="guest_file_content",
    ),
    Template(
        template_id="term_append", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, append the word {text} as a new last line of {filename}.",
        param_space="filename = stem_NN.ext seeded with one line, text = short word",
        setup_id=SETUP_TERMINAL, policy_id="p_term_type_enter", verifier_id="guest_file_content",
    ),
    Template(
        template_id="term_rm", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, delete the file {filename} from the home directory.",
        param_space="filename = stem_NN.ext seeded with one line", setup_id=SETUP_TERMINAL,
        policy_id="p_term_type_enter", verifier_id="guest_path_absent",
    ),
    Template(
        template_id="term_cp", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, copy {src} to a new file named {dst}.",
        param_space="src/dst = distinct stem_NN.ext, src seeded with one line",
        setup_id=SETUP_TERMINAL, policy_id="p_term_type_enter", verifier_id="guest_file_content",
    ),
    Template(
        template_id="term_chmod_x", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, make the script {filename} executable.",
        param_space="filename = stem_NN.sh seeded with a shebang", setup_id=SETUP_TERMINAL,
        policy_id="p_term_type_enter", verifier_id="guest_file_executable",
    ),
    Template(
        template_id="term_special_typing", category="terminal", tier_b=False, single_action=True,
        instruction_fmt="Using the terminal, write the exact line {payload} into a new file named {filename}.",
        param_space="payload = escape-stress text with double quotes and backslashes",
        setup_id=SETUP_TERMINAL, policy_id="p_term_type_enter", verifier_id="guest_file_content",
    ),
    Template(
        template_id="term_two_commands", category="terminal", tier_b=False, single_action=False,
        instruction_fmt="Using the terminal, create a directory named {dirname} and then an empty file named {filename} inside it.",
        param_space="dirname = word_NN, filename = stem_NN.ext", setup_id=SETUP_TERMINAL,
        policy_id="p_term_two_commands", verifier_id="guest_path_exists",
    ),
    Template(
        template_id="term_launch_editor", category="terminal", tier_b=False, single_action=False,
        instruction_fmt="Using the terminal, open the file {filename} in the gedit text editor and wait until its window is up.",
        param_space="filename = stem_NN.ext seeded with one line, waits = 1..2 NO_OP turns",
        setup_id=SETUP_TERMINAL, policy_id="p_term_launch_editor", verifier_id="guest_window_title",
    ),
    Template(
        template_id="gedit_write_save", category="editor", tier_b=False, single_action=False,
        instruction_fmt="Type the line {sentence} into the open text editor and save the file.",
        param_space="filename = stem_NN.ext opened empty, sentence = short sentence",
        setup_id=SETUP_EDITOR, policy_id="p_gedit_write_save", verifier_id="guest_file_content",
    ),
    Template(
        template_id="gedit_select_all_delete", category="editor", tier_b=True, single_action=False,
        instruction_fmt="Delete all of the text in the open text editor and save the now empty file.",
        param_space="filename = stem_NN.ext opened with one seeded sentence",
        setup_id=SETUP_EDITOR, policy_id="p_key_combos", verifier_id="guest_file_content",
    ),
    Template(
        template_id="key_close_editor", category="editor", tier_b=False, single_action=True,
        instruction_fmt="Close the open text editor window with its keyboard shortcut.",
        param_space="filename = stem_NN.ext opened with one seeded sentence",
        setup_id=SETUP_EDITOR, policy_id="p_key_combos", verifier_id="guest_process_absent",
    ),
    Template(
        template_id="key_terminal_newtab", category="editor", tier_b=False, single_action=True,
        instruction_fmt="Open a second tab in the open terminal window with its keyboard shortcut.",
        param_space="workdir = word_NN created before the terminal opens",
        setup_id=SETUP_TERMINAL, policy_id="p_key_combos", verifier_id="guest_terminal_tabs",
    ),
    Template(
        template_id="fx_click_button", category="fixture", tier_b=False, single_action=False,
        instruction_fmt="In the fixture window, click the {label} button, then press Enter to confirm.",
        param_space="4 labelled buttons on distinct grid cells, one target",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_click_commit", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_click_color", category="fixture", tier_b=False, single_action=False,
        instruction_fmt="In the fixture window, click the {color} square, then press Enter to confirm.",
        param_space="4 coloured squares on distinct grid cells, one target colour",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_click_commit", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_double_click", category="fixture", tier_b=False, single_action=False,
        instruction_fmt="In the fixture window, double-click the {label} button, then press Enter to confirm.",
        param_space="3 labelled buttons on distinct grid cells, one target",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_double_click_commit", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_right_click", category="fixture", tier_b=True, single_action=False,
        instruction_fmt="In the fixture window, right-click the {label} button, then press Enter to confirm.",
        param_space="3 labelled buttons on distinct grid cells, one target",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_right_click_commit", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_drag_slider", category="fixture", tier_b=False, single_action=False,
        instruction_fmt="In the fixture window, drag the {label} slider handle from {start_value} to {target_value}, then press Enter to confirm.",
        param_space="10-tick track with seeded y and x0, start 0..3, target 6..10",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_drag_slider", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_scroll_counter", category="fixture", tier_b=False, single_action=False,
        instruction_fmt="In the fixture window, put the pointer over the scroll pad and scroll {direction} {count} notches, then press Enter to confirm.",
        param_space="seeded pane box, notches in -5..-3 / 3..5", setup_id=SETUP_FIXTURE,
        policy_id="p_fx_scroll_commit", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_scroll_find_click", category="fixture", tier_b=True, single_action=False,
        instruction_fmt="In the fixture window, scroll the list to its end and click the row {row}.",
        param_space="14 seeded rows, 5 visible, target row inside the clamped bottom window",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_scroll_find_click", verifier_id="fixture_state",
    ),
    Template(
        template_id="fx_two_buttons_order", category="fixture", tier_b=False, single_action=False,
        instruction_fmt="In the fixture window, click the {first} button and then the {second} button.",
        param_space="2 labelled buttons on distinct grid cells, seeded order",
        setup_id=SETUP_FIXTURE, policy_id="p_fx_two_clicks", verifier_id="fixture_state",
    ),
    Template(
        template_id="web_click_link", category="browser", tier_b=False, single_action=True,
        instruction_fmt="On the open page, click the link labelled {label}.",
        param_space="6 links on distinct grid cells, seeded target and title token",
        setup_id=SETUP_PAGE, policy_id="p_web_click", verifier_id="browser_title",
    ),
    Template(
        template_id="web_type_input", category="browser", tier_b=False, single_action=False,
        instruction_fmt="On the open page, click the text field, type {text} and press Enter.",
        param_space="one field on a seeded middle-row cell, seeded text and title token",
        setup_id=SETUP_PAGE, policy_id="p_web_click_type", verifier_id="browser_title",
    ),
    Template(
        template_id="web_scroll_click", category="browser", tier_b=True, single_action=False,
        instruction_fmt="On the open page, scroll to the bottom and click the {label} button.",
        param_space="2160px page, button 240..880px below the fold, seeded title token",
        setup_id=SETUP_PAGE, policy_id="p_web_scroll_click", verifier_id="browser_title",
    ),
)

_DRAWS: dict[str, Callable[[random.Random], dict[str, Any]]] = {
    "term_touch": _draw_term_touch,
    "term_mkdir": _draw_term_mkdir,
    "term_echo_create": _draw_term_echo_create,
    "term_append": _draw_term_append,
    "term_rm": _draw_term_rm,
    "term_cp": _draw_term_cp,
    "term_chmod_x": _draw_term_chmod_x,
    "term_special_typing": _draw_term_special_typing,
    "term_two_commands": _draw_term_two_commands,
    "term_launch_editor": _draw_term_launch_editor,
    "gedit_write_save": _draw_gedit_write_save,
    "gedit_select_all_delete": _draw_gedit_select_all_delete,
    "key_close_editor": _draw_key_close_editor,
    "key_terminal_newtab": _draw_key_terminal_newtab,
    "fx_click_button": _draw_fx_click_button,
    "fx_click_color": _draw_fx_click_color,
    "fx_double_click": _draw_fx_double_click,
    "fx_right_click": _draw_fx_right_click,
    "fx_drag_slider": _draw_fx_drag_slider,
    "fx_scroll_counter": _draw_fx_scroll_counter,
    "fx_scroll_find_click": _draw_fx_scroll_find_click,
    "fx_two_buttons_order": _draw_fx_two_buttons_order,
    "web_click_link": _draw_web_click_link,
    "web_type_input": _draw_web_type_input,
    "web_scroll_click": _draw_web_scroll_click,
}


def _check_catalog() -> dict[str, Template]:
    by_id: dict[str, Template] = {}
    for template in TEMPLATES:
        if template.template_id in by_id:
            raise ValueError(f"duplicate template id: {template.template_id}")
        if template.category not in CATEGORIES:
            raise ValueError(f"unknown category {template.category!r} for {template.template_id}")
        if template.setup_id not in (SETUP_TERMINAL, SETUP_EDITOR, SETUP_FIXTURE, SETUP_PAGE):
            raise ValueError(f"unknown setup {template.setup_id!r} for {template.template_id}")
        if template.template_id not in _DRAWS:
            raise ValueError(f"template {template.template_id} has no param draw")
        by_id[template.template_id] = template
    if len(by_id) != len(_DRAWS):
        raise ValueError(f"catalog/draw mismatch: {sorted(set(_DRAWS) - set(by_id))}")
    return by_id


TEMPLATES_BY_ID: dict[str, Template] = _check_catalog()
N_TEMPLATES = len(TEMPLATES)
TIER_B_TEMPLATE_IDS: tuple[str, ...] = tuple(t.template_id for t in TEMPLATES if t.tier_b)


def draw_params(template_id: str, seed: int) -> dict[str, Any]:
    """The seeded parameter draw for one ``(template_id, seed)`` task."""
    if template_id not in TEMPLATES_BY_ID:
        raise KeyError(f"unknown template id: {template_id!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    params = _DRAWS[template_id](_rng(template_id, seed))
    if "fixture_spec" in params:
        fixture.validate_spec(params["fixture_spec"])
    return params


def task_id(template_id: str, seed: int) -> str:
    """The stable task id of one seeded draw."""
    if template_id not in TEMPLATES_BY_ID:
        raise KeyError(f"unknown template id: {template_id!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 99:
        raise ValueError(f"seed must be an int in [0,99], got {seed!r}")
    return f"{template_id}__s{seed:02d}"


def concrete_task(template_id: str, seed: int) -> ConcreteTask:
    """One resolved task: instruction text plus the seeded params."""
    template = TEMPLATES_BY_ID[template_id]
    params = draw_params(template_id, seed)
    return ConcreteTask(
        task_id=task_id(template_id, seed),
        template_id=template_id,
        seed=seed,
        category=template.category,
        tier_b=template.tier_b,
        single_action=template.single_action,
        instruction=template.instruction_fmt.format(**params),
        params=params,
        setup_id=template.setup_id,
        policy_id=template.policy_id,
        verifier_id=template.verifier_id,
    )


def concrete_tasks(n_seeds: int = N_SEEDS) -> list[ConcreteTask]:
    """Every template resolved at seeds ``0..n_seeds-1``, in catalog order."""
    if not isinstance(n_seeds, int) or not 2 <= n_seeds <= 99:
        raise ValueError(f"n_seeds must be an int in [2,99], got {n_seeds!r}")
    return [
        concrete_task(template.template_id, seed)
        for template in TEMPLATES
        for seed in range(n_seeds)
    ]


def build_split_manifest(n_seeds: int = N_SEEDS) -> dict[str, Any]:
    """The train / tier_a / tier_b task-id manifest, validated disjoint."""
    tasks = concrete_tasks(n_seeds)
    tier_a_seed = n_seeds - 1
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for task in tasks:
        if task.tier_b:
            splits["tier_b"].append(task.task_id)
        elif task.seed == tier_a_seed:
            splits["tier_a"].append(task.task_id)
        else:
            splits["train"].append(task.task_id)
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        ids = splits[name]
        if len(set(ids)) != len(ids):
            raise ValueError(f"split {name} repeats a task id")
        overlap = seen & set(ids)
        if overlap:
            raise ValueError(f"split {name} overlaps an earlier split: {sorted(overlap)}")
        seen |= set(ids)
    if len(seen) != len(tasks):
        raise ValueError(f"splits cover {len(seen)} of {len(tasks)} tasks")
    return {
        "n_seeds": n_seeds,
        "tier_a_seed": tier_a_seed,
        "tier_b_templates": list(TIER_B_TEMPLATE_IDS),
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "n_tasks": len(tasks),
        **splits,
    }


OVERFIT1_TASK_ID = "term_touch__s00"


def _train_rotation() -> list[str]:
    lanes = [
        [t.template_id for t in TEMPLATES if t.category == category and not t.tier_b]
        for category in CATEGORIES
    ]
    rotation: list[str] = []
    for index in range(max(len(lane) for lane in lanes)):
        rotation.extend(lane[index] for lane in lanes if index < len(lane))
    return rotation


def _overfit32_task_ids() -> tuple[str, ...]:
    rotation = _train_rotation()
    ids = [task_id(template_id, 0) for template_id in rotation]
    ids.extend(task_id(template_id, 1) for template_id in rotation[: 32 - len(ids)])
    if len(set(ids)) != 32:
        raise ValueError(f"overfit32 needs 32 distinct ids, got {len(set(ids))}")
    if {TEMPLATES_BY_ID[i.split('__s')[0]].category for i in ids} != set(CATEGORIES):
        raise ValueError("overfit32 must span every category")
    return tuple(ids)


OVERFIT32_TASK_IDS: tuple[str, ...] = _overfit32_task_ids()
