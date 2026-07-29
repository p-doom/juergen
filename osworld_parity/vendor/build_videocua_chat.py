#!/usr/bin/env python3
"""Build crowd-cast dense-action chat records from VideoCUA tasks."""

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SYSPROMPT = "You operate a desktop computer. The first user turn shows the initial screen and the user's goal; subsequent user turns show the current screen. Reply with the next action toward that goal as `<dx> <dy> <scroll>` optionally followed by ` ; +KEY -KEY` events, or `NO_OP` if no action."
TERMINAL_ACTIONS = {"TERMINATE", "TERMINATE_SUCCESS", "AFTER_LAST_ACTION"}
TYPING_ACTIONS = {"TYPING", "TYPE", "TEXT"}
BUTTON_KEYS = {"LMB", "RMB", "MMB"}

COUNT_KEYS = (
    "defaulted_button",
    "moveto_degenerate",
    "typing_unmapped_chars",
    "unmapped_keys",
    "clicks",
    "typing_chars",
    "press_n",
    "hotkey_n",
    "drag_n",
    "scroll_events",
    "idm_move_frames",
    "idm_keypress_n",
    "idm_click_n",
    "idm_scroll_n",
    "idm_malformed",
    "forced_release",
    "numclicks_capped",
    "scroll_x_dropped",
    "keydown_open",
    "mousedown_as_key",
    "unknown_action_type",
    "anchor_oob",
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_row(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def empty_counts():
    return {key: 0 for key in COUNT_KEYS}


def clean_name(raw):
    text = "" if raw is None else str(raw)
    text = text.strip().strip('"\'').strip()
    return re.sub(r"\s+", " ", text).lower()


def normalize_button(raw):
    if raw is None:
        return "LMB"
    original = clean_name(raw)
    alpha = re.sub(r"[^a-z]", "", original)
    if (
        "left" in alpha
        or alpha.startswith("lef")
        or alpha == "l"
        or original in {"click", "mouse click", "mouseclick", "na", "n/a", ""}
    ):
        return "LMB"
    if "right" in alpha:
        return "RMB"
    if "middle" in alpha:
        return "MMB"
    return None


def normalize_key(raw, key_map, typing=False):
    if raw is None:
        return None
    original = str(raw).strip()
    name = clean_name(original)
    entry = key_map.get("aliases", {}).get(name)
    if entry is not None:
        return entry
    if len(original) == 1 and original.isascii() and original.isalpha():
        base = f"Key{original.upper()}"
        if typing and original.isupper():
            return ["ShiftLeft", base]
        return base
    if len(original) == 1 and original.isascii() and original.isdigit():
        return f"Num{original}"
    return None


def split_chord_tokens(raw, key_map):
    text = str(raw).strip().replace('"', "").replace("'", "")
    explicit = re.split(r"\s*\+\s*|(?<=\w)\s*-\s*(?=\w)", text)
    if len(explicit) > 1:
        return [part.strip() for part in explicit if part.strip()]

    if normalize_key(text, key_map) is not None or clean_name(text) in key_map.get("modifiers", {}):
        return [text]

    words = text.split()
    if len(words) <= 1:
        return [text] if text else []
    known = set(key_map.get("aliases", {})) | set(key_map.get("modifiers", {}))
    tokens = []
    index = 0
    while index < len(words):
        match = None
        for end in range(len(words), index, -1):
            candidate = clean_name(" ".join(words[index:end]))
            if candidate in known or (end == index + 1 and normalize_key(candidate, key_map) is not None):
                match = " ".join(words[index:end])
                index = end
                break
        if match is None:
            return [text]
        tokens.append(match)
    return tokens


def normalize_chord(raw, key_map):
    if raw is None:
        return None
    modifiers = []
    base_keys = []
    for token in split_chord_tokens(raw, key_map):
        name = clean_name(token)
        modifier = key_map.get("modifiers", {}).get(name)
        if modifier is not None:
            if modifier not in modifiers:
                modifiers.append(modifier)
            continue
        entry = normalize_key(token, key_map)
        if entry is None:
            return None
        if isinstance(entry, list):
            for item in entry[:-1]:
                if item not in modifiers:
                    modifiers.append(item)
            base_keys.append(entry[-1])
        else:
            base_keys.append(entry)
    if not modifiers and not base_keys:
        return None
    return modifiers, base_keys


def char_to_chord(char, key_map):
    if char in "\r\n":
        return [], ["Return"]
    if char == "\t":
        return [], ["Tab"]
    if char == " ":
        return [], ["Space"]
    entry = normalize_key(char, key_map, typing=True)
    if entry is None:
        entry = key_map.get("chars", {}).get(char)
    if entry is None:
        return None
    if isinstance(entry, list):
        return list(entry[:-1]), [entry[-1]]
    return [], [entry]


def chord_events(modifiers, base_keys, mode="tap"):
    events = []
    if mode == "tap":
        events.extend(("press", key) for key in modifiers)
        for key in base_keys:
            events.append(("press", key))
            events.append(("release", key))
        events.extend(("release", key) for key in reversed(modifiers))
    elif mode == "down":
        events.extend(("press", key) for key in modifiers)
        events.extend(("press", key) for key in base_keys)
    elif mode == "up":
        events.extend(("release", key) for key in reversed(base_keys))
        events.extend(("release", key) for key in reversed(modifiers))
    return events


def interval_index(timestamp, n_intervals):
    if n_intervals <= 0:
        return 0
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        timestamp = 0.0
    return min(max(math.floor(timestamp * 2.0), 0), n_intervals - 1)


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def action_value(params):
    for key in ("keys", "key", "text"):
        if params.get(key) is not None:
            return params[key]
    return None


def append_unmapped(unmapped, task, action_type, raw_value, context):
    unmapped.append({
        "task": task,
        "action_type": action_type,
        "raw_value": raw_value,
        "context": context,
    })


def add_anchor(anchors, counts, timestamp, x, y, width, height, order):
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return
    if x < 0 or y < 0 or x > width or y > height:
        counts["anchor_oob"] += 1
        return
    anchors.append((float(timestamp), x, y, len(anchors)))


def parse_idm(idm, n_intervals, width, height, counts):
    pixel_sums = [[0.0, 0.0] for _ in range(n_intervals)]
    normalized_moves = defaultdict(lambda: [0.0, 0.0])
    for chunk in idm.get("chunks") or []:
        if not chunk or "error" in chunk:
            continue
        start = integer(chunk.get("start_frame"), 0)
        for prediction in chunk.get("predictions") or []:
            prediction_type = str(prediction.get("type") or "")
            lowered = prediction_type.lower()
            if lowered == "mousemove":
                match = re.fullmatch(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*", str(prediction.get("details", "")))
                frame_match = re.fullmatch(r"F(\d+)", str(prediction.get("frame", "")), re.IGNORECASE)
                if match is None or frame_match is None:
                    counts["idm_malformed"] += 1
                    continue
                dx_norm, dy_norm = int(match.group(1)), int(match.group(2))
                frame = start + int(frame_match.group(1))
                target = min(max(math.floor(frame * 2 / 5), 0), n_intervals - 1)
                pixel_sums[target][0] += dx_norm / 1000.0 * width
                pixel_sums[target][1] += dy_norm / 1000.0 * height
                normalized_moves[frame][0] += dx_norm
                normalized_moves[frame][1] += dy_norm
                counts["idm_move_frames"] += 1
            elif "key" in lowered:
                counts["idm_keypress_n"] += 1
            elif "click" in lowered:
                counts["idm_click_n"] += 1
            elif "scroll" in lowered:
                counts["idm_scroll_n"] += 1
    return pixel_sums, normalized_moves


def convert_actions(task, action_log, key_map, n_intervals, width, height, counts, unmapped):
    actions = action_log.get("action_log") or []
    interval_events = [[] for _ in range(n_intervals)]
    scroll_acc = [0 for _ in range(n_intervals)]
    anchors = []
    open_keys = Counter()
    open_buttons = []
    event_order = 0

    def emit(timestamp, kind, key):
        nonlocal event_order
        interval_events[interval_index(timestamp, n_intervals)].append(
            (float(number(timestamp)), event_order, kind, key)
        )
        event_order += 1

    def emit_many(timestamp, pairs):
        for kind, key in pairs:
            emit(timestamp, kind, key)

    def record_open(kind, key):
        if key in BUTTON_KEYS:
            if kind == "press":
                open_buttons.append(key)
            elif key in open_buttons:
                reverse_index = len(open_buttons) - 1 - open_buttons[::-1].index(key)
                open_buttons.pop(reverse_index)
        else:
            if kind == "press":
                open_keys[key] += 1
            elif open_keys[key] > 0:
                open_keys[key] -= 1

    def emit_typing_text(text, timestamp, next_timestamp):
        counts["typing_chars"] += len(text)
        duration_end = timestamp + len(text) / 8.0
        if next_timestamp is not None:
            duration_end = min(next_timestamp, duration_end)
        duration_end = max(timestamp, duration_end)
        for char_index, char in enumerate(text):
            char_time = timestamp if len(text) <= 1 else timestamp + (duration_end - timestamp) * char_index / (len(text) - 1)
            chord = char_to_chord(char, key_map)
            if chord is None:
                counts["typing_unmapped_chars"] += 1
                append_unmapped(unmapped, task, "TYPING", char, "typing_char")
                continue
            emit_many(char_time, chord_events(*chord, mode="tap"))

    for action_index, action in enumerate(actions):
        action_type = str(action.get("action_type") or "").upper()
        params = action.get("action_params") or {}
        timestamp = number(action.get("timestamp"), 0.0)
        next_timestamp = None
        if action_index + 1 < len(actions):
            next_timestamp = number(actions[action_index + 1].get("timestamp"), timestamp)

        if action_type == "CLICK":
            raw_button = params.get("text", params.get("button"))
            button = normalize_button(raw_button)
            if button is None:
                append_unmapped(unmapped, task, action_type, raw_button, "button")
                counts["unmapped_keys"] += 1
                button = "LMB"
                counts["defaulted_button"] += 1
            elif raw_button is None or clean_name(raw_button) in {"", "na", "n/a"}:
                counts["defaulted_button"] += 1
            click_count = integer(params.get("numClicks"), 1) or 1
            if click_count > 3:
                counts["numclicks_capped"] += 1
                click_count = 3
            click_count = max(click_count, 1)
            for _ in range(click_count):
                emit(timestamp, "press", button)
                emit(timestamp, "release", button)
            counts["clicks"] += click_count
            add_anchor(anchors, counts, timestamp, params.get("x"), params.get("y"), width, height, action_index)

        elif action_type == "MOVE_TO":
            has_endpoint = params.get("xEnd") is not None and params.get("yEnd") is not None
            all_zero = (
                number(params.get("x")) == 0
                and number(params.get("y")) == 0
                and has_endpoint
                and number(params.get("xEnd")) == 0
                and number(params.get("yEnd")) == 0
            )
            if all_zero:
                counts["moveto_degenerate"] += 1
            else:
                add_anchor(anchors, counts, timestamp, params.get("x"), params.get("y"), width, height, action_index * 2)
                if has_endpoint:
                    add_anchor(anchors, counts, timestamp, params.get("xEnd"), params.get("yEnd"), width, height, action_index * 2 + 1)

        elif action_type in TYPING_ACTIONS:
            text = str(params.get("text") or "")
            emit_typing_text(text, timestamp, next_timestamp)

        elif action_type in {"PRESS", "HOTKEY"}:
            if action_type == "PRESS":
                counts["press_n"] += 1
            else:
                counts["hotkey_n"] += 1
            raw_value = action_value(params)
            chord_raw = raw_value
            chord = None
            if isinstance(raw_value, list):
                chord_raw = " + ".join(str(item) for item in raw_value)
            if isinstance(chord_raw, str) and ":" in chord_raw:
                prefix, suffix = chord_raw.split(":", 1)
                prefix_chord = normalize_chord(prefix, key_map)
                if prefix_chord is not None:
                    chord = prefix_chord
                    append_unmapped(unmapped, task, action_type, suffix.strip(), "hotkey_paste_suffix")
            if chord is None:
                chord = normalize_chord(chord_raw, key_map)
            if chord is not None:
                emit_many(timestamp, chord_events(*chord, mode="tap"))
            elif action_type == "PRESS" and isinstance(raw_value, str) and len(raw_value) > 1:
                typed = [char_to_chord(char, key_map) for char in raw_value]
                if all(item is not None for item in typed):
                    emit_typing_text(raw_value, timestamp, next_timestamp)
                else:
                    counts["unmapped_keys"] += 1
                    append_unmapped(unmapped, task, action_type, raw_value, "key")
            else:
                counts["unmapped_keys"] += 1
                append_unmapped(unmapped, task, action_type, raw_value, "key")
            if action_type == "HOTKEY":
                add_anchor(anchors, counts, timestamp, params.get("x"), params.get("y"), width, height, action_index)

        elif action_type in {"KEY_DOWN", "KEY_UP"}:
            raw_value = action_value(params)
            chord = normalize_chord(raw_value, key_map)
            if chord is None:
                counts["unmapped_keys"] += 1
                append_unmapped(unmapped, task, action_type, raw_value, "key")
            else:
                mode = "down" if action_type == "KEY_DOWN" else "up"
                pairs = chord_events(*chord, mode=mode)
                emit_many(timestamp, pairs)
                for kind, key in pairs:
                    record_open(kind, key)

        elif action_type == "MOUSE_DOWN":
            raw_button = params.get("text", params.get("button"))
            button = normalize_button(raw_button)
            if button is None:
                chord = normalize_chord(raw_button, key_map)
                if chord is not None:
                    pairs = chord_events(*chord, mode="down")
                    emit_many(timestamp, pairs)
                    for kind, key in pairs:
                        record_open(kind, key)
                    counts["mousedown_as_key"] += 1
                else:
                    append_unmapped(unmapped, task, action_type, raw_button, "button")
                    counts["unmapped_keys"] += 1
                    button = "LMB"
                    counts["defaulted_button"] += 1
            if button is not None:
                emit(timestamp, "press", button)
                record_open("press", button)

        elif action_type == "DRAG_TO":
            counts["drag_n"] += 1
            add_anchor(anchors, counts, timestamp, params.get("x"), params.get("y"), width, height, action_index)

        elif action_type == "MOUSE_UP":
            raw_button = params.get("text", params.get("button"))
            button = normalize_button(raw_button)
            if button is None:
                button = open_buttons[-1] if open_buttons else "LMB"
            emit(timestamp, "release", button)
            record_open("release", button)

        elif action_type == "SCROLL":
            counts["scroll_events"] += 1
            scroll_acc[interval_index(timestamp, n_intervals)] += integer(params.get("scrollY"), 0)
            if params.get("scrollX") not in (None, 0, 0.0, "0", "0.0", ""):
                counts["scroll_x_dropped"] += 1

        elif action_type in TERMINAL_ACTIONS:
            pass

        else:
            counts["unknown_action_type"] += 1
            append_unmapped(unmapped, task, action_type, action.get("action_params"), "action_type")

    last_event_timestamp = max(
        (event[0] for events in interval_events for event in events),
        default=number(actions[-1].get("timestamp"), 0.0) if actions else 0.0,
    )
    for button in reversed(open_buttons):
        emit(last_event_timestamp, "release", button)
        counts["forced_release"] += 1
    for key, count in list(open_keys.items()):
        if count > 0:
            counts["keydown_open"] += count
            for _ in range(count):
                emit(last_event_timestamp, "release", key)
                counts["forced_release"] += 1

    anchors.sort(key=lambda item: (item[0], item[3]))
    return interval_events, scroll_acc, anchors


def percentile_nearest(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(percentile * len(ordered)) - 1, 0)
    return ordered[index]


def anchor_stats(anchors, normalized_moves, width, height):
    errors = []
    relative = []
    for first, second in zip(anchors, anchors[1:]):
        t1, x1, y1, _ = first
        t2, x2, y2, _ = second
        if t2 <= t1 + 1e-9:
            continue
        gt_dx = (x2 - x1) / width * 1000.0
        gt_dy = (y2 - y1) / height * 1000.0
        idm_dx = 0.0
        idm_dy = 0.0
        for frame, delta in normalized_moves.items():
            frame_time = frame / 5.0
            if t1 <= frame_time < t2:
                idm_dx += delta[0]
                idm_dy += delta[1]
        error = math.hypot(gt_dx - idm_dx, gt_dy - idm_dy)
        errors.append(error)
        relative.append(error / max(math.hypot(gt_dx, gt_dy), 50.0))
    return {
        "n_anchor_pairs": len(errors),
        "anchor_err_mean": statistics.fmean(errors) if errors else None,
        "anchor_err_median": statistics.median(errors) if errors else None,
        "anchor_err_p90": percentile_nearest(errors, 0.9),
        "anchor_rel_median": statistics.median(relative) if relative else None,
        "n_pairs_err_gt300": sum(error > 300 for error in errors),
    }


def render_labels(pixel_sums, scroll_acc, interval_events):
    labels = []
    for index, events in enumerate(interval_events):
        dx = round(pixel_sums[index][0])
        dy = round(pixel_sums[index][1])
        scroll = scroll_acc[index]
        ordered = sorted(events, key=lambda event: (event[0], event[1]))
        tokens = [("+" if kind == "press" else "-") + key for _, _, kind, key in ordered]
        if dx == 0 and dy == 0 and scroll == 0 and not tokens:
            labels.append("NO_OP")
        else:
            label = f"{dx} {dy} {scroll}"
            if tokens:
                label += " ; " + " ".join(tokens)
            labels.append(label)
    return labels


def make_record(app, task_id, action_log, meta, frames_dir, labels):
    instruction = action_log.get("task_instruction")
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSPROMPT}]}]
    for index, label in enumerate(labels):
        content = [{
            "type": "image",
            "image": os.path.abspath(os.path.join(frames_dir, "bc_frames", f"frame_{index:06d}.jpg")),
        }]
        if index == 0:
            content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})
    return {
        "sample_id": f"videocua_{app}_{task_id}",
        "recording_id": f"{app}/{task_id}",
        "app": app,
        "platform": action_log.get("platform"),
        "instruction": instruction,
        "n_frames": integer(meta.get("n_bc_frames")),
        "duration_s": meta.get("duration"),
        "messages": messages,
    }


def base_stats(task, app, n_actions, n_bc_frames, counts, skip_reason=None):
    row = {
        "task": task,
        "app": app,
        "n_actions": n_actions,
        "n_bc_frames": n_bc_frames,
        "n_anchor_pairs": 0,
        "anchor_err_mean": None,
        "anchor_err_median": None,
        "anchor_err_p90": None,
        "anchor_rel_median": None,
        "n_pairs_err_gt300": 0,
        "counts": counts,
        "skip_reason": skip_reason,
    }
    return row


def convert_task(app, task_id, action_path, frames_root, idm_root, key_map):
    task = f"{app}/{task_id}"
    counts = empty_counts()
    unmapped = []
    try:
        action_log = read_json(action_path)
    except (OSError, ValueError, TypeError):
        return None, base_stats(task, app, 0, 0, counts, "action_log_unreadable"), unmapped
    actions = action_log.get("action_log")
    if not isinstance(actions, list) or not actions:
        return None, base_stats(task, app, 0, 0, counts, "action_log_empty"), unmapped

    frames_dir = os.path.join(frames_root, app, task_id)
    meta_path = os.path.join(frames_dir, "extract_meta.json")
    try:
        meta = read_json(meta_path)
    except (OSError, ValueError, TypeError):
        return None, base_stats(task, app, len(actions), 0, counts, "no_bc_frames"), unmapped
    n_intervals = integer(meta.get("n_bc_frames"), 0)
    first_frame = os.path.join(frames_dir, "bc_frames", "frame_000000.jpg")
    if n_intervals <= 0 or not os.path.isfile(first_frame):
        return None, base_stats(task, app, len(actions), n_intervals, counts, "no_bc_frames"), unmapped
    width = number(meta.get("width"), 0.0)
    height = number(meta.get("height"), 0.0)
    if width <= 0 or height <= 0:
        return None, base_stats(task, app, len(actions), n_intervals, counts, "no_bc_frames"), unmapped

    idm_path = os.path.join(idm_root, app, f"{task_id}.idm.json")
    try:
        idm = read_json(idm_path)
    except (OSError, ValueError, TypeError):
        return None, base_stats(task, app, len(actions), n_intervals, counts, "no_idm"), unmapped
    if integer(idm.get("n_errors"), 0) == integer(idm.get("n_windows"), 0):
        return None, base_stats(task, app, len(actions), n_intervals, counts, "no_idm"), unmapped

    pixel_sums, normalized_moves = parse_idm(idm, n_intervals, width, height, counts)
    interval_events, scroll_acc, anchors = convert_actions(
        task, action_log, key_map, n_intervals, width, height, counts, unmapped
    )
    labels = render_labels(pixel_sums, scroll_acc, interval_events)
    if not labels:
        return None, base_stats(task, app, len(actions), n_intervals, counts, "zero_assistant_intervals"), unmapped
    record = make_record(app, task_id, action_log, meta, frames_dir, labels)
    stats = base_stats(task, app, len(actions), n_intervals, counts)
    stats.update(anchor_stats(anchors, normalized_moves, width, height))
    return record, stats, unmapped


def discover_tasks(data_dir):
    tasks = []
    for app_dir in sorted(Path(data_dir).iterdir(), key=lambda path: path.name):
        if not app_dir.is_dir():
            continue
        for action_path in sorted(app_dir.glob("*/action_log.json"), key=lambda path: path.parent.name):
            tasks.append((app_dir.name, action_path.parent.name, str(action_path)))
    return tasks


def validate_args(args):
    if args.num_shards <= 0:
        raise SystemExit("--num_shards must be positive")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < num_shards")


def run_build(args):
    validate_args(args)
    key_map = read_json(args.key_map)
    tasks = discover_tasks(args.data_dir)
    tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_idx]
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"{args.shard_idx:03d}.jsonl"
    sample_path = os.path.join(args.out_dir, f"samples_shard_{suffix}")
    stats_path = os.path.join(args.out_dir, f"stats_shard_{suffix}")
    unmapped_path = os.path.join(args.out_dir, f"unmapped_shard_{suffix}")
    converted = 0
    skipped = 0
    with (
        open(sample_path, "w", encoding="utf-8") as sample_handle,
        open(stats_path, "w", encoding="utf-8") as stats_handle,
        open(unmapped_path, "w", encoding="utf-8") as unmapped_handle,
    ):
        for index, (app, task_id, action_path) in enumerate(tasks, 1):
            record, stats, unmapped = convert_task(
                app, task_id, action_path, args.frames_root, args.idm_root, key_map
            )
            if record is not None:
                write_jsonl_row(sample_handle, record)
                converted += 1
            else:
                skipped += 1
            write_jsonl_row(stats_handle, stats)
            for row in unmapped:
                write_jsonl_row(unmapped_handle, row)
            if index % 100 == 0:
                print(f"processed {index}/{len(tasks)} tasks", flush=True)
    print(
        f"shard {args.shard_idx}/{args.num_shards}: tasks={len(tasks)} converted={converted} skipped={skipped}",
        flush=True,
    )


def stub_key_map():
    return {
        "aliases": {
            "ctrl": "ControlLeft",
            "control": "ControlLeft",
            "shift": "ShiftLeft",
            "left arrow": "LeftArrow",
            ">": ["ShiftLeft", "Dot"],
        },
        "modifiers": {
            "ctrl": "ControlLeft",
            "control": "ControlLeft",
            "shift": "ShiftLeft",
        },
        "chars": {
            "!": ["ShiftLeft", "Num1"],
        },
    }


def run_self_test():
    key_map = stub_key_map()
    action_log = {
        "task_id": "synthetic",
        "task_instruction": "Exercise dense actions",
        "platform": "linux",
        "action_log": [
            {"timestamp": 0.05, "action_type": "CLICK", "action_params": {"x": 10, "y": 10, "text": "Left", "numClicks": 2}},
            {"timestamp": 0.10, "action_type": "TYPING", "action_params": {"text": "Hi!"}},
            {"timestamp": 0.20, "action_type": "HOTKEY", "action_params": {"keys": "Ctrl+Shift+>"}},
            {"timestamp": 0.30, "action_type": "MOUSE_DOWN", "action_params": {"text": "left"}},
            {"timestamp": 0.70, "action_type": "DRAG_TO", "action_params": {"x": 30, "y": 20}},
            {"timestamp": 1.05, "action_type": "MOUSE_UP", "action_params": {"text": "left"}},
            {"timestamp": 1.10, "action_type": "SCROLL", "action_params": {"scrollY": 2}},
            {"timestamp": 1.20, "action_type": "SCROLL", "action_params": {"scrollY": -1}},
        ],
    }
    idm = {
        "n_windows": 1,
        "n_errors": 0,
        "chunks": [{
            "start_frame": 0,
            "predictions": [
                {"frame": "F00", "type": "MouseMove", "details": "100, -100"},
                {"frame": "F01", "type": "MouseMove", "details": " 50, 50 "},
                {"frame": "F02", "type": "MouseMove", "details": "25,0"},
            ],
        }],
    }
    counts = empty_counts()
    pixel_sums, _ = parse_idm(idm, 4, 200, 100, counts)
    unmapped = []
    interval_events, scroll_acc, _ = convert_actions(
        "Test/synthetic", action_log, key_map, 4, 200, 100, counts, unmapped
    )
    labels = render_labels(pixel_sums, scroll_acc, interval_events)
    expected = [
        "35 -5 0 ; +LMB -LMB +LMB -LMB +ShiftLeft +KeyH -KeyH -ShiftLeft +KeyI -KeyI +ShiftLeft +Num1 -Num1 -ShiftLeft +ControlLeft +ShiftLeft +Dot -Dot -ShiftLeft -ControlLeft +LMB",
        "NO_OP",
        "0 0 1 ; -LMB",
        "NO_OP",
    ]
    assert labels == expected, f"labels mismatch:\nactual={labels!r}\nexpected={expected!r}"
    assert counts["clicks"] == 2
    assert counts["typing_chars"] == 3
    assert counts["hotkey_n"] == 1
    assert counts["drag_n"] == 1
    assert counts["scroll_events"] == 2
    assert counts["idm_move_frames"] == 3
    assert not unmapped
    print("self_test passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--frames_root")
    parser.add_argument("--idm_root")
    parser.add_argument("--out_dir")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--key_map")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        missing = [name for name in ("data_dir", "frames_root", "idm_root", "out_dir", "key_map") if getattr(args, name) is None]
        if missing:
            parser.error("the following arguments are required unless --self_test is used: " + ", ".join(f"--{name}" for name in missing))
    return args


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_build(args)


if __name__ == "__main__":
    main()
