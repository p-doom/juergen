#!/usr/bin/env python3
"""Build crowd-cast dense-action chat records from PSAI parquet shards."""

import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


SYSPROMPT = "You operate a desktop computer. The first user turn shows the initial screen and the user's goal; subsequent user turns show the current screen. Reply with the next action toward that goal as `<dx> <dy> <scroll>` optionally followed by ` ; +KEY -KEY` events, or `NO_OP` if no action."

COUNT_KEYS = (
    "n_batched_events",
    "n_batches",
    "max_batch_len",
    "debatch_window_max_s",
    "n_ts_regressions",
    "scroll_x_dropped",
    "n_scroll_flipped_rows",
    "forced_release_buttons",
    "forced_release_keys",
    "orphan_release",
    "click_pos_jump",
    "unmapped_keys",
    "unmapped_buttons",
    "n_app_opened",
    "n_window_focus",
    "n_modifier_change_events",
    "n_modifier_release_events",
    "n_dom_snapshot",
    "n_pause",
    "n_unknown_actions",
)

PARQUET_COLUMNS = (
    "unique_data_id",
    "taskId",
    "task_name",
    "category",
    "subCategory",
    "application_website",
    "os",
    "events",
    "metadata",
    "video_file",
)

MODIFIER_KEY_NAMES = {
    "shift",
    "shift_l",
    "shift_r",
    "ctrl",
    "ctrl_l",
    "ctrl_r",
    "alt",
    "alt_l",
    "alt_r",
    "alt_gr",
    "cmd",
    "cmd_l",
    "cmd_r",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_row(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def empty_counts():
    counts = {key: 0 for key in COUNT_KEYS}
    counts["debatch_window_max_s"] = 0.0
    return counts


def number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def text_value(value):
    return "" if value is None else str(value)


def task_fields(row):
    unique_data_id = text_value(row.get("unique_data_id"))
    task_id = text_value(row.get("taskId"))
    app = text_value(row.get("application_website"))
    task = f"{app}/{task_id}"
    return unique_data_id, task_id, app, task


def clean_instruction(value):
    instruction = text_value(value).strip()
    if len(instruction) >= 2 and instruction[0] == instruction[-1] and instruction[0] in {'"', "'"}:
        instruction = instruction[1:-1]
    return instruction


def raw_events_text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_value(value):
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def first_list_number(mapping, key):
    values = mapping.get(key)
    if not isinstance(values, list) or not values:
        return None
    return number(values[0])


def event_counts(events):
    actions = Counter(text_value(event.get("action")) for event in events if isinstance(event, dict))
    return {
        "n_events": len(events),
        "n_moves": actions["move"],
        "n_clicks": actions["click"],
        "n_scrolls": actions["scroll"],
        "n_press": actions["press"],
        "n_release": actions["release"],
    }


def recorder_version(events):
    actions = Counter(text_value(event.get("action")) for event in events if isinstance(event, dict))
    version_b = (actions["release"] > 0 and actions["press"] == 0) or actions["modifier_change"] > 0
    return "b" if version_b else "a"


def base_stats(row, events_sha1, counts, parsed_events=None, skip_reason=None):
    unique_data_id, _, app, task = task_fields(row)
    action_counts = event_counts(parsed_events or [])
    return {
        "task": task,
        "sample_id": f"psai_{unique_data_id}",
        "app": app,
        **action_counts,
        "recorder_version": recorder_version(parsed_events or []),
        "n_bc_frames": 0,
        "duration_s": None,
        "events_sha1": events_sha1,
        "skip_reason": skip_reason,
        "first_event_rel_s": None,
        "last_event_rel_s": None,
        "obs_stop_rel_s": None,
        "screen_w": None,
        "screen_h": None,
        "video_w": None,
        "video_h": None,
        "aspect_mismatch": False,
        "screen_dims_missing": False,
        "move_px_total": 0.0,
        "intervals_nonzero_frac": 0.0,
        "counts": counts,
    }


def append_unmapped(unmapped, task, action_type, raw_value, context):
    unmapped.append({
        "task": task,
        "action_type": action_type,
        "raw_value": raw_value,
        "context": context,
    })


def resolve_key(raw_value, key_map):
    if raw_value is None:
        return None
    value = str(raw_value)
    for section in ("named", "chars", "control_chars"):
        resolved = key_map.get(section, {}).get(value)
        if resolved is not None:
            return resolved
    return None


def resolve_named_key(raw_value, key_map):
    if raw_value is None:
        return None
    return key_map.get("named", {}).get(str(raw_value))


def resolve_button(raw_value, key_map):
    if raw_value is None:
        return None
    return key_map.get("buttons", {}).get(str(raw_value))


def bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "down", "pressed", "press"}:
            return True
        if lowered in {"false", "0", "no", "up", "released", "release"}:
            return False
    return bool(value)


def interval_index(video_time, n_intervals):
    if video_time < 0:
        return 0
    if video_time >= n_intervals * 0.5:
        return n_intervals - 1
    return min(max(math.floor(video_time * 2.0), 0), n_intervals - 1)


def prepare_events(events, counts):
    prepared = []
    running_max = None
    previous_raw_timestamp = None
    for original_index, event in enumerate(events):
        copied = dict(event)
        timestamp = number(copied.get("time_stamp"), 0.0)
        if previous_raw_timestamp is not None and timestamp < previous_raw_timestamp:
            counts["n_ts_regressions"] += 1
        previous_raw_timestamp = timestamp
        if running_max is not None and timestamp < running_max:
            timestamp = running_max
        running_max = timestamp if running_max is None else max(running_max, timestamp)
        copied["_timestamp"] = timestamp
        copied["_original_index"] = original_index
        prepared.append(copied)

    index = 0
    while index < len(prepared):
        end = index + 1
        run_timestamp = prepared[index]["_timestamp"]
        while end < len(prepared) and prepared[end]["_timestamp"] == run_timestamp:
            end += 1
        run_length = end - index
        if run_length > 1:
            counts["n_batches"] += 1
            counts["n_batched_events"] += run_length
            counts["max_batch_len"] = max(counts["max_batch_len"], run_length)
            previous_timestamp = prepared[index - 1]["_timestamp"] if index > 0 else None
            if previous_timestamp is None:
                window = min(run_length / 8.0, 4.0)
            else:
                window = min(run_timestamp - previous_timestamp, run_length / 8.0, 4.0)
            counts["debatch_window_max_s"] = max(counts["debatch_window_max_s"], window)
            for offset in range(run_length):
                shift = (run_length - 1 - offset) / run_length * window
                prepared[index + offset]["_timestamp"] = run_timestamp - shift
        index = end

    running_max = None
    for event in prepared:
        if running_max is not None and event["_timestamp"] < running_max:
            event["_timestamp"] = running_max
        running_max = event["_timestamp"]
    return prepared


def render_labels(move_sums, scroll_acc, interval_events):
    labels = []
    for index in range(len(move_sums)):
        dx = round(move_sums[index][0])
        dy = round(move_sums[index][1])
        scroll = scroll_acc[index]
        tokens = [token for _, _, token in sorted(interval_events[index], key=lambda item: (item[0], item[1]))]
        if dx == 0 and dy == 0 and scroll == 0 and not tokens:
            labels.append("NO_OP")
            continue
        label = f"{dx} {dy} {scroll}"
        if tokens:
            label += " ; " + " ".join(tokens)
        labels.append(label)
    return labels


def convert_labels(
    task, events, anchor, n_intervals, key_map, counts, unmapped, version_b=False, scroll_flipped=False
):
    prepared = prepare_events(events, counts)
    move_sums = [[0.0, 0.0] for _ in range(n_intervals)]
    scroll_acc = [0 for _ in range(n_intervals)]
    interval_events = [[] for _ in range(n_intervals)]
    open_buttons = Counter()
    open_keys = Counter()
    last_move = None
    move_px_total = 0.0
    last_timestamp = prepared[-1]["_timestamp"]
    forced_order = len(prepared)

    def emit(event, token):
        video_time = event["_timestamp"] - anchor
        target = interval_index(video_time, n_intervals)
        interval_events[target].append((video_time, event["_original_index"], token))

    for event in prepared:
        action = text_value(event.get("action"))
        video_time = event["_timestamp"] - anchor
        target = interval_index(video_time, n_intervals)

        if action == "move":
            x = number(event.get("x"))
            y = number(event.get("y"))
            if x is None or y is None:
                continue
            if last_move is not None:
                dx = x - last_move[0]
                dy = y - last_move[1]
                move_sums[target][0] += dx
                move_sums[target][1] += dy
                move_px_total += abs(dx) + abs(dy)
            last_move = (x, y)
            continue

        if action == "scroll":
            dx = number(event.get("dx"), 0.0)
            if dx != 0:
                counts["scroll_x_dropped"] += 1
            scroll_dy = integer(event.get("dy"), 0)
            scroll_acc[target] += -scroll_dy if scroll_flipped else scroll_dy
            continue

        if action == "click":
            x = number(event.get("x"))
            y = number(event.get("y"))
            if last_move is not None and x is not None and y is not None:
                if math.hypot(x - last_move[0], y - last_move[1]) > 2.0:
                    counts["click_pos_jump"] += 1
            raw_button = event.get("button")
            button = resolve_button(raw_button, key_map)
            if button is None:
                counts["unmapped_buttons"] += 1
                append_unmapped(unmapped, task, action, raw_button, "button")
                continue
            if bool_value(event.get("pressed")):
                open_buttons[button] += 1
                emit(event, f"+{button}")
            elif open_buttons[button] > 0:
                open_buttons[button] -= 1
                emit(event, f"-{button}")
            else:
                counts["orphan_release"] += 1
            continue

        if action in {"press", "release"}:
            raw_key = event.get("name")
            if action == "release" and text_value(raw_key).lower() in MODIFIER_KEY_NAMES:
                counts["n_modifier_release_events"] += 1
            key = resolve_key(raw_key, key_map)
            if key is None:
                counts["unmapped_keys"] += 1
                append_unmapped(unmapped, task, action, raw_key, f"key_{action}")
                continue
            if version_b and action == "release":
                emit(event, f"+{key}")
                emit(event, f"-{key}")
            elif action == "press":
                open_keys[key] += 1
                emit(event, f"+{key}")
            elif open_keys[key] > 0:
                open_keys[key] -= 1
                emit(event, f"-{key}")
            else:
                counts["orphan_release"] += 1
            continue

        if action == "modifier_change":
            counts["n_modifier_change_events"] += 1
            state = text_value(event.get("state")).strip().lower()
            raw_key = event.get("name")
            key = resolve_named_key(raw_key, key_map)
            if key is None:
                counts["unmapped_keys"] += 1
                append_unmapped(unmapped, task, action, raw_key, "key_modifier_change")
                continue
            if state == "pressed":
                open_keys[key] += 1
                emit(event, f"+{key}")
            elif state == "released" and open_keys[key] > 0:
                open_keys[key] -= 1
                emit(event, f"-{key}")
            elif state == "released":
                counts["orphan_release"] += 1
            continue

        if action == "app_opened":
            counts["n_app_opened"] += 1
        elif action == "window_focus":
            counts["n_window_focus"] += 1
        elif action == "dom_snapshot":
            counts["n_dom_snapshot"] += 1
        elif action == "pause":
            counts["n_pause"] += 1
        else:
            counts["n_unknown_actions"] += 1
            append_unmapped(unmapped, task, action, action, "action_type")

    last_video_time = last_timestamp - anchor
    last_target = interval_index(last_video_time, n_intervals)
    for button, count in open_buttons.items():
        for _ in range(count):
            interval_events[last_target].append((last_video_time, forced_order, f"-{button}"))
            forced_order += 1
            counts["forced_release_buttons"] += 1
    for key, count in open_keys.items():
        for _ in range(count):
            interval_events[last_target].append((last_video_time, forced_order, f"-{key}"))
            forced_order += 1
            counts["forced_release_keys"] += 1

    return render_labels(move_sums, scroll_acc, interval_events), move_px_total, prepared


def make_record(row, frames_dir, meta, labels):
    unique_data_id, task_id, app, task = task_fields(row)
    instruction = clean_instruction(row.get("task_name"))
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
        "sample_id": f"psai_{unique_data_id}",
        "recording_id": task,
        "app": app,
        "platform": row.get("os"),
        "instruction": instruction,
        "unique_data_id": row.get("unique_data_id"),
        "task_template_id": row.get("taskId"),
        "category": row.get("category"),
        "n_frames": integer(meta.get("n_bc_frames")),
        "duration_s": meta.get("duration"),
        "messages": messages,
    }


def convert_row(work_item):
    row, frames_root, key_map, duplicate = work_item
    unique_data_id, _, _, task = task_fields(row)
    counts = empty_counts()
    unmapped = []
    events_raw = raw_events_text(row.get("events"))
    events_sha1 = hashlib.sha1(events_raw.encode("utf-8")).hexdigest()

    try:
        events = parse_json_value(row.get("events"))
    except (TypeError, ValueError, json.JSONDecodeError):
        events = None
    if duplicate:
        parsed_events = events if isinstance(events, list) else []
        return None, base_stats(row, events_sha1, counts, parsed_events, "dup_in_shard"), unmapped
    if not isinstance(events, list) or not events or not all(isinstance(event, dict) for event in events):
        return None, base_stats(row, events_sha1, counts, skip_reason="events_unreadable"), unmapped

    stats = base_stats(row, events_sha1, counts, events)
    try:
        metadata = parse_json_value(row.get("metadata"))
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    screen_w = number(metadata.get("screen_width"))
    screen_h = number(metadata.get("screen_height"))
    stats["screen_w"] = screen_w
    stats["screen_h"] = screen_h
    stats["screen_dims_missing"] = screen_w is None or screen_h is None
    scroll_flipped = integer(metadata.get("scroll_direction"), 1) == -1
    counts["n_scroll_flipped_rows"] = int(scroll_flipped)

    timings = metadata.get("obs_record_state_timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    anchor = first_list_number(timings, "OBS_WEBSOCKET_OUTPUT_STARTED")
    if anchor is None:
        stats["skip_reason"] = "no_obs_anchor"
        return None, stats, unmapped
    stopping = first_list_number(timings, "OBS_WEBSOCKET_OUTPUT_STOPPING")
    stats["obs_stop_rel_s"] = None if stopping is None else stopping - anchor

    frames_dir = os.path.join(frames_root, unique_data_id)
    meta_path = os.path.join(frames_dir, "extract_meta.json")
    try:
        meta = read_json(meta_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        stats["skip_reason"] = "no_frames"
        return None, stats, unmapped
    if not isinstance(meta, dict):
        stats["skip_reason"] = "no_frames"
        return None, stats, unmapped

    n_intervals = integer(meta.get("n_bc_frames"), 0)
    stats["n_bc_frames"] = n_intervals
    stats["duration_s"] = meta.get("duration")
    stats["video_w"] = number(meta.get("width"))
    stats["video_h"] = number(meta.get("height"))
    if n_intervals <= 0:
        stats["skip_reason"] = "zero_intervals"
        return None, stats, unmapped

    video_w = stats["video_w"]
    video_h = stats["video_h"]
    if (
        screen_w is not None
        and screen_h not in (None, 0)
        and video_w is not None
        and video_h not in (None, 0)
    ):
        screen_aspect = screen_w / screen_h
        video_aspect = video_w / video_h
        stats["aspect_mismatch"] = abs(screen_aspect - video_aspect) / abs(video_aspect) > 0.02

    version_b = stats["recorder_version"] == "b"
    labels, move_px_total, prepared = convert_labels(
        task, events, anchor, n_intervals, key_map, counts, unmapped, version_b, scroll_flipped
    )
    stats["first_event_rel_s"] = prepared[0]["_timestamp"] - anchor
    stats["last_event_rel_s"] = prepared[-1]["_timestamp"] - anchor
    stats["move_px_total"] = move_px_total
    stats["intervals_nonzero_frac"] = sum(label != "NO_OP" for label in labels) / n_intervals
    record = make_record(row, frames_dir, meta, labels)
    return record, stats, unmapped


def load_shard_rows(parquet_glob, shard_idx, num_shards):
    import pyarrow.parquet as pq

    selected = []
    global_index = 0
    for parquet_path in sorted(glob.glob(parquet_glob)):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=list(PARQUET_COLUMNS)):
            columns = batch.to_pydict()
            for row_index in range(batch.num_rows):
                if global_index % num_shards == shard_idx:
                    selected.append({name: columns[name][row_index] for name in PARQUET_COLUMNS})
                global_index += 1
    return selected


def validate_args(args):
    if args.num_shards <= 0:
        raise SystemExit("--num_shards must be positive")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < num_shards")
    if args.num_workers <= 0:
        raise SystemExit("--num_workers must be positive")


def run_build(args):
    validate_args(args)
    key_map = read_json(args.key_map)
    rows = load_shard_rows(args.parquet_glob, args.shard_idx, args.num_shards)
    seen = set()
    work_items = []
    for row in rows:
        unique_data_id = text_value(row.get("unique_data_id"))
        duplicate = unique_data_id in seen
        seen.add(unique_data_id)
        work_items.append((row, args.frames_root, key_map, duplicate))

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"{args.shard_idx:03d}.jsonl"
    sample_path = os.path.join(args.out_dir, f"samples_shard_{suffix}")
    stats_path = os.path.join(args.out_dir, f"stats_shard_{suffix}")
    unmapped_path = os.path.join(args.out_dir, f"unmapped_shard_{suffix}")
    converted = 0
    skipped = 0
    version_counts = Counter()
    with (
        open(sample_path, "w", encoding="utf-8") as sample_handle,
        open(stats_path, "w", encoding="utf-8") as stats_handle,
        open(unmapped_path, "w", encoding="utf-8") as unmapped_handle,
        ProcessPoolExecutor(max_workers=args.num_workers) as executor,
    ):
        for index, (record, stats, unmapped) in enumerate(executor.map(convert_row, work_items), 1):
            version_counts[stats["recorder_version"]] += 1
            if record is None:
                skipped += 1
            else:
                write_jsonl_row(sample_handle, record)
                converted += 1
            write_jsonl_row(stats_handle, stats)
            for unmapped_row in unmapped:
                write_jsonl_row(unmapped_handle, unmapped_row)
            if index % 100 == 0:
                print(f"processed {index}/{len(work_items)} tasks", flush=True)
    print(
        f"shard {args.shard_idx}/{args.num_shards}: tasks={len(work_items)} converted={converted} "
        f"skipped={skipped} version_a={version_counts['a']} version_b={version_counts['b']}",
        flush=True,
    )


def run_self_test():
    key_map = {
        "named": {"a": "KeyA", "shift": "ShiftLeft"},
        "chars": {"h": "KeyH", "i": "KeyI", "!": "Num1"},
        "control_chars": {"\x16": "KeyV"},
        "buttons": {"left": "LMB"},
    }
    events = [
        {"time_stamp": 100.1, "action": "move", "x": 10, "y": 10},
        {"time_stamp": 100.2, "action": "move", "x": 30, "y": 25},
        {"time_stamp": 100.5, "action": "app_opened"},
        {"time_stamp": 100.55, "action": "move", "x": 40, "y": 25},
        {"time_stamp": 100.55, "action": "move", "x": 50, "y": 25},
        {"time_stamp": 100.60, "action": "click", "button": "left", "pressed": True, "x": 50, "y": 25},
        {"time_stamp": 100.61, "action": "click", "button": "left", "pressed": False, "x": 50, "y": 25},
        {"time_stamp": 100.7, "action": "move", "x": 60, "y": 25},
        {"time_stamp": 101.1, "action": "press", "name": "a"},
        {"time_stamp": 101.2, "action": "release", "name": "a"},
        {"time_stamp": 101.3, "action": "press", "name": "\x16"},
        {"time_stamp": 101.4, "action": "scroll", "dx": 0, "dy": -2},
    ]
    with tempfile.TemporaryDirectory() as temporary_directory:
        unique_data_id = "synthetic"
        frames_dir = os.path.join(temporary_directory, unique_data_id)
        bc_frames_dir = os.path.join(frames_dir, "bc_frames")
        os.makedirs(bc_frames_dir)
        for index in range(3):
            with open(os.path.join(bc_frames_dir, f"frame_{index:06d}.jpg"), "wb") as handle:
                handle.write(b"fake")
        with open(os.path.join(frames_dir, "extract_meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"n_bc_frames": 3, "duration": 1.5, "width": 1000, "height": 500}, handle)
        row = {
            "unique_data_id": unique_data_id,
            "taskId": "task-1",
            "task_name": '"Synthetic task"',
            "category": "COMPUTER_TASK",
            "subCategory": [],
            "application_website": "synthetic_app",
            "os": "LINUX",
            "events": json.dumps(events),
            "metadata": json.dumps({
                "screen_width": 1000,
                "screen_height": 500,
                "obs_record_state_timings": {
                    "OBS_WEBSOCKET_OUTPUT_STARTED": [100.0],
                    "OBS_WEBSOCKET_OUTPUT_STOPPING": [101.5],
                },
            }),
            "video_file": "synthetic.mp4",
        }
        record, stats, unmapped = convert_row((row, temporary_directory, key_map, False))
    labels = [
        message["content"][0]["text"]
        for message in record["messages"]
        if message["role"] == "assistant"
    ]
    expected = [
        "20 15 0",
        "30 0 0 ; +LMB -LMB",
        "0 0 -2 ; +KeyA -KeyA +KeyV -KeyV",
    ]
    assert labels == expected, f"labels mismatch:\nactual={labels!r}\nexpected={expected!r}"
    assert stats["counts"]["n_batched_events"] == 2
    assert stats["counts"]["forced_release_keys"] == 1
    assert not unmapped

    version_b_events = [
        {"time_stamp": 100.9, "action": "modifier_change", "name": "shift", "state": "pressed"},
        {"time_stamp": 101.0, "action": "release", "name": "h"},
        {"time_stamp": 101.5, "action": "release", "name": "i"},
        {"time_stamp": 101.5, "action": "release", "name": "!"},
        {"time_stamp": 101.55, "action": "scroll", "dx": 0, "dy": 3},
        {"time_stamp": 101.6, "action": "modifier_change", "name": "shift", "state": "released"},
    ]
    with tempfile.TemporaryDirectory() as temporary_directory:
        unique_data_id = "synthetic-version-b"
        frames_dir = os.path.join(temporary_directory, unique_data_id)
        bc_frames_dir = os.path.join(frames_dir, "bc_frames")
        os.makedirs(bc_frames_dir)
        for index in range(4):
            with open(os.path.join(bc_frames_dir, f"frame_{index:06d}.jpg"), "wb") as handle:
                handle.write(b"fake")
        with open(os.path.join(frames_dir, "extract_meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"n_bc_frames": 4, "duration": 2.0, "width": 1000, "height": 500}, handle)
        row = {
            "unique_data_id": unique_data_id,
            "taskId": "task-version-b",
            "task_name": "Synthetic version B task",
            "category": "COMPUTER_TASK",
            "subCategory": [],
            "application_website": "synthetic_app",
            "os": "DARWIN",
            "events": json.dumps(version_b_events),
            "metadata": json.dumps({
                "screen_width": 1000,
                "screen_height": 500,
                "scroll_direction": -1,
                "obs_record_state_timings": {
                    "OBS_WEBSOCKET_OUTPUT_STARTED": [100.0],
                    "OBS_WEBSOCKET_OUTPUT_STOPPING": [102.0],
                },
            }),
            "video_file": "synthetic-version-b.mp4",
        }
        record, stats, unmapped = convert_row((row, temporary_directory, key_map, False))
    labels = [
        message["content"][0]["text"]
        for message in record["messages"]
        if message["role"] == "assistant"
    ]
    expected = [
        "NO_OP",
        "0 0 0 ; +ShiftLeft",
        "0 0 0 ; +KeyH -KeyH +KeyI -KeyI",
        "0 0 -3 ; +Num1 -Num1 -ShiftLeft",
    ]
    assert labels == expected, f"version B labels mismatch:\nactual={labels!r}\nexpected={expected!r}"
    assert stats["recorder_version"] == "b"
    assert stats["counts"]["n_batched_events"] == 2
    assert stats["counts"]["debatch_window_max_s"] == 0.25
    assert stats["counts"]["n_modifier_change_events"] == 2
    assert stats["counts"]["n_scroll_flipped_rows"] == 1
    assert not unmapped
    print("self_test passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_glob")
    parser.add_argument("--frames_root")
    parser.add_argument("--key_map")
    parser.add_argument("--out_dir")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        required = ("parquet_glob", "frames_root", "key_map", "out_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(
                "the following arguments are required unless --self_test is used: "
                + ", ".join(f"--{name}" for name in missing)
            )
    return args


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_build(args)


if __name__ == "__main__":
    main()
