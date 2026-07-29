#!/usr/bin/env python3
"""Assemble converter shards into the split PSAI v1 corpus."""

import argparse
import copy
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


SYSPROMPT = (
    "You operate a desktop computer. The first user turn shows the initial screen and the "
    "user's goal; subsequent user turns show the current screen. Reply with the next action "
    "toward that goal as `<dx> <dy> <scroll>` optionally followed by ` ; +KEY -KEY` events, "
    "or `NO_OP` if no action."
)
JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def read_jsonl_files(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def write_jsonl(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def as_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values, fields):
    if not values:
        return {field: None for field in fields}
    result = {}
    for field in fields:
        if field == "min":
            result[field] = min(values)
        elif field == "max":
            result[field] = max(values)
        elif field == "mean":
            result[field] = statistics.fmean(values)
        elif field == "median":
            result[field] = statistics.median(values)
        elif field.startswith("p") and field[1:].isdigit():
            result[field] = percentile(values, int(field[1:]))
        else:
            raise ValueError(f"unsupported distribution field: {field}")
    return result


def exclusion_reasons(stats, min_frames):
    reasons = []
    if stats.get("skip_reason") is not None:
        reasons.append("skip_reason")
    if as_int(stats.get("n_bc_frames")) < min_frames:
        reasons.append("n_bc_frames_lt_min")
    if stats.get("aspect_mismatch") is True:
        reasons.append("aspect_mismatch")
    first_event_rel_s = as_float(stats.get("first_event_rel_s"))
    if first_event_rel_s is not None and first_event_rel_s < -1.0:
        reasons.append("first_event_rel_s_lt_neg1")
    obs_stop_rel_s = as_float(stats.get("obs_stop_rel_s"))
    last_event_rel_s = as_float(stats.get("last_event_rel_s"))
    if (
        obs_stop_rel_s is not None
        and last_event_rel_s is not None
        and last_event_rel_s > obs_stop_rel_s + 1.0
    ):
        reasons.append("last_event_rel_s_gt_obs_stop_plus1")
    if stats.get("screen_dims_missing") is True:
        reasons.append("screen_dims_missing")
    return reasons


def jpeg_dimensions(path):
    try:
        with open(path, "rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None
            while True:
                byte = handle.read(1)
                if not byte:
                    return None
                if byte != b"\xff":
                    continue
                while byte == b"\xff":
                    byte = handle.read(1)
                if not byte:
                    return None
                marker = byte[0]
                if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7 or marker == 0x01:
                    continue
                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    return None
                segment_length = int.from_bytes(length_bytes, "big")
                if segment_length < 2:
                    return None
                if marker in JPEG_SOF_MARKERS:
                    data = handle.read(5)
                    if len(data) != 5:
                        return None
                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return (width, height) if width > 0 and height > 0 else None
                handle.seek(segment_length - 2, os.SEEK_CUR)
    except (OSError, ValueError):
        return None


def content_items(message):
    content = message.get("content", []) if isinstance(message, dict) else []
    if isinstance(content, list):
        return content
    return []


def item_text(item):
    if isinstance(item, dict) and item.get("type") == "text":
        value = item.get("text")
        return value if isinstance(value, str) else str(value or "")
    return ""


def assistant_text(message):
    return "".join(item_text(item) for item in content_items(message))


def first_image_item(message):
    for item in content_items(message):
        if isinstance(item, dict) and item.get("type") == "image":
            return item
    raise ValueError("user turn has no image item")


def image_path(item):
    value = item.get("image") if isinstance(item, dict) else None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "url"):
            if isinstance(value.get(key), str):
                return value[key]
    return None


def record_pairs(record):
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError(f"record {record.get('sample_id')!r} lacks a leading system turn")
    remaining = messages[1:]
    if len(remaining) % 2:
        raise ValueError(f"record {record.get('sample_id')!r} has unpaired turns")
    pairs = []
    for index in range(0, len(remaining), 2):
        user = remaining[index]
        assistant = remaining[index + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError(f"record {record.get('sample_id')!r} has non-alternating turns")
        first_image_item(user)
        pairs.append((user, assistant))
    return messages[0], pairs


def frame_token_cost(record):
    _, pairs = record_pairs(record)
    dimensions = None
    if pairs:
        dimensions = jpeg_dimensions(image_path(first_image_item(pairs[0][0])) or "")
    original_width, original_height = dimensions or (960, 540)
    resized_height = 540
    resized_width = max(2, round(original_width * resized_height / original_height / 2) * 2)
    return math.ceil(resized_height / 32) * math.ceil(resized_width / 32) + 8


def split_windows(pair_costs, overhead, budget, min_subrecord_frames):
    windows = []
    start = 0
    current_cost = overhead
    for index, pair_cost in enumerate(pair_costs):
        if index > start and current_cost + pair_cost > budget:
            windows.append((start, index))
            start = index
            current_cost = overhead
        current_cost += pair_cost
    if start < len(pair_costs):
        windows.append((start, len(pair_costs)))
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < min_subrecord_frames:
        previous_start, _ = windows[-2]
        windows[-2:] = [(previous_start, windows[-1][1])]
    return windows


def make_user_turn(image_item, instruction=None):
    content = [copy.deepcopy(image_item)]
    if instruction is not None:
        content.append({"type": "text", "text": instruction})
    return {"role": "user", "content": content}


def estimate_subrecord_tokens(frame_tokens, assistants, instruction):
    overhead = len(SYSPROMPT) // 3 + len(instruction) // 3 + 16
    return overhead + sum(frame_tokens + len(text) // 3 + 8 for text in assistants)


def split_record(record, split_token_budget, min_subrecord_frames):
    system, pairs = record_pairs(record)
    if not pairs:
        raise ValueError(f"record {record.get('sample_id')!r} has no user/assistant pairs")
    instruction_value = record.get("instruction")
    instruction = instruction_value if isinstance(instruction_value, str) else str(instruction_value or "")
    frame_tokens = frame_token_cost(record)
    assistant_texts = [assistant_text(assistant) for _, assistant in pairs]
    overhead = len(SYSPROMPT) // 3 + len(instruction) // 3 + 16
    pair_costs = [frame_tokens + len(text) // 3 + 8 for text in assistant_texts]
    windows = split_windows(pair_costs, overhead, split_token_budget, min_subrecord_frames)
    n_subrecords = len(windows)
    output = []
    for subrecord_idx, (start, end) in enumerate(windows):
        subrecord = {
            key: copy.deepcopy(record.get(key))
            for key in (
                "sample_id", "recording_id", "app", "platform", "instruction", "duration_s",
                "unique_data_id", "task_template_id", "category",
            )
            if key in record
        }
        original_sample_id = record.get("sample_id")
        if n_subrecords > 1:
            subrecord["sample_id"] = f"{original_sample_id}_c{subrecord_idx:02d}"
        subrecord["n_frames"] = end - start
        subrecord["subrecord_idx"] = subrecord_idx
        subrecord["n_subrecords"] = n_subrecords
        messages = [copy.deepcopy(system)]
        selected_assistants = []
        for local_index, (user, assistant) in enumerate(pairs[start:end]):
            messages.append(make_user_turn(first_image_item(user), instruction if local_index == 0 else None))
            messages.append(copy.deepcopy(assistant))
            selected_assistants.append(assistant_text(assistant))
        subrecord["messages"] = messages
        subrecord["est_tokens"] = estimate_subrecord_tokens(
            frame_tokens, selected_assistants, instruction
        )
        output.append(subrecord)
    return output


def summarize_unmapped(rows):
    counts = Counter()
    for row in rows:
        raw_value = row.get("raw_value")
        stable_raw = json.dumps(raw_value, ensure_ascii=False, sort_keys=True)
        context = str(row.get("context") or "")
        counts[(stable_raw, context)] += 1
    summary = []
    for (stable_raw, context), count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )[:40]:
        summary.append({"raw_value": json.loads(stable_raw), "context": context, "count": count})
    return summary


def sum_converter_counts(stats_rows):
    totals = defaultdict(int)
    for row in stats_rows:
        counts = row.get("counts")
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            totals[key] += as_int(value)
    return {key: totals[key] for key in sorted(totals)}


def debatch_summary(stats_rows):
    return {
        "n_batched_events": sum(
            as_int(row.get("counts", {}).get("n_batched_events"))
            for row in stats_rows if isinstance(row.get("counts"), dict)
        ),
        "n_batches": sum(
            as_int(row.get("counts", {}).get("n_batches"))
            for row in stats_rows if isinstance(row.get("counts"), dict)
        ),
        "max_batch_len": max(
            (as_int(row.get("counts", {}).get("max_batch_len"))
             for row in stats_rows if isinstance(row.get("counts"), dict)),
            default=0,
        ),
        "debatch_max_shift_s": max(
            (as_float(row.get("counts", {}).get("debatch_max_shift_s"), 0.0)
             for row in stats_rows if isinstance(row.get("counts"), dict)),
            default=0.0,
        ),
    }


def append_reason(excluded, sample_id, reasons):
    current = excluded.setdefault(sample_id, [])
    for reason in reasons:
        if reason not in current:
            current.append(reason)


def classify_stats(stats_rows, min_frames):
    seen_sample_ids = set()
    seen_events_sha1 = set()
    included_stats = []
    included_sample_counts = Counter()
    excluded = {}
    excluded_tasks = set()
    excluded_by = Counter()

    for row in stats_rows:
        sample_id = str(row.get("sample_id") or "")
        events_sha1 = str(row.get("events_sha1") or "")
        reasons = exclusion_reasons(row, min_frames)
        if sample_id and sample_id in seen_sample_ids:
            reasons.append("dup_sample_id")
        if events_sha1 and events_sha1 in seen_events_sha1:
            reasons.append("dup_events_sha1")
        if sample_id:
            seen_sample_ids.add(sample_id)
        if events_sha1:
            seen_events_sha1.add(events_sha1)

        if reasons:
            append_reason(excluded, sample_id, reasons)
            excluded_by.update(reasons)
            excluded_tasks.add(str(row.get("task") or row.get("recording_id") or sample_id))
        else:
            included_stats.append(row)
            included_sample_counts[sample_id] += 1

    return included_stats, included_sample_counts, excluded, excluded_tasks, excluded_by


def select_included_records(sample_rows, included_sample_counts):
    remaining = Counter(included_sample_counts)
    included_records = []
    for record in sample_rows:
        sample_id = str(record.get("sample_id") or "")
        if remaining[sample_id] > 0:
            included_records.append(record)
            remaining[sample_id] -= 1
    return included_records


def alignment_summary(included_stats):
    first_values = []
    stop_overrun_values = []
    for row in included_stats:
        first_event_rel_s = as_float(row.get("first_event_rel_s"))
        if first_event_rel_s is not None:
            first_values.append(first_event_rel_s)
        last_event_rel_s = as_float(row.get("last_event_rel_s"))
        obs_stop_rel_s = as_float(row.get("obs_stop_rel_s"))
        if last_event_rel_s is not None and obs_stop_rel_s is not None:
            stop_overrun_values.append(last_event_rel_s - obs_stop_rel_s)
    fields = ("mean", "median", "p90", "min", "max")
    return {
        "first_event_rel_s": distribution(first_values, fields),
        "last_event_rel_s_minus_obs_stop_rel_s": distribution(stop_overrun_values, fields),
    }


def assemble(args):
    conv_dir = Path(args.conv_dir)
    stats_rows = read_jsonl_files(sorted(conv_dir.glob("stats_shard_*.jsonl")))
    sample_rows = read_jsonl_files(sorted(conv_dir.glob("samples_shard_*.jsonl")))
    unmapped_rows = read_jsonl_files(sorted(conv_dir.glob("unmapped_shard_*.jsonl")))

    included_stats, included_sample_counts, excluded, excluded_tasks, excluded_by = classify_stats(
        stats_rows, args.min_frames
    )
    included_records = select_included_records(sample_rows, included_sample_counts)
    output_records = []
    subrecords_per_task = []
    for record in included_records:
        subrecords = split_record(record, args.split_token_budget, args.min_subrecord_frames)
        subrecords_per_task.append(len(subrecords))
        output_records.extend(subrecords)
    write_jsonl(args.out_jsonl, output_records)

    included_sample_ids = [str(record.get("sample_id") or "") for record in included_records]
    split_map = {
        "included_sample_ids": included_sample_ids,
        "excluded": {key: excluded[key] for key in sorted(excluded)},
    }
    write_json(args.split_map_out, split_map)

    per_app = Counter(str(record.get("app") or "") for record in output_records)
    est_tokens = [as_int(record.get("est_tokens")) for record in output_records]
    total_est_tokens = sum(est_tokens)
    distinct_instructions = {
        str(record.get("instruction"))
        for record in included_records
        if record.get("instruction") is not None
    }
    digest = {
        "n_tasks_stats": len(stats_rows),
        "n_records_in": len(sample_rows),
        "n_records_out": len(output_records),
        "n_records_after_split": len(output_records),
        "n_excluded_by": {key: excluded_by[key] for key in sorted(excluded_by)},
        "excluded_tasks": sorted(excluded_tasks),
        "alignment": alignment_summary(included_stats),
        "debatch_summary": debatch_summary(stats_rows),
        "n_dup_sample_id": excluded_by["dup_sample_id"],
        "n_dup_events_sha1": excluded_by["dup_events_sha1"],
        "per_app_record_counts": {key: per_app[key] for key in sorted(per_app)},
        "instructions": {"count_distinct": len(distinct_instructions)},
        "corpus_totals": {
            "n_frames": sum(as_int(record.get("n_frames")) for record in output_records),
            "est_tokens_total": total_est_tokens,
            "per_app": {key: per_app[key] for key in sorted(per_app)},
        },
        "unmapped_summary": summarize_unmapped(unmapped_rows),
        "counts_summary": sum_converter_counts(stats_rows),
        "subrecords_per_task": distribution(
            subrecords_per_task, ("min", "p50", "p90", "max", "mean")
        ),
        "est_tokens_per_subrecord": distribution(
            est_tokens, ("min", "p50", "p90", "p99", "max", "mean")
        ),
        "total_est_tokens": total_est_tokens,
    }
    write_json(args.digest_out, digest)
    return digest, output_records, split_map


def synthetic_record(sample_id, recording_id, n_frames, instruction="Synthetic goal"):
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSPROMPT}]}]
    for index in range(n_frames):
        content = [{"type": "image", "image": f"/missing/frame_{index:06d}.jpg"}]
        if index == 0:
            content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"action-{index:02d}"}],
        })
    return {
        "sample_id": sample_id,
        "recording_id": recording_id,
        "app": recording_id.split("/", 1)[0],
        "platform": "linux",
        "instruction": instruction,
        "unique_data_id": sample_id.removeprefix("psai_"),
        "task_template_id": recording_id.split("/", 1)[-1],
        "category": "synthetic",
        "duration_s": n_frames / 2,
        "n_frames": n_frames,
        "messages": messages,
    }


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def run_self_test():
    base = {
        "app": "TestApp",
        "n_bc_frames": 20,
        "skip_reason": None,
        "aspect_mismatch": False,
        "screen_dims_missing": False,
        "first_event_rel_s": 0.1,
        "last_event_rel_s": 8.0,
        "obs_stop_rel_s": 10.0,
        "counts": {
            "n_batched_events": 1,
            "n_batches": 1,
            "max_batch_len": 2,
            "debatch_max_shift_s": 0.01,
            "clicks": 1,
        },
    }
    definitions = (
        ("psai_skip", "TestApp/skip", "sha-skip", {"skip_reason": "no_frames"}),
        ("psai_short", "TestApp/short", "sha-short", {"n_bc_frames": 1}),
        ("psai_aspect", "TestApp/aspect", "sha-aspect", {"aspect_mismatch": True}),
        ("psai_early", "TestApp/early", "sha-early", {"first_event_rel_s": -1.1}),
        ("psai_late", "TestApp/late", "sha-late", {"last_event_rel_s": 11.1}),
        ("psai_dims", "TestApp/dims", "sha-dims", {"screen_dims_missing": True}),
        ("psai_long", "TestApp/long", "sha-long", {}),
        ("psai_single", "TestApp/single", "sha-single", {"n_bc_frames": 2}),
        ("psai_long", "TestApp/dup-id", "sha-dup-id", {}),
        ("psai_dup_events", "TestApp/dup-events", "sha-long", {}),
    )
    stats_rows = []
    for sample_id, task, events_sha1, changes in definitions:
        row = dict(base)
        row["counts"] = dict(base["counts"])
        row.update({"sample_id": sample_id, "task": task, "events_sha1": events_sha1})
        row.update(changes)
        stats_rows.append(row)

    with tempfile.TemporaryDirectory() as temp_dir:
        conv_dir = Path(temp_dir) / "conv"
        conv_dir.mkdir()
        write_rows(conv_dir / "stats_shard_000.jsonl", stats_rows[:8])
        write_rows(conv_dir / "stats_shard_001.jsonl", stats_rows[8:])
        samples = [
            synthetic_record("psai_skip", "TestApp/skip", 20),
            synthetic_record("psai_short", "TestApp/short", 1),
            synthetic_record("psai_aspect", "TestApp/aspect", 20),
            synthetic_record("psai_early", "TestApp/early", 20),
            synthetic_record("psai_late", "TestApp/late", 20),
            synthetic_record("psai_dims", "TestApp/dims", 20),
            synthetic_record("psai_long", "TestApp/long", 20, "Long goal"),
            synthetic_record("psai_single", "TestApp/single", 2, "Single goal"),
            synthetic_record("psai_long", "TestApp/dup-id", 20),
            synthetic_record("psai_dup_events", "TestApp/dup-events", 20),
        ]
        write_rows(conv_dir / "samples_shard_000.jsonl", samples[:8])
        write_rows(conv_dir / "samples_shard_001.jsonl", samples[8:])
        write_rows(conv_dir / "unmapped_shard_000.jsonl", [
            {"raw_value": "Mystery", "context": "key"},
            {"raw_value": "Mystery", "context": "key"},
            {"raw_value": ["A", "B"], "context": "hotkey"},
        ])
        args = argparse.Namespace(
            conv_dir=str(conv_dir),
            out_jsonl=str(Path(temp_dir) / "out.jsonl"),
            digest_out=str(Path(temp_dir) / "digest.json"),
            split_map_out=str(Path(temp_dir) / "split_map.json"),
            min_frames=2,
            split_token_budget=1800,
            min_subrecord_frames=2,
        )
        digest, output, split_map = assemble(args)

        assert digest["n_excluded_by"] == {
            "aspect_mismatch": 1,
            "dup_events_sha1": 1,
            "dup_sample_id": 1,
            "first_event_rel_s_lt_neg1": 1,
            "last_event_rel_s_gt_obs_stop_plus1": 1,
            "n_bc_frames_lt_min": 1,
            "screen_dims_missing": 1,
            "skip_reason": 1,
        }
        assert digest["n_dup_sample_id"] == 1
        assert digest["n_dup_events_sha1"] == 1
        assert digest["debatch_summary"] == {
            "n_batched_events": 10,
            "n_batches": 10,
            "max_batch_len": 2,
            "debatch_max_shift_s": 0.01,
        }
        assert digest["alignment"]["first_event_rel_s"]["mean"] == 0.1
        assert digest["alignment"]["last_event_rel_s_minus_obs_stop_rel_s"]["mean"] == -2.0
        assert digest["instructions"] == {"count_distinct": 2}
        assert split_map["included_sample_ids"] == ["psai_long", "psai_single"]
        assert split_map["excluded"]["psai_dup_events"] == ["dup_events_sha1"]
        assert split_map["excluded"]["psai_long"] == ["dup_sample_id"]

        long_parts = [record for record in output if record["recording_id"] == "TestApp/long"]
        single_parts = [record for record in output if record["recording_id"] == "TestApp/single"]
        assert len(long_parts) > 1
        assert len(single_parts) == 1
        assert single_parts[0]["sample_id"] == "psai_single"
        assert all(part["n_subrecords"] == len(long_parts) for part in long_parts)
        assert [part["subrecord_idx"] for part in long_parts] == list(range(len(long_parts)))
        assert sum(part["n_frames"] for part in long_parts) == 20
        assert all(part["n_frames"] >= 2 for part in long_parts)
        assert all(part["unique_data_id"] == "long" for part in long_parts)
        assert all(part["task_template_id"] == "long" for part in long_parts)
        assert all(part["category"] == "synthetic" for part in long_parts)
        for part in long_parts:
            messages = part["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert [item["type"] for item in messages[1]["content"]] == ["image", "text"]
            assert messages[1]["content"][1]["text"] == part["instruction"]
            for index in range(3, len(messages), 2):
                assert [item["type"] for item in messages[index]["content"]] == ["image"]
        concatenated = [
            assistant_text(message)
            for part in long_parts
            for message in part["messages"]
            if message["role"] == "assistant"
        ]
        original = [
            assistant_text(message)
            for message in samples[6]["messages"]
            if message["role"] == "assistant"
        ]
        assert concatenated == original
        cursor = 0
        for part in long_parts:
            assert first_image_item(part["messages"][1])["image"].endswith(f"{cursor:06d}.jpg")
            cursor += part["n_frames"]
        assert digest["n_records_after_split"] == len(output)
        assert digest["corpus_totals"]["n_frames"] == 22
        assert digest["per_app_record_counts"] == {"TestApp": len(output)}
        assert digest["total_est_tokens"] == sum(record["est_tokens"] for record in output)
    print("self_test passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv_dir")
    parser.add_argument("--out_jsonl")
    parser.add_argument("--digest_out")
    parser.add_argument("--split_map_out")
    parser.add_argument("--min_frames", type=int, default=2)
    parser.add_argument("--split_token_budget", type=int, default=7600)
    parser.add_argument("--min_subrecord_frames", type=int, default=2)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def validate_args(args):
    missing = [
        name for name in ("conv_dir", "out_jsonl", "digest_out", "split_map_out")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit("missing required flags: " + ", ".join("--" + name for name in missing))
    if args.min_frames < 0:
        raise SystemExit("--min_frames must be nonnegative")
    if args.min_subrecord_frames < 1:
        raise SystemExit("--min_subrecord_frames must be positive")
    if args.split_token_budget <= 0:
        raise SystemExit("--split_token_budget must be positive")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    validate_args(args)
    assemble(args)


if __name__ == "__main__":
    main()
