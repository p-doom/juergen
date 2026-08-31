"""Probe whole-action OPD across OEV3-relative and native-absolute actions.

This is the stage-0 experiment for the variable-length action mapping idea:

1. Qwen3.5 samples an OEV3 action on a real CUA-Gym validation screenshot.
2. The action is parsed and deterministically converted to Qwen's native
   ``computer_use`` grammar, including relative -> absolute mouse coordinates.
3. Qwen3.5 scores the sampled OEV3 action and Qwen3.8 scores the converted
   native action.  Only the complete action spans (plus EOS) are summed.

No token-to-token alignment is assumed.  The resulting Monte-Carlo reverse-KL
sample is ``student_logp - teacher_logp`` for one semantic action.  This script
does not update weights; it validates the signal and its scale before wiring it
into an online trainer.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
import requests
from array_record.python.array_record_module import ArrayRecordReader

from action_parser import OrderedPrimitive, parse_ordered_action
from oev3_agent import extract_action_line, rdev_to_pyautogui
from osworld_system_prompts import SYSTEM_PROMPTS


ROOT = Path(__file__).resolve().parent.parent
OEV3_PROMPT = (
    ROOT / "data_pipeline" / "realigned_pipeline" / "system_prompts" / "cua_v3_cuagym.txt"
).read_text().strip()
GRID = 1000
_INSTRUCTION_RE = re.compile(r"Instruction:\s*(.*?)(?:\n\nPrevious actions:|\Z)", re.DOTALL)
_IO_LOCK = threading.Lock()

# Prefer the short spellings seen in native Qwen CUA data.  The OEV3 executor
# distinguishes left/right modifiers, but the native key action generally
# does not; these aliases preserve the desktop effect while giving the teacher
# the string it was trained to score.
_NATIVE_KEY_ALIASES = {
    "ctrlleft": "ctrl",
    "ctrlright": "ctrl",
    "shiftleft": "shift",
    "shiftright": "shift",
    "winleft": "win",
    "winright": "win",
}


@dataclass(frozen=True)
class NativeTranslation:
    text: str
    calls: tuple[dict[str, Any], ...]
    cursor_after: tuple[int, int]


def _grid_position(cursor_px: tuple[int, int], screen: tuple[int, int]) -> list[int]:
    x, y = cursor_px
    width, height = screen
    return [
        max(0, min(GRID, round(x * GRID / width))),
        max(0, min(GRID, round(y * GRID / height))),
    ]


def _move_cursor(
    cursor_px: tuple[int, int], primitive: OrderedPrimitive, screen: tuple[int, int]
) -> tuple[int, int]:
    width, height = screen
    return (
        max(0, min(width - 1, cursor_px[0] + round(primitive.dx * width / GRID))),
        max(0, min(height - 1, cursor_px[1] + round(primitive.dy * height / GRID))),
    )


def _call(action: str, **kwargs: Any) -> dict[str, Any]:
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def _mouse_name(primitive: OrderedPrimitive) -> str | None:
    if primitive.mouse_button is None:
        return None
    return {1: "left", 2: "middle", 3: "right"}.get(primitive.mouse_button)


def _click_call(button: str, count: int, coordinate: list[int]) -> dict[str, Any] | None:
    if button == "left" and count in (1, 2, 3):
        action = {1: "left_click", 2: "double_click", 3: "triple_click"}[count]
    elif count == 1 and button in ("right", "middle"):
        action = f"{button}_click"
    else:
        return None
    return _call(action, coordinate=coordinate)


def _balanced_keys(primitives: tuple[OrderedPrimitive, ...], start: int) -> tuple[list[str], int] | None:
    """Recognize down(k1); down(k2); up(k2); up(k1)."""
    downs: list[str] = []
    i = start
    while i < len(primitives) and primitives[i].kind == "down" and _mouse_name(primitives[i]) is None:
        downs.append(primitives[i].name or "")
        i += 1
    if not downs:
        return None
    expected = list(reversed(downs))
    ups: list[str] = []
    while i < len(primitives) and len(ups) < len(expected):
        p = primitives[i]
        if p.kind != "up" or _mouse_name(p) is not None:
            return None
        ups.append(p.name or "")
        i += 1
    if ups != expected:
        return None
    keys = [rdev_to_pyautogui(key) for key in downs]
    return [_NATIVE_KEY_ALIASES.get(key, key) for key in keys], i


def oev3_to_native(
    action_line: str,
    cursor_px: tuple[int, int],
    screen: tuple[int, int],
) -> NativeTranslation:
    """Map one OEV3 program to canonical native ``computer_use`` calls.

    We collapse common primitive programs back to the native actions from
    which the training corpus was translated.  Actions with no lossless native
    representation (for example a mouse button held across turns) are rejected
    instead of receiving a misleading teacher score.
    """
    line = action_line.strip()
    if line == "NO_OP":
        calls = [_call("wait", time=1)]
        return NativeTranslation(_render_calls(calls), tuple(calls), cursor_px)
    if line == "TERMINATE":
        calls = [_call("terminate", status="success")]
        return NativeTranslation(_render_calls(calls), tuple(calls), cursor_px)
    if line == "FAIL":
        calls = [_call("terminate", status="failure")]
        return NativeTranslation(_render_calls(calls), tuple(calls), cursor_px)

    action = parse_ordered_action(line)
    primitives = action.primitives
    calls: list[dict[str, Any]] = []
    cursor = cursor_px
    i = 0
    while i < len(primitives):
        p = primitives[i]

        # down(LMB); move(...); up(LMB) is a native drag.
        if (
            p.kind == "down"
            and _mouse_name(p) == "left"
            and i + 2 < len(primitives)
            and primitives[i + 1].kind == "move"
            and primitives[i + 2].kind == "up"
            and _mouse_name(primitives[i + 2]) == "left"
        ):
            cursor = _move_cursor(cursor, primitives[i + 1], screen)
            calls.append(_call("left_click_drag", coordinate=_grid_position(cursor, screen)))
            i += 3
            continue

        # A move followed by one or more complete clicks is one native click
        # action at the resulting absolute coordinate.
        if p.kind == "move":
            cursor = _move_cursor(cursor, p, screen)
            j = i + 1
            button = None
            count = 0
            while j + 1 < len(primitives):
                down, up = primitives[j], primitives[j + 1]
                name = _mouse_name(down)
                if down.kind != "down" or up.kind != "up" or name is None or _mouse_name(up) != name:
                    break
                if button is None:
                    button = name
                if name != button:
                    break
                count += 1
                j += 2
            click = _click_call(button, count, _grid_position(cursor, screen)) if button else None
            if click is not None:
                calls.append(click)
                i = j
            else:
                calls.append(_call("mouse_move", coordinate=_grid_position(cursor, screen)))
                i += 1
            continue

        # One or more clicks without movement happen at the current cursor.
        if p.kind == "down" and _mouse_name(p) is not None:
            button = _mouse_name(p)
            j = i
            count = 0
            while j + 1 < len(primitives):
                down, up = primitives[j], primitives[j + 1]
                if (
                    down.kind != "down"
                    or up.kind != "up"
                    or _mouse_name(down) != button
                    or _mouse_name(up) != button
                ):
                    break
                count += 1
                j += 2
            click = _click_call(button or "", count, _grid_position(cursor, screen))
            if click is None:
                raise ValueError(f"no native click for {line!r} at primitive {i}")
            calls.append(click)
            i = j
            continue

        if p.kind == "scroll":
            if p.dy:
                calls.append(_call("scroll", pixels=p.dy))
            if p.dx:
                calls.append(_call("hscroll", pixels=p.dx))
            i += 1
            continue

        if p.kind == "type":
            calls.append(_call("type", text=p.text or ""))
            i += 1
            continue

        if p.kind == "down":
            key_run = _balanced_keys(primitives, i)
            if key_run is None:
                raise ValueError(f"unbalanced key program has no native equivalent: {line!r}")
            keys, i = key_run
            calls.append(_call("key", keys=keys))
            continue

        raise ValueError(f"unsupported OEV3 primitive {p.kind!r} in {line!r}")

    if not calls:
        calls = [_call("wait", time=1)]
    return NativeTranslation(_render_calls(calls), tuple(calls), cursor)


def _render_calls(calls: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>" for call in calls
    )


def _data_url(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def _instruction(record: dict) -> str:
    for message in record["messages"]:
        if message["role"] != "user":
            continue
        text = _joined_text(message["content"])
        if match := _INSTRUCTION_RE.search(text):
            return match.group(1).strip()
    raise ValueError("could not recover Instruction: field from record")


def _joined_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _messages(system: str, instruction: str, image_url: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": (
                        "Please generate the next move according to the UI screenshot and instruction.\n\n"
                        f"Instruction: {instruction}"
                    ),
                },
            ],
        },
    ]


def _template_messages(system: str, instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "unused-by-template"},
                {
                    "type": "text",
                    "text": (
                        "Please generate the next move according to the UI screenshot and instruction.\n\n"
                        f"Instruction: {instruction}"
                    ),
                },
            ],
        },
    ]


def _response_parts(response: str) -> tuple[str, str]:
    action = extract_action_line(response)
    start = response.rfind(action)
    if start < 0:
        raise ValueError("action is not a suffix of response")
    if response[start + len(action) :].strip():
        raise ValueError("non-whitespace content follows action")
    return response[:start], action


def _score_action(
    endpoint: str,
    api_key: str,
    tokenizer,
    system: str,
    instruction: str,
    image_url: str,
    response_prefix: str,
    action: str,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        _template_messages(system, instruction), tokenize=False, add_generation_prompt=True
    )
    prefix_text = prompt + response_prefix
    full_ids = tokenizer.encode(prefix_text + action, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("action boundary merged with its prefix; cannot score span safely")
    suffix_ids = full_ids[len(prefix_ids) :] + [tokenizer.eos_token_id]
    payload = {
        "input_ids": full_ids + [tokenizer.eos_token_id],
        "image_data": [image_url],
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    response = requests.post(
        endpoint.rstrip("/").removesuffix("/v1") + "/generate",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=900,
    )
    response.raise_for_status()
    body = response.json()
    entries = body["meta_info"]["input_token_logprobs"][-len(suffix_ids) :]
    got_ids = [entry[1] for entry in entries]
    if got_ids != suffix_ids:
        raise ValueError(f"scored suffix IDs differ: expected={suffix_ids}, got={got_ids}")
    logps = [entry[0] for entry in entries]
    if any(value is None for value in logps):
        raise ValueError(f"null action logprob: {entries}")
    return {
        "sum_logp": float(sum(logps)),
        "mean_logp": float(statistics.mean(logps)),
        "n_tokens": len(logps),
        "token_logps": logps,
        "token_ids": suffix_ids,
    }


def _generate_student(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    client = openai.OpenAI(base_url=endpoint.rstrip("/") + "/v1", api_key=api_key, timeout=900, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    return message.content or getattr(message, "reasoning_content", None) or ""


def _summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if "sequence_rkl_sample" in row]
    values = [row["sequence_rkl_sample"] for row in valid]
    teacher_gold_margins = [
        row["teacher_gold_minus_sample_logp"]
        for row in valid
        if row.get("teacher_gold_minus_sample_logp") is not None
    ]
    return {
        "n_requested": len(rows),
        "n_valid": len(valid),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "sequence_rkl_mean": statistics.mean(values) if values else None,
        "sequence_rkl_median": statistics.median(values) if values else None,
        "sequence_rkl_min": min(values) if values else None,
        "sequence_rkl_max": max(values) if values else None,
        "teacher_gold_minus_sample_logp_mean": (
            statistics.mean(teacher_gold_margins) if teacher_gold_margins else None
        ),
        "n_teacher_prefers_gold": sum(x > 0 for x in teacher_gold_margins),
        "n_teacher_gold_scored": len(teacher_gold_margins),
    }


def main() -> None:
    # Keep this heavyweight import out of converter-only unit tests.  On the
    # shared filesystem, Transformers' model discovery can take a while.
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-url", required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--student-model", default="oev3-student")
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--api-key", default="sequence-opd-probe")
    parser.add_argument("--val-shard", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-records", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_tokenizer, local_files_only=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    # Deterministic evenly-spaced coverage is more useful than a lucky random subset.
    if args.num_records and args.num_records < len(manifest):
        stride = len(manifest) / args.num_records
        manifest = [manifest[math.floor(i * stride)] for i in range(args.num_records)]
    reader = ArrayRecordReader(args.val_shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows_lock = threading.Lock()

    def one(item: dict) -> dict:
        row = {k: item.get(k) for k in ("idx", "app", "pool", "recording_id", "target_step", "cursor", "screen")}
        try:
            with _IO_LOCK:
                record = json.loads(reader.read([item["idx"]])[0])
            instruction = _instruction(record)
            image_url = _data_url(item["image_a"])
            student_messages = _messages(OEV3_PROMPT, instruction, image_url)
            raw = _generate_student(
                args.student_url,
                args.api_key,
                args.student_model,
                student_messages,
                args.temperature,
                args.max_tokens,
            )
            thought, sampled_action = _response_parts(raw)
            mapped = oev3_to_native(sampled_action, tuple(item["cursor"]), tuple(item["screen"]))
            student_score = _score_action(
                args.student_url,
                args.api_key,
                student_tok,
                OEV3_PROMPT,
                instruction,
                image_url,
                thought,
                sampled_action,
            )
            teacher_score = _score_action(
                args.teacher_url,
                args.api_key,
                teacher_tok,
                SYSTEM_PROMPTS["computer_use_v1"],
                instruction,
                image_url,
                thought,
                mapped.text,
            )
            row.update(
                {
                    "instruction": instruction,
                    "student_response": raw,
                    "student_action": sampled_action,
                    "native_action": mapped.text,
                    "cursor_after": list(mapped.cursor_after),
                    "student_action_score": student_score,
                    "teacher_native_score": teacher_score,
                    "sequence_rkl_sample": student_score["sum_logp"] - teacher_score["sum_logp"],
                    # A length-normalized diagnostic only.  It is not the exact
                    # sequence reverse-KL estimator.
                    "mean_logp_gap": student_score["mean_logp"] - teacher_score["mean_logp"],
                }
            )
            try:
                gold = extract_action_line(_joined_text(record["messages"][-1]["content"]))
                gold_native = oev3_to_native(gold, tuple(item["cursor"]), tuple(item["screen"]))
                gold_score = _score_action(
                    args.teacher_url,
                    args.api_key,
                    teacher_tok,
                    SYSTEM_PROMPTS["computer_use_v1"],
                    instruction,
                    image_url,
                    thought,
                    gold_native.text,
                )
                row["gold_action"] = gold
                row["gold_native_action"] = gold_native.text
                row["teacher_gold_score"] = gold_score
                row["teacher_gold_minus_sample_logp"] = (
                    gold_score["sum_logp"] - teacher_score["sum_logp"]
                )
            except Exception as exc:  # gold is an optional diagnostic
                row["gold_score_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        with rows_lock:
            print(json.dumps({k: row.get(k) for k in ("idx", "student_action", "sequence_rkl_sample", "error")}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, manifest))

    report = {
        "experiment": "oev3_native_whole_action_sequence_opd_signal_probe",
        "student_model": args.student_model,
        "student_tokenizer": args.student_tokenizer,
        "teacher_tokenizer": args.teacher_tokenizer,
        "temperature": args.temperature,
        "objective": "student OEV3 action logp - teacher mapped-native action logp; action+EOS only",
        "summary": _summary(rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    if report["summary"]["n_valid"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
