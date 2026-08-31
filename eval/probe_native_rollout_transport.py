"""Probe native-to-OEV3 sequence-score transport on real CUA-Gym states.

Both models stay in the native Qwen computer-use language:

1. Qwen3.5 samples a native absolute-coordinate response.
2. Qwen3.5 and Qwen3.8 score that same native action suffix.
3. The complete native action is deterministically converted to OEV3.
4. Qwen3.5 scores the converted action under the OEV3 system prompt.

The native teacher/student sequence log-ratio is the scalar that a sampled
transport objective would attach to the complete converted OEV3 sequence.  It
is never copied position by position, so native and OEV3 token counts may
differ.  This is a signal probe only; it does not update weights.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from array_record.python.array_record_module import ArrayRecordReader

from action_parser import parse_qwen3vl_computer_use_action
from cuagym_pipeline.translate import DropStep, norm_to_px, translate_step
from osworld_system_prompts import SYSTEM_PROMPTS
from probe_sequence_opd import (
    OEV3_PROMPT,
    _data_url,
    _generate_student,
    _instruction,
    _messages,
    _score_action,
    _template_messages,
)


NATIVE_PROMPT = SYSTEM_PROMPTS["computer_use_v1"]
_IO_LOCK = threading.Lock()


def native_response_parts(response: str) -> tuple[str, str]:
    """Split a native response into its unchanged reasoning and tool suffix."""
    think_end = response.rfind("</think>")
    start = response.find("<tool_call>", max(0, think_end))
    if start < 0:
        raise ValueError("native response has no <tool_call> suffix")
    prefix = response[:start]
    action = response[start:].strip()
    parse_qwen3vl_computer_use_action(action)
    return prefix, action


def native_to_oev3(
    native_action: str,
    cursor_px: tuple[int, int],
    screen: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """Convert complete native calls into one canonical OEV3 action program."""
    calls = parse_qwen3vl_computer_use_action(native_action)
    cursor = cursor_px
    lines: list[str] = []
    for call in calls:
        try:
            converted = translate_step(call.arguments, cursor, screen)
        except DropStep as exc:
            raise ValueError(f"native action has no lossless OEV3 mapping: {exc}") from exc
        if converted.dropped_reason:
            raise ValueError(
                f"native action has no lossless OEV3 mapping: {converted.dropped_reason}"
            )
        if converted.line:
            lines.append(converted.line)
        if converted.target_norm is not None:
            cursor = norm_to_px(converted.target_norm, screen)

    if not lines:
        raise ValueError("native action converted to an empty OEV3 program")
    if len(lines) > 1 and any(line in {"NO_OP", "TERMINATE", "FAIL"} for line in lines):
        raise ValueError("cannot combine a terminal/no-op marker with another OEV3 action")
    return "; ".join(lines), cursor


def _sample_native_action_suffix(
    endpoint: str,
    api_key: str,
    tokenizer,
    instruction: str,
    image_url: str,
    thought: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Sample another native action after an already fixed reasoning prefix."""
    prompt = tokenizer.apply_chat_template(
        _template_messages(NATIVE_PROMPT, instruction),
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prompt + thought, add_special_tokens=False)
    payload = {
        "input_ids": prefix_ids,
        "image_data": [image_url],
        "sampling_params": {
            "temperature": temperature,
            "top_p": 0.95,
            "max_new_tokens": max_tokens,
            "skip_special_tokens": False,
        },
    }
    response = requests.post(
        endpoint.rstrip("/").removesuffix("/v1") + "/generate",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=900,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise ValueError(f"unexpected SGLang generation response: {body!r}")
    action = body["text"].strip()
    parse_qwen3vl_computer_use_action(action)
    return action


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [candidate for row in rows for candidate in row.get("candidates", [])]
    valid = [candidate for candidate in candidates if "native_teacher_log_ratio" in candidate]
    ratios = [candidate["native_teacher_log_ratio"] for candidate in valid]
    native_lengths = [candidate["student_native_score"]["n_tokens"] for candidate in valid]
    oev3_lengths = [candidate["student_oev3_score"]["n_tokens"] for candidate in valid]
    valid_states = [row for row in rows if row.get("transported_distribution")]
    return {
        "n_states_requested": len(rows),
        "n_states_valid": len(valid_states),
        "n_candidates_requested": len(candidates),
        "n_candidates_valid": len(valid),
        "candidate_valid_rate": len(valid) / len(candidates) if candidates else 0.0,
        "native_teacher_log_ratio_mean": statistics.mean(ratios) if ratios else None,
        "native_teacher_log_ratio_median": statistics.median(ratios) if ratios else None,
        "native_teacher_log_ratio_min": min(ratios) if ratios else None,
        "native_teacher_log_ratio_max": max(ratios) if ratios else None,
        "student_native_action_tokens_mean": (
            statistics.mean(native_lengths) if native_lengths else None
        ),
        "student_oev3_action_tokens_mean": (
            statistics.mean(oev3_lengths) if oev3_lengths else None
        ),
        "n_token_lengths_differ": sum(a != b for a, b in zip(native_lengths, oev3_lengths)),
        "importance_ess_mean": (
            statistics.mean(row["importance_ess"] for row in valid_states)
            if valid_states
            else None
        ),
    }


def _normalized_importance(log_ratios: list[float]) -> list[float]:
    peak = max(log_ratios)
    weights = [math.exp(value - peak) for value in log_ratios]
    total = sum(weights)
    return [weight / total for weight in weights]


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-url", required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--student-model", default="qwen35-native-student")
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--api-key", default="native-transport-probe")
    parser.add_argument("--val-shard", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-records", type=int, default=4)
    parser.add_argument("--samples-per-state", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_tokenizer, local_files_only=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line]
    if args.num_records and args.num_records < len(manifest):
        stride = len(manifest) / args.num_records
        manifest = [manifest[math.floor(i * stride)] for i in range(args.num_records)]
    reader = ArrayRecordReader(args.val_shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def one(item: dict[str, Any]) -> dict[str, Any]:
        row = {
            key: item.get(key)
            for key in ("idx", "app", "pool", "recording_id", "target_step", "cursor", "screen")
        }
        row["candidates"] = []
        try:
            with _IO_LOCK:
                record = json.loads(reader.read([item["idx"]])[0])
            instruction = _instruction(record)
            image_url = _data_url(item["image_a"])
            row["instruction"] = instruction
            first_raw = _generate_student(
                args.student_url,
                args.api_key,
                args.student_model,
                _messages(NATIVE_PROMPT, instruction, image_url),
                args.temperature,
                args.max_tokens,
            )
            thought, first_action = native_response_parts(first_raw)
            row["fixed_student_reasoning"] = thought
            for sample_index in range(args.samples_per_state):
                candidate: dict[str, Any] = {"sample_index": sample_index}
                try:
                    native_action = (
                        first_action
                        if sample_index == 0
                        else _sample_native_action_suffix(
                            args.student_url,
                            args.api_key,
                            student_tok,
                            instruction,
                            image_url,
                            thought,
                            args.temperature,
                            min(args.max_tokens, 512),
                        )
                    )
                    oev3_action, cursor_after = native_to_oev3(
                        native_action, tuple(item["cursor"]), tuple(item["screen"])
                    )

                    student_native = _score_action(
                        args.student_url,
                        args.api_key,
                        student_tok,
                        NATIVE_PROMPT,
                        instruction,
                        image_url,
                        thought,
                        native_action,
                    )
                    teacher_native = _score_action(
                        args.teacher_url,
                        args.api_key,
                        teacher_tok,
                        NATIVE_PROMPT,
                        instruction,
                        image_url,
                        thought,
                        native_action,
                    )
                    student_oev3 = _score_action(
                        args.student_url,
                        args.api_key,
                        student_tok,
                        OEV3_PROMPT,
                        instruction,
                        image_url,
                        thought,
                        oev3_action,
                    )
                    candidate.update(
                        {
                            "native_action": native_action,
                            "oev3_action": oev3_action,
                            "cursor_after": list(cursor_after),
                            "student_native_score": student_native,
                            "teacher_native_score": teacher_native,
                            "student_oev3_score": student_oev3,
                            # q_native / p_native is the importance ratio for
                            # a candidate drawn from the native student.
                            "native_teacher_log_ratio": (
                                teacher_native["sum_logp"] - student_native["sum_logp"]
                            ),
                        }
                    )
                except Exception as exc:
                    candidate["error"] = f"{type(exc).__name__}: {exc}"
                row["candidates"].append(candidate)

            valid = [
                candidate
                for candidate in row["candidates"]
                if "native_teacher_log_ratio" in candidate
            ]
            if valid:
                weights = _normalized_importance(
                    [candidate["native_teacher_log_ratio"] for candidate in valid]
                )
                merged: dict[str, float] = {}
                for candidate, weight in zip(valid, weights, strict=True):
                    candidate["transported_weight"] = weight
                    action = candidate["oev3_action"]
                    merged[action] = merged.get(action, 0.0) + weight
                row["transported_distribution"] = [
                    {"oev3_action": action, "probability": probability}
                    for action, probability in sorted(merged.items(), key=lambda pair: -pair[1])
                ]
                row["importance_ess"] = 1.0 / sum(weight * weight for weight in weights)
            else:
                row["error"] = "no valid native candidates"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in (
                        "idx",
                        "transported_distribution",
                        "importance_ess",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, manifest))

    report = {
        "experiment": "native_student_native_teacher_to_oev3_sequence_transport",
        "student_model": args.student_model,
        "student_tokenizer": args.student_tokenizer,
        "teacher_tokenizer": args.teacher_tokenizer,
        "temperature": args.temperature,
        "samples_per_state": args.samples_per_state,
        "objective": (
            "sample under Qwen3.5 native policy; teacher/student score native action+EOS; "
            "self-normalize q_native/p_native over each state's candidates; merge equivalent "
            "converted actions; train the resulting OEV3 sequence distribution"
        ),
        "strict_opd_caveat": (
            "Native-prompt samples are off-policy for the OEV3-prompt language-model policy. "
            "This is online transformed distillation unless corrected or followed by OEV3 rollouts."
        ),
        "summary": _summary(rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    if report["summary"]["n_states_valid"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
