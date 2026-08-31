"""Inspect off-the-shelf Qwen OEV3 rollouts and top-K action-token logits.

The historical-rollout pass measures whether the model obeys the OEV3 syntax.
The replay pass sends representative saved first-step screenshots through a
fresh model server with ``logprobs=True`` and records top-K alternatives only
for the final action line (reasoning is intentionally ignored).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
from pathlib import Path

import openai

from action_parser import parse_ordered_action
from oev3_agent import Oev3Agent, _to_jpeg_b64, extract_action_line
from sglang_runner import sglang_server


def inspect_existing(root: Path) -> dict:
    counts = {
        "tasks": 0,
        "successful_tasks": 0,
        "steps": 0,
        "valid_oev3_actions": 0,
        "invalid_oev3_actions": 0,
        "move_primitives": 0,
        "large_moves": 0,
        "consecutive_exact_repeats": 0,
    }
    move_abs_x: list[int] = []
    move_abs_y: list[int] = []
    examples: list[dict] = []
    for result_path in sorted(root.glob("*/*/result.json")):
        result = json.loads(result_path.read_text())
        counts["tasks"] += 1
        counts["successful_tasks"] += result.get("scores", {}).get("reward", 0) > 0
        previous = None
        trajectory_path = result_path.parent / "traj.jsonl"
        for line in trajectory_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("step_num") == 0:
                continue
            counts["steps"] += 1
            response = row.get("response") or ""
            try:
                action = extract_action_line(response)
                if action in {"NO_OP", "TERMINATE", "FAIL"}:
                    parsed = None
                else:
                    parsed = parse_ordered_action(action)
                counts["valid_oev3_actions"] += 1
            except (TypeError, ValueError):
                counts["invalid_oev3_actions"] += 1
                continue
            if action == previous:
                counts["consecutive_exact_repeats"] += 1
            previous = action
            if parsed is None:
                continue
            for primitive in parsed.primitives:
                if primitive.kind != "move":
                    continue
                counts["move_primitives"] += 1
                move_abs_x.append(abs(primitive.dx))
                move_abs_y.append(abs(primitive.dy))
                if abs(primitive.dx) > 500 or abs(primitive.dy) > 500:
                    counts["large_moves"] += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "app": result["params"]["app"],
                            "task_id": result["params"]["task_id"],
                            "step": row["step_num"],
                            "action": action,
                        }
                    )
    counts["syntax_valid_rate"] = (
        counts["valid_oev3_actions"] / counts["steps"] if counts["steps"] else 0.0
    )
    counts["large_move_rate"] = (
        counts["large_moves"] / counts["move_primitives"]
        if counts["move_primitives"]
        else 0.0
    )
    counts["median_abs_dx"] = statistics.median(move_abs_x) if move_abs_x else None
    counts["median_abs_dy"] = statistics.median(move_abs_y) if move_abs_y else None
    counts["examples"] = examples
    return counts


def select_tasks(root: Path, apps: list[str]) -> list[Path]:
    selected: list[Path] = []
    for app in apps:
        candidates = []
        for path in sorted((root / app).glob("*/result.json")):
            result = json.loads(path.read_text())
            screenshot = path.parent / "steps" / "step_000.png"
            if screenshot.exists():
                candidates.append((result.get("scores", {}).get("n_steps_taken", 0), path))
        if candidates:
            selected.append(max(candidates)[1])
    return selected


def token_rows(choice, response_text: str, action_region: str, top_k: int) -> list[dict]:
    content = choice.logprobs.content or []
    joined = "".join(item.token for item in content)
    # SGLang normally makes joined == response_text.  Use the generated-token
    # string for offsets so byte-fallback token display cannot shift the span.
    action_start = joined.rfind(action_region)
    if action_start < 0:
        raise RuntimeError(
            f"could not find final action region {action_region!r} "
            f"in logprob token text tail {joined[-300:]!r}"
        )
    action_end = action_start + len(action_region)
    rows: list[dict] = []
    cursor = 0
    for item in content:
        start, end = cursor, cursor + len(item.token)
        cursor = end
        if end <= action_start or start >= action_end:
            continue
        alternatives = [
            {
                "token": alt.token.replace("\n", "\\n"),
                "logprob": alt.logprob,
                "probability": math.exp(alt.logprob),
            }
            for alt in item.top_logprobs[:top_k]
        ]
        rows.append(
            {
                "token": item.token.replace("\n", "\\n"),
                "logprob": item.logprob,
                "probability": math.exp(item.logprob),
                "topk_mass": sum(x["probability"] for x in alternatives),
                "topk": alternatives,
            }
        )
    return rows


def replay_one(client, model: str, result_path: Path, *, top_k: int, max_tokens: int) -> dict:
    result = json.loads(result_path.read_text())
    screenshot_path = result_path.parent / "steps" / "step_000.png"
    agent = Oev3Agent(model=model, temperature=0.0, max_tokens=max_tokens)
    screenshot_b64 = _to_jpeg_b64(screenshot_path.read_bytes())
    messages = agent._build_messages(result["params"]["task_instruction"], screenshot_b64)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_k,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    action = extract_action_line(text)
    error = None
    try:
        if action not in {"NO_OP", "TERMINATE", "FAIL"}:
            parse_ordered_action(action)
        valid = True
    except (TypeError, ValueError) as exc:
        valid = False
        error = str(exc)
    # If Qwen ignores OEV3 and falls back to its native tool grammar, retain
    # the whole tool call rather than only the final ``</tool_call>`` line.
    action_region = action
    if not valid:
        post_think = text.rsplit("</think>", 1)[-1].strip()
        tool_start = post_think.find("<tool_call>")
        action_region = post_think[tool_start:] if tool_start >= 0 else post_think
    rows = token_rows(choice, text, action_region, top_k)
    return {
        "app": result["params"]["app"],
        "task_id": result["params"]["task_id"],
        "instruction": result["params"]["task_instruction"],
        "screenshot": str(screenshot_path),
        "finish_reason": choice.finish_reason,
        "action": action,
        "action_region": action_region,
        "syntax_valid": valid,
        "parse_error": error,
        "reasoning": text[: text.rfind(action)].strip(),
        "action_tokens": rows,
        "mean_chosen_token_probability": (
            statistics.mean(row["probability"] for row in rows) if rows else None
        ),
        "mean_topk_mass": statistics.mean(row["topk_mass"] for row in rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url")
    source.add_argument("--model-path")
    parser.add_argument("--model", default="qwen38-oev3-probe")
    parser.add_argument("--api-key", default="oev3-topk-probe")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("/fast/project/HFMI_SynergyUnit/yll/eval_logs/oev3_qwen38_27b_zeroshot/offshelf/0"),
    )
    parser.add_argument(
        "--apps",
        nargs="+",
        default=["libreoffice_calc", "chrome", "gimp", "vs_code", "os"],
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    report = {
        "model": args.model_path or args.model,
        "top_k": args.top_k,
        "historical_rollouts": inspect_existing(args.rollout_root),
        "replays": [],
    }
    selected = select_tasks(args.rollout_root, args.apps)
    if len(selected) != len(args.apps):
        print(f"warning: selected {len(selected)}/{len(args.apps)} requested apps", flush=True)

    with contextlib.ExitStack() as stack:
        if args.model_path:
            endpoint = stack.enter_context(
                sglang_server(
                    model_path=args.model_path,
                    port=args.port,
                    api_key=args.api_key,
                    log_path=args.out.with_suffix(".sglang.log"),
                    mem_fraction_static=0.80,
                    chunked_prefill_size=2048,
                    served_model_name=args.model,
                    tp_size=args.tp_size,
                )
            )
            base_url = endpoint
        else:
            base_url = args.base_url
        client = openai.OpenAI(base_url=base_url, api_key=args.api_key, timeout=1800)
        for i, path in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] replaying {path.parent}", flush=True)
            try:
                replay = replay_one(
                    client,
                    args.model,
                    path,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:
                replay = {"result_path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            report["replays"].append(replay)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: replay.get(k) for k in ("app", "action", "syntax_valid", "error")}), flush=True)

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
