"""Probe action-only OPD across Qwen native absolute and OEV3 relative actions.

This is deliberately a *finite-candidate* probe, not the production loss.  A
single frozen model is scored twice:

* teacher context: the native Qwen computer-use prompt and absolute actions;
* student context: the OEV3 prompt and cursor-relative actions.

The teacher probability of every native candidate is moved to the equivalent
OEV3 candidate.  A prefix trie then turns the resulting sequence distribution
into next-token targets.  The reported chain-rule check is useful because it
catches the tempting but incorrect implementation that copies individual
absolute-coordinate digit logits onto relative-coordinate digits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from transformers import AutoTokenizer

from osworld_system_prompts import SYSTEM_PROMPTS
from sglang_runner import sglang_server


ROOT = Path(__file__).resolve().parent.parent
OEV3_PROMPT = (
    ROOT
    / "data_pipeline"
    / "realigned_pipeline"
    / "system_prompts"
    / "cua_v3_cuagym.txt"
).read_text()


@dataclass(frozen=True)
class Candidate:
    kind: str
    x: int
    y: int
    native: str
    relative: str


def candidate_actions(
    cursor: tuple[int, int],
    target: tuple[int, int],
    *,
    radius: int,
    step: int,
    include_click: bool,
) -> list[Candidate]:
    """Construct a small absolute grid and its one-to-one relative image."""
    cx, cy = cursor
    tx, ty = target
    points = sorted(
        {
            (min(1000, max(0, tx + ix * step)), min(1000, max(0, ty + iy * step)))
            for ix in range(-radius, radius + 1)
            for iy in range(-radius, radius + 1)
        }
    )
    kinds = ("mouse_move", "left_click") if include_click else ("mouse_move",)
    out: list[Candidate] = []
    for kind in kinds:
        for x, y in points:
            native = (
                '<tool_call>\n{"name":"computer_use","arguments":'
                f'{{"action":"{kind}","coordinate":[{x},{y}]}}}}\n</tool_call>'
            )
            dx, dy = x - cx, y - cy
            relative = f"move({dx},{dy})"
            if kind == "left_click":
                relative += "; down(LMB); up(LMB)"
            out.append(Candidate(kind, x, y, native, relative))
    return out


def normalized(log_weights: Iterable[float]) -> list[float]:
    values = list(log_weights)
    peak = max(values)
    weights = [math.exp(x - peak) for x in values]
    total = sum(weights)
    return [x / total for x in weights]


def prefix_children(
    sequences: list[tuple[int, ...]], probabilities: list[float], prefix: tuple[int, ...]
) -> dict[int, float]:
    """Return P(next token | prefix) for a finite sequence distribution."""
    mass: dict[int, float] = {}
    for seq, prob in zip(sequences, probabilities, strict=True):
        if len(seq) > len(prefix) and seq[: len(prefix)] == prefix:
            token = seq[len(prefix)]
            mass[token] = mass.get(token, 0.0) + prob
    total = sum(mass.values())
    if not total:
        return {}
    return {token: value / total for token, value in mass.items()}


def sequence_kl(q: list[float], p: list[float]) -> float:
    return sum(qi * math.log(qi / pi) for qi, pi in zip(q, p, strict=True))


def trie_kl(
    sequences: list[tuple[int, ...]], q: list[float], p: list[float]
) -> float:
    """KL by token conditionals; must equal KL over complete sequences."""
    prefixes = {seq[:i] for seq in sequences for i in range(len(seq))}
    total = 0.0
    for prefix in prefixes:
        q_mass = sum(
            prob
            for seq, prob in zip(sequences, q, strict=True)
            if seq[: len(prefix)] == prefix
        )
        if not q_mass:
            continue
        q_next = prefix_children(sequences, q, prefix)
        p_next = prefix_children(sequences, p, prefix)
        total += q_mass * sum(
            prob * math.log(prob / p_next[token])
            for token, prob in q_next.items()
        )
    return total


def make_prefix(tokenizer, system: str, user: str, thought: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Qwen3.5's multimodal tokenizer wrapper returns BatchEncoding here,
    # whereas ordinary text tokenizers return list[int].
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise RuntimeError(f"unexpected chat-template batch size: {len(ids)}")
        ids = ids[0]
    ids = list(ids)
    # Qwen3.5's generation prompt already opens <think>.  In the actual OPD
    # rollout, `thought` is the student's sampled reasoning copied into both
    # contexts; only the following action tokens receive loss.
    return ids + tokenizer.encode(f"{thought}\n</think>\n", add_special_tokens=False)


def score_sequences(
    endpoint: str,
    prefix: list[int],
    sequences: list[list[int]],
    *,
    batch_size: int,
    api_key: str,
) -> list[float]:
    """Teacher-force candidate suffixes and return their summed logprobs."""
    endpoint = endpoint.removesuffix("/v1")
    scores: list[float] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        payload = {
            "input_ids": [prefix + seq for seq in chunk],
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True,
            # Include one prefix token as a guard, then select the final
            # len(seq) records.  This avoids SGLang's meaningless first-token
            # logprob when scoring starts at position zero.
            "logprob_start_len": [max(0, len(prefix) - 1)] * len(chunk),
            "return_text_in_logprobs": False,
        }
        response = requests.post(
            f"{endpoint}/generate",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
        )
        response.raise_for_status()
        rows = response.json()
        if isinstance(rows, dict):
            rows = [rows]
        if len(rows) != len(chunk):
            raise RuntimeError(f"expected {len(chunk)} score rows, got {len(rows)}")
        for seq, row in zip(chunk, rows, strict=True):
            entries = row["meta_info"]["input_token_logprobs"][-len(seq) :]
            got_ids = [entry[1] for entry in entries]
            if got_ids != seq:
                raise RuntimeError(f"scored token IDs differ: expected {seq}, got {got_ids}")
            values = [entry[0] for entry in entries]
            if any(value is None for value in values):
                raise RuntimeError(f"received null candidate logprob: {entries}")
            scores.append(sum(values))
    return scores


def tok(tokenizer, token_id: int) -> str:
    if token_id == tokenizer.eos_token_id:
        return "<EOS>"
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def print_mode_path(
    tokenizer,
    sequences: list[tuple[int, ...]],
    q: list[float],
    mode: int,
) -> list[dict]:
    seq = sequences[mode]
    rows: list[dict] = []
    for pos, actual in enumerate(seq):
        prefix = seq[:pos]
        choices = prefix_children(sequences, q, prefix)
        # Common deterministic wrapper tokens are uninteresting.  Keep every
        # branch plus the signed/numeric span so tokenisation is visible.
        decoded = tok(tokenizer, actual)
        if len(choices) > 1 or any(ch.isdigit() for ch in decoded) or "-" in decoded:
            row = {
                "position": pos,
                "chosen": decoded,
                "target": [
                    {"token": tok(tokenizer, token), "p": round(prob, 6)}
                    for token, prob in sorted(choices.items(), key=lambda x: -x[1])
                ],
            }
            rows.append(row)
            print(
                f"  pos {pos:>2} chosen={decoded!r}: "
                + ", ".join(f"{item['token']!r}={item['p']:.4f}" for item in row["target"])
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url")
    source.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--cursor", type=int, nargs=2, default=(400, 600))
    parser.add_argument("--target", type=int, nargs=2, default=(700, 350))
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--include-click", action="store_true")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--api-key", default="action-transport-probe")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path
    if not tokenizer_path:
        parser.error("--tokenizer-path is required with --base-url")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")

    cursor = tuple(args.cursor)
    target = tuple(args.target)
    candidates = candidate_actions(
        cursor,
        target,
        radius=args.radius,
        step=args.step,
        include_click=args.include_click,
    )
    user = (
        f"The cursor is at [{cursor[0]},{cursor[1]}]. The target is at "
        f"[{target[0]},{target[1]}]. Move the cursor to the target."
    )
    thought = (
        f"The cursor is at ({cursor[0]},{cursor[1]}) and the target is at "
        f"({target[0]},{target[1]}), so I should move to the target."
    )
    teacher_prefix = make_prefix(tokenizer, SYSTEM_PROMPTS["computer_use_v1"], user, thought)
    student_prefix = make_prefix(tokenizer, OEV3_PROMPT, user, thought)
    native_ids = [
        tokenizer.encode(c.native, add_special_tokens=False) + [tokenizer.eos_token_id]
        for c in candidates
    ]
    relative_ids = [
        tokenizer.encode(c.relative, add_special_tokens=False) + [tokenizer.eos_token_id]
        for c in candidates
    ]

    stack = contextlib.ExitStack()
    with stack:
        if args.model_path:
            endpoint = stack.enter_context(
                sglang_server(
                    model_path=args.model_path,
                    port=args.port,
                    api_key="action-transport-probe",
                    log_path=(args.out or ROOT / "eval_logs" / "action_transport_probe.json")
                    .with_suffix(".sglang.log"),
                    mem_fraction_static=0.75,
                    chunked_prefill_size=2048,
                    served_model_name="qwen35-self-teacher",
                )
            )
        else:
            endpoint = args.base_url

        teacher_logp = score_sequences(
            endpoint,
            teacher_prefix,
            native_ids,
            batch_size=args.batch_size,
            api_key=args.api_key,
        )
        student_logp = score_sequences(
            endpoint,
            student_prefix,
            relative_ids,
            batch_size=args.batch_size,
            api_key=args.api_key,
        )

    q = normalized(teacher_logp)
    p = normalized(student_logp)
    rel_sequences = [tuple(ids) for ids in relative_ids]
    seq_kl = sequence_kl(q, p)
    token_kl = trie_kl(rel_sequences, q, p)
    if not math.isclose(seq_kl, token_kl, rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError(f"chain-rule mismatch: sequence={seq_kl}, token={token_kl}")

    teacher_order = sorted(range(len(candidates)), key=lambda i: -q[i])
    student_order = sorted(range(len(candidates)), key=lambda i: -p[i])
    print("\nTeacher native distribution transported to OEV3 (top 5):")
    for i in teacher_order[:5]:
        c = candidates[i]
        print(f"  q={q[i]:.6f}  abs=({c.x},{c.y}) {c.kind} -> {c.relative}")
    print("\nStudent OEV3 distribution over the same candidates (top 5):")
    for i in student_order[:5]:
        c = candidates[i]
        print(f"  p={p[i]:.6f}  {c.relative}")
    print("\nTransported token targets along the teacher-mode OEV3 action:")
    token_rows = print_mode_path(tokenizer, rel_sequences, q, teacher_order[0])
    print(f"\nKL(sequence)={seq_kl:.8f}")
    print(f"KL(token trie, q-prefix weighted)={token_kl:.8f}")
    print("PASS: mapped complete-action mass gives a valid token-level OEV3 target.")

    report = {
        "scope": "finite candidate set; action tokens only",
        "same_model_teacher_and_student": True,
        "cursor": cursor,
        "target": target,
        "sequence_kl": seq_kl,
        "token_trie_kl": token_kl,
        "teacher_top": [
            {
                "probability": q[i],
                "absolute": [candidates[i].x, candidates[i].y],
                "kind": candidates[i].kind,
                "relative": candidates[i].relative,
            }
            for i in teacher_order[:5]
        ],
        "student_top": [
            {"probability": p[i], "relative": candidates[i].relative}
            for i in student_order[:5]
        ],
        "teacher_mode_token_targets": token_rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
