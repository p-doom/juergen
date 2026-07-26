"""Offline teacher-forced scorer for goal-conditioned thinking-SFT checkpoints.

When every online rollout fails, this is the instrument that still separates
checkpoints and data recipes: it teacher-forces a HF checkpoint on HELD-OUT
DAYS of the stage_04 thinking-SFT dataset (conversations.jsonl) and reports
per-token NLL / greedy-exactness metrics per day and in aggregate.

RENDERING CONTRACT (parity with training)
    Training renders raw ChatML via omegalax ``qwen3_encoding.build_chatml_text``
    (assistant content VERBATIM — no chat-template ``<think>`` injection) and
    tokenizes with ``tokenizer.encode(text, add_special_tokens=False)``.
    This module REPLICATES that logic (``build_chatml_text`` below is a
    verbatim copy) instead of importing omegalax, so the scorer has no runtime
    dependency on a sibling checkout. ``test_offline_thinking_score.py``
    contains a parity test that imports the actual omegalax source by path and
    asserts identical output, plus a loss-mask parity test against
    ``collator_qwen3._build_assistant_loss_mask``.

LOSS MASKING (mirrors omegalax ``_build_assistant_loss_mask``)
    Only assistant CONTENT tokens are scored: everything after the 3-token
    header ``<|im_start|> assistant \\n`` up to and INCLUDING ``<|im_end|>``
    (the model must learn to terminate the turn); the ``\\n`` after
    ``<|im_end|>`` is not scored. Sub-span classification within a turn rides
    on special-token boundaries — ``<think>`` / ``</think>`` are single
    special tokens in the Qwen3 tokenizer, so spans are exact by construction
    (no BPE straddle is possible across them):

      thought span  tokens from ``<think>`` through ``</think>`` inclusive
      action  span  every other content token, i.e. the action text, the
                    separator ``\\n`` after ``</think>`` (when present), and
                    the trailing ``<|im_end|>``

METRICS (each with counts; per day + aggregate)
    action_nll      mean NLL/token over action spans of non-memory-update
                    assistant turns (terminate turns INCLUDED — their target
                    is an action line; ``terminate_nll`` reports that subset
                    separately so it can be discounted)
    thought_nll     mean NLL/token over thought spans only
    memory_nll      mean NLL/token over memory-update assistant turns
    action_top1     fraction of pure action turns (non-terminate, non-memory)
                    whose FULL action span (incl. ``<|im_end|>``) is exactly
                    reproduced by position-wise greedy argmax under teacher
                    forcing
    terminate_recall  same exactness, over terminate turns
                      (record metadata ``terminate`` != null -> final turn)
    terminate_false_alarm  over non-terminate action turns: greedy prefers the
                    terminate opening over the target. Computed at the first
                    token where the target action tokens DIVERGE from the
                    tokenized terminate opening (``TERMINATE``, or
                    ``\\nTERMINATE`` after a thought) — the shared prefix is
                    identical to the gold context, so the comparison is valid
                    under teacher forcing.

USAGE (future GPU run; no GPU needed for --smoke)
    cd /fast/project/HFMI_SynergyUnit/yll/juergen
    HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface HF_HUB_OFFLINE=1 \\
    uv run --no-sync python eval/offline_thinking_score.py \\
        --checkpoint <hf checkpoint dir> \\
        --conversations <stage_04 output dir or conversations.jsonl> \\
        --days 2026-05-12,2026-05-13 \\
        --batch-size 1 --device cuda --dtype bfloat16 \\
        --output report.json

    ``ar://`` image refs need the ``array_record`` package (present in the
    workspace-root venv; add ``--with array-record`` when running from the
    eval venv).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

TERMINATE_TOKEN = "TERMINATE"
TURN_ACTION = "action"
TURN_TERMINATE = "terminate"
TURN_MEMORY = "memory"

# ---------------------------------------------------------------------------
# Rendering — VERBATIM replica of omegalax qwen3_encoding.build_chatml_text
# (/fast/project/HFMI_SynergyUnit/yll/omegalax/omegalax/data/qwen3_encoding.py).
# Do not "improve" this function: byte-identical output to training is the
# whole point. Parity-tested in test_offline_thinking_score.py.
# ---------------------------------------------------------------------------


def build_chatml_text(
    messages: list[dict[str, Any]],
    image_grids: list[tuple[int, int, int]],
    merge_size: int,
) -> str:
    """Build a ChatML string from messages, inserting image pad tokens."""

    parts: list[str] = []
    img_idx = 0

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        parts.append(f"<|im_start|>{role}\n")

        if isinstance(content, str):
            parts.append(content)
        else:
            for block in content:
                if block["type"] == "text":
                    parts.append(block["text"])
                elif block["type"] == "image":
                    grid_t, grid_h, grid_w = image_grids[img_idx]
                    img_idx += 1
                    n_tokens = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
                    parts.append("<|vision_start|>" + "<|image_pad|>" * n_tokens + "<|vision_end|>")

        parts.append("<|im_end|>\n")

    return "".join(parts)


def extract_image_refs(messages: list[dict[str, Any]]) -> list[Any]:
    """Image refs in message order (mirrors omegalax ``extract_images``)."""
    refs: list[Any] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") != "image":
                continue
            if "image" in block:
                refs.append(block["image"])
            elif "url" in block:
                refs.append(block["url"])
    return refs


# ---------------------------------------------------------------------------
# ar:// image loading — minimal replica of
# data_pipeline/realigned_pipeline/lib/image_store.py (same URI scheme).
# ---------------------------------------------------------------------------

_AR_SCHEME = "ar"


def _parse_ar_uri(uri: str) -> tuple[Path, int]:
    parsed = urlparse(uri)
    if parsed.scheme != _AR_SCHEME:
        raise ValueError(f"not an ArrayRecord image URI: {uri!r}")
    if parsed.netloc or not parsed.path or not parsed.fragment:
        raise ValueError(f"malformed ArrayRecord image URI: {uri!r}")
    idx = int(parsed.fragment)
    if idx < 0:
        raise ValueError(f"ArrayRecord record index must be non-negative: {uri!r}")
    return Path(unquote(parsed.path)), idx


@lru_cache(maxsize=32)
def _ar_reader(shard_path: str):
    from array_record.python.array_record_module import ArrayRecordReader  # noqa: PLC0415

    return ArrayRecordReader(shard_path)


def open_image(ref: Any):
    """Open an image ref (``ar://`` URI or plain path) as a PIL RGB image."""
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    if isinstance(ref, str) and ref.startswith(f"{_AR_SCHEME}://"):
        shard, idx = _parse_ar_uri(ref)
        jpeg = _ar_reader(str(shard)).read([idx])[0]
        with Image.open(io.BytesIO(jpeg)) as img:
            return img.convert("RGB")
    with Image.open(ref) as img:
        return img.convert("RGB")


# ---------------------------------------------------------------------------
# Token-span logic (pure python; unit-tested against the real Qwen tokenizer
# and against the omegalax loss-mask reference)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialIds:
    """Token ids the span logic pivots on. All are SINGLE special tokens in
    the Qwen3 tokenizer (asserted at load), so span boundaries are exact."""

    im_start: int
    im_end: int
    assistant: int
    think_open: int
    think_close: int
    terminate_ids: tuple[int, ...]              # tokenize("TERMINATE")
    terminate_after_think_ids: tuple[int, ...]  # tokenize("\nTERMINATE")

    @classmethod
    def from_tokenizer(cls, tokenizer) -> SpecialIds:
        def _one(tok_str: str) -> int:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"{tok_str!r} must be a single special token in this tokenizer, got {ids}"
                )
            return ids[0]

        return cls(
            im_start=_one("<|im_start|>"),
            im_end=_one("<|im_end|>"),
            # mirrors omegalax collator: first token of "assistant"
            assistant=tokenizer.encode("assistant", add_special_tokens=False)[0],
            think_open=_one("<think>"),
            think_close=_one("</think>"),
            terminate_ids=tuple(tokenizer.encode(TERMINATE_TOKEN, add_special_tokens=False)),
            terminate_after_think_ids=tuple(
                tokenizer.encode("\n" + TERMINATE_TOKEN, add_special_tokens=False)
            ),
        )


@dataclass
class TurnSpans:
    """Token-index spans of one assistant turn. Half-open ``[start, end)``;
    ``action`` includes the trailing ``<|im_end|>`` (a supervised target)."""

    content_start: int
    im_end_pos: int
    thought: tuple[int, int] | None
    action: tuple[int, int]


def find_assistant_turns(input_ids: list[int], sp: SpecialIds) -> list[TurnSpans]:
    """Assistant-turn spans, mirroring omegalax ``_build_assistant_loss_mask``:
    pair the i-th ``<|im_start|>`` with the i-th ``<|im_end|>``; a turn is an
    assistant turn when the token after ``<|im_start|>`` is ``assistant``;
    content starts 3 tokens after ``<|im_start|>`` and ends at ``<|im_end|>``
    inclusive. Within the content, a leading ``<think>`` opens a thought span
    that ends at the first ``</think>`` (both single special tokens)."""
    n = len(input_ids)
    starts = [i for i, t in enumerate(input_ids) if t == sp.im_start]
    ends = [i for i, t in enumerate(input_ids) if t == sp.im_end]
    k = min(len(starts), len(ends))
    turns: list[TurnSpans] = []
    for s, e in zip(starts[:k], ends[:k]):
        if not (s + 1 < n and input_ids[s + 1] == sp.assistant):
            continue
        content_start = s + 3
        if not (content_start <= e < n):
            raise ValueError(f"malformed assistant turn: im_start@{s}, im_end@{e}")
        thought: tuple[int, int] | None = None
        action_start = content_start
        if content_start < e and input_ids[content_start] == sp.think_open:
            close = next(
                (i for i in range(content_start + 1, e) if input_ids[i] == sp.think_close),
                None,
            )
            if close is None:
                raise ValueError(f"unterminated <think> in assistant turn at token {content_start}")
            thought = (content_start, close + 1)
            action_start = close + 1
        turns.append(
            TurnSpans(
                content_start=content_start,
                im_end_pos=e,
                thought=thought,
                action=(action_start, e + 1),
            )
        )
    return turns


def loss_mask_from_turns(n: int, turns: list[TurnSpans]) -> list[int]:
    """Training loss mask implied by the spans (for the parity test against
    the omegalax reference): 1 on assistant content + ``<|im_end|>``."""
    mask = [0] * n
    for t in turns:
        for i in range(t.content_start, t.im_end_pos + 1):
            mask[i] = 1
    return mask


def first_divergence(a: tuple[int, ...] | list[int], b: tuple[int, ...] | list[int]) -> int | None:
    """First index where ``a`` and ``b`` differ, or None if one is a prefix
    of the other within the common length."""
    for j in range(min(len(a), len(b))):
        if a[j] != b[j]:
            return j
    return None


# ---------------------------------------------------------------------------
# Record encoding
# ---------------------------------------------------------------------------


@dataclass
class EncodedRecord:
    record: dict[str, Any]
    input_ids: list[int]
    pixel_values: Any | None       # torch tensor [total_patches, dim] or None
    image_grid_thw: Any | None     # torch tensor [n_images, 3] or None
    turns: list[tuple[str, TurnSpans]] = field(default_factory=list)  # (kind, spans)


def encode_record(record: dict[str, Any], tokenizer, image_processor, sp: SpecialIds
                  ) -> EncodedRecord:
    """Render one stage_04 conversation exactly as training does and locate
    the assistant spans. Turn kinds come from record metadata: ``terminate``
    != null -> final assistant turn is the terminate turn; ``memory_update``
    -> final assistant turn is the memory-update turn."""
    messages = record["messages"]
    refs = extract_image_refs(messages)
    pixel_values = None
    image_grid_thw = None
    grids: list[tuple[int, int, int]] = []
    if refs:
        images = [open_image(r) for r in refs]
        processed = image_processor.preprocess(images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]
        grids = [tuple(int(x) for x in row) for row in image_grid_thw.tolist()]
    merge_size = int(getattr(image_processor, "merge_size", 1))
    text = build_chatml_text(messages, grids, merge_size)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    spans = find_assistant_turns(input_ids, sp)
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    if len(spans) != n_assistant:
        raise ValueError(
            f"{record.get('conversation_id')}: found {len(spans)} assistant spans for "
            f"{n_assistant} assistant messages — rendering/span logic mismatch"
        )

    kinds = [TURN_ACTION] * len(spans)
    if spans:
        if record.get("terminate"):
            kinds[-1] = TURN_TERMINATE
            act_s, act_e = spans[-1].action
            decoded = tokenizer.decode(input_ids[act_s: act_e - 1]).strip()
            if decoded != TERMINATE_TOKEN:
                raise ValueError(
                    f"{record.get('conversation_id')}: metadata says terminate but the final "
                    f"action span decodes to {decoded!r}"
                )
        elif record.get("memory_update"):
            kinds[-1] = TURN_MEMORY

    enc = EncodedRecord(
        record=record,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    enc.turns = list(zip(kinds, spans))
    return enc


# ---------------------------------------------------------------------------
# Teacher-forced scoring
# ---------------------------------------------------------------------------


def _model_param_dtype(model):
    import torch  # noqa: PLC0415

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _forward_hidden(model, **inputs):
    """Run the checkpoint's BASE model (no lm_head) — computing full-vocab
    logits for a 32k-token sequence would materialize tens of GB; instead we
    take hidden states and apply the output head only at scored positions."""
    base = model.model if hasattr(model, "model") else model
    out = base(**inputs, use_cache=False)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0]
    return hidden


def score_batch(model, encs: list[EncodedRecord], *, pad_token_id: int, device: str,
                head_chunk: int = 1024) -> list[dict[str, Any]]:
    """Teacher-forced NLL + greedy argmax for every scored token of a batch.
    Returns one row per assistant turn (see ``aggregate`` for the schema)."""
    import torch  # noqa: PLC0415

    max_len = max(len(e.input_ids) for e in encs)
    ids = torch.full((len(encs), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(encs), max_len), dtype=torch.long)
    for b, e in enumerate(encs):
        ids[b, : len(e.input_ids)] = torch.tensor(e.input_ids, dtype=torch.long)
        attn[b, : len(e.input_ids)] = 1

    inputs: dict[str, Any] = {
        "input_ids": ids.to(device),
        "attention_mask": attn.to(device),
    }
    pvs = [e.pixel_values for e in encs if e.pixel_values is not None]
    if pvs:
        dtype = _model_param_dtype(model)
        inputs["pixel_values"] = torch.cat(pvs, dim=0).to(device=device, dtype=dtype)
        inputs["image_grid_thw"] = torch.cat(
            [e.image_grid_thw for e in encs if e.image_grid_thw is not None], dim=0
        ).to(device)

    head = model.get_output_embeddings()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        hidden = _forward_hidden(model, **inputs)
        for b, enc in enumerate(encs):
            positions: list[int] = []
            slices: list[tuple[str, TurnSpans, int, int, int]] = []
            for kind, spans in enc.turns:
                th_off = len(positions)
                if spans.thought is not None:
                    positions.extend(range(*spans.thought))
                act_off = len(positions)
                positions.extend(range(*spans.action))
                slices.append((kind, spans, th_off, act_off, len(positions)))
            if not positions:
                continue

            pos = torch.tensor(positions, dtype=torch.long, device=hidden.device)
            prev = hidden[b].index_select(0, pos - 1)
            tgt = inputs["input_ids"][b].index_select(0, pos)
            nll_parts, argmax_parts = [], []
            for c in range(0, prev.shape[0], head_chunk):
                logits = head(prev[c: c + head_chunk]).float()
                logprobs = torch.log_softmax(logits, dim=-1)
                t = tgt[c: c + head_chunk]
                nll_parts.append(-logprobs.gather(1, t[:, None])[:, 0])
                argmax_parts.append(logits.argmax(dim=-1))
            nll = torch.cat(nll_parts).cpu()
            argmax = torch.cat(argmax_parts).cpu()

            rec = enc.record
            sp_ids = enc.input_ids
            for kind, spans, th_off, act_off, end_off in slices:
                n_th = act_off - th_off
                n_act = end_off - act_off
                act_target = sp_ids[spans.action[0]: spans.action[1]]
                act_argmax = argmax[act_off:end_off].tolist()
                rows.append({
                    "conversation_id": rec.get("conversation_id"),
                    "day_tag": rec.get("day_tag"),
                    "kind": kind,
                    "has_thought": spans.thought is not None,
                    "n_thought_tokens": n_th,
                    "thought_nll_sum": float(nll[th_off:act_off].sum()) if n_th else 0.0,
                    "n_action_tokens": n_act,
                    "action_nll_sum": float(nll[act_off:end_off].sum()) if n_act else 0.0,
                    "exact": act_argmax == list(act_target) if n_act else None,
                    "action_target_ids": tuple(act_target),
                    "action_argmax_ids": tuple(act_argmax),
                })
    return rows


def attach_false_alarms(rows: list[dict[str, Any]], sp: SpecialIds) -> None:
    """Annotate non-terminate action turns with the terminate-false-alarm
    outcome (see module docstring for the divergence-position definition)."""
    for r in rows:
        r["false_alarm"] = None
        if r["kind"] != TURN_ACTION or not r["n_action_tokens"]:
            continue
        term = sp.terminate_after_think_ids if r["has_thought"] else sp.terminate_ids
        j = first_divergence(r["action_target_ids"], term)
        if j is None:
            continue  # target is a prefix of the terminate opening — not comparable
        r["false_alarm"] = r["action_argmax_ids"][j] == term[j]


def score_records(records: list[dict[str, Any]], model, tokenizer, image_processor,
                  *, device: str, batch_size: int, progress_every: int = 10
                  ) -> list[dict[str, Any]]:
    sp = SpecialIds.from_tokenizer(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id (Qwen3-VL does)")
    rows: list[dict[str, Any]] = []
    for i in range(0, len(records), batch_size):
        encs = [encode_record(r, tokenizer, image_processor, sp)
                for r in records[i: i + batch_size]]
        rows.extend(score_batch(model, encs, pad_token_id=pad_id, device=device))
        done = min(i + batch_size, len(records))
        if done % progress_every == 0 or done == len(records):
            print(f"  scored {done}/{len(records)} conversations", flush=True)
    attach_false_alarms(rows, sp)
    for r in rows:  # drop the token dumps once false alarms are computed
        r.pop("action_target_ids", None)
        r.pop("action_argmax_ids", None)
    return rows


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def _nll_metric(rows: list[dict[str, Any]], token_key: str, sum_key: str) -> dict[str, Any]:
    n_tokens = sum(r[token_key] for r in rows)
    n_turns = sum(1 for r in rows if r[token_key])
    total = sum(r[sum_key] for r in rows)
    return {
        "mean": (total / n_tokens) if n_tokens else None,
        "n_tokens": n_tokens,
        "n_turns": n_turns,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_rows = [r for r in rows if r["kind"] in (TURN_ACTION, TURN_TERMINATE)]
    pure_action = [r for r in rows if r["kind"] == TURN_ACTION and r["exact"] is not None]
    term_rows = [r for r in rows if r["kind"] == TURN_TERMINATE]
    mem_rows = [r for r in rows if r["kind"] == TURN_MEMORY]
    fa_rows = [r for r in rows if r.get("false_alarm") is not None]
    fa_skipped = sum(1 for r in rows
                     if r["kind"] == TURN_ACTION and r.get("false_alarm") is None)

    n_top1 = sum(1 for r in pure_action if r["exact"])
    n_term_ok = sum(1 for r in term_rows if r["exact"])
    n_alarm = sum(1 for r in fa_rows if r["false_alarm"])
    return {
        "action_nll": _nll_metric(action_rows, "n_action_tokens", "action_nll_sum"),
        "thought_nll": _nll_metric(
            [r for r in rows if r["has_thought"]], "n_thought_tokens", "thought_nll_sum"),
        "memory_nll": _nll_metric(mem_rows, "n_action_tokens", "action_nll_sum"),
        "terminate_nll": _nll_metric(term_rows, "n_action_tokens", "action_nll_sum"),
        "action_top1": {
            "rate": (n_top1 / len(pure_action)) if pure_action else None,
            "n_correct": n_top1,
            "n_turns": len(pure_action),
        },
        "terminate_recall": {
            "rate": (n_term_ok / len(term_rows)) if term_rows else None,
            "n_correct": n_term_ok,
            "n_turns": len(term_rows),
        },
        "terminate_false_alarm": {
            "rate": (n_alarm / len(fa_rows)) if fa_rows else None,
            "n_alarms": n_alarm,
            "n_turns": len(fa_rows),
            "n_skipped": fa_skipped,
        },
        "n_turns_total": len(rows),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_day: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        per_day.setdefault(str(r.get("day_tag")), []).append(r)
    return {
        "aggregate": compute_metrics(rows),
        "per_day": {day: compute_metrics(day_rows)
                    for day, day_rows in sorted(per_day.items())},
    }


def flat_scores(metrics: dict[str, Any], prefix: str = "offline_thinking/") -> dict[str, Any]:
    """Flatten the aggregate metrics into a pmanager-style scores dict."""
    out: dict[str, Any] = {}
    for name in ("action_nll", "thought_nll", "memory_nll", "terminate_nll"):
        out[f"{prefix}{name}"] = metrics[name]["mean"]
    out[f"{prefix}action_top1"] = metrics["action_top1"]["rate"]
    out[f"{prefix}terminate_recall"] = metrics["terminate_recall"]["rate"]
    out[f"{prefix}terminate_false_alarm"] = metrics["terminate_false_alarm"]["rate"]
    return out


def _fmt(v: float | None, spec: str = ".4f") -> str:
    return format(v, spec) if v is not None else "-"


def format_table(agg: dict[str, Any], n_records_per_day: dict[str, int]) -> str:
    cols = ("day", "convs", "turns", "act_nll", "th_nll", "mem_nll",
            "top1", "term_rec", "term_fa")
    lines = ["{:<14} {:>5} {:>6} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12}".format(*cols)]

    def _row(day: str, m: dict[str, Any], n_convs: int | str) -> str:
        top1 = m["action_top1"]
        rec = m["terminate_recall"]
        fa = m["terminate_false_alarm"]
        return "{:<14} {:>5} {:>6} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12}".format(
            day, n_convs, m["n_turns_total"],
            _fmt(m["action_nll"]["mean"]),
            _fmt(m["thought_nll"]["mean"]),
            _fmt(m["memory_nll"]["mean"]),
            f"{_fmt(top1['rate'], '.3f')} ({top1['n_correct']}/{top1['n_turns']})",
            f"{_fmt(rec['rate'], '.3f')} ({rec['n_correct']}/{rec['n_turns']})",
            f"{_fmt(fa['rate'], '.3f')} ({fa['n_alarms']}/{fa['n_turns']})",
        )

    for day, m in agg["per_day"].items():
        lines.append(_row(day, m, n_records_per_day.get(day, "-")))
    lines.append(_row("ALL", agg["aggregate"], sum(n_records_per_day.values())))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stub model (tests + --smoke): same interface surface the scorer touches —
# ``.model`` returning ``last_hidden_state`` and ``get_output_embeddings()``.
# The "hidden state" is [B, T, 1] carrying the NEXT token id, so the head can
# emit deterministic logits at gathered positions only (no [T, V] blow-up).
# ---------------------------------------------------------------------------


def make_stub_model(vocab_size: int, mode: str = "oracle", *, peak: float = 20.0,
                    const_token_id: int | None = None):
    """Deterministic no-GPU stand-in for a checkpoint.

    modes:
      oracle   logits peak at the true next token -> NLL = log1p((V-1)e^-peak),
               greedy always exact
      uniform  all-zero logits -> NLL = log(V), argmax = token 0
      const    logits peak at ``const_token_id`` everywhere (e.g. the first
               TERMINATE token, to exercise terminate_false_alarm = 1)
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    if mode not in ("oracle", "uniform", "const"):
        raise ValueError(f"unknown stub mode {mode!r}")
    if mode == "const" and const_token_id is None:
        raise ValueError("const mode needs const_token_id")

    class _Output:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class _Inner(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
            b, t = input_ids.shape
            nxt = torch.zeros((b, t, 1), dtype=torch.float32)
            nxt[:, :-1, 0] = input_ids[:, 1:].float()
            return _Output(nxt)

    class _Head(nn.Module):
        def forward(self, rows):
            n = rows.shape[0]
            logits = torch.zeros((n, vocab_size), dtype=torch.float32)
            if mode == "oracle":
                idx = rows[:, 0].long().clamp(0, vocab_size - 1)
                logits[torch.arange(n), idx] = peak
            elif mode == "const":
                logits[:, const_token_id] = peak
            return logits

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self._head = _Head()

        def get_output_embeddings(self):
            return self._head

    return _Stub().eval()


def oracle_expected_nll(vocab_size: int, peak: float = 20.0) -> float:
    """Closed-form per-token NLL of the oracle stub (for tests/smoke)."""
    return math.log1p((vocab_size - 1) * math.exp(-peak))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_days(spec: str | None) -> set[str] | None:
    """``--days``: comma-separated day tags, or a path to a file with one day
    tag per line (blank lines / #comments ignored). None -> all days."""
    if spec is None:
        return None
    p = Path(spec)
    if p.is_file():
        days = {ln.strip() for ln in p.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")}
    else:
        days = {d.strip() for d in spec.split(",") if d.strip()}
    if not days:
        raise ValueError(f"--days {spec!r} resolved to an empty day set")
    return days


def resolve_conversations_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "conversations.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(f"no conversations.jsonl under {path}")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"no such conversations file: {path}")
    return path


def load_records(conversations: Path, days: set[str] | None, max_records: int | None
                 ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with resolve_conversations_path(conversations).open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            if days is not None and str(rec.get("day_tag")) not in days:
                continue
            records.append(rec)
            if max_records is not None and len(records) >= max_records:
                break
    return records


def load_model_and_processor(checkpoint: str, *, device: str, dtype: str, smoke: bool):
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    if smoke:
        return make_stub_model(len(tokenizer)), tokenizer, image_processor
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch_dtype, attn_implementation="sdpa"
    )
    model.to(device)
    model.eval()
    return model, tokenizer, image_processor


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True,
                    help="HF checkpoint dir (or hub id in the offline cache); its processor "
                         "supplies the tokenizer + image processor even in --smoke mode.")
    ap.add_argument("--conversations", required=True, type=Path,
                    help="stage_04 output dir (containing conversations.jsonl) or the "
                         "conversations.jsonl itself.")
    ap.add_argument("--days", default=None,
                    help="Held-out day tags: comma-separated list, or a file with one day "
                         "tag per line. Default: all days in the file.")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Conversations per forward pass (24-frame windows are ~30k tokens; "
                         "keep 1 unless memory allows more).")
    ap.add_argument("--device", default=None,
                    help="Default: cuda when available, else cpu. --smoke forces cpu.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--output", type=Path, default=Path("offline_thinking_report.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="Full pipeline wiring test on <=3 records with a deterministic "
                         "stub model (no GPU, no weights load).")
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argparser().parse_args(argv)
    t0 = time.time()

    days = parse_days(args.days)
    max_records = args.max_records
    if args.smoke:
        max_records = min(max_records or 3, 3)
    records = load_records(args.conversations, days, max_records)
    if not records:
        raise SystemExit("no records to score (check --conversations / --days)")

    if args.smoke:
        device = "cpu"
    elif args.device is not None:
        device = args.device
    else:
        import torch  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, image_processor = load_model_and_processor(
        args.checkpoint, device=device, dtype=args.dtype, smoke=args.smoke
    )
    print(f"[offline_thinking] {len(records)} conversations | days="
          f"{sorted(days) if days else 'ALL'} | device={device}"
          f"{' | SMOKE (stub model)' if args.smoke else ''}", flush=True)

    rows = score_records(records, model, tokenizer, image_processor,
                         device=device, batch_size=args.batch_size)
    agg = aggregate(rows)

    n_per_day: dict[str, int] = {}
    for rec in records:
        day = str(rec.get("day_tag"))
        n_per_day[day] = n_per_day.get(day, 0) + 1

    report = {
        "task": "offline_thinking_score",
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "conversations": str(args.conversations),
        "days": sorted(days) if days else None,
        "smoke": bool(args.smoke),
        "n_records": len(records),
        "n_records_per_day": dict(sorted(n_per_day.items())),
        "params": {
            "batch_size": args.batch_size,
            "device": device,
            "dtype": args.dtype if not args.smoke else "stub",
            "max_records": max_records,
        },
        "scores": flat_scores(agg["aggregate"]),
        **agg,
        "elapsed_s": int(time.time() - t0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(format_table(agg, n_per_day))
    print(f"[offline_thinking] report -> {args.output}", flush=True)
    return report


if __name__ == "__main__":
    main()
