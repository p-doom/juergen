"""Teacher-forced (gold, pred) pair generator for bc_offline_score.py.

Reads per-step ArrayRecord chat records (stage_06 val split), replays every
message except the final assistant turn against an OpenAI-compatible sglang
server, and writes a jsonl of {"idx", "gold", "pred"} action-line pairs.
Gold is the last non-blank line of the final assistant message (after any
</think>); pred is extracted from the model response the same way. Sampling is
deterministic: every (total // num_records)-th record.

Server: pass --base_url for an already-running server, or --model_path to
boot a dedicated sglang server for the duration of the run (the export dir is
completed with tokenizer sidecars from the HF snapshot first, unless it IS
the snapshot).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from array_record.python.array_record_module import ArrayRecordReader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.realigned_pipeline.lib.image_store import read_jpeg_bytes
from hf_complete import complete_export_dir, find_hf_snapshot
from oev3_agent import extract_action_line
from sglang_runner import sglang_server

_IO_LOCK = threading.Lock()


def _joined_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _image_data_url(ref: str) -> str:
    with _IO_LOCK:
        raw = read_jpeg_bytes(ref)
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"


def build_request_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages[:-1]:
        role = m["role"]
        content = m["content"]
        if role in ("system", "assistant") or isinstance(content, str):
            out.append({"role": role, "content": _joined_text(content)})
            continue
        parts: list[dict] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "image":
                parts.append({"type": "image_url", "image_url": {"url": _image_data_url(p["image"])}})
            else:
                parts.append({"type": "text", "text": p.get("text", "") if isinstance(p, dict) else str(p)})
        out.append({"role": role, "content": parts})
    return out


def gold_action(messages: list[dict]) -> str:
    final = messages[-1]
    assert final["role"] == "assistant"
    return extract_action_line(_joined_text(final["content"]))


def sample_indices(total: int, want: int) -> list[int]:
    if want <= 0 or want >= total:
        return list(range(total))
    stride = total // want
    return [i * stride for i in range(want)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val_shard", required=True, help="path to part-NNNNN.array_record")
    ap.add_argument("--num_records", type=int, default=400)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--mem_fraction_static", type=float, default=0.80)
    ap.add_argument("--chunked_prefill_size", type=int, default=2048)
    ap.add_argument("--api_key", default="probe")
    ap.add_argument("--model", default="bc-probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()
    if bool(args.base_url) == bool(args.model_path):
        ap.error("pass exactly one of --base_url or --model_path")

    reader = ArrayRecordReader(args.val_shard)
    total = reader.num_records()
    indices = sample_indices(total, args.num_records)
    print(f"[pairs] {len(indices)}/{total} records from {args.val_shard}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        if args.model_path:
            snapshot = find_hf_snapshot(args.model_id, Path(os.environ["HF_HOME"]))
            if Path(args.model_path).resolve() != snapshot.resolve():
                completion = complete_export_dir(Path(args.model_path), snapshot)
                print(f"[hf_complete] copied={completion['copied']} patched={completion['patched']}", flush=True)
            base_url = stack.enter_context(
                sglang_server(
                    model_path=args.model_path,
                    port=args.port,
                    api_key=args.api_key,
                    log_path=out_path.parent / "sglang_server.log",
                    mem_fraction_static=args.mem_fraction_static,
                    chunked_prefill_size=args.chunked_prefill_size,
                    served_model_name=args.model,
                )
            )
        else:
            base_url = args.base_url
        run_pairs(args, reader, indices, base_url, out_path)


def run_pairs(args, reader, indices: list[int], base_url: str, out_path: Path) -> None:
    client = openai.OpenAI(base_url=base_url, api_key=args.api_key, timeout=600, max_retries=0)
    done = 0
    done_lock = threading.Lock()

    def run_one(idx: int) -> dict:
        nonlocal done
        with _IO_LOCK:
            rec = json.loads(reader.read([idx])[0])
        messages = rec["messages"]
        gold = gold_action(messages)
        request_messages = build_request_messages(messages)
        pred = ""
        raw = ""
        err = ""
        for attempt in range(args.retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=request_messages,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                msg = resp.choices[0].message
                raw = msg.content or getattr(msg, "reasoning_content", None) or ""
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(2**attempt, 30))
        try:
            pred = extract_action_line(raw)
        except ValueError:
            pred = ""
        with done_lock:
            done += 1
            if done % 25 == 0:
                print(f"[pairs] {done}/{len(indices)}", flush=True)
        row = {"idx": idx, "gold": gold, "pred": pred}
        if not raw and err:
            row["error"] = err
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(run_one, indices))

    n_err = sum(1 for r in rows if r.get("error"))
    with out_path.open("w") as fh:
        for r in sorted(rows, key=lambda r: r["idx"]):
            fh.write(json.dumps(r) + "\n")
    print(f"[pairs] wrote {len(rows)} pairs to {out_path} ({n_err} request failures)", flush=True)
    if n_err > len(rows) // 4:
        sys.exit(3)


if __name__ == "__main__":
    main()
