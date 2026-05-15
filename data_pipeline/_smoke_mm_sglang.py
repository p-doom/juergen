"""Smoke test: confirm sglang serves Qwen3-VL multimodal chat completions.

Validates the wire format we'll use for the on-policy multimodal generation
pipeline before we commit to running it at scale. Specifically checks:
  * sglang launches under the eval venv with the off-shelf model_path
  * the OAI vision API accepts ``image_url`` blocks
  * file paths can be referenced (we'll switch to base64 if file:// URLs
    don't work)
  * ``finish_reason`` is returned per request (we couldn't capture it on
    smoltalk2 because we forgot; this time we plumb it through)
  * generations look qualitatively sensible on a few FineVision rows
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/fast/home/franz.srambical/eval")
sys.path.insert(0, "/fast/home/franz.srambical/data_pipeline")
import openai
from _sglang import sglang_server  # type: ignore[import-not-found]

SOURCE_JSONL = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/datasets/"
    "2026-05-01_replay_finevision_mixed_v1_chat_jsonl/train/chat.jsonl"
)
TEACHER_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _img_to_data_url(path: str) -> str:
    """Encode local jpeg/png as data: URL — the format sglang's OAI shim
    accepts most reliably across versions."""
    p = Path(path)
    b = p.read_bytes()
    ext = p.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(b).decode('ascii')}"


def _row_to_oai_messages(row: dict) -> list:
    """Convert one FineVision-style chat row's prefix (everything up to
    the first assistant turn) into OAI vision-chat-completions format."""
    out = []
    for m in row["messages"]:
        if m.get("role") == "assistant":
            break
        content_blocks = []
        c = m.get("content")
        if isinstance(c, str):
            content_blocks.append({"type": "text", "text": c})
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "text":
                    content_blocks.append({"type": "text", "text": blk.get("text", "")})
                elif t == "image":
                    url = blk.get("url") or blk.get("path")
                    if not url:
                        continue
                    if not url.startswith(("http://", "https://", "data:")):
                        url = _img_to_data_url(url)
                    content_blocks.append({"type": "image_url", "image_url": {"url": url}})
                elif t == "image_url":
                    content_blocks.append(blk)
        if content_blocks:
            out.append({"role": m.get("role", "user"), "content": content_blocks})
    return out


async def _gen_one(client, messages, idx):
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            seed=idx,
        )
    except Exception as e:
        return {"idx": idx, "ok": False, "err": repr(e)[:300]}
    elapsed = time.time() - t0
    choice = resp.choices[0] if resp.choices else None
    return {
        "idx": idx,
        "ok": True,
        "elapsed_s": round(elapsed, 1),
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        "content_len": len(getattr(choice.message, "content", "") or "") if choice else 0,
        "content_preview": (getattr(choice.message, "content", "") or "")[:200] if choice else "",
        "usage_completion_tokens": getattr(resp.usage, "completion_tokens", None)
        if resp.usage
        else None,
    }


async def _run(rows: list[dict], base_url: str, api_key: str) -> list[dict]:
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    tasks = []
    for idx, row in enumerate(rows):
        msgs = _row_to_oai_messages(row)
        if not msgs:
            tasks.append(asyncio.sleep(0, result={"idx": idx, "ok": False, "err": "empty msgs"}))
            continue
        tasks.append(_gen_one(client, msgs, idx))
    return await asyncio.gather(*tasks)


def main():
    n_samples = int(os.environ.get("SMOKE_N", "5"))
    print(f"[smoke] loading {n_samples} rows from {SOURCE_JSONL}", flush=True)
    rows = []
    with Path(SOURCE_JSONL).open() as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            rows.append(json.loads(line))
    print(
        f"[smoke] loaded {len(rows)} rows; sources: {[r.get('_source') for r in rows]}", flush=True
    )
    for i, r in enumerate(rows):
        m0 = r["messages"][0]
        prompts = [
            b.get("text", "")[:80]
            for b in m0["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        print(f"  row {i}: source={r.get('_source')}, prompt_preview={prompts}", flush=True)

    sglang_port = 30000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 10000
    sglang_log = Path("/tmp") / f"smoke_mm_sglang_{sglang_port}.log"
    print(f"[smoke] launching sglang on port {sglang_port}, log={sglang_log}", flush=True)

    api_key = "smoke"
    with sglang_server(
        model_path=TEACHER_MODEL,
        port=sglang_port,
        api_key=api_key,
        log_path=sglang_log,
        mem_fraction_static=0.80,
        chunked_prefill_size=2048,
        ready_timeout_s=1800,
    ) as base_url:
        print(f"[smoke] sglang ready at {base_url}; firing requests", flush=True)
        results = asyncio.run(_run(rows, base_url, api_key))

    print("\n=== RESULTS ===")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    n_ok = sum(1 for r in results if r.get("ok"))
    n_truncated = sum(1 for r in results if r.get("ok") and r.get("finish_reason") == "length")
    print(f"\n[smoke] ok={n_ok}/{len(results)} truncated={n_truncated}")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
