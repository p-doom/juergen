"""On-policy multimodal-replay prep: replace assistant turns with
teacher-generated ones over FineVision (image+text) prompts.

Multimodal sibling of ``generate_onpolicy_completions.py``. Reads the
prepped FineVision chat.jsonl, extracts each row's prefix up to the
first assistant turn, sends it through SGLang's OAI-compatible
vision-chat-completions endpoint with image blocks encoded as base64
data URLs (file:// is unreliable across sglang versions), captures the
returned ``finish_reason`` per row, and writes a new chat.jsonl with
the same {messages, _source} schema consumed by stage_c.

Differences from the smoltalk2 (text-only) version, learned from the
last round:

  * we capture ``finish_reason`` per row at the source — no separate
    filter pipeline needed downstream.
  * truncated rows (finish_reason == "length") can optionally be
    dropped at write-time via ``--drop_truncated`` (default True). This
    bakes the lesson from the smoltalk2 38% post-hoc-filter into a
    single-pass design.
  * images are loaded once per row, base64-encoded, and shipped via
    ``data:image/...;base64,...`` URLs. sglang's OAI shim accepts these
    consistently across the JustinTong0323 fork's release window.
  * concurrency is configurable separately from text — vision encoding
    plus image decode adds to per-request work; default is half of
    text's 64.

Run inside the eval repo's uv venv (sglang + openai pinned there).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

from absl import app, flags

sys.path.insert(0, "/fast/home/franz.srambical/eval")
sys.path.insert(0, "/fast/home/franz.srambical/data_pipeline")
import openai
from _sglang import sglang_server  # type: ignore[import-not-found]

from _manifest import write_manifest  # type: ignore[import-not-found]

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Output dataset root.", required=True)
flags.DEFINE_string(
    "source_chat_jsonl",
    None,
    "Path to the source FineVision chat.jsonl whose prefixes we re-complete.",
    required=True,
)
flags.DEFINE_string(
    "teacher_model",
    "Qwen/Qwen3-VL-2B-Instruct",
    "HF model_id served by SGLang. Must be the off-shelf VLM we want the BC student to preserve.",
)
flags.DEFINE_integer(
    "max_samples",
    50000,
    "Maximum prompts to generate completions for (0 = full source). "
    "Subsamples deterministically from the head of the source jsonl.",
)
flags.DEFINE_float("temperature", 0.7, "Sampling temperature.")
flags.DEFINE_float("top_p", 0.95, "Nucleus sampling top_p.")
flags.DEFINE_integer(
    "max_tokens",
    2048,
    "Max generated tokens. Higher than smoltalk2's 1280 because OlmOCR "
    "full-page transcriptions in FineVision frequently exceed 1k tokens.",
)
flags.DEFINE_integer("seed", 0, "Sampling seed (passed to OpenAI request).")
flags.DEFINE_integer(
    "concurrency",
    16,
    "Concurrent in-flight chat-completion requests against the SGLang "
    "endpoint. Half the text default because vision encoding adds work.",
)
flags.DEFINE_boolean(
    "drop_truncated",
    True,
    "If True, rows whose generation hit the max_tokens cap "
    "(finish_reason == 'length') are dropped at write-time. The smoltalk2 "
    "round taught us that truncated reasoning chains are bad replay "
    "material — they teach the student to stop mid-thought. Drop them at "
    "the source instead of via a post-hoc filter pipeline.",
)
flags.DEFINE_integer(
    "sglang_port",
    0,
    "SGLang port. 0 = derive from $SLURM_JOB_ID (30000 + jid % 10000).",
)
flags.DEFINE_string("sglang_api_key", "inspectai", "SGLang api key.")
flags.DEFINE_float("mem_fraction_static", 0.80, "SGLang mem fraction.")
flags.DEFINE_integer("chunked_prefill_size", 2048, "SGLang chunked prefill.")
flags.DEFINE_integer(
    "ready_timeout_s",
    1800,
    "Max seconds to wait for SGLang /health_generate.",
)
flags.DEFINE_integer(
    "log_every",
    200,
    "Print throughput + truncation-rate stats every N completed prompts.",
)


def _img_to_data_url(path: str) -> str:
    p = Path(path)
    b = p.read_bytes()
    ext = p.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(b).decode('ascii')}"


def _row_prefix_to_oai(row: dict) -> tuple[list, list]:
    """Return (oai_messages, original_prefix_messages_in_chat_schema).

    The first list is what we send to sglang; the second is what we
    write back into the new chat.jsonl alongside the model-generated
    assistant turn (preserves the original schema with image blocks).
    """
    oai = []
    prefix = []
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            break
        prefix.append(m)
        c = m.get("content")
        content_blocks: list = []
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
            oai.append({"role": m.get("role", "user"), "content": content_blocks})
    return oai, prefix


async def _gen_one(
    client: openai.AsyncOpenAI,
    *,
    teacher_model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> tuple[str | None, str | None]:
    try:
        resp = await client.chat.completions.create(
            model=teacher_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception as e:
        print(f"[gen_mm] request failed: {e!r}", flush=True)
        return None, None
    if not resp.choices:
        return None, None
    choice = resp.choices[0]
    return getattr(choice.message, "content", None) or "", choice.finish_reason


async def _run(rows: list[dict], cfg: dict, out_path: Path) -> dict:
    base_url = cfg["base_url"]
    client = openai.AsyncOpenAI(base_url=base_url, api_key=cfg["api_key"])
    sem = asyncio.Semaphore(cfg["concurrency"])

    n_ok = 0
    n_failed = 0
    n_truncated = 0
    n_written = 0
    t_start = time.time()
    last_log_t = t_start

    out_f = out_path.open("w")
    try:

        async def _bound(idx: int, row: dict):
            nonlocal n_ok, n_failed, n_truncated, n_written, last_log_t
            try:
                oai_msgs, prefix = _row_prefix_to_oai(row)
            except Exception as e:
                print(f"[gen_mm] row {idx} preprocessing failed: {e!r}", flush=True)
                n_failed += 1
                return
            if not oai_msgs:
                n_failed += 1
                return
            async with sem:
                completion, finish_reason = await _gen_one(
                    client,
                    teacher_model=cfg["teacher_model"],
                    messages=oai_msgs,
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    max_tokens=cfg["max_tokens"],
                    seed=cfg["seed"] + idx,
                )
            if completion is None:
                n_failed += 1
                return
            n_ok += 1
            if finish_reason == "length":
                n_truncated += 1
                if cfg["drop_truncated"]:
                    return
            new_messages = [*list(prefix), {"role": "assistant", "content": completion}]
            record = {
                "messages": new_messages,
                "_source": (row.get("_source", "") or "") + "::onpolicy_qwen3vl2b_instruct",
                "_finish_reason": finish_reason,
            }
            if "_finevision_config" in row:
                record["_finevision_config"] = row["_finevision_config"]
            out_f.write(json.dumps(record, ensure_ascii=False))
            out_f.write("\n")
            n_written += 1
            if n_ok % cfg["log_every"] == 0:
                now = time.time()
                rate = cfg["log_every"] / max(now - last_log_t, 1e-6)
                cum_rate = n_ok / max(now - t_start, 1e-6)
                trunc_rate = n_truncated / max(n_ok, 1)
                print(
                    f"[gen_mm] ok={n_ok} failed={n_failed} truncated={n_truncated} "
                    f"({trunc_rate:.1%}) written={n_written} "
                    f"rate={rate:.1f}/s cum={cum_rate:.1f}/s",
                    flush=True,
                )
                last_log_t = now

        tasks = [asyncio.create_task(_bound(idx, row)) for idx, row in enumerate(rows)]
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        out_f.close()

    return {
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_truncated": n_truncated,
        "n_written": n_written,
        "elapsed_s": int(time.time() - t_start),
    }


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "chat.jsonl"

    src_path = Path(FLAGS.source_chat_jsonl)
    if not src_path.is_file():
        raise FileNotFoundError(f"source_chat_jsonl not found: {src_path}")

    rows: list[dict] = []
    with src_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if FLAGS.max_samples and len(rows) >= FLAGS.max_samples:
                break
    print(f"[gen_mm] loaded {len(rows)} source rows from {src_path}", flush=True)

    if FLAGS.sglang_port == 0:
        jid = int(os.environ.get("SLURM_JOB_ID", "0"))
        sglang_port = 30000 + (jid % 10000)
        print(
            f"[gen_mm] auto-derived sglang_port={sglang_port} from SLURM_JOB_ID={jid}",
            flush=True,
        )
    else:
        sglang_port = FLAGS.sglang_port

    sglang_log = output_dir / "sglang_server.log"

    cfg = {
        "teacher_model": FLAGS.teacher_model,
        "temperature": FLAGS.temperature,
        "top_p": FLAGS.top_p,
        "max_tokens": FLAGS.max_tokens,
        "seed": FLAGS.seed,
        "concurrency": FLAGS.concurrency,
        "log_every": FLAGS.log_every,
        "api_key": FLAGS.sglang_api_key,
        "drop_truncated": FLAGS.drop_truncated,
    }

    t_total_start = time.time()
    with sglang_server(
        model_path=FLAGS.teacher_model,
        port=sglang_port,
        api_key=FLAGS.sglang_api_key,
        log_path=sglang_log,
        mem_fraction_static=FLAGS.mem_fraction_static,
        chunked_prefill_size=FLAGS.chunked_prefill_size,
        ready_timeout_s=FLAGS.ready_timeout_s,
    ) as base_url:
        cfg["base_url"] = base_url
        stats = asyncio.run(_run(rows, cfg, out_path))

    write_manifest(
        output_dir,
        stage="replay_prep_finevision_mixed_onpolicy",
        params={
            "source_chat_jsonl": str(src_path),
            "teacher_model": FLAGS.teacher_model,
            "max_samples": FLAGS.max_samples,
            "temperature": FLAGS.temperature,
            "top_p": FLAGS.top_p,
            "max_tokens": FLAGS.max_tokens,
            "seed": FLAGS.seed,
            "concurrency": FLAGS.concurrency,
            "drop_truncated": FLAGS.drop_truncated,
        },
        inputs={"source_chat_jsonl": str(src_path)},
        stats={
            "n_loaded": len(rows),
            **stats,
            "elapsed_total_s": int(time.time() - t_total_start),
        },
    )
    print(
        f"[gen_mm] wrote {out_path} "
        f"(ok={stats['n_ok']}, failed={stats['n_failed']}, "
        f"truncated={stats['n_truncated']}, written={stats['n_written']})",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)
