"""On-policy text-replay prep: replace assistant turns with teacher-generated ones.

Reads an existing chat.jsonl (e.g. the prepped smoltalk2 artifact),
extracts each row's prefix up to the first assistant turn, batch-generates
ONE assistant completion per row from a teacher model via SGLang's
OpenAI-compatible endpoint, and writes a new chat.jsonl under
``<output_dir>/train/chat.jsonl`` in the same {messages, _source} schema
consumed by stage_c (omegalax/scripts/compile_sft_dataset.py).

Run inside the eval repo's uv venv (which already pins sglang + openai +
flashinfer); the script itself lives here for reproducibility alongside
the other prep scripts.

Single-turn protocol: the source chat.jsonl rows may be multi-turn but we
always condition on everything strictly before the first assistant
message and emit a single (prefix, teacher_completion) pair. Rows whose
first message is already an assistant turn, or that have no user turn
before the first assistant, are skipped. This is intentional: we want
the assistant-turn distribution to be on-policy w.r.t. the teacher we
want to preserve, not a mixture of human-written prefixes and teacher
suffixes within one row.

Why we don't generate every assistant turn in a multi-turn conversation:
that would compound teacher hallucinations across turns and make the
output increasingly off-distribution from the smoltalk2 prompt mix. For
testing the on-distribution-vs-off-distribution hypothesis a clean
single-turn signal is enough.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from absl import app, flags

# We invoke this script with ``uv --project=/fast/home/franz.srambical/eval``
# so eval/_sglang.py and eval/_manifest helpers are import-resolvable via
# the eval repo's source dir on PYTHONPATH.
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
    "Path to the source chat.jsonl whose user prefixes we re-complete.",
    required=True,
)
flags.DEFINE_string(
    "teacher_model",
    "Qwen/Qwen3-VL-2B-Instruct",
    "HF model_id served by SGLang. Use the off-shelf instruct model whose "
    "behaviour we want the BC student to preserve.",
)
flags.DEFINE_integer(
    "max_samples",
    100000,
    "Maximum prompts to generate completions for (0 = full source). "
    "Subsamples deterministically from the head of the source jsonl.",
)
flags.DEFINE_float("temperature", 0.7, "Sampling temperature.")
flags.DEFINE_float("top_p", 0.95, "Nucleus sampling top_p.")
flags.DEFINE_integer("max_tokens", 1280, "Max generated tokens per completion.")
flags.DEFINE_integer("seed", 0, "Sampling seed (passed to OpenAI request).")
flags.DEFINE_integer(
    "concurrency",
    64,
    "Concurrent in-flight chat-completion requests against the SGLang "
    "endpoint. SGLang batches internally; this just keeps the queue full.",
)
flags.DEFINE_integer(
    "sglang_port",
    0,
    "SGLang port. 0 = derive from $SLURM_JOB_ID at runtime "
    "(30000 + jid % 10000) so concurrent jobs on the same node don't "
    "collide.",
)
flags.DEFINE_string("sglang_api_key", "inspectai", "SGLang api key.")
flags.DEFINE_float("mem_fraction_static", 0.80, "SGLang mem fraction.")
flags.DEFINE_integer("chunked_prefill_size", 2048, "SGLang chunked prefill.")
flags.DEFINE_integer(
    "ready_timeout_s",
    1500,
    "Max seconds to wait for SGLang /health_generate. Cold flashinfer JIT "
    "compiles can take several minutes.",
)
flags.DEFINE_integer(
    "log_every",
    500,
    "Print throughput stats every N completed prompts.",
)


def _extract_prefix_messages(messages: list) -> list | None:
    """Return everything strictly before the first assistant turn, or None.

    ``None`` means the row is unusable (no user turn before any assistant
    turn). We keep system prompts and any consecutive user/tool turns.
    """
    prefix: list = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            break
        prefix.append(m)
    if not any(m.get("role") == "user" for m in prefix):
        return None
    return prefix


def _content_to_str(content) -> str:
    """Flatten Qwen-style block-list content into a plain string for the
    OpenAI chat-completions API. SGLang's OAI shim accepts both, but
    coercing to ``str`` here keeps the wire format minimal.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def _normalize_for_openai(messages: list) -> list:
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        out.append({"role": role, "content": _content_to_str(m.get("content", ""))})
    return out


async def _gen_one(
    client: openai.AsyncOpenAI,
    *,
    teacher_model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> str | None:
    try:
        resp = await client.chat.completions.create(
            model=teacher_model,
            messages=_normalize_for_openai(messages),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception as e:
        print(f"[gen] request failed: {e!r}", flush=True)
        return None
    if not resp.choices:
        return None
    msg = resp.choices[0].message
    return getattr(msg, "content", None) or ""


async def _run(rows: list[dict], cfg: dict, out_path: Path) -> dict:
    base_url = cfg["base_url"]
    client = openai.AsyncOpenAI(base_url=base_url, api_key=cfg["api_key"])
    sem = asyncio.Semaphore(cfg["concurrency"])

    n_done = 0
    n_failed = 0
    t_start = time.time()
    last_log_t = t_start

    out_f = out_path.open("w")
    try:

        async def _bound(idx: int, row: dict, prefix: list):
            nonlocal n_done, n_failed, last_log_t
            async with sem:
                completion = await _gen_one(
                    client,
                    teacher_model=cfg["teacher_model"],
                    messages=prefix,
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    max_tokens=cfg["max_tokens"],
                    seed=cfg["seed"] + idx,
                )
            if completion is None:
                n_failed += 1
                return
            new_messages = [*list(prefix), {"role": "assistant", "content": completion}]
            record = {
                "messages": new_messages,
                "_source": row.get("_source", "") + "::onpolicy_qwen3vl2b_instruct",
            }
            out_f.write(json.dumps(record, ensure_ascii=False))
            out_f.write("\n")
            n_done += 1
            if n_done % cfg["log_every"] == 0:
                now = time.time()
                rate = cfg["log_every"] / max(now - last_log_t, 1e-6)
                cum_rate = n_done / max(now - t_start, 1e-6)
                print(
                    f"[gen] done={n_done} failed={n_failed} rate={rate:.1f}/s cum={cum_rate:.1f}/s",
                    flush=True,
                )
                last_log_t = now

        tasks = []
        for idx, row in enumerate(rows):
            prefix = _extract_prefix_messages(row.get("messages", []))
            if prefix is None:
                n_failed += 1
                continue
            tasks.append(asyncio.create_task(_bound(idx, row, prefix)))
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        out_f.close()

    return {
        "n_done": n_done,
        "n_failed": n_failed,
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
    print(f"[gen] loaded {len(rows)} source rows from {src_path}", flush=True)

    if FLAGS.sglang_port == 0:
        jid = int(os.environ.get("SLURM_JOB_ID", "0"))
        sglang_port = 30000 + (jid % 10000)
        print(
            f"[gen] auto-derived sglang_port={sglang_port} from SLURM_JOB_ID={jid}",
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
        stage="replay_prep_smoltalk2_onpolicy",
        params={
            "source_chat_jsonl": str(src_path),
            "teacher_model": FLAGS.teacher_model,
            "max_samples": FLAGS.max_samples,
            "temperature": FLAGS.temperature,
            "top_p": FLAGS.top_p,
            "max_tokens": FLAGS.max_tokens,
            "seed": FLAGS.seed,
            "concurrency": FLAGS.concurrency,
        },
        inputs={"source_chat_jsonl": str(src_path)},
        stats={
            "n_loaded": len(rows),
            "n_done": stats["n_done"],
            "n_failed": stats["n_failed"],
            "elapsed_gen_s": stats["elapsed_s"],
            "elapsed_total_s": int(time.time() - t_total_start),
        },
    )
    print(
        f"[gen] wrote {out_path} (done={stats['n_done']}, failed={stats['n_failed']})",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)
