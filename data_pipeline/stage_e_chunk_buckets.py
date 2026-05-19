"""Stage E: chunk into fixed token-budget buckets (8k–128k).

Reads Stage B cleaned frames and Stage C/D annotations. For each bucket size,
the recording is sliced into consecutive chunks of approximately that token
length. Kimi K2.6 generates a per-chunk instruction at the appropriate
granularity:

  8k  (~6s)  → atomic UI primitive ("Open the Config tab")
  16k (~12s) → short action sequence ("Search Google for X")
  32k (~25s) → step-level ("Inspect temperature hyperparameters")
  64k (~50s) → multi-step ("Compare distillation training runs")
  128k(~103s)→ sub-goal ("Research temperature effects")

Input:  Stage B output (--stage_a_dir), Stage C captions, Stage D task tree
Output: one JSONL per bucket in --output_dir, each line is a training-ready sample.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from absl import app, flags
from openai import OpenAI

FLAGS = flags.FLAGS

flags.DEFINE_string("stage_a_dir", None, "Stage A output dir.", required=True)
flags.DEFINE_string("task_tree_path", None, "Path to task_tree.json.", required=True)
flags.DEFINE_string("captions_path", None, "Path to captions.json.", required=True)
flags.DEFINE_string(
    "keyframes_meta_path", None, "Path to keyframes_meta.json.", required=True
)
flags.DEFINE_string("output_dir", None, "Output dir for bucket JSONL files.", required=True)
flags.DEFINE_string(
    "recording_id",
    "d64dc788-6466-4ade-8f4e-b55c21d8311b",
    "Recording ID prefix.",
)
flags.DEFINE_string("split", "train", "Stage A split.")
flags.DEFINE_integer("tokens_per_frame", 634, "Tokens per frame (vision + overhead).")
flags.DEFINE_integer("fps", 2, "Frames per second in Stage A output.")
flags.DEFINE_multi_integer(
    "buckets", [8192, 16384, 32768, 65536, 131072], "Token bucket sizes."
)
flags.DEFINE_string("model", "kimi-k2.6", "LLM model for instruction generation.")
flags.DEFINE_boolean("dry_run", False, "Skip LLM calls, use placeholder instructions.")
flags.DEFINE_integer("llm_batch_size", 40, "Chunks per LLM instruction-generation call.")

BUCKET_LABELS = {
    8192: "8k",
    16384: "16k",
    32768: "32k",
    65536: "64k",
    131072: "128k",
}

GRANULARITY_DESCRIPTIONS = {
    "8k": (
        "an atomic UI action (1-5 words). Examples: 'Open a new browser tab', "
        "'Click the Config button', 'Type sbatch launch.sh in the terminal', "
        "'Scroll down in the wandb dashboard'. Be extremely specific to what "
        "actually happens in the captions."
    ),
    "16k": (
        "a short action sequence (5-12 words). Examples: 'Search Google for "
        "on-policy distillation papers', 'Open the wandb run config and check "
        "temperature'. Reference specific names/URLs from the captions."
    ),
    "32k": (
        "a step-level task (8-20 words). Examples: 'Inspect temperature "
        "hyperparameters for the distillation run in wandb', 'Review the "
        "upload-orphans dry-run output and approve execution'. Should describe "
        "the complete mini-task being performed."
    ),
    "64k": (
        "a multi-step goal (10-25 words). Examples: 'Compare validation "
        "metrics across distillation runs and investigate anomalies', "
        "'Set up and configure the Kimi inference server on the cluster'. "
        "Should capture the broader intent spanning multiple actions."
    ),
    "128k": (
        "a high-level sub-goal (10-30 words). Examples: 'Research temperature "
        "effects on on-policy distillation by reviewing papers and analyzing "
        "training metrics', 'Diagnose training instability at step 10 by "
        "correlating metrics with hyperparameter schedules'. Should describe "
        "the overarching objective."
    ),
}


def _load_all_frames(stage_a_dir: Path, split: str, recording_id: str) -> list[dict]:
    frames = []
    for seg_idx in range(100):
        seg_name = f"recording_{recording_id}_seg{seg_idx:04d}"
        chat_path = stage_a_dir / split / seg_name / "chat_line.json"
        if not chat_path.exists():
            continue
        data = json.loads(chat_path.read_text())
        msgs = data["messages"]
        for i in range(0, len(msgs), 2):
            user_msg = msgs[i]
            asst_msg = msgs[i + 1] if i + 1 < len(msgs) else None
            if asst_msg is None:
                continue
            frames.append({
                "seg_idx": seg_idx,
                "frame_idx": i // 2,
                "user": user_msg,
                "assistant": asst_msg,
            })
    return frames


def _build_caption_timeline(
    captions: list[dict],
    kf_meta: list[dict],
    fps_annotation: int = 5,
    keyframe_every_n: int = 75,
) -> list[dict]:
    """Build time-indexed caption list. Each entry has seg_idx, time_in_seg, caption."""
    seg_kf_count: dict[int, int] = {}
    entries = []
    for kf in kf_meta:
        seg = kf["seg_idx"]
        local_k = seg_kf_count.get(seg, 0)
        seg_kf_count[seg] = local_k + 1
        time_in_seg = (local_k * keyframe_every_n) / fps_annotation
        caption = captions[kf["idx"]].get("caption", "")
        entries.append({
            "seg_idx": seg,
            "time_in_seg": time_in_seg,
            "caption": caption,
        })
    return entries


def _captions_for_chunk(
    caption_timeline: list[dict],
    chunk_frames: list[dict],
    fps: int,
) -> list[str]:
    if not chunk_frames:
        return []
    first_seg = chunk_frames[0]["seg_idx"]
    last_seg = chunk_frames[-1]["seg_idx"]
    first_time = chunk_frames[0]["frame_idx"] / fps
    last_time = chunk_frames[-1]["frame_idx"] / fps

    result = []
    for ce in caption_timeline:
        seg = ce["seg_idx"]
        if seg < first_seg or seg > last_seg:
            continue
        if seg == first_seg and ce["time_in_seg"] < first_time - 15:
            continue
        if seg == last_seg and ce["time_in_seg"] > last_time + 15:
            continue
        if ce["caption"]:
            result.append(ce["caption"])
    return result


def _generate_instructions_batch(
    client: OpenAI,
    model: str,
    chunks_with_captions: list[tuple[int, list[str]]],
    bucket_label: str,
    task_tree: dict,
) -> list[str]:
    """Call Kimi to generate per-chunk instructions for a batch."""
    granularity = GRANULARITY_DESCRIPTIONS[bucket_label]
    goal = task_tree.get("goal", "")

    lines = []
    for chunk_idx, caps in chunks_with_captions:
        caps_text = " | ".join(caps) if caps else "(no captions available)"
        lines.append(f"[{chunk_idx}] {caps_text}")

    prompt = (
        f"You are generating training instructions for a computer-use agent. "
        f"The recording session's overall goal: \"{goal}\"\n\n"
        f"Below are {len(chunks_with_captions)} consecutive chunks from this "
        f"session. Each chunk shows what the user was doing (from screen captions). "
        f"For each chunk, write ONE imperative instruction at this granularity: "
        f"{granularity}\n\n"
        f"Rules:\n"
        f"- Instructions must be imperative ('Open X', 'Click Y', NOT 'The user opens X')\n"
        f"- Be specific: use actual file names, URLs, app names from the captions\n"
        f"- Each instruction must be different — even if adjacent chunks show similar activity, "
        f"focus on what distinguishes THIS chunk\n"
        f"- If the user is idle or watching something, describe what they're monitoring\n\n"
        f"Chunks:\n" + "\n".join(lines) + "\n\n"
        f"Return one instruction per chunk in this exact format:\n"
        f"[0] instruction\n"
        f"[1] instruction\n..."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=16384,
    )
    raw = resp.choices[0].message.content or ""
    if not raw:
        raw = getattr(resp.choices[0].message, "reasoning_content", "") or ""
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()

    instructions = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            try:
                num = int(line[1:bracket_end])
                instructions[num] = line[bracket_end + 1:].strip()
            except ValueError:
                continue

    result = []
    for chunk_idx, _ in chunks_with_captions:
        result.append(instructions.get(chunk_idx, "Continue working on the current task."))
    return result


def _make_system_message(instruction: str, context: str = "") -> dict:
    text = f"You are a computer-use agent. Complete the following task: {instruction}"
    if context:
        text += f"\n\nContext: {context}"
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def main(_) -> None:
    stage_a_dir = Path(FLAGS.stage_a_dir)
    task_tree = json.loads(Path(FLAGS.task_tree_path).read_text())
    captions = json.loads(Path(FLAGS.captions_path).read_text())
    kf_meta = json.loads(Path(FLAGS.keyframes_meta_path).read_text())
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frames...", flush=True)
    all_frames = _load_all_frames(stage_a_dir, FLAGS.split, FLAGS.recording_id)
    print(f"Loaded {len(all_frames)} frames", flush=True)

    caption_timeline = _build_caption_timeline(captions, kf_meta)
    print(f"Built {len(caption_timeline)} caption entries", flush=True)

    client = None
    if not FLAGS.dry_run:
        endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError(
                "Set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_API_KEY "
                "or use --dry_run"
            )
        client = OpenAI(base_url=endpoint, api_key=api_key)

    sys_msg_tokens = 60

    for bucket in FLAGS.buckets:
        label = BUCKET_LABELS.get(bucket, f"{bucket // 1024}k")
        frames_per_chunk = (bucket - sys_msg_tokens) // FLAGS.tokens_per_frame
        chunk_duration = frames_per_chunk / FLAGS.fps

        print(
            f"\n=== {label} bucket: {frames_per_chunk} frames/chunk, "
            f"{chunk_duration:.1f}s ===",
            flush=True,
        )

        chunks = []
        for start in range(0, len(all_frames), frames_per_chunk):
            end = min(start + frames_per_chunk, len(all_frames))
            chunk_frames = all_frames[start:end]
            if len(chunk_frames) < frames_per_chunk * 0.5:
                continue
            caps = _captions_for_chunk(caption_timeline, chunk_frames, FLAGS.fps)
            chunks.append({"frames": chunk_frames, "captions": caps})

        if FLAGS.dry_run:
            instructions = [
                f"[DRY RUN] {label} chunk {i}: {' | '.join(c['captions'][:2])}"
                for i, c in enumerate(chunks)
            ]
        else:
            print(f"  Generating instructions for {len(chunks)} chunks...", flush=True)
            instructions = []
            batch_size = FLAGS.llm_batch_size
            for b_start in range(0, len(chunks), batch_size):
                b_end = min(b_start + batch_size, len(chunks))
                batch = [(i, chunks[i]["captions"]) for i in range(b_start, b_end)]
                print(
                    f"    batch {b_start // batch_size + 1}/"
                    f"{(len(chunks) + batch_size - 1) // batch_size} "
                    f"({len(batch)} chunks)...",
                    flush=True,
                )
                t0 = time.time()
                batch_instructions = _generate_instructions_batch(
                    client, FLAGS.model, batch, label, task_tree,
                )
                elapsed = time.time() - t0
                print(f"      done in {elapsed:.1f}s", flush=True)
                instructions.extend(batch_instructions)

        samples = []
        for i, chunk in enumerate(chunks):
            instruction = instructions[i]
            sys_msg = _make_system_message(instruction)
            messages = [sys_msg]
            for f in chunk["frames"]:
                messages.append(f["user"])
                messages.append(f["assistant"])

            n_frames = len(chunk["frames"])
            samples.append({
                "sample_id": f"{label}_{i:04d}",
                "bucket": label,
                "bucket_tokens": bucket,
                "instruction": instruction,
                "n_frames": n_frames,
                "est_tokens": sys_msg_tokens + n_frames * FLAGS.tokens_per_frame,
                "duration_s": n_frames / FLAGS.fps,
                "start_seg": chunk["frames"][0]["seg_idx"],
                "end_seg": chunk["frames"][-1]["seg_idx"],
                "messages": messages,
            })

        out_path = output_dir / f"samples_{label}.jsonl"
        with out_path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        print(f"  {len(samples)} samples written to {out_path}", flush=True)
        for s in samples[:5]:
            print(f"    {s['sample_id']}: {s['duration_s']:.0f}s — {s['instruction'][:80]}", flush=True)
        if len(samples) > 5:
            print(f"    ... and {len(samples) - 5} more", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    app.run(main)
