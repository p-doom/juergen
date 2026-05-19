"""Stage D: build hierarchical task tree from Stage C captions.

Reads captions.json from Stage C and calls Kimi K2.6 (text-only) to produce
a hierarchical decomposition: goal → sub-goals → steps, each with imperative
instructions and keyframe ranges.

Input:  Stage C output dir (--captions_path pointing to captions.json)
Output: <output_dir>/task_tree.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from absl import app, flags
from openai import OpenAI

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Output directory for task tree.", required=True)
flags.DEFINE_string("captions_path", None, "Path to captions.json from Stage C.", required=True)
flags.DEFINE_string("model", "kimi-k2.6", "Azure AI Foundry model deployment name.")
flags.DEFINE_float("temperature", 0.3, "LLM temperature.")
flags.DEFINE_integer("max_tokens", 32768, "Max output tokens for task-tree call.")
flags.DEFINE_boolean("dry_run", False, "Print prompt but skip LLM call.")


def _build_task_tree_messages(captions: list[dict]) -> list[dict]:
    timeline_lines = []
    current_seg = -1
    for cap in captions:
        if cap["seg_idx"] != current_seg:
            current_seg = cap["seg_idx"]
            timeline_lines.append(f"\n--- Segment {current_seg} (5-minute recording chunk) ---")
        timeline_lines.append(f"[kf{cap['idx']}] {cap['caption']}")

    timeline = "\n".join(timeline_lines)

    prompt = (
        "Below is a timeline of captioned keyframes from a screen recording session "
        "of a software engineer working. Keyframes are sampled every ~5 seconds, "
        "grouped by 5-minute recording segments.\n\n"
        f"{timeline}\n\n"
        "From this timeline, produce a hierarchical task decomposition as JSON:\n\n"
        "1. **goal**: One imperative sentence describing the overall task for this "
        "entire session (what an orchestrator would tell an agent to do).\n"
        "2. **sub_goals**: A list of major phases. Each sub-goal has:\n"
        "   - \"instruction\": imperative sentence (a task you'd give an agent)\n"
        "   - \"start_kf\": first keyframe index\n"
        "   - \"end_kf\": last keyframe index\n"
        "   - \"segments\": list of segment indices this sub-goal spans\n"
        "   - \"completed\": boolean — did the user finish this sub-task?\n"
        "   - \"steps\": list of atomic steps within this sub-goal, each with:\n"
        "     - \"instruction\": imperative sentence\n"
        "     - \"start_kf\": first keyframe index\n"
        "     - \"end_kf\": last keyframe index\n\n"
        "Rules:\n"
        "- Instructions must be imperative (\"Open the file\", \"Run the test\"), "
        "not descriptive (\"The user opens the file\").\n"
        "- Every keyframe must belong to exactly one step and one sub-goal.\n"
        "- Sub-goals should span consecutive segments. Steps span consecutive keyframes.\n"
        "- Be specific: reference actual file names, URLs, commands, variable names "
        "you saw in the captions.\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )

    return [{"role": "user", "content": prompt}]


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _call_llm(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    if not content:
        content = getattr(msg, "reasoning_content", "") or ""
    return _strip_think(content)


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captions = json.loads(Path(FLAGS.captions_path).read_text())
    print(f"Loaded {len(captions)} captions", flush=True)

    messages = _build_task_tree_messages(captions)

    if FLAGS.dry_run:
        print("Dry run — prompt:", flush=True)
        print(messages[0]["content"][:2000], flush=True)
        return

    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_API_KEY or use --dry_run"
        )
    client = OpenAI(base_url=endpoint, api_key=api_key)

    print("Building task tree from captions...", flush=True)
    t0 = time.time()
    raw = _call_llm(client, messages, FLAGS.model, FLAGS.temperature, FLAGS.max_tokens)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, {len(raw)} chars", flush=True)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        tree = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  WARNING: failed to parse task tree JSON: {e}", flush=True)
        (output_dir / "task_tree_raw.txt").write_text(raw)
        print(f"  raw response saved to task_tree_raw.txt", flush=True)
        tree = {"_raw": raw, "_parse_error": str(e)}

    tree_path = output_dir / "task_tree.json"
    tree_path.write_text(json.dumps(tree, indent=2))
    print(f"Saved task tree to {tree_path}", flush=True)

    n_sub_goals = len(tree.get("sub_goals", []))
    n_steps = sum(len(sg.get("steps", [])) for sg in tree.get("sub_goals", []))
    print(f"Done: {n_sub_goals} sub-goals, {n_steps} steps", flush=True)


if __name__ == "__main__":
    app.run(main)
