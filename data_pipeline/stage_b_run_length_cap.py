"""Stage B: cap NO_OP runs in chat_line.json, write to a NEW output dir.

Reads stage A's output at --source_path. For each segment, applies the
run-length cap rule to the action sequence and writes a filtered
chat_line.json + meta.json to --output_dir/<split>/<segment_id>/. Then
regenerates per-split chat.jsonl from the filtered chat_line.json files.

Frames are NOT copied — chat_line.json embeds absolute frame paths that
already point at stage A's frames. Stage B's output is small (just JSONs)
and references the source dataset's frames in place.

Filter rule: for any contiguous run of M consecutive NO_OP action labels,
keep the first k frames and drop the rest. k = round(k_seconds * target_fps),
floor 1.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

from absl import app, flags

from _manifest import write_manifest

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Filtered dataset output dir.", required=True)
flags.DEFINE_string("source_path", None, "Stage A dataset root.", required=True)
# Stage-specific:
flags.DEFINE_float(
    "k_seconds",
    None,
    "Agent response-time budget. Each NO_OP run is capped at "
    "k_seconds * target_fps frames (rounded, floor 1).",
    required=True,
)
flags.DEFINE_integer(
    "num_workers", None, "Multiprocessing workers (0 = mp.cpu_count()).", required=True
)


def _filter_segment(args: dict) -> dict:
    src_seg_dir = Path(args["src_seg_dir"])
    out_seg_dir = Path(args["out_seg_dir"])
    k_seconds = float(args["k_seconds"])

    src_chat = src_seg_dir / "chat_line.json"
    src_meta = src_seg_dir / "meta.json"
    if not src_chat.exists() or not src_meta.exists():
        return {"segment_id": src_seg_dir.name, "skip_reason": "missing files"}

    chat = json.loads(src_chat.read_text())
    meta = json.loads(src_meta.read_text())

    target_fps = int(meta.get("target_fps", 0))
    if target_fps <= 0:
        return {"segment_id": src_seg_dir.name, "skip_reason": "no target_fps in meta"}
    k_frames = max(1, round(k_seconds * target_fps))

    messages = chat["messages"]
    assert len(messages) % 2 == 0, f"odd message count in {src_chat}"
    n_frames = len(messages) // 2
    actions = [messages[2 * i + 1]["content"][0]["text"] for i in range(n_frames)]

    kept = [False] * n_frames
    run_pos = 0
    for i, a in enumerate(actions):
        if a == "NO_OP":
            run_pos += 1
            if run_pos <= k_frames:
                kept[i] = True
        else:
            run_pos = 0
            kept[i] = True

    kept_indices = [i for i, k in enumerate(kept) if k]
    filtered_messages: list[dict] = []
    for i in kept_indices:
        filtered_messages.append(messages[2 * i])
        filtered_messages.append(messages[2 * i + 1])

    chat["messages"] = filtered_messages
    n_no_op_kept = sum(1 for i in kept_indices if actions[i] == "NO_OP")
    meta["filter"] = {
        "rule": "run_length_cap",
        "k_seconds": k_seconds,
        "k_frames": k_frames,
        "n_frames_pre": n_frames,
        "n_frames_kept": len(kept_indices),
        "n_no_op_pre": meta.get("n_no_op", 0),
        "n_no_op_kept": n_no_op_kept,
    }
    meta["kept_indices"] = kept_indices
    meta["source_segment_dir"] = str(src_seg_dir)

    out_seg_dir.mkdir(parents=True, exist_ok=True)
    (out_seg_dir / "chat_line.json").write_text(json.dumps(chat))
    (out_seg_dir / "meta.json").write_text(json.dumps(meta))

    return {
        "segment_id": src_seg_dir.name,
        "split": meta.get("split", ""),
        "n_frames_pre": n_frames,
        "n_frames_kept": len(kept_indices),
        "n_no_op_pre": meta.get("n_no_op", 0),
        "n_no_op_kept": n_no_op_kept,
        "skip_reason": "",
    }


def _concat_chat_jsonl(split_dir: Path) -> int:
    chat_path = split_dir / "chat.jsonl"
    segment_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    n = 0
    with chat_path.open("w") as out_f:
        for seg_dir in segment_dirs:
            line_path = seg_dir / "chat_line.json"
            if not line_path.exists():
                continue
            out_f.write(line_path.read_text().rstrip("\n") + "\n")
            n += 1
    return n


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_args: list[dict] = []
    for split in ("train", "val", "test"):
        src_split = source_path / split
        out_split = output_dir / split
        if not src_split.is_dir():
            continue
        out_split.mkdir(parents=True, exist_ok=True)
        for seg_dir in sorted(src_split.iterdir()):
            if seg_dir.is_dir():
                seg_args.append(
                    {
                        "src_seg_dir": str(seg_dir),
                        "out_seg_dir": str(out_split / seg_dir.name),
                        "k_seconds": FLAGS.k_seconds,
                    }
                )

    n_workers = FLAGS.num_workers if FLAGS.num_workers > 0 else mp.cpu_count()
    n_workers = min(n_workers, max(1, len(seg_args)))
    print(f"Filtering {len(seg_args)} segments with {n_workers} workers (k={FLAGS.k_seconds}s)")

    summaries: list[dict] = []
    with mp.Pool(processes=n_workers) as pool:
        for s in pool.imap_unordered(_filter_segment, seg_args):
            summaries.append(s)
            if len(summaries) % 200 == 0:
                print(f"  processed {len(summaries)}/{len(seg_args)}", flush=True)

    skipped = [s for s in summaries if s.get("skip_reason")]
    if skipped:
        print(f"WARNING: {len(skipped)} segments skipped:")
        for s in skipped[:10]:
            print(f"  {s['segment_id']}: {s['skip_reason']}")

    total_pre = sum(s.get("n_frames_pre", 0) for s in summaries)
    total_kept = sum(s.get("n_frames_kept", 0) for s in summaries)
    no_op_pre = sum(s.get("n_no_op_pre", 0) for s in summaries)
    no_op_kept = sum(s.get("n_no_op_kept", 0) for s in summaries)

    print()
    print("=== Aggregate ===")
    print(f"  segments processed: {len(summaries) - len(skipped)}")
    print(f"  frames pre-filter:  {total_pre:,}")
    print(f"  frames kept:        {total_kept:,}  ({100 * total_kept / max(total_pre, 1):.1f}%)")
    print(f"  NO_OPs pre-filter:  {no_op_pre:,}")
    print(f"  NO_OPs kept:        {no_op_kept:,}")

    print()
    print("Regenerating per-split chat.jsonl ...")
    for split in ("train", "val", "test"):
        out_split = output_dir / split
        if not out_split.is_dir():
            continue
        n = _concat_chat_jsonl(out_split)
        print(f"  {split}: wrote {n} sessions to {out_split / 'chat.jsonl'}")

    write_manifest(
        output_dir,
        stage="run_length_cap",
        params={"k_seconds": FLAGS.k_seconds},
        inputs={"source": str(source_path)},
        stats={
            "segments_processed": len(summaries) - len(skipped),
            "segments_skipped": len(skipped),
            "frames_pre": total_pre,
            "frames_kept": total_kept,
            "n_no_op_pre": no_op_pre,
            "n_no_op_kept": no_op_kept,
            "split_session_counts": {
                split: sum(
                    1 for s in summaries if s.get("split") == split and not s.get("skip_reason")
                )
                for split in ("train", "val", "test")
            },
        },
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
