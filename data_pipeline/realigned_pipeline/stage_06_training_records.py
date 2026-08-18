"""Training-record stage (payload-free): build inline SFT records from chat.jsonl.

Wrapper around omegalax/scripts/build_sft_records_from_chat.py. Payload-free
variant of stage_d_chunk_index.py: reads the stage-04 conversations dataset's
single <source>/chat.jsonl directly (NO grain payload) and writes self-contained
inline records per split under <output_dir>/<split>/. Each record IS a training
example (message slice with ar:// image refs preserved), not a pointer into a
shared payload; the stage 01 master image store is unchanged.

The recording-level train/val split is applied HERE via --val_fraction (> 0 ->
train/ + val/; 0 -> train/ only). Because the split lives here, the stage-05
measure cache stays split-agnostic (a single message_lengths.jsonl) and is
reused for every split, so changing --val_fraction re-runs only this stage and
never re-tokenizes.

Reuses the measure-stage cache (--message_lengths_path) so re-running at a
different max_length / overflow_mode / val_fraction never re-tokenizes.

--min_length puts a floor under the same budget --max_length caps: a chunk is
written only when min_length <= measured_length <= max_length. It exists for
overflow_mode=split, where a conversation's last chunk is whatever is left over
and can be a one-frame stub; the floor drops those instead of training on them.

Each split's builder also writes token_stats.json; this stage turns its
``vision_variability.vision_tokens_per_chunk`` frequency table into a weighted
percentile CDF (<out>/<split>/vision_tokens_percentile_cdf.png) and folds the
key percentiles into manifest.json, so how much of a chunk's budget goes to
pixels is visible without opening the stats blob.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from absl import app, flags

try:
    import matplotlib

    # Headless: this stage runs as a SLURM batch job and only ever writes a PNG.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional -- the percentiles still reach the manifest
    plt = None

# Make the ``realigned_pipeline`` package importable when this stage is run
# directly as a script (mirrors the other stages' PYTHONPATH setup).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.manifest import write_manifest  # noqa: E402

FLAGS = flags.FLAGS

# Filename of the per-message length cache written by the measure stage
# (mirrors omegalax grain_pipeline.MESSAGE_LENGTHS_FILENAME). The measure stage
# writes a single split-agnostic cache at <message_lengths_path>/.
MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

# Per-split token statistics written by the builder (omegalax grain_pipeline
# TOKEN_STATS_FILENAME); holds vision_variability.vision_tokens_per_chunk, the
# {vision_tokens: n_chunks} frequency table this stage plots.
TOKEN_STATS_FILENAME = "token_stats.json"
VISION_CDF_FILENAME = "vision_tokens_percentile_cdf.png"
VISION_CDF_PERCENTILES = (50, 75, 90, 95, 99)

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Inline-records output dir.", required=True)
flags.DEFINE_string(
    "source_path", None, "Conversations dataset root (stage 04, with a single chat.jsonl).",
    required=True,
)
# Stage-specific:
flags.DEFINE_string(
    "omegalax_repo", None, "Path to omegalax repo root (used as uv --project).", required=True
)
flags.DEFINE_string("model_id", None, "Model id (resolves the tokenizer).", required=True)
flags.DEFINE_string(
    "processor", None, "HF repo for image processor config (defaults to model_id).", required=True
)
flags.DEFINE_integer("max_length", None, "Max sequence length.", required=True)
flags.DEFINE_integer(
    "min_length",
    0,
    "Min sequence length: drop any chunk shorter than this many tokens (0 = no "
    "floor, the default). Measured against the same content-only length "
    "--max_length budgets, so a chunk is kept only when min_length <= measured "
    "<= max_length. Mainly for overflow_mode=split, where a conversation's last "
    "chunk is whatever is left over and can be a near-empty tail carrying one or "
    "two frames. Forwarded to build_sft_records_from_chat.py --min_length (only "
    "when > 0, so older omegalax checkouts without the flag still run); per-split "
    "counts land under min_length in truncation_stats.json.",
    lower_bound=0,
)
flags.DEFINE_integer("records_per_shard", None, "Records per output shard.", required=True)
flags.DEFINE_integer(
    "num_workers",
    None,
    "Parallel workers for message-length measurement (>=2), used only when the "
    "measure cache is absent. Forwarded to build_sft_records_from_chat.py.",
    required=True,
    lower_bound=2,
)
flags.DEFINE_enum(
    "overflow_mode",
    "split",
    ["split", "truncate", "drop"],
    "Behaviour for conversations longer than max_length. 'split' (default): "
    "pack into multiple consecutive chunks at turn boundaries (no turns "
    "dropped). 'truncate': keep only the first fitting chunk and drop the "
    "overflowing turn plus the rest of the conversation. 'drop': discard the "
    "whole conversation if it does not fit in a single chunk. Forwarded to "
    "omegalax/scripts/build_sft_records_from_chat.py --overflow_mode; per-split "
    "truncation stats land in each split's truncation_stats.json.",
)
flags.DEFINE_string(
    "message_lengths_path",
    None,
    "Root of a measure-stage artifact holding the split-agnostic "
    f"<root>/{MESSAGE_LENGTHS_FILENAME}. Forwarded to build_sft_records_from_chat.py "
    "so the tokenizer pass is skipped (per-message lengths are independent of "
    "max_length / overflow_mode / split, so one cache serves every sequence length "
    "and every split). Optional: omit to tokenize in-line.",
)
flags.DEFINE_float(
    "val_fraction",
    0.0,
    "Recording-level val fraction, applied HERE (records stage) over the single "
    "<source>/chat.jsonl: > 0 writes <out>/train/ and <out>/val/ (split by "
    "recording_id), 0 writes <out>/train/ only. Because the split is applied here, "
    "the measure cache stays split-agnostic and is reused when you change this value.",
)
flags.DEFINE_bool(
    "carry_goal_context",
    False,
    "overflow_mode=split ONLY: re-attach the system prompt and the first user "
    "turn's goal/[CONTEXT] text to EVERY chunk of a split conversation, so a "
    "goal-conditioned conversation that is split at max_length yields standalone "
    "goal-conditioned chunks instead of headless screenshot->action tails. Budget "
    "for the carried header is reserved up front; the stage-05 measure cache is "
    "reused unchanged (the goal/context text is priced separately at build time). "
    "Forwarded to build_sft_records_from_chat.py --carry_goal_context; no-op for "
    "other overflow modes. Per-split truncation_stats.json reports injected_chunks.",
)


flags.DEFINE_bool(
    "plot_vision_tokens",
    True,
    "After each split is built, render its vision-tokens-per-chunk percentile CDF "
    f"to <out>/<split>/{VISION_CDF_FILENAME} from the builder's "
    f"{TOKEN_STATS_FILENAME}. Key percentiles land in manifest.json either way; "
    "only the PNG is skipped when disabled (or when matplotlib is unavailable).",
)


def weighted_cdf(freq_map: dict[str, int]) -> tuple[list[int], list[float], int, float]:
    """Frequency table {vision_tokens: n_chunks} -> (token values, cumulative
    percentiles, total chunks, chunk-weighted mean vision tokens)."""
    items = sorted((int(k), int(v)) for k, v in freq_map.items())
    total_freq = sum(f for _, f in items)
    if not total_freq:
        return [], [], 0, 0.0
    weighted_sum = sum(t * f for t, f in items)

    tokens: list[int] = []
    percentiles: list[float] = []
    cumulative = 0
    for token, freq in items:
        cumulative += freq
        tokens.append(token)
        percentiles.append(100.0 * cumulative / total_freq)
    return tokens, percentiles, total_freq, weighted_sum / total_freq


def percentile_at(tokens: list[int], percentiles: list[float], p: float) -> int:
    """Smallest token value whose cumulative percentile is >= p."""
    for token, pct in zip(tokens, percentiles, strict=True):
        if pct >= p:
            return token
    return tokens[-1]


def _plot_vision_cdf(
    tokens: list[int],
    percentiles: list[float],
    key_values: dict[int, int],
    title: str,
    out_path: Path,
) -> bool:
    """Render the CDF to ``out_path``. Returns False if matplotlib is missing."""
    if plt is None:
        print("[stage_records] matplotlib unavailable; skipping vision-token CDF", flush=True)
        return False

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(tokens, percentiles, color="#2563eb", linewidth=1.8, label="CDF")
    ax.fill_between(tokens, percentiles, alpha=0.12, color="#2563eb")

    for p, v in key_values.items():
        ax.axhline(p, color="#94a3b8", linewidth=0.7, linestyle="--", alpha=0.8)
        ax.axvline(v, color="#94a3b8", linewidth=0.7, linestyle=":", alpha=0.8)
        ax.scatter([v], [p], color="#dc2626", s=28, zorder=5)
        ax.annotate(
            f"p{p}={v}",
            xy=(v, p),
            xytext=(8, -10 if p < 95 else 8),
            textcoords="offset points",
            fontsize=8,
            color="#334155",
        )

    ax.set_title(title)
    ax.set_xlabel("Vision tokens per chunk")
    ax.set_ylabel("Percentile (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _vision_token_summary(split: str, out_split_dir: Path) -> dict | None:
    """Read the split's token_stats.json, print + plot the vision-tokens-per-chunk
    CDF, and return the summary folded into manifest.json (None if unavailable)."""
    stats_path = out_split_dir / TOKEN_STATS_FILENAME
    if not stats_path.is_file():
        print(f"[stage_records] {split}: no {TOKEN_STATS_FILENAME}; skipping vision-token CDF",
              flush=True)
        return None
    freq_map = (
        json.loads(stats_path.read_text())
        .get("vision_variability", {})
        .get("vision_tokens_per_chunk", {})
    )
    tokens, percentiles, n_chunks, mean = weighted_cdf(freq_map)
    if not tokens:
        print(f"[stage_records] {split}: no vision tokens recorded; skipping CDF", flush=True)
        return None

    key_values = {p: percentile_at(tokens, percentiles, p) for p in VISION_CDF_PERCENTILES}
    print(
        f"[stage_records] {split}: vision tokens/chunk over {n_chunks} chunks -- "
        f"mean {mean:.1f}, min {tokens[0]}, max {tokens[-1]}, "
        + ", ".join(f"p{p}={v}" for p, v in key_values.items()),
        flush=True,
    )

    summary = {
        "n_chunks": n_chunks,
        "mean": round(mean, 1),
        "min": tokens[0],
        "max": tokens[-1],
        "percentiles": {f"p{p}": v for p, v in key_values.items()},
    }
    if FLAGS.plot_vision_tokens:
        png_path = out_split_dir / VISION_CDF_FILENAME
        if _plot_vision_cdf(
            tokens,
            percentiles,
            key_values,
            f"Vision tokens per chunk — percentile CDF ({split})",
            png_path,
        ):
            print(f"[stage_records] {split}: wrote {png_path}", flush=True)
            summary["plot"] = png_path.name
    return summary


def _run_split(split: str, src_chat: Path, out_split_dir: Path, cache_path: Path | None) -> dict:
    """One build_sft_records_from_chat.py invocation for one recording-level split.
    ``--split`` makes the builder emit only that split from the single chat.jsonl;
    ``cache_path`` is the (split-agnostic) message_lengths.jsonl to reuse."""
    out_split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        FLAGS.omegalax_repo,
        "python",
        "scripts/build_sft_records_from_chat.py",
        f"--data_path={src_chat}",
        f"--out_dir={out_split_dir}",
        f"--model_id={FLAGS.model_id}",
        f"--processor={FLAGS.processor}",
        f"--max_length={FLAGS.max_length}",
        f"--records_per_shard={FLAGS.records_per_shard}",
        f"--num_workers={FLAGS.num_workers}",
        f"--overflow_mode={FLAGS.overflow_mode}",
        f"--val_fraction={FLAGS.val_fraction}",
        f"--split={split}",
        ("--carry_goal_context" if FLAGS.carry_goal_context else "--nocarry_goal_context"),
        "--overwrite",
    ]
    if cache_path is not None:
        cmd.append(f"--message_lengths_path={cache_path}")
    # Only pass the floor when it is actually set: 0 is the builder's own default,
    # so omitting it keeps this stage runnable against an omegalax checkout that
    # predates --min_length.
    if FLAGS.min_length > 0:
        cmd.append(f"--min_length={FLAGS.min_length}")
    print(f"[stage_records] {split}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=FLAGS.omegalax_repo, check=False).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"build_sft_records_from_chat.py failed (rc={rc}) for {split}")
    n_shards = sum(1 for _ in out_split_dir.glob("*.array_record"))
    stats = {"split": split, "n_shards": n_shards, "elapsed_s": int(elapsed)}
    if FLAGS.min_length > 0:
        trunc_path = out_split_dir / "truncation_stats.json"
        if trunc_path.is_file():
            floor = json.loads(trunc_path.read_text()).get("min_length")
            if floor:
                stats["min_length"] = floor
                print(
                    f"[stage_records] {split}: min_length={floor['threshold']} dropped "
                    f"{floor['chunks_dropped']} short chunk(s) "
                    f"({floor['tokens_dropped']} tokens)",
                    flush=True,
                )
    vision = _vision_token_summary(split, out_split_dir)
    if vision is not None:
        stats["vision_tokens_per_chunk"] = vision
    return stats


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    lengths_root = Path(FLAGS.message_lengths_path) if FLAGS.message_lengths_path else None
    # Fail here rather than after spawning a builder that would reject it anyway.
    if FLAGS.min_length > FLAGS.max_length:
        raise ValueError(
            f"--min_length={FLAGS.min_length} exceeds --max_length={FLAGS.max_length}; "
            "no chunk could satisfy both bounds."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 04 writes a single split-agnostic <source>/chat.jsonl; apply the
    # recording-level split HERE from --val_fraction, reusing the one root cache for
    # every split (no re-tokenization when val_fraction changes). > 0 writes
    # <out>/train/ and <out>/val/; 0 writes <out>/train/ only.
    src_chat = source_path / "chat.jsonl"
    if not src_chat.is_file():
        raise FileNotFoundError(
            f"no chat.jsonl under {source_path} (stage 04 writes a single "
            f"<source>/chat.jsonl)"
        )
    cache_path = (lengths_root / MESSAGE_LENGTHS_FILENAME) if lengths_root else None
    splits = ("train", "val") if FLAGS.val_fraction > 0.0 else ("train",)
    per_split = [_run_split(s, src_chat, output_dir / s, cache_path) for s in splits]

    write_manifest(
        output_dir,
        stage="inline_records",
        params={
            "model_id": FLAGS.model_id,
            "processor": FLAGS.processor,
            "max_length": FLAGS.max_length,
            "min_length": FLAGS.min_length,
            "records_per_shard": FLAGS.records_per_shard,
            "num_workers": FLAGS.num_workers,
            "omegalax_repo": FLAGS.omegalax_repo,
            "overflow_mode": FLAGS.overflow_mode,
            "message_lengths_path": FLAGS.message_lengths_path,
            "val_fraction": FLAGS.val_fraction,
            "carry_goal_context": FLAGS.carry_goal_context,
            "plot_vision_tokens": FLAGS.plot_vision_tokens,
        },
        inputs={"source": str(source_path)},
        stats={"per_split": per_split},
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
