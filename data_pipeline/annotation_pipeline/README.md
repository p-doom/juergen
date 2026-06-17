# V3 Crowd-Cast Trajectory Pipeline

Turns Crowd-Cast screen recordings + msgpack keylogs into Omegalax-compatible
SFT JSONL. **Single validated configuration:** Qwen3.6-27B (BF16, served with
sglang) annotates the recordings, a grounded verification pass is the quality
gate, and stage 04 exports the canonical SFT artifact. Labctl + omegalax own
training token counts and bucket materialization.

## Configuration (the one we run)

| Knob | Value | Why |
|---|---|---|
| Annotator | `Qwen/Qwen3.6-27B` (BF16) via sglang, TP=2 DP=4 | local, zero API cost; never FP8 |
| Thinking | off | verification pass replaces self-confidence; thinking is ~5x slower + fragile |
| Frame height | 720p | 540p left dense UI text (terminal/tree) unreadable for the trainee |
| Segment window | 90 s, 30 s overlap | finer, more atomic segments; no oversize samples |
| Quality gate | stage 02 pass C verification | Qwen's self-reported confidence is uncalibrated (~0.95 flat) |
| Trainee tokenizer | `Qwen/Qwen3-VL-2B-Instruct` | exact token buckets that match training |
| Output root | `<dataset>/processed/` | processed data lives next to raw, not in the code tree |

Overridable via env (`V3_VLM_MODEL`, `V3_VLM_BASE_URL`, `V3_PROCESSED_ROOT`,
`V3_MAX_CONCURRENCY`, …) or CLI flags; the defaults in `config.py` are the
validated setup.

## Stages

```
(discover_clips.py)  enumerate recordings -> clips_dataset.json, pre-gate idle,
                     chunk long recordings into <=12-segment clips
00 manifest          MP4 + keylog pairs for a clip's segment slice          \
01 frames+actions    2fps 720p frames + per-frame action strings (cached)   / CPU
02 segment+name+verify                                                        GPU
     pass A  segment 90s windows into candidate task intervals
     pass B  write the imperative instruction + refined bounds per segment
     pass C  verify each on its frames (active / action_visible /
             start_grounded / end_reached) -> writes a `verified` flag
03 assemble          verified trajectories -> neutral per-clip samples      \
04 canonical         run-level canonical SFT artifact                       / CPU
05 buckets           optional local token/bucket distribution inspector      / CPU
```

A clip = up to 12 contiguous segments of one recording. All VLM work (A/B/C) is
in stage 02 against the one sglang server, run concurrently. Stages 00+01 cache
per (clip, fps, height) under `processed/cache/`; runs live in `processed/runs/`.

`run_pipeline.py --stages {all,frames,annotate,assemble}` selects which stages
run, so the CPU work (frames, assemble) runs as 0-GPU jobs and the GPU phase
(annotate) never idles on ffmpeg/tokenization.

## Run it

One-time environment setup (syncs the pipeline env, builds the serving venv, and downloads weights):

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
sbatch annotation_pipeline/slurm/setup_env.sbatch
```

**A curated set** (clips in `clips.json`) on one node, all stages:

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
sbatch annotation_pipeline/slurm/run_pipeline.sbatch --run-name v1 --clips all
sbatch annotation_pipeline/slurm/run_pipeline.sbatch --run-name v1 --clips bbbf_s0000-0003   # one clip
```

**The whole dataset** — three decoupled phases so GPUs only ever run the VLM.
Discover once, then run each phase as a sharded array (`N` = array size; phases
resume independently, re-run a phase's line after any timeout):

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
uv run --project . --locked python -m annotation_pipeline.discover_clips  # -> clips_dataset.json (+ report)
sbatch --array=0-15 annotation_pipeline/slurm/extract.sbatch  dataset_v1   # phase 1: frames
sbatch --array=0-1  annotation_pipeline/slurm/annotate.sbatch dataset_v1   # phase 2: VLM
sbatch --array=0-7  annotation_pipeline/slurm/assemble.sbatch dataset_v1   # phase 3: assemble+canonical
```

`--gres=gpu:0` extract/assemble jobs use idle CPU cores and hold no GPUs, so the
8-GPU annotate nodes stay on the VLM. Annotate/assemble **skip** clips whose
frames aren't cached yet (never fall back to ffmpeg). A finished clip (stage-03
`trajectories.jsonl`) is skipped on resume; a bad clip is isolated to
`failed_clips.jsonl` instead of killing the shard. Stage 02 issues
`--max-concurrency` requests at once (default 6 after a 720p OOM; raise via
`V3_MAX_CONCURRENCY`) to keep the DP=4 replicas busy.

## Environments

Two uv-managed environments, both built by `setup_env.sbatch`:

- **`juergen` workspace / `data_pipeline` project** — the whole pipeline,
  stages 00–04 (cv2, msgpack, openai client, and the annotation-side
  dependencies). Stage 05 is an optional local token/bucket distribution
  inspector. **No omegalax dependency.**
- **`yll/venvs/vllm-annotate`** — sglang serving only (torch cu126, cuDNN 9.16).

The SLURM entrypoints run stages 00–04 with
`uv run --project <juergen>/data_pipeline --locked python -m annotation_pipeline...`;
the model is served from `vllm-annotate` alongside.

## Outputs

All under `<dataset>/processed/` (`config.PROCESSED_ROOT`):

```
processed/
  cache/frames/<rec8>_s<a>-<b>_2fps_720p_noop2/   stage 00+01, shared, ffmpeg once
  runs/<run_name>/
    run_config.json
    <clip_id>/
      stage_02_segment/
        pass_a_merged_segments.jsonl       # eyeball boundaries here
        pass_b_response_segment_*.txt       # raw instruction responses
        pass_c_verify_segment_*.txt         # raw verification verdicts
        trajectories_raw.json               # each carries `verified` + `verify_checks`
        stage02_summary.json                # incl. n_verified
      stage_03_assemble/trajectories.jsonl  # verified == true only; neutral
    stage_04_canonical_sft/
      chat.jsonl, manifest.json, sample_manifest.jsonl, split_manifest.jsonl
    stage_05_length_buckets/                # optional local iteration only
      chat.jsonl, chat_8k.jsonl … chat_256k.jsonl
      bucket_summary.json, trajectory_manifest.jsonl, rejected_oversize.jsonl
```

Raw VLM responses (A/B/C) are persisted and reused on reruns, so iterating on
downstream stages never re-spends VLM calls.

## Key design facts

- **VLM decides boundaries + instructions only.** Keylogs are the source of
  truth for assistant actions; each action supervises the next 500 ms at 2 fps.
- **Verification, not confidence, is the gate.** Pass C asks discrete grounded
  yes/no questions on the segment's frames; stage 03 keeps a trajectory iff
  `active ∧ action_visible ∧ start_grounded ∧ end_reached`. Deliberate
  scrolling/reading to find or review information counts as `active`.
- **SFT frames are the clean stage 01 stream** (2 fps, 720 p, no-op capped); the
  stage 02 annotation renders add a timestamp overlay and are VLM inputs only.
  Stage 03 refuses to run if pointed at the annotation renders.
- **~96% of a sample's tokens are vision tokens** (~900 per 720 p frame), so
  sequence length ≈ `n_frames × ~900`. Token budget is driven by frame count,
  not instruction text. (At 540 p it was ~510/frame — downsample later by
  re-extracting or capping the processor's `max_pixels`, then re-bucketing.)
- **Never FP8**; the cluster driver is CUDA 12.8 (serving venv pins cu126 wheels
  + cuDNN ≥9.15 for the Conv3d path).

`qwen3_encoding.py` is kept for the optional local stage-05 distribution
inspector. Labctl + omegalax own the training token counts and buckets.

## Inspect a run

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
uv run --project . --locked python -m annotation_pipeline.visualize_pipeline
```

## Layout

```
discover_clips.py            clip discovery + idle pre-gate -> clips_dataset.json
run_pipeline.py              orchestrates 00->04, stage 05 optional
config.py  common.py  qwen3_encoding.py
stage_00…stage_05 (.py)
visualize_pipeline.py        run inspector
clips.json                   curated clips;  clips_dataset.json = discovered (gitignored)
slurm/  _serve_qwen.sh  setup_env.sbatch  run_pipeline.sbatch
        extract.sbatch  annotate.sbatch  assemble.sbatch   dataset phases
experiments/single_pass_test.py   one-call probe for the single-pass idea
```
