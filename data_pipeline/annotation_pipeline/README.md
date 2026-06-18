# Crowd-Cast Annotation Pipeline

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

Overridable via env (`JUERGEN_ANNOTATION_VLM_MODEL`,
`JUERGEN_ANNOTATION_VLM_BASE_URL`, `JUERGEN_ANNOTATION_PROCESSED_ROOT`,
`JUERGEN_ANNOTATION_MAX_CONCURRENCY`, ...) or CLI flags; the defaults in
`config.py` are the validated setup.

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
03 assemble          verified trajectories -> run-level neutral samples     \
04 canonical         run-level canonical SFT artifact                       / CPU
05 buckets           optional local token/bucket distribution inspector      / CPU
```

A clip = up to 12 contiguous segments of one recording. It is only a sharding
and resume unit. Durable stage outputs are run-level tables under
`processed/runs/<run>/stage_*`; each stage also owns a `progress.jsonl` ledger
used for resume. Stages 00+01 cache decoded frames per (clip, fps, height) under
`processed/cache/`.

`run_pipeline.py --stages {all,frames,annotate,assemble}` selects which stages
run, so the CPU work (frames, assemble) runs as 0-GPU jobs and the GPU phase
(annotate) never idles on ffmpeg/tokenization.

## Run it

### Production path: labctl

Submit the whole dataset pipeline through labctl so every materialized stage is
registered with run provenance and artifact lineage:

```bash
cd /fast/project/HFMI_SynergyUnit/yll/slurm/dev/yll/berlin/labctl
labctl run-pipeline pipelines/annotation_sft_buckets.toml
```

That pipeline runs and records:

```
discover_clips          clips_dataset.json + discovery report
stage_00_01_frames      stage 00 manifests + stage 01 frame cache
stage_02_annotate       VLM segment/name/verify outputs
stage_03_assemble       neutral verified trajectory samples
stage_04_canonical      canonical SFT JSONL with labctl-owned prompt policy
count_tokens            omegalax exact token-count artifact
bucket_8k/16k/32k/64k   omegalax train/val chunk-index datasets
```

The training recipes consume the final bucket aliases, e.g.
`juergen_annotation_qwen3vl2b_bucket_32768_65536_chunk_index` for 64k.

### Low-level/debug path

One-time environment setup (syncs the pipeline env, builds the serving venv, and downloads weights):

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
sbatch annotation_pipeline/slurm/setup_env.sbatch
```

**A curated set** (clips in `clips.json`) on one node, all stages:

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
sbatch annotation_pipeline/slurm/run_pipeline.sbatch --run-name pilot --clips all
sbatch annotation_pipeline/slurm/run_pipeline.sbatch --run-name pilot --clips bbbf_s0000-0003   # one clip
```

**The whole dataset** — decoupled phases so GPUs only ever run the VLM.
Discover once, then run each phase as a sharded array (`N` = array size; phases
resume independently, re-run a phase's line after any timeout):

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
uv run --project . --locked python -m annotation_pipeline.discover_clips  # -> clips_dataset.json (+ report)
sbatch --array=0-15 annotation_pipeline/slurm/extract.sbatch  dataset_full_20260615   # phase 1: frames
sbatch --array=0-1  annotation_pipeline/slurm/annotate.sbatch dataset_full_20260615   # phase 2: VLM
sbatch --array=0-7  annotation_pipeline/slurm/assemble.sbatch dataset_full_20260615   # phase 3: assemble
sbatch             annotation_pipeline/slurm/finalize.sbatch dataset_full_20260615   # phase 4: canonical
```

`--gres=gpu:0` extract/assemble jobs use idle CPU cores and hold no GPUs, so the
8-GPU annotate nodes stay on the VLM. Annotate/assemble **skip** clips whose
frames aren't cached yet. A finished clip in the stage `progress.jsonl` ledger is skipped on resume; a bad
clip is isolated to `failed_clips.jsonl` instead of killing the shard. Stage 02 issues
`--max-concurrency` requests at once (default 6 after a 720p OOM; raise via
`JUERGEN_ANNOTATION_MAX_CONCURRENCY`) to keep the DP=4 replicas busy.

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
    stage_00_manifest/                      # run-level raw segment manifest
      manifest.jsonl, manifest_summary.json, clip_summaries.jsonl, progress.jsonl
    stage_01_frames_actions/                # run-level frame/action table
      frame_records.jsonl, segment_summaries.jsonl, clip_summaries.jsonl, progress.jsonl
    stage_02_segment/                       # run-level annotation tables
      trajectories_raw.jsonl, pass_a_candidates.jsonl
      pass_a_merged_segments.jsonl, naming_rejected.jsonl, clip_summaries.jsonl, progress.jsonl
    stage_03_assemble/                      # run-level neutral SFT samples
      trajectories.jsonl, rejected_trajectories.jsonl, clip_summaries.jsonl, progress.jsonl
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

To inspect the canonical stage-04 SFT JSONL directly:

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
uv run --project . --locked python -m annotation_pipeline.visualize_canonical_sft
```

## Migrate the legacy per-clip run

The old `dataset_full_20260615` output under `processed/runs/` stored stage
outputs inside each clip directory. To materialize the current run-level layout
without touching the legacy run or the shared frame cache:

```bash
cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline/annotation_pipeline
sbatch slurm/legacy_migrate_audit.sbatch
sbatch slurm/legacy_migrate_aggregate.sbatch
sbatch --array=0-15 slurm/legacy_migrate_assemble.sbatch
sbatch slurm/legacy_migrate_finalize.sbatch
```

The default destination is
`processed/runs/dataset_full_20260615_runlevel_migrated`. Stage 00/01 tables
are rebuilt from `processed/cache/frames`; stage 02 is converted from the
legacy per-clip annotation files; stage 03 is rerun with the current assembler
so samples have `clip_id` and no obsolete system messages; stage 04 is then
exported with the current canonical writer.

## Layout

```
discover_clips.py            clip discovery + idle pre-gate -> clips_dataset.json
run_pipeline.py              orchestrates 00->04, stage 05 optional
config.py  common.py  qwen3_encoding.py  migrate_legacy_clip_run.py
stage_00…stage_05 (.py)
visualize_pipeline.py        run inspector
clips.json                   curated clips;  clips_dataset.json = discovered (gitignored)
slurm/  _serve_qwen.sh  setup_env.sbatch  run_pipeline.sbatch
        extract.sbatch  annotate.sbatch  assemble.sbatch  finalize.sbatch
experiments/single_pass_test.py   one-call probe for the single-pass idea
```
