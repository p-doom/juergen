# VideoCUA thinking-trace insertion (side-quest)

Insert first-person **thinking traces with timestamps** into the
[ServiceNow/VideoCUA](https://huggingface.co/datasets/ServiceNow/VideoCUA)
dataset (~10k desktop computer-use tasks: an mp4 + an explicit `task_instruction`
+ a timestamped action log per task).

This is **decoupled** from the crowd-cast (`realigned_pipeline`) annotation
method: no day-long context thread, no hindsight goal-recovery, no
`FilterArtifact` / `DayStream`, no stage_04t SFT build. Each VideoCUA task is
short and self-contained with a known goal, so we annotate **each task
independently**. We borrow only the *technique* (first-person reasoning at
decision points + a future-blind verify gate) and the low-level tooling (the
labeler client and the `ar://` ArrayRecord frame store) from `realigned_pipeline`.

## Pipeline (2 stages + a metadata step)

```
raw VideoCUA app dirs
   │  build_manifest.py            (stage A0 — metadata only, no decode)
   ├─► clips_manifest.jsonl        (stage_01-shaped: segment_id/video_path/video_ok/video_fps/…)
   └─► tasks.jsonl                 (self-contained: task_instruction + normalized actions)
   │
   │  realigned_pipeline/stage_01_master_frames.py --master-fps 15   (stage A1 — REUSED verbatim)
   └─► 15fps JPEG ArrayRecord master store   (decode once via ffmpeg; the sampling CEILING)
   │
   │  annotate_thinking.py          (stage B — the actual thinking insertion)
   └─► thinking.jsonl               ◄── the deliverable
```

**Why a 15fps master then subsample:** exactly the ccast approach — decode each
video once at the fps ceiling, then the annotator subsamples *down* (default
5fps) and slides a ≤15-frame window (under the VLM vision-image cap) over it,
force-including the frame nearest each action so every decision point has a
frame. Sending raw 15fps to a VLM is infeasible (a 16s task = 240 images).

## `thinking.jsonl` schema (one row per verified thought)

```json
{"task_id": 111433, "segment_id": "bash__111433", "platform": "Bash",
 "task_instruction": "List files in the current directory.",
 "t_s": 2.13, "frame_idx": 32, "image": "ar:///…/images.array_record#32",
 "before_action": {"t_s": 2.5, "type": "TYPING", "params": {"text": "ls"}, "groundcua_id": "…"},
 "kind": "plan", "thought": "…first-person reasoning…",
 "verify": {"verdict": "pass", "violations": [], "reason": "…"},
 "window_idx": 0, "model": "Qwen/Qwen3.6-27B"}
```

`kind ∈ {plan, reorient, decide, react, monitor, wait}`. Only `verdict == "pass"`
thoughts are emitted unless `--keep-fails`.

## Run via labctl (the artifact form)

Recipes live in `…/labctl/recipes/videocua/`:
`vcua_00_manifest.toml` → `vcua_01_frames.toml` → `vcua_03_thinking.toml`.
Edit the `inputs.*.path` fields to point at your unzipped data / prior-stage
outputs, then launch each in order. The thinking recipe's `[env]` selects the
backend: **local sglang Qwen** for the pilot; point `LABELER_MODEL` /
`LABELER_BASE_URL` / `LABELER_API_KEY` at Azure Kimi for the full run (no code
change). An sglang server must be reachable at `LABELER_BASE_URL` from the job.

## Run directly (dev / smoke)

```bash
cd data_pipeline
# 0) download + unzip the apps you want (pilot: Bash + Inkscape)
#    from https://huggingface.co/datasets/ServiceNow/VideoCUA/tree/main/raw_data
# 1) manifest
uv run python videocua_thinking/build_manifest.py \
    --raw-dir /path/to/videocua/raw --output-dir OUT/manifest --platforms Bash Inkscape
# 2) 15fps master frames (needs ffmpeg; JUERGEN_ANNOTATION_FFMPEG_BIN)
uv run python realigned_pipeline/stage_01_master_frames.py \
    --clips-manifest OUT/manifest/clips_manifest.jsonl --output-dir OUT/frames --master-fps 15
# 3) thinking traces (needs a labeler endpoint; LABELER_BASE_URL/MODEL/API_KEY)
LABELER_MODEL=Qwen/Qwen3.6-27B LABELER_BASE_URL=http://localhost:8011/v1 LABELER_API_KEY=EMPTY \
uv run python videocua_thinking/annotate_thinking.py \
    --frames-dir OUT/frames --tasks OUT/manifest/tasks.jsonl --output-dir OUT/thinking \
    --vlm-fps 5 --window 15 --max-thoughts 5
```

Useful `annotate_thinking.py` flags: `--no-verify` (skip the gate), `--keep-fails`
(emit failed thoughts for audit), `--limit N`, `--platforms`, `--force` (re-run
cached tasks). Re-runs resume from `units/<segment_id>.json` + a per-clip
response cache under `calls/`.

## Dataset quirks (observed)

- Action vocabulary is `CLICK / MOVE_TO / DRAG_TO / MOUSE_DOWN / MOUSE_UP /
  PRESS / HOTKEY / KEY_DOWN / KEY_UP / TYPING`. **`SCROLL` never actually
  appears** despite the dataset card; unknown types are rendered generically.
- `platform` casing is inconsistent (`Inkscape` vs `inkscape`); the `--platforms`
  filter is case-insensitive and segment-id slugs are normalized, so it's a
  non-issue.
- A few task folders ship without `video/video.mp4` (7 of 121 Inkscape); these
  are excluded automatically. Tasks with corrupt `video_metadata.json`
  (sentinel duration) fail the `video_ok` gate.
