# frames-master (stage 01a + 01b) — decode once, sample any fps for free

`stage_01_frames_actions` couples the expensive ffmpeg **decode** to the cheap
**action binning + NO_OP thinning**, so changing `--target-fps` re-decodes the
whole corpus. This folder splits that in two:

- **01a `build_frames_master.py`** turns every segment's mp4 into a JPEG
  `images.array_record` **once**, at a fixed *master* fps.
- **01b `sample_frames_actions.py`** samples any `target_fps ≤ master_fps` out of
  that store — picking the nearest master record per bin and binning the keylog —
  a metadata-only pass that writes **no new JPEG bytes**. A new-fps dataset costs
  seconds, not a re-decode.

> **Caveat:** 01b removes the *decode* work, not the *annotation* work. A new
> `target_fps` changes which frames the VLM sees, so stage 02 must still re-run.
> The win is byte-identical pixels across fps variants (one decode, immune to
> ffmpeg-version drift) and near-instant fps sweeps.

The master store is **alignment-agnostic**: the decode never reads the keylog,
so realignment (the keylog↔video time fix) is *not* upstream of it. Realignment
is a 01b concern — 01b joins the realigned keylog to these master shards by
`segment_id` at sampling time.

```
discover                      stage 01a: build_frames_master.py       stage 01b (sampler)
clips_manifest.jsonl  ─────▶  mp4 → images.array_record @ master_fps  ─────▶  frame_records.jsonl
(no realign needed)           + per-record source_time_s                      (join realigned keylog
                                                                               by segment_id; target_fps,
                                                                               NO_OP thinning, actions)
```

## Build the master (01a)

```bash
# On a CPU allocation (ffmpeg-heavy), from the data_pipeline root:
export JUERGEN_ANNOTATION_FFMPEG_BIN=/path/to/ffmpeg   # or rely on PATH
uv run python realignment_fix/build_frames_master.py \
    --clips-manifest <discover_out>/clips_manifest.jsonl \
    --output-dir     <frames_master_out> \
    --master-fps 4 --num-workers 16
```

Each worker spawns one ffmpeg (capped by `JUERGEN_ANNOTATION_FFMPEG_THREADS`,
default 4). Resumable: a segment with an existing `frame_manifest.jsonl` is
skipped unless `--force`.

## Sample any fps (01b)

```bash
# Cheap (no ffmpeg). Joins the master frames to the REALIGNED clips_manifest by
# segment_id: frames from --frames-master-dir, keylog+alignment from --clips-manifest.
uv run python realignment_fix/sample_frames_actions.py \
    --frames-master-dir <frames_master_out> \
    --clips-manifest    <stage00_realign_out>/clips_manifest.jsonl \
    --output-dir        <sampled_out> \
    --target-fps 0.5 --num-workers 16
```

`--target-fps` must be ≤ the store's `master_fps` (enforced; you can only sample
down). Resumable: a segment already sampled at the same fps + NO_OP params is
skipped unless `--force`. Because the master store is alignment-agnostic, you can
re-run realignment and re-sample here **without** rebuilding the master.

The output is a **drop-in for the annotation runner** — its
`clips/<seg>/stage_01/frame_records.jsonl` layout is exactly what
`run_dataset.py --phase annotate` expects, so stage 02 runs straight off it with
no ffmpeg:

```bash
PYTHONPATH=. python -m annotation_pipeline.run_dataset --phase annotate \
    --frames-root <sampled_out> --manifest <stage00_realign_out>/clips_manifest.jsonl \
    --run-name <run> --models Kimi-K2.6,Kimi-K2.5 ...
```

## The one rule: master fps is the sampling ceiling

Frames are stored **CFR at `--master-fps`** (ffmpeg's `fps=` filter resamples
any VFR source), so master record `i` sits at `source_time_s = i / master_fps`.
Downstream you can only sample **down** to a target fps ≤ master — never up.
Pick master as the highest fps you will ever want; storage scales linearly with
it (4 fps ≈ 8× the current 0.5 fps default). `source_frame_idx` (nearest frame
in the *original* video) is recorded for provenance only.

Not done here (fps-dependent → belongs to 01b): action binning, NO_OP head/tail
thinning, per-frame action strings.

## 01a outputs (`--output-dir`)

| Path | Contents |
| --- | --- |
| `frames/<segment_id>/images.array_record` | one grain shard per segment; record `i` = raw JPEG for master frame `i`. |
| `frames/<segment_id>/frame_manifest.jsonl` | per record: `record_index`, `image` (`ar:///…#idx`), `source_time_s`, `source_frame_idx`, `jpeg_bytes`, `sha256`. |
| `segment_index.jsonl` | one row per segment — shard path, `num_records`, `master_fps`, video-relative timing + video provenance. Keyed by `segment_id`; keylog-free and alignment-agnostic (01b joins the realigned keylog by `segment_id`). |
| `frames_master_summary.json` | aggregate stats + status counts. |
| `manifest.json` | artifact marker (`artifact_type: juergen_annotation_frames_master`). |

`ar://` refs resolve through `annotation_pipeline/image_store.py` exactly like
stage 01's store, so stage 02 and the viewers consume them unchanged.

Per-segment status in `segment_index.jsonl`: `ok`, `empty` (0 frames decoded),
`cached` (resume), `skipped_video_not_ok`, `skipped_no_video`, `failed` (+`error`).

## Viewer — `visualize_frame_records.py` (stage 01b)

A browser viewer for the **01b** output, the thin successor to the old
`ylli_visualizer` realignment inspector. Because 01b has already sampled to the
target fps, thinned NO_OPs, and binned the actions — with each kept frame an
`ar://` ref into the 01a master store — the viewer needs no ffmpeg, no video
transcode, and no raw-vs-realigned dual clock. It just steps through the
trajectory 01b produced: one frame + one action per bin.

```bash
cd .../data_pipeline
uv run python realignment_fix/visualize_frame_records.py \
    --dataset <01b_output_dir> \
    --port 8770
# SSH-forward and open http://127.0.0.1:8770/
#   ssh -L 8770:127.0.0.1:8770 <host>
```

`--dataset` takes one or more datasets (01b output dirs, `frame_records.jsonl`
files, or 01a frames-master stores) and you switch between them in the UI's
**dataset** dropdown. Each is auto-detected (master vs sample) and built lazily
on first selection; a build failure (e.g. an empty / not-yet-generated dir) is
reported inline while the other datasets keep working. A single 01b dir finds
`frame_records.jsonl` at its root or one level down (e.g. `train/`, `val/`).
Pick a segment from the dropdown; step with ◀ / ▶ (or <kbd>←</kbd>/<kbd>→</kbd>),
<kbd>space</kbd> to play, <kbd>a</kbd>/<kbd>d</kbd> to jump to the previous/next
active (non-NO_OP) frame. The bin strip and action table both jump on click.
The frame fills the left pane; the keyboard, mouse radar, typed text, and action
table live in the **right sidebar** — drag its left edge to resize it (the width
persists across reloads, and the keyboard auto-fits the chosen width).

**Action HUD** — the current bin's `action` string is parsed client-side and
shown three ways:

- **Keyboard** — an on-screen keyboard lights each key: *bright* = pressed in
  this bin, *dim* = held from an earlier bin (state is replayed from frame 0).
  Keys not on the drawn layout appear as chips so nothing is lost.
- **Mouse radar** — an arrow in a circle points along the summed `(dx, dy)` for
  the bin, length ∝ magnitude; `LMB`/`MMB`/`RMB` pills light on press/hold and a
  scroll readout shows `±scroll`.
- **Typed text** — key events are replayed into a materialized text buffer
  (Shift → case, Backspace deletes, Space/Return/Tab handled); the characters
  typed **in the current bin** are colorized so you can watch text appear as you
  step. Note this reflects *keystrokes*, not focus — it concatenates everything
  typed in the segment regardless of which field had focus.

**Input contract** — one JSON object per line (the `frame_records.jsonl` schema
stage 01 emits and 01b must reproduce): `segment_id` (grouping key, rows
contiguous + in order), `action` (`"NO_OP"` or a formatted action string), and
an image ref in `image_path` (or `image`) — normally an
`ar:///…/images.array_record#idx` into the master store. `local_bin_idx`,
`local_time_s`, `source_frame_idx`, `recording_id` are shown when present; extra
keys are ignored. Frames resolve through `annotation_pipeline/image_store.py`,
so plain image-file paths render too. The browser requests frames by
`(segment, index)`, never a path, so there is no path-traversal surface.

Until 01b lands, the viewer runs unchanged on any existing stage-01
`frame_records.jsonl` (identical schema) — the same tool serves both.

**Raw frames-master (01a) mode.** Point `--dataset` at a 01a frames-master store
(`segment_index.jsonl` + `frames/<seg>/{images.array_record,frame_manifest.jsonl}`)
and the viewer auto-detects it and browses the *raw* decoded frames directly — no
sampler run required. Segments are listed from `segment_index.jsonl` up front and
each segment's frames are read lazily from its `frame_manifest.jsonl` on demand
(the store can be hundreds of thousands of frames).

**Raw keylog overlay (no 01b needed).** The frames-master is keylog-free, but the
viewer can overlay the keylog itself: it finds the keylog per segment from the
master's `manifest.json` `source_clips_manifest` (a stage-00 clips_manifest;
override with `--clips-manifest`), then shows

- a **raw event table** — individual keylog events `[t_s, type, detail]` at their
  own timestamps (unbinned; consecutive mouse-moves coalesced), auto-scrolled and
  highlighted to the frame's time window as you step; and
- the **keyboard / mouse / typed-text HUD**, driven by bucketing the keylog at
  `master_fps` (bin *i* = the frame at `[i/fps, (i+1)/fps)`).

This is **raw / as-recorded**: events sit on their own clock against the raw-clock
master frames, with **no realignment** applied — so for a misaligned segment you
see the drift (that's the point). If no keylog is linked, the HUD stays empty and
a banner points you to `--clips-manifest` or 01b. To get *realigned* actions with
NO_OP thinning, run 01b and view the `_sample` dataset instead.

**Alignment overlay — dual-clock table + "aligned + trims" timeline.** When a
stage-00 `alignment.jsonl` is available (auto-discovered from a `*realign*` sibling
of the master's `source_clips_manifest`, matched by dataset-family prefix; or set
explicitly with `--alignment <alignment.jsonl | realign-clip-manifest-dir>`), the
master view shows *how stage-00 trimmed the keylog*:

- the event table becomes **dual-clock** — every event listed at both its **raw**
  and **aligned** (`realign_lib.keylog_to_video`) timestamp; events inside a
  collapsed idle span or past video-end are flagged **trimmed** (red / strikethrough).
  Highlight + click now track the *aligned* clock (frames are video-clock).
- the **frame strip** shades the **cut frames** amber — the raw-keylog span
  `[kp, kp+collapse]` (in frame units) that realignment folds away to a single
  video frame at `vp`.
- a **second timeline** appears under the frame strip: green where an aligned action
  lands, a red **■ collapse marker** at each folded idle span (tooltip = seconds
  collapsed, `kp→vp`), and a status line (`status · collapse Xs · residual Ys ·
  N events trimmed past video`). Compare it against the raw strip above to see the shift.
- a **`keys: raw ⇄ aligned` toggle** (header) switches the keyboard / mouse /
  typed-text HUD between the raw keylog binning and the realigned binning (the
  latter via `aggregate_actions(..., timemap=keylog_to_video)`). The two strips
  always show both clocks; the toggle only drives the HUD/status readout.

Master frames stay raw; only the overlay is realigned. `aligned` segments (no splices)
show identity times and no markers. Uses the frozen `alignment.jsonl` splices (not a
re-derivation), so the view matches what stage-00 actually did.

## 01b outputs (`--output-dir`)

| Path | Contents |
| --- | --- |
| `clips/<segment_id>/stage_01/frame_records.jsonl` | one row per kept frame: `image_path` (`ar://` into the master shard), `action`, 0-based `global_frame_idx`, `local_bin_idx`, `source_frame_idx`, `master_record_index`. |
| `clips/<segment_id>/stage_01/{segment_summaries,frames_actions_summary}.json` | per-clip stats (mirrors stage 01's schema, so `visualize_run` reads them unchanged). |
| `clips/<segment_id>/stage_00/manifest.jsonl` | single-row provenance (the realigned manifest row + `sampled_target_fps`); satisfies the `run_dataset --phase annotate` layout. |
| `sample_index.jsonl` | one row per segment — `keylog_path`, `alignment_status`, `target_fps`, frame counts, status. |
| `sample_summary.json` | aggregate stats + status counts. |
| `manifest.json` | artifact marker (`artifact_type: juergen_annotation_frames_sampled`). |

No new JPEGs are written — every `image_path` points back into the 01a master
shard. Per-segment status in `sample_index.jsonl`: `ok`, `empty` (0 frames after
NO_OP thinning), `cached` (resume), `no_master_frames` (segment absent from the
master store), `master_<status>` (master store skipped/failed it), `empty_master`
(0 master records), `failed` (+`error`, e.g. `target_fps > master_fps`).
