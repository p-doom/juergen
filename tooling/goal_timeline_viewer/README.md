# Goal-timeline viewer (full recording)

Full-recording viewer that overlays our **auto-extracted goal hierarchy**
(`fold → restructure → assign → stitch`) on a continuous video timeline: the HEVC
`<video>` plays the whole ~6h session, and a **goal-timeline canvas** shows each
overarching goal as a colored group of task lanes, every task drawn as its scattered
multi-interval bars across the day — so you can see, at a glance, how tasks interleave.
A sidebar lists the hierarchy; click a task (lane, gutter, or sidebar row) to jump to it.

Adapted from `p-doom/slurm:dev/franz/berlin/crowd-cast-bc/goal_annotation_manual` (the
manual human-labeling tool) — same video/keylog/day-timeline infra; we add a read-only
goal overlay. The manual-annotation keys (`[`/`]`/`i`/…) still work; see the doc below.

## Run (3 steps)

```bash
cd tooling/goal_timeline_viewer
UP=/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-06-18/uploads
REC=e2c26556-7861-4109-9f70-cd0847695612

# 1) build the single-recording day.json (segment timing from mp4 creation_time)
uv run --script build_recording_day.py --uploads $UP --recording $REC --out data/$REC/day.json

# 2) resolve the extracted goals to t_day -> overlay_goals.json
R=/fast/project/HFMI_SynergyUnit/yll/hindsight_fold/runs/exp_e2c26556
python build_overlay.py --restructured $R/restructured_v2/restructured.json \
    --spans $R/assigned_v2/task_spans.json --day data/$REC/day.json \
    --out data/$REC/overlay_goals.json

# 3) serve (login node OK — IO only), then SSH-tunnel the port and open in a browser
uv run --script annotator.py serve --data_dir data/$REC --port 8791 --no_prewarm
#   on your Mac:  ssh -N -L 8791:127.0.0.1:8791 <host>   then  http://localhost:8791
```

New files (ours): `build_recording_day.py`, `build_overlay.py`; the overlay is served at
`/api/overlay` (read from `<DATA>/overlay_goals.json`) and rendered by the `#goaltl`
canvas + `#ovlist` sidebar in `annotator.html`. Drop in a new `overlay_goals.json` and
refresh — no restart needed. HEVC decodes client-side (Safari, or Chrome≥107/Firefox≥134).

---

# Manual goal annotation — full-day viewer

Infrastructure for **hand-labelling goals** on a crowd-cast participant's
**entire working day**, to gauge how hard goal extraction is on real human
recordings and to build a ground-truth set for benchmarking automated
goal-extraction (the VLM hindsight track is separate; this is the human one).

Two files — one script and one frontend:

| file | role |
|---|---|
| `annotator.py` | the whole tool: a CLI with four subcommands (`index`, `align`, `prewarm`, `serve`) |
| `annotator.html` | the viewer: `<video>` playback, synchronized keylog + mouse-path, goal authoring (served from the same dir) |

`annotator.py` subcommands:

| subcommand | role |
|---|---|
| `index` | order one participant's segmented recordings into days; freeze one day → `day.json` |
| `align` | (re)build the keylog↔video realignment map → `alignment.json` (`serve` auto-runs this) |
| `prewarm` | parallel-remux the day to faststart mp4s (moov-first) → `<DATA>/cache_faststart/` (`serve` self-warms too) |
| `serve` | local server: streams the faststart mp4s (HTTP Range) + keylog, persists goals |

Optional one-off diagnostics live under `analysis/` (`realign_stats.py`,
`classify_unverified.py`) — read-only stats over a prepared day; not needed for the
normal flow.

## Why this shape

- **Ordering.** A participant (`<user-dir-id>`) appears under several upload
  schema-version dirs, and each recording is cut into ~5-min segments whose
  `seg` numbers reset per recording. The reliable global order is the OBS
  `creation_time` baked into each mp4 container (absolute UTC) — *not* file
  mtime (= upload time) and *not* seg number. `annotator.py index` reads that, dedupes
  re-exports across versions, and buckets into local-tz days.
- **Direct video, no transcode.** Recordings are **HEVC**. Modern browsers decode
  HEVC client-side via the OS hardware decoder (Safari always; Chrome ≥107,
  Firefox ≥134, all hardware-gated) — and the browser runs on your **Mac**, not
  the Linux cluster, so playback just works.
- **Faststart is mandatory over a tunnel.** OBS writes the `moov` atom at
  *end-of-file*, so a browser must download the **entire** segment (tens of MB)
  before it can build a seek index or show one frame — 10–20s/segment over SSH,
  plus intermittent HEVC decode errors from incomplete data. We remux each segment
  to a faststart copy (moov-first, lossless packet copy, no re-encode) so the
  browser reads a tiny index up front and then fetches only the GOP it needs.
  `annotator.py prewarm` builds the cache ahead of time; `serve` also warms it
  lazily/on-startup. The server implements HTTP Range (required for `<video>`
  seeking). No ffmpeg, no pre-extracted frames.
- **Continuous day.** The 92 segments + gaps are stitched into one virtual
  timeline (`t_day` seconds since the day's first frame). The viewer maps any
  `t_day` to `(segment, offset)`, auto-advances across segments, and visibly
  skips recorder-off gaps.

## Run

```bash
# 1) survey a participant's days (reads mp4 metadata; ~15s for ~3.5k segments)
uv run --script annotator.py index --user 87776db2-8265-4b9a-8199-ba43f1baf479 --tz Europe/Berlin

# 2) freeze one working day -> day.json (put data outside the repo)
DATA=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/goal_annotation/87776db2_2026-05-21
uv run --script annotator.py index --user 87776db2-8265-4b9a-8199-ba43f1baf479 \
    --tz Europe/Berlin --date 2026-05-21 --out $DATA/day.json

# 3) (optional) prebuild the faststart cache in parallel — else the server warms
#    it lazily/on-startup. ~4 GB for a 7h day; run on a compute node if you like.
uv run --script annotator.py prewarm --data_dir $DATA --workers 8

# 4) serve it (login node is fine — IO only)
#    On startup `serve` auto-builds alignment.json (runs the `align` logic) and
#    self-warms any missing faststart segments, so steps 3 and `align` are optional.
uv run --script annotator.py serve --data_dir $DATA --port 8753   # --verbose to log every range request
```

To (re)build just the realignment map without serving — e.g. before running the
`analysis/` diagnostics:

```bash
uv run --script annotator.py align --data_dir $DATA   # writes $DATA/alignment.json
```

Then from your laptop:

```bash
ssh -L 8753:localhost:8753 <login-node>
# open http://localhost:8753   (use Safari/Chrome/Firefox on macOS for HEVC)
```

Switching participant/day is just different `--user` / `--date`.

## Annotating

Goals are authored at two horizons and persisted **immediately** to
`$DATA/goals.jsonl` (write-through; a crash never loses work).

| | keys |
|---|---|
| navigate | `Space` play/pause · `+`/`−` speed 0.5–16× · `a` idle-skip · `→`/`←` ±0.5s · `⇧→`/`⇧←` ±5s · `.`/`,` ±1 frame · `↑`/`↓` ±1 min · `J`/`K` segment · `Home`/`End` · click timeline |
| annotate | `[` mark start · `]` mark end + write text · `i` instant/event goal · `n`/`p` select goal · `e` edit · `x` delete · `Esc` cancel |
| editor | `Enter` save · `Tab` cycle horizon · `Esc` cancel |

**Horizons:** `short` = one-breath sub-task · `long` = goal spanning many
segments · `event` = a single moment. The synchronized panel highlights keys
pressed approaching the playhead, draws the **mouse-motion path** (deltas
integrated over a ~2.5s window — recency = brightness, clicks ringed, current
position the white dot), shows mouse buttons / scroll / total travel, the active
app, and a recent-event ticker.

## `goals.jsonl` schema

One JSON object per line (sorted by `t_start`):

```json
{"id": 1, "t_start": 1234.5, "t_end": 1450.0,
 "start_utc": "2026-05-21T...Z", "end_utc": "2026-05-21T...Z",
 "horizon": "short|long|event", "text": "...", "seg_idx": 12, "app": "ghostty"}
```

`t_start`/`t_end` are seconds on the day timeline; `*_utc` are absolute wall-clock
(from `day.json` `day_start_utc`). This is the ground-truth contract a benchmark
harness reads.

## Iterating on the viewer

`annotator.html` is read fresh from disk per request and served `no-store`, so a
browser refresh shows edits immediately. Editing `annotator.py` needs a
server restart; only `/video` responses are cached client-side.
