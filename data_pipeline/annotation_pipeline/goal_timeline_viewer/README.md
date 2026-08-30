# Goal-timeline viewer

This local viewer presents segmented recordings as one continuous day. It can
record manual goals in `goals.jsonl` or overlay an extracted goal hierarchy from
`overlay_goals.json`.

## Prepare and serve a day

```bash
data=/absolute/viewer-data

uv run annotator.py index \
    --uploads /absolute/uploads \
    --user <recording-owner> \
    --tz UTC \
    --date YYYY-MM-DD \
    --out "$data/day.json"

uv run annotator.py prewarm --data_dir "$data" --workers 8
uv run annotator.py serve --data_dir "$data" --port 8753
```

`index` orders segments from media creation timestamps. `prewarm` remuxes them
for ranged browser playback without re-encoding. `serve` rebuilds alignment
metadata and serves only on `127.0.0.1` by default.

## Extracted-goal overlay

```bash
uv run build_overlay.py \
    --restructured /absolute/goals/restructured.json \
    --spans /absolute/goals/task_spans.json \
    --day "$data/day.json" \
    --out "$data/overlay_goals.json"
```

The browser reads `overlay_goals.json` on refresh. Manual annotations are
written through immediately to `goals.jsonl`; each record carries day-relative
start/end times, UTC timestamps, horizon, text, segment index, and application.

Playback requires a browser with HEVC support for the source recordings.
