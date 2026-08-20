# realigned_pipeline

The realigned crowd-cast chain: raw screen recordings + keylogs → per-app or
per-goal SFT conversations → inline training records. "Realigned" = stage 02 fixes
the keylog↔video clock skew (the OBS compositor pause clock), so actions land on
the frames they actually produced.

| Stage | Script | Role |
| --- | --- | --- |
| 00 | `stage_00_clip_manifest.py` | Walk the upload tree → one row per segment (video + keylog paths, probe metadata). |
| 01 | `stage_01_master_frames.py` | Decode each mp4 ONCE into a JPEG `images.array_record` at a fixed MASTER fps. Keylog-free and alignment-agnostic. |
| 02 | `stage_02_realign.py` | Re-stamp every keylog event keylog-clock → video-PTS. Structure-lossless: only timestamps change. Repoints `clips_manifest.keylog_path` at the corrected keylog. |
| 03 | `stage_03_sample_frames.py` | Sample the master store at `--target-fps` (no decode), bin the realigned keylog into actions, drop black frames, thin NO_OPs, **label the foreground app**. |
| 03 (alt) | `stage_03_filter.py` | Mask-only variant: a keep/drop mask over the master axis, consumed at any fps via `lib/views`. |
| 03b | `stage_03b_reference_goals.py` | Bridge annotated goals onto realigned frames → `goal_frame_index.jsonl`. |
| 04 | `stage_04_build_conversations.py` | Assemble conversations (per segment, per app run, or per goal); re-derive action labels; **filter by application**. |
| 04b | `stage_04b_filter_key_combo.py` | Re-cut a stage-04 artifact into short trajectories that each **begin with a chord**. |
| 04c | `stage_04c_annotate_conversations.py` | Hindsight-annotate an **already cut** conversations artifact: one goal instruction per conversation. |
| 05 | `stage_05_measure_lengths.py` | Payload-free tokenizer cache (`message_lengths.jsonl`), shardable as a SLURM array. |
| 06 | `stage_06_training_records.py` | Inline SFT records straight from `chat.jsonl`, reusing the stage-05 cache; plots the per-chunk vision-token CDF. |

Every stage writes `<output_dir>/manifest.json` as its completion marker.

---

## Application filtering

Conversations can be selected — or cut — by the **foreground application**.

### Where the label comes from

The crowd-cast recorder emits a `ContextChanged` event carrying the focused app's
bundle id into the *same* keylog as the input events:

```
[0,         ['ContextChanged', ['com.apple.Safari']]]
[224099997, ['KeyPress',       [59, 'KeyI']]]
```

`lib/events.iter_events` skips it (it is state, not input); `lib/events.iter_context`
parses it and `lib/app_context` folds it onto the master tick axis. The join is
exact, not heuristic, for two reasons:

* **Same clock.** `stage_02_realign.write_corrected_keylog` re-stamps every entry
  through `keylog_to_video` with no type dispatch, so app switches ride the
  identical splice map as the actions. Verified on 400 spliced segments: the
  corrected `ContextChanged` timestamps equal `keylog_to_video(raw_ts, splices)`
  exactly, 400/400.
* **Not part of the realign model.** `realign_lib.INPUT_TYPES` excludes it, so an
  app switch can never suppress a pause detection — the recorder pauses on *input*
  idleness and the model must agree with the recorder.

Consequence of that second point: `keylog_to_video` **clamps** a timestamp inside a
collapsed pause to the splice point, so a run of switches during a pause lands on
one instant. The fold takes **last-wins** at equal timestamps, which resolves to
the app in focus when recording resumed.

The track is *state*: forward-filled and deliberately **not** run through
`lib/events.apply_label_policy`. A switch inside a dead zone is how you know which
app came back, so the label policy would delete exactly the information that
explains the blackout.

### Sentinels

| Id | Meaning |
| --- | --- |
| `UNCAPTURED` | The recorder's privacy blackout (app on a do-not-capture list). Those spans are also (near-)black video — the frames `--drop-black-frames` already removes — so they cost a filtered dataset almost nothing. |
| `UNKNOWN` | No `ContextChanged` in force. Recorder versions before 0.1.1 emit none at all. |

Neither is ever a filter target or a dominant `app`, but both appear in the raw
count maps (`app_ticks`, `app_mix`) because they are real recorder states.

Windows/Linux agents report process names where macOS reports bundle ids, so
`lib/app_context.normalize_app_id` canonicalizes them (`firefox` →
`org.mozilla.firefox`, `code` → `com.microsoft.VSCode`,
`com.google.antigravity-ide` → `com.google.antigravity`, …). CLI selectors also
accept friendly short names (`resolve_app_selector`): `firefox`, `safari`, `chrome`,
`arc`, `cursor`, `vscode`, `zed`, `ghostty`, `terminal`, `antigravity`, `preview`,
`finder`, `zotero`, `obsidian`, `notion`, `slack`, `discord`, `spotify`, `claude`,
`codex`, `chatgpt`, `inkscape`, `drawio`, `linear`. A raw bundle id always works
verbatim.

### Stage 03: writing the labels

| Flag | Default | Effect |
| --- | --- | --- |
| `--app-context BOOL` | `true` | Label every kept frame with the foreground app. Metadata only — one extra msgpack pass over a keylog the stage already opens. Toggling it invalidates the resume cache (the labels are per-record). |

Per **frame record**: `app`, `app_window_switches` (switches strictly inside that
frame's label window `[tick_i, tick_{i+1})` — the turns whose action aggregates two
apps).

Per **segment** (`clips/<seg>/stage_01/segment_summaries.json`) — two weightings,
named apart on purpose:

| Frame-weighted (what filtering gates on) | Tick-weighted (coverage) |
| --- | --- |
| `app`, `app_frac`, `app_mix`, `app_frame_counts` | `app_by_ticks`, `app_frac_by_ticks`, `app_ticks`, `apps`, `app_uncaptured_frac` |

They genuinely disagree — black/idle thinning keeps a small, biased slice of the
ticks, so a segment's tick-dominant app is often not its frame-dominant one. Plus
`n_app_switches` (tick-weighted) and `n_app_seam_turns`.

Per **segment index row** (`sample_index.jsonl`): `app`, `app_frac`,
`n_app_switches`, `n_app_seam_turns`, `app_frame_counts` — enough to prefilter
without opening every records file. Frame-weighted, so it can never disagree with
the stage-04 gate.

Per **run** (`sample_summary.json` / `manifest.json`) — the app inventory and its
distribution. Both maps are ordered descending; below is a real 8-segment run:

```json
"app_frame_counts":            {"company.thebrowser.Browser": 140, "notion.mail.id": 75,
                                "com.mitchellh.ghostty": 26, "notion.id": 13,
                                "com.todesktop.230313mzl4w4u92": 1},
"app_dominant_segment_counts": {"notion.mail.id": 3, "company.thebrowser.Browser": 2,
                                "com.mitchellh.ghostty": 1},
"n_app_seam_turns_total":      26,
"app_context":                 true
```

`list(app_frame_counts)` *is* the vocabulary a stage-04 `--include-app` can select
from; the frame counts predict how many turns each app would contribute.

### Stage 04: filtering

Two modes. **Gate** keeps or drops whole segments on their dominant app —
trajectories stay intact, but a kept conversation still carries its minority apps.
**Split** cuts each segment into one conversation per maximal same-app run — pure
per-app data, at the cost of cutting segments.

Split is usually what you want: measured over 22.9k ccast0618d segments, only
**~31%** touch a single application, the median segment has **3** same-app runs, and
the dominant app holds a median **0.90** of the segment's labeled frames.

| Flag | Default | Mode | Effect |
| --- | --- | --- | --- |
| `--include-app APP` | — | both | Keep only conversations whose app is APP. **Repeatable or comma-separated.** |
| `--exclude-app APP` | — | both | Drop conversations whose app is APP. Applied after `--include-app`. |
| `--app-min-frac F` | `0.0` | gate | Require the dominant app to hold ≥ F of the conversation's labeled frames. `0.8` keeps ~65% of segments, `0.95` ~43%. Ignored under `--split-by-app` (runs are pure by construction). |
| `--app-unknown {keep,drop}` | `keep` | both | Conversations with no app label. `keep` passes them only when no `--include-app` is set; `drop` removes every unlabeled conversation. |
| `--split-by-app BOOL` | `false` | — | One conversation per maximal same-app run. An `UNCAPTURED` gap does not break a run when the same app resumes. **Rejected with `--goal-index`** (a goal window already defines the unit). |
| `--app-min-run-frames N` | `1` | split | Skip runs shorter than N frames (== turns; == seconds at `--target-fps 1`). Runs ≥ 30 hold ~91% of captured foreground time. |
| `--app-drop-seam-turns BOOL` | `true` | split | Drop each run's final turn when its action window straddles the switch — that label aggregates input from *both* apps, the same argument as the black-frame dead zones. ~2.5% of turns @1 fps. |

Interactions: `--min-frames` also applies to each app-run span (the effective floor
is `max(--app-min-run-frames, --min-frames, 1)`); `--limit` still counts *segments*
(or goals), not the conversations a split produces.

An unlabeled or `UNCAPTURED` conversation can never satisfy an `--include-app`
list — the filter never claims a segment is Firefox because nothing said otherwise.

**Ordering is load-bearing.** Action labels are re-derived and typing coalesced over
the whole segment *first*, and only then is the frame list cut, so every surviving
turn's action string is byte-identical to the unfiltered build. Verified: each split
conversation's assistant turns are a contiguous run of the unsplit build's.

#### Passing several apps

```bash
# shell: repeat the flag, or comma-separate
--include-app firefox --include-app safari
--include-app=firefox,safari,arc
```

In a **labctl recipe use the comma form** — an `[args]` table renders each key
exactly once as `--key=value`, so a repeated flag is unreachable:

```toml
[params]
include_app = "firefox,safari,arc"

[args]
"include-app" = "{params.include_app}"
```

#### What lands on the output

Per conversation: `app`, `app_frac`, `app_mix`, `app_seam_turns`, and under
`--split-by-app` also `app_run_idx`, `n_app_runs`, `app_seam_turn_dropped`. The
`conversation_id` gains an `_app<NN>` suffix so one segment's runs stay distinct.

In `conversations_summary.json` / `manifest.json`: the criteria as given
(`include_app`, `exclude_app`, `app_min_frac`, `app_unknown`, `split_by_app`,
`app_min_run_frames`, `app_drop_seam_turns`) plus `app_conversation_counts`,
`n_app_seam_turns_dropped`, `n_skipped_app_filter` (segment mode; goal mode folds
app rejections into `n_skipped`) and `n_segments_without_app_labels`.

With the filter **off**, records carry no `app*` keys at all — the existing dataset
schema is untouched.

### Running against a label-free sample

Stage 04 back-fills labels from the realigned keylogs when pointed at a stage-03
sample built before `--app-context` existed (it prints a NOTE and does it), so you
can filter an existing artifact without re-sampling. Prefer a labeled sample: the
back-fill re-reads every keylog on each stage-04 variant, and an unlabeled artifact
does not describe itself (the frame-records viewer sees no `app` either).

### Caveats worth planning around

* **~22% of segments carry no `ContextChanged`** (recorder v0.1.0). They are
  unfilterable, so an app-filtered set silently omits them — put the same
  restriction (`--app-unknown drop`) on any baseline you compare against, or the
  two sets are not comparable.
* **Splitting inherits goal mode's windowing caveat.** A formatter's cross-turn
  state (a key held across the cut) can leave an `up(X)` whose `down(X)` sits in
  another conversation. Dropping the seam turn removes the boundary turn itself but
  not a key held across it.
* **936 of 22.9k segments are `needs_review`/`UNDER`** (the realign closure
  certificate did not close). There, actions and app switches are matched but
  *equally* suspect — the app label inherits whatever timing error the actions have.

### Recipes

`slurm/dev/alfred/berlin/labctl/recipes/data_pipeline/dataset_v5_application_filter/`
holds the full chain: `stage_03_…_app_context` (writes the labels) →
`stage_04_…_split_by_app` (all apps, pure runs) or `stage_04_…_app_firefox`
(single app, the clone-per-app template) → `stage_05` (+ merge) → `stage_06`.
That lineage is goal-free, since `--split-by-app` and `--goal-index` are mutually
exclusive.

### Tests

`tests/test_app_context.py` — the tick fold (forward fill, last-wins on the
pause-collapse clamp), run merging across `UNCAPTURED`, selector flattening, and
every gate/split branch.

## Key-combo windows (stage 04b)

The application filter picks **which app**; `stage_04b_filter_key_combo.py` picks
**what the user did**. It is a filter *between* stages 04 and 05: it reads a
stage-04 artifact's `chat.jsonl` and re-cuts it into short trajectories that each
**begin with a chord** and run for at most N turns after it, writing the same
schema back out — so stages 05/06 consume it unchanged.

This is as fine-grained as the corpus gets. `ContextChanged` carries a bundle id
and **nothing else** — no window title, no URL, verified across all nine recorder
versions — so there is no site or topic key to filter on. The chord is the intent
proxy: `Meta+KeyT` is "opened a new tab", `Meta+KeyL` is "went to the address bar",
`Meta+KeyF` is "searched within the page".

```bash
uv run --locked -- python -m realigned_pipeline.stage_04b_filter_key_combo \
  --source-dir  <a stage-04 artifact> \
  --output-dir  <out> \
  --key-combo=Meta+KeyT,Meta+KeyL \
  --max-frames-after 15
```

### Matching

The **last** token of a spec is the trigger key whose *press* opens the window;
every earlier token is a modifier that must be held at that moment.
`Meta`/`Cmd`/`Ctrl`/`Shift`/`Alt` are side-agnostic groups (`Meta` matches
`MetaLeft` **or** `MetaRight`); anything else is a raw input name passed through
verbatim, so the whole vocabulary works (`KeyT`, `Return`, `Tab`, `LMB`,
`PageDown`, `Escape`). A spec with no `+` matches an unmodified press.

Held state is tracked **across** turns, because a v3 program really does carry a
key over a turn boundary — a chord commonly reads `down(MetaLeft); down(KeyT)` in
one turn and `up(KeyT); up(MetaLeft)` in the next. A press is matched *before* it
joins the held set, so a key can never satisfy a combo through itself.

* `--combo-scope turn` (default) also requires the modifiers to go down in the
  **same turn** as the trigger, so the whole chord is visible inside the window's
  first assistant turn and the trajectory explains itself. `conversation` accepts a
  modifier held from an earlier turn (~+30% matches on the browser set) but then a
  window can open on a `down(KeyT)` whose `down(MetaLeft)` was cut away.
* `--strict-modifiers` rejects a match carrying modifiers the spec did not ask for,
  so `Meta+KeyT` stops matching a `Meta+Shift+KeyT` press.

### Windows

`trigger turn + ≤ --max-frames-after turns`, truncated at the end of the source
conversation (at `--target-fps 1` that bound is also seconds). `--min-frames-after`
drops stubs at the tail. By default a trigger firing **inside** an already-open
window does not start another one; `--allow-overlap` emits those too, at the cost
of duplicating frames across rows.

Comma-separate several combos (`--key-combo=Meta+KeyT,Meta+KeyL`) — in a labctl
recipe the comma form is the only one that works, exactly as for `--include-app`.

### What lands on the output

Per window: `source_conversation_id`, `key_combo`, `trigger_key`,
`combo_turn_index` (the anchor's index in the *source* conversation),
`combo_window_idx`, `combo_frames_after`, `source_n_turns`, and recomputed
`n_frames`/`n_turns`/`n_non_noop`. Every other source field (`app`, `segment_id`,
`recording_id`, …) is carried through; `conversation_id` gains a `_kc<NNN>` suffix.
In the summary: the settings as given plus `combo_window_counts`,
`trigger_key_counts`, `frames_after_histogram`, and
`n_source_conversations_with_match`.

Source `action_format` must carry a key-transition stream — `ordered_events_v2/v3`
or the aggregate `sampled`/`canonical`. `computer_use_rel_v1` is rejected outright.

### Measured on the browser artifact

Whole-corpus chord presses over the 8024-conversation browser build: `Meta+KeyT`
2590, `Meta+KeyW` 1877, `Meta+KeyR` 948, `Meta+KeyL` 808, `Meta+KeyF` 590,
`Meta+KeyN` 176. Windowed at `--max-frames-after 15`, the three-combo set
`Meta+KeyT,Meta+KeyL,Meta+KeyF` yields **2342 windows / 35189 turns** from
1784/8024 source conversations, median 16 turns each (~8 s for the full pass).

Recipe: `…/dataset_v5_application_filter/stage_04b_key_combo_new_tab.toml` — the
clone-per-combo template.

## Annotating cut conversations (stage 04c)

Goals normally enter at stage 03b, over a *filter view* of the corpus, and the
goal windows then define what a conversation is. `stage_04c_annotate_conversations.py`
inverts that for the case where **the cut is the filter**: a 04b row is "the user
pressed `Meta+KeyT`, then did this for ≤30 turns", and the instruction has to
describe *that window* — a segment-level goal spanning ten minutes of browsing
says nothing about the new tab.

```bash
uv run --locked -- python -m realigned_pipeline.stage_04c_annotate_conversations \
  --source-dir <a stage-04 or 04b artifact> --output-dir <out> \
  --method describe_extract --models Kimi-K2.6,Kimi-K2.5 --limit 3
```

### Why it needs nothing but `chat.jsonl`

A stage-04 row is a lossless description of its own frames' master coordinates.
The user turns carry `ar://<master store>/frames/<seg>/images.array_record#N`
and that **`N` is the master tick** — stage 01 packs one record per tick
(`source_time_s = record_index / master_fps`) — while the assistant turns carry
the derived action label the keystroke-burst start-snap needs. So each row
rebuilds an exact `SegmentView` (real ticks, real label windows, gaps where
stage 03 dropped black/idle frames) and becomes one `AnnotationUnit`. Goal spans
land as genuine master intervals, joinable to any other view of the corpus.
Conversations past the context budget split into several units exactly as a
segment does; their goals pool back per conversation before one is selected.

Only `INPUT_KIND=frames` methods apply (`describe_extract`); `days`/`goals`
methods consume something a cut conversation is not.

| Flag | Default | Effect |
| --- | --- | --- |
| `--goal-select {cover-first,longest,first}` | `cover-first` | Which extracted goal conditions the row when a method returns several. `cover-first` takes the one whose span covers turn 0 — for a key-combo window, the goal the chord itself opened. |
| `--on-no-goal {drop,keep}` | `drop` | Rows the method found no bounded goal for. `drop` keeps the output purely goal-conditioned. |
| `--system-prompt-id ID` | — | Replace the source's system message (same table and grammar check as 04b). A goal-*free* source prompt stops describing the artifact once an instruction is on the first turn. |
| `--target-fps` / `--master-fps` | from the row / the store manifest | Override the frame period told to the labeler, or skip the master-store lookup. |

Everything else — `--models`, `--target-tpm`, `--param`, the window knobs, the
governor — matches `annotation/stage_annotate.py`, and so does the resume story:
`progress.jsonl` skips finished conversations, `calls/` makes re-running an
unfinished one free.

### What lands on the output

`chat.jsonl` / `conversations.jsonl` are the SAME rows with the instruction
prepended as a leading text block on the first user turn (stage 04's canonical
layout), plus `instruction`, `instruction_variants`, `goal_conditioned: true`,
`goal_{start,end}_turn_idx`, `goal_{start,end}_master_idx` and
`annotation_goals` (every candidate, for audit). Assistant turns are copied
verbatim, so stages 05/06 consume it unchanged. `goals.jsonl` holds every
extracted goal as a uniform goals row.

Recipe: `…/dataset_v5_application_filter/stage_04c_annotate_goals_key_combo_new_tab.toml`.
Note it resolves the caveat in the 04b recipe: `cua_ordered_typing_v1` says "the
first user turn states the goal", which only becomes true after this stage.

### Tests

`tests/test_stage_04c_conversations.py` — the tick round-trip (`ar://#N` →
master idx → projected span), label windows across dropped-frame gaps, goal
selection, and the message rewrite (first turn only, assistant turns untouched,
source row not mutated).
