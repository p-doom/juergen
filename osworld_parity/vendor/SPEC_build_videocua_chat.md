# Spec: build_videocua_chat.py — VideoCUA -> crowd-cast dense-action chat.jsonl

Write a single python3 script `build_videocua_chat.py` (stdlib only; no third-party deps)
in this directory. It merges VideoCUA ground-truth action logs with IDM MouseMove
pseudo-labels into crowd-cast-format chat records.

## Inputs (CLI flags, argparse)
- `--data_dir`   : VideoCUA extracted root: `<data_dir>/<App>/<task_id>/action_log.json`
- `--frames_root`: extract_frames.py output: `<frames_root>/<App>/<task_id>/{bc_frames/frame_%06d.jpg, extract_meta.json}`
- `--idm_root`   : idm_label_videocua.py output: `<idm_root>/<App>/<task_id>.idm.json`
- `--out_dir`    : output dir
- `--shard_idx`, `--num_shards` (defaults 0/1): shard the sorted task list by index mod.
- `--key_map`    : path to videocua_key_map.json (see below; load it, do not hardcode).

## Outputs (per shard, under out_dir)
- `samples_shard_{shard_idx:03d}.jsonl` — one record per task (see Record format)
- `stats_shard_{shard_idx:03d}.jsonl` — one stats row per task (incl. skipped ones, with `skip_reason`)
- `unmapped_shard_{shard_idx:03d}.jsonl` — one row per unmapped key/button/char occurrence:
  {task, action_type, raw_value, context}

## Frame/interval geometry
- BC grid: 2.0 fps. extract_meta.json has n_bc_frames; frame j image = bc_frames/frame_{j:06d}.jpg,
  covering wall-clock interval [j/2, (j+1)/2). n_intervals = n_bc_frames.
- Events with ts >= n_intervals*0.5 are clamped into the LAST interval. Events with ts < 0 -> interval 0.
- IDM grid: 5.0 fps. IDM prediction for frame k (from a chunk with start_frame s and tag F%02d -> k = s + int(tag[1:]))
  covers [k/5, (k+1)/5). Assign IDM frame k to BC interval min(floor(k*2/5), n_intervals-1).

## IDM label handling
From each task's .idm.json: iterate chunks; skip chunks with "error". For each prediction:
- type MouseMove: details "dx,dy" integers (may have stray spaces; parse robustly, skip malformed -> count in stats as idm_malformed).
  Normalized 0-1000 where 1000 = full screen width (dx) / height (dy).
  Convert to pixels with the task's native W,H from extract_meta.json (width/height fields):
  dx_px = dx/1000*W, dy_px = dy/1000*H. Sum per BC interval (float), round once per interval at the end.
- All other IDM types are IGNORED for labels (GT wins) but COUNTED in stats
  (idm_keypress_n, idm_click_n, idm_scroll_n) for the GT-vs-IDM agreement report.
- Task missing its .idm.json or with n_errors == n_windows -> skip_reason="no_idm".

## Ground-truth event conversion
Parse action_log.json: task_id, task_instruction, platform, action_log[]. Convert each action to
zero or more timed events `(ts, kind, key)` with kind in {press, release} and key in the crowd-cast
rdev vocabulary, plus per-interval scroll accumulation, plus position anchors.

Normalization machinery (key names are messy human annotations — census showed "Lef", "LefT",
"Left Mouse BUtton", "Down Arrow Key", "space bar", "ctrl + comma", '"cmd" + "="', etc.):
- normalize_button(raw): lowercase, strip non-alpha; if contains "left" or startswith("lef") or == "l"
  or in {"click","mouse click","mouseclick","na","n/a",""} or raw is None -> "LMB" (the na/None/empty
  default applies to CLICK and to MOUSE_UP-pairing fallback, see below); contains "right" -> "RMB";
  contains "middle" -> "MMB"; else None.
- normalize_key(raw): case-insensitive; strip whitespace; the key_map JSON maps alias->vocab entry.
  A vocab entry is either a string (e.g. "Return") or a list for shifted chars (["ShiftLeft","Equal"]
  meaning Shift chord around the base key). Also handle: single letter a-z -> "Key<upper>";
  single digit -> "Num<d>"; uppercase single letter -> shifted ["ShiftLeft","Key<letter>"] ONLY when it
  comes from TYPING text (PRESS "G" means the G key, unshifted). Unknown -> None (log to unmapped).
- normalize_chord(raw): for PRESS/HOTKEY values. Split on "+", "-" (only when between tokens, e.g.
  "Ctrl-A"), and whitespace runs; also handle quoted tokens like '"cmd" + "="' (strip quotes).
  Words {ctrl, control, strg} -> ControlLeft; {shift} -> ShiftLeft; {alt, option} -> Alt;
  {altgr} -> AltGr; {cmd, command, meta, super, win, windows, windows key, winkey} -> MetaLeft.
  Non-modifier tokens go through normalize_key. Emits (modifiers, base_keys).
  Special cases:
  * "Enter + left arrow" (two non-modifier keys) -> sequence of two separate presses.
  * Value containing ":" where the prefix parses as a chord (e.g. "command + v: Featuring 100...")
    -> use the prefix chord, log the suffix to unmapped with context "hotkey_paste_suffix".
  * PRESS value that fails chord parsing but is a multi-char string whose chars are all typeable
    -> treat as TYPING text (census: PRESS "123456", PRESS "scholarships for international students").
  * Anything else unparseable -> log unmapped, emit nothing.
- char_to_events(c) for TYPING: letter -> KeyX (+Shift chord if uppercase); digit -> NumD; " " -> Space;
  "\n"/"\r" -> Return; "\t" -> Tab; punctuation via key_map "chars" section (e.g. "." -> Dot,
  ">" -> ["ShiftLeft","Dot"], "!" -> ["ShiftLeft","Num1"], ...). Unknown char -> log unmapped
  (context "typing_char"), skip it; count per task (stats typing_unmapped_chars).

Per action type:
- CLICK{x,y,text,numClicks} at ts: button = normalize_button(text) (None -> LMB, count as
  defaulted_button). n = numClicks or 1, capped at 3 (census max 20 is annotation noise; cap and count).
  Emit n x (press BTN, release BTN) at ts. (x,y) is a position ANCHOR (authoritative over MOVE_TO).
- MOVE_TO{x,y[,xEnd,yEnd]} at ts: NO events. Anchor at (x,y) unless the degenerate all-zero
  pattern (x==0 and y==0 and xEnd/yEnd present) -> ignore entirely (count moveto_degenerate).
  If xEnd/yEnd present and not all-zero: anchor at (x,y) with this ts AND anchor at (xEnd,yEnd)
  at the same ts (endpoint marker: order (x,y) first).
- TYPING/TYPE/TEXT{text} at ts: chars spread uniformly over [ts, t_end] where
  t_end = min(next_action_ts if any else ts + len/8.0, ts + len/8.0) and at least ts (len = #chars).
  Each char's chord: +mods, +key, -key, -mods in order at its char time.
- PRESS{key|text} / HOTKEY{keys|key|text} at ts: normalize_chord; for each chord: press modifiers
  in order, press+release each base key, release modifiers in reverse. All at ts.
  HOTKEY may carry x,y -> treat as anchor too.
- KEY_DOWN{key|text} -> press events of the chord's keys (no auto-release; count keydown_open if
  never matched by KEY_UP). KEY_UP -> release. Match leniency: normalize the same way.
- MOUSE_DOWN{text} at ts: button normalize; if None, try normalize_chord on the raw value (census:
  MOUSE_DOWN "Ctrl") and emit key press instead (count mousedown_as_key); if still nothing -> LMB
  (count defaulted_button). Emit press BTN. Track open buttons.
- DRAG_TO{x,y} at ts: NO events; anchor at (x,y).
- MOUSE_UP{text} at ts: button = normalize_button(text); if None -> the most recently opened
  still-open button, else LMB. Emit release BTN.
- SCROLL{scrollY[,scrollX]} at ts: scroll_acc[interval(ts)] += int(scrollY or 0). scrollX ignored
  but counted (stats scroll_x_dropped). VideoCUA scrollY follows pyautogui semantics (positive = up);
  crowd-cast scroll assumed same convention (identity mapping) — see notes.md D8.
- TERMINATE / TERMINATE_SUCCESS / AFTER_LAST_ACTION: ignore (stage_d injects TERMINATE downstream).
- Unknown action_type: count + log unmapped (context "action_type").

At the end, force-release any still-open buttons/keys at the last event's ts (count forced_release).

## Assistant label strings (crowd-cast grammar, exact)
Per BC interval j:
- dx, dy = rounded IDM pixel-delta sums for the interval (0 if none).
- scroll = GT scroll_acc[j] (int).
- events = GT timed events in interval j, sorted by (ts, original order), rendered "+Key"/"-Key".
- If dx==0 and dy==0 and scroll==0 and no events: label = "NO_OP".
- Else: "{dx} {dy} {scroll}" + (" ; " + " ".join(event tokens) if events else "").

## Record format (one JSON line per task; MUST match the crowd-cast corpus shape)
{
  "sample_id": "videocua_<App>_<task_id>",
  "recording_id": "<App>/<task_id>",
  "app": "<App>", "platform": <platform from action_log>,
  "instruction": <task_instruction verbatim>,
  "n_frames": n_bc_frames, "duration_s": <duration from extract_meta>,
  "messages": [
    {"role":"system","content":[{"type":"text","text": SYSPROMPT}]},
    {"role":"user","content":[{"type":"image","image": <abs path frame_000000.jpg>},
                               {"type":"text","text": <instruction>}]},
    {"role":"assistant","content":[{"type":"text","text": <label interval 0>}]},
    {"role":"user","content":[{"type":"image","image": <abs frame_000001.jpg>}]},
    {"role":"assistant","content":[{"type":"text","text": <label 1>}]},
    ... (n_bc_frames user turns total, each followed by its assistant label)
  ]
}
SYSPROMPT (exact, single line): "You operate a desktop computer. The first user turn shows the initial screen and the user's goal; subsequent user turns show the current screen. Reply with the next action toward that goal as `<dx> <dy> <scroll>` optionally followed by ` ; +KEY -KEY` events, or `NO_OP` if no action."
Image order in first user turn: image FIRST, then instruction text (matches corpus).

## Anchor-consistency sanity check (per task, into stats row)
- anchors: chronological (ts, x, y) from CLICK, valid MOVE_TO (incl. xEnd/yEnd endpoints), DRAG_TO,
  HOTKEY-with-xy. Drop anchors with x<0 or y<0 or x>W or y>H (count anchor_oob).
- For consecutive pairs (t1,p1)->(t2,p2) with t2 > t1 + 1e-9:
  gt = ((x2-x1)/W*1000, (y2-y1)/H*1000)
  idm = sum over IDM 5fps frames k with k/5 in [t1, t2) of the NORMALIZED (0-1000) deltas.
  err = hypot(gt - idm); rel = err / max(hypot(gt), 50).
- stats row: {task, app, n_actions, n_bc_frames, n_anchor_pairs, anchor_err_mean, anchor_err_median,
  anchor_err_p90, anchor_rel_median, n_pairs_err_gt300, counts: {defaulted_button, moveto_degenerate,
  typing_unmapped_chars, unmapped_keys, clicks, typing_chars, press_n, hotkey_n, drag_n, scroll_events,
  idm_move_frames, idm_keypress_n, idm_click_n, idm_scroll_n, idm_malformed, forced_release,
  numclicks_capped, scroll_x_dropped}, skip_reason: null|str}
- Tasks are NOT excluded here — write every convertible record; exclusion happens downstream from
  the stats (threshold chosen after inspecting the corpus distribution).
- Skip (no record, stats row with skip_reason) when: action_log unreadable/empty, no bc frames /
  extract_meta.json missing, no_idm (above), or zero assistant intervals.

## Style
Plain python3.12 stdlib. Deterministic. No prints except a per-100-tasks progress line and a final
summary line. Keep functions small and testable; include `if __name__ == "__main__": main()`.
Write a tiny self-test mode `--self_test` that builds a synthetic 3-interval task in-memory
(fabricated action log + idm chunks) and asserts the rendered labels equal expected strings
(cover: click with numClicks=2, typing "Hi!", hotkey Ctrl+Shift+>, drag press-move-release across
intervals, NO_OP interval, scroll accumulation, IDM dx,dy pixel conversion + interval binning).
