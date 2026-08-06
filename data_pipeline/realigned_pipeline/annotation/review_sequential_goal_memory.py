#!/usr/bin/env python3
"""Build a static human-review page for a sequential_goal_memory artifact."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path
from typing import Any

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.image_store import open_image_pil, read_jpeg_bytes  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2), quote=False)


def _checkpoint_text(checkpoint: dict[str, Any] | None) -> str:
    return str((checkpoint or {}).get("text") or "")


def _goal_path(snapshot: dict[str, Any], goals: dict[str, dict[str, Any]]) -> str:
    chips = []
    for goal_id in snapshot.get("active_goal_path") or []:
        goal = goals.get(str(goal_id))
        if goal is None:
            chips.append(f'<span class="goal unresolved">unresolved: {_e(goal_id)}</span>')
            continue
        chips.append(
            f'<span class="goal {_e(goal["level"])}">'
            f'<b>{_e(goal["level"])}</b> · {_e(goal["text"])} '
            f'<small>{_e(goal.get("provenance") or "")}</small></span>'
        )
    return "".join(chips) or '<span class="muted">No active goal path</span>'


def _card(
    snapshot: dict[str, Any], *, sequence: int, image_name: str, image_data_url: str,
    goals: dict[str, dict[str, Any]], checkpoints: dict[str, dict[str, Any]],
) -> str:
    thought = str(snapshot.get("thought") or "").strip()
    checkpoint = checkpoints.get(str(snapshot.get("checkpoint_id") or ""))
    event_id = str(snapshot["anchor_semantic_event_id"])
    classes = ["event-card"]
    if thought:
        classes.append("has-thought")
    if checkpoint:
        classes.append("has-checkpoint")
    action = snapshot.get("upcoming_tool_calls") or []
    memory_before = str(snapshot.get("memory_before") or "")
    memory_after = str(snapshot.get("memory_after") or "")
    return f"""
    <article class="{' '.join(classes)}" data-event="{_e(event_id)}">
      <header class="event-head">
        <div><span class="index">{sequence + 1}</span>
          <b>Event {_e(snapshot['anchor_event_index'])}</b>
          <code>{_e(event_id)}</code>
        </div>
        <div class="badges">
          <span>t={float(snapshot.get('t_day_s') or 0):.2f}s</span>
          <span>master {_e(snapshot.get('anchor_master_idx'))}</span>
          <span>{len(snapshot.get('raw_event_ids') or [])} raw events</span>
          {('<span class="accent">thought</span>' if thought else '<span>no thought</span>')}
          {('<span class="checkpoint-badge">checkpoint</span>' if checkpoint else '')}
        </div>
      </header>

      <div class="goal-path">{_goal_path(snapshot, goals)}</div>

      <div class="event-grid">
        <section class="visual">
          <a href="{image_data_url}" download="{_e(image_name)}"
             title="Download this full-resolution screenshot">
            <img src="{image_data_url}" loading="lazy" alt="Screenshot at {_e(event_id)}">
          </a>
          <p class="caption">Current screenshot before the recorded action · click to download</p>
        </section>

        <section class="annotation">
          <h3>Immediate thought</h3>
          <div class="thought {('present' if thought else 'empty')}">
            {_e(thought) if thought else 'No thought emitted for this event.'}
          </div>

          <h3>Upcoming recorded action</h3>
          <pre class="action">{_json(action)}</pre>

          <div class="review-box">
            <b>Your review</b>
            <label>Rating
              <select class="rating">
                <option value="unreviewed">Unreviewed</option>
                <option value="good">Good</option>
                <option value="needs_edit">Needs edit</option>
                <option value="bad">Bad</option>
              </select>
            </label>
            <label><input class="grounded" type="checkbox"> Grounded in screenshot/history</label>
            <label><input class="consistent" type="checkbox"> Thought agrees with action</label>
            <label><input class="memory_ok" type="checkbox"> Memory update is causal/useful</label>
            <textarea class="notes" rows="3" placeholder="Review note…"></textarea>
          </div>
        </section>
      </div>

      <details class="memory" open>
        <summary>Rolling memory transition</summary>
        <div class="memory-grid">
          <section><h3>Memory before</h3><p>{_e(memory_before)}</p></section>
          <section><h3>Memory after</h3><p>{_e(memory_after)}</p></section>
        </div>
      </details>

      <details class="evidence">
        <summary>Evidence and provenance</summary>
        <dl>
          <dt>Visible event IDs</dt><dd>{_e(', '.join(snapshot.get('visible_event_ids') or []))}</dd>
          <dt>Prior action event IDs</dt><dd>{_e(', '.join(snapshot.get('prior_action_event_ids') or [])) or 'None'}</dd>
          <dt>Explicit references</dt><dd>{_e(', '.join(snapshot.get('references') or [])) or 'None'}</dd>
          <dt>Segment</dt><dd><code>{_e(snapshot.get('segment_id'))}</code></dd>
          <dt>Action specification</dt><dd><code>{_e(snapshot.get('action_spec'))}</code></dd>
        </dl>
      </details>

      {f'''<details class="checkpoint" open>
        <summary>Checkpoint projected at this anchor</summary>
        <pre>{_e(_checkpoint_text(checkpoint))}</pre>
      </details>''' if checkpoint else ''}
    </article>
    """


def build_review(artifact_dir: Path, output_dir: Path, *, image_width: int) -> Path:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest.json under {artifact_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("method") != "sequential_goal_memory":
        raise SystemExit(
            f"artifact method is {manifest.get('method')!r}, not 'sequential_goal_memory'"
        )
    snapshots = sorted(
        _rows(artifact_dir / "memory_snapshots.jsonl"),
        key=lambda row: (str(row.get("day_tag")), int(row["anchor_event_index"])),
    )
    if not snapshots:
        raise SystemExit("artifact has no memory_snapshots.jsonl rows")
    goals = {str(row["goal_id"]): row for row in _rows(artifact_dir / "goal_nodes.jsonl")}
    checkpoints = {
        str(row["checkpoint_id"]): row for row in _rows(artifact_dir / "checkpoints.jsonl")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for sequence, snapshot in enumerate(snapshots):
        event_id = str(snapshot["anchor_semantic_event_id"])
        image_name = f"{sequence + 1:03d}_{event_id}.jpg"
        if image_width > 0:
            image = open_image_pil(str(snapshot["image"])).convert("RGB")
            if image.width > image_width:
                height = round(image.height * image_width / image.width)
                image = image.resize((image_width, height))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95, optimize=False)
            jpeg = buffer.getvalue()
        else:
            # ArrayRecord holds the original full-resolution JPEG, so embedding
            # these bytes avoids a lossy decode/re-encode round trip.
            jpeg = read_jpeg_bytes(str(snapshot["image"]))
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        cards.append(_card(
            snapshot, sequence=sequence, image_name=image_name,
            image_data_url=image_data_url,
            goals=goals, checkpoints=checkpoints,
        ))

    n_thoughts = sum(bool(str(row.get("thought") or "").strip()) for row in snapshots)
    n_checkpoints = sum(bool(row.get("checkpoint_id")) for row in snapshots)
    artifact_key = str(manifest.get("goals_id") or artifact_dir.resolve())
    metadata = json.dumps({
        "artifact_dir": str(artifact_dir.resolve()),
        "method_schema_version": manifest.get("method_schema_version"),
        "prompt_versions": manifest.get("prompt_versions"),
        "n_events": len(snapshots),
    }, ensure_ascii=False).replace("<", "\\u003c")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sequential goal-memory review</title>
<style>
  :root {{ color-scheme: dark; --bg:#101218; --panel:#181c25; --soft:#222938;
    --line:#333c50; --text:#edf1f7; --muted:#9da9bb; --accent:#80d4ff;
    --green:#70df9b; --amber:#ffc66d; --red:#ff8a8a; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
    font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
  code,pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .top {{ position:sticky; top:0; z-index:10; padding:16px 24px; background:#101218f2;
    border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
  .top h1 {{ margin:0 0 5px; font-size:22px; }} .summary {{ color:var(--muted); }}
  .toolbar {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:12px; }}
  button,select,textarea {{ color:var(--text); background:var(--soft); border:1px solid var(--line);
    border-radius:7px; padding:7px 10px; }} button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
  main {{ max-width:1500px; margin:auto; padding:20px; }}
  .event-card {{ margin:0 0 24px; border:1px solid var(--line); border-radius:12px;
    background:var(--panel); overflow:hidden; box-shadow:0 10px 30px #0004; }}
  .event-card.hidden {{ display:none; }} .event-head {{ display:flex; justify-content:space-between;
    gap:12px; padding:13px 16px; border-bottom:1px solid var(--line); background:#1d2230; }}
  .index {{ display:inline-grid; place-items:center; width:27px; height:27px; margin-right:8px;
    border-radius:50%; background:var(--accent); color:#081018; font-weight:800; }}
  .event-head code {{ color:var(--muted); margin-left:8px; }} .badges {{ display:flex; gap:7px;
    flex-wrap:wrap; justify-content:flex-end; }} .badges span {{ border:1px solid var(--line);
    border-radius:999px; padding:2px 8px; color:var(--muted); font-size:12px; }}
  .badges .accent {{ color:var(--green); border-color:#34724a; }}
  .badges .checkpoint-badge {{ color:var(--amber); border-color:#725d34; }}
  .goal-path {{ display:flex; flex-wrap:wrap; gap:8px; padding:12px 16px; border-bottom:1px solid var(--line); }}
  .goal {{ padding:5px 9px; border-radius:6px; background:var(--soft); }}
  .goal b {{ color:var(--accent); text-transform:uppercase; font-size:11px; }}
  .goal small {{ color:var(--muted); margin-left:5px; }}
  .event-grid {{ display:grid; grid-template-columns:minmax(420px,1.2fr) minmax(380px,1fr); gap:18px; padding:16px; }}
  .visual img {{ display:block; width:100%; max-height:700px; object-fit:contain; background:#06070a;
    border:1px solid var(--line); border-radius:8px; }} .caption {{ margin:6px 0 0; color:var(--muted); font-size:12px; }}
  h3 {{ margin:0 0 7px; font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
  .thought {{ margin-bottom:18px; padding:12px; border-radius:8px; border-left:4px solid var(--green); background:#16251e; }}
  .thought.empty {{ color:var(--muted); border-left-color:#566174; background:var(--soft); }}
  pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:0; padding:12px; border-radius:8px;
    background:#0d1016; border:1px solid var(--line); }} .action {{ max-height:280px; overflow:auto; }}
  .review-box {{ display:grid; gap:8px; margin-top:18px; padding:12px; border:1px solid #4d5e7a;
    border-radius:8px; background:#1a2230; }} .review-box label {{ display:block; }}
  .review-box textarea {{ width:100%; resize:vertical; }}
  details {{ border-top:1px solid var(--line); }} summary {{ cursor:pointer; padding:12px 16px; font-weight:650; }}
  .memory-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:0 16px 16px; }}
  .memory-grid section {{ background:var(--soft); padding:12px; border-radius:8px; }}
  .memory-grid p {{ margin:0; white-space:pre-wrap; }} .evidence dl {{ display:grid;
    grid-template-columns:180px 1fr; margin:0; padding:0 16px 16px; }}
  .evidence dt,.evidence dd {{ margin:0; padding:6px; border-bottom:1px solid var(--line); }}
  .evidence dt {{ color:var(--muted); }} .checkpoint pre {{ margin:0 16px 16px; color:#ffe5b0; }}
  .muted {{ color:var(--muted); }} #progress {{ color:var(--accent); font-weight:700; }}
  @media(max-width:900px) {{ .event-grid,.memory-grid {{ grid-template-columns:1fr; }}
    .event-head {{ flex-direction:column; }} .badges {{ justify-content:flex-start; }} }}
</style>
</head>
<body>
<header class="top">
  <h1>Sequential goal-memory pilot review</h1>
  <div class="summary">{len(snapshots)} events · {n_thoughts} thoughts · {n_checkpoints} checkpoints ·
    schema {_e(manifest.get('method_schema_version'))} · <span id="progress">0 reviewed</span></div>
  <div class="toolbar">
    <label><input id="thoughtsOnly" type="checkbox"> Thoughts only</label>
    <label><input id="checkpointsOnly" type="checkbox"> Checkpoints only</label>
    <label>Rating <select id="ratingFilter"><option value="all">All</option>
      <option value="unreviewed">Unreviewed</option><option value="good">Good</option>
      <option value="needs_edit">Needs edit</option><option value="bad">Bad</option></select></label>
    <button id="expandMemory">Toggle all memory</button>
    <button id="export">Export review JSON</button>
    <button id="clear">Clear saved review</button>
  </div>
</header>
<main>{''.join(cards)}</main>
<script>
const META = {metadata};
const STORAGE_KEY = "sgm-review:" + {json.dumps(artifact_key)};
const cards = [...document.querySelectorAll('.event-card')];
function readCard(card) {{ return {{
  semantic_event_id: card.dataset.event,
  rating: card.querySelector('.rating').value,
  grounded: card.querySelector('.grounded').checked,
  thought_action_consistent: card.querySelector('.consistent').checked,
  memory_causal_useful: card.querySelector('.memory_ok').checked,
  notes: card.querySelector('.notes').value.trim()
}}; }}
function state() {{ return Object.fromEntries(cards.map(c => [c.dataset.event, readCard(c)])); }}
function save() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state())); update(); }}
function restore() {{
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
  cards.forEach(card => {{ const row=saved[card.dataset.event]; if(!row)return;
    card.querySelector('.rating').value=row.rating || 'unreviewed';
    card.querySelector('.grounded').checked=!!row.grounded;
    card.querySelector('.consistent').checked=!!row.thought_action_consistent;
    card.querySelector('.memory_ok').checked=!!row.memory_causal_useful;
    card.querySelector('.notes').value=row.notes || '';
  }});
}}
function update() {{
  const s=state(), reviewed=Object.values(s).filter(x=>x.rating!=='unreviewed').length;
  document.querySelector('#progress').textContent=`${{reviewed}}/${{cards.length}} reviewed`;
  const t=document.querySelector('#thoughtsOnly').checked, c=document.querySelector('#checkpointsOnly').checked;
  const rating=document.querySelector('#ratingFilter').value;
  cards.forEach(card => {{ const show=(!t||card.classList.contains('has-thought')) &&
    (!c||card.classList.contains('has-checkpoint')) && (rating==='all'||s[card.dataset.event].rating===rating);
    card.classList.toggle('hidden',!show); }});
}}
document.querySelectorAll('.review-box input,.review-box select,.review-box textarea').forEach(x=>x.addEventListener('change',save));
document.querySelectorAll('.review-box textarea').forEach(x=>x.addEventListener('input',save));
document.querySelectorAll('#thoughtsOnly,#checkpointsOnly,#ratingFilter').forEach(x=>x.addEventListener('change',update));
document.querySelector('#expandMemory').onclick=()=>{{ const ds=[...document.querySelectorAll('details.memory')];
  const open=ds.some(x=>!x.open); ds.forEach(x=>x.open=open); }};
document.querySelector('#clear').onclick=()=>{{ if(confirm('Clear all locally saved ratings and notes?')){{
  localStorage.removeItem(STORAGE_KEY); location.reload(); }} }};
document.querySelector('#export').onclick=()=>{{ const review=Object.values(state());
  const payload={{...META, reviewed_at:new Date().toISOString(), summary:{{
    good:review.filter(x=>x.rating==='good').length,
    needs_edit:review.filter(x=>x.rating==='needs_edit').length,
    bad:review.filter(x=>x.rating==='bad').length,
    unreviewed:review.filter(x=>x.rating==='unreviewed').length}}, events:review}};
  const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}), a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='sequential_goal_memory_review.json'; a.click(); URL.revokeObjectURL(a.href);
}};
restore(); update();
</script>
</body></html>"""
    output = output_dir / "review.html"
    output.write_text(page)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-width", type=int, default=0,
        help="Optional maximum embedded width; 0 preserves original JPEG bytes (default).",
    )
    args = parser.parse_args()
    output = build_review(args.artifact_dir, args.output_dir, image_width=args.image_width)
    print(f"[review] wrote {output}")


if __name__ == "__main__":
    main()
