"""Parse-rate report for a Track-1 format spot-check (NOT a parity number).

Reads {base}/{app}/{task_id}/result.json (n_steps + parse_errors from
format_eval_shard) and computes the closed-loop parse-rate = fraction of the
policy's steps whose output parsed to a valid action. Also prints a few RAW
model outputs from trajectory.jsonl so the grammar can be eyeballed.
"""
import json
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/fast/home/franz.srambical/osworld_parity_split/spotcheck_moverel")
rows = []
tot_steps = tot_perr = 0
for rp in sorted(base.glob("*/*/result.json")):
    d = json.loads(rp.read_text())
    p = d.get("params", {}); s = d.get("scores", {})
    nst = s.get("n_steps_taken") or 0
    perr = p.get("parse_errors")
    rows.append((p.get("app"), (p.get("task_id") or "")[:12], nst, perr,
                 p.get("stop_reason"), rp.parent))
    tot_steps += nst or 0
    tot_perr += (perr or 0)

print(f"=== parse-rate spot-check: {base.name} ({len(rows)} tasks) ===")
for app, tid, nst, perr, stop, _ in rows:
    pr = "n/a" if not nst else f"{100*(nst-(perr or 0))/nst:.0f}%"
    print(f"  {app:<16}{tid}  steps={nst}  parse_errors={perr}  parse_rate={pr}  stop={stop}")
if tot_steps:
    print(f"\nOVERALL parse-rate: {100*(tot_steps-tot_perr)/tot_steps:.1f}%  "
          f"({tot_steps-tot_perr}/{tot_steps} steps parsed, {tot_perr} parse errors)")
else:
    print("\nno steps recorded yet")

for app, tid, nst, perr, stop, dirp in rows[:2]:
    tj = dirp / "trajectory.jsonl"
    if not tj.is_file():
        continue
    print(f"\n--- raw model outputs: {app}/{tid} ---")
    for line in tj.read_text().splitlines()[:4]:
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        at = e.get("action_text") or e.get("response") or ""
        parsed = e.get("parsed"); perr_f = e.get("parse_error")
        print(f"  step {e.get('step')}: parsed={bool(parsed)} parse_error={perr_f} :: {at[:200]!r}")
