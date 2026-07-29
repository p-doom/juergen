"""Aggregate OSWorld baseline result.json files into a task-success score.

reward is in [0,1] per task (OSWorld env.evaluate()); NaN counts as 0.0
(crashed/unscored). Prints overall + per-app + stop-reason breakdown.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/ckpt_evals/baseline_8b_instruct")
# Exclude gdrive-unscorable tasks (no Google OAuth here; OSWorld test_nogdrive drops
# them too) so EVERY format scores over the SAME 107-task denominator (apples-to-apples),
# regardless of whether a format wrote them as nan or left them absent.
_gd = Path("/fast/home/franz.srambical/osworld_parity_split/gdrive_unscorable.txt")
EXCLUDE = {l.strip() for l in _gd.read_text().splitlines() if l.strip()} if _gd.exists() else set()
rows = []
for rp in base.glob("*/*/result.json"):
    key = f"{rp.parent.parent.name}/{rp.parent.name}"
    if key in EXCLUDE:
        continue
    try:
        d = json.loads(rp.read_text())
    except Exception:
        continue
    sc = d.get("scores", {})
    r = sc.get("reward")
    r = 0.0 if (r is None or (isinstance(r, float) and math.isnan(r))) else float(r)
    app = d.get("params", {}).get("app", rp.parent.parent.name)
    rows.append((app, r, d.get("params", {}).get("stop_reason", "?")))

n = len(rows)
if n == 0:
    print("no results yet"); sys.exit(0)
solved = sum(1 for _, r, _ in rows if r >= 0.999)
partial = sum(1 for _, r, _ in rows if 0 < r < 0.999)
succ = sum(r for _, r, _ in rows) / n
per_app = defaultdict(lambda: [0, 0.0])
for a, r, _ in rows:
    per_app[a][0] += 1; per_app[a][1] += r
stops = defaultdict(int)
for _, _, s in rows:
    stops[s] += 1

print(f"{base.name}: {n}/107 scorable held-out tasks (gdrive-3 excluded)")
print(f"  success rate (mean reward): {succ*100:.1f}%   solved(=1.0): {solved}   partial: {partial}")
print("  per-app:")
for a in sorted(per_app):
    c, s = per_app[a]
    print(f"    {a:<20} {s/c*100:5.1f}%  ({int(round(s))}/{c})")
print("  stop reasons:", dict(stops))
