"""Paired bootstrap on the difference between two probe runs.

Both runs replay the same val records, so a two-sample interval throws away
most of the information: the record is the resampling unit and the comparison
is paired. Without this, run-to-run differences of a few hundredths on 150-400
records read as movement when they are noise.

  aim     difference in mean move-direction cosine (bc_aim_probe pairs.jsonl)
  cursor  difference in the cursor-compensation slope (cursor rows.jsonl)
  rate    difference in frac_compensating — steadier than the slope, which is
          one fit over a heavy-tailed ratio distribution and can swing while
          the underlying behaviour holds still
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from bc_offline_score import analyze_pair
from cursor_probe import observations

N_BOOT = 5000
AIM_ROOT = Path("/fast/project/HFMI_SynergyUnit/yll/eval_logs/bc_aim_probe")
CURSOR_ROOT = Path("/fast/project/HFMI_SynergyUnit/yll/eval_logs/bc_cursor_probe")


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def aim_cosines(root: Path, label: str) -> dict[int, float]:
    out = {}
    for r in _rows(root / label / "pairs.jsonl"):
        a = analyze_pair(r["gold"], r["pred"], action_format="oev3")
        if a["gold_moved"]:
            out[r["idx"]] = a["cosine"]
    return out


def cursor_obs(root: Path, label: str) -> tuple[dict[int, list], set[int]]:
    rows = _rows(root / label / "rows.jsonl")
    out = {}
    for r in rows:
        got = observations([r])
        if got:
            out[r["idx"]] = [(o["expect"], o["obs"]) for o in got]
    return out, {r["idx"] for r in rows}


def _rate(pairs, floor: float = 20.0) -> float:
    usable = [(x, y) for x, y in pairs if abs(x) > floor]
    if not usable:
        return 0.0
    return sum(1 for x, y in usable if 0.5 <= y / x <= 1.5) / len(usable)


def _slope(pairs) -> float:
    sxx = sum(x * x for x, _ in pairs)
    return (sum(x * y for x, y in pairs) / sxx) if sxx else 0.0


def _report(name: str, a_label: str, b_label: str, n: int, sa: float, sb: float, boots: list[float]):
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    verdict = "significant" if lo > 0 or hi < 0 else "NOT significant"
    print(f"{b_label} - {a_label}   [{name}, n={n}]")
    print(f"  {sa:.4f} -> {sb:.4f}   delta {sb - sa:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")


def aim_delta(root: Path, a_label: str, b_label: str) -> None:
    a, b = aim_cosines(root, a_label), aim_cosines(root, b_label)
    keys = sorted(set(a) & set(b))
    rng = random.Random(0)
    n = len(keys)
    diffs = [b[k] - a[k] for k in keys]
    boots = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT)]
    _report("aim cosine", a_label, b_label, n,
            sum(a[k] for k in keys) / n, sum(b[k] for k in keys) / n, boots)


def cursor_delta(root: Path, a_label: str, b_label: str, stat=_slope, name="cursor slope") -> None:
    a, all_a = cursor_obs(root, a_label)
    b, all_b = cursor_obs(root, b_label)
    keys = sorted(all_a & all_b)
    sa = stat([p for k in keys for p in a.get(k, [])])
    sb = stat([p for k in keys for p in b.get(k, [])])
    rng = random.Random(0)
    boots = []
    for _ in range(N_BOOT):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        boots.append(
            stat([p for k in draw for p in b.get(k, [])])
            - stat([p for k in draw for p in a.get(k, [])])
        )
    _report(name, a_label, b_label, len(keys), sa, sb, boots)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metric", choices=("aim", "cursor", "rate"))
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    default = AIM_ROOT if args.metric == "aim" else CURSOR_ROOT
    root = Path(args.root) if args.root else default
    if args.metric == "aim":
        aim_delta(root, args.baseline, args.candidate)
    elif args.metric == "rate":
        cursor_delta(root, args.baseline, args.candidate, stat=_rate, name="frac_compensating")
    else:
        cursor_delta(root, args.baseline, args.candidate)


if __name__ == "__main__":
    main()
