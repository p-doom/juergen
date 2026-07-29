"""Differential across sweep arms: trace-quality metrics per arm x domain.

Reads a sweep root (dir of arm subdirs, each = a collect output dir with task_*/
result.json + trajectory.jsonl). Computes, per arm (and per OSWorld domain):
  - setup_success_rate  (n_setup_ok / n_runs)   [SetupController worked]
  - valid_toolcall_rate (steps with a parsed computer_use / total steps) [format]
  - prose_rate + avg_prose_chars (reasoning present?)  [distill-ability]
  - terminate_rate (rollouts ending terminate success)  [task-attempt proxy]
  - avg_n_steps
Also dumps a few sample reasoning-prose snippets per arm for MANUAL quality read
(the metric that can't be auto-scored). task-SUCCESS needs the VLM judge or
standalone evaluate() — run separately.
"""
from __future__ import annotations
import argparse, json, glob, os, re, collections, statistics

_TOOLCALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def _prose(resp: str) -> str:
    return _TOOLCALL.sub("", resp or "").strip()


def arm_metrics(arm_dir: str) -> dict:
    runs = sorted(d for d in glob.glob(os.path.join(arm_dir, "*")) if os.path.isdir(d) and os.path.isfile(os.path.join(d, "result.json")))
    n = len(runs)
    per_domain = collections.defaultdict(lambda: {"n": 0, "setup_ok": 0})
    n_setup_ok = 0
    step_total = step_valid = 0
    prose_steps = 0
    prose_chars = []
    term = 0
    nsteps = []
    prose_samples = []
    succ = []          # deterministic evaluate() scores (if --evaluate was used)
    for d in runs:
        rp = os.path.join(d, "result.json")
        if not os.path.isfile(rp):
            continue
        r = json.load(open(rp))
        app = r.get("app", "?")
        per_domain[app]["n"] += 1
        if r.get("setup_ok"):
            n_setup_ok += 1
            per_domain[app]["setup_ok"] += 1
        if r.get("stop_reason") == "setup_failed":
            continue
        tp = os.path.join(d, "trajectory.jsonl")
        if not os.path.isfile(tp):
            continue
        steps = [json.loads(l) for l in open(tp) if l.strip() and json.loads(l).get("step_num", 0) > 0]
        nsteps.append(len(steps))
        if r.get("stop_reason") == "terminate":
            term += 1
        if r.get("task_success") is not None:
            succ.append(float(r["task_success"]))
        for s in steps:
            step_total += 1
            info = s.get("info", {}) or {}
            if info.get("parsed") and not info.get("parse_error"):
                step_valid += 1
            pr = _prose(s.get("action") or s.get("response") or "")
            if len(pr) > 3:
                prose_steps += 1
                prose_chars.append(len(pr))
                if len(prose_samples) < 4:
                    prose_samples.append(pr[:240])
    done = [r for r in (json.load(open(os.path.join(d, "result.json")))
                        for d in runs if os.path.isfile(os.path.join(d, "result.json")))
            if r.get("stop_reason") != "setup_failed"]
    ndone = len(done)
    return {
        "arm": os.path.basename(arm_dir), "n_runs": n, "n_setup_ok": n_setup_ok,
        "setup_success_rate": round(n_setup_ok / n, 3) if n else 0,
        "n_rolled_out": ndone,
        "valid_toolcall_rate": round(step_valid / step_total, 3) if step_total else 0,
        "prose_rate": round(prose_steps / step_total, 3) if step_total else 0,
        "avg_prose_chars": round(statistics.mean(prose_chars), 1) if prose_chars else 0,
        "terminate_rate": round(term / ndone, 3) if ndone else 0,
        "det_success_rate": round(statistics.mean(succ), 3) if succ else None,  # gold: deterministic evaluate()
        "n_evaluated": len(succ),
        "avg_n_steps": round(statistics.mean(nsteps), 1) if nsteps else 0,
        "per_domain_setup": {k: f"{v['setup_ok']}/{v['n']}" for k, v in sorted(per_domain.items())},
        "prose_samples": prose_samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True, help="dir containing arm subdirs")
    ap.add_argument("--show_prose", action="store_true")
    args = ap.parse_args()
    arms = sorted(d for d in glob.glob(os.path.join(args.sweep_root, "*")) if os.path.isdir(d))
    rows = [arm_metrics(a) for a in arms if glob.glob(os.path.join(a, "*", "result.json"))]
    print(f"\n===== SWEEP DIFFERENTIAL: {args.sweep_root} =====")
    hdr = f"{'arm':<34}{'setup%':>8}{'valid_tc%':>10}{'prose%':>8}{'prose_ch':>9}{'term%':>7}{'steps':>7}{'n':>5}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ds = f"{r['det_success_rate']*100:>5.0f}%" if r['det_success_rate'] is not None else "   NA"
        print(f"{r['arm']:<34}{r['setup_success_rate']*100:>7.0f}%{r['valid_toolcall_rate']*100:>9.0f}%"
              f"{r['prose_rate']*100:>7.0f}%{r['avg_prose_chars']:>9.0f}{r['terminate_rate']*100:>6.0f}%"
              f"{r['avg_n_steps']:>7.1f}{r['n_runs']:>5}  det_succ={ds}(n={r['n_evaluated']})")
    print("\n--- per-domain setup-success (arm -> {domain: ok/n}) ---")
    for r in rows:
        print(f"  {r['arm']}: {r['per_domain_setup']}")
    if args.show_prose:
        print("\n--- sample reasoning prose per arm (MANUAL quality read) ---")
        for r in rows:
            print(f"\n### {r['arm']}")
            for p in r["prose_samples"]:
                print("   |", p.replace(chr(10), " ⏎ "))
    out = os.path.join(args.sweep_root, "differential.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
