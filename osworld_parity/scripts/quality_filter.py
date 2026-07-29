"""Quality-filter teacher rollouts before abs->rel conversion.

Real DesktopEnv.evaluate() task-success is VM-provider-blocked, so we approximate
trajectory quality with three cheap signals and keep the intersection:

  1. HEURISTIC prefilter (no GPU): drop model_error / screenshot_error rollouts,
     drop rollouts whose parse-error fraction is too high, drop trivially-short
     ones, and score terminate-plausibility (terminated success after >= a few
     real actions is good; terminating on step 1 or never terminating is worse).
  2. VLM-JUDGE progress score (SGLang OAI vision, same mechanism as
     data_pipeline/generate_onpolicy_completions_mm.py): a strict judge VLM sees
     the goal + initial/final (+ sampled middle) frames and returns a 0-10 score.
  3. BEST-OF-N: when the collector sampled N rollouts per instruction, keep only
     the highest-judge-score survivor(s) per instruction — this raises keep-rate
     and quality (absolute generation keeps the baseline high enough that best-of-N
     yields usable relative data).

Writes ``keep_slugs.txt`` (one kept run-slug per line, consumed by
convert_abs_to_relative.py --keep_slugs) and ``filter_report.json``.

Run in the juergen/eval uv venv (openai + sglang pinned there). --dry_run does the
heuristic pass only (no GPU / no judge) for quick iteration.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_EVAL = "/fast/home/franz.srambical/juergen/eval"
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)
from osworld_runtime import _pil_to_data_url  # noqa: E402

_JUDGE_SYSTEM = (
    "You are a STRICT evaluator of desktop-GUI-agent trajectories. You are shown a "
    "task GOAL, the INITIAL screen, and the FINAL screen (plus maybe intermediate "
    "frames). Judge ONLY whether the SPECIFIC action described in the goal is VISIBLY "
    "reflected in the FINAL screen.\n"
    "CRITICAL RULES (a manual audit found the teacher OVER-TERMINATES and a lenient "
    "judge rubber-stamped false successes):\n"
    "- Opening the correct APPLICATION is NOT sufficient. If the goal names a specific "
    "menu/button/item to click, that menu must be visibly open OR its effect visible "
    "(e.g. a dialog opened, a new slide added, text typed). App-open-but-action-not-done "
    "= score <= 3.\n"
    "- If the goal's target does not exist / the pane is empty (nothing to click) and no "
    "meaningful action occurred = score <= 3.\n"
    "- A terminal command counts as done ONLY if the command output is visible in the "
    "final screen.\n"
    "- Do NOT reward mere activity, cursor movement, or a plausible-looking screen.\n"
    "Output ONLY a single-line JSON object:\n"
    '{"score": <integer 0-10>, "success": <true|false>, "reason": "<cite the specific evidence in the FINAL screen>"}\n'
    "score 0 = no progress/worse; 5 = the specific action partially done; 8-10 = the "
    "specific instructed action is clearly, verifiably completed in the final screen. "
    "success=true ONLY at score>=8 with explicit final-screen evidence."
)


def _heuristic(result: dict) -> tuple[bool, float, str]:
    """Return (pass_prefilter, terminate_plausibility[0..1], note)."""
    stop = result.get("stop_reason", "")
    n_steps = int(result.get("n_steps", 0))
    parse_err = int(result.get("parse_errors", 0))
    if stop in ("model_error", "screenshot_error"):
        return False, 0.0, f"infra_error:{stop}"
    if n_steps < 1:
        return False, 0.0, "empty"
    if n_steps and parse_err / max(n_steps, 1) > 0.5:
        return False, 0.0, f"parse_err_frac={parse_err}/{n_steps}"
    # terminate plausibility
    if stop == "terminate":
        plaus = 0.4 if n_steps <= 1 else (0.8 if n_steps <= 2 else 1.0)
        note = "terminate_ok" if n_steps > 2 else "terminate_early"
    elif stop.startswith("terminate_"):
        plaus = 0.3
        note = stop
    else:  # max_steps (never terminated) — looping risk
        plaus = 0.5
        note = "no_terminate"
    return True, plaus, note


def _sample_frames(run_dir: Path, n_steps: int, max_frames: int) -> list[Path]:
    steps_dir = run_dir / "steps"
    all_png = sorted(steps_dir.glob("step_*.png"))
    if not all_png:
        return []
    if len(all_png) <= max_frames:
        return all_png
    # always include first and last; sample the rest evenly
    idxs = [0]
    inner = max_frames - 2
    if inner > 0:
        step = (len(all_png) - 1) / (inner + 1)
        idxs += [int(round(step * (i + 1))) for i in range(inner)]
    idxs.append(len(all_png) - 1)
    idxs = sorted(set(min(i, len(all_png) - 1) for i in idxs))
    return [all_png[i] for i in idxs]


def _judge(client, model, instruction, frame_paths, max_tokens=256) -> dict:
    from PIL import Image
    content = [{"type": "text", "text": f"GOAL: {instruction}"}]
    labels = ["INITIAL screen:"] + [f"frame {i}:" for i in range(1, len(frame_paths) - 1)] + ["FINAL screen:"]
    if len(frame_paths) == 1:
        labels = ["screen:"]
    for lbl, fp in zip(labels, frame_paths):
        content.append({"type": "text", "text": lbl})
        img = Image.open(fp).convert("RGB")
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}})
    content.append({"type": "text", "text": "Judge now. Output only the JSON object."})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                  {"role": "user", "content": content}],
        temperature=0.0, max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return _parse_judge(text)


def _parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"score": float(d.get("score", 0)), "success": bool(d.get("success", False)),
                    "reason": str(d.get("reason", ""))[:200], "raw": text[:400]}
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"score\D+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    score = float(m2.group(1)) if m2 else 0.0
    return {"score": score, "success": score >= 8, "reason": "regex_fallback", "raw": text[:400]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--judge_model", default=None,
                   help="HF dir/id served as the judge VLM. Required unless --dry_run.")
    p.add_argument("--min_score", type=float, default=5.0, help="keep iff judge score >= this")
    p.add_argument("--best_of_n", action="store_true",
                   help="keep only the top-judge-score survivor per instruction")
    p.add_argument("--max_judge_frames", type=int, default=4)
    p.add_argument("--dry_run", action="store_true", help="heuristics only, no judge/GPU")
    p.add_argument("--sglang_port", type=int, default=0)
    p.add_argument("--sglang_api_key", default="judge")
    p.add_argument("--mem_fraction_static", type=float, default=0.85)
    args = p.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(rollouts_dir.iterdir()):
        if not d.is_dir() or not (d / "result.json").is_file():
            continue
        result = json.loads((d / "result.json").read_text())
        pf, plaus, note = _heuristic(result)
        runs.append({"slug": d.name, "dir": d, "instruction": result.get("instruction"),
                     "stop_reason": result.get("stop_reason"), "n_steps": int(result.get("n_steps", 0)),
                     "parse_errors": int(result.get("parse_errors", 0)),
                     "prefilter_pass": pf, "terminate_plausibility": plaus, "heuristic_note": note,
                     "judge_score": None, "judge_success": None, "judge_reason": None})
    print(f"[filter] {len(runs)} rollouts; {sum(r['prefilter_pass'] for r in runs)} pass heuristic prefilter")

    survivors = [r for r in runs if r["prefilter_pass"]]

    if not args.dry_run:
        if not args.judge_model:
            print("ERROR: --judge_model required unless --dry_run", file=sys.stderr)
            return 1
        import openai
        from sglang_runner import sglang_server
        with sglang_server(model_path=args.judge_model, port=args.sglang_port,
                           api_key=args.sglang_api_key, log_path=out_dir / "judge_sglang.log",
                           mem_fraction_static=args.mem_fraction_static) as base_url:
            client = openai.OpenAI(base_url=base_url, api_key=args.sglang_api_key)
            for r in survivors:
                frames = _sample_frames(r["dir"], r["n_steps"], args.max_judge_frames)
                if not frames:
                    r["judge_score"] = 0.0
                    r["judge_reason"] = "no_frames"
                    continue
                try:
                    j = _judge(client, args.judge_model, r["instruction"], frames)
                except Exception as e:
                    j = {"score": 0.0, "success": False, "reason": f"judge_error:{e}", "raw": ""}
                r["judge_score"], r["judge_success"], r["judge_reason"] = j["score"], j["success"], j["reason"]
                print(f"[judge] {r['slug']:<45} score={j['score']:.1f} success={j['success']} "
                      f"stop={r['stop_reason']} n={r['n_steps']} :: {j['reason']}")

    # keep decision
    def _kept(r):
        if not r["prefilter_pass"]:
            return False
        if args.dry_run:
            return True  # heuristic-only mode keeps all prefilter survivors
        return (r["judge_score"] or 0) >= args.min_score

    for r in runs:
        r["kept_prejudge"] = _kept(r)

    if args.best_of_n and not args.dry_run:
        by_instr = defaultdict(list)
        for r in runs:
            if r["kept_prejudge"]:
                by_instr[r["instruction"]].append(r)
        keep_slugs = set()
        for instr, group in by_instr.items():
            best = max(group, key=lambda r: (r["judge_score"] or 0, r["terminate_plausibility"]))
            keep_slugs.add(best["slug"])
    else:
        keep_slugs = {r["slug"] for r in runs if r["kept_prejudge"]}

    for r in runs:
        r["kept"] = r["slug"] in keep_slugs
        r.pop("dir", None)

    (out_dir / "keep_slugs.txt").write_text("\n".join(sorted(keep_slugs)) + ("\n" if keep_slugs else ""))
    report = {"rollouts_dir": str(rollouts_dir), "n_total": len(runs),
              "n_prefilter_pass": sum(r["prefilter_pass"] for r in runs),
              "n_kept": len(keep_slugs), "min_score": args.min_score,
              "best_of_n": args.best_of_n, "dry_run": args.dry_run,
              "judge_model": args.judge_model, "runs": runs}
    (out_dir / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(f"[filter] kept {len(keep_slugs)}/{len(runs)} -> {out_dir/'keep_slugs.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
