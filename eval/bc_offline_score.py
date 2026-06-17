"""Offline imitation-fidelity monitor for BC SFT checkpoints.

This is NOT a correctness oracle. The held-out actions are noisy single-human
demonstrations in a *relative*, *multi-step*, *cursorless* action space, so we
cannot decide offline whether a predicted action "clicked the right thing"
(there is no golden target and no recoverable cursor anchor). Instead we measure
how well a checkpoint *imitates the human action distribution* — a training-
health / regression / collapse signal, meant to be tracked **relative across
checkpoints and against the base model**, never read as a capability score.

Teacher-forced setup (the generation runner feeds us the pairs): the model sees
the real screenshot history up to step t and predicts action t; we compare that
prediction to the human's action t.

Metrics
  - format_validity_rate     fraction of predictions that parse at all
  - type_accuracy / confusion over {no_op, move, scroll, click, key, terminate}
                             (reported overall AND on decision points, i.e.
                             gold != no_op, so idle steps can't inflate it)
  - move_dir_cosine          cosine(pred_delta, gold_delta) on gold-move steps
  - move_mag_relerr          |‖pred‖-‖gold‖| / ‖gold‖ on gold-move steps
  - click / terminate        precision & recall of the discrete decision

Modes
  profile   gold-only distribution over a val.jsonl. No model needed — runnable
            now, to sanity-check the eval set and establish baselines.
  score     given a jsonl of {"gold": <str>, "pred": <str>} pairs, compute the
            metrics and write the pmanager/labctl result.json. The per-checkpoint
            sglang teacher-forcing runner produces that pairs file.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from action_parser import Action, parse_action, parse_action_tolerant
from result import write_result

# Action-type taxonomy, in priority order: a step that both moves and clicks is
# a "click" (the click is the salient intent; the move is just the approach).
ACTION_TYPES = ("terminate", "click", "key", "scroll", "move", "no_op")
_TERMINATE = "TERMINATE"


def classify(act: Action | None, *, is_terminate: bool) -> str:
    """Map a parsed action to its salient type. ``None`` act + terminate flag."""
    if is_terminate:
        return "terminate"
    assert act is not None
    if act.no_op:
        return "no_op"
    if any(e.mouse_button == 1 for e in act.events):  # LMB press or release
        return "click"
    if act.events:  # keyboard or other mouse-button transitions
        return "key"
    if act.scroll != 0 and act.dx == 0 and act.dy == 0:
        return "scroll"
    return "move" if (act.dx or act.dy) else "no_op"


def parse_any(text: str, *, tolerant: bool) -> tuple[Action | None, bool, bool]:
    """Parse one action string.

    Returns ``(action, is_terminate, ok)``. ``TERMINATE`` is handled out-of-band
    because it is not part of the mouse/key grammar the parser accepts.
    """
    if text is not None and text.strip().split("\n", 1)[0].strip() == _TERMINATE:
        return None, True, True
    try:
        act = parse_action_tolerant(text) if tolerant else parse_action(text)
        return act, False, True
    except (ValueError, TypeError):
        return None, False, False


def _has_left_click(act: Action | None) -> bool:
    return act is not None and (act.has_left_click_press or act.has_left_click_release)


def score_pairs(pairs: list[tuple[str, str]]) -> dict:
    """Compute imitation-fidelity metrics over (gold, pred) action strings."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no (gold, pred) pairs to score")

    n_valid = 0
    confusion: dict[str, Counter] = defaultdict(Counter)  # confusion[gold][pred]
    cosines: list[float] = []
    mag_relerrs: list[float] = []
    # discrete-decision tallies for precision/recall
    click_tp = click_fp = click_fn = 0
    term_tp = term_fp = term_fn = 0

    for gold_s, pred_s in pairs:
        g_act, g_term, _ = parse_any(gold_s, tolerant=False)  # gold is clean data
        gtype = classify(g_act, is_terminate=g_term)

        p_act, p_term, p_ok = parse_any(pred_s, tolerant=True)
        if p_ok:
            n_valid += 1
            ptype = classify(p_act, is_terminate=p_term)
        else:
            ptype = "INVALID"
        confusion[gtype][ptype] += 1

        # Move geometry: only where the human actually moved the cursor and the
        # prediction is a parseable move too.
        gold_moved = not g_term and g_act is not None and (g_act.dx or g_act.dy)
        pred_moved = p_ok and not p_term and p_act is not None and (p_act.dx or p_act.dy)
        if gold_moved and pred_moved:
            gx, gy, px, py = g_act.dx, g_act.dy, p_act.dx, p_act.dy
            gnorm = math.hypot(gx, gy)
            pnorm = math.hypot(px, py)
            if gnorm > 0 and pnorm > 0:
                cosines.append((gx * px + gy * py) / (gnorm * pnorm))
                mag_relerrs.append(abs(pnorm - gnorm) / gnorm)

        # Click decision (LMB).
        g_click, p_click = _has_left_click(g_act), (p_ok and _has_left_click(p_act))
        click_tp += g_click and p_click
        click_fp += (not g_click) and p_click
        click_fn += g_click and (not p_click)

        # Terminate decision.
        term_tp += g_term and p_term
        term_fp += (not g_term) and p_term
        term_fn += g_term and (not p_term)

    # Aggregate.
    correct = sum(confusion[t].get(t, 0) for t in ACTION_TYPES)
    decision_total = sum(sum(confusion[g].values()) for g in ACTION_TYPES if g != "no_op")
    decision_correct = sum(confusion[g].get(g, 0) for g in ACTION_TYPES if g != "no_op")

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    cp, cr, cf = _prf(click_tp, click_fp, click_fn)
    tp_, tr, tf = _prf(term_tp, term_fp, term_fn)

    scores = {
        "format_validity_rate": n_valid / n,
        "type_accuracy_overall": correct / n,
        "type_accuracy_decision": (decision_correct / decision_total) if decision_total else 0.0,
        "move_dir_cosine_mean": (sum(cosines) / len(cosines)) if cosines else 0.0,
        "move_mag_relerr_median": _median(mag_relerrs) if mag_relerrs else 0.0,
        "click_precision": cp,
        "click_recall": cr,
        "click_f1": cf,
        "terminate_precision": tp_,
        "terminate_recall": tr,
        "terminate_f1": tf,
    }
    confusion_plain = {g: dict(confusion[g]) for g in confusion}
    return {"scores": scores, "confusion": confusion_plain, "n_move_steps": len(cosines)}


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def iter_gold_actions(val_jsonl: Path, limit: int = 0):
    """Yield every assistant action string from a val.jsonl (BC samples format)."""
    n = 0
    with Path(val_jsonl).open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            for m in rec.get("messages", []):
                if m.get("role") == "assistant":
                    yield _text_of(m.get("content")).strip()
            n += 1
            if limit and n >= limit:
                return


def profile(val_jsonl: Path, limit: int = 0) -> dict:
    """Gold-only distribution over the val set — no model required."""
    type_counts: Counter = Counter()
    n_unparseable = 0
    abs_dx: list[int] = []
    abs_dy: list[int] = []
    euclid: list[float] = []
    total = 0
    for s in iter_gold_actions(val_jsonl, limit=limit):
        total += 1
        act, term, ok = parse_any(s, tolerant=False)
        if not ok:
            n_unparseable += 1
            type_counts["UNPARSEABLE"] += 1
            continue
        type_counts[classify(act, is_terminate=term)] += 1
        if act is not None and (act.dx or act.dy):
            abs_dx.append(abs(act.dx))
            abs_dy.append(abs(act.dy))
            euclid.append(math.hypot(act.dx, act.dy))
    decision = total - type_counts.get("no_op", 0) - type_counts.get("UNPARSEABLE", 0)
    return {
        "n_actions": total,
        "n_unparseable_gold": n_unparseable,
        "decision_point_fraction": (decision / total) if total else 0.0,
        "type_distribution": {t: type_counts.get(t, 0) for t in (*ACTION_TYPES, "UNPARSEABLE")},
        "move_delta_stats": {
            "n": len(euclid),
            "median_euclid": _median(euclid) if euclid else 0.0,
            "p95_euclid": (sorted(euclid)[int(0.95 * len(euclid))] if euclid else 0.0),
            "median_abs_dx": _median([float(x) for x in abs_dx]) if abs_dx else 0.0,
            "median_abs_dy": _median([float(y) for y in abs_dy]) if abs_dy else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_prof = sub.add_parser("profile", help="gold-only distribution over a val.jsonl")
    p_prof.add_argument("--val_jsonl", required=True)
    p_prof.add_argument("--limit", type=int, default=0, help="max sessions (0 = all)")

    p_score = sub.add_parser("score", help="score (gold,pred) pairs → result.json")
    p_score.add_argument("--pairs_jsonl", required=True, help='jsonl of {"gold","pred"}')
    p_score.add_argument("--output_dir", required=True)
    p_score.add_argument("--task", default="bc_offline_imitation")

    args = ap.parse_args()

    if args.mode == "profile":
        out = profile(Path(args.val_jsonl), limit=args.limit)
        print(json.dumps(out, indent=2))
        return

    pairs = []
    with Path(args.pairs_jsonl).open() as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                rec = json.loads(line)
                pairs.append((rec["gold"], rec["pred"]))
    t0 = time.time()
    res = score_pairs(pairs)
    out_dir = Path(args.output_dir)
    write_result(
        out_dir / "result.json",
        task=args.task,
        scores={f"bc_offline/{k}": v for k, v in res["scores"].items()},
        params={},
        inputs={"pairs_jsonl": args.pairs_jsonl, "n_pairs": len(pairs)},
        n_samples=len(pairs),
        elapsed_s=int(time.time() - t0),
        extra={"confusion": res["confusion"], "n_move_steps": res["n_move_steps"]},
    )
    print(json.dumps({f"bc_offline/{k}": v for k, v in res["scores"].items()}, indent=2))


if __name__ == "__main__":
    main()
