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
  - move_coverage            fraction of gold-move steps where the prediction
                             also produced a nonzero net move
  - move_dir_cosine          cosine(pred_delta, gold_delta) over ALL gold-move
                             steps; a gold-move step the model answers without a
                             move scores 0, so the denominator is the gold set
                             and cannot drift with the checkpoint. Reported as
                             mean / median / frac(>0.9) / frac(<0), plus the
                             matched-only mean under ``_matched`` for continuity
                             with older runs.
  - move_mag_relerr          |‖pred‖-‖gold‖| / ‖gold‖ on matched move steps
  - move_big_*               same family restricted to gold moves of at least
                             MAG_FLOOR thousandths — below that a move is finer
                             than one merged vision token and unresolvable
  - click / terminate        precision & recall of the discrete decision

Every metric above is also reported per ``pool`` ("success" / "failure") when
the pairs file carries that field.

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

from action_parser import (
    Action,
    KeyEvent,
    OrderedAction,
    parse_action,
    parse_action_tolerant,
    parse_ordered_action,
    parse_ordered_action_tolerant,
)
from result import write_result

ACTION_FORMATS = ("legacy", "oev3")

MAG_FLOOR = 50

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


def _ordered_to_action(oa: OrderedAction) -> Action:
    """Project an ``ordered_events_v3`` action onto the legacy ``Action`` shape.

    Move deltas are summed (net cursor displacement), scroll keeps the
    vertical component (horizontal as fallback), down/up become press/release
    events, and each ``type(...)`` becomes a synthetic TYPE press+release so
    typing classifies as "key".
    """
    if oa.no_op:
        return Action(dx=0, dy=0, scroll=0, events=(), no_op=True)
    dx = dy = scroll = 0
    events: list[KeyEvent] = []
    for p in oa.primitives:
        if p.kind == "move":
            dx += p.dx
            dy += p.dy
        elif p.kind == "scroll":
            scroll += p.dy if p.dy else (p.dx or 0)
        elif p.kind == "down":
            events.append(KeyEvent(kind="press", what=p.name, mouse_button=p.mouse_button))
        elif p.kind == "up":
            events.append(KeyEvent(kind="release", what=p.name, mouse_button=p.mouse_button))
        elif p.kind == "type":
            events.append(KeyEvent(kind="press", what="TYPE", mouse_button=None))
            events.append(KeyEvent(kind="release", what="TYPE", mouse_button=None))
    return Action(dx=dx, dy=dy, scroll=scroll, events=tuple(events), no_op=False)


def parse_any(
    text: str, *, tolerant: bool, action_format: str = "legacy"
) -> tuple[Action | None, bool, bool]:
    """Parse one action string.

    Returns ``(action, is_terminate, ok)``. ``TERMINATE`` is handled out-of-band
    because it is not part of the mouse/key grammar the parser accepts.
    """
    if text is not None and text.strip().split("\n", 1)[0].strip() == _TERMINATE:
        return None, True, True
    try:
        if action_format == "oev3":
            oa = parse_ordered_action_tolerant(text) if tolerant else parse_ordered_action(text)
            return _ordered_to_action(oa), False, True
        act = parse_action_tolerant(text) if tolerant else parse_action(text)
        return act, False, True
    except (ValueError, TypeError):
        return None, False, False


def _has_left_click(act: Action | None) -> bool:
    return act is not None and (act.has_left_click_press or act.has_left_click_release)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def analyze_pair(gold_s: str, pred_s: str, action_format: str = "legacy") -> dict:
    """Reduce one (gold, pred) action pair to the per-step facts we aggregate."""
    g_act, g_term, _ = parse_any(gold_s, tolerant=False, action_format=action_format)
    p_act, p_term, p_ok = parse_any(pred_s, tolerant=True, action_format=action_format)

    gold_moved = bool(not g_term and g_act is not None and (g_act.dx or g_act.dy))
    pred_moved = bool(p_ok and not p_term and p_act is not None and (p_act.dx or p_act.dy))
    gold_norm = math.hypot(g_act.dx, g_act.dy) if gold_moved else 0.0
    cosine = relerr = None
    if gold_moved:
        # A gold move the model answers without any move is a total aim miss,
        # not a step to drop: it scores 0 so the denominator stays the gold set.
        cosine = 0.0
        if pred_moved:
            pnorm = math.hypot(p_act.dx, p_act.dy)
            cosine = (g_act.dx * p_act.dx + g_act.dy * p_act.dy) / (gold_norm * pnorm)
            relerr = abs(pnorm - gold_norm) / gold_norm

    return {
        "gold_type": classify(g_act, is_terminate=g_term),
        "pred_type": classify(p_act, is_terminate=p_term) if p_ok else "INVALID",
        "pred_valid": p_ok,
        "gold_moved": gold_moved,
        "pred_moved": pred_moved,
        "gold_norm": gold_norm,
        "cosine": cosine,
        "relerr": relerr,
        "gold_click": _has_left_click(g_act),
        "pred_click": bool(p_ok and _has_left_click(p_act)),
        "gold_term": bool(g_term),
        "pred_term": bool(p_ok and p_term),
    }


def _move_scores(steps: list[dict], prefix: str, mag_floor: float = 0.0) -> dict:
    """Aim metrics over the gold-move steps whose |delta| clears ``mag_floor``."""
    sel = [s for s in steps if s["gold_moved"] and s["gold_norm"] >= mag_floor]
    matched = [s for s in sel if s["pred_moved"]]
    cosines = [s["cosine"] for s in sel]
    relerrs = [s["relerr"] for s in matched]
    n = len(sel)
    return {
        f"{prefix}n": n,
        f"{prefix}coverage": (len(matched) / n) if n else 0.0,
        f"{prefix}dir_cosine_mean": (sum(cosines) / n) if n else 0.0,
        f"{prefix}dir_cosine_median": _median(cosines) if cosines else 0.0,
        f"{prefix}dir_cosine_frac_above_0p9": (
            sum(1 for c in cosines if c > 0.9) / n if n else 0.0
        ),
        f"{prefix}dir_cosine_frac_negative": (
            sum(1 for c in cosines if c < 0) / n if n else 0.0
        ),
        f"{prefix}dir_cosine_mean_matched": (
            sum(s["cosine"] for s in matched) / len(matched) if matched else 0.0
        ),
        f"{prefix}mag_relerr_median": _median(relerrs) if relerrs else 0.0,
    }


def aggregate_steps(steps: list[dict]) -> dict:
    """Turn per-step facts into the reported score dict."""
    n = len(steps)
    if n == 0:
        raise ValueError("no (gold, pred) pairs to score")
    confusion: dict[str, Counter] = defaultdict(Counter)
    for s in steps:
        confusion[s["gold_type"]][s["pred_type"]] += 1

    correct = sum(confusion[t].get(t, 0) for t in ACTION_TYPES)
    decision_total = sum(sum(confusion[g].values()) for g in ACTION_TYPES if g != "no_op")
    decision_correct = sum(confusion[g].get(g, 0) for g in ACTION_TYPES if g != "no_op")

    cp, cr, cf = _prf(
        sum(1 for s in steps if s["gold_click"] and s["pred_click"]),
        sum(1 for s in steps if not s["gold_click"] and s["pred_click"]),
        sum(1 for s in steps if s["gold_click"] and not s["pred_click"]),
    )
    tp_, tr, tf = _prf(
        sum(1 for s in steps if s["gold_term"] and s["pred_term"]),
        sum(1 for s in steps if not s["gold_term"] and s["pred_term"]),
        sum(1 for s in steps if s["gold_term"] and not s["pred_term"]),
    )

    scores = {
        "n_pairs": n,
        "format_validity_rate": sum(1 for s in steps if s["pred_valid"]) / n,
        "type_accuracy_overall": correct / n,
        "type_accuracy_decision": (decision_correct / decision_total) if decision_total else 0.0,
        **_move_scores(steps, "move_"),
        **_move_scores(steps, "move_big_", mag_floor=MAG_FLOOR),
        "click_precision": cp,
        "click_recall": cr,
        "click_f1": cf,
        "terminate_precision": tp_,
        "terminate_recall": tr,
        "terminate_f1": tf,
    }
    return {"scores": scores, "confusion": {g: dict(confusion[g]) for g in confusion}}


def score_pairs(
    pairs: list[tuple[str, str]],
    action_format: str = "legacy",
    pools: list | None = None,
) -> dict:
    """Compute imitation-fidelity metrics over (gold, pred) action strings."""
    steps = [analyze_pair(g, p, action_format=action_format) for g, p in pairs]
    out = aggregate_steps(steps)
    out["n_move_steps"] = sum(1 for s in steps if s["gold_moved"])
    out["n_move_steps_matched"] = sum(1 for s in steps if s["gold_moved"] and s["pred_moved"])

    if pools:
        by_pool: dict[str, dict] = {}
        for pool in sorted({p for p in pools if p}):
            sub = [s for s, p in zip(steps, pools) if p == pool]
            if sub:
                by_pool[pool] = aggregate_steps(sub)["scores"]
        if by_pool:
            out["by_pool"] = by_pool
    return out


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


def profile(val_jsonl: Path, limit: int = 0, action_format: str = "legacy") -> dict:
    """Gold-only distribution over the val set — no model required."""
    type_counts: Counter = Counter()
    n_unparseable = 0
    abs_dx: list[int] = []
    abs_dy: list[int] = []
    euclid: list[float] = []
    total = 0
    for s in iter_gold_actions(val_jsonl, limit=limit):
        total += 1
        act, term, ok = parse_any(s, tolerant=False, action_format=action_format)
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
    p_prof.add_argument("--action_format", choices=ACTION_FORMATS, default="legacy")

    p_score = sub.add_parser("score", help="score (gold,pred) pairs → result.json")
    p_score.add_argument("--pairs_jsonl", required=True, help='jsonl of {"gold","pred"}')
    p_score.add_argument("--output_dir", required=True)
    p_score.add_argument("--task", default="bc_offline_imitation")
    p_score.add_argument("--action_format", choices=ACTION_FORMATS, default="legacy")

    args = ap.parse_args()

    if args.mode == "profile":
        out = profile(Path(args.val_jsonl), limit=args.limit, action_format=args.action_format)
        print(json.dumps(out, indent=2))
        return

    pairs = []
    pools = []
    with Path(args.pairs_jsonl).open() as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                rec = json.loads(line)
                pairs.append((rec["gold"], rec["pred"]))
                pools.append(rec.get("pool"))
    t0 = time.time()
    res = score_pairs(pairs, action_format=args.action_format, pools=pools)
    out_dir = Path(args.output_dir)
    extra = {
        "confusion": res["confusion"],
        "n_move_steps": res["n_move_steps"],
        "n_move_steps_matched": res["n_move_steps_matched"],
    }
    if "by_pool" in res:
        extra["by_pool"] = res["by_pool"]
    write_result(
        out_dir / "result.json",
        task=args.task,
        scores={f"bc_offline/{k}": v for k, v in res["scores"].items()},
        params={"action_format": args.action_format, "mag_floor": MAG_FLOOR},
        inputs={"pairs_jsonl": args.pairs_jsonl, "n_pairs": len(pairs)},
        n_samples=len(pairs),
        elapsed_s=int(time.time() - t0),
        extra=extra,
    )
    print(json.dumps({f"bc_offline/{k}": v for k, v in res["scores"].items()}, indent=2))


if __name__ == "__main__":
    main()
