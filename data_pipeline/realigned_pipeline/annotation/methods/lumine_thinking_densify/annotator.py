"""lumine_thinking_densify: text-only densification of an existing run.

AD-HOC / INTERIM (variant A): reads a finished lumine_thinking artifact's
per-clip ledgers (log + carried memory + existing thoughts) plus the
deterministic per-frame action stream, and writes ADDITIONAL thoughts at
decision points the first pass skipped — WITHOUT sending any frames. The
audit is text-only against the same evidence (weaker than the frame-based
gate; rows are stamped verify mode "text" and method
"lumine_thinking_densify" so provenance never blurs with Track A rows).

Needs ``--param source_dir=<lumine_thinking artifact or snapshot>`` whose
``units/<day_tag>/clip_*.json`` ledgers exist for the processed days. The
source is opened READ-ONLY; all outputs land in this run's own artifact.

Anchoring: the day stream is rebuilt deterministically (same filter artifact,
manifest, fps, gap-cut => identical day indices as the source run; each
clip's frame slice is cross-checked against the ledger's recorded range and
skipped loudly on any mismatch). New anchors map day_idx -> (segment_id,
master_idx) exactly like the parent method.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from realigned_pipeline.annotation.lib.days import DayFrame, DayStream, fmt_t, frame_label
from realigned_pipeline.annotation.lib.labeler import ContentFilteredError
from realigned_pipeline.annotation.lib.registry import MethodContext
from realigned_pipeline.lib.common import write_json

INPUT_KIND = "days"
LABELER_DEFAULTS = {"temperature": 0.2, "reasoning_effort": "low"}

KINDS = ("plan", "reorient", "decide", "react", "monitor", "wait")
DEFAULT_MAX_NEW_PER_CLIP = 4
DEFAULT_WRITER_MAX_TOKENS = 8192
DEFAULT_VERIFY_MAX_TOKENS = 8192
# Unlike the parent mechanic, densify clips are INDEPENDENT (no carried
# memory — each clip's memory_in is already in its ledger), so a day can
# process several clips concurrently. Kept small: total concurrency is
# in-flight days x this.
DEFAULT_CLIP_WORKERS = 4


def _tokens(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    return usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0)
                                         + (usage.get("completion_tokens") or 0))


def _norm(text: Any) -> str:
    return " ".join(str(text).split())


def _existing_block(existing: list[dict[str, Any]]) -> str:
    if not existing:
        return "(none)"
    return "\n".join(f"- frame {t['day_idx']} ({t['t_label']}), {t['kind']}: {t['text']}"
                     for t in existing)


def _parse_verdict(obj: Any) -> dict[str, Any]:
    obj = obj if isinstance(obj, dict) else {}
    verdict = "pass" if str(obj.get("verdict", "")).lower() == "pass" else "fail"
    return {"verdict": verdict, "violations": obj.get("violations") or [],
            "reason": _norm(obj.get("reason", "")), "mode": "text"}


def run_unit(item: dict[str, Any], ctx: MethodContext) -> dict[str, Any]:
    day: DayStream = item["day"]
    p = ctx.params
    source_dir = Path(str(p.get("source_dir") or ""))
    if not source_dir.is_dir():
        raise ValueError("lumine_thinking_densify needs --param source_dir=<parent artifact>")
    max_new = int(p.get("max_new_per_clip", DEFAULT_MAX_NEW_PER_CLIP))
    writer_max = int(p.get("writer_max_tokens", DEFAULT_WRITER_MAX_TOKENS))
    verify_max = int(p.get("verify_max_tokens", DEFAULT_VERIFY_MAX_TOKENS))
    # "text" (default) audits each clip's new thoughts; "off" halves cost and
    # wall clock, stamping verdicts pass/mode="none" (unaudited — rows remain
    # distinguishable and a text audit can be run later over the ledgers).
    verify = str(p.get("verify", "text"))
    if verify not in ("text", "off"):
        raise ValueError(f"verify must be 'text' or 'off', got {verify!r}")
    day_units_dir = Path(p["day_units_dir"])
    force = bool(p.get("force"))
    report = p.get("report_tokens") or (lambda n: None)
    spent = {"total": 0}
    spent_lock = threading.Lock()

    def track(n: int) -> None:
        with spent_lock:
            spent["total"] += int(n)
        report(n)

    src_day = source_dir / "units" / day.day_tag
    if not src_day.is_dir():
        return {"thoughts": [], "n_clips": 0, "skipped": "day_not_in_source",
                "verify_mode": "text", "selftest": {"status": "skipped",
                "reason": "text pass"}, "actual_tokens": 0}

    by_idx = {fr.day_idx: fr for fr in day.frames}
    system = ctx.prompts.get("system")
    clip_workers = int(p.get("clip_workers", DEFAULT_CLIP_WORKERS))
    all_new: list[dict[str, Any]] = []
    n_clips = n_mismatch = n_dropped = 0

    def process_clip(rec: dict[str, Any]) -> dict[str, Any]:
        clip_key = str(rec["clip_key"])
        out_path = day_units_dir / f"{clip_key}.json"
        d0, d1 = (int(x) for x in rec["day_idx_range"])
        clip = [by_idx[i] for i in range(d0, d1 + 1) if i in by_idx]
        if len(clip) != int(rec.get("n_frames") or 0):
            return {"status": "mismatch", "thoughts": [], "n_dropped": 0}
        existing = [{"day_idx": int(t["day_idx"]), "t_label": t["t_label"],
                     "kind": t["kind"], "text": t["text"]}
                    for t in rec.get("thoughts", [])]
        taken = {t["day_idx"] for t in existing}
        actions_block = "\n".join(frame_label(fr) for fr in clip)
        prompt = ctx.prompts.render(
            "densify", max_new=max_new, memory=str(rec.get("memory_in", "")),
            log=str(rec.get("log", "")), actions_block=actions_block,
            existing_block=_existing_block(existing))
        try:
            parsed, res = ctx.labeler.call_json_full(
                system, prompt, cache_path=ctx.cache_dir / f"{clip_key}_densify.txt",
                no_cache=ctx.no_cache, max_completion_tokens=writer_max)
        except ContentFilteredError:
            return {"status": "content_filtered", "thoughts": [], "n_dropped": 0}
        track(_tokens(res.usage))

        dropped = 0
        new: list[dict[str, Any]] = []
        for th in (parsed.get("thoughts") or [])[:max_new]:
            if not isinstance(th, dict):
                dropped += 1
                continue
            try:
                di = int(th.get("frame"))
            except (TypeError, ValueError):
                dropped += 1
                continue
            fr: DayFrame | None = by_idx.get(di)
            text = _norm(th.get("text", ""))
            if fr is None or not (d0 <= di <= d1) or di in taken or not text:
                dropped += 1
                continue
            taken.add(di)
            kind = str(th.get("kind", "")).strip()
            new.append({"day_idx": di, "t_day_s": fr.t_day_s, "t_label": fmt_t(fr.t_day_s),
                        "segment_id": fr.segment_id, "recording_id": fr.recording_id,
                        "master_idx": fr.master_idx,
                        "kind": kind if kind in KINDS else "plan", "text": text})
        new.sort(key=lambda t: t["day_idx"])

        if new and verify == "off":
            for t in new:
                t["verify"] = {"verdict": "pass", "violations": [],
                               "reason": "unverified (densify verify=off)", "mode": "none"}
        elif new:
            block = "\n".join(
                f'{k}) anchor frame {t["day_idx"]} ({t["t_label"]}), kind={t["kind"]}: "{t["text"]}"'
                for k, t in enumerate(new, start=1))
            try:
                vparsed, vres = ctx.labeler.call_json_full(
                    ctx.prompts.get("verifier_system"),
                    ctx.prompts.render("verify_text", n_thoughts=len(new),
                                       memory=str(rec.get("memory_in", "")),
                                       log=str(rec.get("log", "")),
                                       actions_block=actions_block,
                                       existing_block=_existing_block(existing),
                                       thoughts_block=block),
                    cache_path=ctx.cache_dir / f"{clip_key}_verify.txt",
                    no_cache=ctx.no_cache, max_completion_tokens=verify_max)
                track(_tokens(vres.usage))
                by_n: dict[int, dict[str, Any]] = {}
                for v in (vparsed.get("verdicts") or []):
                    if isinstance(v, dict):
                        try:
                            by_n[int(v.get("n"))] = v
                        except (TypeError, ValueError):
                            continue
                for k, t in enumerate(new, start=1):
                    t["verify"] = (_parse_verdict(by_n[k]) if k in by_n else
                                   {"verdict": "fail", "violations": ["no_verdict"],
                                    "reason": "auditor returned no verdict", "mode": "text"})
            except ContentFilteredError:
                for t in new:
                    t["verify"] = {"verdict": "fail", "violations": ["content_filter"],
                                   "reason": "text audit blocked", "mode": "text"}

        write_json(out_path, {"clip_key": clip_key, "chunk_index": rec.get("chunk_index"),
                              "day_idx_range": rec["day_idx_range"],
                              "n_existing": len(existing), "thoughts": new,
                              "n_dropped_anchor": dropped})
        return {"status": "ok", "thoughts": new, "n_dropped": dropped}

    todo: list[dict[str, Any]] = []
    for rec_path in sorted(src_day.glob("clip_*.json")):
        rec = json.loads(rec_path.read_text())
        out_path = day_units_dir / f"{rec['clip_key']}.json"
        if out_path.exists() and not force:
            prev = json.loads(out_path.read_text())
            all_new.extend(prev.get("thoughts", []))
            n_clips += 1
            continue
        if rec.get("content_filtered") or not str(rec.get("log", "")).strip():
            continue
        todo.append(rec)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, clip_workers)) as ex:
            for out in ex.map(process_clip, todo):
                if out["status"] == "mismatch":
                    n_mismatch += 1
                    continue
                if out["status"] == "ok":
                    n_clips += 1
                all_new.extend(out["thoughts"])
                n_dropped += int(out["n_dropped"])

    n_pass = sum(1 for t in all_new if (t.get("verify") or {}).get("verdict") == "pass")
    return {
        "thoughts": all_new,
        "n_clips": n_clips,
        "n_clip_mismatch": n_mismatch,
        "n_thoughts": len(all_new),
        "n_pass": n_pass,
        "n_dropped_anchor": n_dropped,
        "verify_mode": "text",
        "selftest": {"status": "skipped", "reason": "text-only pass (variant A)"},
        "actual_tokens": spent["total"],
    }
