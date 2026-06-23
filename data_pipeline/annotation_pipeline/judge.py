#!/usr/bin/env python3
"""Independent LLM judge + diversity metrics for an iteration run.

Separate from stage 02's own verify pass: re-scores each finished
(instruction, trajectory) example from scratch on the 6 quality axes, so the
run-level numbers are a measurement, not the annotator grading its own work.
Also computes programmatic diversity metrics over the instruction set (the old
pipeline's failure: every instruction started "In the <app>, ...").

    python -m annotation_pipeline.judge --run-dir iteration_runs/v1
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2

from annotation_pipeline import prompts
from annotation_pipeline.common import read_jsonl
from annotation_pipeline.keylog_transcript import build_transcript
from annotation_pipeline.labeler import Labeler

# Prompt text lives in prompts.yaml (loaded via annotation_pipeline.prompts).
JUDGE_SYSTEM = prompts.get("judge_system")

BANNED_OPENINGS = re.compile(
    r"^\s*(in the\b|open\b|click\b|navigate\b|go to\b|select\b|scroll\b|press\b|type\b)",
    re.IGNORECASE,
)


def judge_prompt(instruction: str, s: float, e: float, transcript_text: str) -> str:
    return prompts.render(
        "judge",
        instruction=repr(instruction),
        start_s=f"{s:.1f}", end_s=f"{e:.1f}", transcript_text=transcript_text,
    )


def sample_frame_urls(video_path: Path, s: float, e: float, n: int = 10, height: int = 540) -> tuple[list[str], list[str]]:
    """Return (data-url images, per-frame `original_t=..s` text labels) — clean
    frames; the time is interleaved as text, not burned in."""
    cap = cv2.VideoCapture(str(video_path))
    urls: list[str] = []
    labels: list[str] = []
    try:
        if not cap.isOpened():
            return urls, labels
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        span = max(0.0, e - s)
        for i in range(n):
            t = s + (span * i / max(1, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, int(round(t * fps))), max(0, count - 1)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[0] != height:
                sc = height / frame.shape[0]
                frame = cv2.resize(frame, (max(2, int(frame.shape[1] * sc)), height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                urls.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
                labels.append(f"original_t={t:.1f}s")
    finally:
        cap.release()
    return urls, labels


AXES = ["achieves", "monotonic", "boundary_tight", "grounded", "user_prompt_register"]
PASS_AXES = ["achieves", "monotonic", "grounded", "user_prompt_register"]


def diversity(instructions: list[str]) -> dict[str, Any]:
    if not instructions:
        return {"n": 0}
    banned = sum(1 for i in instructions if BANNED_OPENINGS.match(i))
    first_words = [i.strip().split()[0].lower() for i in instructions if i.strip()]
    lengths = [len(i.split()) for i in instructions]
    return {
        "n": len(instructions),
        "banned_opening_rate": round(banned / len(instructions), 3),
        "distinct_first_word_ratio": round(len(set(first_words)) / len(first_words), 3) if first_words else 0,
        "mean_words": round(sum(lengths) / len(lengths), 1),
        "min_words": min(lengths), "max_words": max(lengths),
    }


def judge_run(run_dir: Path, no_cache: bool = False) -> dict[str, Any]:
    lab = Labeler()
    per_clip = []
    all_instructions: list[str] = []
    n_pass = n_total = 0
    for clip_dir in sorted((run_dir / "clips").iterdir()):
        if not clip_dir.is_dir():
            continue
        stage02 = clip_dir / "stage_02"
        traj_path = stage02 / "trajectories_raw.json"
        if not traj_path.exists():
            continue
        manifest = read_jsonl(clip_dir / "stage_00" / "manifest.jsonl")
        if not manifest:
            continue
        row = manifest[0]
        video_path = Path(row["video_path"])
        transcript = build_transcript(Path(row["keylog_path"]))
        cache_dir = stage02 / "judge_cache"
        results = []
        for i, t in enumerate(json.loads(traj_path.read_text()).get("trajectories", [])):
            s, e = t["start_time_s"], t["end_time_s"]
            all_instructions.append(t.get("instruction", ""))
            urls, labels = sample_frame_urls(video_path, s, e)
            try:
                verdict = lab.call_json(
                    JUDGE_SYSTEM,
                    judge_prompt(t.get("instruction", ""), s, e, transcript.render(s, e, max_text_chars=800)),
                    images=urls, image_labels=labels, cache_path=cache_dir / f"judge_{i:04d}.txt", no_cache=no_cache,
                )
            except Exception as exc:  # noqa: BLE001
                verdict = {"error": str(exc)}
            passed = all(bool(verdict.get(a)) for a in PASS_AXES)
            n_total += 1
            n_pass += int(passed)
            results.append({"idx": i, "start_time_s": s, "end_time_s": e,
                            "instruction": t.get("instruction", ""), "pass": passed, "verdict": verdict})
        per_clip.append({"clip_key": clip_dir.name, "n": len(results),
                         "n_pass": sum(r["pass"] for r in results), "results": results})

    report = {
        "run": run_dir.name,
        "n_examples": n_total,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n_total, 3) if n_total else 0.0,
        "diversity": diversity(all_instructions),
        "per_clip": per_clip,
    }
    (run_dir / "judge.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    rep = judge_run(args.run_dir, no_cache=args.no_cache)
    d = rep["diversity"]
    print(f"\nJUDGE {rep['run']}: pass {rep['n_pass']}/{rep['n_examples']} = {rep['pass_rate']:.0%}")
    print(f"diversity: banned_opening_rate={d.get('banned_opening_rate')} "
          f"distinct_first_word_ratio={d.get('distinct_first_word_ratio')} "
          f"mean_words={d.get('mean_words')}")
    print(f"-> {args.run_dir/'judge.json'}")


if __name__ == "__main__":
    main()
