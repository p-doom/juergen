#!/usr/bin/env python3
"""True closed-loop evaluator for one absolute or move_rel checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from .contract import (
        Contract,
        ContractError,
        EXPECTED_ACTION,
        SAMPLING,
        Semantic,
        data_url,
        load_frozen,
        load_jsonl,
        oscillates,
        request_seed,
        sha256_bytes,
        sha256_file,
        strict_schema_ok,
        unit_range_ok,
    )
    from .metrics import summarize
except ImportError:  # direct script execution
    from contract import (
    Contract,
    ContractError,
    EXPECTED_ACTION,
    SAMPLING,
    Semantic,
    data_url,
    load_frozen,
    load_jsonl,
    oscillates,
    request_seed,
    sha256_bytes,
    sha256_file,
    strict_schema_ok,
    unit_range_ok,
    )
    from metrics import summarize

TRUSTED_OUTPUTS = ("rows.jsonl", "report.json", "eval_manifest.json")


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"manifest is not an object: {path}")
    return value


def validate_episode_artifact(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    manifest = _load_manifest(root / "build_manifest.json")
    expected = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_phasea_episodes",
        "status": "complete",
        "preamble": False,
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items()
                if manifest.get(key) != value}
    if mismatch:
        raise ContractError(f"wrong episode artifact: {mismatch}")
    for name, digest in manifest.get("artifact_sha256", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ContractError(f"episode artifact hash mismatch: {path}")
    identity = manifest.get("step1_identity", {})
    if not (
        identity.get("checked") == manifest.get("n_episodes")
        and identity.get("byte_equal") == manifest.get("n_episodes")
        and identity.get("geometry_equal") == manifest.get("n_episodes")
    ):
        raise ContractError(f"step-1 identity guard absent: {identity}")
    if any(value != 1.0 for value in manifest.get("oracle_rate", {}).values()):
        raise ContractError(f"oracle is not 100%: {manifest.get('oracle_rate')}")
    if not manifest.get("oracle_observations_and_states_identical"):
        raise ContractError("oracle cross-semantic identity guard absent")
    specs = load_jsonl(root / "episode_specs.jsonl")
    if len(specs) != manifest["n_episodes"]:
        raise ContractError("episode spec count mismatch")
    return manifest, specs


def model_provenance(
    model_dir: Path,
    semantic: Semantic,
    checkpoint_alias: str,
    *,
    preamble: bool = False,
    comparison_label: str = "primary",
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        raise ContractError(f"model config missing: {model_dir / 'config.json'}")
    candidates = [
        model_dir.parent / "train_export_manifest.json",
        model_dir.parent / "export_manifest.json",
        model_dir.parent / "curriculum_train_export_manifest.json",
    ]
    manifests = [path for path in candidates if path.is_file()]
    if len(manifests) != 1:
        raise ContractError(f"expected one model export manifest beside {model_dir}: {manifests}")
    path = manifests[0]
    manifest = _load_manifest(path)
    expected_arm = {
        ("absolute_toolcall", False): "abstool_act",
        ("absolute_toolcall", True): "abstool_pre",
        ("move_rel", False): "reltool_act",
        ("move_rel", True): "reltool_pre",
        ("deltatype_raw", False): "relraw_act",
        ("deltatype_raw", True): "relraw_pre",
    }[(semantic, preamble)]
    if comparison_label in ("curriculum_transfer", "curriculum_transfer_lr5e5"):
        frozen = load_frozen()["curriculum_transfer"]
        models = (frozen["stage2_models"] if comparison_label == "curriculum_transfer"
                  else frozen["low_lr_rescue_prepared"]["models"])
        inverse = {alias: branch for branch, alias in models.items()}
        branch = inverse.get(checkpoint_alias)
        fixed = {
            "artifact_type": "synthetic_multistep_curriculum_hf_checkpoint",
            "schema_version": 1, "status": "complete", "branch": branch,
            "target_format": "deltatype_raw_pre", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "step": 750, "fresh_optimizer": True, "lora_rank": 256,
            "lora_alpha": 256, "hf_subdir": "hf",
        }
        if comparison_label == "curriculum_transfer_lr5e5":
            fixed["learning_rate"] = 5e-5
    else:
        fixed = {
            "artifact_type": "relative_factorial_hf_checkpoint",
            "schema_version": 1,
            "status": "complete",
            "arm": expected_arm,
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "step": 750,
            "hf_subdir": "hf",
        }
    mismatch = {key: (manifest.get(key), value) for key, value in fixed.items()
                if manifest.get(key) != value}
    if mismatch:
        raise ContractError(f"wrong model for {semantic}: {mismatch}")
    if model_dir != (path.parent / manifest["hf_subdir"]).resolve():
        raise ContractError("model path is not the export manifest's HF subdirectory")
    weights = sorted(
        {p.name: p.stat().st_size for p in model_dir.glob("*.safetensors")}.items()
    )
    if not weights:
        raise ContractError(f"no safetensors weights in {model_dir}")
    return {
        "checkpoint_alias": checkpoint_alias,
        "arm": expected_arm,
        "branch": manifest.get("branch"),
        "lora_rank": manifest.get("lora_rank"),
        "lora_alpha": manifest.get("lora_alpha"),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "model_dir": str(model_dir),
        "export_manifest": str(path.resolve()),
        "export_manifest_sha256": sha256_file(path),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "weights": [{"name": name, "size": size} for name, size in weights],
    }


def _call_model(
    client: Any,
    *,
    model: str,
    system: str,
    user_text: str,
    png: bytes,
    history: list[dict[str, Any]],
    seed: int,
    max_tokens: int,
) -> tuple[str, Any, dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": data_url(png)}},
        ],
    })
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        seed=seed,
        extra_body={"top_k": SAMPLING["top_k"]},
    )
    message = completion.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    content = message.content or ""
    raw = content
    if tool_calls:
        raw += " | tool_calls=" + json.dumps([
            {
                "name": getattr(getattr(call, "function", call), "name", None),
                "arguments": getattr(getattr(call, "function", call), "arguments", None),
            }
            for call in tool_calls
        ])
    usage = getattr(completion, "usage", None)
    return raw, tool_calls, {
        "completion_tokens": getattr(usage, "completion_tokens", None),
        # Keep exactly what the model supplied as assistant prose.  The raw field
        # additionally carries structured-call diagnostics for auditability.
        "history_text": content if content else raw,
    }


def run_episode(
    client: Any,
    *,
    model: str,
    semantic: Semantic,
    contract: Contract,
    episode_root: Path,
    spec: dict[str, Any],
    max_attempts: int,
    history_turns: int,
    max_tokens: int,
    preamble: bool = False,
    k: int = 0,
) -> dict[str, Any]:
    episode_id = spec["episode_id"]
    cursor = tuple(spec["initial_cursor"])
    history: list[dict[str, Any]] = []
    prior: list[str] = []
    steps: list[dict[str, Any]] = []
    previous_move: tuple[int, int] | None = None
    outcome = "completed"
    reached_targets = 0
    global_step = 0

    for target_index, target_spec in enumerate(spec["targets"]):
        bbox = target_spec["bbox"]
        target = tuple(target_spec["target_center"])
        previous_move = None
        hit = False
        for attempt in range(1, max_attempts + 1):
            global_step += 1
            cursor_before = cursor
            if target_index == 0 and attempt == 1:
                png = (episode_root / spec["step1_image"]).read_bytes()
                if sha256_bytes(png) != spec["step1_png_sha256"]:
                    raise ContractError(f"step-1 image hash drift: {episode_id}")
                # Re-rendering equality catches both stale cursor state and canvas drift.
                if png != contract.render_png(bbox, cursor):
                    raise ContractError(f"step-1 render identity failed at eval: {episode_id}")
            else:
                png = contract.render_png(bbox, cursor)
            user_text = contract.user_text(
                semantic,
                cursor,
                target,
                target_index=target_index,
                target_count=len(spec["targets"]),
                preamble=preamble,
                prior=prior[-history_turns:] if history_turns else None,
            )
            seed = request_seed(episode_id, k, target_index, attempt)
            distance_before = contract.distance_to_box(cursor, bbox)
            raw, tool_calls, meta = _call_model(
                client,
                model=model,
                system=contract.system_prompt(semantic),
                user_text=user_text,
                png=png,
                history=history[-2 * history_turns:] if history_turns else [],
                seed=seed,
                max_tokens=max_tokens,
            )
            parse_text = raw.split(" | tool_calls=", 1)[0]
            move = contract.parse(semantic, parse_text, tool_calls)
            if move.coord is None:
                cursor_after = cursor
                movement = (0, 0)
            else:
                cursor_after = contract.apply_coord(semantic, cursor, move.coord)
                movement = (
                    cursor_after[0] - cursor_before[0],
                    cursor_after[1] - cursor_before[1],
                )
            # Recompute, do not trust mutable state: this is the cursor-state guard.
            if move.coord is not None and cursor_after != contract.apply_coord(
                semantic, cursor_before, move.coord
            ):
                raise ContractError("cursor update is not a pure application of the emitted coord")
            cursor = cursor_after
            distance_after = contract.distance_to_box(cursor, bbox)
            hit = contract.in_bbox(cursor, bbox)
            schema_ok = strict_schema_ok(semantic, parse_text, move.coord)
            step = {
                "global_step": global_step,
                "target_index": target_index,
                "attempt": attempt,
                "cursor_before": list(cursor_before),
                "cursor_after": list(cursor_after),
                "bbox": bbox,
                "target_center": list(target),
                "observation_sha256": sha256_bytes(png),
                "sampling_seed": seed,
                "sampling": dict(SAMPLING),
                "raw_output": raw,
                "completion_tokens": meta["completion_tokens"],
                "action": move.action,
                "expected_action": EXPECTED_ACTION[semantic],
                "coord": list(move.coord) if move.coord is not None else None,
                "parse_ok": move.parse_ok,
                "schema_ok": schema_ok,
                "unit_range_ok": unit_range_ok(semantic, move.coord),
                "terminate": move.terminate,
                "movement_px": list(movement),
                "distance_before": distance_before,
                "distance_after": distance_after,
                "progress_px": distance_before - distance_after,
                "regression": distance_after > distance_before,
                "oscillation": oscillates(previous_move, movement),
                "hit": hit,
            }
            steps.append(step)
            # Full output is retained in both explicit prior prose and history.
            prior.append(raw)
            history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": meta["history_text"]},
            ])
            if move.coord is not None:
                previous_move = movement
            if move.terminate:
                outcome = "terminate"
                break
            if hit:
                reached_targets += 1
                break
        if outcome == "terminate":
            break
        if not hit:
            outcome = "attempt_budget_exhausted"
            break

    completed = reached_targets == len(spec["targets"])
    if completed:
        outcome = "completed"
    return {
        "episode_id": episode_id,
        "episode_index": spec["episode_index"],
        "kind": spec["kind"],
        "semantic": semantic,
        "preamble": preamble,
        "k": k,
        "initial_cursor": spec["initial_cursor"],
        "target_count": len(spec["targets"]),
        "reached_targets": reached_targets,
        "completed": completed,
        "outcome": outcome,
        "final_cursor": list(cursor),
        "n_steps": len(steps),
        "steps": steps,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in TRUSTED_OUTPUTS:
        (out / name).unlink(missing_ok=True)
    episode_manifest, specs = validate_episode_artifact(args.episodes)
    frozen = load_frozen()
    if args.comparison_label == "primary":
        expected_alias = frozen["primary_checkpoints"][args.semantic]
        if args.checkpoint_alias != expected_alias:
            raise ContractError(
                f"primary alias mismatch for {args.semantic}: "
                f"{args.checkpoint_alias} != {expected_alias}"
            )
    elif args.comparison_label == "capacity_sensitivity":
        allowed = set(frozen["capacity_sensitivity"]["candidate_checkpoints"].values())
        if args.semantic != "move_rel" or args.checkpoint_alias not in allowed:
            raise ContractError(
                f"capacity sensitivity requires a frozen move_rel candidate: "
                f"{args.checkpoint_alias}"
            )
    elif args.comparison_label == "production_movement_bridge":
        bridge = frozen["production_movement_bridge"]
        if not args.preamble or args.checkpoint_alias != bridge["checkpoints"].get(
            args.semantic
        ):
            raise ContractError(
                f"production bridge requires a frozen preamble checkpoint: "
                f"{args.semantic}/{args.checkpoint_alias}"
            )
    elif args.comparison_label == "curriculum_transfer":
        bridge = frozen["curriculum_transfer"]
        if (args.semantic != "deltatype_raw" or not args.preamble
                or args.checkpoint_alias not in bridge["stage2_models"].values()):
            raise ContractError(
                "curriculum transfer requires a frozen stage-2 raw/preamble model: "
                f"{args.semantic}/{args.checkpoint_alias}"
            )
    elif args.comparison_label == "curriculum_transfer_lr5e5":
        rescue = frozen["curriculum_transfer"]["low_lr_rescue_prepared"]
        if (args.semantic != "deltatype_raw" or not args.preamble
                or args.checkpoint_alias not in rescue["models"].values()):
            raise ContractError(
                "low-LR curriculum transfer requires a frozen stage-2 raw/preamble model: "
                f"{args.semantic}/{args.checkpoint_alias}"
            )
    provenance = model_provenance(
        args.model_dir,
        args.semantic,
        args.checkpoint_alias,
        preamble=args.preamble,
        comparison_label=args.comparison_label,
    )
    contract = Contract(args.audit_dir)
    from openai import OpenAI

    client = OpenAI(
        base_url=args.base_url, api_key=args.api_key, timeout=600.0, max_retries=3
    )
    lock = threading.Lock()
    completed = 0

    def work(spec: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        row = run_episode(
            client,
            model=args.model,
            semantic=args.semantic,
            contract=contract,
            episode_root=args.episodes,
            spec=spec,
            max_attempts=args.max_attempts,
            history_turns=args.history_turns,
            max_tokens=args.max_tokens,
            preamble=args.preamble,
        )
        with lock:
            completed += 1
            if completed % 10 == 0:
                print(f"[synthetic-multistep] {completed}/{len(specs)} episodes", flush=True)
        return row

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(executor.map(work, specs))
    report = {
        "schema_version": 1,
        "semantic": args.semantic,
        "checkpoint_alias": args.checkpoint_alias,
        "comparison_label": args.comparison_label,
        "preamble": args.preamble,
        "metrics": summarize(rows, max_attempts=args.max_attempts),
    }
    _atomic_text(out / "rows.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    _atomic_text(out / "report.json", json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_multistep_phasea_eval",
        "status": "complete",
        "semantic": args.semantic,
        "checkpoint_alias": args.checkpoint_alias,
        "comparison_label": args.comparison_label,
        "preamble": args.preamble,
        "model_provenance": provenance,
        "episode_artifact": str(args.episodes.resolve()),
        "episode_manifest_sha256": sha256_file(args.episodes / "build_manifest.json"),
        "n_episodes": len(rows),
        "max_attempts": args.max_attempts,
        "history_turns": args.history_turns,
        "sampling": {**SAMPLING, "max_tokens": args.max_tokens},
        "sampling_seed_scheme": frozen["episode_contract"]["sampling_seed_scheme"],
        "prose_policy": "verbatim raw output stored; complete output replayed without stripping/truncation",
        "elapsed_seconds": time.time() - started,
        "rows_sha256": sha256_file(out / "rows.jsonl"),
        "report_sha256": sha256_file(out / "report.json"),
    }
    if report["metrics"]["request_error_count"]:
        raise ContractError("request errors invalidate evaluation")
    _atomic_text(out / "eval_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    frozen = load_frozen()["episode_contract"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--api-key", default="x")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-alias", required=True)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-dir", type=Path, default=Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/audit_operand"
    ))
    parser.add_argument(
        "--semantic", required=True,
        choices=("absolute_toolcall", "move_rel", "deltatype_raw"),
    )
    parser.add_argument("--preamble", action="store_true")
    parser.add_argument("--comparison-label", default="primary",
                        choices=("primary", "capacity_sensitivity", "preamble_sensitivity",
                                 "production_movement_bridge", "curriculum_transfer",
                                 "curriculum_transfer_lr5e5"))
    parser.add_argument("--max-attempts", type=int, default=frozen["max_attempts_per_target"])
    parser.add_argument("--history-turns", type=int, default=frozen["history_turns"])
    parser.add_argument("--max-tokens", type=int, default=frozen["sampling"]["max_tokens"])
    parser.add_argument("--concurrency", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    manifest = evaluate(parse_args())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
