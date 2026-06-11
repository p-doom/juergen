"""Roundtrip BC offline-imitation eval: omegalax export → HF dir → SGLang →
teacher-forced action generation over a held-out val set → imitation metrics.

Mirrors ``roundtrip_ifeval.py`` exactly for stages 1-2 (orbax -> HF export, then
``hf_complete``) and the SGLang serve in stage 3; only the evaluation body
differs: instead of an inspect-ai task, we teacher-force the model over real
crowdcast trajectories (real screenshots + real prior actions as history) and,
at each assistant turn, ask the served model for the next action. The (gold,
pred) pairs are scored by ``bc_offline_score`` and written as result.json.

This is an imitation-fidelity monitor, NOT a correctness oracle — see
``bc_offline_score`` for why (noisy single-human demos, relative/multi-step/
cursorless action space). Track its metrics relative across checkpoints.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path

from absl import app, flags
from openai import OpenAI

import bc_offline_score
from hf_complete import complete_export_dir, find_hf_snapshot
from result import write_result
from sglang_runner import sglang_server

FLAGS = flags.FLAGS

# Export params (identical contract to roundtrip_ifeval.py — pmanager/labctl
# injects checkpoint_path from the parent checkpoint at fire time).
flags.DEFINE_string("output_dir", None, "Eval task dir.", required=True)
flags.DEFINE_string("checkpoint_path", "", "Orbax checkpoint dir; '' = pretrained export.")
flags.DEFINE_string("model_id", "Qwen/Qwen3-VL-2B-Instruct", "HF model id.")
flags.DEFINE_string("omegalax_repo", None, "omegalax repo root (uv --project).", required=True)
flags.DEFINE_string("hf_home", None, "HF cache root (hub/<model_id>/snapshots/...).", required=True)
flags.DEFINE_integer("tp_size", 1, "Export tensor parallelism.")
flags.DEFINE_integer("fsdp_size", 1, "Export FSDP size.")
flags.DEFINE_integer("dp_size", 1, "Export DP size.")
flags.DEFINE_float("max_grad_norm", 1.0, "Optimizer-state shape: training run's.")
flags.DEFINE_integer("grad_accum_steps", 1, "Optimizer-state shape: training run's.")

# BC eval params.
flags.DEFINE_string("val_jsonl", None, "Held-out BC samples jsonl to score against.", required=True)
flags.DEFINE_integer("max_trajectories", 0, "Cap trajectories (0 = all).")
flags.DEFINE_integer("max_pairs", 1000, "Cap total scored assistant turns (0 = all).")
flags.DEFINE_integer("max_history_turns", 0, "Cap history turns kept before each step (0 = full).")
flags.DEFINE_float("temperature", 0.0, "Generation temperature.")
flags.DEFINE_integer("max_tokens", 64, "Max new tokens per action (actions are short).")
flags.DEFINE_integer("seed", 0, "Generation seed.")

# SGLang params (same names as roundtrip_ifeval.py).
flags.DEFINE_integer("sglang_port", 0, "0 = auto-derive from SLURM_JOB_ID.")
flags.DEFINE_string("sglang_api_key", "bceval", "SGLang API key.")
flags.DEFINE_float("mem_fraction_static", 0.80, "SGLang static mem fraction.")
flags.DEFINE_integer("chunked_prefill_size", 2048, "SGLang chunked prefill size.")


def _data_url(image_path: str) -> str:
    """Encode a frame as a base64 data URL for the OpenAI-compatible API."""
    raw = Path(image_path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def _to_openai_content(content) -> object:
    """Convert a BC message's content into OpenAI chat format (image→image_url)."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content if isinstance(content, list) else []:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image":
            parts.append({"type": "image_url", "image_url": {"url": _data_url(p["image"])}})
    return parts


def _assistant_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def _gold_assistant_indices(msgs: list) -> list[int]:
    """Message-indices of assistant turns with non-empty gold text."""
    return [
        i for i, m in enumerate(msgs)
        if m.get("role") == "assistant" and _assistant_text(m.get("content", ""))
    ]


# Fraction of the scored budget reserved for terminal (last/TERMINATE) turns so
# the terminate metric has real support. Each trajectory ends with one TERMINATE
# (~1% of all turns), so a naive prefix/position-sweep would score ~0 of them.
_TERMINAL_RESERVE = 0.15


def iter_eval_steps(
    val_jsonl: Path, max_trajectories: int, max_history_turns: int, max_pairs: int
):
    """Yield (history_messages_openai, gold_action_str) for a representative,
    terminal-covering spread of assistant turns (teacher-forced history).

    A naive prefix of ``max_pairs`` turns only covers the first couple of long
    trajectories and almost no TERMINATE (one per trajectory, ~1% of turns).
    Instead we sample ACROSS trajectories and turn-positions:
      * budget <= #trajectories: pick that many trajectories strided across the
        set; reserve ~15% for terminal turns (so TERMINATE is measurable) and
        sweep the rest over non-terminal positions 0..m-2.
      * budget  > #trajectories: take evenly-spaced turns per trajectory, always
        including the terminal turn.
    ``max_history_turns`` keeps only the most recent N non-system turns.
    NOTE: terminals are deliberately mildly oversampled vs their ~1% natural
    rate; read per-class terminate P/R/F1 (prevalence-independent) for that
    action. Sampling is deterministic, so checkpoint-to-checkpoint trends hold.
    """
    # --- Pass 1: cheap index of gold assistant-turn counts (no image encoding) ---
    traj_turns: list[int] = []
    with val_jsonl.open() as fh:
        for line in fh:
            row = line.strip()
            if not row:
                continue
            gi = _gold_assistant_indices(json.loads(row).get("messages", []))
            if not gi:
                continue
            traj_turns.append(len(gi))
            if max_trajectories and len(traj_turns) >= max_trajectories:
                break
    n_traj = len(traj_turns)
    if n_traj == 0:
        return
    budget = max_pairs if (max_pairs and max_pairs > 0) else sum(traj_turns)

    # --- Decide which (trajectory, turn-ordinal) steps to score ---
    selection: dict[int, set[int]] = {}

    def add(t: int, ordn: int) -> None:
        selection.setdefault(t, set()).add(max(0, min(ordn, traj_turns[t] - 1)))

    if n_traj >= budget:
        n_term = min(n_traj, max(1, round(budget * _TERMINAL_RESERVE)))
        n_rep = max(0, budget - n_term)
        for j in range(n_term):  # terminal turns, strided across trajectories
            t = (j * n_traj) // n_term
            add(t, traj_turns[t] - 1)
        for k in range(n_rep):  # representative position sweep, strided
            t = (k * n_traj) // n_rep
            m = traj_turns[t]
            frac = k / (n_rep - 1) if n_rep > 1 else 0.0
            add(t, round(frac * max(0, m - 2)))
    else:
        per, extra = divmod(budget, n_traj)
        for t, m in enumerate(traj_turns):
            k = min(per + (1 if t < extra else 0), m)
            if k <= 0:
                continue
            if k == 1:
                add(t, m - 1)  # terminal turn
            else:
                for j in range(k):  # evenly spaced, incl. first and terminal
                    add(t, round(j * (m - 1) / (k - 1)))

    # --- Pass 2: encode + yield only the selected trajectories' selected turns ---
    t = -1
    with val_jsonl.open() as fh:
        for line in fh:
            row = line.strip()
            if not row:
                continue
            msgs = json.loads(row).get("messages", [])
            gi = _gold_assistant_indices(msgs)
            if not gi:
                continue
            t += 1
            if max_trajectories and t >= max_trajectories:
                break
            ords = selection.get(t)
            if not ords:
                continue
            oa = [{"role": m["role"], "content": _to_openai_content(m["content"])} for m in msgs]
            for ordn in sorted(ords):
                i = gi[ordn]
                gold = _assistant_text(msgs[i]["content"])
                history = oa[:i]
                if max_history_turns > 0:
                    sys_part = [h for h in history[:1] if h["role"] == "system"]
                    history = sys_part + history[len(sys_part):][-max_history_turns:]
                yield history, gold


def main(_):
    output_dir = Path(FLAGS.output_dir)
    export_dir = output_dir / "exported_hf"
    sglang_log = output_dir / "sglang_server.log"
    orbax_path = Path(FLAGS.checkpoint_path) if FLAGS.checkpoint_path else None
    sglang_port = (
        30000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 10000)
        if FLAGS.sglang_port == 0
        else FLAGS.sglang_port
    )
    t_start = time.time()

    # --- Stage 1: omegalax export → HF safetensors (CPU; srun for jax.dist) ---
    export_dir.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        "srun", "uv", "run", "--project", FLAGS.omegalax_repo,
        "python", "scripts/export_to_hf.py",
        f"--model_id={FLAGS.model_id}", f"--out_dir={export_dir}",
        f"--tp_size={FLAGS.tp_size}", f"--fsdp_size={FLAGS.fsdp_size}", f"--dp_size={FLAGS.dp_size}",
    ]
    if orbax_path is not None:
        export_cmd += [
            f"--checkpoint_path={orbax_path}",
            f"--max_grad_norm={FLAGS.max_grad_norm}",
            f"--grad_accum_steps={FLAGS.grad_accum_steps}",
        ]
    print(f"[bc_roundtrip] export: {' '.join(export_cmd)}", flush=True)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    rc = subprocess.run(export_cmd, cwd=FLAGS.omegalax_repo, check=False, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"omegalax export failed (rc={rc})")

    # --- Stage 2: complete the HF dir (tokenizer sidecars + config patches) ---
    snapshot_dir = find_hf_snapshot(FLAGS.model_id, Path(FLAGS.hf_home))
    completion = complete_export_dir(export_dir, snapshot_dir)
    print(f"[hf_complete] copied={completion['copied']} patched={completion['patched']}")

    # --- Stage 3: SGLang serve + teacher-forced action generation ------------
    pairs: list[tuple[str, str]] = []
    with sglang_server(
        model_path=str(export_dir),
        port=sglang_port,
        api_key=FLAGS.sglang_api_key,
        log_path=sglang_log,
        mem_fraction_static=FLAGS.mem_fraction_static,
        chunked_prefill_size=FLAGS.chunked_prefill_size,
    ) as server_url:
        client = OpenAI(base_url=server_url, api_key=FLAGS.sglang_api_key)
        for history, gold in iter_eval_steps(
            Path(FLAGS.val_jsonl), FLAGS.max_trajectories, FLAGS.max_history_turns,
            FLAGS.max_pairs,
        ):
            resp = client.chat.completions.create(
                model=str(export_dir),
                messages=history,
                temperature=FLAGS.temperature,
                max_tokens=FLAGS.max_tokens,
                seed=FLAGS.seed,
            )
            pairs.append((gold, resp.choices[0].message.content or ""))
            if FLAGS.max_pairs and len(pairs) >= FLAGS.max_pairs:
                break

    # --- Stage 4: score + write the pmanager/labctl result.json --------------
    (output_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps({"gold": g, "pred": p}) for g, p in pairs)
    )
    res = bc_offline_score.score_pairs(pairs)
    write_result(
        output_dir / "result.json",
        task="bc_offline_imitation",
        scores={f"bc_offline/{k}": v for k, v in res["scores"].items()},
        params={
            "checkpoint_path": FLAGS.checkpoint_path,
            "model_id": FLAGS.model_id,
            "temperature": FLAGS.temperature,
            "max_history_turns": FLAGS.max_history_turns,
        },
        inputs={"val_jsonl": FLAGS.val_jsonl},
        n_samples=len(pairs),
        elapsed_s=int(time.time() - t_start),
        extra={"confusion": res["confusion"], "n_move_steps": res["n_move_steps"]},
    )
    print(json.dumps({f"bc_offline/{k}": v for k, v in res["scores"].items()}, indent=2))


if __name__ == "__main__":
    app.run(main)
