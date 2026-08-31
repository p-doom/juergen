# HANDOFF: OSWorld pass@4 eval on Horeka (4×A100 40GB)

**Goal:** evaluate the fp32 parity checkpoints (Qwen3.5-9B, LoRA merged into base, oev3 *relative* action format) on OSWorld with **pass@4 (k=4)**. Each checkpoint is served from HuggingFace with sglang; OSWorld tasks run as guest VMs; `osworld_passk_score.py` produces `pass_at_k`.

This file is the reference. The authoritative, working implementation of the whole flow is the origin-cluster recipe:
`recipes/eval/osworld_strat38_pass4_oev3_lr5e5.toml` (in the sibling labctl tree). Its `command` bash block is the loop to port — read it alongside this doc.

---

## 0. Fill these in before starting (human)

- **`HF_REPO_IDS`** — the uploaded checkpoints + their step, e.g. `yll/parity-oev3-lr5e5@step5000`. On origin they are `checkpoints/oev3_exports/qwen35_9b_lora_success_cuagym_oev3_stage_06_seqlen_24576_lr5e5_run_01a05470.../{002500,005000,007500,010000}/`. **Best checkpoint so far = step 5000.**
- **Horeka SLURM** — account, a GPU partition with 4×A100-40GB nodes, and whether that partition exposes hardware virtualization (see §1).
- **OSWorld VM** — is `Ubuntu.qcow2` (21 GB) transferred to Horeka scratch, and is a working `qemu-system-x86_64` available? (see §2)

---

## 1. ⚠️ #1 BLOCKER — prove OSWorld VMs run on Horeka BEFORE anything else

`eval/osworld_oev3_kvm.py` imports `qemu_kvm_provider` and calls `qemu_kvm_provider.install()`, so **every OSWorld task boots a full guest VM from `Ubuntu.qcow2` via QEMU/KVM.** KVM acceleration needs `/dev/kvm` (hardware virt) *inside the SLURM allocation*. On shared HPC this is frequently NOT granted.

**Verify first (one interactive job):**
```bash
srun -A <ACCT> -p <PART> --gres=gpu:1 -t 0:20:00 --pty bash -c \
  'ls -l /dev/kvm; grep -Ec "(vmx|svm)" /proc/cpuinfo; echo "kvm writable: $([ -w /dev/kvm ] && echo yes || echo no)"'
```
- `/dev/kvm` present + writable → KVM works, proceed.
- Absent / not writable → options, in order of preference:
  1. Ask Horeka support / pick a partition that exposes `/dev/kvm`.
  2. QEMU **TCG software emulation** (no `/dev/kvm`): functionally works but ~10–20× slower — a 100-step rollout may not finish in the time limit. Only viable for a tiny smoke, probably not the full 152.
  3. OSWorld's **docker/podman provider** if Horeka supports rootless containers (needs the OSWorld container image; different `--provider_name`).

**Do not run the full eval until ONE OSWorld task boots and completes end-to-end.** A broken VM provider produces all-zero rewards that look like a bad model.

---

## 2. Assets to stage on Horeka

| Asset | Origin path | Notes |
|---|---|---|
| Eval repo | this repo, branch **`yll/oev3-cuagym`** | entrypoints `eval/osworld_oev3_kvm.py`, `eval/osworld_passk_score.py`; system prompt `data_pipeline/realigned_pipeline/system_prompts/cua_v3_cuagym.txt` (bundled — the agent reads it from the repo) |
| OSWorld harness | `osworld-pinned/` repo | set `OSWORLD_ROOT` to its path |
| VM disk (21 GB) | `p-doom_shared/franz/osworld_vm/Ubuntu.qcow2` | transfer to Horeka scratch → `OSWORLD_QCOW2` |
| QEMU | `p-doom_shared/franz/qemu/bin/qemu-system-x86_64-wrapped` (24 KB wrapper) | it wraps a system qemu; on Horeka either transfer + fix its inner path, or `module load qemu`/use system qemu and point `OSWORLD_QEMU_BIN` at it |
| Task subset (38) | `osworld_verified_strat38_no_gdrive.json` (1.8 KB) | copy the file, or use `osworld-pinned/evaluation_examples/test_small.json` |

Full-OSWorld option: use `osworld-pinned/evaluation_examples/test_nogdrive.json` (361 tasks) or `test_all.json` (369) as `--test_split_path` instead — ~10× cost, tighter confidence. Keep any native-baseline comparison on the **same** split (the 47.37% reference is on the 38-set).

---

## 3. Environment

- Install **uv**, then in the repo: `uv sync --all-packages` (creates `.venv`). Python **3.12**. Torch comes from the **cu128** wheel index (needs a CUDA ≥12.8 driver). sglang is pinned in `uv.lock`.
- **Do NOT set `HF_HUB_OFFLINE=1`** — you are pulling checkpoints from the HF hub. Set `HF_TOKEN` if the repos are private.
- Set `CUDA_HOME` to the Horeka CUDA (module or `/usr/local/cuda-*`); put `$CUDA_HOME/bin` on `PATH`.

---

## 4. Per-checkpoint flow (1 GPU serves the 9B model; VMs are CPU/RAM)

**a) Serve the checkpoint** (from the HF repo id directly):
```bash
PORT=31234
.venv/bin/python -m sglang.launch_server \
  --model-path <HF_REPO_ID> --host 127.0.0.1 --port $PORT \
  --api-key osworld --served-model-name oev3-ckpt \
  --mem-fraction-static 0.80 --chunked-prefill-size 2048 \
  > sglang_server.log 2>&1 &
# wait until healthy:
until curl -sf -H "Authorization: Bearer osworld" http://127.0.0.1:$PORT/health_generate >/dev/null; do sleep 3; done
export SGLANG_URL=http://127.0.0.1:$PORT/v1
```
(9B in bf16 ≈ 18 GB → fits one A100-40GB. Inference is entirely through `SGLANG_URL`; the runner does not load the model itself.)

**b) VM env:**
```bash
export OSWORLD_ROOT=/path/to/osworld-pinned
export OSWORLD_QCOW2=/scratch/.../Ubuntu.qcow2
export OSWORLD_QEMU_BIN=/path/to/qemu-system-x86_64
export OSWORLD_VM_BOOT_TIMEOUT_S=600
```

**c) Rollouts — k=4 × 38 tasks = 152 jobs**, run with parallel workers. Port the `run_worker` loop from the recipe; each job is:
```bash
.venv/bin/python eval/osworld_oev3_kvm.py \
  --base_output_dir=$OUT --test_split_path=$SUBSET \
  --task_index=$IDX --sample_index=$S \
  --model_path=<HF_REPO_ID> --served_model_name=oev3-ckpt \
  --path_to_vm=$OSWORLD_QCOW2 \
  --max_steps=100 --temperature=0.6 --top_p=0.95 \
  --max_tokens=32768 --history_n=4 \
  --coordinate_type=relative --screen_width=1920 --screen_height=1080 \
  --sglang_port=$PORT --sglang_api_key=osworld --retry_on_env_error
```
`task_index` 0..37, `sample_index` 0..3. Results land at `$OUT/<app>/<task_id>[/sample_<s>]/result.json`.

**d) Score pass@4:**
```bash
.venv/bin/python eval/osworld_passk_score.py \
  --base_output_dir=$OUT --test_split_path=$SUBSET \
  --k=4 --output_path=$OUT/score.json \
  --checkpoint=<HF_REPO_ID> --arm=lora_success_lr5e5
# -> score.json: pass_at_k, pass_at_1, per_app
```

---

## 5. NON-NEGOTIABLE flags (getting these wrong = silently wrong numbers)

- **`--coordinate_type=relative`** — the model emits relative-mouse deltas on a 1000-grid. `absolute` mis-scales every click with no error.
- **NO sglang `--reasoning-parser`.** The model **thinks inline** — it emits `<think>…</think>` in `content`, and the oev3 agent strips it itself. A reasoning-parser shunts it to `reasoning_content`, leaving `content` empty → episodes die. (This is only the *native* Qwen3VL path's flag; the oev3 path must not use it.)
- **`--max_tokens=32768`** (room for the think block), **`--history_n=4`**, **k=4** (pass@4 — never single-sample; single-seed at n=38 has ~7pp noise).
- `served-model-name` must be **identical** at serve time (`oev3-ckpt`) and in the runner (`--served_model_name=oev3-ckpt`).
- System prompt is `cua_v3_cuagym.txt` (repo default — do not swap it).

---

## 6. Sanity anchor (origin-cluster numbers, same 38-set)

- step 2500 → pass@4 **28.95%** (11/38)
- step 5000 → pass@4 **31.6% & 36.8%** (two seeds; ~34% mean, 12–14/38) ← current best
- step 7500 → pass@4 **26.3%** (10/38, single seed — within noise of the above)

If Horeka yields wildly different numbers, suspect (in order): `coordinate_type`, an accidental reasoning-parser, screen resolution ≠ 1920×1080, or a broken VM provider — **not** the model.

---

## 7. Horeka node utilization

A 4×A100 node: **1 GPU serves the 9B model** (bf16 ≈ 18 GB). Use the other 3 GPUs to eval **up to 4 checkpoints in parallel** — one sglang per GPU (`CUDA_VISIBLE_DEVICES=k`, distinct `--port`, distinct `SGLANG_URL`). OSWorld VMs are CPU/RAM: budget ~2 vCPU + a few GB each; `n_workers=8` ≈ 16 cores + ~64–192 GB.

---

## 8. Agent: your task, in order

1. Run the §1 `/dev/kvm` check. If it fails, STOP and report which fallback the human should authorize — do not run the full eval on a broken provider.
2. Stage §2 assets; set up §3 env; confirm `uv sync` + sglang import succeed.
3. **Smoke: ONE task, ONE sample** end-to-end (serve → 1 rollout → non-error `result.json`). Confirm the VM boots and a click lands.
4. Only then run the full **k=4 × 38** and score. Report `pass_at_k`, `pass_at_1`, per-app, and wall-clock. Compare to §6.
5. If evaluating multiple checkpoints, parallelize across the 4 GPUs (§7).
