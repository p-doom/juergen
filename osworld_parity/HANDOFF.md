# HANDOFF — Post-training FROM a crowd-cast checkpoint (OSWorld-parity pipelines)

**Author:** franz (prepared 2026-07-29) · **Audience:** colleague seeding our two OSWorld-parity
post-training pipelines from a checkpoint that has ALREADY been SFT'd on crowd-cast.

## TL;DR / the hypothesis you're testing
Crowd-cast SFT may already have imparted the desktop-control capability; our pipelines just need to
**ELICIT** it. So instead of starting pipelines (A) and (B) from off-the-shelf `Qwen/Qwen3-VL-8B-Instruct`,
you **seed them from a crowd-cast HF-export checkpoint**. The seed swap is a one-liner per stage:

- **Pipeline (A) distillation-SFT (omegalax):** in the labctl training recipe `[args]`, set
  `model_id` (mirror into `processor`) to a crowd-cast `bc_export_hf_*/<step>/` directory, and give the
  run a **fresh** `save_dir`. (Enabled by the uncommitted `omegalax/vlm/api.py` change — captured on the
  omegalax handoff branch — that lets `--model_id` be a local HF dir, not just a hub id.)
- **Pipeline (B) cold-start + RFT/GRPO (prime-rl):** in `grpo_movebox/configs/rl.toml`, change **all six**
  `*.model.name` / `*.tokenizer.name` fields (currently the moverel cold-start export
  `fmt_sft_8b_osw_moverel_hf`) to the crowd-cast HF-export dir — or, better, to a *cold-start SFT of the
  crowd-cast checkpoint*. Must be a **full HF export**, never a LoRA adapter.

Everything below is on four **additive** handoff branches (nothing in the live working trees the
in-flight jobs read was touched). This doc is duplicated at
`p-doom/juergen:franz/osworld-parity-handoff:osworld_parity/HANDOFF.md`.

---

## 1. Multi-repo map (branch `franz/osworld-parity-handoff` in every p-doom repo)

| Repo (GitHub) | Branch base → tip | What it provides |
|---|---|---|
| **p-doom/juergen** | `a82741e` → **`e937a27`** | Eval/rollout harness + the whole distillation pipeline. `eval/freeroll.py` (native qemu+KVM closed-loop rollout), `eval/action_parser.py::parse_deltatype`, `eval/osworld_system_prompts.py`, `eval/osworld_{runtime,vm_client,grounding_runner}.py`. Plus **`osworld_parity/`**: `scripts/` (converters + collectors + builders), `labctl/` (labctl-native recipes), `split/` (no-leak 259/110 + prompts + eval shards), `vendor/` (converter code deps). |
| **p-doom/omegalax** | `b3f32c0` → **`f26b81e`** (off `feat/extra-transforms-hook`) | JAX SFT trainer `scripts/train_vlm_sft.py`, `scripts/build_sft_records_from_chat.py` (tokenize), `scripts/export_to_hf.py` (orbax→merged HF). **The tip commit is the seed-swap enabler**: `omegalax/vlm/api.py` `load_pretrained()` now accepts a local HF-export dir as `model_id`. |
| **p-doom/reinforcement-learning** | `2da5657f` → **`52782d7e`** | On-policy RL envs `rl/movebox/` (id `rl_movebox`, the env job 134308 trains) + `rl/grounding/` (+ wrapper pkgs `rl_movebox/`,`rl_grounding/`); `configs/prime_rl/*.toml`; **`osworld_parity_runs/`** = the vendored GRPO recipe (`grpo_movebox/rl.toml`+split), RFT/STaR (`movebox_gen/`), transfer-eval launchers. |
| **p-doom/prime-rl** | `41dc85c7` → **`a0d0f10d`** | GRPO/SFT trainer. Tip = the **LoRA-merge weight-export fix** (`trainer/lora.py::merge_lora_state_dict`, `weights.py`, `ckpt.py`, `model.py`, `sft/train.py`, `utils/pathing.py`). |

Notes:
- These branches are **static snapshots** of the live working trees as of 2026-07-29 (they reproduce the
  uncommitted state the in-flight jobs run from). The live checkouts stay on their original branches.
- `juergen` local HEAD `a82741e` is *behind* `origin/main`; the branch is intentionally based on the
  local HEAD so it matches what the running jobs use.
- **OSWorld** (`/fast/home/franz.srambical/OSWorld`, `xlang-ai/OSWorld` upstream — no push access) was
  NOT branched. Its only local delta is the **broken apptainer path** (see §6); the working eval
  substrate lives entirely in juergen. Capture commands in §9 if you ever want it.

---

## 2. Where crowd-cast checkpoints live + how to pick a seed

**Root:** `/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/{alfred.nguyen,yll.kryeziu,franz.srambical,mihir.mahajan}/`
(on disk crowd-cast is spelled **`ccast`** in dataset names, `crowdcast` in eval logs.)

Two on-disk formats — the distinction is the crux:

**(A) Orbax-native run dirs — resume-only, NOT a HF seed.** `<name>_run_<id>/` with zero-padded step
folders + run-root `config.json` + `lora_metadata.json`. Each step folder has `train_state/` (ocdbt) +
`input_iter/`. LoRA runs store only the adapter delta — **cannot** be handed to HF `from_pretrained`.
Loadable only by omegalax (`--resume`, or as `--checkpoint_path` to the exporter).

**(B) `bc_export_hf_*/<step>/` — merged HF safetensors — THIS is the seed.** Full `model.safetensors` +
`config.json` + full tokenizer/processor set. Directly loadable as a HF model path and as the omegalax
`--model_id` / prime-rl `model.name` seed.

**Concrete seed examples (format B):**
- 8B: `…/labctl/checkpoints/alfred.nguyen/bc_export_hf_artifact_8b08675ccfbaf762/035000/`
- 4B thinking (yll, self-describing name): `…/labctl/checkpoints/yll.kryeziu/bc_export_hf_4b_thinkingbase_fullft_w12_manual2600/002600/`
- Frontier **canonical 8B, job 130027** (realigned-v3 + goals + cua_v1, LoRA r32, 16k/truncate) is orbax only:
  `…/labctl/checkpoints/alfred.nguyen/qwen3vl8b_lora_ds_v3_goals_canonical_fps_1_cua_v1_seqlen_16k_truncate_run_019f85ca70d87ab1b9714d946b48cb5b/` (latest step `033750`) → **export first** (see §3).
- yll thinking-only 8B LoRA (orbax): `…/yll.kryeziu/qwen3vl8b_thinkingbase_16k_lora_fsdp4_w12_snap390d_run_019f9287…/` (latest `013000`).

**How to select:** pick by lineage+arch first (canonical generalist = alfred `ds_v3_goals…cua_v1` 8B;
thinking-only = yll `thinkingbase`). Confirm arch from the export's `config.json` `text_config`
(authoritative; sizes only approximate): 8B=`hidden_size 4096`/`36 layers`, 4B=`2560`/`36`, 2B=`2048`/`28`.
HF exports are always full merged weights (the exporter re-injects+merges LoRA). Verify the chosen step
dir is non-empty (some are 0-byte crashes: needs `train_state/` for orbax, `model.safetensors` for HF).
Hash-named `bc_export_hf_artifact_<hash>` dirs don't back-link to their training run (that edge is in the
labctl DB, currently **down** — see §7); match by arch+step or use the self-describing names.

---

## 3. The seed swap — exact one-line change per stage

### Pipeline (A) — omegalax SFT
The base weights come from **`--model_id`** (tokenizer/image config from **`--processor`**, defaults to
`model_id`). In the labctl training recipes (`osworld_parity/labctl/recipes/training/fmt_sft_8b_osw_*_v1.toml`)
this is under `[args]`:
```toml
# BEFORE (off-the-shelf)
model_id  = "Qwen/Qwen3-VL-8B-Instruct"
processor = "Qwen/Qwen3-VL-8B-Instruct"
save_dir  = "{outputs.checkpoint.path}"
# AFTER (seed from crowd-cast)
model_id  = "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/alfred.nguyen/bc_export_hf_artifact_8b08675ccfbaf762/035000"
processor = "Qwen/Qwen3-VL-8B-Instruct"   # tokenizer identical; or point at the same export dir (it has tokenizer.json)
save_dir  = "<A FRESH, EMPTY run dir>"     # must be new so it warm-starts, not resumes
```
- Requires the **omegalax handoff-branch `api.py`** (local-dir `model_id`). `resume="if_present"` stays safe
  (empty `save_dir` ⇒ fresh start via `load_pretrained`, still auto-resumes on SLURM requeue).
- Keep `enable_lora="true"` to LoRA-on-top of the crowd-cast base, or `="false"` for full-FT.
- **`resume` is NOT a seeding knob** — it restores the full orbax train_state (weights+optimizer+step+data
  position) from `save_dir` to continue the *same* run; when resuming, `model_id` is ignored for weights.

### Pipeline (B) — prime-rl GRPO
In `osworld_parity_runs/grpo_movebox/rl.toml` (live copy:
`/fast/project/…/franz/rl_scratch/osworld_rl/grpo_movebox/configs/rl.toml`) change **all six** to the same
full-HF-export dir:
```
[model].name, [trainer.model].name, [trainer.tokenizer].name,
[orchestrator.model].name, [orchestrator.tokenizer].name, [inference.model].name
# currently: /fast/project/.../franz/onpolicy_distill/checkpoints/fmt_sft_8b_osw_moverel_hf
```
GRPO freezes this base and trains a fresh LoRA (`rank=8, alpha=32`) on top, so the seed must be a full HF
model, not an adapter. (If you split `trainer.toml`/`orchestrator.toml`/`inference.toml` are re-consumed,
mirror the same edit there.)

### If your crowd-cast checkpoint is orbax-only (no HF export yet)
Export it first (both pipelines seed from a HF dir):
```bash
cd /fast/home/franz.srambical/omegalax
JAX_PLATFORMS=cpu uv run --project . -- python scripts/export_to_hf.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --checkpoint_path <orbax run dir or a specific step dir> \
  --out_dir <dest HF dir> --tp_size 1 --fsdp_size 1 --dp_size 1 \
  --max_grad_norm 1.0 --grad_accum_steps 8    # optimizer-shape flags MUST match the source SFT recipe
```
(or `labctl run osworld_parity/labctl/recipes/data_pipeline/export_hf_fmt_sft_8b_osw_v1.toml` with
`inputs.checkpoint` = the orbax step). The exporter reads `lora_metadata.json` and merges the adapter into
the base automatically.

---

## 4. Pipeline (A) run steps — native rollout → convert → SFT → export → eval

Prefer the **labctl-native** recipes in `osworld_parity/labctl/` (franz wants labctl-native going forward).
`osworld_parity/labctl/launch.sh` is the verified launch guide.

0. **`ssh hai-login2`** — the labctl PG registry is a login-node-local unix socket; from a compute node
   `labctl doctor` shows registry connect = FAIL. Check: `labctl doctor | grep 'registry connect'`.
   (⚠ the registry DB may be down right now — see §7.)
1. **Teacher rollout capture** over the **259 train** tasks via the native-qemu harness (see §6): loose
   drivers `osworld_parity/scripts/{collect_osworld_rollouts.py, collect_absolute_rollouts.py,
   adapt_osworld_rollouts.py}` (+ `split/baseline_eval_shard.py`, sbatch
   `scripts/collect_osworld_train.sbatch`). Off-the-shelf teacher emits **absolute** 0–1000 coords.
2. **Convert** absolute → target format → `_normalized/{train,val}/chat.jsonl`:
   `osworld_parity/scripts/convert_abs_to_deltatype.py` (crowd-cast-native; or `_diffabs`/`_moverel`/
   `_absolute`). deltatype takes `--coord_space raw|normalized`.
3. **Tokenize** (labctl, produces a tracked dataset artifact):
   `labctl run osworld_parity/labctl/recipes/data_pipeline/osw_tokenize_fmt_records_deltatype_raw_v1.toml`
   (wraps omegalax `build_sft_records_from_chat.py`; tokenizer = Qwen3-VL-2B, shared vocab with 8B).
4. **SFT** — with the §3 seed swap:
   `labctl run osworld_parity/labctl/recipes/training/fmt_sft_8b_osw_deltatype_raw_v1.toml`
   (LoRA r32/a32, single-GPU tp=fsdp=dp=1 ⇒ no multi-GPU NCCL, 300 steps, save every 150).
5. **Export** orbax → merged HF:
   `labctl run osworld_parity/labctl/recipes/data_pipeline/export_hf_fmt_sft_8b_osw_v1.toml`
   (`inputs.checkpoint` = the SFT step).
6. **Eval** on the **110 held-out** tasks via the native-qemu closed-loop:
   `osworld_parity/split/format_eval.sbatch` (`ACTION_FORMAT=deltatype`, greedy, sharded) → `aggregate.py`.

For non-deltatype formats: the other 5 training recipes consume the already-built
`onpolicy_distill/converted/osworld_train_<fmt>` datasets directly (`type="external"`). deltatype_raw is
the one whose tokenized dataset must be built first (step 3), which is why it has its own build recipe.

---

## 5. Pipeline (B) run steps — cold-start SFT → RFT/GRPO → transfer eval

1. **Cold-start SFT** (produces the GRPO seed). Two supported engines:
   - **omegalax (production path for movebox/moverel):** run `fmt_sft_8b_osw_moverel_v1.toml` then the HF
     export → `fmt_sft_8b_osw_moverel_hf`. **Seed it from crowd-cast per §3.**
   - **prime-rl-native (grounding track):** `configs/prime_rl/grounding_sft_cold_start.toml`
     (`uv run sft`, `[model.lora] rank=8` to match RL rank). It already seeds from a crowd-cast export
     (`…/labctl/checkpoints/franz.srambical/bc_export_hf_8b_nativerel_artifact_6a26a4e2452eb6c5/001125`) —
     swap that `[model].name` to your chosen crowd-cast dir. Then merge adapter→base before RL.
2. **GRPO** — `osworld_parity_runs/grpo_movebox/rl.sbatch` → `uv run rl @ configs/rl.toml`
   (1 node/4 GPU: 2 train + 2 infer; `[orchestrator.algo] type="grpo"`, `group_size=16`, `max_steps=400`,
   trainer LR `1e-6`). **Seed = §3 pipeline (B).** Env = `rl_movebox` (container-free "move cursor into
   target box" on train-split screenshots; move_rel format). Checkpoints land in `grpo_movebox/weights/step_*`
   every 50 steps — with the prime-rl handoff-branch fix, these contain the **real** merged policy (see §7).
3. **RFT / STaR** — `osworld_parity_runs/movebox_gen/`: `worker*.sbatch`+`gen_worker.py` (pass@k self-gen)
   → `traj_to_chat.py` (successful, loop-free trajs → chat SFT) → `rft_sft.sbatch` (STaR SFT, seeded from
   the moverel export) → `rft_export.sbatch` (merge → `movebox_rft_moverel_iter1_hf`).
4. **Held-out transfer eval** — `osworld_parity_runs/transfer_eval/grpo_final_transfer.sbatch` picks the
   newest `weights/step_*` and calls `transfer_launch.sh`, which submits `split/format_eval.sbatch`
   (110-task held-out OSWorld eval, `ACTION_FORMAT=move_rel`, greedy, 6-shard).

---

## 6. The no-leak split + the KVM eval substrate

**No-leak rule (never violate).** Source = OSWorld-Verified `test_all.json` (369), stratified by app,
seed 20260728, disjoint: **259 train** (rollout generation ONLY) / **110 held-out** (the parity metric
ONLY). Manifests in `osworld_parity/split/{osworld_train.json, osworld_eval_heldout.json}` (+ `.tasks`,
`SPLIT_README.txt`, reproducible via `make_split.py`). 3 held-out tasks need Google-Drive OAuth
(absent) ⇒ structurally unscorable ⇒ effective denominator **107** (`aggregate.py` counts only tasks with
`result.json`, so baseline and treatments drop them identically). **Do not train on the 110; ideally keep
the benchmark eval-only** — Phase-1 (in-dist, proves the pipeline) then Phase-2 (clean, benchmark eval-only).

**KVM eval substrate (the WORKING path).** The VM is booted directly by **`juergen/eval/freeroll.py`** as
**native qemu+KVM** (extracted from the tianon SIF; sidesteps the apptainer-userns KVM block on hai-*
nodes), then sglang serves the policy for a closed-loop rollout:
- qcow2: `/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2`
- qemu wrapper: `/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/qemu/bin/qemu-system-x86_64-wrapped`
- q35 + `-enable-kvm -cpu host`, `snapshot=on` (throwaway overlay ⇒ pristine desktop, shared qcow2 never
  mutated). Overridable via `--qcow2` / `--qemu_bin`. Defaults in `eval/osworld_runtime.py`.
- **Do NOT use the OSWorld DesktopEnv `apptainer` provider** — that is the BROKEN path (the untracked
  `OSWorld/desktop_env/providers/apptainer/` + `desktop_env.py` edits in the OSWorld working tree). It is
  the abandoned attempt; the native-qemu launcher above is what works.

---

## 7. Known gotchas
- **labctl DB is DOWN right now** (`PgStore::connect … pool timed out`) — `labctl status`/`run` may fail
  until it recovers. Team-wide labctl slowness = TOAST-fragmented run JSON; `VACUUM FULL` co-locates
  (12–14s→3s). Filesystem paths in this doc are authoritative regardless.
- **Run labctl from `hai-login2`** (registry = login-node-local unix socket). No dataset registration is
  needed for these — training recipes use `type="external"` and resolve raw absolute paths at run time.
- **Fair-share / node pinning:** recipes `--exclude=hai001`. Historically LoRA recipes over-asked
  `mem=300G` (full-FT leftover) → jobs stuck PENDING/Resources despite idle GPUs; these use **150G**. Check
  free MEM, not just free GPUs; pin to idle nodes via `ReqNodeList`/`--nodelist` if fair-share starves you.
- **/fast/home is full** → write all outputs/checkpoints/caches to **/fast/project**
  (`…/p-doom_shared/franz/…`). (This capture and its clones live under
  `…/p-doom_shared/franz/handoff_capture/`.)
- **enable_lora freezes the vision tower** regardless of `freeze_vision_tower=false` (omegalax `lora.py`);
  the current omegalax checkout also *rejects* `enable_lora=true`+`freeze_vision_tower=true`, hence the
  recipes set `false`. So the LoRA SFTs do NOT adapt the vision encoder.
- **prime-rl LoRA-export bug:** stock prime-rl `clean_lora_state_dict` dropped the adapter and exported
  the frozen base ⇒ every `weights/step_*` was byte-identical to the base (trained delta lost; earlier
  cold-start evals secretly scored the base). Fixed ONLY on the **prime-rl handoff branch**
  (`merge_lora_state_dict`). Ensure `[trainer.ckpt.weights] save_adapter_separately=false` so the merge
  path is active. omegalax export is unaffected (always merges).
- **deltatype format spec** (crowd-cast-native; parser `juergen/eval/action_parser.py::parse_deltatype`):
  one action line per turn — `NO_OP | TERMINATE | FAIL | "dx dy scroll" [ ; EVENTS ]`. `dx dy scroll` =
  three ints (relative mouse move in pixels [raw] or thousandths-of-screen [normalized, ±999] + scroll
  ticks), move applied first, then EVENTS after `;`: `+X` press / `-X` release (mouse `LMB/RMB/MMB`, keys
  by rdev name `Return`,`ControlLeft`,…), and `type("...")` for literal text. Click = `dx dy 0 ; +LMB -LMB`.
- **Converter code deps are vendored** into `juergen:osworld_parity/vendor/` (`build_videocua_chat.py`,
  `videocua_key_map.json`, `move_rel_format.py`). Originals were LOOSE (untracked) at
  `/home/franz.srambical/slurm/dev/franz/berlin/crowd-cast-bc/videocua_golden_v1/` and
  `…/datasets/franz.srambical/videocua_moverel/`. If you run the loose copies in place, they import from
  those absolute paths; the vendored copies make the pipeline self-contained.

---

## 8. Key findings context (so you don't re-derive)
- **SFT teaches ABSOLUTE to off-the-shelf parity, but RELATIVE collapses closed-loop.** This is a
  **structural policy collapse** (BC emits a ~constant ~10px nudge and never navigates), NOT a decoding
  artifact — presence-penalty and temperature sweeps do not fix it. **On-policy RL (GRPO) is the lever.**
- The single-step "grounding wall" (~1%) was an **artifact**: an absolute model scored through a relative
  harness + a fan-out collision bug. Off-the-shelf single-step grounding is ~90.5%; cold-starts score well
  single-step in matched convention. The *closed-loop* relative-move collapse is the real problem.
- **Stay relative.** crowd-cast is relative-only; absolute is already solved. Do NOT pivot to absolute.
- **`deltatype` = the crowd-cast-native format** (bare-token diffabs grammar + two fixes: coalesced
  `type("...")` and first-class documented `TERMINATE`/`FAIL`). It is round-trippable with crowd-cast and
  ScaleAugment-compatible. move_rel is the JSON-tool-call sibling the movebox GRPO env currently uses.
- **This handoff's bet:** if crowd-cast SFT already imparted navigation, seeding (A)/(B) from a crowd-cast
  checkpoint should elicit closed-loop competence that off-the-shelf seeds could not reach. Calibrate any
  metric against a known-good ref (off-the-shelf Qwen3-VL-8B ≈ 33.9% OSWorld-Verified) through the exact
  harness before trusting it.

---

## 9. What was committed where / what still needs a human hand

**Committed + pushed (all additive, live trees untouched, all four jobs kept running):**

| Repo | Branch | Tip commit |
|---|---|---|
| p-doom/juergen | franz/osworld-parity-handoff | `e937a27` (on `a82741e`) |
| p-doom/omegalax | franz/osworld-parity-handoff | `f26b81e` (on `b3f32c0`) |
| p-doom/reinforcement-learning | franz/osworld-parity-handoff | `52782d7e` (on `2da5657f`) |
| p-doom/prime-rl | franz/osworld-parity-handoff | `a0d0f10d` (on `41dc85c7`) |

Capture workspace (clones; safe to delete): `/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/handoff_capture/`.

**Intentionally NOT captured (documented instead — do at a safe moment / by a human):**
- **OSWorld broken-apptainer delta** — origin is `xlang-ai/OSWorld` (no push). If wanted, capture to a fork:
  ```bash
  cd /fast/home/franz.srambical/OSWorld && git remote add pdoom <your fork>   # do NOT push to xlang-ai
  git stash create   # snapshots tracked mods without touching the tree; then branch off it + add providers/apptainer/
  ```
  (Low value — it is the broken path; the working substrate is juergen native-qemu.)
- **prime-rl submodule pointer bump** `deps/research-environments` — not captured (needs the submodule
  checked out). To fold it in later: in a fresh clone with submodules, `git submodule update --init
  deps/research-environments`, check out the target commit the live tree points to
  (`git -C /fast/project/.../prime-rl/deps/research-environments rev-parse HEAD`), then commit the pointer.
- **The de-indexed reinforcement-learning live tree** — the live repo is in a `git rm -r --cached .` state
  (whole tree shows staged-deleted + untracked). It is harmless (files intact) but a `git checkout`/`reset`
  to "clean it up" WOULD destroy the loose work the jobs read — **do not**. To re-index in place at a safe
  moment: `cd …/reinforcement-learning && git add -A && git status` (verify) — but coordinate first; the
  handoff branch already contains the load-bearing loose code, so this is optional housekeeping.
- **Large binaries / data** (the qcow2 image, the qemu wrapper, rollout/checkpoint dirs, HF exports) are
  not git-tracked by design; paths are documented above.
