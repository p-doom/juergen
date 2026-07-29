# Repo-architecture proposal — OSWorld-parity SFT + RL stack

**Status:** PROPOSAL for franz's approval. Nothing restructured yet. The four additive
`franz/osworld-parity-handoff` capture branches (see HANDOFF.md §1) are an *interim* home; the final
placement is exactly what this doc asks you to pick. **No risky moves until a pattern is chosen.**

---

## 1. Current topology (what actually exists today)

| Repo | pyproject `name` / role | How it's packaged | Consumes | Consumed via |
|---|---|---|---|---|
| **juergen** | `juergen` — *"Crowd-cast SFT supply-chain monorepo: data pipeline, eval harnesses, opd-slime venv builder. The model trainer (omegalax) is a separate generic library, kept as its own repo."* | **uv workspace**, members `data_pipeline`, `eval` (+`opd-slime`). Own env (torch cu128 for sglang eval). | omegalax (as a **subprocess**, not an import) | labctl `[repos] juergen=…`; runs omegalax via `uv run --project <omegalax>` |
| **omegalax** | generic JAX VLM trainer | own repo, own uv env (JAX/XLA) | — (upstream-ish) | labctl `[repos] omegalax=…` name→path; recipe `repo="omegalax"` + `uv run --project`. **Multiple pinned checkouts** already exist as separate `[repos]` entries (`omegalax_qwen35`, `omegalax_gbs16`, `omegalax_qwen35_fix`). |
| **reinforcement-learning** | `verifiers-diy` — *"Cluster-native Verifiers OSWorld environment experiments."* | wheel packages = `rl`, `osworld_desktop_host`; env pulls `verifiers>=0.2.0` | prime-rl (framework: verifiers V1 + GRPO trainer + vLLM serving) | **git submodule** `prime-rl` pinned at commit `41dc85c7`; `[tool.uv.sources]` makes `verifiers` + `prime-rl-configs` **editable paths into the submodule**. `rl/*` does `import verifiers.v1 as vf`. |
| **prime-rl** | upstream (p-doom fork of PrimeIntellect) | its own repo + its own submodules (`deps/verifiers`, …) | — (upstream) | submodule of reinforcement-learning |

**Key facts that constrain the design:**
- The two project repos live in **different runtime worlds** — juergen SFT = **omegalax/JAX**; RL = **prime-rl/torch/vLLM/verifiers**. They do **not** import each other and cannot share one Python env (JAX vs vLLM/torch closures conflict).
- **Two different consumption models for the two upstreams:** prime-rl is **imported** (→ submodule + editable path is the right pin); omegalax is **run as a subprocess** (`uv run --project`) → a submodule is optional; a path + recorded SHA already suffices.
- **labctl `[repos]` is a hard interface:** recipe `repo="<name>"` → cluster.toml `[repos] <name> = "<checkout path>"` → `uv run --project <path>`, and labctl records that checkout's **git SHA at dispatch** (that recorded SHA is today's real reproducibility anchor). Any architecture must keep this name→path→uv-run pattern.
- **Loose today (in no repo):** all convert/build/collect scripts + labctl recipes + the no-leak split (on `/fast/project/.../onpolicy_distill/` and `/fast/home/.../osworld_parity_split/`); the `rl/movebox` + `rl/grounding` envs (never committed) + the `rl_scratch/osworld_rl/*` GRPO/RFT/transfer recipes; plus three vendored grammar deps (`build_videocua_chat.py`, `videocua_key_map.json`, `move_rel_format.py`).
- Two upstream **fixes are still uncommitted** and must land upstream before they can be pinned: the omegalax `api.py` local-HF-dir seed knob, and the prime-rl LoRA-merge export fix (both captured on the handoff branches).

**Bottom line: the topology franz wants ALREADY mostly exists.** juergen = SFT/eval/data; reinforcement-learning = RL infra (envs) not-for-prime-rl-upstream; omegalax + prime-rl = consumed upstreams. The gaps are (i) loose code not yet homed, and (ii) omegalax pinning is convention-only (no committed pin).

---

## 2. Design goals
1. Everything WE build lives in **juergen** (training/eval/data scripts) or **reinforcement-learning** (RL infra not for prime-rl upstream). No loose `/fast/project` scripts.
2. **Reproducible pinning** of fast-moving upstreams (omegalax, prime-rl), staying **labctl-compatible** (`[repos]` name→path, recorded SHA).
3. Preserve the **SFT=omegalax / RL=prime-rl** runtime split (no env merge).
4. **Easy colleague clone-and-run** for the crowd-cast-post-train handoff.

---

## 3. Options + pros/cons

### (a) Import reinforcement-learning as a package into juergen (juergen → RL → prime-rl)
- **Pros:** one clone; single dependency graph.
- **Cons:** **env conflict is fatal** — juergen (JAX + sglang torch cu128) and RL (vLLM/torch/verifiers) cannot coexist in one uv env. Buries "RL infra" inside the SFT monorepo (violates goal #1's intent). Fights labctl, which wants them as separate `[repos]` checkouts run in separate envs. **Reject.**

### (b) Two repos; each project repo pins the upstream IT consumes (refinement of franz's (b))
Not "juergen pins everything" — rather: reinforcement-learning pins prime-rl (already, submodule); **juergen pins omegalax**. Sub-choices for the omegalax pin:
- **(b1) omegalax as a git submodule of juergen** (`juergen/omegalax`, cluster.toml `omegalax=".../juergen/omegalax"`).
  - Pros: git-native exact pin; `clone --recursive` gets it; stable path for labctl.
  - Cons: a submodule pins **one** commit, but the team keeps **several** omegalax checkouts (`_qwen35`, `_gbs16`, …); submodule ergonomics (detached HEAD, easy-to-forget bumps); the canonical omegalax for this pipeline is currently a **non-main branch** (`feat/extra-transforms-hook`) with an **uncommitted** api.py — must be landed first.
- **(b2) uv git-dependency `omegalax @ git+…@<sha>` in juergen's pyproject.**
  - Pros: uv-lockfile reproducibility.
  - Cons: **wrong consumption model** — omegalax is *run as a subprocess in its own JAX env*, not installed into juergen's env; a uv dep would drag JAX/XLA into juergen's env and conflict. **Reject for omegalax** (fine for tiny pure-python packages only).
- **(b3) committed `UPSTREAM_PINS.toml` + `scripts/bootstrap.sh` in each project repo.** Records `{url, rev}` per upstream (multiple named pins allowed, mirroring the cluster.toml variants); bootstrap clones/checks-out each upstream at its rev and writes/updates cluster.toml `[repos]`.
  - Pros: matches the existing `uv run --project` + labctl pattern exactly; supports multiple pinned variants; human-readable; no submodule pain; a colleague runs one script to reproduce the exact `[repos]` layout.
  - Cons: not auto-fetched by `git clone` (needs the bootstrap step); the pin is a file convention, not a git object.

### (c) Monorepo (merge juergen + reinforcement-learning)
- **Pros:** one clone/place.
- **Cons:** either one env (same fatal conflict as (a)) or a uv workspace of two separate-env members = (b) with extra coupling and a bigger blast radius; loses the clean SFT-repo / RL-infra-repo separation. Heavier for a colleague who only needs one side. **Reject as primary** (viable only if the team strongly prefers a single URL; keep as fallback).

### (d) RECOMMENDED — "Two project repos, upstreams pinned per consumption model, orchestrated by labctl"
The pattern that best fits what's already there:
- **reinforcement-learning** keeps **prime-rl as a submodule** (imported → submodule is correct; already pinned `41dc85c7`).
- **juergen** pins **omegalax via a committed `UPSTREAM_PINS.toml` + bootstrap** (b3) — because omegalax is run, not imported, and there are multiple pinned variants. (A `juergen/omegalax` submodule (b1) is an acceptable alternative for the single canonical pin if you prefer git-native; the two can coexist — submodule for the canonical, pins-file for the variants.)
- **All loose code is homed** into the two project repos (see §4).
- **labctl stays the orchestrator**: keep `repo="omegalax"` for SFT/tokenize/export; **add `repo="juergen"` recipes** for convert/collect/eval (juergen is already a `[repos]` entry, and there are currently **zero** `repo="juergen"` recipes — this is the missing half that makes the pipeline fully labctl-native).

---

## 4. Migration plan (where each loose thing lands) — for approval, then execute

**Into `juergen` (SFT / eval / data-processing):**
- Converters `convert_abs_to_*.py`, `build_osworld_format_records.py` → `data_pipeline/osworld_parity/` (data-processing).
- Rollout capture `collect_osworld_rollouts.py`, `collect_absolute_rollouts.py`, `adapt_osworld_rollouts.py` → `eval/osworld_rollout/` (they already import `eval/` modules).
- Vendored grammar deps `build_videocua_chat.py`, `videocua_key_map.json`, `move_rel_format.py` → a shared `data_pipeline/grammar/` (imported by the converters); replace the loose absolute-path `sys.path` hacks with package imports.
- No-leak split + per-format prompts + eval shards (`osworld_train.json`/`osworld_eval_heldout.json`/`make_split.py`/`*_system_prompt.txt`/`format_eval*.sbatch`) → `eval/osworld_parity/`.
- **labctl recipes** → a new tracked `juergen/labctl/recipes/{data_pipeline,training}/` + `policies_templates/`; add the missing `repo="juergen"` recipes for the convert/collect/eval stages.
- (Interim: all of the above currently sits under one `osworld_parity/` dir on the juergen handoff branch — easy to review, then fan out into `data_pipeline/`+`eval/` as above, or keep as one cohesive `osworld_parity/` subpackage if you prefer a single mount point.)

**Into `reinforcement-learning` (RL infra):**
- `rl/movebox/`, `rl/grounding/` (+ wrappers) → already the repo's `rl/` package — just commit them (currently only on the handoff branch / de-indexed live tree).
- GRPO/RFT/transfer run recipes (`grpo_movebox/`, `movebox_gen/`, `transfer_eval/`) → the repo's existing **`experiments/`** (or `configs/` + `sbatch/`), not a new top-level. Parameterize the hard-coded seed path so the crowd-cast swap is one field.
- prime-rl-native SFT/GRPO configs → the existing `configs/prime_rl/`.

**Upstream landing + pinning (prerequisite for reproducibility):**
1. Land the omegalax `api.py` seed-knob change into `p-doom/omegalax` (PR the handoff branch `franz/osworld-parity-handoff`, tip `f26b81e`), then record that SHA in juergen's `UPSTREAM_PINS.toml` (and/or the submodule).
2. Land the prime-rl LoRA-merge fix into `p-doom/prime-rl` (PR handoff branch tip `a0d0f10d`), then **bump the reinforcement-learning `prime-rl` submodule** to that SHA and commit the pointer.
3. Add `juergen/scripts/bootstrap.sh` that clones omegalax at the pinned rev to a chosen path and writes the labctl `cluster.toml [repos]` entries — the colleague's one-command setup.

**Colleague clone-and-run after migration:**
```
git clone --recursive git@github.com:p-doom/reinforcement-learning.git   # gets prime-rl@pin
git clone git@github.com:p-doom/juergen.git && juergen/scripts/bootstrap.sh  # clones omegalax@pin, writes cluster.toml [repos]
# then: labctl run juergen/labctl/recipes/... (SFT/convert/eval) ; RL via reinforcement-learning/experiments/...
# seed swap = one field (HANDOFF.md §3)
```

---

## 5. Open decisions for franz (pick before I execute)
1. **omegalax pin:** committed `UPSTREAM_PINS.toml`+bootstrap (recommended, multi-variant friendly) vs a `juergen/omegalax` git submodule (git-native, single canonical) vs both.
2. **juergen layout:** fan the loose code into existing `data_pipeline/`+`eval/` (idiomatic) vs keep one cohesive `osworld_parity/` subpackage (easier mental model / mount point).
3. **RL run recipes:** land under existing `experiments/` vs a new `runs/`/`configs/osworld_parity/`.
4. **Fallback:** if you'd rather a single URL, monorepo-as-uv-workspace (option c) — say so and I'll re-plan.
5. Confirm it's OK to open the two upstream PRs (omegalax api.py, prime-rl LoRA fix) — they're the pinning prerequisite.
