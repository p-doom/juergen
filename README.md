# Juergen

Juergen owns Crowd-Cast action grammars, data preparation, and closed-loop model
evaluation. Training lives in Omegalax. Desktop/QEMU lifecycle, Labctl run
authorization, Slurm production controllers, and the collector are separate
repositories and separate release boundaries.

There is no CI. A Juergen release is tested locally from a clean commit:

```bash
tooling/local_release_gate.sh
```

That command rebuilds a fresh environment from the hash-locked test manifest,
requires the clean sibling `desktop` checkout and its `origin/main` to equal the
exact revision in `pyproject.toml`, checks both Git stores and the Juergen lock,
then runs the complete `pytest -q tests grammars` gate. A shared mutable venv does
not earn release credit. Do not tag or publish a commit unless its final local
gate is green.

## Evaluation

The one eval in this repo is the CUA micro-eval, under `eval/`. It scores a
checkpoint on a small state-verifiable suite of atomic and short-horizon desktop
tasks: each attempt starts from a fresh VM, every step is semantically gated
against realized guest state, and four sampled attempts per task give pass@1 and
pass@4.

```bash
cd eval && python cua_micro_eval.py --model_path <hf-checkout> --output_dir <dir>
```

Run it from `eval/`: its modules import each other flat, so the directory has to
be on `sys.path`. It launches its own SGLang server out of `eval/pyproject.toml`
-- a separate resolve, deliberately not a member of this workspace, because that
stack is the CUDA/torch one this repo otherwise does not carry.

`eval/README.md` documents the suite, the action format and the task table. Its
unit tests are not part of the `tests grammars` gate and run separately:

```bash
cd eval && python -m pytest -q .
```

Three of the suite's own contract tests are `xfail(strict)`: they assert a shape
`cua_micro_tasks.json` no longer has, and they fail the same way on the branch
this was ported from. The task set is frozen because the suite's value is a
calibrated reference, so neither side was edited to agree.

## Nightly model-estate check

`tooling/estate_gate.sh` covers only Juergen, `data_pipeline`, desktop,
`desktop_fleet`, and Omegalax. It does **not** cover Labctl, Slurm production
recipes/controllers, the collector, or database/backup operations, so its green
verdict is a model-estate reading rather than full production readiness.

The durable wrapper is armed only from a clean, remotely published checkout:

```bash
tooling/estate_gate_cron.sh \
  /canonical/absolute/juergen \
  <full-40-character-published-head> \
  refs/remotes/origin/<exact-release-branch>
```

An authoritative `ESTATE GATE: RED` records an alert and still queues exactly one
successor. Wrapper/provenance errors, a missing or contradictory verdict, dirty or
moved code, and malformed scheduler responses queue nothing. Before replacing a
stale live chain, inspect the exact pending job command and arguments, cancel only
that numeric job ID, then submit the remotely verified command above.
