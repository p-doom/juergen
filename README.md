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

## Evaluation results

`python -m evals.signoflife` requires an absent absolute output path. It creates
that run ID before external/server/desktop resources; a crash leaves a visible
uncommitted orphan that must be audited before manual removal. Completion exists
only when `RESULT_COMMITTED.json` is present and the exhaustive generation can be
read through:

```python
from pathlib import Path
from evals.signoflife.__main__ import read_committed_result

result = read_committed_result(Path("/absolute/run/path"))
```

That marker proves atomic transport completion, not authorization or promotion.
Promotion additionally requires the Labctl DB-bound exhaustive result receipt
named by `result["promotion_evidence"]["required_receipt"]`.

External OpenAI-compatible servers are unsupported by the canonical evaluator.
It rejects the presence of `SIGN_OF_LIFE_API_KEY` before argument parsing or
resource acquisition and launches only a local no-auth loopback SGLang server.
Causal evaluation remains blocked until the registered model generation, sealed
runtime, inherited listener, host GPU/driver boundary, and registrar contracts
are all production-authoritative.

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
