# labctl recipes for the rearchitected sign-of-life gate

Nine recipes: four scripted controls (CPU + KVM) and five model arms (GPU + KVM),
all against `python -m evals.signoflife` at this repo's HEAD.

## Three things here are not obvious and cost a day each if rediscovered

### 1. `provenance.repo_path`, never the staged snapshot

`labctl` stages **one** repo, by copying that repo's `git ls-files` into
`<run_dir>/source/<repo>`. It has no notion of sibling checkouts. This repo's
`desktop-env` dependency is a path source with no remote and no index presence:

```toml
# pyproject.toml
desktop-env = { path = "../desktop-env" }
```

`../desktop-env` does not exist inside a staged snapshot, and nothing records
which commit of it a run used — a path source resolves to whatever is on disk. So
a staged run cannot resolve the dependency and, worse, would not be able to say so.

Every recipe here therefore does what the working sign-of-life recipes did:

```bash
SOURCE="$(jq -er '.provenance.repo_path' "$LABCTL_CONTEXT")"
```

`provenance.repo_path` is the *registered* checkout (`[repos] juergen` in
`cluster.toml`), which has its siblings. The snapshot is then only a record, not
the thing that runs — an honest trade, because the alternative does not run at all.

There is no `// .source_path` alternative in that expression. `repo_path` is
non-optional in labctl's `RepoProvenance`, so the fallback could never fire; had
it fired it would have run the snapshot's code instead of the registered
checkout's, silently, since `desktop-env` reaches the interpreter through
`PYTHONPATH` either way. `jq -er` under `set -e` is the assertion.

Two consequences to hold onto:

* **The checkout must be clean at dispatch.** `provenance.git_head` +
  `git_diff_head` are the only record of what executed; a dirty tree records the
  diff, an *untracked* file records nothing (this is how the arm-T sweep of record
  ended up with its recipe TOMLs unrecorded).
* **`desktop-env`'s commit is not recorded by labctl at all**, so each recipe
  declares it as an explicit `[inputs.desktop_env]` and the command prints
  `git -C "$DESKTOP_ENV" rev-parse HEAD` into the job log. Read it from there.

The durable fix is to give `desktop-env` a remote and replace the path source with
`{ git = ..., rev = ... }`, at which point the staged snapshot resolves and this
whole section becomes unnecessary.

### 2. Two venvs, on purpose

`[inputs.runtime]` is a small prebuilt venv that has `verifiers` and no CUDA.
`[inputs.sglang]` is the existing 14 GB sglang venv, and it is **never imported by
the harness** — it is only the `--sglang-python` the runner spawns
`sglang.launch_server` with, as a subprocess.

That split is deliberate. The alternative — installing `verifiers` into the sglang
venv — mutates a runtime that finished, recorded runs already point at as an
input, which retroactively changes what those runs mean. Building a single 14 GB
combined venv would work too, and is what to do if the subprocess split ever gets
in the way.

`desktop-env` reaches the runtime through `PYTHONPATH`, not an install, for the
same reason as §1: it is a sibling checkout whose identity is its commit.

### 3. Node pinning is a measurement decision

Every recipe pins `--nodelist=hai003`. Controls and model arms must run on the
same host or the controls do not calibrate anything: the previously published
comparison had its controls on hai003 and arm D on hai002, so control conformance
was never established on the node the model arm was measured on. Change the node
if you must, but change it in all nine files at once.

### 4. Paths concatenated onto an input are checked before anything starts

`model-path`, `qcow` and `qemu` are built by appending to `{inputs.X.path}`, and
that template encodes a LAYOUT. `model-path = "{inputs.model.path}/hf"` is right
for a labctl checkpoint export and wrong for a raw HF hub snapshot, whose
`config.json` sits at the snapshot root — the off-the-shelf arm carried the `/hf`
form, matched nothing, and surfaced 56 s later as an `HFValidationError` about
repo-id format, because `transformers` fell through to treating the whole string
as a repo id.

Every recipe therefore walks its own rendered `"$@"` in the prologue and refuses
a `--qcow`/`--qemu` that does not exist or a `--model-path` with no `config.json`,
before the venv, the VM or sglang are touched.

Two `[inputs.model]` shapes exist and they are not interchangeable:
`type = "artifact"` + `artifact = "<alias>"` for anything labctl produced (the
alias is the directory name under `artifact_roots[checkpoint]/<user>/`, and the
reference records lineage), `type = "external"` + `path` only for the HF hub
snapshot, which the registry does not own.

## Trials

The model-arm recipes pass `trials = "3"`. This is not caution, it is the
instrument's own documentation: `desktop_open_chrome` is flagged race-prone —
*"a Chrome that starts but never maps a window flips PASS→FAIL; read pass_rate
over trials"* — and a single draw on that cell cannot be read. The runner reports
`pass_rate` per cell; a scalar `n/4` for an arm is not a result.

The control arms run `trials = "1"`: they are deterministic scripts, and their
calibrated readings (oracle 4/4, negative 0/4) are per-cell values, not rates.
