# venv manifests

The estate runs on ten hand-built virtual environments. Nine are under
`/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/venvs/`; the sign-of-life
serving runtime is a sibling named directly by the eval recipes. This directory
is the record — one
`<name>.requirements.txt` and one `rebuild-<name>.sh` per venv, captured
2026-08-23 with `uv pip freeze` (uv 0.7.19).

**The rule from now on: a venv change goes through the manifest.** Install into a
venv, then re-freeze it into its `.requirements.txt` in the same change. A venv
whose manifest no longer reproduces it is a venv nobody can rebuild.

## What serves what

| venv | py | dists | role |
| --- | --- | --- | --- |
| `juergen-testgate-venv` | 3.12.11 | 113 | Estate gate, suites `juergen` (`tests/` + `grammars/`) and `desktop`. `JUERGEN_PYTHON` and `DESKTOP_PYTHON` in `tooling/estate_gate.sh`. |
| `data-pipeline-testgate-venv` | 3.12.11 | 18 | Estate gate, suite `data_pipeline` (`juergen/data_pipeline/tests`). `DATA_PIPELINE_PYTHON`. Separate from the juergen testgate venv for one reason: `cv2`, which that one does not have. |
| `omegalax-rearch-gate-venv` | 3.13.5 | 166 | Estate gate, suite `omegalax_rearch`. `OMEGALAX_PYTHON`. Built by `uv sync --locked --extra torch-tests --python 3.13` from omegalax-rearch's lockfile; deliberately not that repo's own `.venv`, which the training jobs share and which a bare `uv sync --locked` would strip of torch mid-gate. |
| `omegalax-rearch-venv` | 3.13.5 | 166 | omegalax-rearch working environment. Package set **identical** to the gate venv, all 166 lines — the isolation costs nothing, as the gate's header claims. |
| `omegalax-venv` | 3.13.5 | 141 | Pre-rearchitecture omegalax run environment. No pytest; not a gate. |
| `omegalax-rearch-testgate-venv` | 3.12.11 | 87 | **Retired.** Referenced by nothing. This is the venv `estate_gate.sh` blames by name: drifted months ahead of the lock (transformers 5.14.1 vs 5.2.0, jax 0.11.0 vs 0.9.2) and failed the Qwen3-VL MoE loader on a renamed config key. Kept so that failure stays reproducible. |
| `osworld-parity-venv` | 3.12.11 | 323 | OSWorld parity evaluation — the one environment where juergen's dependencies and OSWorld's coexist. It exists because they could not be merged into an existing venv: a numpy 1.x/2.x ABI conflict forced a dedicated build. |
| `eval-venv` | 3.12.9 | 309 | Serving/eval: sglang 0.5.6.post1 from PyPI, torch 2.9.1, transformers 5.0.0rc0. Not on any gate path. |
| `sglang-eval-venv` | 3.12.9 | 248 | Serving/eval against a forked sglang (`JustinTong0323/sglang@39b37ca6`, `subdirectory=python`). Oldest venv in the estate (2026-03-31), and the only one that still carries its own pip. |
| `sign_of_life_eval_v2_venv` | 3.12.11 | 344 | Active SGLang serving runtime named by the sign-of-life eval recipes. Torch 2.9.1+cu128, sglang 0.5.10.post1, transformers 5.3.0. |

Interpreters: `3.12.11` and `3.13.5` are uv-managed
(`cpython-*-linux-x86_64-gnu`) and the rebuild scripts name them by version, so
uv fetches exactly those builds. `3.12.9` is **conda-forge**, at
`/opt/miniforge3/bin/python3.12`; those two scripts pin it by absolute path,
because `uv venv --python 3.12.9` would fetch a uv-managed 3.12.9 instead — same
version, different build.

## Pinning

Pins are exact versions, plus shas for VCS requirements. Only
`juergen-testgate-venv.requirements.txt` carries `--hash` pins and installs under
`--require-hashes`: it is the only venv in the estate with no local-path and no
VCS requirement, and `--require-hashes` is all-or-nothing — one
`pkg @ file:///…` line disqualifies a whole file.

Re-freezing after a change. For nine of the ten:

    NO_COLOR=1 uv pip freeze --no-config --python <venv>/bin/python

and paste the result under the existing header. `juergen-testgate-venv` needs the
hashes regenerated from that freeze:

    uv pip compile --no-config --generate-hashes --python-version 3.12 \
      --no-annotate --no-header --emit-index-url <freeze> -o <out>

Check that the compile neither adds nor drops a distribution before committing it.

Every `uv` call in these scripts passes `--no-config`. The scripts live inside
the juergen checkout, whose `pyproject.toml` carries
`[tool.uv] override-dependencies = ["openai>=2.26.0"]`; uv discovers it by
walking up from the script and applies it to an unrelated environment, and it
defeats `--require-hashes` outright since an override is not `==` pinned.

## Provenance holes

Fourteen requirements across eight venvs are installed from a path rather than an
identity. The manifest records what the path held at capture time, but the path
is what gets installed, so a rebuild takes whatever is there when it runs. These
are the holes, and closing them means publishing those trees and pinning a sha:

| venv | requirement | source at capture |
| --- | --- | --- |
| `data-pipeline-testgate-venv` | `desktop @ file://…/desktop` | `b1be21fa`, `main`, clean |
| `eval-venv` | `-e file://…/eval` | **directory no longer exists** |
| `omegalax-rearch-gate-venv` | `-e file://…/omegalax-rearch` | `156394891cb2`, `omegalax-rearchitected`, 1 uncommitted |
| `omegalax-rearch-venv` | `-e file://…/omegalax-rearch` | same |
| `omegalax-venv` | `-e file://…/omegalax` | `b3f32c002998`, `feat/extra-transforms-hook`, 5 uncommitted |
| `omegalax-rearch-testgate-venv` | `renderers @ file://…/prime-rl/deps/renderers` | `bdb96b0c84a3`, detached, clean |
| `omegalax-rearch-testgate-venv` | `tokamax @ file://…/tokamax` | `d81bc23bc80f`, `fix/shard-map-nested-mask`, 1 uncommitted |
| `osworld-parity-venv` | `juergen @ file://…/juergen-nn` | `03a7b04680`, `native-normalized`, 1 uncommitted — a *second* juergen checkout, not this one |
| `sglang-eval-venv` | `-e file://…/crowd-cast-eval` | **directory no longer exists** |
| `sign_of_life_eval_v2_venv` | editable `juergen` | `119b2e252b9a`, `franz/sign-of-life-eval-v2-20260803`, clean |
| `sign_of_life_eval_v2_venv` | editable `crowdcast-data-pipeline` | same commit, `data_pipeline` subdirectory |
| `sign_of_life_eval_v2_venv` | editable `crowdcast-eval` | same commit, `eval` subdirectory |

Two of them are unrecoverable. `eval-venv`'s `-e file:///fast/home/franz.srambical/eval`
and `sglang-eval-venv`'s `-e file:///fast/home/franz.srambical/crowd-cast-eval`
point at directories that are gone; nothing on disk records what they contained.
Those lines are commented out in their manifests so the rebuild completes, and
the rebuild is **not** equivalent to the captured venv. Both venvs are
serving-only and off every gate path, which is the only reason this is survivable.

The sign-of-life source commit is not on `origin`. Its rebuild script accepts a
repository containing that object and archives the exact commit rather than
installing whatever a checkout currently holds.

Three requirements do pin, because the sha is in the URL:
`tokamax @ git+https://github.com/p-doom/tokamax.git@d81bc23b` (the three omegalax
venvs), `instruction-following-eval @ git+…@0c495b2f` (both eval venvs),
`sglang @ git+…@39b37ca6` (`sglang-eval-venv`), and
`desktop @ git+file://…/desktop.git@5ed77a58` (`osworld-parity-venv`) — note that
last one is three days behind the `desktop` working tree the data-pipeline gate
venv installs from.

## Non-pip state

There is none worth reproducing. No venv has a `sitecustomize.py`, and no
`bin/activate` sets `LD_PRELOAD`, `LD_LIBRARY_PATH` or any other variable beyond
the standard `VIRTUAL_ENV`/`PATH` pair. Every `.pth` file is either installed by
a distribution already in the manifest (`_virtualenv.pth`,
`distutils-precedence.pth`, `a1_coverage.pth`, `nvidia_cutlass_dsl.pth`,
`_cuda_bindings_redirector.pth`, the matplotlib and google-generativeai nspkg
shims) or is the editable-install marker for a hole already listed above
(`_editable_impl_omegalax.pth`, `_crowd_cast_eval.pth`,
`__editable__.crowdcast_eval-0.1.0.pth`).

The NCCL `LD_PRELOAD` the training jobs need is a property of those jobs, not of
any venv here, and nothing in these environments expects it.

## Known defect

`osworld-parity-venv` has `opencv-python==5.0.0.93` and
`opencv-python-headless==4.8.1.78` installed side by side — two versions of the
same extension module in one import path. The manifest reproduces it faithfully,
which is the point; it is recorded here so it is not mistaken for a capture
error.

## Verified

`juergen-testgate-venv` was rebuilt from its manifest into a scratch directory on
2026-08-23 and checked two ways. `uv pip freeze` on the rebuild is identical to
the capture, 113 distributions, no drift. Under the rebuilt interpreter,
`pytest tests/test_suite_and_gate.py grammars/` with the gate's own
`PYTEST_ADDOPTS="-q -p no:cacheprovider"`: **756 passed**, matching the same
slice under the original venv.

`sign_of_life_eval_v2_venv` was captured from its live recipe interpreter:
Python 3.12.11, 344 distributions, raw freeze sha256
`997ae8ba16caa70eac29f2f4316691323bf2f041bcc32c6f3fe0f62a3c8b1b23`.
