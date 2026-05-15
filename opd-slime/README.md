# opd-slime

Lab-side wrapper around [slime](https://github.com/THUDM/slime) for
on-policy distillation (OPD) experiments running through `labctl`.

This repo is small on purpose. It contains:

```
scripts/
├── build_slime_venv.sh      # builds the slime CUDA stack as a labctl
│                            # "environment" artifact (atomic; idempotent
│                            # patches; pinned SHAs)
└── run_opd.sh               # OPD training launcher; consumes a built env
                             # and student/teacher/prompts artifacts

patches/
└── apex.patch               # nvcc-vs-torch minor-CUDA-version-mismatch
                             # check bypass; needed because our cluster has
                             # nvcc 12.8 + torch cu129 wheels

opd_plugins/
├── __init__.py
└── null_rm.py               # zero-reward RM for pure OPD (no task reward);
                             # imported by slime via --custom-rm-path
                             # opd_plugins.null_rm.reward_func
```

It does NOT contain a vendored copy of slime; slime is cloned at a pinned
SHA inside `build_slime_venv.sh` along with Megatron-LM, sglang, and apex,
each at their own pinned SHAs and with their respective patches applied.
Slime's own `docker/patch/v0.5.9/{megatron,sglang}.patch` are reused
in-place from the slime clone; only our cluster-specific apex patch is
maintained here.

## Reproducibility

The four pinned SHAs in `scripts/build_slime_venv.sh` plus the three
patches (slime-shipped megatron.patch, sglang.patch + our apex.patch)
define the slime stack deterministically. labctl captures this state via
its source-tree snapshot of opd-slime per recipe run, so a colleague's
reproduction (clone opd-slime + run the labctl pipeline) builds the same
stack byte-for-byte at the wheel level (compiled `.so` files vary across
build hosts, but that's the same boundary every CUDA-heavy lab project
treats as out of scope).

## Bumping slime / Megatron-LM / sglang / apex

Edit the corresponding SHA constant in `scripts/build_slime_venv.sh`,
verify the in-place patches still apply via `git apply --check`, and
commit. The next labctl pipeline run rebuilds the environment artifact
under a new alias (or under the same alias if you delete the old one;
see the marker-bail discussion in the surrounding lab docs).

## Not a uv project

opd-slime does not declare a `pyproject.toml` because the slime CUDA
stack is fundamentally outside what `uv lock` can capture (in-place
patches against external repos, custom build flags, build-from-source
CUDA extensions). The build script is the canonical install path,
analogous to slime upstream's `Dockerfile` / `build_conda.sh`.
