"""The CUA micro-eval suite — per-interaction-type canary tasks in a real VM.

A STAGED PORT. This is the suite as it ran on `feat/cua-micro-evals`, moved
under `evals/` and wired to this repo's grammars and prompt registry, but still
carrying its own VM lifecycle, its own sglang launch and its own turn loop. It
runs today; it is not yet an `evals/` task family.

What that means concretely: several thousand lines here duplicate machinery this
repo already has, and each duplicated block is marked with a `PORT:` comment
naming its counterpart. The map:

    _launch_vm / _port_free / _preflight_ports   -> evals/vm.py:kvm_desktop_pool
    run_attempt / run_multiturn_attempt          -> evals/harness.py DesktopHarness
    _run_one_task_attempt / _run_vm_slot / main  -> verifiers.v1.cli + Harness
    in_bbox / distance_to_bbox / cursor starts   -> evals/tasks.py (verbatim)
    denormalize_* / qwen3vl_native_to_ordered    -> a codec's compile()
    _launch_chrome / _launch_native_app / …      -> evals/fixtures/{chrome,apps,web,tk}
    aggregate_results                            -> verifiers scoring
    osworld_vm_client                            -> the `desktop` package transport
    cua_micro_harvest                            -> datasets/convert.py

What is NOT duplicated, and is why the suite still exists in this form:

  * `action_matches_expected` scores a single-turn task by matching the model's
    emitted ACTION against an expected intent. Every oracle in `evals/` decides
    from realized guest state instead (`StateOracle`: "never from the trace"),
    so this has no counterpart and porting it needs a design decision, not a
    move.
  * `osworld_vm_client.patch_xcursor_leak` fixes a guest-image bug — the OSWorld
    agent leaks one X connection per `/screenshot`, and Xorg refuses clients past
    256, so a multiturn task dies around turn ~55. `evals/vm.py` and the pinned
    `desktop` package carry no equivalent; his suites never hit it because
    cuagym caps at 25 steps and sign-of-life at 12.
  * `sampling.py`'s Qwen regime detection and `cua_micro_wandb.py`'s
    artifact-lineage walk (group = the producing TRAINING recipe).

Imports are package-absolute (`evals.micro_evals.x`), not flat: `evals/` is a
package, so the flat form only resolved from a cwd that happened to be the old
`eval/` directory.
"""
