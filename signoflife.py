"""The sign-of-life gate under a **flat** plugin id.

This module exists for one reason: `verifiers` 0.2.1 cannot resolve a dotted
plugin id, and `evals.signoflife` is dotted.

`loaders._import_plugin` (`verifiers/v1/loaders.py:31-43`) does

    namespaced = f"verifiers.v1.tasksets.{module}"
    target = namespaced if importlib.util.find_spec(namespaced) else module

and `find_spec` imports the *parent* of the name it is given. For
`evals.signoflife` that parent is `verifiers.v1.tasksets.evals`, which does not
exist, so `find_spec` raises `ModuleNotFoundError` **out of the guard that was
meant to catch it** — the `else module` fallback is unreachable for any id
containing a dot. The failure is upstream's, not ours, and it is a crash rather
than a miss:

    >>> taskset_class("evals.signoflife")
    ModuleNotFoundError: No module named 'verifiers.v1.tasksets.evals'

A flat id takes the intended path: `find_spec("verifiers.v1.tasksets.signoflife")`
returns `None`, and `importlib.import_module("signoflife")` finds this file at the
repo root. So the gate's plugin id is `signoflife`, and `evals/signoflife/` stays
where it belongs in the package layout.

Chosen over the two alternatives on purpose. Vendoring a patched `loaders.py` puts
us on a fork of the framework we pin exactly (`verifiers==0.2.1`) for its internal
API; moving the package to the repo root would drag `evals/harness.py`,
`evals/tasks.py` and the fixtures with it, because the taskset is a thin shell over
those. A twelve-line alias is the smaller thing to own.

**The `__all__` below is a verifiers contract, not documentation.**
`loaders._plugin_class` scans it and requires *exactly one* `Taskset` subclass and,
for the harness, *at most one* `Harness` subclass. Re-exporting anything else from
`evals.signoflife` here — `DevelopmentSuite`, the preparers, `ARMS` — is harmless;
re-exporting a second `Taskset` or `Harness` breaks resolution with a `TypeError`
at dispatch. Two names, and a reason to think before adding a third.

`rl.movebox`, `rl.grounding` and `rl.target_box` have the identical defect and no
alias yet; they are unreachable through the plugin loader until they get one.
"""

from evals.harness import DesktopHarness
from evals.signoflife.taskset import SignOfLifeTaskset

__all__ = ["DesktopHarness", "SignOfLifeTaskset"]

PLUGIN_ID = "signoflife"
"""The id to pass as `--taskset.id` / `--harness.id`. `default_harness_id` returns
the taskset id when the taskset module also exports a `Harness`, so one id names
both."""
