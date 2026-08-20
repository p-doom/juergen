"""The sign-of-life gate under a flat plugin id.

`verifiers` 0.2.1 cannot resolve a dotted plugin id, and `evals.signoflife` is
dotted.

`loaders._import_plugin` (`verifiers/v1/loaders.py:31-43`) does

    namespaced = f"verifiers.v1.tasksets.{module}"
    target = namespaced if importlib.util.find_spec(namespaced) else module

and `find_spec` imports the parent of the name it is given. For
`evals.signoflife` that parent is `verifiers.v1.tasksets.evals`, which does not
exist, so `find_spec` raises `ModuleNotFoundError` out of the guard meant to catch
it — the `else module` fallback is unreachable for any id containing a dot, and
the failure is a crash rather than a miss:

    >>> taskset_class("evals.signoflife")
    ModuleNotFoundError: No module named 'verifiers.v1.tasksets.evals'

A flat id takes the intended path: `find_spec("verifiers.v1.tasksets.signoflife")`
returns `None`, and `importlib.import_module("signoflife")` finds this file at the
repo root. So the gate's plugin id is `signoflife`, and `evals/signoflife/` stays
where it belongs in the package layout.

Alternatives rejected: vendoring a patched `loaders.py` forks the framework we pin
exactly (`verifiers==0.2.1`) for its internal API, and moving the package to the
repo root would drag `evals/harness.py`, `evals/tasks.py` and the fixtures with
it, since the taskset is a thin shell over those.

The `__all__` below is a verifiers contract. `loaders._plugin_class` scans it and
requires exactly one `Taskset` subclass and, for the harness, at most one
`Harness` subclass. Re-exporting anything else from `evals.signoflife` here —
`DevelopmentSuite`, the preparers, `ARMS` — is harmless; re-exporting a second
`Taskset` or `Harness` breaks resolution with a `TypeError` at dispatch.

`rl.movebox`, `rl.grounding` and `rl.target_box` have the identical defect and the
same remedy: `rl_movebox.py`, `rl_grounding.py` and `rl_target_box.py`.
`tests/test_rl_plugin_ids.py` resolves all three by id.
"""

from evals.harness import DesktopHarness
from evals.signoflife.taskset import SignOfLifeTaskset

__all__ = ["DesktopHarness", "SignOfLifeTaskset"]

PLUGIN_ID = "signoflife"
"""The id to pass as `--taskset.id` / `--harness.id`. `default_harness_id` returns
the taskset id when the taskset module also exports a `Harness`, so one id names
both."""
