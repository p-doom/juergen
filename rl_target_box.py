"""`rl.target_box` under a flat plugin id.

`signoflife.py` has the mechanism: verifiers 0.2.1 crashes on any dotted plugin id,
so an env is unreachable from `--taskset.id` / `HarnessConfig.id` until a root
module names it. The name here is `TargetBoxHarnessConfig.id`, which is the id
`loaders.harness_class()` imports.
"""

from rl.target_box import TargetBoxHarness, TargetBoxTaskset

__all__ = ["TargetBoxHarness", "TargetBoxTaskset"]
