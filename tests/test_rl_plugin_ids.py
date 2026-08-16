"""The three RL envs, reached the way a run reaches them: by plugin id.

`load_harness` is `harness_class(config.id)(config)`, so the harness config's own
`id` is the module verifiers imports. Each env set that field long before a module
with that name existed, and the failure is a `ModuleNotFoundError` at dispatch, not
at import — a rename that misses the root module is silent until a run spends a VM
on it.

`default_harness_id` is the other half: it returns `"bash"` for anything it cannot
resolve, so an unreachable env does not raise, it quietly runs the wrong harness.
"""

from __future__ import annotations

import pytest
from verifiers.v1.loaders import default_harness_id, harness_class, taskset_class

from rl.grounding import GroundingHarness, GroundingHarnessConfig, GroundingTaskset
from rl.movebox import MoveBoxHarness, MoveBoxHarnessConfig, MoveBoxTaskset
from rl.target_box import TargetBoxHarness, TargetBoxHarnessConfig, TargetBoxTaskset

ENVS = [
    ("rl_target_box", TargetBoxHarnessConfig, TargetBoxTaskset, TargetBoxHarness),
    ("rl_movebox", MoveBoxHarnessConfig, MoveBoxTaskset, MoveBoxHarness),
    ("rl_grounding", GroundingHarnessConfig, GroundingTaskset, GroundingHarness),
]


@pytest.mark.parametrize("plugin_id,config_type,taskset,harness", ENVS)
def test_the_harness_config_default_id_is_a_module_that_resolves(
    plugin_id, config_type, taskset, harness
) -> None:
    assert config_type().id == plugin_id
    assert harness_class(config_type().id) is harness


@pytest.mark.parametrize("plugin_id,config_type,taskset,harness", ENVS)
def test_the_same_id_names_the_taskset_and_its_harness(
    plugin_id, config_type, taskset, harness
) -> None:
    assert taskset_class(plugin_id) is taskset
    assert default_harness_id(plugin_id) == plugin_id
