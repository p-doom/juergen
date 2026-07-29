"""Standalone DETERMINISTIC OSWorld task-success evaluate() against the freeroll VM.

OSWorld task-success is fully deterministic (per-task getter/checker functions that
read real VM state — files, process output, config); there is NO VLM judge in the
real benchmark. We wire DesktopEnv.evaluate() standalone (no DesktopEnv, no apptainer
fork) by building a minimal `env` shim whose `.controller` is the OSWorld
PythonController pointed at the SAME freeroll qemu VM we roll out on, and resolving
the task's evaluator spec (func->metric, result/expected->getter) exactly as
DesktopEnv.__init__ does. Call at EPISODE END (VM still in its final state).

This is the GOLD filter signal for which teacher traces are worth distilling —
replaces the leaky VLM judge for real-OSWorld tasks.

Caveat: cloud/a11y-tree getters (get_cloud_file, get_accessibility_tree, ...) need
internet / the a11y endpoint; tasks whose evaluator needs those return eval_error and
are simply excluded from the deterministic-success filter (still usable via other
signals). File/process/config checkers — the majority — work over the VM Flask agent.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_OSWORLD = os.environ.get("OSWORLD_ROOT", "/fast/home/franz.srambical/OSWorld")
if _OSWORLD not in sys.path:
    sys.path.insert(0, _OSWORLD)
_JUERGEN_EVAL = "/fast/home/franz.srambical/juergen/eval"
if _JUERGEN_EVAL not in sys.path:
    sys.path.insert(0, _JUERGEN_EVAL)

from desktop_env.controllers.python import PythonController  # noqa: E402
from desktop_env.evaluators import metrics, getters  # noqa: E402
# reuse the grounding runner's working SetupController subclass (for postconfig)
import osworld_grounding_runner as gr  # noqa: E402

_LOGGER = logging.getLogger("osworld_evaluate")


class _EvalEnv:
    """Minimal DesktopEnv stand-in exposing exactly what evaluate() + getters read."""

    def __init__(self, task: dict, *, vm_port: int, cache_dir: Path,
                 chromium_port: int, vlc_port: int, screen_w: int, screen_h: int,
                 action_history=None):
        self.vm_ip = "localhost"
        self.server_port = vm_port
        self.cache_dir = str(cache_dir)
        self.vm_platform = "Ubuntu"
        self.vm_screen_size = (screen_w, screen_h)
        self.enable_proxy = False
        self.action_history = list(action_history or [])
        self.controller = PythonController(vm_ip="localhost", server_port=vm_port)
        self.setup_controller = gr._GroundingSetupController(
            vm_ip="localhost", server_port=vm_port, chromium_port=chromium_port,
            vlc_port=vlc_port, cache_dir=str(cache_dir), client_password="",
            screen_width=screen_w, screen_height=screen_h)

        # --- evaluator resolution: copied verbatim from DesktopEnv.__init__ ---
        self.evaluator = task["evaluator"]
        self.metric = ([getattr(metrics, f) for f in self.evaluator["func"]]
                       if isinstance(self.evaluator["func"], list)
                       else getattr(metrics, self.evaluator["func"]))
        self.metric_conj = self.evaluator.get("conj", "and")
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter = ([getattr(getters, "get_{:}".format(r["type"])) for r in self.evaluator["result"]]
                                  if isinstance(self.evaluator["result"], list)
                                  else getattr(getters, "get_{:}".format(self.evaluator["result"]["type"])))
        else:
            self.result_getter = [None] * len(self.metric) if isinstance(self.metric, list) else None
        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter = ([getattr(getters, "get_{:}".format(e["type"])) if e else None for e in self.evaluator["expected"]]
                                    if isinstance(self.evaluator["expected"], list)
                                    else getattr(getters, "get_{:}".format(self.evaluator["expected"]["type"])))
        else:
            self.expected_getter = [None] * len(self.metric) if isinstance(self.metric, list) else None
        self.metric_options = ([opt if opt else {} for opt in self.evaluator["options"]]
                               if isinstance(self.evaluator.get("options", {}), list)
                               else self.evaluator["options"] if "options" in self.evaluator
                               else [{}] * len(self.metric) if isinstance(self.metric, list) else {})

    def evaluate(self) -> float:
        """DesktopEnv.evaluate() logic, verbatim (postconfig -> getters -> metrics)."""
        postconfig = self.evaluator.get("postconfig", [])
        if postconfig:
            try:
                self.setup_controller.setup(postconfig, self.enable_proxy)
            except TypeError:
                self.setup_controller.setup(postconfig)
        if self.evaluator["func"] == "infeasible":
            if self.action_history and (self.action_history[-1] == "FAIL"):
                return 1
            return 0
        if self.action_history and (self.action_history[-1] == "FAIL"):
            return 0

        if isinstance(self.metric, list):
            results = []
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = self.result_getter[idx](self, config)
                except FileNotFoundError:
                    if self.metric_conj == 'and':
                        return 0
                    continue
                if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                    expected_state = self.expected_getter[idx](self, self.evaluator["expected"][idx])
                    m = metric(result_state, expected_state, **self.metric_options[idx])
                else:
                    m = metric(result_state, **self.metric_options[idx])
                if self.metric_conj == 'and' and float(m) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(m) == 1.0:
                    return 1
                results.append(m)
            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            try:
                result_state = self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                return 0
            if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                expected_state = self.expected_getter(self, self.evaluator["expected"])
                return float(self.metric(result_state, expected_state, **self.metric_options))
            return float(self.metric(result_state, **self.metric_options))


def evaluate_task(task: dict, *, vm_port: int, cache_dir, chromium_port: int, vlc_port: int,
                  screen_w: int = 1920, screen_h: int = 1080, action_history=None) -> tuple[float | None, str | None]:
    """Return (score in [0,1], None) or (None, error) if the evaluator can't run here."""
    try:
        env = _EvalEnv(task, vm_port=vm_port, cache_dir=Path(cache_dir), chromium_port=chromium_port,
                       vlc_port=vlc_port, screen_w=screen_w, screen_h=screen_h, action_history=action_history)
        return float(env.evaluate()), None
    except Exception as e:  # cloud/a11y getter, missing attr, network, etc.
        return None, f"{type(e).__name__}: {str(e)[:200]}"
