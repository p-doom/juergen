"""OSWorld's task setup and its benchmark scorer, against a desktop we own.

`OSWorldEvaluateOracle` says the reward is `DesktopEnv.evaluate()`. This module
produces that number without letting OSWorld own the VM. The OSWorld task
preparer owns one bridge per episode; core desktop actions and
observations continue to use ``DesktopSession`` directly.

`DesktopEnv` itself is unusable here: it owns a VM lifecycle through its own
provider layer, and the lifecycle is ours (qemu + KVM under `desktop`'s pool) —
OSWorld's apptainer provider strips KVM ioctls on the hai-* nodes. Two pieces do
generalise across providers, and they are the ones we take:

  * `SetupController`, which turns a task JSON's `config` list into guest side
    effects over the in-VM HTTP agent, on the same port as the desktop client.
  * `evaluators.getters` + `evaluators.metrics`, which are the benchmark's
    definition of success.

`OSWorldBridge` is therefore both the caller of those and the `env` object they
are handed: OSWorld's getters take `(env, config)` and read `env.controller`,
`env.vm_ip`, `env.cache_dir`, `env.vm_platform`, `env.chromium_port`,
`env.server_port`, `env.vlc_port`, `env.getters`, `env.vm_machine`,
`env.current_use_proxy`, `env.setup_controller` and `env.evaluators`. Every one of
those is a plain attribute here, named and set in `__init__` or `bind()`.

`evaluate()` is a faithful port of `DesktopEnv.evaluate`
(OSWorld's `desktop_env/desktop_env.py:458-524`)
— the postconfig pass, the `infeasible` inversion, the FAIL short-circuit, the
`and`/`or` conjunction, the mean-vs-max, and `FileNotFoundError` counting as 0
under `and`. It is a port and not a call because `DesktopEnv.evaluate` is a method
on the object that owns the VM. Do not "improve" the arithmetic here: a difference
from it is a difference from the published benchmark.

The import is lazy, and has to be: OSWorld's `controllers/setup.py` pulls
playwright, pydrive and requests_toolbelt at module import, and
`evaluators.getters.__init__` is worse; a text-only eval that merely imports
`evals.vm` must not drag that in. `osworld_modules()` is the one seam that touches
`sys.path` and OSWorld's namespace, so a test can substitute it wholesale and
exercise every line of the binding and the arithmetic without OSWorld installed.

⚠️ `desktop_env` below is OSWorld's package, not ours — ours is `desktop`, and
it must stay that way: `sys.path` cannot override an entry already in
`sys.modules`, so any local distribution claiming the name `desktop_env` makes
`import desktop_env.controllers.setup` resolve to it instead, whatever
`OSWORLD_ROOT` says. Do not "tidy" these names to match ours; they are the other
package's.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

__all__ = [
    "OSWorldBridge",
    "OSWorldNotAvailable",
    "bridge_for_desktop",
    "osworld_modules",
    "osworld_root",
]

_LOGGER = logging.getLogger(__name__)


class OSWorldNotAvailable(RuntimeError):
    """OSWorld's tree is missing, unreadable, or not importable.

    Distinct from a scoring failure on purpose: a run that cannot reach the
    benchmark's own definition of success has produced no measurement, and must
    never be recorded as a task the model failed.
    """


def osworld_root() -> Path:
    """`$OSWORLD_ROOT`, validated as an actual OSWorld checkout.

    Validated rather than trusted: a re-clone or a wrong path otherwise surfaces
    as an ImportError naming a submodule, instead of naming the environment
    variable that chose it.
    """
    raw = os.environ.get("OSWORLD_ROOT", "").strip()
    if not raw:
        raise OSWorldNotAvailable(
            "OSWorld scoring needs $OSWORLD_ROOT (the checkout that owns "
            "evaluation_examples/ and desktop_env/evaluators/); it is unset"
        )
    root = Path(raw).expanduser()
    if not (root / "desktop_env" / "evaluators").is_dir():
        raise OSWorldNotAvailable(
            f"$OSWORLD_ROOT={raw!r} is not an OSWorld checkout: "
            f"{root / 'desktop_env' / 'evaluators'} does not exist"
        )
    return root


def osworld_modules() -> tuple[Any, Any, Any, Any]:
    """`(SetupController, PythonController, getters, metrics)` — the import seam.

    One function, so exactly one place imports OSWorld and exactly one place has
    to be substituted to test everything above it.
    """
    root = osworld_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from desktop_env.controllers.python import PythonController
        from desktop_env.controllers.setup import SetupController
        from desktop_env.evaluators import getters, metrics
    except ImportError as exc:  # pragma: no cover - needs a real OSWorld tree
        installed = sys.modules.get("desktop_env")
        raise OSWorldNotAvailable(
            f"OSWorld is at {root} but `desktop_env` imported as "
            f"{getattr(installed, '__file__', 'not importable')}: {exc}. If that "
            "path is not under $OSWORLD_ROOT, another distribution owns the "
            "import name and shadows OSWorld's."
        ) from exc
    return SetupController, PythonController, getters, metrics


class OSWorldBridge:
    """One desktop's OSWorld setup + scorer, and the `env` its getters read.

    Bound to a task by `bind()` / `setup()`, then asked for a number by
    `evaluate()`. Nothing here boots, resets or releases a VM: the lease is the
    session pool's, and this object is discarded with the episode.
    """

    def __init__(
        self,
        *,
        base_url: str,
        chromium_port: int = 9222,
        vlc_port: int = 8080,
        cache_dir: str | Path | None = None,
        vm_platform: str = "Ubuntu",
        vm_machine: str = "",
        screen_size: tuple[int, int] = (1920, 1080),
        client_password: str = "",
        loader: Callable[[], tuple[Any, Any, Any, Any]] = osworld_modules,
    ) -> None:
        split = urlsplit(base_url)
        self.vm_ip: str = split.hostname or "127.0.0.1"
        self.server_port: int = int(split.port or 5000)
        self.chromium_port: int = int(chromium_port)
        self.vlc_port: int = int(vlc_port)
        self.vm_platform: str = vm_platform
        self.vm_machine: str = vm_machine
        self.current_use_proxy: bool = False
        self.screen_width, self.screen_height = (int(screen_size[0]), int(screen_size[1]))
        self.action_history: list[Any] = []
        """OSWorld reads only the *last* entry, and only to spot a declared FAIL.
        Our harness declares that through the episode's OSWorld bridge."""

        self._loader = loader
        self._client_password = client_password
        self._cache_root = Path(cache_dir) if cache_dir is not None else None
        self._modules: tuple[Any, Any, Any, Any] | None = None
        self._controller: Any = None
        self._setup_controller: Any = None

        # Bound per task by `bind()`; named here so the shape is one place.
        self.task_id: str | None = None
        self.instruction: str = ""
        self.config: list[dict[str, Any]] = []
        self.evaluator: dict[str, Any] | None = None
        self.cache_dir: str = ""
        self.metric: Any = None
        self.metric_conj: str = "and"
        self.metric_options: Any = {}
        self.result_getter: Any = None
        self.expected_getter: Any = None

    def _load(self) -> tuple[Any, Any, Any, Any]:
        if self._modules is None:
            self._modules = self._loader()
        return self._modules

    @property
    def getters(self) -> Any:
        """`env.getters` — several getters recurse through it by name."""
        return self._load()[2]

    @property
    def getter(self) -> Any:
        """`env.getter`, singular. One getter spells it that way; both are the
        same module, and a missing alias would be an `AttributeError` deep inside
        a scorer rather than here."""
        return self.getters

    @property
    def evaluators(self) -> Any:
        return self._load()[3]

    @property
    def controller(self) -> Any:
        """`env.controller` — 106 of the getters' env reads. A `PythonController`
        against the same in-VM agent our transport drives."""
        if self._controller is None:
            _, python_controller, _, _ = self._load()
            self._controller = python_controller(
                vm_ip=self.vm_ip, server_port=self.server_port
            )
        return self._controller

    @property
    def setup_controller(self) -> Any:
        if self._setup_controller is None:
            setup_controller, _, _, _ = self._load()
            self._setup_controller = setup_controller(
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                chromium_port=self.chromium_port,
                vlc_port=self.vlc_port,
                cache_dir=str(self._cache_base()),
                client_password=self._client_password,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
            )
        return self._setup_controller

    def _cache_base(self) -> Path:
        if self._cache_root is None:
            self._cache_root = Path(tempfile.mkdtemp(prefix="osworld-cache-"))
        self._cache_root.mkdir(parents=True, exist_ok=True)
        return self._cache_root

    def bind(self, task_config: dict[str, Any]) -> None:
        """`DesktopEnv._set_task_info` + `_set_evaluator_info`, same arithmetic.

        A task JSON with no `evaluator` binds fine and leaves `evaluate()` unable
        to score — that is the grounding family, which uses OSWorld *setup* and
        our own bbox oracle, and must not be forced to carry a benchmark
        evaluator it does not have.
        """
        self.task_id = str(task_config.get("id") or "task")
        self.instruction = str(task_config.get("instruction") or "")
        self.config = list(task_config.get("config") or [])
        self.cache_dir = str(self._cache_base() / self.task_id)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        if self._setup_controller is not None:
            self._setup_controller.reset_cache_dir(self.cache_dir)
        evaluator = task_config.get("evaluator")
        self.evaluator = dict(evaluator) if isinstance(evaluator, dict) else None
        if self.evaluator is None:
            self.metric = None
            self.result_getter = None
            self.expected_getter = None
            self.metric_options = {}
            return
        self._bind_evaluator(self.evaluator)

    def _bind_evaluator(self, evaluator: dict[str, Any]) -> None:
        metrics = self.evaluators
        getters = self.getters
        func = evaluator["func"]
        listed = isinstance(func, list)
        self.metric = (
            [getattr(metrics, name) for name in func] if listed else getattr(metrics, func)
        )
        self.metric_conj = evaluator.get("conj", "and")

        def _bind_getters(key: str) -> Any:
            spec = evaluator.get(key)
            if not spec:
                return [None] * len(self.metric) if listed else None
            if isinstance(spec, list):
                return [
                    getattr(getters, f"get_{item['type']}") if item else None for item in spec
                ]
            return getattr(getters, f"get_{spec['type']}")

        self.result_getter = _bind_getters("result")
        self.expected_getter = _bind_getters("expected")
        options = evaluator.get("options")
        if isinstance(options, list):
            self.metric_options = [option or {} for option in options]
        elif options is not None:
            self.metric_options = options
        else:
            self.metric_options = [{}] * len(self.metric) if listed else {}
        if listed and not (
            len(self.metric)
            == len(self.result_getter)
            == len(self.expected_getter)
            == len(self.metric_options)
        ):
            raise ValueError(
                f"OSWorld task {self.task_id!r} declares {len(self.metric)} metrics but "
                f"{len(self.result_getter)} result getters, "
                f"{len(self.expected_getter)} expected getters and "
                f"{len(self.metric_options)} option sets"
            )

    def setup(self, task_config: dict[str, Any]) -> int:
        """Bind the task and run its `config` steps. Returns the step count.

        Binding here rather than in a separate call is what makes `evaluate()`
        answerable with no arguments: by the time the episode ends, the only task
        this desktop was ever prepared for is the one to score.
        """
        self.bind(task_config)
        if not self.config:
            return 0
        self.setup_controller.setup(self.config, self.current_use_proxy)
        return len(self.config)

    def run_setup_steps(self, steps: list[dict[str, Any]]) -> int:
        """Run already-materialized setup steps without binding an OSWorld task."""
        self.setup_controller.setup(steps, self.current_use_proxy)
        return len(steps)

    def declare_terminal(self, control: str | None) -> None:
        """Record the model's terminal control so `infeasible` can be scored.

        OSWorld's only use of `action_history` is the last entry, and only to ask
        whether the model said FAIL: on an `infeasible` task saying so IS the
        success condition, and on every other task it forfeits. Our grammars call
        that control `fail`; nothing else in the history matters, so nothing else
        is kept.
        """
        if control == "fail":
            self.action_history.append("FAIL")

    def evaluate(self) -> float:
        """`DesktopEnv.evaluate()` (OSWorld's `desktop_env.py:458-524`), ported verbatim."""
        if self.evaluator is None:
            raise OSWorldNotAvailable(
                f"task {self.task_id!r} has no `evaluator` block, so there is no "
                "OSWorld score to compute; evaluate_on_finish only applies to the "
                "benchmark family"
            )
        postconfig = self.evaluator.get("postconfig", [])
        if postconfig:
            self.setup_controller.setup(postconfig, self.current_use_proxy)

        declared_fail = bool(self.action_history) and self.action_history[-1] == "FAIL"
        if self.evaluator["func"] == "infeasible":
            return 1.0 if declared_fail else 0.0
        if declared_fail:
            return 0.0

        if isinstance(self.metric, list):
            results: list[float] = []
            for index, metric in enumerate(self.metric):
                try:
                    result_state = self.result_getter[index](
                        self, self.evaluator["result"][index]
                    )
                except FileNotFoundError:
                    # Two deliberate deviations, both in the `or` branch, both
                    # places where upstream cannot produce a number at all:
                    # OSWorld falls through to use an unbound `result_state`
                    # (OSWorld's `desktop_env.py:491-495` — an UnboundLocalError, not a
                    # score), and an all-missing `or` list then divides by zero.
                    # Skipping the metric and returning 0.0 for "no metric could
                    # be read" is the same verdict its `and` branch already
                    # gives. The `and` path — the one every published number
                    # goes through — is byte-for-byte upstream's.
                    _LOGGER.error("OSWorld getter: file not found")
                    if self.metric_conj == "and":
                        return 0.0
                    continue
                if self.evaluator.get("expected") and self.expected_getter[index] is not None:
                    expected_state = self.expected_getter[index](
                        self, self.evaluator["expected"][index]
                    )
                    score = metric(
                        result_state, expected_state, **self.metric_options[index]
                    )
                else:
                    score = metric(result_state, **self.metric_options[index])
                if self.metric_conj == "and" and float(score) == 0.0:
                    return 0.0
                if self.metric_conj == "or" and float(score) == 1.0:
                    return 1.0
                results.append(float(score))
            if not results:
                return 0.0
            return (
                sum(results) / len(results)
                if self.metric_conj == "and"
                else max(results)
            )

        try:
            result_state = self.result_getter(self, self.evaluator["result"])
        except FileNotFoundError:
            _LOGGER.error("OSWorld getter: file not found")
            return 0.0
        if self.evaluator.get("expected") and self.expected_getter is not None:
            expected_state = self.expected_getter(self, self.evaluator["expected"])
            return float(self.metric(result_state, expected_state, **self.metric_options))
        return float(self.metric(result_state, **self.metric_options))


def bridge_for_desktop(
    session: Any, *, cache_dir: str | Path | None = None
) -> OSWorldBridge:
    """Bind OSWorld's controller/scorer to one live ``DesktopSession``."""
    base_url = session.base_url
    ports = session.ports
    chromium_port = getattr(ports, "chromium", None)
    vlc_port = getattr(ports, "vlc", None)
    if type(chromium_port) is not int or chromium_port <= 0:
        raise RuntimeError("desktop session has no forwarded Chromium port")
    if type(vlc_port) is not int or vlc_port <= 0:
        raise RuntimeError("desktop session has no forwarded VLC port")
    return OSWorldBridge(
        base_url=base_url,
        chromium_port=chromium_port,
        vlc_port=vlc_port,
        cache_dir=cache_dir,
        screen_size=session.screen_size(),
    )
