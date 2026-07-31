from __future__ import annotations

from typing import Any

from ..rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ..rung1b_realapps.training.env import VmEnvironmentBackend
from ..rung1b_realapps.vm import JSON_MARKER, _guest_dir, _run_json, probe_fixture, probe_geometry
from .actions import RecoveryGeometry
from .env import BackendSnapshot
from .spec import RecoveryTask


class VmRecoveryBackend:
    """CPU/KVM backend; no model runtime is imported or invoked."""

    def __init__(self, session: Any) -> None:
        self.base = VmEnvironmentBackend(session)
        self.geometry = RecoveryGeometry()

    @property
    def transport(self) -> Any:
        if self.base.transport is None:
            raise RuntimeError("VM recovery backend was not reset")
        return self.base.transport

    def _wrong_file_point(self, task: RecoveryTask, name: str) -> tuple[int, int]:
        code = f"""
import json,pyatspi
wanted={name!r}; found=None
def walk(node,depth=0):
 global found
 if depth>12 or found is not None: return
 try:
  if node.name == wanted:
   e=node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
   if e.width>4 and e.height>4: found=[int(e.x+e.width//2),int(e.y+e.height//2)]; return
  for child in node: walk(child,depth+1)
 except Exception: pass
walk(pyatspi.Registry.getDesktop(0))
print({JSON_MARKER!r}+json.dumps({{'point':found}},sort_keys=True))
""".strip()
        value = _run_json(self.transport, ["/usr/bin/python3", "-c", code])
        point = value.get("point")
        if not isinstance(point, list) or len(point) != 2:
            raise RuntimeError(f"wrong-file accessibility point missing: {value}")
        return int(point[0]), int(point[1])

    def reset(self, task: RecoveryTask) -> BackendSnapshot:
        snapshot = self.base.reset(task.fixture)
        base_geometry = snapshot.geometry
        wrong_file_source = RecoveryGeometry().wrong_file_source
        if task.perturbation == "wrong_file_drag":
            wrong_name = f"wrong-parcel-{task.fixture.parameter_seed}.txt"
            path = _guest_dir(task.fixture) / wrong_name
            source = f"controlled wrong-file distractor {task.fixture.parameter_seed}\n"
            code = (
                "import pathlib;"
                f"p=pathlib.Path({str(path)!r});p.write_text({source!r},encoding='utf-8')"
            )
            self.transport.execute_argv(["python3", "-c", code])
            NativeAbsoluteExecutor(self.transport).execute(
                {"action": "key", "keys": ["F5"]}
            )
            self.transport.wait(1.0)
            base_geometry = probe_geometry(self.transport, task.fixture)
            wrong_file_source = self._wrong_file_point(task, wrong_name)
            snapshot = type(snapshot)(
                self.base._screenshot(),
                probe_fixture(self.transport, task.fixture),
                base_geometry,
                self.transport.cursor_position(),
            )
        self.geometry = RecoveryGeometry(
            base=base_geometry,
            wrong_file_source=wrong_file_source,
        )
        return BackendSnapshot(
            snapshot.screenshot_png,
            snapshot.hidden_state,
            self.geometry,
            snapshot.cursor,
            "ok",
        )

    def dispatch(
        self, task: RecoveryTask, arm: str, action: dict[str, Any] | str
    ) -> BackendSnapshot:
        status = "error"
        try:
            if arm == "native_absolute_control":
                if not isinstance(action, dict):
                    raise TypeError("native recovery action must be an object")
                result = NativeAbsoluteExecutor(self.transport).execute(action)
            elif arm == "compact_raw_phaseb":
                if not isinstance(action, str):
                    raise TypeError("compact recovery action must be text")
                result = CompactRawExecutor(self.transport).execute(action)
            else:
                raise ValueError(f"unknown recovery arm: {arm}")
            status = result.executor_dispatch_status
        except Exception:
            # Preserve the label boundary. The environment records an executor
            # failure and the VM runner fails the cell without calling it a
            # perturbation or natural ineffective action.
            status = "error"
        return BackendSnapshot(
            self.base._screenshot(),
            probe_fixture(self.transport, task.fixture),
            self.geometry,
            self.transport.cursor_position(),
            status,
        )
