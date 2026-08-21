"""In-guest setup for CUA-Gym bundles, behind the `Preparer` seam.

`prepare` runs the bundle's `config` steps — `download` (a `./`-relative bundle
member written to a guest path), `execute` (a shell line or argv), `sleep`, and
`open` (fire-and-forget `xdg-open`; the bundles pair it with an explicit
`sleep`, which is the settle) — after making sure the document libraries the
graders import exist in the guest, since a snapshot restore wipes user-site
packages. `probe` is read-only, as the seam requires.

`run_reward` is not a probe and is deliberately not called from one: the
family's verifier is an in-guest script with side effects (5 of the 28 graders
even carry a `postconfig` that presses ctrl-s before reading state). It runs
from the oracle at scoring time, inside the lease's grace window — the same
allowance OSWorld's `evaluate()` gives its own `postconfig`.
"""

from __future__ import annotations

import base64
import logging
import shlex
import time
from pathlib import Path
from typing import Any

from evals.tasks import DesktopTaskData, register_preparer

__all__ = [
    "GUEST_PACKAGES",
    "CuaGymPreparer",
    "ensure_guest_packages",
    "run_reward",
    "run_steps",
    "write_guest_file",
]

_LOGGER = logging.getLogger(__name__)

_REWARD_PATH = "/tmp/cuagym_reward.py"
_PROVISION_LOG = "/tmp/cuagym_provision.log"
_PROVISION_TIMEOUT_S = 300.0
_PROVISION_POLL_S = 5.0
# printf's argument rides the guest agent's argv; stay far under ARG_MAX.
_WRITE_CHUNK_BYTES = 256 * 1024

# (pip distribution, importable module) — what the bundles' scripts import.
GUEST_PACKAGES: tuple[tuple[str, str], ...] = (
    ("PyPDF2", "PyPDF2"),
    ("odfpy", "odf"),
    ("openpyxl", "openpyxl"),
    ("pandas", "pandas"),
    ("pdfplumber", "pdfplumber"),
    ("pymupdf", "fitz"),
    ("python-docx", "docx"),
    ("python-pptx", "pptx"),
)


def _stdout(result: dict[str, Any]) -> str:
    return str(result.get("output") or "")


def write_guest_file(session: Any, guest_path: str, data: bytes) -> None:
    """Write bytes to a guest path via the command channel, chunked."""

    quoted = shlex.quote(guest_path)
    session.execute_argv(
        ["bash", "-lc", f"mkdir -p {shlex.quote(str(Path(guest_path).parent))}"]
    )
    session.execute_argv(["bash", "-lc", f": > {quoted}"])
    for offset in range(0, len(data) or 1, _WRITE_CHUNK_BYTES):
        encoded = base64.b64encode(data[offset : offset + _WRITE_CHUNK_BYTES]).decode()
        session.execute_argv(
            ["bash", "-lc", f"printf '%s' {shlex.quote(encoded)} | base64 -d >> {quoted}"]
        )


def run_steps(session: Any, steps: list[dict[str, Any]], bundle_dir: Path) -> int:
    """Run `config`/`postconfig` steps. Returns how many ran."""

    ran = 0
    for step in steps:
        kind = str(step.get("type"))
        parameters = dict(step.get("parameters") or {})
        if kind == "download":
            for entry in parameters.get("files") or []:
                url = str(entry["url"])
                if not url.startswith("./"):
                    raise ValueError(
                        f"cuagym download url must be a ./-relative bundle member, got {url!r}"
                    )
                write_guest_file(
                    session, str(entry["path"]), (bundle_dir / url[2:]).read_bytes()
                )
        elif kind == "execute":
            command = parameters.get("command")
            argv = list(command) if isinstance(command, list) else ["bash", "-lc", str(command)]
            session.execute_argv(argv)
        elif kind == "sleep":
            time.sleep(float(parameters.get("seconds") or 0.0))
        elif kind == "open":
            path = shlex.quote(str(parameters["path"]))
            session.execute_argv(
                ["bash", "-lc", f"nohup xdg-open {path} >/dev/null 2>&1 &"]
            )
        else:
            raise ValueError(f"unsupported cuagym setup step type {kind!r}")
        ran += 1
    return ran


def ensure_guest_packages(session: Any) -> bool:
    """Install the grader imports if the guest lost them. Returns True if it installed."""

    probe = ["python3", "-c", "import " + ", ".join(module for _, module in GUEST_PACKAGES)]
    if "ImportError" not in _err(session.execute_argv(probe)) and _ok(session, probe):
        return False
    distributions = " ".join(distribution for distribution, _ in GUEST_PACKAGES)
    session.execute_argv(
        [
            "bash",
            "-lc",
            "setsid bash -c 'python3 -m pip install --user --no-input "
            f"--disable-pip-version-check {distributions} > {_PROVISION_LOG} 2>&1' "
            ">/dev/null 2>&1 < /dev/null &",
        ]
    )
    deadline = time.monotonic() + _PROVISION_TIMEOUT_S
    while time.monotonic() < deadline:
        if _ok(session, probe):
            return True
        time.sleep(_PROVISION_POLL_S)
    tail = _stdout(session.execute_argv(["bash", "-lc", f"tail -5 {_PROVISION_LOG}"]))
    raise RuntimeError(f"cuagym guest provisioning did not complete; pip log tail:\n{tail}")


def _ok(session: Any, argv: list[str]) -> bool:
    result = session.execute_argv(argv)
    if "returncode" in result:
        return int(result["returncode"]) == 0
    return "Error" not in _stdout(result) and not _err(result)


def _err(result: dict[str, Any]) -> str:
    return str(result.get("error") or "")


def run_reward(
    session: Any,
    reward_source: str,
    postconfig: list[dict[str, Any]],
    bundle_dir: Path,
) -> str:
    """Run the bundle's verifier in the guest and return its stdout."""

    if postconfig:
        run_steps(session, postconfig, bundle_dir)
    write_guest_file(session, _REWARD_PATH, reward_source.encode("utf-8"))
    return _stdout(session.execute_argv(["python3", _REWARD_PATH]))


class CuaGymPreparer:
    """CUA-Gym bundles: provision grader imports, then run the `config` steps."""

    kind = "cuagym"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        provisioned = ensure_guest_packages(session)
        steps = run_steps(
            session,
            list(task.setup.get("config") or []),
            Path(str(task.setup["bundle_dir"])),
        )
        return {"prepared": "cuagym", "steps": steps, "provisioned": provisioned}

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del task
        return {
            "cursor": list(session.cursor_position()),
            "screen": list(session.screen_size()),
        }


register_preparer(CuaGymPreparer())
