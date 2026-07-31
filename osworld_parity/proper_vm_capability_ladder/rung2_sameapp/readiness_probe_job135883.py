"""Development-only reproduction probe for the job 135883 readiness boundary.

This never opens evaluation fixtures and never invokes a model. It replays the
existing rung-1 development self-check verbatim while preserving evidence that
the historical harness discarded when ``wait_ready`` raised.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import urllib.request
from pathlib import Path
from typing import Any

from ..rung1.selfcheck import run_vm_selfcheck
from ..rung1.vm import DEFAULT_PROVIDER, DEFAULT_QCOW, DEFAULT_QEMU, KvmFixtureSession


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _guest_diagnostics(session: KvmFixtureSession, url: str) -> dict[str, Any]:
    transport = session.transport
    if transport is None:
        return {"error": "transport missing"}
    code = f"""
import json,pathlib,subprocess
def run(command):
 try:
  value=subprocess.run(command,capture_output=True,text=True,timeout=8)
  return {{'rc':value.returncode,'stdout':value.stdout[-12000:],'stderr':value.stderr[-4000:]}}
 except Exception as exc: return {{'error':type(exc).__name__+': '+str(exc)}}
logs={{}}
for raw in ['/tmp/rung1a_chrome.log']:
 p=pathlib.Path(raw)
 if p.exists(): logs[raw]=p.read_text(errors='replace')[-12000:]
value={{'processes':run(['ps','-eo','pid,ppid,stat,etime,comm,args']),
 'windows':run(['wmctrl','-lx']), 'fixture_curl':run(['curl','-v','--max-time','8',{url!r}]),
 'chrome_debug':run(['curl','-fsS','--max-time','5','http://127.0.0.1:9222/json']), 'logs':logs}}
print('RUNG1A_READINESS_DIAGNOSTIC='+json.dumps(value,sort_keys=True))
""".strip()
    result = transport.execute_argv(["python3", "-c", code])
    output = result.get("output")
    if not isinstance(output, str):
        return {"error": "diagnostic command returned no output"}
    prefix = "RUNG1A_READINESS_DIAGNOSTIC="
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    return json.loads(lines[-1][len(prefix) :]) if lines else {"raw_output": output}


def run_probe(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    original = KvmFixtureSession.launch_fixture

    def instrumented(self: KvmFixtureSession, fixture_server: Any, fixture: Any, *, timeout_s: float = 60.0) -> dict[str, Any]:
        try:
            return original(self, fixture_server, fixture, timeout_s=timeout_s)
        except Exception as exc:
            url = fixture_server.guest_url(fixture)
            provider_state: dict[str, Any] = {}
            if self.provider is not None:
                raw = self.provider.state(str(self.qcow))
                provider_state = {
                    "ports": raw.get("ports"),
                    "qemu_poll": raw["proc"].poll(),
                    "qemu_log": str(raw.get("log")),
                    "timings": list(getattr(self.provider, "timings", [])),
                }
            evidence = {
                "schema_version": 1,
                "classification": "development_only_job135883_reproduction",
                "fixture_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "timeout_s": timeout_s,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "host_fixture_state": fixture_server.store.snapshot(fixture.id),
                "provider": provider_state,
                "guest": _guest_diagnostics(self, url),
            }
            if self.transport is not None:
                try:
                    with urllib.request.urlopen(
                        self.transport.base_url + "/screenshot", timeout=10
                    ) as response:
                        screenshot = response.read()
                    (output / "readiness_failure.png").write_bytes(screenshot)
                    evidence["screenshot_bytes"] = len(screenshot)
                except Exception as screenshot_exc:
                    evidence["screenshot_error"] = (
                        f"{type(screenshot_exc).__name__}: {screenshot_exc}"
                    )
            _write_json(output / "readiness_diagnostic.json", evidence)
            raise

    KvmFixtureSession.launch_fixture = instrumented
    try:
        result = run_vm_selfcheck(
            output=output,
            qcow=qcow,
            qemu=qemu,
            provider_path=provider,
            expected_provider_sha256=None,
        )
        payload = {
            "schema_version": 1,
            "status": "passed_without_readiness_failure",
            "development_only": True,
            "selfcheck_cell_count": result["selfcheck_cell_count"],
        }
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "reproduced_failure",
            "development_only": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "readiness_diagnostic_present": (output / "readiness_diagnostic.json").is_file(),
        }
    finally:
        KvmFixtureSession.launch_fixture = original
    _write_json(output / "probe.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    args = parser.parse_args(argv)
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise SystemExit("/dev/kvm unavailable")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("GPU allocation is forbidden")
    payload = run_probe(
        output=args.output,
        qcow=args.qcow,
        qemu=args.qemu,
        provider=args.provider,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
