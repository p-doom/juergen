from __future__ import annotations

import base64
import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from ..rung1.transport import HttpVmTransport, TransportError
from ..rung1.vm import (  # audited/cherry-picked pinned provider boundary
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    KvmFixtureSession,
    sha256_file,
)
from .fixtures import Fixture
from .states import base_state, drag_state, focus_state, scroll_state
from .trajectory import UiGeometry


GUEST_ROOT_NAME = ".r1b_realapps_development"
JSON_MARKER = "RUNG1B_JSON="
READINESS_STABLE_PROBES = 3
SETTLE_STABLE_PROBES = 3


class AppReadinessError(TransportError):
    """A phase-specific setup failure with guest evidence attached."""

    def __init__(self, *, fixture_id: str, failed_phase: str, evidence: dict[str, Any]):
        self.fixture_id = fixture_id
        self.failed_phase = failed_phase
        self.evidence = evidence
        super().__init__(
            f"{fixture_id}: readiness failed at {failed_phase}; "
            f"last_error={evidence.get('last_error')!r}"
        )


class AppSettleTimeout(TransportError):
    """An action was not acknowledged and observed stable before its deadline."""

    def __init__(self, *, fixture_id: str, phase: str, evidence: dict[str, Any]):
        self.fixture_id = fixture_id
        self.phase = phase
        self.evidence = evidence
        super().__init__(
            f"{fixture_id}: action settle timed out at {phase}; "
            f"acknowledged={evidence.get('acknowledged')!r}; "
            f"last_error={evidence.get('last_error')!r}"
        )

# This program runs as the VM agent user.  The pinned image used to expose
# /home/oai/share, but that is not part of the agent contract: some snapshots
# have HOME=/home/oai while the directory itself is absent.  Pick only a
# standard per-user application base that already exists and is writable, then
# prove the private fixture root is owned and writable by this process.
GUEST_ROOT_RESOLVER = f"""
import json,os,pathlib,tempfile
name={GUEST_ROOT_NAME!r}
raw=[os.environ.get('XDG_RUNTIME_DIR'),os.environ.get('HOME'),tempfile.gettempdir()]
seen=set(); errors=[]
for item in raw:
 try:
  if not item: continue
  base=pathlib.Path(item).resolve(strict=True)
  if base in seen: continue
  seen.add(base)
  if not base.is_dir() or not os.access(base,os.R_OK|os.W_OK|os.X_OK):
   errors.append(str(base)+':not-writable'); continue
  root=base/name
  root.mkdir(mode=0o700,parents=False,exist_ok=True)
  resolved=root.resolve(strict=True)
  if resolved.parent != base or resolved.name != name:
   errors.append(str(root)+':escaped-base'); continue
  if resolved.stat().st_uid != os.geteuid():
   errors.append(str(resolved)+':wrong-owner'); continue
  os.chmod(resolved,0o700)
  fd,probe=tempfile.mkstemp(prefix='.write-probe-',dir=resolved)
  os.close(fd); os.unlink(probe)
  print({JSON_MARKER!r}+json.dumps({{'root':str(resolved)}},sort_keys=True))
  break
 except (OSError,RuntimeError) as exc:
  errors.append(str(item)+':'+str(exc))
else:
 raise RuntimeError('no authorized writable guest app root: '+'; '.join(errors))
""".strip()


@dataclass(frozen=True)
class GuestFixture:
    state: dict[str, Any]
    geometry: UiGeometry
    readiness: dict[str, Any]


@dataclass(frozen=True)
class SettledFixture:
    state: dict[str, Any]
    acknowledgement: dict[str, Any]
    polls: tuple[dict[str, Any], ...]
    stable_probe_count: int


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _stdout(result: dict[str, Any]) -> str:
    value = result.get("output")
    if not isinstance(value, str):
        raise TransportError("guest command returned no stdout")
    return value


def _run_json(transport: HttpVmTransport, argv: list[str]) -> dict[str, Any]:
    output = _stdout(transport.execute_argv(argv))
    lines = [line for line in output.splitlines() if line.startswith(JSON_MARKER)]
    if len(lines) != 1:
        raise TransportError(f"guest JSON marker count was {len(lines)}: {output[-500:]!r}")
    value = json.loads(lines[0][len(JSON_MARKER) :])
    if not isinstance(value, dict):
        raise TransportError("guest JSON payload was not an object")
    return value


def resolve_guest_root(transport: HttpVmTransport) -> PurePosixPath:
    cached = getattr(transport, "_r1b_guest_root", None)
    if isinstance(cached, PurePosixPath):
        return cached
    value = _run_json(transport, ["python3", "-c", GUEST_ROOT_RESOLVER])
    raw = value.get("root")
    if not isinstance(raw, str):
        raise TransportError("guest root resolver returned no root")
    root = PurePosixPath(raw)
    if not root.is_absolute() or root.name != GUEST_ROOT_NAME or ".." in root.parts:
        raise TransportError(f"guest root resolver returned an unsafe path: {raw!r}")
    setattr(transport, "_r1b_guest_root", root)
    return root


def _guest_dir(transport: HttpVmTransport, fixture: Fixture) -> PurePosixPath:
    return resolve_guest_root(transport) / fixture.id


def _focus_setup_script(fixture: Fixture, root: PurePosixPath) -> str:
    path = root / str(fixture.params["file_name"])
    encoded = _b64(str(fixture.params["initial_text"]))
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}
path={shlex.quote(str(path))}
rm -rf "$root"
mkdir -p "$root"
printf '%s' {shlex.quote(encoded)} | base64 -d > "$path"
code_bin="$(command -v code || command -v codium)"
test -n "$code_bin"
nohup "$code_bin" --new-window --disable-extensions --skip-welcome \
  --disable-workspace-trust "$path" >"$root/vscode.log" 2>&1 </dev/null &
for _ in $(seq 1 80); do
  if wmctrl -lx 2>/dev/null | grep -Eqi 'code|codium'; then break; fi
  sleep 0.25
done
wmctrl -r :ACTIVE: -b add,maximized 2>/dev/null || true
""".strip()


def _scroll_html(fixture: Fixture, token: str) -> str:
    lines = "".join(
        f"<p>Development line {index:03d} — seed {fixture.parameter_seed}</p>"
        for index in range(1, int(fixture.params["document_lines"]) + 1)
    )
    initial_y = int(fixture.params["initial_y"])
    return f"""<!doctype html><meta charset="utf-8">
<title>Local development scroll document</title>
<style>body{{font:22px sans-serif;max-width:900px;margin:36px auto;line-height:1.7}}p{{margin:22px}}</style>
<h1>Local development scroll document</h1>{lines}
<script>
const report=()=>fetch('/event/{token}',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{scroll_y:Math.round(scrollY)}})}});
addEventListener('scroll',report,{{passive:true}});
addEventListener('load',()=>requestAnimationFrame(()=>{{scrollTo(0,{initial_y});requestAnimationFrame(report)}}));
</script>"""


def _scroll_server_source(
    fixture: Fixture, token: str, root: PurePosixPath
) -> str:
    html_b64 = _b64(_scroll_html(fixture, token))
    root_value = str(root)
    return f"""
import base64,json,os,tempfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
TOKEN={token!r}; ROOT={root_value!r}; HTML=base64.b64decode({html_b64!r})
STATE=os.path.join(ROOT,'scroll-state.json')
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  if self.path != '/document/'+TOKEN: self.send_error(404); return
  self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(HTML)
 def do_POST(self):
  if self.path != '/event/'+TOKEN: self.send_error(404); return
  try:
   n=int(self.headers.get('Content-Length','0')); value=json.loads(self.rfile.read(n)); y=int(value['scroll_y'])
   fd,tmp=tempfile.mkstemp(dir=ROOT); os.chmod(tmp,0o600)
   with os.fdopen(fd,'w') as f: json.dump({{'scroll_y':y}},f)
   os.replace(tmp,STATE); self.send_response(204); self.end_headers()
  except Exception: self.send_error(400)
ThreadingHTTPServer(('127.0.0.1',{int(fixture.params['port'])}),H).serve_forever()
""".strip()


def _scroll_setup_script(fixture: Fixture, root: PurePosixPath) -> str:
    token = f"dev-{fixture.parameter_seed}-{fixture.fixture_sha256[:16]}"
    server_b64 = _b64(_scroll_server_source(fixture, token, root))
    port = int(fixture.params["port"])
    url = f"http://127.0.0.1:{port}/document/{token}"
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}
rm -rf "$root"
mkdir -p "$root"
printf '%s' {shlex.quote(server_b64)} | base64 -d > "$root/server.py"
chmod 700 "$root/server.py"
nohup python3 "$root/server.py" >"$root/server.log" 2>&1 </dev/null &
browser="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
test -n "$browser"
for _ in $(seq 1 40); do
  if curl -fsS {shlex.quote(url)} >/dev/null; then break; fi
  sleep 0.25
done
nohup "$browser" --no-first-run --no-default-browser-check --disable-session-crashed-bubble \
  --disable-features=TranslateUI --start-maximized {shlex.quote(url)} \
  >"$root/chrome.log" 2>&1 </dev/null &
for _ in $(seq 1 80); do
  if wmctrl -l 2>/dev/null | grep -Fq 'Local development scroll document'; then break; fi
  sleep 0.25
done
wmctrl -a 'Local development scroll document'
wmctrl -r :ACTIVE: -b add,maximized 2>/dev/null || true
""".strip()


def _drag_setup_script(fixture: Fixture, root: PurePosixPath) -> str:
    encoded = _b64(str(fixture.params["content"]))
    source = str(fixture.params["source_name"])
    destination = str(fixture.params["destination_name"])
    decoy = str(fixture.params["decoy_name"])
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}
rm -rf "$root"
mkdir -p "$root/{destination}" "$root/{decoy}"
printf '%s' {shlex.quote(encoded)} | base64 -d > "$root/{source}"
gsettings set org.gnome.nautilus.preferences default-folder-viewer 'list-view' 2>/dev/null || true
nohup nautilus --new-window "file://$root" >"$root/files.log" 2>&1 </dev/null &
for _ in $(seq 1 80); do
  if wmctrl -lx 2>/dev/null | grep -qi nautilus; then break; fi
  sleep 0.25
done
wmctrl -r :ACTIVE: -b add,maximized 2>/dev/null || true
""".strip()


def setup_fixture(
    transport: HttpVmTransport,
    fixture: Fixture,
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.5,
) -> GuestFixture:
    started = time.monotonic()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "fixture_id": fixture.id,
        "guest_controller": "setup_command_pending",
        "required_identical_geometry_probes": READINESS_STABLE_PROBES,
        "phases": [],
        "polls": [],
        "last_error": None,
    }
    try:
        root = _guest_dir(transport, fixture)
    except (OSError, ValueError, TransportError, json.JSONDecodeError) as exc:
        evidence["last_error"] = f"{type(exc).__name__}: {exc}"
        raise AppReadinessError(
            fixture_id=fixture.id,
            failed_phase="guest_root_resolution",
            evidence=evidence,
        ) from exc
    evidence["phases"].append(
        {
            "phase": "guest_root_resolution",
            "status": "ok",
            "elapsed_s": round(time.monotonic() - started, 3),
            "root": str(root),
        }
    )
    if fixture.template == "vscode_focus_type":
        script = _focus_setup_script(fixture, root)
    elif fixture.template == "local_document_scroll":
        script = _scroll_setup_script(fixture, root)
    else:
        script = _drag_setup_script(fixture, root)
    try:
        transport.execute_argv(["bash", "-lc", script])
    except TransportError as exc:
        evidence["last_error"] = f"{type(exc).__name__}: {exc}"
        evidence["diagnostics"] = collect_readiness_diagnostics(transport, root)
        raise AppReadinessError(
            fixture_id=fixture.id,
            failed_phase="app_process_launch",
            evidence=evidence,
        ) from exc
    evidence["guest_controller"] = "accepted_setup_command"
    evidence["phases"].append(
        {
            "phase": "app_process_launch",
            "status": "ok",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    )
    deadline = started + timeout_s
    state_seen = False
    geometry_seen = False
    previous_geometry: UiGeometry | None = None
    identical_geometry = 0
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        poll: dict[str, Any] = {
            "index": len(evidence["polls"]) + 1,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        try:
            state = probe_fixture(transport, fixture)
            last_state = state
            poll["state"] = state
            if not state_seen:
                evidence["phases"].append(
                    {
                        "phase": "initial_state_probe",
                        "status": "ok",
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
            state_seen = True
            geometry = probe_geometry(transport, fixture)
            poll["geometry"] = asdict(geometry)
            if not geometry_seen:
                evidence["phases"].append(
                    {
                        "phase": "app_window_geometry",
                        "status": "ok",
                        "elapsed_s": round(time.monotonic() - started, 3),
                    }
                )
            geometry_seen = True
            if geometry == previous_geometry:
                identical_geometry += 1
            else:
                previous_geometry = geometry
                identical_geometry = 1
            poll["identical_geometry_probe_count"] = identical_geometry
            evidence["polls"].append(poll)
            if identical_geometry >= READINESS_STABLE_PROBES:
                evidence["phases"].append(
                    {
                        "phase": "stable_geometry",
                        "status": "ok",
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "identical_probe_count": identical_geometry,
                    }
                )
                evidence["stable_geometry"] = asdict(geometry)
                evidence["stable_geometry_probe_count"] = identical_geometry
                return GuestFixture(state, geometry, evidence)
        except (OSError, KeyError, ValueError, TransportError, json.JSONDecodeError) as exc:
            evidence["last_error"] = f"{type(exc).__name__}: {exc}"
            poll["error"] = evidence["last_error"]
            evidence["polls"].append(poll)
            previous_geometry = None
            identical_geometry = 0
        time.sleep(poll_interval_s)
    evidence["last_state"] = last_state
    evidence["diagnostics"] = collect_readiness_diagnostics(transport, root)
    failed_phase = (
        "initial_state_probe"
        if not state_seen
        else "app_window_geometry"
        if not geometry_seen
        else "stable_geometry"
    )
    raise AppReadinessError(
        fixture_id=fixture.id,
        failed_phase=failed_phase,
        evidence=evidence,
    )


def wait_for_action_settle(
    transport: HttpVmTransport,
    fixture: Fixture,
    initial_state: dict[str, Any],
    *,
    phase: str,
    timeout_s: float = 15.0,
    poll_interval_s: float = 0.25,
    stable_probe_count: int = SETTLE_STABLE_PROBES,
) -> SettledFixture:
    """Require a causal state change, then fresh identical state probes.

    The acknowledgement observation is deliberately not counted as one of the
    stable probes.  This prevents a transient first change from being handed to
    the hidden oracle and makes timeout fail closed instead of returning the
    last (possibly stale) state.
    """
    if stable_probe_count < 3:
        raise ValueError("action settle requires at least three identical probes")
    started = time.monotonic()
    deadline = started + timeout_s
    polls: list[dict[str, Any]] = []
    acknowledgement: dict[str, Any] | None = None
    stable_state: dict[str, Any] | None = None
    identical = 0
    last_error: str | None = None
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        poll: dict[str, Any] = {
            "index": len(polls) + 1,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        try:
            current = probe_fixture(transport, fixture)
            last_state = current
            poll["state"] = current
            if acknowledgement is None:
                if current != initial_state:
                    acknowledgement = {
                        "kind": "hidden_state_changed",
                        "poll_index": poll["index"],
                        "elapsed_s": poll["elapsed_s"],
                        "state": current,
                    }
                    poll["action_acknowledged"] = True
                    # Acknowledgement is never also a stability observation.
                    stable_state = None
                    identical = 0
            else:
                if current == stable_state:
                    identical += 1
                else:
                    stable_state = current
                    identical = 1
                poll["identical_post_ack_probe_count"] = identical
        except (OSError, KeyError, ValueError, TransportError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            poll["error"] = last_error
            identical = 0
            stable_state = None
        polls.append(poll)
        if acknowledgement is not None and identical >= stable_probe_count:
            if stable_state is None:  # defensive: impossible by construction
                raise AssertionError("settle stability count has no state")
            return SettledFixture(
                state=stable_state,
                acknowledgement=acknowledgement,
                polls=tuple(polls),
                stable_probe_count=identical,
            )
        time.sleep(poll_interval_s)
    raise AppSettleTimeout(
        fixture_id=fixture.id,
        phase=phase,
        evidence={
            "schema_version": 1,
            "phase": phase,
            "acknowledged": acknowledgement is not None,
            "acknowledgement": acknowledgement,
            "required_identical_post_ack_probes": stable_probe_count,
            "last_identical_post_ack_probe_count": identical,
            "initial_state": initial_state,
            "last_state": last_state,
            "last_error": last_error,
            "polls": polls,
        },
    )


def probe_fixture(transport: HttpVmTransport, fixture: Fixture) -> dict[str, Any]:
    root = _guest_dir(transport, fixture)
    if fixture.template == "vscode_focus_type":
        path = root / str(fixture.params["file_name"])
        code = (
            "import base64,hashlib,json,pathlib;"
            f"p=pathlib.Path({str(path)!r});b=p.read_bytes();"
            f"v={base_state(fixture)!r};"
            "v.update({'application':'vscode','file_name':p.name,'content_b64':base64.b64encode(b).decode('ascii'),'content_sha256':hashlib.sha256(b).hexdigest()});"
            f"print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))"
        )
        return _run_json(transport, ["python3", "-c", code])
    if fixture.template == "local_document_scroll":
        path = root / "scroll-state.json"
        code = (
            "import json,pathlib;"
            f"p=pathlib.Path({str(path)!r});raw=json.loads(p.read_text());"
            f"v={base_state(fixture)!r};"
            "v.update({'application':'chrome','document_kind':'guest_local_development_document','scroll_y':int(raw['scroll_y'])});"
            f"print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))"
        )
        return _run_json(transport, ["python3", "-c", code])
    source = root / str(fixture.params["source_name"])
    destination = root / str(fixture.params["destination_name"]) / str(fixture.params["source_name"])
    decoy = root / str(fixture.params["decoy_name"]) / str(fixture.params["source_name"])
    code = (
        "import hashlib,json,pathlib;"
        f"s=pathlib.Path({str(source)!r});d=pathlib.Path({str(destination)!r});x=pathlib.Path({str(decoy)!r});"
        "h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None;"
        f"v={base_state(fixture)!r};"
        "v.update({'application':'files','drag_backend':'filesystem','source_exists':s.is_file(),'destination_sha256':h(d),'decoy_sha256':h(x)});"
        f"print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))"
    )
    return _run_json(transport, ["python3", "-c", code])


def probe_geometry(transport: HttpVmTransport, fixture: Fixture) -> UiGeometry:
    width, height = transport.screen_size()
    center = (width // 2, height // 2)
    if fixture.template != "files_drag":
        return UiGeometry(editor=center, scroll_surface=center)
    names = [
        str(fixture.params["source_name"]),
        str(fixture.params["destination_name"]),
        str(fixture.params["decoy_name"]),
    ]
    code = f"""
import json,pyatspi
wanted=set({names!r}); found={{}}
def walk(node,depth=0):
 if depth>12 or len(found)==len(wanted): return
 try:
  name=node.name
  if name in wanted:
   e=node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
   if e.width>4 and e.height>4: found[name]=[int(e.x+e.width//2),int(e.y+e.height//2)]
  for child in node: walk(child,depth+1)
 except Exception: pass
walk(pyatspi.Registry.getDesktop(0))
print({JSON_MARKER!r}+json.dumps({{'points':found}},sort_keys=True))
""".strip()
    value = _run_json(transport, ["/usr/bin/python3", "-c", code])
    points = value.get("points")
    if not isinstance(points, dict) or any(name not in points for name in names):
        raise TransportError(f"Files accessibility geometry incomplete: {points}")
    parsed = {name: tuple(int(v) for v in points[name]) for name in names}
    return UiGeometry(
        editor=center,
        scroll_surface=center,
        drag_source=parsed[names[0]],
        drag_destination=parsed[names[1]],
        drag_decoy=parsed[names[2]],
    )


def collect_readiness_diagnostics(
    transport: HttpVmTransport, root: PurePosixPath
) -> dict[str, Any]:
    """Collect bounded guest logs/window/process evidence without masking failure."""
    code = f"""
import json,pathlib,subprocess
r=pathlib.Path({str(root)!r}); logs={{}}
for p in r.glob('*.log'):
 try: logs[p.name]=p.read_text(errors='replace')[-12000:]
 except OSError as exc: logs[p.name]={{'read_error':str(exc)}}
def run(cmd):
 try:
  x=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
  return {{'rc':x.returncode,'stdout':x.stdout[-12000:],'stderr':x.stderr[-4000:]}}
 except Exception as exc: return {{'error':type(exc).__name__+': '+str(exc)}}
v={{'logs':logs,'processes':run(['ps','-eo','pid,stat,comm,args']),'windows':run(['wmctrl','-lGx']),'active_window':run(['xprop','-root','_NET_ACTIVE_WINDOW'])}}
print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))
""".strip()
    try:
        return _run_json(transport, ["python3", "-c", code])
    except Exception as exc:  # diagnostics preserve the primary failure
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def collect_fixture_diagnostics(
    transport: HttpVmTransport, fixture: Fixture
) -> dict[str, Any]:
    return collect_readiness_diagnostics(transport, _guest_dir(transport, fixture))


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_QCOW",
    "DEFAULT_QEMU",
    "READY_SNAPSHOT",
    "KvmFixtureSession",
    "sha256_file",
    "GuestFixture",
    "SettledFixture",
    "AppReadinessError",
    "AppSettleTimeout",
    "READINESS_STABLE_PROBES",
    "SETTLE_STABLE_PROBES",
    "GUEST_ROOT_NAME",
    "GUEST_ROOT_RESOLVER",
    "resolve_guest_root",
    "setup_fixture",
    "probe_fixture",
    "probe_geometry",
    "wait_for_action_settle",
    "collect_readiness_diagnostics",
    "collect_fixture_diagnostics",
]
