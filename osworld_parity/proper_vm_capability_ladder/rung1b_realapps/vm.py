from __future__ import annotations

import base64
import json
import shlex
import time
from dataclasses import dataclass
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
    return f"""
import base64,json,os,tempfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
TOKEN={token!r}; ROOT={root!r}; HTML=base64.b64decode({html_b64!r})
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


def setup_fixture(transport: HttpVmTransport, fixture: Fixture) -> GuestFixture:
    root = _guest_dir(transport, fixture)
    if fixture.template == "vscode_focus_type":
        script = _focus_setup_script(fixture, root)
    elif fixture.template == "local_document_scroll":
        script = _scroll_setup_script(fixture, root)
    else:
        script = _drag_setup_script(fixture, root)
    transport.execute_argv(["bash", "-lc", script])
    deadline = time.monotonic() + 60.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            state = probe_fixture(transport, fixture)
            geometry = probe_geometry(transport, fixture)
            return GuestFixture(state, geometry)
        except (OSError, KeyError, ValueError, TransportError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise TransportError(f"real-application setup did not become ready: {last_error}")


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


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_QCOW",
    "DEFAULT_QEMU",
    "READY_SNAPSHOT",
    "KvmFixtureSession",
    "sha256_file",
    "GuestFixture",
    "GUEST_ROOT_NAME",
    "GUEST_ROOT_RESOLVER",
    "resolve_guest_root",
    "setup_fixture",
    "probe_fixture",
    "probe_geometry",
]
