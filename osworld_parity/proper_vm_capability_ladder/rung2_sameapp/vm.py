from __future__ import annotations

import base64
import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from ..rung1.transport import HttpVmTransport, TransportError
from ..rung1.vm import KvmFixtureSession
from .fixtures import Fixture
from .oracle import reset_signature


GUEST_ROOT_NAME = ".r2_sameapp"
JSON_MARKER = "RUNG2_JSON="


class AppReadinessError(TransportError):
    def __init__(self, *, fixture_id: str, failed_phase: str, evidence: dict[str, Any]):
        self.fixture_id = fixture_id
        self.failed_phase = failed_phase
        self.evidence = evidence
        super().__init__(
            f"{fixture_id}: readiness failed at {failed_phase}; "
            f"last_error={evidence.get('last_error')!r}"
        )


@dataclass(frozen=True)
class GuestFixture:
    state: dict[str, Any]
    geometry: dict[str, tuple[int, int]]
    reset_signature: str
    readiness: dict[str, Any]


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
    cached = getattr(transport, "_r2_guest_root", None)
    if isinstance(cached, PurePosixPath):
        return cached
    code = f"""
import json,os,pathlib,tempfile
name={GUEST_ROOT_NAME!r}; candidates=[os.environ.get('XDG_RUNTIME_DIR'),os.environ.get('HOME'),tempfile.gettempdir()]
errors=[]
for raw in candidates:
 try:
  if not raw: continue
  base=pathlib.Path(raw).resolve(strict=True)
  if not base.is_dir() or not os.access(base,os.R_OK|os.W_OK|os.X_OK): continue
  root=base/name; root.mkdir(mode=0o700,exist_ok=True)
  root=root.resolve(strict=True)
  if root.parent != base or root.name != name or root.stat().st_uid != os.geteuid(): continue
  os.chmod(root,0o700)
  print({JSON_MARKER!r}+json.dumps({{'root':str(root)}},sort_keys=True)); break
 except Exception as exc: errors.append(str(exc))
else: raise RuntimeError('no private writable guest root: '+'; '.join(errors))
""".strip()
    raw = _run_json(transport, ["python3", "-c", code]).get("root")
    if not isinstance(raw, str):
        raise TransportError("guest root resolver returned no root")
    root = PurePosixPath(raw)
    if not root.is_absolute() or root.name != GUEST_ROOT_NAME or ".." in root.parts:
        raise TransportError(f"unsafe guest root: {raw!r}")
    setattr(transport, "_r2_guest_root", root)
    return root


def _fixture_root(transport: HttpVmTransport, fixture: Fixture) -> PurePosixPath:
    return resolve_guest_root(transport) / fixture.id


def _writer_script(fixture: Fixture, root: PurePosixPath) -> str:
    source = root / "initial.txt"
    target = root / str(fixture.params["file_name"])
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}; rm -rf "$root"; mkdir -p "$root"
printf '%s' {shlex.quote(_b64(str(fixture.params['initial_text'])))} | base64 -d > {shlex.quote(str(source))}
timeout 45 libreoffice --headless --convert-to odt --outdir "$root" {shlex.quote(str(source))} >"$root/convert.log" 2>&1
mv "$root/initial.odt" {shlex.quote(str(target))}
nohup libreoffice --writer {shlex.quote(str(target))} >"$root/writer.log" 2>&1 </dev/null &
for _ in $(seq 1 120); do wmctrl -l 2>/dev/null | grep -Fq {shlex.quote(str(fixture.params['file_name']))} && break; sleep 0.25; done
wmctrl -a {shlex.quote(str(fixture.params['file_name']))}
wmctrl -r {shlex.quote(str(fixture.params['file_name']))} -b add,maximized
sleep 0.5
""".strip()


def _calc_script(fixture: Fixture, root: PurePosixPath) -> str:
    csv = root / "initial.csv"
    target = root / str(fixture.params["file_name"])
    cell = str(fixture.params["cell"])
    column = ord(cell[0].upper()) - ord("A")
    row = int(cell[1:]) - 1
    rows = [[""] * (column + 1) for _ in range(row + 1)]
    rows[row][column] = str(fixture.params["initial_value"])
    csv_text = "\n".join(",".join(values) for values in rows) + "\n"
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}; rm -rf "$root"; mkdir -p "$root"
printf '%s' {shlex.quote(_b64(csv_text))} | base64 -d > {shlex.quote(str(csv))}
timeout 45 libreoffice --headless --convert-to ods --outdir "$root" {shlex.quote(str(csv))} >"$root/convert.log" 2>&1
mv "$root/initial.ods" {shlex.quote(str(target))}
nohup libreoffice --calc {shlex.quote(str(target))} >"$root/calc.log" 2>&1 </dev/null &
for _ in $(seq 1 120); do wmctrl -l 2>/dev/null | grep -Fq {shlex.quote(str(fixture.params['file_name']))} && break; sleep 0.25; done
win="$(wmctrl -l | awk -v title={shlex.quote(str(fixture.params['file_name']))} 'index($0,title){{print $1; exit}}')"
test -n "$win"
# The clean snapshot sometimes restores Calc's last geometry as a 16-pixel
# sliver.  Address the resolved X11 window, wait past LibreOffice's own late
# geometry restore, then normalize and verify the mapped client.
sleep 1
wmctrl -ir "$win" -b remove,shaded,hidden,maximized_vert,maximized_horz
wmctrl -ir "$win" -e 0,50,27,1316,741
wmctrl -ir "$win" -b add,maximized_vert,maximized_horz
wmctrl -ia "$win"
for _ in $(seq 1 40); do
  read -r _ _ x y w h _ < <(wmctrl -lG | awk -v id="$win" '$1==id{{print; exit}}')
  test "${{w:-0}}" -gt 1000 && test "${{h:-0}}" -gt 600 && break
  sleep 0.25
done
test "${{w:-0}}" -gt 1000 && test "${{h:-0}}" -gt 600
sleep 0.75
""".strip()


def _files_script(fixture: Fixture, root: PurePosixPath) -> str:
    p = fixture.params
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}; rm -rf "$root"; mkdir -p "$root/{p['destination_name']}" "$root/{p['decoy_name']}"
printf '%s' {shlex.quote(_b64(str(p['content'])))} | base64 -d > "$root/{p['source_name']}"
gsettings set org.gnome.nautilus.preferences default-folder-viewer 'list-view' 2>/dev/null || true
nohup nautilus --new-window "file://$root" >"$root/.files.log" 2>&1 </dev/null &
for _ in $(seq 1 120); do wmctrl -lx 2>/dev/null | grep -Fqi {shlex.quote(fixture.id)} && break; sleep 0.25; done
win="$(wmctrl -lx | awk -v title={shlex.quote(fixture.id)} 'tolower($0) ~ /nautilus/ && index($0,title){{print $1; exit}}')"
test -n "$win"
wmctrl -ir "$win" -b remove,shaded,hidden,maximized_vert,maximized_horz
wmctrl -ir "$win" -e 0,100,100,900,600
wmctrl -ia "$win"
sleep 0.75
""".strip()


def _chrome_html(fixture: Fixture) -> str:
    p = fixture.params
    return f"""<!doctype html><meta charset="utf-8"><title>Same-app settings</title>
<style>body{{font:22px sans-serif;margin:0}}nav{{position:sticky;top:0;background:white;padding:20px}}button{{margin:8px;padding:14px}}main{{height:1700px;padding:40px}}#settings{{margin-top:900px}}</style>
<nav><button id="nav">{p['section']}</button><button id="decoy_nav">appearance</button></nav>
<main><h1>Local deterministic Chrome settings</h1><div id="settings"><label><input id="toggle" type="checkbox">{p['setting']}</label><br><label><input id="decoy_toggle" type="checkbox">Unrelated setting</label></div></main>
<script>
const send=()=>fetch('/event',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{
 ready:true,section:window.section||'root',scroll_y:Math.round(scrollY),setting_enabled:toggle.checked,
 geometry:Object.fromEntries(['nav','decoy_nav','toggle','decoy_toggle'].map(id=>{{let e=document.getElementById(id),r=e.getBoundingClientRect(),top=Math.max(0,outerHeight-innerHeight);return [id,[Math.round(screenX+r.left+r.width/2),Math.round(screenY+top+r.top+r.height/2)]]}}).concat([['scroll_surface',[Math.round(screenX+innerWidth/2),Math.round(screenY+(outerHeight-innerHeight)+innerHeight/2)]]]))
}})}});
nav.onclick=()=>{{window.section={json.dumps(str(p['section']))};send()}};decoy_nav.onclick=()=>{{window.section='appearance';send()}};
toggle.onchange=send;decoy_toggle.onchange=send;addEventListener('scroll',send,{{passive:true}});addEventListener('load',()=>requestAnimationFrame(send));
</script>"""


def _chrome_script(fixture: Fixture, root: PurePosixPath) -> str:
    port = int(fixture.params["port"])
    source = f"""
import base64,json,os,tempfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT={str(root)!r}; HTML=base64.b64decode({_b64(_chrome_html(fixture))!r}); STATE=os.path.join(ROOT,'state.json')
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  if self.path!='/settings': self.send_error(404); return
  self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(HTML)
 def do_POST(self):
  if self.path!='/event': self.send_error(404); return
  try:
   n=int(self.headers.get('Content-Length','0')); value=json.loads(self.rfile.read(n)); fd,tmp=tempfile.mkstemp(dir=ROOT)
   with os.fdopen(fd,'w') as f: json.dump(value,f,sort_keys=True)
   os.replace(tmp,STATE); self.send_response(204); self.end_headers()
  except Exception: self.send_error(400)
ThreadingHTTPServer(('127.0.0.1',{port}),H).serve_forever()
""".strip()
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}; rm -rf "$root"; mkdir -p "$root"
printf '%s' {shlex.quote(_b64(source))} | base64 -d > "$root/server.py"
nohup python3 "$root/server.py" >"$root/server.log" 2>&1 </dev/null &
for _ in $(seq 1 40); do curl -fsS http://127.0.0.1:{port}/settings >/dev/null && break; sleep 0.25; done
browser="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"; test -n "$browser"
nohup "$browser" --no-first-run --no-default-browser-check --disable-session-crashed-bubble --disable-features=TranslateUI --start-maximized http://127.0.0.1:{port}/settings >"$root/chrome.log" 2>&1 </dev/null &
""".strip()


def setup_fixture(transport: HttpVmTransport, fixture: Fixture, *, timeout_s: float = 45.0) -> GuestFixture:
    root = _fixture_root(transport, fixture)
    script = {
        "writer": _writer_script,
        "calc": _calc_script,
        "files": _files_script,
        "chrome": _chrome_script,
    }[fixture.app](fixture, root)
    started = time.monotonic()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "guest_controller": "setup_command_pending",
        "fixture_id": fixture.id,
        "phases": [],
        "last_error": None,
    }
    try:
        transport.execute_argv(["bash", "-lc", script])
        evidence["guest_controller"] = "accepted_setup_command"
        evidence["phases"].append(
            {
                "phase": "app_process_launch",
                "elapsed_s": round(time.monotonic() - started, 3),
                "status": "ok",
            }
        )
    except TransportError as exc:
        evidence["last_error"] = f"{type(exc).__name__}: {exc}"
        evidence["diagnostics"] = collect_readiness_diagnostics(transport, root)
        raise AppReadinessError(
            fixture_id=fixture.id,
            failed_phase="app_process_launch",
            evidence=evidence,
        ) from exc
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        try:
            state = probe_state(transport, fixture)
            evidence["phases"].append({"phase": "state_probe", "elapsed_s": round(time.monotonic() - started, 3), "status": "ok"})
            geometry = probe_geometry(transport, fixture, state)
            evidence["phases"].append({"phase": "app_window_geometry", "elapsed_s": round(time.monotonic() - started, 3), "status": "ok"})
            signature = reset_signature(fixture, state)
            evidence["phases"].append({"phase": "initial_oracle", "elapsed_s": round(time.monotonic() - started, 3), "status": "ok"})
            return GuestFixture(state, geometry, signature, evidence)
        except (KeyError, OSError, ValueError, json.JSONDecodeError, TransportError) as exc:
            evidence["last_error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
    evidence["diagnostics"] = collect_readiness_diagnostics(transport, root)
    failed_phase = "browser_document_ready" if fixture.app == "chrome" else "app_window_geometry"
    raise AppReadinessError(fixture_id=fixture.id, failed_phase=failed_phase, evidence=evidence)


def reset_and_setup(session: KvmFixtureSession, fixture: Fixture) -> tuple[HttpVmTransport, GuestFixture]:
    transport = session.reset_to_ready()
    return transport, setup_fixture(transport, fixture)


def probe_state(transport: HttpVmTransport, fixture: Fixture) -> dict[str, Any]:
    root = _fixture_root(transport, fixture)
    base = {
        "schema_version": 1,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "app": fixture.app,
    }
    if fixture.app == "writer":
        path = root / str(fixture.params["file_name"])
        code = f"""
import hashlib,json,zipfile,xml.etree.ElementTree as E
p={str(path)!r}; ns={{'text':'urn:oasis:names:tc:opendocument:xmlns:text:1.0','style':'urn:oasis:names:tc:opendocument:xmlns:style:1.0','fo':'urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0'}}
with zipfile.ZipFile(p) as z: content=z.read('content.xml'); root=E.fromstring(content)
text=''.join(''.join(n.itertext()) for n in root.findall('.//text:p',ns)); bold=b'font-weight="bold"' in content
v={base!r}; v.update({{'text':text,'content_sha256':hashlib.sha256(text.encode()).hexdigest(),'bold':bold,'saved':text!={str(fixture.params['initial_text'])!r}}}); print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))
""".strip()
    elif fixture.app == "calc":
        path = root / str(fixture.params["file_name"])
        code = f"""
import json,zipfile,xml.etree.ElementTree as E
p={str(path)!r}; target={str(fixture.params['cell'])!r}; col=ord(target[0])-65; row=int(target[1:])-1
ns={{'t':'urn:oasis:names:tc:opendocument:xmlns:table:1.0','o':'urn:oasis:names:tc:opendocument:xmlns:office:1.0','x':'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}}
with zipfile.ZipFile(p) as z: root=E.fromstring(z.read('content.xml'))
logical_row=0; c=None
for row_node in root.findall('.//t:table-row',ns):
 row_repeat=int(row_node.get('{{'+ns['t']+'}}number-rows-repeated','1'))
 if logical_row <= row < logical_row+row_repeat:
  logical_col=0
  for cell_node in row_node.findall('t:table-cell',ns):
   col_repeat=int(cell_node.get('{{'+ns['t']+'}}number-columns-repeated','1'))
   if logical_col <= col < logical_col+col_repeat: c=cell_node; break
   logical_col+=col_repeat
  break
 logical_row+=row_repeat
assert c is not None, 'target cell missing'
formula=c.get('{{'+ns['t']+'}}formula'); value=c.get('{{'+ns['o']+'}}value') or ''.join(c.itertext()); v={base!r}; v.update({{'cell':target,'formula':formula,'display_value':value,'saved':formula is not None}}); print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))
""".strip()
    elif fixture.app == "files":
        p = fixture.params
        code = f"""
import hashlib,json,pathlib
r=pathlib.Path({str(root)!r}); source=r/{str(p['source_name'])!r}; expected=r/{str(p['destination_name'])!r}; decoy=r/{str(p['decoy_name'])!r}
found=[]
for name,dest in [({str(p['destination_name'])!r},expected),({str(p['decoy_name'])!r},decoy)]:
 for f in dest.iterdir():
  if f.is_file(): found.append((name,f))
item=found[0] if len(found)==1 else (None,source); data=item[1].read_bytes() if item[1].is_file() else b''; v={base!r}; v.update({{'source_exists':source.is_file(),'destination':item[0],'final_name':item[1].name,'content_sha256':hashlib.sha256(data).hexdigest(),'saved':not source.exists()}}); print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))
""".strip()
    else:
        path = root / "state.json"
        code = f"""
import json,pathlib
p=pathlib.Path({str(path)!r}); raw=json.loads(p.read_text());
assert raw.get('ready') is True
v={base!r}; v.update({{'section':raw['section'],'scroll_y':int(raw['scroll_y']),'setting_enabled':bool(raw['setting_enabled']),'saved':bool(raw['setting_enabled'])}}); v['_geometry']=raw['geometry']; print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))
""".strip()
    return _run_json(transport, ["python3", "-c", code])


def probe_geometry(
    transport: HttpVmTransport, fixture: Fixture, state: dict[str, Any]
) -> dict[str, tuple[int, int]]:
    if fixture.app == "chrome":
        raw = state.pop("_geometry", None)
        if not isinstance(raw, dict):
            raise TransportError("Chrome document did not report geometry")
        geometry = {name: (int(point[0]), int(point[1])) for name, point in raw.items()}
        required = {"nav", "decoy_nav", "toggle", "decoy_toggle", "scroll_surface"}
    else:
        if fixture.app == "writer":
            code = f"""
import json,subprocess,time
value=subprocess.run(['wmctrl','-lG'],capture_output=True,text=True,check=True)
matches=[]
for line in value.stdout.splitlines():
 parts=line.split(None,7)
 if len(parts)==8 and 'libreoffice writer' in parts[7].lower():
  x,y,w,h=map(int,parts[2:6]); matches.append((w*h,parts[0],x,y,w,h,parts[7]))
assert matches, 'Writer window missing'
_,window_id,x,y,w,h,title=max(matches)
assert w>500 and h>400 and x>=0 and y>=0, 'Writer window not visibly mapped'
subprocess.run(['wmctrl','-ia',window_id],check=True); time.sleep(0.5)
active=subprocess.run(['xprop','-root','_NET_ACTIVE_WINDOW'],capture_output=True,text=True,check=True).stdout.lower()
assert window_id.lower().lstrip('0x').lstrip('0') in active.replace('0x','').lstrip('0'), 'Writer window not active'
point=[x+w//2,y+int(h*0.58)]
print({JSON_MARKER!r}+json.dumps({{'geometry':{{'editor':point}},'window':{{'x':x,'y':y,'width':w,'height':h,'title':title}}}},sort_keys=True))
""".strip()
            raw = _run_json(transport, ["python3", "-c", code]).get("geometry")
            if not isinstance(raw, dict) or "editor" not in raw:
                raise TransportError("Writer window geometry incomplete")
            return {"editor": (int(raw["editor"][0]), int(raw["editor"][1]))}
        elif fixture.app == "calc":
            # Calc exposes a virtual million-row grid through AT-SPI.  A
            # recursive walk can wedge both the probe and soffice.  Select the
            # requested cell through Calc's visible Name Box instead; its
            # location is fixed relative to the verified, normalized window.
            code = f"""
import json,subprocess,time
value=subprocess.run(['wmctrl','-lG'],capture_output=True,text=True,check=True)
matches=[]
for line in value.stdout.splitlines():
 parts=line.split(None,7)
 if len(parts)==8 and 'libreoffice calc' in parts[7].lower():
  x,y,w,h=map(int,parts[2:6]); matches.append((w*h,parts[0],x,y,w,h,parts[7]))
assert matches, 'Calc window missing'
_,window_id,x,y,w,h,title=max(matches)
assert w>1000 and h>600 and x>=0 and y>=0, 'Calc window not visibly mapped'
subprocess.run(['wmctrl','-ia',window_id],check=True); time.sleep(0.35)
# Center of the Name Box in the pinned LibreOffice UI.  Typing the target cell
# there avoids any dependence on virtual spreadsheet accessibility children.
point=[x+55,y+84]
print({JSON_MARKER!r}+json.dumps({{'geometry':{{'cell':point}},'window':{{'x':x,'y':y,'width':w,'height':h,'title':title}}}},sort_keys=True))
""".strip()
            raw = _run_json(transport, ["python3", "-c", code]).get("geometry")
            if not isinstance(raw, dict) or "cell" not in raw:
                raise TransportError("Calc window geometry incomplete")
            return {"cell": (int(raw["cell"][0]), int(raw["cell"][1]))}
        else:
            # The pinned Nautilus list view sorts the two directories before
            # the source file.  The log is hidden, and the window geometry is
            # normalized during setup, so row centers are deterministic.
            code = f"""
import json,subprocess,time
value=subprocess.run(['wmctrl','-lGx'],capture_output=True,text=True,check=True)
matches=[]
for line in value.stdout.splitlines():
 parts=line.split(None,8)
 if len(parts)==9 and 'nautilus' in parts[6].lower() and {fixture.id!r} in parts[8]:
  x,y,w,h=map(int,parts[2:6]); matches.append((w*h,parts[0],x,y,w,h,parts[8]))
assert matches, 'Files window missing'
_,window_id,x,y,w,h,title=max(matches)
assert w>700 and h>450 and x>=0 and y>=0, 'Files window not visibly mapped'
subprocess.run(['wmctrl','-ia',window_id],check=True); time.sleep(0.35)
geometry={{'decoy':[x+250,y+15],'destination':[x+250,y+63],'source':[x+250,y+131],'moved':[x+250,y+15]}}
print({JSON_MARKER!r}+json.dumps({{'geometry':geometry,'window':{{'x':x,'y':y,'width':w,'height':h,'title':title}}}},sort_keys=True))
""".strip()
            raw = _run_json(transport, ["python3", "-c", code]).get("geometry")
            if not isinstance(raw, dict):
                raise TransportError("Files window geometry incomplete")
            geometry = {
                name: (int(point[0]), int(point[1]))
                for name, point in raw.items()
            }
            required = {"source", "destination", "decoy", "moved"}
    if not required.issubset(geometry):
        raise TransportError(f"{fixture.app} geometry incomplete: {geometry}")
    return geometry


def collect_readiness_diagnostics(
    transport: HttpVmTransport, root: PurePosixPath
) -> dict[str, Any]:
    code = f"""
import glob,json,pathlib,subprocess
r=pathlib.Path({str(root)!r}); logs={{}}
for p in r.glob('*.log'):
 try: logs[p.name]=p.read_text(errors='replace')[-4000:]
 except OSError: pass
def run(cmd):
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=5); return {{'rc':x.returncode,'stdout':x.stdout[-4000:],'stderr':x.stderr[-1000:]}}
v={{'logs':logs,'processes':run(['ps','-eo','pid,stat,comm,args']),'windows':run(['wmctrl','-lx']),'chrome_debug':run(['curl','-fsS','http://127.0.0.1:9222/json'])}}
print({JSON_MARKER!r}+json.dumps(v,sort_keys=True))
""".strip()
    try:
        return _run_json(transport, ["python3", "-c", code])
    except Exception as exc:  # diagnostics must preserve the primary phase failure
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def readiness_as_dict(fixture: GuestFixture) -> dict[str, Any]:
    value = asdict(fixture)
    value["geometry"] = {name: list(point) for name, point in fixture.geometry.items()}
    return value
