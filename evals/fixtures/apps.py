"""Desktop-app task fixtures: Writer, Calc, Files, and an in-guest Chrome.

These are *tasks*, so they live here and not in the VM layer: a session boots a
desktop, and a task that needs a spreadsheet with a known value in B7 brings the
spreadsheet.

Each fixture is one bash script driven through the guest, plus one read-only state
probe. Four properties are load-bearing and preserved:

  * **A private, per-fixture guest root**, resolved once by trying
    `XDG_RUNTIME_DIR`, then `HOME`, then the temp dir, and requiring the created
    directory to be mode 0700, owned by us and a direct child of the base. A shared
    `/tmp/fixture` would let one cell's leftovers decide another cell's outcome.
  * **Documents are built, not shipped.** Text and CSV go in as base64, then
    LibreOffice converts them headlessly to `.odt`/`.ods`. Committing binary office
    documents makes the initial state unreviewable, and `--convert-to` is the only
    way to get a file the same LibreOffice build will open without a recovery dialog.
  * **Window geometry is normalised and then verified.** The clean snapshot
    sometimes restores Calc's last geometry as a 16-pixel sliver; unmaximising,
    setting an explicit rectangle, re-maximising and then *polling until the mapped
    client is actually larger than 1000x600* is what makes the window usable. Setting
    geometry without verifying it silently yields a task the model cannot see.
  * **The in-guest Chrome fixture serves itself.** A local `ThreadingHTTPServer` on
    the loopback interface serves one settings page and accepts its state posts, so
    the task needs no host round-trip and no port forward — unlike `web.py`'s
    fixtures, which are reached from the guest at `10.0.2.2`.
"""

from __future__ import annotations

import base64
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

__all__ = [
    "APPS",
    "AppFixture",
    "AppFixtureError",
    "GUEST_ROOT_NAME",
    "JSON_MARKER",
    "probe_app_state",
    "resolve_guest_root",
    "setup_app_fixture",
]

JSON_MARKER = "FIXTURE_JSON="
GUEST_ROOT_NAME = "juergen_app_fixtures"
APPS = ("writer", "calc", "files", "chrome")


class AppFixtureError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppFixture:
    """One app fixture instance. `params` are the template's knobs."""

    id: str
    app: str
    instruction: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if self.app not in APPS:
            raise AppFixtureError(f"unknown app {self.app!r}")


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _stdout(result: dict[str, Any]) -> str:
    value = result.get("output")
    if not isinstance(value, str):
        raise AppFixtureError("guest command returned no stdout")
    return value


def _run_json(session: Any, argv: list[str]) -> dict[str, Any]:
    """One guest command, one JSON marker line. Ambiguity fails closed."""
    output = _stdout(session.execute_argv(argv))
    lines = [line for line in output.splitlines() if line.startswith(JSON_MARKER)]
    if len(lines) != 1:
        raise AppFixtureError(
            f"guest JSON marker count was {len(lines)}: {output[-500:]!r}"
        )
    value = json.loads(lines[0][len(JSON_MARKER) :])
    if not isinstance(value, dict):
        raise AppFixtureError("guest JSON payload was not an object")
    return value


def resolve_guest_root(session: Any) -> PurePosixPath:
    """A private, writable, 0700 guest directory owned by us.

    Cached on the session object: one resolution per session, and re-resolving would
    be a second `python3` start per fixture.
    """
    cached = getattr(session, "_juergen_fixture_root", None)
    if isinstance(cached, PurePosixPath):
        return cached
    code = f"""
import json,os,pathlib,tempfile
name={GUEST_ROOT_NAME!r}
candidates=[os.environ.get('XDG_RUNTIME_DIR'),os.environ.get('HOME'),tempfile.gettempdir()]
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
    raw = _run_json(session, ["python3", "-c", code]).get("root")
    if not isinstance(raw, str):
        raise AppFixtureError("guest root resolver returned no root")
    root = PurePosixPath(raw)
    if not root.is_absolute() or root.name != GUEST_ROOT_NAME or ".." in root.parts:
        raise AppFixtureError(f"unsafe guest root: {raw!r}")
    setattr(session, "_juergen_fixture_root", root)
    return root


def _fixture_root(session: Any, fixture: AppFixture) -> PurePosixPath:
    return resolve_guest_root(session) / fixture.id


def _writer_script(fixture: AppFixture, root: PurePosixPath) -> str:
    params = fixture.params
    source = root / "initial.txt"
    target = root / str(params["file_name"])
    name = shlex.quote(str(params["file_name"]))
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}; rm -rf "$root"; mkdir -p "$root"
printf '%s' {shlex.quote(_b64(str(params['initial_text'])))} | base64 -d > {shlex.quote(str(source))}
timeout 45 libreoffice --headless --convert-to odt --outdir "$root" {shlex.quote(str(source))} >"$root/convert.log" 2>&1
mv "$root/initial.odt" {shlex.quote(str(target))}
nohup libreoffice --writer {shlex.quote(str(target))} >"$root/writer.log" 2>&1 </dev/null &
for _ in $(seq 1 120); do wmctrl -l 2>/dev/null | grep -Fq {name} && break; sleep 0.25; done
wmctrl -a {name}
wmctrl -r {name} -b add,maximized
sleep 0.5
""".strip()


def _calc_script(fixture: AppFixture, root: PurePosixPath) -> str:
    params = fixture.params
    csv = root / "initial.csv"
    target = root / str(params["file_name"])
    cell = str(params["cell"])
    column = ord(cell[0].upper()) - ord("A")
    row = int(cell[1:]) - 1
    rows = [[""] * (column + 1) for _ in range(row + 1)]
    rows[row][column] = str(params["initial_value"])
    csv_text = "\n".join(",".join(values) for values in rows) + "\n"
    name = shlex.quote(str(params["file_name"]))
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}; rm -rf "$root"; mkdir -p "$root"
printf '%s' {shlex.quote(_b64(csv_text))} | base64 -d > {shlex.quote(str(csv))}
timeout 45 libreoffice --headless --convert-to ods --outdir "$root" {shlex.quote(str(csv))} >"$root/convert.log" 2>&1
mv "$root/initial.ods" {shlex.quote(str(target))}
nohup libreoffice --calc {shlex.quote(str(target))} >"$root/calc.log" 2>&1 </dev/null &
for _ in $(seq 1 120); do wmctrl -l 2>/dev/null | grep -Fq {name} && break; sleep 0.25; done
win="$(wmctrl -l | awk -v title={name} 'index($0,title){{print $1; exit}}')"
test -n "$win"
# The clean snapshot sometimes restores Calc's geometry as a 16-pixel sliver.
# Address the resolved X11 window, wait past LibreOffice's own late geometry
# restore, then normalize and VERIFY the mapped client — setting geometry without
# verifying it silently yields a window the model cannot read.
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


def _files_script(fixture: AppFixture, root: PurePosixPath) -> str:
    p = fixture.params
    return f"""
set -euo pipefail
root={shlex.quote(str(root))}
rm -rf "$root"; mkdir -p "$root/{p['destination_name']}" "$root/{p['decoy_name']}"
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


def _chrome_html(fixture: AppFixture) -> str:
    """A deterministic local settings page with a below-the-fold control.

    The scroll margin flips with direction: a down-scroll task hides the control
    below the fold, an up-scroll task starts below a control near the top. Without
    that, "scroll up" would be satisfiable by not scrolling at all.
    """
    p = fixture.params
    initial_scroll_y = int(p.get("initial_scroll_y", 0))
    settings_margin = 400 if p.get("scroll_direction") == "up" else 900
    return f"""<!doctype html><meta charset="utf-8"><title>Same-app settings</title>
<style>body{{font:22px sans-serif;margin:0}}nav{{position:sticky;top:0;background:white;padding:20px}}button{{margin:8px;padding:14px}}main{{height:1700px;padding:40px}}#settings{{margin-top:{settings_margin}px}}</style>
<nav><button id="nav">{p['section']}</button><button id="decoy_nav">appearance</button></nav>
<main><h1>Local deterministic Chrome settings</h1><div id="settings"><label><input id="toggle" type="checkbox">{p['setting']}</label><br><label><input id="decoy_toggle" type="checkbox">Unrelated setting</label></div></main>
<script>
const send=()=>fetch('/event',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{
 ready:true,section:window.section||'root',scroll_y:Math.round(scrollY),setting_enabled:toggle.checked,
 geometry:Object.fromEntries(['nav','decoy_nav','toggle','decoy_toggle'].map(id=>{{let e=document.getElementById(id),r=e.getBoundingClientRect(),top=Math.max(0,outerHeight-innerHeight);return [id,[Math.round(screenX+r.left+r.width/2),Math.round(screenY+top+r.top+r.height/2)]]}}).concat([['scroll_surface',[Math.round(screenX+innerWidth/2),Math.round(screenY+(outerHeight-innerHeight)+innerHeight/2)]]]))
}})}});
nav.onclick=()=>{{window.section={json.dumps(str(p['section']))};send()}};decoy_nav.onclick=()=>{{window.section='appearance';send()}};
toggle.onchange=send;decoy_toggle.onchange=send;addEventListener('scroll',send,{{passive:true}});
addEventListener('load',()=>requestAnimationFrame(()=>{{scrollTo(0,{initial_scroll_y});requestAnimationFrame(send)}}));
</script>"""


def _chrome_script(fixture: AppFixture, root: PurePosixPath) -> str:
    """Serve the page from inside the guest, on loopback.

    State lands in `state.json` via `mkstemp` + `os.replace`, so a probe can never
    read a half-written file — the probe and the page are concurrent by construction.
    """
    port = int(fixture.params["port"])
    source = f"""
import base64,json,os,tempfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT={str(root)!r}
HTML=base64.b64decode({_b64(_chrome_html(fixture))!r})
STATE=os.path.join(ROOT,'state.json')
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  if self.path!='/settings': self.send_error(404); return
  self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
  self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(HTML)
 def do_POST(self):
  if self.path!='/event': self.send_error(404); return
  try:
   n=int(self.headers.get('Content-Length','0')); value=json.loads(self.rfile.read(n))
   fd,tmp=tempfile.mkstemp(dir=ROOT)
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
browser="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
test -n "$browser"
nohup "$browser" --no-first-run --no-default-browser-check --disable-session-crashed-bubble \
 --disable-features=TranslateUI --start-maximized http://127.0.0.1:{port}/settings \
 >"$root/chrome.log" 2>&1 </dev/null &
""".strip()


_SCRIPTS: dict[str, Callable[[AppFixture, PurePosixPath], str]] = {
    "writer": _writer_script,
    "calc": _calc_script,
    "files": _files_script,
    "chrome": _chrome_script,
}


def setup_app_fixture(
    session: Any, fixture: AppFixture, *, timeout_s: float = 45.0
) -> dict[str, Any]:
    """Run one app fixture's setup script and return its evidence."""
    root = _fixture_root(session, fixture)
    started = time.monotonic()
    script = _SCRIPTS[fixture.app](fixture, root)
    try:
        result = session.execute_argv(["bash", "-lc", script])
        status = "ready"
        error: str | None = None
    except Exception as exc:  # noqa: BLE001 - reported, never silently swallowed
        result = {}
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    del timeout_s  # the guest scripts carry their own bounded waits
    return {
        "fixture_id": fixture.id,
        "app": fixture.app,
        "guest_root": str(root),
        "status": status,
        "error": error,
        "elapsed_s": round(time.monotonic() - started, 3),
        "setup_output": result.get("output") if isinstance(result, dict) else None,
    }


def probe_app_state(session: Any, fixture: AppFixture) -> dict[str, Any]:
    """Read-only realized state for one app fixture.

    Per app: Writer/Calc convert the live document back to text/CSV in a temp copy
    (never touching the file the model is editing), Files lists the tree, and Chrome
    reads the atomically-replaced `state.json`.
    """
    root = _fixture_root(session, fixture)
    params = json.dumps(fixture.params, sort_keys=True)
    code = f"""
import glob,json,os,pathlib,subprocess,tempfile
root=pathlib.Path({str(root)!r}); app={fixture.app!r}; params=json.loads({params!r})
def text(path):
 try: return pathlib.Path(path).read_text(encoding='utf-8')
 except (FileNotFoundError,UnicodeDecodeError,OSError): return None
value={{'schema_version':1,'fixture_id':{fixture.id!r},'app':app,'root':str(root)}}
if app in ('writer','calc'):
 target=root/str(params['file_name'])
 value['document_exists']=target.is_file()
 out=tempfile.mkdtemp(dir=str(root))
 fmt='txt' if app=='writer' else 'csv'
 try:
  subprocess.run(['libreoffice','--headless','--convert-to',fmt,'--outdir',out,str(target)],
                 capture_output=True,timeout=60,check=False)
  found=sorted(glob.glob(os.path.join(out,'*.'+fmt)))
  value['content']=text(found[0]) if found else None
 except Exception as exc: value['content']=None; value['probe_error']=str(exc)
elif app=='files':
 value['entries']=sorted(str(p.relative_to(root)) for p in root.rglob('*'))
 value['source_exists']=(root/str(params['source_name'])).is_file()
 value['destination_content']=text(root/str(params['destination_name'])/str(params['source_name']))
 value['decoy_content']=text(root/str(params['decoy_name'])/str(params['source_name']))
else:
 raw=text(root/'state.json')
 value['page_state']=json.loads(raw) if raw else None
print({JSON_MARKER!r}+json.dumps(value,ensure_ascii=False,sort_keys=True))
""".strip()
    return _run_json(session, ["python3", "-c", code])
