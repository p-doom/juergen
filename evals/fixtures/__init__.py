"""Task fixtures: a task that needs Chrome brings its own Chrome.

pixeldesk is grammar-free *and* task-free: it owns isolation and port locking,
reset attestation, a process-**group** reaper, the guest JSON-marker protocol
(`GuestScript`) and `reset_to_checkpoint`. It deliberately does not know what a
checkbox, a spreadsheet cell or a Nautilus window is. Those are task knowledge, so
they live here, next to tasks and oracles.

The seam is `Preparer` (`evals/tasks.py`): a fixture-backed task registers a preparer
that sets its fixture up in `prepare` and reads it back in `probe`. Nothing about
Chrome, LibreOffice or Nautilus reaches the session API.

  * `cdp.py`    — a dependency-free CDP client (one `Runtime.evaluate` over a raw
                  websocket), because reading a page's DOM should not require a
                  browser-automation stack.
  * `web.py`    — the four browser fixtures (click / focus_type / drag / scroll) and
                  the **host** HTTP server that serves them and receives their state.
  * `chrome.py` — Chrome launch under one absolute deadline, plus browser diagnostics.
  * `apps.py`   — Writer / Calc / Files fixtures and an **in-guest** Chrome fixture
                  that serves its own page on loopback.

Two conventions worth keeping straight: `web.py` fixtures are loaded from the guest at
`10.0.2.2` (qemu user-mode networking exposes the host there, so no forward is needed
inbound), while `apps.py`'s Chrome fixture serves itself on guest loopback and needs no
host round-trip at all.
"""

from evals.fixtures.apps import (
    APPS,
    AppFixture,
    AppFixtureError,
    probe_app_state,
    resolve_guest_root,
    setup_app_fixture,
)
from evals.fixtures.cdp import CdpError, cdp_evaluate, find_page_target
from evals.fixtures.chrome import (
    ChromeFixtureError,
    ChromeLaunch,
    capture_browser_diagnostics,
    capture_chrome_log,
)
from evals.fixtures.chrome import launch as launch_chrome_fixture
from evals.fixtures.web import (
    TEMPLATES,
    FixtureServerError,
    FixtureStateStore,
    WebFixture,
    WebFixtureServer,
    render_fixture_html,
)

__all__ = [
    "APPS",
    "TEMPLATES",
    "AppFixture",
    "AppFixtureError",
    "CdpError",
    "ChromeFixtureError",
    "ChromeLaunch",
    "FixtureServerError",
    "FixtureStateStore",
    "WebFixture",
    "WebFixtureServer",
    "capture_browser_diagnostics",
    "capture_chrome_log",
    "cdp_evaluate",
    "find_page_target",
    "launch_chrome_fixture",
    "probe_app_state",
    "render_fixture_html",
    "resolve_guest_root",
    "setup_app_fixture",
]
