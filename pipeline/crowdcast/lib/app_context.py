"""Foreground-application state on the master tick axis.

The crowd-cast recorder emits, alongside the six input event types, a
``ContextChanged`` event carrying the bundle id / process name of the app that
just took focus::

    [0,         ['ContextChanged', ['com.apple.Safari']]]
    [224099997, ['KeyPress',       [59, 'KeyI']]]

``lib/events.iter_events`` deliberately skips it (it is not an input event), so
this module is the only consumer. Two properties make the join exact:

  * SAME CLOCK. ``stage_02_realign.write_corrected_keylog`` re-stamps EVERY
    keylog entry through ``realign_lib.keylog_to_video`` with no type dispatch,
    so app switches ride the identical splice map as the actions. Reading the
    ``keylog_path`` off the realigned clips manifest / stage-03 sample index
    (already repointed to the corrected keylog) puts them on the master clock.
  * NOT PART OF THE REALIGN MODEL. ``realign_lib.INPUT_TYPES`` excludes
    ``ContextChanged``, so an app switch can never suppress a pause detection --
    the recorder pauses on INPUT idleness, and the model must agree with the
    recorder. The consequence is handled here: ``keylog_to_video`` CLAMPS a
    timestamp inside a collapsed pause to the splice point, so a run of switches
    during a pause collapses onto one instant. LAST WINS at equal timestamps,
    which resolves to the app in focus when recording resumed.

The track is STATE, not events: it is forward-filled and deliberately NOT run
through ``lib/events.apply_label_policy``. A switch inside a dead zone is how you
know which app came back, so the label policy would delete exactly the
information that explains the blackout.

``UNCAPTURED`` is the recorder's privacy blackout (the app is on a
do-not-capture list) and ``UNKNOWN`` its could-not-resolve fallback; both are
"no app" for filtering purposes (``UNRESOLVED_APPS``). Empirically UNCAPTURED
spans are also (near-)black video, i.e. the frames the black filter already
drops -- so they cost a filtered dataset almost nothing.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.crowdcast.lib.events import iter_context

# Sentinel app ids: not real applications, never a filter target.
UNCAPTURED = "UNCAPTURED"
UNKNOWN = "UNKNOWN"
UNRESOLVED_APPS = frozenset({UNCAPTURED, UNKNOWN})

# The recorder reports macOS bundle ids, but the Windows/Linux agents report bare
# process names, so the same application arrives under two spellings. Canonical
# form is the bundle id (the majority spelling in the corpus).
_ALIASES = {
    "firefox": "org.mozilla.firefox",
    "chrome": "com.google.Chrome",
    "msedge": "com.microsoft.edgemac",
    "code": "com.microsoft.VSCode",
    "com.google.antigravity-ide": "com.google.antigravity",
    "vitis-ide": "com.xilinx.vitis",
    "zotero": "org.zotero.zotero",
    "spotify": "com.spotify.client",
    "explorer": "com.apple.finder",
    "notepad": "com.apple.TextEdit",
    "snippingtool": "com.apple.screenshot",
    "excel": "com.microsoft.Excel",
    "powerpnt": "com.microsoft.Powerpoint",
    "windowsterminal": "com.microsoft.windowsterminal",
    "illustrator": "com.adobe.Illustrator",
    "chimerax": "edu.ucsf.rbvi.ChimeraX",
    "crowd-cast-agent": "dev.crowd-cast.agent",
}

# Short, human-typable aliases for the CLI (--include-app firefox). Resolved
# against the canonical ids above; anything unrecognized is passed through, so a
# bundle id always works verbatim.
_FRIENDLY = {
    "safari": "com.apple.Safari",
    "firefox": "org.mozilla.firefox",
    "chrome": "com.google.Chrome",
    "arc": "company.thebrowser.Browser",
    "cursor": "com.todesktop.230313mzl4w4u92",
    "vscode": "com.microsoft.VSCode",
    "zed": "dev.zed.Zed",
    "ghostty": "com.mitchellh.ghostty",
    "terminal": "com.apple.Terminal",
    "alacritty": "Alacritty",
    "antigravity": "com.google.antigravity",
    "preview": "com.apple.Preview",
    "finder": "com.apple.finder",
    "zotero": "org.zotero.zotero",
    "obsidian": "md.obsidian",
    "notion": "notion.id",
    "slack": "com.tinyspeck.slackmacgap",
    "discord": "com.hnc.Discord",
    "spotify": "com.spotify.client",
    "claude": "com.anthropic.claudefordesktop",
    "codex": "com.openai.codex",
    "chatgpt": "com.openai.chat",
    "inkscape": "org.inkscape.Inkscape",
    "drawio": "com.jgraph.drawio.desktop",
    "linear": "com.linear",
}


def normalize_app_id(raw: Any) -> str:
    """Canonical app id: collapse the bundle-id vs process-name spellings. Unknown
    ids pass through unchanged (the corpus grows; a new app must not become
    UNKNOWN)."""
    app = str(raw).strip()
    if not app:
        return UNKNOWN
    return _ALIASES.get(app, app)


def resolve_app_selector(name: str) -> str:
    """One ``--include-app``/``--exclude-app`` value -> canonical id. Accepts a
    friendly short name (``firefox``, ``cursor``), a raw process name, or a bundle
    id verbatim."""
    key = str(name).strip()
    if not key:
        raise ValueError("empty app selector")
    if key.upper() in UNRESOLVED_APPS:
        return key.upper()
    return _FRIENDLY.get(key.lower(), normalize_app_id(key))


@dataclass(frozen=True)
class AppRun:
    """A maximal same-app span on the master tick axis, ``[start, end)`` ticks."""

    app: str
    start: int
    end: int

    @property
    def n_ticks(self) -> int:
        return self.end - self.start

    def duration_s(self, master_fps: float) -> float:
        return self.n_ticks / master_fps


@dataclass
class AppTrack:
    """Forward-filled foreground app over one segment's master ticks.

    ``switch_ticks`` are the ticks at which the app CHANGES (the tick containing
    the ``ContextChanged``); ``apps`` is the app in force from that tick onward.
    Both are parallel and ascending, and ``switch_ticks[0]`` may be > 0 when the
    keylog opens without a context event (that head span is ``UNKNOWN``).
    """

    switch_ticks: list[int]
    apps: list[str]
    n_ticks: int
    master_fps: float

    def at(self, tick: int) -> str:
        """The app in force at ``tick`` (forward fill). ``UNKNOWN`` before the
        first context event."""
        i = bisect_right(self.switch_ticks, int(tick)) - 1
        return self.apps[i] if i >= 0 else UNKNOWN

    def counts(self, lo: int, hi: int) -> Counter:
        """Ticks per app over ``[lo, hi)`` -- the interval form of ``at()``. Used
        to score a whole conversation, where a point sample would hide a switch."""
        out: Counter = Counter()
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return out
        i = max(0, bisect_right(self.switch_ticks, lo) - 1)
        while i < len(self.switch_ticks) and self.switch_ticks[i] < hi:
            start = max(self.switch_ticks[i], lo)
            end = min(
                self.switch_ticks[i + 1] if i + 1 < len(self.switch_ticks) else hi, hi
            )
            if end > start:
                out[self.apps[i]] += end - start
            i += 1
        if not self.switch_ticks or self.switch_ticks[0] > lo:
            head_end = min(self.switch_ticks[0], hi) if self.switch_ticks else hi
            if head_end > lo:
                out[UNKNOWN] += head_end - lo
        return out

    def switches_in(self, lo: int, hi: int) -> list[int]:
        """Switch ticks strictly inside ``(lo, hi)`` -- the seams that make a
        window's action label span two apps."""
        i = bisect_right(self.switch_ticks, int(lo))
        return [t for t in self.switch_ticks[i:] if t < int(hi)]

    def runs(self, *, merge_across_unresolved: bool = True) -> list[AppRun]:
        """Maximal same-app spans. With ``merge_across_unresolved`` (the default) an
        UNCAPTURED/UNKNOWN span between two spans of the SAME app does not break the
        run: those ticks are the privacy blackout (already black-filtered), so the
        app never really changed. A span of a DIFFERENT app always breaks it."""
        out: list[AppRun] = []
        for i, app in enumerate(self.apps):
            start = self.switch_ticks[i]
            end = (
                self.switch_ticks[i + 1]
                if i + 1 < len(self.switch_ticks)
                else self.n_ticks
            )
            if end <= start:
                continue
            if app in UNRESOLVED_APPS:
                continue
            # extend the open run when the same app resumes: always if we merge
            # across blackouts, otherwise only when the spans actually touch
            resumes = out and out[-1].app == app and (
                merge_across_unresolved or out[-1].end == start
            )
            if resumes:
                out[-1] = AppRun(app, out[-1].start, end)
            else:
                out.append(AppRun(app, start, end))
        return out

    def summary(self) -> dict[str, Any]:
        """Segment-level COVERAGE provenance, weighted by master ticks (i.e. by
        wall-clock, including the black/idle ticks no frame was sampled from). The
        ``*_by_ticks`` keys are deliberately distinct from ``frame_app_stats``:
        filtering must gate on frames, so only this one is named by ticks."""
        counts = self.counts(0, self.n_ticks)
        captured = {a: n for a, n in counts.items() if a not in UNRESOLVED_APPS}
        total = sum(captured.values())
        dominant = max(captured, key=lambda a: captured[a]) if captured else None
        return {
            "apps": sorted(captured, key=lambda a: -captured[a]),
            "app_ticks": dict(counts),
            "app_by_ticks": dominant,
            "app_frac_by_ticks": round(captured[dominant] / total, 6) if dominant else 0.0,
            "app_uncaptured_frac": (
                round(sum(counts[a] for a in UNRESOLVED_APPS if a in counts) / self.n_ticks, 6)
                if self.n_ticks
                else 0.0
            ),
            "n_app_switches": max(0, len(self.switch_ticks) - 1),
        }


def build_app_track(
    events: Iterable[tuple[float, str]],
    *,
    n_ticks: int,
    master_fps: float,
) -> AppTrack:
    """Fold ``(t_s, raw_app)`` context events onto the master tick axis.

    Timestamps are on the master clock already (see the module docstring), so the
    tick is ``floor(t * master_fps)``, clamped to the axis. LAST WINS when two
    events land on the same tick -- both for genuinely fast switching and for the
    pause-collapse clamp, where the last one is the app in focus at resume."""
    ticks: list[int] = []
    apps: list[str] = []
    for t_s, raw in events:
        tick = max(int(t_s * master_fps), 0)
        if n_ticks and tick >= n_ticks:
            tick = n_ticks - 1
        app = normalize_app_id(raw)
        if ticks and ticks[-1] == tick:
            apps[-1] = app          # last wins
            continue
        if apps and apps[-1] == app:
            continue                # no-op switch (re-focus of the same app)
        ticks.append(tick)
        apps.append(app)
    return AppTrack(switch_ticks=ticks, apps=apps, n_ticks=int(n_ticks), master_fps=float(master_fps))


def load_app_track(
    keylog_path: Path | str | None,
    *,
    n_ticks: int,
    master_fps: float,
) -> AppTrack:
    """``build_app_track`` over a (realigned) keylog. A missing/absent keylog, or
    one from a recorder version that predates ``ContextChanged``, yields an EMPTY
    track whose ``at()`` is ``UNKNOWN`` everywhere -- absence of the event is not
    evidence of an app."""
    if keylog_path is None:
        return AppTrack([], [], int(n_ticks), float(master_fps))
    path = Path(keylog_path)
    if not path.exists():
        return AppTrack([], [], int(n_ticks), float(master_fps))
    return build_app_track(iter_context(path), n_ticks=n_ticks, master_fps=master_fps)


def frame_app_stats(frames: list[dict[str, Any]], *, key: str = "app") -> dict[str, Any]:
    """App scoring over SAMPLED FRAMES (one frame == one conversation turn):
    dominant app, its share of the labeled frames, the full mix, and how many turns
    straddle a switch.

    Deliberately frame-weighted, not tick-weighted: a segment's tick-dominant app
    (see ``AppTrack.summary``) can differ from its frame-dominant one, because
    black/idle thinning keeps only a fraction of the ticks. Everything that GATES a
    conversation must agree on this one, so the stage-03 index row and the stage-04
    filter both use it."""
    counts: Counter = Counter()
    for f in frames:
        app = f.get(key)
        if app:
            counts[normalize_app_id(app)] += 1
    captured = {a: n for a, n in counts.items() if a not in UNRESOLVED_APPS}
    total = sum(captured.values())
    dominant = max(captured, key=lambda a: captured[a]) if captured else None
    return {
        "app": dominant,
        "app_frac": round(captured[dominant] / total, 6) if dominant else 0.0,
        "app_mix": dict(counts.most_common()) if counts else None,
        "app_seam_turns": sum(1 for f in frames if f.get("app_window_switches")),
    }


def iter_app_spans(
    frames: list[dict[str, Any]],
    *,
    key: str = "app",
) -> Iterator[tuple[str, int, int]]:
    """``(app, lo, hi)`` index spans of consecutive same-``key`` frames -- the
    conversation-splitting form, over ALREADY-SAMPLED frames (so it needs no
    keylog and no tick axis). ``hi`` is exclusive. Frames whose app is missing or
    unresolved break a span."""
    lo: int | None = None
    cur: str | None = None
    for i, f in enumerate(frames):
        app = f.get(key)
        app = normalize_app_id(app) if app else None
        if app is None or app in UNRESOLVED_APPS:
            if lo is not None and cur is not None:
                yield (cur, lo, i)
            lo, cur = None, None
            continue
        if cur is None:
            lo, cur = i, app
        elif app != cur:
            yield (cur, lo, i)  # type: ignore[arg-type]
            lo, cur = i, app
    if lo is not None and cur is not None:
        yield (cur, lo, len(frames))
