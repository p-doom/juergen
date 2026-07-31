#!/usr/bin/env python3
"""Guest-side dynamic image/event app for the unlaunched roadmap stage-2 design.

The host supplies a byte-exact PNG after every on-policy transition.  This app
reloads it only through a monotonic, hash-checked render command and advances its
active target only after a physical left-button release inside the current box.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from typing import Any, Sequence


class GuestContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(point: tuple[int, int], bbox: Sequence[int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def release_transition(
    target_index: int,
    targets: Sequence[Sequence[int]],
    point: tuple[int, int],
) -> tuple[int, bool, bool]:
    if not (0 <= target_index < len(targets)):
        raise GuestContractError("release without an active target")
    hit = _inside(point, targets[target_index])
    next_index = target_index + int(hit)
    completed = next_index == len(targets)
    return (target_index if completed else next_index), hit, completed


def validate_render_command(
    command: dict[str, Any],
    *,
    episode_revision: str,
    previous_sequence: int,
    target_index: int,
    targets: Sequence[Sequence[int]],
    image_path: Path,
) -> tuple[int, tuple[int, int], str]:
    if command.get("episode_revision") != episode_revision:
        raise GuestContractError("stale episode render command")
    sequence = command.get("sequence")
    if not isinstance(sequence, int) or sequence != previous_sequence + 1:
        raise GuestContractError("nonmonotonic render command")
    if command.get("target_index") != target_index:
        raise GuestContractError("render command target differs from guest state")
    if target_index >= len(targets):
        raise GuestContractError("cannot render a completed episode")
    if command.get("bbox") != list(targets[target_index]):
        raise GuestContractError("render command bbox differs from active target")
    cursor = command.get("cursor")
    if (
        not isinstance(cursor, list)
        or len(cursor) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in cursor)
    ):
        raise GuestContractError("render command has invalid cursor")
    expected_sha = command.get("image_sha256")
    if not isinstance(expected_sha, str) or _sha256(image_path) != expected_sha:
        raise GuestContractError("dynamic render PNG hash mismatch")
    return sequence, (cursor[0], cursor[1]), expected_sha


class DynamicGuestApp:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.revision = str(config["episode_revision"])
        self.targets = config["targets"]
        self.image_path = Path(config["image_path"])
        self.command_path = Path(config["command_path"])
        self.state_path = Path(config["state_path"])
        self.target_index = 0
        self.completed = False
        self.command_sequence = 0
        self.render_revision = "initial"
        self.rendered_cursor = tuple(config["initial_cursor"])
        self.image_sha256 = str(config["initial_image_sha256"])
        self.down = False
        self.button_presses = 0
        self.button_releases = 0
        self.last_release_position: tuple[int, int] | None = None
        self.last_hit: bool | None = None
        self.error: str | None = None
        if _sha256(self.image_path) != self.image_sha256:
            raise GuestContractError("initial dynamic PNG hash mismatch")
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{config['screen'][0]}x{config['screen'][1]}+0+0")
        self.root.configure(cursor="none")
        self.root.attributes("-topmost", True)
        self.image = tk.PhotoImage(file=str(self.image_path))
        self.label = tk.Label(self.root, image=self.image, borderwidth=0, highlightthickness=0)
        self.label.place(x=0, y=0)
        self.root.bind_all("<ButtonPress-1>", self._press)
        self.root.bind_all("<ButtonRelease-1>", self._release)
        self.root.after_idle(self._ready)
        self.root.after(50, self._poll_command)

    def _atomic_state(self) -> None:
        value = {
            "episode_revision": self.revision,
            "ready": True,
            "error": self.error,
            "target_index": self.target_index,
            "completed": self.completed,
            "down": self.down,
            "button_presses": self.button_presses,
            "button_releases": self.button_releases,
            "last_release_position": list(self.last_release_position) if self.last_release_position else None,
            "last_hit": self.last_hit,
            "command_sequence": self.command_sequence,
            "render_revision": self.render_revision,
            "rendered_cursor": list(self.rendered_cursor),
            "image_sha256": self.image_sha256,
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _ready(self) -> None:
        self.root.focus_force()
        self._atomic_state()

    def _press(self, _event: tk.Event) -> None:
        if self.completed or self.error:
            return
        if self.down:
            self.error = "duplicate mouseDown"
        else:
            self.down = True
            self.button_presses += 1
        self._atomic_state()

    def _release(self, event: tk.Event) -> None:
        if self.completed or self.error:
            return
        if not self.down:
            self.error = "mouseUp without mouseDown"
            self._atomic_state()
            return
        self.down = False
        self.button_releases += 1
        point = (int(event.x_root), int(event.y_root))
        self.last_release_position = point
        self.target_index, self.last_hit, self.completed = release_transition(
            self.target_index, self.targets, point
        )
        self._atomic_state()

    def _poll_command(self) -> None:
        if not self.error and not self.completed and self.command_path.is_file():
            try:
                command = json.loads(self.command_path.read_text(encoding="utf-8"))
                if command.get("sequence") != self.command_sequence:
                    sequence, cursor, image_sha = validate_render_command(
                        command,
                        episode_revision=self.revision,
                        previous_sequence=self.command_sequence,
                        target_index=self.target_index,
                        targets=self.targets,
                        image_path=self.image_path,
                    )
                    replacement = tk.PhotoImage(file=str(self.image_path))
                    self.label.configure(image=replacement)
                    self.image = replacement
                    self.command_sequence = sequence
                    self.rendered_cursor = cursor
                    self.image_sha256 = image_sha
                    self.render_revision = str(command["render_revision"])
                    self.root.update_idletasks()
                    self._atomic_state()
            except BaseException as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._atomic_state()
        self.root.after(50, self._poll_command)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: dynamic_guest_app.py CONFIG.json")
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    DynamicGuestApp(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
