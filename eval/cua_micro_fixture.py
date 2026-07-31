"""Deterministic, instrumented desktop fixtures for one-turn CUA evals.

This script is copied into the OSWorld guest and launched there.  It uses only
the Python standard library, renders a real desktop window, and atomically
writes semantic UI state plus exact widget bounding boxes to JSON.  The model
never sees the state file; the evaluator uses it after the action so progress is
machine-verifiable rather than judged from screenshots.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import tkinter as tk
from pathlib import Path


class Fixture:
    def __init__(self, mode: str, state_path: Path) -> None:
        self.mode = mode
        self.state_path = state_path
        self.root = tk.Tk()
        self.root.title(
            {
                "editor": "Text Editor — CUA Micro Eval",
                "terminal": "Terminal — CUA Micro Eval",
                "calculator": "Calculator — CUA Micro Eval",
                "files": "Files — CUA Micro Eval",
                "settings": "Settings — CUA Micro Eval",
            }[mode]
        )
        self.root.geometry("1100x720+410+180")
        self.root.minsize(900, 600)
        self.widgets: dict[str, tk.Widget] = {}
        self.item_widgets: dict[str, tuple[tk.Listbox, int]] = {}
        self.values: dict[str, object] = {}

        getattr(self, f"build_{mode}")()
        self.root.after(250, self._ready)

    def _base(self, title: str, subtitle: str) -> tk.Frame:
        frame = tk.Frame(self.root, bg="#f6f7fb", padx=48, pady=32)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text=title,
            font=("Sans", 28, "bold"),
            bg="#f6f7fb",
            fg="#172033",
        ).pack(anchor="w")
        tk.Label(
            frame,
            text=subtitle,
            font=("Sans", 14),
            bg="#f6f7fb",
            fg="#526078",
            pady=8,
        ).pack(anchor="w")
        return frame

    def build_editor(self) -> None:
        frame = self._base("Text Editor", "Type the requested text into the focused document.")
        text = tk.Text(frame, font=("Monospace", 20), padx=20, pady=20, wrap="word")
        text.pack(fill="both", expand=True, pady=(20, 0))
        text.bind("<KeyRelease>", lambda _event: self._set("text", text.get("1.0", "end-1c")))
        self.widgets["editor"] = text
        self.values["text"] = ""
        self.root.after(350, text.focus_force)

    def build_terminal(self) -> None:
        frame = tk.Frame(self.root, bg="#171421", padx=40, pady=35)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="CUA deterministic terminal",
            font=("Monospace", 18, "bold"),
            bg="#171421",
            fg="#ffffff",
        ).pack(anchor="w")
        row = tk.Frame(frame, bg="#171421")
        row.pack(fill="x", pady=(45, 0))
        tk.Label(
            row,
            text="eval@ubuntu:~$ ",
            font=("Monospace", 22),
            bg="#171421",
            fg="#8ae234",
        ).pack(side="left")
        value = tk.StringVar()
        entry = tk.Entry(
            row,
            textvariable=value,
            font=("Monospace", 22),
            bg="#171421",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
        )
        entry.pack(side="left", fill="x", expand=True)
        value.trace_add("write", lambda *_: self._set("command", value.get()))
        self.widgets["terminal_input"] = entry
        self.values["command"] = ""
        self.root.after(350, entry.focus_force)

    def build_calculator(self) -> None:
        frame = self._base("Calculator", "Click exactly one digit or operator.")
        display = tk.StringVar(value="0")
        entry = tk.Entry(
            frame,
            textvariable=display,
            font=("Sans", 34),
            justify="right",
            state="readonly",
            readonlybackground="#ffffff",
        )
        entry.pack(fill="x", pady=(18, 16), ipady=12)
        grid = tk.Frame(frame, bg="#f6f7fb")
        grid.pack(fill="both", expand=True)
        labels = (
            ("7", "8", "9", "+"),
            ("4", "5", "6", "-"),
            ("1", "2", "3", "x"),
            ("0", ".", "=", "/"),
        )

        def press(label: str) -> None:
            display.set(label)
            self._set("display", label)

        for row, values in enumerate(labels):
            grid.rowconfigure(row, weight=1)
            for col, label in enumerate(values):
                grid.columnconfigure(col, weight=1)
                button = tk.Button(
                    grid,
                    text=label,
                    font=("Sans", 24, "bold"),
                    command=lambda value=label: press(value),
                    bg="#ffffff" if col < 3 else "#dbe7ff",
                )
                button.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
                key = {
                    "+": "plus",
                    "-": "minus",
                    "x": "multiply",
                    "/": "divide",
                    "=": "equals",
                    ".": "decimal",
                }.get(label, f"digit_{label}")
                self.widgets[key] = button
        self.values["display"] = "0"

    def build_files(self) -> None:
        frame = self._base("Files", "Select the requested folder from the current directory.")
        listing = tk.Listbox(
            frame,
            font=("Sans", 22),
            activestyle="none",
            selectbackground="#3584e4",
            selectforeground="#ffffff",
        )
        listing.pack(fill="both", expand=True, pady=(20, 0))
        items = ("Documents", "Downloads", "EvalTarget", "Pictures", "Projects")
        for item in items:
            listing.insert("end", f"📁  {item}")
        listing.bind(
            "<<ListboxSelect>>",
            lambda _event: self._set(
                "selected",
                items[listing.curselection()[0]] if listing.curselection() else "",
            ),
        )
        self.item_widgets["folder_eval_target"] = (listing, items.index("EvalTarget"))
        self.values["selected"] = ""

    def build_settings(self) -> None:
        frame = self._base("Settings", "Mouse & Touchpad")
        card = tk.Frame(frame, bg="#ffffff", padx=32, pady=28, relief="groove", borderwidth=1)
        card.pack(fill="x", pady=(30, 0))
        enabled = tk.BooleanVar(value=False)
        check = tk.Checkbutton(
            card,
            text="Natural scrolling",
            variable=enabled,
            font=("Sans", 21),
            bg="#ffffff",
            activebackground="#ffffff",
            command=lambda: self._set("natural_scroll", bool(enabled.get())),
            padx=20,
            pady=20,
        )
        check.pack(fill="x", anchor="w")
        self.widgets["natural_scroll"] = check
        self.values["natural_scroll"] = False

    def _set(self, key: str, value: object) -> None:
        self.values[key] = value
        self.write_state()

    @staticmethod
    def _widget_bbox(widget: tk.Widget) -> list[int]:
        return [
            widget.winfo_rootx(),
            widget.winfo_rooty(),
            widget.winfo_rootx() + widget.winfo_width(),
            widget.winfo_rooty() + widget.winfo_height(),
        ]

    def _ready(self) -> None:
        self.root.update_idletasks()
        self.write_state()

    def write_state(self) -> None:
        self.root.update_idletasks()
        boxes = {key: self._widget_bbox(widget) for key, widget in self.widgets.items()}
        boxes["__window_content__"] = self._widget_bbox(self.root)
        for key, (listing, index) in self.item_widgets.items():
            item = listing.bbox(index)
            if item:
                x, y, width, height = item
                boxes[key] = [
                    listing.winfo_rootx() + x,
                    listing.winfo_rooty() + y,
                    listing.winfo_rootx() + x + width,
                    listing.winfo_rooty() + y + height,
                ]
        payload = {"ready": True, "mode": self.mode, "values": self.values, "widgets": boxes}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.state_path.name, dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
            Path(tmp).replace(self.state_path)
        finally:
            if Path(tmp).exists():
                Path(tmp).unlink()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("editor", "terminal", "calculator", "files", "settings"), required=True
    )
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    Fixture(args.mode, args.state).run()


if __name__ == "__main__":
    main()
