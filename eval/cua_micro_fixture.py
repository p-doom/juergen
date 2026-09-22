"""Deterministic calculator fixture for the CUA micro-evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tkinter as tk
from pathlib import Path

STATE_PATH = Path("/tmp/cua_micro_fixture_state.json")


class Calculator:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Calculator — CUA Micro Eval")
        self.root.geometry("1100x720+410+180")
        self.root.minsize(900, 600)
        self.expression = ""
        self.values = {"display": "0", "expression": "", "submitted": ""}

        frame = tk.Frame(self.root, bg="#f6f7fb", padx=48, pady=32)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="Calculator",
            font=("Sans", 28, "bold"),
            bg="#f6f7fb",
            fg="#172033",
        ).pack(anchor="w")
        self.display = tk.StringVar(value="0")
        tk.Entry(
            frame,
            textvariable=self.display,
            font=("Sans", 34),
            justify="right",
            state="readonly",
            readonlybackground="#ffffff",
        ).pack(fill="x", pady=(18, 16), ipady=12)
        grid = tk.Frame(frame, bg="#f6f7fb")
        grid.pack(fill="both", expand=True)
        labels = (
            ("7", "8", "9", "+"),
            ("4", "5", "6", "-"),
            ("1", "2", "3", "x"),
            ("0", ".", "=", "/"),
        )
        for row, row_labels in enumerate(labels):
            grid.rowconfigure(row, weight=1)
            for column, label in enumerate(row_labels):
                grid.columnconfigure(column, weight=1)
                tk.Button(
                    grid,
                    text=label,
                    font=("Sans", 24, "bold"),
                    command=lambda value=label: self.press(value),
                    bg="#ffffff" if column < 3 else "#dbe7ff",
                ).grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        self.root.bind_all("<KeyPress>", self.keypress)
        self.root.after(250, self.ready)

    def press(self, label: str) -> None:
        if label.isdigit():
            self.expression += label
        elif label == "+" and self.expression and self.expression[-1].isdigit():
            self.expression += label
        elif label == "=":
            match = re.fullmatch(r"(\d+)\+(\d+)", self.expression)
            if match:
                self.expression = str(int(match[1]) + int(match[2]))
                self.values["submitted"] = self.expression
        self.display.set(self.expression or "0")
        self.values.update({"display": self.display.get(), "expression": self.expression})
        self.write_state()

    def keypress(self, event: tk.Event) -> str | None:
        if event.char.isdigit() or event.char == "+":
            self.press(event.char)
            return "break"
        if event.keysym in {"Return", "KP_Enter"}:
            self.press("=")
            return "break"
        return None

    def ready(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.write_state()

    def write_state(self) -> None:
        payload = {"ready": True, "mode": "calculator", "values": self.values}
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=STATE_PATH.name + ".", dir=STATE_PATH.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, STATE_PATH)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Calculator().run()
