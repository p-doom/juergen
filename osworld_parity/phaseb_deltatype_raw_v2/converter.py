#!/usr/bin/env python3
"""Byte-preserving assistant action-span replacement for raw deltatype-v2."""

from __future__ import annotations

from typing import Any


class ConversionError(RuntimeError):
    pass


def replace_action_span(
    conversion: Any, text: str, label: str
) -> tuple[str, str]:
    """Replace only the audited action span and return output plus old span."""
    before, old_action, after = conversion.split_assistant_turn(text)
    output = conversion.convert_assistant_turn(
        text,
        lambda _old: label,
        keep_prose=True,
    )
    new_before, new_action, new_after = conversion.split_assistant_turn(output)
    if (new_before, new_after) != (before, after):
        raise ConversionError("assistant bytes outside the action span changed")
    if new_action != label:
        raise ConversionError("converted assistant action span does not equal label")
    return output, old_action
