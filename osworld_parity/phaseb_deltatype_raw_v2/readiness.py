#!/usr/bin/env python3
"""Fail-closed response readiness checks for raw deltatype-v2."""

from __future__ import annotations

from action_v2 import DeltaTypeV2Action, format_deltatype_v2, parse_deltatype_v2
from prompt import SYSTEM_PROMPT


def validate_response(text: str) -> DeltaTypeV2Action:
    action = parse_deltatype_v2(text)
    canonical = format_deltatype_v2(action)
    if canonical != [line.strip() for line in text.splitlines() if line.strip()][-1]:
        raise ValueError("raw deltatype-v2 response is not canonical")
    return action


def selftest() -> None:
    required = (
        "RAW PIXEL",
        "initial_dx initial_dy 0 ; +LMB MOVE(drag_dx,drag_dy) -LMB",
        "MOVE(0,0)",
    )
    if any(fragment not in SYSTEM_PROMPT for fragment in required):
        raise RuntimeError("raw deltatype-v2 system prompt is incomplete")
    validate_response("0 0 0 ; +LMB MOVE(1051,254) -LMB")
    validate_response("-793 -229 0 ; +LMB MOVE(547,321) -LMB")


if __name__ == "__main__":
    selftest()
    print("raw deltatype-v2 readiness: PASS")
