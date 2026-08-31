"""Construct the QEMU desktop pool used by the evaluation harness.

The desktop package exposes one complete ``DesktopSession`` surface. The
harness receives that session directly through ``CheckedOutDesktopSession``;
Juergen owns no transport facade and no second reset policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["kvm_desktop_pool"]


def kvm_desktop_pool(
    *,
    image: str | Path,
    root_dir: str | Path,
    qemu_binary: str | Path | None = None,
    qemu_img_binary: str | Path | None = None,
    smp: int | None = None,
    memory: str | None = None,
    accelerator: str | None = None,
    transport_timeout_s: float = 60.0,
    min_ready_sessions: int = 1,
    max_sessions: int = 1,
    max_rollouts_per_session: int = 1,
    checkout_timeout_s: float = 1800.0,
    lease_timeout_s: float = 1800.0,
    startup_timeout_s: float = 1200.0,
    status_dir: str | Path | None = None,
) -> Any:
    """Build a pool from JSON-safe arguments.

    ``desktop`` resets every reusable session before marking it ready. Juergen
    therefore does not reset on checkout and cannot accidentally advertise a
    dirty VM through a second, diverging reuse policy.
    """
    from desktop.vm.factory import build_desktop_pool
    from desktop.vm.pool import DesktopPoolConfig

    config = DesktopPoolConfig(
        min_ready_sessions=min_ready_sessions,
        max_sessions=max_sessions,
        max_rollouts_per_session=max_rollouts_per_session,
        checkout_timeout_s=checkout_timeout_s,
        lease_timeout_s=lease_timeout_s,
        startup_timeout_s=startup_timeout_s,
        status_dir=Path(status_dir) if status_dir is not None else None,
    )
    runtime_options: dict[str, Any] = {"transport_timeout_s": transport_timeout_s}
    if qemu_binary is not None:
        runtime_options["qemu_binary"] = qemu_binary
    if qemu_img_binary is not None:
        runtime_options["qemu_img_binary"] = qemu_img_binary
    if smp is not None:
        runtime_options["smp"] = int(smp)
    if memory is not None:
        runtime_options["memory"] = memory
    if accelerator is not None:
        runtime_options["accelerator"] = accelerator
    return build_desktop_pool(
        root_dir=Path(root_dir),
        image=Path(image),
        config=config,
        **runtime_options,
    )
