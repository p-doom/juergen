"""`evals/vm.py`'s contract with `desktop`, with NOTHING substituted.

Every other test that reaches `kvm_desktop_pool` runs under an autouse fixture
that replaces `desktop.vm.factory.build_desktop_pool` with a fake
(`test_signoflife_cli.py`), which is correct for those tests -- they are about the
CLI and must not boot a VM -- and is also exactly why a `startup_timeout_s` that
the real factory REFUSES survived here unnoticed. This file exists so at least one
test constructs the real thing.
"""

from __future__ import annotations

import inspect

import pytest


def test_the_real_factory_accepts_the_startup_budget_we_pass_it(tmp_path) -> None:
    """`desktop`'s `qemu_session_factory` refuses a `startup_timeout_s` below the
    QEMU runtime's own worst-case start (QMP connect 60 + boot 300 + snapshot
    600 = 960 s). We shipped 900, so the production path raised `ConfigError` at
    pool construction while the gate stayed green.

    Hermetic: constructing a pool boots nothing, `DesktopSessionPool.start()` does.
    """
    from desktop.vm.factory import ConfigError

    from evals.vm import kvm_desktop_pool

    from desktop.vm.factory import build_qemu_runtime

    image = tmp_path / "desktop.qcow2"
    image.write_bytes(b"\x00" * 64)

    # Read off the runtime rather than pinned at 960, so a phase timeout that grows
    # moves our default instead of silently passing a stale one.
    budget = build_qemu_runtime(image=image).start_budget_s
    default = inspect.signature(kvm_desktop_pool).parameters["startup_timeout_s"].default
    assert default >= budget, f"our default {default} is below the runtime's {budget}"

    kvm_desktop_pool(image=image, root_dir=tmp_path / "ok", accelerator="tcg").close()

    with pytest.raises(ConfigError, match="below this runtime's own worst-case"):
        kvm_desktop_pool(
            image=image,
            root_dir=tmp_path / "too-small",
            accelerator="tcg",
            startup_timeout_s=budget - 1.0,
        )
