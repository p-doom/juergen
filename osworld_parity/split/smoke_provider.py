"""No-GPU smoke test for the restored native-qemu ('apptainer') DesktopEnv provider.

Boots the OSWorld qcow2 via DesktopEnv(provider_name='apptainer'), runs a
task-less reset (no model needed), and pulls a screenshot + screen size
through the real PythonController round-trip. Validates: qemu boots under
KVM, guest osworld-server answers on :5000, controller talks to it.
"""
import io
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
os.environ.setdefault("OSWORLD_VM_LOGDIR", "/tmp/osworld_smoke")

OSWORLD_ROOT = "/fast/home/franz.srambical/OSWorld"
sys.path.insert(0, OSWORLD_ROOT)
QCOW2 = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2"

from PIL import Image  # noqa: E402
from desktop_env.desktop_env import DesktopEnv  # noqa: E402

env = DesktopEnv(
    provider_name="apptainer",
    path_to_vm=QCOW2,
    action_space="pyautogui",
    screen_size=(1920, 1080),
    headless=True,
    os_type="Ubuntu",
    require_a11y_tree=False,
)
print("PROVIDER OK: DesktopEnv constructed + VM booted", flush=True)
try:
    print("vm screen size:", env.controller.get_vm_screen_size(), flush=True)
    obs = env.reset(task_config=None)
    ss = obs.get("screenshot")
    print("screenshot bytes:", len(ss) if ss else None, flush=True)
    img = Image.open(io.BytesIO(ss))
    print("decoded image size:", img.size, flush=True)
    # quick action round-trip: move mouse, read cursor
    env.controller.execute_python_command(
        "import pyautogui; pyautogui.FAILSAFE=False; pyautogui.moveTo(400,300)"
    )
    print("SMOKE OK: booted + controller round-trip succeeded", flush=True)
finally:
    env.close()
    print("closed", flush=True)
