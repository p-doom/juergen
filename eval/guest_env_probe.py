import json
import os
import sys

from absl import app, flags

sys.path.insert(0, os.environ.get("OSWORLD_ROOT", "/fast/project/HFMI_SynergyUnit/yll/osworld-pinned"))

FLAGS = flags.FLAGS
flags.DEFINE_string("path_to_vm", "", "qcow2 path")
flags.DEFINE_string("out", "guest_env_probe.json", "output json")

PROBE = (
    "import importlib.util, json, sys; "
    "mods=['openpyxl','docx','pptx','PIL','fitz','pymupdf','lxml','numpy','PyPDF2','pypdf','pandas','odf','requests','pyautogui','bs4','yaml']; "
    "ok={m: bool(importlib.util.find_spec(m)) for m in mods}; "
    "print('PROBE_JSON:'+json.dumps({'python': sys.version.split()[0], 'mods': ok, 'pip': bool(importlib.util.find_spec('pip'))}))"
)


def main(_):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qemu_kvm_provider
    from osworld_fullbench_kvm import _lease_vm_ports

    _lease_vm_ports()
    qemu_kvm_provider.install()
    from desktop_env.desktop_env import DesktopEnv

    env = DesktopEnv(
        provider_name="docker",
        path_to_vm=FLAGS.path_to_vm,
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
    )
    env.reset(task_config={"id": "guest-env-probe", "instruction": "probe", "config": [], "evaluator": {"func": "infeasible"}})
    out = env.controller.execute_python_command(PROBE)
    env.close()
    payload = None
    for line in (out.get("output") or "").splitlines():
        if line.startswith("PROBE_JSON:"):
            payload = json.loads(line[len("PROBE_JSON:"):])
    result = {"raw": out, "probe": payload}
    with open(FLAGS.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(payload or result))


if __name__ == "__main__":
    app.run(main)
