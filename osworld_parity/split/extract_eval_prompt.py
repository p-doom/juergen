"""Extract the BYTE-EXACT system prompt the eval harness feeds the model.

Instantiates mm_agents.qwen3vl_agent.Qwen3VLAgent exactly as the eval harness
does (computer_use / pyautogui, screenshot-only, coordinate_type=relative),
intercepts predict()'s self.call_llm to capture the real assembled messages,
and dumps messages[0] (the system prompt) verbatim + length + sha256.

Also compares against osworld_system_prompts['computer_use_v1'] (the stand-in
a4b609f6 is baking into abs-control training) so we know if they differ.
No GPU / no VM / no model call needed.
"""
import hashlib
import sys
from io import BytesIO

sys.path.insert(0, "/fast/home/franz.srambical/OSWorld")
sys.path.insert(0, "/fast/home/franz.srambical/juergen/eval")
from PIL import Image
from mm_agents.qwen3vl_agent import Qwen3VLAgent

agent = Qwen3VLAgent(
    platform="ubuntu", model="Qwen3-VL-8B-Instruct", max_tokens=1024, top_p=0.9,
    temperature=0.0, action_space="pyautogui", observation_type="screenshot",
    history_n=4, coordinate_type="relative", api_backend="openai",
)
agent.reset(None)

cap = {}
class _Stop(Exception):
    pass
def fake_call_llm(payload, model):
    cap["messages"] = payload["messages"]
    raise _Stop()
agent.call_llm = fake_call_llm

buf = BytesIO()
Image.new("RGB", (1920, 1080), (255, 255, 255)).save(buf, format="PNG")
try:
    agent.predict("PLACEHOLDER_INSTRUCTION", {"screenshot": buf.getvalue()})
except _Stop:
    pass

sp = cap["messages"][0]["content"][0]["text"]
data = sp.encode("utf-8")
out = "/fast/home/franz.srambical/osworld_parity_split/eval_system_prompt.txt"
open(out, "w").write(sp)

print("=== EVAL-HARNESS SYSTEM PROMPT (Qwen3VLAgent, computer_use, screenshot, relative) ===")
print("len_chars:", len(sp))
print("len_bytes:", len(data))
print("sha256:", hashlib.sha256(data).hexdigest())
print("written_to:", out)
print("screen-size dependent?:", "NO (resolution hardcoded 1000x1000 for relative)")

# compare to the stand-in a4b609f6 bakes into abs-control training
try:
    from osworld_system_prompts import SYSTEM_PROMPTS
    cu1 = SYSTEM_PROMPTS.get("computer_use_v1")
    if cu1 is not None:
        print("\n=== vs osworld_system_prompts['computer_use_v1'] ===")
        print("computer_use_v1 len_chars:", len(cu1))
        print("EQUAL:", cu1 == sp)
        if cu1 != sp:
            # first divergence
            import difflib
            i = next((k for k in range(min(len(cu1), len(sp))) if cu1[k] != sp[k]), min(len(cu1), len(sp)))
            print("first divergence at char", i)
            print("  eval[...]:", repr(sp[max(0,i-40):i+40]))
            print("  cu1 [...]:", repr(cu1[max(0,i-40):i+40]))
    else:
        print("\ncomputer_use_v1 not found in SYSTEM_PROMPTS; keys:", list(SYSTEM_PROMPTS)[:20])
except Exception as e:
    print("\n(could not load osworld_system_prompts:", e, ")")

print("\n----BEGIN EVAL SYSTEM PROMPT----")
print(sp)
print("----END EVAL SYSTEM PROMPT----")

# ---- per-turn user scaffold + input spec (follow-up #1) ----
# messages[1] is the (only) user turn for a no-history step:
#   content = [ {image_url}, {type:text, text: instruction_prompt} ]
user_turn = cap["messages"][1]
user_text = next(c["text"] for c in user_turn["content"] if c.get("type") == "text")
img_part = next(c for c in user_turn["content"] if c.get("type") == "image_url")
img_url_prefix = img_part["image_url"]["url"][:32]

spec = f"""EVAL-HARNESS INPUT SPEC — mm_agents.qwen3vl_agent.Qwen3VLAgent
(computer_use / pyautogui, observation_type=screenshot, coordinate_type=relative, history_n=4)
The abs-control TRAINING input must byte-match ALL of the following for a true zero-shift control.

[1] SYSTEM PROMPT
    file: eval_system_prompt.txt   len={len(sp)}   sha256={hashlib.sha256(data).hexdigest()}
    screen-size independent (resolution hard-coded 1000x1000 for coordinate_type=relative).

[2] MESSAGE STRUCTURE (per predict call)
    messages[0] = {{role: system,  content:[{{type:text, text: <SYSTEM PROMPT>}}]}}
    then the last history_n=4 turns are interleaved, oldest->newest:
      user      content:[{{image_url}}] (+ the instruction text ONLY on the FIRST in-window user turn)
      assistant content:[{{type:text, text: <that turn's raw model response>}}]
    then the CURRENT frame:
      user      content:[{{image_url}}]                       (if there IS history)
      user      content:[{{image_url}}, {{type:text, text: <USER TURN TEXT>}}]  (if NO history: instruction rides here)
    NOTE: the instruction/USER-TURN-TEXT is attached to the FIRST in-window user turn only
    (qwen3vl_agent.py:224-311). History uses the last 4 (frame,response) turns.

[3] USER-TURN TEXT (byte-exact; {{instruction}} and {{previous_actions_str}} are substituted;
    previous_actions_str = "None" on step 1, else "Step 1: <act>\\nStep 2: <act>...")
    file: eval_user_turn_template.txt
----BEGIN USER TURN TEXT (instruction=PLACEHOLDER_INSTRUCTION, previous=None)----
{user_text}
----END USER TURN TEXT----

[4] IMAGE
    each frame is a data URL: '{img_url_prefix}...'  (PNG, base64)
    frames are smart_resize-processed BEFORE encoding: max_pixels = 16*16*4*12800 = {16*16*4*12800}
    (qwen3vl_agent.py process_image/smart_resize); e.g. a 1920x1080 screen -> ~1920x1088 processed.

[5] COORDINATES
    relative 0-999 grid. On PARSE, x_px = int(x * original_width/999), y_px = int(y * original_height/999)
    (qwen3vl_agent.py:382-385). Eval screen = 1920x1080.

[6] DECODING (eval harness): temperature=0.0, top_p=0.9, max_tokens=1024, history_n=4.
"""
open("/fast/home/franz.srambical/osworld_parity_split/eval_user_turn_template.txt", "w").write(user_text)
open("/fast/home/franz.srambical/osworld_parity_split/eval_input_spec.txt", "w").write(spec)
print("\n=== INPUT SPEC ===")
print(spec)
