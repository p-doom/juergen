from __future__ import annotations

import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS

DEFAULT_TASKS_PARQUET = "/fast/project/HFMI_SynergyUnit/yll/gym_rollout_assets/dataset/CUA-Gym/data/tasks.parquet"
DEFAULT_TASKS_V1_DIR = "/fast/project/HFMI_SynergyUnit/yll/gym_rollout_assets/dataset/tasks_v1"
DEFAULT_ROLLOUTS_JSONL = "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/datasets--p-doom--cuagym-qwen35-rollouts/snapshots/ac6a484bd3e9dbe232163e200e80613cc99146b2/p2_9b_think/trajectories.jsonl"
DEFAULT_STRAT38_JSON = "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu/osworld_verified_strat38_no_gdrive.json"
DEFAULT_OSWORLD_EXAMPLES = "/fast/project/HFMI_SynergyUnit/yll/osworld-pinned/evaluation_examples/examples"
DEFAULT_REGISTERED_BLOCKLIST = "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu/cuagym_osworld_blocklist_v1.json"

PERSIST_KEYWORDS = {
    "file_io": [
        "with open",
        "open(",
        "os.path",
        "os.listdir",
        "os.walk",
        "os.stat",
        "os.remove",
        "os.makedirs",
        "pathlib",
        "Path(",
        "glob(",
        "import glob",
        "shutil",
    ],
    "document_parse": [
        "import fitz",
        "fitz.open",
        "from docx",
        "import docx",
        "openpyxl",
        "load_workbook",
        "pypdf",
        "PyPDF",
        "import odf",
        "from odf",
        "ezodf",
        "zipfile",
        "ElementTree",
        "etree.parse",
        "csv.reader",
        "csv.DictReader",
        "json.load",
        "configparser",
        "yaml.safe_load",
        "tarfile",
        "Image.open",
    ],
    "process_command_state": [
        "subprocess",
        "psutil",
        "/proc/",
        "os.popen",
        "check_output",
        "getoutput",
        "os.system",
    ],
    "database": [
        "sqlite3",
        "sqlite",
    ],
    "http_state": [
        "requests.get",
        "requests.post",
        "urllib.request",
    ],
}

TRANSIENT_KEYWORDS = {
    "screen_capture": [
        "pyautogui.screenshot",
        "ImageGrab.grab",
        "import mss",
        "mss.mss(",
        "gnome-screenshot",
        "scrot",
        "xwd",
    ],
    "live_pixel_inspection": [
        "locateOnScreen",
        "pyautogui.pixel",
        "pixel_color",
        "matchTemplate",
    ],
    "accessibility": [
        "pyatspi",
        "Atspi",
        "dogtail",
        "AT-SPI",
        "accessibility_tree",
    ],
    "window_inspection": [
        "wmctrl -l",
        "xdotool search",
        "xdotool getactivewindow",
        "xdotool getwindow",
        "xprop",
        "xdpyinfo",
        "Wnck",
        "_NET_WM",
        "Xlib",
    ],
}

GAP_SIGNATURES = {
    "calc_dialog_list_entry": {
        "app_types": ["libreoffice_calc"],
        "keywords": [
            "data validation",
            "validity",
            "list entry",
            "dropdown",
            "drop-down",
            "conditional formatting",
            "format cells",
            "autofilter",
            "custom sort",
            "sort dialog",
            "define name",
            "named range",
        ],
    },
    "spinbox_numeric_edit": {
        "app_types": [],
        "keywords": [
            "spinbox",
            "spin box",
            "row height",
            "column width",
            "font size to",
            "margin to",
            "margins to",
            "indent",
            "line spacing",
            "spacing to",
            "zoom to",
            "opacity to",
            "brightness",
            "transparency to",
        ],
    },
    "file_open_double_click": {
        "app_types": [],
        "keywords": [
            "double-click",
            "double click",
            "file manager",
            "nautilus",
            "from the desktop",
            "open the file",
            "open the document",
            "open the folder",
        ],
    },
    "multi_app_export_chain": {
        "app_types": ["multi_apps"],
        "keywords": [
            "export",
            "save as pdf",
            "save it as pdf",
            "convert",
            "then open",
            "import it",
            "paste it into",
            "copy it into",
            "attach",
        ],
    },
    "app_menu_navigation": {
        "app_types": [],
        "keywords": [
            "menu",
            "toolbar",
            "preferences",
            "options dialog",
            "settings dialog",
            "tools >",
            "format >",
            "insert >",
            "view >",
            "file >",
        ],
    },
}


def define_flags() -> None:
    flags.DEFINE_string("tasks_parquet", DEFAULT_TASKS_PARQUET, "CUA-Gym tasks parquet")
    flags.DEFINE_string("tasks_v1_dir", DEFAULT_TASKS_V1_DIR, "Per-task verifier dir")
    flags.DEFINE_string("rollouts_jsonl", DEFAULT_ROLLOUTS_JSONL, "SFT rollouts with rewards")
    flags.DEFINE_string("strat38_json", DEFAULT_STRAT38_JSON, "OSWorld eval subset app->ids")
    flags.DEFINE_string("osworld_examples_dir", DEFAULT_OSWORLD_EXAMPLES, "OSWorld examples root")
    flags.DEFINE_string(
        "registered_blocklist", DEFAULT_REGISTERED_BLOCKLIST, "Registered blocklist (read-only)"
    )
    flags.DEFINE_string("out_dir", None, "Output dir", required=True)
    flags.DEFINE_string("prior_cache", "", "Prior-rewards cache path (default <out_dir>/prior_rewards_cache.json)")
    flags.DEFINE_integer("round_size", 600, "Round-0 task count")
    flags.DEFINE_float("persist_frac", 0.6, "Target persist-verified fraction (0.5-0.7)")
    flags.DEFINE_float("tier_a_boost", 3.0, "Weight multiplier for tier_a tasks")
    flags.DEFINE_float("w_solved_sometimes", 2.0, "Weight for 0<rate<=sometimes_max")
    flags.DEFINE_float("w_reliably_solved", 0.5, "Weight for rate>sometimes_max")
    flags.DEFINE_float("w_never_solved", 0.25, "Weight for rate==0 with data")
    flags.DEFINE_float("w_no_data", 1.0, "Weight for tasks without prior rollouts")
    flags.DEFINE_float("sometimes_max", 0.3, "Upper bound of solved-sometimes bucket")
    flags.DEFINE_float("solve_threshold", 0.999, "Reward threshold counting as solved")
    flags.DEFINE_float("jaccard_threshold", 0.8, "Token-set Jaccard near-dup threshold")
    flags.DEFINE_integer("substring_min_tokens", 5, "Min tokens for substring near-dup rule")
    flags.DEFINE_integer("seed", 0, "Sampling seed")


def strip_comments_and_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        text = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", source)
        return re.sub(r"(?m)^\s*#.*$", "", text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def pattern_hit(code: str, pattern: str) -> bool:
    if not pattern[0].isalnum() and pattern[0] != "_":
        return pattern in code
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(pattern), code) is not None


def classify_verifier(source: str) -> tuple[str, dict[str, list[str]]]:
    if not source.strip():
        return "empty", {"persist": [], "transient": []}
    code = strip_comments_and_docstrings(source)
    hits: dict[str, list[str]] = {"persist": [], "transient": []}
    for label, patterns in PERSIST_KEYWORDS.items():
        if any(pattern_hit(code, p) for p in patterns):
            hits["persist"].append(label)
    for label, patterns in TRANSIENT_KEYWORDS.items():
        if any(pattern_hit(code, p) for p in patterns):
            hits["transient"].append(label)
    n_persist = len(hits["persist"])
    n_transient = len(hits["transient"])
    if n_transient and n_transient >= n_persist:
        return "transient_screen", hits
    if n_persist:
        return "persist_verified", hits
    return "unknown", hits


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_near_duplicate(
    a: str, b: str, threshold: float, min_tokens: int
) -> tuple[bool, float, bool]:
    ta, tb = normalize_tokens(a), normalize_tokens(b)
    j = jaccard(ta, tb)
    if j >= threshold:
        return True, j, False
    if min(len(ta), len(tb)) >= min_tokens:
        na, nb = normalize_text(a), normalize_text(b)
        if na and nb and (na in nb or nb in na):
            return True, j, True
    return False, j, False


def match_gap_signatures(instruction: str, app_type: str) -> list[str]:
    instr = instruction.lower()
    matched = []
    for name, spec in GAP_SIGNATURES.items():
        if spec["app_types"] and app_type not in spec["app_types"]:
            continue
        for kw in spec["keywords"]:
            if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", instr):
                matched.append(name)
                break
    return matched


def load_prior_rewards(rollouts_path: Path, cache_path: Path) -> dict[str, list[float]]:
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    rewards: dict[str, list[float]] = defaultdict(list)
    with rollouts_path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line, strict=False)
            tid = re.sub(r"__r\d+$", "", d["task_id"])
            rewards[tid].append(float(d.get("reward") or 0.0))
            if (i + 1) % 2000 == 0:
                print(f"  rollouts parsed: {i + 1}", file=sys.stderr)
    out = dict(rewards)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(out, f)
    return out


def prior_bucket(
    rewards: list[float] | None, solve_threshold: float, sometimes_max: float
) -> tuple[str, float | None, float | None, int]:
    if not rewards:
        return "no_data", None, None, 0
    n = len(rewards)
    rate = sum(1 for r in rewards if r >= solve_threshold) / n
    mean = sum(rewards) / n
    if rate == 0:
        return "never_solved", rate, mean, n
    if rate <= sometimes_max:
        return "solved_sometimes", rate, mean, n
    return "reliably_solved", rate, mean, n


def load_eval_instructions(strat38_path: Path, examples_dir: Path) -> list[dict]:
    with strat38_path.open() as f:
        subset = json.load(f)
    out = []
    for eval_app, ids in subset.items():
        for eval_id in ids:
            p = examples_dir / eval_app / f"{eval_id}.json"
            if not p.exists():
                print(f"WARNING: missing eval example {p}", file=sys.stderr)
                continue
            with p.open() as f:
                instr = json.load(f).get("instruction", "")
            out.append({"app": eval_app, "id": eval_id, "instruction": instr})
    return out


def load_registered_blocklist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    ids: set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                ids.add(item)
            elif isinstance(item, dict):
                for key in ("task_id", "cuagym_id", "id"):
                    if key in item:
                        ids.add(item[key])
                        break
    elif isinstance(data, dict):
        ids.update(str(k) for k in data.keys())
    return ids


def weighted_sample_without_replacement(
    items: list[tuple[str, float]], k: int, rng: random.Random
) -> list[str]:
    keyed = [(rng.random() ** (1.0 / max(w, 1e-9)), tid) for tid, w in items]
    keyed.sort(reverse=True)
    return [tid for _, tid in keyed[:k]]


def rate_histogram(rates: list[float]) -> list[tuple[str, int]]:
    bins = [("0.0", 0)]
    edges = [(i / 10, (i + 1) / 10) for i in range(10)]
    counts = Counter()
    zero = 0
    for r in rates:
        if r == 0:
            zero += 1
            continue
        for lo, hi in edges:
            if lo < r <= hi:
                counts[(lo, hi)] += 1
                break
    rows = [("0.0", zero)]
    for lo, hi in edges:
        rows.append((f"({lo:.1f},{hi:.1f}]", counts.get((lo, hi), 0)))
    return rows


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main(argv):
    del argv
    import pandas as pd

    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(FLAGS.seed)
    if not 0.5 <= FLAGS.persist_frac <= 0.7:
        print(f"WARNING: persist_frac {FLAGS.persist_frac} outside 0.5-0.7", file=sys.stderr)

    df = pd.read_parquet(FLAGS.tasks_parquet)
    tasks = df.to_dict("records")
    for t in tasks:
        for key in ("instruction", "app_type", "app_family"):
            if not isinstance(t[key], str):
                t[key] = "unknown"
    tasks_v1 = Path(FLAGS.tasks_v1_dir)

    print("classifying verifiers ...", file=sys.stderr)
    verifier_class: dict[str, str] = {}
    verifier_hits: dict[str, dict] = {}
    missing_verifier = 0
    n_done = 0
    for t in tasks:
        if t["app_family"] == "mock_web":
            continue
        p = tasks_v1 / t["id"] / "reward.py"
        try:
            source = p.read_text()
        except OSError:
            missing_verifier += 1
            verifier_class[t["id"]] = "unknown"
            verifier_hits[t["id"]] = {"persist": [], "transient": []}
            continue
        cls, hits = classify_verifier(source)
        verifier_class[t["id"]] = cls
        verifier_hits[t["id"]] = hits
        n_done += 1
        if n_done % 2000 == 0:
            print(f"  classified: {n_done}", file=sys.stderr)

    print("loading prior rewards ...", file=sys.stderr)
    cache_path = Path(FLAGS.prior_cache) if FLAGS.prior_cache else out_dir / "prior_rewards_cache.json"
    prior_rewards = load_prior_rewards(Path(FLAGS.rollouts_jsonl), cache_path)

    print("loading eval instructions ...", file=sys.stderr)
    eval_tasks = load_eval_instructions(Path(FLAGS.strat38_json), Path(FLAGS.osworld_examples_dir))

    print("contamination scan ...", file=sys.stderr)
    blocklist_candidates = []
    contaminated_ids: set[str] = set()
    near_misses = []
    for t in tasks:
        best_j, best_et = -1.0, None
        for et in eval_tasks:
            dup, j, sub = is_near_duplicate(
                t["instruction"], et["instruction"], FLAGS.jaccard_threshold, FLAGS.substring_min_tokens
            )
            if j > best_j:
                best_j, best_et = j, et
            if dup:
                contaminated_ids.add(t["id"])
                blocklist_candidates.append(
                    {
                        "cuagym_id": t["id"],
                        "cuagym_app_type": t["app_type"],
                        "cuagym_instruction": t["instruction"],
                        "eval_app": et["app"],
                        "eval_id": et["id"],
                        "eval_instruction": et["instruction"],
                        "jaccard": round(j, 4),
                        "substring_match": sub,
                    }
                )
        if best_et is not None:
            near_misses.append((best_j, t, best_et))
    near_misses.sort(key=lambda x: -x[0])
    with (out_dir / "onpolicy_blocklist_candidates.json").open("w") as f:
        json.dump(blocklist_candidates, f, indent=1)

    registered = load_registered_blocklist(Path(FLAGS.registered_blocklist))

    eligible = []
    excluded = Counter()
    for t in tasks:
        if t["app_family"] == "mock_web":
            excluded["mock_web"] += 1
            continue
        if t["id"] in contaminated_ids:
            excluded["contamination"] += 1
            continue
        if t["id"] in registered:
            excluded["registered_blocklist"] += 1
            continue
        cls = verifier_class[t["id"]]
        if cls == "empty":
            excluded["empty_verifier"] += 1
            continue
        bucket, rate, mean_r, n_roll = prior_bucket(
            prior_rewards.get(t["id"]), FLAGS.solve_threshold, FLAGS.sometimes_max
        )
        signatures = match_gap_signatures(t["instruction"], t["app_type"])
        tier = "tier_a" if signatures else "tier_b"
        bucket_w = {
            "solved_sometimes": FLAGS.w_solved_sometimes,
            "reliably_solved": FLAGS.w_reliably_solved,
            "never_solved": FLAGS.w_never_solved,
            "no_data": FLAGS.w_no_data,
        }[bucket]
        tier_w = FLAGS.tier_a_boost if tier == "tier_a" else 1.0
        weight = tier_w * bucket_w
        eligible.append(
            {
                "task_id": t["id"],
                "app_family": t["app_family"],
                "app_type": t["app_type"],
                "verifier_class": cls,
                "prior_rate": None if rate is None else round(rate, 4),
                "prior_bucket": bucket,
                "prior_mean_reward": None if mean_r is None else round(mean_r, 4),
                "n_prior_rollouts": n_roll,
                "tier": tier,
                "gap_signatures": signatures,
                "weight": weight,
            }
        )

    persist_pool = [e for e in eligible if e["verifier_class"] == "persist_verified"]
    other_pool = [e for e in eligible if e["verifier_class"] != "persist_verified"]
    n_target = min(FLAGS.round_size, len(eligible))
    n_persist_target = min(round(n_target * FLAGS.persist_frac), len(persist_pool))
    n_other_target = min(n_target - n_persist_target, len(other_pool))
    n_persist_target = min(n_target - n_other_target, len(persist_pool))

    by_id = {e["task_id"]: e for e in eligible}
    picked_persist = weighted_sample_without_replacement(
        [(e["task_id"], e["weight"]) for e in persist_pool], n_persist_target, rng
    )
    picked_other = weighted_sample_without_replacement(
        [(e["task_id"], e["weight"]) for e in other_pool], n_other_target, rng
    )
    selected_ids = picked_persist + picked_other
    selected = [by_id[tid] for tid in selected_ids]
    selected.sort(key=lambda e: e["task_id"])

    with (out_dir / "round0_tasks.jsonl").open("w") as f:
        for e in selected:
            f.write(json.dumps(e) + "\n")

    class_counts_all = Counter(verifier_class.values())
    persist_cat_counts = Counter()
    transient_cat_counts = Counter()
    for hits in verifier_hits.values():
        for c in hits["persist"]:
            persist_cat_counts[c] += 1
        for c in hits["transient"]:
            transient_cat_counts[c] += 1
    sig_counts = Counter()
    for e in eligible:
        for s in e["gap_signatures"]:
            sig_counts[s] += 1
    tier_counts = Counter(e["tier"] for e in eligible)
    bucket_counts = Counter(e["prior_bucket"] for e in eligible)
    rates_with_data = [e["prior_rate"] for e in eligible if e["prior_rate"] is not None]
    hist = rate_histogram(rates_with_data)

    sel_class = Counter(e["verifier_class"] for e in selected)
    sel_tier = Counter(e["tier"] for e in selected)
    sel_bucket = Counter(e["prior_bucket"] for e in selected)
    sel_family = Counter(e["app_family"] for e in selected)
    sel_app = Counter(e["app_type"] for e in selected)
    sel_weight_total = sum(e["weight"] for e in selected)
    sel_persist_weight = sum(e["weight"] for e in selected if e["verifier_class"] == "persist_verified")

    n_sel = len(selected)
    report = []
    report.append("# Round-0 On-Policy Curriculum Report\n")
    report.append(
        f"Inputs: {len(tasks)} CUA-Gym tasks; {len(eval_tasks)} strat38 eval instructions; "
        f"{sum(len(v) for v in prior_rewards.values())} prior rollouts over {len(prior_rewards)} tasks.\n"
    )
    report.append(f"Missing reward.py: {missing_verifier}\n")
    report.append(
        f"Flags: round_size={FLAGS.round_size} persist_frac={FLAGS.persist_frac} "
        f"tier_a_boost={FLAGS.tier_a_boost} sometimes_max={FLAGS.sometimes_max} "
        f"solve_threshold={FLAGS.solve_threshold} "
        f"w=[sometimes {FLAGS.w_solved_sometimes}, reliably {FLAGS.w_reliably_solved}, "
        f"never {FLAGS.w_never_solved}, no_data {FLAGS.w_no_data}] "
        f"jaccard_threshold={FLAGS.jaccard_threshold} seed={FLAGS.seed}\n"
    )
    report.append("## Verifier classification (non-mock_web tasks)\n")
    report.append(
        md_table(
            ["class", "count"],
            [[k, v] for k, v in class_counts_all.most_common()],
        )
    )
    report.append("\n### Persist keyword-category hits\n")
    report.append(md_table(["category", "verifiers"], [[k, v] for k, v in persist_cat_counts.most_common()]))
    report.append("\n### Transient keyword-category hits\n")
    report.append(md_table(["category", "verifiers"], [[k, v] for k, v in transient_cat_counts.most_common()]))
    report.append("\n## Prior success under SFT policy (eligible tasks)\n")
    report.append(md_table(["bucket", "count"], [[k, v] for k, v in bucket_counts.most_common()]))
    report.append("\n### Prior-rate histogram (tasks with data)\n")
    report.append(md_table(["rate bin", "count"], [[k, v] for k, v in hist]))
    report.append("\n## Gap-neighbor tiers (eligible tasks)\n")
    report.append(md_table(["tier", "count"], [[k, v] for k, v in sorted(tier_counts.items())]))
    report.append("\n### Signature matches\n")
    report.append(md_table(["signature", "count"], [[k, v] for k, v in sig_counts.most_common()]))
    report.append("\n## Contamination guard\n")
    report.append(
        f"Blocklist candidates: {len(blocklist_candidates)} matches covering "
        f"{len(contaminated_ids)} distinct CUA-Gym tasks (excluded)."
    )
    report.append(f"Registered blocklist entries applied: {len(registered)}.\n")
    for c in blocklist_candidates[:3]:
        report.append(
            f"- `{c['cuagym_id']}` vs `{c['eval_app']}/{c['eval_id']}` "
            f"(jaccard={c['jaccard']}, substring={c['substring_match']}): "
            f"\"{c['cuagym_instruction'][:140]}\""
        )
    report.append("\n### Nearest sub-threshold pairs\n")
    for j, t, et in near_misses[:3]:
        report.append(
            f"- jaccard={j:.3f} `{t['id']}` vs `{et['app']}/{et['id']}`\n"
            f"  - gym: \"{t['instruction'][:140]}\"\n"
            f"  - eval: \"{et['instruction'][:140]}\""
        )
    report.append("\n## Exclusions\n")
    report.append(md_table(["reason", "count"], [[k, v] for k, v in excluded.most_common()]))
    report.append(f"\nEligible pool: {len(eligible)} tasks.\n")
    report.append(f"## Round-0 selection ({n_sel} tasks, seed={FLAGS.seed})\n")
    report.append("### By verifier class\n")
    report.append(
        md_table(
            ["class", "count", "task share", "weight share"],
            [
                [
                    k,
                    v,
                    f"{v / n_sel:.1%}",
                    f"{sum(e['weight'] for e in selected if e['verifier_class'] == k) / sel_weight_total:.1%}",
                ]
                for k, v in sel_class.most_common()
            ],
        )
    )
    report.append(
        f"\nPersist-verified: {sel_class.get('persist_verified', 0) / n_sel:.1%} of tasks, "
        f"{sel_persist_weight / sel_weight_total:.1%} of weight "
        f"(target {FLAGS.persist_frac:.0%}, band 50-70%).\n"
    )
    if n_other_target == len(other_pool) and sel_class.get("persist_verified", 0) / n_sel > 0.7:
        report.append(
            f"NOTE: the non-persist pool is exhausted ({len(other_pool)} eligible "
            f"transient/unknown tasks corpus-wide), so the persist share exceeds the "
            f"50-70% band by necessity: CUA-Gym verifiers are post-hoc artifact "
            f"checkers by construction.\n"
        )
    report.append("### By tier\n")
    report.append(
        md_table(
            ["tier", "count", "share"],
            [[k, v, f"{v / n_sel:.1%}"] for k, v in sorted(sel_tier.items())],
        )
    )
    report.append("\n### By prior bucket\n")
    report.append(
        md_table(
            ["bucket", "count", "share"],
            [[k, v, f"{v / n_sel:.1%}"] for k, v in sel_bucket.most_common()],
        )
    )
    report.append("\n### By app_family\n")
    report.append(
        md_table(
            ["app_family", "count", "share"],
            [[k, v, f"{v / n_sel:.1%}"] for k, v in sel_family.most_common()],
        )
    )
    report.append("\n### Top app_types\n")
    report.append(md_table(["app_type", "count"], [[k, v] for k, v in sel_app.most_common(12)]))
    report.append("")

    report_text = "\n".join(report)
    with (out_dir / "curriculum_report.md").open("w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nwrote {out_dir / 'round0_tasks.jsonl'} ({n_sel} tasks)", file=sys.stderr)


if __name__ == "__main__":
    define_flags()
    app.run(main)
