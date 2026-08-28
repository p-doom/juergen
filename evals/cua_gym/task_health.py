"""Snapshot-pinned task-health findings and the blocklist derived from them."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tarfile
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import SnapshotValidationError
from .manifest import CompatibilityManifest
from .models import DatasetSnapshotConfig, TaskId

BLOCKLIST_VERSION = 1
JOURNAL_HEADER_RECORD = "journal_header"

EMPTY_REWARD_SCRIPT = "empty_reward_script"
UNPARSEABLE_REWARD_SCRIPT = "unparseable_reward_script"
REWARD_SCRIPT_EMITS_NO_SCORE = "reward_script_emits_no_score"
GRADER_PAYS_BEFORE_AGENT_ACTS = "grader_pays_before_agent_acts"

STATIC_DEFECTS = (
    EMPTY_REWARD_SCRIPT,
    UNPARSEABLE_REWARD_SCRIPT,
    REWARD_SCRIPT_EMITS_NO_SCORE,
)
BLOCKING_DEFECTS = (*STATIC_DEFECTS, GRADER_PAYS_BEFORE_AGENT_ACTS)

SHIPPED_DOCUMENT_SETUP_KINDS = frozenset({"docx", "pptx", "xlsx"})
SCRIPT_SETUP_KINDS = frozenset({"py", "sh"})

NOT_BLOCKED_MEANING = (
    "A task absent from blocked_tasks has no measured defect. That is not "
    "evidence that the task is solvable, that a correct solution reaches a "
    "non-zero score, or that its grader is correct."
)
RESET_PROBE_MEANING = (
    "grader_score_at_reset is the grader's score with setup applied and no "
    "agent action taken. Above zero proves the task pays for nothing. Zero "
    "proves only that; it measures nothing about solvability."
)

_ENDPOINT_PLACEHOLDER_RE = re.compile(r"__CUA_GYM_([A-Z0-9_]+?)_(?:URL|HOST)__")
_SCHEMA_KEY_RE = re.compile(r"`([^`]+)`")
_STATE_SCHEMA_HEADING = "state schema"
_TASK_FILE_NAMES = frozenset({"task.json", "reward.py"})
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SCORE_PATTERNS = (
    re.compile(rf"^\s*REWARD\s*:\s*({_NUMBER})\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(rf"REWARD\s*:\s*({_NUMBER})", re.IGNORECASE),
    re.compile(rf"^\s*Total score\s*:\s*({_NUMBER})", re.MULTILINE | re.IGNORECASE),
    re.compile(rf"^\s*Score\s*:\s*({_NUMBER})\s*/", re.MULTILINE | re.IGNORECASE),
    re.compile(rf"^\s*({_NUMBER})\s*$", re.MULTILINE),
)


def parse_grader_score(stdout: str) -> float | None:
    """Read a score from grader stdout under every contract the corpus uses."""

    for pattern in _SCORE_PATTERNS:
        matches = pattern.findall(stdout)
        if matches:
            return float(matches[-1])
    return None


@dataclass(frozen=True)
class SnapshotIdentity:
    """Dataset revision and file digests every finding is keyed to."""

    dataset: str
    revision: str
    file_digests: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "file_digests": dict(sorted(self.file_digests.items())),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SnapshotIdentity:
        digests = payload["file_digests"]
        if not isinstance(digests, dict):
            raise SnapshotValidationError("file_digests must be an object")
        return cls(
            dataset=str(payload["dataset"]),
            revision=str(payload["revision"]),
            file_digests={str(key): str(value) for key, value in digests.items()},
        )


@dataclass(frozen=True)
class RewardScriptFindings:
    """What a reward script's bytes and syntax tree say about its ability to score."""

    byte_count: int
    score_statements: tuple[str, ...]
    parse_error: str | None
    defects: tuple[str, ...]


@dataclass(frozen=True)
class StaticTaskFindings:
    """VM-free findings for one task in a pinned snapshot."""

    task_id: TaskId
    app_type: str
    platform: str | None
    setup_kind: str
    setup_delivery: str
    reward: RewardScriptFindings
    seeded_state_endpoints: tuple[str, ...]
    undocumented_state_keys: Mapping[str, tuple[str, ...]]

    @property
    def defects(self) -> tuple[str, ...]:
        return self.reward.defects

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "app_type": self.app_type,
            "platform": self.platform,
            "setup_kind": self.setup_kind,
            "setup_delivery": self.setup_delivery,
            "reward_script_bytes": self.reward.byte_count,
            "reward_score_statements": list(self.reward.score_statements),
            "reward_parse_error": self.reward.parse_error,
            "static_defects": list(self.reward.defects),
            "seeded_state_endpoints": list(self.seeded_state_endpoints),
            "undocumented_state_keys": {
                endpoint: list(keys)
                for endpoint, keys in sorted(self.undocumented_state_keys.items())
            },
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> StaticTaskFindings:
        undocumented = payload.get("undocumented_state_keys") or {}
        platform = payload.get("platform")
        return cls(
            task_id=TaskId(str(payload["task_id"])),
            app_type=str(payload["app_type"]),
            platform=None if platform is None else str(platform),
            setup_kind=str(payload["setup_kind"]),
            setup_delivery=str(payload["setup_delivery"]),
            reward=RewardScriptFindings(
                byte_count=int(payload["reward_script_bytes"]),
                score_statements=tuple(payload["reward_score_statements"]),
                parse_error=payload["reward_parse_error"],
                defects=tuple(payload["static_defects"]),
            ),
            seeded_state_endpoints=tuple(payload.get("seeded_state_endpoints") or ()),
            undocumented_state_keys={
                str(endpoint): tuple(keys) for endpoint, keys in undocumented.items()
            },
        )


@dataclass(frozen=True)
class ResetProbeRecord:
    """One task graded immediately after setup, with no agent action."""

    task_id: TaskId
    setup_completed: bool
    setup_failure: str | None
    pre_grade_completed: bool
    pre_grade_failure: str | None
    grader_exit_code: int | None
    grader_emitted_score: bool
    grader_score_at_reset: float | None
    grader_stdout_tail: str
    seconds: float
    error: str | None = None

    @property
    def pays_before_agent_acts(self) -> bool:
        return (
            self.grader_score_at_reset is not None and self.grader_score_at_reset > 0.0
        )

    @property
    def prepared(self) -> bool:
        return self.setup_completed and self.pre_grade_completed

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "setup_completed": self.setup_completed,
            "setup_failure": self.setup_failure,
            "pre_grade_completed": self.pre_grade_completed,
            "pre_grade_failure": self.pre_grade_failure,
            "grader_exit_code": self.grader_exit_code,
            "grader_emitted_score": self.grader_emitted_score,
            "grader_score_at_reset": self.grader_score_at_reset,
            "grader_stdout_tail": self.grader_stdout_tail,
            "pays_before_agent_acts": self.pays_before_agent_acts,
            "seconds": self.seconds,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ResetProbeRecord:
        score = payload["grader_score_at_reset"]
        exit_code = payload["grader_exit_code"]
        return cls(
            task_id=TaskId(str(payload["task_id"])),
            setup_completed=bool(payload["setup_completed"]),
            setup_failure=payload["setup_failure"],
            pre_grade_completed=bool(payload["pre_grade_completed"]),
            pre_grade_failure=payload["pre_grade_failure"],
            grader_exit_code=None if exit_code is None else int(exit_code),
            grader_emitted_score=bool(payload["grader_emitted_score"]),
            grader_score_at_reset=None if score is None else float(score),
            grader_stdout_tail=str(payload["grader_stdout_tail"]),
            seconds=float(payload["seconds"]),
            error=payload.get("error"),
        )


def read_snapshot_identity(
    config: DatasetSnapshotConfig, manifest: CompatibilityManifest
) -> SnapshotIdentity:
    """Digest the pinned snapshot files and fail closed on any drift."""

    if config.revision != manifest.revision:
        raise SnapshotValidationError(
            f"Configured revision {config.revision!r} does not match manifest "
            f"revision {manifest.revision!r}"
        )
    digests: dict[str, str] = {}
    for name, fingerprint in manifest.files.items():
        path = snapshot_file(config, manifest, name)
        if not path.is_file():
            raise SnapshotValidationError(f"Missing CUA-Gym snapshot file: {path}")
        digest = sha256_file(path)
        if digest != fingerprint.sha256:
            raise SnapshotValidationError(
                f"SHA-256 mismatch for {fingerprint.path}: "
                f"expected {fingerprint.sha256}, found {digest}"
            )
        digests[name] = digest
    return SnapshotIdentity(
        dataset=manifest.dataset, revision=manifest.revision, file_digests=digests
    )


def snapshot_file(
    config: DatasetSnapshotConfig, manifest: CompatibilityManifest, name: str
) -> Path:
    """Resolve one manifest-declared file inside the snapshot root."""

    try:
        relative = PurePosixPath(manifest.files[name].path)
    except KeyError as error:
        raise SnapshotValidationError(f"Manifest has no {name!r} file") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise SnapshotValidationError(
            f"Manifest file path must stay inside the snapshot: {relative}"
        )
    return config.snapshot_root.joinpath(*relative.parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata_rows(
    config: DatasetSnapshotConfig, manifest: CompatibilityManifest
) -> tuple[dict[str, Any], ...]:
    """Return every metadata row, including the non-web tasks the runtime skips."""

    table = pq.read_table(snapshot_file(config, manifest, "metadata"))
    rows = table.to_pylist()
    if len(rows) != manifest.metadata_row_count:
        raise SnapshotValidationError(
            f"Expected {manifest.metadata_row_count} metadata rows, found {len(rows)}"
        )
    return tuple(rows)


def iter_task_files(
    archive_path: Path, task_ids: frozenset[str] | None = None
) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Stream the artifact once, yielding each task's text-bearing bundle files."""

    pending: dict[str, dict[str, bytes]] = defaultdict(dict)
    open_task_id: str | None = None
    with pa.input_stream(str(archive_path), compression="zstd") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = member.name.split("/", 1)
                if len(parts) != 2:
                    continue
                task_id, name = parts
                if name.startswith("._") or task_id.startswith("._"):
                    continue
                if task_ids is not None and task_id not in task_ids:
                    continue
                if name not in _TASK_FILE_NAMES and not name.startswith(
                    "initial_setup."
                ):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SnapshotValidationError(f"Unreadable member: {member.name}")
                if task_id != open_task_id and open_task_id in pending:
                    yield open_task_id, pending.pop(open_task_id)
                open_task_id = task_id
                pending[task_id][name] = extracted.read()
    for task_id in list(pending):
        yield task_id, pending.pop(task_id)


def analyze_reward_script(content: bytes) -> RewardScriptFindings:
    """Classify a reward script by size, syntax, and score-emitting statements."""

    if not content.strip():
        return RewardScriptFindings(
            byte_count=len(content),
            score_statements=(),
            parse_error=None,
            defects=(EMPTY_REWARD_SCRIPT,),
        )
    try:
        source = content.decode("utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        return RewardScriptFindings(
            byte_count=len(content),
            score_statements=(),
            parse_error=f"{type(error).__name__}: {error}",
            defects=(UNPARSEABLE_REWARD_SCRIPT,),
        )
    statements = score_emitting_statements(tree)
    return RewardScriptFindings(
        byte_count=len(content),
        score_statements=statements,
        parse_error=None,
        defects=() if statements else (REWARD_SCRIPT_EMITS_NO_SCORE,),
    )


def score_emitting_statements(tree: ast.AST) -> tuple[str, ...]:
    """Return the kinds of executable statement that could write a score to stdout."""

    kinds: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "print":
            kinds.add("print")
        elif isinstance(function, ast.Attribute) and function.attr == "write":
            if _is_stdout_expression(function.value):
                kinds.add("stdout_write")
    return tuple(sorted(kinds))


def _is_stdout_expression(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id == "stdout"
    return isinstance(expression, ast.Attribute) and expression.attr == "stdout"


def classify_setup_delivery(setup_kind: str) -> str:
    """Name how a task's start state arrives: generated by a script, or shipped."""

    if setup_kind in SCRIPT_SETUP_KINDS:
        return "script"
    if setup_kind in SHIPPED_DOCUMENT_SETUP_KINDS:
        return "shipped_document"
    return "unknown"


def seeded_state_keys(
    setup_source: str, known_endpoints: frozenset[str]
) -> dict[str, frozenset[str]]:
    """Map each mock endpoint to the top-level state keys a setup seeds into it."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(setup_source)
    dict_bindings: dict[str, ast.Dict] = {}
    string_bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, ast.Dict):
                dict_bindings[target.id] = node.value
            literal = _static_string(node.value, string_bindings)
            if literal is not None:
                string_bindings[target.id] = literal

    seeded: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        endpoint = _seeding_endpoint(node, string_bindings, known_endpoints)
        if endpoint is None:
            continue
        assert isinstance(node, ast.Call)
        payload = _resolve_dict(_keyword_value(node, ("json", "data")), dict_bindings)
        if payload is None:
            continue
        entries = _string_keyed_entries(payload)
        action = entries.get("action")
        if not (isinstance(action, ast.Constant) and action.value == "set"):
            continue
        state = _resolve_dict(entries.get("state"), dict_bindings)
        if state is None:
            continue
        seeded[endpoint].update(_string_keyed_entries(state))
    return {endpoint: frozenset(keys) for endpoint, keys in seeded.items()}


def _seeding_endpoint(
    node: ast.AST, string_bindings: dict[str, str], known_endpoints: frozenset[str]
) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in {"post", "put"}:
        return None
    url = _static_string(node.args[0], string_bindings) if node.args else None
    if url is None:
        url = _static_string(_keyword_value(node, ("url",)), string_bindings)
    if url is None:
        return None
    match = _ENDPOINT_PLACEHOLDER_RE.search(url)
    if match is None:
        return None
    endpoint = match.group(1).lower()
    return endpoint if endpoint in known_endpoints else None


def _keyword_value(node: ast.Call, names: tuple[str, ...]) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg in names:
            return keyword.value
    return None


def _resolve_dict(
    expression: ast.expr | None, bindings: dict[str, ast.Dict]
) -> ast.Dict | None:
    if isinstance(expression, ast.Dict):
        return expression
    if isinstance(expression, ast.Name):
        return bindings.get(expression.id)
    return None


def _string_keyed_entries(node: ast.Dict) -> dict[str, ast.expr]:
    return {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _static_string(expression: ast.expr | None, bindings: dict[str, str]) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.Name):
        return bindings.get(expression.id)
    if isinstance(expression, ast.JoinedStr):
        parts: list[str] = []
        for value in expression.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append(_static_string(value.value, bindings) or "")
        return "".join(parts) or None
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _static_string(expression.left, bindings) or ""
        right = _static_string(expression.right, bindings) or ""
        return f"{left}{right}" or None
    return None


def documented_state_keys(schema_markdown: str) -> frozenset[str]:
    """Read a mock's canonical top-level state keys from its SCHEMA.md table."""

    keys: set[str] = set()
    inside = False
    table_rows = 0
    for line in schema_markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            inside = heading.startswith(_STATE_SCHEMA_HEADING)
            table_rows = 0
            continue
        if not inside or not stripped.startswith("|"):
            continue
        table_rows += 1
        if table_rows <= 2:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        match = _SCHEMA_KEY_RE.fullmatch(cells[0]) if cells else None
        if match is not None:
            keys.add(match.group(1))
    return frozenset(keys)


def load_documented_schemas(hub_checkout: Path) -> dict[str, frozenset[str]]:
    """Index every mock's documented state keys by its endpoint name."""

    schemas: dict[str, frozenset[str]] = {}
    websites = hub_checkout / "websites"
    if not websites.is_dir():
        raise SnapshotValidationError(f"Hub checkout has no websites/: {hub_checkout}")
    for schema_path in sorted(websites.glob("*/SCHEMA.md")):
        endpoint = schema_path.parent.name.removesuffix("_mock").lower()
        keys = documented_state_keys(schema_path.read_text(encoding="utf-8"))
        if keys:
            schemas[endpoint] = keys
    return schemas


def undocumented_state_keys(
    seeded: Mapping[str, frozenset[str]],
    documented: Mapping[str, frozenset[str]],
) -> dict[str, tuple[str, ...]]:
    """Return per-endpoint seeded keys that the mock's SCHEMA.md does not document."""

    deviations: dict[str, tuple[str, ...]] = {}
    for endpoint, keys in seeded.items():
        expected = documented.get(endpoint)
        if expected is None:
            continue
        extra = tuple(sorted(keys - expected))
        if extra:
            deviations[endpoint] = extra
    return deviations


def analyze_task(
    row: Mapping[str, Any],
    files: Mapping[str, bytes],
    known_endpoints: frozenset[str],
    documented: Mapping[str, frozenset[str]],
) -> StaticTaskFindings:
    """Run every VM-free check over one task's metadata row and bundle files."""

    setup_kind = str(row["setup_kind"])
    reward = analyze_reward_script(files.get("reward.py", b""))
    endpoints: dict[str, frozenset[str]] = {}
    deviations: dict[str, tuple[str, ...]] = {}
    setup_source = files.get("initial_setup.py")
    if setup_source is not None:
        try:
            source = setup_source.decode("utf-8")
            endpoints = seeded_state_keys(source, known_endpoints)
            deviations = undocumented_state_keys(endpoints, documented)
        except (UnicodeDecodeError, SyntaxError, ValueError):
            endpoints = {}
            deviations = {}
    platform = row.get("platform")
    return StaticTaskFindings(
        task_id=TaskId(str(row["id"])),
        app_type=str(row["app_type"]),
        platform=None if platform is None else str(platform),
        setup_kind=setup_kind,
        setup_delivery=classify_setup_delivery(setup_kind),
        reward=reward,
        seeded_state_endpoints=tuple(sorted(endpoints)),
        undocumented_state_keys=deviations,
    )


def shard_task_ids(
    task_ids: Sequence[str], shard: int, shard_count: int
) -> tuple[str, ...]:
    """Split a deterministic task order into one interleaved shard."""

    if shard_count < 1 or not 0 <= shard < shard_count:
        raise ValueError("shard must satisfy 0 <= shard < shard_count")
    ordered = sorted(task_ids)
    return tuple(ordered[index] for index in range(shard, len(ordered), shard_count))


def journal_header(
    identity: SnapshotIdentity, kind: str, *, grading: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Header for a journal. `grading` records the conditions a probe measured
    under, since a score is only comparable to a runtime that grades the same way."""

    header: dict[str, Any] = {
        "record": JOURNAL_HEADER_RECORD,
        "kind": kind,
        **identity.to_json(),
    }
    if grading is not None:
        header["grading"] = dict(grading)
    return header


def append_journal_line(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_journal(
    path: Path, header: Mapping[str, Any], records: Iterable[Mapping[str, Any]]
) -> int:
    """Rewrite a journal in one pass, for findings cheap enough to redo wholesale."""

    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
    return written


def read_journal(path: Path, identity: SnapshotIdentity | None) -> list[dict[str, Any]]:
    """Read one journal file, rejecting records measured against another snapshot."""

    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("record") == JOURNAL_HEADER_RECORD:
            header = SnapshotIdentity.from_json(payload)
            if identity is not None and header != identity:
                raise SnapshotValidationError(
                    f"{path} was measured against {header.revision}, not "
                    f"{identity.revision}"
                )
            continue
        records.append(payload)
    return records


def read_journal_identity(path: Path) -> SnapshotIdentity:
    """Read the snapshot a journal file was measured against."""

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("record") == JOURNAL_HEADER_RECORD:
                return SnapshotIdentity.from_json(payload)
            break
    raise SnapshotValidationError(f"{path} has no snapshot header")


def read_probe_journals(
    journal_dir: Path, identity: SnapshotIdentity | None
) -> dict[TaskId, ResetProbeRecord]:
    """Merge every reset-probe shard in a journal directory, last record winning."""

    merged: dict[TaskId, ResetProbeRecord] = {}
    for path in sorted(journal_dir.glob("reset_probe.*.jsonl")):
        for payload in read_journal(path, identity):
            record = ResetProbeRecord.from_json(payload)
            merged[record.task_id] = record
    return merged


def read_static_journal(
    journal_dir: Path, identity: SnapshotIdentity | None
) -> dict[TaskId, StaticTaskFindings]:
    findings: dict[TaskId, StaticTaskFindings] = {}
    for payload in read_journal(journal_dir / "static.jsonl", identity):
        record = StaticTaskFindings.from_json(payload)
        findings[record.task_id] = record
    return findings


def build_blocklist(
    identity: SnapshotIdentity,
    static_findings: Mapping[TaskId, StaticTaskFindings],
    probes: Mapping[TaskId, ResetProbeRecord],
) -> dict[str, Any]:
    """Merge static findings and reset probes into the checked-in blocklist document."""

    blocked: dict[str, dict[str, Any]] = {}
    for task_id, findings in static_findings.items():
        if findings.defects:
            blocked[str(task_id)] = {"reasons": list(findings.defects)}
    for task_id, probe in probes.items():
        if not probe.pays_before_agent_acts:
            continue
        entry = blocked.setdefault(str(task_id), {"reasons": []})
        entry["reasons"] = sorted({*entry["reasons"], GRADER_PAYS_BEFORE_AGENT_ACTS})
        entry["grader_score_at_reset"] = probe.grader_score_at_reset

    graded = [probe for probe in probes.values() if probe.grader_emitted_score]
    advisories = {
        "reset_probe_emitted_no_score": sorted(
            str(probe.task_id)
            for probe in probes.values()
            if not probe.grader_emitted_score
        ),
        "reset_preparation_did_not_complete": sorted(
            str(probe.task_id) for probe in probes.values() if not probe.prepared
        ),
        "undocumented_seeded_state_keys": {
            str(task_id): {
                endpoint: list(keys)
                for endpoint, keys in sorted(findings.undocumented_state_keys.items())
            }
            for task_id, findings in sorted(static_findings.items())
            if findings.undocumented_state_keys
        },
    }
    return {
        "blocklist_version": BLOCKLIST_VERSION,
        **identity.to_json(),
        "measured": {
            "static_checks": list(STATIC_DEFECTS),
            "static_task_count": len(static_findings),
            "reset_probe_task_count": len(probes),
            "reset_probe_graded_task_count": len(graded),
            "reset_probe_task_ids": sorted(str(task_id) for task_id in probes),
        },
        "meaning": {
            "blocked_tasks": (
                "Tasks with a measured defect that makes them useless for GRPO."
            ),
            "not_blocked": NOT_BLOCKED_MEANING,
            "grader_score_at_reset": RESET_PROBE_MEANING,
            "reset_probe_task_ids": (
                "Exactly the tasks a reset probe ran on. Every other task's grader "
                "was never run, so nothing is known about what it pays at reset."
            ),
            "advisories": (
                "Observations that are not proof of a task defect on their own."
            ),
        },
        "blocked_tasks": dict(sorted(blocked.items())),
        "advisories": advisories,
    }
