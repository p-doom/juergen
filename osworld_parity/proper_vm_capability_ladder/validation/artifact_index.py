from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "proper_vm_executor_artifact_index_v1"
BUILD_SCHEMA_VERSION = "proper_vm_executor_build_v1"
PINNED_SUBSTRATE_SHA256 = {
    "provider": "76a8f44fab16c6dd38a4378a270e38758ba8d31885f244baedb95d8178f588d7",
    "qemu_wrapper": "7e8c97b98d1b31448dc9053b6712b27d871ba8e0c21f19dc7a10a8c3ef6b9f56",
    "qemu_binary": "467326456c87802faa98ea60eebe6356b5079fbaff85b764ec42e8ffff97c0d1",
    "qemu_loader": "8c1cf687490f3f7858528ceb4e651b979f625e43396431bec65e5bb89cb63860",
    "base_qcow": "6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313",
}
PINNED_BASE_QCOW_SIZE = 24_460_197_888


class ArtifactIndexError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ArtifactIndexError(f"indexed input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIndexError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactIndexError(f"JSON value is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def _named_paths(values: Iterable[str], *, option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path:
            raise ArtifactIndexError(f"{option} must be NAME=/absolute/path: {raw!r}")
        if name in parsed:
            raise ArtifactIndexError(f"duplicate {option} name: {name}")
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ArtifactIndexError(f"{option} path must be absolute: {candidate}")
        parsed[name] = candidate.resolve()
    return parsed


def _file_records(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }


def _input_bindings(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = context.get("inputs")
    if not isinstance(raw, list):
        raise ArtifactIndexError("labctl context inputs must be a list")
    bindings: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ArtifactIndexError("labctl context contains an invalid input record")
        role = item.get("role")
        resolved = item.get("resolved_path")
        artifact_id = item.get("artifact_id")
        if not isinstance(role, str) or not role:
            raise ArtifactIndexError("labctl input record has no role")
        if role in bindings:
            raise ArtifactIndexError(f"labctl context contains duplicate input role: {role}")
        if not isinstance(resolved, str) or not Path(resolved).is_absolute():
            raise ArtifactIndexError(f"labctl input {role} has no absolute resolved path")
        if artifact_id is not None and not isinstance(artifact_id, str):
            raise ArtifactIndexError(f"labctl input {role} has an invalid artifact id")
        bindings[role] = {
            "artifact_id": artifact_id,
            "resolved_path": resolved,
        }
    return bindings


def _context_provenance(context: dict[str, Any], lock_file: Path) -> dict[str, Any]:
    provenance = context.get("provenance")
    if not isinstance(provenance, dict):
        raise ArtifactIndexError("labctl context has no provenance object")
    commit = provenance.get("git_head")
    source_hash = context.get("source_hash")
    status = provenance.get("git_status_porcelain")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ArtifactIndexError("labctl context has no exact git HEAD")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ArtifactIndexError("labctl context has no source SHA256")
    if status != "":
        raise ArtifactIndexError("executor certification source was not clean at submission")
    repo_path = provenance.get("repo_path")
    if not isinstance(repo_path, str):
        raise ArtifactIndexError("labctl context has no submitted repository path")
    try:
        tree = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", f"{commit}^{{tree}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactIndexError("cannot resolve the submitted commit tree") from exc
    if len(tree) != 40:
        raise ArtifactIndexError("submitted commit has no exact git tree")
    lock_sha = sha256_file(lock_file)
    if provenance.get("uv_lock_hash") != lock_sha:
        raise ArtifactIndexError("snapshot uv.lock differs from submitted lock hash")
    recipe_hash = context.get("recipe_hash")
    if not isinstance(recipe_hash, str) or len(recipe_hash) != 64:
        raise ArtifactIndexError("labctl context has no recipe SHA256")
    return {
        "git_commit": commit,
        "git_tree": tree,
        "source_sha256": source_hash,
        "git_status_porcelain": status,
        "tracked_patch_sha256": provenance.get("diff_hash"),
        "untracked_patch_sha256": provenance.get("untracked_files_hash"),
        "lock": {
            "path": str(lock_file),
            "sha256": lock_sha,
            "submitted_sha256": provenance.get("uv_lock_hash"),
        },
        "recipe_name": context.get("recipe_name"),
        "recipe_sha256": recipe_hash,
    }


def _substrate_records(
    *, provider: Path, qemu_components: Mapping[str, Path], base_image: Path
) -> dict[str, Any]:
    if set(qemu_components) != {"wrapper", "binary", "loader"}:
        raise ArtifactIndexError("QEMU components must be exactly wrapper, binary, and loader")
    records = {
        "provider": {
            "path": str(provider),
            "size": provider.stat().st_size,
            "sha256": sha256_file(provider),
        },
        "qemu": _file_records(qemu_components),
        "base_qcow": {
            "path": str(base_image),
            "size": base_image.stat().st_size,
            "sha256": sha256_file(base_image),
        },
    }
    observed = {
        "provider": records["provider"]["sha256"],
        "qemu_wrapper": records["qemu"]["wrapper"]["sha256"],
        "qemu_binary": records["qemu"]["binary"]["sha256"],
        "qemu_loader": records["qemu"]["loader"]["sha256"],
        "base_qcow": records["base_qcow"]["sha256"],
    }
    if observed != PINNED_SUBSTRATE_SHA256:
        raise ArtifactIndexError(
            f"external VM substrate hash mismatch: {observed!r} != {PINNED_SUBSTRATE_SHA256!r}"
        )
    if records["base_qcow"]["size"] != PINNED_BASE_QCOW_SIZE:
        raise ArtifactIndexError("base qcow size differs from the pinned image")
    return records


def write_build_result(
    *,
    path: Path,
    context: dict[str, Any],
    lock_file: Path,
    commands: list[str],
    junit_xml: Path,
) -> dict[str, Any]:
    provenance = _context_provenance(context, lock_file)
    try:
        root = ET.parse(junit_xml).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ArtifactIndexError(f"cannot parse pytest JUnit evidence: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    if tests < 109 or failures or errors:
        raise ArtifactIndexError(
            f"build suite gate failed: tests={tests} failures={failures} errors={errors}"
        )
    value = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "status": "passed",
        "git_commit": provenance["git_commit"],
        "git_tree": provenance["git_tree"],
        "source_sha256": provenance["source_sha256"],
        "lock_sha256": provenance["lock"]["sha256"],
        "commands": commands,
        "baseline_test_count": 109,
        "current_test_count": tests,
        "failure_count": failures,
        "error_count": errors,
        "skipped_count": skipped,
        "junit_sha256": sha256_file(junit_xml),
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }
    _atomic_json(path, value)
    return value


def create_index(
    *,
    kind: str,
    context: dict[str, Any],
    lock_file: Path,
    results: Mapping[str, Path],
    fixtures: Mapping[str, Path],
    recipes: Mapping[str, Path],
    commands: list[str],
    provider: Path | None = None,
    qemu_components: Mapping[str, Path] | None = None,
    base_image: Path | None = None,
    vm_metadata_path: Path | None = None,
    build_result: Path | None = None,
) -> dict[str, Any]:
    if not results:
        raise ArtifactIndexError("at least one explicit terminal result is required")
    if not commands or any(not command.strip() for command in commands):
        raise ArtifactIndexError("at least one nonempty executed command is required")
    provenance = _context_provenance(context, lock_file)
    is_vm = vm_metadata_path is not None
    supplied_substrate = (provider, qemu_components, base_image)
    if is_vm != all(value is not None for value in supplied_substrate):
        raise ArtifactIndexError("VM indexes require metadata and the complete external substrate")
    substrate = None
    vm_metadata = None
    if is_vm:
        assert provider is not None and qemu_components is not None and base_image is not None
        substrate = _substrate_records(
            provider=provider, qemu_components=qemu_components, base_image=base_image
        )
        assert vm_metadata_path is not None
        vm_metadata = _load_object(vm_metadata_path)
        if vm_metadata.get("schema_version") != "proper_vm_isolation_v1":
            raise ArtifactIndexError("VM metadata has the wrong schema")
        if vm_metadata.get("closed") is not True:
            raise ArtifactIndexError("VM metadata was indexed before clean VM shutdown")
        if vm_metadata.get("cuda_visible_devices") != "":
            raise ArtifactIndexError("VM metadata reports visible CUDA devices")
        if vm_metadata.get("one_vm_per_task") is not True:
            raise ArtifactIndexError("VM metadata does not prove one VM per task")
        if vm_metadata.get("cleanup_errors") != []:
            raise ArtifactIndexError("VM metadata reports cleanup errors")
        overlay = vm_metadata.get("overlay")
        if (
            not isinstance(overlay, dict)
            or overlay.get("removed") is not True
            or overlay.get("job_unique_scratch") is not True
        ):
            raise ArtifactIndexError("VM overlay cleanup/job isolation is unproven")
        ports = vm_metadata.get("ports")
        if not isinstance(ports, dict) or len(ports) != 4:
            raise ArtifactIndexError("VM metadata has no complete port map")
        parsed_ports = [int(value) for value in ports.values()]
        if len(set(parsed_ports)) != 4 or any(not 1024 <= port <= 65535 for port in parsed_ports):
            raise ArtifactIndexError("VM metadata contains colliding or invalid host ports")
    build_dependency = None
    if build_result is not None:
        build_payload = _load_object(build_result)
        if (
            build_payload.get("schema_version") != BUILD_SCHEMA_VERSION
            or build_payload.get("status") != "passed"
        ):
            raise ArtifactIndexError("build dependency is not a passed executor build")
        if build_payload.get("git_commit") != provenance["git_commit"]:
            raise ArtifactIndexError("build dependency commit differs from this run")
        if build_payload.get("git_tree") != provenance["git_tree"]:
            raise ArtifactIndexError("build dependency tree differs from this run")
        if build_payload.get("source_sha256") != provenance["source_sha256"]:
            raise ArtifactIndexError("build dependency source hash differs from this run")
        build_dependency = {
            "path": str(build_result),
            "sha256": sha256_file(build_result),
        }
    input_bindings = _input_bindings(context)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source": {
            key: provenance[key]
            for key in (
                "git_commit",
                "git_tree",
                "source_sha256",
                "git_status_porcelain",
                "tracked_patch_sha256",
                "untracked_patch_sha256",
            )
        },
        "lock": provenance["lock"],
        "submission": {
            "run_id": context.get("run_id"),
            "recipe_name": provenance["recipe_name"],
            "recipe_sha256": provenance["recipe_sha256"],
            "stage_name": context.get("stage_name"),
            "job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("LABCTL_JOB_ID")),
            "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
            "input_artifacts": input_bindings,
        },
        "recipe_files": _file_records(recipes),
        "fixtures": _file_records(fixtures),
        "commands": list(commands),
        "terminal_results": _file_records(results),
        "build_dependency": build_dependency,
        "substrate": substrate,
        "vm": vm_metadata,
    }
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return {**payload, "content_address": f"sha256:{digest}"}


def validate_index(value: dict[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactIndexError("artifact index has the wrong schema")
    address = value.get("content_address")
    if not isinstance(address, str) or not address.startswith("sha256:"):
        raise ArtifactIndexError("artifact index has no content address")
    payload = dict(value)
    del payload["content_address"]
    expected = "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if address != expected:
        raise ArtifactIndexError("artifact index content address mismatch")
    terminal = value.get("terminal_results")
    if not isinstance(terminal, dict) or not terminal:
        raise ArtifactIndexError("artifact index has no terminal results")
    for record in terminal.values():
        if not isinstance(record, dict):
            raise ArtifactIndexError("artifact index terminal record is invalid")
        path = Path(str(record.get("path", "")))
        if sha256_file(path) != record.get("sha256"):
            raise ArtifactIndexError(f"terminal result hash changed: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--context", type=Path, default=os.environ.get("LABCTL_CONTEXT"))
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--recipe", action="append", default=[])
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--qemu-component", action="append", default=[])
    parser.add_argument("--base-image", type=Path)
    parser.add_argument("--vm-metadata", type=Path)
    parser.add_argument("--build-result", type=Path)
    parser.add_argument("--write-build-result", type=Path)
    parser.add_argument("--junit-xml", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.context is None:
        raise ArtifactIndexError("--context or LABCTL_CONTEXT is required")
    context = _load_object(args.context.resolve())
    lock_file = args.lock_file.resolve()
    commands = list(args.command)
    if args.write_build_result is not None:
        if args.junit_xml is None:
            raise ArtifactIndexError("--write-build-result requires --junit-xml")
        write_build_result(
            path=args.write_build_result.resolve(),
            context=context,
            lock_file=lock_file,
            commands=commands,
            junit_xml=args.junit_xml.resolve(),
        )
    value = create_index(
        kind=args.kind,
        context=context,
        lock_file=lock_file,
        results=_named_paths(args.result, option="--result"),
        fixtures=_named_paths(args.fixture, option="--fixture"),
        recipes=_named_paths(args.recipe, option="--recipe"),
        commands=commands,
        provider=args.provider.resolve() if args.provider else None,
        qemu_components=_named_paths(args.qemu_component, option="--qemu-component"),
        base_image=args.base_image.resolve() if args.base_image else None,
        vm_metadata_path=args.vm_metadata.resolve() if args.vm_metadata else None,
        build_result=args.build_result.resolve() if args.build_result else None,
    )
    validate_index(value)
    _atomic_json(args.output_index.resolve(), value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
