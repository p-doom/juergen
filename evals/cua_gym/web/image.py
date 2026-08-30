"""Pinned CUA-Gym-Hub image build and provenance contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .manifest import CuaGymWebRuntimeManifest, load_default_web_runtime_manifest

HUB_BASE_IMAGE = (
    "node@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5"
)
HUB_IMAGE_MANIFEST_VERSION = 1
_LEGACY_FUNCTION = """function isLegacyCompatEnabled() {
  return process.env.CUA_GYM_LEGACY_COMPAT !== '0' && process.env.CUA_GYM_LEGACY_COMPAT !== 'false';
}

"""
_LEGACY_BRANCH = "        if (isLegacyCompatEnabled()) return next();\n"
HUB_SOURCE_PATCH_SHA256 = hashlib.sha256(
    (_LEGACY_FUNCTION + _LEGACY_BRANCH * 3).encode("utf-8")
).hexdigest()
_MAX_MANIFEST_BYTES = 65_536
_MANIFEST_KEYS = {
    "apps_sha256",
    "base_image",
    "dataset_revision",
    "definition_sha256",
    "hub_revision",
    "image_bytes",
    "image_sha256",
    "manifest_version",
    "source_patch_sha256",
}


@dataclass(frozen=True)
class CuaGymHubImageManifest:
    manifest_version: int
    dataset_revision: str
    hub_revision: str
    base_image: str
    definition_sha256: str
    apps_sha256: str
    source_patch_sha256: str
    image_sha256: str
    image_bytes: int

    def __post_init__(self) -> None:
        if self.manifest_version != HUB_IMAGE_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported CUA-Gym Hub image manifest: {self.manifest_version!r}"
            )
        if self.base_image != HUB_BASE_IMAGE:
            raise ValueError("CUA-Gym Hub image base does not match the pinned digest")
        for name in (
            "dataset_revision",
            "hub_revision",
            "definition_sha256",
            "apps_sha256",
            "source_patch_sha256",
            "image_sha256",
        ):
            value = getattr(self, name)
            expected_length = 40 if name.endswith("revision") else 64
            if not isinstance(value, str) or len(value) != expected_length or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"CUA-Gym Hub image manifest has invalid {name}")
        if type(self.image_bytes) is not int or self.image_bytes < 1:
            raise ValueError("CUA-Gym Hub image size must be positive")

    @classmethod
    def for_image(
        cls,
        image_path: Path,
        *,
        web_manifest: CuaGymWebRuntimeManifest,
    ) -> CuaGymHubImageManifest:
        definition = render_hub_definition(web_manifest).encode("utf-8")
        apps = render_hub_apps(web_manifest).encode("utf-8")
        return cls(
            manifest_version=HUB_IMAGE_MANIFEST_VERSION,
            dataset_revision=web_manifest.dataset_revision,
            hub_revision=web_manifest.hub_revision,
            base_image=HUB_BASE_IMAGE,
            definition_sha256=hashlib.sha256(definition).hexdigest(),
            apps_sha256=hashlib.sha256(apps).hexdigest(),
            source_patch_sha256=HUB_SOURCE_PATCH_SHA256,
            image_sha256=_file_sha256(image_path),
            image_bytes=image_path.stat().st_size,
        )

    @classmethod
    def read(cls, path: Path) -> CuaGymHubImageManifest:
        if path.is_symlink():
            raise ValueError("CUA-Gym Hub image manifest must not be a symlink")
        raw = path.read_bytes()
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise ValueError("CUA-Gym Hub image manifest is too large")
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("CUA-Gym Hub image manifest is not valid JSON") from None
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
            raise ValueError("CUA-Gym Hub image manifest has an unexpected schema")
        return cls(
            manifest_version=_integer(payload, "manifest_version"),
            dataset_revision=_string(payload, "dataset_revision"),
            hub_revision=_string(payload, "hub_revision"),
            base_image=_string(payload, "base_image"),
            definition_sha256=_string(payload, "definition_sha256"),
            apps_sha256=_string(payload, "apps_sha256"),
            source_patch_sha256=_string(payload, "source_patch_sha256"),
            image_sha256=_string(payload, "image_sha256"),
            image_bytes=_integer(payload, "image_bytes"),
        )

    def validate(
        self,
        image_path: Path,
        *,
        web_manifest: CuaGymWebRuntimeManifest,
    ) -> None:
        expected = CuaGymHubImageManifest.for_image(
            image_path,
            web_manifest=web_manifest,
        )
        if self != expected:
            raise RuntimeError("CUA-Gym Hub image provenance or SHA-256 differs")

    def write(self, path: Path) -> None:
        payload = (json.dumps(asdict(self), indent=2, sort_keys=True) + "\n").encode()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CuaGymHubImageBuildConfig:
    hub_checkout: Path
    output_image: Path
    apptainer_binary: Path

    def __post_init__(self) -> None:
        for name in ("hub_checkout", "output_image", "apptainer_binary"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
            object.__setattr__(self, name, path)

    @property
    def output_manifest(self) -> Path:
        return hub_image_manifest_path(self.output_image)


class CuaGymHubImageProducer:
    def __init__(self, config: CuaGymHubImageBuildConfig) -> None:
        self.config = config
        self.web_manifest = load_default_web_runtime_manifest()

    def build(self) -> CuaGymHubImageManifest:
        self._validate_inputs()
        config = self.config
        config.output_image.parent.mkdir(parents=True, exist_ok=True)
        source_epoch = self._source_epoch()
        with tempfile.TemporaryDirectory(
            dir=config.output_image.parent,
            prefix=".cua-gym-hub-build-",
        ) as temporary_name:
            workspace = Path(temporary_name)
            archive_path = workspace / "hub.tar"
            source_root = workspace / "hub"
            source_root.mkdir()
            self._run(
                [
                    "git",
                    "-C",
                    str(config.hub_checkout),
                    "archive",
                    "--format=tar",
                    "--output",
                    str(archive_path),
                    self.web_manifest.hub_revision,
                ]
            )
            with tarfile.open(archive_path) as archive:
                archive.extractall(source_root, filter="data")
            archive_path.unlink()
            _remove_legacy_compatibility(source_root)

            definition_path = workspace / "cua_gym_hub.def"
            apps_path = workspace / "apps.txt"
            definition_path.write_text(
                render_hub_definition(self.web_manifest), encoding="utf-8"
            )
            apps_path.write_text(render_hub_apps(self.web_manifest), encoding="utf-8")
            partial_image = workspace / "cua-gym-hub.sif"
            environment = dict(os.environ)
            environment.update(
                {
                    "LC_ALL": "C",
                    "SOURCE_DATE_EPOCH": source_epoch,
                    "TZ": "UTC",
                }
            )
            self._run(
                [
                    str(config.apptainer_binary),
                    "build",
                    str(partial_image),
                    str(definition_path),
                ],
                cwd=workspace,
                env=environment,
            )
            if not partial_image.is_file() or partial_image.stat().st_size == 0:
                raise RuntimeError("Apptainer did not produce a CUA-Gym Hub image")
            partial_image.chmod(0o444)
            image_manifest = CuaGymHubImageManifest.for_image(
                partial_image,
                web_manifest=self.web_manifest,
            )
            image_manifest.validate(partial_image, web_manifest=self.web_manifest)
            os.replace(partial_image, config.output_image)
            image_manifest.write(config.output_manifest)
            return image_manifest

    def _validate_inputs(self) -> None:
        config = self.config
        if not config.hub_checkout.is_dir():
            raise FileNotFoundError(
                f"CUA-Gym-Hub checkout is missing: {config.hub_checkout}"
            )
        if not config.apptainer_binary.is_file() or not os.access(
            config.apptainer_binary, os.X_OK
        ):
            raise FileNotFoundError(
                f"Apptainer executable is missing: {config.apptainer_binary}"
            )
        for output in (config.output_image, config.output_manifest):
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"CUA-Gym Hub image output exists: {output}")
        revision = self._git_output(["rev-parse", "HEAD"])
        if revision != self.web_manifest.hub_revision:
            raise ValueError(
                "CUA-Gym-Hub checkout revision differs from the pinned web manifest"
            )

    def _source_epoch(self) -> str:
        value = self._git_output(
            ["show", "-s", "--format=%ct", self.web_manifest.hub_revision]
        )
        if not value.isdecimal():
            raise RuntimeError("CUA-Gym-Hub commit timestamp is invalid")
        return value

    def _git_output(self, arguments: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.config.hub_checkout), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not inspect CUA-Gym-Hub: {result.stderr.strip()}")
        return result.stdout.strip()

    @staticmethod
    def _run(
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"CUA-Gym Hub image command failed: {detail}")


def hub_image_manifest_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".manifest.json")


def render_hub_apps(manifest: CuaGymWebRuntimeManifest) -> str:
    return "".join(f"{app}\n" for app in manifest.apps)


def render_hub_definition(manifest: CuaGymWebRuntimeManifest) -> str:
    return f"""Bootstrap: docker
From: {HUB_BASE_IMAGE}

%files
    hub /opt/cua-gym-hub
    apps.txt /opt/cua-gym-apps.txt

%post
    set -eu
    mkdir -p /var/lib/cua-gym
    while IFS= read -r app; do
        app_root="/opt/cua-gym-hub/websites/$app"
        test -f "$app_root/package-lock.json"
        npm --prefix "$app_root" ci --no-audit --no-fund
        npm --prefix "$app_root" run build
        for directory in .mock-secure-states .mock-states .mock-state .mock-files .mock-uploads; do
            rm -rf "$app_root/$directory"
            ln -s "/var/lib/cua-gym/$app/$directory" "$app_root/$directory"
        done
    done < /opt/cua-gym-apps.txt
    npm cache clean --force
    chmod -R a+rX /opt/cua-gym-hub

%environment
    export CUA_GYM_HARDENED=1

%labels
    org.opencontainers.image.title CUA-Gym-Hub browser mocks
    org.opencontainers.image.source https://github.com/xlang-ai/CUA-Gym-Hub
    org.opencontainers.image.revision {manifest.hub_revision}
    org.opencontainers.image.base.digest {HUB_BASE_IMAGE.removeprefix("node@")}
    io.juergen.cua-gym.source-patch.sha256 {HUB_SOURCE_PATCH_SHA256}

%runscript
    exec npm "$@"
"""


def _file_sha256(path: Path) -> str:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("CUA-Gym Hub image must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat(follow_symlinks=False)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ):
        raise RuntimeError("CUA-Gym Hub image changed while hashing")
    return digest.hexdigest()


def _remove_legacy_compatibility(source_root: Path) -> None:
    plugin = source_root / "shared" / "secureMockApiPlugin.mjs"
    source = plugin.read_text(encoding="utf-8")
    if source.count(_LEGACY_FUNCTION) != 1 or source.count(_LEGACY_BRANCH) != 3:
        raise RuntimeError("Pinned CUA-Gym-Hub compatibility patch no longer applies")
    plugin.write_text(
        source.replace(_LEGACY_FUNCTION, "").replace(_LEGACY_BRANCH, ""),
        encoding="utf-8",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("CUA-Gym Hub image manifest has a duplicate key")
        payload[key] = value
    return payload


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"CUA-Gym Hub image manifest has invalid {key}")
    return value


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"CUA-Gym Hub image manifest has invalid {key}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-checkout", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--apptainer-binary", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = CuaGymHubImageProducer(
        CuaGymHubImageBuildConfig(
            hub_checkout=arguments.hub_checkout,
            output_image=arguments.output_image,
            apptainer_binary=arguments.apptainer_binary,
        )
    ).build()
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
