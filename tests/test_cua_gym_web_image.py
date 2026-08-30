from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from evals.cua_gym.models import EndpointName
from evals.cua_gym.web.image import (
    HUB_BASE_IMAGE,
    HUB_SOURCE_PATCH_SHA256,
    CuaGymHubImageBuildConfig,
    CuaGymHubImageManifest,
    CuaGymHubImageProducer,
    hub_image_manifest_path,
    render_hub_apps,
    render_hub_definition,
)
from evals.cua_gym.web.manifest import CuaGymWebRuntimeManifest


def _manifest(hub_revision: str = "e" * 40) -> CuaGymWebRuntimeManifest:
    return CuaGymWebRuntimeManifest(
        manifest_version=1,
        dataset_revision="d" * 40,
        hub_revision=hub_revision,
        endpoint_apps={EndpointName("gmail"): "gmail_mock"},
        writable_directories=(".mock-states",),
        unsupported_endpoints={},
        unsupported_tasks={},
        task_scratch_paths=("/tmp/task_sid",),
        supported_task_count=1,
    )


def test_hub_definition_and_app_set_are_one_pinned_contract() -> None:
    manifest = _manifest()
    definition = render_hub_definition(manifest)

    assert f"From: {HUB_BASE_IMAGE}" in definition
    assert f"org.opencontainers.image.revision {manifest.hub_revision}" in definition
    assert "export CUA_GYM_HARDENED=1" in definition
    assert "CUA_GYM_LEGACY_COMPAT" not in definition
    assert HUB_SOURCE_PATCH_SHA256 in definition
    assert render_hub_apps(manifest) == "gmail_mock\n"


def test_image_manifest_roundtrip_detects_hash_and_schema_tampering(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    image = tmp_path / "hub.sif"
    image.write_bytes(b"pinned image bytes")
    path = hub_image_manifest_path(image)
    expected = CuaGymHubImageManifest.for_image(image, web_manifest=manifest)
    expected.write(path)

    loaded = CuaGymHubImageManifest.read(path)
    assert loaded == expected
    loaded.validate(image, web_manifest=manifest)

    image.write_bytes(b"changed image bytes")
    with pytest.raises(RuntimeError, match="provenance or SHA-256"):
        loaded.validate(image, web_manifest=manifest)

    path.write_text('{"manifest_version":1,"manifest_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        CuaGymHubImageManifest.read(path)


def test_image_producer_archives_the_pin_and_publishes_no_personal_paths(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "hub-source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "CUA test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "cua@example.invalid"],
        check=True,
    )
    source = checkout / "source.txt"
    source.write_text("pinned\n", encoding="utf-8")
    shared = checkout / "shared"
    shared.mkdir()
    (shared / "secureMockApiPlugin.mjs").write_text(
        """function isLegacyCompatEnabled() {
  return process.env.CUA_GYM_LEGACY_COMPAT !== '0' && process.env.CUA_GYM_LEGACY_COMPAT !== 'false';
}

export function strictOnly() {
        if (isLegacyCompatEnabled()) return next();
        if (isLegacyCompatEnabled()) return next();
        if (isLegacyCompatEnabled()) return next();
}
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "--no-gpg-sign", "-m", "pin"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("dirty working tree\n", encoding="utf-8")

    apptainer = tmp_path / "apptainer"
    apptainer.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys
            from pathlib import Path

            assert sys.argv[1] == "build"
            assert (Path.cwd() / "hub" / "source.txt").read_text() == "pinned\\n"
            plugin = (Path.cwd() / "hub" / "shared" / "secureMockApiPlugin.mjs").read_text()
            assert "LegacyCompat" not in plugin
            assert "CUA_GYM_LEGACY_COMPAT" not in plugin
            definition = Path(sys.argv[3]).read_bytes()
            Path(sys.argv[2]).write_bytes(b"SIF\\0" + definition)
            """
        ),
        encoding="utf-8",
    )
    apptainer.chmod(0o755)
    output = tmp_path / "artifacts" / "cua-gym-hub.sif"
    producer = CuaGymHubImageProducer(
        CuaGymHubImageBuildConfig(
            hub_checkout=checkout,
            output_image=output,
            apptainer_binary=apptainer,
        )
    )
    producer.web_manifest = _manifest(revision)

    built = producer.build()

    assert CuaGymHubImageManifest.read(hub_image_manifest_path(output)) == built
    built.validate(output, web_manifest=producer.web_manifest)
    serialized = hub_image_manifest_path(output).read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert json.loads(serialized)["image_sha256"] == built.image_sha256
    with pytest.raises(FileExistsError, match="output exists"):
        producer.build()
