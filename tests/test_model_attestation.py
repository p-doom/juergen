"""A scored model is the registered bytes that answered, or the run is refused."""

from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _external_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SIGN_OF_LIFE_API_KEY", "test-only-secret")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _register_model(root: Path) -> tuple[Path, dict]:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        )
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(b"registered weights")
    (model / "tokenizer.json").write_text('{"version":"1.0"}')
    files = [
        {
            "path": path.relative_to(model).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(model.rglob("*"))
        if path.is_file()
    ]
    artifact_sha256 = hashlib.sha256(_canonical(files)).hexdigest()
    config_sha256 = next(row["sha256"] for row in files if row["path"] == "config.json")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "artifact_sha256": artifact_sha256,
        "config_sha256": config_sha256,
        "files": files,
    }
    manifest_path = root / "artifact_manifest.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    registration = {
        "id": "artifact_test_model",
        "producer_run_id": "run_test_model",
        "artifact_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (root / ".meta.json").write_text(json.dumps(registration))
    return model, {**manifest, **registration}


def test_registered_artifact_verifies_every_file_and_config_identity(tmp_path) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    model, expected = _register_model(tmp_path)
    artifact = _verify_model_artifact(model)

    assert artifact.artifact_id == expected["id"]
    assert artifact.artifact_sha256 == expected["artifact_sha256"]
    assert artifact.config_sha256 == expected["config_sha256"]
    assert artifact.served_model.endswith(expected["artifact_sha256"])
    assert artifact.file_count == 3


def test_a_weight_byte_mutation_is_refused(tmp_path) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    model, _ = _register_model(tmp_path)
    weight = model / "model-00001-of-00001.safetensors"
    weight.write_bytes(weight.read_bytes()[:-1] + b"X")

    with pytest.raises(RuntimeError, match="artifact inventory mismatch"):
        _verify_model_artifact(model)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_model_files_are_refused(tmp_path, mutation) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    model, _ = _register_model(tmp_path)
    if mutation == "missing":
        (model / "tokenizer.json").unlink()
    else:
        (model / "unregistered.safetensors").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="artifact inventory mismatch"):
        _verify_model_artifact(model)


def test_a_manifest_mutation_breaks_its_registration(tmp_path) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    model, _ = _register_model(tmp_path)
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="manifest registration mismatch"):
        _verify_model_artifact(model)


def test_an_external_model_arm_still_requires_registered_local_bytes(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_run_attempts",
        lambda *args, **kwargs: pytest.fail(
            "artifact validation ran after resource setup"
        ),
    )
    with pytest.raises(SystemExit, match="--model-path"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--base-url",
                "http://127.0.0.1:9000/v1",
            ]
        )
    assert not (tmp_path / "run").exists()


def test_changed_artifact_bytes_are_refused_before_output_or_resources(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    monkeypatch.setattr(
        dispatcher,
        "_attest_external_server",
        lambda *args, **kwargs: pytest.fail(
            "server attestation ran before artifact validation"
        ),
    )
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="artifact inventory mismatch"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(output),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--model-path",
                str(model),
                "--base-url",
                "http://127.0.0.1:9000/v1",
            ]
        )
    assert not output.exists()


def test_external_attestation_failure_leaves_only_an_uncommitted_run_id(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    monkeypatch.setattr(
        dispatcher,
        "_attest_external_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("model attestation mismatch")
        ),
    )
    monkeypatch.setattr(
        dispatcher,
        "_run_attempts",
        lambda *args, **kwargs: pytest.fail(
            "attempt allocation ran before server attestation"
        ),
    )
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="model attestation mismatch"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(output),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--model-path",
                str(model),
                "--base-url",
                "http://127.0.0.1:9000/v1",
            ]
        )
    assert output.is_dir()
    assert not (output / "RESULT_COMMITTED.json").exists()


def test_missing_local_runtime_prerequisite_does_not_consume_run_id(tmp_path) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    output = tmp_path / "never-created"
    with pytest.raises(SystemExit, match="explicit --sglang-python"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(output),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--model-path",
                str(model),
            ]
        )

    assert not output.exists()


class _AttestationResponse:
    def __init__(self, request, artifact, *, change=None):
        nonce = request.get_header("X-attestation-nonce")
        self.payload = {
            "schema_version": 1,
            "nonce": nonce,
            "served_model": artifact.served_model,
            "artifact_id": artifact.artifact_id,
            "artifact_sha256": artifact.artifact_sha256,
            "manifest_sha256": artifact.manifest_sha256,
            "config_sha256": artifact.config_sha256,
        }
        if change is not None:
            self.payload.update(change)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, limit):
        assert limit == 65537
        return json.dumps(self.payload).encode()


def test_exact_external_attestation_is_bound_to_the_artifact(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    artifact = dispatcher._verify_model_artifact(model)
    seen = []

    def urlopen(request, timeout):
        seen.append(
            (
                request.full_url,
                timeout,
                request.get_header("Authorization"),
            )
        )
        return _AttestationResponse(request, artifact)

    monkeypatch.setattr(dispatcher.urllib.request, "urlopen", urlopen)
    record = dispatcher._attest_external_server("http://127.0.0.1:9000/v1", artifact)

    assert seen == [
        (
            "http://127.0.0.1:9000/model-attestation",
            10.0,
            "Bearer test-only-secret",
        )
    ]
    assert record["source"] == "external_endpoint"
    assert record["artifact_sha256"] == artifact.artifact_sha256
    assert "test-only-secret" not in json.dumps(record)


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_sha256": "0" * 64},
        {"config_sha256": "0" * 64},
        {"served_model": "wrong"},
        {"nonce": None},
    ],
)
def test_external_attestation_mismatch_is_refused(tmp_path, monkeypatch, change) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    artifact = dispatcher._verify_model_artifact(model)
    monkeypatch.setattr(
        dispatcher.urllib.request,
        "urlopen",
        lambda request, timeout: _AttestationResponse(request, artifact, change=change),
    )

    with pytest.raises(RuntimeError, match="model attestation mismatch"):
        dispatcher._attest_external_server("http://127.0.0.1:9000/v1", artifact)


def test_missing_or_unreachable_external_attestation_is_refused(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    artifact = dispatcher._verify_model_artifact(model)
    monkeypatch.setattr(
        dispatcher.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )

    with pytest.raises(RuntimeError, match="external model attestation failed"):
        dispatcher._attest_external_server("http://127.0.0.1:9000/v1", artifact)
