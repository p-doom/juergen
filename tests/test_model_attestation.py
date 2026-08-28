"""A scored model is the registered bytes that answered, or the run is refused."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


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


def test_external_server_flag_is_not_part_of_the_canonical_cli(
    tmp_path, monkeypatch, capsys
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.delenv(dispatcher.API_KEY_VAR, raising=False)
    with pytest.raises(SystemExit):
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
    assert "unrecognized arguments: --base-url" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_inherited_external_credential_is_refused_before_any_acquisition(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setenv(dispatcher.API_KEY_VAR, "must-never-reach-a-child")
    monkeypatch.setattr(
        dispatcher,
        "load_suite",
        lambda: pytest.fail("credential validation ran after suite/model acquisition"),
    )
    output = tmp_path / "run"

    with pytest.raises(SystemExit, match="owned local no-auth SGLang"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(output),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
            ]
        )

    assert not output.exists()


def test_changed_artifact_bytes_are_refused_before_output_or_resources(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    monkeypatch.setattr(
        dispatcher,
        "_sglang",
        lambda *args, **kwargs: pytest.fail(
            "local server started before artifact validation"
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
                "--sglang-python",
                sys.executable,
                "--sglang-port",
                "29500",
            ]
        )
    assert not output.exists()


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


def test_missing_listener_port_is_refused_before_artifact_or_output(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    output = tmp_path / "never-created"
    monkeypatch.setattr(
        dispatcher,
        "_verify_model_artifact",
        lambda *_args, **_kwargs: pytest.fail("artifact verification acquired resources"),
    )

    with pytest.raises(SystemExit, match="--sglang-port in \\[1, 55535\\]"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(output),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
                "--model-path",
                str(tmp_path / "model"),
                "--sglang-python",
                sys.executable,
            ]
        )

    assert not output.exists()
