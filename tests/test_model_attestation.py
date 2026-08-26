from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _register_model(root: Path) -> tuple[Path, dict]:
    model = root / "model"
    model.mkdir()
    config_identity = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
    }
    (model / "config.json").write_text(json.dumps(config_identity))
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text('{"version":"1.0"}')
    return model, {
        "path": str(model.resolve()),
        "config_identity": config_identity,
        "served_model": str(model.resolve()),
    }


def test_local_model_path_and_config_are_the_serving_identity(tmp_path) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    model, expected = _register_model(tmp_path)

    artifact = _verify_model_artifact(model)

    assert artifact.model_path == model.resolve()
    assert artifact.config_identity == expected["config_identity"]
    assert artifact.served_model == expected["served_model"]


def test_relative_model_path_is_refused(tmp_path, monkeypatch) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    _register_model(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="must be absolute"):
        _verify_model_artifact(Path("model"))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_config", "invalid model config"),
        ("missing_architectures", "requires architectures"),
        ("missing_weights", "no safetensors weights"),
    ],
)
def test_incomplete_model_directories_are_refused(tmp_path, mutation, error) -> None:
    from evals.signoflife.__main__ import _verify_model_artifact

    model, _ = _register_model(tmp_path)
    if mutation == "missing_config":
        (model / "config.json").unlink()
    elif mutation == "missing_architectures":
        (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
    else:
        (model / "model.safetensors").unlink()

    with pytest.raises(RuntimeError, match=error):
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


def test_inherited_external_credential_is_refused_before_acquisition(
    tmp_path, monkeypatch
) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setenv(dispatcher.API_KEY_VAR, "must-never-reach-a-child")
    monkeypatch.setattr(
        dispatcher,
        "load_suite",
        lambda: pytest.fail("credential validation ran after suite acquisition"),
    )

    with pytest.raises(SystemExit, match="owned local no-auth SGLang"):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(tmp_path / "run"),
                "--qcow",
                str(tmp_path / "desktop.qcow2"),
            ]
        )


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


def test_caller_chosen_listener_port_is_refused(tmp_path, capsys) -> None:
    import evals.signoflife.__main__ as dispatcher

    model, _ = _register_model(tmp_path)
    with pytest.raises(SystemExit):
        dispatcher.main(
            [
                "--arm",
                "ordered",
                "--output",
                str(tmp_path / "run"),
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
    assert "unrecognized arguments: --sglang-port" in capsys.readouterr().err
