import json

from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.fixtures import (
    MANIFEST_PATH,
    load_manifest,
)


def test_manifest_is_development_only_and_sealed():
    manifest = load_manifest()
    assert len(manifest.fixtures) == 6
    assert {fixture.split for fixture in manifest.fixtures} == {"development"}
    assert len(manifest.manifest_payload_sha256) == 64


def test_templates_are_real_applications_and_drag_is_files_state():
    manifest = load_manifest()
    assert {fixture.template for fixture in manifest.fixtures} == {
        "vscode_focus_type",
        "local_document_scroll",
        "files_drag",
    }
    drag = [fixture for fixture in manifest.fixtures if fixture.template == "files_drag"]
    assert drag and all(fixture.expected["destination"].startswith("Delivered-") for fixture in drag)
    assert "slider" not in MANIFEST_PATH.read_text(encoding="utf-8").lower()


def test_unicode_and_signed_scroll_coverage():
    manifest = load_manifest()
    typed = [fixture.expected["text"] for fixture in manifest.fixtures if fixture.template == "vscode_focus_type"]
    assert any("🧭" in text and "Δ" in text for text in typed)
    assert any("東京" in text and "🚲" in text for text in typed)
    directions = {fixture.params["direction"] for fixture in manifest.fixtures if fixture.template == "local_document_scroll"}
    assert directions == {"up", "down"}


def test_no_official_task_identifiers_or_assets():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(payload["fixtures"], ensure_ascii=False).lower()
    for forbidden in ("task_config", "evaluation_examples", "heldout", "osworld_eval"):
        assert forbidden not in serialized

