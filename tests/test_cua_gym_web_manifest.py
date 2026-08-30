from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from evals.cua_gym.manifest import load_default_manifest
from evals.cua_gym.models import EndpointName, TaskId
from evals.cua_gym.web.manifest import (
    CuaGymWebRuntimeManifest,
    load_default_web_runtime_manifest,
)


def _payload() -> dict[str, object]:
    return {
        "dataset_revision": "d" * 40,
        "endpoints": {"gmail": "gmail_mock", "notion": "notion_mock"},
        "hub_revision": "e" * 40,
        "manifest_version": 1,
        "supported_task_count": 2,
        "task_scratch_paths": ["/tmp/task_sid"],
        "unsupported_endpoints": {},
        "unsupported_tasks": {},
        "writable_directories": [".mock-files", ".mock-states"],
    }


def test_web_manifest_parses_one_exact_immutable_contract() -> None:
    manifest = CuaGymWebRuntimeManifest.from_dict(_payload())

    assert manifest.apps == ("gmail_mock", "notion_mock")
    assert manifest.hostname(EndpointName("google_docs")) == (
        "google-docs.cua.internal"
    )
    assert isinstance(manifest.endpoint_apps, MappingProxyType)
    with pytest.raises(TypeError):
        manifest.endpoint_apps[EndpointName("extra")] = "extra_mock"  # type: ignore[index]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({**_payload(), "extra": True}, "unexpected=.*extra"),
        (
            {key: value for key, value in _payload().items() if key != "hub_revision"},
            "missing=.*hub_revision",
        ),
        ({**_payload(), "manifest_version": 2}, "Unsupported.*version"),
        ({**_payload(), "hub_revision": "short"}, "full lowercase Git revision"),
        (
            {**_payload(), "writable_directories": [".mock-states", ".mock-files"]},
            "sorted, unique",
        ),
        (
            {**_payload(), "task_scratch_paths": ["/tmp/nested/file"]},
            "flat /tmp paths",
        ),
    ],
)
def test_web_manifest_rejects_malformed_payloads(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CuaGymWebRuntimeManifest.from_dict(payload)


def test_web_manifest_rejects_colliding_hostnames_and_non_string_keys() -> None:
    values = _payload()
    with pytest.raises(ValueError, match="Duplicate.*hostname"):
        CuaGymWebRuntimeManifest(
            manifest_version=1,
            dataset_revision="d" * 40,
            hub_revision="e" * 40,
            endpoint_apps={
                EndpointName("google_docs"): "docs_mock",
                EndpointName("google-docs"): "other_mock",
            },
            writable_directories=(".mock-states",),
            unsupported_endpoints={},
            unsupported_tasks={},
            task_scratch_paths=("/tmp/task_sid",),
            supported_task_count=1,
        )

    values["endpoints"] = {1: "gmail_mock"}
    with pytest.raises(ValueError, match="invalid endpoints"):
        CuaGymWebRuntimeManifest.from_dict(values)


def test_web_manifest_reports_task_and_endpoint_incompatibilities() -> None:
    task_id = TaskId("task-1")
    endpoint = EndpointName("notion")
    manifest = CuaGymWebRuntimeManifest(
        manifest_version=1,
        dataset_revision="d" * 40,
        hub_revision="e" * 40,
        endpoint_apps={endpoint: "notion_mock"},
        writable_directories=(".mock-states",),
        unsupported_endpoints={endpoint: "endpoint defect"},
        unsupported_tasks={task_id: "task defect"},
        task_scratch_paths=("/tmp/task_sid",),
        supported_task_count=1,
    )

    assert manifest.incompatibilities_for(task_id, (endpoint,)) == (
        "task task-1: task defect",
        "endpoint notion: endpoint defect",
    )


def test_web_manifest_validates_the_existing_dataset_contract() -> None:
    dataset = load_default_manifest()
    manifest = CuaGymWebRuntimeManifest(
        manifest_version=1,
        dataset_revision=dataset.revision,
        hub_revision="e" * 40,
        endpoint_apps={
            endpoint: f"{endpoint}_mock" for endpoint in dataset.endpoint_specs
        },
        writable_directories=(".mock-states",),
        unsupported_endpoints={},
        unsupported_tasks={},
        task_scratch_paths=("/tmp/task_sid",),
        supported_task_count=dataset.eligible_task_count,
    )

    manifest.validate_dataset(dataset)
    with pytest.raises(ValueError, match="supported task count differs"):
        replace(
            manifest, supported_task_count=manifest.supported_task_count - 1
        ).validate_dataset(dataset)


def test_packaged_web_manifest_matches_the_packaged_dataset() -> None:
    manifest = load_default_web_runtime_manifest()

    manifest.validate_dataset(load_default_manifest())
    assert manifest.supported_task_count == 966
    assert manifest.hub_revision == "53205689c3d88078c1375f76466d5bd799478828"
