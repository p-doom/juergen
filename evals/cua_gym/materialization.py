"""Safe, in-memory endpoint materialization for CUA-Gym task sources."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from types import MappingProxyType
from urllib.parse import urlparse

from .errors import MaterializationError
from .manifest import CompatibilityManifest
from .models import (
    BundleFile,
    EndpointName,
    MaterializedTaskBundle,
    TaskBundle,
    TaskCompatibility,
    clone_frozen_json_object,
)

_PLACEHOLDER_RE = re.compile(r"__CUA_GYM_[A-Z0-9_]+__")
_SAFE_BASE_URL_RE = re.compile(r"[A-Za-z0-9._~:/\[\]@!$&()*+,;=%-]+")


def derive_required_endpoints(
    compatibility: TaskCompatibility,
    setup_sources: tuple[str, ...],
    reward_source: str,
    manifest: CompatibilityManifest,
) -> tuple[EndpointName, ...]:
    """Derive mock dependencies from setup/reward contents, never app metadata."""

    combined = "\n".join((*setup_sources, reward_source))
    found: set[EndpointName] = set()
    for endpoint, spec in manifest.endpoint_specs.items():
        if any(token in combined for token in (*spec.url_tokens, *spec.host_tokens)):
            found.add(endpoint)
    for endpoint, url in compatibility.hard_coded_endpoint_urls:
        if url in combined:
            found.add(endpoint)
    return tuple(sorted(found))


def materialize_task_bundle(
    bundle: TaskBundle,
    gateway_urls: Mapping[str | EndpointName, str],
    manifest: CompatibilityManifest,
) -> MaterializedTaskBundle:
    """Replace only manifest-known deployment tokens without executing any code."""

    compatibility = bundle.metadata.compatibility
    normalized = _normalize_gateway_urls(gateway_urls)
    required = set(compatibility.required_endpoints)
    missing = required - set(normalized)
    if missing:
        raise MaterializationError(
            f"Missing gateway URLs for task {bundle.task_id}: "
            + ", ".join(sorted(missing))
        )

    replacements: dict[str, str] = {}
    token_owners: dict[str, EndpointName] = {}

    def add_replacement(token: str, replacement: str, endpoint: EndpointName) -> None:
        existing_replacement = replacements.setdefault(token, replacement)
        existing_owner = token_owners.setdefault(token, endpoint)
        if existing_replacement != replacement or existing_owner != endpoint:
            raise MaterializationError(
                f"Conflicting endpoint replacement token for task {bundle.task_id}: "
                f"{token!r}"
            )

    for endpoint in compatibility.required_endpoints:
        spec = manifest.endpoint_specs[endpoint]
        url = normalized[endpoint]
        host = urlparse(url).netloc
        for token in spec.url_tokens:
            add_replacement(token, url, endpoint)
        for token in spec.host_tokens:
            add_replacement(token, host, endpoint)
    for endpoint, hard_coded_url in compatibility.hard_coded_endpoint_urls:
        add_replacement(hard_coded_url, normalized[endpoint], endpoint)
    ordered_tokens = sorted(replacements, key=lambda token: (-len(token), token))
    replacement_pattern = (
        re.compile("|".join(re.escape(token) for token in ordered_tokens))
        if ordered_tokens
        else None
    )

    replacement_counts = {endpoint: 0 for endpoint in required}

    def materialize_source(source: str) -> tuple[str, str]:
        if replacement_pattern is None:
            return source, source

        validation_parts: list[str] = []
        validation_start = 0

        def replace_match(match: re.Match[str]) -> str:
            nonlocal validation_start
            token = match.group()
            # Validate only unmatched original text; replacement values may
            # legitimately contain text shaped like another deployment token.
            validation_parts.append(source[validation_start : match.start()])
            validation_start = match.end()
            replacement_counts[token_owners[token]] += 1
            return replacements[token]

        materialized = replacement_pattern.sub(replace_match, source)
        validation_parts.append(source[validation_start:])
        return materialized, "".join(validation_parts)

    setup_files: list[BundleFile] = []
    validation_sources: list[str] = []
    for setup_file in bundle.setup_files:
        try:
            source = setup_file.text()
        except UnicodeDecodeError as error:
            raise MaterializationError(
                f"Setup source is not UTF-8 text: {setup_file.name}"
            ) from error
        materialized_source, validation_source = materialize_source(source)
        setup_files.append(
            BundleFile(
                name=setup_file.name,
                content=materialized_source.encode("utf-8"),
            )
        )
        validation_sources.append(validation_source)
    reward_source, reward_validation_source = materialize_source(bundle.reward_source)
    validation_sources.append(reward_validation_source)
    for setup_file in setup_files:
        if setup_file.name.endswith(".py"):
            _validate_python_syntax(setup_file.name, setup_file.text())
    _validate_python_syntax("reward.py", reward_source)

    missing_replacements = sorted(
        endpoint for endpoint, count in replacement_counts.items() if count == 0
    )
    if missing_replacements:
        raise MaterializationError(
            f"Manifest endpoint tokens were absent for task {bundle.task_id}: "
            + ", ".join(missing_replacements)
        )
    _validate_no_deployment_tokens(
        "\n".join(validation_sources), compatibility, manifest
    )

    used_gateways = MappingProxyType(
        {
            endpoint: normalized[endpoint]
            for endpoint in compatibility.required_endpoints
        }
    )
    return MaterializedTaskBundle(
        metadata=bundle.metadata,
        task_config=clone_frozen_json_object(bundle.task_config),
        reward_source=reward_source,
        setup_files=tuple(setup_files),
        gateway_urls=used_gateways,
    )


def _normalize_gateway_urls(
    gateway_urls: Mapping[str | EndpointName, str],
) -> dict[EndpointName, str]:
    normalized: dict[EndpointName, str] = {}
    for raw_endpoint, raw_url in gateway_urls.items():
        endpoint = EndpointName(str(raw_endpoint))
        if not isinstance(raw_url, str):
            raise MaterializationError(f"Gateway URL for {endpoint} must be a string")
        url = raw_url.rstrip("/")
        if (
            not url.isascii()
            or any(character in url for character in ("'", '"', "\\", "{", "}"))
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in url
            )
            or _SAFE_BASE_URL_RE.fullmatch(url) is None
        ):
            raise MaterializationError(
                f"Gateway URL for {endpoint} contains characters unsafe for "
                "Python source materialization"
            )
        try:
            parsed = urlparse(url)
            parsed_port = parsed.port
        except ValueError as error:
            raise MaterializationError(
                f"Gateway URL for {endpoint} has an invalid host or port"
            ) from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (parsed_port is not None and not 1 <= parsed_port <= 65535)
        ):
            raise MaterializationError(
                f"Gateway URL for {endpoint} must be an HTTP(S) base URL "
                "without credentials, query, or fragment"
            )
        normalized[endpoint] = url
    return normalized


def _validate_python_syntax(name: str, source: str) -> None:
    try:
        ast.parse(source, filename=name)
    except SyntaxError as error:
        raise MaterializationError(
            f"Endpoint materialization produced invalid Python in {name}: {error.msg}"
        ) from error


def _validate_no_deployment_tokens(
    source: str,
    compatibility: TaskCompatibility,
    manifest: CompatibilityManifest,
) -> None:
    placeholders = sorted(set(_PLACEHOLDER_RE.findall(source)))
    if placeholders:
        raise MaterializationError(
            "Deployment placeholders remain after materialization: "
            + ", ".join(placeholders)
        )
    remaining: set[str] = set()
    for spec in manifest.endpoint_specs.values():
        for token in (*spec.url_tokens, *spec.host_tokens):
            if token in source:
                remaining.add(token)
    for _, url in compatibility.hard_coded_endpoint_urls:
        if url in source:
            remaining.add(url)
    if remaining:
        raise MaterializationError(
            "Known hosted endpoints remain after materialization: "
            + ", ".join(sorted(remaining))
        )
