"""Emits the setup-phase execution-contract manifest.

Writes are atomic so a partially written manifest can never be read by `run`.
Validation is dependency-free: the schema is walked directly rather than
pulling jsonschema into the orchestrator env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from indic_runner.config import DIRS
from schemas.manifest_schema import MANIFEST_SCHEMA, MANIFEST_VERSION

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


class ManifestValidationError(ValueError):
    """Raised when an assembled manifest violates the contract."""


def _check(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        types = tuple(
            t
            for name in allowed
            for t in (_TYPES[name] if isinstance(_TYPES[name], tuple) else (_TYPES[name],))
        )
        # bool is a subclass of int; keep them distinct.
        if isinstance(value, bool) and "boolean" not in allowed:
            errors.append(f"{path}: expected {allowed}, got boolean")
            return
        if not isinstance(value, types):
            errors.append(f"{path}: expected {allowed}, got {type(value).__name__}")
            return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    if schema.get("type") == "object":
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                _check(value[key], subschema, f"{path}.{key}", errors)


def validate_manifest(manifest: dict) -> None:
    """Raise ManifestValidationError if the manifest breaks the contract."""
    errors: list[str] = []
    _check(manifest, MANIFEST_SCHEMA, "manifest", errors)
    if errors:
        raise ManifestValidationError(
            "manifest failed validation:\n  - " + "\n  - ".join(errors)
        )


def build_manifest(
    model_alias: str,
    base_repo: str,
    revision: str,
    plan,
    hardware,
    artifacts_dir: Path,
    tokenizer_dir: Path,
    binary_or_env: str,
    library_dir: Path | None = None,
    artifact_source: dict | None = None,
) -> dict:
    """Assemble the execution contract. Paths are absolute by construction."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "model_alias": model_alias,
        "base_repo": base_repo,
        "revision": revision,
        "task": plan.task,
        "target_hardware": hardware.accelerator,
        "precision": plan.precision,
        "engine": {
            "name": plan.engine,
            "mode": plan.mode,
            "binary_or_env": binary_or_env,
            "library_dir": str(library_dir) if library_dir else None,
        },
        "artifact_source": artifact_source
        or {"repo": base_repo, "provenance": "official"},
        "paths": {
            "artifacts_dir": str(Path(artifacts_dir).resolve()),
            "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
        },
        "runtime_parameters": dict(plan.runtime_parameters),
    }
    validate_manifest(manifest)
    return manifest


def manifest_path(model_alias: str) -> Path:
    return DIRS["manifests"] / f"{model_alias}.json"


def write_manifest(manifest: dict) -> Path:
    """Validate and atomically write the manifest; returns its path."""
    validate_manifest(manifest)
    destination = manifest_path(manifest["model_alias"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    tmp = destination.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
    return destination
