"""Reads and validates the setup-phase execution-contract manifest.

`run` executes strictly from this file: no network, no guessing. Anything the
manifest points at that is missing on disk fails here, before an engine starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from indic_runner.setup.manifest_writer import (
    ManifestValidationError,
    manifest_path,
    validate_manifest,
)
from schemas.manifest_schema import MANIFEST_VERSION


class ManifestError(RuntimeError):
    """The manifest is absent, malformed, stale or points at missing files."""


@dataclass(frozen=True)
class Manifest:
    raw: dict

    @property
    def alias(self) -> str: return self.raw["model_alias"]
    @property
    def task(self) -> str: return self.raw["task"]
    @property
    def engine(self) -> str: return self.raw["engine"]["name"]
    @property
    def mode(self) -> str: return self.raw["engine"]["mode"]
    @property
    def binary_or_env(self) -> str: return self.raw["engine"]["binary_or_env"]
    @property
    def library_dir(self) -> str | None: return self.raw["engine"].get("library_dir")
    @property
    def artifacts_dir(self) -> Path: return Path(self.raw["paths"]["artifacts_dir"])
    @property
    def tokenizer_dir(self) -> Path: return Path(self.raw["paths"]["tokenizer_dir"])
    @property
    def precision(self) -> str: return self.raw["precision"]
    @property
    def hardware(self) -> str: return self.raw["target_hardware"]
    @property
    def max_batch_size(self) -> int: return self.raw["runtime_parameters"]["max_batch_size"]
    @property
    def context_length(self) -> int: return self.raw["runtime_parameters"]["context_length"]
    @property
    def threads(self) -> int | None: return self.raw["runtime_parameters"].get("threads")


def load_manifest(alias: str, check_paths: bool = True) -> Manifest:
    path = manifest_path(alias)
    if not path.is_file():
        raise ManifestError(
            f"no manifest for {alias!r} at {path}; run `indic-runner setup {alias}` first"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    try:
        validate_manifest(raw)
    except ManifestValidationError as exc:
        raise ManifestError(f"{path}: {exc}; re-run setup with --force") from exc
    if raw["manifest_version"] != MANIFEST_VERSION:
        raise ManifestError(
            f"{path} is manifest v{raw['manifest_version']}, this runner reads "
            f"v{MANIFEST_VERSION}; re-run setup with --force"
        )
    manifest = Manifest(raw)
    if check_paths:
        _check_paths(manifest)
    return manifest


def _check_paths(m: Manifest) -> None:
    missing = [
        str(p) for p in (m.artifacts_dir, m.tokenizer_dir) if not p.exists()
    ]
    if m.mode == "daemon" and not Path(m.binary_or_env).exists() and m.engine == "llama.cpp":
        missing.append(m.binary_or_env)
    if missing:
        raise ManifestError(
            "manifest references missing files (was setup interrupted or the "
            "home directory moved?): " + ", ".join(missing)
        )
