"""Orchestrates the AOT compile: profile -> decide -> compile -> manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from indic_runner.config import DIRS, hf_token
from indic_runner.models.registry import MODEL_REGISTRY
from indic_runner.setup import binary_manager, env_manager
from indic_runner.setup.artifact_compiler import (
    ENV_OVERLAYS,
    compiler_for,
    model_dir,
    prune_orphan_sources,
    runtime_env,
    validate_artifact_format,
)
from indic_runner.setup.decision_matrix import ExecutionPlan, select_plan
from indic_runner.setup.hardware_profiler import HardwareProfile, profile_hardware
from indic_runner.setup.manifest_writer import (
    build_manifest,
    manifest_path,
    write_manifest,
)
from indic_runner.setup.model_profiler import ModelProfile, profile_model


class AmbiguousTask(ValueError):
    """Raised when a model serves several tasks and none was specified."""


class UnknownModel(ValueError):
    """Raised when a target is neither a registry alias nor a repo id."""


class ThirdPartyWeights(ValueError):
    """Raised when a registry entry points at another org's GGUF build.

    The project does not run third-party weights: community quantizers differ
    in imatrix and base revision, so mixing them across the roster would mean
    partly measuring the quantizer rather than the model. Setup converts from
    official safetensors instead.
    """

    def __init__(self, alias: str, repo: str, gguf_repo: str):
        super().__init__(
            f"{alias}: gguf_repo {gguf_repo!r} belongs to a different org than "
            f"{repo!r}. Remove it and let setup convert from the official "
            "weights, or point it at a build published by the model's own org."
        )


class NotOnHub(RuntimeError):
    """Raised for a registered model that is not distributed via the Hub.

    EasyOCR is the current case: it fetches its own weights at first use, so
    there is nothing for the AOT compiler to profile or convert.
    """

    def __init__(self, alias: str):
        self.alias = alias
        super().__init__(
            f"{alias} does not publish to the Hugging Face Hub; its weights are "
            "fetched by the package itself at first use, so `setup` has nothing "
            "to compile. It is provisioned with its OCR runtime environment instead."
        )


@dataclass(frozen=True)
class ResolvedTarget:
    alias: str
    repo_id: str | None
    task: str
    registry_params_b: float | None
    ocr_runtime: str | None = None
    # True when the target came from the curated roster. Gates execution of a
    # repo's custom modelling code: roster entries are vetted, arbitrary
    # Hub repos are not.
    is_registry: bool = False
    # Curated GGUF build for models whose own org publishes none.
    gguf_repo: str | None = None


def _slugify(repo_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", repo_id.split("/")[-1].lower()).strip("-")


def tasks_for_alias(target: str) -> dict[str, dict]:
    """Registry entries matching an alias, keyed by task."""
    matches = {}
    for task, entries in MODEL_REGISTRY.items():
        for entry in entries:
            if entry["alias"] == target:
                matches[task] = entry
    return matches


def _check_same_org(alias: str, repo: str, gguf_repo: str | None) -> None:
    """Reject a curated GGUF published by anyone but the model's own org."""
    if gguf_repo and gguf_repo.split("/")[0] != repo.split("/")[0]:
        raise ThirdPartyWeights(alias, repo, gguf_repo)


def resolve_target(target: str, task: str | None = None) -> ResolvedTarget:
    """Map a CLI target (alias or repo id) onto a task and repo.

    The registry is metadata, not a whitelist: an unknown ``org/name`` is
    accepted as long as ``--task`` says what it is for.
    """
    matches = tasks_for_alias(target)

    if matches:
        if task:
            if task not in matches:
                raise UnknownModel(
                    f"{target!r} is not registered for task {task!r}; "
                    f"it serves {sorted(matches)}"
                )
            entry = matches[task]
            _check_same_org(target, entry["repo"], entry.get("gguf_repo"))
            return ResolvedTarget(
                target, entry["repo"], task, entry.get("params_b"),
                entry.get("runtime"), is_registry=True,
                gguf_repo=entry.get("gguf_repo"),
            )
        if len(matches) > 1:
            raise AmbiguousTask(
                f"{target!r} serves several tasks ({sorted(matches)}); "
                "pass --task to choose one"
            )
        only_task, entry = next(iter(matches.items()))
        _check_same_org(target, entry["repo"], entry.get("gguf_repo"))
        return ResolvedTarget(
            target, entry["repo"], only_task, entry.get("params_b"),
            entry.get("runtime"), is_registry=True,
            gguf_repo=entry.get("gguf_repo"),
        )

    if "/" not in target:
        raise UnknownModel(
            f"{target!r} is not a known alias and is not a Hugging Face repo id "
            "(expected 'org/name')"
        )
    if not task:
        raise AmbiguousTask(
            f"{target!r} is not in the registry, so --task is required"
        )
    return ResolvedTarget(_slugify(target), target, task, None)


def _hub_client():
    from functools import partial

    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    api = HfApi(token=token)
    # HfApi has no hf_hub_download method; adapt the module function onto the
    # injected-client Protocol so tests can supply a single fake object.
    api.hf_hub_download = partial(hf_hub_download, token=token)  # type: ignore[attr-defined]
    return api


@dataclass
class SetupResult:
    target: ResolvedTarget
    hardware: HardwareProfile
    model: ModelProfile
    plan: ExecutionPlan
    manifest: dict
    manifest_file: Path | None
    dry_run: bool


def run_setup(
    target: str,
    task: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    client=None,
    hardware: HardwareProfile | None = None,
) -> SetupResult:
    """Execute the AOT compile for one model."""
    resolved = resolve_target(target, task)

    existing = manifest_path(resolved.alias)
    if existing.exists() and not force and not dry_run:
        raise FileExistsError(
            f"{resolved.alias} is already set up at {existing}; pass --force to rebuild"
        )

    hw = hardware or profile_hardware()
    if resolved.repo_id is None:
        raise NotOnHub(resolved.alias)

    model = profile_model(
        resolved.repo_id,
        client or _hub_client(),
        registry_params_b=resolved.registry_params_b,
        ocr_runtime=resolved.ocr_runtime,
        gguf_repo=resolved.gguf_repo,
    )
    plan = select_plan(resolved.task, hw, model)

    bundle = None
    binary_or_env = plan.engine
    env_spec = runtime_env(plan)
    if env_spec:  # in-process engines run under a named isolated env
        binary_or_env = env_spec[0]
    library_dir = None
    artifact_source = None

    if dry_run:
        # A dry run that reports OK for a plan no compiler can build is worse
        # than no dry run at all. Checked without constructing the compiler,
        # which for GGUF would need a bundle a dry run never downloads.
        validate_artifact_format(plan)
        # Report the real destinations without touching the network or disk,
        # so the printed manifest matches what a real run would write.
        artifacts_dir = model_dir(resolved.alias, plan.artifact_format)
        tokenizer_dir = artifacts_dir
        if plan.engine == "llama.cpp":
            slug = binary_manager.asset_slug(hw.os, hw.arch, hw.accelerator)
            root = DIRS["bin"] / f"llama.cpp-{binary_manager.PINNED_BUILD}-{slug}"
            binary_or_env = str(root / "llama-server")
            library_dir = root
    else:
        if plan.engine == "llama.cpp":
            bundle = binary_manager.ensure_llama_cpp(hw.os, hw.arch, hw.accelerator)
            binary_or_env = str(bundle.executable("llama-server"))
            library_dir = bundle.library_dir

        compiler = compiler_for(
            plan, resolved.alias, bundle, hw.accelerator, trusted=resolved.is_registry
        )
        artifacts = compiler.ensure(plan, model)
        if env_spec:
            env_manager.ensure_env(
                *env_spec, accelerator=hw.accelerator, overlay=ENV_OVERLAYS.get(env_spec[0], ())
            )
        artifacts_dir = artifacts.artifacts_dir
        tokenizer_dir = artifacts.tokenizer_dir
        # A compiler evicts the source it consumed; this clears one abandoned
        # by a previous run whose plan chose a different lane.
        prune_orphan_sources(resolved.alias, keep=artifacts_dir)
        artifact_source = {
            "repo": artifacts.source_repo or resolved.repo_id,
            "provenance": artifacts.provenance,
        }

    manifest = build_manifest(
        model_alias=resolved.alias,
        base_repo=resolved.repo_id,
        revision=model.revision,
        plan=plan,
        hardware=hw,
        artifacts_dir=artifacts_dir,
        tokenizer_dir=tokenizer_dir,
        binary_or_env=binary_or_env,
        library_dir=library_dir,
        artifact_source=artifact_source,
    )

    manifest_file = None if dry_run else write_manifest(manifest)
    return SetupResult(resolved, hw, model, plan, manifest, manifest_file, dry_run)
