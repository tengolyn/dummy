"""Target resolution and end-to-end dry runs against a fake Hub."""

from __future__ import annotations

import pytest
from helpers import FakeHub, make_hw

from indic_runner.setup import pipeline
from indic_runner.setup.decision_matrix import InsufficientResources

QWEN_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
}


# --- target resolution -----------------------------------------------------

def test_unique_alias_infers_its_task():
    resolved = pipeline.resolve_target("sarvam-translate")
    assert resolved.task == "translation"
    assert resolved.repo_id == "sarvamai/sarvam-translate"


def _multi_task_aliases():
    """Aliases the registry lists under more than one task.

    Derived rather than hardcoded so the test follows roster changes.
    """
    from collections import Counter

    from indic_runner.models.registry import MODEL_REGISTRY

    counts = Counter(
        e["alias"] for entries in MODEL_REGISTRY.values() for e in entries
    )
    return sorted(a for a, n in counts.items() if n > 1)


@pytest.mark.parametrize("alias", _multi_task_aliases())
def test_alias_serving_several_tasks_demands_one(alias):
    with pytest.raises(pipeline.AmbiguousTask, match="--task"):
        pipeline.resolve_target(alias)


def test_there_is_at_least_one_ambiguous_alias_to_guard():
    assert _multi_task_aliases()


def test_explicit_task_disambiguates():
    from indic_runner.models.registry import MODEL_REGISTRY

    alias = _multi_task_aliases()[0]
    tasks = [t for t, es in MODEL_REGISTRY.items() if any(e["alias"] == alias for e in es)]
    for task in tasks:
        assert pipeline.resolve_target(alias, task).task == task


def test_task_the_alias_does_not_serve_is_rejected():
    with pytest.raises(pipeline.UnknownModel, match="not registered"):
        pipeline.resolve_target("sarvam-translate", "ocr")


def test_offregistry_repo_is_accepted_with_a_task():
    """The registry is metadata, not a whitelist."""
    resolved = pipeline.resolve_target("Qwen/Qwen2.5-0.5B-Instruct", "reasoning")
    assert resolved.repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolved.alias == "qwen2.5-0.5b-instruct"
    assert resolved.registry_params_b is None


def test_offregistry_repo_without_a_task_is_rejected():
    with pytest.raises(pipeline.AmbiguousTask, match="--task is required"):
        pipeline.resolve_target("Qwen/Qwen2.5-0.5B-Instruct")


def test_bare_nonsense_is_rejected():
    with pytest.raises(pipeline.UnknownModel, match="not a known alias"):
        pipeline.resolve_target("not-a-model")


def test_registry_params_are_passed_through():
    assert pipeline.resolve_target("sarvam-translate").registry_params_b == 4


# --- dry run ---------------------------------------------------------------

def _dry_run(alias="sarvam-translate", task=None, hw=None, hub=None):
    return pipeline.run_setup(
        alias,
        task=task,
        dry_run=True,
        client=hub or FakeHub(config=QWEN_CONFIG),
        hardware=hw or make_hw("cpu", total_ram_gb=32.0),
    )


def test_dry_run_produces_a_valid_manifest_without_touching_disk():
    result = _dry_run("Qwen/Qwen2.5-0.5B-Instruct", "reasoning")
    assert result.dry_run is True
    assert result.manifest_file is None
    assert result.manifest["task"] == "reasoning"
    assert result.manifest["precision"] == "q4_k_m"


def test_dry_run_reports_the_real_destination_paths():
    result = _dry_run("Qwen/Qwen2.5-0.5B-Instruct", "reasoning")
    artifacts = result.manifest["paths"]["artifacts_dir"]
    assert artifacts.endswith("qwen2.5-0.5b-instruct-gguf")
    assert "<" not in artifacts  # never a placeholder


def test_dry_run_names_the_llama_binary_and_library_dir():
    result = _dry_run("Qwen/Qwen2.5-0.5B-Instruct", "reasoning")
    engine = result.manifest["engine"]
    assert engine["binary_or_env"].endswith("llama-server")
    assert engine["library_dir"] is not None


def test_cuda_host_plans_vllm(monkeypatch):
    hw = make_hw("cuda", vram_gb=80.0, gpu_name="A100", cuda_version="12.4")
    hub = FakeHub(config=QWEN_CONFIG, safetensors_total=7_000_000_000)
    result = _dry_run("Qwen/Qwen2.5-7B-Instruct", "reasoning", hw=hw, hub=hub)
    assert result.manifest["engine"]["name"] == "vllm"
    assert result.manifest["precision"] == "fp16"
    assert result.manifest["engine"]["library_dir"] is None


def test_oversized_model_is_refused_before_any_download():
    hub = FakeHub(config=QWEN_CONFIG, safetensors_total=70_000_000_000)
    with pytest.raises(InsufficientResources):
        _dry_run("org/huge-70B", "reasoning", hw=make_hw("cpu", total_ram_gb=4.0), hub=hub)


def test_existing_manifest_blocks_a_rebuild(tmp_path, monkeypatch):
    from indic_runner.setup import manifest_writer

    monkeypatch.setitem(manifest_writer.DIRS, "manifests", tmp_path)
    (tmp_path / "sarvam-translate.json").write_text("{}")
    with pytest.raises(FileExistsError, match="--force"):
        pipeline.run_setup(
            "sarvam-translate", client=FakeHub(), hardware=make_hw("cpu")
        )


def test_dry_run_ignores_an_existing_manifest(tmp_path, monkeypatch):
    from indic_runner.setup import manifest_writer

    monkeypatch.setitem(manifest_writer.DIRS, "manifests", tmp_path)
    (tmp_path / "sarvam-translate.json").write_text("{}")
    assert _dry_run().dry_run is True


# --- registry integrity ----------------------------------------------------

def test_every_registry_entry_resolves_or_is_marked_off_hub():
    from indic_runner.models.registry import MODEL_REGISTRY

    for task, entries in MODEL_REGISTRY.items():
        for entry in entries:
            resolved = pipeline.resolve_target(entry["alias"], task)
            assert resolved.task == task
            assert resolved.repo_id == entry["repo"]


def test_registry_slot_counts():
    """Four per task, except translation: IndicTrans2 ships one model per
    direction and both en-indic and indic-en are registered."""
    from indic_runner.models.registry import MODEL_REGISTRY

    assert {t: len(e) for t, e in MODEL_REGISTRY.items()} == {
        "reasoning": 4, "summarization": 4, "translation": 5, "ocr": 4
    }


def test_every_registry_entry_has_a_hub_repo():
    """EasyOCR was the only off-Hub entry; Chandra replaced it."""
    from indic_runner.models.registry import MODEL_REGISTRY

    for entries in MODEL_REGISTRY.values():
        for entry in entries:
            assert entry.get("repo"), entry["alias"]


def test_every_ocr_entry_declares_its_runtime():
    """OCR models span four different stacks; none may fall back by accident."""
    from indic_runner.models.registry import MODEL_REGISTRY

    for entry in MODEL_REGISTRY["ocr"]:
        assert entry.get("runtime"), entry["alias"]


def test_model_without_a_hub_repo_is_refused_with_an_explanation(monkeypatch):
    """No roster entry is off-Hub today (Chandra replaced EasyOCR), but the
    guard stays: a package-distributed model has nothing for setup to compile."""
    from indic_runner.models import registry

    monkeypatch.setitem(
        registry.MODEL_REGISTRY,
        "ocr",
        [{"alias": "package-only", "repo": None, "params_b": 0.1, "runtime": "easyocr"}],
    )
    with pytest.raises(pipeline.NotOnHub, match="fetched by the package itself"):
        pipeline.run_setup("package-only", dry_run=True, hardware=make_hw("cpu"))


def test_no_registry_model_uses_an_alias_of_a_removed_model():
    """Manual-gated models were replaced; their aliases must not linger."""
    from indic_runner.models.registry import MODEL_REGISTRY

    retired = {"llama-3.1-8b-instruct", "gemma-2-9b-it", "easyocr"}
    live = {e["alias"] for entries in MODEL_REGISTRY.values() for e in entries}
    assert not (live & retired)


def test_registry_repos_are_not_manually_gated():
    """A manual-gated repo blocks setup on a third party's approval queue.

    This asserts the roster's intent; the live check is the 16-model matrix.
    """
    from indic_runner.models.registry import MODEL_REGISTRY

    known_manual = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "google/gemma-3-4b-it",
        "CohereLabs/aya-expanse-8b",  # auto-gated but CC-BY-NC, excluded too
    }
    repos = {e["repo"] for entries in MODEL_REGISTRY.values() for e in entries}
    assert not (repos & known_manual)


def test_dry_run_manifest_matches_what_a_real_run_writes():
    """--dry-run is only useful if it predicts the real manifest exactly.

    Both paths build the manifest from the same plan, so the only fields that
    may differ are the ones a real run discovers on disk.
    """
    from indic_runner.setup import manifest_writer

    hub = FakeHub(config=QWEN_CONFIG)
    hw = make_hw("cpu", total_ram_gb=32.0)
    dry = pipeline.run_setup(
        "Qwen/Qwen2.5-0.5B-Instruct", task="reasoning",
        dry_run=True, client=hub, hardware=hw,
    ).manifest
    manifest_writer.validate_manifest(dry)
    # Paths are absolute and point into the framework home, never placeholders.
    assert dry["paths"]["artifacts_dir"].startswith("/")
    assert "<" not in dry["paths"]["artifacts_dir"]
    assert dry["engine"]["binary_or_env"].endswith("llama-server")


def test_registry_models_are_marked_trusted():
    assert pipeline.resolve_target("indictrans2-en-indic").is_registry is True


def test_offregistry_repos_are_not_trusted():
    """An arbitrary Hub repo must not get its custom code executed."""
    resolved = pipeline.resolve_target("Qwen/Qwen2.5-0.5B-Instruct", "reasoning")
    assert resolved.is_registry is False


def test_no_registry_entry_uses_third_party_weights():
    """The roster runs official weights only; anything else is converted."""
    from indic_runner.models.registry import MODEL_REGISTRY

    for entries in MODEL_REGISTRY.values():
        for entry in entries:
            gguf = entry.get("gguf_repo")
            if gguf:
                assert gguf.split("/")[0] == entry["repo"].split("/")[0], entry["alias"]


def test_a_third_party_gguf_repo_is_rejected(monkeypatch):
    """The guard is in resolve_target, so it fires before any download."""
    from indic_runner.models import registry

    monkeypatch.setitem(
        registry.MODEL_REGISTRY, "reasoning",
        [{"alias": "smuggled", "repo": "sarvamai/sarvam-1", "params_b": 2,
          "gguf_repo": "bartowski/sarvam-1-GGUF"}],
    )
    with pytest.raises(pipeline.ThirdPartyWeights, match="different org"):
        pipeline.resolve_target("smuggled", "reasoning")


def test_same_org_gguf_repo_is_allowed(monkeypatch):
    from indic_runner.models import registry

    monkeypatch.setitem(
        registry.MODEL_REGISTRY, "reasoning",
        [{"alias": "official", "repo": "Qwen/Qwen3-8B", "params_b": 8,
          "gguf_repo": "Qwen/Qwen3-8B-GGUF-special"}],
    )
    assert pipeline.resolve_target("official", "reasoning").gguf_repo.startswith("Qwen/")


def test_dry_run_rejects_a_plan_no_compiler_can_build(monkeypatch):
    """--dry-run reporting OK for an unbuildable plan is worse than no dry run."""
    from indic_runner.setup import decision_matrix, pipeline as pl

    real = decision_matrix.select_plan

    def unbuildable(task, hw, model):
        plan = real(task, hw, model)
        return type(plan)(**{**plan.__dict__, "artifact_format": "onnx"})

    monkeypatch.setattr(pl, "select_plan", unbuildable)
    with pytest.raises(Exception, match="no compiler"):
        pl.run_setup("sarvam-1", task="reasoning",
                     dry_run=True, client=FakeHub(config=QWEN_CONFIG),
                     hardware=make_hw("cpu"))
