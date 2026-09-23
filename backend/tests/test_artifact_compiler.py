"""Compiler dispatch, eviction safety, and the AWQ limitation."""

from __future__ import annotations

import pytest
from helpers import FakeHub, make_model

from indic_runner.setup import artifact_compiler as ac
from indic_runner.setup.binary_manager import BinaryBundle
from indic_runner.setup.decision_matrix import ExecutionPlan


def plan_for(fmt, precision="int8"):
    return ExecutionPlan("translation", "e", "in-process", precision, fmt, {})


@pytest.mark.parametrize(
    "fmt,precision,expected",
    [
        ("ct2", "int8", ac.Ct2Compiler),
        ("awq", "awq", ac.AwqCompiler),
        ("safetensors", "fp16", ac.SafetensorsCompiler),
        ("safetensors", "fp32", ac.SafetensorsCompiler),
    ],
)
def test_dispatch_by_artifact_format(fmt, precision, expected):
    assert isinstance(compiler := ac.compiler_for(plan_for(fmt, precision), "a"), expected)
    assert compiler is not None


def test_gguf_dispatch_requires_a_binary_bundle(tmp_path):
    with pytest.raises(ac.CompilationError, match="bundle is required"):
        ac.compiler_for(plan_for("gguf", "q4_k_m"), "a")
    bundle = BinaryBundle(root=tmp_path, build="b11118")
    assert isinstance(
        ac.compiler_for(plan_for("gguf", "q4_k_m"), "a", bundle), ac.GgufCompiler
    )


def test_unknown_format_is_rejected():
    with pytest.raises(ac.CompilationError, match="no compiler"):
        ac.compiler_for(plan_for("onnx"), "a")


def test_fp32_and_fp16_land_in_separate_directories():
    fp32 = ac.compiler_for(plan_for("safetensors", "fp32"), "alias")
    fp16 = ac.compiler_for(plan_for("safetensors", "fp16"), "alias")
    assert fp32.suffix != fp16.suffix


def test_awq_without_a_published_build_fails_actionably():
    compiler = ac.AwqCompiler("alias")
    model = make_model(params_b=7.0, prebuilt={})
    with pytest.raises(ac.CompilationError) as excinfo:
        compiler.ensure(plan_for("awq", "awq"), model)
    message = str(excinfo.value)
    assert "no published AWQ build" in message
    assert "calibration corpus" in message  # says why, not just that it failed


def test_verify_rejects_a_missing_artifact(tmp_path):
    with pytest.raises(ac.CompilationError, match="was not produced"):
        ac._verify_non_empty(tmp_path / "absent.gguf", "artifact")


def test_verify_rejects_an_empty_file(tmp_path):
    empty = tmp_path / "empty.gguf"
    empty.touch()
    with pytest.raises(ac.CompilationError, match="is empty"):
        ac._verify_non_empty(empty, "artifact")


def test_verify_rejects_an_empty_directory(tmp_path):
    hollow = tmp_path / "hollow"
    hollow.mkdir()
    with pytest.raises(ac.CompilationError, match="is empty"):
        ac._verify_non_empty(hollow, "artifact")


def test_verify_accepts_a_real_artifact(tmp_path):
    good = tmp_path / "model.gguf"
    good.write_bytes(b"GGUF")
    ac._verify_non_empty(good, "artifact")


def test_eviction_removes_the_source_tree(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    assert ac._evict(source) is True
    assert not source.exists()


def test_eviction_is_a_noop_when_there_is_nothing_to_remove(tmp_path):
    assert ac._evict(tmp_path / "never-existed") is False


def test_loader_path_var_matches_the_platform(monkeypatch):
    import platform

    for system, expected in [
        ("Linux", "LD_LIBRARY_PATH"),
        ("Darwin", "DYLD_LIBRARY_PATH"),
        ("Windows", None),  # DLLs resolve from the executable's own directory
    ]:
        monkeypatch.setattr(platform, "system", lambda s=system: s)
        assert ac.loader_path_var() == expected


def test_model_dirs_live_under_the_framework_home():
    assert ac.model_dir("alias", "gguf").parent == ac.DIRS["models"]
    assert ac.model_dir("alias", "gguf").name == "alias-gguf"


def test_isolated_envs_are_distinct_per_toolchain():
    names = {ac.GGUF_ENV[0], ac.CT2_ENV[0], ac.OCR_ENV[0]}
    assert len(names) == 3  # never a shared dependency resolution


# --- redundant weight formats ----------------------------------------------

def test_safetensors_repos_skip_duplicate_weight_formats():
    """Repos often publish .safetensors and .bin with identical tensors.

    IndicTrans2-en-indic ships both at 4.46GB each; fetching both doubles the
    download and the disk for no benefit.
    """
    model = make_model(params_b=1.1)
    model = type(model)(**{**model.__dict__, "available_formats": frozenset({"safetensors"})})
    patterns = ac.source_ignore_patterns(model)
    assert "*.bin" in patterns


def test_legacy_only_repos_still_download_their_weights():
    model = make_model(params_b=1.1)
    model = type(model)(**{**model.__dict__, "available_formats": frozenset()})
    assert ac.source_ignore_patterns(model) is None


def test_ignore_patterns_reach_the_hub_call(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    model = make_model(params_b=1.1)
    model = type(model)(**{**model.__dict__, "available_formats": frozenset({"safetensors"})})
    ac._snapshot("org/repo", tmp_path, ignore_patterns=ac.source_ignore_patterns(model))
    assert "*.bin" in captured["ignore_patterns"]


# --- custom remote code ----------------------------------------------------

def _remote_code_model():
    model = make_model(params_b=1.1, is_encoder_decoder=True)
    return type(model)(**{**model.__dict__, "requires_remote_code": True})


def test_untrusted_remote_code_is_refused_before_execution():
    """IndicTrans2 ships custom modelling code; converting it runs that code.

    Allowed only for the curated roster, never for an arbitrary Hub repo.
    """
    compiler = ac.Ct2Compiler("alias", trusted=False)
    with pytest.raises(ac.UntrustedRemoteCode) as excinfo:
        compiler.ensure(plan_for("ct2"), _remote_code_model())
    message = str(excinfo.value)
    assert "custom modelling code" in message
    assert "registry.py" in message  # says how to opt in


def test_models_without_remote_code_need_no_trust():
    """The common case must not require the roster."""
    compiler = ac.Ct2Compiler("alias", trusted=False)
    model = make_model(params_b=1.1, is_encoder_decoder=True)
    assert model.requires_remote_code is False
    # Reaches the snapshot step rather than raising UntrustedRemoteCode.
    with pytest.raises(Exception) as excinfo:
        compiler.ensure(plan_for("ct2"), model)
    assert not isinstance(excinfo.value, ac.UntrustedRemoteCode)


def test_trusted_flag_reaches_the_compiler():
    plan = plan_for("ct2")
    assert ac.compiler_for(plan, "a", trusted=True).trusted is True
    assert ac.compiler_for(plan, "a", trusted=False).trusted is False


def test_ct2_env_pins_transformers_below_5():
    """transformers 5 dropped transformers.onnx, which IndicTrans2's custom
    modelling code imports. Only the CT2 path needs the pin."""
    assert "<5" in ac.TRANSFORMERS_PIN
    assert ac.TRANSFORMERS_PIN in ac.CT2_ENV[1]
    assert "transformers" not in ac.CT2_ENV[1]  # the bare, unpinned name


def test_gguf_env_is_not_held_back_by_the_ct2_pin():
    """llama.cpp's converter runs on transformers 5; pinning it here would
    hold the GGUF lane back for a problem it does not have."""
    assert "transformers" in ac.GGUF_ENV[1]
    assert ac.TRANSFORMERS_PIN not in ac.GGUF_ENV[1]


def test_pytorch_only_repo_keeps_its_weights():
    """NLLB-200 publishes only pytorch_model.bin. Skipping .bin there would
    download a config with no model and fail the converter."""
    model = make_model(params_b=0.6)
    model = type(model)(**{**model.__dict__, "available_formats": frozenset({"pytorch"})})
    assert ac.source_ignore_patterns(model) is None


def test_repo_with_both_formats_skips_the_duplicate():
    model = make_model(params_b=0.6)
    model = type(model)(
        **{**model.__dict__, "available_formats": frozenset({"safetensors", "pytorch"})}
    )
    assert "*.bin" in ac.source_ignore_patterns(model)


def test_orphan_source_is_pruned(tmp_path, monkeypatch):
    """A re-plan can switch lanes (IndicTrans2 moved CT2 -> transformers),
    leaving the abandoned lane's multi-GB source with nothing to evict it."""
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)
    orphan = ac.model_dir("alias", "src")
    orphan.mkdir(parents=True)
    (orphan / "model.safetensors").write_bytes(b"weights")

    removed = ac.prune_orphan_sources("alias")
    assert orphan in removed
    assert not orphan.exists()


def test_pruning_never_deletes_the_artifact_in_use(tmp_path, monkeypatch):
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)
    keep = ac.model_dir("alias", "src")
    keep.mkdir(parents=True)
    (keep / "model.bin").write_bytes(b"x")

    assert ac.prune_orphan_sources("alias", keep=keep) == []
    assert keep.exists()


def test_pruning_is_a_noop_when_nothing_is_orphaned(tmp_path, monkeypatch):
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)
    assert ac.prune_orphan_sources("never-set-up") == []


# --- quantization verification & provenance --------------------------------

def test_wrong_quantization_is_refused(tmp_path, monkeypatch):
    """A Q5 file under a manifest claiming q4_k_m would misreport the run."""
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)
    target = ac.model_dir("alias", "gguf")
    target.mkdir(parents=True)
    (target / "model-Q5_K_M.gguf").write_bytes(b"GGUF")

    with pytest.raises(ac.QuantizationMismatch) as excinfo:
        ac.GgufCompiler._verify_quantization(target, "q4_k_m", "someone/model-GGUF")
    message = str(excinfo.value)
    assert "q4_k_m" in message
    assert "Q5_K_M" in message  # names what it found instead


def test_matching_quantization_passes(tmp_path):
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    ac.GgufCompiler._verify_quantization(tmp_path, "q4_k_m", "org/repo")


def test_empty_gguf_directory_is_refused(tmp_path):
    with pytest.raises(ac.QuantizationMismatch, match="no .gguf files"):
        ac.GgufCompiler._verify_quantization(tmp_path, "q4_k_m", "org/repo")


@pytest.mark.parametrize(
    "base,prebuilt,expected",
    [
        ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-GGUF", "official"),
        ("sarvamai/sarvam-1", "bartowski/sarvam-1-GGUF", "third-party"),
        ("ai4bharat/Airavata", None, "converted"),
    ],
)
def test_provenance_reflects_who_published_the_artifact(base, prebuilt, expected):
    assert ac.GgufCompiler._provenance(base, prebuilt) == expected


# --- uniform quantization --------------------------------------------------

def test_published_gguf_is_not_used_by_default():
    """Uniform conversion: sourcing some models from their author's build and
    converting the rest makes part of every result a quantizer property."""
    assert ac.GgufCompiler.USE_PUBLISHED_GGUF is False


def test_official_sibling_is_ignored_in_favour_of_converting(tmp_path, monkeypatch):
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)
    compiler = ac.GgufCompiler("alias", BinaryBundle(root=tmp_path, build="b11118"))
    model = make_model(params_b=1.0, repo_id="Qwen/Qwen3-8B",
                       prebuilt={"gguf": "Qwen/Qwen3-8B-GGUF"})

    captured = {}
    monkeypatch.setattr(ac, "_snapshot",
                        lambda repo, dest, **kw: captured.setdefault("repo", repo))
    monkeypatch.setattr(ac.env_manager, "ensure_env", lambda *a, **k: None)
    monkeypatch.setattr(ac.converter_manager, "ensure_converter",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached converter")))

    with pytest.raises(RuntimeError, match="reached converter"):
        compiler.ensure(plan_for("gguf", "q4_k_m"), model)
    # It fetched the official safetensors, not the published GGUF.
    assert captured["repo"] == "Qwen/Qwen3-8B"


def test_sibling_detection_still_runs_for_awq():
    """Detection is kept: AWQ still uses it, and GGUF may again one day."""
    from indic_runner.setup import model_profiler as mp

    hub = FakeHub(config={"architectures": ["X"]}, existing_repos={"org/m-AWQ"})
    assert mp.profile_model("org/m", hub).prebuilt_repos.get("awq") == "org/m-AWQ"


# --- every emittable format has a compiler ---------------------------------

def test_every_format_the_matrix_emits_has_a_compiler():
    """The gap this caught: `paddle` had no compiler, so PaddleOCR would fail
    at real setup while --dry-run still reported OK."""
    from indic_runner.setup.decision_matrix import OCR_ARTIFACT_FORMAT

    emittable = {"gguf", "ct2", "safetensors", "awq"} | set(OCR_ARTIFACT_FORMAT.values())
    assert emittable <= ac.SUPPORTED_ARTIFACT_FORMATS


@pytest.mark.parametrize("fmt", sorted(ac.SUPPORTED_ARTIFACT_FORMATS))
def test_each_supported_format_dispatches(fmt, tmp_path):
    bundle = BinaryBundle(root=tmp_path, build="b11118")
    assert ac.compiler_for(plan_for(fmt), "alias", bundle) is not None


def test_validate_accepts_without_a_bundle():
    """--dry-run must be able to check GGUF plans it has no bundle for."""
    ac.validate_artifact_format(plan_for("gguf", "q4_k_m"))


def test_validate_rejects_an_unknown_format():
    with pytest.raises(ac.CompilationError, match="no compiler"):
        ac.validate_artifact_format(plan_for("onnx"))


def test_paddle_compiler_requires_paddle_weights(tmp_path, monkeypatch):
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)
    monkeypatch.setattr(ac, "_snapshot", lambda repo, dest, **kw: dest.mkdir(parents=True, exist_ok=True))
    with pytest.raises(ac.CompilationError, match="pdiparams"):
        ac.PaddleCompiler("alias").ensure(plan_for("paddle"), make_model(params_b=0.1))


def test_paddle_compiler_accepts_inference_weights(tmp_path, monkeypatch):
    monkeypatch.setitem(ac.DIRS, "models", tmp_path)

    def fake(repo, dest, **kw):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "inference.pdiparams").write_bytes(b"w")

    monkeypatch.setattr(ac, "_snapshot", fake)
    paths = ac.PaddleCompiler("alias").ensure(plan_for("paddle"), make_model(params_b=0.1))
    assert (paths.artifacts_dir / "inference.pdiparams").exists()
