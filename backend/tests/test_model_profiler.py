"""Model profiling driven by a fake Hub - no network."""

from __future__ import annotations

import pytest
from helpers import FakeHub

from indic_runner.setup import model_profiler as mp

QWEN_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
}
NLLB_CONFIG = {
    "architectures": ["M2M100ForConditionalGeneration"],
    "d_model": 1024,
    "num_layers": 12,
    "vocab_size": 256206,
    "max_position_embeddings": 1024,
    "is_encoder_decoder": True,
}


def test_profiles_a_decoder_model():
    hub = FakeHub(config=QWEN_CONFIG, sha="7ae5576", safetensors_total=494_000_000)
    profile = mp.profile_model("Qwen/Qwen2.5-0.5B-Instruct", hub)
    assert profile.architecture == "Qwen2ForCausalLM"
    assert profile.param_count_b == 0.49
    assert profile.context_length == 32768
    assert profile.is_encoder_decoder is False
    assert profile.revision == "7ae5576"


def test_detects_encoder_decoder_by_architecture():
    hub = FakeHub(config=NLLB_CONFIG)
    assert mp.profile_model("facebook/nllb-200", hub).is_encoder_decoder is True


def test_detects_encoder_decoder_by_config_flag():
    hub = FakeHub(config={"architectures": ["CustomArch"], "is_encoder_decoder": True})
    assert mp.profile_model("org/custom", hub).is_encoder_decoder is True


def test_finds_prebuilt_gguf_repo():
    hub = FakeHub(config=QWEN_CONFIG, existing_repos={"org/model-GGUF"})
    profile = mp.profile_model("org/model", hub)
    assert profile.prebuilt_repos["gguf"] == "org/model-GGUF"


def test_absent_prebuilt_repos_are_not_reported():
    hub = FakeHub(config=QWEN_CONFIG)
    assert mp.profile_model("org/model", hub).prebuilt_repos == {}


@pytest.mark.parametrize(
    "files,expected",
    [
        (["model.safetensors"], {"safetensors"}),
        (["model-q4_k_m.gguf"], {"gguf"}),
        (["model.bin", "config.json"], {"ct2"}),
        (["pytorch_model.bin"], {"pytorch"}),
        (["model.safetensors", "pytorch_model.bin"], {"safetensors", "pytorch"}),
        (["README.md"], set()),
    ],
)
def test_classify_formats(files, expected):
    assert mp.classify_formats(files) == expected


def test_param_count_prefers_safetensors_total():
    assert mp.estimate_params_b(QWEN_CONFIG, safetensors_total=7_000_000_000) == 7.0


def test_param_count_falls_back_to_config_estimate():
    estimate = mp.estimate_params_b(QWEN_CONFIG)
    assert 0.2 < estimate < 1.0  # 0.5B-class model


def test_param_count_falls_back_to_repo_name():
    hub = FakeHub(files=["README.md"], config={})
    profile = mp.profile_model("org/Airavata-7B", hub)
    assert profile.param_count_b == 7.0


def test_param_count_falls_back_to_registry_value():
    hub = FakeHub(files=["README.md"], config={})
    profile = mp.profile_model("org/mystery", hub, registry_params_b=3.3)
    assert profile.param_count_b == 3.3


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"max_position_embeddings": 8192}, 8192),
        ({"n_positions": 2048}, 2048),
        ({"seq_length": 4096}, 4096),
        ({}, 4096),  # documented default
    ],
)
def test_resolve_context_length(config, expected):
    assert mp.resolve_context_length(config) == expected


def test_unreadable_config_does_not_abort_profiling():
    class Broken(FakeHub):
        def hf_hub_download(self, repo_id, filename, **kwargs):
            raise OSError("network died mid-download")

    profile = mp.profile_model("org/Model-2B", Broken(config={}))
    assert profile.param_count_b == 2.0  # recovered from the name
    assert profile.architecture == "unknown"


# --- gated repositories ----------------------------------------------------

class _GatedRepoError(Exception):
    """Mimics huggingface_hub.errors.GatedRepoError by name."""


def test_gated_repo_fails_loudly_instead_of_guessing():
    """A swallowed 401 yields an empty config, and the decision matrix would
    then pick a plausible but wrong engine that only fails at download time."""

    class Gated(FakeHub):
        def hf_hub_download(self, repo_id, filename, **kwargs):
            raise _GatedRepoError("401 Client Error")

    _GatedRepoError.__name__ = "GatedRepoError"
    with pytest.raises(mp.GatedRepository) as excinfo:
        mp.profile_model("meta-llama/Llama-3.1-8B-Instruct", Gated(config={}))
    message = str(excinfo.value)
    assert "gated" in message
    # No token configured (see the isolate_credentials fixture), so the message
    # must point at the token, not at licence acceptance.
    assert "HF_TOKEN" in message


@pytest.mark.parametrize("status", [401, 403])
def test_http_auth_failures_are_treated_as_gating(status):
    class Response:
        status_code = status

    class HttpError(Exception):
        response = Response()

    assert mp._is_gated(HttpError()) is True


@pytest.mark.parametrize("status", [404, 500, None])
def test_other_failures_are_not_gating(status):
    class Response:
        status_code = status

    class HttpError(Exception):
        response = Response()

    assert mp._is_gated(HttpError()) is False


def test_ocr_runtime_is_carried_onto_the_profile():
    hub = FakeHub(config=QWEN_CONFIG)
    profile = mp.profile_model("org/ocr", hub, ocr_runtime="paddleocr")
    assert profile.ocr_runtime == "paddleocr"


@pytest.mark.parametrize(
    "gating,has_token,expected",
    [
        ("auto", True, "immediately"),      # one click, instant
        ("manual", True, "owner"),          # request and wait
        (None, True, "most likely lacks"),  # unknown gating type
        ("manual", False, "HF_TOKEN"),      # no token: fix that first
    ],
)
def test_gated_message_matches_what_the_user_must_do(gating, has_token, expected):
    error = mp.GatedRepository("org/repo", has_token=has_token, gating=gating)
    assert expected in str(error)
    assert "https://huggingface.co/org/repo" in str(error)


def test_gating_type_is_taken_from_hub_metadata():
    class GatedInfo(FakeHub):
        def model_info(self, repo_id, **kwargs):
            info = super().model_info(repo_id, **kwargs)
            info.gated = "manual"
            return info

        def hf_hub_download(self, repo_id, filename, **kwargs):
            raise _GatedRepoError("401")

    _GatedRepoError.__name__ = "GatedRepoError"
    with pytest.raises(mp.GatedRepository) as excinfo:
        mp.profile_model("org/repo", GatedInfo(config={}))
    assert excinfo.value.gating == "manual"


def test_auto_map_marks_a_model_as_needing_remote_code():
    """IndicTrans2's config carries auto_map, meaning custom modelling code."""
    config = {
        "architectures": ["IndicTransForConditionalGeneration"],
        "auto_map": {"AutoModelForSeq2SeqLM": "modeling_indictrans.IndicTrans"},
    }
    profile = mp.profile_model("ai4bharat/indictrans2", FakeHub(config=config))
    assert profile.requires_remote_code is True


def test_standard_models_need_no_remote_code():
    profile = mp.profile_model("Qwen/Qwen2.5-7B", FakeHub(config=QWEN_CONFIG))
    assert profile.requires_remote_code is False
