"""The full task x accelerator grid, plus fit-check edges.

select_plan is pure, so every lane is asserted here without the hardware.
"""

from __future__ import annotations

import pytest
from helpers import make_hw, make_model

from indic_runner.setup.decision_matrix import (
    BYTES_PER_PARAM,
    CT2_CONVERTIBLE_ARCHITECTURES,
    TASKS,
    ExecutionPlan,
    InsufficientResources,
    has_ample_vram,
    required_memory_gb,
    select_plan,
)

CPU = make_hw("cpu", total_ram_gb=64.0)
MPS = make_hw("mps", os_name="darwin", arch="arm64", total_ram_gb=64.0)
CUDA_AMPLE = make_hw("cuda", vram_gb=80.0, gpu_name="A100", cuda_version="12.4")
CUDA_TIGHT = make_hw("cuda", vram_gb=8.0, gpu_name="RTX3070", cuda_version="12.4")

LLM = make_model(params_b=7.0, is_encoder_decoder=False)
# NLLB-200's architecture: a seq2seq translator CTranslate2 can convert.
SEQ2SEQ = make_model(params_b=1.1, is_encoder_decoder=True)
SEQ2SEQ = type(SEQ2SEQ)(**{**SEQ2SEQ.__dict__, "architecture": "M2M100ForConditionalGeneration"})
OCR = make_model(params_b=0.8, is_encoder_decoder=True)

# (task, hardware, model) -> (engine, mode, precision, artifact_format)
GRID = [
    ("reasoning", CPU, LLM, "llama.cpp", "daemon", "q4_k_m", "gguf"),
    ("reasoning", MPS, LLM, "llama.cpp", "daemon", "q4_k_m", "gguf"),
    ("reasoning", CUDA_AMPLE, LLM, "vllm", "daemon", "fp16", "safetensors"),
    ("reasoning", CUDA_TIGHT, LLM, "vllm", "daemon", "awq", "awq"),
    ("summarization", CPU, LLM, "llama.cpp", "daemon", "q4_k_m", "gguf"),
    ("summarization", MPS, LLM, "llama.cpp", "daemon", "q4_k_m", "gguf"),
    ("summarization", CUDA_AMPLE, LLM, "vllm", "daemon", "fp16", "safetensors"),
    ("summarization", CUDA_TIGHT, LLM, "vllm", "daemon", "awq", "awq"),
    ("translation", CPU, SEQ2SEQ, "ctranslate2", "in-process", "int8", "ct2"),
    ("translation", MPS, SEQ2SEQ, "ctranslate2", "in-process", "int8", "ct2"),
    ("translation", CUDA_AMPLE, SEQ2SEQ, "ctranslate2", "in-process", "fp16", "ct2"),
    ("ocr", CPU, OCR, "transformers", "in-process", "bf16", "safetensors"),
    ("ocr", MPS, OCR, "transformers", "in-process", "fp16", "safetensors"),
    ("ocr", CUDA_AMPLE, OCR, "transformers", "in-process", "fp16", "safetensors"),
    ("ocr", CUDA_TIGHT, OCR, "transformers", "in-process", "fp16", "safetensors"),
]


@pytest.mark.parametrize("task,hw,model,engine,mode,precision,fmt", GRID)
def test_grid(task, hw, model, engine, mode, precision, fmt):
    plan = select_plan(task, hw, model)
    assert (plan.engine, plan.mode, plan.precision, plan.artifact_format) == (
        engine, mode, precision, fmt
    )


def test_every_task_is_covered_by_the_grid():
    assert {row[0] for row in GRID} == set(TASKS)


def test_vllm_is_never_selected_without_cuda():
    for task in ("reasoning", "summarization"):
        for hw in (CPU, MPS):
            assert select_plan(task, hw, LLM).engine != "vllm"


def test_ocr_is_never_low_bit():
    low_bit = {"int8", "awq", "q4_k_m"}
    for hw in (CPU, MPS, CUDA_AMPLE, CUDA_TIGHT):
        assert select_plan("ocr", hw, OCR).precision not in low_bit


def test_ocr_avoids_fp16_on_cpu():
    # fp16 is not a real CPU dtype in PyTorch; bf16 is, and it keeps fp32's
    # exponent range, so visual resolution survives at half the memory.
    assert select_plan("ocr", CPU, OCR).precision == "bf16"


def test_ocr_on_cpu_halves_memory_versus_fp32():
    """This is what lets a 5B-class OCR model run on a 32GB machine."""
    assert required_memory_gb(5.3, "bf16") < 24.0 < required_memory_gb(5.3, "fp32")


def test_decoder_only_translator_uses_the_llm_lane():
    decoder_translator = make_model(params_b=4.0, is_encoder_decoder=False)
    plan = select_plan("translation", CPU, decoder_translator)
    assert plan.engine == "llama.cpp"
    assert plan.artifact_format == "gguf"


def test_unknown_task_rejected():
    with pytest.raises(ValueError, match="unknown task"):
        select_plan("summarisation", CPU, LLM)  # British spelling is not a task


def test_insufficient_resources_reports_the_shortfall():
    tiny = make_hw("cpu", total_ram_gb=2.0)
    with pytest.raises(InsufficientResources) as excinfo:
        select_plan("reasoning", tiny, make_model(params_b=70.0))
    error = excinfo.value
    assert error.needed_gb > error.budget_gb
    assert "70.0B" in str(error)


def test_model_that_just_fits_is_accepted():
    params = 7.0
    needed = required_memory_gb(params, "q4_k_m")
    hw = make_hw("cpu", total_ram_gb=needed / 0.75 + 0.1)
    assert select_plan("reasoning", hw, make_model(params_b=params)).precision == "q4_k_m"


@pytest.mark.parametrize("precision", sorted(BYTES_PER_PARAM))
def test_required_memory_scales_with_precision(precision):
    assert required_memory_gb(1.0, precision) > 0


def test_has_ample_vram_is_false_without_cuda():
    assert has_ample_vram(CPU, 1.0) is False
    assert has_ample_vram(MPS, 1.0) is False


def test_ample_vram_depends_on_model_size():
    # The same card is ample for a small model and constrained for a big one.
    assert has_ample_vram(CUDA_TIGHT, 1.1) is True
    assert has_ample_vram(CUDA_TIGHT, 7.0) is False


def test_context_length_is_carried_from_the_model():
    model = make_model(params_b=1.0, context_length=131072)
    assert select_plan("reasoning", CPU, model).runtime_parameters["context_length"] == 131072


def test_llama_cpp_plans_carry_thread_count():
    hw = make_hw("cpu", cpu_threads=12)
    plan = select_plan("reasoning", hw, make_model(params_b=1.0))
    assert plan.runtime_parameters["threads"] == 12


def test_non_llama_plans_omit_thread_count():
    plan = select_plan("ocr", CUDA_AMPLE, OCR)
    assert "threads" not in plan.runtime_parameters


def test_batch_size_shrinks_as_memory_tightens():
    roomy = select_plan("reasoning", make_hw("cpu", total_ram_gb=64.0), LLM)
    cramped = select_plan("reasoning", make_hw("cpu", total_ram_gb=8.0), LLM)
    assert roomy.runtime_parameters["max_batch_size"] > cramped.runtime_parameters["max_batch_size"]


def test_plan_is_immutable():
    plan = select_plan("reasoning", CPU, LLM)
    with pytest.raises(AttributeError):
        plan.precision = "fp16"  # type: ignore[misc]
    assert isinstance(plan, ExecutionPlan)


# --- OCR runtimes are not one family ---------------------------------------

@pytest.mark.parametrize(
    "runtime,fmt",
    [
        ("transformers", "safetensors"),
        ("surya", "safetensors"),
        ("paddleocr", "paddle"),   # Paddle weights, not safetensors
        ("easyocr", "package"),    # ships its own weights
    ],
)
def test_ocr_runtime_drives_engine_and_format(runtime, fmt):
    model = make_model(params_b=0.5, is_encoder_decoder=True)
    model = type(model)(**{**model.__dict__, "ocr_runtime": runtime})
    plan = select_plan("ocr", CPU, model)
    assert plan.engine == runtime
    assert plan.artifact_format == fmt


def test_ocr_defaults_to_transformers_when_unspecified():
    assert select_plan("ocr", CPU, make_model(params_b=0.5)).engine == "transformers"


def test_ocr_precision_ignores_the_runtime():
    for runtime in ("transformers", "surya", "paddleocr", "easyocr"):
        model = make_model(params_b=0.5)
        model = type(model)(**{**model.__dict__, "ocr_runtime": runtime})
        assert select_plan("ocr", CPU, model).precision == "bf16"
        assert select_plan("ocr", CUDA_AMPLE, model).precision == "fp16"


# --- CTranslate2 convertibility --------------------------------------------

def _seq2seq(architecture, params_b=1.1):
    model = make_model(params_b=params_b, is_encoder_decoder=True)
    return type(model)(**{**model.__dict__, "architecture": architecture})


@pytest.mark.parametrize("architecture", sorted(CT2_CONVERTIBLE_ARCHITECTURES))
def test_convertible_seq2seq_uses_ctranslate2(architecture):
    assert select_plan("translation", CPU, _seq2seq(architecture)).engine == "ctranslate2"


def test_unconvertible_seq2seq_falls_back_to_transformers():
    """IndicTrans2's custom config has no CTranslate2 converter; routing it
    there would fail only after its weights had been downloaded."""
    plan = select_plan("translation", CPU, _seq2seq("IndicTransForConditionalGeneration"))
    assert plan.engine == "transformers"
    assert plan.artifact_format == "safetensors"


def test_unconvertible_seq2seq_is_never_low_bit():
    low_bit = {"int8", "awq", "q4_k_m"}
    for hw in (CPU, MPS, CUDA_AMPLE, CUDA_TIGHT):
        plan = select_plan("translation", hw, _seq2seq("IndicTransForConditionalGeneration"))
        assert plan.precision not in low_bit


def test_registry_seq2seq_models_route_somewhere_runnable():
    """Every encoder-decoder translator resolves to an engine that can load it."""
    for architecture in ("M2M100ForConditionalGeneration", "T5ForConditionalGeneration",
                         "IndicTransForConditionalGeneration"):
        plan = select_plan("translation", CPU, _seq2seq(architecture))
        assert plan.engine in ("ctranslate2", "transformers")
