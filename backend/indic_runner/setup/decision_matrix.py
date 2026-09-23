"""Selects engine + precision from task, hardware, and model profile.

Pure: no I/O, no subprocesses, no network. Every decision is a lookup against
the tables below, so adding an accelerator or task is a data change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from indic_runner.setup.hardware_profiler import HardwareProfile
from indic_runner.setup.model_profiler import ModelProfile

TASKS = ("reasoning", "summarization", "translation", "ocr")
ACCELERATORS = ("cuda", "mps", "cpu")

# Bytes per parameter for each precision, used for the fit check.
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "awq": 0.6,
    "q4_k_m": 0.55,
}

# Activations, KV cache, and runtime overhead, as a fraction of weight size.
RUNTIME_OVERHEAD_FACTOR = 1.2

# A CUDA card must hold fp16 weights plus overhead to be considered "ample";
# otherwise the constrained (AWQ) lane is selected.
AMPLE_VRAM_MARGIN_GB = 2.0


class InsufficientResources(RuntimeError):
    """Raised when no supported precision fits in the available memory."""

    def __init__(self, task: str, params_b: float, budget_gb: float, needed_gb: float):
        self.task = task
        self.params_b = params_b
        self.budget_gb = budget_gb
        self.needed_gb = needed_gb
        super().__init__(
            f"{task}: a {params_b}B model needs ~{needed_gb:.1f}GB at the smallest "
            f"supported precision, but only {budget_gb:.1f}GB is available"
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """The engine/precision contract handed to the artifact compiler."""

    task: str
    engine: str  # "llama.cpp" | "vllm" | "ctranslate2" | "transformers"
    mode: str  # "daemon" | "in-process"
    precision: str
    artifact_format: str  # "gguf" | "ct2" | "safetensors" | "awq"
    runtime_parameters: dict = field(default_factory=dict)


def weight_size_gb(params_b: float, precision: str) -> float:
    """Approximate on-device size of the weights at a given precision."""
    return params_b * BYTES_PER_PARAM[precision]


def required_memory_gb(params_b: float, precision: str) -> float:
    """Weights plus runtime overhead."""
    return weight_size_gb(params_b, precision) * RUNTIME_OVERHEAD_FACTOR


def has_ample_vram(hw: HardwareProfile, params_b: float) -> bool:
    """True when fp16 weights plus overhead and a safety margin fit in VRAM."""
    if hw.accelerator != "cuda" or hw.vram_gb is None:
        return False
    return hw.vram_gb >= required_memory_gb(params_b, "fp16") + AMPLE_VRAM_MARGIN_GB


# --------------------------------------------------------------------------
# Per-task lane selection. Each returns (engine, mode, precision, format).
# --------------------------------------------------------------------------

def _llama_cpp_lane() -> tuple[str, str, str, str]:
    return "llama.cpp", "daemon", "q4_k_m", "gguf"


def _decoder_lane(hw: HardwareProfile, model: ModelProfile) -> tuple[str, str, str, str]:
    """Decoder-only LLMs: reasoning, summarization, decoder translators."""
    if hw.accelerator == "cuda":
        if has_ample_vram(hw, model.param_count_b):
            return "vllm", "daemon", "fp16", "safetensors"
        return "vllm", "daemon", "awq", "awq"
    # vLLM has no production CPU backend, and MPS is memory-bandwidth bound:
    # both route to llama.cpp with a 4-bit GGUF.
    return _llama_cpp_lane()


# Encoder-decoder architectures CTranslate2 can actually convert. A model
# outside this set fails the converter with "No conversion is registered",
# and only after its weights have been downloaded -- IndicTrans2 ships a
# custom IndicTransConfig and is the case that surfaced this.
CT2_CONVERTIBLE_ARCHITECTURES = frozenset(
    {
        "BartForConditionalGeneration",
        "M2M100ForConditionalGeneration",
        "MBartForConditionalGeneration",
        "MT5ForConditionalGeneration",
        "MarianMTModel",
        "PegasusForConditionalGeneration",
        "T5ForConditionalGeneration",
        "WhisperForConditionalGeneration",
    }
)


def _translation_lane(hw: HardwareProfile, model: ModelProfile) -> tuple[str, str, str, str]:
    if not model.is_encoder_decoder:
        return _decoder_lane(hw, model)

    if model.architecture not in CT2_CONVERTIBLE_ARCHITECTURES:
        # No CTranslate2 path: run the model as published, under transformers.
        # Never low-bit here either -- these are the seq2seq translators whose
        # quality the roster is chosen for.
        precision = "bf16" if hw.accelerator == "cpu" else "fp16"
        return "transformers", "in-process", precision, "safetensors"

    # CTranslate2 runs in-process and is int8-optimised on CPU; fp16 is only
    # worth it when there is a CUDA device with room to spare.
    if hw.accelerator == "cuda" and has_ample_vram(hw, model.param_count_b):
        return "ctranslate2", "in-process", "fp16", "ct2"
    return "ctranslate2", "in-process", "int8", "ct2"


# OCR engines are not one family: only some are transformers models, so the
# runtime comes from the registry rather than being assumed.
OCR_ARTIFACT_FORMAT = {
    "transformers": "safetensors",
    "surya": "safetensors",
    "paddleocr": "paddle",
    "easyocr": "package",
}


def _ocr_lane(hw: HardwareProfile, model: ModelProfile) -> tuple[str, str, str, str]:
    # Never low-bit: quantisation destroys the feature resolution needed to
    # separate Indic matras. bf16 keeps fp32's exponent range (only the mantissa
    # narrows) and, unlike fp16, is a real CPU dtype in PyTorch -- so it halves
    # memory without the precision objection. On CPUs lacking native bf16 it is
    # emulated, trading some speed for the ability to run larger OCR models at all.
    precision = "bf16" if hw.accelerator == "cpu" else "fp16"
    runtime = model.ocr_runtime or "transformers"
    return runtime, "in-process", precision, OCR_ARTIFACT_FORMAT.get(runtime, "safetensors")


_LANES = {
    "reasoning": _decoder_lane,
    "summarization": _decoder_lane,
    "translation": _translation_lane,
    "ocr": _ocr_lane,
}


def _batch_size(budget_gb: float, weights_gb: float) -> int:
    """Scale batch size with the memory left after weights are resident."""
    spare = max(budget_gb - weights_gb, 0.0)
    if spare >= 16:
        return 64
    if spare >= 8:
        return 32
    if spare >= 4:
        return 16
    if spare >= 2:
        return 8
    return 4


def select_plan(task: str, hw: HardwareProfile, model: ModelProfile) -> ExecutionPlan:
    """Choose engine, mode, precision and artifact format for a workload.

    Raises:
        ValueError: unknown task.
        InsufficientResources: the model cannot fit at any supported precision.
    """
    if task not in _LANES:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")

    engine, mode, precision, artifact_format = _LANES[task](hw, model)

    budget = hw.memory_budget_gb
    needed = required_memory_gb(model.param_count_b, precision)
    if needed > budget:
        raise InsufficientResources(task, model.param_count_b, budget, needed)

    weights = weight_size_gb(model.param_count_b, precision)
    runtime_parameters: dict = {
        "max_batch_size": _batch_size(budget, weights),
        "context_length": model.context_length,
    }
    if engine == "llama.cpp":
        runtime_parameters["threads"] = hw.cpu_threads

    return ExecutionPlan(
        task=task,
        engine=engine,
        mode=mode,
        precision=precision,
        artifact_format=artifact_format,
        runtime_parameters=runtime_parameters,
    )
