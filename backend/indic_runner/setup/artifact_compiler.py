"""Downloads base weights, converts to the target format, evicts raw FP16.

Each artifact format has a compiler. Conversion runs inside an isolated env so
the orchestrator never imports torch, and source weights are deleted only once
the converted artifact has been verified on disk.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from indic_runner.config import DIRS, hf_token
from indic_runner.setup import converter_manager, env_manager
from indic_runner.setup.binary_manager import BinaryBundle
from indic_runner.setup.decision_matrix import ExecutionPlan
from indic_runner.setup.model_profiler import ModelProfile

# Isolated environments, each pinned to one toolchain.
# transformers 5 removed ``transformers.onnx``, which custom modelling code in
# older repos still imports -- IndicTrans2's configuration_indictrans.py does,
# and CTranslate2 conversion dies on ModuleNotFoundError. The pin is scoped to
# that path: llama.cpp's converter runs fine on transformers 5, and pinning it
# there would hold the GGUF lane back for no reason. The version is part of the
# env fingerprint, so changing it rebuilds rather than reusing a stale env.
TRANSFORMERS_PIN = "transformers>=4.40,<5"

# The converter tree is vendored (see converter_manager), so this env supplies
# only its runtime dependencies. gguf comes from the vendored gguf-py.
GGUF_ENV = ("gguf-convert", ("torch", "numpy", "sentencepiece", "transformers", "protobuf"))
CT2_ENV = ("ct2-convert", ("torch", "ctranslate2", TRANSFORMERS_PIN, "sentencepiece"))
OCR_ENV = ("ocr-runtime", ("torch", "torchvision", "transformers", "pillow"))

# Run-phase envs for the in-process engines, keyed by (task, engine). The pins
# conflict, which is why they are separate: IndicTrans2's remote code and
# surya 0.6 need transformers<5, while Chandra (qwen3_5) and Bodhan's IndicOCR
# need >=5.x. The CTranslate2 worker reuses the conversion env.
TRANSLATE_ENV = (
    "tx-translate",
    ("torch", TRANSFORMERS_PIN, "sentencepiece", "protobuf", "accelerate", "IndicTransToolkit"),
)
OCR_HF_ENV = (
    "ocr-hf",
    ("torch", "torchvision", "transformers>=5.7", "accelerate>=1.1", "pillow", "numpy",
     "huggingface_hub>=1.0", "chandra-ocr[hf]"),
)
OPENCV_VERSION = "4.10.0.84"
# surya-ocr 0.6.x is the generation that used the roster's vikp/surya_rec2.
# Its config classes break on transformers>=4.5x (KeyError: 'encoder'), so it
# gets a 2024-era pin. Its opencv-python needs libxcb, absent on servers, so the
# headless build of the same version is overlaid (see ENV_OVERLAYS).
OCR_SURYA_ENV = (
    "ocr-surya",
    ("torch", "surya-ocr==0.6.13", "transformers==4.45.2", "tokenizers<0.21", "pillow<11",
     f"opencv-python=={OPENCV_VERSION}"),
)
OCR_SURYA_OVERLAY = (f"opencv-python-headless=={OPENCV_VERSION}",)
# paddlex's dependency check wants the opencv-contrib-python dist, whose cv2
# links libGL (absent on servers and containers); the headless build of the same
# version is overlaid so the import works while the check still passes.
OCR_PADDLE_ENV = (
    "ocr-paddle",
    ("paddlepaddle", "paddleocr", "pillow", f"opencv-contrib-python=={OPENCV_VERSION}"),
)
OCR_PADDLE_OVERLAY = (f"opencv-contrib-python-headless=={OPENCV_VERSION}",)

RUNTIME_ENVS = {
    ("translation", "ctranslate2"): CT2_ENV,
    ("translation", "transformers"): TRANSLATE_ENV,
    ("ocr", "transformers"): OCR_HF_ENV,
    ("ocr", "surya"): OCR_SURYA_ENV,
    ("ocr", "paddleocr"): OCR_PADDLE_ENV,
    ("ocr", "easyocr"): OCR_ENV,
}


ENV_OVERLAYS = {OCR_PADDLE_ENV[0]: OCR_PADDLE_OVERLAY, OCR_SURYA_ENV[0]: OCR_SURYA_OVERLAY}


def runtime_env(plan: ExecutionPlan) -> tuple[str, tuple[str, ...]] | None:
    """(env name, packages) an in-process plan runs in; None for daemons."""
    return RUNTIME_ENVS.get((plan.task, plan.engine))


QUANT_TYPE = "Q4_K_M"


class CompilationError(RuntimeError):
    """Raised when an artifact could not be produced."""


class QuantizationMismatch(CompilationError):
    """Raised when a fetched artifact is not the quantization that was planned.

    Third-party GGUF repos publish assorted quant types. Accepting whatever is
    there would make the manifest claim a precision the weights do not have,
    which is worse than failing: the run would be silently unattributable.
    """

    def __init__(self, repo_id: str, wanted: str, found: list[str]):
        self.repo_id = repo_id
        self.wanted = wanted
        self.found = found
        available = ", ".join(sorted(found)) if found else "no .gguf files"
        super().__init__(
            f"{repo_id} does not publish a {wanted} build (found: {available}). "
            "Point the registry's gguf_repo at a build that does, rather than "
            "running weights the manifest would describe incorrectly."
        )


class UntrustedRemoteCode(CompilationError):
    """Raised when converting a model would execute unvetted repo code.

    Some repos (IndicTrans2 among them) ship custom modelling code that
    transformers must import to load the model. That is arbitrary Python
    running on the host, so it is allowed only for the curated roster.
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        super().__init__(
            f"{repo_id} ships custom modelling code that must be executed to "
            "convert it, and it is not part of the curated roster. Review the "
            f"repo's .py files at https://huggingface.co/{repo_id}/tree/main and "
            "add it to indic_runner/models/registry.py if you trust it."
        )


@dataclass(frozen=True)
class ArtifactPaths:
    """Where the inference-ready artifact and its tokenizer landed."""

    artifacts_dir: Path
    tokenizer_dir: Path
    evicted_source: bool = False
    # Which repo the artifact actually came from. A GGUF may be published by
    # a third-party quantizer rather than the model's own org, and results are
    # only attributable if the manifest says which.
    source_repo: str | None = None
    provenance: str = "official"  # "official" | "third-party" | "converted"


class Compiler(Protocol):
    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths: ...


def model_dir(alias: str, suffix: str) -> Path:
    return DIRS["models"] / f"{alias}-{suffix}"


def prune_orphan_sources(alias: str, keep: Path | None = None) -> list[Path]:
    """Delete leftover source downloads for an alias.

    A re-plan can change which compiler runs -- IndicTrans2 moved from the
    CTranslate2 lane to the transformers one -- and the abandoned lane's
    source tree is then never evicted by the compiler that created it. Left
    alone it is a silent multi-GB leak, so setup sweeps it on the way out.
    """
    removed = []
    for suffix in ("src",):
        candidate = model_dir(alias, suffix)
        if candidate == keep or not candidate.is_dir():
            continue
        shutil.rmtree(candidate, ignore_errors=True)
        removed.append(candidate)
    return removed


# Weight formats that duplicate safetensors. Many repos publish both, so a
# plain snapshot fetches the same tensors twice -- IndicTrans2-en-indic ships
# model.safetensors and pytorch_model.bin at 4.46GB each, 8.9GB for 4.5GB of
# weights. Skipped whenever safetensors are present.
REDUNDANT_WEIGHT_PATTERNS = ("*.bin", "*.h5", "*.msgpack", "*.ckpt")


def source_ignore_patterns(model: ModelProfile) -> list[str] | None:
    """Patterns to skip when snapshotting source weights.

    Returns None when the repo publishes no safetensors, so a pytorch-only
    repo (NLLB-200 among them) still downloads the weights it actually has.
    Skipping them would leave the converter with a config and no model.
    """
    if "safetensors" not in model.available_formats:
        return None
    return list(REDUNDANT_WEIGHT_PATTERNS)


def _snapshot(
    repo_id: str,
    destination: Path,
    allow_patterns=None,
    ignore_patterns=None,
) -> Path:
    """Download a repo snapshot, resuming an interrupted transfer.

    Gated repos need the same token the profiler used, or the download fails
    after the plan has already been committed to.
    """
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=hf_token(),
    )
    return Path(path)


def _verify_non_empty(path: Path, what: str) -> None:
    if not path.exists():
        raise CompilationError(f"{what} was not produced at {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise CompilationError(f"{what} at {path} is empty")
    if path.is_dir() and not any(path.iterdir()):
        raise CompilationError(f"{what} at {path} is empty")


def _evict(source: Path) -> bool:
    """Delete source weights once the converted artifact is verified."""
    if source.exists() and source.is_dir():
        shutil.rmtree(source, ignore_errors=True)
        return True
    return False


class GgufCompiler:
    """Builds a q4_k_m GGUF from the model's official safetensors.

    Published GGUFs are deliberately not used, even the ones an org ships for
    its own model. The roster exists for head-to-head comparison, and a build
    made by the model's authors may be imatrix-calibrated with settings we
    cannot see -- so sourcing some models that way and converting the rest
    would make part of every result a property of the quantizer. Converting
    everything through one pipeline leaves a constant offset that cancels in
    comparison, instead of a bias correlated with which orgs publish GGUFs
    (well-resourced labs do; the Indic-specific models mostly do not).

    ``USE_PUBLISHED_GGUF`` turns the prebuilt path back on. Flipping it
    reintroduces that confound, so it exists for deliberate experiments, not
    for saving setup time.
    """

    # See the class docstring before changing this.
    USE_PUBLISHED_GGUF = False

    def __init__(self, alias: str, bundle: BinaryBundle, accelerator: str = "cpu"):
        self.alias = alias
        self.bundle = bundle
        self.accelerator = accelerator

    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths:
        target = model_dir(self.alias, "gguf")
        prebuilt = (
            model.prebuilt_repos.get("gguf") if self.USE_PUBLISHED_GGUF else None
        )
        provenance = self._provenance(model.repo_id, prebuilt)

        if target.exists() and any(target.glob("*.gguf")):
            self._verify_quantization(target, plan.precision, prebuilt or model.repo_id)
            return ArtifactPaths(target, target, source_repo=prebuilt, provenance=provenance)

        if prebuilt:
            quant = plan.precision
            _snapshot(
                prebuilt,
                target,
                allow_patterns=[f"*{quant}*", f"*{quant.upper()}*", "*.json", "tokenizer*"],
            )
            # The artifact must be the quantization the plan committed to, or
            # the manifest would misreport what was actually run.
            self._verify_quantization(target, quant, prebuilt)
            _verify_non_empty(target, "prebuilt GGUF")
            return ArtifactPaths(target, target, source_repo=prebuilt, provenance=provenance)

        # No prebuilt artifact: convert locally, then drop the FP16 source.
        source = model_dir(self.alias, "src")
        _snapshot(model.repo_id, source, ignore_patterns=source_ignore_patterns(model))

        env_manager.ensure_env(*GGUF_ENV, accelerator=self.accelerator)
        target.mkdir(parents=True, exist_ok=True)
        f16_path = target / f"{self.alias}-f16.gguf"

        converter = converter_manager.ensure_converter(self.bundle.build)
        env_manager.run_in_env(
            GGUF_ENV[0],
            [
                str(converter.script),
                str(source),
                "--outfile", str(f16_path),
                "--outtype", "f16",
            ],
            extra_python_path=converter.python_path,
        )
        _verify_non_empty(f16_path, "intermediate f16 GGUF")

        quantized = target / f"{self.alias}-{QUANT_TYPE.lower()}.gguf"
        result = _run_binary(self.bundle, "llama-quantize", [str(f16_path), str(quantized), QUANT_TYPE])
        if result.returncode != 0:
            raise CompilationError(f"llama-quantize failed:\n{result.stderr.strip()}")
        _verify_non_empty(quantized, "quantized GGUF")

        f16_path.unlink(missing_ok=True)
        evicted = _evict(source)
        return ArtifactPaths(
            target, target, evicted_source=evicted,
            source_repo=model.repo_id, provenance="converted",
        )


    @staticmethod
    def _provenance(base_repo: str, prebuilt: str | None) -> str:
        if not prebuilt:
            return "converted"
        # Same org publishing its own GGUF counts as official.
        return "official" if prebuilt.split("/")[0] == base_repo.split("/")[0] else "third-party"

    @staticmethod
    def _verify_quantization(directory: Path, wanted: str, repo_id: str) -> None:
        found = sorted(p.name for p in directory.glob("*.gguf"))
        if not any(wanted.lower() in name.lower() for name in found):
            raise QuantizationMismatch(repo_id, wanted, found)


class Ct2Compiler:
    """Converts a seq2seq model to CTranslate2, quantized per the plan."""

    def __init__(self, alias: str, accelerator: str = "cpu", trusted: bool = False):
        self.alias = alias
        self.accelerator = accelerator
        self.trusted = trusted

    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths:
        target = model_dir(self.alias, "ct2")
        if (target / "model.bin").exists():
            return ArtifactPaths(target, target)

        # Decided before the download: refusing after fetching gigabytes that
        # can never be converted wastes the user's time and disk.
        if model.requires_remote_code and not self.trusted:
            raise UntrustedRemoteCode(model.repo_id)

        source = model_dir(self.alias, "src")
        _snapshot(model.repo_id, source, ignore_patterns=source_ignore_patterns(model))

        env_manager.ensure_env(*CT2_ENV, accelerator=self.accelerator)
        converter = env_manager.env_bin(CT2_ENV[0], "ct2-transformers-converter")
        import subprocess

        argv = [
            str(converter),
            "--model", str(source),
            "--output_dir", str(target),
            "--quantization", plan.precision,
            "--force",
        ]
        if model.requires_remote_code:
            argv.append("--trust_remote_code")

        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CompilationError(f"ct2-transformers-converter failed:\n{result.stderr.strip()}")
        _verify_non_empty(target / "model.bin", "CTranslate2 model")

        # The tokenizer is not part of the CT2 artifact; keep it beside it.
        tokenizer_dir = model_dir(self.alias, "tokenizer")
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("tokenizer*", "*.model", "special_tokens_map.json"):
            for item in source.glob(pattern):
                shutil.copy2(item, tokenizer_dir / item.name)

        evicted = _evict(source)
        return ArtifactPaths(target, tokenizer_dir, evicted_source=evicted)


class SafetensorsCompiler:
    """No conversion: snapshot the weights as published (fp16/fp32/awq)."""

    def __init__(self, alias: str, suffix: str = "hf"):
        self.alias = alias
        self.suffix = suffix

    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths:
        target = model_dir(self.alias, self.suffix)
        if target.exists() and any(target.iterdir()):
            return ArtifactPaths(target, target)
        _snapshot(model.repo_id, target, ignore_patterns=source_ignore_patterns(model))
        _verify_non_empty(target, "model snapshot")
        return ArtifactPaths(target, target)


class PaddleCompiler:
    """Snapshots PaddleOCR inference weights.

    Paddle ships its own inference format (``inference.pdiparams`` plus a
    json/yml pair) rather than safetensors, so there is nothing to convert --
    but the artifact must still be verified, or a repo reshuffle would leave
    an empty directory that only fails at run time.
    """

    WEIGHTS_SUFFIX = ".pdiparams"

    def __init__(self, alias: str):
        self.alias = alias

    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths:
        target = model_dir(self.alias, "paddle")
        if any(target.glob(f"*{self.WEIGHTS_SUFFIX}")) if target.exists() else False:
            return ArtifactPaths(target, target, source_repo=model.repo_id)

        _snapshot(model.repo_id, target)
        if not any(target.glob(f"*{self.WEIGHTS_SUFFIX}")):
            raise CompilationError(
                f"{model.repo_id} published no {self.WEIGHTS_SUFFIX} weights; "
                "PaddleOCR cannot load this repo"
            )
        return ArtifactPaths(target, target, source_repo=model.repo_id)


class PackageCompiler:
    """For runtimes that fetch their own weights (EasyOCR).

    There is no Hub artifact to compile, so setup only provisions the runtime
    environment; the package downloads its weights on first use.
    """

    def __init__(self, alias: str, accelerator: str = "cpu"):
        self.alias = alias
        self.accelerator = accelerator

    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths:
        env_manager.ensure_env(*OCR_ENV, accelerator=self.accelerator)
        target = env_manager.env_path(OCR_ENV[0])
        return ArtifactPaths(target, target, source_repo=None, provenance="converted")


class AwqCompiler:
    """Uses a pre-quantised AWQ repo; local quantisation is not attempted."""

    def __init__(self, alias: str):
        self.alias = alias

    def ensure(self, plan: ExecutionPlan, model: ModelProfile) -> ArtifactPaths:
        prebuilt = model.prebuilt_repos.get("awq")
        if not prebuilt:
            raise CompilationError(
                f"{model.repo_id} has no published AWQ build, and local AWQ "
                "quantisation needs a calibration corpus plus a CUDA device. "
                "Use a host with more VRAM, or point setup at a pre-quantised repo."
            )
        target = model_dir(self.alias, "awq")
        if target.exists() and any(target.iterdir()):
            return ArtifactPaths(target, target)
        _snapshot(prebuilt, target)
        _verify_non_empty(target, "AWQ snapshot")
        return ArtifactPaths(target, target)


def loader_path_var() -> str | None:
    """Environment variable the platform uses for the shared-library path.

    Windows resolves DLLs from the executable's own directory, so it needs no
    override; the llama.cpp bundle is not self-contained on Linux or macOS.
    """
    system = platform.system().lower()
    if system == "darwin":
        return "DYLD_LIBRARY_PATH"
    if system == "windows":
        return None
    return "LD_LIBRARY_PATH"


def _run_binary(bundle: BinaryBundle, name: str, args: list[str]):
    """Invoke a llama.cpp executable with its own directory on the loader path."""
    import os
    import subprocess

    env = os.environ.copy()
    key = loader_path_var()
    if key:
        env[key] = os.pathsep.join(filter(None, [str(bundle.library_dir), env.get(key, "")]))
    return subprocess.run(
        [str(bundle.executable(name)), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


# Every artifact format the decision matrix can emit must appear here, or a
# plan will dispatch to nothing and fail only after the weights are fetched.
SUPPORTED_ARTIFACT_FORMATS = frozenset(
    {"gguf", "ct2", "safetensors", "awq", "paddle", "package"}
)


def validate_artifact_format(plan: ExecutionPlan) -> None:
    """Check a compiler exists, without building one.

    Lets `--dry-run` prove the plan is buildable even though it has not
    downloaded the llama.cpp bundle a GgufCompiler would need.
    """
    if plan.artifact_format not in SUPPORTED_ARTIFACT_FORMATS:
        raise CompilationError(
            f"no compiler for artifact format {plan.artifact_format!r}; "
            f"supported: {sorted(SUPPORTED_ARTIFACT_FORMATS)}"
        )


def compiler_for(
    plan: ExecutionPlan,
    alias: str,
    bundle: BinaryBundle | None = None,
    accelerator: str = "cpu",
    trusted: bool = False,
) -> Compiler:
    """Pick the compiler that produces this plan's artifact format."""
    validate_artifact_format(plan)
    if plan.artifact_format == "gguf":
        if bundle is None:
            raise CompilationError("a llama.cpp bundle is required to build GGUF artifacts")
        return GgufCompiler(alias, bundle, accelerator)
    if plan.artifact_format == "ct2":
        return Ct2Compiler(alias, accelerator, trusted=trusted)
    if plan.artifact_format == "awq":
        return AwqCompiler(alias)
    if plan.artifact_format == "paddle":
        return PaddleCompiler(alias)
    if plan.artifact_format == "package":
        return PackageCompiler(alias, accelerator)
    if plan.artifact_format == "safetensors":
        return SafetensorsCompiler(alias, "fp32" if plan.precision == "fp32" else "hf")
    raise CompilationError(f"no compiler for artifact format {plan.artifact_format!r}")
