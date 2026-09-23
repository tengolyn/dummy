"""Queries the HF API for model architecture, parameter count, and config.

The Hugging Face client is injected rather than constructed here so the whole
module is testable from a fake without network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from indic_runner.config import hf_token

# Architectures that are sequence-to-sequence rather than decoder-only. Drives
# the translation branch of the decision matrix (CTranslate2 vs llama.cpp).
ENCODER_DECODER_ARCHITECTURES = frozenset(
    {
        "M2M100ForConditionalGeneration",
        "MarianMTModel",
        "T5ForConditionalGeneration",
        "MT5ForConditionalGeneration",
        "BartForConditionalGeneration",
        "IndicTransForConditionalGeneration",
    }
)

_PARAM_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?\b")


class GatedRepository(RuntimeError):
    """Raised when a repo needs authentication or a licence acceptance.

    Swallowing this would be worse than failing: without the config the
    profiler cannot see the architecture, so the decision matrix would pick a
    plausible-looking but wrong engine and only fail much later, at download.
    """

    def __init__(self, repo_id: str, has_token: bool = False, gating: str | None = None):
        self.repo_id = repo_id
        self.has_token = has_token
        # "auto": accepting the terms grants access immediately.
        # "manual": the repo owner reviews each request, so approval takes time.
        self.gating = gating
        page = f"https://huggingface.co/{repo_id}"

        if not has_token:
            detail = (
                "No Hugging Face token is configured. Put HF_TOKEN in "
                f"backend/.env (see .env.example), then accept the terms at {page}."
            )
        elif gating == "manual":
            detail = (
                f"Access is granted manually by the repo owner. Request it at {page}; "
                "approval is not immediate, so retry once it is granted."
            )
        elif gating == "auto":
            detail = (
                f"Accepting the terms at {page} grants access immediately. "
                "Use the same account the token belongs to."
            )
        else:
            detail = (
                f"A token is configured, so it most likely lacks access: open {page} "
                "and accept the terms with the same account, then retry."
            )
        super().__init__(f"{repo_id} is gated. {detail}")


class HubClient(Protocol):
    """The slice of ``huggingface_hub.HfApi`` this module depends on."""

    def model_info(self, repo_id: str, **kwargs: Any) -> Any: ...

    def repo_exists(self, repo_id: str, **kwargs: Any) -> bool: ...

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class ModelProfile:
    """What `setup` knows about a model before any weights are fetched."""

    repo_id: str
    revision: str
    architecture: str
    param_count_b: float
    context_length: int
    is_encoder_decoder: bool
    available_formats: frozenset[str] = field(default_factory=frozenset)
    tokenizer_files: tuple[str, ...] = ()
    prebuilt_repos: dict[str, str] = field(default_factory=dict)
    # Set for OCR models, whose runtimes differ per model (see registry).
    ocr_runtime: str | None = None
    # True when the repo ships custom modelling code that transformers must
    # execute to load the model (config carries an ``auto_map``).
    requires_remote_code: bool = False


def classify_formats(filenames: list[str]) -> frozenset[str]:
    """Infer which artifact formats a repo already publishes."""
    formats: set[str] = set()
    for name in filenames:
        lower = name.lower()
        if lower.endswith(".gguf"):
            formats.add("gguf")
        elif lower.endswith(".safetensors"):
            formats.add("safetensors")
        elif lower.endswith(".bin") and "pytorch_model" in lower:
            formats.add("pytorch")
        elif lower == "model.bin" or lower.endswith("/model.bin"):
            # CTranslate2 emits model.bin alongside a config.json.
            formats.add("ct2")
    return frozenset(formats)


def estimate_params_b(config: dict, safetensors_total: int | None = None) -> float | None:
    """Best-effort parameter count in billions.

    Prefers the exact total reported by safetensors metadata, then a structural
    estimate from the config, and finally the size hint in the repo name.
    """
    if safetensors_total:
        return round(safetensors_total / 1e9, 2)

    hidden = config.get("hidden_size") or config.get("d_model")
    layers = config.get("num_hidden_layers") or config.get("num_layers")
    vocab = config.get("vocab_size")
    if hidden and layers and vocab:
        # 12 * L * H^2 approximates the transformer blocks; embeddings are
        # counted separately (tied input/output not assumed).
        blocks = 12 * int(layers) * int(hidden) ** 2
        embeddings = 2 * int(vocab) * int(hidden)
        return round((blocks + embeddings) / 1e9, 2)
    return None


def params_from_name(repo_id: str) -> float | None:
    """Pull a size hint such as ``-7B`` out of a repo name."""
    match = _PARAM_SUFFIX.search(repo_id)
    return float(match.group(1)) if match else None


def resolve_context_length(config: dict) -> int:
    for key in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 4096


def _is_gated(exc: BaseException) -> bool:
    """True when an exception means "authenticate", not "unavailable"."""
    if type(exc).__name__ in ("GatedRepoError", "LocalTokenNotFoundError"):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def _siblings(info: Any) -> list[str]:
    raw = getattr(info, "siblings", None) or []
    names = []
    for sibling in raw:
        name = getattr(sibling, "rfilename", None)
        if name:
            names.append(name)
    return names


def _safetensors_total(info: Any) -> int | None:
    meta = getattr(info, "safetensors", None)
    if meta is None:
        return None
    total = getattr(meta, "total", None)
    if total is None and isinstance(meta, dict):
        total = meta.get("total")
    return total if isinstance(total, int) else None


def profile_model(
    repo_id: str,
    client: HubClient,
    registry_params_b: float | None = None,
    ocr_runtime: str | None = None,
    gguf_repo: str | None = None,
) -> ModelProfile:
    """Build a ModelProfile from the Hub, without downloading weights."""
    info = client.model_info(repo_id, files_metadata=False)
    filenames = _siblings(info)

    config: dict = {}
    if "config.json" in filenames:
        try:
            path = client.hf_hub_download(repo_id, "config.json")
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception as exc:
            if _is_gated(exc):
                raise GatedRepository(
                    repo_id,
                    has_token=hf_token() is not None,
                    gating=getattr(info, "gated", None) or None,
                ) from exc
            # Anything else (a transient error, malformed JSON) leaves the
            # config advisory: profiling falls back to metadata and the name.
            config = {}

    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else (
        getattr(info, "pipeline_tag", None) or "unknown"
    )

    params = (
        estimate_params_b(config, _safetensors_total(info))
        or params_from_name(repo_id)
        or registry_params_b
        or 0.0
    )

    is_enc_dec = architecture in ENCODER_DECODER_ARCHITECTURES or bool(
        config.get("is_encoder_decoder")
    )

    prebuilt = {}
    if gguf_repo:
        # Curated in the registry; trusted over anything discovered here.
        prebuilt["gguf"] = gguf_repo
    for suffix, fmt in (("-GGUF", "gguf"), ("-AWQ", "awq")):
        if fmt in prebuilt:
            continue
        candidate = f"{repo_id}{suffix}"
        try:
            if client.repo_exists(candidate):
                prebuilt[fmt] = candidate
        except Exception:  # noqa: BLE001,S112 - absence is the common case, not an error
            continue

    return ModelProfile(
        repo_id=repo_id,
        revision=getattr(info, "sha", None) or "main",
        architecture=architecture,
        param_count_b=params,
        context_length=resolve_context_length(config),
        is_encoder_decoder=is_enc_dec,
        available_formats=classify_formats(filenames),
        tokenizer_files=tuple(
            n for n in filenames if "tokenizer" in n.lower() or n.endswith(".model")
        ),
        prebuilt_repos=prebuilt,
        ocr_runtime=ocr_runtime,
        requires_remote_code=bool(config.get("auto_map")),
    )
