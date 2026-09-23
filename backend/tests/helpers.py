"""Builders and fakes shared by the test suite.

Kept out of conftest so tests can import them directly.
"""

from __future__ import annotations

from indic_runner.setup.hardware_profiler import HardwareProfile
from indic_runner.setup.model_profiler import ModelProfile


def make_hw(
    accelerator: str = "cpu",
    os_name: str = "linux",
    arch: str = "x86_64",
    total_ram_gb: float = 32.0,
    vram_gb: float | None = None,
    gpu_name: str | None = None,
    cuda_version: str | None = None,
    cpu_threads: int = 8,
) -> HardwareProfile:
    return HardwareProfile(
        os=os_name,
        arch=arch,
        accelerator=accelerator,
        total_ram_gb=total_ram_gb,
        usable_ram_gb=round(total_ram_gb * 0.75, 2),
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        cpu_threads=cpu_threads,
    )


def make_model(
    params_b: float = 7.0,
    is_encoder_decoder: bool = False,
    context_length: int = 8192,
    repo_id: str = "org/model",
    prebuilt: dict | None = None,
) -> ModelProfile:
    return ModelProfile(
        repo_id=repo_id,
        revision="deadbeef",
        architecture="TestArch",
        param_count_b=params_b,
        context_length=context_length,
        is_encoder_decoder=is_encoder_decoder,
        prebuilt_repos=prebuilt or {},
    )


# The four lanes the decision matrix distinguishes.
LANES = {
    "cpu": make_hw("cpu"),
    "mps": make_hw("mps", os_name="darwin", arch="arm64"),
    "cuda_ample": make_hw("cuda", vram_gb=80.0, gpu_name="A100", cuda_version="12.4"),
    "cuda_constrained": make_hw("cuda", vram_gb=8.0, gpu_name="RTX3070", cuda_version="12.4"),
}


class FakeSibling:
    def __init__(self, name: str):
        self.rfilename = name


class FakeModelInfo:
    def __init__(self, files, sha="abc123", safetensors_total=None, pipeline_tag=None):
        self.siblings = [FakeSibling(f) for f in files]
        self.sha = sha
        self.safetensors = {"total": safetensors_total} if safetensors_total else None
        self.pipeline_tag = pipeline_tag


class FakeHub:
    """Stands in for huggingface_hub.HfApi with no network access."""

    def __init__(self, files=None, config=None, sha="abc123",
                 safetensors_total=None, existing_repos=()):
        self.files = files or ["config.json", "model.safetensors", "tokenizer.json"]
        self.config = config or {}
        self.sha = sha
        self.safetensors_total = safetensors_total
        self.existing_repos = set(existing_repos)
        self.downloads: list[str] = []

    def model_info(self, repo_id, **kwargs):
        return FakeModelInfo(self.files, self.sha, self.safetensors_total)

    def repo_exists(self, repo_id, **kwargs):
        return repo_id in self.existing_repos

    def hf_hub_download(self, repo_id, filename, **kwargs):
        import json
        import tempfile

        self.downloads.append(filename)
        # delete=False so the path outlives this call, like a real download.
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(self.config, handle)
        handle.close()
        return handle.name
