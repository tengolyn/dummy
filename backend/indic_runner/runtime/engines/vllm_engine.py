"""vLLM OpenAI-compatible server, launched from its isolated env."""

from __future__ import annotations

from indic_runner.runtime.engines.base import EngineError
from indic_runner.runtime.engines.openai_daemon import OpenAIDaemon
from indic_runner.runtime.manifest_loader import Manifest
from indic_runner.setup.env_manager import python_path


class VllmEngine(OpenAIDaemon):
    def __init__(self, manifest: Manifest):
        super().__init__(workers=manifest.max_batch_size)
        self.m = manifest
        self.model_name = str(manifest.artifacts_dir)

    def command(self) -> list[str]:
        py = python_path(self.m.binary_or_env)
        if not py.exists():
            raise EngineError(f"env {self.m.binary_or_env!r} is not provisioned; re-run setup")
        cmd = [
            str(py), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.m.artifacts_dir), "--tokenizer", str(self.m.tokenizer_dir),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--max-model-len", str(self.m.context_length),
            "--max-num-seqs", str(self.m.max_batch_size),
        ]
        if self.m.precision == "awq":
            cmd += ["--quantization", "awq"]
        return cmd
