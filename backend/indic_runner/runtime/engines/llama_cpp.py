"""llama-server daemon."""

from __future__ import annotations

import sys
from pathlib import Path

from indic_runner.runtime.engines.base import EngineError
from indic_runner.runtime.engines.openai_daemon import OpenAIDaemon
from indic_runner.runtime.manifest_loader import Manifest


def find_gguf(artifacts_dir: Path) -> Path:
    if artifacts_dir.is_file():
        return artifacts_dir
    found = sorted(artifacts_dir.glob("*.gguf"))
    if not found:
        raise EngineError(f"no .gguf file in {artifacts_dir}")
    return found[0]


# The KV cache is allocated up front for slots x per-slot context, and the
# manifest's batch size only accounts for weights. Uncapped, 16 x 8192 tokens
# asked for ~14GB on a 2.5B model. CPU decoding gains little past a few slots.
MAX_SLOTS = 8
MAX_SLOT_CONTEXT = 4096


class LlamaCppEngine(OpenAIDaemon):
    def __init__(self, manifest: Manifest):
        self.slots = min(manifest.max_batch_size, MAX_SLOTS)
        self.slot_context = min(manifest.context_length, MAX_SLOT_CONTEXT)
        super().__init__(workers=self.slots)
        self.m = manifest

    def command(self) -> list[str]:
        cmd = [
            self.m.binary_or_env, "-m", str(find_gguf(self.m.artifacts_dir)),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.slot_context * self.slots),
            "--parallel", str(self.slots),
        ]
        if self.m.threads:
            cmd += ["-t", str(self.m.threads)]
        return cmd

    def environment(self) -> dict[str, str]:
        if not self.m.library_dir:
            return {}
        import os
        var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        prior = os.environ.get(var, "")
        return {var: self.m.library_dir + (f":{prior}" if prior else "")}
