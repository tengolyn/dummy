"""Builds the engine a manifest names and guarantees teardown."""

from __future__ import annotations

from contextlib import contextmanager

from indic_runner.runtime.engines.base import Engine, EngineError
from indic_runner.runtime.manifest_loader import Manifest


def build_engine(m: Manifest) -> Engine:
    if m.engine == "llama.cpp":
        from indic_runner.runtime.engines.llama_cpp import LlamaCppEngine
        return LlamaCppEngine(m)
    if m.engine == "vllm":
        from indic_runner.runtime.engines.vllm_engine import VllmEngine
        return VllmEngine(m)
    from indic_runner.runtime.engines.worker_engine import make_worker_engine
    return make_worker_engine(m)  # raises EngineError for engines with no worker


@contextmanager
def running_engine(engine: Engine):
    """start() on entry; stop() always, including on error and Ctrl-C."""
    try:
        engine.start()
        yield engine
    finally:
        engine.stop()
