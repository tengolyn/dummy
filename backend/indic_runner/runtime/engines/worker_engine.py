"""In-process engines, run as one long-lived worker in the engine's isolated env.

The orchestrator cannot import torch/CTranslate2 (dependency stacks stay
separate), so "in-process" means: load the model once in a worker process using
the env's interpreter, and exchange JSON lines over stdio.
Protocol: worker prints {"ready": true}; per request line
{"id", ...payload, max_new_tokens, num_beams} it prints {"id", "text"|"error"}.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from indic_runner.runtime.engines.base import (
    Engine, EngineError, GenerationConfig, Request, Response,
)
from indic_runner.runtime.manifest_loader import Manifest
from indic_runner.setup.env_manager import python_path

WORKERS_DIR = Path(__file__).resolve().parent.parent / "workers"


class WorkerEngine(Engine):
    def __init__(self, argv: list[str], startup_timeout_s: float = 900.0):
        self.argv = argv
        self.startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, bufsize=1,
            )
        except OSError as exc:
            raise EngineError(f"could not launch worker: {exc}") from exc
        ready: list = []
        t = threading.Thread(target=lambda: ready.append(self._proc.stdout.readline()), daemon=True)
        t.start()
        t.join(self.startup_timeout_s)
        line = ready[0] if ready else ""
        try:
            ok = json.loads(line).get("ready") is True
        except ValueError:
            ok = False
        if not ok:
            self.stop()
            raise EngineError(f"worker failed to start (first output: {line.strip()!r})")

    def infer(self, batch: list[Request], gen: GenerationConfig) -> list[Response]:
        p = self._proc
        try:
            for r in batch:
                msg = {"id": r.id, **r.payload, "max_new_tokens": gen.max_new_tokens, "num_beams": gen.num_beams}
                p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            p.stdin.flush()
            out = []
            for r in batch:
                line = p.stdout.readline()
                if not line:
                    raise EngineError("worker died mid-run")
                d = json.loads(line)
                out.append(Response(d["id"], d.get("text", ""), d.get("error")))
            return out
        except (BrokenPipeError, ValueError, KeyError) as exc:
            raise EngineError(f"worker protocol failure: {exc}") from exc

    def stop(self) -> None:
        p = self._proc
        if not p:
            return
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


WORKER_SCRIPTS = {"ctranslate2": "ctranslate2_worker.py", "transformers": "transformers_worker.py"}
# OCR engines are named for their runtime; one worker serves all of them.
OCR_RUNTIMES = ("transformers", "surya", "paddleocr")


def make_worker_engine(m: Manifest) -> WorkerEngine:
    if m.task == "ocr" and m.engine in OCR_RUNTIMES:
        script, extra = "ocr_worker.py", ["--runtime", m.engine, "--alias", m.alias]
    else:
        script, extra = WORKER_SCRIPTS.get(m.engine), []
    if script is None:
        raise EngineError(
            f"engine {m.engine!r} has no run-phase worker yet (supported: "
            f"{', '.join(sorted({*WORKER_SCRIPTS, *OCR_RUNTIMES}))})"
        )
    py = python_path(m.binary_or_env)
    if not py.exists():
        raise EngineError(f"env {m.binary_or_env!r} is not provisioned; re-run setup")
    return WorkerEngine([
        str(py), str(WORKERS_DIR / script),
        "--artifacts", str(m.artifacts_dir), "--tokenizer", str(m.tokenizer_dir),
        "--device", m.hardware, "--precision", m.precision, *extra,
    ])
