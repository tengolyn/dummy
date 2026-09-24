"""Shared base for daemons speaking the OpenAI-compatible HTTP API."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from indic_runner.runtime.engines.base import (
    Engine, EngineError, GenerationConfig, Request, Response,
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _text(choice: dict) -> str:
    """Completion text. Servers that split out a thinking trace (llama.cpp's
    reasoning_content) have it re-wrapped in <think> so the adapter's
    split_reasoning sees the same shape whichever server produced it."""
    if "text" in choice:
        return choice["text"] or ""
    msg = choice["message"]
    content = msg.get("content") or ""
    thought = msg.get("reasoning_content") or msg.get("reasoning")
    return f"<think>{thought}</think>{content}" if thought else content


class OpenAIDaemon(Engine):
    startup_timeout_s = 600.0

    def __init__(self, workers: int = 4):
        self.port = free_port()
        self.workers = workers
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._pool: ThreadPoolExecutor | None = None

    # -- subclass hooks -------------------------------------------------
    def command(self) -> list[str]: raise NotImplementedError
    def environment(self) -> dict[str, str]: return {}
    model_name = "model"

    # -------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._log = tempfile.NamedTemporaryFile(prefix="indic-runner-engine-", suffix=".log", delete=False)
        env = {**os.environ, **self.environment()}
        try:
            self._proc = subprocess.Popen(
                self.command(), stdout=self._log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
        except OSError as exc:
            raise EngineError(f"could not launch engine: {exc}") from exc
        self._pool = ThreadPoolExecutor(max_workers=self.workers)
        self._wait_healthy()

    def _tail(self) -> str:
        try:
            with open(self._log.name, errors="replace") as fh:
                return "".join(fh.readlines()[-15:])
        except OSError:
            return ""

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise EngineError(
                    f"engine exited during startup (code {self._proc.returncode}):\n{self._tail()}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        raise EngineError(f"engine not healthy after {self.startup_timeout_s:.0f}s:\n{self._tail()}")

    def _one(self, req: Request, gen: GenerationConfig) -> Response:
        if "prompt" in req.payload:  # base / hand-templated models
            path = "/v1/completions"
            body = {"model": self.model_name, "prompt": req.payload["prompt"]}
            if req.payload.get("stop"):
                body["stop"] = req.payload["stop"]
        else:
            path = "/v1/chat/completions"
            body = {"model": self.model_name, "messages": req.payload["messages"]}
        body.update(max_tokens=gen.max_new_tokens, temperature=gen.temperature, stream=False)
        http = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(http, timeout=gen.timeout_s) as r:
                data = json.load(r)
            return Response(req.id, _text(data["choices"][0]), latency_s=time.monotonic() - t0)
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as exc:
            if self._proc is not None and self._proc.poll() is not None:
                raise EngineError(f"engine died mid-run:\n{self._tail()}") from exc
            return Response(req.id, error=f"{type(exc).__name__}: {exc}")

    def infer(self, batch: list[Request], gen: GenerationConfig) -> list[Response]:
        return list(self._pool.map(lambda r: self._one(r, gen), batch))

    def stop(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
        proc = self._proc
        if proc and proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if self._log:
            self._log.close()
            try:
                os.unlink(self._log.name)
            except OSError:
                pass
