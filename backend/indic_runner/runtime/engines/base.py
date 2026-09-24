"""Engine interface shared by daemons (HTTP) and in-process workers (stdio)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EngineError(RuntimeError):
    """Engine failed to start, died, or is unavailable."""


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    num_beams: int = 1
    timeout_s: float = 120.0

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass(frozen=True)
class Request:
    """One unit of work. `payload` is adapter-built: {"messages": [...]},
    {"text": ...} or {"image_path": ..., "language": ...}."""

    id: str
    payload: dict


@dataclass(frozen=True)
class Response:
    id: str
    text: str = ""
    error: str | None = None
    latency_s: float | None = None


class Engine(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def infer(self, batch: list[Request], gen: GenerationConfig) -> list[Response]:
        """Return one Response per Request, in order. Per-row failures go in
        Response.error; EngineError means the engine itself is gone."""

    @abstractmethod
    def stop(self) -> None: ...
