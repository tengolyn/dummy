"""Task adapters: dataset row -> engine Request, engine text -> (response, reasoning)."""

from __future__ import annotations

import re
from typing import Protocol

from indic_runner.runtime.engines.base import Request


class Adapter(Protocol):
    def build(self, row: dict) -> Request: ...
    def parse(self, text: str) -> tuple[str, str]: ...
    def image_sha256(self, row: dict) -> str | None: ...


_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate <think>...</think> blocks (Qwen3-style) from the answer."""
    thoughts = "\n".join(m.strip() for m in _THINK.findall(text))
    answer = _THINK.sub("", text)
    if "</think>" in answer:  # template prefilled the opening tag
        head, _, answer = answer.partition("</think>")
        thoughts = (thoughts + "\n" + head.strip()).strip()
    return answer.strip(), thoughts


def build_llm_request(task: str, alias: str, row: dict) -> Request:
    """Render a row into the prompt format the model was trained on."""
    from indic_runner.models.prompts import (
        render_completion, render_system, render_user, spec_for,
    )

    spec = spec_for(alias)
    if spec.mode == "completion":
        payload = {"prompt": render_completion(spec, task, row), "stop": list(spec.stop)}
        return Request(row["id"], payload)
    messages = []
    system = render_system(spec, task, row)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": render_user(spec, task, row)})
    return Request(row["id"], {"messages": messages})


def get_adapter(task: str, engine: str, model_alias: str = "") -> Adapter:
    from indic_runner.runtime.task_adapters import ocr, reasoning, summarization, translation

    if task == "reasoning":
        return reasoning.ReasoningAdapter(model_alias)
    if task == "summarization":
        return summarization.SummarizationAdapter(model_alias)
    if task == "translation":
        return translation.TranslationAdapter(engine, model_alias)
    if task == "ocr":
        return ocr.OcrAdapter(model_alias)
    raise ValueError(f"unknown task {task!r}")
