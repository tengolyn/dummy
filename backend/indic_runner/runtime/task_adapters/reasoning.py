from indic_runner.runtime.engines.base import Request
from indic_runner.runtime.task_adapters import build_llm_request, split_reasoning


class ReasoningAdapter:
    def __init__(self, model_alias: str = ""):
        self.alias = model_alias

    def build(self, row: dict) -> Request:
        return build_llm_request("reasoning", self.alias, row)

    def parse(self, text: str) -> tuple[str, str]:
        return split_reasoning(text)

    def image_sha256(self, row):
        return None
