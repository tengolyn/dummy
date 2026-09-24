"""Per-model prompt specs, keyed by registry alias.

Each model was trained on one prompt format; sending every model the same chat
message measures the format mismatch rather than the model. The spec is code,
not manifest data: it is fixed per model, so `run` stays offline and the
setup/run contract is unchanged.

``mode``:
  chat        -- messages go to /v1/chat/completions; the server applies the
                 template embedded in the model. ``system`` is optional.
  completion  -- the full prompt is rendered here and sent to /v1/completions.
                 Used where the model has no (or an unreliable) embedded chat
                 template: base models and Alpaca/Zephyr-style fine-tunes.
  seq2seq     -- language-code driven translators; ``codes`` names the scheme.

Templates use {language}, {target_language} and {input}.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptSpec:
    mode: str = "chat"
    system: dict[str, str] = field(default_factory=dict)  # task -> system prompt
    user: dict[str, str] = field(default_factory=dict)  # task -> user/instruction template
    completion: dict[str, str] = field(default_factory=dict)  # task -> full prompt template
    stop: tuple[str, ...] = ()
    codes: str = "flores"  # seq2seq only: flores | madlad | indictrans2
    src_prefix: str = ""  # seq2seq only: prepended to the source, e.g. "<2{tgt}> "


DEFAULT_USER = {
    "ocr": "Transcribe all {language} text in this image exactly as written. Output only the text.",
    "reasoning": "{input}",
    "summarization": "Summarize the following {language} text concisely, in {language}.\n\n{input}",
    "translation": (
        "Translate the following text from {language} to {target_language}. "
        "Output only the translation.\n\n{input}"
    ),
}

_ALPACA = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

PROMPTS: dict[str, PromptSpec] = {
    # Instruct models with a reliable embedded chat template.
    "qwen2.5-7b-instruct": PromptSpec(),
    "qwen3-8b": PromptSpec(),
    "phi-4-mini-instruct": PromptSpec(),
    "qwen3-4b-instruct-2507": PromptSpec(),
    # Sarvam-1 is a base model: no chat template, so it gets a plain
    # question/answer frame and a stop before it invents the next question.
    "sarvam-1": PromptSpec(
        mode="completion",
        completion={"reasoning": "Question: {input}\nAnswer:"},
        stop=("\nQuestion:",),
    ),
    # Navarasa 2.0 was tuned on the Alpaca instruction/input/response format.
    "navarasa-2.0-7b": PromptSpec(
        mode="completion",
        completion={
            "summarization": _ALPACA.replace(
                "{instruction}", "Summarize the following {language} text concisely, in {language}."
            ),
        },
        stop=("### Instruction", "<eos>"),
    ),
    # Airavata (Llama-2 based) uses <|user|>/<|assistant|> turns. Rendered here
    # so the result does not depend on the converted GGUF carrying a template.
    "airavata-7b": PromptSpec(
        mode="completion",
        completion={
            "summarization": (
                "<|user|>\nSummarize the following {language} text concisely, in {language}."
                "\n\n{input}\n<|assistant|>\n"
            ),
        },
        stop=("<|user|>", "</s>"),
    ),
    # Sarvam-Translate takes the target language in the system prompt and the
    # bare text as the user turn.
    "sarvam-translate": PromptSpec(
        system={"translation": "Translate the text below to {target_language}."},
        user={"translation": "{input}"},
    ),
    # Seq2seq translators.
    "nllb-200": PromptSpec(mode="seq2seq", codes="flores"),
    "madlad-400": PromptSpec(mode="seq2seq", codes="madlad", src_prefix="<2{tgt}> "),
    "indictrans2-en-indic": PromptSpec(mode="seq2seq", codes="indictrans2"),
    "indictrans2-indic-en": PromptSpec(mode="seq2seq", codes="indictrans2"),
    # OCR: prompts only matter for the VLM-style models.
    "chandra-ocr": PromptSpec(
        user={"ocr": "Transcribe all {language} text in this image exactly as written. "
                     "Output only the text."},
    ),
    "indic-ocr-bodhan": PromptSpec(
        user={"ocr": "Transcribe all {language} text in this image exactly as written. "
                     "Output only the text."},
    ),
    "surya-ocr": PromptSpec(mode="seq2seq"),
    "paddleocr-pp-ocrv4": PromptSpec(mode="seq2seq"),
}


def spec_for(alias: str) -> PromptSpec:
    """Spec for an alias; unknown aliases (ad-hoc repos) get chat defaults."""
    return PROMPTS.get(alias, PromptSpec())


def render_user(spec: PromptSpec, task: str, row: dict) -> str:
    template = spec.user.get(task) or DEFAULT_USER[task]
    return _fill(template, row)


def render_system(spec: PromptSpec, task: str, row: dict) -> str | None:
    template = spec.system.get(task)
    return _fill(template, row) if template else None


def render_completion(spec: PromptSpec, task: str, row: dict) -> str:
    template = spec.completion.get(task)
    if template is None:
        raise ValueError(f"no completion prompt for task {task!r}")
    return _fill(template, row)


def _fill(template: str, row: dict) -> str:
    # str.replace, not str.format: dataset text may contain braces.
    out = template
    for key in ("language", "target_language"):
        out = out.replace("{" + key + "}", str(row.get(key, "")))
    return out.replace("{input}", str(row.get("input", "")))
