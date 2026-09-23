"""JSON Schema (draft-07) for the setup-phase execution contract.

This is the manifest `setup` emits and `run` consumes. It is distinct from
``schemas/result_schema.py``, which describes the output of a completed run.
"""

MANIFEST_VERSION = "1.0"

MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "manifest_version": {"type": "string"},
        "model_alias": {"type": "string"},
        "base_repo": {"type": "string"},
        "revision": {"type": "string"},
        "task": {
            "type": "string",
            "enum": ["reasoning", "summarization", "translation", "ocr"],
        },
        "target_hardware": {
            "type": "string",
            "enum": ["cuda", "mps", "cpu"],
        },
        "precision": {
            "type": "string",
            "enum": ["fp32", "fp16", "bf16", "int8", "awq", "q4_k_m"],
        },
        "engine": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    # The OCR runtimes are separate engines, not transformers
                    # variants: PaddleOCR and EasyOCR ship their own stacks.
                    "enum": [
                        "llama.cpp",
                        "vllm",
                        "ctranslate2",
                        "transformers",
                        "surya",
                        "paddleocr",
                        "easyocr",
                    ],
                },
                "mode": {"type": "string", "enum": ["daemon", "in-process"]},
                "binary_or_env": {"type": "string"},
                "library_dir": {"type": ["string", "null"]},
            },
            "required": ["name", "mode", "binary_or_env"],
        },
        "artifact_source": {
            "type": "object",
            "description": (
                "Where the inference-ready weights actually came from. A GGUF "
                "may be published by a third-party quantizer rather than the "
                "model's own org; results are only attributable if this says so."
            ),
            "properties": {
                "repo": {"type": ["string", "null"]},
                "provenance": {
                    "type": "string",
                    "enum": ["official", "third-party", "converted"],
                },
            },
            "required": ["repo", "provenance"],
        },
        "paths": {
            "type": "object",
            "properties": {
                "artifacts_dir": {"type": "string"},
                "tokenizer_dir": {"type": "string"},
            },
            "required": ["artifacts_dir", "tokenizer_dir"],
        },
        "runtime_parameters": {
            "type": "object",
            "properties": {
                "max_batch_size": {"type": "integer"},
                "context_length": {"type": "integer"},
                "threads": {"type": "integer"},
            },
            "required": ["max_batch_size", "context_length"],
        },
    },
    "required": [
        "manifest_version",
        "model_alias",
        "base_repo",
        "task",
        "target_hardware",
        "precision",
        "engine",
        "artifact_source",
        "paths",
        "runtime_parameters",
    ],
}
