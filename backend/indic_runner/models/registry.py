"""Fixed roster of sub-10B open-weight models per task. See CLAUDE.md.

Four per task, except translation, which carries five: IndicTrans2 is published
as one model per direction, and both en-indic and indic-en are registered.

Every repo here is ungated or ``auto``-gated (one licence click): models
behind ``manual`` gating, where the owner reviews each request, were replaced so
setup is never blocked waiting on a third party. Licences are permissive
(Apache-2.0 / MIT) -- no non-commercial weights.

No third-party weights. A GGUF is used only when the model's own org publishes
``<repo>-GGUF``; otherwise setup converts from the official safetensors itself
(see converter_manager). Community quantizers vary in imatrix and base revision,
so mixing them in would mean partly measuring the quantizer rather than the
model. ``gguf_repo`` remains supported for a same-org build the sibling naming
does not cover, and is rejected if it points at another org.

OCR entries carry a ``runtime`` because the four OCR models are not one family:
only two are transformers models. PaddleOCR ships Paddle weights (its config
declares no ``architectures``) and EasyOCR does not publish to the Hub at all,
so each needs its own in-process runtime and isolated environment.
"""

MODEL_REGISTRY = {
    "reasoning": [
        {"alias": "qwen2.5-7b-instruct", "repo": "Qwen/Qwen2.5-7B-Instruct", "params_b": 7},
        {"alias": "qwen3-8b", "repo": "Qwen/Qwen3-8B", "params_b": 8.19},
                {"alias": "phi-4-mini-instruct", "repo": "microsoft/Phi-4-mini-instruct", "params_b": 3.84},
                {"alias": "sarvam-1", "repo": "sarvamai/sarvam-1", "params_b": 2},
    ],
    "summarization": [
        {"alias": "navarasa-2.0-7b", "repo": "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0", "params_b": 7},
                {"alias": "airavata-7b", "repo": "ai4bharat/Airavata", "params_b": 7},
        {"alias": "qwen2.5-7b-instruct", "repo": "Qwen/Qwen2.5-7B-Instruct", "params_b": 7},
                {"alias": "qwen3-4b-instruct-2507", "repo": "Qwen/Qwen3-4B-Instruct-2507", "params_b": 4.02},
    ],
    "translation": [
        # IndicTrans2 ships one model per direction and gates each separately,
        # so the two directions are separate entries rather than one slot.
        {"alias": "indictrans2-en-indic", "repo": "ai4bharat/indictrans2-en-indic-1B", "params_b": 1.1},
        {"alias": "indictrans2-indic-en", "repo": "ai4bharat/indictrans2-indic-en-1B", "params_b": 1.1},
        {"alias": "nllb-200", "repo": "facebook/nllb-200-3.3B", "params_b": 3.3},
        {"alias": "madlad-400", "repo": "google/madlad400-7b-mt", "params_b": 7.2},
                {"alias": "sarvam-translate", "repo": "sarvamai/sarvam-translate", "params_b": 4},
    ],
    "ocr": [
        {"alias": "indic-ocr-bodhan", "repo": "bodhan-ai/indic-ocr", "params_b": 0.8, "runtime": "transformers"},
        {"alias": "surya-ocr", "repo": "vikp/surya_rec2", "params_b": 0.65, "runtime": "surya"},
        {"alias": "paddleocr-pp-ocrv4", "repo": "PaddlePaddle/PP-OCRv4_server_rec", "params_b": 0.1, "runtime": "paddleocr"},
        {"alias": "chandra-ocr", "repo": "datalab-to/chandra-ocr-2", "params_b": 5.3, "runtime": "transformers"},
    ],
}
