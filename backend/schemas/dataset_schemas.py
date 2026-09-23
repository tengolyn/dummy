"""Required input-dataset columns per task. See CLAUDE.md "Input dataset schemas"."""

TRANSLATION_COLUMNS = ["id", "language", "target_language", "reference", "input"]

OCR_COLUMNS = ["id", "image_path", "language", "reference"]

REASONING_COLUMNS = ["id", "input", "language", "reference", "judge_rubric"]

SUMMARIZATION_COLUMNS = ["id", "language", "input", "reference"]

DATASET_SCHEMAS = {
    "translation": TRANSLATION_COLUMNS,
    "ocr": OCR_COLUMNS,
    "reasoning": REASONING_COLUMNS,
    "summarization": SUMMARIZATION_COLUMNS,
}
