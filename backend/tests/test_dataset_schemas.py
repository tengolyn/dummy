"""Per-task input dataset columns."""

from __future__ import annotations

import pytest

from schemas.dataset_schemas import DATASET_SCHEMAS


def test_all_tasks_present():
    assert set(DATASET_SCHEMAS) == {"translation", "ocr", "reasoning", "summarization"}


@pytest.mark.parametrize("task,columns", sorted(DATASET_SCHEMAS.items()))
def test_every_task_carries_id_language_and_reference(task, columns):
    assert {"id", "language", "reference"} <= set(columns)


def test_translation_declares_a_target_language():
    assert "target_language" in DATASET_SCHEMAS["translation"]


def test_ocr_reads_images_not_text():
    assert "image_path" in DATASET_SCHEMAS["ocr"]
    assert "input" not in DATASET_SCHEMAS["ocr"]


def test_reasoning_carries_a_judge_rubric():
    # Carried through for downstream evaluation; the runner never scores it.
    assert "judge_rubric" in DATASET_SCHEMAS["reasoning"]


@pytest.mark.parametrize("task", ["translation", "reasoning", "summarization"])
def test_text_tasks_take_an_input_column(task):
    assert "input" in DATASET_SCHEMAS[task]


def test_no_task_declares_duplicate_columns():
    for task, columns in DATASET_SCHEMAS.items():
        assert len(columns) == len(set(columns)), task
