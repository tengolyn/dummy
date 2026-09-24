"""Chunked dataset reader and batch dispatcher.

Rows are validated against schemas/dataset_schemas.py as they stream; nothing
holds the dataset in memory. Only JSONL is supported.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from schemas.dataset_schemas import DATASET_SCHEMAS


class DatasetError(ValueError):
    """The dataset is unreadable or a row breaks the task's schema."""


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    row_count: int
    languages: list[str]


def _check_path(path: Path) -> None:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise DatasetError(f"unsupported dataset format {path.suffix!r}; expected .jsonl")


def iter_rows(path: Path, task: str) -> Iterator[dict]:
    """Yield validated rows one at a time; fail on the first bad row."""
    path = Path(path)
    _check_path(path)
    if task not in DATASET_SCHEMAS:
        raise DatasetError(f"unknown task {task!r}")
    required = DATASET_SCHEMAS[task]
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path.name}:{lineno}: row is not a JSON object")
            missing = [c for c in required if c not in row]
            if missing:
                raise DatasetError(
                    f"{path.name}:{lineno}: missing {task} columns {missing}"
                )
            row["id"] = str(row["id"])
            if row["id"] in seen:
                raise DatasetError(f"{path.name}:{lineno}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            yield row


def scan_dataset(path: Path, task: str) -> DatasetInfo:
    """One validating pre-pass for row count and languages (O(languages) memory
    plus ids, which iter_rows needs to reject duplicates)."""
    langs: Counter = Counter()
    count = 0
    for row in iter_rows(path, task):
        count += 1
        langs[str(row["language"])] += 1
    return DatasetInfo(Path(path).name, count, sorted(langs))


def batched(rows: Iterable[dict], size: int) -> Iterator[list[dict]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
