"""Appends per-row results to disk as they stream, then assembles the final
{manifest, results} JSON by copying lines -- never holding results in memory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from schemas.result_schema import RESULT_SCHEMA
from indic_runner.setup.manifest_writer import _check

ROW_FIELDS = RESULT_SCHEMA["properties"]["results"]["items"]


class ResultValidationError(ValueError):
    pass


def validate(schema: dict, value, label: str) -> None:
    errors: list[str] = []
    _check(value, schema, label, errors)
    if errors:
        raise ResultValidationError("; ".join(errors))


class OutputSink:
    def __init__(self, run_dir: Path, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.partial = self.run_dir / "results.partial.jsonl"
        self.final = self.run_dir / "results.json"
        if not resume and self.partial.exists():
            self.partial.unlink()
        self._fh = None

    def done_ids(self) -> set[str]:
        """Ids already written (for resume). Drops a torn trailing line."""
        ids: set[str] = set()
        if not self.partial.exists():
            return ids
        good = 0
        with self.partial.open("rb") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["id"])
                    good += len(line)
                except (ValueError, KeyError):
                    break
        with self.partial.open("r+b") as fh:
            fh.truncate(good)
        return ids

    def append(self, row: dict) -> None:
        validate(ROW_FIELDS, row, f"result[{row.get('id')}]")
        if self._fh is None:
            self._fh = self.partial.open("a", encoding="utf-8")
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def assemble(self, manifest: dict) -> Path:
        """Write results.json atomically: manifest, then rows copied line by line."""
        self.close()
        validate(RESULT_SCHEMA["properties"]["manifest"], manifest, "manifest")
        tmp = self.final.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as out:
            out.write('{"manifest": ' + json.dumps(manifest, ensure_ascii=False, indent=2))
            out.write(', "results": [')
            first = True
            if self.partial.exists():
                with self.partial.open(encoding="utf-8") as src:
                    for line in src:
                        line = line.strip()
                        if line:
                            out.write(("" if first else ",") + "\n" + line)
                            first = False
            out.write("\n]}\n")
        os.replace(tmp, self.final)
        return self.final
