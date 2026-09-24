"""Orchestrates one run: manifest -> engine -> stream rows -> sink -> results.json."""

from __future__ import annotations

import platform
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from indic_runner.config import DIRS
from indic_runner.runtime.engine_lifecycle import build_engine, running_engine
from indic_runner.runtime.engines.base import Engine, GenerationConfig, Request
from indic_runner.runtime.manifest_loader import Manifest
from indic_runner.runtime.output_sink import OutputSink
from indic_runner.runtime.streaming_pipeline import batched, iter_rows, scan_dataset
from indic_runner.runtime.task_adapters import get_adapter


@dataclass(frozen=True)
class RunOptions:
    gen: GenerationConfig = GenerationConfig()
    max_retries: int = 1
    resume_run_id: str | None = None
    ocr_variant: str | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    results_file: Path
    status: str
    successful: int
    failed: int


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _error_row(row: dict, message: str, latency: float = 0.0) -> dict:
    return {"id": row["id"], "input": _input_text(row), "response": "", "reasoning": "",
            "latency_s": latency, "status": "error", "error": message, "image_sha256": None}


def _input_text(row: dict) -> str:
    return str(row.get("input", row.get("image_path", "")))


def run(manifest: Manifest, dataset: Path, options: RunOptions = RunOptions(),
        engine: Engine | None = None) -> RunResult:
    if manifest.task == "ocr" and options.ocr_variant not in ("normal", "scanned"):
        raise ValueError("OCR runs need ocr_variant 'normal' or 'scanned'")
    if manifest.task != "ocr" and options.ocr_variant:
        raise ValueError("ocr_variant only applies to OCR runs")

    info = scan_dataset(dataset, manifest.task)  # validates every row before any engine starts
    adapter = get_adapter(manifest.task, manifest.engine, manifest.alias)
    resumed = options.resume_run_id is not None
    run_id = options.resume_run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    sink = OutputSink(DIRS["runs"] / run_id, resume=resumed)
    done = sink.done_ids() if resumed else set()
    created = datetime.now(timezone.utc).isoformat()

    latencies: list[float] = []
    ok = failed = 0
    started = time.monotonic()
    engine = engine or build_engine(manifest)
    try:
        with running_engine(engine):
            todo = (r for r in iter_rows(dataset, manifest.task) if r["id"] not in done)
            for batch in batched(todo, manifest.max_batch_size):
                for out in _process(batch, adapter, engine, options):
                    sink.append(out)
                    latencies.append(out["latency_s"])
                    if out["status"] == "ok":
                        ok += 1
                    else:
                        failed += 1
    finally:
        sink.close()
    wall = time.monotonic() - started

    if resumed:  # counts must cover the whole run, not just this session
        ok = failed = 0
        for line in _iter_status(sink.partial):
            ok, failed = (ok + 1, failed) if line == "ok" else (ok, failed + 1)
    status = "failed" if ok == 0 and failed else "partial" if failed else "completed"
    rows_now = ok + failed
    doc = {
        "run_id": run_id,
        "model": {"name": manifest.alias, "revision": manifest.raw.get("revision", "unknown"),
                  "backend": manifest.engine, "path": str(manifest.artifacts_dir)},
        "dataset": {"name": info.name, "use_case": manifest.task,
                    "language": info.languages[0] if len(info.languages) == 1 else "multi",
                    "languages": info.languages, "row_count": info.row_count,
                    "ocr_variant": options.ocr_variant},
        "runtime": {"device_requested": manifest.hardware, "device_used": manifest.hardware,
                    "hardware": f"{platform.system()}/{platform.machine()}",
                    "quantisation": manifest.precision, "dtype": manifest.precision,
                    "fallback_reason": None,
                    "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
                    "latency_p95_s": _pct(latencies, 0.95),
                    "throughput_rows_per_s": (len(latencies) / wall) if wall > 0 else 0.0,
                    "peak_vram_gb": None,
                    "performance_notes": "latency covers this session's rows only" if resumed else "",
                    "timeout_s": options.gen.timeout_s, "max_retries": options.max_retries,
                    "resumed": resumed, "python_version": platform.python_version()},
        "generation": {"batch_size": manifest.max_batch_size,
                       "max_new_tokens": options.gen.max_new_tokens,
                       "temperature": options.gen.temperature,
                       "do_sample": options.gen.do_sample, "num_beams": options.gen.num_beams},
        "run": {"created_at": created, "status": status, "successful_count": ok,
                "failed_count": failed, "resumed": resumed},
    }
    final = sink.assemble(doc)
    return RunResult(run_id, final, status, ok, failed)


def _iter_status(partial: Path):
    import json
    with partial.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["status"]


def _process(batch: list[dict], adapter, engine: Engine, options: RunOptions):
    """Yield one result row per input row, retrying failed rows."""
    rows = {r["id"]: r for r in batch}
    order = [r["id"] for r in batch]
    built: dict[str, Request] = {}
    results: dict[str, dict] = {}
    for rid in order:
        try:
            built[rid] = adapter.build(rows[rid])
        except Exception as exc:  # noqa: BLE001 - bad row is a row failure, not a run failure
            results[rid] = _error_row(rows[rid], f"{type(exc).__name__}: {exc}")

    pending = [rid for rid in order if rid in built]
    for attempt in range(options.max_retries + 1):
        if not pending:
            break
        t0 = time.monotonic()
        responses = engine.infer([built[r] for r in pending], options.gen)
        elapsed = time.monotonic() - t0
        retry = []
        for rid, resp in zip(pending, responses):
            latency = resp.latency_s if resp.latency_s is not None else elapsed
            if resp.error is not None or not resp.text.strip():
                message = resp.error or "empty response"
                results[rid] = _error_row(rows[rid], message, latency)
                retry.append(rid)
                continue
            text, reasoning = adapter.parse(resp.text)
            results[rid] = {"id": rid, "input": _input_text(rows[rid]), "response": text,
                            "reasoning": reasoning, "latency_s": latency, "status": "ok",
                            "error": None, "image_sha256": _sha(adapter, rows[rid])}
        pending = retry
    for rid in order:
        yield results[rid]


def _sha(adapter, row):
    try:
        return adapter.image_sha256(row)
    except OSError:
        return None
